"""Simulate a set of images at a given coordinate in the sky

A local GAIA database is used to obtain star positions.


View a (4, 3) array with DS9 as follows:

ds9 -tile grid layout 3 4 'polaris.fits[3]' 'polaris.fits[2]' 'polaris.fits[1]' 'polaris.fits[6]' 'polaris.fits[5]' 'polaris.fits[4]' 'polaris.fits[9]' 'polaris.fits[8]' 'polaris.fits[7]' 'polaris.fits[12]' 'polaris.fits[11]' 'polaris.fits[10]'

"""

from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import logging
from pathlib import Path
import tomllib

from astropy.coordinates import SkyCoord
import astropy.io.fits as pyfits
from astropy.table import Table, vstack
from astropy import units
from astropy.wcs import WCS
import duckdb
import numpy as np
from photutils.datasets import make_model_image, apply_poisson_noise
from photutils.psf import MoffatPSF


LOGLEVELS = {
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
MATRIX = {
    # Unnormalized modulation matrix
    # inner lists are the rows
    3: [[1, 1, 0], [1, -0.5, -np.sqrt(3) / 2], [1, -0.5, np.sqrt(3) / 2]],
    # 4: [[1, 1, 1, 1], [2, 0, -2, 0], [0, -2, 0, -2]],
}
DEFAULT_OMEGACAM_HEADER_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "OMEGACAM-master.txt"
)
DEFAULT_OMEGACAM_BIAS_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "OMEGACAM.2018-02-19T05:32:25.059.fits"
)
PRIMARY_HEADER_SKIP = {"SIMPLE", "NAXIS", "EXTEND", "CHECKSUM", "DATASUM"}
IMAGE_HEADER_SKIP = {
    "XTENSION",
    "BITPIX",
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "PCOUNT",
    "GCOUNT",
    "BSCALE",
    "BZERO",
    "CHECKSUM",
    "DATASUM",
}
DYNAMIC_HEADER_KEYS = {
    "MAGZP",
    "BETA",
    "BKG",
    "POLANGLE",
    "ESO TPL ID",
    "ESO TPL NEXP",
    "ESO TPL EXPNO",
}
WCS_HEADER_PREFIXES = (
    "WCSAXES",
    "CTYPE",
    "CRPIX",
    "CRVAL",
    "CDELT",
    "CUNIT",
    "CD",
    "PC",
    "PV",
    "LONPOLE",
    "LATPOLE",
    "RADESYS",
)


logger = logging.getLogger(__package__)


def _normalize_header_key(key: str) -> str:
    return key.removeprefix("HIERARCH ").upper()


def _is_wcs_header_key(key: str) -> bool:
    key = _normalize_header_key(key)
    return key.startswith(WCS_HEADER_PREFIXES)


def _copy_header_cards(
    target: pyfits.Header,
    template: pyfits.Header | None,
    skip: set[str],
    skip_wcs: bool = False,
):
    if template is None:
        return

    existing = {_normalize_header_key(key) for key in target.keys()}
    for card in template.cards:
        key = card.keyword
        norm = _normalize_header_key(key)
        if norm in skip:
            continue
        if skip_wcs and _is_wcs_header_key(norm):
            continue
        if key in ("COMMENT", "HISTORY", ""):
            target.append(card)
            continue
        if norm in existing:
            continue
        target[key] = (card.value, card.comment)
        existing.add(norm)


def _pad_header_to_size(target: pyfits.Header, size: int):
    if size <= 0:
        return
    while len(target.tostring(endcard=True, padding=True)) < size:
        target.append(pyfits.Card.fromstring(" " * 80), end=True)


def _to_utc_timestamp(timestamp: datetime | None) -> datetime:
    if timestamp is None:
        return datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _format_iso_millis(timestamp: datetime | None) -> str:
    return _to_utc_timestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


@lru_cache(maxsize=4)
def _load_header_template(
    path: str,
) -> tuple[pyfits.Header | None, tuple[pyfits.Header, ...]]:
    template_path = Path(path)
    if not template_path.exists():
        logger.warning("header template not found: %s", template_path)
        return None, ()
    if template_path.suffix.lower() == ".txt":
        headers = []
        current = None
        for line in template_path.read_text().splitlines():
            if line.startswith("# HDU "):
                if current is not None:
                    headers.append(current)
                current = pyfits.Header()
                continue
            if current is None:
                continue
            if len(line) == 0:
                continue
            card_line = line if len(line) >= 80 else line.ljust(80)
            current.append(pyfits.Card.fromstring(card_line[:80]), end=True)
        if current is not None:
            headers.append(current)
        if not headers:
            logger.warning("header template has no HDUs: %s", template_path)
            return None, ()
        primary_header = headers[0]
        image_headers = tuple(headers[1:])
    else:
        with pyfits.open(
            template_path, memmap=False, do_not_scale_image_data=True
        ) as hdul:
            primary_header = hdul[0].header.copy()
            image_headers = tuple(
                hdu.header.copy()
                for hdu in hdul[1:]
                if isinstance(hdu, pyfits.ImageHDU)
            )

    if not image_headers:
        logger.warning("header template has no image HDUs: %s", template_path)

    return primary_header, image_headers


def _get_detector_sections(
    header: pyfits.Header | None, fallback_shape: tuple[int, int]
) -> dict[str, int | slice]:
    def _infer_output_low_side(axis: str, active_size: int) -> bool:
        if header is None:
            return True

        out_key = f"ESO DET OUT1 {axis}"
        out_coord = header.get(out_key)
        if out_coord is None:
            return True
        out_coord = int(out_coord)

        chip_key = f"ESO DET CHIP {axis}"
        chip_index = header.get(chip_key)
        if chip_index is not None and active_size > 0:
            chip_index = int(chip_index)
            chip_start = (chip_index - 1) * active_size + 1
            chip_end = chip_index * active_size
            return abs(out_coord - chip_start) <= abs(out_coord - chip_end)

        if active_size > 0:
            out_local = ((out_coord - 1) % active_size) + 1
            return out_local <= (active_size + 1) // 2

        return True

    if header is None:
        nx, ny = fallback_shape
        prscx, prscy, ovscx, ovscy = 0, 0, 0, 0
    else:
        nx = int(
            header.get(
                "ESO DET OUT1 NX", header.get("ESO DET CHIP NX", fallback_shape[0])
            )
        )
        ny = int(
            header.get(
                "ESO DET OUT1 NY", header.get("ESO DET CHIP NY", fallback_shape[1])
            )
        )
        prscx = int(header.get("ESO DET OUT1 PRSCX", 0))
        prscy = int(header.get("ESO DET OUT1 PRSCY", 0))
        ovscx = int(header.get("ESO DET OUT1 OVSCX", 0))
        ovscy = int(header.get("ESO DET OUT1 OVSCY", 0))

    total_x = prscx + nx + ovscx
    total_y = prscy + ny + ovscy
    x_output_low = _infer_output_low_side("X", nx)
    y_output_low = _infer_output_low_side("Y", ny)

    if x_output_low:
        prex_x = slice(0, prscx)
        active_x = slice(prscx, prscx + nx)
        ovx_x = slice(active_x.stop, active_x.stop + ovscx)
    else:
        ovx_x = slice(0, ovscx)
        active_x = slice(ovscx, ovscx + nx)
        prex_x = slice(active_x.stop, active_x.stop + prscx)

    if y_output_low:
        prey_y = slice(0, prscy)
        active_y = slice(prscy, prscy + ny)
        ovy_y = slice(active_y.stop, active_y.stop + ovscy)
    else:
        ovy_y = slice(0, ovscy)
        active_y = slice(ovscy, ovscy + ny)
        prey_y = slice(active_y.stop, active_y.stop + prscy)
    return {
        "nx": nx,
        "ny": ny,
        "prscx": prscx,
        "prscy": prscy,
        "ovscx": ovscx,
        "ovscy": ovscy,
        "total_x": total_x,
        "total_y": total_y,
        "x_output_low": x_output_low,
        "y_output_low": y_output_low,
        "active_x": active_x,
        "active_y": active_y,
        "prex_x": prex_x,
        "ovx_x": ovx_x,
        "prey_y": prey_y,
        "ovy_y": ovy_y,
    }


@lru_cache(maxsize=2)
def _load_bias_levels(path: str) -> dict[str, dict[str, float]]:
    ref_path = Path(path)
    if not ref_path.exists():
        logger.warning("bias template not found: %s", ref_path)
        return {}

    levels = {}
    with pyfits.open(ref_path, memmap=False) as hdul:
        for idx, hdu in enumerate(hdul[1:], start=1):
            if not isinstance(hdu, pyfits.ImageHDU) or hdu.data is None:
                continue
            data = np.asarray(hdu.data, dtype=np.float64)
            fallback = (data.shape[1], data.shape[0])
            sections = _get_detector_sections(hdu.header, fallback)
            ysec = sections["active_y"]

            prex = None
            if sections["prscx"] > 0:
                prex = float(np.median(data[ysec, sections["prex_x"]]))

            ovx = None
            if sections["ovscx"] > 0:
                ovx = float(np.median(data[ysec, sections["ovx_x"]]))

            prey = None
            if sections["prscy"] > 0:
                prey = float(np.median(data[sections["prey_y"], :]))

            ovy = None
            if sections["ovscy"] > 0:
                ovy = float(np.median(data[sections["ovy_y"], :]))

            scan_values = [
                value for value in (prex, ovx, prey, ovy) if value is not None
            ]
            if not scan_values:
                continue

            default_bias = float(np.median(scan_values))
            active_bias_candidates = [
                value for value in (prex, ovx) if value is not None
            ]
            if active_bias_candidates:
                active_bias = float(np.median(active_bias_candidates))
            else:
                active_bias = default_bias
            extname = hdu.header.get("EXTNAME", f"EXT{idx}")
            levels[extname] = {
                "active": active_bias,
                "prex": prex if prex is not None else default_bias,
                "ovx": ovx if ovx is not None else default_bias,
                "prey": prey if prey is not None else default_bias,
                "ovy": ovy if ovy is not None else default_bias,
            }

    return levels


def mod_vector(angle: float):
    """Return a module vector for a specific angle"""

    angle = np.deg2rad(angle)
    return [1, np.cos(2 * angle), -np.sin(2 * angle)]


class CCDImage:
    def __init__(self, shape: tuple[int], pixelsize: float = 0.2, ron: float = 0.0):
        self.shape = shape
        self.pixelsize = pixelsize * units.arcsec
        self.ron = ron

    def run(
        self,
        table: Table,
        wcs: WCS,
        beta: float = 3.5,
        shift=None,
        shape: tuple[int, int] | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
    ):
        """Simulate a single CCD with a given WCS for stars given in `table`"""
        if shape is None:
            shape = self.shape

        coords = SkyCoord(table["ra"], table["dec"], unit="degree")
        xcoords, ycoords = coords.to_pixel(wcs)
        if shift:
            xcoords += shift["x"]
            ycoords += shift["y"]
        xcoords -= origin[0]
        ycoords -= origin[1]

        table["x_0"] = xcoords
        table["y_0"] = ycoords
        sel = (xcoords > 0) & (xcoords < shape[0])
        sel &= (ycoords > 0) & (ycoords < shape[1])
        seltable = table[sel]
        psf_model = MoffatPSF(bbox_factor=150.0 / beta)
        image = make_model_image(tuple(shape[::-1]), psf_model, seltable)

        return image


def query_catalog(name, coord: SkyCoord, fov: dict[str, units.Quantity]):
    # radius = np.sqrt((fov['width'] / 2)**2 + (fov['height'] / 2)**2)
    ra = coord.ra.degree
    decl = coord.dec.degree
    width = fov["width"].to(units.degree).value
    height = fov["height"].to(units.degree).value
    ra_range = [ra - width / 2, ra + width / 2]
    dec_range = [decl - height / 2, decl + height / 2]

    with duckdb.connect(name, read_only=True) as conn:
        query = conn.execute(
            """
select ra, "dec", mag from catalog where
ra between ? and ? and "dec" between ? and ?
    """,
            (*ra_range, *dec_range),
        )
        df = query.df()
    catalog = Table.from_pandas(df)

    return catalog


def create_catalog(name, stars: Table):
    """Save a table of stars to a DuckDB database

    The required columns are 'ra', 'dec' and 'mag'

    """

    df = stars.to_pandas()
    if len({"ra", "dec", "mag"} & set(df.columns)) != 3:
        raise ValueError("missing column(s) in table")

    with duckdb.connect(name, read_only=False) as conn:
        conn.execute("drop table if exists catalog")
        conn.execute("create table catalog as select * from df")
    logger.info("Stored selected stars in catalog %s", name)


def create_wcs(center: SkyCoord, ccd: CCDImage):
    ra = center.ra.degree
    decl = center.dec.degree
    nx, ny = ccd.shape

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2, ny / 2]
    wcs.wcs.cdelt = np.array([-ccd.pixelsize.value, ccd.pixelsize.value]) / 3600
    wcs.wcs.crval = [ra, decl]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]

    return wcs


class StarSimulation:
    def __init__(
        self,
        magrange: tuple[float],
        psfparams: dict[str, float],
        catalog: str | None = None,
        catalog_fraction: float = -1,
    ):
        """

        - magminmax: magnitude range, [minimum, maximum]
             Note that the lowest value is always interpreted as the minimum magnitude

        - fwhm: full width half maximum in pixels

        - fwhm_std: spread in fwhm, in pixels

        - beta: "wings" parameter of the Moffat PSF function
            Alpha is determined from the fwhm

        - catalog: catalog name of the DuckDB database file

        - catalog_fraction: if a positive fraction, don't use the catalog as input,
            but save a fraction of the simulated stars to the catalog as output

        """

        self.magrange = magrange
        self.fwhm = psfparams.get("fwhm", 2.5)
        self.fwhm_std = psfparams.get("fwhm_std", 0.0)
        self.beta = psfparams.get("beta", 3.5)
        self.catalog = catalog
        self.catalog_fraction = catalog_fraction

    def copy(self):
        return StarSimulation(
            magrange=self.magrange[:],
            psfparams={"fwhm": self.fwhm, "fwhm_std": self.fwhm_std, "beta": self.beta},
            catalog=self.catalog,
            catalog_fraction=self.catalog_fraction,
        )

    def run(
        self,
        center: SkyCoord,
        fov: dict[str, units.Quantity],
        t_exp: float,
        nstars: int = 1000,
        zeropoint: float = 30.0,
        rng=None,
    ):
        """Create a list of random stars, and optionally add stars from an existing catalog to it"""

        rng = np.random.default_rng(rng)

        stars = Table()
        logger.info("Simulating %d random stars", nstars)
        low = (center.ra - fov["width"] / 2).value
        high = (center.ra + fov["width"] / 2).value
        stars["ra"] = rng.uniform(low, high, nstars) * units.degree
        low = (center.dec - fov["height"] / 2).value
        high = (center.dec + fov["height"] / 2).value
        stars["dec"] = rng.uniform(low, high, nstars) * units.degree
        unirand = np.random.uniform(0, 1, nstars)
        mmin, mmax = sorted(self.magrange)
        stars["mag"] = (1 / 0.6) * np.log10(
            unirand * (10 ** (0.6 * mmax) - 10 ** (0.6 * mmin)) + 10 ** (0.6 * mmin)
        )

        # Use catalog stars or save a random fraction to a catalog
        if self.catalog_fraction <= 0 and self.catalog:
            catalog = query_catalog(self.catalog, center, fov)
            logger.info("Using %d stars from the database catalog", len(catalog))
            stars = vstack([stars, catalog])
        elif self.catalog_fraction > 0 and self.catalog:
            n = len(stars)
            size = int(self.catalog_fraction * n)
            logger.info("Selecting %d stars as catalog stars", size)
            indices = rng.choice(len(stars), size=size, replace=False)
            create_catalog(self.catalog, stars[indices])

        stars["flux"] = t_exp * 10.0 ** (-0.4 * (stars["mag"] - zeropoint))
        stars["fwhm"] = rng.normal(self.fwhm, self.fwhm_std, len(stars))

        return stars


class Mosaic:
    def __init__(
        self, ccd: CCDImage, layout: tuple[int] = (1, 1), gaps: tuple[float] = (0, 0)
    ):
        self.layout = layout
        self.gaps = gaps
        self.ccd = ccd

        self.fov = {
            "width": ccd.pixelsize
            * (ccd.shape[0] * layout[0] + gaps[0] * (layout[0] - 1)),
            "height": ccd.pixelsize
            * (ccd.shape[1] * layout[1] + gaps[1] * (layout[1] - 1)),
        }

    def simulate(
        self,
        center: SkyCoord,
        starsim: StarSimulation,
        t_exp: float,
        stars: int | Table = 1000,
        skylevel: float = 1,
        zeropoint: float = 30.0,
        outfile: str | Path = "polaris.fits",
        angle: float = None,
        tpl_id: str | None = None,
        tpl_nexp: int | None = None,
        tpl_expno: int | None = None,
        header_template: str | Path | None = DEFAULT_OMEGACAM_HEADER_TEMPLATE,
        polstars=None,
        poisson_noise: bool = True,
        shift: dict[int, float] | None = None,
        obs_start: datetime | None = None,
        tpl_start: datetime | None = None,
        date_obs: datetime | None = None,
        rng=None,
    ):
        if isinstance(stars, int):
            table = starsim.run(
                center, self.fov, t_exp, stars, zeropoint=zeropoint, rng=rng
            )
        else:
            table = stars
        field_ra = center.ra.degree
        field_dec = center.dec.degree
        date_obs_ts = _to_utc_timestamp(date_obs)
        obs_start_ts = _to_utc_timestamp(
            obs_start if obs_start is not None else date_obs_ts
        )
        tpl_start_ts = _to_utc_timestamp(
            tpl_start if tpl_start is not None else obs_start_ts
        )
        date_ts = date_obs_ts + timedelta(seconds=t_exp)
        date_str = _format_iso_millis(date_ts)
        date_obs_str = _format_iso_millis(date_obs_ts)
        obs_start_str = _format_iso_millis(obs_start_ts)
        tpl_start_str = _format_iso_millis(tpl_start_ts)

        centers = self._calc_centers(center)
        ordered_centers = sorted(
            centers.items(), key=lambda item: (item[0][1], item[0][0])
        )
        template_primary = None
        template_images = ()
        primary_header_size = None
        bias_profiles = _load_bias_levels(str(DEFAULT_OMEGACAM_BIAS_TEMPLATE))
        if bias_profiles:
            keys = ("active", "prex", "ovx", "prey", "ovy")
            default_profile = {
                key: float(
                    np.median([profile[key] for profile in bias_profiles.values()])
                )
                for key in keys
            }
        else:
            default_profile = {
                "active": 0.0,
                "prex": 0.0,
                "ovx": 0.0,
                "prey": 0.0,
                "ovy": 0.0,
            }
        if header_template:
            template_primary, template_images = _load_header_template(
                str(header_template)
            )
            if template_primary is not None:
                primary_header_size = len(
                    template_primary.tostring(endcard=True, padding=True)
                )

        common_cards = [
            ("magzp", zeropoint, ""),
            ("beta", starsim.beta, ""),
            ("bkg", skylevel, ""),
        ]
        image_cards = common_cards[:]
        primary_cards = common_cards[:]
        if angle is not None:
            card = ("polangle", angle, "polarizer angle")
            primary_cards.append(card)
            image_cards.append(card)
        if tpl_id is not None:
            primary_cards.append(
                ("HIERARCH ESO TPL ID", tpl_id, "Template signature ID")
            )
        if tpl_nexp is not None:
            primary_cards.append(
                (
                    "HIERARCH ESO TPL NEXP",
                    tpl_nexp,
                    "Number of exposures within template",
                )
            )
        if tpl_expno is not None:
            primary_cards.append(
                (
                    "HIERARCH ESO TPL EXPNO",
                    tpl_expno,
                    "Exposure number within template",
                )
            )

        hdus = [pyfits.PrimaryHDU()]
        _copy_header_cards(
            hdus[0].header,
            template_primary,
            skip=PRIMARY_HEADER_SKIP | DYNAMIC_HEADER_KEYS,
        )
        if template_primary is not None and "BITPIX" in template_primary:
            hdus[0].header["BITPIX"] = (
                template_primary["BITPIX"],
                template_primary.comments["BITPIX"],
            )
        if "RA" in hdus[0].header:
            hdus[0].header["RA"] = field_ra
        if "DEC" in hdus[0].header:
            hdus[0].header["DEC"] = field_dec
        hdus[0].header.extend(primary_cards)
        hdus[0].header["DATE"] = date_str
        hdus[0].header["DATE-OBS"] = date_obs_str
        hdus[0].header["HIERARCH ESO OBS START"] = obs_start_str
        hdus[0].header["HIERARCH ESO TPL START"] = tpl_start_str
        if primary_header_size is not None:
            _pad_header_to_size(hdus[0].header, primary_header_size)
        for i in range(len(ordered_centers)):
            logger.info("Creating single CCD image")
            template_hdu = template_images[i] if i < len(template_images) else None
            template_header = None
            if template_hdu is not None:
                template_header = template_hdu.copy()
                if "CRVAL1" in template_header:
                    template_header["CRVAL1"] = field_ra
                if "CRVAL2" in template_header:
                    template_header["CRVAL2"] = field_dec
                wcs = WCS(template_header)
            else:
                _, chip_center = ordered_centers[i]
                wcs = create_wcs(chip_center, self.ccd)
            sections = _get_detector_sections(template_header, self.ccd.shape)
            active_shape = (sections["nx"], sections["ny"])
            origin = (sections["active_x"].start, sections["active_y"].start)
            active_image = self.ccd.run(
                table, wcs, starsim.beta, shift=shift, shape=active_shape, origin=origin
            )
            if polstars:
                # Add polarised stars
                active_image = active_image + self.ccd.run(
                    polstars, wcs, starsim.beta, shape=active_shape, origin=origin
                )
            active_image = active_image + skylevel * t_exp
            if poisson_noise:
                active_image = apply_poisson_noise(active_image, seed=rng)
            # Add read-out noise
            active_image = active_image + rng.normal(
                0.0, self.ccd.ron, size=active_image.shape
            )
            extname = (
                template_header.get("EXTNAME", f"EXT{i + 1}")
                if template_header is not None
                else f"EXT{i + 1}"
            )
            profile = bias_profiles.get(extname, default_profile)
            active_bias = profile["active"]
            image = rng.normal(
                loc=active_bias,
                scale=self.ccd.ron,
                size=(sections["total_y"], sections["total_x"]),
            )
            image[sections["active_y"], sections["active_x"]] = (
                active_bias + active_image
            )
            if sections["prscx"] > 0:
                image[sections["active_y"], sections["prex_x"]] = rng.normal(
                    loc=profile["prex"],
                    scale=self.ccd.ron,
                    size=(sections["ny"], sections["prscx"]),
                )
            if sections["ovscx"] > 0:
                image[sections["active_y"], sections["ovx_x"]] = rng.normal(
                    loc=profile["ovx"],
                    scale=self.ccd.ron,
                    size=(sections["ny"], sections["ovscx"]),
                )
            if sections["prscy"] > 0:
                image[sections["prey_y"], :] = rng.normal(
                    loc=profile["prey"],
                    scale=self.ccd.ron,
                    size=(sections["prscy"], sections["total_x"]),
                )
            if sections["ovscy"] > 0:
                image[sections["ovy_y"], :] = rng.normal(
                    loc=profile["ovy"],
                    scale=self.ccd.ron,
                    size=(sections["ovscy"], sections["total_x"]),
                )
            image = np.clip(np.rint(image), 0, np.iinfo(np.uint16).max).astype(
                np.uint16
            )
            header = template_header if template_header is not None else wcs.to_header()
            hdu = pyfits.ImageHDU(image, header=header)
            hdu.header["DATE"] = date_str
            hdu.header.extend(image_cards)
            hdus.append(hdu)
        logger.info("Writing mosaic to file %s", outfile)
        pyfits.HDUList(hdus).writeto(outfile, overwrite=True)

    def _calc_centers(self, center: SkyCoord):
        # Global x and y center
        gxc = self.layout[0] / 2
        gyc = self.layout[1] / 2
        offsets = {}  # offsets in arcseconds from the FoV center
        for x in range(self.layout[0]):
            # local chip center
            xc = x - (gxc - 0.5)
            xc = self.ccd.pixelsize * (self.ccd.shape[0] + self.gaps[0]) * xc
            for y in range(self.layout[1]):
                yc = y - (gyc - 0.5)
                yc = self.ccd.pixelsize * (self.ccd.shape[1] + self.gaps[1]) * yc
                offsets[(x, y)] = (xc, yc)

        centers = {
            key: center.spherical_offsets_by(*offset) for key, offset in offsets.items()
        }

        return centers


def save_to_region(sources, qfractions, ufractions, regfile):
    """Save the polarized stars to a region file, with annotation"""

    with open(regfile, "w") as file:
        file.write("# Region file format: DS9 version 4.1\n")
        file.write(
            'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 "
            "source=1\nfk5\n"
        )
        for x, y, q, u in zip(sources["ra"], sources["dec"], qfractions, ufractions):
            file.write(f'circle({x},{y},3.0") # text={{{q:.3f}, {u:.3f}}}\n')


def add_polarization(mosaic, center, stars, params, starsim, rng=None):
    """Add polarized stars to the existing simulation"""

    rng = np.random.default_rng(rng)

    # Create 'polarized' Q and U images
    # for selected stars
    polpars = params["polarisation"]
    starsim.magrange = polpars["magrange"]
    # Don't use stars from the catalog as polarized stars
    starsim.catalog = None
    nstars = polpars["nstars"]
    qrange = polpars["qrange"]
    urange = polpars["urange"]
    # Add `nstars` stars to the I images
    stokes_i = starsim.run(
        center,
        mosaic.fov,
        params["t_exp"],
        nstars,
        zeropoint=params["zeropoint"],
        rng=rng,
    )

    # The Q & U frames are copies of I
    stokes_q = stokes_i.copy()
    stokes_u = stokes_i.copy()
    # with their fluxes changed by random amount
    qfractions = rng.uniform(*qrange, size=nstars)
    stokes_q["flux"] *= qfractions
    ufractions = rng.uniform(*urange, size=nstars)
    stokes_u["flux"] *= ufractions

    # Save the 'polarized' stars to a DS9 region file
    if regfile := params["region"]["regfile"]:
        save_to_region(stokes_i, qfractions, ufractions, regfile)

    return {"I": stokes_i, "Q": stokes_q, "U": stokes_u}


def run(params):
    rng = np.random.default_rng(params["seed"])
    center = SkyCoord(params["sky"]["ra"], params["sky"]["dec"], unit="degree")
    psfparams = params["psf"]
    action = params["catalog"]["action"]
    catfraction = -1
    if action == "create":
        catfraction = params["catalog"]["fraction"]
        print(catfraction, type(catfraction))
    elif action != "use":
        raise ValueError(
            '[simulation.catalog]: action should be one of "create" or "use"'
        )

    starsim = StarSimulation(
        magrange=params["magrange"],
        psfparams=psfparams,
        catalog=params["catalog"]["dbname"],
        catalog_fraction=catfraction,
    )
    ccdparams = params["ccd"]
    ccd = CCDImage(
        shape=ccdparams["shape"], pixelsize=ccdparams["pixelsize"], ron=ccdparams["ron"]
    )
    mosaic = Mosaic(ccd, params["mosaic"]["layout"], params["mosaic"]["gaps"])
    stars = starsim.run(
        center,
        mosaic.fov,
        params["t_exp"],
        params["nstars"],
        zeropoint=params["zeropoint"],
        rng=rng,
    )

    if "polarisation" in params:
        stokes = add_polarization(mosaic, center, stars, params, starsim, rng=rng)

        # Convert the I, Q and U frames to frames at polarizer angles
        polpars = params["polarisation"]
        shifts = params["shifts"]
        angles = polpars["angles"]
        nexp = len(angles)
        obs_start = datetime.now(timezone.utc)
        tpl_start = obs_start
        for expno, angle in enumerate(angles, start=1):
            # Allow for shifts in the polarizer frames
            shift = {}
            if f"x_{angle}" in shifts:
                shift["x"] = shifts[f"x_{angle}"]
            if f"y_{angle}" in shifts:
                shift["y"] = shifts[f"y_{angle}"]
            date_obs = obs_start + timedelta(seconds=(expno - 1) * params["t_exp"])
            polstars = stokes["I"].copy()
            modvec = mod_vector(angle)
            # modulate the flux for the current angle
            polstars["flux"] = (
                stokes["I"]["flux"] * modvec[0]
                + stokes["Q"]["flux"] * modvec[1]
                + stokes["U"]["flux"] * modvec[2]
            ) / 2

            logger.info("Creating mosaic for polarization angle %.1f", angle)
            outfile = f"polaris-deg{angle:.1f}".replace(".", "_") + ".fits"

            # Save the output file, with normal stars and polarized stars
            mosaic.simulate(
                center,
                starsim,
                t_exp=params["t_exp"],
                stars=stars,
                skylevel=params["skylevel"],
                outfile=outfile,
                angle=angle,
                tpl_id="OMEGACAM_img_pol",
                tpl_nexp=nexp,
                tpl_expno=expno,
                polstars=polstars,
                poisson_noise=params["poisson_noise"],
                shift=shift,
                obs_start=obs_start,
                tpl_start=tpl_start,
                date_obs=date_obs,
                rng=rng,
            )

    else:
        logger.info("Creating mosaic")
        obs_start = datetime.now(timezone.utc)
        mosaic.simulate(
            center,
            starsim,
            t_exp=params["t_exp"],
            stars=stars,
            skylevel=params["skylevel"],
            obs_start=obs_start,
            tpl_start=obs_start,
            date_obs=obs_start,
            rng=rng,
        )


def setup_logging(level: str = "warning"):
    """Set up some default logging configuration.

    Note: this doesn't use a dict- or file-config; it is felt that the
    logging setup should still be relatively simple, with only options
    for the logging level and whether or not to (also) log to file.

    """

    level = LOGLEVELS[level.lower()]
    fmt = "%(asctime)s  [%(levelname)-5s] - %(module)s.%(funcName)s():%(lineno)d: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", help="configuration file")
    parser.add_argument(
        "--loglevel",
        choices=["warning", "info", "debug"],
        default="warning",
        help="logging level",
    )
    args = parser.parse_args()
    with open(args.config, "rb") as fp:
        config = tomllib.load(fp)
    return config


def main():
    config = parse_args()
    setup_logging(level=config["logging"]["level"])

    params = config["simulation"]
    params["seed"] = None if params["random_seed"] < 0 else params["random_seed"]

    run(params)


if __name__ == "__main__":
    main()
