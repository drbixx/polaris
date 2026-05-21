"""Simulate a set of images at a given coordinate in the sky

A local GAIA database is used to obtain star positions.


View a (4, 3) array with DS9 as follows:

ds9 -tile grid layout 3 4 'polaris.fits[3]' 'polaris.fits[2]' 'polaris.fits[1]' 'polaris.fits[6]' 'polaris.fits[5]' 'polaris.fits[4]' 'polaris.fits[9]' 'polaris.fits[8]' 'polaris.fits[7]' 'polaris.fits[12]' 'polaris.fits[11]' 'polaris.fits[10]'

"""

from argparse import ArgumentParser
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


def mod_vector(angle: float):
    """Return a module vector for a specific angle"""

    angle = np.deg2rad(angle)
    return [1, np.cos(2 * angle), -np.sin(2 * angle)]


class CCDImage:
    def __init__(self, shape: tuple[int], pixelsize: float = 0.2, ron: float = 0.0):
        self.shape = shape
        self.pixelsize = pixelsize * units.arcsec
        self.ron = ron

    def run(self, table: Table, wcs: WCS, beta: float = 3.5, shift=None):
        """Simulate a single CCD with a given WCS for stars given in `table`"""

        coords = SkyCoord(table["ra"], table["dec"], unit="degree")
        xcoords, ycoords = coords.to_pixel(wcs)
        if shift:
            xcoords += shift["x"]
            ycoords += shift["y"]

        table["x_0"] = xcoords
        table["y_0"] = ycoords
        sel = (xcoords > 0) & (xcoords < self.shape[0])
        sel &= (ycoords > 0) & (ycoords < self.shape[1])
        seltable = table[sel]
        psf_model = MoffatPSF(bbox_factor=150.0 / beta)
        image = make_model_image(tuple(self.shape[::-1]), psf_model, seltable)

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

        centers = self._calc_centers(center)
        ordered_centers = sorted(
            centers.items(), key=lambda item: (item[0][1], item[0][0])
        )
        template_primary = None
        template_images = ()
        primary_header_size = None
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
            image = self.ccd.run(table, wcs, starsim.beta, shift=shift)
            if polstars:
                # Add polarised stars
                image = image + self.ccd.run(polstars, wcs, starsim.beta)
            image = image + skylevel * t_exp
            if poisson_noise:
                image = apply_poisson_noise(image, seed=rng)
            # Add read-out noise
            image = image + rng.normal(0.0, self.ccd.ron, size=image.shape)
            image = np.clip(np.rint(image), 0, np.iinfo(np.uint16).max).astype(
                np.uint16
            )
            header = template_header if template_header is not None else wcs.to_header()
            hdu = pyfits.ImageHDU(image, header=header)
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
        for expno, angle in enumerate(angles, start=1):
            # Allow for shifts in the polarizer frames
            shift = {}
            if f"x_{angle}" in shifts:
                shift["x"] = shifts[f"x_{angle}"]
            if f"y_{angle}" in shifts:
                shift["y"] = shifts[f"y_{angle}"]
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
                rng=rng,
            )

    else:
        logger.info("Creating mosaic")
        mosaic.simulate(
            center,
            starsim,
            t_exp=params["t_exp"],
            stars=stars,
            skylevel=params["skylevel"],
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
