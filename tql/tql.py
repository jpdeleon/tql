#!/usr/bin/env python
import sys
import os
from os.path import join, exists
import traceback
import itertools
import warnings
from glob import glob

# import matplotlib.colors as mcolors
import matplotlib.pyplot as pl
import numpy as np
import time
import logging
from imp import reload
import pandas as pd
from tqdm import tqdm
import lightkurve as lk
from astropy import units as u
from scipy.ndimage import zoom  # , rotate
from scipy.stats.kde import gaussian_kde
from astropy.coordinates import SkyCoord, Distance
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astroplan import FixedTarget
from astroplan.plots import plot_finder_image
import deepdish as dd
from tqdm import tqdm
from transitleastsquares import transitleastsquares, catalog

# from transitleastsquares.tls_constants import DEFAULT_U
import getpass

warnings.filterwarnings("ignore", module="astropy.io.votable.tree")
if getpass.getuser() == "muscat":
    #    #non-GUI back-end
    pass
    # import matplotlib; matplotlib.use('agg')

MISSION = "TESS"
TESS_JD_offset = 2457000
SG_FILTER_WINDOW_SC = 401  # short-cadence: 361x2min = 722min= 12 hr
SG_FILTER_WINDOW_LC = 11  # long-cadence:  25x30min = 750min = 12.5 hr
TESS_pix_scale = 21 * u.arcsec  # /pix
FFI_CUTOUT_SIZE = 8  # pix
PHOTMETHOD = "aperture"  # or 'prf'
BINSIZE_SC = 5  # bin == 10 minutes
# APPHOTMETHOD  =  'pipeline'  or 'all' or threshold --> uses tpf.extract_aperture_photometry
# QUALITY_FLAGS  = lk.utils.TessQualityFlags()
# PGMETHOD       = 'lombscargle' # or 'boxleastsquares'; deprecated in favor of TLS
SFF_CHUNKSIZE = 27  # 27 chunks for a 27-day baseline
# there is a 3-day gap in all TESS dataset due to data downlink
# use chunksize larger than the gap i.e. chunksize>27/3
SFF_BINSIZE = 360  # 0.5 day for 2-minute cadence
quality_bitmask = "hard"  # or default?
time_format = "btjd"
time_scale = "tdb"  #'tt', 'ut1', or 'utc'
TLS_PERIOD_MIN = 0.01
TLS_PERIOD_MAX = 27  # defaults to baseline/2== 3 transits!
N_TRANSITS_MIN = 2
DEFAULT_U = [
    0.4804,
    0.1867,
]  # quadratic limb darkening for a G2V star in the Kepler bandpass
IMAGING_SURVEY = "DSS2 Red"
MAX_SECTORS = (
    5
)  # number of sectors to analyze if target is osberved in multiple sectors
MULTISEC_BIN = (
    10 * u.min
)  # binning for very dense data (observed >MAX_SECTORS)
FONTSIZE = 16
PLOT_KWARGS_LC = {"linewidth": 3}
PLOT_KWARGS_SC = {"linewidth": 1}
LOG_FILENAME = r"tql.log"
YLIMIT = (0.98, 1.02)  # flux limits

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "lime",
]
reload(logging)
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

def get_tpf(
    target_coord,
    tic=None,
    apphot_method="sap",
    apply_data_quality_mask=True,
    sector=None,
    verbose=False,
    clobber=False,
    sap_mask="pipeline",
    fitsoutdir=".",
    return_df=True,
):
    """Download tpf from MAST given coordinates
       though using TIC id yields unique match.

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    tic : int
        TIC id
    use_pld : bool
        use PLD for systematics correction
    toi : float
        TOI id
    sector : int
        TESS sector
    apphot_method : str
        aperture photometry method
    sap_mask : str
        SAP mask type
    fitsoutdir : str
        fits output directory
    figoutdir : str
        figure output directory
    clobber : bool
        re-download files

    Returns
    -------
    tpf and/or df: lk.targetpixelfile, pd.DataFrame
    """
    # sector = None searches for all tpf; which sector to download is specified later
    if tic:
        ticstr = f"TIC {tic}"
        if verbose:
            print(f"\nSearching mast for {ticstr}\n")
        res = lk.search_targetpixelfile(ticstr, mission=MISSION, sector=None)
    else:
        if verbose:
            print(
                f"\nSearching mast for ra,dec=({target_coord.to_string()})\n"
            )
        res = lk.search_targetpixelfile(
            target_coord, mission=MISSION, sector=None
        )
    df = res.table.to_pandas()

    if len(df) > 0:
        all_sectors = [int(i) for i in df["sequence_number"].values]
        if sector:
            sector_idx = df["sequence_number"][
                df["sequence_number"].isin([sector])
            ].index.tolist()
            if len(sector_idx) == 0:
                raise ValueError(
                    "sector {} data is unavailable".format(sector)
                )
            obsid = df.iloc[sector_idx]["obs_id"].values[0]
            ticid = int(df.iloc[sector_idx]["target_name"].values[0])
            fitsfilename = df.iloc[sector_idx]["productFilename"].values[0]
        else:
            sector_idx = 0
            sector = int(df.iloc[sector_idx]["sequence_number"])
            obsid = df.iloc[sector_idx]["obs_id"]
            ticid = int(df.iloc[sector_idx]["target_name"])
            fitsfilename = df.iloc[sector_idx]["productFilename"]

        msg = f"{len(df)} tpf(s) found in sector(s) {all_sectors}\n"
        msg += f"Using data from sector {sector} only\n"
        if verbose:
            logging.info(msg)
            print(msg)

        filepath = join(fitsoutdir, "mastDownload/TESS", obsid, fitsfilename)
        if not exists(filepath) or clobber:
            if verbose:
                print(f"Downloading TIC {ticid} ...\n")
            ticstr = f"TIC {ticid}"
            res = lk.search_targetpixelfile(
                ticstr, mission=MISSION, sector=sector
            )
            tpf = res.download(
                quality_bitmask=quality_bitmask, download_dir=fitsoutdir
            )
        else:
            if verbose:
                print("Loading TIC {} from {}/...\n".format(ticid, fitsoutdir))
            tpf = lk.TessTargetPixelFile(filepath)
        # assert tpf.mission == MISSION
        if apply_data_quality_mask:
            tpf = remove_bad_data(tpf, sector=tpf.sector)
        if return_df:
            return tpf, df
        else:
            return tpf
    else:
        msg = "No tpf file found! Check FFI data using --cadence=long\n"
        logging.info(msg)
        raise FileNotFoundError(msg)


def get_ffi_cutout(
    target_coord=None,
    tic=None,
    sector=None,  # cutout_size=10,
    apply_data_quality_mask=True,
    verbose=False,
    clobber=False,
    fitsoutdir=".",
    return_df=True,
):
    """Download a tpf cutout from full-frame images.
       Caveat: stars from FFI do not have TIC id.
       Does Gaia id make a good proxy for TIC id?

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    tic : int
        TIC id
    sector : int
        TESS sector
    cutout_size : int
        cutout size in pixel
    fitsoutdir : str
        fits output directory
    verbose : bool
        print texts
    clobber : bool
        re-download files
    """
    try:
        import eleanor as el
    except:
        raise ImportError("pip install eleanor")

    if tic is not None:
        ticstr = f"TIC {tic}"
        if verbose:
            print(f"\nSearching mast for {ticstr}\n")
        res = lk.search_tesscut(ticstr, sector=None)
        ticid = int(tic)
    else:
        if verbose:
            print(
                f"\nSearching mast for ra,dec=({target_coord.to_string()})\n"
            )
        res = lk.search_tesscut(target_coord, sector=None)
        # search using eleanor
        # ra, dec = target_coord.ra.deg, target_coord.dec.deg
        # star = el.Source(coords=(ra, dec), sector=sector)
        # ticid = int(star.tic)
        stars = Catalogs.query_region(target_coord, radius=3*u.arcsec, catalog="TIC").to_pandas()
        ticid = int(stars.loc[0,'ID'])
        ticstr = f"TIC {ticid}"
    df = res.table.to_pandas()

    if len(df) > 0:
        all_sectors = [int(i) for i in df["sequence_number"].values]
        if sector:
            sector_idx = df["sequence_number"][
                df["sequence_number"].isin([sector])
            ].index.tolist()
            if len(sector_idx) == 0:
                raise ValueError(
                    "sector {} data is unavailable".format(sector)
                )
            target = df.iloc[sector_idx]["targetid"].values[0]
        else:
            sector_idx = 0
            sector = int(df.iloc[sector_idx]["sequence_number"])
            target = df.iloc[sector_idx]["targetid"]

        msg = "{} tpf(s) found in sector(s) {}\n".format(len(df), all_sectors)
        msg += "Using data from sector {} only\n".format(sector)
        if verbose:
            logging.info(msg)
            print(msg)

        sr = lk.SearchResult(res.table)
        filepath = sr._fetch_tesscut_path(
            target, sector, fitsoutdir, FFI_CUTOUT_SIZE
        )

        if not exists(filepath) or clobber:
            if ticid is not None:
                res = lk.search_tesscut(ticstr, sector=sector)
                if verbose:
                    print("Downloading TIC {} ...\n".format(ticid))
            else:
                res = lk.search_tesscut(target_coord, sector=sector)
                if verbose:
                    print("Downloading {} ...\n".format(target_coord))
            tpf = res.download(
                quality_bitmask=quality_bitmask,
                cutout_size=FFI_CUTOUT_SIZE,
                download_dir=fitsoutdir,
            )
        else:
            if verbose:
                print("Loading TIC {} from {}/...\n".format(ticid, fitsoutdir))
            tpf = lk.TessTargetPixelFile(filepath)
        if apply_data_quality_mask:
            tpf = remove_bad_data(tpf, sector=tpf.sector)
        # set
        tpf.targetid = ticid
        if return_df:
            return tpf, df
        else:
            return tpf
    else:
        msg = "No full-frame tpf file found!\n"
        logging.info(msg)
        # sys.exit(msg)
        raise ValueError(msg)


def get_ffi_cutout_eleanor(
    target_coord=None, tic=None, sector=None, verbose=False
):
    """
    """
    raise NotImplementedError

    if target_coord is not None:
        star = eleanor.Source(coords=target_coord, sector=sector)
    elif tic is not None:
        star = eleanor.Source(tic=tic, sector=sector)
    data = eleanor.TargetData(
        star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=True
    )
    return data


def ffi_cutout_to_lc(
    tpf,
    sap_mask="threshold",
    aper_radius=None,
    percentile=None,
    use_sff=False,
    use_pld=True,
    use_gp=False,
    period=None,
    t0=None,
    t14=None,
    flatten=True,
    return_trend=True,
    verbose=False,
    clobber=False,
):
    """correct ffi tpf

    Parameters
    ----------
    tpf : lk.targetpixelfile
        ffi tpf
    sap_mask : str
        SAP mask type
    aper_radius : int
        aperture radius
    use_pld : bool
        use PLD for systematics correction
    use_gp : bool
        use GP
    verbose : bool
        print texts
    clobber : bool
        re-download files
    """
    # make aperture mask
    mask = parse_aperture_mask(
        tpf,
        sap_mask=sap_mask,
        aper_radius=aper_radius,
        percentile=percentile,
        verbose=verbose,
    )
    # tpf to lc
    raw_lc = tpf.to_lightcurve(aperture_mask=mask)
    raw_lc = raw_lc.remove_nans().remove_outliers().normalize()

    if verbose:
        print(f"ndata={len(raw_lc.time)}\n")
    # correct systematics/ filter long-term variability
    # see https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/correctors.py
    if np.any([use_pld,use_sff]):
        msg = "Applying systematics correction:\n"
        if use_pld:
            msg += "using PLD (gp={})".format(use_gp)
            if verbose:
                logging.info(msg)
                print(msg)
            if np.all([period, t0, t14]):
                cadence_mask_tpf = make_cadence_mask(
                    tpf.time, period, t0, t14, verbose=verbose
                )
            else:
                cadence_mask_tpf = None
            npix = tpf.pipeline_mask.sum()
            npix_lim = 24
            if sap_mask == "pipeline" and npix > npix_lim:
                # GP will create MemoryError so limit mask
                msg = f"More than {npix_lim} pixels (npix={npix}) are used in PLD\n"
                sap_mask, aper_radius = "square", 1
                mask = parse_aperture_mask(
                    tpf,
                    sap_mask=sap_mask,
                    aper_radius=aper_radius,
                    percentile=percentile,
                    verbose=verbose,
                )
                msg += f"Try changing to --aper_mask={sap_mask} (s={aper_radius}; npix={mask.sum()}) to avoid memory error.\n"
                if verbose:
                    logging.info(msg)
                    print(msg)
                raise ValueError(msg)
            # pld = tpf.to_corrector(method='pld')
            pld = lk.PLDCorrector(tpf)
            corr_lc = (
                pld.correct(
                    aperture_mask=mask,
                    use_gp=use_gp,
                    # True means cadence is considered in the noise model
                    cadence_mask=~cadence_mask_tpf if cadence_mask_tpf.sum()>0 else None,
                    # True means the pixel is chosen when selecting the PLD basis vectors
                    pld_aperture_mask=mask,
                    # gp_timescale=30, n_pca_terms=10, pld_order=2,
                )
                .remove_nans()
                .remove_outliers()
                .normalize()
            )
        else:
            # use_sff without restoring trend
            msg += "using SFF\n"
            if verbose:
                logging.info(msg)
                print(msg)
            sff = lk.SFFCorrector(raw_lc)
            corr_lc = (
                sff.correct(
                    centroid_col=raw_lc.centroid_col,
                    centroid_row=raw_lc.centroid_row,
                    polyorder=5,
                    niters=3,
                    bins=SFF_BINSIZE,
                    windows=SFF_CHUNKSIZE,
                    sigma_1=3.0,
                    sigma_2=5.0,
                    restore_trend=True,
                )
                .remove_nans()
                .remove_outliers()
            )
        # get transit mask of corr lc
        msg = "Flattening corrected light curve using Savitzky-Golay filter"
        if verbose:
            logging.info(msg)
            print(msg)
        if np.all([period, t0, t14]):
            cadence_mask_corr = make_cadence_mask(
                corr_lc.time, period, t0, t14, verbose=verbose
            )
        else:
            cadence_mask_corr = None

        # finally flatten
        t14_ncadences = t14*u.day.to(cadence_in_minutes)
        errmsg = f'use sg_filter_window> {t14_ncadences}'
        assert t14_ncadences<sg_filter_window_SC, errmsg
        flat_lc, trend = corr_lc.flatten(
            window_length=SG_FILTER_WINDOW_LC,
            mask=cadence_mask_corr,
            return_trend=True,
        )
    else:
        if verbose:
            msg = "Flattening raw light curve using Savitzky-Golay filter"
            logging.info(msg)
            print(msg)
        if np.all([period, t0, t14]):
            cadence_mask_raw = make_cadence_mask(
                raw_lc.time, period, t0, t14, verbose=verbose
            )
        else:
            cadence_mask_raw = None
        flat_lc, trend = raw_lc.flatten(
            window_length=SG_FILTER_WINDOW_LC,
            mask=cadence_mask_raw,
            return_trend=True,
        )
    # remove obvious outliers and NaN in time
    raw_time_mask = ~np.isnan(raw_lc.time)
    raw_flux_mask = (raw_lc.flux > YLIMIT[0]) & (raw_lc.flux < YLIMIT[1])
    raw_lc = raw_lc[raw_time_mask & raw_flux_mask]
    flat_time_mask = ~np.isnan(flat_lc.time)
    flat_flux_mask = (flat_lc.flux > YLIMIT[0]) & (flat_lc.flux < YLIMIT[1])
    flat_lc = flat_lc[flat_time_mask & flat_flux_mask]
    if np.any([use_pld,use_sff]):
        trend = trend[flat_time_mask & flat_flux_mask]
    else:
        trend = trend[raw_time_mask & raw_flux_mask]

    if np.any([use_pld,use_sff]):
        if flatten:
            return (flat_lc, trend) if return_trend else flat_lc
        else:
            return (corr_lc, trend) if return_trend else corr_lc
    else:
        return (raw_lc, trend) if return_trend else raw_lc


def plot_ffi_apers(tpf, percentiles=np.arange(40, 100, 10), figsize=(14, 6)):
    """
    """
    fig, axs = pl.subplots(2, 3, figsize=figsize)
    ax = axs.flatten()

    for n, perc in enumerate(percentiles):
        mask = parse_aperture_mask(tpf, sap_mask="percentile", percentile=perc)
        a = tpf.plot(
            ax=ax[n], aperture_mask=mask, origin="lower", show_colorbar=False
        )
        a.axis("off")
        a.set_title(perc)

    return fig


def get_pipeline_lc(
    target_coord=None,
    tic=None,
    sector=None,
    flux_type="pdcsap",
    verbose=False,
    clobber=False,
    fitsoutdir=".",
):
    """fetch pipeline generated (corrected) light curve

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    tic : int
        TIC id
    sector : int
        TESS sector
    flux_type : str
        PDCSAP or SAP
    fitsoutdir : str
        fits output directory
    verbose : bool
        print texts
    clobber : bool
        re-download files
    """
    if tic is not None:
        ticstr = "TIC {}".format(tic)
        if verbose:
            print("\nSearching mast for {}\n".format(ticstr))
        res = lk.search_lightcurvefile(ticstr, mission=MISSION, sector=sector)
    else:
        if verbose:
            print(
                "\nSearching mast for ra,dec=({})\n".format(
                    target_coord.to_string()
                )
            )
        res = lk.search_lightcurvefile(
            target_coord, mission=MISSION, sector=sector
        )
    df = res.table.to_pandas()

    if len(df) > 0:
        all_sectors = [int(i) for i in df["sequence_number"].values]
        if sector is not None:
            sector_idx = df["sequence_number"][
                df["sequence_number"].isin([sector])
            ].index.tolist()
            if len(sector_idx) == 0:
                raise ValueError(
                    "sector {} data is unavailable".format(sector)
                )
            obsid = df.iloc[sector_idx]["obs_id"].values[0]
            ticid = int(df.iloc[sector_idx]["target_name"].values[0])
            fitsfilename = df.iloc[sector_idx]["productFilename"].values[0]
        else:
            sector_idx = 0
            sector = int(df.iloc[sector_idx]["sequence_number"])
            obsid = df.iloc[sector_idx]["obs_id"]
            ticid = int(df.iloc[sector_idx]["target_name"])
            fitsfilename = df.iloc[sector_idx]["productFilename"]

        msg = "{} tpf(s) found in sector(s) {}\n".format(len(df), all_sectors)
        msg += "Using data from sector {} only\n".format(sector)
        if verbose:
            logging.info(msg)
            print(msg)

        filepath = join(fitsoutdir, "mastDownload/TESS", obsid, fitsfilename)
        if not exists(filepath) or clobber:
            if verbose:
                print("Downloading TIC {} ...\n".format(ticid))
            ticstr = "TIC {}".format(ticid)
            res = lk.search_lightcurvefile(
                ticstr, mission=MISSION, sector=sector
            )
            lc = res.download(
                quality_bitmask=quality_bitmask, download_dir=fitsoutdir
            )
        else:
            if verbose:
                print("Loading TIC {} from {}/...\n".format(ticid, fitsoutdir))
            lc = lk.TessLightCurveFile(filepath)

        if flux_type == "pdcsap":
            flux_type = "PDCSAP_FLUX"
        else:
            flux_type = "SAP_FLUX"
        lc = lc.get_lightcurve(flux_type=flux_type)
        return lc
    else:
        msg = "No light curve file found!\n"
        logging.info(msg)
        raise FileNotFoundError(msg)


def compare_custom_pipeline_lc(
    target_coord=None,
    tic=None,
    sector=None,
    flux_type="sap",
    verbose=False,
    clobber=False,
    fitsoutdir=".",
):
    """

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    tic : int
        TIC id
    sector : int
        TESS sector
    flux_type : str
        PDCSAP or SAP
    fitsoutdir : str
        fits output directory
    verbose : bool
        print texts
    clobber : bool
        re-download files
    """
    sap = get_pipeline_lc(
        target_coord=target_coord,
        tic=None,
        sector=None,
        flux_type="sap",
        verbose=False,
        clobber=False,
        fitsoutdir=".",
    )
    pdcsap = get_pipeline_lc(
        target_coord=target_coord,
        tic=None,
        sector=None,
        flux_type="pdcsap",
        verbose=False,
        clobber=False,
        fitsoutdir=".",
    )

    fig, ax = pl.subplots(3, 1, figsize=(10, 5))
    sap.errorbar(ax=ax[0])
    pdcsap.errorbar(ax=ax[1])
    return fig


def run_tls_on_pdcsap(
    target_coord=None,
    tic=None,
    sector=None,
    flux_type="pdcsap",
    verbose=False,
    fitsoutdir=".",
):
    """run tls periodogram on pdcsap lc

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    tic : int
        TIC id
    sector : int
        TESS sector
    flux_type : str
        PDCSAP or SAP
    fitsoutdir : str
        fits output directory
    verbose : bool
        print texts
    clobber : bool
        re-download files
    """
    lc = get_pipeline_lc(
        target_coord=None,
        tic=None,
        sector=None,
        flux_type="pdcsap",
        verbose=False,
        fitsoutdir=".",
    )
    # run tls
    t, fcor = lc.time, lc.flux
    model = transitleastsquares(t, fcor)
    try:
        ((u1, u2), Ms_tic, _, _, Rs_tic, _, _) = catalog.catalog_info(
            TIC_ID=int(tic)
        )
        Teff_tic, logg_tic, _, _, _, _, _, _ = catalog.catalog_info_TIC(
            int(tic)
        )
        u1, u2 = DEFAULT_U if not np.all([u1, u2]) else [u1, u2]
        Rs_tic = 1.0 if Rs_tic is None else Rs_tic
        Ms_tic = 1.0 if Ms_tic is None else Ms_tic
    except:
        (u1, u2), Ms_tic, Rs_tic = DEFAULT_U, 1.0, 1.0  # assume G2 star
    if verbose:
        if u1 == DEFAULT_U[0] and u2 == DEFAULT_U[1]:
            print("Using default limb-darkening coefficients\n")
        else:
            print(
                "Using u1={:.4f},u2={:.4f} based on TIC catalog\n".format(
                    u1, u2
                )
            )

    results = model.power(
        u=[u1, u2], limb_dark="quadratic", n_transits_min=N_TRANSITS_MIN
    )
    period = results.period
    t0 = results.T0
    t14 = results.duration

    return (period, t0, t14)


def generate_QL(
    target_coord,
    toi=None,
    tic=None,
    sector=None,  # cutout_size=10,
    use_pld=True,
    use_gp=False,
    use_sff=False,
    apphot_method="sap",
    sap_mask="pipeline",
    aper_radius=None,
    percentile=None,
    apply_data_quality_mask=True,
    cadence="short",
    fitsoutdir=".",
    figoutdir=".",
    savefig=True,
    clobber=False,
    verbose=True,
):
    """Create quick look light curve with archival image

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    tic : int
        TIC id
    use_pld : bool
        use PLD for systematics correction
    toi : float
        TOI id
    cadence : str
        short: 2-min or long: 30-min (FFI cutout)
    cutout_size : int
        FFI cutout size in pixels
    use_sff : bool
        use SFF for systematics correction
    sector : int
        TESS sector
    apply_data_quality_mask : bool
        apply data quality mask identified in data release notes
    apphot_method : str
        aperture photometry method
    sap_mask : str
        SAP mask type
    aper_radius : int
        aperture radius
    percentile : float
        flux percentile to make mask
    fitsoutdir : str
        fits output directory
    figoutdir : str
        figure output directory
    savefig : bool
        save figure
    clobber : bool
        re-download files
    verbose : bool
        print texts
    """
    start = time.time()
    plot_kwargs = PLOT_KWARGS_LC if cadence == "long" else PLOT_KWARGS_SC
    try:
        # download or load tpf
        if cadence == "short":
            tpf, df = get_tpf(
                target_coord=target_coord,
                tic=tic,
                sector=sector,
                verbose=verbose,
                clobber=clobber,
                sap_mask=sap_mask,
                apply_data_quality_mask=apply_data_quality_mask,
                fitsoutdir=fitsoutdir,
                return_df=True,
            )
            sg_filter_window = SG_FILTER_WINDOW_SC
            cadence_in_minutes = 2 * u.minute

        else:
            tpf, df = get_ffi_cutout(
                target_coord=target_coord,
                tic=tic,
                clobber=clobber,
                sector=sector,  # cutout_size=cutout_size,
                apply_data_quality_mask=apply_data_quality_mask,
                verbose=verbose,
                fitsoutdir=fitsoutdir,
                return_df=True,
            )
            sg_filter_window = SG_FILTER_WINDOW_LC
            cadence_in_minutes = 30 * u.minute
            assert (
                sap_mask != "pipeline"
            ), "--aper_mask=pipeline (default) is not available for FFI data."
        all_sectors = [int(i) for i in df["sequence_number"].values]
        if sector is None:
            sector = all_sectors[0]

        # check tpf size
        ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]
        diag = np.sqrt(nx ** 2 + ny ** 2)
        fov_rad = (0.6 * diag * TESS_pix_scale).to(u.arcmin)
        if fov_rad > 1 * u.deg:
            tpf = cutout_tpf(tpf)
            # redefine dimensions
            ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]
            diag = np.sqrt(nx ** 2 + ny ** 2)
            fov_rad = (0.6 * diag * TESS_pix_scale).to(u.arcmin)

        if tpf.targetid is None:
            if cadence == "short":
                ticid = df.iloc[0]["target_name"]
            else:
                if tic is not None:
                    ticid = int(str(df.iloc[0]["target_name"]).split()[-1])
                else:
                    ticid = None
        else:
            ticid = tpf.targetid
        msg = "#----------TIC {}----------#\n".format(ticid)
        if verbose:
            logging.info(msg)
            print(msg)

        # check if target is TOI from tess alerts
        try:
            q = get_toi(
                tic=ticid,
                toi=toi,
                clobber=clobber,
                outdir="../data/",
                verbose=verbose,
            )
        except Exception as e:
            print(e)
        try:
            period, t0, t14, depth, toiid = get_transit_params(
                toi=toi, tic=ticid, verbose=False
            )
        except Exception as e:
            print(e)
            period, t0, t14, depth, toiid = None, None, None, None, None

        if verbose:
            print("Generating QL figure...\n")

        # make aperture mask
        mask = parse_aperture_mask(
            tpf,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            percentile=percentile,
            verbose=verbose,
        )

        # make lc
        raw_lc = tpf.to_lightcurve(method=PHOTMETHOD, aperture_mask=mask)
        raw_lc = raw_lc.remove_nans().remove_outliers().normalize()
        if verbose:
            print("ndata={}\n".format(len(raw_lc.time)))
        # correct systematics/ filter long-term variability
        # see https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/correctors.py
        cadence_mask_tpf = make_cadence_mask(
            tpf.time, period, t0, t14, verbose=verbose
        )
        if np.any([use_pld,use_sff]):
            msg = "Applying systematics correction:\n"
            if use_pld:
                msg += "using PLD (gp={})\n".format(use_gp)
                if verbose:
                    logging.info(msg)
                    print(msg)

                npix = tpf.pipeline_mask.sum()
                npix_lim = 24
                if sap_mask == "pipeline" and npix > npix_lim:
                    # GP will create MemoryError so limit mask
                    msg = "More than {} pixels (npix={}) are used in PLD\n".format(
                        npix_lim, npix
                    )
                    sap_mask, aper_radius = "square", 1.0
                    mask = parse_aperture_mask(
                        tpf,
                        sap_mask=sap_mask,
                        aper_radius=aper_radius,
                        percentile=percentile,
                        verbose=verbose,
                    )
                    msg += "Try changing to {} mask (s={}; npix={}) to avoid memory error.\n".format(
                        aper_mask, aper_radius, mask.sum()
                    )
                    if verbose:
                        logging.info(msg)
                        print(msg)

                pld = lk.PLDCorrector(tpf)
                corr_lc = (
                    pld.correct(
                        aperture_mask=mask,
                        use_gp=use_gp,
                        # True means cadence is considered in the noise model
                        cadence_mask=~cadence_mask_tpf if cadence_mask_tpf.sum()>0 else None,
                        # True means the pixel is chosen when selecting the PLD basis vectors
                        pld_aperture_mask=mask,
                        # gp_timescale=30, n_pca_terms=10, pld_order=2,
                    )
                    .remove_nans()
                    .remove_outliers()
                    .normalize()
                )

            else:
                # use_sff without restoring trend
                msg += "using SFF\n"
                if verbose:
                    logging.info(msg)
                    print(msg)
                sff = lk.SFFCorrector(raw_lc)
                corr_lc = (
                    sff.correct(
                        centroid_col=raw_lc.centroid_col,
                        centroid_row=raw_lc.centroid_row,
                        polyorder=5,
                        niters=3,
                        bins=SFF_BINSIZE,
                        windows=SFF_CHUNKSIZE,
                        sigma_1=3.0,
                        sigma_2=5.0,
                        restore_trend=True,
                    )
                    .remove_nans()
                    .remove_outliers()
                )
            # get transit mask of corr lc
            msg = (
                "Flattening corrected light curve using Savitzky-Golay filter"
            )
            if verbose:
                logging.info(msg)
                print(msg)
            cadence_mask_corr = make_cadence_mask(
                corr_lc.time, period, t0, t14, verbose=False
            )
            # finally flatten
            t14_ncadences = t14*u.day.to(cadence_in_minutes)
            errmsg = f'use sg_filter_window> {t14_ncadences}'
            assert t14_ncadences<sg_filter_window, errmsg
            flat_lc, trend = corr_lc.flatten(
                window_length=sg_filter_window,
                mask=cadence_mask_corr,
                return_trend=True,
            )
        # elif apphot_method=='pdcsap':
        #    flat_lc = res2.get_lightcurve(flux_type='PDCSAP_FLUX').remove_nans().remove_outliers()
        #    if verbose:
        #        msg='Using PDCSAP light curve\n'
        #        print(msg)
        #        logging.info(msg)

        else:
            if verbose:
                msg = "Flattening raw light curve using Savitzky-Golay filter"
                logging.info(msg)
                print(msg)
            cadence_mask_raw = make_cadence_mask(
                raw_lc.time, period, t0, t14, verbose=verbose
            )
            flat_lc, trend = raw_lc.flatten(
                window_length=sg_filter_window,
                mask=cadence_mask_raw,
                return_trend=True,
            )
        # remove obvious outliers and NaN in time
        raw_time_mask = ~np.isnan(raw_lc.time)
        raw_flux_mask = (raw_lc.flux > YLIMIT[0]) & (raw_lc.flux < YLIMIT[1])
        raw_lc = raw_lc[raw_time_mask & raw_flux_mask]
        flat_time_mask = ~np.isnan(flat_lc.time)
        flat_flux_mask = (flat_lc.flux > YLIMIT[0]) & (
            flat_lc.flux < YLIMIT[1]
        )
        flat_lc = flat_lc[flat_time_mask & flat_flux_mask]
        # if apphot_method!='pdcsap':
        if np.any([use_pld,use_sff]):
            trend = trend[flat_time_mask & flat_flux_mask]
        else:
            trend = trend[raw_time_mask & raw_flux_mask]

        # periodogram; see also https://docs.lightkurve.org/tutorials/02-recover-a-planet.html
        # pg = corr_lc.to_periodogram(minimum_period=min_period,
        #                                maximum_period=max_period,
        #                                method=PGMETHOD,
        #                                oversample_factor=10)
        if verbose:
            print("Periodogram with TLS\n")
        t = flat_lc.time
        fcor = flat_lc.flux
        idx = np.isnan(t) | np.isnan(fcor)
        t = t[~idx]
        fcor = fcor[~idx]

        # TLS
        model = transitleastsquares(t, fcor)
        # get TIC catalog info: https://github.com/hippke/tls/blob/master/transitleastsquares/catalog.py
        # see defaults: https://github.com/hippke/tls/blob/master/transitleastsquares/tls_constants.py
        try:
            ((u1, u2), Ms_tic, _, _, Rs_tic, _, _) = catalog.catalog_info(
                TIC_ID=int(ticid)
            )
            Teff_tic, logg_tic, _, Rs_min_tic, Rs_max_tic, _, _, _ = catalog.catalog_info_TIC(
                int(ticid)
            )
            Rs_err_tic = np.sqrt(Rs_min_tic ** 2 + Rs_max_tic ** 2)
            u1, u2 = DEFAULT_U if not np.all([u1, u2]) else [u1, u2]
            Rs_tic = 1.0 if Rs_tic is None else Rs_tic
            Ms_tic = 1.0 if Ms_tic is None else Ms_tic
        except:
            (u1, u2), Ms_tic, Rs_tic = DEFAULT_U, 1.0, 1.0  # assume G2 star
            Rs_err_tic = 0.01
            Teff_tic, logg_tic = None, None
        if verbose:
            if u1 == DEFAULT_U[0] and u2 == DEFAULT_U[1]:
                print("Using default limb-darkening coefficients\n")
            else:
                print(
                    "Using u1={:.4f},u2={:.4f} based on TIC catalog\n".format(
                        u1, u2
                    )
                )

        results = model.power(
            u=[u1, u2],
            limb_dark="quadratic",
            # R_star  = Rs_tic,
            # M_star  = Ms_tic,
            # oversampling_factor=3,
            # duration_grid_step =1.1
            # transit_depth_min=ppm*10**-6,
            period_min=TLS_PERIOD_MIN,
            period_max=TLS_PERIOD_MAX,
            n_transits_min=N_TRANSITS_MIN,
        )
        results["u"] = [u1, u2]
        results["Rstar_tic"] = Rs_tic
        results["Mstar_tic"] = Ms_tic
        results["Teff_tic"] = Teff_tic

        if verbose:
            print(
                "Odd-Even transit mismatch: {:.2f} sigma\n".format(
                    results.odd_even_mismatch
                )
            )
            print(
                "Best period from periodogram: {:.4f} {}\n".format(
                    results.period, u.day
                )
            )
        # phase fold
        fold_lc = flat_lc.fold(period=results.period, t0=results.T0)
        fold_lc_2P = flat_lc.fold(period=results.period * 2, t0=results.T0)
        fold_lc_halfP = flat_lc.fold(
            period=results.period * 0.5, t0=results.T0
        )

        maskhdr = tpf.hdu[2].header
        tpfwcs = WCS(maskhdr)
        # ------------------------create figure-----------------------#
        # FIXME: line below is run again after here to define projection
        if verbose:
            print(
                "Querying {0} ({1:.2f} x {1:.2f}) archival image\n".format(
                    IMAGING_SURVEY, fov_rad
                )
            )
        ax, hdu = plot_finder_image(
            target_coord,
            fov_radius=fov_rad,
            survey=IMAGING_SURVEY,
            reticle=True,
        )
        pl.close()

        fig = pl.figure(figsize=(15, 15))
        ax0 = fig.add_subplot(321)
        ax1 = fig.add_subplot(322, projection=WCS(hdu.header))
        ax2 = fig.add_subplot(323)
        ax3 = fig.add_subplot(324)
        ax4 = fig.add_subplot(325)
        ax5 = fig.add_subplot(326)
        axs = [ax0, ax1, ax2, ax3, ax4, ax5]

        # ----------ax0: tpf plot----------
        i = 0
        ax1 = tpf.plot(aperture_mask=mask, frame=10, origin="lower", ax=axs[i])
        ax1.text(
            0.95,
            0.10,
            f"mask={sap_mask}",
            verticalalignment="top",
            horizontalalignment="right",
            transform=axs[i].transAxes,
            color="w",
            fontsize=12,
        )
        ax1.set_title(f"sector={sector}", fontsize=FONTSIZE)

        # centroid shift analysis
        if (cadence_mask_tpf is None) | (
            np.sum(cadence_mask_tpf) == 0
        ):  # non-TOIs
            # make cadence_mask based on TLS results
            cadence_mask_tpf = make_cadence_mask(
                tpf.time, results.period, results.T0
            )
        flux_intransit = np.nanmedian(tpf.flux[cadence_mask_tpf], axis=0)
        flux_outtransit = np.nanmedian(tpf.flux[~cadence_mask_tpf], axis=0)

        # centroid based on TIC coordinates (identical to Gaia coordinates)
        y, x = tpf.wcs.all_world2pix(np.c_[tpf.ra, tpf.dec], 1)[0]
        y = round(y)
        x = round(x)
        y += tpf.row + 0.5
        x += tpf.column + 0.5
        # centroid based on difference image centroid
        y2, x2 = get_2d_centroid(flux_outtransit - flux_intransit)
        y2 += tpf.row - 0.5
        x2 += tpf.column + 0.5
        # ax.matshow(flux_outtransit-flux_intransit, origin='lower')
        ax1.plot(x, y, "rx", ms=16, label="TIC")
        ax1.plot(x2, y2, "bx", ms=16, label="out-in")
        ax1.legend(title="centroid")
        axs[i].invert_xaxis()

        # ----------ax1: archival image with superposed aper mask ----------
        i = 1
        nax, hdu = plot_finder_image(
            target_coord,
            fov_radius=fov_rad,
            survey=IMAGING_SURVEY,
            reticle=True,
            ax=axs[i],
        )
        nwcs = WCS(hdu.header)
        mx, my = hdu.data.shape

        # plot TESS aperture
        contour = np.zeros((ny, nx))
        contour[np.where(mask)] = 1
        contour = np.lib.pad(contour, 1, PadWithZeros)
        highres = zoom(contour, 100, order=0, mode="nearest")
        extent = np.array([-1, nx, -1, ny])
        # superpose aperture mask
        cs1 = axs[i].contour(
            highres,
            levels=[0.5],
            extent=extent,
            origin="lower",
            colors="y",
            transform=nax.get_transform(tpfwcs),
        )

        # plot gaia sources
        gaia_sources = Catalogs.query_region(
            target_coord, radius=fov_rad, catalog="Gaia", version=2
        ).to_pandas()
        tic_sources = Catalogs.query_region(
            target_coord, radius=fov_rad, catalog="TIC"
        ).to_pandas()
        idx = tic_sources["ID"].astype(int).isin([ticid])
        if np.any(idx):
            gaia_id = tic_sources.loc[idx, "GAIA"].values[0]
            gaia_id = int(gaia_id) if str(gaia_id) != "nan" else None
        else:
            gaia_id = None

        for r, d in gaia_sources[["ra", "dec"]].values:
            pix = nwcs.all_world2pix(np.c_[r, d], 1)[0]
            nax.scatter(
                pix[0],
                pix[1],
                marker="s",
                s=100,
                edgecolor="r",
                facecolor="none",
            )
        pl.setp(nax, xlim=(0, mx), ylim=(0, my))
        nax.set_title(
            "{0} ({1:.2f}' x {1:.2f}')".format(IMAGING_SURVEY, fov_rad.value),
            fontsize=FONTSIZE,
        )
        # get gaia stellar params
        star = get_gaia_params_from_dr2(
            target_coord,
            tic=ticid,
            gaia_sources=gaia_sources,
            gaia_id=gaia_id,
            return_star=True,
            search_radius=fov_rad,
            verbose=verbose,
        )
        Rs_gaia, Teff_gaia = star["radius_val"], star["teff_val"]
        Rs_err_gaia = np.sqrt(
            star["radius_percentile_upper"] ** 2
            + star["radius_percentile_lower"] ** 2
        )

        # ----------ax2: lc plot----------
        i = 2
        ax2 = raw_lc.errorbar(label="raw lc", ax=axs[i], **plot_kwargs)
        # some weird outliers do not get clipped, so force ylim
        y1, y2 = ax2.get_ylim()
        if (y1 < YLIMIT[0]) & (y2 > YLIMIT[1]):
            axs[i].set_ylim(YLIMIT[0], YLIMIT[1])
        elif y1 < YLIMIT[0]:
            axs[i].set_ylim(YLIMIT[0], y2)
        elif y2 > YLIMIT[1]:
            axs[i].set_ylim(y1, YLIMIT[1])

        # plot trend in raw flux if no correction is applied in sap flux
        # no trend plot if pdcsap or pld or sff is used
        if (use_pld == use_sff == False) & (apphot_method == "sap"):
            trend.plot(
                color="r", linewidth=3, label="Savgol_filter", ax=axs[i]
            )
        text = "cdpp: {:.2f}".format(raw_lc.flatten().estimate_cdpp())
        axs[i].text(
            0.95,
            0.15,
            text,
            verticalalignment="top",
            horizontalalignment="right",
            transform=axs[i].transAxes,
            color="green",
            fontsize=15,
        )
        axs[i].legend(loc="upper left")

        # ----------ax3: long-term variability (+ systematics) correction----------
        i = 3
        # plot trend in corrected light curve
        if np.any([use_pld,use_sff]):
            # ----------ax3: systematics-corrected----------
            ax3 = corr_lc.errorbar(ax=axs[i], label="corr lc", **plot_kwargs)
            text = "PLD={}, SFF={}, gp={}, cdpp={:.2f}".format(
                use_pld, use_sff, use_gp, flat_lc.estimate_cdpp()
            )
            trend.plot(
                color="r", linewidth=3, label="Savgol_filter", ax=axs[i]
            )
            # axs[i].plot(t[cadence_mask_corr], f[cadence_mask_corr], '')
        else:
            # ----------ax3: long-term variability-corrected----------
            ax3 = flat_lc.errorbar(ax=axs[i], label="flat lc", **plot_kwargs)
            text = "PLD={} (gp={}), SFF={}, cdpp={:.2f}".format(
                use_pld, use_gp, use_sff, flat_lc.estimate_cdpp()
            )
        # plot detected transits in panel 4:
        if np.all([results.period, results.T0]):
            # period and t0 from TLS
            if use_pld:
                tns = get_tns(corr_lc.time, results.period, results.T0)
            else:
                tns = get_tns(flat_lc.time, results.period, results.T0)
            for t in tns:
                ax3.axvline(t, 0, 1, linestyle="--", color="k", alpha=0.5)
        axs[i].legend(loc="upper left")
        axs[i].text(
            0.95,
            0.15,
            text,
            verticalalignment="top",
            horizontalalignment="right",
            transform=axs[i].transAxes,
            color="green",
            fontsize=15,
        )

        # ----------ax4: periodogram----------
        i = 4
        # pg.plot(ax=axs[i], c='k', unit=u.day, view='Period', scale='log', label='periodogram')
        axs[i].axvline(results.period, alpha=0.4, lw=3)
        axs[i].set_xlim(np.min(results.periods), np.max(results.periods))
        # plot harmonics: period multiples
        for n in range(2, 10):
            axs[i].axvline(
                n * results.period, alpha=0.4, lw=1, linestyle="dashed"
            )
            axs[i].axvline(
                results.period / n, alpha=0.4, lw=1, linestyle="dashed"
            )
        axs[i].set_ylabel(r"SDE")
        axs[i].set_xlabel("Period [days]")
        axs[i].plot(
            results.periods,
            results.power,
            color="black",
            lw=0.5,
            label="TLS periodogram",
        )
        axs[i].set_xlim(0, max(results.periods))
        text = "Best period={:.2f} {}".format(results.period, u.day)
        # text = 'Best period={:.2f} {}'.format(period.value, period.unit)
        # axs[i].axvline(period.value, 0, 1, linestyle='--', color='k', linewidth=3)
        axs[i].text(
            0.95,
            0.85,
            text,
            verticalalignment="top",
            horizontalalignment="right",
            transform=axs[i].transAxes,
            color="green",
            fontsize=15,
        )
        axs[i].legend()

        # ----------ax5: phase folded lc----------
        i = 5
        fold_lc.scatter(ax=axs[i], color="k", alpha=0.1, label="unbinned")
        # ax[i].scatter(results.folded_phase-phase_offset,results.folded_y,
        #              color='k', marker='.', label='unbinned', alpha=0.1, zorder=2)
        fold_lc.bin(BINSIZE_SC).scatter(
            ax=axs[i], color="C1", label="binned (10-min)", **plot_kwargs
        )
        # lc folded at period multiples to check for EB
        flux_offset = (1 - results.depth) * 3
        # axs[i].plot(
        #     fold_lc_2P.bin(BINSIZE_SC).time,
        #     fold_lc_2P.bin(BINSIZE_SC).flux - flux_offset,
        #     "ks",
        #     label="2xPeriod",
        #     alpha=0.1,
        # )
        # axs[i].plot(
        #     fold_lc_halfP.bin(BINSIZE_SC).time,
        #     fold_lc_halfP.bin(BINSIZE_SC).flux - flux_offset * 2,
        #     "k^",
        #     label="0.5xPeriod",
        #     alpha=0.1,
        # )
        # TLS model
        axs[i].plot(
            results.model_folded_phase - 0.5,
            results.model_folded_model,
            color="red",
            label="TLS model",
        )
        rprs = results["rp_rs"]
        rprs_err = np.sqrt(np.nanstd(results["transit_depths"]))
        t14 = results.duration * u.day.to(u.hour)
        t0 = results["T0"]

        star_source = "TIC"
        rstar, teff = Rs_tic, Teff_tic
        rstar_err = Rs_err_tic
        if str(rstar) == "nan":
            star_source = "Gaia DR2"
            rstar, teff = Rs_gaia, Teff_gaia
            rstar_err = Rs_err_gaia
        # Rp = rprs*rstar*u.Rsun.to(u.Rearth)
        Rp, Rpsighi, Rpsiglo = get_Rp_monte_carlo(
            RpRs=(rprs, rprs_err), Rs=(rstar, rstar_err), verbose=False
        )
        Rp_err = np.sqrt(Rpsighi ** 2 + Rpsiglo ** 2)
        text1 = "Rp/Rs={:.4f}\nt14={:.2f} hr\nt0={:.6f}".format(rprs, t14, t0)
        text2 = "Source: {}\nRs={:.2f}+/-{:.2f} Rsun\nTeff={:.0f} K\nRp={:.2f}+/-{:.2f} Re".format(
            star_source, rstar, rstar_err, teff, Rp, Rp_err
        )
        if verbose:
            print(f"{text1}\n\n{text2}")
        axs[i].text(
            0.3,
            0.25,
            text1,
            verticalalignment="top",
            horizontalalignment="left",
            transform=axs[i].transAxes,
            color="g",
            fontsize=FONTSIZE,
        )
        axs[i].text(
            0.6,
            0.3,
            text2,
            verticalalignment="top",
            horizontalalignment="left",
            transform=axs[i].transAxes,
            color="g",
            fontsize=FONTSIZE,
        )
        axs[i].set_xlim(-0.1, 0.1)
        axs[i].legend(title="phase-folded lc")
        axs[i].legend(loc=3)

        # manually set ylimit for shallow transits
        if rprs <= 0.1:
            ylo, yhi = 1 - 15 * rprs ** 2, 1 + 5 * rprs ** 2
            axs[i].set_ylim(ylo, yhi if yhi < 1.02 else 1.02)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        end = time.time()

        cad = "sc" if cadence == "short" else "lc"
        # save tls results
        if results:
            results["tic"] = tpf.targetid
            results["sector"] = sector
            results["Rs_gaia"] = Rs_gaia
            results["Teff_gaia"] = Teff_gaia
            results["Rp"] = Rp
            fp = join(figoutdir, f"tic{tpf.targetid}_s{sector}_{cad}_tls.hdf5")
            if savefig:
                dd.io.save(fp, results)
                print(f"Saved: {fp}\n")

        if toi or toiid:
            # toiid is TOIXXX determined from TESS release queried using TIC or coordinates
            id = toi if toi is not None else toiid
            figname = f"TIC{tpf.targetid}_TOI{str(id)}_s{sector}_{cad}.png"
            lcname1 = (
                f"TIC{tpf.targetid}_TOI{str(id)}_s{sector}_lc_flat_{cad}.txt"
            )
            lcname2 = (
                f"TIC{tpf.targetid}_TOI{str(id)}_s{sector}_lc_fold_{cad}.txt"
            )
            pl.suptitle(f"TIC {ticid} (TOI {id})", fontsize=FONTSIZE)
        else:
            figname = f"TIC{tpf.targetid}_s{sector}_{cad}.png"
            lcname1 = f"TIC{tpf.targetid}_s{sector}_lc_flat_{cad}.txt"
            lcname2 = f"TIC{tpf.targetid}_s{sector}_lc_fold_{cad}.txt"
            pl.suptitle(f"TIC {ticid}", fontsize=FONTSIZE)
        figoutpath = join(figoutdir, figname)
        lcoutpath1 = join(figoutdir, lcname1)
        lcoutpath2 = join(figoutdir, lcname2)

        if savefig:
            fig.savefig(figoutpath, bbox_inches="tight")
            print(f"Saved: {figoutpath}\n")
            # np.savetxt(lcoutpath, np.c_[t,f], fmt='%.8f')
            flat_lc.to_pandas().to_csv(lcoutpath1, index=False, sep=" ")
            fold_lc.to_pandas().to_csv(lcoutpath2, index=False, sep=" ")
            print(f"Saved:\n{lcoutpath1}\n{lcoutpath2}")
        else:
            pl.show()
        msg = "#----------Runtime: {:.2f} s----------#\n".format(end - start)
        if verbose:
            logging.info(msg)
            print(msg)
        pl.close()
    except:
        print(f"Error occured:\n{traceback.format_exc()}")
        print_recommendations()
        # print('tpf size=(x,x) pix seems too big\n\n')
        logging.error(str(traceback.format_exc()))
    # save logs
    logfile = open(LOG_FILENAME, "rt")
    try:
        body = logfile.read()
    finally:
        logfile.close()


def generate_all_lc(
    target_coord,
    toi=None,
    tic=None,
    use_pld=False,
    use_gp=True,
    use_sff=False,
    apphot_method="sap",
    sap_mask="pipeline",
    aper_radius=None,
    percentile=None,
    sectors=None,
    apply_data_quality_mask=True,
    fitsoutdir=".",
    figoutdir=".",
    savefig=True,
    clobber=False,
    verbose=True,
):
    """Create multi-sector light curves

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    tic : int
        TIC id
    use_pld : bool
        use PLD for systematics correction
    toi : float
        TOI id
    use_sff : bool
        use SFF for systematics correction
    sectors : int
        TESS sectors
    apply_data_quality_mask : bool

    apphot_method : str
        aperture photometry method
    sap_mask : str
        SAP mask type
    aper_radius : int
        aperture radius
    fitsoutdir : str
        fits output directory
    figoutdir : str
        figure output directory
    savefig : bool
        save figure
    clobber : bool
        re-download files
    """
    # try:
    #     from brokenaxes import brokenaxes
    # except:
    #     sys.exit('pip install git+https://github.com/bendichter/brokenaxes')

    start = time.time()
    plot_kwargs = PLOT_KWARGS_LC if cadence == "long" else PLOT_KWARGS_SC
    try:
        if cadence == "short":
            tpf, df = get_tpf(
                target_coord=target_coord,
                tic=tic,
                sector=None,
                verbose=verbose,
                clobber=clobber,
                sap_mask=sap_mask,
                apply_data_quality_mask=apply_data_quality_mask,
                fitsoutdir=fitsoutdir,
                return_df=True,
            )
            sg_filter_window = SG_FILTER_WINDOW_SC
        else:
            tpf, df = get_ffi_cutout(
                target_coord=target_coord,
                tic=tic,
                clobber=clobber,
                sector=None,  # cutout_size=cutout_size,
                apply_data_quality_mask=apply_data_quality_mask,
                verbose=verbose,
                fitsoutdir=fitsoutdir,
                return_df=True,
            )
            sg_filter_window = SG_FILTER_WINDOW_LC
            assert (
                sap_mask != "pipeline"
            ), "--aper_mask=pipeline (default) is not available for FFI data."
        all_sectors = [int(i) for i in df["sequence_number"].values]

        if verbose:
            print(
                f"\nSearching mast for ra,dec=({target_coord.to_string()})\n"
            )

        all_sectors = [int(i) for i in df["sequence_number"].values]
        msg = "{} tpf(s) found in sector(s) {}\n".format(len(df), all_sectors)

        ticid = df.iloc[0]["target_name"]
        if len(df) == 1:
            msg = "Target is observed in just 1 sector. Re-run script without -a argument.\n"
            logging.info(msg)
            print(msg)
        elif len(df) > 1:
            msg = f"#----------TIC {ticid}----------#\n"
            logging.info(msg)

            # check if target is TOI from TESS alerts
            ticid = int(df.iloc[0]["target_name"])
            try:
                q = get_toi(
                    tic=ticid,
                    toi=toi,
                    clobber=clobber,
                    outdir="../data/",
                    verbose=verbose,
                )
            except Exception as e:
                print(e)
            if len(df) > 1:
                # if tic is observed in multiple sectors
                if sectors is not None:
                    # use only specified sectors
                    sector_idx = df["sequence_number"][
                        df["sequence_number"].isin(sectors)
                    ].index.tolist()
                    # reduce all sectors to specified sectors
                    all_sectors = np.array(all_sectors)[sector_idx]
                    if verbose:
                        print(f"Analyzing data from sectors: {sectors}\n")
                else:
                    # use all sectors
                    sector_idx = range(len(df))
                    if verbose:
                        print(f"Downloading all sectors: {all_sectors}\n")
            else:
                # if tic is observed only once, take first available sector
                sector_idx = [0]

            msg += f"Using data from sectors {all_sectors}\n"
            if verbose:
                logging.info(msg)
                print(msg)

            try:
                period, t0, t14, depth, toiid = get_transit_params(
                    toi=toi, tic=ticid, verbose=False
                )
            except Exception as e:
                print(e)
                period, t0, t14, depth, toiid = None, None, None, None, None

            tpfs = []
            masks = []
            # sectors = []
            for n in tqdm(sector_idx):
                # load or download per sector
                obsid = df.iloc[n]["obs_id"]
                ticid = int(df.iloc[n]["target_name"])
                sector = int(
                    df.iloc[n]["sequence_number"]
                )  # ==obsid.split('-')[1][-1]
                fitsfilename = df.iloc[n]["productFilename"]

                filepath = join(
                    fitsoutdir, "mastDownload/TESS", obsid, fitsfilename
                )
                if not exists(filepath) or clobber:
                    if verbose:
                        print(
                            f"Downloading TIC {ticid} (sector {sector})...\n"
                        )
                    # re-search tpf with specified sector
                    # FIXME: related issue: https://github.com/KeplerGO/lightkurve/issues/533
                    ticstr = f"TIC {ticid}"
                    tpf = lk.search_targetpixelfile(
                        ticstr, mission=MISSION, sector=sector
                    ).download(
                        quality_bitmask=quality_bitmask,
                        download_dir=fitsoutdir,
                    )
                else:
                    if verbose:
                        print(
                            f"Loading TIC {ticid} (sector {sector}) from {fitsoutdir}/...\n"
                        )
                    tpf = lk.TessTargetPixelFile(filepath)
                assert tpf.mission == MISSION

                # check tpf size
                ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]
                diag = np.sqrt(nx ** 2 + ny ** 2)
                fov_rad = (0.6 * diag * TESS_pix_scale).to(u.arcmin)
                if fov_rad > 1 * u.deg:
                    tpf = cutout_tpf(tpf)
                    # redefine dimensions
                    ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]
                    diag = np.sqrt(nx ** 2 + ny ** 2)
                    fov_rad = (0.6 * diag * TESS_pix_scale).to(u.arcmin)

                # remove bad data identified in data release notes
                if apply_data_quality_mask:
                    tpf = remove_bad_data(tpf, sector, verbose=verbose)

                # make aperture mask
                mask = parse_aperture_mask(
                    tpf,
                    sap_mask=sap_mask,
                    verbose=verbose,
                    aper_radius=aper_radius,
                    percentile=percentile,
                )

                tpfs.append(tpf)
                masks.append(mask)
                # sectors.append(sector)

            # ------------------------create figure-----------------------#
            # FIXME: line below is run again after here to define projection
            if verbose:
                print(
                    "Querying {0} ({1:.2f} x {1:.2f}) archival image\n".format(
                        IMAGING_SURVEY, fov_rad
                    )
                )
            ax, hdu = plot_finder_image(
                target_coord,
                fov_radius=fov_rad,
                survey=IMAGING_SURVEY,
                reticle=True,
            )
            pl.close()

            fig = pl.figure(figsize=(15, 15))
            ax0 = fig.add_subplot(321)
            ax1 = fig.add_subplot(322, projection=WCS(hdu.header))
            ax2 = fig.add_subplot(323)
            ax3 = fig.add_subplot(324)
            ax4 = fig.add_subplot(325)
            ax5 = fig.add_subplot(326)
            ax = [ax0, ax1, ax2, ax3, ax4, ax5]

            # ----------ax0: tpf plot----------
            # plot only the first tpf
            i = 0
            tpf = tpfs[i]
            mask = masks[i]
            ax1 = tpf.plot(
                aperture_mask=mask, frame=10, origin="lower", ax=ax[i]
            )
            ax1.text(
                0.95,
                0.10,
                f"mask={sap_mask}",
                verticalalignment="top",
                horizontalalignment="right",
                transform=ax[i].transAxes,
                color="w",
                fontsize=12,
            )
            ax1.set_title(f"sector={all_sectors[i]}", fontsize=FONTSIZE)

            cadence_mask_tpf = make_cadence_mask(tpf.time, period, t0, t14)
            # centroid shift analysis
            if (cadence_mask_tpf is not None) | (
                np.sum(cadence_mask_tpf) != 0
            ):
                # Does not work for non-TOIs because period is not given a priori
                flux_intransit = np.nanmedian(
                    tpf.flux[cadence_mask_tpf], axis=0
                )
                flux_outtransit = np.nanmedian(
                    tpf.flux[~cadence_mask_tpf if cadence_mask_tpf.sum()>0 else None], axis=0
                )

                # centroid based on TIC coordinates (identical to Gaia coordinates)
                y, x = tpf.wcs.all_world2pix(np.c_[tpf.ra, tpf.dec], 1)[0]
                y = round(y)
                x = round(x)
                y += tpf.row + 0.5
                x += tpf.column + 0.5
                # centroid based on difference image centroid
                y2, x2 = get_2d_centroid(flux_outtransit - flux_intransit)
                y2 += tpf.row + 0.5
                x2 += tpf.column + 0.5
                # ax.matshow(flux_outtransit-flux_intransit, origin='lower')
                ax1.plot(x, y, "rx", ms=16, label="TIC")
                ax1.plot(x2, y2, "bx", ms=16, label="out-in")
                ax1.legend(title="centroid")
            ax[i].invert_xaxis()

            # ----------ax1: archival image with superposed aper mask ----------
            i = 1
            # if verbose:
            #     print('Querying {0} ({1} x {1}) archival image'.format(IMAGING_SURVEY,fov_rad))
            nax, hdu = plot_finder_image(
                target_coord,
                fov_radius=fov_rad,
                survey=IMAGING_SURVEY,
                reticle=True,
                ax=ax[i],
            )
            nwcs = WCS(hdu.header)
            mx, my = hdu.data.shape

            # plot gaia sources
            gaia_sources = Catalogs.query_region(
                target_coord, radius=fov_rad, catalog="Gaia", version=2
            ).to_pandas()
            for r, d in gaia_sources[["ra", "dec"]].values:
                pix = nwcs.all_world2pix(np.c_[r, d], 1)[0]
                nax.scatter(
                    pix[0],
                    pix[1],
                    marker="s",
                    s=100,
                    edgecolor="r",
                    facecolor="none",
                )
            pl.setp(nax, xlim=(0, mx), ylim=(0, my))
            nax.set_title(
                "{0} ({1:.2f}' x {1:.2f}')".format(
                    IMAGING_SURVEY, fov_rad.value
                ),
                fontsize=FONTSIZE,
            )

            # get gaia stellar params
            tic_sources = Catalogs.query_region(
                target_coord, radius=fov_rad, catalog="TIC"  # version=8
            ).to_pandas()
            idx = tic_sources["ID"].astype(int).isin([ticid])
            if np.any(idx):
                gaia_id = tic_sources.loc[idx, "GAIA"].values[0]
                gaia_id = int(gaia_id) if str(gaia_id) != "nan" else None
            else:
                gaia_id = None

            star = get_gaia_params_from_dr2(
                target_coord,
                tic=ticid,
                gaia_sources=gaia_sources,
                gaia_id=gaia_id,
                return_star=True,
                search_radius=fov_rad,
                verbose=verbose,
            )
            Rs_gaia, Teff_gaia = star["radius_val"], star["teff_val"]

            lcs = []
            corr_lcs = []
            flat_lcs = []
            times = []
            fluxes = []
            flux_errs = []
            trends = []
            cdpps_raw = []
            cdpps_corr = []
            for j, n in tqdm(enumerate(sector_idx)):
                print(f"\n----------sector {all_sectors[j]}----------\n")
                tpf = tpfs[j]
                mask = masks[j]
                maskhdr = tpf.hdu[2].header
                tpfwcs = WCS(maskhdr)

                # plot TESS aperture
                contour = np.zeros((ny, nx))
                contour[np.where(mask)] = 1
                contour = np.lib.pad(contour, 1, PadWithZeros)
                highres = zoom(contour, 100, order=0, mode="nearest")
                extent = np.array([-1, nx, -1, ny])
                # superpose aperture mask
                cs1 = ax[1].contour(
                    highres,
                    levels=[0.5],
                    extent=extent,
                    origin="lower",
                    colors=COLORS[j],
                    transform=nax.get_transform(tpfwcs),
                )
                # make lc
                raw_lc = tpf.to_lightcurve(
                    method=PHOTMETHOD, aperture_mask=masks[j]
                )
                raw_lc = raw_lc.remove_nans().remove_outliers().normalize()

                # remove obvious outliers and NaN in time
                time_mask = ~np.isnan(raw_lc.time)
                flux_mask = (raw_lc.flux > YLIMIT[0]) | (
                    raw_lc.flux < YLIMIT[1]
                )
                raw_lc = raw_lc[time_mask & flux_mask]

                lcs.append(raw_lc)
                cdpps_raw.append(raw_lc.flatten().estimate_cdpp())

                cadence_mask_tpf = make_cadence_mask(
                    tpf.time, period, t0, t14, verbose=verbose
                )

                # correct systematics/ filter long-term variability
                if np.any([use_pld,use_sff]):
                    msg = "Applying systematics correction:\n"
                    if use_pld:
                        msg += f"using PLD (gp={use_gp})"
                        if verbose:
                            logging.info(msg)
                            print(msg)
                        # pld = tpf.to_corrector(method='pld')
                        pld = lk.PLDCorrector(tpf)
                        corr_lc = (
                            pld.correct(
                                aperture_mask=mask,
                                use_gp=use_gp,
                                # True means cadence is considered in the noise model
                                cadence_mask=~cadence_mask_tpf if cadence_mask_tpf.sum()>0 else None,
                                # True means the pixel is chosen when selecting the PLD basis vectors
                                pld_aperture_mask=mask,
                                # gp_timescale=30, n_pca_terms=10, pld_order=2,
                            )
                            .remove_nans()
                            .remove_outliers()
                            .normalize()
                        )
                    else:
                        # use_sff without restoring trend
                        msg += "using SFF\n"
                        if verbose:
                            logging.info(msg)
                            print(msg)
                        sff = lk.SFFCorrector(raw_lc)
                        corr_lc = (
                            sff.correct(
                                centroid_col=raw_lc.centroid_col,
                                centroid_row=raw_lc.centroid_row,
                                polyorder=5,
                                niters=3,
                                bins=SFF_BINSIZE,
                                windows=SFF_CHUNKSIZE,
                                sigma_1=3.0,
                                sigma_2=5.0,
                                restore_trend=True,
                            )
                            .remove_nans()
                            .remove_outliers()
                        )
                    corr_lcs.append(corr_lc)

                    # get transit mask of corr lc
                    msg = "Flattening corrected light curve using Savitzky-Golay filter"
                    if verbose:
                        logging.info(msg)
                        print(msg)
                    cadence_mask_corr = make_cadence_mask(
                        corr_lc.time, period, t0, t14, verbose=False
                    )
                    # finally flatten
                    t14_ncadences = t14*u.day.to(cadence_in_minutes)
                    errmsg = f'use sg_filter_window> {t14_ncadences}'
                    assert t14_ncadences<sg_filter_window_SC, errmsg
                    flat_lc, trend = corr_lc.flatten(
                        window_length=SG_FILTER_WINDOW_SC,
                        mask=cadence_mask_corr,
                        return_trend=True,
                    )
                else:
                    if verbose:
                        msg = "Flattening raw light curve using Savitzky-Golay filter"
                        logging.info(msg)
                        print(msg)
                    cadence_mask_raw = make_cadence_mask(
                        raw_lc.time, period, t0, t14, verbose=verbose
                    )
                    flat_lc, trend = raw_lc.flatten(
                        window_length=SG_FILTER_WINDOW_SC,
                        mask=cadence_mask_raw,
                        return_trend=True,
                    )
                import pdb; pdb.set_trace()
                # remove obvious outliers and NaN in time
                raw_time_mask = ~np.isnan(raw_lc.time)
                raw_flux_mask = (raw_lc.flux > YLIMIT[0]) | (
                    raw_lc.flux < YLIMIT[1]
                )
                raw_lc = raw_lc[raw_time_mask & raw_flux_mask]
                flat_time_mask = ~np.isnan(flat_lc.time)
                flat_flux_mask = (flat_lc.flux > YLIMIT[0]) & (
                    flat_lc.flux < YLIMIT[1]
                )
                flat_lc = flat_lc[flat_time_mask & flat_flux_mask]
                # if apphot_method!='pdcsap':
                if np.any([use_pld,use_sff]):
                    trend = trend[flat_time_mask & flat_flux_mask]
                else:
                    trend = trend[raw_time_mask & raw_flux_mask]

                # append with trend
                flat_lcs.append(flat_lc)
                times.append(flat_lc.time)
                fluxes.append(flat_lc.flux)
                flux_errs.append(flat_lc.flux_err)
                trends.append(trend.flux)
                # sectors.append(str(sector))
                cdpps_corr.append(flat_lc.estimate_cdpp())

            if verbose:
                print("Periodogram with TLS\n")
            t = np.array(list(itertools.chain(*times)))
            # idx = np.argsort(t)
            # t = sorted(t)
            f = np.array(list(itertools.chain(*fluxes)))
            e = np.array(list(itertools.chain(*flux_errs)))
            tr = np.array(list(itertools.chain(*trends)))

            if len(all_sectors) >= MAX_SECTORS:
                if CADENCE == "short":
                    cadence_in_minutes = (
                        2 * u.minute
                    )  # np.median(np.diff(t))*u.day.to(u.second)
                else:
                    cadence_in_minutes = 30 * u.minute
                # count binsize given old and new cadences
                msg = f"Number of sectors exceeds {MAX_SECTORS} (ndata={len(t)}).\n"
                binsize = int(MULTISEC_BIN.to(cadence_in_minutes).value)
                t = binned(t, binsize=binsize)
                f = binned(f, binsize=binsize)
                e = binned(e, binsize=binsize)
                tr = binned(tr, binsize=binsize)
                msg += f"Data was binned to {MULTISEC_BIN} (ndata={len(t)}).\n"
                if verbose:
                    logging.info(msg)
                    print(msg)

            # concatenate time series into one light curve
            full_lc = lk.TessLightCurve(
                time=t,
                flux=f,
                flux_err=e,
                time_format=time_format,
                time_scale=time_scale,
            )

            # run TLS with default parameters
            model = transitleastsquares(t, f)
            try:
                ((u1, u2), Ms_tic, _, _, Rs_tic, _, _) = catalog.catalog_info(
                    TIC_ID=int(ticid)
                )
                Teff_tic, logg_tic, _, _, _, _, _, _ = catalog.catalog_info_TIC(
                    int(ticid)
                )
                u1, u2 = DEFAULT_U if not np.all([u1, u2]) else [u1, u2]
                Rs_tic = 1.0 if Rs_tic is None else Rs_tic
                Ms_tic = 1.0 if Ms_tic is None else Ms_tic
                star_source = "TIC"
            except:
                (u1, u2), Ms_tic, Rs_tic = (
                    DEFAULT_U,
                    1.0,
                    1.0,
                )  # assume G2 star
                star_source = "default"
            if verbose:
                if u1 == DEFAULT_U[0] and u2 == DEFAULT_U[1]:
                    print("Using default limb-darkening coefficients\n")
                else:
                    print(
                        "Using u1={:.4f},u2={:.4f} based on TIC catalog\n".format(
                            u1, u2
                        )
                    )

            # FIXME: limit period allowing single transits for each sector
            results = model.power(
                u=[u1, u2],
                limb_dark="quadratic",
                # R_star  = Rs_tic,
                # M_star  = Ms_tic,
                # oversampling_factor=3,
                # duration_grid_step =1.1
                # transit_depth_min=ppm*10**-6,
                period_min=TLS_PERIOD_MIN,
                period_max=TLS_PERIOD_MAX,
                n_transits_min=N_TRANSITS_MIN,
            )
            results["u"] = [u1, u2]
            results["Rstar_tic"] = Rs_tic
            results["Mstar_tic"] = Ms_tic
            results["Teff_tic"] = Teff_tic

            if verbose:
                # FIXME: compare period from TESS alerts
                msg = "Odd-Even transit mismatch: {:.2f} sigma\n".format(
                    results.odd_even_mismatch
                )
                msg += "Best period from periodogram: {:.4f} {}\n".format(
                    results.period, u.day
                )
                logging.info(msg)
                print(msg)

            # phase fold
            fold_lc = flat_lc.fold(period=results.period, t0=results.T0)
            fold_lc_2P = flat_lc.fold(period=results.period * 2, t0=results.T0)
            fold_lc_halfP = flat_lc.fold(
                period=results.period * 0.5, t0=results.T0
            )

            # ----------ax2: raw lc plot----------
            i = 2
            # ax2 = full_lc.errorbar(label='raw lc',ax=ax[i]) <-- plot with single color
            for lc, col, sec in zip(lcs, COLORS, all_sectors):
                cdpp = lc.flatten().estimate_cdpp()
                ax2 = lc.errorbar(
                    color=col,
                    label="s{}: {:.2f}".format(sec, cdpp),
                    ax=ax[i],
                    **plot_kwargs,
                )
            # some weird outliers do not get clipped, so force ylim
            y1, y2 = ax[i].get_ylim()
            if (y1 < YLIMIT[0]) & (y2 > YLIMIT[1]):
                ax[i].set_ylim(YLIMIT[0], YLIMIT[1])
            elif y1 < YLIMIT[0]:
                ax[i].set_ylim(YLIMIT[0], y2)
            elif y2 > YLIMIT[1]:
                ax[i].set_ylim(y1, YLIMIT[1])

            if use_pld == use_sff == False:
                ax[i].plot(
                    t, tr, color="r", linewidth=3, label="Savgol_filter"
                )

            # text = ['s{}: {:.2f}'.format(sector,cdpp) for sector,cdpp in zip(sectors,cdpps_raw)]
            ax[i].legend(title="raw lc cdpp", loc="upper left")

            # ----------ax3: long-term variability (+ systematics) correction----------
            i = 3
            colors = COLORS[: len(all_sectors)]
            if np.any([use_pld,use_sff]):
                # ----------ax3: systematics-corrected----------
                for corr_lc, col, sec in zip(corr_lcs, colors, all_sectors):
                    cdpp = corr_lc.flatten().estimate_cdpp()
                    ax3 = corr_lc.errorbar(
                        color=col,
                        ax=ax[i],
                        label="s{}: {:.2f}".format(sec, cdpp),
                        **plot_kwargs,
                    )
                ax[i].legend(title="corr lc cdpp", loc="upper left")
                ax[i].plot(t, tr, color="r", label="Savgol_filter")
            else:
                # ax1 long-term variability-corrected
                for flat_lc, col, sec in zip(flat_lcs, colors, all_sectors):
                    cdpp = flat_lc.estimate_cdpp()
                    ax3 = flat_lc.errorbar(
                        color=col,
                        ax=ax[i],
                        label="s{}: {:.2f}".format(sec, cdpp),
                        **plot_kwargs,
                    )
                ax[i].legend(title="flat lc cdpp", loc="upper left")

            if np.all([results.period, results.T0]):
                try:
                    # index error due to either per or t0
                    tns = get_tns(t, results.period, results.T0)
                    for tt in tns:
                        ax[i].axvline(
                            tt, 0, 1, linestyle="--", color="k", alpha=0.5
                        )
                except:
                    pass
            text = f"PLD={use_pld} (gp={use_gp}), SFF={use_sff}"
            ax[i].text(
                0.95,
                0.15,
                text,
                verticalalignment="top",
                horizontalalignment="right",
                transform=ax[i].transAxes,
                color="green",
                fontsize=15,
            )

            # ----------ax4: periodogram----------
            i = 4
            # pg.plot(ax=axs[i], c='k', unit=u.day, view='Period', scale='log', label='periodogram')
            ax[i].axvline(results.period, alpha=0.4, lw=3)
            # plot harmonics: period multiples
            for n in range(2, 10):
                ax[i].axvline(
                    n * results.period, alpha=0.4, lw=1, linestyle="dashed"
                )
                ax[i].axvline(
                    results.period / n, alpha=0.4, lw=1, linestyle="dashed"
                )
            ax[i].set_ylabel(r"SDE")
            ax[i].set_xlabel("Period [days]")
            ax[i].plot(
                results.periods,
                results.power,
                color="black",
                lw=0.5,
                label="TLS periodogram",
            )
            ax[i].set_xlim(np.min(results.periods), np.max(results.periods))

            # text = 'Best period={:.2f} {}'.format(period.value, period.unit)
            text = "Best period={:.2f} {}".format(results.period, u.day)
            # axs[i].axvline(period.value, 0, 1, linestyle='--', color='k', linewidth=3)
            ax[i].text(
                0.95,
                0.75,
                text,
                verticalalignment="top",
                horizontalalignment="right",
                transform=ax[i].transAxes,
                color="green",
                fontsize=15,
            )
            ax[i].legend()

            # ----------ax5: phase folded lc----------
            i = 5
            phase_offset = 0.5
            ax[i].plot(
                results.model_folded_phase - phase_offset,
                results.model_folded_model,
                color="red",
                label="TLS model",
            )
            fold_lc.bin(BINSIZE_SC).scatter(
                ax=ax[i], color="C1", label="binned (10-min)", **plot_kwargs
            )
            fold_lc.scatter(ax=ax[i], color="k", alpha=0.1, label="unbinned")

            flux_offset = (1 - results.depth) * 3
            # ax[i].plot(
            #     fold_lc_2P.bin(BINSIZE_SC).time,
            #     fold_lc_2P.bin(BINSIZE_SC).flux - flux_offset,
            #     "ks",
            #     label="2xPeriod",
            #     alpha=0.1,
            # )
            # ax[i].plot(
            #     fold_lc_halfP.bin(BINSIZE_SC).time,
            #     fold_lc_halfP.bin(BINSIZE_SC).flux - flux_offset * 2,
            #     "k^",
            #     label="0.5xPeriod",
            #     alpha=0.1,
            # )
            # ax[i].legend(loc=3)

            rprs = results["rp_rs"]
            t14 = results.duration * u.day.to(u.hour)
            t0 = results["T0"]

            # get gaia stellar params
            gaia_sources = Catalogs.query_region(
                target_coord, radius=fov_rad, catalog="Gaia", version=2
            ).to_pandas()
            tic_sources = Catalogs.query_region(
                target_coord, radius=fov_rad, catalog="TIC"  # version=8
            ).to_pandas()
            idx = tic_sources["ID"].astype(int).isin([ticid])
            if np.any(idx):
                gaia_id = tic_sources.loc[idx, "GAIA"].values[0]
                gaia_id = int(gaia_id) if str(gaia_id) != "nan" else None
            else:
                gaia_id = None

            star = get_gaia_params_from_dr2(
                target_coord,
                tic=ticid,
                gaia_sources=gaia_sources,
                gaia_id=gaia_id,
                return_star=True,
                search_radius=fov_rad,
                verbose=verbose,
            )
            Rs_gaia, Teff_gaia = star["radius_val"], star["teff_val"]

            rstar, teff = Rs_tic, Teff_tic
            Rp = rprs * rstar * u.Rsun.to(u.Rearth)
            if str(rstar) == "nan":
                star_source = "Gaia"
                rstar, teff = Rs_gaia, Teff_gaia
                Rp = rprs * rstar * u.Rsun.to(u.Rearth)

            text1 = "Rp/Rs={:.4f}\nt14={:.2f} hr\nt0={:.6f}".format(
                rprs, t14, t0
            )
            text2 = "Source: {}\nRs={:.2f} Rsun\nTeff={:.0f} K\nRp={:.2f} Re".format(
                star_source, rstar, teff, Rp
            )
            if verbose:
                print(f"{text1}\n\n{text2}")
            ax[i].text(
                0.3,
                0.25,
                text1,
                verticalalignment="top",
                horizontalalignment="left",
                transform=ax[i].transAxes,
                color="g",
                fontsize=FONTSIZE,
            )
            ax[i].text(
                0.6,
                0.3,
                text2,
                verticalalignment="top",
                horizontalalignment="left",
                transform=ax[i].transAxes,
                color="g",
                fontsize=FONTSIZE,
            )

            ax[i].legend(loc=3, title="phase-folded lc")
            pl.setp(
                ax[i],
                xlim=(-0.2, 0.2),
                xlabel="Phase",
                ylabel="Normalized flux",
            )

            # manually set ylimit for shallow transits
            if rprs <= 0.1:
                ylo, yhi = 1 - 15 * rprs ** 2, 1 + 5 * rprs ** 2
                ax[i].set_ylim(ylo, yhi if yhi < 1.02 else 1.02)

            all_sectors = [str(s) for s in all_sectors]
            if results:
                # append info to tls results
                results["tic"] = ticid
                results["sector"] = all_sectors
                results["Rs_gaia"] = Rs_gaia
                results["Teff_gaia"] = Teff_gaia
                results["Rs source"] = star_source
                results["Rp"] = Rp

                fp = join(
                    figoutdir,
                    "tic{}_s{}_tls.hdf5".format(
                        tpf.targetid, "-".join(all_sectors)
                    ),
                )
                if savefig:
                    dd.io.save(fp, results)
                    print(f"Saved: {fp}\n")

            if ticid is None:
                ticid = tpf.targetid
            if toi or toiid:
                # toiid is TOIXXX determined from TESS release queried using TIC or coordinates
                id = toi if toi is not None else toiid
                figname = "TIC{}_TOI{}_s{}.png".format(
                    tpf.targetid, str(id), "-".join(all_sectors)
                )
                lcname = "TIC{}_TOI{}_s{}_lc_flat.txt".format(
                    tpf.targetid, str(id), "-".join(all_sectors)
                )
                pl.suptitle(
                    "TIC {} (TOI {})".format(ticid, id), fontsize=FONTSIZE
                )
            else:
                figname = "TIC{}_s{}.png".format(
                    tpf.targetid, "-".join(all_sectors)
                )
                lcname = "TIC{}_s{}.txt".format(
                    tpf.targetid, "-".join(all_sectors)
                )
                pl.suptitle("TIC {})".format(ticid), fontsize=FONTSIZE)
            figoutpath = join(figoutdir, figname)
            lcoutpath = join(figoutdir, lcname)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            if savefig:
                fig.savefig(figoutpath, bbox_inches="tight")
                print(f"Saved: {figoutpath}\n")
                # np.savetxt(lcoutpath, np.c_[t,f], fmt='%.8f')
                final = pd.DataFrame(np.c_[t, f])
                final.to_csv(lcoutpath, index=False, sep=" ")
                print(f"Saved: {lcoutpath}\n")
            else:
                pl.show()
            end = time.time()
            msg = "#----------Runtime: {:.2f} s----------#\n".format(
                end - start
            )
            logging.info(msg)
            if verbose:
                print(msg)
            pl.close()
        else:
            msg = "No tpf file found! Check FFI data using --cadence=long\n"
            logging.info(msg)
            if verbose:
                print(msg)
    except:
        print(f"Error occured:\n{traceback.format_exc()}")
        print_recommendations()
        logging.error(str(traceback.format_exc()))
    # save logs
    logfile = open(LOG_FILENAME, "rt")
    try:
        body = logfile.read()
    finally:
        logfile.close()
    return res


def generate_FOV(
    target_coord,
    tic=None,
    toi=None,
    sector=None,
    apphot_method="sap",
    sap_mask="pipeline",
    aper_radius=None,
    percentile=None,
    apply_data_quality_mask=True,
    fitsoutdir=".",
    figoutdir=".",
    savefig=True,
    clobber=False,
    verbose=True,
):
    """Create DSS2 blue and red FOV images

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    tic : int
        TIC id
    toi : float
        TOI id
    sector : int
        TESS sector
    apphot_method : str
        aperture photometry method
    sap_mask : str
        SAP mask type
    aper_radius : int
        aperture radius
    fitsoutdir : str
        fits output directory
    figoutdir : str
        figure output directory
    savefig : bool
        save figure
    """
    start = time.time()
    try:
        tpf, df = get_tpf(
            target_coord,
            tic=tic,
            apphot_method="sap",
            sector=sector,
            verbose=verbose,
            clobber=clobber,
            apply_data_quality_mask=apply_data_quality_mask,
            sap_mask=sap_mask,
            fitsoutdir=fitsoutdir,
            return_df=True,
        )
        # check tpf size
        ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]
        diag = np.sqrt(nx ** 2 + ny ** 2)
        fov_rad = (0.6 * diag * TESS_pix_scale).to(u.arcmin)
        if fov_rad > 1 * u.deg:
            tpf = cutout_tpf(tpf)
            # redefine dimensions
            ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]
            diag = np.sqrt(nx ** 2 + ny ** 2)
            fov_rad = (0.6 * diag * TESS_pix_scale).to(u.arcmin)

        # make aperture mask
        mask = parse_aperture_mask(
            tpf,
            sap_mask=sap_mask,
            aper_radius=aper_radius,
            percentile=percentile,
            verbose=verbose,
        )
        maskhdr = tpf.hdu[2].header
        photwcs = WCS(maskhdr)

        if tpf.targetid is not None:
            ticid = tpf.targetid
        else:
            ticid = df["target_name"].values[0]
        # query tess alerts/ toi release
        try:
            q = get_toi(
                tic=ticid,
                toi=toi,
                clobber=clobber,
                outdir="../data/",
                verbose=False,
            )
        except Exception as e:
            print(e)
        try:
            period, t0, t14, depth, toiid = get_transit_params(
                toi=toi, tic=ticid, verbose=False
            )
        except Exception as e:
            print(e)
            period, t0, t14, depth, toiid = None, None, None, None, None

        if verbose:
            print(
                "Querying {0} ({1:.2f} x {1:.2f}) archival image".format(
                    "DSS2 Blue", fov_rad
                )
            )
        nax1, hdu1 = plot_finder_image(
            target_coord, fov_radius=fov_rad, survey="DSS2 Blue", reticle=True
        )
        pl.close()
        if verbose:
            print(
                "Querying {0} ({1:.2f} x {1:.2f}) archival image".format(
                    "DSS2 Red", fov_rad
                )
            )
        nax2, hdu2 = plot_finder_image(
            target_coord, fov_radius=fov_rad, survey="DSS2 Red", reticle=True
        )
        pl.close()
        wcs1 = WCS(hdu1.header)
        wcs2 = WCS(hdu2.header)

        # -----------create figure---------------#
        fig = pl.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(121, projection=wcs1)
        ax2 = fig.add_subplot(122, projection=wcs2)

        gaia_sources = Catalogs.query_region(
            target_coord, radius=fov_rad, catalog="Gaia", version=2
        ).to_pandas()

        obj = FixedTarget(target_coord)
        nax1, hdu1 = plot_finder_image(
            obj, fov_radius=fov_rad, survey="DSS2 Blue", reticle=True, ax=ax1
        )
        nax2, hdu2 = plot_finder_image(
            obj, fov_radius=fov_rad, survey="DSS2 Red", reticle=True, ax=ax2
        )
        wcs1 = WCS(hdu1.header)
        wcs2 = WCS(hdu2.header)
        mx, my = hdu1.data.shape

        contour = np.zeros((ny, nx))
        contour[np.where(mask)] = 1
        contour = np.lib.pad(contour, 1, PadWithZeros)
        highres = zoom(contour, 100, order=0, mode="nearest")

        extent = np.array([-1, nx, -1, ny])
        # aperture mask
        cs1 = ax1.contour(
            highres,
            levels=[0.5],
            extent=extent,
            origin="lower",
            colors="y",
            transform=nax1.get_transform(photwcs),
        )
        cs2 = ax2.contour(
            highres,
            levels=[0.5],
            extent=extent,
            origin="lower",
            colors="y",
            transform=nax2.get_transform(photwcs),
        )

        for r, d in gaia_sources[["ra", "dec"]].values:
            pix1 = wcs1.all_world2pix(np.c_[r, d], 1)[0]
            pix2 = wcs2.all_world2pix(np.c_[r, d], 1)[0]
            nax1.scatter(
                pix1[0],
                pix1[1],
                marker="s",
                s=100,
                edgecolor="r",
                facecolor="none",
            )
            nax2.scatter(
                pix2[0],
                pix2[1],
                marker="s",
                s=100,
                edgecolor="r",
                facecolor="none",
            )
        pl.setp(
            nax1,
            xlim=(0, mx),
            ylim=(0, my),
            title="DSS2 Blue ({0:.2f}' x {0:.2f}')".format(fov_rad.value),
        )
        pl.setp(
            nax2,
            xlim=(0, mx),
            ylim=(0, my),
            title="DSS2 Red ({0:.2f}' x {0:.2f}')".format(fov_rad.value),
        )

        if toi or toiid:
            id = toi if toi is not None else toiid
            figname = "TIC{}_TOI{}_FOV_s{}.png".format(tic, id, sector)
            pl.suptitle("TIC {} (TOI {})".format(ticid, id), fontsize=FONTSIZE)
        else:
            figname = "TIC{}_FOV_s{}.png".format(tic, sector)
            pl.suptitle("TIC {}".format(ticid), fontsize=FONTSIZE)
        figoutpath = join(figoutdir, figname)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if savefig:
            fig.savefig(figoutpath, bbox_inches="tight")
            print(f"Saved: {figoutpath}\n")
        else:
            pl.show()
        end = time.time()
        msg = "#----------Runtime: {:.2f} s----------#\n".format(end - start)
        logging.info(msg)
        if verbose:
            print(msg)
        pl.close()

    except:
        print(f"Error occured:\n{traceback.format_exc()}")
        print_recommendations()
        logging.error(str(traceback.format_exc()))


def get_tois(
    clobber=True,
    outdir="../data/",
    verbose=False,
    remove_FP=True,
    remove_known_planets=False,
):
    """Download TOI list from TESS Alert/TOI Release.

    Parameters
    ----------
    clobber : bool
        re-download table and save as csv file
    outdir : str
        download directory location
    verbose : bool
        print texts

    Returns
    -------
    d : pandas.DataFrame
        TOI table as dataframe
    """
    dl_link = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    fp = join(outdir, "TOIs.csv")
    if not exists(outdir):
        os.makedirs(outdir)

    if not exists(fp) or clobber:
        d = pd.read_csv(dl_link)  # , dtype={'RA': float, 'Dec': float})
        # remove False Positives
        if remove_FP:
            d = d[d["TFOPWG Disposition"] != "FP"]
            if verbose:
                print("TOIs with TFPWG disposition==FP are removed.\n")
        if remove_known_planets:
            planet_keys = [
                "WASP",
                "SWASP",
                "HAT",
                "HATS",
                "KELT",
                "QATAR",
                "K2",
                "Kepler",
            ]
            keys = []
            for key in planet_keys:
                idx = ~np.array(
                    d["Comments"].str.contains(key).tolist(), dtype=bool
                )
                d = d[idx]
                if idx.sum() > 0:
                    keys.append(key)
            if verbose:
                print(f"{keys} planets are removed.\n")
        d.to_csv(fp, index=False)
        if verbose:
            print(f"Saved: {fp}\n")
    else:
        d = pd.read_csv(fp)
        # remove False Positives
        if remove_FP:
            d = d[d["TFOPWG Disposition"] != "FP"]
        if verbose:
            print(f"Loaded: {fp}")

    return d.sort_values("TOI")


def get_toi(toi=None, tic=None, clobber=True, outdir="../data/", verbose=True):
    """Query TOI from TOI list

    Parameters
    ----------
    tic : int
        TIC id
    toi : float
        TOI id
    clobber : bool
        re-download csv file
    outdir : str
        csv path
    verbose : bool
        print texts

    Returns
    -------
    q : pandas.DataFrame
        TOI match else None
    """
    if (toi is None) and (tic is None):
        raise ValueError('Provide toi or tic')
    else:
        df = get_tois(clobber=clobber, verbose=verbose, outdir=outdir)

        if toi is not None:
            if isinstance(toi, int):
                toi = float(str(toi) + ".01")
            else:
                planet = str(toi).split(".")[1]
                assert len(planet) == 2, "use pattern: TOI.01"
            idx = df["TOI"].isin([toi])
        elif tic is not None:
            idx = df["TIC ID"].isin([tic])

        q = df.loc[idx]
        # return if empty, else continue
        if len(q) == 0:
            raise ValueError("TOI not found!")

        q.index = q["TOI"].values
        if verbose:
            print("Data from TOI Release:\n")
            columns = [
                "Period (days)",
                "Epoch (BJD)",
                "Duration (hours)",
                "Depth (ppm)",
                "Comments",
            ]
            print(f"{q[columns].T}\n")

        if q["TFOPWG Disposition"].isin(["FP"]).any():
            print("\nTFOPWG disposition is a False Positive!\n")

        return q.sort_values(by="TOI", ascending=True)


def get_Rp_monte_carlo(RpRs, Rs, nsamples=10000, verbose=True):
    """Monte Carlo estimation of planet radius

    Parameters
    ----------
    RoRs : tuple (val,err)
        Rp/Rs = sqrt(transit depth)
    Rs : tuple (val,err)
        stellar radius
    nsamples : int
        number of samples

    Returns
    -------
    Rp : tuple (val, -siglo, +sighi)
        planet radius [Rearth]
    """

    RpRs_s = np.random.rand(nsamples) * RpRs[1] + RpRs[0]
    Rs_s = np.random.rand(nsamples) * Rs[1] + Rs[0]

    med, lo, hi = np.percentile(RpRs_s * Rs_s, [50, 16, 84]) * u.Rsun.to(
        u.Rearth
    )
    if verbose:
        print(f"Range: {lo},{hi}")
        print(f"{med} -{med-lo} +{hi-med}")
    return (med, med - lo, hi - med)


def get_transit_params(
    toi=None, tic=None, clobber=False, verbose=False, outdir="../data/"
):
    """
    """
    if (toi is not None) or (tic is not None):
        q = get_toi(
            toi=toi, tic=tic, clobber=clobber, verbose=verbose, outdir=outdir
        )
    else:
        raise ValueError('Provide toi or tic')

    if len(q) > 0:
        if toi is not None:
            toiid = str(toi).split(".")[0]
            assert (
                toi in q["TOI"].values
            ), f"{toi} not found! Check: {q['TOI'].values}"
            planet_num = int(str(toi).split(".")[1])
            idx = planet_num - 1 if len(q) > 1 else 0
            period = q["Period (days)"].values[idx]
            t0 = q["Epoch (BJD)"].values[idx]
            t14 = q["Duration (hours)"].values[idx] * u.hour.to(u.day)
            depth = q["Depth (ppm)"].values[idx]
        else:
            # assume first planet since tic cannot specify which planet
            idx = 0
            if tic in q["TIC ID"].values:
                toi = q["TOI"].values[idx]
                toiid = str(q["TOI"].values[idx]).split(".")[0]
                period = q["Period (days)"].values[idx]
                t0 = q["Epoch (BJD)"].values[idx]
                t14 = q["Duration (hours)"].values[idx] * u.hour.to(u.day)
                depth = q["Depth (ppm)"].values[idx]

                if verbose:
                    print(
                        "TIC {} is identified as TOI {} from TESS alerts/TOI Releases...\n".format(
                            tic, toiid
                        )
                    )
            else:
                toiid = str(q["TOI"].values[idx]).split(".")[0]
                period, t0, t14, depth = None, None, None, None
                msg = "tic not found!"
                print(msg)
                logging.error(msg)

            ticid = q["TIC ID"].values[idx]
            tois = list(set(q["TOI"].values))
            if verbose and len(tois) > 1:
                assert ticid in q["TIC ID"].values
                msg = "TIC {} (TOI {}) has {} planets!\n".format(
                    ticid, toiid, len(q)
                )
                logging.info(msg)
                print(msg)
        return (period, t0, t14, depth, toiid)
    else:
        raise ValueError('empty query')



def make_cadence_mask(time, period, t0, t14=0.5, padding=0.5, verbose=False):
    """Make cadence mask for PLD if ephemeris is known

    Parameters
    ----------
    time : array
        time [JD]
    period : float
        orbital period [day]
    t14 : float
        transit duration [day] (default 12 hr)
    padding : float
        factor padding to t12 and t34 in units of t14 to include in mask

    Returns
    -------
    cadence_mask : bool array
        masked candences
    """
    if np.all([period, t0]):
        if (time_format == "btjd") and (t0 > 2450000):
            t0 = t0 - TESS_JD_offset
        tns = get_tns(time, period, t0, allow_half_period=True)
        ntransit = len(tns)
        cadence_mask = np.zeros_like(time).astype(bool)
        for tn in tns:
            ix = (time > tn - t14 * padding) & (time < tn + t14 * padding)
            cadence_mask[ix] = True
        ncadence_mask = cadence_mask.sum()
        mask_duration = 2 * padding * t14 * u.day.to(u.hour)
        msg = f"n={ntransit} transits with {mask_duration} hr mask duration\nyields n={ncadence_mask} masked cadences\n"
        assert len(time) == len(cadence_mask)
    else:
        # FIXME: no transit mask usually overfits!
        cadence_mask = None
        msg = "Provide period & t0.\n"
    if verbose:
        logging.info(msg)
        print(msg)
    return cadence_mask


def plot_gaia_sources(
    tpf,
    target_gaiaid,
    gaia_sources=None,
    fov_rad=1 * u.arcmin,
    depth=0.0,
    kmax=1.0,
    sap_mask="pipeline",
    survey="DSS2 Red",
    verbose=True,
    ax=None,
    **kwargs,
):
    """Plot (superpose) Gaia sources on archival image

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    gaia_sources : pd.DataFrame
        gaia sources table
    fov_rad : astropy.unit
        FOV radius
    survey : str
        image survey; see from astroquery.skyview import SkyView;
        SkyView.list_surveys()
    verbose : bool
        print texts
    ax : axis
        subplot axis
    kwargs : dict
        keyword arguments for aper_radius, percentile
    Returns
    -------
    ax : axis
        subplot axis
    """
    ny, nx = tpf.flux.shape[1:]
    if fov_rad is None:
        diag = np.sqrt(nx ** 2 + ny ** 2)
        fov_rad = (0.4 * diag * TESS_pix_scale).to(u.arcmin)
    target_coord = SkyCoord(ra=tpf.ra * u.deg, dec=tpf.dec * u.deg)
    if gaia_sources is None:
        gaia_sources = Catalogs.query_region(
            target_coord, radius=fov_rad, catalog="Gaia", version=2
        ).to_pandas()
    # make aperture mask
    mask = parse_aperture_mask(tpf, sap_mask=sap_mask, **kwargs)
    maskhdr = tpf.hdu[2].header
    # make aperture mask outline
    contour = np.zeros((ny, nx))
    contour[np.where(mask)] = 1
    contour = np.lib.pad(contour, 1, PadWithZeros)
    highres = zoom(contour, 100, order=0, mode="nearest")
    extent = np.array([-1, nx, -1, ny])

    if tpf.targetid is not None:
        ticid = tpf.targetid
    else:
        ticid = df["target_name"].values[0]

    if verbose:
        print(
            f"Querying {survey} ({fov_rad:.2f} x {fov_rad:.2f}) archival image"
        )
    # get img hdu
    nax, hdu = plot_finder_image(
        target_coord, fov_radius=fov_rad, survey=survey, reticle=True
    )
    pl.close()

    # -----------create figure---------------#
    fig = pl.figure(figsize=(6, 6))
    # define scaling in projection
    ax = fig.add_subplot(111, projection=WCS(hdu.header))
    nax, hdu = plot_finder_image(
        target_coord, ax=ax, fov_radius=fov_rad, survey=survey, reticle=False
    )
    imgwcs = WCS(hdu.header)
    mx, my = hdu.data.shape
    # plot mask
    cs = ax.contour(
        highres,
        levels=[0.5],
        extent=extent,
        origin="lower",
        colors="b",
        transform=nax.get_transform(WCS(maskhdr)),
    )
    idx = gaia_sources["source_id"].astype(int).isin([target_gaiaid])
    target_gmag = gaia_sources.loc[idx, "phot_g_mean_mag"].values[0]
    # marker & size
    marker, s = "o", 100
    NEBs = []
    for r, d, mag, id in gaia_sources[
        ["ra", "dec", "phot_g_mean_mag", "source_id"]
    ].values:
        pix = imgwcs.all_world2pix(np.c_[r, d], 1)[0]
        if int(id) != target_gaiaid:
            dmag = mag - target_gmag
            gamma = 1 + 10 ** (0.4 * dmag)
            if depth > kmax / gamma:
                # too deep to have originated from secondary star
                edgecolor = "b"
                alpha = 0.1
            else:
                # possible NEBs
                NEBs.append(id)
                edgecolor = "r"
                alpha = 1
        else:
            edgecolor = "y"
            alpha = 1
        nax.scatter(
            pix[0],
            pix[1],
            marker=marker,
            s=s,
            edgecolor=edgecolor,
            alpha=alpha,
            facecolor="none",
        )
    if verbose:
        bad = gaia_sources.loc[gaia_sources.source_id.astype(int).isin(NEBs)]
        bad['distance']=bad['distance'].apply(lambda x: x*u.arcmin.to(u.arcsec))
        print(bad[['source_id','distance','parallax','phot_g_mean_mag']])
    # set img limits
    pl.setp(
        nax,
        xlim=(0, mx),
        ylim=(0, my),
        title="{0} ({1:.2f}' x {1:.2f}')".format(survey, fov_rad.value),
    )
    return fig, NEBs


def get_gaia_params_from_tic(target_coord=None, toi=None, tic=None):
    if not np.any([target_coord, tic, toi, gaia_id]):
        raise ValueError("Provide target_coord or toi or tic")
    tic_sources = Catalogs.query_region(
        target_coord, radius=radius, catalog="TIC"  # version=8
    ).to_pandas()
    return tic_sources


def get_gaia_params_from_dr2(
    target_coord=None,
    toi=None,
    tic=None,
    gaia_id=None,
    gaia_sources=None,
    return_basic=False,
    return_phot=False,
    return_star=False,
    verbose=False,
    search_radius=10 * u.arcsec,
):
    """ """
    if (target_coord is None) and (toi is None) and (tic is None):
        raise ValueError("Provide target_coord or toi or tic")
    try:
        q = get_toi(
            tic=tic, toi=toi, clobber=False, outdir="../data/", verbose=False
        )
        target_coord = SkyCoord(
            ra=q["RA"], dec=q["Dec"], unit=(u.hourangle, u.deg)
        )[0]
        Tmag = q["TESS Mag"].values[0]
    except Exception as e:
        print(e)
    if gaia_sources is None:
        gaia_sources = Catalogs.query_region(
            target_coord, radius=search_radius, catalog="Gaia", version=2
        ).to_pandas()
    gcoords = SkyCoord(
        ra=gaia_sources["ra"], dec=gaia_sources["dec"], unit="deg"
    )[0]
    # FIXME: may not correspond to the host if binary or has confusing background star
    if gaia_id is not None:
        # search gaia id
        idx = np.where(gaia_sources["source_id"].astype(int).isin([gaia_id]))[
            0
        ][0]
    else:
        # assume closest coordinate match
        idx = target_coord.separation(gcoords).argmin()
        gaia_id = int(gaia_sources.loc[idx, "source_id"])
    # perform match
    d = gaia_sources.iloc[idx]
    # FIXME: compute Tmag using ticgen

    # T = Star(Gmag=d['phot_g_mean_mag'])
    # assert abs(T - Tmag)<1, '|G-T|>1'
    params = d.index
    d.name = gaia_id

    if len(d) > 1:
        print(
            "\nThere are {} Gaia sources within r={:.2f} of ra={:.2f} dec={:.2f}\n".format(
                len(d),
                search_radius,
                target_coord.ra.deg,
                target_coord.dec.deg,
            )
        )

    if return_basic:
        idx1 = [10, 11, 12, 13, 14, 15, 16, 33, 34, 67, 68]
        vals = d[params[idx1]]
        if verbose:
            print(f"{vals}\n")
        return vals

    elif return_phot:  # photometry
        idx2 = [51, 56, 61]
        vals = d[params[idx2]]
        if verbose:
            print(f"{vals}\n")
        return vals

    elif return_star:  # stellar
        idx3 = [78, 79, 80, 88, 89, 90, 81, 82, 83]
        vals = d[params[idx3]]
        if verbose:
            print(f"{vals}\n")
        return vals

    else:
        return d

    if d["astrometric_excess_noise_sig"] > 2:
        print(
            "The target has significant astrometric excess noise: {:.2f}\n".format(
                d["astrometric_excess_noise_sig"]
            )
        )
    rstar = star["radius_val"]
    rstar_lo = rstar - star["radius_percentile_lower"]
    rstar_hi = star["radius_percentile_upper"] - rstar
    teff = star["teff_val"]
    teff_lo = teff - star["teff_percentile_lower"]
    teff_hi = star["teff_percentile_upper"] - teff
    if verbose:
        print("Rstar={:.2f} +{:.2f} -{:.2f}".format(rstar, rstar_lo, rstar_hi))
        print("Teff={:.0f} +{:.0f} -{:.0f}\n".format(teff, teff_lo, teff_hi))
    return rstar, teff


def get_tns(t, p, t0, allow_half_period=False):
    """Get transit occurrences

    Parameters
    ----------
    t : float
        time
    p : float
        period
    t0 : float
        epoch of periastron
    allow_half_period : bool

    Returns
    -------
    tns : array
        transit occurrences
    """
    baseline = t[-1] - t[0]
    # assert baseline>p, 'period longer than baseline'
    if allow_half_period and (baseline < p):
        print("P={:.2f}d > time baseline=({:.2f})d\n".format(p, baseline))
        p = p / 2.0
        print("Using P/2={:.2f}".format(p))

    idx = t != 0
    t = t[idx]

    while t0 - p > t.min():
        t0 -= p
    if t0 < t.min():
        t0 += p

    tns = [t0 + p * i for i in range(int((t.max() - t0) / p + 1))]

    while tns[-1] > t.max():
        tns.pop()

    while tns[0] < t.min():
        tns = tns[1:]

    return np.array(tns)


def PadWithZeros(vector, pad_width, iaxis, kwargs):
    """ """
    vector[: pad_width[0]] = 0
    vector[-pad_width[1] :] = 0
    return vector


def collate_tls_results(results_dir, save_csv=False):
    """Collate TLS results

    Parameters
    ----------
    results_dir : str
        directory with tls.hdf5 files
    save_csv : bool
        save csv files

    Returns
    -------
    df : pd.DataFrame
        collated table
    """
    if results_dir is None:
        results_dir = "."
    colnames = "tic sector snr SDE FAP rp_rs period T0 duration depth_mean_odd depth_mean_even chi2 transit_count".split()
    if not exists(results_dir):
        raise FileNotFoundError("{} does not exist!".format(results_dir))
    fl = glob(join(results_dir, "*.hdf5"))
    if len(fl) > 0:
        pars = []
        for f in tqdm(fl):
            results = dd.io.load(f)
            s = pd.Series([results[col] for col in colnames], index=colnames)
            pars.append(s)
        df = pd.DataFrame(pars).sort_values(by="SDE", ascending=False)
        if save_csv:
            fp = join(results_dir, "tls_summary.csv")
            df.to_csv(fp, index=False)
            print(f"Saved: {fp}\n")
        else:
            # SDE=9 == FP rate<1e−4 in the limiting case of white noise
            print(df)
    else:
        raise FileNotFoundError(
            "No .hdf5 files found in {}\n".format(results_dir)
        )
    return df


def binned(a, binsize, fun=np.mean):
    """bin data"""
    a_b = []
    for i in range(0, a.shape[0], binsize):
        a_b.append(fun(a[i : i + binsize], axis=0))
    # make sure no NaN
    # a_b = a_b[~np.isnan(a_b)]
    return a_b


def remove_bad_data(tpf, sector=None, verbose=True):
    """Remove bad cadences identified in data releae notes

    Parameters
    ----------
    tpf : lk.targetpixelfile

    sector : int
        TESS sector
    verbose : bool
        print texts
    """
    if sector is None:
        sector = tpf.sector
    if verbose:
        print(
            "Applying data quality mask identified in Data Release Notes (sector {}):".format(
                sector
            )
        )
    if sector == 1:
        pointing_jitter_start = 1346
        pointing_jitter_end = 1350
        if verbose:
            print(
                "t<{}|t>{}\n".format(
                    pointing_jitter_start, pointing_jitter_end
                )
            )
        tpf = tpf[
            (tpf.time < pointing_jitter_start)
            | (tpf.time > pointing_jitter_end)
        ]
    if sector == 2:
        if verbose:
            print("None.\n")
    if sector == 3:
        science_data_start = 1385.89663
        science_data_end = 1406.29247
        if verbose:
            print("t>{}|t<{}\n".format(science_data_start, science_data_end))
        tpf = tpf[
            (tpf.time > science_data_start) | (tpf.time < science_data_end)
        ]
    if sector == 4:
        guidestar_tables_replaced = 1413.26468
        instru_anomaly_start = 1418.53691
        data_collection_resumed = 1421.21168
        if verbose:
            print(
                "t>{}|t<{}|t>{}\n".format(
                    guidestar_tables_replaced,
                    instru_anomaly_start,
                    data_collection_resumed,
                )
            )
        tpf = tpf[
            (tpf.time > guidestar_tables_replaced)
            | (tpf.time < instru_anomaly_start)
            | (tpf.time > data_collection_resumed)
        ]
    if sector == 5:
        # use of Cam1 in attitude control was disabled for the
        # last ~0.5 days of orbit due to o strong scattered light
        cam1_guide_disabled = 1463.93945
        if verbose:
            print("t<{}\n".format(cam1_guide_disabled))
        tpf = tpf[tpf.time < cam1_guide_disabled]
    if sector == 6:
        # ~3 days of orbit 19 were used to collect calibration
        # data for measuring the PRF of cameras;
        # reaction wheel speeds were reset with momentum dumps
        # every 3.125 days
        data_collection_start = 1468.26998
        if verbose:
            print("t>{}\n".format(data_collection_start))
        tpf = tpf[tpf.time > data_collection_start]
    if sector == 8:
        # interruption in communications between instru and spacecraft occurred
        cam1_guide_enabled = 1517.39566
        orbit23_end = 1529.06510
        cam1_guide_enabled2 = 1530.44705
        instru_anomaly_start = 1531.74288
        data_colletion_resumed = 1535.00264
        if verbose:
            print(
                "t>{}|t<{}|t>{}|t<{}|t>{}\n".format(
                    cam1_guide_enabled,
                    orbit23_end,
                    cam1_guide_enabled2,
                    instru_anomaly_start,
                    data_colletion_resumed,
                )
            )
        tpf = tpf[
            (tpf.time > cam1_guide_enabled)
            | (tpf.time <= orbit23_end)
            | (tpf.time > cam1_guide_enabled2)
            | (tpf.time < instru_anomaly_start)
            | (tpf.time > data_colletion_resumed)
        ]
    if sector == 9:
        # use of Cam1 in attitude control was disabled at the
        # start of both orbits due to strong scattered light
        cam1_guide_enabled = 1543.75080
        orbit25_end = 1555.54148
        cam1_guide_enabled2 = 1543.75080
        if verbose:
            print(
                "t>{}|t<{}|t>{}\n".format(
                    cam1_guide_enabled, orbit25_end, cam1_guide_enabled2
                )
            )
        tpf = tpf[
            (tpf.time > cam1_guide_enabled)
            | (tpf.time <= orbit25_end)
            | (tpf.time > cam1_guide_enabled2)
        ]
    if sector == 10:
        # use of Cam1 in attitude control was disabled at the
        # start of both orbits due to strong scattered light
        cam1_guide_enabled = 1570.87620
        orbit27_end = 1581.78453
        cam1_guide_enabled2 = 1584.72342
        if verbose:
            print(
                "t>{}|t<{}|t>{}\n".format(
                    cam1_guide_enabled, orbit27_end, cam1_guide_enabled2
                )
            )
        tpf = tpf[
            (tpf.time > cam1_guide_enabled)
            | (tpf.time <= orbit27_end)
            | (tpf.time > cam1_guide_enabled2)
        ]
    if sector == 11:
        # use of Cam1 in attitude control was disabled at the
        # start of both orbits due to strong scattered light
        cam1_guide_enabled = 1599.94148
        orbit29_end = 1609.69425
        cam1_guide_enabled2 = 1614.19842
        if verbose:
            print(
                "t>{}|t<{}|t>{}\n".format(
                    cam1_guide_enabled, orbit29_end, cam1_guide_enabled2
                )
            )
        tpf = tpf[
            (tpf.time > cam1_guide_enabled)
            | (tpf.time <= orbit29_end)
            | (tpf.time > cam1_guide_enabled2)
        ]
    return tpf


def cutout_tpf(tpf):
    """create a smaller cutout of original tpf

    Parameters
    ----------
    tpf : targetpixelfile

    Note
    ----
    This requires new method in trim-tpfs branch of lightkurve
    """
    ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]
    print(
        "tpf size=({},{}) pix chosen by TESS pipeline seems too big!\n".format(
            ny, nx
        )
    )
    # choose the length of size
    min_dim = min(tpf.flux.shape)
    new_size = 12 if (min_dim < 8) | (min_dim > 20) else min_dim
    new_center = (min_dim // 2, min_dim // 2)
    print("Setting tpf size to ({}, {}) pix.\n".format(new_size, new_size))
    print("This process may take some time.\n")
    tpf = tpf.cutout(center=new_center, size=new_size)
    return tpf


def parse_aperture_mask(
    tpf, sap_mask="pipeline", aper_radius=None, percentile=None, verbose=False
):
    """Parse and make aperture mask"""
    if verbose:
        if sap_mask == "round":
            print(
                "aperture photometry mask: {} (r={} pix)\n".format(
                    sap_mask, aper_radius
                )
            )
        elif sap_mask == "square":
            print(
                "aperture photometry mask: {0} ({1}x{1} pix)\n".format(
                    sap_mask, aper_radius
                )
            )
        elif sap_mask == "percentile":
            print(
                "aperture photometry mask: {} ({}%)\n".format(
                    sap_mask, percentile
                )
            )
        else:
            print("aperture photometry mask: {}\n".format(sap_mask))

    # stacked_img = np.median(tpf.flux,axis=0)
    if sap_mask == "all":
        mask = np.ones((tpf.shape[1], tpf.shape[2]), dtype=bool)
    elif sap_mask == "round":
        assert aper_radius is not None, "supply aper_radius"
        mask = make_round_mask(tpf.flux[0], radius=aper_radius)
    elif sap_mask == "square":
        assert aper_radius is not None, "supply aper_radius/size"
        mask = make_square_mask(tpf.flux[0], size=aper_radius, angle=None)
    elif sap_mask == "threshold":
        mask = tpf.create_threshold_mask()
    elif sap_mask == "percentile":
        assert percentile is not None, "supply percentile"
        median_img = np.nanmedian(tpf.flux, axis=0)
        mask = median_img > np.nanpercentile(median_img, percentile)
    else:
        mask = tpf.pipeline_mask  # default
    return mask


def make_round_mask(img, radius, xy_center=None):
    """Make round mask in units of pixels

    Parameters
    ----------
    img : numpy ndarray
        image
    radius : int
        aperture mask radius or size
    xy_center : tuple
        aperture mask center position

    Returns
    -------
    mask : np.ma.masked_array
        aperture mask
    """
    h, w = img.shape
    if xy_center is None:  # use the middle of the image
        y, x = np.unravel_index(np.argmax(img), img.shape)
        xy_center = [x, y]
        # check if near edge
        if np.any([x >= h - 1, x >= w - 1, y >= h - 1, y >= w - 1]):
            print("Brightest star is detected near the edges.")
            print("Aperture mask is placed at the center instead.\n")
            xy_center = [img.shape[0] // 2, img.shape[1] // 2]

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt(
        (X - xy_center[0]) ** 2 + (Y - xy_center[1]) ** 2
    )

    mask = dist_from_center <= radius
    return np.ma.masked_array(img, mask=mask).mask


def make_square_mask(img, size, xy_center=None, angle=None):
    """Make rectangular mask with optional rotation

    Parameters
    ----------
    img : numpy ndarray
        image
    size : int
        aperture mask size
    xy_center : tuple
        aperture mask center position
    angle : int
        rotation

    Returns
    -------
    mask : np.ma.masked_array
        aperture mask
    """
    h = w = size
    if xy_center is None:  # use the middle of the image
        y, x = np.unravel_index(np.argmax(img), img.shape)
        xy_center = [x, y]
        # check if near edge
        if np.any([x >= h - 1, x >= w - 1, y >= h - 1, y >= w - 1]):
            print(
                "Brightest star detected is near the edges.\nAperture mask is placed at the center instead.\n"
            )
            x, y = img.shape[0] // 2, img.shape[1] // 2
            xy_center = [x, y]
    mask = np.zeros_like(img, dtype=bool)
    mask[y - h : y + h + 1, x - w : x + w + 1] = True
    # if angle:
    #    #rotate mask
    #    mask = rotate(mask, angle, axes=(1, 0), reshape=True, output=bool, order=0)
    return mask


def get_2d_centroid(image):
    """ """
    img = np.copy(image)
    w, h = img.shape
    y, x = np.unravel_index(np.argmax(img), img.shape)
    # check if centroid is near image boundary
    while np.any([x >= h - 1, x >= w - 1, y >= h - 1, y >= w - 1]):
        img[y, x] = 0
        y, x = np.unravel_index(np.argmax(img), img.shape)
    return y, x


def plot_centroid_shift(tpf, cadence_mask_tpf, ax=None):
    """the residual from the difference between the mean out-of-transit
    flux value and the mean in-transit shows the location of the signal.
    """
    flux_intransit = np.nanmean(tpf.flux[cadence_mask_tpf], axis=0)
    flux_outtransit = np.nanmean(tpf.flux[~cadence_mask_tpf if cadence_mask_tpf.sum()>0 else None], axis=0)

    # centroid based on TIC coordinates
    y, x = tpf.wcs.all_world2pix(np.c_[tpf.ra, tpf.dec], 1)[0]
    # centroid based on out-of-transit centroid
    y2, x2 = get_2d_centroid(flux_outtransit)

    if ax is None:
        fig, ax = pl.subplots()

    ax.matshow(flux_outtransit - flux_intransit, origin="lower")
    ax.plot(x, y, "rx", ms=18, label="TIC")
    ax.plot(x2, y2, "bx", ms=18, label="OOT")
    pl.colorbar()
    pl.legend()
    return ax


def generate_multi_aperture_lc(
    target_coord,
    aper_radii=None,
    tic=None,
    toi=None,
    sector=None,
    use_pld=False,
    use_gp=False,
    use_sff=False,
    percentiles=None,
    apphot_method="sap",
    sap_mask="pipeline",
    apply_data_quality_mask=True,
    fitsoutdir=".",
    figoutdir=".",
    savefig=True,
    clobber=False,
    verbose=True,
):
    """Create lc with two aperture masks

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    aper_radii : array
        aperture radii
    percentiles : array
        aperture percentiles
    tic : int
        TIC id
    use_pld : bool
        use PLD for systematics correction
    toi : float
        TOI id
    use_sff : bool
        use SFF for systematics correction
    sector : int
        TESS sector
    apply_data_quality_mask : bool
        apply quality mask identified in TESS Notes
    apphot_method : str
        aperture photometry method
    sap_mask : str
        SAP mask type
    fitsoutdir : str
        fits output directory
    figoutdir : str
        figure output directory
    savefig : bool
        save figure
    clobber : bool
        re-download files
    """
    start = time.time()
    try:
        tpf, df = get_tpf(
            target_coord,
            tic=tic,
            apphot_method=apphot_method,
            sap_mask=sap_mask,
            apply_data_quality_mask=apply_data_quality_mask,
            sector=sector,
            verbose=verbose,
            clobber=clobber,
            fitsoutdir=fitsoutdir,
            return_df=True,
        )
        all_sectors = [int(i) for i in df["sequence_number"].values]
        if sector is None:
            sector = all_sectors[0]
        # check tpf size
        ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]
        diag = np.sqrt(nx ** 2 + ny ** 2)
        fov_rad = (0.6 * diag * TESS_pix_scale).to(u.arcmin)
        if fov_rad > 1 * u.deg:
            tpf = cutout_tpf(tpf)
            # redefine dimensions
            ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]
            diag = np.sqrt(nx ** 2 + ny ** 2)
            fov_rad = (0.6 * diag * TESS_pix_scale).to(u.arcmin)

        # check if target is TOI from TESS alerts
        if tpf.targetid is not None:
            ticid = tpf.targetid
        else:
            ticid = df["target_name"].values[0]

        try:
            q = get_toi(
                tic=ticid,
                toi=toi,
                clobber=clobber,
                outdir="../data/",
                verbose=False,
            )
        except Exception as e:
            print(e)
        try:
            period, t0, t14, depth, toiid = get_transit_params(
                toi=toi, tic=ticid, verbose=False
            )
        except Exception as e:
            print(e)
            period, t0, t14, depth, toiid = None, None, None, None, None

        maskhdr = tpf.hdu[2].header
        tpfwcs = WCS(maskhdr)

        if verbose:
            print(
                "Querying {0} ({1:.2f} x {1:.2f}) archival image".format(
                    "DSS2 Blue", fov_rad
                )
            )
        _, hdu = plot_finder_image(
            target_coord, fov_radius=fov_rad, survey="DSS2 Blue", reticle=True
        )
        pl.close()

        # -----------create figure---------------#
        fig = pl.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222, projection=WCS(hdu.header))

        # -----ax2-----#
        # query gaia
        gaia_sources = Catalogs.query_region(
            target_coord, radius=fov_rad, catalog="Gaia", version=2
        ).to_pandas()
        tic_sources = Catalogs.query_region(
            target_coord, radius=fov_rad, catalog="TIC"  # version=8
        ).to_pandas()
        idx = tic_sources["ID"].astype(int).isin([ticid])
        if np.any(idx):
            gaia_id = tic_sources.loc[idx, "GAIA"].values[0]
            gaia_id = int(gaia_id) if str(gaia_id) != "nan" else None
        else:
            gaia_id = None

        # get gaia stellar params
        star = get_gaia_params_from_dr2(
            target_coord,
            tic=ticid,
            gaia_sources=gaia_sources,
            gaia_id=gaia_id,
            return_star=True,
            search_radius=fov_rad,
            verbose=verbose,
        )
        Rs_gaia, Teff_gaia = star["radius_val"], star["teff_val"]
        Rs_err_gaia = np.sqrt(
            star["radius_percentile_upper"] ** 2
            + star["radius_percentile_lower"] ** 2
        )
        # plot gaia sources on archival image
        nax, archival_img = plot_gaia_sources(
            target_coord,
            gaia_sources,
            verbose=verbose,
            survey="DSS2 Blue",
            fov_rad=fov_rad,
            reticle=True,
            ax=ax2,
        )

        lcs = []
        corr_lcs = []
        flat_lcs = []
        trends = []
        folded_lcs = []
        raw_cdpps = []
        corr_cdpps = []
        masks = []
        gcas = []
        colors = ["#1f77b4", "#ff7f0e"]
        if sap_mask == "square" or sap_mask == "round":
            aper_args = aper_radii
        else:
            aper_args = percentiles
        for n, (color, aper_arg, axn) in enumerate(
            zip(colors, aper_args, [223, 224])
        ):
            print("\n----------aperture {}----------\n".format(n + 1))
            # make aperture mask
            mask = parse_aperture_mask(
                tpf,
                sap_mask=sap_mask,
                aper_radius=aper_arg,
                percentile=aper_arg,
                verbose=verbose,
            )
            # -----ax0-----#
            if n == 0:
                nax1 = tpf.plot(aperture_mask=mask, origin="lower", ax=ax1)
                ax1.text(
                    0.95,
                    0.10,
                    "mask={}".format(sap_mask),
                    verticalalignment="top",
                    horizontalalignment="right",
                    transform=nax1.transAxes,
                    color="w",
                    fontsize=12,
                )
                ax1.set_title("sector={}".format(sector), fontsize=FONTSIZE)

            # make lc
            # correct systematics/ filter long-term variability
            # see https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/correctors.py

            # -----ax1-----#
            contour = np.zeros((ny, nx))
            contour[np.where(mask)] = 1
            contour = np.lib.pad(contour, 1, PadWithZeros)
            highres = zoom(contour, 100, order=0, mode="nearest")
            extent = np.array([-1, nx, -1, ny])
            # aperture mask outline
            cs2 = ax2.contour(
                highres,
                levels=[0.5],
                extent=extent,
                origin="lower",
                colors=color,
                transform=nax.get_transform(tpfwcs),
            )

            raw_lc = tpf.to_lightcurve(method=PHOTMETHOD, aperture_mask=mask)
            raw_lc = raw_lc.remove_nans().remove_outliers().normalize()
            lcs.append(raw_lc)

            cadence_mask_tpf = make_cadence_mask(
                tpf.time, period, t0, t14, verbose=verbose
            )
            if np.any([use_pld,use_sff]):
                msg = "Applying systematics correction:\n".format(use_gp)
                if use_pld is not None:
                    msg += "using PLD (gp={})".format(use_gp)
                    if verbose:
                        logging.info(msg)
                        print(msg)
                    # pld = tpf.to_corrector(method='pld')
                    pld = lk.PLDCorrector(tpf)
                    corr_lc = (
                        pld.correct(
                            aperture_mask=mask,
                            use_gp=use_gp,
                            # True means cadence is considered in the noise model
                            cadence_mask=~cadence_mask_tpf,
                            # True means the pixel is chosen when selecting the PLD basis vectors
                            pld_aperture_mask=mask,
                            # gp_timescale=30, n_pca_terms=10, pld_order=2,
                        )
                        .remove_nans()
                        .remove_outliers()
                        .normalize()
                    )
                else:
                    # use_sff without restoring trend
                    msg += "using SFF\n"
                    if verbose:
                        logging.info(msg)
                        print(msg)
                    sff = lk.SFFCorrector(raw_lc)
                    corr_lc = (
                        sff.correct(
                            centroid_col=raw_lc.centroid_col,
                            centroid_row=raw_lc.centroid_row,
                            polyorder=5,
                            niters=3,
                            bins=SFF_BINSIZE,
                            windows=SFF_CHUNKSIZE,
                            sigma_1=3.0,
                            sigma_2=5.0,
                            restore_trend=True,
                        )
                        .remove_nans()
                        .remove_outliers()
                    )
                corr_lcs.append(corr_lc)
                # get transit mask of corr lc
                msg = "Flattening corrected light curve using Savitzky-Golay filter"
                if verbose:
                    logging.info(msg)
                    print(msg)
                cadence_mask_corr = make_cadence_mask(
                    corr_lc.time, period, t0, t14, verbose=False
                )
                # finally flatten
                t14_ncadences = t14*u.day.to(cadence_in_minutes)
                errmsg = f'use sg_filter_window> {t14_ncadences}'
                assert t14_ncadences<sg_filter_window_SC, errmsg
                flat_lc, trend = corr_lc.flatten(
                    window_length=SG_FILTER_WINDOW_SC,
                    mask=cadence_mask_corr,
                    return_trend=True,
                )
            else:
                if verbose:
                    msg = "Flattening raw light curve using Savitzky-Golay filter"
                    logging.info(msg)
                    print(msg)
                cadence_mask_raw = make_cadence_mask(
                    raw_lc.time, period, t0, t14, verbose=verbose
                )
                flat_lc, trend = raw_lc.flatten(
                    window_length=SG_FILTER_WINDOW_SC,
                    mask=cadence_mask_raw,
                    return_trend=True,
                )

            # remove obvious outliers and NaN in time
            raw_time_mask = ~np.isnan(raw_lc.time)
            raw_flux_mask = (raw_lc.flux > YLIMIT[0]) | (
                raw_lc.flux < YLIMIT[1]
            )
            raw_lc = raw_lc[raw_time_mask & raw_flux_mask]
            flat_time_mask = ~np.isnan(flat_lc.time)
            flat_flux_mask = (flat_lc.flux > YLIMIT[0]) & (
                flat_lc.flux < YLIMIT[1]
            )
            flat_lc = flat_lc[flat_time_mask & flat_flux_mask]

            if np.any([use_pld,use_sff]):
                trend = trend[flat_time_mask & flat_flux_mask]
            else:
                trend = trend[raw_time_mask & raw_flux_mask]

            flat_lcs.append(flat_lc)
            trends.append(trend)

            if verbose:
                print("Periodogram with TLS\n")
            t = flat_lc.time
            fcor = flat_lc.flux

            # TLS
            model = transitleastsquares(t, fcor)
            # get TIC catalog info: https://github.com/hippke/tls/blob/master/transitleastsquares/catalog.py
            # see defaults: https://github.com/hippke/tls/blob/master/transitleastsquares/tls_constants.py
            try:
                ((u1, u2), Ms_tic, _, _, Rs_tic, _, _) = catalog.catalog_info(
                    TIC_ID=int(ticid)
                )
                Teff_tic, logg_tic, _, _, _, _, _, _ = catalog.catalog_info_TIC(
                    int(ticid)
                )
                u1, u2 = DEFAULT_U if not np.all([u1, u2]) else [u1, u2]
                Rs_tic = 1.0 if Rs_tic is None else Rs_tic
                Ms_tic = 1.0 if Ms_tic is None else Ms_tic
            except:
                (u1, u2), Ms_tic, Rs_tic = (
                    DEFAULT_U,
                    1.0,
                    1.0,
                )  # assume G2 star
            if verbose:
                if u1 == DEFAULT_U[0] and u2 == DEFAULT_U[1]:
                    print("Using default limb-darkening coefficients\n")
                else:
                    print(
                        "Using u1={:.4f},u2={:.4f} based on TIC catalog\n".format(
                            u1, u2
                        )
                    )

            results = model.power(
                u=[u1, u2],
                limb_dark="quadratic",
                n_transits_min=N_TRANSITS_MIN,
            )
            results["u"] = [u1, u2]
            results["Rstar_tic"] = Rs_tic
            results["Mstar_tic"] = Ms_tic
            results["Teff_tic"] = Teff_tic

            if verbose:
                print(
                    "Odd-Even transit mismatch: {:.2f} sigma\n".format(
                        results.odd_even_mismatch
                    )
                )
                print(
                    "Best period from periodogram: {:.4f} {}\n".format(
                        results.period, u.day
                    )
                )

            # phase fold
            if ~np.any(pd.isnull([results.period, results.T0])):
                # check if TLS input are not np.nan or np.NaN or None
                fold_lc = flat_lc.fold(period=results.period, t0=results.T0)
            # elif ~np.any(pd.isnull([period,t0])):
            #     #check if TLS input are not np.nan or np.NaN or None
            #     fold_lc = flat_lc.fold(period=period, t0=t0)
            else:
                msg = (
                    "TLS period and t0 search did not yield useful results.\n"
                )
                logging.info(msg)
                raise ValueError(msg)

            # -----folded lc-----#
            ax = fig.add_subplot(axn)
            fold_lc.scatter(ax=ax, color="k", alpha=0.1, label="unbinned")
            fold_lc.bin(BINSIZE_SC).scatter(
                ax=ax, color=color, label="binned (10-min)"
            )
            ax.plot(
                results.model_folded_phase - 0.5,
                results.model_folded_model,
                color="red",
                label="TLS model",
            )
            # compare depths
            rprs = results["rp_rs"]
            ax.axhline(1 - rprs ** 2, 0, 1, color="k", linestyle="--")
            if sap_mask == "round":
                ax.set_title(
                    "{} mask (r={} pix)\n".format(sap_mask, aper_arg), pad=0.1
                )
            elif sap_mask == "square":
                ax.set_title(
                    "{0} mask ({1}x{1} pix)\n".format(sap_mask, aper_arg),
                    pad=0.1,
                )
            else:
                ax.set_title(
                    "{0} mask ({1}\%)\n".format(sap_mask, aper_arg), pad=0.1
                )

            # manually set ylimit for shallow transits
            if rprs <= 0.1:
                if n == 0:
                    ylo, yhi = 1 - 10 * rprs ** 2, 1 + 5 * rprs ** 2
                    ax.set_ylim(ylo, yhi if yhi < 1.02 else 1.02)
                elif n == 1:
                    ax.set_ylim(*gcas[0])
            t14 = results.duration * u.day.to(u.hour)
            t0 = results["T0"]

            star_source = "tic"
            rstar, teff = Rs_tic, Teff_tic
            if str(rstar) != "nan":
                star_source = "gaia"
                rstar, teff = Rs_gaia, Teff_gaia
            Rp = rprs * rstar * u.Rsun.to(u.Rearth)

            text1 = "Rp/Rs={:.4f}\nt14={:.2f} hr\nt0={:.6f}".format(
                rprs, t14, t0
            )
            text2 = "Source: {}\nRs={:.2f} Rsun\nTeff={:.0f} K\nRp={:.2f} Re".format(
                star_source, rstar, teff, Rp
            )
            if verbose:
                print(f"{text1}\n\n{text2}")
            ax.text(
                0.3,
                0.25,
                text1,
                verticalalignment="top",
                horizontalalignment="left",
                transform=ax.transAxes,
                color="g",
                fontsize=FONTSIZE,
            )
            ax.text(
                0.6,
                0.3,
                text2,
                verticalalignment="top",
                horizontalalignment="left",
                transform=ax.transAxes,
                color="g",
                fontsize=FONTSIZE,
            )
            ax.set_xlim(-0.2, 0.2)
            ax.legend(title="phase-folded lc")
            ax.legend(loc=3)
            gcas.append(ax.get_ylim())

        if toi or toiid:
            id = toi if toi is not None else toiid
            figname = f"TIC{tic}_TOI{id}_FOV_s{sector}_pla.png"
            pl.suptitle(f"TIC {ticid} (TOI {id})", fontsize=FONTSIZE)
        else:
            ticid = tpf.targetid
            figname = f"TIC{ticid}_FOV_s{sector}_pla.png"
            pl.suptitle(f"TIC {ticid}", fontsize=FONTSIZE)
        figoutpath = join(figoutdir, figname)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if savefig:
            fig.savefig(figoutpath, bbox_inches="tight")
            print(f"Saved: {figoutpath}\n")
        else:
            pl.show()
        end = time.time()
        msg = "#----------Runtime: {:.2f} s----------#\n".format(end - start)
        if verbose:
            logging.info(msg)
            print(msg)
        pl.close()

    except:
        print(f"Error occured:\n{traceback.format_exc()}")
        print_recommendations()


def getDistance(x1, y1, x2, y2):
    """Get pythagorean distance"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def isInside(border, target):
    """Check if star is inside aperture mask"""
    degree = 0
    for i in range(len(border) - 1):
        a = border[i]
        b = border[i + 1]

        # calculate distance of vector
        A = getDistance(a[0], a[1], b[0], b[1])
        B = getDistance(target[0], target[1], a[0], a[1])
        C = getDistance(target[0], target[1], b[0], b[1])

        # calculate direction of vector
        ta_x = a[0] - target[0]
        ta_y = a[1] - target[1]
        tb_x = b[0] - target[0]
        tb_y = b[1] - target[1]

        cross = tb_y * ta_x - tb_x * ta_y
        clockwise = cross < 0

        # calculate sum of angles
        if clockwise:
            degree = degree + math.degrees(
                math.acos((B * B + C * C - A * A) / (2.0 * B * C))
            )
        else:
            degree = degree - math.degrees(
                math.acos((B * B + C * C - A * A) / (2.0 * B * C))
            )

    if abs(round(degree) - 360) <= 3:
        return True
    return False


def impact_parameter(a, inc):
    """
    a : [Rs]
    inc : [rad]
    """
    return a * np.cos(inc)


def tshape_approx(a, k, b):
    """
    Seager & Mallen-Ornelas 2003, eq. 15
    """
    i = np.arccos(b / a)
    alpha = (1 - k) ** 2 - b ** 2
    beta = (1 + k) ** 2 - b ** 2
    return np.sqrt(alpha / beta)


def max_k(tshape):
    """
    Seager & Mallen-Ornelas 2003, eq. 21
    """
    return (1 - tshape) / (1 + tshape)


def apply_dmag(delta_prime, t12, t13):
    """
    compute the maximumum delta mag to reproduce the observed depth

    See Sec. 4 of Vanderburg+2019:
    delta_prime < (t12/t13)^2 gamma
    where the contamination factor
    gamma = Fsource/Ftotal = 10^(-0.4*dmag)

    Parameters
    ----------
    delta_prime : float
        observed/apparent depth
    t12 : float
        time between first and second contact (ingress duration)
    t13 : float
        time between first and third contact (egress duration)

    Returns
    -------
    dmag_max : float
        maximum delta mag between host and contaminating source
    """
    dmag_max = 2.5 * np.log10(t12 ** 2 / (delta_prime * t13 ** 2))
    return dmag_max


def print_recommendations():
    print("\n-----------Some recommendations-----------\n")
    print(
        "Try -c if [buffer is too small for requested array] or to update TOI list"
    )
    print("Try -no_gp if [MemoryError: std::bad_alloc]")
    print(
        "Try --aper={pipeline,threshold,percentile,all} if [assert k <= min(m, n)] or tpf seems corrupted"
    )
    print(
        "Re-run tql if [json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)]"
    )
    print(
        "Re-run tql if [TypeError: unsupported format string passed to NoneType.__format__]"
    )
    print(
        "Check internet connection if [Failed to establish a new connection]\n"
    )


def catch_IERS_warning():
    import warnings

    # Set up the thing to catch the warning (and potentially others)
    with warnings.catch_warnings(record=True) as w:
        # import the modules
        # from astroplan import Observer
        # from astroplan import OldEarthOrientationDataWarning
        # One want to know aout the first time a warning is thrown
        warnings.simplefilter("once")

    # Look through all the warnings to see if one is OldEarthOrientationDataWarning,
    # update the table if it is.
    for i in w:
        if i.category == OldEarthOrientationDataWarning:
            # This new_mess statement isn't really needed I just didn't want to print
            #  all the information that is produce in the warning.
            new_mess = ".".join(str(i.message).split(".")[:3])
            print("WARNING:", new_mess)
            print("Updating IERS bulletin table...")
            from astroplan import download_IERS_A

            download_IERS_A()


def get_cluster_members_Bouma2019(dataloc="../data/"):
    """Bouma+2019 open cluster, young association, and young star catalogs
    """
    df = pd.read_csv(
        dataloc + "TablesBouma2019/OC_MG_FINAL_v0.3_publishable.csv",
        header=0,
        sep=";",
    )
    df = df.rename(
        columns={"cluster": "clusters", "unique_cluster_name": "Cluster"}
    )
    # remove negative parallaxes for the meantime
    # see: https://arxiv.org/pdf/1804.09366.pdf
    df = df[df["parallax"] > 0]

    icrs = SkyCoord(
        ra=df["ra"].values * u.deg,
        dec=df["dec"].values * u.deg,
        distance=Distance(parallax=df["parallax"].values * u.mas),
        #                 radial_velocity=df['RV'].values*u.km/u.s,
        pm_ra_cosdec=df["pmra"].values * u.mas / u.yr,
        pm_dec=df["pmdec"].values * u.mas / u.yr,
        frame="fk5",
        equinox="J2000.0",
    )
    gal = icrs.transform_to("galactic")
    df["gal_l"] = gal.l.deg
    df["gal_b"] = gal.b.deg
    df["distance"] = gal.distance
    df["gal_pm_b"] = gal.pm_b
    df["gal_pm_l_cosb"] = gal.pm_l_cosb
    # galactocentric
    xyz = gal.galactocentric
    df["X"] = xyz.x
    df["Y"] = xyz.y
    df["Z"] = xyz.z
    df["U"] = xyz.v_x
    df["V"] = xyz.v_y
    df["W"] = xyz.v_z

    # abs G magnitude
    df["bp_rp"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]
    df["abs_gmag"] = (
        df["phot_g_mean_mag"] - 5.0 * (np.log10(df["distance"])) - 1
    )
    df["abs_gmag"].unit = u.mag
    return df


def get_clusters_Bouma2019(dataloc="../data/"):
    d = get_cluster_members_Bouma2019(dataloc=dataloc)
    # count unique cluster group members
    g = d.groupby("Cluster").groups
    members = pd.Series({k: len(g[k]) for k in g.keys()}, name="count")
    df = pd.pivot_table(d, index=["Cluster"], aggfunc=np.median)
    df = df.drop("source_id", axis=1)
    df = pd.merge(df, members, left_index=True, right_index=True)
    return df


def plot_rdp_pmrv(df, target_gaia_id=None, target_label=None):
    fig, axs = pl.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    ax = axs.flatten()

    n = 0
    x, y = "ra", "dec"
    df.plot.scatter(x=x, y=y, ax=ax[n])
    if target_gaia_id is not None:
        idx = df.source_id.isin([target_gaia_id])
        assert sum(idx) > 0, "gaia_id not in df"
        ax[n].plot(
            df.loc[idx, x],
            df.loc[idx, y],
            marker=r"$\star$",
            c="y",
            ms="25",
            label=target_label,
        )
    ax[n].set_xlabel("R.A. [deg]")
    ax[n].set_ylabel("Dec. [deg]")
    text = len(df[["ra", "dec"]].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    if target_label is not None:
        ax[n].legend()
    n = 1
    par = "parallax"
    df[par].plot.kde(ax=ax[n])
    if target_gaia_id is not None:
        idx = df.source_id.isin([target_gaia_id])
        assert sum(idx) > 0, "gaia_id not in df"
        ax[n].axvline(
            df.loc[idx, par].values[0],
            0,
            1,
            c="k",
            ls="--",
            label=target_label,
        )
        if target_label is not None:
            ax[n].legend()
    ax[n].set_xlabel("Parallax [mas]")
    text = len(df[par].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    n = 2
    x, y = "pmra", "pmdec"
    df.plot.scatter(x=x, y=y, ax=ax[n])
    if target_gaia_id is not None:
        idx = df.source_id.isin([target_gaia_id])
        assert sum(idx) > 0, "gaia_id not in df"
        ax[n].plot(
            df.loc[idx, x], df.loc[idx, y], marker=r"$\star$", c="y", ms="25"
        )
    ax[n].set_xlabel("PM R.A. [deg]")
    ax[n].set_ylabel("PM Dec. [deg]")
    text = len(df[["pmra", "pmdec"]].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    n = 3
    par = "radial_velocity"
    try:
        df[par].plot.kde(ax=ax[n])
        if target_gaia_id is not None:
            idx = df.source_id.isin([target_gaia_id])
            assert sum(idx) > 0, "gaia_id not in df"
            ax[n].axvline(
                df.loc[idx, par].values[0],
                0,
                1,
                c="k",
                ls="--",
                label=target_label,
            )
        ax[n].set_xlabel("RV [km/s]")
        text = len(df[par].dropna())
        ax[n].text(
            0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        )
    except Exception as e:
        print(e)
        raise ValueError(
            f"radial_velocity is not available in {self.catalog_name}"
        )
    return fig


def plot_xyz_uvw(df, target_gaia_id=None):
    fig, axs = pl.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    ax = axs.flatten()

    n = 0
    for (i, j) in itertools.combinations(["X", "Y", "Z"], r=2):
        if target_gaia_id is not None:
            idx = df.source_id.isin([target_gaia_id])
            assert sum(idx) > 0, "gaia_id not in df"
            ax[n].plot(
                df.loc[idx, i],
                df.loc[idx, j],
                marker=r"$\star$",
                c="y",
                ms="25",
            )
        df.plot.scatter(x=i, y=j, ax=ax[n])
        ax[n].set_xlabel(i + " [pc]")
        ax[n].set_ylabel(j + " [pc]")
        text = len(df[[i, j]].dropna())
        ax[n].text(
            0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        )
        n += 1

    n = 3
    for (i, j) in itertools.combinations(["U", "V", "W"], r=2):
        if target_gaia_id is not None:
            idx = df.source_id.isin([target_gaia_id])
            assert sum(idx) > 0, "gaia_id not in df"
            ax[n].plot(
                df.loc[idx, i],
                df.loc[idx, j],
                marker=r"$\star$",
                c="y",
                ms="25",
            )
        df.plot.scatter(x=i, y=j, ax=ax[n])
        ax[n].set_xlabel(i + " [km/s]")
        ax[n].set_ylabel(j + " [km/s]")
        text = len(df[[i, j]].dropna())
        ax[n].text(
            0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        )
        n += 1

    return fig


def plot_hrd(df, target_gaia_id=None, target_label=None, figsize=(8, 8)):
    fig, ax = pl.subplots(1, 1, figsize=figsize)
    if target_gaia_id is not None:
        idx = df.source_id.isin([target_gaia_id])
        assert sum(idx) > 0, "gaia_id not in df"
        ax.plot(
            df.loc[idx, "bp_rp"],
            df.loc[idx, "abs_gmag"],
            marker=r"$\star$",
            c="y",
            ms="25",
            label=target_label,
        )
        if target_label is not None:
            ax.legend()
    df.plot.scatter(ax=ax, x="bp_rp", y="abs_gmag", marker=".")
    ax.set_xlabel(r"Bp $-$ Rp", fontsize=16)
    ax.set_ylabel(r"M$_{\mathrm{G}}$", fontsize=16)
    pl.gca().invert_yaxis()

    text = len(df[["bp_rp", "abs_gmag"]].dropna())
    pl.text(0.8, 0.9, f"n={text}", fontsize=14, transform=pl.gca().transAxes)
    return fig


def get_2167_open_clusters(dataloc="../data/TablesDias2014/"):
    """Dias+2004-2015; compiled until 2016:
    https://ui.adsabs.harvard.edu/abs/2014yCat....102022D/abstract"""
    fp = join(dataloc, "2167_open_clusters_and_candidates.tsv")
    df = pd.read_csv(fp, delimiter="\t", comment="#")
    return df


def get_1229_open_clusters(dataloc="../data/TablesCantatGaudin2018/"):
    """Cantat-Gaudin+2018:
    http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/618/A93"""
    fp = join(dataloc, "Table1_1229_open_clusters.tsv")
    df = pd.read_csv(fp, delimiter="\t", comment="#")
    coords = SkyCoord(ra=df["RAJ2000"], dec=df["DEJ2000"], unit="deg")
    df.Cluster = df.Cluster.apply(lambda x: x.strip())
    df["RA"] = coords.ra.deg
    df["Dec"] = coords.dec.deg
    return df


def get_1229_open_cluster_members(dataloc="../data/TablesCantatGaudin2018/"):
    """Cantat-Gaudin+2018:
    http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/618/A93"""
    fp = join(dataloc, "Membership probabilities of all individual stars.tsv")
    df = pd.read_csv(fp, delimiter="\t", comment="#")
    df.Cluster = df.Cluster.apply(lambda x: x.strip())
    return df


def get_269_open_clusters(dataloc="../data/"):
    """Bossini+2019"""
    fp = join(dataloc, "Bossini2019_269_open_clusters.tsv")
    df = pd.read_table(fp, delimiter="\t", skiprows=69, comment="#")
    # convert distance and age
    df["D_est[pc]"] = Distance(distmod=df["Dist"]).pc
    df["Age[Myr]"] = 10 ** df["logA"] / 1e6
    # rename Dist to Dmod (distance modulus)
    df["Cluster"] = df.Cluster.apply(lambda x: x.strip())
    df[["Cluster", "Age[Myr]", "D_est[pc]"]].sort_values(by="Age[Myr]")
    df = df.rename(
        columns={
            "Dist": "Dmod",
            "e_Dist": "e_Dmod",
            "E_Dist": "E_Dmod",
            #'RA_ICRS': 'RA', 'DE_ICRS':'Dec'
        }
    )
    return df


def get_open_clusters(dataloc="../data/TablesGaiaDR2HRDpaper/"):
    """summary table of 32 open clusters"""
    fp = join(dataloc, "Table2_32 open clusters.csv")
    df_open = pd.read_csv(fp, delimiter=",", comment="#")
    df_open.Cluster = df_open.Cluster.apply(lambda x: x.replace("_", ""))
    df_open["D_est[pc]"] = Distance(distmod=df_open["DM"]).pc
    df_open["Age[Myr]"] = 10 ** df_open["log(age)"] / 1e6
    return df_open


def get_NGC_clusters(dataloc="../data/TablesGaiaDR2HRDpaper/"):
    """summary table of 14 globular clusters"""
    fp = join(dataloc, "Table3_14 globular clusters.csv")
    df_glob = pd.read_csv(fp, delimiter=" ", comment="#")
    df_glob["Cluster"] = df_glob.NGC.apply(lambda x: "NGC" + str(x).strip())
    return df_glob


def get_open_cluster_members_near(dataloc="../data/TablesGaiaDR2HRDpaper/"):
    # <250pc
    fp = join(dataloc, "TableA1a_9 open cluster members within 250 pc.csv")
    df = pd.read_csv(fp, delimiter=",", comment="#")
    df.columns = [c.strip() for c in df.columns]
    df.Cluster = df.Cluster.apply(lambda x: x.strip())
    return df


def get_open_cluster_members_far(dataloc="../data/TablesGaiaDR2HRDpaper/"):
    # >250pc
    fp = join(dataloc, "TableA1b_37 open cluster members beyond 250 pc.csv")
    df = pd.read_csv(fp, delimiter=",", comment="#")
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df.columns = [c.strip() for c in df.columns]
    df.Cluster = df.Cluster.apply(lambda x: x.strip())
    return df


def get_open_clusters_near(dataloc="../data/TablesGaiaDR2HRDpaper/"):
    # <250pc
    fp = join(
        dataloc, "Table3_Mean parameters for 9 open clusters within 250pc.tsv"
    )
    df = pd.read_csv(fp, delimiter="\t", comment="#")
    df.columns = [c.strip() for c in df.columns]
    df.Cluster = df.Cluster.apply(lambda x: x.strip())
    return df


def get_open_clusters_far(dataloc="../data/TablesGaiaDR2HRDpaper/"):
    # >250pc
    fp = join(
        dataloc,
        "TableA4_Mean parameters for 37 open clusters beyond 250pc.tsv",
    )
    df = pd.read_csv(fp, delimiter="\t", comment="#")
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df.columns = [c.strip() for c in df.columns]
    df.Cluster = df.Cluster.apply(lambda x: x.strip())
    return df


def get_open_cluster_members_far_parallax(
    dataloc="../data/TablesGaiaDR2HRDpaper/", save_csv=True
):
    """gaia ari provides plx values for d>250pc cluster members"""

    df = get_open_cluster_members_far()
    plx = []
    for (r, d, m) in tqdm(df[["ra", "dec", "Gmag"]].values):
        coord = SkyCoord(ra=r, dec=d, unit="deg")
        g = Gaia.query_object(coord, radius=10 * u.arcsec).to_pandas()
        gcoords = SkyCoord(ra=g["ra"], dec=g["dec"], unit="deg")
        # FIXME: get minimum or a few stars around the minimum?
        idx = coord.separation(gcoords).argmin()
        if abs(m - g.loc[idx, "g_mean_mag"]) > 1.0:
            star = g.loc[idx]
        else:
            star = g.loc[1]

        plx.append(star["parallax"].values[0])
    if save_csv:
        df["plx"] = plx
        fp = "open_cluster_members_far_parallax.csv"
        df_open_mem_concat.to_csv(join(dataloc, fp), index=False)
        print(f"Saved: {fp}")
    return df


def combine_open_clusters(dataloc="../data/", clobber=False, save_csv=False):
    """ Gaia DR2 open cluster + 269 clusters with ages"""
    fp = join(dataloc, "merged_open_clusters.csv")
    if not exists(fp) or clobber:
        df1 = tql.get_269_open_clusterss(dataloc=dataloc)
        df2 = tql.get_open_clusters()
        df2 = df2.rename(columns={"RA": "RA_ICRS", "Dec": "DE_ICRS"})
        df2.Cluster = df2.Cluster.apply(lambda x: x.replace(" ", ""))
        df_all = pd.merge(
            left=df1[
                ["Cluster", "RA_ICRS", "DE_ICRS", "D_est[pc]", "Age[Myr]"]
            ],
            right=df2[
                ["Cluster", "RA_ICRS", "DE_ICRS", "D_est[pc]", "Age[Myr]"]
            ],
            on="Cluster",
            how="outer",
        )
        print("\n=====Edit columns=====\n")
        if save_csv:
            df_all.to_csv(fp)
            print(f"Saved: {fp}")
    else:
        df_all = pd.read_csv(fp)
    return df_all


def combine_open_cluster_members_near_far(
    dataloc="../data/TablesGaiaDR2HRDpaper/", save_csv=False
):
    # make uniform column
    df_open_near_mem = get_open_cluster_members_near(dataloc)
    df_open_far_mem = get_open_cluster_members_far(dataloc)
    # concatenate
    df_open_mem_concat = pd.concat(
        [df_open_near_mem, df_open_far_mem], sort=True, join="outer"
    )
    if save_csv:
        fp = "open_cluster_members.csv"
        df_open_mem_concat.to_csv(join(dataloc, fp), index=False)
        print(f"Saved: {fp}")
    return df_open_mem_concat


def combine_open_clusters_near_far(
    dataloc="../data/TablesGaiaDR2HRDpaper/", save_csv=False
):
    # make uniform column
    df_open_near = get_open_clusters_near(dataloc)
    df_open_far = get_open_clusters_far(dataloc)
    df_open_far["RA_ICRS"] = df_open_far["RAJ2000"]
    df_open_far["DE_ICRS"] = df_open_far["DEJ2000"]
    df_open_far = df_open_far.drop(["RAJ2000", "DEJ2000"], axis=1)

    # concatenate
    df_open_concat = pd.concat(
        [df_open_near, df_open_far], sort=True, join="outer"
    )
    if save_csv:
        fp = "open_clusters.csv"
        df_open_concat.to_csv(join(dataloc, fp), index=False)
        print(f"Saved: {fp}")
    return df_open_concat


def compute_separation_from_clusters(target_coord, sep_3d=True, verbose=False):
    """compute 3d separation between target and all known clusters"""
    if target_coord.distance.value == 1.0:
        target_coord = get_target_coord_3d(target_coord, verbose=verbose)

    # try:
    #     #merged Bossini+2019 & gaia DR2 paper
    #     df = pd.read_csv('../data/merged_open_clusters.csv')
    #     catalog = SkyCoord(ra=df['RA_ICRS'].values*u.deg,
    #                   dec=df['DE_ICRS'].values*u.deg,
    #                   distance=df['D_est[pc]'].values*u.pc,
    #                   #radial_velocity=df['RV'].values*u.km/u.s,
    #                   #pm_ra_cosdec=df['pmRA'].values*u.mas/u.yr,
    #                   #pm_dec=df['pmDE'].values*u.mas/u.yr,
    #                   frame='icrs'
    #                   )
    # except:
    #     #using gaia DR2 paper
    #     df = pd.read_csv('../data/TablesGaiaDR2HRDpaper/open_clusters.csv')
    #     # df = combine_open_clusters_near_far()
    #     catalog = SkyCoord(ra=df['RA_ICRS'].values*u.deg,
    #                   dec=df['DE_ICRS'].values*u.deg,
    #                   distance=Distance(parallax=df['plx'].values*u.mas),
    #                   #radial_velocity=df['RV'].values*u.km/u.s,
    #                   #pm_ra_cosdec=df['pmRA'].values*u.mas/u.yr,
    #                   #pm_dec=df['pmDE'].values*u.mas/u.yr,
    #                   frame='icrs'
    #                   )
    df = get_1229_open_clusters()
    catalog = SkyCoord(
        ra=df["RA"].values * u.deg,
        dec=df["Dec"].values * u.deg,
        distance=Distance(parallax=df["plx"].values * u.mas),
        #                      pm_ra_cosdec=df0['pmRA'].values*u.mas/u.yr,
        #                      pm_dec=df0['pmDE'].values*u.mas/u.yr
    )

    if sep_3d:
        return catalog.separation_3d(target_coord)
    else:
        raise NotImplementedError


def get_cluster_near_target(
    target_coord,
    distance=None,
    unit=u.pc,
    sep_3d=True,
    catalog="CantatGaudin2018",
    verbose=False,
):
    """get nearest cluster to target within specified distance"""
    if target_coord.distance.value == 1.0:
        target_coord = get_target_coord_3d(target_coord, verbose=verbose)
    # FIXME: include clusters not only open
    # try:
    #     df = pd.read_csv('../data/merged_open_clusters.csv')
    # except:
    #     df = pd.read_csv('../data/TablesGaiaDR2HRDpaper/open_clusters.csv')
    #     # df = combine_open_clusters_near_far()
    if catalog == "CantatGaudin2018":
        df = get_1229_open_clusters()
    elif catalog == "Bouma2019":
        df = get_clusters_Bouma2019()

    catalog_sep = compute_separation_from_clusters(target_coord, sep_3d=sep_3d)
    if distance is not None:
        idx = catalog_sep < distance * unit
        cluster = df.Cluster.loc[idx].values
    else:
        idx = catalog_sep.argmin()
        cluster = df.Cluster.iloc[idx]
    sep = catalog_sep[idx]
    # if verbose:
    #     print('Nearest cluster to target: {} (d={:.2f})'.format(cluster[0], sep[0]))
    return (cluster, sep)


def get_target_coord(toi=None, tic=None, name=None):
    """get target coordinate
    """
    # TIC
    if (toi is not None) or (tic is not None):
        toi = get_toi(toi=toi, tic=tic, clobber=True, verbose=False)
        target_coord = SkyCoord(
            ra=toi["RA"].values[0],
            dec=toi["Dec"].values[0],
            unit=(u.hourangle, u.degree),
        )
    # name resolver
    else:
        target_coord = SkyCoord.from_name(name)
    return target_coord


def get_target_coord_3d(target_coord, verbose=False):
    """append distance to target coordinate"""
    if verbose:
        print("Querying parallax of target from Gaia\n")
    g = Gaia.query_object(target_coord, radius=10 * u.arcsec).to_pandas()
    gcoords = SkyCoord(ra=g["ra"], dec=g["dec"], unit="deg")
    # FIXME: get minimum or a few stars around the minimum?
    idx = target_coord.separation(gcoords).argmin()
    star = g.loc[idx]
    # get distance from parallax
    target_dist = Distance(parallax=star["parallax"] * u.mas)
    # redefine skycoord with coord and distance
    target_coord = SkyCoord(
        ra=target_coord.ra, dec=target_coord.dec, distance=target_dist
    )
    return target_coord


def get_cluster_members_near_target(
    target_coord, distance=50, unit=u.pc, verbose=False
):
    """get cluster members to target within specified distance
    target_coord : target coordinates
    distance : target's 3d distance from nearest cluster
    """
    if target_coord.distance.value == 1.0:
        target_coord = get_target_coord_3d(target_coord, verbose=verbose)

    # get clusters and separation
    (cluster, sep) = get_cluster_near_target(
        target_coord, distance=distance, unit=unit, verbose=verbose
    )

    if len(cluster) > 0:
        # try:
        #     mem = pd.read_csv('../data/TablesGaiaDR2HRDpaper/open_cluster_members.csv')
        # except:
        #     mem = combine_open_cluster_members_near_far()
        mem = get_1229_open_cluster_members()
        idx = mem.Cluster.isin(cluster)
        # mcoord = SkyCoord(ra=m.loc[idx].ra.values*u.deg,
        #                  dec=m.loc[idx].dec.values*u.deg,
        #                  distance=Distance(parallax=m.loc[idx].par.values*u.mas))
        return mem[idx]
    else:
        # raise ValueError('target not near any known clusters')
        pass


def get_cluster_diameter(coords, verbose=False):
    """get size of cluster in pc by getting mutual 3d-separation"""
    # FIXME: this gets only ra,dec-extreme positions; search parallax too
    idxs = [
        np.nanargmin(coords.ra),
        np.nanargmax(coords.ra),
        np.nanargmin(coords.dec),
        np.nanargmax(coords.dec),
        np.nanargmin(coords.distance),
        np.nanargmax(coords.distance),
    ]
    max_sep = np.nanmax([coords[i].separation_3d(coords[idxs]) for i in idxs])
    if verbose:
        print("cluster diameter estimate: {:.2f} pc\n".format(max_sep))
    return max_sep * u.pc


def get_all_cluster_diameters(df=None):
    if df is None:
        try:
            df = pd.read_csv(
                "../data/TablesGaiaDR2HRDpaper/open_cluster_members.csv"
            )
        except:
            df = combine_open_cluster_members_near_far()

    cluster_diameters = {}
    for cluster, d in df.groupby(by="Cluster"):
        mcoords = SkyCoord(
            ra=d.ra.values * u.deg,
            dec=d.dec.values * u.deg,
            distance=Distance(parallax=d.par.values * u.mas),
        )
        max_sep = get_cluster_diameter(mcoords)
        cluster_diameters[cluster] = max_sep
    return cluster_diameters


def check_if_cluster_in_database(cluster, verbose=False):
    # FIXME:
    try:
        df = pd.read_csv("../data/TablesGaiaDR2HRDpaper/open_clusters.csv")
    except:
        df = combine_open_clusters_near_far()
    cnames = df.Cluster
    idx = cnames.isin(cluster)
    if idx.sum() > 0:
        print("{} is not in {}".format(cluster, cnames.tolist()))
    return cnames


def get_cluster_members_gaia_params(
    cluster_name, df, clobber=False, verbose=True, dataloc="../data"
):
    """query gaia params for each cluster member"""
    # fp=join(dataloc,f'TablesGaiaDR2HRDpaper/{cluster_name}_members.hdf5')
    fp = join(dataloc, f"{cluster_name}_members.hdf5")
    if not exists(fp) or clobber:
        gaia_data = {}
        for n in tqdm(df.index.tolist()):
            assert np.all(df.Cluster.isin([cluster_name]))
            # sid,ra,dec = df.loc[n,['SourceId','ra','dec']].values
            sid, ra, dec = df.loc[n, ["Source", "RA_ICRS", "DE_ICRS"]].values
            coord = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame="icrs")
            radius = u.Quantity(1, u.arcsec)
            g = Gaia.query_object(coordinate=coord, radius=radius)
            gaia_data[sid] = g.to_pandas()
        dd.io.save(fp, gaia_data)
        if verbose:
            print(f"Saved: {fp}")
    else:
        gaia_data = dd.io.load(fp)
        if verbose:
            print(f"Loaded: {fp}")

    df_gaia = pd.concat(gaia_data.values(), ignore_index=True)
    return df_gaia


def merge_gaia_params(df_open, df_gaia):
    """ """
    df_gaia_merged = pd.merge(
        df_open,
        df_gaia,
        how="inner",
        left_on="SourceId",
        right_on="source_id",
        left_index=False,
        right_index=False,
    )
    return df_gaia_merged


def check_parallax_difference(df_gaia, cluster_name):
    """applicable only for nearby open cluster members with known parallaxes"""
    df_open_near = get_open_cluster_members_near()
    idx = df_open_near.Cluster == cluster_name
    return df_open_near.loc[idx, "par"] - df_gaia["parallax"]


def check_gaia_id_in_cluster_members(target_gaia_id):
    df = combine_open_clusters_near_far()
    return df.SourceId.isin([target_gaia_id]).sum()


def compute_sigma(
    df_gaia,
    target_gaia_id,
    parameters=["parallax", "pmra", "pmdec", "radial_velocity"],
):
    sigmas = {}
    for p in parameters:
        idx = df_gaia.source_id == target_gaia_id
        val = df_gaia.loc[idx, p].values[0]
        sigma = (df_gaia[p].median() - val) / df_gaia[p].std()
        sigmas[p] = sigma
        print(p, sigma)
    return sigmas


def get_toi_coord_3d(toi, clobber=False, verbose=False):
    all_tois = get_tois(clobber=clobber, verbose=verbose)
    idx = all_tois["TOI"].isin([toi])
    columns = ["RA", "Dec", "Stellar Distance (pc)"]
    ra, dec, dist = all_tois.loc[idx, columns].values[0]
    target_coord = SkyCoord(
        ra=ra, dec=dec, distance=dist, unit=(u.hourangle, u.deg, u.pc)
    )
    return target_coord


def get_all_tois_gaia_params(dataloc="../data/", clobber=False, verbose=False):
    all_tois = get_tois(clobber=clobber, verbose=verbose)

    fp = join(dataloc, "all_toi_gaia_params.hdf5")
    if not exists(fp) or clobber:
        toi_gaia = {}
        for toi in tqdm(all_tois["TOI"].values):
            target_coord = get_toi_coord_3d(toi, verbose=False)
            g = Gaia.query_object(
                target_coord, radius=15 * u.arcsec
            ).to_pandas()
            toi_gaia[toi] = g
        dd.io.save(fp, toi_gaia)
        if verbose:
            print(f"Saved: {fp}")
    else:
        toi_gaia = dd.io.load(fp)
        if verbose:
            print(f"Loaded: {fp}")
    return toi_gaia


def get_tois_near_cluster(
    tois=None,
    distance=None,
    unit=u.pc,
    clobber=False,
    verbose=False,
    remove_known_planets=False,
):
    """

    """
    if tois is None:
        tois = get_tois(
            clobber=clobber,
            verbose=verbose,
            remove_known_planets=remove_known_planets,
        )

    toi_list = {}
    for toi in tqdm(tois["TOI"].values):
        target_coord = get_toi_coord_3d(toi)
        if distance is not None:
            cluster, sep = get_cluster_near_target(
                target_coord, distance=distance, unit=unit
            )
            if len(np.concatenate([cluster, sep])) > 1:
                toi_list[toi] = (cluster, sep)
        else:
            cluster, sep = get_cluster_near_target(target_coord)
            if len(cluster) > 0:
                toi_list[toi] = (cluster, sep.pc)
        if verbose:
            print(toi, cluster, sep)
    df = pd.DataFrame(toi_list).T
    df.columns = ["cluster", "distance"]
    return df


def plot_parallax_density(target_gaia_id, df, ax=None, verbose=True):
    # FIXME: par is for df_open, parallax is for df_gaia
    if ax is None:
        fig, ax = pl.subplots()

    # get median
    # vals,bins = np.histogram(df['par'],bins=100)
    # peak = vals[np.argmax(bins))]
    # ax.axvline(peak, 0, 1, color='k', linestyle='-')

    df["par"] = df["par"].dropna()
    density = gaussian_kde(df["par"])
    x = np.linspace(df["par"].min(), df["par"].max(), 100)
    ax.plot(x, density(x))

    idx = df.SourceId == target_gaia_id
    target_plx = df.loc[idx, "par"].values[0]
    ax.axvline(target_plx, 0, 1, color="k", linestyle="--")
    target_dist = Distance(parallax=target_plx * u.mas)
    text = "d={:.1f}".format(target_dist)
    ax.text(0.7, 0.9, text, fontsize=14, transform=ax.transAxes)
    ax.set_xlabel("Parallax [mas]")
    if verbose:
        sigma = (df["par"].median() - target_plx) / df["par"].std()
        print(sigma)
    return ax


def plot_rv_density(target_gaia_id, df_gaia, ax=None, verbose=True):

    if ax is None:
        fig, ax = pl.subplots()

    df_gaia = df_gaia["radial_velocity"].dropna()
    idx = df_gaia.source_id == target_gaia_id
    rv = df_gaia.loc[idx, "radial_velocity"].values[0]
    x = np.linspace(df_gaia.min(), df_gaia.max(), 100)
    ax.plot(x, density(x))
    ax.axvline(rv, 0, 1, color="k", linestyle="--")
    text = "RV={:.1f}".format(rv)
    ax.text(0.7, 0.8, text, fontsize=14, transform=ax[n + 1].transAxes)
    ax.set_xlabel("RV [km/s]")
    if verbose:
        sigma = (df["radial_velocity"].median() - rv) / df[
            "radial_velocity"
        ].std()
        print(sigma)
    return ax


def plot_target_in_cluster(
    target_gaia_id,
    df_gaia,
    params=["ra", "dec"],
    show_centroid=True,
    ax=None,
    cluster_name=None,
    verbose=True,
):
    """scatter plot of either cluster member positions or proper motions"""
    if ax is None:
        fig, ax = pl.subplots()

    df_gaia.plot.scatter(x=params[0], y=params[1], ax=ax)
    member_count = "n={}".format(len(df_gaia[params].dropna()))
    ax.text(0.8, 0.9, member_count, fontsize=14, transform=ax.transAxes)

    if show_centroid:
        r, d = df_gaia[params].median.values
        ax.plot(pr, pd, "ro", label="centroid")

    idx = df_gaia.source_id == target_gaia_id
    pmra, pmdec = df_gaia.loc[idx, params].values[0]
    ax.plot(pmra, pmdec, marker=r"$\star$", c="y", ms="25", label="target")
    if cluster_name is not None:
        ax.set_title(cluster_name)
    pl.legend()
    if verbose:
        sigma1 = (df[params[0]].median() - r) / df[params[0]].std()
        sigma2 = (df[params[1]].median() - d) / df[params[1]].std()
        print(params, sigma1, sigma2)
    return ax


def plot_cluster_membership(
    target_coord,
    cluster=None,
    target_gaia_id=None,
    min_cluster_diameter=100,
    verbose=False,
    figoutdir=".",
    savefig=False,
):
    """basic cluster membership plots"""
    # min_cluster_diameter = get_cluster_diameter(coords).value
    if cluster is None:
        df = get_cluster_members_near_target(
            target_coord, distance=2 * min_cluster_diameter, unit=u.pc
        )
    else:
        try:
            mem = pd.read_csv(
                "../data/TablesGaiaDR2HRDpaper/open_cluster_members.csv"
            )
        except:
            mem = combine_open_cluster_members_near_far()
        idx = mem.Cluster.isin(cluster)
        df = mem[idx]

    if target_gaia_id is None:
        g = Gaia.query_object(target_coord, radius=1 * u.arcsec).to_pandas()
        target_gaia_id = g.source_id.values[0]

    # check if 3d-separation is smaller than cluster size
    (cluster, sep) = get_cluster_near_target(
        target_coord,
        distance=2 * min_cluster_diameter,
        unit=u.pc,
        sep_3d=True,
        verbose=verbose,
    )
    #
    params = ["ra", "dec", "parallax"]
    if not np.all(df.isin(params)):
        df_gaia = get_cluster_members_gaia_params(
            cluster[0], verbose=True, dataloc="../data/TablesGaiaDR2HRDpaper/"
        )
        df_gaia = merge_gaia_params(df, df_gaia)

    # idx = df.SourceId==target_gaia_id
    # ra,dec,plx = df.loc[idx,params].values[0]
    # target_coord = SkyCoord(ra=ra*u.deg,dec=dec*u.deg,
    #                         distance=Distance(parallax=plx*u.mas))

    # get coordinates of cluster members
    mcoords = SkyCoord(
        ra=df_gaia.ra.values * u.deg,
        dec=df_gaia.dec.values * u.deg,
        distance=Distance(parallax=df_gaia.parallax.values * u.mas),
    )
    # estimate cluster size
    min_cluster_diameter = get_cluster_diameter(mcoords)

    # FIXME: sep[0]?
    if sep[0] > min_cluster_diameter:
        raise ValueError(
            f"{target_coord} > diameter of {cluster[0]} (min_cluster_diameter)"
        )
    # if ax is None:
    fig, ax = pl.subplots(2, 2, figsize=(10, 10))

    # position
    n = 0
    plot_target_in_cluster(
        target_gaia_id,
        df_gaia,
        params=["ra", "dec"],
        show_centroid=True,
        ax=ax[n],
        cluster_name=cluster[0],
    )
    # kernel density
    n = 1
    plot_parallax_density(target_gaia_id, df, ax=ax[n])
    # proper motion
    n = 2
    plot_target_in_cluster(
        target_gaia_id,
        df_gaia,
        params=["pmra", "pmdec"],
        show_centroid=True,
        ax=ax[n],
        cluster_name=None,
    )
    # radial radial_velocity
    n = 3
    plot_rv_density(target_gaia_id, df_gaia, ax=ax[n])

    if savefig:
        figname = join(figoutdir, f"_{cluster[0]}_membership.png")
        fig.savefig(figname, bbox_inches="tight")
        print(f"Saved: {figname}")
        pl.close()
    else:
        pl.show()
    return None


def distance_modulus(d):
    """
    dmag = M-m
    """
    dmag = 5 * np.log10(d) - 5
    return dmag


def dmag2dist(dmag):
    return 10 ** ((dmag + 5) / 5)


def FFI_video_urls():
    return "https://www.youtube.com/user/ethank18/videos"
