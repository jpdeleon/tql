#!/usr/bin/env python
import sys
import os
from os.path import join, exists
import traceback
import itertools
import warnings
from glob import glob
import matplotlib.colors as mcolors
import matplotlib.pyplot as pl
import numpy as np
import time
import logging
from imp import reload
import pandas as pd
from tqdm import tqdm
import eleanor as el
import lightkurve as lk
from astropy import units as u
from scipy.ndimage import zoom, rotate
# from astroquery.gaia import Gaia
from astroplan import FixedTarget
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.mast import Catalogs
from astroplan.plots import plot_finder_image
import deepdish as dd
from matplotlib.gridspec import GridSpec
from transitleastsquares import transitleastsquares, catalog
#from transitleastsquares.tls_constants import DEFAULT_U
import getpass
if getpass.getuser()=='muscat':
#    #non-GUI back-end
    pass
    #import matplotlib; matplotlib.use('agg')

MISSION        = 'TESS'
TESS_JD_offset = 2457000
#Savitzky-Golay filter window size (odd)
SG_FILTER_WINDOW_SC = 361    #short-cadence: 361x2min = 722min= 12 hr
SG_FILTER_WINDOW_LC = 11     #long-cadence:  25x30min = 750min = 12.5 hr
TESS_pix_scale      = 21*u.arcsec #/pix
FFI_CUTOUT_SIZE     = 5          #pix
PHOTMETHOD     = 'aperture'  #or 'prf'
# APPHOTMETHOD  =  'pipeline'  or 'all' or threshold --> uses tpf.extract_aperture_photometry
SFF_CHUNKSIZE  = 27          #27 chunks for a 27-day baseline
                             #there is a 3-day gap in all TESS dataset due to data downlink
                             #use chunksize larger than the gap i.e. chunksize>27/3
SFF_BINSIZE    = 360         #0.5 day for 2-minute cadence
#QUALITY_FLAGS  = lk.utils.TessQualityFlags()
quality_bitmask= 'hard'      #or default?
time_format    = 'btjd'
time_scale     = 'tdb'       #'tt', 'ut1', or 'utc'
PGMETHOD       = 'lombscargle' # 'boxleastsquares'
IMAGING_SURVEY = 'DSS2 Red'
FONTSIZE       = 16
LOG_FILENAME   = r'kurasta.log'
MAX_SECTORS    = 5           # sectors to analyze if target is osberved in multiple sectors
MULTISEC_BIN   = 10*u.min    # binning for very dense data (observed >MAX_SECTORS)
YLIMIT       = (0.8,1.2)   # flux limits
DEFAULT_U      = [0.4804, 0.1867] #quadratic limb darkening for a G2V star in the Kepler bandpass
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf', 'lime']
reload(logging)
logging.basicConfig(filename=LOG_FILENAME ,level=logging.DEBUG)


def get_tpf(targ_coord, tic=None, apphot_method='sap',
            apply_data_quality_mask=True,
            sector=None, verbose=False, clobber=False, sap_mask='pipeline',
            fitsoutdir='.', return_df=True):
    '''Download tpf from MAST given coordinates
       though using TIC id yields unique match.

    Parameters
    ----------
    targ_coord : astropy.coordinates
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
    '''
    # sector = None searches for all tpf; which sector to download is specified later
    if tic:
        ticstr = 'TIC {}'.format(tic)
        if verbose:
            print('\nSearching mast for {}\n'.format(ticstr))
        res = lk.search_targetpixelfile(ticstr, mission=MISSION, sector=None)
    else:
        if verbose:
            print('\nSearching mast for ra,dec=({})\n'.format(targ_coord.to_string()))
        res = lk.search_targetpixelfile(targ_coord, mission=MISSION, sector=None)
    df = res.table.to_pandas()

    if len(df)>0:
        all_sectors = [int(i) for i in df['sequence_number'].values]
        if sector:
            sector_idx = df['sequence_number'][df['sequence_number'].isin([sector])].index.tolist()
            if len(sector_idx)==0:
                sys.exit('sector {} data is unavailable'.format(sector))
            obsid = df.iloc[sector_idx]['obs_id'].values[0]
            ticid = int(df.iloc[sector_idx]['target_name'].values[0])
            fitsfilename = df.iloc[sector_idx]['productFilename'].values[0]
        else:
            sector_idx = 0
            sector = int(df.iloc[sector_idx]['sequence_number'])
            obsid = df.iloc[sector_idx]['obs_id']
            ticid = int(df.iloc[sector_idx]['target_name'])
            fitsfilename = df.iloc[sector_idx]['productFilename']

        msg='{} tpf(s) found in sector(s) {}\n'.format(len(df), all_sectors)
        msg+='Using data from sector {} only\n'.format(sector)
        if verbose:
            logging.info(msg); print(msg)

        filepath = join(fitsoutdir,'mastDownload/TESS',obsid,fitsfilename)
        if not exists(filepath) or clobber:
            if verbose:
                print('Downloading TIC {} ...\n'.format(ticid))
            ticstr = 'TIC {}'.format(ticid)
            res = lk.search_targetpixelfile(ticstr, mission=MISSION, sector=sector)
            tpf = res.download(quality_bitmask=quality_bitmask, download_dir=fitsoutdir)
        else:
            if verbose:
                print('Loading TIC {} from {}/...\n'.format(ticid,fitsoutdir))
            tpf = lk.TessTargetPixelFile(filepath)
        #assert tpf.mission == MISSION
        if apply_data_quality_mask:
            tpf = remove_bad_data(tpf, sector=tpf.sector)
        if return_df:
            return tpf, df
        else:
            return tpf
    else:
        msg='No tpf file found! Check FFI data using --cadence=long\n'
        logging.info(msg)
        sys.exit(msg)


def get_ffi_cutout(targ_coord=None, tic=None, sector=None, #cutout_size=10,
               apply_data_quality_mask=True,
               verbose=False, clobber=False, fitsoutdir='.', return_df=True):
    '''Download a tpf cutout from full-frame images.
       Caveat: stars from FFI do not have TIC id.
       Does Gaia id make a good proxy for TIC id?

    Parameters
    ----------
    targ_coord : astropy.coordinates
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
    '''
    if tic:
        ticstr = 'TIC {}'.format(tic)
        if verbose:
            print('\nSearching mast for {}\n'.format(ticstr))
        res = lk.search_tesscut(ticstr, sector=None)
        ticid = int(tic)
    else:
        if verbose:
            print('\nSearching mast for ra,dec=({})\n'.format(targ_coord.to_string()))
        res = lk.search_tesscut(targ_coord, sector=None)
        #search using eleanor
        ra, dec = targ_coord.ra.deg, targ_coord.dec.deg
        star = el.Source(coords=(ra,dec), sector=sector)
        ticid = int(star.tic)
        ticstr = 'TIC {}'.format(ticid)
    df = res.table.to_pandas()

    if len(df)>0:
        all_sectors = [int(i) for i in df['sequence_number'].values]
        if sector:
            sector_idx = df['sequence_number'][df['sequence_number'].isin([sector])].index.tolist()
            if len(sector_idx)==0:
                sys.exit('sector {} data is unavailable'.format(sector))
            target = df.iloc[sector_idx]['targetid'].values[0]
        else:
            sector_idx = 0
            sector = int(df.iloc[sector_idx]['sequence_number'])
            target = df.iloc[sector_idx]['targetid']

        msg='{} tpf(s) found in sector(s) {}\n'.format(len(df), all_sectors)
        msg+='Using data from sector {} only\n'.format(sector)
        if verbose:
            logging.info(msg); print(msg)

        sr = lk.SearchResult(res.table)
        filepath = sr._fetch_tesscut_path(target, sector, fitsoutdir, FFI_CUTOUT_SIZE)

        if not exists(filepath) or clobber:
            if ticid is not None:
                res = lk.search_tesscut(ticstr, sector=sector)
                if verbose:
                    print('Downloading TIC {} ...\n'.format(ticid))
            else:
                res = lk.search_tesscut(targ_coord, sector=sector)
                if verbose:
                    print('Downloading {} ...\n'.format(targ_coord))
            tpf = res.download(quality_bitmask=quality_bitmask,
                               cutout_size=FFI_CUTOUT_SIZE,
                               download_dir=fitsoutdir)
        else:
            if verbose:
                print('Loading TIC {} from {}/...\n'.format(ticid,fitsoutdir))
            tpf = lk.TessTargetPixelFile(filepath)
        if apply_data_quality_mask:
            tpf = remove_bad_data(tpf, sector=tpf.sector)
        #set
        tpf.targetid = ticid
        if return_df:
            return tpf, df
        else:
            return tpf
    else:
        msg='No full-frame tpf file found!\n'
        logging.info(msg); #sys.exit(msg)
        raise ValueError(msg)


def get_ffi_cutout_eleanor(targ_coord=None, tic=None, sector=None, verbose=False):
    '''
    '''
    raise NotImplementedError

    if targ_coord is not None:
        star = eleanor.Source(coords=targ_coord, sector=sector)
    elif tic is not None:
        star = eleanor.Source(tic=tic, sector=sector)
    data = eleanor.TargetData(star, height=15, width=15,
                              bkg_size=31,
                              do_psf=False,
                              do_pca=True)
    return data


def ffi_cutout_to_lc(tpf, sap_mask='threshold', aper_radius=None, percentile=None,
                  use_pld=True, use_gp=False,
                  period=None, t0=None, t14=None,
                  flatten=True, return_trend=True,
                  verbose=False, clobber=False):
    '''correct ffi tpf

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
    '''
    #make aperture mask
    mask = parse_aperture_mask(tpf, sap_mask=sap_mask, aper_radius=aper_radius,
                               percentile=percentile, verbose=verbose)
    #tpf to lc
    raw_lc = tpf.to_lightcurve(aperture_mask=mask)
    raw_lc = raw_lc.remove_nans().remove_outliers().normalize()

    if verbose:
        print('ndata={}\n'.format(len(raw_lc.time)))
    #correct systematics/ filter long-term variability
    #see https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/correctors.py
    if use_pld or use_sff:
        msg = 'Applying systematics correction:\n'.format(use_gp)
        if use_pld:
            msg += 'using PLD (gp={})\n'.format(use_gp)
            if verbose:
                logging.info(msg); print(msg)
            if np.all([period, t0, t14]):
                cadence_mask_tpf = make_cadence_mask(tpf.time, period, t0,
                                                 t14, verbose=verbose)
            else:
                cadence_mask_tpf = None
            npix = tpf.pipeline_mask.sum()
            if sap_mask=='pipeline' and npix>20:
                #GP will create MemoryError so limit mask
                msg = 'More than 20 pixels (npix={}) are used in PLD\n'.format(npix)
                sap_mask, aper_radius = 'round', 3.0
                mask = parse_aperture_mask(tpf, sap_mask, aper_radius, verbose=verbose)
                msg += 'Changing to round mask (r={}; npix={}) \
                        to avoid memory error.\n'.format(aper_radius, mask.sum())
            if verbose:
                logging.info(msg); print(msg)
            # pld = tpf.to_corrector(method='pld')
            pld = lk.PLDCorrector(tpf)
            corr_lc = pld.correct(cadence_mask=cadence_mask_tpf,
                    aperture_mask=mask, use_gp=use_gp,
                    pld_aperture_mask=mask,
                    #gp_timescale=30, n_pca_terms=10, pld_order=2,
                    ).remove_nans().remove_outliers().normalize()
        else:
            #use_sff without restoring trend
            msg += 'using SFF\n'
            if verbose:
                logging.info(msg); print(msg)
            sff = lk.SFFCorrector(raw_lc)
            corr_lc = sff.correct(centroid_col=raw_lc.centroid_col,
                                  centroid_row=raw_lc.centroid_row,
                                  polyorder=5, niters=3, bins=SFF_BINSIZE,
                                  windows=SFF_CHUNKSIZE,
                                  sigma_1=3.0, sigma_2=5.0,
                                  restore_trend=True
                                  ).remove_nans().remove_outliers()
        # get transit mask of corr lc
        msg='Flattening corrected light curve using Savitzky-Golay filter'
        if verbose:
            logging.info(msg); print(msg)
        if np.all([period, t0, t14]):
            cadence_mask_corr = make_cadence_mask(corr_lc.time, period, t0,
                                                 t14, verbose=verbose)
        else:
            cadence_mask_corr = None

        #finally flatten
        flat_lc, trend = corr_lc.flatten(window_length=SG_FILTER_WINDOW_LC,
                                         mask=cadence_mask_corr,
                                         return_trend=True)
    else:
        if verbose:
            msg='Flattening raw light curve using Savitzky-Golay filter'
            logging.info(msg); print(msg)
        if np.all([period, t0, t14]):
            cadence_mask_raw = make_cadence_mask(raw_lc.time, period, t0,
                                                 t14, verbose=verbose)
        else:
            cadence_mask_raw = None
        flat_lc, trend = raw_lc.flatten(window_length=SG_FILTER_WINDOW_LC,
                                        mask=cadence_mask_raw,
                                        return_trend=True)
    # remove obvious outliers and NaN in time
    raw_time_mask = ~np.isnan(raw_lc.time)
    raw_flux_mask = (raw_lc.flux > YLIMIT[0]) | (raw_lc.flux < YLIMIT[1])
    raw_lc = raw_lc[raw_time_mask & raw_flux_mask]
    flat_time_mask = ~np.isnan(flat_lc.time)
    flat_flux_mask = (flat_lc.flux > YLIMIT[0]) | (flat_lc.flux < YLIMIT[1])
    flat_lc = flat_lc[flat_time_mask | flat_flux_mask]
    if use_pld or use_sff:
        trend = trend[flat_time_mask & flat_flux_mask]
    else:
        trend = trend[raw_time_mask & raw_flux_mask]

    if use_pld or use_sff:
        if flatten:
            return (flat_lc, trend) if return_trend else flat_lc
        else:
            return (corr_lc, trend) if return_trend else corr_lc
    else:
        return (raw_lc, trend) if return_trend else raw_lc


def plot_ffi_apers(tpf, percentiles=np.arange(40,100,10), figsize=(14,6)):
    fig, axs = pl.subplots(2,3,figsize=figsize)
    ax = axs.flatten()

    for n,perc in enumerate(percentiles):
        mask = parse_aperture_mask(tpf, sap_mask='percentile', percentile=perc)
        a = tpf.plot(ax=ax[n], aperture_mask=mask, show_colorbar=False)
        a.axis('off')
        a.set_title(perc)

    return fig


def get_pipeline_lc(targ_coord=None, tic=None, sector=None, flux_type='pdcsap',
                    verbose=False, clobber=False, fitsoutdir='.'):
    '''fetch pipeline generated (corrected) light curve

    Parameters
    ----------
    targ_coord : astropy.coordinates
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
    '''
    if tic:
        ticstr = 'TIC {}'.format(tic)
        if verbose:
            print('\nSearching mast for {}\n'.format(ticstr))
        res = lk.search_lightcurvefile(ticstr, mission=MISSION, sector=sector)
    else:
        if verbose:
            print('\nSearching mast for ra,dec=({})\n'.format(targ_coord.to_string()))
        res = lk.search_lightcurvefile(targ_coord, mission=MISSION, sector=sector)
    df = res.table.to_pandas()

    if len(df)>0:
        all_sectors = [int(i) for i in df['sequence_number'].values]
        if sector:
            sector_idx = df['sequence_number'][df['sequence_number'].isin([sector])].index.tolist()
            if len(sector_idx)==0:
                sys.exit('sector {} data is unavailable'.format(sector))
            obsid = df.iloc[sector_idx]['obs_id'].values[0]
            ticid = int(df.iloc[sector_idx]['target_name'].values[0])
            fitsfilename = df.iloc[sector_idx]['productFilename'].values[0]
        else:
            sector_idx = 0
            sector = int(df.iloc[sector_idx]['sequence_number'])
            obsid = df.iloc[sector_idx]['obs_id']
            ticid = int(df.iloc[sector_idx]['target_name'])
            fitsfilename = df.iloc[sector_idx]['productFilename']

        msg='{} tpf(s) found in sector(s) {}\n'.format(len(df), all_sectors)
        msg+='Using data from sector {} only\n'.format(sector)
        if verbose:
            logging.info(msg); print(msg)

        filepath = join(fitsoutdir,'mastDownload/TESS',obsid,fitsfilename)
        if not exists(filepath) or clobber:
            if verbose:
                print('Downloading TIC {} ...\n'.format(ticid))
            ticstr = 'TIC {}'.format(ticid)
            res = lk.search_lightcurvefile(ticstr, mission=MISSION, sector=sector)
            lc = res.download(quality_bitmask=quality_bitmask, download_dir=fitsoutdir)
        else:
            if verbose:
                print('Loading TIC {} from {}/...\n'.format(ticid,fitsoutdir))
            lc = lk.TessLightCurveFile(filepath)

        if flux_type=='pdcsap':
            flux_type='PDCSAP_FLUX'
        else:
            flux_type='SAP_FLUX'
        lc = lc.get_lightcurve(flux_type=flux_type)
        return lc
    else:
        msg='No light curve file found!\n'
        logging.info(msg)
        sys.exit(msg)


def compare_custom_pipeline_lc(targ_coord=None, tic=None, sector=None, flux_type='sap',
                                verbose=False, clobber=False, fitsoutdir='.'):
    '''

    Parameters
    ----------
    targ_coord : astropy.coordinates
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
    '''
    sap = get_pipeline_lc(targ_coord=targ_coord, tic=None, sector=None, flux_type='sap',
                        verbose=False, clobber=False, fitsoutdir='.')
    pdcsap = get_pipeline_lc(targ_coord=targ_coord, tic=None, sector=None, flux_type='pdcsap',
                        verbose=False, clobber=False, fitsoutdir='.')

    fig, ax = pl.subplots(3, 1, figsize=(10,5))
    sap.errorbar(ax=ax[0])
    pdcsap.errorbar(ax=ax[1])
    return fig


def run_tls_on_pdcsap(targ_coord=None, tic=None, sector=None, flux_type='pdcsap',
                     verbose=False, fitsoutdir='.'):
    '''run tls periodogram on pdcsap lc

    Parameters
    ----------
    targ_coord : astropy.coordinates
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
    '''
    lc = get_pipeline_lc(targ_coord=None, tic=None, sector=None, flux_type='pdcsap',
                         verbose=False, fitsoutdir='.')
    #run tls
    t, fcor = lc.time, lc.flux
    model = transitleastsquares(t, fcor)
    try:
        ((u1, u2), Ms_tic, _, _, Rs_tic, _, _) = catalog.catalog_info(TIC_ID=int(ticid))
        u1, u2 = DEFAULT_U if not np.all([u1, u2]) else [u1,u2]
        Rs_tic = 1.0 if Rs_tic is None else Rs_tic
        Ms_tic = 1.0 if Ms_tic is None else Ms_tic
    except:
        (u1, u2), Ms_tic, Rs_tic =  DEFAULT_U, 1.0, 1.0 #assume G2 star
    if verbose:
        if u1==DEFAULT_U[0] and u2==DEFAULT_U[1]:
            print('Using default limb-darkening coefficients\n')
        else:
            print('Using u1={:.4f},u2={:.4f} based on TIC catalog\n'.format(u1,u2))

    results = model.power(u = [u1,u2], limb_dark = 'quadratic')
    period = results.period
    t0     = results.T0
    t14    = results.duration

    return (period,t0,t14)


def get_tpf(targ_coord, tic=None, apphot_method='sap',
            apply_data_quality_mask=True,
            sector=None, verbose=False, clobber=False, sap_mask='pipeline',
            fitsoutdir='.', return_df=True):
    '''Fetch tpf from MAST

    Parameters
    ----------
    targ_coord : astropy.coordinates
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
    '''
    if tic:
        ticstr = 'TIC {}'.format(tic)
        # sector = None searches for all tpf
        # which sector to download is specified later
        if verbose:
            print('\nSearching mast for {}\n'.format(ticstr))
        res = lk.search_targetpixelfile(ticstr, mission=MISSION, sector=None)
    else:
        if verbose:
            print('\nSearching mast for ra,dec=({})\n'.format(targ_coord.to_string()))
        res = lk.search_targetpixelfile(targ_coord, mission=MISSION, sector=sector)
    df = res.table.to_pandas()

    if len(df)>0:
        all_sectors = [int(i) for i in df['sequence_number'].values]
        if sector:
            sector_idx = df['sequence_number'][df['sequence_number'].isin([sector])].index.tolist()
            if len(sector_idx)==0:
                sys.exit('sector {} data is unavailable'.format(sector))
            obsid = df.iloc[sector_idx]['obs_id'].values[0]
            ticid = int(df.iloc[sector_idx]['target_name'].values[0])
            fitsfilename = df.iloc[sector_idx]['productFilename'].values[0]
        else:
            sector_idx = 0
            sector = int(df.iloc[sector_idx]['sequence_number'])
            obsid = df.iloc[sector_idx]['obs_id']
            ticid = int(df.iloc[sector_idx]['target_name'])
            fitsfilename = df.iloc[sector_idx]['productFilename']

        msg='{} tpf(s) found in sector(s) {}\n'.format(len(df), all_sectors)
        msg+='Using data from sector {} only\n'.format(sector)
        if verbose:
            logging.info(msg); print(msg)

        filepath = join(fitsoutdir,'mastDownload/TESS',obsid,fitsfilename)
        if not exists(filepath) or clobber:
            if verbose:
                print('Downloading TIC {} ...\n'.format(ticid))
            ticstr = 'TIC {}'.format(ticid)
            res = lk.search_targetpixelfile(ticstr, mission=MISSION, sector=sector)
            tpf = res.download(quality_bitmask=quality_bitmask, download_dir=fitsoutdir)
        else:
            if verbose:
                print('Loading TIC {} from {}/...\n'.format(ticid,fitsoutdir))
            tpf = lk.TessTargetPixelFile(filepath)
        #assert tpf.mission == MISSION
        if apply_data_quality_mask:
            tpf = remove_bad_data(tpf, sector=tpf.sector)
        if return_df:
            return tpf, df
        else:
            return tpf
    else:
        msg='No tpf file found! Check FFI data using --cadence=long\n'
        logging.info(msg)
        #sys.exit(msg)
        raise ValueError(msg)


def generate_QL(targ_coord,toi=None,tic=None,sector=None,#cutout_size=10,
                use_pld=True,use_gp=False,use_sff=False,
                apphot_method='sap',sap_mask='pipeline',
                aper_radius=None,percentile=None,
                apply_data_quality_mask=True, cadence='short',
                fitsoutdir='.',figoutdir='.',savefig=True,
                clobber=False,verbose=True):
    '''Create quick look light curve with archival image

    Parameters
    ----------
    targ_coord : astropy.coordinates
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
    '''
    start = time.time()
    try:
        #download or load tpf
        if cadence=='short':
            tpf, df = get_tpf(targ_coord=targ_coord, tic=tic, sector=sector, verbose=verbose,
                              clobber=clobber, sap_mask=sap_mask,
                              apply_data_quality_mask=apply_data_quality_mask,
                              fitsoutdir=fitsoutdir, return_df=True)
            sg_filter_window = SG_FILTER_WINDOW_SC
        else:
            tpf, df = get_ffi_cutout(targ_coord=targ_coord, tic=tic, clobber=clobber,
                           sector=sector, #cutout_size=cutout_size,
                           apply_data_quality_mask=apply_data_quality_mask,
                           verbose=verbose, fitsoutdir=fitsoutdir, return_df=True)
            sg_filter_window = SG_FILTER_WINDOW_LC
            assert sap_mask!='pipeline', 'Use any aper_mask except pipeline for FFI data.'
        all_sectors = [int(i) for i in df['sequence_number'].values]
        if sector is None:
            sector = all_sectors[0]

        #check tpf size
        ny, nx  = tpf.flux.shape[1], tpf.flux.shape[2]
        diag    = np.sqrt(nx**2+ny**2)
        fov_rad = (0.6*diag*TESS_pix_scale).to(u.arcmin)
        if fov_rad>1*u.deg:
            tpf = cutout_tpf(tpf)
            # redefine dimensions
            ny, nx  = tpf.flux.shape[1], tpf.flux.shape[2]
            diag    = np.sqrt(nx**2+ny**2)
            fov_rad = (0.6*diag*TESS_pix_scale).to(u.arcmin)

        if tpf.targetid is None:
            if cadence=='short':
                ticid = df.iloc[0]['target_name']
            else:
                if tic is not None:
                    ticid = int(str(df.iloc[0]['target_name']).split()[-1])
                else:
                    ticid = None
        else:
            ticid = tpf.targetid
        msg='#----------TIC {}----------#\n'.format(ticid)
        if verbose:
            logging.info(msg); print(msg)

        # check if target is TOI from tess alerts
        q = query_toi(tic=ticid, toi=toi, clobber=clobber,
                      outdir='../data', verbose=verbose)
        period, t0, t14, depth, toiid = get_transit_params(q, toi=toi,
                                                    tic=ticid, verbose=verbose)
        if verbose:
            print('Generating QL figure...\n')

        #make aperture mask
        mask = parse_aperture_mask(tpf, sap_mask=sap_mask,
               aper_radius=aper_radius, percentile=percentile, verbose=verbose)

        # make lc
        raw_lc = tpf.to_lightcurve(method=PHOTMETHOD, aperture_mask=mask)
        raw_lc = raw_lc.remove_nans().remove_outliers().normalize()
        if verbose:
            print('ndata={}\n'.format(len(raw_lc.time)))
        #correct systematics/ filter long-term variability
        #see https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/correctors.py
        if use_pld or use_sff:
            msg = 'Applying systematics correction:\n'.format(use_gp)
            if use_pld:
                msg += 'using PLD (gp={})\n'.format(use_gp)
                if verbose:
                    logging.info(msg); print(msg)
                cadence_mask_tpf = make_cadence_mask(tpf.time, period, t0,
                                                     t14, verbose=verbose)
                npix = tpf.pipeline_mask.sum()
                if sap_mask=='pipeline' and npix>20:
                    #GP will create MemoryError so limit mask
                    msg = 'More than 20 pixels (npix={}) are used in PLD\n'.format(npix)
                    sap_mask, aper_radius = 'round', 3.0
                    mask = parse_aperture_mask(tpf, sap_mask, aper_radius, verbose=verbose)
                    msg += 'Changing to round mask (r={}; npix={}) \
                            to avoid memory error.\n'.format(aper_radius, mask.sum())
                if verbose:
                    logging.info(msg); print(msg)
                # pld = tpf.to_corrector(method='pld')
                pld = lk.PLDCorrector(tpf)
                corr_lc = pld.correct(cadence_mask=cadence_mask_tpf,
                        aperture_mask=mask, use_gp=use_gp,
                        pld_aperture_mask=mask,
                        #gp_timescale=30, n_pca_terms=10, pld_order=2,
                        ).remove_nans().remove_outliers().normalize()

            else:
                #use_sff without restoring trend
                msg += 'using SFF\n'
                if verbose:
                    logging.info(msg); print(msg)
                sff = lk.SFFCorrector(raw_lc)
                corr_lc = sff.correct(centroid_col=raw_lc.centroid_col,
                                      centroid_row=raw_lc.centroid_row,
                                      polyorder=5, niters=3, bins=SFF_BINSIZE,
                                      windows=SFF_CHUNKSIZE,
                                      sigma_1=3.0, sigma_2=5.0,
                                      restore_trend=True
                                      ).remove_nans().remove_outliers()
            # get transit mask of corr lc
            msg='Flattening corrected light curve using Savitzky-Golay filter'
            if verbose:
                logging.info(msg); print(msg)
            cadence_mask_corr = make_cadence_mask(corr_lc.time, period, t0,
                                                 t14, verbose=verbose)
            #finally flatten
            flat_lc, trend = corr_lc.flatten(window_length=sg_filter_window,
                                             mask=cadence_mask_corr,
                                             return_trend=True)
        #elif apphot_method=='pdcsap':
        #    flat_lc = res2.get_lightcurve(flux_type='PDCSAP_FLUX').remove_nans().remove_outliers()
        #    if verbose:
        #        msg='Using PDCSAP light curve\n'
        #        print(msg)
        #        logging.info(msg)

        else:
            if verbose:
                msg='Flattening raw light curve using Savitzky-Golay filter'
                logging.info(msg); print(msg)
            cadence_mask_raw = make_cadence_mask(raw_lc.time, period, t0,
                                                 t14, verbose=verbose)
            flat_lc, trend = raw_lc.flatten(window_length=sg_filter_window,
                                            mask=cadence_mask_raw,
                                            return_trend=True)
        # remove obvious outliers and NaN in time
        raw_time_mask = ~np.isnan(raw_lc.time)
        raw_flux_mask = (raw_lc.flux > YLIMIT[0]) | (raw_lc.flux < YLIMIT[1])
        raw_lc = raw_lc[raw_time_mask & raw_flux_mask]
        flat_time_mask = ~np.isnan(flat_lc.time)
        flat_flux_mask = (flat_lc.flux > YLIMIT[0]) | (flat_lc.flux < YLIMIT[1])
        flat_lc = flat_lc[flat_time_mask | flat_flux_mask]
        #if apphot_method!='pdcsap':
        if use_pld or use_sff:
            trend = trend[flat_time_mask & flat_flux_mask]
        else:
            trend = trend[raw_time_mask & raw_flux_mask]

        #periodogram; see also https://docs.lightkurve.org/tutorials/02-recover-a-planet.html
        #pg = corr_lc.to_periodogram(minimum_period=min_period,
        #                                maximum_period=max_period,
        #                                method=PGMETHOD,
        #                                oversample_factor=10)
        if verbose:
            print('Periodogram with TLS\n')
        t = flat_lc.time
        fcor = flat_lc.flux

        # TLS
        model = transitleastsquares(t, fcor)
        #get TIC catalog info: https://github.com/hippke/tls/blob/master/transitleastsquares/catalog.py
        #see defaults: https://github.com/hippke/tls/blob/master/transitleastsquares/tls_constants.py
        try:
            ((u1, u2), Ms_tic, _, _, Rs_tic, _, _) = catalog.catalog_info(TIC_ID=int(ticid))
            u1, u2 = DEFAULT_U if not np.all([u1, u2]) else [u1,u2]
            Rs_tic = 1.0 if Rs_tic is None else Rs_tic
            Ms_tic = 1.0 if Ms_tic is None else Ms_tic
        except:
            (u1, u2), Ms_tic, Rs_tic =  DEFAULT_U, 1.0, 1.0 #assume G2 star
        if verbose:
            if u1==DEFAULT_U[0] and u2==DEFAULT_U[1]:
                print('Using default limb-darkening coefficients\n')
            else:
                print('Using u1={:.4f},u2={:.4f} based on TIC catalog\n'.format(u1,u2))

        results = model.power(u       = [u1,u2],
                              limb_dark = 'quadratic',
                              #R_star  = Rs_tic,
                              #M_star  = Ms_tic,
                              #oversampling_factor=3,
                              #duration_grid_step =1.1
                              #transit_depth_min=ppm*10**-6,
                              )
        results['u'] = [u1,u2]
        results['Rstar_tic'] = Rs_tic
        results['Mstar_tic'] = Ms_tic

        if verbose:
            print('Odd-Even transit mismatch: {:.2f} sigma\n'.format(results.odd_even_mismatch))
            print('Best period from periodogram: {:.4f} {}\n'.format(results.period,u.day))
        #phase fold
        fold_lc = flat_lc.fold(period=results.period, t0=results.T0)

        maskhdr = tpf.hdu[2].header
        tpfwcs = WCS(maskhdr)
        #------------------------create figure-----------------------#
        #FIXME: line below is run again after here to define projection
        if verbose:
            print('Querying {0} ({1:.2f} x {1:.2f}) archival image\n'.format(IMAGING_SURVEY,fov_rad))
        ax, hdu = plot_finder_image(targ_coord, fov_radius=fov_rad,
                                    survey=IMAGING_SURVEY, reticle=True)
        pl.close()

        fig = pl.figure(figsize=(15,15))
        ax0 = fig.add_subplot(321)
        ax1 = fig.add_subplot(322, projection=WCS(hdu.header))
        ax2 = fig.add_subplot(323)
        ax3 = fig.add_subplot(324)
        ax4 = fig.add_subplot(325)
        ax5 = fig.add_subplot(326)
        axs = [ax0,ax1,ax2,ax3,ax4,ax5]

        #----------ax0: tpf plot----------
        i=0
        ax1=tpf.plot(aperture_mask=mask, #frame=10,
                     origin='lower', ax=axs[i]);
        ax1.text(0.95, 0.10, 'mask={}'.format(sap_mask),
            verticalalignment='top', horizontalalignment='right',
            transform=axs[i].transAxes, color='w', fontsize=12)
        ax1.set_title('sector={}'.format(sector), fontsize=FONTSIZE)
        axs[i].invert_yaxis()

        #----------ax1: archival image with superposed aper mask ----------
        i=1
        # if verbose:
        #     print('Querying {0} ({1} x {1}) archival image'.format(IMAGING_SURVEY,fov_rad))
        nax, hdu = plot_finder_image(targ_coord, fov_radius=fov_rad,
                                     survey=IMAGING_SURVEY, reticle=True, ax=axs[i])
        nwcs = WCS(hdu.header)
        mx, my = hdu.data.shape

        #plot TESS aperture
        contour = np.zeros((ny, nx))
        contour[np.where(mask)] = 1
        contour = np.lib.pad(contour, 1, PadWithZeros)
        highres = zoom(contour, 100, order=0, mode='nearest')
        extent = np.array([-1, nx, -1, ny])
        #superpose aperture mask
        cs1 = axs[i].contour(highres, levels=[0.5], extent=extent,
                             origin='lower', colors='y',
                             transform=nax.get_transform(tpfwcs))

        #plot gaia sources
        gaia_sources = Catalogs.query_region(targ_coord, radius=fov_rad,
                                    catalog="Gaia", version=2).to_pandas()
        for r,d in gaia_sources[['ra','dec']].values:
            pix = nwcs.all_world2pix(np.c_[r,d],1)[0]
            nax.scatter(pix[0], pix[1], marker='s', s=100, edgecolor='r',
                                                        facecolor='none')
        pl.setp(nax, xlim=(0,mx), ylim=(0,my))
        nax.set_title("{0} ({1:.2f}\' x {1:.2f}\')".format(IMAGING_SURVEY,fov_rad.value),
                                                        fontsize=FONTSIZE)
        #get gaia stellar params
        rstar, teff = get_gaia_params(targ_coord,gaia_sources,verbose=verbose)

        #----------ax2: lc plot----------
        i=2
        ax2 = raw_lc.errorbar(label='raw lc',ax=axs[i])
        #some weird outliers do not get clipped, so force ylim
        y1,y2=axs[i].get_ylim()
        if y1<YLIMIT[0]:
            axs[i].set_ylim(YLIMIT[0],y2)
        if y2>YLIMIT[1]:
            axs[i].set_ylim(y1,YLIMIT[1])

        #plot trend in raw flux if no correction is applied in sap flux
        #no trend plot if pdcsap or pld or sff is used
        if (use_pld==use_sff==False) & (apphot_method=='sap'):
            trend.plot(color='r', linewidth=3, label='Savgol_filter',ax=axs[i])
        text = 'cdpp: {:.2f}'.format(raw_lc.flatten().estimate_cdpp())
        axs[i].text(0.95, 0.05, text,
                verticalalignment='top', horizontalalignment='right',
                transform=axs[i].transAxes, color='green', fontsize=15)
        axs[i].legend(loc='upper left')

        #----------ax3: long-term variability (+ systematics) correction----------
        i=3
        #plot trend in corrected light curve
        if use_pld or use_sff:
            #----------ax3: systematics-corrected----------
            ax3 = corr_lc.errorbar(ax=axs[i], label='corr lc');
            text = 'PLD={}, SFF={}, gp={}, cdpp={:.2f}'.format(use_pld,
                                        use_sff,use_gp,flat_lc.estimate_cdpp())
            trend.plot(color='r', linewidth=3, label='Savgol_filter',ax=axs[i])
            #axs[i].plot(t[cadence_mask_corr], f[cadence_mask_corr], '')
        else:
            #----------ax3: long-term variability-corrected----------
            ax3 = flat_lc.errorbar(ax=axs[i],label='flat lc');
            text = 'PLD={} (gp={}), SFF={}, cdpp={:.2f}'.format(use_pld,use_gp,
                                        use_sff,flat_lc.estimate_cdpp())
        #plot detected transits in panel 4:
        if np.all([results.period,results.T0]):
            # period and t0 from TLS
            if use_pld:
                tns = get_tns(corr_lc.time, results.period, results.T0)
            else:
                tns = get_tns(flat_lc.time, results.period, results.T0)
            for t in tns:
                ax3.axvline(t, 0, 1, linestyle='--', color='k', alpha=0.5)
        axs[i].legend(loc='upper left')
        axs[i].text(0.95, 0.15, text,
            verticalalignment='top', horizontalalignment='right',
            transform=axs[i].transAxes, color='green', fontsize=15)

        #----------ax4: periodogram----------
        i=4
        # pg.plot(ax=axs[i], c='k', unit=u.day, view='Period', scale='log', label='periodogram')
        axs[i].axvline(results.period, alpha=0.4, lw=3)
        axs[i].set_xlim(np.min(results.periods), np.max(results.periods))
        # plot harmonics: period multiples
        for n in range(2, 10):
            axs[i].axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
            axs[i].axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
        axs[i].set_ylabel(r'SDE')
        axs[i].set_xlabel('Period [days]')
        axs[i].plot(results.periods, results.power, color='black', lw=0.5, label='TLS periodogram')
        axs[i].set_xlim(0, max(results.periods));
        text = 'Best period={:.2f} {}'.format(results.period, u.day)
        # text = 'Best period={:.2f} {}'.format(period.value, period.unit)
        # axs[i].axvline(period.value, 0, 1, linestyle='--', color='k', linewidth=3)
        axs[i].text(0.95, 0.85, text,
                verticalalignment='top', horizontalalignment='right',
                transform=axs[i].transAxes, color='green', fontsize=15)
        axs[i].legend()

        #----------ax5: phase folded lc----------
        i=5
        fold_lc.scatter(ax=axs[i], color='k', alpha=0.1, label='unbinned')
        #ax[i].scatter(results.folded_phase-phase_offset,results.folded_y,
        #              color='k', marker='.', label='unbinned', alpha=0.1, zorder=2)
        fold_lc.bin(5).scatter(ax=axs[i], color='C1', label='binned (10-min)')
        axs[i].plot(results.model_folded_phase-0.5, results.model_folded_model, color='red', label='TLS model')
        rprs= results['rp_rs']
        t14 = results.duration*u.day.to(u.hour)
        t0  = results['T0']

        Rp = rprs*rstar*u.Rsun.to(u.Rearth)
        if str(rstar)!='nan':
            text = 'Rp={:.2f} Re\nt14={:.2f} hr\nt0={:.6f}'.format(Rp, t14, t0)
        else:
            text = 'Rp/Rs={:.2f}\nt14={:.2f} hr\nt0={:.6f}'.format(rprs, t14, t0)
        if verbose:
            print(text)
        axs[i].text(0.65, 0.20, text,
                verticalalignment='top', horizontalalignment='left',
                transform=axs[i].transAxes, color='green', fontsize=15)
        axs[i].set_xlim(-0.2,0.2)
        axs[i].legend(title='phase-folded lc')
        axs[i].legend(loc=3)

        # manually set ylimit for shallow transits
        if rprs<=0.1:
            ylo,yhi = 1-10*rprs**2,1+5*rprs**2
            axs[i].set_ylim(ylo, yhi if yhi<1.02 else 1.02)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        end = time.time()

        cad = 'sc' if cadence=='short' else 'lc'
        #save tls results
        if results:
            results['tic'] = tpf.targetid
            results['sector'] = sector
            results['Rs_gaia'] = rstar
            results['Teff_gaia'] = teff
            results['Rp'] = Rp
            fp = join(figoutdir,'tic{}_s{}_{}_tls.hdf5'.format(tpf.targetid,sector,cad))
            if savefig:
                dd.io.save(fp, results)
                print('Saved: {}\n'.format(fp))

        if toi or toiid:
            #toiid is TOIXXX determined from TESS release queried using TIC or coordinates
            id = toi if toi is not None else toiid
            figname='TIC{}_TOI{}_s{}_{}.png'.format(tpf.targetid, str(id),sector,cad)
            lcname1='TIC{}_TOI{}_s{}_lc_flat_{}.txt'.format(tpf.targetid, str(id),sector,cad)
            lcname2='TIC{}_TOI{}_s{}_lc_fold_{}.txt'.format(tpf.targetid, str(id),sector,cad)
            pl.suptitle('TIC {} (TOI {})'.format(ticid,id), fontsize=FONTSIZE)
        else:
            figname='TIC{}_s{}_{}.png'.format(str(tpf.targetid),sector,cad)
            lcname1='TIC{}_s{}_lc_flat_{}.txt'.format(str(tpf.targetid),sector,cad)
            lcname2='TIC{}_s{}_lc_fold_{}.txt'.format(str(tpf.targetid),sector,cad)
            pl.suptitle('TIC {}'.format(ticid), fontsize=FONTSIZE)
        figoutpath=join(figoutdir,figname)
        lcoutpath1 =join(figoutdir,lcname1)
        lcoutpath2 =join(figoutdir,lcname2)

        if savefig:
            fig.savefig(figoutpath, bbox_inches='tight')
            print('Saved: {}\n'.format(figoutpath))
            # np.savetxt(lcoutpath, np.c_[t,f], fmt='%.8f')
            flat_lc.to_pandas().to_csv(lcoutpath1, index=False, sep=' ')
            fold_lc.to_pandas().to_csv(lcoutpath2, index=False, sep=' ')
            print('Saved:\n{}\n{}'.format(lcoutpath1,lcoutpath2))
        else:
            pl.show()
        msg='#----------Runtime: {:.2f} s----------#\n'.format(end - start)
        if verbose:
            logging.info(msg); print(msg)
        pl.close()
    except:
        print('Error occured:\n{}'.format(traceback.format_exc()))
        print('\n-----------Some recommendations-----------\n')
        print('Try -c if [buffer is too small for requested array]')
        print('Try using -no_gp if [MemoryError: std::bad_alloc]')
        print('Try --aper={pipeline,threshold,all} if tpf seems corrupted\n')
        #print('tpf size=(x,x) pix seems too big\n\n')
        logging.error(str(traceback.format_exc()))
    #save logs
    logfile = open(LOG_FILENAME, 'rt')
    try:
        body = logfile.read()
    finally:
        logfile.close()


def generate_all_lc(targ_coord,toi=None,tic=None,
                    use_pld=False,use_gp=True,use_sff=False,
                    apphot_method='sap',sap_mask='pipeline',
                    aper_radius=None,sectors=None,
                    apply_data_quality_mask=True,
                    fitsoutdir='.',figoutdir='.',savefig=True,
                    clobber=False,verbose=True):
    '''Create multi-sector light curves

    Parameters
    ----------
    targ_coord : astropy.coordinates
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
    '''
    # try:
    #     from brokenaxes import brokenaxes
    # except:
    #     sys.exit('pip install git+https://github.com/bendichter/brokenaxes')

    start = time.time()
    try:
        if verbose:
            print('\nSearching mast for ra,dec=({})\n'.format(targ_coord.to_string()))
        #search for tpf
        #sector=None searches all available tpf
        #sector to be used is specified later
        if tic:
            #search by tic
            ticstr = 'TIC {}'.format(tic)
            res = lk.search_targetpixelfile(ticstr, mission=MISSION, sector=None)
        else:
            #search by coordinates
            res = lk.search_targetpixelfile(targ_coord, mission=MISSION, sector=None)

        df = res.table.to_pandas()
        all_sectors = [int(i) for i in df['sequence_number'].values]
        msg='{} tpf(s) found in sector(s) {}\n'.format(len(df), all_sectors)

        ticid = df.iloc[0]['target_name']
        if len(df)==1:
             msg='Target is observed in just 1 sector. Re-run script without -a argument.\n'
             logging.info(msg)
             print(msg)
        elif len(df)>1:
            msg='#----------TIC {}----------#\n'.format(ticid)
            logging.info(msg)

            # check if target is TOI from TESS alerts
            ticid = int(df.iloc[0]['target_name'])
            q = query_toi(tic=ticid, toi=toi, clobber=clobber, outdir='../data', verbose=verbose)

            if len(df)>1:
                #if tic is observed in multiple sectors
                if sectors:
                    #use only specified sectors
                    sector_idx=df['sequence_number'][df['sequence_number'].isin(sectors)].index.tolist()
                    #reduce all sectors to specified sectors
                    all_sectors = np.array(all_sectors)[sector_idx]
                    if verbose:
                        print('Analyzing data from sectors: {}\n'.format(sectors))
                else:
                    #use all sectors
                    sector_idx=range(len(df))
                    if verbose:
                        print('Downloading all sectors: {}\n'.format(all_sectors))
            else:
                #if tic is observed only once, take first available sector
                sector_idx=[0]

            msg+='Using data from sectors {}\n'.format(all_sectors)
            if verbose:
                logging.info(msg); print(msg)

            tpfs = []
            masks = []
            # sectors = []
            for n in tqdm(sector_idx):
                #load or download per sector
                obsid = df.iloc[n]['obs_id']
                ticid = int(df.iloc[n]['target_name'])
                sector = int(df.iloc[n]['sequence_number']) #==obsid.split('-')[1][-1]
                fitsfilename = df.iloc[n]['productFilename']

                filepath = join(fitsoutdir,'mastDownload/TESS',obsid,fitsfilename)
                if not exists(filepath) or clobber:
                    if verbose:
                        print('Downloading TIC {} (sector {})...\n'.format(ticid,sector))
                    #re-search tpf with specified sector
                    #FIXME: related issue: https://github.com/KeplerGO/lightkurve/issues/533
                    ticstr = 'TIC {}'.format(ticid)
                    tpf = lk.search_targetpixelfile(ticstr, mission=MISSION,
                            sector=sector).download(quality_bitmask=quality_bitmask,
                                                    download_dir=fitsoutdir)
                else:
                    if verbose:
                        print('Loading TIC {} (sector {}) from {}/...\n'.format(ticid,sector,fitsoutdir))
                    tpf = lk.TessTargetPixelFile(filepath)
                assert tpf.mission==MISSION

                #check tpf size
                ny, nx  = tpf.flux.shape[1], tpf.flux.shape[2]
                diag    = np.sqrt(nx**2+ny**2)
                fov_rad = (0.6*diag*TESS_pix_scale).to(u.arcmin)
                if fov_rad>1*u.deg:
                    tpf = cutout_tpf(tpf)
                    # redefine dimensions
                    ny, nx  = tpf.flux.shape[1], tpf.flux.shape[2]
                    diag    = np.sqrt(nx**2+ny**2)
                    fov_rad = (0.6*diag*TESS_pix_scale).to(u.arcmin)

                # remove bad data identified in data release notes
                if apply_data_quality_mask:
                    tpf = remove_bad_data(tpf,sector,verbose=verbose)

                #make aperture mask
                mask = parse_aperture_mask(tpf, sap_mask, aper_radius, verbose=verbose)

                tpfs.append(tpf)
                masks.append(mask)
                # sectors.append(sector)

            #query tess alerts/ toi release
            period, t0, t14, depth, toiid = get_transit_params(q, toi=toi,
                                            tic=ticid, verbose=verbose)

            #------------------------create figure-----------------------#
            #FIXME: line below is run again after here to define projection
            if verbose:
                print('Querying {0} ({1:.2f} x {1:.2f}) archival image\n'.format(IMAGING_SURVEY,fov_rad))
            ax, hdu = plot_finder_image(targ_coord, fov_radius=fov_rad,
                                        survey=IMAGING_SURVEY, reticle=True)
            pl.close()

            fig = pl.figure(figsize=(15,15))
            ax0 = fig.add_subplot(321)
            ax1 = fig.add_subplot(322, projection=WCS(hdu.header))
            ax2 = fig.add_subplot(323)
            ax3 = fig.add_subplot(324)
            ax4 = fig.add_subplot(325)
            ax5 = fig.add_subplot(326)
            ax = [ax0,ax1,ax2,ax3,ax4,ax5]

            #----------ax0: tpf plot----------
            # plot only the first tpf
            i=0
            tpf = tpfs[i]
            ax1=tpf.plot(aperture_mask=mask, #frame=10,
                         origin='lower', ax=ax[i]);
            ax1.text(0.95, 0.10, 'mask={}'.format(sap_mask),
                verticalalignment='top', horizontalalignment='right',
                transform=ax[i].transAxes, color='w', fontsize=12)
            ax1.set_title('sector={}'.format(all_sectors[i]), fontsize=FONTSIZE)
            #FIXME: when should axis be inverted?
            ax[i].invert_yaxis()

            #----------ax1: archival image with superposed aper mask ----------
            i=1
            # if verbose:
            #     print('Querying {0} ({1} x {1}) archival image'.format(IMAGING_SURVEY,fov_rad))
            nax, hdu = plot_finder_image(targ_coord, fov_radius=fov_rad,
                                         survey=IMAGING_SURVEY, reticle=True, ax=ax[i])
            nwcs = WCS(hdu.header)
            mx, my = hdu.data.shape

            #plot gaia sources
            gaia_sources = Catalogs.query_region(targ_coord, radius=fov_rad,
                                        catalog="Gaia", version=2).to_pandas()
            for r,d in gaia_sources[['ra','dec']].values:
                pix = nwcs.all_world2pix(np.c_[r,d],1)[0]
                nax.scatter(pix[0], pix[1], marker='s', s=100, edgecolor='r',
                                                            facecolor='none')
            pl.setp(nax, xlim=(0,mx), ylim=(0,my))
            nax.set_title("{0} ({1:.2f}\' x {1:.2f}\')".format(IMAGING_SURVEY,fov_rad.value),
                                                            fontsize=FONTSIZE)

            #get gaia stellar params
            rstar, teff = get_gaia_params(targ_coord,gaia_sources,verbose=verbose)

            lcs = []
            corr_lcs = []
            flat_lcs = []
            times = []
            fluxes = []
            flux_errs = []
            trends = []
            cdpps_raw = []
            cdpps_corr = []
            for j,n in tqdm(enumerate(sector_idx)):
                print('\n----------sector {}----------\n'.format(all_sectors[j]))
                tpf = tpfs[j]
                maskhdr = tpf.hdu[2].header
                tpfwcs = WCS(maskhdr)

                #plot TESS aperture
                contour = np.zeros((ny, nx))
                contour[np.where(mask)] = 1
                contour = np.lib.pad(contour, 1, PadWithZeros)
                highres = zoom(contour, 100, order=0, mode='nearest')
                extent = np.array([-1, nx, -1, ny])
                #superpose aperture mask
                cs1 = ax[1].contour(highres, levels=[0.5], extent=extent,
                                     origin='lower', colors=COLORS[j],
                                     transform=nax.get_transform(tpfwcs))
                # make lc
                raw_lc = tpf.to_lightcurve(method=PHOTMETHOD, aperture_mask=masks[j])
                raw_lc = raw_lc.remove_nans().remove_outliers().normalize()

                # remove obvious outliers and NaN in time
                time_mask = ~np.isnan(raw_lc.time)
                flux_mask = (raw_lc.flux > YLIMIT[0]) | (raw_lc.flux < YLIMIT[1])
                raw_lc = raw_lc[time_mask & flux_mask]

                lcs.append(raw_lc)
                cdpps_raw.append(raw_lc.flatten().estimate_cdpp())

                #correct systematics/ filter long-term variability
                if use_pld or use_sff:
                    msg = 'Applying systematics correction:\n'.format(use_gp)
                    if use_pld:
                        msg += 'using PLD (gp={})'.format(use_gp)
                        if verbose:
                            logging.info(msg); print(msg)
                        cadence_mask_tpf = make_cadence_mask(tpf.time, period, t0,
                                                             t14, verbose=verbose)
                        # pld = tpf.to_corrector(method='pld')
                        pld = lk.PLDCorrector(tpf)
                        corr_lc = pld.correct(cadence_mask=cadence_mask_tpf,
                                aperture_mask=mask, use_gp=use_gp,
                                pld_aperture_mask=mask,
                                #gp_timescale=30, n_pca_terms=10, pld_order=2,
                                ).remove_nans().remove_outliers().normalize()
                    else:
                        #use_sff without restoring trend
                        msg += 'using SFF\n'
                        if verbose:
                            logging.info(msg); print(msg)
                        sff = lk.SFFCorrector(raw_lc)
                        corr_lc = sff.correct(centroid_col=raw_lc.centroid_col,
                                              centroid_row=raw_lc.centroid_row,
                                              polyorder=5, niters=3, bins=SFF_BINSIZE,
                                              windows=SFF_CHUNKSIZE,
                                              sigma_1=3.0, sigma_2=5.0,
                                              restore_trend=True
                                              ).remove_nans().remove_outliers()
                    corr_lcs.append(corr_lc)

                    # get transit mask of corr lc
                    msg='Flattening corrected light curve using Savitzky-Golay filter'
                    if verbose:
                        logging.info(msg); print(msg)
                    cadence_mask_corr = make_cadence_mask(corr_lc.time, period, t0,
                                                         t14, verbose=verbose)
                    #finally flatten
                    flat_lc, trend = corr_lc.flatten(window_length=SG_FILTER_WINDOW_SC,
                                                     mask=cadence_mask_corr,
                                                     return_trend=True)
                else:
                    if verbose:
                        msg='Flattening raw light curve using Savitzky-Golay filter'
                        logging.info(msg); print(msg)
                    cadence_mask_raw = make_cadence_mask(raw_lc.time, period, t0,
                                                         t14, verbose=verbose)
                    flat_lc, trend = raw_lc.flatten(window_length=SG_FILTER_WINDOW_SC,
                                                    mask=cadence_mask_raw,
                                                    return_trend=True)

                # remove obvious outliers and NaN in time
                raw_time_mask = ~np.isnan(raw_lc.time)
                raw_flux_mask = (raw_lc.flux > YLIMIT[0]) | (raw_lc.flux < YLIMIT[1])
                raw_lc = raw_lc[raw_time_mask & raw_flux_mask]
                flat_time_mask = ~np.isnan(flat_lc.time)
                flat_flux_mask = (flat_lc.flux > YLIMIT[0]) | (flat_lc.flux < YLIMIT[1])
                flat_lc = flat_lc[flat_time_mask | flat_flux_mask]
                #if apphot_method!='pdcsap':
                if use_pld or use_sff:
                    trend = trend[flat_time_mask & flat_flux_mask]
                else:
                    trend = trend[raw_time_mask & raw_flux_mask]

                #append with trend
                flat_lcs.append(flat_lc)
                times.append(flat_lc.time)
                fluxes.append(flat_lc.flux)
                flux_errs.append(flat_lc.flux_err)
                trends.append(trend.flux)
                # sectors.append(str(sector))
                cdpps_corr.append(flat_lc.estimate_cdpp())


            if verbose:
                print('Periodogram with TLS\n')
            t = np.array(list(itertools.chain(*times)))
            #idx = np.argsort(t)
            #t = sorted(t)
            f = np.array(list(itertools.chain(*fluxes)))
            e = np.array(list(itertools.chain(*flux_errs)))
            tr = np.array(list(itertools.chain(*trends)))

            if len(all_sectors)>=MAX_SECTORS:
                if CADENCE=='short':
                    cadence_in_minutes = 2*u.minute #np.median(np.diff(t))*u.day.to(u.second)
                else:
                    cadence_in_minutes = 30*u.minute
                #count binsize given old and new cadences
                msg='Number of sectors exceeds {} (ndata={}).\n'.format(MAX_SECTORS,len(t))
                binsize=int(MULTISEC_BIN.to(cadence_in_minutes).value)
                t = binned(t, binsize=binsize)
                f = binned(f, binsize=binsize)
                e = binned(e, binsize=binsize)
                tr= binned(tr, binsize=binsize)
                msg+='Data was binned to {} (ndata={}).\n'.format(MULTISEC_BIN,len(t))
                if verbose:
                    logging.info(msg); print(msg)

            #concatenate time series into one light curve
            full_lc = lk.TessLightCurve(time=t, flux=f, flux_err=e,
                                time_format=time_format, time_scale=time_scale)

            #run TLS with default parameters
            model = transitleastsquares(t, f)
            try:
                ((u1, u2), Ms_tic, _, _, Rs_tic, _, _) = catalog.catalog_info(TIC_ID=int(ticid))
                u1, u2 = DEFAULT_U if not np.all([u1, u2]) else [u1,u2]
                Rs_tic = 1.0 if Rs_tic is None else Rs_tic
                Ms_tic = 1.0 if Ms_tic is None else Ms_tic
            except:
                (u1, u2), Ms_tic, Rs_tic =  DEFAULT_U, 1.0, 1.0 #assume G2 star
            if verbose:
                if u1==DEFAULT_U[0] and u2==DEFAULT_U[1]:
                    print('Using default limb-darkening coefficients\n')
                else:
                    print('Using u1={:.4f},u2={:.4f} based on TIC catalog\n'.format(u1,u2))

            #FIXME: limit period allowing single transits for each sector
            results = model.power(u       = [u1,u2],
                                  limb_dark = 'quadratic',
                                  #R_star  = Rs_tic,
                                  #M_star  = Ms_tic,
                                  #oversampling_factor=3,
                                  #duration_grid_step =1.1
                                  #transit_depth_min=ppm*10**-6,
                                  period_max = 27,
                                  )
            results['u'] = [u1,u2]
            results['Rstar_tic'] = Rs_tic
            results['Mstar_tic'] = Ms_tic

            if verbose:
                #FIXME: compare period from TESS alerts
                msg='Odd-Even transit mismatch: {:.2f} sigma\n'.format(results.odd_even_mismatch)
                msg+='Best period from periodogram: {:.4f} {}\n'.format(results.period,u.day)
                logging.info(msg); print(msg)

            #phase fold
            fold_lc = flat_lc.fold(period=results.period, t0=results.T0)

            #----------ax2: raw lc plot----------
            i=2
            # ax2 = full_lc.errorbar(label='raw lc',ax=ax[i]) <-- plot with single color
            for lc,col,sec in zip(lcs,COLORS,all_sectors):
                cdpp = lc.flatten().estimate_cdpp()
                ax2 = lc.errorbar(color=col,
                      label='s{}: {:.2f}'.format(sec,cdpp), ax=ax[i])
            #some weird outliers do not get clipped, so force ylim
            y1,y2=ax[i].get_ylim()
            if y1<YLIMIT[0]:
                ax[i].set_ylim(YLIMIT[0],y2)
            if y2>YLIMIT[1]:
                ax[i].set_ylim(y1,YLIMIT[1])

            if use_pld==use_sff==False:
                ax[i].plot(t, tr, color='r', linewidth=3, label='Savgol_filter')

            #text = ['s{}: {:.2f}'.format(sector,cdpp) for sector,cdpp in zip(sectors,cdpps_raw)]
            ax[i].legend(title='raw lc cdpp', loc='upper left')

            #----------ax3: long-term variability (+ systematics) correction----------
            i=3
            colors = COLORS[:len(all_sectors)]
            if use_pld or use_sff:
                #----------ax3: systematics-corrected----------
                for corr_lc,col,sec in zip(corr_lcs,colors,all_sectors):
                    cdpp = corr_lc.flatten().estimate_cdpp()
                    ax3 = corr_lc.errorbar(color=col, label='s{}: {:.2f}'.format(sec,cdpp), ax=ax[i])
                ax[i].legend(title='corr lc cdpp', loc='upper left')
                ax[i].plot(t, tr, color='r', linewidth=3, label='Savgol_filter')
            else:
                #ax1 long-term variability-corrected
                for flat_lc,col,sec in zip(flat_lcs,colors,all_sectors):
                    cdpp = flat_lc.estimate_cdpp()
                    ax3 = flat_lc.errorbar(color=col, label='s{}: {:.2f}'.format(sec,cdpp), ax=ax[i])
                ax[i].legend(title='flat lc cdpp', loc='upper left')

            if np.all([results.period,results.T0]):
                try:
                    #index error due to either per or t0
                    tns = get_tns(t, results.period, results.T0)
                    for tt in tns:
                        ax[i].axvline(tt, 0, 1, linestyle='--', color='k', alpha=0.5)
                except:
                    pass
            text = 'PLD={} (gp={}), SFF={}'.format(use_pld,use_gp,use_sff)
            ax[i].text(0.95, 0.15, text,
                verticalalignment='top', horizontalalignment='right',
                transform=ax[i].transAxes, color='green', fontsize=15)

            #----------ax4: periodogram----------
            i=4
            # pg.plot(ax=axs[i], c='k', unit=u.day, view='Period', scale='log', label='periodogram')
            ax[i].axvline(results.period, alpha=0.4, lw=3)
            # plot harmonics: period multiples
            for n in range(2, 10):
                ax[i].axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
                ax[i].axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
            ax[i].set_ylabel(r'SDE')
            ax[i].set_xlabel('Period [days]')
            ax[i].plot(results.periods, results.power,
                       color='black', lw=0.5, label='TLS periodogram')
            ax[i].set_xlim(np.min(results.periods), np.max(results.periods))

            # text = 'Best period={:.2f} {}'.format(period.value, period.unit)
            text = 'Best period={:.2f} {}'.format(results.period, u.day)
            # axs[i].axvline(period.value, 0, 1, linestyle='--', color='k', linewidth=3)
            ax[i].text(0.95, 0.75, text,
                    verticalalignment='top', horizontalalignment='right',
                    transform=ax[i].transAxes, color='green', fontsize=15)
            ax[i].legend()

            #----------ax5: phase folded lc----------
            i=5
            phase_offset=0.5
            ax[i].plot(results.model_folded_phase-phase_offset,
                       results.model_folded_model,
                       color='red', label='TLS model')
            fold_lc.bin(5).scatter(ax=ax[i], color='C1', label='binned (10-min)')
            fold_lc.scatter(ax=ax[i], color='k', alpha=0.1, label='unbinned')
            #ax[i].scatter(results.folded_phase-phase_offset,results.folded_y,
            #              color='k', marker='.', label='unbinned', alpha=0.1, zorder=2)
            ax[i].legend(loc=3)

            rprs= results['rp_rs']
            t14 = results.duration*u.day.to(u.hour)
            t0  = results['T0']

            # manually set lower ylimit for shallow transits
            if rprs<=0.1:
                ylo,yhi = 1-10*rprs**2,1+5*rprs**2
                ax[i].set_ylim(ylo, yhi if yhi<1.02 else 1.02)

            #get gaia stellar params
            gaia_sources = Catalogs.query_region(targ_coord, radius=5*u.arcsec,
                                        catalog="Gaia", version=2).to_pandas()
            gcoords=SkyCoord(ra=gaia_sources['ra'],dec=gaia_sources['dec'],unit='deg')
            idx=targ_coord.separation(gcoords).argmin()
            star=gaia_sources.iloc[idx]
            if star['astrometric_excess_noise_sig']>2:
                print('The target has significant astrometric excess noise: {:.2f}\n'.format(star['astrometric_excess_noise_sig']))
            rstar = star['radius_val']
            rstar_lo = rstar-star['radius_percentile_lower']
            rstar_hi = star['radius_percentile_upper']-rstar
            teff = star['teff_val']
            teff_lo = teff-star['teff_percentile_lower']
            teff_hi = star['teff_percentile_upper']-teff
            if verbose:
                print('Rstar={:.2f} +{:.2f} -{:.2f}'.format(rstar,rstar_lo,rstar_hi))
                print('Teff={:.0f} +{:.0f} -{:.0f}\n'.format(teff,teff_lo,teff_hi))

            Rp = rprs*rstar*u.Rsun.to(u.Rearth)
            if str(rstar)!='nan':
               text = 'Rp={:.2f} Re\nt14={:.2f} hr\nt0={:.6f}'.format(Rp, t14, t0)
            else:
                text = 'Rp/Rs={:.2f}\nt14={:.2f} hr\nt0={:.6f}'.format(rprs, t14, t0)
            if verbose:
                print(text)
            ax[i].text(0.65, 0.25, text,
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax[i].transAxes, color='green', fontsize=15)
            ax[i].legend(loc=3, title='phase-folded lc')
            pl.setp(ax[i], xlim=(-0.2, 0.2), xlabel='Phase', ylabel='Normalized flux');

            #manually set ylim
            if rprs<=0.1:
                ylo,yhi = 1-10*rprs**2,1+5*rprs**2
                ax[i].set_ylim(ylo, yhi if yhi<1.015 else 1.015)

            all_sectors = [str(s) for s in all_sectors]
            if results:
                #append info to tls results
                results['tic'] = ticid
                results['sector'] = all_sectors
                results['Rs'] = rstar
                results['Teff'] = teff
                results['Rp'] = Rp

                fp = join(figoutdir,'tic{}_s{}_tls.hdf5'.format(tpf.targetid,'-'.join(all_sectors)))
                if savefig:
                    dd.io.save(fp, results)
                    print('Saved: {}\n'.format(fp))

            if ticid is None:
                ticid = tpf.targetid
            if toi or toiid:
                #toiid is TOIXXX determined from TESS release queried using TIC or coordinates
                id = toi if toi is not None else toiid
                figname='TIC{}_TOI{}_s{}.png'.format(tpf.targetid, str(id),'-'.join(all_sectors))
                lcname='TIC{}_TOI{}_s{}_lc_flat.txt'.format(tpf.targetid, str(id),'-'.join(all_sectors))
                pl.suptitle('TIC {} (TOI {})'.format(ticid,id), fontsize=FONTSIZE)
            else:
                figname='TIC{}_s{}.png'.format(tpf.targetid,'-'.join(all_sectors))
                lcname='TIC{}_s{}.txt'.format(tpf.targetid,'-'.join(all_sectors))
                pl.suptitle('TIC {})'.format(ticid), fontsize=FONTSIZE)
            figoutpath=join(figoutdir,figname)
            lcoutpath =join(figoutdir,lcname)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            if savefig:
                fig.savefig(figoutpath, bbox_inches='tight')
                print('Saved: {}\n'.format(figoutpath))
                # np.savetxt(lcoutpath, np.c_[t,f], fmt='%.8f')
                final = pd.DataFrame(np.c_[t,f])
                final.to_csv(lcoutpath, index=False, sep=' ')
                print('Saved: {}\n'.format(lcoutpath))
            else:
                pl.show()
            end = time.time()
            msg='#----------Runtime: {:.2f} s----------#\n'.format(end - start)
            logging.info(msg)
            if verbose:
                print(msg)
            pl.close()
        else:
            msg='No tpf file found! Check FFI data using --cadence=long\n'
            logging.info(msg)
            if verbose:
                print(msg)
    except:
        print('Error occured:\n{}'.format(traceback.format_exc()))
        print('\n-----------Some recommendations-----------\n')
        print('Try using -no_gp if [MemoryError: std::bad_alloc]')
        print('Try -c if [buffer is too small for requested array]')
        print('Try --aper={pipeline,threshold,all} if tpf seems corrupted\n\n')
        logging.error(str(traceback.format_exc()))
    #save logs
    logfile = open(LOG_FILENAME, 'rt')
    try:
        body = logfile.read()
    finally:
        logfile.close()
    return res


def generate_FOV(targ_coord,tic=None,toi=None,sector=None,
                 apphot_method='sap',sap_mask='pipeline',aper_radius=None,
                 percentile=None,apply_data_quality_mask=True,
                 fitsoutdir='.',figoutdir='.',savefig=True,
                 clobber=False,verbose=True):
    """Create DSS2 blue and red FOV images

    Parameters
    ----------
    targ_coord : astropy.coordinates
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
        tpf, df = get_tpf(targ_coord, tic=tic, apphot_method='sap',
                          sector=sector, verbose=verbose, clobber=clobber,
                          apply_data_quality_mask=apply_data_quality_mask,
                          sap_mask=sap_mask, fitsoutdir=fitsoutdir, return_df=True)
        #check tpf size
        ny, nx  = tpf.flux.shape[1], tpf.flux.shape[2]
        diag    = np.sqrt(nx**2+ny**2)
        fov_rad = (0.6*diag*TESS_pix_scale).to(u.arcmin)
        if fov_rad>1*u.deg:
            tpf = cutout_tpf(tpf)
            # redefine dimensions
            ny, nx  = tpf.flux.shape[1], tpf.flux.shape[2]
            diag    = np.sqrt(nx**2+ny**2)
            fov_rad = (0.6*diag*TESS_pix_scale).to(u.arcmin)

        #make aperture mask
        mask = parse_aperture_mask(tpf, sap_mask=sap_mask,
               aper_radius=aper_radius, percentile=percentile, verbose=verbose)
        maskhdr = tpf.hdu[2].header
        photwcs = WCS(maskhdr)

        if tpf.targetid is not None:
            ticid = tpf.targetid
        else:
            ticid = df['target_name'].values[0]
        #query tess alerts/ toi release
        q = query_toi(tic=ticid, toi=toi, clobber=clobber, outdir='../data', verbose=False)
        period, t0, t14, depth, toiid = get_transit_params(q, toi=toi, tic=ticid, verbose=False)

        if verbose:
            print('Querying {0} ({1:.2f} x {1:.2f}) archival image'.format('DSS2 Blue',fov_rad))
        nax1, hdu1 = plot_finder_image(targ_coord, fov_radius=fov_rad,
                                        survey='DSS2 Blue', reticle=True)
        pl.close()
        if verbose:
            print('Querying {0} ({1:.2f} x {1:.2f}) archival image'.format('DSS2 Red',fov_rad))
        nax2, hdu2 = plot_finder_image(targ_coord, fov_radius=fov_rad,
                                        survey='DSS2 Red', reticle=True)
        pl.close()
        wcs1 = WCS(hdu1.header)
        wcs2 = WCS(hdu2.header)

        #-----------create figure---------------#
        fig = pl.figure(figsize=(10,6))
        ax1 = fig.add_subplot(121, projection=wcs1)
        ax2 = fig.add_subplot(122, projection=wcs2)

        gaia_sources = Catalogs.query_region(targ_coord, radius=fov_rad,
                                    catalog="Gaia", version=2).to_pandas()

        obj = FixedTarget(targ_coord)
        nax1, hdu1 = plot_finder_image(obj, fov_radius=fov_rad,
                                survey='DSS2 Blue', reticle=True, ax=ax1)
        nax2, hdu2 = plot_finder_image(obj, fov_radius=fov_rad,
                                survey='DSS2 Red', reticle=True, ax=ax2)
        wcs1 = WCS(hdu1.header)
        wcs2 = WCS(hdu2.header)
        mx, my = hdu1.data.shape

        contour = np.zeros((ny, nx))
        contour[np.where(mask)] = 1
        contour = np.lib.pad(contour, 1, PadWithZeros)
        highres = zoom(contour, 100, order=0, mode='nearest')

        extent = np.array([-1, nx, -1, ny])
        #aperture mask
        cs1=ax1.contour(highres, levels=[0.5], extent=extent, origin='lower',
                        colors='y', transform=nax1.get_transform(photwcs))
        cs2=ax2.contour(highres, levels=[0.5], extent=extent, origin='lower',
                        colors='y', transform=nax2.get_transform(photwcs))

        for r,d in gaia_sources[['ra','dec']].values:
            pix1 = wcs1.all_world2pix(np.c_[r,d],1)[0]
            pix2 = wcs2.all_world2pix(np.c_[r,d],1)[0]
            nax1.scatter(pix1[0], pix1[1], marker='s', s=100, edgecolor='r', facecolor='none')
            nax2.scatter(pix2[0], pix2[1], marker='s', s=100, edgecolor='r', facecolor='none')
        pl.setp(nax1, xlim=(0,mx), ylim=(0,my),
                title="DSS2 Blue ({0:.2f}\' x {0:.2f}\')".format(fov_rad.value))
        pl.setp(nax2, xlim=(0,mx), ylim=(0,my),
                title="DSS2 Red ({0:.2f}\' x {0:.2f}\')".format(fov_rad.value))

        if toi or toiid:
            id = str(toi).split('.')[0] if toi is not None else toiid
            figname='TIC{}_TOI{}_FOV_s{}.png'.format(tic,id,sector)
            pl.suptitle('TIC {} (TOI {})'.format(ticid,id), fontsize=FONTSIZE)
        else:
            figname='TIC{}_FOV_s{}.png'.format(tic,sector)
            pl.suptitle('TIC {}'.format(ticid), fontsize=FONTSIZE)
        figoutpath=join(figoutdir,figname)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if savefig:
            fig.savefig(figoutpath, bbox_inches='tight')
            print('Saved: {}\n'.format(figoutpath))
        else:
            pl.show()
        end = time.time()
        msg='#----------Runtime: {:.2f} s----------#\n'.format(end - start)
        logging.info(msg)
        if verbose:
            print(msg)
        pl.close()

    except:
        print('Error occured:\n{}'.format(traceback.format_exc()))
        print('\n-----------Some recommendations-----------\n')
        print('Try using -no_gp if [MemoryError: std::bad_alloc]')
        print('Try -c if [buffer is too small for requested array]')
        print('Try --aper={pipeline,threshold,all} if tpf seems corrupted\n')


def query_toi(toi=None, tic=None, clobber=True, outdir='../data', verbose=True):
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
    assert np.any([tic,toi]), 'Supply toi or tic'
    #TOI csv file from TESS alerts
    dl_link = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
    fp = join(outdir,'TOIs.csv')

    if not exists(fp) or clobber:
        d = pd.read_csv(dl_link)
        d.to_csv(fp, index=False)
        print('Saved: {}\n'.format(fp))
    else:
        d = pd.read_csv(fp)
        #if verbose:
        #    print('Loaded: {}\n'.format(fp))

    if tic:
        q=d[d['TIC ID']==tic]
        #return if empty, else continue
        if len(q)==0:
            return []
    else:
        if isinstance(toi, int):
            toi = float(str(toi)+'.01')
        else:
            planet = str(toi).split('.')[1]
            assert len(planet)==2, 'use pattern: TOI.01'
        q = d[d['TOI']==toi]

    if verbose:
        tic = q['TIC ID'].values[0]
        per = q['Period (days)'].values[0]
        t0  = q['Epoch (BJD)'].values[0]
        t14 = q['Duration (hours)'].values[0]
        dep = q['Depth (ppm)'].values[0]
        comments=q[['TOI','Comments']].values
        print('Data from TOI Releases:\nTIC ID\t{}\nP(d)\t{}\nT0(BJD)\t{} \
               \nT14(hr)\t{}\ndepth(ppm)\t{}\n'.format(tic,per,t0,t14,dep))
        print('Comment:\n{}\n'.format(comments))

    if q['TFOPWG Disposition'].isin(['FP']).any():
        print('\nTFOPWG disposition is a False Positive!\n')

    return q.sort_values(by='TOI')


def get_tois(clobber=True, outdir='../data', verbose=False,
             remove_FP=True, remove_known_planets=True):
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
    dl_link = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
    fp = join(outdir,'TOIs.csv')
    if not exists(outdir):
        makedirs(outdir)

    if not exists(fp) or clobber:
        d = pd.read_csv(dl_link)#, dtype={'RA': float, 'Dec': float})
        #remove False Positives
        if remove_FP:
            d = d[d['TFOPWG Disposition']!='FP']
            print('TOIs with TFPWG disposition==FP are removed.\n')
        if remove_known_planets:
            # d = d['Comments'].str.contains('WASP')
            d = d[~np.array(d['Comments'].str.contains('WASP').tolist(),dtype=bool)]
            # d = d['Comments'].str.contains('HAT')
            d = d[~np.array(d['Comments'].str.contains('HAT').tolist(),dtype=bool)]
            print('WASP and HAT planets are removed.\n')
        d.to_csv(fp, index=False)
        print('Saved: {}\n'.format(fp))
    else:
        d = pd.read_csv(fp)
        #remove False Positives
        d = d[d['TFOPWG Disposition']!='FP']
        print('Loaded: {}'.format(fp))

    return d.sort_values('TOI')

def get_transit_params(q, toi=None, tic=None, verbose=False):
    if len(q)>0:
        if toi:
            toiid = str(toi).split('.')[0]
            assert toi in q['TOI'].values, 'toi not found!'
            planet_num = int(str(toi).split('.')[1])
            idx = planet_num-1
            period = q['Period (days)'].values[idx]
            t0     = q['Epoch (BJD)'].values[idx]
            t14    = q['Duration (hours)'].values[idx]*u.hour.to(u.day)
            depth  = q['Depth (ppm)'].values[idx]
        else:
            #assume first planet since tic cannot specify which planet
            idx = 0
            if tic in q['TIC ID'].values:
                toi = q['TOI'].values[idx]
                toiid = str(q['TOI'].values[idx]).split('.')[0]
                period = q['Period (days)'].values[idx]
                t0     = q['Epoch (BJD)'].values[idx]
                t14    = q['Duration (hours)'].values[idx]*u.hour.to(u.day)
                depth  = q['Depth (ppm)'].values[idx]

                if verbose:
                    print('TIC {} is identified as TOI {} from TESS alerts/TOI Releases...\n'.format(tic, toiid))
            else:
                toiid = str(q['TOI'].values[idx]).split('.')[0]
                period, t0, t14, depth = None, None, None, None
                msg = 'tic not found!'
                print(msg)
                logging.error(msg)

            tois = list(set(q['TOI'].values))
            if verbose and len(tois)>1:
                assert ticid in q['TIC ID'].values
                msg='TIC {} (TOI {}) has {} planets!\n'.format(ticid,toiid,len(q))
                logging.info(msg)
                print(msg)
    else:
        #target is not a TOI
        period, t0, t14, depth = None, None, None, None
        toiid = None
    return period, t0, t14, depth, toiid


def make_cadence_mask(time, period, t0, t14, verbose=False):
    '''Make cadence mask for PLD if ephemeris is known
    '''
    if np.all([period, t0]):
        #if toi with known ephemeris, mask transit so PLD cannot overfit
        #FIXME: PLD should be re-run with ephemeris from TLS
        if (time_format=='btjd') and (t0 > 2450000):
            t0 = t0 - TESS_JD_offset
        w = t14 if t14 is not None else 0.5 # 12 hour width
        tns = get_tns(time, period, t0, allow_half_period=True)
        cadence_mask = np.zeros_like(time).astype(bool)
        for tn in tns:
            #instead of factor of 1/2, 2/3 is used to add padding
            ix = (time > tn-2.*w/3.) & (time < tn+2.*w/3.)
            cadence_mask[ix] = True
        msg = 'with n={} masked transits\n'.format(len(tns))
        assert len(time)==len(cadence_mask)

    else:
        #FIXME: no transit mask usually overfits!
        cadence_mask = None #np.zeros_like(time).astype(bool)
        msg = 'without masked transits\n'
    if verbose:
        logging.info(msg); print(msg)
    return cadence_mask


def plot_gaia_sources(targ_coord, gaia_sources, survey='DSS2 Blue', verbose=True,
                     fov_rad=1*u.arcmin, reticle=True, ax=None):
    """Plot (superpose) Gaia sources on archival image

    Parameters
    ----------
    targ_coord : astropy.coordinates
        target coordinate
    gaia_sources : pd.DataFrame
        gaia sources table
    fov_rad : astropy.unit
        FOV radius
    reticle : bool
        show reticle
    verbose : bool
        print texts
    ax : axis
        subplot axis

    Returns
    -------
    ax : axis
        subplot axis
    """
    #if verbose:
    #    print('Querying {0} ({1} x {1}) archival image'.format(IMAGING_SURVEY,fov_rad))
    nax, hdu = plot_finder_image(targ_coord, fov_radius=fov_rad,
                                 survey=survey, reticle=reticle, ax=ax)
    wcs = WCS(hdu.header)
    img = hdu.data
    mx, my = img.shape[0], img.shape[1]

    for r,d in gaia_sources[['ra','dec']].values:
        pix = wcs.all_world2pix(np.c_[r,d],1)[0]
        ax.scatter(pix[0], pix[1], marker='s', s=100, edgecolor='r', facecolor='none')
    pl.setp(nax, xlim=(0,mx), ylim=(0,my))
    nax.set_title("{0} ({1:.2f}\' x {1:.2f}\')".format(survey,fov_rad.value), fontsize=FONTSIZE)
    return nax, img


def get_gaia_params(targ_coord,gaia_sources,verbose=True):
    '''Get rstar and teff
    '''
    gcoords=SkyCoord(ra=gaia_sources['ra'],dec=gaia_sources['dec'],unit='deg')
    #FIXME: may not correspond to the host if binary
    idx=targ_coord.separation(gcoords).argmin()
    star=gaia_sources.iloc[idx]

    if star['astrometric_excess_noise_sig']>2:
        print('The target has significant astrometric excess noise: {:.2f}\n'.format(star['astrometric_excess_noise_sig']))
    rstar = star['radius_val']
    rstar_lo = rstar-star['radius_percentile_lower']
    rstar_hi = star['radius_percentile_upper']-rstar
    teff = star['teff_val']
    teff_lo = teff-star['teff_percentile_lower']
    teff_hi = star['teff_percentile_upper']-teff
    if verbose:
        print('Rstar={:.2f} +{:.2f} -{:.2f}'.format(rstar,rstar_lo,rstar_hi))
        print('Teff={:.0f} +{:.0f} -{:.0f}\n'.format(teff,teff_lo,teff_hi))
    return rstar, teff


def get_tns(t, p, t0, allow_half_period=False):
    '''Get transit occurrences

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
    '''
    baseline = t[-1]-t[0]
    #assert baseline>p, 'period longer than baseline'
    if allow_half_period and (baseline<p):
        print('P={:.2f}d > time baseline=({:.2f})d\n'.format(p,baseline))
        p=p/2.0
        print('Using P/2={:.2f}'.format(p))

    idx = t != 0
    t = t[idx]

    while t0-p > t.min():
        t0 -= p
    if t0 < t.min():
        t0 += p

    tns = [t0+p*i for i in range(int((t.max()-t0)/p+1))]

    while tns[-1] > t.max():
        tns.pop()

    while tns[0] < t.min():
        tns = tns[1:]

    return np.array(tns)


def PadWithZeros(vector, pad_width, iaxis, kwargs):
    ''' '''
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def collate_tls_results(results_dir,save_csv=False):
    '''Collate TLS results

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
    '''
    if results_dir is None:
        results_dir = '.'
    colnames='tic sector snr SDE FAP rp_rs period T0 duration depth_mean_odd depth_mean_even chi2 transit_count'.split()
    if not exists(results_dir):
        sys.exit('{} does not exist!'.format(results_dir))
    fl = glob(join(results_dir,'*.hdf5'))
    if len(fl)>0:
        pars = []
        for f in tqdm(fl):
            results = dd.io.load(f)
            s=pd.Series([results[col] for col in colnames],index=colnames)
            pars.append(s)
        df=pd.DataFrame(pars).sort_values(by='SDE',ascending=False)
        if save_csv:
            fp = join(results_dir,'tls_summary.csv')
            df.to_csv(fp,index=False)
            print('Saved: {}\n'.format(fp))
        else:
            #SDE=9 == FP rate<1e−4 in the limiting case of white noise
            print(df)
    else:
        sys.exit('No .hdf5 files found in {}\n'.format(results_dir))
    return df

def binned(a, binsize, fun=np.mean):
    '''bin data'''
    a_b = []
    for i in range(0, a.shape[0], binsize):
        a_b.append(fun(a[i:i+binsize], axis=0))
    #make sure no NaN
    #a_b = a_b[~np.isnan(a_b)]
    return a_b


def remove_bad_data(tpf,sector=None,verbose=True):
    '''Remove bad cadences identified in data releae notes

    Parameters
    ----------
    tpf : lk.targetpixelfile

    sector : int
        TESS sector
    verbose : bool
        print texts
    '''
    if sector is None:
        sector = tpf.sector
    if verbose:
        print('Applying data quality mask identified in Data Release Notes (sector {}):'.format(sector))
    if sector==1:
        pointing_jitter_start = 1346
        pointing_jitter_end   = 1350
        if verbose:
            print('t<{}|t>{}\n'.format(pointing_jitter_start,pointing_jitter_end))
        tpf = tpf[(tpf.time < pointing_jitter_start) |
                  (tpf.time > pointing_jitter_end)]
    if sector==2:
        if verbose:
            print('None.\n')
    if sector==3:
        science_data_start = 1385.89663
        science_data_end   = 1406.29247
        if verbose:
            print('t>{}|t<{}\n'.format(science_data_start,science_data_end))
        tpf = tpf[(tpf.time > science_data_start) |
                  (tpf.time < science_data_end)]
    if sector==4:
        guidestar_tables_replaced = 1413.26468
        instru_anomaly_start      = 1418.53691
        data_collection_resumed   = 1421.21168
        if verbose:
            print('t>{}|t<{}|t>{}\n'.format(guidestar_tables_replaced,instru_anomaly_start,data_collection_resumed))
        tpf = tpf[(tpf.time > guidestar_tables_replaced) |
                  (tpf.time < instru_anomaly_start)      |
                  (tpf.time > data_collection_resumed)]
    if sector==5:
        #use of Cam1 in attitude control was disabled for the
        #last ~0.5 days of orbit due to o strong scattered light
        cam1_guide_disabled = 1463.93945
        if verbose:
            print('t<{}\n'.format(cam1_guide_disabled))
        tpf = tpf[tpf.time < cam1_guide_disabled]
    if sector==6:
        #~3 days of orbit 19 were used to collect calibration
        #data for measuring the PRF of cameras;
        #reaction wheel speeds were reset with momentum dumps
        #every 3.125 days
        data_collection_start = 1468.26998
        if verbose:
            print('t>{}\n'.format(data_collection_start))
        tpf = tpf[tpf.time > data_collection_start]
    if sector==8:
        #interruption in communications between instru and spacecraft occurred
        cam1_guide_enabled  = 1517.39566
        orbit23_end         = 1529.06510
        cam1_guide_enabled2 = 1530.44705
        instru_anomaly_start= 1531.74288
        data_colletion_resumed = 1535.00264
        if verbose:
            print('t>{}|t<{}|t>{}|t<{}|t>{}\n'.format(cam1_guide_enabled,
                                       orbit23_end,cam1_guide_enabled2,
                                       instru_anomaly_start,data_colletion_resumed))
        tpf = tpf[(tpf.time > cam1_guide_enabled)   |
                  (tpf.time <=orbit23_end)          |
                  (tpf.time > cam1_guide_enabled2)  |
                  (tpf.time < instru_anomaly_start) |
                  (tpf.time > data_colletion_resumed)]
    if sector==9:
        #use of Cam1 in attitude control was disabled at the
        #start of both orbits due to strong scattered light
        cam1_guide_enabled  = 1543.75080
        orbit25_end         = 1555.54148
        cam1_guide_enabled2 = 1543.75080
        if verbose:
            print('t>{}|t<{}|t>{}\n'.format(cam1_guide_enabled,orbit25_end,cam1_guide_enabled2))
        tpf = tpf[(tpf.time > cam1_guide_enabled) |
                  (tpf.time <=orbit25_end)        |
                  (tpf.time > cam1_guide_enabled2)]
    if sector==10:
        #use of Cam1 in attitude control was disabled at the
        #start of both orbits due to strong scattered light
        cam1_guide_enabled  = 1570.87620
        orbit27_end         = 1581.78453
        cam1_guide_enabled2 = 1584.72342
        if verbose:
            print('t>{}|t<{}|t>{}\n'.format(cam1_guide_enabled,orbit27_end,cam1_guide_enabled2))
        tpf = tpf[(tpf.time > cam1_guide_enabled) |
                  (tpf.time <=orbit27_end)        |
                  (tpf.time > cam1_guide_enabled2)]
    if sector==11:
        #use of Cam1 in attitude control was disabled at the
        #start of both orbits due to strong scattered light
        cam1_guide_enabled  = 1599.94148
        orbit29_end         = 1609.69425
        cam1_guide_enabled2 = 1614.19842
        if verbose:
            print('t>{}|t<{}|t>{}\n'.format(cam1_guide_enabled,orbit29_end,cam1_guide_enabled2))
        tpf = tpf[(tpf.time > cam1_guide_enabled) |
                  (tpf.time <=orbit29_end)        |
                  (tpf.time > cam1_guide_enabled2)]
    return tpf


def cutout_tpf(tpf):
    '''create a smaller cutout of original tpf

    Parameters
    ----------
    tpf : targetpixelfile

    Note
    ----
    This requires new method in trim-tpfs branch of lightkurve
    '''
    ny, nx  = tpf.flux.shape[1], tpf.flux.shape[2]
    print('tpf size=({},{}) pix chosen by TESS pipeline seems too big!\n'.format(ny,nx))
    #choose the length of size
    min_dim = min(tpf.flux.shape)
    new_size = 12 if (min_dim < 8) | (min_dim > 20) else min_dim
    new_center = (min_dim//2, min_dim//2)
    print('Setting tpf size to ({}, {}) pix.\n'.format(new_size,new_size))
    print('This process may take some time.\n')
    tpf = tpf.cutout(center=new_center,size=new_size)
    return tpf


def parse_aperture_mask(tpf, sap_mask='pipeline', aper_radius=None,
                        percentile=None, verbose=False):
    '''Parse and make aperture mask'''
    if verbose:
        if sap_mask=='round':
            print('aperture photometry mask: {} (r={} pix)\n'.format(sap_mask,aper_radius))
        elif sap_mask=='square':
            print('aperture photometry mask: {0} ({1}x{1} pix)\n'.format(sap_mask,aper_radius))
        elif sap_mask=='percentile':
            print('aperture photometry mask: {} ({}%)\n'.format(sap_mask,percentile))
        else:
            print('aperture photometry mask: {}\n'.format(sap_mask))

    #stacked_img = np.median(tpf.flux,axis=0)
    if sap_mask=='all':
        mask = np.ones((tpf.shape[1], tpf.shape[2]), dtype=bool)
    elif sap_mask=='round':
        assert aper_radius is not None, 'supply aper_radius'
        mask = make_round_mask(tpf.flux[0], radius=aper_radius)
    elif sap_mask=='square':
        assert aper_radius is not None, 'supply aper_radius/size'
        mask = make_square_mask(tpf.flux[0], size=aper_radius, angle=None)
    elif sap_mask=='threshold':
        mask = tpf.create_threshold_mask()
    elif sap_mask=='percentile':
        assert percentile is not None, 'supply percentile'
        median_img = np.nanmedian(tpf.flux, axis=0)
        mask = median_img > np.nanpercentile(median_img, percentile)
    else:
         mask = tpf.pipeline_mask #default
    return mask


def make_round_mask(img, radius, xy_center=None):
    '''Make round mask in units of pixels

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
    '''
    h, w = img.shape
    if xy_center is None: # use the middle of the image
        y,x = np.unravel_index(np.argmax(img), img.shape)
        xy_center = [x,y]
        #check if near edge
        if np.any([x>=h-1,x>=w-1,y>=h-1,y>=w-1]):
           print('Brightest star detected is near the edges.')
           print('Aperture mask is placed at the center instead.\n')
           xy_center = [img.shape[0]//2, img.shape[1]//2]

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X-xy_center[0])**2 + (Y-xy_center[1])**2)

    mask = dist_from_center <= radius
    return np.ma.masked_array(img,mask=mask).mask


def make_square_mask(img, size, xy_center=None, angle=None):
    '''Make rectangular mask with optional rotation

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
    '''
    h=w=size
    if xy_center is None: # use the middle of the image
        y,x = np.unravel_index(np.argmax(img), img.shape)
        xy_center = [x,y]
        #check if near edge
        if np.any([x>=h-1,x>=w-1,y>=h-1,y>=w-1]):
           print('Brightest star detected is near the edges.\nAperture mask is placed at the center instead.\n')
           x, y = img.shape[0]//2, img.shape[1]//2
           xy_center = [x,y]
    mask = np.zeros_like(img, dtype=bool)
    mask[y-h:y+h+1,x-w:x+w+1] = True
    if angle:
        #rotate mask
        mask = rotate(mask, angle, axes=(1, 0), reshape=True, output=bool, order=0)
    return mask


def generate_multi_aperture_lc(targ_coord,aper_radii=None,tic=None,toi=None,sector=None,
                 use_pld=False,use_gp=False,use_sff=False,percentiles=None,
                 apphot_method='sap',sap_mask='pipeline',apply_data_quality_mask=True,
                 fitsoutdir='.',figoutdir='.',savefig=True,clobber=False,verbose=True):
    '''Create lc with two aperture masks

    Parameters
    ----------
    targ_coord : astropy.coordinates
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
    '''
    start = time.time()
    try:
        tpf, df = get_tpf(targ_coord, tic=tic,
                          apphot_method=apphot_method, sap_mask=sap_mask,
                          apply_data_quality_mask=apply_data_quality_mask,
                          sector=sector, verbose=verbose, clobber=clobber,
                          fitsoutdir=fitsoutdir, return_df=True)
        all_sectors = [int(i) for i in df['sequence_number'].values]
        if sector is None:
            sector = all_sectors[0]
        #check tpf size
        ny, nx  = tpf.flux.shape[1], tpf.flux.shape[2]
        diag    = np.sqrt(nx**2+ny**2)
        fov_rad = (0.6*diag*TESS_pix_scale).to(u.arcmin)
        if fov_rad>1*u.deg:
            tpf = cutout_tpf(tpf)
            # redefine dimensions
            ny, nx  = tpf.flux.shape[1], tpf.flux.shape[2]
            diag    = np.sqrt(nx**2+ny**2)
            fov_rad = (0.6*diag*TESS_pix_scale).to(u.arcmin)

        # check if target is TOI from TESS alerts
        if tpf.targetid is not None:
            ticid = tpf.targetid
        else:
            ticid = df['target_name'].values[0]
        q = query_toi(tic=ticid, toi=toi, clobber=clobber, outdir='../data', verbose=verbose)
        period, t0, t14, depth, toiid = get_transit_params(q, toi=toi, tic=ticid, verbose=verbose)

        maskhdr = tpf.hdu[2].header
        tpfwcs = WCS(maskhdr)

        if verbose:
            print('Querying {0} ({1:.2f} x {1:.2f}) archival image'.format('DSS2 Blue',fov_rad))
        _, hdu = plot_finder_image(targ_coord, fov_radius=fov_rad,
                                        survey='DSS2 Blue', reticle=True)
        pl.close()

        #-----------create figure---------------#
        fig = pl.figure(figsize=(15,8))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222, projection=WCS(hdu.header))

        #-----ax2-----#
        #query gaia
        gaia_sources = Catalogs.query_region(targ_coord, radius=fov_rad,
                                    catalog="Gaia", version=2).to_pandas()
        #plot gaia sources on archival image
        nax, archival_img = plot_gaia_sources(targ_coord, gaia_sources, verbose=verbose,
                             survey='DSS2 Blue', fov_rad=fov_rad, reticle=True, ax=ax2)

        #get gaia stellar params
        rstar, teff = get_gaia_params(targ_coord,gaia_sources,verbose=verbose)

        lcs = []
        corr_lcs = []
        flat_lcs = []
        trends = []
        folded_lcs = []
        raw_cdpps = []
        corr_cdpps = []
        masks = []
        gcas = []
        colors = ['#1f77b4','#ff7f0e']
        if sap_mask=='square' or sap_mask=='round':
            aper_args = aper_radii
        else:
            aper_args = percentiles
        for n,(color,aper_arg,axn) in enumerate(zip(colors,aper_args,[223,224])):
            print('\n----------aperture {}----------\n'.format(n+1))
            #make aperture mask
            mask = parse_aperture_mask(tpf, sap_mask=sap_mask, aper_radius=aper_arg,
                   percentile=aper_arg, verbose=verbose)
            #-----ax0-----#
            if n==0:
                nax1 = tpf.plot(aperture_mask=mask, ax=ax1)
                ax1.text(0.95, 0.10, 'mask={}'.format(sap_mask),
                    verticalalignment='top', horizontalalignment='right',
                    transform=nax1.transAxes, color='w', fontsize=12)
                ax1.set_title('sector={}'.format(sector), fontsize=FONTSIZE)

            # make lc
            #correct systematics/ filter long-term variability
            #see https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/correctors.py

            #-----ax1-----#
            contour = np.zeros((ny, nx))
            contour[np.where(mask)] = 1
            contour = np.lib.pad(contour, 1, PadWithZeros)
            highres = zoom(contour, 100, order=0, mode='nearest')
            extent = np.array([-1, nx, -1, ny])
            #aperture mask outline
            cs2=ax2.contour(highres, levels=[0.5], extent=extent, origin='lower',
                            colors=color, transform=nax.get_transform(tpfwcs))

            raw_lc = tpf.to_lightcurve(method=PHOTMETHOD, aperture_mask=mask)
            raw_lc = raw_lc.remove_nans().remove_outliers().normalize()
            lcs.append(raw_lc)

            if use_pld or use_sff:
                msg = 'Applying systematics correction:\n'.format(use_gp)
                if use_pld:
                    msg += 'using PLD (gp={})'.format(use_gp)
                    if verbose:
                        logging.info(msg); print(msg)
                    cadence_mask_tpf = make_cadence_mask(tpf.time, period, t0,
                                                         t14, verbose=verbose)
                    # pld = tpf.to_corrector(method='pld')
                    pld = lk.PLDCorrector(tpf)
                    corr_lc = pld.correct(cadence_mask=cadence_mask_tpf,
                            aperture_mask=mask, use_gp=use_gp,
                            pld_aperture_mask=mask,
                            #gp_timescale=30, n_pca_terms=10, pld_order=2,
                            ).remove_nans().remove_outliers().normalize()
                else:
                    #use_sff without restoring trend
                    msg += 'using SFF\n'
                    if verbose:
                        logging.info(msg); print(msg)
                    sff = lk.SFFCorrector(raw_lc)
                    corr_lc = sff.correct(centroid_col=raw_lc.centroid_col,
                                          centroid_row=raw_lc.centroid_row,
                                          polyorder=5, niters=3, bins=SFF_BINSIZE,
                                          windows=SFF_CHUNKSIZE,
                                          sigma_1=3.0, sigma_2=5.0,
                                          restore_trend=True
                                          ).remove_nans().remove_outliers()
                corr_lcs.append(corr_lc)
                # get transit mask of corr lc
                msg='Flattening corrected light curve using Savitzky-Golay filter'
                if verbose:
                    logging.info(msg); print(msg)
                cadence_mask_corr = make_cadence_mask(corr_lc.time, period, t0,
                                                     t14, verbose=verbose)
                #finally flatten
                flat_lc, trend = corr_lc.flatten(window_length=SG_FILTER_WINDOW_SC,
                                                 mask=cadence_mask_corr,
                                                 return_trend=True)
            else:
                if verbose:
                    msg='Flattening raw light curve using Savitzky-Golay filter'
                    logging.info(msg); print(msg)
                cadence_mask_raw = make_cadence_mask(raw_lc.time, period, t0,
                                                     t14, verbose=verbose)
                flat_lc, trend = raw_lc.flatten(window_length=SG_FILTER_WINDOW_SC,
                                                mask=cadence_mask_raw,
                                                return_trend=True)

            # remove obvious outliers and NaN in time
            raw_time_mask = ~np.isnan(raw_lc.time)
            raw_flux_mask = (raw_lc.flux > YLIMIT[0]) | (raw_lc.flux < YLIMIT[1])
            raw_lc = raw_lc[raw_time_mask & raw_flux_mask]
            flat_time_mask = ~np.isnan(flat_lc.time)
            flat_flux_mask = (flat_lc.flux > YLIMIT[0]) | (flat_lc.flux < YLIMIT[1])
            flat_lc = flat_lc[flat_time_mask | flat_flux_mask]

            if use_pld or use_sff:
                trend = trend[flat_time_mask & flat_flux_mask]
            else:
                trend = trend[raw_time_mask & raw_flux_mask]

            flat_lcs.append(flat_lc)
            trends.append(trend)

            if verbose:
                print('Periodogram with TLS\n')
            t = flat_lc.time
            fcor = flat_lc.flux

            # TLS
            model = transitleastsquares(t, fcor)
            #get TIC catalog info: https://github.com/hippke/tls/blob/master/transitleastsquares/catalog.py
            #see defaults: https://github.com/hippke/tls/blob/master/transitleastsquares/tls_constants.py
            try:
                ((u1, u2), Ms_tic, _, _, Rs_tic, _, _) = catalog.catalog_info(TIC_ID=int(ticid))
                u1, u2 = DEFAULT_U if not np.all([u1, u2]) else [u1,u2]
                Rs_tic = 1.0 if Rs_tic is None else Rs_tic
                Ms_tic = 1.0 if Ms_tic is None else Ms_tic
            except:
                (u1, u2), Ms_tic, Rs_tic =  DEFAULT_U, 1.0, 1.0 #assume G2 star
            if verbose:
                if u1==DEFAULT_U[0] and u2==DEFAULT_U[1]:
                    print('Using default limb-darkening coefficients\n')
                else:
                    print('Using u1={:.4f},u2={:.4f} based on TIC catalog\n'.format(u1,u2))

            results = model.power(u=[u1,u2], limb_dark='quadratic')
            # results['u'] = [u1,u2]
            # results['Rstar_tic'] = Rs_tic
            # results['Mstar_tic'] = Ms_tic

            if verbose:
                print('Odd-Even transit mismatch: {:.2f} sigma\n'.format(results.odd_even_mismatch))
                print('Best period from periodogram: {:.4f} {}\n'.format(results.period,u.day))

            #phase fold
            if ~np.any(pd.isnull([results.period,results.T0])):
                #check if TLS input are not np.nan or np.NaN or None
                fold_lc = flat_lc.fold(period=results.period, t0=results.T0)
            # elif ~np.any(pd.isnull([period,t0])):
            #     #check if TLS input are not np.nan or np.NaN or None
            #     fold_lc = flat_lc.fold(period=period, t0=t0)
            else:
                msg='TLS period and t0 search did not yield useful results.\n'
                logging.info(msg)
                sys.exit(msg)

            #-----folded lc-----#
            ax = fig.add_subplot(axn)
            fold_lc.scatter(ax=ax, color='k', alpha=0.1, label='unbinned')
            fold_lc.bin(5).scatter(ax=ax, color=color, label='binned (10-min)')
            ax.plot(results.model_folded_phase-0.5,
                    results.model_folded_model,
                    color='red', label='TLS model')
            #compare depths
            rprs= results['rp_rs']
            ax.axhline(1-rprs**2, 0, 1, color='k', linestyle='--')
            if sap_mask=='round':
                ax.set_title('{} mask (r={} pix)\n'.format(sap_mask,aper_arg), pad=0.1)
            elif sap_mask=='square':
                ax.set_title('{0} mask ({1}x{1} pix)\n'.format(sap_mask,aper_arg), pad=0.1)
            else:
                ax.set_title('{0} mask ({1}\%)\n'.format(sap_mask,aper_arg), pad=0.1)

            # manually set ylimit for shallow transits
            if rprs<=0.1:
                if n==0:
                    ylo,yhi = 1-10*rprs**2,1+5*rprs**2
                    ax.set_ylim(ylo, yhi if yhi<1.02 else 1.02)
                elif n==1:
                    ax.set_ylim(*gcas[0])
            t14 = results.duration*u.day.to(u.hour)
            t0  = results['T0']
            Rp = rprs*rstar*u.Rsun.to(u.Rearth)
            if str(rstar)!='nan':
                text = 'Rp={:.3f} Re\nt14={:.2f} hr\nt0={:.6f}'.format(Rp, t14, t0)
            else:
                text = 'Rp/Rs={:.3f}\nt14={:.2f} hr\nt0={:.6f}'.format(rprs, t14, t0)
            if verbose:
                print(text)
            ax.text(0.6, 0.30, text,
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes, color='k', fontsize=FONTSIZE)
            ax.set_xlim(-0.2,0.2)
            ax.legend(title='phase-folded lc')
            ax.legend(loc=3)
            gcas.append(ax.get_ylim())

        if toi or toiid:
            id = toi if toi is not None else toiid
            figname='TIC{}_TOI{}_FOV_s{}_pla.png'.format(tic,id,sector)
            pl.suptitle('TIC {} (TOI {})'.format(ticid,id), fontsize=FONTSIZE)
        else:
            ticid = tpf.targetid
            figname='TIC{}_FOV_s{}_pla.png'.format(ticid,sector)
            pl.suptitle('TIC {}'.format(ticid), fontsize=FONTSIZE)
        figoutpath=join(figoutdir,figname)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if savefig:
            fig.savefig(figoutpath, bbox_inches='tight')
            print('Saved: {}\n'.format(figoutpath))
        else:
            pl.show()
        end = time.time()
        msg='#----------Runtime: {:.2f} s----------#\n'.format(end - start)
        if verbose:
            logging.info(msg); print(msg)
        pl.close()

    except:
        print('Error occured:\n{}'.format(traceback.format_exc()))
        print('\n-----------Some recommendations-----------\n')
        print('Try -c if [buffer is too small for requested array]')
        print('Try using -no_gp if [MemoryError: std::bad_alloc]')
        print('Try --aper={pipeline,threshold,all} if tpf seems corrupted\n')

def getDistance(x1, y1, x2, y2):
    '''Get pythagorean distance'''
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def isInside(border, target):
    '''Check is star is inside aperture mask'''
    degree = 0
    for i in range(len(border) - 1):
        a = border[i]
        b = border[i + 1]

        # calculate distance of vector
        A = getDistance(a[0], a[1], b[0], b[1]);
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
        if(clockwise):
            degree = degree + math.degrees(math.acos((B * B + C * C - A * A) / (2.0 * B * C)))
        else:
            degree = degree - math.degrees(math.acos((B * B + C * C - A * A) / (2.0 * B * C)))

    if(abs(round(degree) - 360) <= 3):
        return True
    return False
