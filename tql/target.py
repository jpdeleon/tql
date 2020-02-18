# -*- coding: utf-8 -*-

r"""
classes for searching object
"""

# Import standard library
from os.path import join, exists
import logging
import re

# Import modules
import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
from astroquery.mast import Catalogs
from astroquery.mast.tesscut import Tesscut
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u
import lightkurve as lk

# from lightkurve.search import _query_mast as query_mast
# Import from package
from tql.utils import (
    get_toi,
    get_tois,
    get_target_coord,
    get_target_coord_3d,
    get_mamajek_table,
)

log = logging.getLogger(__name__)

__all__ = ["Target"]


class Target:
    def __init__(
        self,
        name=None,
        toiid=None,
        ticid=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        search_radius=3 * u.arcsec,
        verbose=True,
        clobber=False,
    ):
        self.toiid = toiid  # e.g. 837
        self.ticid = ticid  # e.g. 364107753
        self.epicid = epicid  # 201270176
        self.gaiaid = gaiaDR2id  # e.g. Gaia DR2 5251470948229949568
        self.target_name = name  # e.g. Pi Mensae
        self.search_radius = search_radius
        self.tic_params = None
        self.gaia_params = None
        self.toi_params = None
        self.gmag = None
        self.tesscut_tpf = None
        self.corrector = None
        self.clobber = clobber
        self.verbose = verbose

        if name:
            if name[:4].lower() == "epic":
                if epicid is None:
                    self.epicid = int(name.strip()[4:])
            elif name[:3].lower() == "tic":
                if ticid is None:
                    self.ticid = int(name.strip()[3:])
            elif name[:4].lower() == "gaia":
                if gaiaDR2id is None:
                    self.gaiaid = int(name.strip()[7:])
        # self.sector = sector
        # self.mission = mission
        self.ra = ra_deg
        self.dec = dec_deg
        # self.distance = None
        if (self.ticid is not None) and (self.toiid is None):
            tois = get_tois(clobber=True, verbose=False)
            idx = tois["TIC ID"].isin([self.ticid])
            if sum(idx) > 0:
                self.toiid = tois.loc[idx, "TOI"].values[0]
                if self.verbose:
                    print(f"TIC {self.ticid} is TOI {int(self.toiid)}!")
        if self.toiid is not None:
            self.toi_params = get_toi(
                toi=self.toiid, clobber=self.clobber, verbose=False
            )
        if (self.ticid is None) and (self.toiid is not None):
            self.ticid = int(self.toi_params["TIC ID"].values[0])
        self.target_coord = get_target_coord(
            ra=self.ra,
            dec=self.dec,
            toi=self.toiid,
            tic=self.ticid,
            epic=self.epicid,
            gaiaid=self.gaiaid,
            name=self.target_name,
        )
        self.ccd_info = Tesscut.get_sectors(self.target_coord).to_pandas()

    def get_all_sectors(self):
        """
        """
        df = self.ccd_info
        all_sectors = [int(i) for i in df["sector"].values]
        return all_sectors

    def get_sector_cam_ccd(self):
        """get TESS sector, camera, and ccd numbers using Tesscut
        """
        all_sectors = self.get_all_sectors()
        df = self.ccd_info
        if self.sector:
            sector = self.sector
            sector_idx = df["sector"][
                df["sector"].isin([self.sector])
            ].index.tolist()
            if len(sector_idx) == 0:
                raise ValueError(f"Available sector(s): {all_sectors}")
            cam = str(df.iloc[sector_idx]["camera"].values[0])
            ccd = str(df.iloc[sector_idx]["ccd"].values[0])
        else:
            sector_idx = 0
            sector = str(df.iloc[sector_idx]["sector"])
            cam = str(df.iloc[sector_idx]["camera"])
            ccd = str(df.iloc[sector_idx]["ccd"])
        return sector, cam, ccd

    def estimate_Av(self, map="sfd", constant=3.1):
        """
        compute the extinction Av from color index E(B-V)
        estimated from dustmaps via Av=constant*E(B-V)

        Parameters
        ----------
        map : str
            dust map; see https://dustmaps.readthedocs.io/en/latest/maps.html
        """
        try:
            import dustmaps
        except ImportError:
            print("pip install dustmaps")

        if map == "sfd":
            from dustmaps import sfd

            # sfd.fetch()
            dust_map = sfd.SFDQuery()
        elif map == "planck":
            from dustmaps import planck

            # planck.fetch()
            dust_map = planck.PlanckQuery()
        else:
            raise ValueError(f"Available maps: (sfd,planck)")

        ebv = dust_map(self.target_coord)
        Av = constant * ebv
        return Av

    def query_toi(self, toi=None, **kwargs):
        d = get_toi(toi=toi, **kwargs)
        if d is not None:
            self.toi_params = d
        return d

    def query_gaia_dr2_catalog(self, radius=None, return_nearest_xmatch=False):
        """
        Parameter
        ---------
        radius : float
            query radius in arcsec
        return_nearest_xmatch : bool
            return nearest single star if True else possibly more matches

        Returns
        -------
        tab : pandas.DataFrame
            table of star match(es)

        Notes:
        1. See column meaning here: https://mast.stsci.edu/api/v0/_c_a_o_mfields.html

        From Carillo+2019:
        2. the sample with the low parallax errors i.e. 0 < f < 0.1,
        has distances derived from simply inverting the parallax

        Whereas, the sample with higher parallax errors i.e. f > 0.1
        has distances derived from a Bayesian analysis following Bailer-Jones (2015),
        where they use a weak distance prior (i.e. exponentially decreasing space
        density prior) that changes with Galactic latitude and longitude

        3. See also Gaia DR2 Cross-match for the celestial reference system (ICRS)
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_proc/ssec_cu3ast_proc_xmatch.html
        and
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_cali/ssec_cu3ast_cali_frame.html
        """
        radius = self.search_radius if radius is None else radius * u.arcsec
        if self.verbose:
            print(
                f"""Querying Gaia DR2 catalog for {self.target_coord.to_string()} within {radius}.\n"""
            )
        # load gaia params for all TOIs
        tab = Catalogs.query_region(
            self.target_coord, radius=radius, catalog="Gaia", version=2
        ).to_pandas()
        tab["source_id"] = tab.source_id.astype(int)

        # check if results from DR2 (epoch 2015.5)
        assert np.all(
            tab["ref_epoch"].isin([2015.5])
        ), "Epoch not 2015 (version<2?)"
        assert len(tab) > 0, f"use radius>{radius}"

        if np.any(tab["parallax"] < 0):
            # use positive parallaxes only
            # tab = tab[tab["parallax"]> 0] #this drops NaN too
            tab = tab[(tab["parallax"] >= 0) | (tab["parallax"].isnull())]
        """
        check parallax error here and apply corresponding distance calculation: see Note 1
        """
        if self.gaiaid is not None:
            errmsg = "Catalog does not contain target gaia id."
            assert np.any(tab["source_id"].isin([self.gaiaid])), errmsg

        # add gaia distance to target_coord
        # FIXME: https://docs.astropy.org/en/stable/coordinates/transforming.html
        gcoords = SkyCoord(
            ra=tab["ra"],
            dec=tab["dec"],
            unit="deg",
            frame="icrs",
            obstime="J2015.5",
        )
        # precess coordinate from Gaia DR2 epoch to J2000
        gcoords = gcoords.transform_to("icrs")
        if self.gaiaid is not None:
            # find by id match for gaiaDR2id input
            idx = tab.source_id.isin([self.gaiaid]).argmax()
        else:
            # find by nearest distance (for toiid or ticid input)
            idx = self.target_coord.separation(gcoords).argmin()
        star = tab.loc[idx]
        # get distance from parallax
        target_dist = Distance(parallax=star["parallax"] * u.mas)
        # redefine skycoord with coord and distance
        target_coord = SkyCoord(
            ra=self.target_coord.ra,
            dec=self.target_coord.dec,
            distance=target_dist,
        )
        self.target_coord = target_coord

        if return_nearest_xmatch or (len(tab) == 1):
            self.gaiaid = int(tab.loc[0, "source_id"])
            self.gaia_params = tab.iloc[0]
            self.gmag = self.gaia_params["phot_g_mean_mag"]
            return tab.iloc[0]  # return series of len 1
        else:
            # if self.verbose:
            #     d = self.get_nearby_gaia_sources()
            #     print(d)
            self.gaia_params = tab
            return tab  # return dataframe of len 2 or more

    def get_nearby_gaia_sources(self, radius=60.0, add_column=None):
        """
        get information about stars within 60 arcsec and
        dilution factor from delta Gmag

        Parameters
        ----------
        radius : float
            query radius in arcsec
        add_column : str
            additional Gaia column name to show (e.g. radial_velocity)
        """
        if self.gaia_params is None:
            _ = self.query_gaia_dr2_catalog(radius=60)
        if len(self.gaia_params) == 1:
            _ = self.query_gaia_dr2_catalog(radius=60)
        d = self.gaia_params.copy()

        if self.gaiaid is None:
            # nearest match (first entry row=0) is assumed as the target
            gaiaid = int(d.iloc[0]["source_id"])
        else:
            gaiaid = self.gaiaid
        idx = d.source_id == gaiaid
        target_gmag = d.loc[idx, "phot_g_mean_mag"].values[0]
        d["distance"] = d["distance"].apply(
            lambda x: x * u.arcmin.to(u.arcsec)
        )
        d["delta_Gmag"] = d["phot_g_mean_mag"] - target_gmag
        # compute dilution factor
        d["gamma"] = 1 + 10 ** (0.4 * d["delta_Gmag"])
        columns = [
            "source_id",
            "distance",
            "parallax",
            "phot_g_mean_mag",
            "delta_Gmag",
            "gamma",
        ]
        if add_column is not None:
            assert (isinstance(add_column, str)) & (add_column in d.columns)
            columns.append(add_column)
        return d[columns]

    def get_possible_NEBs(self, depth, gaiaid=None, kmax=1.0):
        """
        depth is useful to rule out deep eclipses when kmax/gamma>depth

        kmax : float [0,1]
            maximum eclipse depth (default=1)
        """
        assert (kmax >= 0.0) & (kmax <= 1.0), "eclipse depth is between 0 & 1"

        d = self.get_nearby_gaia_sources()

        good, bad = [], []
        for id, dmag, gamma in d[["source_id", "delta_Gmag", "gamma"]].values:
            if int(id) != gaiaid:
                if depth > kmax / gamma:
                    # observed depth is too deep to have originated from the secondary star
                    good.append(id)
                else:
                    # uncertain signal source
                    bad.append(id)
        uncleared = d.loc[d.source_id.isin(bad)]
        return uncleared

    def query_tic_catalog(self, radius=None, return_nearest_xmatch=False):
        """
        Parameter
        ---------
        radius : float
            query radius in arcsec

        Returns
        -------
        tab : pandas.DataFrame
            table of star match(es)
        """
        radius = self.search_radius if radius is None else radius * u.arcsec
        if self.verbose:
            print(
                f"""Querying TIC catalog for {self.target_coord.to_string()}
            within {radius}.\n"""
            )
        # NOTE: check tic version
        tab = Catalogs.query_region(
            self.target_coord, radius=radius, catalog="TIC"
        ).to_pandas()
        if return_nearest_xmatch:
            tab = tab.iloc[0]
        self.tic_params = tab
        return tab

    def get_spec_type(
        self,
        columns="Teff B-V J-H H-Ks".split(),
        nsamples=int(1e4),
        return_samples=False,
        clobber=False,
    ):
        """
        Interpolate spectral type from Mamajek table from
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        based on observables Teff and color indices.

        Parameters
        ----------
        columns : list
            column names of input parameters
        nsamples : int
            number of Monte Carlo samples (default=1e4)
        clobber : bool (default=False)
            re-download Mamajek table

        Returns
        -------
        interpolated spectral type

        Notes:
        It may be good to check which color index yields most accurate result
        """
        df = get_mamajek_table(clobber=clobber, verbose=self.verbose)
        if self.gaia_params is None:
            self.gaia_params = self.query_gaia_dr2_catalog(
                return_nearest_xmatch=True
            )
        if self.tic_params is None:
            self.tic_params = self.query_tic_catalog(
                return_nearest_xmatch=True
            )

        # effective temperature
        col = "teff"
        teff = self.gaia_params[f"{col}_val"]
        siglo = (
            self.gaia_params[f"{col}_val"]
            - self.gaia_params[f"{col}_percentile_lower"]
        )
        sighi = (
            self.gaia_params[f"{col}_percentile_upper"]
            - self.gaia_params[f"{col}_val"]
        )
        uteff = np.sqrt(sighi ** 2 + siglo ** 2)
        s_teff = (
            teff + np.random.randn(nsamples) * uteff
        )  # Monte Carlo samples

        # B-V color index
        bv_color = self.tic_params["Bmag"] - self.tic_params["Vmag"]
        ubv_color = (
            self.tic_params["e_Bmag"] + self.tic_params["e_Vmag"]
        )  # uncertainties add
        s_bv_color = (
            bv_color + np.random.randn(nsamples) * ubv_color
        )  # Monte Carlo samples

        # J-H color index
        jh_color = self.tic_params["Jmag"] - self.tic_params["Hmag"]
        ujh_color = (
            self.tic_params["e_Jmag"] + self.tic_params["e_Hmag"]
        )  # uncertainties add
        s_jh_color = (
            jh_color + np.random.randn(nsamples) * ujh_color
        )  # Monte Carlo samples

        # H-K color index
        hk_color = self.tic_params["Hmag"] - self.tic_params["Kmag"]
        uhk_color = (
            self.tic_params["e_Hmag"] + self.tic_params["e_Kmag"]
        )  # uncertainties add
        s_hk_color = (
            hk_color + np.random.randn(nsamples) * uhk_color
        )  # Monte Carlo samples

        # Interpolate
        interp = NearestNDInterpolator(
            df[columns].values, df["#SpT"].values, rescale=False
        )
        samples = interp(s_teff, s_bv_color, s_jh_color, s_hk_color)
        # encode category
        spt_cats = pd.Series(samples, dtype="category")  # .cat.codes
        spt = spt_cats.mode().values[0]
        if return_samples:
            return spt, samples
        else:
            return spt

    def make_custom_ffi_lc(
        self,
        sector=None,
        cutout_size=(50, 50),
        mask_threshold=3,
        pca_nterms=5,
        with_offset=True,
    ):
        """
        create a custom lightcurve based on this tutorial:
        https://docs.lightkurve.org/tutorials/04-how-to-remove-tess-scattered-light-using-regressioncorrector.html

        Parameters
        ----------
        sector : int or str
            specific sector or all
        cutout_size : tuple
            tpf cutout size
        mask_threshold : float
            threshold (sigma) to create aperture mask
        pca_nterms : int
            number of pca terms to use

        Returns
        -------
        corrected_lc : lightkurve object
        """
        if sector is None:
            all_sectors = self.get_all_sectors()
            sector = all_sectors[0]
            print(f"Available sectors: {all_sectors}")
            print(f"sector {sector} is used.\n")
        if self.tesscut_tpf is not None:
            if self.tesscut_tpf.sector == sector:
                tpf = self.tesscut_tpf
            else:
                tpf = lk.search_tesscut(
                    self.target_coord, sector=sector
                ).download(cutout_size=cutout_size)
                self.tesscut_tpf = tpf
        else:
            tpf = lk.search_tesscut(self.target_coord, sector=sector).download(
                cutout_size=cutout_size
            )
            self.tesscut_tpf = tpf

        # remove zeros
        zero_mask = (tpf.flux_err == 0).all(axis=(1, 2))
        if zero_mask.sum() > 0:
            tpf = tpf[~zero_mask]
        # Make an aperture mask and a raw light curve
        aper = tpf.create_threshold_mask(threshold=mask_threshold)
        raw_lc = tpf.to_lightcurve(aperture_mask=aper)

        # Make a design matrix and pass it to a linear regression corrector
        regressors = tpf.flux[:, ~aper]
        dm = (
            lk.DesignMatrix(regressors, name="regressors")
            .pca(nterms=pca_nterms)
            .append_constant()
        )
        rc = lk.RegressionCorrector(raw_lc)
        # if remove_outliers:
        #     clean_lc, outlier_mask = tpf.to_lightcurve(
        #         aperture_mask=aper
        #     ).remove_outliers(return_mask=True)
        #     regressors = tpf.flux[:, ~aper][~outlier_mask]
        #     dm = lk.DesignMatrix(regressors, name="regressors").pca(nterms=pca_nterms).append_constant()
        #     rc = lk.RegressionCorrector(clean_lc)
        self.corrector = rc
        corrected_lc = rc.correct(dm)

        # Optional: Remove the scattered light, allowing for the large offset from scattered light
        if with_offset:
            corrected_lc = (
                raw_lc - rc.model_lc + np.percentile(rc.model_lc.flux, q=5)
            )
        return corrected_lc.normalize()
