#!/usr/bin/env python
import numpy as np
from astropy import units as u
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord, Distance
from ticgen import Star

try:
    from tql import tql
except:
    # relative import
    # import sys
    # sys.path.append('..')
    import tql


def validate_inputs(toi=None, tic=None):
    if (toi is not None) & (len(str(toi).split(".")) == 2):
        toi_id = int(str(toi).split(".")[0])
        return toi_id
    if (tic is not None) & (len(str(tic).split(".")) == 2):
        tic_id = int(str(tic).split(".")[0])
        return tic_id


class TOIParams:
    """get parameters from TOI table

    see column meanings here:
    https://outerspace.stsci.edu/display/TESS/TIC+v8+and+CTL+v8.xx+Data+Release+Notes
    """

    def __init__(
        self,
        target_coord=None,
        toi_id=None,
        tic_id=None,
        gaia_id=None,
        search_radius=15 * u.arcsec,
        clobber=False,
        verbose=False,
    ):
        self.target_coord = target_coord
        self.toi_id = toi_id
        self.tic_id = tic_id
        self.gaia_id = gaia_id
        self.search_radius = search_radius
        self.verbose = verbose
        self.clobber = clobber

        if not np.any([self.target_coord, self.toi_id, self.tic_id]):
            raise ValueError("Provide target_coord or toi_id or tic_id")

        if np.any([self.toi_id, self.tic_id]):
            self.toi_params = tql.get_toi(
                toi=self.toi_id,
                tic=self.tic_id,
                clobber=self.clobber,
                verbose=self.verbose,
            )
            if len(self.toi_params) > 0:
                if self.target_coord is None:
                    self.target_coord = SkyCoord(
                        ra=self.toi_params["RA"],
                        dec=self.toi_params["Dec"],
                        unit=(u.hourangle, u.degree),
                    )[0]
                # set values if TOI
                # photometry
                self.Tmag = self.toi_params["TESS Mag"].values[0] * u.mag
                # parallax & proper motion
                self.pmra = (
                    self.toi_params["PM RA (mas/yr)"].values[0]
                    * u.mas
                    / u.year
                )
                self.pmdec = (
                    self.toi_params["PM Dec (mas/yr)"].values[0]
                    * u.mas
                    / u.year
                )
                self.distance = (
                    self.toi_params["Stellar Distance (pc)"].values[0] * u.pc
                )
                self.parallax = Distance(self.distance).parallax.value * u.mas
                # stellar parameters
                self.Teff_tic = (
                    self.toi_params["Stellar Eff Temp (K)"].values[0] * u.K
                )
                self.Rstar_tic = (
                    self.toi_params["Stellar Radius (R_Sun)"].values[0]
                    * u.Rsun
                )

                # set tic_id or toi_id
                if self.tic_id is None:
                    self.tic_id = self.toi_params["TIC ID"].values[0]
                elif self.toi_id is None:
                    self.toi_id = int(
                        self.toi_params["TOI"]
                        .astype(str)
                        .values[0]
                        .split(".")[0]
                    )
        #     else:
        #         raise ValueError(f'{self.target_coord} does not match TOI Release catalog!')
        else:
            # position match
            all_tois = tql.get_tois(
                clobber=False,
                verbose=False,
                remove_known_planets=False,
                remove_FP=True,
            )
            toi_coords = SkyCoord(
                ra=all_tois["RA"],
                dec=all_tois["Dec"],
                unit=(u.hourangle, u.degree),
            )
            separation = target_coord.separation(toi_coords)
            idx = separation < self.search_radius
            self.toi_params = all_tois.loc(idx)
            if len(self.toi_params) == 0:
                if self.verbose:
                    print(
                        f"{self.target_coord} does not match TOI Release catalog!"
                    )
            elif len(self.toi_params) > 1:
                # take the nearest toi
                self.toi_params = self.toi_params.iloc[separation.argmin()]


class TICParams(TOIParams):
    """get parameters from TIC catalog"""

    def __init__(
        self,
        target_coord=None,
        toi_id=None,
        tic_id=None,
        gaia_id=None,
        search_radius=15 * u.arcsec,
        clobber=False,
        verbose=False,
    ):
        super().__init__(
            target_coord,
            toi_id,
            tic_id,
            gaia_id,
            search_radius,
            clobber,
            verbose,
        )
        if self.target_coord is not None:
            # search_radius can be changed
            self.tic_catalog = self.query_tic()
        else:
            raise ValueError("Provide target_coord")

        if (self.toi_id is None) | (self.Tmag is None):
            self.match_single_star()

    def query_tic(self, radius=None):
        # set by: v.tic_catalog = v.query_tic()
        radius = self.search_radius if radius is None else radius * u.arcsec
        cat = Catalogs.query_region(
            self.target_coord, radius=radius, catalog="TIC"
        ).to_pandas()
        if self.verbose:
            print(
                f"""Querying TIC catalog for {self.target_coord} within
                                                    {self.search_radius}.\n"""
            )
        return cat

    def match_single_star(self):
        if len(self.tic_catalog) == 0:
            raise ValueError(
                f"{self.target_coord} has NO match in TIC catalog!"
            )
        elif len(self.tic_catalog) == 1:
            assert self.toi is None
            if self.gaia_id is None:
                self.gaia_id = self.tic_catalog["GAIA"].values[0] * u.mag
            # photometry
            self.Tmag = self.tic_catalog["Tmag"].values[0] * u.mag
            self.Vmag = self.tic_catalog["Vmag"].values[0] * u.mag
            self.Gmag = self.tic_catalog["GAIAmag"].values[0] * u.mag
            # parallax & proper motion
            self.pmra = self.tic_catalog["pmRA"].values[0] * u.mas / u.year
            self.pmdec = self.tic_catalog["pmDEC"].values[0] * u.mas / u.year
            self.parallax = self.tic_catalog["plx"].values[0] * u.mas
            self.distance = Distance(parallax=self.parallax).pc * u.pc
            # stellar parameters
            self.Teff_tic = self.tic_catalog["Teff"].values[0] * u.K
            self.Rstar_tic = self.tic_catalog["rad"].values[0] * u.Rsun
            self.Mstar_tic = self.tic_catalog["mass"].values[0] * u.Msun
            self.Rhostar_tic = self.tic_catalog["rho"].values[0] * u.g / u.cm3

        else:  # len(self.tic_catalog)>1:
            # match if gaia_id is known
            if self.gaia_id is not None:
                idx = self.tic_catalog["GAIA"].isin([self.gaia_id])
                star = self.tic_catalog.loc[idx]
                if len(star) > 0:
                    return star
                else:
                    raise ValueError(
                        f"""Gaia source id: {self.source_id} not found
                                     in TIC catalog within {self.search_radius} of
                                     {self.target_coord}.\n"""
                    )
            else:
                # match given position and magnitude
                tic_coords = SkyCoord(
                    ra=self.tic_catalog["ra"],
                    dec=self.tic_catalog["dec"],
                    unit=(u.degree),
                )
                separation = target_coord.separation(tic_coords)
                idx = separation < self.search_radius / 2.0
                self.tic_catalog = self.tic_catalog.loc(idx)
                if len(self.tic_catalog) == 0:
                    if self.verbose:
                        print(
                            f"{self.target_coord} does not match any star in TIC catalog!"
                        )
                elif len(self.tic_catalog) > 1:
                    # take the nearest tic
                    self.tic_catalog = self.tic_catalog.iloc[
                        separation.argmin()
                    ]


class GaiaParams(TICParams):
    """
    see column meanings here:
    https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/
    chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
    """

    def __init__(
        self,
        target_coord=None,
        toi_id=None,
        tic_id=None,
        gaia_id=None,
        search_radius=15 * u.arcsec,
        clobber=False,
        verbose=False,
    ):
        super().__init__(
            target_coord,
            toi_id,
            tic_id,
            gaia_id,
            search_radius,
            clobber,
            verbose,
        )
        self.gaia_catalog = self.query_gaia_dr2()

    def query_gaia_dr2(self):
        if self.verbose:
            print(
                f"""Querying Gaia DR2 catalog for {self.target_coord} within
                                                    {self.search_radius}.\n"""
            )
        cat = Catalogs.query_region(
            self.target_coord,
            radius=self.search_radius,
            catalog="Gaia",
            version=2,
        ).to_pandas()
        return cat

    def match_single_star(self):
        if self.gaia_id is not None:
            idx = self.gaia_catalog["source_id"].isin([self.gaia_id])
            star = self.gaia_catalog.loc[idx]
            if len(star) == 0:
                raise ValueError(
                    f"""Gaia source id: {self.source_id} not found
                                 in Gaia DR2 catalog within {self.search_radius}
                                 of {self.target_coord}.\n"""
                )
            elif len(star) == 1:
                return star
            else:
                # possible duplicate matches so get one only
                star_coords = SkyCoord(
                    ra=star["ra"], dec=star["dec"], unit="deg"
                )
                sep = self.target_coord.separation(star_coords)
                # sep < sep_tol
                pass
        else:
            # match given position and magnitude
            if len(self.gaia_catalog) == 1:
                gaia_par = self.gaia_catalog["parallax"].values[0]
                Gmag = self.gaia_catalog["phot_g_mean_mag"].values[0]
                assert round(self.parallax) != round(gaia_par)
                # if Vmag with Gmag is available, use ticgen to predict Tmag
                assert abs(Tmag - Gmag) < 1
            else:
                # nearest neighbor search
                pass
