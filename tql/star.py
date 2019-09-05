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
    if (toi is not None) & (len(str(toi).split('.'))==2):
        toi_id = int(str(toi).split('.')[0])
        return toi_id
    if (tic is not None) (len(str(tic).split('.'))==2):
        tic_id = int(str(tic).split('.')[0])
        return tic_id

class TICParams:
    '''
    see column meanings here:
    https://outerspace.stsci.edu/display/TESS/TIC+v8+and+CTL+v8.xx+Data+Release+Notes
    '''

    def __init__(self, target_coords=None, toi=None, tic_id=None, gaia_id=None,
                 search_radius=15*u.arcsec, clobber=False, verbose=False):
        self.target_coord = target_coord
        self.toi_id = toi_id
        self.tic_id = tic_id
        self.gaia_id = gaia_id
        self.search_radius = search_radius
        self.verbose = verbose
        self.clobber = clobber

        self.tic_catalog = None
        if np.any([self.toi_id, self.tic_id]):
            self.toi_params = tql.get_toi(toi=self.toi_id, tic=self.tic_id,
                                          clobber=self.clobber, verbose=self.verbose)
            #set tic_id or toi_id
            if self.tic_id is None:
                self.tic_id = self.toi_params['TIC ID'].values[0]
            elif self.toi_id is None:
                self.toi_id = int(self.toi_params['TOI'].astype(str).values[0].split('.')[0])


        if self.target_coord is None:
            if (self.toi_id is not None) | (self.tic_id is not None):
                #FIXME: toi_id can only match TOI.01
                self.target_coord = SkyCoord(ra=self.toi_params['RA'],
                                dec=self.toi_params['Dec'],
                                unit=(u.hourangle, u.deg))[0]
            self.tic_catalog = self.query_tic()

        if self.gaia_id is None:
            #self.gaia_id = self.match_single_star()
            pass

    def query_tic(self):
        cat = Catalogs.query_region(self.target_coord, radius=self.search_radius,
                                        catalog='TIC').to_pandas()
        if self.verbose:
            print(f"""Querying TIC catalog for {self.target_coord} within
                                                    {self.search_radius}.\n""")
        self.tic_catalog = cat
        return cat

    def match_single_star(self):
        if self.gaia_id is not None:
            idx = self.tic_catalog['GAIA'].isin([self.gaia_id])
            star = self.tic_catalog.loc[idx]
            if len(star)>0:
                return star
            else:
                raise ValueError(f"""Gaia source id: {self.source_id} not found
                                 in TIC catalog within {self.search_radius} of
                                 {self.target_coord}.\n""")

        else:
            if len(self.tic_catalog)==1:
                #toi release table info
                # radec = self.toi_params['RA','Dec'].values[0]
                Tmag = self.toi_params['TESS Mag'].values[0]
                pmra = self.toi_params['PM RA (mas/yr)'].values[0]
                pmdec = self.toi_params['PM Dec (mas/yr)'].values[0]
                parallax = Distance(self.toi_params['Stellar Distance (pc)'].values[0],
                                    unit='pc').parallax.value
                Teff = self.toi_params['Stellar Eff Temp (K)'].values[0]
                Rstar = self.toi_params['Stellar Radius (Rsun)'].values[0]

                #TIC catalog info
                cat = self.tic_catalog
                Vmag = cat['Vmag'].values[0]
                Gmag = cat['Gmag'].values[0]
                #compute Tmag using ticgen
                Tmag_ = Star(Vmag=Vmag, Gmag=Gmag)
                assert abs(Tmag-Tmag_)<1
                assert
            else:
                #nearest neighbor search
                pass

class GaiaParams(TICParams):
    '''
    see column meanings here:
    https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/
    chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
    '''

    def __init__(self, target_coord=None, toi_id=None, tic_id=None, gaia_id=None,
                 search_radius=15*u.arcsec, clobber=False, verbose=False):
        super().__init__(target_coord, toi_id, tic_id, gaia_id,
                         search_radius, clobber, verbose)
        self.gaia_catalog = self.query_gaia_dr2()

    def query_gaia_dr2(self):
        if self.verbose:
            print(f"""Querying Gaia DR2 catalog for {self.target_coord} within
                                                    {self.search_radius}.\n""")
        cat = Catalogs.query_region(self.target_coord, radius=self.search_radius,
                                        catalog='Gaia', version=2).to_pandas()
        return cat

    def match_single_star(self):
        if self.gaia_id is not None:
            idx = self.gaia_catalog['source_id'].isin([self.gaia_id])
            star = self.gaia_catalog.loc[idx]
            if len(star)>0:
                return star
            else:
                raise ValueError(f"""Gaia source id: {self.source_id} not found
                                 in Gaia DR2 catalog within {self.search_radius}
                                 of {self.target_coord}.\n""")

        else:
            #toi release table info
            # radec = self.toi_params['RA','Dec'].values[0]
            Tmag = self.toi_params['TESS Mag'].values[0]
            pmra = self.toi_params['PM RA (mas/yr)'].values[0]
            pmdec = self.toi_params['PM Dec (mas/yr)'].values[0]
            parallax = Distance(self.toi_params['Stellar Distance (pc)'].values[0],
                                unit='pc').parallax.value

            if len(self.gaia_catalog)==1:
                cat = self.gaia_catalog
                gaia_par = cat['parallax'].values[0]
                Gmag = cat['phot_g_mean_mag'].values[0]
                assert round(parallax) != round(gaia_par)
                #if Vmag with Gmag is available, use ticgen to predict Tmag
                assert abs(Tmag-Gmag)<1
            else:
                #nearest neighbor search
