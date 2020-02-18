# -*- coding: utf-8 -*-

# Import standard library
import time

# Import from package
from tql.target import Target

__all__ = ["generate_QL"]

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
        if np.any([use_pld, use_sff]):
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
                        cadence_mask=~cadence_mask_tpf
                        if cadence_mask_tpf.sum() > 0
                        else None,
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
            t14_ncadences = t14 * u.day.to(cadence_in_minutes)
            errmsg = f"use sg_filter_window> {t14_ncadences}"
            assert t14_ncadences < sg_filter_window, errmsg
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
        if np.any([use_pld, use_sff]):
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
            (
                Teff_tic,
                logg_tic,
                _,
                Rs_min_tic,
                Rs_max_tic,
                _,
                _,
                _,
            ) = catalog.catalog_info_TIC(int(ticid))
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
        if np.any([use_pld, use_sff]):
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
