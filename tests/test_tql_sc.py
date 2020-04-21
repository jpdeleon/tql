# -*- coding: utf-8 -*-
from matplotlib.figure import Figure
from tql import plot_tql

toiid = 1063
savefig = True
verbose = True
quality_bitmask = "default"
apply_data_quality_mask = False
cutout_size = (15, 15)
window_length = 0.5
lctype = "custom"


def test_ticid():
    # ticid
    fig = fig = plot_tql(
        gaiaid=None,
        toiid=None,
        ticid=460205581,
        name=None,
        sector=None,
        cadence="short",
        lctype="pdcsap",
        sap_mask="pipeline",
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )
    assert isinstance(fig, Figure)


def test_toiid():
    # square mask aper_radius=1
    fig = fig = plot_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="square",
        aper_radius=1,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )
    assert isinstance(fig, Figure)


def test_square_mask():
    # square mask aper_radius=2
    fig = fig = plot_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="square",
        aper_radius=2,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )
    assert isinstance(fig, Figure)


def test_round_mask():
    # round mask aper_radius=1
    fig = fig = plot_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="round",
        aper_radius=1,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )
    assert isinstance(fig, Figure)


def test_mask_size():
    # round mask aper_radius=2, sector 11
    fig = fig = plot_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=11,
        cadence="short",
        lctype=lctype,
        sap_mask="round",
        aper_radius=2,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )
    assert isinstance(fig, Figure)


def test_cutout_size():
    # smaller cutout_size
    fig = plot_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="round",
        aper_radius=2,
        cutout_size=(10, 10),
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )
    assert isinstance(fig, Figure)


def test_name():
    # name search
    fig = plot_tql(
        gaiaid=None,
        toiid=None,
        ticid=None,
        name="Trappist-1",
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="round",
        aper_radius=2,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )
    assert isinstance(fig, Figure)


def test_percentile_mask():
    # sap mask percentile
    fig = plot_tql(
        gaiaid=None,
        toiid=toiid,
        ticid=None,
        name=None,
        sector=None,
        cadence="short",
        lctype=lctype,
        sap_mask="percentile",
        percentile=90,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )
    assert isinstance(fig, Figure)


def test_threshold_mask():
    # sap mask threshold
    fig = plot_tql(
        toiid=toiid,
        cadence="short",
        lctype=lctype,
        sap_mask="threshold",
        threshold_sigma=5,
        cutout_size=cutout_size,
        quality_bitmask=quality_bitmask,
        apply_data_quality_mask=apply_data_quality_mask,
        window_length=window_length,
        savefig=savefig,
        savetls=False,
        outdir=".",
        verbose=verbose,
    )
    assert isinstance(fig, Figure)
