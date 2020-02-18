#!/usr/bin/env python
from os.path import join
import numpy as np

try:
    import stardate as sd
    import corner
    import h5py
except Exception:
    raise ValueError("pip install stardate isochrones h5py corner")


def run_stardate(
    iso_params: dict,
    prot: tuple = None,
    nsteps: int = 1000,
    return_age: bool = True,
    verbose: bool = True,
):
    """Estimate stellar age using isochrone and asteroseismology

    Parameters
    ----------
    iso_params : input parameter for isochrones
    prot : rotation period [day]
    nsteps : number of MCMC steps

    Returns
    -------
    star object

    # NOTE: prot=None defaults to `isochrones`
    """

    prot, prot_err = prot

    # Set up the star object.
    star = sd.Star(
        iso_params, prot=prot, prot_err=prot_err
    )  # Here's where you add a rotation period

    # Run the MCMC
    star.fit(max_n=nsteps)
    if verbose:
        age, errp, errm, age_samples = parse_samples(star, verbose=verbose)
    return star


def parse_samples(
    star,
    return_age: bool = True,
    return_mass: bool = False,
    return_feh: bool = True,
    return_lndist: bool = True,
    return_Av: bool = True,
    verbose: bool = True,
):
    """parse stellar parameter from star samples

    Parameters
    ----------
    star : stardate.Star object
    return_par : parameter of choice

    Returns
    -------
    val, sighi, siglo : 50th, 84th, 16th percentiles
    (best value & 1-sigma error bars) of parameter & MCMC samples
    """
    if return_age:
        age, errp, errm, age_samples = star.age_results()
        if verbose:
            print(
                "stellar age = {0:.2f} + {1:.2f} + {2:.2f}".format(
                    age, errp, errm
                )
            )
        return age, errp, errm, age_samples

    if return_mass:
        mass, mass_errp, mass_errm, mass_samples = star.mass_results(
            burnin=10
        )  # burnin is thin_by x larger than this
        if verbose:
            print(
                "Mass = {0:.2f} + {1:.2f} - {2:.2f} M_sun".format(
                    mass, mass_errp, mass_errm
                )
            )
        return mass, mass_errp, mass_errm, mass_samples

    if return_feh:
        feh, feh_errp, feh_errm, feh_samples = star.feh_results(burnin=10)
        if verbose:
            print(
                "feh = {0:.2f} + {1:.2f} - {2:.2f}".format(
                    feh, feh_errp, feh_errm
                )
            )
        return feh, feh_errp, feh_errm, feh_samples

    if return_lndist:
        (
            lndistance,
            lndistance_errp,
            lndistance_errm,
            lndistance_samples,
        ) = star.distance_results(burnin=10)
        if verbose:
            print(
                "ln(distance) = {0:.2f} + {1:.2f} - {2:.2f} ".format(
                    lndistance, lndistance_errp, lndistance_errm
                )
            )
        return lndistance, lndistance_errp, lndistance_errm, lndistance_samples

    if return_Av:
        Av, Av_errp, Av_errm, Av_samples = star.Av_results(burnin=10)
        if verbose:
            print(
                "Av = {0:.2f} + {1:.2f} - {2:.2f}".format(Av, Av_errp, Av_errm)
            )
        return Av, Av_errp, Av_errm, Av_samples


def load_stardate_samples(loc: str = "./", filename: str = "samples"):
    # Load the samples.
    flatsamples, _3Dsamples, posterior_samples, prior_samples = sd.load_samples(
        join(loc, "{filename}.h5"), burnin=1
    )

    # Extract the median and maximum likelihood parameter estimates from the samples.
    return sd.read_samples(flatsamples)


def plot_corner(flatsamples):
    labels = [
        "EEP",
        "log10(Age [yr])",
        "[Fe/H]",
        "ln(Distance)",
        "Av",
        "ln(probability)",
    ]
    corner.corner(flatsamples, labels=labels)


def get_gyro_age(bprp: float, prot: float, verbose: bool = True):
    """predict stellar age from rotation period using simple gyrochronology
    model without running MCMC, only applicable to FGK and early M dwarfs
    on the main sequence, older than a few hundred Myrs

    Parameters
    ----------
    bprp : Gaia BP - RP color [mag]
    prot : rotation period [day]

    Returns
    -------
    age [year]
    """

    log10_period = np.log10(prot)
    log10_age_yrs = sd.lhf.age_model(log10_period, bprp)

    if verbose:
        age = (10 ** log10_age_yrs) * 1e-6
        print(f"{age} Myr\n")
    return age


def get_prot_from_gyro_age(bprp: float, age: float, verbose: bool = True):
    """predict rotation period from age using gyrochronology model of
    Praesepe cluster stars, only applicable to FGK and early M dwarfs
    on the main sequence, older than a few hundred Myrs

    Parameters
    ----------
    bprp : Gaia BP - RP color [mag]
    age : stellar age [year]

    Returns
    -------
    age [year]
    """

    log10_age_yrs = np.log10(age)
    log10_period = sd.lhf.gyro_model_praesepe(log10_age_yrs, bprp)
    if verbose:
        age = 10 ** log10_period
        print(f"{prot} days")
    return age


if __name__ == "__main__":
    if True:
        # Create a dictionary of observables
        iso_params = {
            "teff": (4386, 50),  # Teff with uncertainty.
            "logg": (4.66, 0.05),  # logg with uncertainty.
            "feh": (0.0, 0.02),  # Metallicity with uncertainty.
            "parallax": (1.48, 0.1),  # Parallax in milliarcseconds.
            "maxAV": 0.1,
        }  # Maximum extinction

        prot, prot_err = 29, 3

        # Set up the star object.
        star = sd.Star(
            iso_params, prot=prot, prot_err=prot_err
        )  # Here's where you add a rotation period

        # Run the MCMC
        star.fit(max_n=1000)

        # max_n is the maximum number of MCMC samples. I recommend setting this
        # much higher when running for real, or using the default value of 100000.

        # Print the median age with the 16th and 84th percentile uncertainties.
        age, errp, errm, samples = star.age_results()
        print(
            "stellar age = {0:.2f} + {1:.2f} + {2:.2f}".format(age, errp, errm)
        )

    if False:
        bprp = 0.82  # Gaia BP - RP color.
        log10_period = np.log10(26)
        log10_age_yrs = sd.lhf.age_model(log10_period, bprp)
        print((10 ** log10_age_yrs) * 1e-6, "Myr")

    if False:
        bprp = 0.82  # Gaia BP - RP color.
        log10_age_yrs = np.log10(4.56 * 1e9)
        log10_period = sd.lhf.gyro_model_praesepe(log10_age_yrs, bprp)
        print(10 ** log10_period, "days")
