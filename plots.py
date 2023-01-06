import copy
import json
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import RVtools.utils as utils
from astropy.time import Time
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.get_module import get_module_from_specs
from RVtools.builder import BaseBuilder, Director
from RVtools.cosmos import Planet
from RVtools.cosmoses.exosims import ExosimsUniverse
from tqdm import tqdm

trail_kwargs = {
    "linestyle": "--",
    "color": "w",
    "alpha": 0.1,
    "zorder": 0,
}
planet_scatter_kwargs_2d = {
    "s": 25,
    "color": "w",
    "edgecolor": "k",
}
planet_scatter_kwargs_3d = {
    "s": 25,
    "color": "w",
    "edgecolor": "k",
}


def create_builder():
    settings_file = Path("/home/corey/Documents/github/AAS_winter_2023/.config.json")
    with open(settings_file, "r") as f:
        settings = json.load(f)
    cache_dir = settings["cache_dir"]
    workers = settings["workers"]

    # Set up director and builder objects
    director = Director()
    builder = BaseBuilder(cache_dir=cache_dir, workers=workers)
    director.builder = builder
    builder.universe_params = {
        "universe_type": "exosims",
        "script": "/home/corey/Documents/github/AAS_winter_2023/caseB.json",
    }
    mission_start = Time(2043, format="decimalyear")
    rv100_25 = {
        "name": "1 m/s",
        "precision": 1 * u.m / u.s,
        "start_time": mission_start - 20 * u.yr,
    }

    rv03_15 = {
        "name": "3 cm/s",
        "precision": 0.03 * u.m / u.s,
        "start_time": mission_start - 10 * u.yr,
    }
    survey = {
        "fit_order": 2,
        "instruments": [rv100_25, rv03_15],
    }
    surveys = [survey]
    base_params = {
        "observation_scheme": "survey",
        "observations_per_night": 2,
        "bad_weather_prob": 0.8,
        "end_time": mission_start,
    }
    nsystems = 10
    builder.preobs_params = {
        "base_params": base_params,
        "surveys": surveys,
        "n_systems_to_observe": nsystems,
        "filters": ["distance"],
    }
    builder.orbitfit_params = {
        "fitting_method": "rvsearch",
        "max_planets": 3,
    }
    construction_method = {"name": "multivariate gaussian", "cov_samples": 1000}
    # construction_method = {"name": "credible interval"}
    builder.pdet_params = {
        "construction_method": construction_method,
        "script": "/home/corey/Documents/github/AAS_winter_2023/caseB.json",
        "number_of_orbits": 10000,
        "start_time": mission_start,
        "end_time": mission_start + 5 * u.yr,
    }
    director.build_orbit_fits()
    builder.probability_of_detection()
    return builder


def pop2planet(pop, system):
    planet_dict = {
        "t0": pop.t0[0],
        "a": pop.a[0],
        "e": pop.e[0],
        "mass": pop.Mp[0],
        "radius": pop.Rp[0],
        "inc": pop.inc[0],
        "W": pop.W[0],
        "w": pop.w[0],
        "M0": pop.M0[0],
        "p": pop.p,
    }
    planet = create_planet(system, planet_dict)
    return planet


def earthLike_setup(system, t0, tf, dt):
    earthLike = {
        "t0": t0,
        "a": 1.0 * u.AU,
        "e": 0,
        "mass": 1 * u.M_earth,
        "radius": 1 * u.R_earth,
        "inc": 90 * u.deg,
        "W": 0 * u.deg,
        "w": 0 * u.deg,
        "M0": 0 * u.rad,
        "p": 0.2,
    }
    plot_times = Time(np.arange(0, tf, dt) + t0, format="mjd")
    setup_system(system, [earthLike])
    generate_system_rv(system, plot_times)
    return system, plot_times


def pop_3d(ax, pop, time_jd, color="r"):
    vectors = utils.calc_position_vectors(pop, Time([time_jd], format="jd"))
    x = (np.arctan((vectors["x"][0] * u.m) / pop.dist_to_star)).to(u.arcsec).value
    y = (np.arctan((vectors["y"][0] * u.m) / pop.dist_to_star)).to(u.arcsec).value
    z = (np.arctan((vectors["z"][0] * u.m) / pop.dist_to_star)).to(u.arcsec).value
    ax.scatter(x, y, z, alpha=0.75, s=0.01, color=color)
    return ax


def init_skyplot(fig, loc):
    ax = fig.add_subplot(loc, projection="aitoff")
    plt.grid(True)
    return ax


def sky_plot(ax, coord):
    gal = coord.galactic
    ax.scatter(gal.l.wrap_at("180d").radian, gal.b.radian)
    return ax


def generate_system_pdet(system, times, IWA, OWA, dMag0):
    pdets = []
    for time in times:
        n_visible = 0
        for planet in system.planets:
            # pos = planet.pos.loc[planet.pos.t == time.jd].reset_index()
            planet.dist = system.star.dist
            WA, dMag = utils.prop_for_imaging(planet, time)
            planet.WAs.append(WA)
            planet.dMags.append(dMag)
            n_visible += (IWA < WA) & (OWA > WA) & (dMag0 > dMag)
        pdets.append((n_visible / len(system.planets))[0])
    system.pdets = pdets


def generate_system_rv(system, times):
    system.propagate(times)
    system.rv_df.rv = -system.rv_df.rv


def earth_3d_plane(ax, azim=45):
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_zlabel("z (AU) ")
    ax.view_init(20, azim, 0)
    return ax


def earth_img_plane(ax):
    ax.set_title("Image plane")
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    return ax


def full_3d_plane(ax, azim=45):
    ax.set_xlim([-9, 9])
    ax.set_ylim([-9, 9])
    ax.set_zlim([-9, 9])
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_zlabel("z (AU) ")
    ax.view_init(20, azim, 0)
    return ax


def full_img_plane(ax):
    ax.set_title("Image plane")
    ax.set_xlim([-9, 9])
    ax.set_ylim([-9, 9])
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    return ax


def planet_marker_size(z, all_z, base_size=5, factor=0.5):
    """
    Make the planet marker smaller when the planet is behind the star in its orbit
    """
    z_range = np.abs(max(all_z) - min(all_z))

    # Want being at max z to correspond to a factor of 1 and min z
    # to be a factor of negative 1
    scaled_z = 2 * (z - min(all_z)) / z_range - 1

    marker_size = base_size * (1 + factor * scaled_z)

    return marker_size


def get_planet_alpha(planet, i, dMag0, dMag_ub):
    dMag = planet.dMags[i]
    if dMag > dMag0:
        alpha = 0
    elif dMag < dMag_ub:
        alpha = 1
    else:
        alpha = ((-dMag + dMag0) / (-dMag_ub + dMag0))[0]
    return alpha


def get_planet_zorder(planet, time):
    # pos = utils.calc_position_vectors(planet, Time([time.jd], format="jd"))
    pos = planet.pos.loc[planet.pos.t == time.jd].reset_index()
    if pos.z[0] > 0:
        zorder = 2
    else:
        zorder = 0
    if zorder > planet.trail_kwargs["zorder"]:
        planet.trail_kwargs["zorder"] = zorder
    planet.scatter_kwargs_2d["zorder"] = zorder
    return planet


def scatter_star(ax, planet_pos, system, projection, ind, unit=u.AU):
    star_cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(
        vmin=min(system.rv_df.rv.values), vmax=max(system.rv_df.rv.values)
    )
    if unit == u.arcsec:
        x = (np.arctan((planet_pos[0] * u.m) / system.star.dist)).to(u.arcsec).value
        y = (np.arctan((planet_pos[1] * u.m) / system.star.dist)).to(u.arcsec).value
        z = (np.arctan((planet_pos[2] * u.m) / system.star.dist)).to(u.arcsec).value
    else:
        x = (planet_pos[0] * u.m).to(unit).value
        y = (planet_pos[1] * u.m).to(unit).value
        z = (planet_pos[2] * u.m).to(unit).value
    if projection == "2d":
        ax.scatter(
            -0.05 * x,
            -0.05 * y,
            c=system.rv_df.rv[ind],
            cmap=star_cmap,
            norm=norm,
            **system.star.scatter_kwargs,
        )
        if hasattr(system.star, "trail_kwargs") and ind != 0:
            pos = system.planets[0].pos
            ax = star_trail(ax, system.star, pos, ind, unit, projection)
    elif projection == "3d":
        ax.scatter(
            -0.05 * x,
            -0.05 * y,
            -0.05 * z,
            c=system.rv_df.rv[ind],
            cmap=star_cmap,
            norm=norm,
            **system.star.scatter_kwargs,
        )
        if hasattr(system.star, "trail_kwargs") and ind != 0:
            pos = system.planets[0].pos
            ax = star_trail(ax, system.star, pos, ind, unit, projection)
    return ax


def star_trail(ax, star, pos, ind, unit, projection):
    x = [-0.05 * x[0] for i, x in enumerate(pos.x) if i < ind]
    y = [-0.05 * y[0] for i, y in enumerate(pos.y) if i < ind]
    z = [-0.05 * z[0] for i, z in enumerate(pos.z) if i < ind]
    if unit == u.arcsec:
        x = (np.arctan((x * u.m) / system.star.dist)).to(u.arcsec).value
        y = (np.arctan((y * u.m) / system.star.dist)).to(u.arcsec).value
        z = (np.arctan((z * u.m) / system.star.dist)).to(u.arcsec).value
    else:
        x = (x * u.m).to(unit).value
        y = (y * u.m).to(unit).value
        z = (z * u.m).to(unit).value
    if projection == "2d":
        ax.plot(x, y, **star.trail_kwargs)
    elif projection == "3d":
        ax.plot(x, y, z, **star.trail_kwargs)
    return ax


def planet_trail(ax, planet, ind, unit, projection):
    # pos = utils.calc_position_vectors(planet, times)
    pos = planet.pos.iloc[:ind]
    x = [x[0] for x in pos.x]
    y = [y[0] for y in pos.y]
    z = [z[0] for z in pos.z]
    if unit == u.arcsec:
        x = (np.arctan((x * u.m) / system.star.dist)).to(u.arcsec).value
        y = (np.arctan((y * u.m) / system.star.dist)).to(u.arcsec).value
        z = (np.arctan((z * u.m) / system.star.dist)).to(u.arcsec).value
    else:
        x = (x * u.m).to(unit).value
        y = (y * u.m).to(unit).value
        z = (z * u.m).to(unit).value
    if projection == "2d":
        ax.plot(x, y, **planet.trail_kwargs)
    elif projection == "3d":
        ax.plot(x, y, z, **planet.trail_kwargs)
    return ax


def scatter_planet(ax, planet, ind, unit=u.arcsec, projection="2d"):

    # Format position data
    # pos = utils.calc_position_vectors(planet, Time([times[ind].jd], format="jd"))
    pos = planet.pos.iloc[ind]
    if unit == u.arcsec:
        x = (np.arctan((pos.x[0] * u.m) / system.star.dist)).to(u.arcsec).value
        y = (np.arctan((pos.y[0] * u.m) / system.star.dist)).to(u.arcsec).value
        z = (np.arctan((pos.z[0] * u.m) / system.star.dist)).to(u.arcsec).value
    else:
        x = (pos.x[0] * u.m).to(unit).value
        y = (pos.y[0] * u.m).to(unit).value
        z = (pos.z[0] * u.m).to(unit).value

    # 2d Projection
    if projection == "2d":
        ax.scatter(x, y, **planet.scatter_kwargs_2d)
        if hasattr(planet, "trail_kwargs") and ind != 0:
            # ax = planet_trail(ax, planet, times[:ind], unit, projection)
            ax = planet_trail(ax, planet, ind, unit, projection)

    # 3d Projection
    elif projection == "3d":
        ax.scatter(x, y, z, **planet.scatter_kwargs_3d)
        if hasattr(planet, "trail_kwargs") and ind != 0:
            ax = planet_trail(ax, planet, ind, unit, projection)
    return ax


def scatter_system(ax, system, times, ind, unit=u.arcsec, projection="2d"):
    for planet in system.planets:
        ax = scatter_planet(ax, planet, times, ind, unit, projection)
    return ax


def scatter_pop(ax, pop, ind, unit=u.arcsec, projection="2d"):
    pos = pop.pos.iloc[ind]
    if unit == u.arcsec:
        x = (np.arctan((pos.x * u.m) / system.star.dist)).to(u.arcsec).value
        y = (np.arctan((pos.y * u.m) / system.star.dist)).to(u.arcsec).value
        z = (np.arctan((pos.z * u.m) / system.star.dist)).to(u.arcsec).value
    else:
        x = (pos.x * u.m).to(unit).value
        y = (pos.y * u.m).to(unit).value
        z = (pos.z * u.m).to(unit).value
    if projection == "2d":
        ax.scatter(x, y, **pop.scatter_kwargs_2d)
    elif projection == "3d":
        # 3d Projection
        ax.scatter(x, y, z, **pop.scatter_kwargs_3d)
    return ax


def add_sep_plot(ax, planet, system, i, IWA, dMag0, unit, add_lines=True):
    if unit == u.arcsec:
        IWA_dist = IWA.to(u.arcsec).value
        WA_dist = planet.WAs[i].to(u.arcsec).value
    else:
        IWA_dist = (np.tan(IWA) * system.star.dist).to(unit).value
        WA_dist = (np.tan(planet.WAs[i]) * system.star.dist).to(u.AU).value
    tmp_scatter = planet.scatter_kwargs_2d.copy()
    tmp_scatter["alpha"] = 1
    ax.scatter(WA_dist, planet.dMags[i], **tmp_scatter)
    if add_lines:
        IWA_line = mpl.lines.Line2D([IWA_dist, IWA_dist], [15, dMag0], color="red")
        dMag_line = mpl.lines.Line2D([IWA_dist, 5], [dMag0, dMag0], color="red")
        ax.add_line(IWA_line)
        ax.add_line(dMag_line)
    return ax


def add_IWA(ax, IWA, system, unit, annotate=False):
    if unit == u.arcsec:
        IWA_radius = IWA.to(u.arcsec).value
    else:
        IWA_radius = (np.tan(IWA) * system.star.dist).to(unit).value
    IWA_patch = mpatches.Circle(
        (0, 0),
        IWA_radius,
        facecolor="grey",
        edgecolor="black",
        alpha=0.5,
        zorder=5,
    )
    ax.add_patch(IWA_patch)
    if annotate:
        ax.annotate(
            "IWA",
            xy=(0, 0),
            xytext=(0, IWA_radius * 1.125),
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="<-"),
            zorder=10,
        )
    return ax


def darken_3d_background(ax):
    ax.w_xaxis.set_pane_color((0.05, 0.05, 0.05, 1.0))
    ax.w_yaxis.set_pane_color((0.05, 0.05, 0.05, 1.0))
    ax.w_zaxis.set_pane_color((0.05, 0.05, 0.05, 1.0))
    return ax


def time_x_label(ax, times):
    ax.set_xlim(
        [
            times[0].decimalyear - times[0].decimalyear,
            times[-1].decimalyear - times[0].decimalyear,
        ]
    )
    return ax


def base_plot(ax, times, val, final_ind=-1):
    cumulative_times = times.value - times[0].value
    ax.set_xlim(
        [
            times[0].value - times[0].value,
            times[-1].value - times[0].value,
        ]
    )
    ax.set_xlabel("Time (yr)")
    ax.plot(
        cumulative_times[:final_ind],
        val[:final_ind],
        # color=ecolor,
    )
    return ax


def pdet_plot(ax, times, pdet, final_ind=-1, hatching_info=None, add_hatching=False):
    ax = base_plot(ax, times, pdet, final_ind=final_ind)
    ax.set_ylim(-0.05, 1.05)
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Probability of detection")
    ax.set_xlabel("Time (yr)")

    # Add hatching for true detectability
    if add_hatching:
        planet, IWA, OWA = hatching_info
        ax, det_sq, not_det_sq = add_true_planet_hatching(
            ax, times, planet, IWA, OWA, -1, return_patches=True
        )
        return ax, det_sq, not_det_sq

    else:
        return ax


def add_true_planet_hatching(
    ax, times, planet, IWA, OWA, dMag0, final_ind, return_patches=False
):
    cumulative_times = times.value - times[0].value
    true_pdet = []
    for i, time in enumerate(times[:final_ind]):
        time_jd = Time(Time(time, format="decimalyear").jd, format="jd")
        planet_alpha, planet_dMag = planet.prop_for_imaging(time_jd)
        true_pdet.append(
            (planet_dMag < dMag0)
            & (OWA.to(u.arcsec).value > planet_alpha.to(u.arcsec).value)
            & (planet_alpha.to(u.arcsec).value > IWA.to(u.arcsec).value)
        )
    change_inds = np.where(np.roll(true_pdet, 1) != true_pdet)[0]
    inds_to_plot = np.concatenate(([0], change_inds, [len(true_pdet) - 1]))
    ymin, ymax = ax.get_ylim()
    box_height = ymax - ymin
    plt.rcParams["hatch.linewidth"] = 0.5
    for i, ind in enumerate(inds_to_plot):
        current_pdet = true_pdet[ind]
        if i + 1 == len(inds_to_plot):
            continue
        if current_pdet == 1:
            start_time = cumulative_times[ind]
            width = cumulative_times[inds_to_plot[i + 1]] - cumulative_times[ind]
            det_sq1 = mpl.patches.Rectangle(
                (start_time, ymin),
                width=width,
                height=box_height,
                zorder=2,
                edgecolor="black",
                fill=False,
            )
            ax.add_patch(det_sq1)
            det_sq = mpl.patches.Rectangle(
                (start_time, ymin),
                width=width,
                height=box_height,
                hatch=r"\\",
                zorder=1,
                alpha=0.5,
                edgecolor="black",
                label="Ground truth planet detectable",
                fill=False,
            )
            ax.add_patch(det_sq)
        else:
            start_time = cumulative_times[ind]
            width = cumulative_times[inds_to_plot[i + 1]] - cumulative_times[ind]
            not_det_sq1 = mpl.patches.Rectangle(
                (start_time, ymin),
                width=width,
                height=box_height,
                zorder=2,
                edgecolor="black",
                fill=False,
            )
            ax.add_patch(not_det_sq1)
            not_det_sq = mpl.patches.Rectangle(
                (start_time, ymin),
                width=width,
                height=box_height,
                hatch="||",
                zorder=1,
                alpha=0.5,
                edgecolor="black",
                label="Ground truth planet not detectable",
                fill=False,
            )
            ax.add_patch(not_det_sq)
    if return_patches:
        return ax, det_sq, not_det_sq
    else:
        return ax


def create_planet(system, planet_dict):
    planet = Planet(planet_dict)
    planet.star = system.star
    planet.solve_dependent_params()
    return planet


def setup_system(system, planet_dicts):
    system.planets = []
    for planet_dict in planet_dicts:
        planet = create_planet(system, planet_dict)
        system.planets.append(planet)


def find_msini_aliases(system, planet, n, Irange=[2, 178] * u.deg):
    C = 0.5 * (np.cos(Irange[0]) - np.cos(Irange[1]))
    incs = np.arccos(np.cos(Irange[0]) - 2.0 * C * np.random.uniform(size=n))
    msini_aliases = []
    for inc in incs:
        msini_aliases.append(create_inc_alias(system, planet, inc))
    return msini_aliases


def create_inc_alias(system, planet, inc):
    K = planet.K
    mass = (
        K
        * (system.star.mass) ** (2 / 3)
        * np.sqrt(1 - planet.e**2)
        / np.sin(inc.to(u.rad))
        * (planet.T / (2 * np.pi * const.G)) ** (1 / 3)
    ).decompose()
    base_params = planet.dump_params()
    new_params = copy.deepcopy(base_params)
    new_params["inc"] = inc
    new_params["mass"] = mass
    new_planet = create_planet(system, new_params)
    return new_planet


def create_mass_alias(system, planet, mass):
    K = planet.K
    inc = np.arcsin(
        K
        * (system.star.mass) ** (2 / 3)
        * np.sqrt(1 - planet.e**2)
        / mass
        * (planet.T / (2 * np.pi * const.G)) ** (1 / 3)
    ).decompose()
    base_params = planet.dump_params()
    new_params = copy.deepcopy(base_params)
    new_params["inc"] = inc
    new_params["mass"] = mass
    new_planet = create_planet(system, new_params)
    return new_planet


def fig_1(system, specs):
    t0 = system.planets[0].t0
    IWA = specs["observingModes"][0]["IWA"]
    plot_times = Time(np.arange(0, 2 * 365, 1) + t0, format="mjd")
    azim_range = np.linspace(15, 75, len(plot_times))
    earthLike = {
        "t0": t0,
        "a": 1.0 * u.AU,
        "e": 0,
        "mass": 1 * u.M_earth,
        "radius": 1 * u.R_earth,
        "inc": 90 * u.deg,
        "W": 0 * u.deg,
        "w": 0 * u.deg,
        "M0": 0 * u.rad,
        "p": 0.2,
    }
    setup_system(system, [earthLike])

    earthLike_system = copy.deepcopy(system)

    massive_planet = create_mass_alias(system, system.planets[0], 1 * u.M_jupiter)
    massive_system = copy.deepcopy(system)
    massive_system.planets = [massive_planet]

    # Generate RV data
    generate_system_rv(earthLike_system, plot_times)
    generate_system_rv(massive_system, plot_times)

    # Get z-vals
    earth = earthLike_system.planets[0]
    earth.pos = utils.calc_position_vectors(earth, plot_times)
    earthLike_system.planets[0].all_z = [val[0] for val in earth.pos.z.values]

    jup = massive_system.planets[0]
    jup.pos = utils.calc_position_vectors(jup, plot_times)
    jup.all_z = [val[0] for val in jup.pos.z.values]

    # Set up plotting stuff
    earthLike_system.star.scatter_kwargs = {"s": 50, "zorder": 1}
    massive_system.star.scatter_kwargs = {"s": 50, "zorder": 1}
    jup_scatter_kwargs = {
        "s": 50,
        "color": "w",
        "edgecolor": "k",
        # "label": "Jupiter mass",
    }
    earth_scatter_kwargs = {
        "s": 25,
        "color": "w",
        "edgecolor": "k",
        # "label": "Earth mass",
    }
    massive_system.planets[0].scatter_kwargs_2d = jup_scatter_kwargs
    massive_system.planets[0].scatter_kwargs_3d = jup_scatter_kwargs
    earthLike_system.planets[0].scatter_kwargs_2d = earth_scatter_kwargs
    earthLike_system.planets[0].scatter_kwargs_3d = earth_scatter_kwargs

    trail_kwargs = {
        "linestyle": "--",
        "color": "w",
        "alpha": 0.5,
        "zorder": 0,
    }
    star_trail_kwargs = {
        "linestyle": "--",
        "color": "w",
        "alpha": 0.5,
        "zorder": 0,
    }
    earthLike_system.star.trail_kwargs = star_trail_kwargs
    massive_system.star.trail_kwargs = star_trail_kwargs
    massive_system.planets[0].trail_kwargs = trail_kwargs
    earthLike_system.planets[0].trail_kwargs = trail_kwargs
    earth.base_s = 25
    earth.title = "Earth mass"
    jup.base_s = 50
    jup.title = "Jupiter mass"

    # alias_planets = find_msini_aliases(system, system.planets[0], 5)

    # Plot work
    fig1 = plt.figure(figsize=[16, 9])
    subfigs = fig1.subfigures(nrows=1, ncols=3, width_ratios=[1.5, 1, 1])
    earthLike_row = [
        subfigs[0].add_subplot(211),
        subfigs[1].add_subplot(211, projection="3d"),
        subfigs[2].add_subplot(211),
    ]
    massive_row = [
        subfigs[0].add_subplot(212),
        subfigs[1].add_subplot(212, projection="3d"),
        subfigs[2].add_subplot(212),
    ]
    systems = [earthLike_system, massive_system]
    star_cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(
        vmin=min(earthLike_system.rv_df.rv.values),
        vmax=max(earthLike_system.rv_df.rv.values),
    )
    for i, time in enumerate(tqdm(plot_times, desc="Figure 1")):
        subfigs[1].suptitle("Orbits in 3d")
        subfigs[2].suptitle("Image plane")
        subfigs[0].suptitle("Radial velocity")
        for im_row, system in zip([earthLike_row, massive_row], systems):
            planet = system.planets[0]
            pos = planet.pos.iloc[i]
            xyz = [pos.x[0], pos.y[0], pos.z[0]]
            planet = get_planet_zorder(planet, time)
            ax_3d = im_row[1]
            ax_3d = darken_3d_background(ax_3d)
            ax_3d = scatter_star(ax_3d, xyz, system, "3d", i)
            ax_3d = scatter_planet(ax_3d, planet, i, unit=u.AU, projection="3d")
            ax_3d = earth_3d_plane(ax_3d, azim_range[i])
            ax_3d.set_title(planet.title)

            # Image plane
            z_val = planet.all_z[i]
            planet.scatter_kwargs_2d["s"] = planet_marker_size(
                z_val, planet.all_z, base_size=planet.base_s
            )
            ax_img = im_row[2]
            ax_img = scatter_star(ax_img, xyz, system, "2d", i)
            ax_img = scatter_planet(ax_img, planet, i, unit=u.AU, projection="2d")
            # ax_img = add_star(ax_img, xyz, system, "2d", i)
            ax_img.set_xlim([-1.5, 1.5])
            ax_img.set_ylim([-1.5, 1.5])
            if not ax_img.get_subplotspec().is_first_row():
                ax_img.set_xlabel("x (AU)")
            ax_img.set_ylabel("y (AU)")
            # ax_img = add_IWA(ax_img, IWA, system, u.AU)
            ax_img.set_title(planet.title)

            # RV data
            ax_rv = im_row[0]
            ax_rv.scatter(
                plot_times.decimalyear[:i] - plot_times[0].decimalyear,
                system.rv_df.rv.values[:i],
                c=system.rv_df.rv.values[:i],
                cmap=star_cmap,
                norm=norm,
            )
            ax_rv = time_x_label(ax_rv, plot_times)
            if not ax_img.get_subplotspec().is_first_row():
                ax_rv.set_xlabel("Time (yr)")
            ax_rv.set_ylim([-0.1, 0.1])
            ax_rv.set_ylabel("RV (m/s)")
            ax_rv.set_title(planet.title)

        fig1.savefig(f"figures/fig1/fig1_{i:003d}.png", bbox_inches="tight")
        for row in [earthLike_row, massive_row]:
            for ax in row:
                ax.clear()


def fig_2(system, specs):
    t0 = system.planets[0].t0
    IWA = specs["observingModes"][0]["IWA"]
    system.star.mass = 1 * u.M_sun
    system.star.dist = 20 * u.pc
    t0 = system.planets[0].t0
    tf = 366
    dt = 1
    system, plot_times = earthLike_setup(system, t0, tf, dt)
    azim_range = np.linspace(15, 75, len(plot_times))
    # alias_planets = find_msini_aliases(system, system.planets[0], 25)
    alias_planets = []
    for inc in np.linspace(2, 178, 25) * u.deg:
        planet = create_inc_alias(system, system.planets[0], inc)
        planet.WAs = []
        planet.dMags = []
        alias_planets.append(planet)
    # system.planets += alias_planets
    system.planets = alias_planets

    # Get z-vals
    # Set up plotting stuff
    system.star.scatter_kwargs = {"s": 50, "zorder": 1}
    # Add planet params
    for planet in system.planets:
        planet.pos = utils.calc_position_vectors(planet, plot_times)
        planet.all_z = [val[0] for val in planet.pos.z.values]
        planet.scatter_kwargs_2d = copy.deepcopy(planet_scatter_kwargs_2d)
        planet.scatter_kwargs_3d = copy.deepcopy(planet_scatter_kwargs_3d)
        planet.trail_kwargs = trail_kwargs
        planet.base_s = (
            (
                10
                + 30
                / (30 * u.M_earth - 1 * u.M_earth).decompose()
                * planet.mass.decompose()
            )
            .decompose()
            .value
        )

    # Plot work
    star_cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(
        vmin=min(system.rv_df.rv.values),
        vmax=max(system.rv_df.rv.values),
    )
    for i, time in enumerate(tqdm(plot_times, desc="Figure 2")):
        fig = plt.figure(figsize=[16, 5])
        subfigs = fig.subfigures(nrows=1, ncols=3, width_ratios=[1.5, 1, 1])
        axes = [
            subfigs[0].add_subplot(),
            subfigs[1].add_subplot(projection="3d"),
            subfigs[2].add_subplot(),
        ]
        ax_rv = axes[0]
        ax_rv.set_title("Radial velocity")
        ax_rv = time_x_label(ax_rv, plot_times)
        ax_rv.set_xlabel("Time (yr)")
        ax_rv.set_ylim([-0.1, 0.1])
        ax_rv.set_ylabel("RV (m/s)")

        ax_3d = axes[1]
        ax_3d.set_title("Orbits in 3d")
        ax_3d = earth_3d_plane(ax_3d, azim_range[i])

        ax_img = axes[2]
        ax_img.set_title("Image plane")
        ax_img = earth_img_plane(ax_img)
        # ax_img = add_IWA(ax_img, IWA, system, u.AU)

        for planet in system.planets:
            # RV data
            ax_rv.scatter(
                plot_times.decimalyear[:i] - plot_times[0].decimalyear,
                system.rv_df.rv.values[:i],
                c=system.rv_df.rv.values[:i],
                cmap=star_cmap,
                norm=norm,
            )

            # 3d
            z_val = planet.all_z[i]
            planet = get_planet_zorder(planet, time)
            ax_3d = darken_3d_background(ax_3d)
            planet.scatter_kwargs_3d["s"] = planet_marker_size(
                z_val, planet.all_z, base_size=planet.base_s, factor=0.25
            )
            ax_3d = scatter_planet(ax_3d, planet, i, unit=u.AU, projection="3d")

            # Image plane
            planet.scatter_kwargs_2d["s"] = planet_marker_size(
                z_val, planet.all_z, base_size=planet.base_s
            )
            ax_img = scatter_planet(ax_img, planet, i, unit=u.AU, projection="2d")

        ax_img.set_aspect("equal", "box")
        ax_3d = scatter_star(ax_3d, [0, 0, 0], system, "3d", i)
        ax_img = scatter_star(ax_img, [0, 0, 0], system, "2d", i)
        fig.savefig(f"figures/fig2/fig2_{i:003d}.png", bbox_inches="tight")
        plt.close()


def fig_3(system, specs):
    t0 = system.planets[0].t0
    IWA = specs["observingModes"][0]["IWA"]
    system.star.mass = 1 * u.M_sun
    system.star.dist = 20 * u.pc
    t0 = system.planets[0].t0
    tf = 366
    dt = 1
    system, plot_times = earthLike_setup(system, t0, tf, dt)
    azim_range = np.linspace(15, 75, len(plot_times))
    alias_planets = []
    for inc in np.linspace(2, 178, 25) * u.deg:
        planet = create_inc_alias(system, system.planets[0], inc)
        planet.WAs = []
        planet.dMags = []
        alias_planets.append(planet)
    system.planets = alias_planets
    generate_system_pdet(system, plot_times, IWA, np.inf, 100)

    # Get z-vals
    # Set up plotting stuff
    system.star.scatter_kwargs = {"s": 50, "zorder": 1}
    # Add planet params
    for planet in system.planets:
        planet.pos = utils.calc_position_vectors(planet, plot_times)
        planet.all_z = [val[0] for val in planet.pos.z.values]
        planet.scatter_kwargs_2d = copy.deepcopy(planet_scatter_kwargs_2d)
        planet.scatter_kwargs_3d = copy.deepcopy(planet_scatter_kwargs_3d)
        planet.trail_kwargs = trail_kwargs
        planet.base_s = (
            (
                10
                + 30
                / (30 * u.M_earth - 1 * u.M_earth).decompose()
                * planet.mass.decompose()
            )
            .decompose()
            .value
        )

    # Plot work
    for i, time in enumerate(tqdm(plot_times, desc="Figure 3")):
        fig = plt.figure(figsize=[16, 5])
        subfigs = fig.subfigures(nrows=1, ncols=3, width_ratios=[1.2, 1, 1])
        axes = [
            subfigs[0].add_subplot(projection="3d"),
            subfigs[1].add_subplot(),
            subfigs[2].add_subplot(),
        ]
        ax_3d = axes[0]
        ax_3d.set_title("Orbits in 3d")
        ax_3d = earth_3d_plane(ax_3d, azim_range[i])

        ax_img = axes[1]
        ax_img = earth_img_plane(ax_img)
        ax_img = add_IWA(ax_img, IWA, system, u.AU)

        ax_pdet = axes[2]
        ax_pdet.set_title("Percent not obscured")
        ax_pdet = time_x_label(ax_pdet, plot_times)
        ax_pdet.set_xlabel("Time (yr)")
        ax_pdet.set_ylim([100 * -0.05, 100 * 1.05])
        ax_pdet.set_yticks(np.arange(0, 101, 20))
        # ax_rv.set_ylabel("RV (m/s)")
        for planet in system.planets:
            z_val = planet.all_z[i]
            planet = get_planet_zorder(planet, time)
            ax_3d = darken_3d_background(ax_3d)
            planet.scatter_kwargs_3d["s"] = planet_marker_size(
                z_val, planet.all_z, base_size=planet.base_s, factor=0.25
            )
            ax_3d = scatter_planet(ax_3d, planet, i, unit=u.AU, projection="3d")

            # Image plane
            planet.scatter_kwargs_2d["s"] = planet_marker_size(
                z_val, planet.all_z, base_size=planet.base_s
            )
            ax_img = scatter_planet(ax_img, planet, i, unit=u.AU, projection="2d")

            # pdet data
            ax_pdet.plot(
                plot_times.decimalyear[:i] - plot_times[0].decimalyear,
                [100 * pdet for pdet in system.pdets[:i]],
                color="w",
            )

        ax_img.set_aspect("equal", "box")
        ax_3d = scatter_star(ax_3d, [0, 0, 0], system, "3d", i)
        ax_img = scatter_star(ax_img, [0, 0, 0], system, "2d", i)
        fig.savefig(f"figures/fig3/fig3_{i:003d}.png")
        plt.close()


def fig_4(system, specs):
    IWA = specs["observingModes"][0]["IWA"]
    dMag0 = specs["dMag0"]
    dMag_ub = 23
    system.star.mass = 1 * u.M_sun
    system.star.dist = 20 * u.pc
    t0 = system.planets[0].t0
    tf = 366
    dt = 1
    system, plot_times = earthLike_setup(system, t0, tf, dt)
    azim_range = np.linspace(15, 75, len(plot_times))

    alias_planets = []
    for inc in np.linspace(2, 178, 25) * u.deg:
        planet = create_inc_alias(system, system.planets[0], inc)
        planet.p = np.random.rand(1) * 0.3 + 0.2
        planet.WAs = []
        planet.dMags = []
        alias_planets.append(planet)
    # system.planets += alias_planets
    system.planets = alias_planets
    generate_system_pdet(system, plot_times, IWA, np.inf, dMag0)

    # Set up plotting stuff
    system.star.scatter_kwargs = {"s": 50, "zorder": 1}
    # Add planet params
    for planet in system.planets:
        planet.pos = utils.calc_position_vectors(planet, plot_times)
        planet.all_z = [val[0] for val in planet.pos.z.values]
        planet.scatter_kwargs_2d = copy.deepcopy(planet_scatter_kwargs_2d)
        planet.scatter_kwargs_3d = copy.deepcopy(planet_scatter_kwargs_3d)
        planet.trail_kwargs = copy.deepcopy(trail_kwargs)
        planet.base_s = (
            (
                10
                + 30
                / (30 * u.M_earth - 1 * u.M_earth).decompose()
                * planet.mass.decompose()
            )
            .decompose()
            .value
        )

    # Plot work
    for i, time in enumerate(tqdm(plot_times, desc="Figure 4")):
        fig = plt.figure(figsize=[10, 10])
        subfigs = fig.subfigures(nrows=2, ncols=2)
        axes = [
            subfigs[0, 0].add_subplot(projection="3d"),
            subfigs[0, 1].add_subplot(),
            subfigs[1, 0].add_subplot(),
            subfigs[1, 1].add_subplot(),
        ]
        ax_3d = axes[0]
        ax_3d.set_title("Orbits in 3d")
        ax_3d = earth_3d_plane(ax_3d, azim_range[i])

        ax_img = axes[1]
        ax_img = earth_img_plane(ax_img)
        ax_img = add_IWA(ax_img, IWA, system, u.AU)

        ax_sep = axes[2]
        ax_sep.set_ylim([15, 30])
        ax_sep.set_xlim([0, 1.5])
        ax_sep.set_ylabel(r"Planet-star $\Delta$mag")
        ax_sep.set_xlabel("Planet-star separation (AU)")

        ax_pdet = axes[3]
        ax_pdet.set_title("Probability of detection")
        ax_pdet = time_x_label(ax_pdet, plot_times)
        ax_pdet.set_xlabel("Time (yr)")
        ax_pdet.set_ylim([-0.05, 1.05])
        ax_pdet.set_yticks(np.arange(0, 1.1, 0.2))
        # ax_rv.set_ylabel("RV (m/s)")
        for planet in system.planets:
            z_val = planet.all_z[i]
            planet = get_planet_zorder(planet, time)
            ax_3d = darken_3d_background(ax_3d)
            planet.scatter_kwargs_3d["s"] = planet_marker_size(
                z_val, planet.all_z, base_size=planet.base_s, factor=0.25
            )
            planet.scatter_kwargs_3d["alpha"] = get_planet_alpha(
                planet, i, dMag0, dMag_ub
            )
            ax_3d = scatter_planet(ax_3d, planet, i, unit=u.AU, projection="3d")

            # Image plane
            planet.scatter_kwargs_2d["s"] = planet_marker_size(
                z_val, planet.all_z, base_size=planet.base_s
            )
            planet.scatter_kwargs_2d["alpha"] = get_planet_alpha(
                planet, i, dMag0, dMag_ub
            )
            ax_img = scatter_planet(ax_img, planet, i, unit=u.AU, projection="2d")

            # Separation dMag plot
            ax_sep = add_sep_plot(ax_sep, planet, system, i, IWA, dMag0, u.AU)

            # pdet data
            ax_pdet.plot(
                plot_times.decimalyear[:i] - plot_times[0].decimalyear,
                system.pdets[:i],
                color="w",
            )

        ax_img.set_aspect("equal", "box")
        ax_3d = scatter_star(ax_3d, [0, 0, 0], system, "3d", i)
        ax_img = scatter_star(ax_img, [0, 0, 0], system, "2d", i)
        fig.savefig(f"figures/fig4/fig4_{i:003d}.png")
        plt.close()


def fig_5(system, specs, SS):
    IWA = specs["observingModes"][0]["IWA"]
    system.star.mass = 1 * u.M_sun
    system.star.dist = 20 * u.pc
    t0 = system.planets[0].t0
    tf = 366
    dt = 1
    system, plot_times = earthLike_setup(system, t0, tf, dt)

    # EXOSIMS stuff
    int_times = [5 * u.d, 10 * u.d, 30 * u.d, 60 * u.d, 100 * u.d]
    dMag0s = []
    for int_time in int_times:
        fZ = SS.ZodiacalLight.fZ0
        fEZ = SS.ZodiacalLight.fEZ0
        TL = SS.TargetList
        mode = list(
            filter(
                lambda mode: mode["detectionMode"],
                SS.TargetList.OpticalSystem.observingModes,
            )
        )[0]
        WA = np.mean([mode["IWA"].value, mode["OWA"].value]) * u.arcsec
        dMag = SS.OpticalSystem.calc_dMag_per_intTime(
            int_time, TL, 0, fZ, fEZ, WA, mode
        )
        dMag0s.append(dMag[0])
    alias_planets = []
    for inc in np.linspace(2, 178, 25) * u.deg:
        planet = create_inc_alias(system, system.planets[0], inc)
        planet.p = np.random.rand(1) * 0.3 + 0.2
        planet.WAs = []
        planet.dMags = []
        alias_planets.append(planet)
    # system.planets += alias_planets
    system.planets = alias_planets
    systems = [copy.deepcopy(system) for i in int_times]
    cmap = plt.get_cmap("viridis")
    i = 0
    for system, dMag0, int_time in zip(systems, dMag0s, int_times):
        generate_system_pdet(system, plot_times, IWA, np.inf, dMag0)
        system.dMag_label = str(int_time.to(u.d))
        system.color = cmap(i / len(systems))
        i += 1
        for planet in system.planets:
            planet.pos = utils.calc_position_vectors(planet, plot_times)
            planet.all_z = [val[0] for val in planet.pos.z.values]
            planet.scatter_kwargs_2d = copy.deepcopy(planet_scatter_kwargs_2d)
            planet.scatter_kwargs_3d = copy.deepcopy(planet_scatter_kwargs_3d)
            planet.trail_kwargs = copy.deepcopy(trail_kwargs)
            planet.base_s = (
                (
                    10
                    + 30
                    / (30 * u.M_earth - 1 * u.M_earth).decompose()
                    * planet.mass.decompose()
                )
                .decompose()
                .value
            )

    # Make single plot of pdet
    fig, ax_pdet = plt.subplots()
    ax_pdet.set_title("Probability of detection")
    ax_pdet = time_x_label(ax_pdet, plot_times)
    ax_pdet.set_xlabel("Time (yr)")
    ax_pdet.set_ylim([-0.05, 1.05])
    ax_pdet.set_yticks(np.arange(0, 1.1, 0.2))
    for system in systems:
        ax_pdet.plot(
            plot_times.decimalyear - plot_times[0].decimalyear,
            system.pdets,
            color=system.color,
            label=system.dMag_label,
        )
    ax_pdet.legend(loc="upper left")
    fig.savefig("figures/pdet_per_int.png", dpi=300)
    plt.close()
    breakpoint()

    for i, time in enumerate(tqdm(plot_times, desc="Figure 5")):
        fig = plt.figure(figsize=[13, 7.3])
        subfigs = fig.subfigures(ncols=2)
        axes = [subfigs[0].add_subplot(), subfigs[1].add_subplot()]
        ax_sep = axes[0]
        ax_sep.set_ylim([15, 30])
        ax_sep.set_xlim([0, 1.5])
        ax_sep.set_ylabel(r"Planet-star $\Delta$mag")
        ax_sep.set_xlabel("Planet-star separation (AU)")

        ax_pdet = axes[1]
        ax_pdet.set_title("Probability of detection")
        ax_pdet = time_x_label(ax_pdet, plot_times)
        ax_pdet.set_xlabel("Time (yr)")
        ax_pdet.set_ylim([-0.05, 1.05])
        ax_pdet.set_yticks(np.arange(0, 1.1, 0.2))
        IWA_dist = (np.tan(IWA) * system.star.dist).to(u.AU).value
        IWA_line = mpl.lines.Line2D([IWA_dist, IWA_dist], [15, dMag0], color="red")
        ax_sep.add_line(IWA_line)
        # ax_rv.set_ylabel("RV (m/s)")
        for system, dMag0 in zip(systems, dMag0s):
            for planet in system.planets:
                # Separation dMag plot
                ax_sep = add_sep_plot(
                    ax_sep, planet, system, i, IWA, dMag0, u.AU, add_lines=False
                )
            # pdet data
            dMag_line = mpl.lines.Line2D(
                [IWA_dist, 5], [dMag0, dMag0], color=system.color
            )
            ax_sep.add_line(dMag_line)
            ax_pdet.plot(
                plot_times.decimalyear[:i] - plot_times[0].decimalyear,
                system.pdets[:i],
                color=system.color,
                label=system.dMag_label,
            )
        ax_pdet.legend(loc="upper left")

        fig.savefig(f"figures/fig5/fig5_{i:003d}.png")
        plt.close()


def fig_6(builder):
    fitted_systems = [
        star.replace("_", " ") for star in builder.rvdata.pdet.pops.keys()
    ]
    fitted_sInds = [
        np.where(builder.rvdata.universe.SU.TargetList.Name == system)[0][0]
        for system in fitted_systems
    ]
    # system_ind = 2
    system_ind = -1
    system_name = fitted_systems[system_ind]
    system_sInd = fitted_sInds[system_ind]
    systems = [builder.rvdata.universe.systems[sInd] for sInd in fitted_sInds]
    system = systems[system_ind]
    pdets = builder.rvdata.pdet.pdets[system_name.replace(" ", "_")]
    rv_info = builder.rvdata.surveys[0].syst_observations[system_sInd]
    rv_sorted = rv_info.sort_values(by="time").reset_index(drop=True)
    plot_times = Time(rv_sorted.time.values, format="jd")
    rv_vals = -rv_sorted.mnvel.values
    rv_err = rv_sorted.errvel.values
    azim_range = np.linspace(15, 75, len(plot_times))
    # alias_planets = find_msini_aliases(system, system.planets[0], 25)

    # generate_system_rv(system, plot_times)
    # Get z-vals
    # Set up plotting stuff
    system.star.scatter_kwargs = {"s": 50, "zorder": 1}
    system.rv_df = pd.DataFrame(
        np.stack((plot_times, rv_vals), axis=1), columns=["t", "rv"]
    )
    # Add planet params
    for planet in system.planets:
        planet.pos = utils.calc_position_vectors(planet, plot_times)
        planet.all_z = [val[0] for val in planet.pos.z.values]
        planet.scatter_kwargs_2d = copy.deepcopy(planet_scatter_kwargs_2d)
        planet.scatter_kwargs_3d = copy.deepcopy(planet_scatter_kwargs_3d)
        planet.trail_kwargs = trail_kwargs
        planet.base_s = (
            (
                10
                + 100
                / (3 * u.M_jupiter - 0.1 * u.M_earth).decompose()
                * planet.mass.decompose()
            )
            .decompose()
            .value
        )

    # Plot work
    star_cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=min(rv_vals), vmax=max(rv_vals))
    for i, time in enumerate(tqdm(plot_times, desc="Figure 6")):
        fig = plt.figure(figsize=[16, 5])
        subfigs = fig.subfigures(nrows=1, ncols=3, width_ratios=[1.5, 1, 1])
        axes = [
            subfigs[0].add_subplot(),
            subfigs[1].add_subplot(projection="3d"),
            subfigs[2].add_subplot(),
        ]
        ax_rv = axes[0]
        ax_rv.set_title("Radial velocity")
        ax_rv = time_x_label(ax_rv, plot_times)
        ax_rv.set_xlabel("Time (yr)")
        ax_rv.set_ylim([-5, 5])
        ax_rv.set_ylabel("RV (m/s)")

        # RV data
        ax_rv.errorbar(
            plot_times.decimalyear[:i] - plot_times[0].decimalyear,
            rv_vals[:i],
            yerr=rv_err[:i],
            ecolor="w",
            alpha=0.2,
            fmt="none",
        )
        ax_rv.scatter(
            plot_times.decimalyear[:i] - plot_times[0].decimalyear,
            rv_vals[:i],
            c=rv_vals[:i],
            cmap=star_cmap,
            norm=norm,
            s=10,
        )

        ax_3d = axes[1]
        ax_3d.set_title("Orbits in 3d")
        ax_3d = full_3d_plane(ax_3d, azim_range[i])

        ax_img = axes[2]
        ax_img.set_title("Image plane")
        ax_img = full_img_plane(ax_img)
        # ax_img = add_IWA(ax_img, IWA, system, u.AU)

        for planet in system.planets:
            # 3d
            z_val = planet.all_z[i]
            planet = get_planet_zorder(planet, time)
            ax_3d = darken_3d_background(ax_3d)
            planet.scatter_kwargs_3d["s"] = planet_marker_size(
                z_val, planet.all_z, base_size=planet.base_s, factor=0.25
            )
            ax_3d = scatter_planet(ax_3d, planet, i, unit=u.AU, projection="3d")

            # Image plane
            planet.scatter_kwargs_2d["s"] = planet_marker_size(
                z_val, planet.all_z, base_size=planet.base_s
            )
            ax_img = scatter_planet(ax_img, planet, i, unit=u.AU, projection="2d")

        ax_img.set_aspect("equal", "box")
        ax_3d = scatter_star(ax_3d, [0, 0, 0], system, "3d", i)
        ax_img = scatter_star(ax_img, [0, 0, 0], system, "2d", i)
        fig.savefig(f"figures/fig6/fig6_{i:003d}.png", bbox_inches="tight")
        plt.close()


def fig7(builder):
    fitted_systems = [
        star.replace("_", " ") for star in builder.rvdata.pdet.pops.keys()
    ]
    fitted_sInds = [
        np.where(builder.rvdata.universe.SU.TargetList.Name == system)[0][0]
        for system in fitted_systems
    ]
    # system_ind = 2
    system_ind = -1
    system_name = fitted_systems[system_ind]
    system_sInd = fitted_sInds[system_ind]
    systems = [builder.rvdata.universe.systems[sInd] for sInd in fitted_sInds]
    system = systems[system_ind]
    pdets = builder.rvdata.pdet.pdets[system_name.replace(" ", "_")]
    pops = builder.rvdata.pdet.pops[system_name.replace(" ", "_")]
    rv_info = builder.rvdata.surveys[0].syst_observations[system_sInd]
    rv_sorted = rv_info.sort_values(by="time").reset_index(drop=True)
    plot_times = Time(rv_sorted.time.values, format="jd")
    rv_vals = -rv_sorted.mnvel.values
    rv_err = rv_sorted.errvel.values
    azim_range = np.linspace(15, 75, len(plot_times))
    # alias_planets = find_msini_aliases(system, system.planets[0], 25)

    generate_system_rv(system, plot_times)
    # Get z-vals
    # Set up plotting stuff
    system.star.scatter_kwargs = {"s": 50, "zorder": 1}
    system.rv_df = pd.DataFrame(
        np.stack((plot_times, rv_vals), axis=1), columns=["t", "rv"]
    )
    # Add planet params
    for planet in system.planets:
        planet.pos = utils.calc_position_vectors(planet, plot_times)
        planet.all_z = [val[0] for val in planet.pos.z.values]
        planet.scatter_kwargs_2d = copy.deepcopy(planet_scatter_kwargs_2d)
        planet.scatter_kwargs_3d = copy.deepcopy(planet_scatter_kwargs_3d)
        planet.trail_kwargs = trail_kwargs
        planet.base_s = (
            (
                10
                + 100
                / (3 * u.M_jupiter - 0.1 * u.M_earth).decompose()
                * planet.mass.decompose()
            )
            .decompose()
            .value
        )
    cmap = plt.get_cmap("viridis")
    cmap_vals = np.linspace(0, 1, len(pops))
    for i, pop in enumerate(pops):
        pop.pos = utils.calc_position_vectors(pop, plot_times)
        pop.all_z = [val[0] for val in pop.pos.z.values]
        tmp2d = copy.deepcopy(planet_scatter_kwargs_2d)
        tmp2d["s"] = 0.1
        tmp2d["alpha"] = 0.5
        tmp2d["c"] = cmap(cmap_vals[i])
        tmp3d = copy.deepcopy(planet_scatter_kwargs_3d)
        tmp3d["s"] = 0.4
        tmp3d["alpha"] = 0.5
        tmp3d["c"] = cmap(cmap_vals[i])
        pop.scatter_kwargs_2d = tmp2d
        pop.scatter_kwargs_3d = tmp3d
        # pop.trail_kwargs = trail_kwargs
        # pop.base_s = (
        #     (
        #         10
        #         + 100
        #         / (3 * u.M_jupiter - 0.1 * u.M_earth).decompose()
        #         * pop.mass.decompose()
        #     )
        #     .decompose()
        #     .value
        # )

    # Create system from the fitted orbits
    fitted_system = copy.deepcopy(system)
    fitted_system.planets = []
    for pop in pops:
        planet = pop2planet(pop, system)
        fitted_system.planets.append(planet)
    generate_system_rv(fitted_system, plot_times)

    # Plot work
    star_cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=min(rv_vals), vmax=max(rv_vals))
    for i, time in enumerate(tqdm(plot_times, desc="Figure 7")):
        fig = plt.figure(figsize=[16, 5])
        subfigs = fig.subfigures(nrows=1, ncols=3, width_ratios=[1.5, 1, 1])
        axes = [
            subfigs[0].add_subplot(),
            subfigs[1].add_subplot(projection="3d"),
            subfigs[2].add_subplot(),
        ]
        ax_rv = axes[0]
        ax_rv.set_title("Radial velocity")
        ax_rv = time_x_label(ax_rv, plot_times)
        ax_rv.set_xlabel("Time (yr)")
        ax_rv.set_ylim([-5, 5])
        ax_rv.set_ylabel("RV (m/s)")

        # RV data
        ax_rv.errorbar(
            plot_times.decimalyear[:i] - plot_times[0].decimalyear,
            rv_vals[:i],
            yerr=rv_err[:i],
            ecolor="w",
            alpha=0.2,
            fmt="none",
        )
        ax_rv.scatter(
            plot_times.decimalyear[:i] - plot_times[0].decimalyear,
            rv_vals[:i],
            c=rv_vals[:i],
            cmap=star_cmap,
            norm=norm,
            s=10,
        )
        ax_rv.plot(
            plot_times.decimalyear[:i] - plot_times[0].decimalyear,
            fitted_system.rv_df.rv.values[:i],
            c=cmap(0.5),
            alpha=0.5
            # c=fitted_system.rv_df.rv.values[:i],
            # cmap=star_cmap,
            # norm=norm,
        )

        ax_3d = axes[1]
        ax_3d.set_title("Orbits in 3d")
        ax_3d = full_3d_plane(ax_3d, azim_range[i])

        ax_img = axes[2]
        ax_img.set_title("Image plane")
        ax_img = full_img_plane(ax_img)
        # ax_img = add_IWA(ax_img, IWA, system, u.AU)

        for pop in pops:
            ax_3d = scatter_pop(ax_3d, pop, i, unit=u.AU, projection="3d")
            ax_img = scatter_pop(ax_img, pop, i, unit=u.AU, projection="2d")

        for planet in system.planets:
            # 3d
            z_val = planet.all_z[i]
            planet = get_planet_zorder(planet, time)
            ax_3d = darken_3d_background(ax_3d)
            planet.scatter_kwargs_3d["s"] = planet_marker_size(
                z_val, planet.all_z, base_size=planet.base_s, factor=0.25
            )
            ax_3d = scatter_planet(ax_3d, planet, i, unit=u.AU, projection="3d")

            # Image plane
            planet.scatter_kwargs_2d["s"] = planet_marker_size(
                z_val, planet.all_z, base_size=planet.base_s
            )
            ax_img = scatter_planet(ax_img, planet, i, unit=u.AU, projection="2d")

        ax_img.set_aspect("equal", "box")
        ax_3d = scatter_star(ax_3d, [0, 0, 0], system, "3d", i)
        ax_img = scatter_star(ax_img, [0, 0, 0], system, "2d", i)
        fig.savefig(f"figures/fig7/fig7_{i:003d}.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    font = {"size": 13}
    plt.rc("font", **font)
    plt.style.use("dark_background")

    # with open("caseA.json") as f:
    #     specs = json.loads(f.read())
    # assert "seed" in specs.keys(), (
    #     "For reproducibility the seed should" " not be randomized by EXOSIMS."
    # )

    # # # Need to use SurveySimulation if we want to have a random seed
    # SS = get_module_from_specs(specs, "SurveySimulation")(**specs)
    # SU = SS.SimulatedUniverse
    # universe_params = {
    #     "universe_type": "exosims",
    #     "script": "caseB.json",
    # }
    # universe_params["missionStart"] = specs["missionStart"]
    # universe_params["nsystems"] = 10
    # universe = ExosimsUniverse(SU, universe_params)

    # system = universe.systems[0]
    # fig_1(system, specs)
    # fig_2(universe.systems[1], specs)
    # fig_3(universe.systems[2], specs)
    # fig_4(universe.systems[3], specs)
    # fig_5(universe.systems[4], specs, SS)

    builder = create_builder()
    # fig_6(builder)
    fig7(builder)
