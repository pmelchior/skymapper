[![PyPI](https://img.shields.io/pypi/v/skymapper.svg)](https://pypi.python.org/pypi/skymapper/)
[![License](https://img.shields.io/github/license/pmelchior/skymapper.svg)](https://github.com/pmelchior/skymapper/blob/master/LICENSE.md)

# Skymapper

*A collection of matplotlib instructions to map astronomical survey data from the celestial sphere onto 2D.*

The purpose of this package is to facilitate interactive work as well as the the creation of publication-quality plots with a python-based workflow many astronomers are accustomed to. The primary motivation is a truthful representation of samples and fields from the curved sky in planar figures, which becomes relevant when sizable portions of the sky are observed.

What can it do? For instance, find the optimal projection for a given list of spherical coordinates and [creating a density map](examples/example1.py) from a catalog in a few lines:

```python
import skymapper as skm

# 1) construct a projection, here Albers
# lon_0, lat_0: longitude/latitude that map onto 0/0
# lat_1, lat_2: reference latitudes for conic projection
lon_0, lat_0, lat_1, lat_2 = 27.35, -37.04, -57.06, -11.34
proj = skm.Albers(lon_0, lat_0, lat_1, lat_2)

# alternative: define the optimal projection for set of coordinates
# by minimizing the variation in distortion
crit = skm.stdDistortion
proj = skm.Albers.optimize(ra, dec, crit=crit)

# 2) construct map: will hold figure and projection
# the outline of the sphere can be styled with kwargs for matplotlib Polygon
map = skm.Map(proj)

# 3) add graticules, separated by 15 deg
# the lines can be styled with kwargs for matplotlib Line2D
# additional arguments for formatting the graticule labels
sep = 15
map.grid(sep=sep)

# 4) add data to the map, e.g.
# make density plot
nside = 32
mappable = map.density(ra, dec, nside=nside)
cb = map.colorbar(mappable, cb_label="$n_g$ [arcmin$^{-2}$]")

# add scatter plot
map.scatter(ra_scatter, dec_scatter, s=size_scatter, edgecolor='k', facecolor='None')

# focus on relevant region
map.focus(ra, dec)
```

![Random density in DES footprint](https://github.com/pmelchior/skymapper/assets/1463403/fa1528f4-2ed0-4945-a342-17bddd64d73d)

The `map` instance has several members, most notably

*  `fig`: the `matplotlib.Figure` that holds the map
* `ax`: the `matplotlib.Axes` that holds the map

The syntax mimics `matplotlib` as closely as possible. Currently supported are canonical plotting functions

* `plot`
* `scatter`
* `hexbin` for binning and interpolating samples
* `colorbar` with an optional argument `cb_label` to set the label
* `text` with an optional `direction in ['parallel','meridian']` argument to align along either graticule

as well as special functions

* `footprint` to show the region covered by a survey
* `vertex` to plot a list of simple convex polygons
* `healpix` to plot a healpix map as a list of polygons
* `density` to create a density map in healpix cells
* `extrapolate` to generate a field from samples over the entire sky or a subregion

Exploratory and interactive workflows are specifically supported. For instance, you can zoom and pan, also scroll in/out (google-maps style), and the `map` will automatically update the location of the graticule labels, which are not regularly spaced.

The styling of graticules can be changed by calling `map.grid()` with different parameters. Finer-grained access is provided by 

* `map.labelParallelsAtFrame()` creates/styles the vertical axis labels at the intersection of the grid parallels
* `map.labelMeridiansAtFrame()` creates/styles the horizontal axis labels at the intersection of the grid meridians
* `map.labelParallelsAtMeridian()` creates/styles parallels at a given meridian (useful for all-sky maps)
* `map.labelMeridiansAtParallel()` creates/styles meridians at a given parallel (useful for all-sky maps)

## Installation and Prerequisites

You can either clone the repo and install with `pip install .` or get the latest release with `pip install skymapper`.

Dependencies:

* numpy
* scipy
* matplotlib
* healpy

For survey footprints, you'll need [`pymangle`](https://github.com/esheldon/pymangle).

## Background

The essential parts of the workflow are

1. Creating the `Projection`, e.g. `Hammer`, `Albers`, `WagnerIV`
2. Setting up a `Map` to hold the projection and matplotlib figure, ax, ...
3. Add data to the map

Several map projections are available, the full list is stored in the dictionary `projection_register`. If the projection you want isn't included, open an issue, or better: create it yourself (see below) and submit a pull request.

There are two conventions for longitudes in astronomy. The standard frame, used for instance for world maps or Galactic maps, has a longitudinal coordinates in the range [-180 .. 180] deg, which increase west to east (in other words, on the map east is right). The equatorial (RA/Dec) frame is left-handed (i.e. on the map east is left) and has coordinates in the range [0 .. 360] deg. To determine the convention, `Projection` has an argument `lon_type`, which can be either `"lon"` or `"ra"` for standard or equatorial, respectively. The default is `lon_type="ra"`.

Map projections can preserve sky area, angles, or distances, but never all three. That means defining a suitable projection must be a compromise. For most applications, sizes should exactly be preserved, which means that angles and distances may not be. The optimal projection for a given list of `ra`, `dec` can be found by calling:


```python
crit = skm.projection.stdDistortion
proj = skm.Albers.optimize(ra, dec, crit=crit)
```

This optimizes the `Albers` projection parameters to minimize the variance of the map distortion (i.e. the apparent ellipticity of a true circle on the sky). Alternative criteria are e.g. `maxDistortion` or `stdScale` (for projections that are not equal-area).

### Creating a custom projection

For constructing your own projection, derive from [`Projection`](skymapper/projection.py). You'll see that every projection needs to implement at least these methods: 

* `transform` to map from spherical to map coordinates x/y
* `invert` to map from x/y to spherical (if not implemented defaults to basic and slow BFGS inversion)

If the projection has several parameters, you will want to create a special `@classmethod optimize` because the default one only determines the best longitude reference. An example for that is given in e.g. `ConicProjection.optimize`.

### Creating/using a survey

Several surveys are predefined and listed in the `survey_register` dictionary. If the survey you want isn't included, don't despair. To create one can derive a class from [`Survey`](skymapper/survey/__init__.py), which only needs to implement one method:

​	`def contains(self, ra, dec)` to determine whether RA, Dec are inside the footprint.

If this looks like the [`pymangle`](https://github.com/esheldon/pymangle) interface: it should. That means that you can avoid the overhead of having to define a survey and e.g. pass a `pymangle.Mangle` object directly to `footprint()`.

### Limitation(s)

The combination of `Map` and `Projection` is *not* a [matplotlib transformation](http://matplotlib.org/users/transforms_tutorial.html). Among several reasons, it is very difficult (maybe impossible) to work with the `matplotlib.Axes` that are not rectangles or ellipses. So, we decided to split the problem: making use of matplotlib for lower-level graphics primitive and layering the map-making on top of it. This way, we can control e.g. the interpolation method on the sphere or the location of the tick labels in a way consistent with visual expectations from hundreds of years of cartography. While `skymapper` tries to follow matplotlib conventions very closely, some methods may not work as expected. Open an issue if you think you found such a case.

In particular, we'd appreciate help to make sure that the interactive features work well on all matplotlib backends.
