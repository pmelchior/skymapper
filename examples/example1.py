# load projection and helper functions
import numpy as np
import skymapper as skm
import matplotlib.pyplot as plt

def getCatalog(size=10000, survey=None):
    # dummy catalog: uniform on sphere
    # Marsaglia (1972)
    xyz = np.random.normal(size=(size, 3))
    r = np.sqrt((xyz**2).sum(axis=1))
    dec = np.arccos(xyz[:,2]/r) / skm.DEG2RAD - 90
    ra = 180 - np.arctan2(xyz[:,0], xyz[:,1]) / skm.DEG2RAD

    # survey selection
    if survey is not None:
        inside = survey.contains(ra, dec)
        ra = ra[inside]
        dec = dec[inside]
    return ra, dec

def makeHealpixMap(ra, dec, nside=1024, nest=False):
    # convert a ra/dec catalog into healpix map with counts per cell
    import healpy as hp
    ipix = hp.ang2pix(nside, ra, dec, nest=nest, lonlat=True)
    return np.bincount(ipix, minlength=hp.nside2npix(nside))

class TestSurvey(skm.survey.Survey):
    def contains(self, ra, dec):
        # simplistic DES like survey
        return (dec < 5) & (dec > -60) & ((ra < 90) | (ra > 300))


if __name__ == "__main__":

    # load RA/Dec from catalog
    size = 100000
    try:
        from skymapper.survey import DES
        survey = DES()
    except ImportError:
        survey = TestSurvey()
    ra, dec = getCatalog(size, survey=survey)

    # define the best Albers projection for the footprint
    # minimizing the variation in distortion
    # alternatively, specify the projection, e.g. proj = skm.Hammer(0)
    crit = skm.stdDistortion
    proj = skm.Albers.optimize(ra, dec, crit=crit)

    # construct map:
    # if fig axis is provided, will use it; otherwise will create figure
    # the outline of the map can be styled with kwargs for matplotlib Polygon
    fig = plt.figure(tight_layout=True, figsize=(10, 6))
    ax = fig.add_subplot(111, aspect='equal')
    map = skm.Map(proj, ax=ax)

    # add graticules, separated by 15 deg
    # the lines can be styled with kwargs for matplotlib Line2D
    # additional arguments for formatting the graticule labels
    sep=15
    map.grid(sep=sep)

    #### 1. plot density in healpix cells ####
    nside = 32
    mappable = map.density(ra, dec, nside=nside)
    cb = map.colorbar(mappable, cb_label="$n$ [arcmin$^{-2}$]")

    # add random scatter plot
    nsamples = 30
    size = 100*np.random.rand(nsamples)
    map.scatter(ra[:nsamples], dec[:nsamples], s=size, edgecolor='w', facecolor='None')

    # focus on relevant region
    map.focus(ra, dec)

    # entitle: access mpl figure
    map.title('Density with random scatter')

    # save
    map.savefig("skymapper-example1.png")

    #### 2. plot healpix map ####
    # clone the map without data contents
    map2 = map.clone()

    # to make healpix map, simply bin the counts of ra/dec
    m = makeHealpixMap(ra, dec, nside=nside)
    m = np.ma.array(m, mask=(m==0))
    mappable2 = map2.healpix(m, cmap="YlOrRd")
    cb2 = map2.colorbar(mappable2, cb_label="Healpix cell count")
    map2.title('Healpix map')

    #### 3. show map distortion over the survey ####
    # compute distortion for all ra, dec and make hexbin plot
    map3 = map.clone()
    a,b = proj.distortion(ra, dec)
    mappable3 = map3.hexbin(ra, dec, C=1-np.abs(b/a), vmin=0, vmax=0.3, cmap='RdYlBu_r')
    cb3 = map3.colorbar(mappable3, cb_label='Distortion')
    map3.title('Projection distortion')

    #### 4. extrapolate sampled values over all sky with an all-sky projection ####
    proj4 = skm.McBrydeThomasFPQ(0)
    map4 = skm.Map(proj4)

    # show with 45 deg graticules
    sep=45
    map4.grid(sep=sep)

    # alter number of labels at the south pole
    map4.labelMeridiansAtParallel(-90, size=8)#, meridians=np.arange(0,360,90))
    map4.labelMeridiansAtParallel(0, color='w')

    # as example: extrapolate declination over sky
    # this is slow when working with lots of samples...
    mappable4 = map4.extrapolate(ra[::10], dec[::10], dec[::10], resolution=100, cmap='Spectral', vmax=90, vmin=-90)
    cb4 = map4.colorbar(mappable4, cb_label='Dec')

    # add footprint shade
    nside = 32
    footprint4 = map4.footprint(survey, nside=nside, zorder=20, facecolors='w', alpha=0.3)
    map4.title('Extrapolation on the sphere')

    # when run as a script: need to show the result
    plt.show()