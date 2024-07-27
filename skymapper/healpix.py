import healpy as hp
import numpy as np

# python 3 compatible
try:
    xrange
except NameError:
    xrange = range

def getHealpixArea(nside):
    return hp.nside2pixarea(nside, degrees=True)

def getHealpixVertices(pixels, nside, nest=False):
    """Get polygon vertices for list of HealPix pixels.

    Args:
        pixels: list of HealPix pixels
        nside: HealPix nside
        nest: HealPix nesting scheme

    Returns:
        vertices: (N,4,2), RA/Dec coordinates of 4 boundary points of cell
    """
    corners = np.transpose(hp.boundaries(nside, pixels, step=1, nest=nest), (0, 2, 1))
    corners_x = corners[:, :, 0].flatten()
    corners_y = corners[:, :, 1].flatten()
    corners_z = corners[:, :, 2].flatten()
    vertices_lon, vertices_lat = hp.rotator.vec2dir(corners_x, corners_y, corners_z, lonlat=True)
    return np.stack([vertices_lon.reshape(-1, 4), vertices_lat.reshape(-1, 4)], axis=-1)


def getGrid(nside, nest=False, return_vertices=False):
    pixels = np.arange(hp.nside2npix(nside))
    lon, lat = hp.pix2ang(nside, pixels, nest=nest, lonlat=True)
    if return_vertices:
        vertices = getHealpixVertices(pixels, nside, nest=nest)
        return pixels, lon, lat, vertices
    return pixels, lon, lat

def getCountAtLocations(lon, lat, nside=512, nest=False, per_area=True, return_vertices=False):
    """Get number density of objects from lon, lat in HealPix cells.

    Args:
        lon: list of longitudes in degrees
        lat: list of latutude in degrees
        nside: HealPix nside
        nest: Healpix NEST scheme
        per_area: return counts in units of 1/arcmin^2
        return_vertices: whether to also return the boundaries of HealPix cells

    Returns:
        bc, [vertices]
        bc: count of objects in a HealPix cell if count > 0
        vertices: (N,4,2), RA/Dec coordinates of 4 boundary points of cell
    """
    # get healpix pixels
    ipix = hp.ang2pix(nside, lon, lat, lonlat=True, nest=nest)
    # count how often each pixel is hit
    bc = np.bincount(ipix, minlength=hp.nside2npix(nside))
    if per_area:
        bc = bc.astype('f8')
        bc /= hp.nside2resol(nside, arcmin=True)**2 # in arcmin^-2

    # for every non-empty pixel: get the vertices that confine it
    if return_vertices:
        pixels = np.nonzero(bc)[0]
        vertices = getHealpixVertices(pixels, nside)
        return bc, vertices
    return bc

def reduceAtLocations(lon, lat, value, reduce_fct=np.mean, nside=512, nest=False, return_vertices=False):
    """Reduce values at given lon, lat in HealPix cells to a scalar.

    Args:
        lon: list of longitudes in degrees
        lat: list of latutude in degrees
        value: list of values to be reduced
        reduce_fct: function to operate on values in each cell
        nside: HealPix nside
        nest: Healpix NEST scheme
        return_vertices: whether to also return the boundaries of HealPix cells

    Returns:
        v, [vertices]
        v: reduction of values in a HealPix cell if count > 0, otherwise masked
        vertices: (N,4,2), RA/Dec coordinates of 4 boundary points of cell
    """
    # get healpix pixels
    ipix = hp.ang2pix(nside, lon, lat, lonlat=True, nest=nest)
    # count how often each pixel is hit, only use non-empty pixels
    bc = np.bincount(ipix, minlength=hp.nside2npix(nside))
    pixels = np.nonzero(bc)[0]

    v = np.ma.empty(bc.size, mask=(bc==0))
    for pixel in pixels:
        sel = (ipix == pixels)
        v.data[pixel] = reduce_fct(value[sel])

    # get the vertices that confine each pixel
    # convert to lon, lat (thanks to Eric Huff)
    if return_vertices:
        vertices = getHealpixVertices(pixels, nside)
        return v, vertices
    return v
