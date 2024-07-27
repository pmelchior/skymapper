import matplotlib
import matplotlib.pyplot
import numpy as np
import math, re, pickle
import scipy.interpolate
from . import healpix
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


DEG2RAD = np.pi/180

def angularDistance(radec1, radec2):
    """Compute distance on the curved sky"""
    ra1, dec1 = radec1 * DEG2RAD
    ra2, dec2 = radec2 * DEG2RAD
    return np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2))


# extrapolation function from
# http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
# improved to order x and y to have ascending x
def extrap(x, xp, yp):
    """np.interp function with linear extrapolation"""
    x_ = np.array(x)
    order = np.argsort(xp)
    xp_ = xp[order]
    yp_ = yp[order]

    y = np.array(np.interp(x_, xp_, yp_))
    y[x_ < xp_[0]] = yp_[0] + (x_[x_ < xp_[0]] -xp_[0]) * (yp_[0] - yp_[1]) / (xp_[0] - xp_[1])
    y[x_ > xp_[-1]] = yp_[-1] + (x_[x_ > xp_[-1]] -xp_[-1])*(yp_[-1] - yp_[-2])/(xp_[-1] - xp_[-2])
    return y

def degFormatter(deg):
    """Formats degrees as X˚

    Args:
        deg: float
    Returns:
        string
    """
    return "${:d}^\circ$".format(int(deg))

def degPMFormatter(deg):
    """Formats degrees as [+-]X˚

    Args:
        deg: float

    Return:
        String
    """
    if deg == 0:
        return degFormatter(deg)
    return "${:+d}^\circ$".format(int(deg))

def deg360Formatter(deg):
        """Formats degrees as X˚ with X in [0..360]

        Args:
            deg: float
        Returns:
            string
        """
        if deg < 0:
            deg += 360
        return degFormatter(deg)

def deg180Formatter(deg):
        """Formats degrees as X˚ with X in [-180..180]

        Args:
            deg: float
        Returns:
            string
        """
        if deg < -180:
            deg += 360
        if deg > 180:
            deg -= 360
        return degPMFormatter(deg)

def hourAngleFormatter(ra):
    """String formatter for "hh:mm"

    Args:
        deg: float

    Return:
        String
    """
    if ra < 0:
        ra += 360
    hours = int(ra//15)
    minutes = int(float(ra - hours*15)/15 * 60)
    if minutes:
        return "${:d}^{{{:>02}}}$h".format(hours, minutes)
    return "${:d}$h".format(hours)

def degEastWestFormatter(deg):
        """Formats degrees as X˚D with X in [0..180] and D in ['E','W']

        Args:
            deg: float
        Returns:
            string
        """
        if deg < -180:
            deg += 360
        if deg > 180:
            deg -= 360

        d = ''
        if deg > 0:
            d = 'E'
        if deg < 0:
            deg *= -1
            d = 'W'
        return degFormatter(deg) + d

def nullFormatter(deg):
    return ""


def _parseArgs(locals):
    """Turn list of arguments (all named or kwargs) into flat dictionary"""
    locals.pop('self')
    kwargs = locals.pop('kwargs', {})
    for k,v in kwargs.items():
        locals[k] = v
    return locals

class Map():
    def __init__(self, proj, ax=None, interactive=True, **kwargs):
        """Create Map with a given projection.

        A `skymapper.Map` holds a predefined projection and `matplotlib` axes
        and figures to enable plotting on the sphere with proper labeling
        and inter/extrapolations.

        It also allows for interactive and exploratory work by updateing the
        maps after pan/zoom events.

        Most of the methods are wrappers of `matplotlib` functions by the same
        names, so that one can mostly interact with a `Map` instance as one
        would do with a `matplotlib.axes`.

        For plotting purposes, it is recommended to switch `interactive` off.

        Args:
            proj: `skymapper.Projection` instance
            ax: `matplotlib.axes` instance, will be created otherwise
            interactive: if pan/zoom is enabled for map updates
            **kwargs: styling of the `matplotlib.patches.Polygon` that shows
                the outline of the map.
        """
        # store arguments to regenerate the map
        self._config = {'__init__': _parseArgs(locals())}
        self.proj = proj
        self._setFigureAx(ax, interactive=interactive)
        self._resolution = 75 # for graticules
        self._setEdge(**kwargs)
        self.ax.relim()
        self.ax.autoscale_view()
        self._setFrame()

    def _setFigureAx(self, ax=None, interactive=True):
        if ax is None:
            self.fig = matplotlib.pyplot.figure(tight_layout=True)
            self.ax = self.fig.add_subplot(111, aspect='equal')
        else:
            self.ax = ax
            self.ax.set_aspect('equal')
            self.fig = self.ax.get_figure()
            self.fig.set_tight_layout(True)

        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.ax.tick_params(axis='both', which='both', length=0)
        self.ax.xaxis.set_ticks_position('none')
        self.ax.yaxis.set_ticks_position('none')

        # attach event handlers
        if interactive:
            self.fig.show()
            self._press_evt = self.fig.canvas.mpl_connect('button_press_event', self._pressHandler)
            self._release_evt = self.fig.canvas.mpl_connect('button_release_event', self._releaseHandler)
            self._scroll_evt = self.fig.canvas.mpl_connect('scroll_event', self._scrollHandler)

    def clone(self, ax=None):
        """Clone map

        Args:
            ax: `matplotlib.axes` instance, will be created otherwise
        Returns:
            New map using the same projections and configuration
        """
        config = dict(self._config)
        config['xlim'] = self.ax.get_xlim()
        config['ylim'] = self.ax.get_ylim()
        return Map._create(config, ax=ax)

    def save(self, filename):
        """Save map configuration to file

        All aspects necessary to reproduce a map are stored in a pickle file.

        Args:
            filename: name for pickle file

        Returns:
            None
        """
        try:
            with open(filename, 'wb') as fp:
                config = dict(self._config)
                config['xlim'] = self.ax.get_xlim()
                config['ylim'] = self.ax.get_ylim()
                pickle.dump(config, fp)
        except IOError as e:
            raise

    @staticmethod
    def load(filename, ax=None):
        """Load map from pickled file

        Args:
            filename: name for pickle file
            ax: `matplotlib.axes` instance, will be created otherwise
        Returns:
            `skymapper.Map`
        """
        try:
            with open(filename, 'rb') as fp:
                config = pickle.load(fp)
                fp.close()
                return Map._create(config, ax=ax)
        except IOError as e:
            raise

    @staticmethod
    def _create(config, ax=None):
        init_args = config.pop('__init__')
        init_args['ax'] = ax
        map = Map(**init_args)

        xlim = config.pop('xlim')
        ylim = config.pop('ylim')
        map.ax.set_xlim(xlim)
        map.ax.set_ylim(ylim)
        map._setFrame()

        meridian_args = config.pop('labelMeridiansAtParallel', {})
        parallel_args = config.pop('labelParallelsAtMeridian', {})
        for method in config.keys():
            getattr(map, method)(**config[method])

        for args in meridian_args.values():
            map.labelMeridiansAtParallel(**args)

        for args in parallel_args.values():
            map.labelParallelsAtMeridian(**args)

        return map

    @property
    def parallels(self):
        """Get the location of the drawn parallels"""
        return [ float(m.group(1)) for c,m in self.artists(r'grid-parallel-([\-\+0-9.]+)', regex=True) ]

    @property
    def meridians(self):
        """Get the location of the drawn meridians"""
        return [ float(m.group(1)) for c,m in self.artists(r'grid-meridian-([\-\+0-9.]+)', regex=True) ]

    def artists(self, gid, regex=False):
        """Get the `matplotlib` artists used in the map

        Args:
            gid: `gid` string of the artist
            regex: if regex matching is done

        Returns:
            list of matching artists
            if `regex==True`, returns list of (artist, match)
        """
        if regex:
            matches = [ re.match(gid, c.get_gid()) if c.get_gid() is not None else None for c in self.ax.get_children() ]
            return [ (c,m) for c,m in zip(self.ax.get_children(), matches) if m is not None ]
        else: # direct match
            return [ c for c in self.ax.get_children() if c.get_gid() is not None and c.get_gid().find(gid) != -1 ]

    def _getParallel(self, p, reverse=False):
        if not reverse:
            return self.proj.transform(self._lon_range, p*np.ones(len(self._lon_range)))
        return self.proj.transform(self._lon_range[::-1], p*np.ones(len(self._lon_range)))

    def _getMeridian(self, m, reverse=False):
        if not reverse:
            return self.proj.transform(m*np.ones(len(self._lat_range)), self._lat_range)
        return self.proj.transform(m*np.ones(len(self._lat_range)), self._lat_range[::-1])

    def _setParallel(self, p, **kwargs):
        x, y = self._getParallel(p)
        artist = Line2D(x, y, **kwargs)
        self.ax.add_line(artist)
        return artist

    def _setMeridian(self, m, **kwargs):
        x, y = self._getMeridian(m)
        artist = Line2D(x, y, **kwargs)
        self.ax.add_line(artist)
        return artist

    def _setEdge(self, **kwargs):
        self._lat_range = np.linspace(-89.9999, 89.9999, self._resolution)
        self._lon_range = np.linspace(-179.9999, 179.9999, self._resolution) + self.proj.lon_0


        # styling: frame needs to be on top of everything, must be transparent
        facecolor = 'None'
        zorder = 1000
        lw = kwargs.pop('lw', 0.7)
        edgecolor = kwargs.pop('edgecolor', 'k')
        # if there is facecolor: clone the polygon and put it in as bottom layer
        facecolor_ = kwargs.pop('facecolor', '#dddddd')

        # polygon of the map edge: top, left, bottom, right
        # don't draw poles if that's a single point
        lines = []
        if not self.proj.poleIsPoint[90]:
            lines.append(self._getParallel(90))
        lines.append(self._getMeridian(self._lon_range[-1], reverse=True))
        if not self.proj.poleIsPoint[-90]:
            lines.append(self._getParallel(-90, reverse=True))
        lines.append(self._getMeridian(self._lon_range[0]))
        xy = np.concatenate(lines, axis=1).T
        self._edge = Polygon(xy, closed=True, edgecolor=edgecolor, facecolor=facecolor, lw=lw, zorder=zorder,gid="edge", **kwargs)
        self.ax.add_patch(self._edge)

        if facecolor_ is not None:
            zorder = -1000
            edgecolor = 'None'
            poly = Polygon(xy, closed=True, edgecolor=edgecolor, facecolor=facecolor_, zorder=zorder, gid="edge-background")
            self.ax.add_patch(poly)

    def contains(self, x, y):
        xy = np.dstack((x,y)).reshape(-1,2)
        return self._edge.get_path().contains_points(xy).reshape(x.shape)

    def xlim(self, left=None, right=None, **kw):
        """Get/set the map limits in x-direction"""
        if left is None and right is None:
            return (self._edge.xy[:, 0].min(), self._edge.xy[:, 0].max())
        else:
            self.ax.set_xlim(left=left, right=right, **kw)
            self._resetFrame()

    def ylim(self, bottom=None, top=None, **kw):
        """Get/set the map limits in x-direction"""
        if bottom is None and top is None:
            return (self._edge.xy[:, 1].min(), self._edge.xy[:, 1].max())
        else:
            self.ax.set_ylim(bottom=bottom, top=top, **kw)
            self._resetFrame()

    def grid(self, sep=30, parallel_fmt=None, meridian_fmt=None, lat_min=-90, lat_max=90, lon_min=-180, lon_max=180, **kwargs):
        """Set map grid / graticules

        Args:
            sep: distance between graticules in deg. if two-element list, distance in lat and long, resp.
            parallel_fmt: formatter for parallel labels
            meridian_fmt: formatter for meridian labels
            lat_min: minimum latitude for graticules
            lat_max: maximum latitude for graticules
            lon_min: minimum longitude for graticules
            lon_max: maximum longitude for graticules
            **kwargs: styling of `matplotlib.lines.Line2D` for the graticules
        """
        try:
            if len(sep)==2:
                sep_lat, sep_lon = sep
        except TypeError:
            sep_lat = sep_lon = sep

        # clone() sets sep_lat, sep_lon, need to get them from kwargs
        # also need to avoid sep being stored in config
        del sep
        sep_lat = kwargs.pop("sep_lat", sep_lat)
        sep_lon = kwargs.pop("sep_lon", sep_lon)

        if parallel_fmt is None:
            parallel_fmt = degPMFormatter
        if meridian_fmt is None:
            if self.proj.lon_type == "ra":
                meridian_fmt = deg360Formatter
            else:
                meridian_fmt = deg180Formatter
        self._config['grid'] = _parseArgs(locals())
        self._lat_range = np.linspace(lat_min, lat_max, self._resolution)
        self._lon_range = np.linspace(lon_min, lon_max, self._resolution) + self.proj.lon_0
        _parallels = np.arange(-90+sep_lat,90,sep_lat)
        if self.proj.lon_0 % sep_lon == 0:
            _meridians = np.arange(sep_lon * ((self.proj.lon_0 + 180) // sep_lon - 1), sep_lon * ((self.proj.lon_0 - 180) // sep_lon), -sep_lon)
        else:
            _meridians = np.arange(sep_lon * ((self.proj.lon_0 + 180) // sep_lon), sep_lon * ((self.proj.lon_0 - 180) // sep_lon), -sep_lon)
        _meridians[_meridians < 0] += 360
        _meridians[_meridians >= 360] -= 360

        # clean up previous grid
        artists = self.artists('grid-meridian') + self.artists('grid-parallel')
        for artist in artists:
                artist.remove()

        # clean up frame meridian and parallel labels because they're tied to the grid
        artists = self.artists('meridian-label') + self.artists('parallel-label')
        for artist in artists:
                artist.remove()

        # styling: based on edge
        ls = kwargs.pop('ls', '-')
        lw = kwargs.pop('lw', self._edge.get_linewidth() / 2)
        c = kwargs.pop('c', self._edge.get_edgecolor())
        alpha = kwargs.pop('alpha', 0.2)
        zorder = kwargs.pop('zorder', self._edge.get_zorder() - 1)

        for p in _parallels:
            self._setParallel(p, gid='grid-parallel-%r' % p, lw=lw, c=c, alpha=alpha, zorder=zorder, **kwargs)
        for m in _meridians:
            self._setMeridian(m, gid='grid-meridian-%r' % m, lw=lw, c=c, alpha=alpha, zorder=zorder, **kwargs)

        # (re)generate the frame labels
        for method in ['labelMeridiansAtFrame', 'labelParallelsAtFrame']:
            if method in self._config.keys():
                getattr(self, method)(**self._config[method])
            else:
                getattr(self, method)()

        # (re)generate edge labels
        for method in ['labelMeridiansAtParallel', 'labelParallelsAtMeridian']:
            if method in self._config.keys():
                args_list = self._config.pop(method, [])
                for args in args_list.values():
                    getattr(self, method)(**args)
            else:
                # label meridians: at the poles if they are not points
                if method == 'labelMeridiansAtParallel':
                    # determine the parallel that has the most space for labels
                    dec = [-90, 0, 90]
                    ra = [self.proj.lon_0,] * 3
                    jac = np.sum(self.proj.gradient(ra, dec)**2, axis=1)
                    p = dec[np.argmax(jac)]
                    getattr(self, method)(p)
                # label both outer meridians
                else:
                    degs = [self.proj.lon_0 + 180, self.proj.lon_0 - 180]
                    for deg in degs:
                        getattr(self, method)(deg)

    def _negateLoc(self, loc):
        if loc == "bottom":
            return "top"
        if loc == "top":
            return "bottom"
        if loc == "left":
            return "right"
        if loc == "right":
            return "left"

    def labelMeridiansAtParallel(self, p, loc=None, meridians=None, pad=None, direction='parallel', fmt=None, 
                                 clear_existing_labels=False, **kwargs):
        """Label the meridians intersecting a given parallel

        The method is called by `grid()` but can be used to overwrite the defaults.

        Args:
            p: parallel in deg
            loc: location of the label with respect to `p`, from `['top', 'bottom']`
            meridians: list of meridians to label, if None labels all of them
            pad: padding of annotation, in units of fontsize
            direction: tangent of the label, from `['parallel', 'meridian']`
            fmt: formatter for labels, if `None` use default
            clear_existing_labels : bool (default=False)
            **kwargs: styling of `matplotlib` annotations for the graticule labels
        """
        arguments = _parseArgs(locals())

        if p in self.proj.poleIsPoint.keys() and self.proj.poleIsPoint[p]:
            return

        myname = 'labelMeridiansAtParallel'
        if myname not in self._config.keys():
            self._config[myname] = dict()

        # remove all existing labels:
        if clear_existing_labels:
            for _p in self._config[myname].keys():
                gid = 'meridian-label-%r' % _p 
                artists = self.artists(gid)
                for artist in artists:
                    artist.remove() 

        # remove exisiting labels at p
        gid = 'meridian-label-%r' % p
        if p in self._config[myname].keys():
            artists = self.artists(gid)
            for artist in artists:
                artist.remove()

        self._config[myname][p] = arguments

        if meridians is None:
            meridians = self.meridians

        # determine rot_base so that central label is upright
        rotation = kwargs.pop('rotation', None)
        if rotation is None or loc is None:
            m = self.proj.lon_0
            dxy = self.proj.gradient(m, p, direction=direction)
            angle = np.arctan2(*dxy) / DEG2RAD
            options = [-90, 90]
            closest = np.argmin(np.abs(options - angle))
            rot_base = options[closest]

            if loc is None:
                if p >= 0:
                    loc = 'top'
                else:
                    loc = 'bottom'
        assert loc in ['top', 'bottom']

        horizontalalignment = kwargs.pop('horizontalalignment', 'center')
        verticalalignment = kwargs.pop('verticalalignment', self._negateLoc(loc))
        zorder = kwargs.pop('zorder', 20)
        size = kwargs.pop('size', matplotlib.rcParams['font.size'])
        # styling consistent with frame, i.e. with edge
        color = kwargs.pop('color', self._edge.get_edgecolor())
        alpha = kwargs.pop('alpha', self._edge.get_alpha())
        zorder = kwargs.pop('zorder', self._edge.get_zorder() + 1) # on top of edge
        if fmt is None:
            fmt = self._config['grid']['meridian_fmt']

        if pad is None:
            pad = size / 3

        for m in meridians:
            # move label along meridian
            xp, yp = self.proj(m, p)
            dxy = self.proj.gradient(m, p, direction="meridian")
            dxy *= pad / np.sqrt((dxy**2).sum())
            if loc == 'bottom': # dxy in positive RA
                dxy *= -1

            if rotation is None:
                dxy_ = self.proj.gradient(m, p, direction=direction)
                angle = rot_base - np.arctan2(*dxy_) / DEG2RAD
            else:
                angle = rotation

            self.ax.annotate(fmt(m), (xp, yp), xytext=dxy, textcoords='offset points', rotation=angle, rotation_mode='anchor', horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, size=size, color=color, alpha=alpha, zorder=zorder, gid=gid, in_layout=False, **kwargs)

    def labelParallelsAtMeridian(self, m, loc=None, parallels=None, pad=None, direction='parallel', fmt=None, 
                                 clear_existing_labels=False, **kwargs):
        """Label the parallel intersecting a given meridian

        The method is called by `grid()` but can be used to overwrite the defaults.

        Args:
            m: meridian in deg
            loc: location of the label with respect to `m`, from `['left', 'right']`
            parallel: list of parallels to label, if None labels all of them
            pad: padding of annotation, in units of fontsize
            direction: tangent of the label, from `['parallel', 'meridian']`
            fmt: formatter for label, if `None` use default
            clear_existing_labels : bool (default=False)
            **kwargs: styling of `matplotlib` annotations for the graticule labels
        """
        arguments = _parseArgs(locals())

        myname = 'labelParallelsAtMeridian'
        if myname not in self._config.keys():
            self._config[myname] = dict()

        # remove all existing labels
        if clear_existing_labels:
            for _m in self._config[myname].keys():
                gid = 'parallel-label-%r' % _m
                artists = self.artists(gid)
                for artist in artists:
                    artist.remove() 

        # remove exisiting labels at m
        gid = 'parallel-label-%r' % m
        if m in self._config[myname].keys():
            artists = self.artists(gid)
            for artist in artists:
                artist.remove()

        self._config[myname][m] = arguments

        # determine rot_base so that central label is upright
        rotation = kwargs.pop('rotation', None)
        if rotation is None or loc is None:
            p = 0
            dxy = self.proj.gradient(m, p, direction=direction)
            angle = np.arctan2(*dxy) / DEG2RAD
            options = [-90, 90]
            closest = np.argmin(np.abs(options - angle))
            rot_base = options[closest]

            if loc is None:
                if m < self.proj.lon_0: # meridians on the left: dx goes in positive RA
                    dxy *= -1
                if dxy[0] > 0:
                    loc = 'right'
                else:
                    loc = 'left'
        assert loc in ['left', 'right']

        if parallels is None:
            parallels = self.parallels

        horizontalalignment = kwargs.pop('horizontalalignment', self._negateLoc(loc))
        verticalalignment = kwargs.pop('verticalalignment', 'center')
        zorder = kwargs.pop('zorder', 20)
        size = kwargs.pop('size', matplotlib.rcParams['font.size'])
        # styling consistent with frame, i.e. with edge
        color = kwargs.pop('color', self._edge.get_edgecolor())
        alpha = kwargs.pop('alpha', self._edge.get_alpha())
        zorder = kwargs.pop('zorder', self._edge.get_zorder() + 1) # on top of edge
        if fmt is None:
            fmt = self._config['grid']['parallel_fmt']

        if pad is None:
            pad = size/2 # more space for horizontal parallels

        for p in parallels:
            # move label along parallel
            xp, yp = self.proj(m, p)
            dxy = self.proj.gradient(m, p, direction="parallel")
            dxy *= pad / np.sqrt((dxy**2).sum())
            if m < self.proj.lon_0: # meridians on the left: dx goes in positive RA
                dxy *= -1

            if rotation is None:
                dxy_ = self.proj.gradient(m, p, direction=direction)
                angle = rot_base - np.arctan2(*dxy_) / DEG2RAD
            else:
                angle = rotation

            self.ax.annotate(fmt(p), (xp, yp), xytext=dxy, textcoords='offset points', rotation=angle, rotation_mode='anchor',  horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, size=size, color=color, alpha=alpha, zorder=zorder, gid=gid, in_layout=False, **kwargs)


    def labelMeridiansAtFrame(self, loc='auto', meridians=None, pad=None, **kwargs):
        """Label the meridians on rectangular frame of the map

        If the view only shows a fraction of the map, a segment or an entire
        rectangular frame is shown and the graticule labels are moved to outside
        that frame. This method is implicitly called, but can be used to overwrite
        the defaults.

        Args:
            loc: location of the label with respect to frame, from `['top', 'bottom']`
            meridians: list of meridians to label, if None labels all of them
            pad: padding of annotation, in units of fontsize
            **kwargs: styling of `matplotlib` annotations for the graticule labels
        """
        assert loc in [None, 'none', 'auto', 'bottom', 'top']

        arguments = _parseArgs(locals())
        myname = 'labelMeridiansAtFrame'
        self._config[myname] = arguments

        # remove existing
        self.ax.xaxis.set_visible(False)
        frame_artists = self.artists('frame-meridian-label')
        for artist in frame_artists:
            artist.remove()

        # check if loc has frame, check both frames for auto
        locs = {"bottom": 0, "top": 1}
        frame_artists = self.artists(r'frame-([a-zA-Z]+)', regex=True)
        frame_locs = [match.group(1) for c,match in frame_artists]
        if loc == "auto":
            loc = None
            xdelta = 0
            for c,match in frame_artists:
                if match.group(1) in locs.keys():
                    xdata = c.get_xdata()
                    xdelta_ = xdata[-1] - xdata[0]
                    if xdelta_ > xdelta:
                        xdelta = xdelta_
                        loc = match.group(1)
                    elif xdelta_ == xdelta: # top and bottom frame equally large
                        # pick the largest meridian gradient in the middle of the frame
                        xmean = xdata.mean()
                        ylim = self.ax.get_ylim()
                        m_bottom, p_ = self.proj.inv(xmean, ylim[0])
                        grad_bottom = np.abs(self.proj.gradient(m_bottom, p_, direction="parallel")[0])
                        m_top, p_ = self.proj.inv(xmean, ylim[1])
                        grad_top = np.abs(self.proj.gradient(m_top, p_, direction="parallel")[0])
                        if grad_top > grad_bottom:
                            loc = 'top'
                        else:
                            loc = 'bottom'
        if loc not in frame_locs:
            return

        size = kwargs.pop('size', matplotlib.rcParams['font.size'])
        # styling consistent with frame, i.e. with edge
        color = kwargs.pop('color', self._edge.get_edgecolor())
        alpha = kwargs.pop('alpha', self._edge.get_alpha())
        zorder = kwargs.pop('zorder', self._edge.get_zorder() + 1) # on top of edge
        horizontalalignment = kwargs.pop('horizontalalignment', 'center')
        verticalalignment = self._negateLoc(loc) # no option along the frame
        _ = kwargs.pop('verticalalignment', None)

        if pad is None:
            pad = size / 3
        pad /= 72

        if meridians is None:
            meridians = self.meridians

        # find all meridian grid lines
        m_artists = self.artists(r'grid-meridian-([\-\+0-9.]+)', regex=True)
        pos = locs[loc]
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xticks = []
        xticklabels = []
        for c,match in m_artists:
            m = float(match.group(1))
            if m in meridians:
                # intersect with axis
                xm, ym = c.get_xdata(), c.get_ydata()
                xm_at_ylim = extrap(ylim, ym, xm)[pos]
                if xm_at_ylim >= xlim[0] and xm_at_ylim <= xlim[1] and self.contains(xm_at_ylim, ylim[pos]):
                    # compute the offset and the location of the label by following the meridian tangent
                    m_, p_ = self.proj.inv(xm_at_ylim, ylim[pos])
                    dxy = self.proj.gradient(m_, p_, direction="meridian")
                    dxy *= pad / np.sqrt((dxy**2).sum())
                    if loc == "bottom":
                        dxy *= -1
                    dxy[1] = 0 # only offset in x

                    # move the position of the label by dx:
                    # go from data coordinates to display, offset in display, go back to data
                    offset = matplotlib.transforms.ScaledTranslation(*dxy, self.fig.dpi_scale_trans)
                    data_offset = self.ax.transData + offset + self.ax.transData.inverted()
                    xy_at_ylim = data_offset.transform((xm_at_ylim, ylim[pos]))
                    xticks.append(xy_at_ylim[0])
                    xticklabels.append(self._config['grid']['meridian_fmt'](m))

        if len(xticks):
            self.ax.xaxis.set_visible(True)
            self.ax.xaxis.set_ticks_position(loc)
            self.ax.xaxis.set_label_position(loc)
            self.ax.xaxis.set_ticks(xticks)
            self.ax.xaxis.set_ticklabels(xticklabels)


    def labelParallelsAtFrame(self, loc='auto', parallels=None, pad=None, **kwargs):
        """Label the parallels on rectangular frame of the map

        If the view only shows a fraction of the map, a segment or an entire
        rectangular frame is shown and the graticule labels are moved to outside
        that frame. This method is implicitly called, but can be used to overwrite
        the defaults.

        Args:
            loc: location of the label with respect to frame, from `['left', 'right']`
            parallels: list of parallels to label, if None labels all of them
            pad: padding of annotation, in units of fontsize
            **kwargs: styling of `matplotlib` annotations for the graticule labels
        """
        assert loc in [None, 'none', 'auto', 'left', 'right']

        arguments = _parseArgs(locals())
        myname = 'labelParallelsAtFrame'
        self._config[myname] = arguments

        # remove existing
        self.ax.yaxis.set_visible(False)
        frame_artists = self.artists('frame-parallel-label')
        for artist in frame_artists:
            artist.remove()

        # check if loc has frame, check both frames for auto
        locs = {"left": 0, "right": 1}
        frame_artists = self.artists(r'frame-([a-zA-Z]+)', regex=True)
        frame_locs = [match.group(1) for c,match in frame_artists]
        if loc == "auto":
            loc = None
            ydelta = 0
            for c,match in frame_artists:
                if match.group(1) in locs.keys():
                    ydata = c.get_ydata()
                    ydelta_ = ydata[-1] - ydata[0]
                    if ydelta_ > ydelta:
                        ydelta = ydelta_
                        loc = match.group(1)
                    elif ydelta_ == ydelta: # top and bottom frame equally large
                        # pick the largest meridian gradient in the middle of the frame
                        ymean = ydata.mean()
                        xlim = self.ax.get_xlim()
                        m_, p_left = self.proj.inv(xlim[0], ymean)
                        grad_left = np.abs(self.proj.gradient(m_, p_left, direction="meridian")[1])
                        m_, p_right = self.proj.inv(xlim[1], ymean)
                        grad_right = np.abs(self.proj.gradient(m_, p_right, direction="meridian")[1])
                        if grad_right > grad_left:
                            loc = 'right'
                        else:
                            loc = 'left'

        if loc not in frame_locs:
            return

        size = kwargs.pop('size', matplotlib.rcParams['font.size'])
        # styling consistent with frame, i.e. with edge
        color = kwargs.pop('color', self._edge.get_edgecolor())
        alpha = kwargs.pop('alpha', self._edge.get_alpha())
        zorder = kwargs.pop('zorder', self._edge.get_zorder() + 1) # on top of edge
        verticalalignment = kwargs.pop('verticalalignment', 'center')
        horizontalalignment = self._negateLoc(loc) # no option along the frame
        _ = kwargs.pop('horizontalalignment', None)

        if pad is None:
            pad = size / 3
        pad /= 72 # in inch

        if parallels is None:
            parallels = self.parallels

        # find all parallel grid lines
        p_artists = self.artists(r'grid-parallel-([\-\+0-9.]+)', regex=True)
        pos = locs[loc]
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        yticks = []
        yticklabels = []
        for c,match in p_artists:
            p = float(match.group(1))
            if p in parallels:
                # intersect with axis
                xp, yp = c.get_xdata(), c.get_ydata()
                yp_at_xlim = extrap(xlim, xp, yp)[pos]
                if yp_at_xlim >= ylim[0] and yp_at_xlim <= ylim[1] and self.contains(xlim[pos], yp_at_xlim):
                    m_, p_ = self.proj.inv(xlim[pos], yp_at_xlim)
                    dxy = self.proj.gradient(m_, p_, direction='parallel')
                    dxy *= pad / np.sqrt((dxy**2).sum())
                    if (self.proj.lon_type == "ra" and loc == "right") or (self.proj.lon_type != "ra" and loc == "left"):
                        dxy *= -1
                    dxy[0] = 0 # only offset in y

                    # move the position of the label by dy:
                    # go from data coordinates to display, offset in display, go back to data
                    offset = matplotlib.transforms.ScaledTranslation(*dxy, self.fig.dpi_scale_trans)
                    data_offset = self.ax.transData + offset + self.ax.transData.inverted()
                    xy_at_xlim = data_offset.transform((xlim[pos], yp_at_xlim))
                    yticks.append(xy_at_xlim[1])
                    yticklabels.append(self._config['grid']['parallel_fmt'](p))

        if len(yticks):
            self.ax.yaxis.set_visible(True)
            self.ax.yaxis.set_ticks_position(loc)
            self.ax.yaxis.set_label_position(loc)
            self.ax.yaxis.set_ticks(yticks)
            self.ax.yaxis.set_ticklabels(yticklabels)


    def _setFrame(self):
        # clean up existing frame
        frame_artists = self.artists(r'frame-([a-zA-Z]+)', regex=True)
        for c,m in frame_artists:
            c.remove()

        locs = ['left', 'bottom', 'right', 'top']
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()

        # use styling of edge for consistent map borders
        ls = '-'
        lw = self._edge.get_linewidth()
        c = self._edge.get_edgecolor()
        alpha = self._edge.get_alpha()
        zorder = self._edge.get_zorder() - 1 # limits imprecise, hide underneath edge

        precision = 1000
        const = np.ones(precision)
        for loc in locs:
            # define line along axis
            if loc == "left":
                line = xlim[0]*const, np.linspace(ylim[0], ylim[1], precision)
            if loc == "right":
                line = xlim[1]*const, np.linspace(ylim[0], ylim[1], precision)
            if loc == "bottom":
                line = np.linspace(xlim[0], xlim[1], precision), ylim[0]*const
            if loc == "top":
                line = np.linspace(xlim[0], xlim[1], precision), ylim[1]*const

            # show axis lines only where line is inside of map edge
            inside = self.contains(*line)
            if (~inside).all():
                continue

            if inside.all():
                startpos, stoppos = 0, -1
                xmin = (line[0][startpos] - xlim[0])/(xlim[1]-xlim[0])
                ymin = (line[1][startpos] - ylim[0])/(ylim[1]-ylim[0])
                xmax = (line[0][stoppos] - xlim[0])/(xlim[1]-xlim[0])
                ymax = (line[1][stoppos] - ylim[0])/(ylim[1]-ylim[0])
                self.ax.plot([xmin,xmax], [ymin, ymax], c=c, ls=ls, lw=lw, alpha=alpha, zorder=zorder, clip_on=False, transform=self.ax.transAxes, gid='frame-%s' % loc)
                continue

            # for piecewise inside: determine limits where it's inside
            # by checking for jumps in inside
            inside = inside.astype("int")
            diff = inside[1:] - inside[:-1]
            jump = np.flatnonzero(diff)
            start = 0
            if inside[0]:
                jump = np.concatenate(((0,),jump))

            while True:
                startpos = jump[start]
                if start+1 < len(jump):
                    stoppos = jump[start + 1]
                else:
                    stoppos = -1

                xmin = (line[0][startpos] - xlim[0])/(xlim[1]-xlim[0])
                ymin = (line[1][startpos] - ylim[0])/(ylim[1]-ylim[0])
                xmax = (line[0][stoppos] - xlim[0])/(xlim[1]-xlim[0])
                ymax = (line[1][stoppos] - ylim[0])/(ylim[1]-ylim[0])
                artist = Line2D([xmin,xmax], [ymin, ymax], c=c, ls=ls, lw=lw, alpha=alpha, zorder=zorder, clip_on=False, transform=self.ax.transAxes, gid='frame-%s' % loc)
                self.ax.add_line(artist)
                if start + 2 < len(jump):
                    start += 2
                else:
                    break

    def _clearFrame(self):
        frame_artists = self.artists('frame-')
        for artist in frame_artists:
            artist.remove()

    def _resetFrame(self):
        self._setFrame()
        for method in ['labelMeridiansAtFrame', 'labelParallelsAtFrame']:
            if method in self._config.keys():
                getattr(self, method)(**self._config[method])

    def _pressHandler(self, evt):
        if evt.button != 1: return
        if evt.dblclick: return
        # remove frame and labels
        self._clearFrame()
        self.fig.canvas.draw()

    def _releaseHandler(self, evt):
        if evt.button != 1: return
        if evt.dblclick: return
        self._resetFrame()
        self.fig.canvas.draw()

    def _scrollHandler(self, evt):
        # mouse scroll for zoom
        if evt.inaxes != self.ax: return
        if evt.step == 0: return

        # remove frame and labels
        self._clearFrame()

        # scroll to fixed pointer position: google maps style
        factor = 0.1
        c = 1 - evt.step*factor # scaling factor
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xdiff, ydiff = xlim[1] - xlim[0], ylim[1] - ylim[0]
        x, y = evt.xdata, evt.ydata
        fx, fy = (x - xlim[0])/xdiff, (y - ylim[0])/ydiff # axis units
        xlim_, ylim_ = x - fx*c*xdiff, y - fy*c*ydiff
        xlim__, ylim__ = xlim_ + c*xdiff, ylim_ + c*ydiff

        self.ax.set_xlim(xlim_, xlim__)
        self.ax.set_ylim(ylim_, ylim__)
        self._resetFrame()
        self.fig.canvas.draw()

    #### common plot type for maps: follow mpl convention ####
    def plot(self, lon, lat, *args, **kwargs):
        """Matplotlib `plot` with `lon/lat` points transformed according to map projection"""
        x, y = self.proj.transform(lon, lat)
        return self.ax.plot(x, y, *args, **kwargs)

    def scatter(self, lon, lat, **kwargs):
        """Matplotlib `scatter` with `lon/lat` points transformed according to map projection"""
        x, y = self.proj.transform(lon, lat)
        return self.ax.scatter(x, y, **kwargs)

    def hexbin(self, lon, lat, C=None, **kwargs):
        """Matplotlib `hexbin` with `lon/lat` points transformed according to map projection"""
        x, y = self.proj.transform(lon, lat)
        # determine proper gridsize: by default x is only needed, y is chosen accordingly
        gridsize = kwargs.pop("gridsize", None)
        mincnt = kwargs.pop("mincnt", 1)
        clip_path = kwargs.pop('clip_path', self._edge)
        if gridsize is None:
            xlim, ylim = (x.min(), x.max()), (y.min(), y.max())
            per_sample_volume = (xlim[1]-xlim[0])**2 / x.size * 10
            gridsize = (int(np.round((xlim[1]-xlim[0]) / np.sqrt(per_sample_volume))),
                        int(np.round((ylim[1]-ylim[0]) / np.sqrt(per_sample_volume)) / np.sqrt(3)))

        # styling: use same default colormap as density for histogram
        if C is None:
            cmap = kwargs.pop("cmap", "YlOrRd")
        else:
            cmap = kwargs.pop("cmap", None)
        zorder = kwargs.pop("zorder", 0) # same as for imshow: underneath everything

        artist = self.ax.hexbin(x, y, C=C, gridsize=gridsize, mincnt=mincnt, cmap=cmap, zorder=zorder, **kwargs)
        artist.set_clip_path(clip_path)
        return artist

    def text(self, lon, lat, s, rotation=None, direction="parallel", **kwargs):
        """Matplotlib `text` with coordinates given by `lon/lat`.

        Args:
            lon: longitude of text
            lat: latitude of text
            s: string
            rotation: if text should be rotated to tangent direction
            direction: tangent direction, from `['parallel', 'meridian']`
            **kwargs: styling arguments for `matplotlib.text`
        """
        x, y = self.proj(lon, lat)

        if rotation is None:
            dxy_ = self.proj.gradient(lon, lat, direction=direction)
            if self.proj.lon_type == "ra":
                dxy_[0] *= -1
            angle = 90-np.arctan2(*dxy_) / DEG2RAD
        else:
            angle = rotation

        return self.ax.text(x, y, s, rotation=angle, rotation_mode="anchor", clip_on=True, **kwargs)

    def colorbar(self, cb_collection, cb_label="", loc="right", size="2%", pad="1%"):
        """Add colorbar to side of map.

        The location of the colorbar will be chosen automatically to not interfere
        with the map frame labels.

        Args:
            cb_collection: a `matplotlib` mappable collection
            cb_label: string for colorbar label
            orientation: from ["vertical", "horizontal"]
            size: fraction of ax size to use for colorbar
            pad: fraction of ax size to use as pad to map frame
        """
        assert loc in ["top", "bottom", "left", "right"]

        # move frame ticks to other side: colorbar side is taken
        if loc in["top", "bottom"]:
            orientation = "horizontal"
            params = self._config['labelMeridiansAtFrame']
            params['loc'] = self._negateLoc(loc)
            self.labelMeridiansAtFrame(**params)
        if loc in["left", "right"]:
            orientation = "vertical"
            params = self._config['labelParallelsAtFrame']
            params['loc'] = self._negateLoc(loc)
            self.labelParallelsAtFrame(**params)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes(loc, size=size, pad=pad)
        cb = self.fig.colorbar(cb_collection, cax=cax, orientation=orientation, ticklocation=loc)
        cb.solids.set_edgecolor("face")
        cb.set_label(cb_label)
        return cb

    def title(self, label, **kwargs):
        return self.ax.set_title(label, **kwargs)

    def focus(self, lon, lat, pad=0.025):
        """Focus onto region of map covered by `lon/lat`

        Adjusts x/y limits to encompass given `lon/lat`.

        Args:
            lon: list of longitudes
            lat: list of latitudes
            pad: distance to edge of the frame, in units of axis size
        """
        # to replace the autoscale function that cannot zoom in
        x, y = self.proj.transform(lon, lat)
        xlim = [x.min(), x.max()]
        ylim = [y.min(), y.max()]
        xrange = xlim[1]-xlim[0]
        yrange = ylim[1]-ylim[0]
        xlim[0] -= pad * xrange
        xlim[1] += pad * xrange
        ylim[0] -= pad * yrange
        ylim[1] += pad * yrange
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self._resetFrame()
        self.fig.canvas.draw()

    def defocus(self, pad=0.025):
        """Show entire map.

        Args:
            pad: distance to edge of the map, in units of axis size
        """
        # to replace the autoscale function that cannot zoom in
        xlim, ylim = list(self.xlim()), list(self.ylim())
        xrange = xlim[1]-xlim[0]
        yrange = ylim[1]-ylim[0]
        xlim[0] -= pad * xrange
        xlim[1] += pad * xrange
        ylim[0] -= pad * yrange
        ylim[1] += pad * yrange
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self._clearFrame()
        self.fig.canvas.draw()

    def show(self, *args, **kwargs):
        """Show `matplotlib` figure"""
        self.fig.show(*args, **kwargs)

    def savefig(self, *args, **kwargs):
        """Save `matplotlib` figure"""
        self.fig.savefig(*args, **kwargs)

    #### special plot types for maps ####

    def vertex(self, vertices, color=None, vmin=None, vmax=None, **kwargs):
        """Plot polygons (e.g. Healpix vertices)

        Args:
            vertices: cell boundaries in RA/Dec, from getCountAtLocations()
            color: string or matplib color, or numeric array to set polygon colors
            vmin: if color is numeric array, use vmin to set color of minimum
            vmax: if color is numeric array, use vmin to set color of minimum
            **kwargs: matplotlib.collections.PolyCollection keywords
        Returns:
            matplotlib.collections.PolyCollection
        """
        vertices_ = np.empty_like(vertices)
        vertices_[:,:,0], vertices_[:,:,1] = self.proj.transform(vertices[:,:,0], vertices[:,:,1])

        # remove vertices which are split at the outer meridians
        # find variance of vertice nodes large compared to dispersion of centers
        centers = np.mean(vertices, axis=1)
        x, y = self.proj.transform(centers[:,0], centers[:,1])
        var = np.sum(np.var(vertices_, axis=1), axis=-1) / (x.var() + y.var())
        sel = var < 0.05
        vertices_ = vertices_[sel]

        from matplotlib.collections import PolyCollection
        zorder = kwargs.pop("zorder", 0) # same as for imshow: underneath everything
        rasterized = kwargs.pop('rasterized', True)
        alpha = kwargs.pop('alpha', 1)
        if alpha < 1:
            lw = kwargs.pop('lw', 0)
        else:
            lw = kwargs.pop('lw', None)
        coll = PolyCollection(vertices_, zorder=zorder, rasterized=rasterized, alpha=alpha, lw=lw, **kwargs)
        if color is not None:
            coll.set_array(color[sel])
            coll.set_clim(vmin=vmin, vmax=vmax)
        coll.set_edgecolor("face")
        self.ax.add_collection(coll)
        self.ax.set_rasterization_zorder(zorder)
        return coll

    def healpix(self, m, nest=False, color_percentiles=[10,90], **kwargs):
        """Plot HealPix map

        Args:
            m: Healpix map array
            nest: HealPix NEST scheme
            color_percentiles: lower and higher cutoff percentile for map coloring
            **kwargs: matplotlib.imshow keywords
        """

        # determine lon, lat for all pixels visible in the map
        clip_path = kwargs.pop('clip_path', self._edge)
        if clip_path is None:
            xlim, ylim = self.xlim(), self.ylim()
        else:
            xlim = clip_path.xy[:, 0].min(), clip_path.xy[:, 0].max()
            ylim = clip_path.xy[:, 1].min(), clip_path.xy[:, 1].max()
        # determine number of pixel in xy limits
        x_pixel_min, y_pixel_min = self.ax.transData.transform((xlim[0], ylim[0]))
        x_pixel_max, y_pixel_max = self.ax.transData.transform((xlim[1], ylim[1]))
        width, height = int(math.ceil(x_pixel_max - x_pixel_min)), int(math.ceil(y_pixel_max - y_pixel_min))
        # make a grid in xy coordinates with the determined number of pixels
        xline = np.linspace(xlim[0], xlim[1], width)
        yline = np.linspace(ylim[0], ylim[1], height)
        xp, yp = np.meshgrid(xline, yline)
        # compute coordinate for the pixels that are inside map
        inside = self.contains(xp, yp)
        xp = np.ma.array(xp, mask=~inside)
        yp = np.ma.array(yp, mask=~inside)
        lonp, latp = self.proj.inv(xp, yp)

        # determine healpix ipix for all visible and valid map pixels
        valid = np.isfinite(lonp.data) & np.isfinite(latp.data)
        inside &= valid
        nside = healpix.hp.npix2nside(m.size)
        ipix = healpix.hp.ang2pix(nside, lonp[inside], latp[inside], lonlat=True, nest=nest) # use lonlat=True!

        # copy values from m into vp
        # but check if m is a masked array to avoid plotting masked values
        vp = np.ma.array(np.empty(xp.shape), mask=~inside)
        if np.__version__ >= "1.1":
            matype = np.ma.core.MaskedArray
        else:
            matype = np.ma.array
        if not isinstance(m, matype):
            vp.data[inside] = m[ipix]
        else:
            vp.data[inside] = m.data[ipix]
            vp.mask[inside] |= m.mask[ipix]

        # color range
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        if vmin is None or vmax is None:
            vlim = np.percentile(vp.data[inside], color_percentiles)
            if vmin is None:
                vmin = vlim[0]
            if vmax is None:
                vmax = vlim[1]

        # show the vp image
        cmap = kwargs.pop("cmap", "YlOrRd")
        zorder = kwargs.pop("zorder", 0) # same as for imshow: underneath everything
        xlim_, ylim_ = self.ax.get_xlim(), self.ax.get_ylim()
        artist = self.ax.imshow(vp, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), vmin=vmin, vmax=vmax, cmap=cmap, zorder=zorder, **kwargs)
        artist.set_clip_path(clip_path)
        # ... because imshow focusses on extent
        self.ax.set_xlim(xlim_)
        self.ax.set_ylim(ylim_)
        return artist

    def footprint(self, survey, nside, **kwargs):
        """Plot survey footprint onto map

        Uses `contains()` method of a `skymapper.Survey` derived class instance

        Args:
            survey: name of the survey, must be in keys of `skymapper.survey_register`
            nside: HealPix nside
            **kwargs: styling of `matplotlib.collections.PolyCollection`
        """

        pixels, rap, decp, vertices = healpix.getGrid(nside, return_vertices=True)
        inside = survey.contains(rap, decp)
        return self.vertex(vertices[inside], **kwargs)

    def density(self, lon, lat, nside=1024, **kwargs):
        """Plot sample density using healpix binning

        Args:
            lon: list of longitudes
            lat: list of latitudes
            nside: HealPix nside
            **kwargs: additional arguments for healpix()
        """
        # get count in healpix cells, restrict to non-empty cells
        bc = healpix.getCountAtLocations(lon, lat, nside=nside)
        bc = np.ma.array(bc, mask=(bc==0))

        # styling
        cmap = kwargs.pop("cmap", "YlOrRd")

        # make map
        return self.healpix(bc, cmap=cmap, **kwargs)

    def extrapolate(self, lon, lat, value, resolution=100, **kwargs):
        """Extrapolate lon,lat,value samples on the entire sphere

        Requires scipy, uses `scipy.interpolate.Rbf`.

        Args:
            lon: list of longitudes
            lat: list of latitudes
            value: list of sample values
            resolution: number of evaluated cells per linear map dimension
            **kwargs: arguments for matplotlib.imshow
        """
        # interpolate samples in lon/lat
        # this should be evaluated with angular distances, but that's very slow and has unstable extrapolation
        # we use Euclidean norm instead
        rbfi = scipy.interpolate.Rbf(lon, lat, value) # , norm=angularDistance)

        # make grid in x/y over the limits of the map or the clip_path
        clip_path = kwargs.pop('clip_path', self._edge)
        if clip_path is None:
            xlim, ylim = self.xlim(), self.ylim()
        else:
            xlim = clip_path.xy[:, 0].min(), clip_path.xy[:, 0].max()
            ylim = clip_path.xy[:, 1].min(), clip_path.xy[:, 1].max()

        if resolution % 1 == 0:
            resolution += 1

        dx = (xlim[1]-xlim[0])/resolution
        xline = np.arange(xlim[0], xlim[1], dx)
        yline = np.arange(ylim[0], ylim[1], dx)
        xp, yp = np.meshgrid(xline, yline) + dx/2 # evaluate center pixel
        inside = self.contains(xp,yp)
        xp = np.ma.array(xp, mask=~inside)
        yp = np.ma.array(yp, mask=~inside)
        lonp, latp = self.proj.inv(xp, yp)

        # evaluate RBF for all visible and valid map pixels
        valid = np.isfinite(lonp.data) & np.isfinite(latp.data)
        inside &= valid
        vp = np.ma.array(np.empty(xp.shape), mask=~inside)
        vp[inside] = rbfi(lonp[inside], latp[inside])

        # xp,yp whose centers aren't in the map have no values:
        # edges of map are not clean
        # construct another rbf in pixel space to populate the values
        # outside of the map region, then use clip-path to clip them at the edges
        # ordinary Euclidean distance now
        rbfi = scipy.interpolate.Rbf(xp[inside], yp[inside], vp[inside])
        vp[~inside] = rbfi(xp[~inside], yp[~inside])

        zorder = kwargs.pop("zorder", 0) # same as for imshow: underneath everything
        xlim_, ylim_ = self.ax.get_xlim(), self.ax.get_ylim()
        artist = self.ax.imshow(vp, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), zorder=zorder, **kwargs)
        artist.set_clip_path(clip_path)
        # ... because imshow focusses on extent
        self.ax.set_xlim(xlim_)
        self.ax.set_ylim(ylim_)
        return artist
