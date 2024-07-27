import skymapper as skm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # cycle through all defined projections and show the full sky
    # with default graticules
    args = {"lon_0": 0}
    conic_args = {"lon_0":0,
                "lat_0": -10,
                "lat_1": -40,
                "lat_2": 10
                }

    for name, proj_cls in skm.projection_register.items():
        proj = None
        try:
            proj = proj_cls(**args)
        except TypeError:
            try:
                proj = proj_cls(**conic_args)
            except TypeError:
                pass

        if proj is not None:
            map = skm.Map(proj, interactive=False)
            map.grid()
            map.title(name)

        plt.show()
