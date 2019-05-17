import ipyvolume as ipv
import ipyvolume.styles
import numpy as np
import colorsys

axis_off = ['light', 'minimal']


def volshow(volume):
    """
    Use jupyter notebook to visualize 3d volumes.
    :param volume: 3d numpy array
    :return:
    """
    import scipy.ndimage
    sampled = scipy.ndimage.zoom(volume, 0.3, order=2)

    fig = ipv.quickvolshow(sampled, level=[0.24, 0.29], opacity=[0.09, 0.14])
    ipv.pylab.style.use(axis_off)
    return fig


def scatter(coords, color='gray'):
    """
    Use jupyter notebook to visualize point clouds.
    :param coords: [N, 3] numpy array with x, y, z coordinates
    :param color: color of points as string, e.g. 'red', '#ff0000' or 'rgb(1, 0, 0)'
    :return:
    """
    xs = coords[:, 0].astype(np.float)
    ys = coords[:, 1].astype(np.float)
    zs = coords[:, 2].astype(np.float)

    fig = ipv.quickscatter(xs, ys, zs, size=0.1, marker='sphere', color=color)
    ipv.pylab.style.use(axis_off)
    return fig


def scatter_with_target(coords, target, color=(0.5, 0.5, 0.5), color_label=(1, 0, 0)):
    """
    Use jupyter notebook to visualize point clouds with an annotated item.
    :param coords: [N, 3] numpy array with x, y, z coordinates
    :param target: [N] binary numpy array with 0 for background and 1 for label
    :param color: color of points as rgb tuple, e.g. (1, 0, 0) for red
    :param color_label: color of labelled points as rgb tuple, e.g. (1, 0, 0) for red
    :return:
    """
    xs = coords[:, 0].astype(np.float)
    ys = coords[:, 1].astype(np.float)
    zs = coords[:, 2].astype(np.float)

    c = np.full_like(target, color, dtype=(float, 3))
    c[np.argwhere(target == 1)] = color_label

    fig = ipv.quickscatter(xs, ys, zs, size=0.1, marker='sphere', color=c)
    ipv.pylab.style.use(axis_off)
    ipv.pylab.view(200, 80)
    return fig


def scatter_with_intensities(coords, intensities):
    """
    Use jupyter notebook to visualize point clouds.
    :param coords: [N, 3] numpy array with x, y, z coordinates
    :param intensities [N] numpy array with corresponding intensities
    :return:
    """
    xs = coords[:, 0].astype(np.float)
    ys = coords[:, 1].astype(np.float)
    zs = coords[:, 2].astype(np.float)

    cs = list(map(lambda i: (1, 1 - i, 0), intensities))

    fig = ipv.quickscatter(xs, ys, zs, size=0.1, marker='sphere', color=cs)
    ipv.pylab.style.use(axis_off)
    ipv.pylab.view(200, 80)
    return fig
