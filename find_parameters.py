from fish_spot_find_photutils import (
    photutils_daostarfinder, process_whole_slide
)
import napari
import skimage.io
import itertools
from joblib import Parallel, delayed
import numpy as np


def load_channel_of_image(img_path, channel=0):
    try:
        img = skimage.io.imread(img_path, key=channel)
    except IndexError:
        img = skimage.io.imread(img_path, key=0)
        img = _normalize_img_channel(img)
        img = img[channel]
    return img

def try_small_image(
    img,
    min_threshold,
    max_threshold,
    num_of_steps_threshold,
    min_fwhm, 
    max_fwhm,
    num_of_steps_fwhm
):
    thresholds = np.linspace(
        min_threshold, max_threshold, num_of_steps_threshold
    )
    fwhms = np.linspace(
        min_fwhm, max_fwhm, num_of_steps_fwhm
    )
    combinations = list(itertools.product(fwhms, thresholds))
    results = Parallel(n_jobs=-3, verbose=0)(
        delayed(photutils_daostarfinder)(img, [t], [f]) 
        for f, t in combinations
    )
    with napari.gui_qt():
        viewer = napari.Viewer()

        viewer.add_image(
            img, contrast_limits=[0, np.percentile(img, 99.5)],
            blending='additive', interpolation='nearest', colormap='magma'
        )
        for idx, r in enumerate(results):
            if not r.empty:
                f, t = combinations[idx]
                name = 't={} f={}'.format(t, f)
                data = np.array([r.ycentroid, r.xcentroid]).T

                viewer.add_points(
                    data=data, 
                    name=name,
                    properties={'sharpness': r.sharpness > 0.5},
                    face_color='#ffffff00',
                    edge_width=4, 
                    edge_color='sharpness',
                    edge_color_cycle=['#d8b365', '#5ab4ac'],
                    size=4/r.sharpness,
                    opacity=0.5,
                    visible=True if idx == len(results) - 1 else False,
                )

def try_full_image(
    img,
    thresholds,
    fwhms,
    additional_imgs=None
):
    result, _ = process_whole_slide(img, thresholds, fwhms)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(
            img, contrast_limits=[0, np.percentile(img, 99.5)],
            blending='additive', interpolation='nearest', colormap='magma'
        )
        if additional_imgs is not None:
            for i in additional_imgs:
                viewer.add_image(
                    i, blending='additive', interpolation='nearest'
                )
        viewer.add_points(
            result[['ycentroid', 'xcentroid']],
            name='t={} f={}'.format(thresholds, fwhms),
            properties={'sharpness': result.sharpness > 0.5},
            face_color='#ffffff00',
            edge_width=4,
            edge_color='sharpness',
            edge_color_cycle=['#d8b365', '#5ab4ac'],
            size=4/result.sharpness,
            opacity=0.5,
            visible=True,
        )

def _normalize_img_channel(img):
    if img.ndim == 2:
        return img.reshape(1, *img.shape)
    elif img.ndim == 3:
        if 3 in img.shape or 4 in img.shape:
            try:
                channel_idx = img.shape.index(3)
            except ValueError:
                channel_idx = img.shape.index(4)
            return np.moveaxis(img, channel_idx, 0)
        else:
            return img
    else:
        raise NotImplementedError(
            'image of shape {} is not supported'.format(img.shape)
        )

def _crop_center(img, center, crop_size):
    assert img.ndim == 2, 'image dimension ({}) not supported'.format(img.ndim)
    center_a = np.array(center)
    assert center_a.shape == (2,), (
        'cropping center must be a list in the form of [center_x, center_y] not {}'
            .format(center)
    )
    crop_size = np.array([crop_size]).flatten()
    if crop_size.shape == (1,):
        crop_size = np.repeat(crop_size, 2)
    assert crop_size.shape == (2,), (
        'cropping size must be a single number or a list in the form of [width, height] not {}'
            .format(center)
    )
    half = 0.5 * crop_size
    vertices = np.round([center_a - half, center_a + half]).astype(int).flatten()
    c_s, r_s, c_e, r_e = vertices
    c_s = 0 if c_s < 0 else c_s
    r_s = 0 if r_s < 0 else r_s
    return img[r_s:r_e, c_s:c_e]

