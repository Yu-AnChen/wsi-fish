# pylab

import numpy as np
from skimage.util import view_as_windows, montage


def view_as_windows_overlap(
    img, block_size, overlap_size, pad_mode='constant',
    return_padding_mask=False
): 
    init_shape = img.shape
    step_size = block_size - overlap_size

    padded_shape = np.array(init_shape) + overlap_size
    n = np.ceil((padded_shape - block_size) / step_size)
    padded_shape = (block_size + (n * step_size)).astype(np.int)

    half = int(overlap_size / 2)
    img = np.pad(img, (
        (half, padded_shape[0] - init_shape[0] - half), 
        (half, padded_shape[1] - init_shape[1] - half),
    ), mode=pad_mode)

    if return_padding_mask:
        padding_mask = np.ones(init_shape)
        padding_mask = np.pad(padding_mask, (
            (half, padded_shape[0] - init_shape[0] - half), 
            (half, padded_shape[1] - init_shape[1] - half),
        ), mode='constant', constant_values=0).astype(np.bool)
        return (
            view_as_windows(img, block_size, step_size),
            view_as_windows(padding_mask, block_size, step_size)
        )
    else:
        return view_as_windows(img, block_size, step_size)

def reconstruct_from_windows(
    window_view, block_size, overlap_size, out_shape=None
):
    grid_shape = window_view.shape[:2]

    start = int(overlap_size / 2)
    end = int(block_size - start)

    window_view = window_view.reshape(
        (-1, block_size, block_size)
    )[..., start:end, start:end]

    if out_shape:
        re, ce = out_shape
    else:
        re, ce = None, None

    return montage(
        window_view, grid_shape=grid_shape, 
    )[:re, :ce]

def crop_with_padding_mask(img, padding_mask, return_mask=False):
    if np.all(padding_mask == 1):
        return (img, padding_mask) if return_mask else img
    (r_s, r_e), (c_s, c_e) = [
        (i.min(), i.max() + 1)
        for i in np.where(padding_mask == 1)
    ]
    padded = np.zeros_like(img)
    img = img[r_s:r_e, c_s:c_e]
    padded[r_s:r_e, c_s:c_e] = 1
    return (img, padded) if return_mask else img