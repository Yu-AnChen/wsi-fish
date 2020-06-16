# pylab
from trackpy import locate
import skimage.io
from joblib import Parallel, delayed
import pandas as pd

import numpy as np
from pathlib import Path
from scipy.stats import percentileofscore 
import warnings
from sklearn.mixture import GaussianMixture
from skimage.util import montage, view_as_windows
import rowit


def locate_with_mask(img, padding_mask=None, min_peak_int=None):
    if padding_mask is not None and np.any(padding_mask == 0):
        img = rowit.crop_with_padding_mask(
            img, padding_mask
        )
    if min_peak_int is None:
        p = 95
    else:
        p = percentileofscore(img.flatten(), min_peak_int)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='Image contains no local maxima.'
        )
        warnings.filterwarnings(
            'ignore', message='All local maxima were in the margins.'
        )
        spot_table = locate(
            img, 3, preprocess=False, percentile=p, separation=2
        )
    return spot_table

def gmm_filter_spots(spots_table):
    # Filtering method 1: Gaissoam Mixture clustering
    gmm = GaussianMixture(
        n_components=2, covariance_type='full', random_state=1001
    )
    subset = spots_table.sample(100000, random_state=1001)
    gmm.fit(np.array([subset['ecc'], subset['size']]).T)
    label = 0
    if gmm.means_[0, 0] < gmm.means_[1, 0]:
        label = 1
    filtered = spots_table[
        gmm.predict_proba(
            np.array([spots_table['ecc'], spots_table['size']]).T
        )[:, label] > 0.9
    ]

    # Filtering method 2: Gating
    # filtered = spots_table[(spots_table['ecc'] > 0.3) & (spots_table['size'] < 0.888)]
    return filtered

def trackpy_spots_to_img(img_shape, locate_df):
    out = np.zeros(img_shape, dtype=np.int8)
    out[
        locate_df.y.transform(np.round).astype('int'),
        locate_df.x.transform(np.round).astype('int')
    ] = 1
    extremes = locate_df[
        locate_df.signal > np.exp(2)*locate_df.signal.min()
    ]
    if not extremes.empty:
        out[
            extremes.y.transform(np.round).astype('int'),
            extremes.x.transform(np.round).astype('int')
        ] += 9
    return out


file_path = r'Z:\JL503_JERRY\152-SABIN_FISH2-2020FEB\ashlar\ome_tiffs\scaled_2.00-29B_8w_T2_20200221.ome.tif'
channels = [1, 2, 5]
save_img = True
out_dir = r'Z:\JL503_JERRY\152-SABIN_FISH2-2020FEB\ashlar\ome_tiffs\fish'

block_size = 2000
overlap_size = 0

file_path = Path(file_path)
img_name = file_path.stem
out_dir = Path(out_dir)

for i, channel in enumerate(channels):
    img = skimage.io.imread(str(file_path), key=channel)

    img_shape = img.shape
    img_dtype = img.dtype
    img, window_mask = rowit.view_as_windows_overlap(
        img, block_size, overlap_size, return_padding_mask=True
    )
    window_view_shape = img.shape
    img = img.reshape(-1, block_size, block_size)
    print('Number of windows:', img.shape[0])
    window_mask = window_mask.reshape(-1, block_size, block_size)

    min_peak_int = np.median(np.percentile(img, 95, axis=(1, 2)))
    spot_results = Parallel(n_jobs=8, verbose=1)(
        delayed(locate_with_mask)(
            i, padding_mask=m, min_peak_int=min_peak_int
        ) for i, m in zip(img, window_mask)
    )

    img = rowit.reconstruct_from_windows(
        img.reshape(window_view_shape),
        block_size, overlap_size, img_shape
    )

    window_position_ref = np.arange(len(spot_results)).reshape(window_view_shape[:2])
    for idx, df in enumerate(spot_results):
        r, c = np.where(window_position_ref == idx)
        step_size = block_size - overlap_size
        df.loc[:, 'y'] += (r*step_size - (0.5*overlap_size*(r != 0)))
        df.loc[:, 'x'] += (c*step_size - (0.5*overlap_size*(c != 0)))

    full_result = pd.concat(spot_results)
    full_result.drop(columns=['mass', 'ep'], inplace=True)

    mass_dtype = 'int64'
    if np.issubdtype(img_dtype, np.floating):
        mass_dtype = 'float64'
    full_result = full_result.astype(
        {'raw_mass': mass_dtype, 'signal': str(img_dtype)}
    )

    filtered = gmm_filter_spots(full_result)

    csv_name = 'fish_trackpy-{}-channel_{}.csv'.format(img_name, channel)
    filtered.to_csv(str(out_dir / csv_name))
    if save_img is True:
        append = True
        if i == 0:
            append = False
        print('append', append)
        skimage.io.imsave(
            str(out_dir / 'fish_trackpy-{}.tif'.format(img_name)),
            trackpy_spots_to_img(img_shape, filtered),
            bigtiff=True, metadata=None, append=append,
            tile=(1024, 1024), photometric='minisblack',
            check_contrast=False
        )


try:
    plot_debug
except NameError:
    plot_debug = False

if plot_debug is True:
    from matplotlib import pyplot as plt

    plt.figure()
    plt.scatter(
        filtered.x, filtered.y, 
        marker='.', linewidths=0, s=8, 
        c=np.log(filtered.signal)
    )
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()

    r_s, r_e, c_s, c_e = [
        6000, 15000, 
        6000, 10000
    ]
    roi = filtered[
        filtered.x.between(c_s, c_e) & filtered.y.between(r_s, r_e)
    ]
    plt.figure()
    plt.imshow(img[r_s:r_e, c_s:c_e], cmap='gray', vmax=2*min_peak_int)
    plt.scatter(
        roi.x-c_s, roi.y-r_s, 
        marker='.', linewidths=0,
        c=roi.signal > np.exp(np.log(min_peak_int) + 2), 
        s=(roi.raw_mass / (min_peak_int / 2))
    )