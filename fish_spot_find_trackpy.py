# pylab
from trackpy import locate

import numpy as np
from scipy.stats import percentileofscore 
import warnings
from sklearn.mixture import GaussianMixture
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

