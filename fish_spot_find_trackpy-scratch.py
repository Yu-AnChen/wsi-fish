# pylab
from trackpy import locate
import skimage.io
from joblib import Parallel, delayed
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt
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
    extremes = locate_df[locate_df.signal > np.exp(2)*locate_df.min()]
    if not extremes.empty:
        out[
            extremes.y.transform(np.round).astype('int'),
            extremes.x.transform(np.round).astype('int')
        ] += 9
    return out


filename = 'scaled_2.00-29B_8w_T2_20200221.ome.tif'
channels = [1, 2, 5]
save_img = True

img = skimage.io.imread(filename)

img_1 = img[1]

block_size = 2000
overlap_size = 0

img_shape = img_1.shape
img_dtype = img_1.dtype
img_1, window_mask = rowit.view_as_windows_overlap(
    img_1, block_size, overlap_size, return_padding_mask=True
)
window_view_shape = img_1.shape
img_1 = img_1.reshape(-1, block_size, block_size)
print('Number of windows:', img_1.shape[0])
window_mask = window_mask.reshape(-1, block_size, block_size)

min_peak_int = np.median(np.percentile(img_1, 95, axis=(1, 2)))
spot_results = Parallel(n_jobs=8, verbose=1)(
    delayed(locate_with_mask)(
        i, padding_mask=m, min_peak_int=min_peak_int
    ) for i, m in zip(img_1, window_mask)
)

img_1 = rowit.reconstruct_from_windows(
    img_1.reshape(window_view_shape),
    block_size, overlap_size, img_shape
)

ref = np.arange(len(spot_results)).reshape(window_view_shape[:2])
for idx, df in enumerate(spot_results):
    r, c = np.where(ref == idx)
    step_size = block_size - overlap_size
    df.loc[:, 'y'] += (r*step_size - (0.5*overlap_size*(r != 0)))
    df.loc[:, 'x'] += (c*step_size - (0.5*overlap_size*(c != 0)))

full_result = pd.concat(spot_results)
full_result.drop(columns=['raw_mass', 'ep'], inplace=True)

full_result = full_result.astype(
    {'mass': 'int64', 'signal': str(img_dtype)}
)

filtered = gmm_filter_spots(full_result)

filtered.to_csv('{}.csv'.format(filename))
if save_img == True:
    skimage.io.imsave(
        '{}.tif'.format(filename),
        trackpy_spots_to_img(img_shape, filtered),
        bigtiff=True, metadata=None
    )


plt.figure()
plt.scatter(filtered.x, filtered.y, marker='.', linewidths=0, s=8, c=np.log(filtered.signal))
plt.gca().set_aspect('equal')
plt.gca().invert_yaxis()

roi = filtered[filtered.x.between(6000, 10000) & filtered.y.between(6000, 15000)]
plt.figure()
plt.imshow(img_1[6000:15000, 6000:10000], cmap='gray', vmax=2*min_peak_int)
plt.scatter(
    roi.x-6000, roi.y-6000, 
    marker='.', linewidths=0,
    c=roi.signal > np.exp(np.log(min_peak_int) + 2), 
    s=(roi.mass / (min_peak_int / 2))
)


# Compare cluster method
subtable = full_result[full_result.x.between(6000, 10000) & full_result.y.between(6000, 15000)]
colors = []
methods = []

colors += [((subtable['ecc'] > 0.3) & (subtable['size'] < 0.888))]
methods += ['cutoff']

gmm1 = GaussianMixture(n_components=2, covariance_type='diag', random_state=1001)
gmm1.fit(np.array([subtable['ecc'], subtable['size']]).T)

colors += [gmm1.predict_proba(np.array([subtable['ecc'], subtable['size']]).T)[:, 1] > 0.9]
methods += ['covariance diag']

gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=1001)
gmm.fit(np.array([subtable['ecc'], subtable['size']]).T)

colors += [gmm.predict_proba(np.array([subtable['ecc'], subtable['size']]).T)[:, 1] > 0.9]
methods += ['covariance full']

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
for c, m, ax in zip(colors, methods, axes.ravel()):
    plt.sca(ax)
    plt.imshow(img[2][6000:15000, 6000:10000], cmap='gray_r', vmax=min_peak_int)
    plt.scatter(
        subtable.x-6000, subtable.y-6000,
        marker='.', linewidths=0,
        c=c
    )
    ax.title.set_text(m)
# This seems to help me establish the filters of
# ['ecc'] > 0.3 & ['size'] < 0.888
# maybe use peak intensity differ by exp(2) as the cutoff of 
# aggregated dots vs single target molecule
subtable = full_result[full_result.x.between(6000, 10000) & full_result.y.between(6000, 15000)]

# figure(); scatter(subtable.signal, subtable.mass, marker='.', s=1, c=subtable['size'])
# figure(); scatter(subtable.signal, subtable.mass, marker='.', s=1, c=subtable['ecc'])
# figure(); subtable['eec'].hist(bins=1000)
# figure(); subtable['size'].hist(bins=1000)



