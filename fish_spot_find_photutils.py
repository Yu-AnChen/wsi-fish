import warnings
from skimage.morphology import binary_dilation, disk
import pandas as pd
import numpy as np
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
import contextlib
with contextlib.redirect_stdout(None):
    from photutils.detection import DAOStarFinder
    from photutils.utils import NoDetectionsWarning

def photutils_daostarfinder(img, thresholds, fwhms):
    mask = np.zeros_like(img, dtype=np.bool)
    results = []

    for t, f in zip(thresholds, fwhms):
        dao = DAOStarFinder(threshold=t, fwhm=f)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=NoDetectionsWarning
            )
            dao_dataframe = dao.find_stars(img, mask=mask)
        dao_dataframe = dao.find_stars(img, mask=mask)
        if dao_dataframe is not None:
            dao_dataframe = dao_dataframe.to_pandas()
            results += [dao_dataframe]
        
            _mask = np.zeros_like(img, dtype=np.bool)
            
            y_limit, x_limit = np.array(img.shape) - 1
            spot_mask_positions = (dao_dataframe
                .loc[:, ['xcentroid', 'ycentroid']]
                .transform(np.round).astype('int')
                .query('({y} >= 0) & ({y} <= @y_limit)'.format(y='ycentroid'))
                .query('({x} >= 0) & ({x} <= @x_limit)'.format(x='xcentroid'))
            )
            _mask[
                spot_mask_positions.ycentroid,
                spot_mask_positions.xcentroid
            ] = True
            # fwhm ~= 2.355 sigma
            mask += binary_dilation(_mask, selem=disk(2*f))
    
    if len(results) > 0:
        return pd.concat(results)
    else:
        return pd.DataFrame(columns=['xcentroid', 'ycentroid'])


from joblib import Parallel, delayed
import rowit

def process_whole_slide(
    img,
    thresholds,
    fwhms,
    block_size=2000,
    overlap_size=0,
    filter_sharpness=True,
):

    assert len(thresholds) == len(fwhms), (
        'number of thresholds ({}) must be equal to number of fwhms ({})'
    ).format(len(thresholds), len(fwhms))

    img_shape = img.shape

    wv_settings = rowit.WindowView(img_shape, block_size, overlap_size)
    img = wv_settings.window_view_list(img)

    print('        number of windows:', img.shape[0])

    spot_results = Parallel(n_jobs=8, verbose=0)(
        delayed(photutils_daostarfinder)(
            # use two-pass approach to catch out-of-focus spots
            i, thresholds, fwhms
        ) for i in img
    )
    y, x = 'ycentroid', 'xcentroid'

    img = wv_settings.reconstruct(img)

    window_position_ref = np.arange(len(spot_results)).reshape(
        wv_settings.window_view_shape[:2]
    )
    for idx, df in enumerate(spot_results):
        r, c = np.where(window_position_ref == idx)
        step_size = block_size - overlap_size
        df.loc[:, y] += (r*step_size - (0.5*overlap_size*(r != 0)))
        df.loc[:, x] += (c*step_size - (0.5*overlap_size*(c != 0)))

    full_result = pd.concat(spot_results)
    out_img = np.zeros(img_shape, dtype=np.uint8)

    if not full_result.empty:
        spot_img_positions = full_result[[y, x]]
        if filter_sharpness:
            spot_img_positions = full_result.query('sharpness > 0.5')[[y, x]]

        y_limit, x_limit = np.array(img_shape) - 1

        spot_img_positions = (spot_img_positions
            .transform(np.round).astype('int')
            .query('({} >= 0) & ({} <= @y_limit)'.format(y, y))
            .query('({} >= 0) & ({} <= @x_limit)'.format(x, x))
        )

        out_img[spot_img_positions[y], spot_img_positions[x]] += 1
        
    return full_result, out_img
