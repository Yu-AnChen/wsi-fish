import sys
import argparse

from pathlib import Path
import skimage.io
import rowit
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from fish_spot_find_trackpy import (
    locate_with_mask, gmm_filter_spots,
    trackpy_spots_to_img
)
from fish_spot_find_photutils import photutils_daostarfinder

parser = argparse.ArgumentParser()

parser.add_argument('input-file-dir')
parser.add_argument('filename-pattern')
parser.add_argument('channels', nargs='+', type=int)
parser.add_argument('--recursive', default=False, action='store_true')
parser.add_argument('--save-result-img', default=True, dest='save_img')
parser.add_argument('output-dir')


args = parser.parse_args(sys.argv[1:])

file_paths = Path(vars(args)['input-file-dir'])
if args.recursive is True:
    do_glob = file_paths.rglob
else:
    do_glob = file_paths.glob
file_paths = sorted(do_glob(vars(args)['filename-pattern']))

if len(file_paths) == 0:
    raise ValueError(
        'No matched file found with pattern "{}" in {}'.format(
            vars(args)['filename-pattern'], file_paths
        )
    )

print('Files to be precessed')
print(''.join(['    {}\n'.format(p.name) for p in file_paths]))
print('File count: {} files\n'.format(len(file_paths)))

method = 'photutils'

channels = args.channels
save_img = args.save_img
out_dir = Path(vars(args)['output-dir'])

block_size = 2000
overlap_size = 0

for file_path in file_paths[:]:
    print('    Processing', file_path.name)
    img_name = file_path.stem

    for i, channel in enumerate(channels):
        img = skimage.io.imread(str(file_path), key=channel)

        img_shape = img.shape
        img_dtype = img.dtype

        wv_settings = rowit.WindowView(img_shape, block_size, overlap_size)
        img = wv_settings.window_view_list(img)
        window_mask = wv_settings.padding_mask()
        
        if i == 0:
            print('        number of windows:', img.shape[0])

        print(
            '       ',
            'processing channel %d/%d' % (i + 1, len(channels)),
            end='\r'
        )

        if method == 'trackpy':
            min_peak_int = np.median(np.percentile(img, 95, axis=(1, 2)))
            spot_results = Parallel(n_jobs=8, verbose=0)(
                delayed(locate_with_mask)(
                    i, padding_mask=m, min_peak_int=min_peak_int
                ) for i, m in zip(img, window_mask)
            )
            y, x = 'y', 'x'

        elif method == 'photutils':
            min_peak_int = np.median(np.percentile(img, 95, axis=(1, 2)))
            print('spot peak threshold:', min_peak_int*np.exp(1))
            spot_results = Parallel(n_jobs=8, verbose=0)(
                delayed(photutils_daostarfinder)(
                    # use two-pass approach to catch out-of-focus spots
                    i, [min_peak_int*np.exp(1)]*2, [2, 6]
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
        
        if method == 'trackpy':
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
                skimage.io.imsave(
                    str(out_dir / 'fish_trackpy-{}.tif'.format(img_name)),
                    trackpy_spots_to_img(img_shape, filtered),
                    bigtiff=True, metadata=None, append=append,
                    tile=(1024, 1024), photometric='minisblack',
                    check_contrast=False
                )

        if method == 'photutils':
            out_img = np.zeros(img_shape, dtype=np.uint8)
            out_img[
                full_result[y].transform(np.round).astype('int'),
                full_result[x].transform(np.round).astype('int')
            ] += 1
            append = True
            if i == 0:
                append = False
            skimage.io.imsave(
                str(out_dir / 'fish_photutils-{}.tif'.format(img_name)),
                out_img,
                bigtiff=True, metadata=None, append=append,
                tile=(1024, 1024), photometric='minisblack',
                check_contrast=False
            )

    print()