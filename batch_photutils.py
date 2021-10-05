import sys
import argparse
import pathlib
import yaml
from fish_spot_find_photutils import process_whole_slide
import skimage.io


def main(argv=sys.argv):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config-file', type=argparse.FileType('r'))

    args = parser.parse_args(argv[1:])

    config_str = vars(args)['config-file'].read()
    config = yaml.load(config_str, Loader=yaml.SafeLoader)

    if validate_config(config) == 1:
        return
    file_paths = get_file_list(config)
    if file_paths == 1:
        return

    print('Files to process')
    print(''.join(['    {}\n'.format(p.name) for p in file_paths]))
    print('File count: {} files\n'.format(len(file_paths)))

    out_dir = pathlib.Path(config['output dir'])

    for file_path in file_paths[:]:
        print('    Processing', file_path.name)
        img_name = file_path.stem 

        for i, setting in enumerate(config['settings']):
            channel = setting['channel']
            img = skimage.io.imread(str(file_path), key=channel)
            print(
                '       ',
                'processing channel %d/%d' % (i + 1, len(config['settings']))
            )
            _, out_img = process_whole_slide(img, setting['thresholds'], setting['fwhms'])
            append = True
            if i == 0:
                append = False
            skimage.io.imsave(
                str(out_dir / 'fish_photutils-{}.tif'.format(img_name)),
                out_img,
                bigtiff=True, append=append,
                metadata=None, ome=False,
                tile=(1024, 1024), photometric='minisblack',
                check_contrast=False
            )
        print()

def validate_config(config):
    # required field exist
    for k in [
        'input file dir',
        'filename pattern',
        'recursive',
        'settings',
        'output dir'
    ]:
        if k not in config.keys():
            print(
                'configuration file missing required field - {}'.format(k)
            )
            return 1
    # write access
    out_dir = pathlib.Path(config['output dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'log.txt', 'w') as f:
        f.write('validated')
    (out_dir / 'log.txt').unlink()
    # match number of thresholds and fwhms
    for s in config['settings']:
        if len(s['thresholds']) != len(s['fwhms']):
            print('number of thresholds ({}) does not match numner of fwhms ({})'.format(
                len(s['thresholds']), len(s['fwhms'])
            ))
            print(yaml.dump(s))
            return 1

def get_file_list(config):
    # glob input files
    file_paths = pathlib.Path(config['input file dir'])
    if not file_paths.exists():
        print('input file dir ({}) does not exist'.format(file_paths))
        return 1
    do_glob = file_paths.glob
    if config['recursive'] is True:
        do_glob = file_paths.rglob
    file_paths = sorted(do_glob(config['filename pattern']))
    if len(file_paths) == 0:
        print('No matched file found with pattern "{}" in {}'.format(
            config['filename pattern'], config['input file dir']
        ))
        return 1
    return file_paths

if __name__ == '__main__':
    main()
