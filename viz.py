import argparse
import sys
import find_parameters
import numpy as np

def main(argv=sys.argv):

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=parser.print_help)
    subparsers = parser.add_subparsers(
        dest='subparser_name',
        help='sub-command help'
    )


    parser_small = subparsers.add_parser('small', 
        description='Try spot detection on a small portion of a channel of an image with a combination of minimum intensities (thresholds) and spot sizes (fwhms)')
    parser_small.add_argument('image', 
        help='an image file to be processed')
    parser_small.add_argument('channel', 
        help='a channel number in the image to process; numbering starts at 0')
    parser_small.add_argument('-x', '--center-x', type=float, default=None, 
        help='x coordinate of the center of the crop')
    parser_small.add_argument('-y', '--center-y', type=float, default=None, 
        help='y coordinate of the center of the crop')
    parser_small.add_argument('--threshold-bottom', dest='t_bottom', type=float, default=None, 
        help='lower limit of the thresholds to try')
    parser_small.add_argument('--threshold-top', dest='t_top', type=float, default=None, 
        help='upper limit of the thresholds to try')
    parser_small.add_argument('--threshold-num-steps', dest='num_t', type=int, default=3, 
        help='number of thresholds to try between the lower and upper threshold limits')
    parser_small.add_argument('--fwhm-bottom', dest='f_bottom', type=float, default=2, 
        help='lower limit of the fwhms to try; small spots are better detected with small fwhms')
    parser_small.add_argument('--fwhm-top', dest='f_top', type=float, default=10, 
        help='upper limit of the fwhms to try; small spots are better detected with small fwhms')
    parser_small.add_argument('--fwhm-num-steps', dest='num_f', type=int, default=3, 
        help='number of fwhms to try between the lower and upper fwhm limits')
    parser_small.set_defaults(func=small_wrapper)


    parser_large = subparsers.add_parser('large',
        description='Test the choosen minimum intensities (thresholds) and spot sizes (fwhms) on a channel of a large image')
    parser_large.add_argument('image',
        help='an image file to be processed')
    parser_large.add_argument('channel', 
        help='a channel number in the image to process; numbering starts at 0')
    parser_large.add_argument('-n', '--nucleus-channel', dest='nucleus_ch', type=int, default=None,
        help='a channel number of the nuclear staining channel, optional; numbering starts at 0')
    parser_large.add_argument('-t', '--thresholds', nargs='+', type=float,
        help='a series of thresholds; number of thresholds must equal to number of fwhms')
    parser_large.add_argument('-f', '--fwhms', nargs='+', type=float,
        help='a series of fwhms; number of thresholds must equal to number of fwhms')
    parser_large.set_defaults(func=large_wrapper)

    args = parser.parse_args(argv[1:])
    if len(argv) == 1:
        args = None
    print()
    return args.func(args)

def small_wrapper(args):
    if args.num_t * args.num_f > 40:
        print(
            'current number of combinations ({}) of thresholds ({}) and fwhms ({}) exceeds 40, reduce number of steps'.format(
                args.num_t * args.num_f, args.num_t, args.num_f
            )
        )
        return 1
    print('loading channel {} of {}\n'.format(args.channel, args.image))
    img = find_parameters.load_channel_of_image(args.image, args.channel)
    if args.center_x is None:
        args.center_x = img.shape[1] * 0.5
    if args.center_y is None:
        args.center_y = img.shape[0] * 0.5
    max_area = np.square(4000)
    crop_size = np.array(img.shape) * np.sqrt(max_area / (img.shape[0] * img.shape[1]) )
    crop_size = crop_size[::-1]
    img = find_parameters._crop_center(img, [args.center_x, args.center_y], crop_size)
    if args.t_bottom is None: 
        args.t_bottom = np.percentile(img, 95)
    if args.t_top is None: 
        args.t_top = np.percentile(img, 99)
    if args.t_bottom < np.percentile(img, 0.01):
        print('--threshold-bottom is smaller than 0.01 percentile of the image, this will often increase runtime and generate bad results.')
    find_parameters.try_small_image(
        img, 
        args.t_bottom, args.t_top, args.num_t,
        args.f_bottom, args.f_top, args.num_f
    )

def large_wrapper(args):
    if len(args.thresholds) != len(args.fwhms):
        print(
            'number of thresholds {} must match number of fwhms {}'
                .format(args.thresholds, args.fwhms)
        )
        return 1
    img = find_parameters.load_channel_of_image(args.image, args.channel)
    max_area = np.square(20000)
    nucleus_img = None
    if args.nucleus_ch is not None:
        nucleus_img = [find_parameters.load_channel_of_image(args.image, args.nucleus_ch)]
    area_ratio = (img.shape[0] * img.shape[1]) / max_area
    if area_ratio > 1:
        crop_size = np.array(img.shape) * np.sqrt(area_ratio)
        crop_size = crop_size[::-1]
        img = find_parameters._crop_center(
            img, 0.5*np.array(img.shape)[::-1], crop_size
        )
        if nucleus_img is not None:
            nucleus_img = [find_parameters._crop_center(
                nucleus_img[0], 0.5*np.array(nucleus_img[0].shape)[::-1], crop_size
            )]
    find_parameters.try_full_image(
        img, args.thresholds, args.fwhms,
        additional_imgs=nucleus_img
    )

if __name__ == '__main__':
    sys.exit(main())