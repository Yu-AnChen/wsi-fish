## Installation

1. Install minicona/anaconda and git
1. In a terminal, navigate to a desired location and download this reopsitory

    ```bash
    cd ~/my/project/directory/
    git clone https://github.com/Yu-AnChen/wsi-fish.git
    ```
1. Navigate to the wsi-fish directory and create the conda environment using the `conda_env-fish.yml` file
    
    Note: replace `<fish>` in the following block with a different conda enviroment name or path 
    
    ```bash
    cd ~/my/project/directory/wsi-fish
    conda env create -n <fish> -f conda_env-fish.yml
    ```
1. Activate the conda environment befor running
    
    Note: change `<fish>` in the following block if a different conda enviroment name or path was used to create the environment

    ```bash
    conda activate <fish>
    ```
## Suggested workflow

### Choosing parameters using `viz.py`

---

`python viz.py small`

```bash
Try spot detection on a small portion of an image with a combination of
minimum intensities (thresholds) and spot sizes (fwhms)

positional arguments:
  image                 an image file to be processed
  channel               a channel number in the image to process; numbering
                        starts at 0

optional arguments:
  -h, --help            show this help message and exit
  -x CENTER_X, --center-x CENTER_X
                        x coordinate of the center of the crop
  -y CENTER_Y, --center-y CENTER_Y
                        y coordinate of the center of the crop
  --threshold-bottom T_BOTTOM
                        lower limit of the thresholds to try
  --threshold-top T_TOP
                        upper limit of the thresholds to try
  --threshold-num-steps NUM_T
                        number of thresholds to try between the lower and
                        upper threshold limits
  --fwhm-bottom F_BOTTOM
                        lower limit of the fwhms to try; small spots are
                        better detected with small fwhms
  --fwhm-top F_TOP      upper limit of the fwhms to try; small spots are
                        better detected with small fwhms
  --fwhm-num-steps NUM_F
                        number of fwhms to try between the lower and upper
                        fwhm limits
```

Example command
```bash
python viz.py small "Z:\project-001\stitched\slide-01.ome.tif" 2 -x 7000 --threshold-bottom 1000 --threshold-top 10000 --threshold-num-steps 3 --fwhm-bottom 1 --fwhm-top 6 --fwhm-num-steps 5
```

---

`python viz.py large`

```bash
Test the choosen minimum intensities (thresholds) and spot sizes (fwhms) on a
channel of a large image

positional arguments:
  image                 an image file to be processed
  channel               a channel number in the image to process; numbering
                        starts at 0

optional arguments:
  -h, --help            show this help message and exit
  -n NUCLEUS_CH, --nucleus-channel NUCLEUS_CH
                        a channel number of the nuclear staining channel,
                        optional; numbering starts at 0
  -t THRESHOLDS [THRESHOLDS ...], --thresholds THRESHOLDS [THRESHOLDS ...]
                        a series of thresholds; number of thresholds must
                        equal to number of fwhms
  -f FWHMS [FWHMS ...], --fwhms FWHMS [FWHMS ...]
                        a series of fwhms; number of thresholds must equal to
                        number of fwhms
```

Example command
```bash
python viz.py large "Z:\project-001\stitched\slide-01.ome.tif" 2 -t 3000 3000 -f 2 6
```

### Batch processing using `batch_photutils.py`

---

The `batch_photutils.py` requires a configuration yaml file. Here's an example yaml file 

```yaml
input file dir: Z:\project-001\stitched\
filename pattern: "slide*.ome.tif"
recursive: False
settings:
- channel: 2
    thresholds:
    - 3000
    - 3000
    fwhms:
    - 2
    - 6
- channel: 3
    thresholds:
    - 1000
    fwhms:
    - 2
output dir: Z:\project-001\stitched\fish-output
```

Notes for a valid configuration yaml file
- the channel numbering starts at 0
- all the top level keys are required (input file dir, filename pattern, recursive, settings, output dir)
- for each channel, more than one minimum intensities (thresholds) and spot sizes (fwhms) can be run in sequential, but the number of thresholds must equal to the number of fwhms

Example command
```bash
python batch_photutils.py project-001-config.yml
```