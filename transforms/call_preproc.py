from __future__ import annotations

from monai.transforms import ScaleIntensityRanged

from transforms.ImageProcessing import *

__all__ = ['call_trans_function']

def call_trans_function(config, ensemble):
    test_transforms = []
    config_names = config["INT_NORM"][ensemble].lower()
    if '+' in config_names: 
        config_names = config_names.split('+')
    else:
        config_names = [config_names]

    for config_name in config_names:
        if config_name in ['scale', 'windowing', 'clip']:
            test_transforms += [
                ScaleIntensityRanged(keys=["image"],
                    a_min=config["CONTRAST"][ensemble][0], a_max=config["CONTRAST"][ensemble][1], 
                    b_min=0, b_max=1, clip=True),
                ]
        elif config_name in ['z_norm', 'znorm', 'z norm']:
            test_transforms += [
                ZNormalizationd(keys=["image"],contrast=config["CONTRAST"][ensemble],clip=True)
            ]
        elif config_name in ['min_max_norm', 'norm', 'min max norm']:
            test_transforms += [
                Normalizationd(keys=["image"])
            ]
        else:
            print('Not Intensity Normalization')
    return test_transforms