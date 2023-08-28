import sys
from config.utils_test.fat_muscle import call_test

config = {
    "CHANNEL_IN"    : 1,
    "CHANNEL_OUT"   : 1,
    "SAVED_MODEL"   : "apricot-smoke-585",
    "MODEL_NAME"    : "nnunet",
    "interp_mode"   : 'area_area',
    "SPACING"       : [1.0, 1.0, 1.0],
    "INPUT_SHAPE"   : [128,128,128],
    "CONTRAST"      : [-250,250],
    "INT_NORM"      : 'znorm',
    "DEEPSUPERVISION": True,
    "MODEL_CHANNEL_IN" : 32,
    "HU_WEIGHTING"  : [False, None, "", 1.0],
    "BATCH_SIZE"    : 12,
    "ACTIVATION"    : 'sigmoid',
    "SAVE_CT"       : False,
    "SAVE_MERGE"    : False,
}
