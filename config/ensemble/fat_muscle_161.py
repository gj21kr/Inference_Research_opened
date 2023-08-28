import sys
from config.utils_test.fat_muscle import call_test

config = {
    "CHANNEL_IN"    : 1,
    "CHANNEL_OUT"   : 4,
    "SAVED_MODEL"   : "solar-armadillo-161",
    "MODEL_NAME"    : "nnunet",
    "SPACING"       : [2.5,2.5,2.5],
    "INPUT_SHAPE"   : [160,160,160],
    "CONTRAST"      : [-150,250],
    "INT_NORM"      : 'znorm',
    "DEEPSUPERVISION": True,
    "MODEL_CHANNEL_IN" : 32,
    "HU_WEIGHTING"  : [False, None, "", 1.0],
    "BATCH_SIZE"    : 24,
    "ACTIVATION"    : 'sigmoid',
    "SAVE_CT"       : False,
    "SAVE_MERGE"    : False,
}
