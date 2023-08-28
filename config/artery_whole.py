config = {
    "CHANNEL_IN"    : [1,1],
    "CHANNEL_OUT"   : [1,1],
    "CLASSES"       : {1: "Whole_Artery"},
    "SAVED_MODEL"   : [
        "blooming-flower-1043",
        "stellar-sound-1052"
        # "zesty-snowflake-1051"
        ],
    "from_raid"     : True,
    "interp_mode"   : ["trilinear_trilinear","area_area"],
    "MODEL_NAME"    : ["nnunet","nnunet"],
    "SPACING"       : [[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]], 
    "INPUT_SHAPE"   : [[128,128,128],[128,128,128]], 
    "DROPOUT"       : [0.,0.,0.],
    "CONTRAST"      : [[-150,300],[-150,300]],
    "INT_NORM"      : ['znorm','znorm'],
    "BATCH_SIZE"    : [12,12],
    "ACTIVATION"    : ['sigmoid','sigmoid'],
    "MODE"          : 'ensemble',
    "WEIGHTS"       : [1.0, 1.0],
    "MODEL_CHANNEL_IN"  : [32,32],
    "DEEPSUPERVISION"   : [True, True],
    "THRESHOLD"     : 0.05,
    "ARGMAX"        : False,
    "FLIP_XYZ"      : [False, False, False],
    "TRANSPOSE"     : [(1,2,0), (2,0,1)],
    "SAVE_CT"       : False,
    "SAVE_MERGE"    : False,
}

from transforms.ImageProcessing import *
# post processing 
transform = [
    Threshold(threshold=[config["THRESHOLD"]]),
    RemoveSamllObjects(min_size=5000),
    # KeepLargestComponent(connectivity=3)
]