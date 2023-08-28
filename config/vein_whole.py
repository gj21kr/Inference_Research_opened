import os, sys
import importlib
from config.utils_test.fat_muscle import call_test
ensemble_models = [
    "vein_whole_581", 
    "vein_whole_586",
    "vein_whole_594"
]

config = {}
for i, e in enumerate(ensemble_models):
    temp = importlib.import_module(f'config.ensemble.{e}')
    for k in temp.config.keys():
        if i == 0 :
            config[k] = [temp.config[k]]
        else:
            config[k].append(temp.config[k])

config.update({
    "CLASSES"       : {1: "Whole_Vein"}, 
    "FLIP_XYZ"      : [False, False, False],
    "TRANSPOSE"     : [(1,2,0), (2,0,1)],
    "MODE"          : "Ensemble", # None, "Ensemble", "TTA"
    "THRESHOLD"     : 0.05,
    "ARGMAX"        : False,
    "WEIGHTS"       : [1.0, 1.0, 1.0],
    "SAVE_CT"       : False,
    "SAVE_MERGE"    : False,
    "from_raid"     : False,
})

from transforms.ImageProcessing import *
# post processing 
transform = [
    RemoveSamllObjects(min_size=5000)
]