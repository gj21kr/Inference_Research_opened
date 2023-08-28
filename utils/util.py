# -*- coding: utf-8 -*-
# Copyright 2023 by Jung-eun Park, Hutom.
# All rights reserved.
from __future__ import annotations

import os, gc, sys
import json
import argparse as ap
import subprocess
import SimpleITK as sitk
import scipy
import numpy as np 
import torch 
from matplotlib import pyplot as plt 

from core.call import call_model
from transforms.Orientation import orientation_revert

__all__ = [
	'add_array', 'sum_list_of_array', 'image_saver',
	'load_saved_model', 'arg2target', 'str2bool',
	'merger', 'saver', 'dict_update', 'gen_models',
	'get_underutilized_gpu'
]


def get_underutilized_gpu(num_gpus=None):
	def list2str(list_):
		re = ''
		for li in list_:
			re += f',{int(li)}'
		return re[1:]
	try:
		if num_gpus is None: num_gpus = 1
		gpus = []
		result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,nounits'], capture_output=True, text=True)
		gpu_memory_info = result.stdout.strip().split('\n')[1:]
		available_memory = []
		for gpu_info in gpu_memory_info:
			gpu_index,used_memory, total_memory = map(float, gpu_info.split(','))
			if float(used_memory/total_memory) < 0.3 and len(gpus)<int(num_gpus):
				gpus += [int(gpu_index)]
		return list2str(gpus)
	except FileNotFoundError:
		return None
	
def add_array(arr1, arr2):
	if arr1 is None: 
		return arr2
	else:
		return np.add(arr1, arr2)

def sum_list_of_array(list1):
	for i, arr in enumerate(list1):
		if i ==0 : 
			result = arr
		else:
			result = np.add(result, arr)
	return result 
	
def image_saver(ct, ct_labels, save_path, p_name=None):
	inds = np.where(ct_labels>0)
	z_min, z_max = min(inds[-1]), max(inds[-1])
	if ct.shape[0] < 5: ct = ct[0]
	for i in range(z_min, z_max):
		ct_image = ct[:,:,i]    
		ct_image[ct_image<-150] = -150; ct_image[ct_image>300] = 300     
		gt_image = np.zeros_like(ct_image)
		for j in range(ct_labels.shape[0]):
			temp = ct_labels[j,:,:,i]			
			gt_image[temp>0] = int(j+1)
		gt_image = np.ma.masked_where(gt_image == 0, gt_image)
		plt.imshow(ct_image, 'gray')
		plt.imshow(gt_image, cmap='tab10', alpha=0.7, vmin=1, vmax=ct_labels.shape[0]+1)
		plt.colorbar()
		img_file = os.path.join(save_path, f'{p_name}_{i}.png')
		plt.savefig(img_file)
		plt.close()
	print('image saving done', p_name)

def load_saved_model(config, seq, model):
	saved_model = ''
	if "MODEL_KEY" not in config.keys():
		key = "model_state_dict"
	else:
		key = config["MODEL_KEY"]
	if "DataParallel" not in config.keys():
		dp = True
	else:
		dp = config["DataParallel"]
	if "from_raid" in config.keys() and config["from_raid"]==True:
		from utils.model_list import path
		saved_model = path[config["SAVED_MODEL"][seq]]
	else:
		saved_model = config["SAVED_MODEL"][seq]
		saved_model = f'./models/{saved_model}'  
	if dp == True:
		model = torch.nn.DataParallel(model)
	model_load = torch.load(saved_model, map_location=config["device"])
	model.load_state_dict(model_load[key], strict=False)
	model.to(config["device"])
	model.eval()
	return model

def arg2targets(arg_t):
	if ',' not in arg_t and ' ' not in arg_t:
		print('This is multi-projects inference code. You can use other code for single target inference.')
	if ',' in arg_t:
		return arg_t.split(',')
	if ' ' in arg_t:
		return arg_t.split(' ')

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise ap.ArgumentTypeError('Boolean value expected.')


def merger(classes, config):
	rst_path = config["rst_path"]
	file_ = os.path.join(rst_path, f'{classes[1]}.nii.gz')
	img_ = sitk.ReadImage(file_)
	arr_ = sitk.GetArrayFromImage(img_)
	for i in range(len(classes)):
		file_ = os.path.join(rst_path, f'{classes[i+1]}.nii.gz')
		img_ = sitk.ReadImage(file_)
		temp_ = sitk.GetArrayFromImage(img_)
		np.putmask(arr_, temp_>0, i+1)
	img_ = sitk.GetImageFromArray(arr_)
	img_.SetSpacing(config["original_spacing"])
	img_.SetOrigin(config["original_origin"])
	direction = [config["original_direction"][0], config["original_direction"][3], 0,
				config["original_direction"][1], config["original_direction"][4], 0,
				0, 0, 1]
	img_.SetDirection(direction)
	save_img: str = os.path.join(rst_path, f'Merged.nii.gz')
	sitk.WriteImage(img_, save_img); os.chmod(save_img , 0o777)
	del temp_, img_, arr_; gc.collect()

def saver(class_, x, config, max_intensity=255) -> None: 
	if len(np.unique(x))==1:
		print('No inference result!', class_)
		return 
	rst_path = config["rst_path"]
	x = orientation_revert(x, config["FLIP_XYZ"][0], config["FLIP_XYZ"][1], config["FLIP_XYZ"][2], config["TRANSPOSE"][1])
	if config['APPLY_TRANSFORM']==True and 'Return_Angles' in config.keys():
		from utils.dcm_reader import rotate_forward
		config["PixelData"] = (x>0).astype(np.uint8)
		config = rotate_forward(
			rot_angles= config["Return_Angles"], 
			meta_info=config, reshape=False, reverse=True
		)
		x = config["PixelData"]
	x = ((x>0).astype(np.uint8)*max_intensity).astype(np.uint8)
	x = sitk.GetImageFromArray(x)
	x.SetSpacing(config["original_spacing"])
	x.SetOrigin(config["original_origin"])
	direction = [config["original_direction"][0], config["original_direction"][3], 0,
				config["original_direction"][1], config["original_direction"][4], 0,
				0, 0, 1]
	x.SetDirection(direction)
	save_img: str = os.path.join(rst_path, f'{class_}.nii.gz')
	sitk.WriteImage(x, save_img); os.chmod(save_img , 0o777)
	print("Saved",save_img)
	del x; gc.collect()

def dict_update(config, common):
	config["original_shape"] = common["original_shape"]
	config["original_spacing"] = common["original_spacing"]
	config["original_origin"] = common["original_origin"]
	config["case_name"] = common["case_name"]
	config["series_name"] = common["series_name"]
	config["rst_path"] = common["rst_path"]
	return config

def gen_models(config):
	if config["MODE"] is not None and config["MODE"].lower() in ['tta']:
		new_config = config.copy()
		new_config["MODEL_NAME"] = config["MODEL_NAME"][0]
		new_config["SPACING"] = config["SPACING"][0]
		new_config["INPUT_SHAPE"] = config["INPUT_SHAPE"][0]
		new_config["CONTRAST"] = config["CONTRAST"][0]
		new_config["DROPOUT"] = config["DROPOUT"][0]
		if "FEATURE_SIZE" in list(config.keys()):
			new_config["FEATURE_SIZE"] = config["FEATURE_SIZE"][0]
		if "PATCH_SIZE" in list(config.keys()):
			new_config["PATCH_SIZE"] = config["PATCH_SIZE"][0]
		return [call_model(new_config)]
	elif config["MODE"] is not None and config["MODE"].lower() in ['ensemble']:
		models = []
		for i in range(len(config["SAVED_MODEL"])):
			new_config = config.copy()
			new_config["CHANNEL_IN"] = config["CHANNEL_IN"][i]
			new_config["CHANNEL_OUT"] = config["CHANNEL_OUT"][i]
			new_config["MODEL_NAME"] = config["MODEL_NAME"][i]
			new_config["SPACING"] = config["SPACING"][i]
			new_config["INPUT_SHAPE"] = config["INPUT_SHAPE"][i]
			new_config["CONTRAST"] = config["CONTRAST"][i]
			if "FEATURE_SIZE" in list(config.keys()):
				new_config["FEATURE_SIZE"] = config["FEATURE_SIZE"][i]
			if "PATCH_SIZE" in list(config.keys()):
				new_config["PATCH_SIZE"] = config["PATCH_SIZE"][i]
			if "MODEL_CHANNEL_IN" in list(config.keys()):
				new_config["MODEL_CHANNEL_IN"] = config["MODEL_CHANNEL_IN"][i]
			models.append(call_model(new_config))
		return models
	else:
		new_config = config.copy()
		new_config["MODEL_NAME"] = config["MODEL_NAME"][0]
		new_config["SPACING"] = config["SPACING"][0]
		new_config["INPUT_SHAPE"] = config["INPUT_SHAPE"][0]
		new_config["CONTRAST"] = config["CONTRAST"][0]
		new_config["DROPOUT"] = config["DROPOUT"][0]
		if "FEATURE_SIZE" in list(config.keys()):
			new_config["FEATURE_SIZE"] = config["FEATURE_SIZE"][0]
		if "PATCH_SIZE" in list(config.keys()):
			new_config["PATCH_SIZE"] = config["PATCH_SIZE"][0]
		return [call_model(new_config)]
