from __future__ import annotations

import os, sys, time, gc

import torch
import torch.nn as nn
import scipy
import numpy as np
import SimpleITK as sitk
from shutil import copyfile
from joblib import Parallel, delayed
import concurrent.futures
import multiprocessing
from monai.transforms import (
		Compose, ScaleIntensityRanged, Affined
)

from utils.inferer import SlidingWindowInferer
from utils.dcm_reader import Hutom_DicomLoader
from utils.util import saver, str2bool, gen_models, merger, load_saved_model, add_array
from utils.kaist_utils import KAISTSlidingwindowInferer
from transforms.Orientation import orientation
from transforms.call_preproc import call_trans_function

job_threshold = 10
class do_inference():
	def __init__(self, config, model, raw, ensemble=0):
		self.original_spacing = [config["original_spacing"][1], config["original_spacing"][0], config["original_spacing"][2]]
		self.original_shape = config["original_shape"]
		self.target_shape = None
		self.model_spacing = None
		self.add_channel = False
		self.invert = False
		self.active = None
		if "interp_mode" in config.keys():
			self.interp_mode = config["interp_mode"][ensemble]
		else:
			self.interp_mode = "trilinear_trilinear"
		
		if config["SPACING"][ensemble][0] is None and config["SPACING"][ensemble][1] is not None:
			self.target_spacing = [*config["original_spacing"][:2], config["SPACING"][i][-1]]
			self.model_spacing = [*config["original_spacing"][:2], config["SPACING"][i][-1]]
		elif config["SPACING"][ensemble][0] is None and config["SPACING"][ensemble][1] is None:
			self.target_spacing = config["original_spacing"]
			self.model_spacing = config["original_spacing"]
		else:
			self.target_spacing = config["SPACING"][ensemble]
			self.model_spacing = config["SPACING"][ensemble]				

		if config["TRANSPOSE"][0]==(0,1,2) or config["TRANSPOSE"][0]==(0,2,1):
			self.original_spacing = [self.original_spacing[-1], *self.original_spacing[:2]]
			self.target_spacing = [self.target_spacing[-1], *self.target_spacing[:2]]  
			self.model_spacing = [self.model_spacing[-1], *self.model_spacing[:2]]

		self.test_transforms = call_trans_function(config, ensemble)
		
		if config["MODE"] is not None and config["MODE"].lower() == 'tta':
			self.invert = True 

		self.model=model[ensemble]
		self.weight=config["WEIGHTS"][ensemble]

		if "DEEPSUPERVISION" in config.keys() and config["DEEPSUPERVISION"][ensemble]==True:
			self.deep_supervision = True
		else: 
			self.deep_supervision = False

		if config["ACTIVATION"][ensemble] is not None:
			if config["ACTIVATION"][ensemble].lower()=='sigmoid':
				self.active=torch.nn.Sigmoid()
			elif config["ACTIVATION"][ensemble].lower()=='softmax':
				self.active=torch.nn.Softmax(dim=0)
		
		if type(config["CHANNEL_OUT"])==list: 
			channel_out = config["CHANNEL_OUT"][ensemble]
		else:
			channel_out = config["CHANNEL_OUT"]
		if channel_out > len(config["CLASSES"].keys()):
			self.include_background = False
		else:
			self.include_background = True

		self.inferer = SlidingWindowInferer(
			roi_size=config["INPUT_SHAPE"][ensemble],
			sw_batch_size=config["BATCH_SIZE"][ensemble],
			sw_device=torch.device("cuda"),
			device=torch.device("cpu"),
			overlap=0.5,
			deep_supervision=self.deep_supervision
		)
		
		## Preprocessing for the inference-framework difference.. 
		x = raw["PixelData"]
		if "HU_WEIGHTING" in list(config.keys()) and config["HU_WEIGHTING"][ensemble][0]:
			config["NEW_CH"] = self.load_new_channel_data(
				config, config["HU_WEIGHTING"][ensemble][1], config["HU_WEIGHTING"][ensemble][2], raw["APPLY_TRANSFORM"])
			config["NEW_CH"] = orientation(
				config["NEW_CH"], config["FLIP_XYZ"][0], config["FLIP_XYZ"][1], 
				config["FLIP_XYZ"][2], config["TRANSPOSE"][0]
				)
			x[config["NEW_CH"]>0] = x[config["NEW_CH"]>0] * config["HU_WEIGHTING"][ensemble][3]
			# from utils.util import image_saver
			# image_saver(x, np.expand_dims(mask,axis=0), config["rst_path"], config["case_name"])

		if "ADD_INPUT_CH" in list(config.keys()) and config["ADD_INPUT_CH"][ensemble][0]==True:
			config["NEW_CH"] = self.load_new_channel_data(
				config, config["ADD_INPUT_CH"][ensemble][1], config["ADD_INPUT_CH"][ensemble][2], raw["APPLY_TRANSFORM"])
			config["NEW_CH"] = orientation(
				config["NEW_CH"], config["FLIP_XYZ"][0], config["FLIP_XYZ"][1],
				config["FLIP_XYZ"][2], config["TRANSPOSE"][0])
			# from utils.util import image_saver
			# image_saver(x, np.expand_dims(config["NEW_CH"],axis=0), config["rst_path"], config["case_name"])

			x = self.resampling(x, out_dtype='tensor')
			m = self.resampling(config["NEW_CH"], out_dtype='tensor')
			self.input_image = {"image":x, "mask":m}		
		else:
			x = self.resampling(x, out_dtype='tensor')
			self.input_image = {"image":x}
			if "CONST_LOG" in list(config.keys()) and config["CONST_LOG"][ensemble]==True:
				self.inferer = KAISTSlidingwindowInferer(
			   								image_size=x.shape, 
											roi_size=config["INPUT_SHAPE"][ensemble],
											window=config["CONTRAST"][ensemble],
	 										sw_batch_size=config["BATCH_SIZE"][ensemble], sw_device=torch.device("cuda"),
					   						device=torch.device("cpu"), workers=10)
  
	def load_new_channel_data(self, config, mask_folder=None, mask_file=None, bool_apply_transform=True):		
		if mask_folder is None: mask_folder = config["rst_path"]
		mask_file = os.path.join(mask_folder, f'{mask_file}.nii.gz')
		assert os.path.isfile(mask_file), "Please run 1-step inference!"

		mask = sitk.ReadImage(mask_file)
		mask = sitk.GetArrayFromImage(mask)
		if bool_apply_transform==True:
			from utils.dcm_reader import rotate_forward 
			ch = ct_dict
			ch["PixelData"] = mask
			ch = rotate_forward(
				rot_angles=ch["Return_Angles"], meta_info=ch, reshape=False
			)
			mask = ch["PixelData"]
		return (mask > 0).astype(np.uint8)

	def pre(self):
		if self.invert==True:
			test_keys = list(self.input_image.keys())
			rotate_angle = np.random.randint(-10,10) * 0.005
			scale_factor = float(np.random.choice([8,9,11,12]) * 0.1)
			rescale_factor = float((1 - scale_factor) +1)

			test_transforms = self.test_transforms
			test_transforms += [
				Affined(
					keys=test_keys, rotate_params=(rotate_angle, rotate_angle, 0), 
					shear_params=None, translate_params=None, scale_params=scale_factor)
			]
			invert_transforms = [
				Affined(
					keys=["pred"], rotate_params=(-rotate_angle, -rotate_angle, 0), 
					shear_params=None, translate_params=None, scale_params=rescale_factor)
			]
		else:
			test_transforms = self.test_transforms
			invert_transforms = None
		data = Compose(test_transforms)(self.input_image)
		return data, invert_transforms

	def post(self, data, invert_transforms=None):
		if self.invert==True:
			data = Compose(invert_transforms)(data)
		x = data["pred"]
		del data["pred"]; gc.collect()
		if self.include_background==False:
			x = x[1:]

		assert x.ndim > 3, 'Check the size of model output!'

		if self.model_spacing is not None:
			stime = time.time()
			x = self.resampling(x, revert=True, out_dtype='ndarray', with_channels=True)
			return x
		else:
			return x.numpy()
	
	def inference(self, data):
		with torch.no_grad():
			x = data["image"]
			if x.dim()==3:			
				if "mask" in list(data.keys()):
					x = torch.reshape(x,(1, *x.shape))
					m = data["mask"]
					m = torch.reshape(m,(1, *m.shape))	
					x = torch.stack([x, m], dim=1)
				else:
					x = torch.reshape(x,(1, 1, *x.shape))
			x = self.inferer(inputs=x, network=self.model)[0]  * self.weight
		data["pred"] = self.active(x) if not self.active is None else x
		del x ; gc.collect()
		torch.cuda.empty_cache()
		return data

	def resampling(self, raw, revert=False, out_dtype='tensor',with_channels=False):
		# nearest | linear | bilinear | bicubic | trilinear | area | nearest-exact
		# align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear
		if revert==True:
			mode = self.interp_mode.split('_')[-1]
		else:
			mode = self.interp_mode.split('_')[0]
		if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
			align_corners=True
		else:
			align_corners=None

		if revert : 
			original_shape = raw.shape
			original_spacing = self.model_spacing
			target_spacing = self.original_spacing
			target_shape = self.original_shape
		else:
			original_shape = self.original_shape
			original_spacing = self.original_spacing			
			target_spacing = self.target_spacing			
			target_shape = self.target_shape
			if self.target_shape is None:
				target_shape = list(np.rint(
					np.array(original_shape)*np.array(original_spacing)/np.array(target_spacing)).astype(int))
		if raw is None:	return None
		if mode.isnumeric():
			if torch.is_tensor(raw): raw = raw.numpy()
			zoom = [t/o for t, o in zip(target_shape, original_shape)]
			to_bool = True if len(np.unique(raw))==2 else False
			if raw.ndim==4:
				x = np.empty((raw.shape[0],*target_shape))
				for c in range(raw.shape[0]): # channel-first only!
					x[c] = scipy.ndimage.zoom(raw[c], zoom, order=int(mode), mode='constant', cval=np.min(raw))
			elif raw.ndim==3:
				x = scipy.ndimage.zoom(raw, zoom, order=int(mode), mode='constant', cval=np.min(raw))
			if to_bool==True: x = (x>0.5).astype(np.uint8)
			if out_dtype=='tensor':
				return torch.from_numpy(x).type(torch.FloatTensor)
			else:
				return x

		if not torch.is_tensor(raw):
			x = torch.from_numpy(raw.copy())
		else:
			x = raw.clone().detach()
		if x.ndim==3:
			channel_in = 1
			with_channels = False
		elif x.ndim==4:
			channel_in = x.shape[0] # channel-first only! 
			with_channels = True
		x = torch.reshape(x.type(torch.FloatTensor),(1, channel_in, *x.shape[-3:]))
		x = torch.nn.functional.interpolate(
			x, size=target_shape, mode=mode, align_corners=align_corners)
		if out_dtype=='tensor':
			if with_channels==True:
				return x[0].type(torch.FloatTensor)
			else:
				return x[0,0].type(torch.FloatTensor)
		else:
			if with_channels==True:
				return x.numpy()[0]
			else:
				return x.numpy()[0,0]

def load_image(config, data_dir):
	if '@eaDir' in data_dir: return config, None, None
	if 'cache' in data_dir: return config, None, None
	start_time = time.time()
	# Data Load
	if os.path.isdir(data_dir):
		if "do_transform" in list(config.keys()) and config["do_transform"]==False:
			do_ = False
		else:
			do_ = True
			config["do_transform"] = True
		if "rot_angles" in list(config.keys()) and config["rot_angles"]!=[0,0,0]:
			rot_angles = config["rot_angles"]
			save_new_dicom = True
		else:
			rot_angles = [0,0,0]; save_new_dicom = False
		ct = Hutom_DicomLoader(
			data_dir, do_transform=do_, 
			rot_angles=rot_angles, save_new_dicom=save_new_dicom)
		if ct is None:
			return config, None, None
		else:
			if ct.data is None: return config, None, None
			ct.data["PixelData"] = orientation(ct.data["PixelData"], config["FLIP_XYZ"][0], config["FLIP_XYZ"][1], config["FLIP_XYZ"][2], config["TRANSPOSE"][0])
			config["original_shape"] 		= ct.data["PixelData"].shape
			config["original_spacing"] 		= [*ct.data["PixelSpacing"], ct.data["SliceThickness"]]
			config["original_origin"] 		= ct.data["ImagePosition(Patient)"]
			config["original_direction"]	= ct.data["ImageOrientation(Patient)"]
			config["series_name"] 			= ct.data["series_name"]
			config["case_name"] 			= ct.data["PatientID"]
			config["APPLY_TRANSFORM"]		= ct.data["APPLY_TRANSFORM"]
			if 'Return_Angles' in ct.data.keys():
				config["Return_Angles"]		= ct.data["Return_Angles"]
			load_time = time.time()

			config["rst_path"] = os.path.join(config["output_dir"], config["case_name"])
			config["rst_path"] = os.path.join(config["rst_path"], config["series_name"])
			if not os.path.isdir(config["rst_path"]): 
				os.makedirs(config["rst_path"], exist_ok=True); os.chmod(config["rst_path"], 0o777)

			if config["SAVE_CT"]: saver('image', x, config, max_intensity=1)
			return config, ct.data, {"load":load_time-start_time}
	else:
		print("Input Path should be the path of directory"); return config, None, None


def post_inference(config, image, results, post_transform):	
	if "EVP_ENHANCING" in config.keys() and config["EVP_ENHANCING"][0]==True:
		for c in range(results.shape[0]):
			results[c,config["NEW_CH"]>0] = results[c,config["NEW_CH"]>0] * config["EVP_ENHANCING"][1]

	if config["ARGMAX"]==True:
		bg = np.expand_dims((-sum(results)),0)
		results = np.concatenate((bg,results),axis=0)
		argmax = np.argmax(results, axis=0)
		results = np.eye(results.shape[0])[...,argmax]
		results = results[1:].astype(np.uint8)

	num_classes, save_classes, n_jobs = 0, 0, 0
	if "SAVE_CLASSES" in list(config.keys()):
		num_classes = len(config["SAVE_CLASSES"])
		save_classes = [config["CLASSES"][i] for i in config["SAVE_CLASSES"]]
		t = [results[i-1] for i in config["SAVE_CLASSES"]]
		results = np.stack(t,axis=0)
	else:
		num_classes = len(config["CLASSES"].keys())
		save_classes = list(config["CLASSES"].values())
  
	if num_classes > job_threshold:
		n_jobs = job_threshold
	else:
		n_jobs = num_classes
 
	def process_index(i, this_class, this_image, image, config, post_transform):
		# folder = '/raid/users/mi_pje_0/inference_results/clip_test/01001ug_10/0014_20221017_000000'
		# np.save(os.path.join(folder, this_class+'.npy'), np.round(this_image*255).astype(np.uint8))
		if len(post_transform)>0:
			for func in post_transform:
				if func.apply_labels is None or i in func.apply_labels: 
					if func.image_require==True :
						this_image = func(this_image, image)
					elif func.image_require=='mask':
						this_image = func(this_image, config["NEW_CH"])
					else:
						this_image = func(this_image)
		# np.save(os.path.join(folder, this_class+'_thres.npy'), this_image)
		saver(this_class, this_image, config, max_intensity=255)

	Parallel(n_jobs=n_jobs)(delayed(process_index)(
		i, save_classes[i], results[i], image, config, post_transform) for i in range(num_classes))

	if config["SAVE_MERGE"]: merger(config["CLASSES"], config)


def main(config, raw, model, post_transform) -> None:
	# Inference	
	num_ensemble = len(config["MODEL_NAME"]) if config["MODE"] is not None and 'ensemble' in config["MODE"].lower() else 1
	num_tta = 3 if config["MODE"] is not None and 'tta' in config["MODE"].lower() else 1
	results = None
	start_time = time.time()
	for i in range(num_ensemble):
		if config["WEIGHTS"][i] == 0 : continue
		inferer = do_inference(config, model, raw, i)
		for _ in range(num_tta):
			data, invert_transforms = inferer.pre()
			pre_time = time.time()
			data = inferer.inference(data)
			inf_time = time.time()#;print('e1', i, inf_time-pre_time)
			data = inferer.post(data, invert_transforms)
			post_time = time.time()#;print('e2', i, post_time-inf_time)
			results = add_array(results, data)
			add_time = time.time()#;print('e3', i, add_time-post_time)
		del inferer; gc.collect()
		torch.cuda.empty_cache()
	del model; gc.collect()
	torch.cuda.empty_cache()
 
	if num_ensemble*num_tta > 1:
		results = results / (num_ensemble*num_tta)
	d_time = time.time()#; print('e3', d_time-post_time)
	# Postprocessing & Save Results
	post_inference(config, raw["PixelData"], results, post_transform)
	e_time = time.time()#; print('e4', e_time-d_time)
	print('elapsed time:',
		'\n\t Preprocessing:\t',pre_time-start_time,
		'\n\t Inferece:\t',inf_time-pre_time,
		'\n\t Post-inference:',add_time-inf_time,
		'\n\t Postprocessing:',e_time-add_time,
		'\n Total Processing:\t',e_time-start_time
	)


def save_config_file(log_dir, config_name):
	config_name += '.py'
	file_path = os.path.join(log_dir,config_name)
	copyfile(f'./config/{config_name}', file_path)
