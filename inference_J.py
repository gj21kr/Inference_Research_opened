import os, shutil
import json
import time
import importlib
import torch
import torch.nn as nn
import argparse as ap

from joblib import Parallel, delayed

from utils.util import gen_models, load_saved_model, str2bool, get_underutilized_gpu
from utils.core import main, load_image, save_config_file

if __name__ == "__main__":
	parser = ap.ArgumentParser()
	parser.add_argument('-i', dest='input_path', default=None)        # directory
	parser.add_argument('-o', dest='output_path', default=None)       # directory
	parser.add_argument('-g', dest='gpus', default=None)              # number of GPUs 
	parser.add_argument('-t', dest='target', default=None)            # target inference name
	parser.add_argument('-save_ct', dest='save_ct', default=False)
	parser.add_argument('-r', dest='reverse', default=False)
	args = parser.parse_args()

	args.input_path = os.path.normpath(args.input_path)
	args.output_path = os.path.normpath(args.output_path)

	module = importlib.import_module(f'config.{args.target}')
	config, post_transform = module.config, module.transform

	if os.path.isdir(args.output_path)==False:
		os.makedirs(args.output_path, exist_ok=True); os.chmod(args.output_path, 0o777)
	config["output_dir"] = args.output_path
	config["SAVE_CT"] = str2bool(args.save_ct)

	## Copy Config files 
	save_config_file(args.output_path, args.target)

	## Get pathes
	# Modify here for your condition
	# args.input_path allows to have only dcm directories
	if 'json' in args.input_path:
		# Multi
		with open(args.input_path, 'r') as f:
			args.input_path = json.load(f)
	elif ',' in args.input_path:
		# Multi
		args.input_path = args.input_path.split(',')
	else: 
		# Single
		args.input_path = [args.input_path]

	## GPU setting
	# automatically select GPUs. 
 	# the function, get_underutilized_gpu, considers total number of gpus you 
	if ',' in args.gpus:
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = get_underutilized_gpu(args.gpus)
	use_cuda = torch.cuda.is_available()
	assert use_cuda==True
	config["device"] = torch.device("cuda" if use_cuda else "cpu")

	## Model Load
	models = gen_models(config)
	models = Parallel(n_jobs=len(models))(
		delayed(load_saved_model)(config, i, model) for i, model in enumerate(models))

	for data_dir in sorted(args.input_path, reverse=args.reverse):
		if data_dir[-1]=='/': data_dir=data_dir[:-1]
		## Data Load
		start_time = time.time()
		config, raw, e_time = load_image(config, data_dir)
		e_time = time.time() - start_time
		if raw is None: continue 
		print(config["case_name"])
		main(config, raw, models, post_transform)
		# print('elapsed time for data reading:\n',e_time)
