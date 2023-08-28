from __future__ import annotations

import os, sys

from monai.data import *

__all__=[
	"call_model"
 ]

def call_model(config):
	model = None
	model_name = config["MODEL_NAME"].lower()
	if model_name in [
    	'clip_universal_unet','clip_universal_swinunetr',
    	'clip_universal_dints','clip_universal_unetpp']:
		backbone = model_name.split('_')[-1]
		from core.CLIPuniversal import Universal_model
		model = Universal_model(
			img_size=config["INPUT_SHAPE"],
			in_channels=config["CHANNEL_IN"],
			out_channels=config["CHANNEL_OUT"],
			backbone=backbone,
			encoding='word_embedding'
		)
	elif model_name=='er_net':
		pass
	elif model_name=='unest':
		from core.UNEST import UNesT
		model = UNesT(
			in_channels = config["CHANNEL_IN"],
			out_channels = config["CHANNEL_OUT"],
			img_size = config["INPUT_SHAPE"],
			feature_size = config["FEATURE_SIZE"],
			patch_size = config["PATCH_SIZE"],
		)
	elif model_name=='swin_unetr':
		from monai.networks.nets import SwinUNETR
		model = SwinUNETR(
			img_size=config["INPUT_SHAPE"],
			in_channels=config["CHANNEL_IN"],
			out_channels=config["CHANNEL_OUT"],
			feature_size=config["FEATURE_SIZE"],
			use_checkpoint=False,
		)
	elif model_name in ['unetr']:
		from monai.networks.nets import UNETR
		model = UNETR(
			in_channels = config["CHANNEL_IN"],
			out_channels = config["CHANNEL_OUT"],
			img_size = config["INPUT_SHAPE"],
			feature_size = config["PATCH_SIZE"],
			hidden_size = config["EMBED_DIM"],
			mlp_dim = config["MLP_DIM"],
			num_heads = config["NUM_HEADS"],
			pos_embed="perceptron",
			norm_name="instance",
			res_block=True,
		)
	elif model_name in ['vnet', 'v-net']:
		from monai.networks.nets import VNet
		model = VNet(
			spatial_dims=3,
			in_channels=config["CHANNEL_IN"],
			out_channels=config["CHANNEL_OUT"],
		)
	elif model_name in ['unet']:
		from monai.networks.nets import UNet
		model = UNet(
			spatial_dims=3,
			in_channels=config["CHANNEL_IN"],
			out_channels=config["CHANNEL_OUT"],
			channels=config["CHANNEL_LIST"], #(32, 64, 128, 256, 512),
			strides=config["STRIDES"], #(2, 2, 2, 2),
			num_res_units=config["NUM_RES_UNITS"], #2,
		)

	assert model is not None, 'Model Error!'    
	return model
