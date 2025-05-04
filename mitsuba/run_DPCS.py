import sys
sys.path.append('../')
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import drjit as dr
import mitsuba as mi
from utils import *
from utils.fov import fov
import cv2
from utils.fov import intrinsic2fov_x
import colour
import torch_ops
mi.set_variant("cuda_ad_rgb")
from mitsuba import TensorXf
import torch
import copy
import time
from utils.cali import *
from utils.utils import *
from utils.validation import *
from utils.ImgProc import *
import torchvision.transforms as transforms
import pytorch_ssim
import torch.nn.functional as F
import torch.nn
from Model import *
from render_utils.file_util import XML_write
from perc_al.differential_color_functions import deltaE, rgb2lab_diff, ciede2000_diff
# Image denoiser using Siggraph24 Target-Aware Image Denoising
from custom_ops.simple_regression import Regression
@dr.wrap_ad(source='drjit', target='torch')
def dr_deltaE_loss(x,y):
    return deltaE(x.permute(2,0,1).unsqueeze(0),y.permute(2,0,1).unsqueeze(0))

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model_name = "DPCS"
import json
# -------------------------------------------------------file loader-----------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='Dataset path configuration')
parser.add_argument('--dataset_root', type=str, required=True,
                    help='specify the dataset_root')
args = parser.parse_args()

dataset_root = args.dataset_root
with open(fullfile(dataset_root,"sets_up","config.json"), 'r') as file:
    config = json.load(file)

for data_name in config["sets_up"]:
    data_list = [f'sets_up/{data_name}']
    print("start train--------------------------"+data_name+"-----------------------------------")
    print()
    print()
    print("---------------------------------------------------------------------------------")

    checkDataList(dataset_root, data_list)
    data_root = fullfile(dataset_root, data_list[0])
    params_path = fullfile(data_root, 'params')
    cam_ref_path = fullfile(data_root, 'cam/raw/ref')
    cam_train_path = fullfile(data_root, 'cam/raw/train')
    prj_train_path = fullfile(dataset_root, 'train')
    cam_valid_path = fullfile(data_root, 'cam/raw/test')
    prj_valid_path = fullfile(dataset_root, 'test')
    prj_ref_path = fullfile(dataset_root, "ref")
    checkr_path = fullfile(dataset_root, "checkr")
    pred_path = fullfile(data_root, "pred")
    mesh_path = fullfile(data_root, "mesh")
    desired_test_path = fullfile(data_root, "cam/desire/test")
    cmp_test_path = fullfile(data_root, "prj/cmp/test")

    #%%
    # -----------------------Train_config---------------------------
    # DPCS currently does not model environment light and projector backlight
    # if the projector backlight light (camera captured black projection image) was high
    # subtract it from the captured image and add it during infer
    train_config ={
        "is_metric":True,# whether to inference and evaluate the model
        "denoiser_op" : "cross_bilatera", # [op=["cross_bilatera","TARGET","OIDN"]]
        "scale": config["sets_up"][data_name]["scale"],
        # training sampling per pixel higher rendering image will have high quality while may waste too many times
        "spp" : 16,
        "Testspp":2070 ,
        "data_name":data_name,
        "model_name":model_name,
        "is_in_direct" : config["sets_up"][data_name]["is_in_direct"],# using direct light mask or in-direct light mask,
        "requried_compensation":False,
        "loss":"l1",# training loss for train a virtual ProCams (l1,l2,ssim)
        "compensation_loss":"l1_smooth",# compensation loss for compensation stage (l1,l1_smooth)
        "num_train_list":[100,50,15,5],
        # num of sampling to render update scene parameters per-iteration.
        "batch_size":4,
        "max_iters":100,
        # whether need to subtract the backlight from the captured image to ensure that
        # the rendered projector black image is the same as the captured image (no backlight in render)
        "backlight_sub":True,
        # whether need to mask the ROI of the image before training, note that
        # the indirect light mask may not be accurate, in that case free the mask, otherwise some area may be lost
        # but for baseline model like DeProCams, the direct light mask is required for better performance
        "masked_require":config['sets_up'][data_name]["masked_require"],
        "num_stages":1,
        "num_test":100,
        "lr":0.02,
        "num_cmp":5,
        "lr_drop_ratio":0,
        "lr_drop_iter":100,
        "benchmark":False,
        "Threshold":2,
        "l2_reg":0.02,
        "params_path": params_path,
        "data_list":data_list,
        "BRDF_lambda":0.02,
        # path depth when,tracing increase can improve indirect light contribution but may suffer computation time
        # effort actually most of the simple direct light main scene use 2 is enough, while use 3 or more can process more complex
        "max_depth" : config["sets_up"][data_name]["max_depth"],
    }
    # ----------------------------------------------------------------------------------------------------------------------
    cali_data = loadCalib(fullfile(params_path, "params.yml"))
    cali_data["cam_w"] = 640
    cali_data["cam_h"] = 360
    cali_data["prj_w"] = 600
    cali_data["prj_h"] = 600
    cx = -int(cali_data["prjK"].squeeze().numpy()[0, 2] - 400)
    cy = -int(cali_data["prjK"].squeeze().numpy()[1, 2] - 300)
    cali_data["prjK"][0,0,0] = cali_data["prjK"][0,0,0]*600/600
    cali_data["prjK"][0,0,2] = (cali_data["prjK"][0,0,2]-100)*600/600
    cali_data["prjK"][0,1,1]  = cali_data["prjK"][0,1,1] *600/600
    cali_data["prjK"][0,1,2] = cali_data["prjK"][0,1,2] *600/600
    # -----------------------------------------load image-----------------------------------------------
    print()
    print(
        "-----------------------------------------------------Start Data Loader----------------------------------------------------")
    print()
    # mask image using camera-captured checkerboard Nayar's TOG'06 method (also see Moreno 3DV'12)
    im_mask,im_mask_torch = compute_im_masks(cam_ref_path,data_root,train_config)

    if train_config["backlight_sub"]:
        backlight = readImgsMT(cam_ref_path, index=[0])
    else:
        backlight = readImgsMT(cam_ref_path, index=[0])*0

    for num_train in train_config["num_train_list"]:
        num_test = train_config["num_test"]
        prj_train = readImgsMT(prj_train_path)[:num_train, :, :, :]
        cam_train = readImgsMT(cam_train_path)[:num_train, :, :, :] - backlight
        cam_train[cam_train<0] = 0
        prj_valid = readImgsMT(prj_valid_path)[:num_test, :, :, :]
        cam_valid = readImgsMT(cam_valid_path)[:num_test, :, :, :]
        cam_ref = readImgsMT(cam_ref_path)[:,:,:,:]
        try:
            if not train_config['benchmark']:
                cam_desire = readImgsMT(desired_test_path)[:num_test, :, :, :] - backlight
                cam_desire[cam_desire < 0] = 0
            else:
                cam_desire = cam_valid - backlight
                cam_desire[cam_desire < 0] = 0
        except:
            print("No desire image found")
        prj_ref = readImgsMT(prj_ref_path)
        if train_config["masked_require"]:
            for i in range(num_train):
                img = cam_train[i,:,:,:].permute((1,2,0)) * im_mask_torch.unsqueeze(2).cpu()
                cam_train[i, :, :, :] = torch.tensor(img).permute((2, 0, 1))
            for i in range(num_test):
                img = cam_valid[i, :, :, :].permute((1, 2, 0)) * im_mask_torch.unsqueeze(2).cpu()
                cam_valid[i, :, :, :] = torch.tensor(img).permute((2, 0, 1))
        im_mask = TensorXf(im_mask_torch.unsqueeze(2).cpu().numpy())
        # convert all data to CUDA tensor if you have sufficient GPU memory (faster), otherwise comment them
        print()
        print(
            "------------------------------------------------All load images from files--------------------------------------------------")

        print(dr.device())
        print("Waiting For Scene Establishing ......")
        # %% establish scene
        # load xml scene file
        # init the extrinbsics and intrainsics of Projector and Camera Systems
        XML_write(cali_data, "./scene/simu.xml", dataset_root,train_config)
        scene_uncom = mi.load_file("./scene/simu.xml")
        params_uncom = mi.traverse(scene_uncom)
        WB = mi.Vector3f(0.533040463924408, 0.5640742778778076, 0.7095497846603394)
        input = prj_valid[3, :, :, :].permute(1,2,0)
        params_uncom["Projector.irradiance.data"] = TensorXf(input.numpy())
        # extrinsics of projector updates
        import numpy as np
        proRT = torch.squeeze(cali_data["prjRT"]).numpy()
        invM = np.linalg.inv(proRT)
        tmat = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        invM = np.matmul(invM, tmat)
        params_uncom["Projector.to_world"] = mi.cuda_ad_rgb.Transform4f(invM)
        params_uncom.update()
        img = mi.render(scene_uncom, params_uncom, spp=1024)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        print("Scene has been established!!! Initialization rendered result is shown in matplotlib")

        #%% optimizer
        opt = mi.ad.Adam(train_config["lr"])
        opt[key] = params_uncom[key]
        opt[key1] = params_uncom[key1]
        opt[key2] = mi.Vector3f(2.1476354598999023, 2.0160303115844727, 2.069981336593628)
        opt[key3] = mi.Vector3f(0.5589925050735474, 0.5731906294822693, 0.5018266439437866)
        opt[key4] = params_uncom[key4]
        opt[key5] = WB
        opt[key6] = params_uncom[key6]
        opt['angle'] = mi.Float(0)
        opt['trans'] = mi.Point3f(0, 0, 0)
        opt.set_learning_rate({"angle": 0.000000020, "trans": 0.000000038})
        params_uncom.update(opt)
        #%% convert into TensorXf backend of training and valid dataset:
        cam_train_list = []
        prj_train_list = []
        cam_valid_list = []
        prj_valid_list = []
        desire_img_list = []
        for i in range(train_config["num_cmp"]):
            desire_img_list.append(TensorXf(cam_desire[i, :, :, :].permute(1, 2, 0).numpy()))
        for i in range(num_train):
            cam_train_list.append(TensorXf(cam_train[i, :, :, :].permute(1, 2, 0).numpy()))
            prj_train_list.append(TensorXf(prj_train[i, :, :, :].permute(1, 2, 0).numpy()))
        for i in range(num_test):
            cam_valid_list.append(TensorXf(cam_valid[i, :, :, :].permute(1, 2, 0).numpy()))
            prj_valid_list.append(TensorXf(prj_valid[i, :, :, :].permute(1, 2, 0).numpy()))
        data = {
            "cam_train_list": cam_train_list,
            "prj_train_list": prj_train_list,
            "cam_valid_list": cam_valid_list,
            "prj_valid_list": prj_valid_list,
            "im_mask": im_mask, # mask used either for training (if mask can be well using checkerboard algorithm)
            # or only for metric calculation (mask not well for indirect light just use for validation# ) drjit backend mask
            "im_mask_torch":im_mask_torch, # torch backend im_mask
            "backlight": backlight,# camera captured scene when projector project pure black image e.g.
            #  project ref/img_0000.png -----------> get the cam/raw/ref/img_0000.png
            "desire_img_list": desire_img_list, # Desired Image for Projector Compensation
            "prj_ref": prj_ref, # ref image to project (black white and gray)
            "num_train": num_train,
        }

        #%% Train a virtual single view ProCams setup
        params_uncom, scene_uncom,opt = train_DPCS(train_config,data, opt, scene_uncom, params_uncom)
        # %% Test simulation
        relit_folder_path = os.path.join(pred_path,
                                         f"relit/test/DPCS_{train_config['loss']}_{num_train}_{train_config['batch_size']}_{train_config['lr']}_{train_config['max_iters']}_spp{train_config['spp']}")

        relight_DPCS(train_config, data,opt,scene_uncom,params_uncom,params_path,relit_folder_path,cam_valid)
        cam_desire = cam_desire.cuda()
        # ----------------------------------------------Compensation image calculation----------------------------------------
        # %% compensation image calculation
        if train_config['requried_compensation']:
            Compensate_DPCS(train_config, data, opt, scene_uncom, params_uncom, cmp_test_path, im_mask, num_train,
                            train_config['batch_size'])
        del prj_train, cam_train, prj_valid, cam_valid, cam_desire, prj_ref, scene_uncom, params_uncom,data,opt
        del desire_img_list, cam_train_list, prj_train_list, cam_valid_list, prj_valid_list
        torch.cuda.empty_cache()

