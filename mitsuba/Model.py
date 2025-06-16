
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
from render_utils.file_util import XML_write
from perc_al.differential_color_functions import deltaE, rgb2lab_diff, ciede2000_diff
# Image denoiser using Siggraph24 Target-Aware Image Denoising
from custom_ops.simple_regression import Regression
from custom_ops.cross_bilateral import GbufferCrossBilateral

bandwidth = 0.002
winSize = 5
regression = Regression(win_size=winSize, bandwidth=bandwidth)
base_cross_bilateral = GbufferCrossBilateral(winSize=winSize)
@dr.wrap_ad(source='drjit', target='torch')
def dr_deltaE_loss(x,y):
    return deltaE(x.permute(2,0,1).unsqueeze(0),y.permute(2,0,1).unsqueeze(0))

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

key = "normal.nested_bsdf.base_color.data"
key1 = "normal.nested_bsdf.roughness.data"
key2 = "prf"
key3 = "crf"
key4 = "normal.normalmap.data"
key5 = "white_balance_coe"
key6 = "normal.nested_bsdf.metallic.data"
key9 = "Projector.scale"
aov_integrator = mi.load_dict({
    'type': 'aov',
    'aovs': 'albedo:albedo, normals:sh_normal, dd.y:depth'
})

def train_DPCS(train_config, data,opt,scene_uncom,params_uncom):
    """
    * \brief : Train the DPCS model for surface plane albedo estimation.
    * This function performs the optimization training process for estimating surface plane albedo.
    * It iterates through several training stages and iterations while rendering scenes using Mitsuba and
    * Dr.Jit, accumulating losses from different denoising and rendering strategies, and updating the parameters
    * with an optimizer.
    * \params :
    *   train_config: A dictionary containing configurations for training, including values such as:
    *                 - num_train: the number of training samples.
    *                 - num_test: the number of testing samples.
    *                 - batch_size: the size of the batch in each iteration.
    *                 - num_stages: the number of training stages.
    *                 - max_iters: the maximum iterations for each stage.
    *                 - lr: the learning rate.
    *                 - lr_drop_iter: the interval at which the learning rate drops.
    *                 - lr_drop_ratio: the ratio for reducing the learning rate.
    *                 - spp: samples per pixel used for rendering.
    *                 - loss: the type of loss ("l2" or "l1", or using SSIM).
    *                 - denoiser_op: specifies the denoising operator to use (e.g., "cross_bilatera" or "OIDN" or "other").
    *                 - masked_require: a flag indicating whether masking is required during training.
    *                 - BRDF_lambda: the weight for the BRDF smooth loss.
    *   data: A dictionary containing various data input for training, such as:
    *         - prj_train_list: list of irradiance inputs for projector estimation.
    *         - cam_train_list: list of camera images for constructing the loss.
    *         - im_mask: an optional image mask used in the loss computation.
    *   opt: An optimizer or a dictionary of parameters used for the optimization process.
    *        It holds values to be updated (e.g., learning rates, camera and projector response function
    *        parameters) and provides methods to update these values during training.
    *   scene_uncom: The scene representation used by the rendering engine (Mitsuba) to generate images.
    *   params_uncom: A container for the scene and sensor parameters, including the transformation from sensor
    *                 to world coordinates. This is updated throughout training to refine the scene's properties.
    * \output :
    *   params_uncom: The updated scene and sensor parameters after training.
    *   scene_uncom: The updated scene representation after training.
    """
    losses = []
    initial_to_world = params_uncom["sensor.to_world"]
    print("Start optimization for surface plane albedo .... ")
    batch_size = min(train_config['batch_size'],4)
    train_config['batch_size'] = batch_size
    num_train = data['num_train']
    for stage in range(train_config["num_stages"]):
        print()
        print("--------------------------------Stat stage %d learning--------------------------------------" % stage)
        for it in range(train_config["max_iters"]):
            if (it % train_config["lr_drop_iter"] == 0 & it!=0):
                opt.set_learning_rate(train_config["lr"] * (1 - train_config["lr_drop_ratio"]))
                opt.set_learning_rate({"angle": 0.000000020, "trans": 0.000000038})
            loss_dr_accum = 0
            ref_num = 0
            num_batch = int(it % int(num_train / batch_size+1))
            for batch_id in range(batch_size):
                seed_f = np.random.randint(2 ** 31)
                seed_b = np.random.randint(2 ** 31)
                #%% shuffle batch in training set
                #idx = random.sample(range(num_train),1)[0]
                idx = (batch_id + num_batch * batch_size) % num_train
                # insure a valid gamma projector response function
                opt[key2] = dr.clamp(opt[key2], 2, 3)
                params_uncom["Projector.irradiance.data"] = PRF(data['prj_train_list'][idx], opt[key2])
                #%%
                params_uncom.update()
                img = mi.render(scene_uncom, params_uncom, spp=train_config['spp'], seed = seed_f, seed_grad=seed_b)
                # insure a valid gamma camera response function
                img_wb = White_Balance_Camera(opt[key5], img)
                opt[key3] = dr.clamp(opt[key3], 1 / 3, 1)
                img_warped = dr.clamp(img_wb,0,1)
                image = CRF(img_warped, opt[key3])  # this is for camera response estimation
                if (train_config["denoiser_op"] =="cross_bilatera"):
                    with dr.suspend_grad():
                        aovs = mi.render(scene_uncom, params_uncom, integrator=aov_integrator, spp=train_config["spp"],
                                         spp_grad=train_config["spp"], seed=seed_f, seed_grad=seed_b)
                        albedo = aovs[:, :, 0:3]
                        normal = aovs[:, :, 3:6]
                        depth = aovs[:, :, 6:7]
                        albedo = dr.clamp(albedo, 0.0, 1.0)
                        max_depth = dr.max(depth)
                        depth /= max_depth
                    image_denoised = torch_ops.base_cross_bilateral_op(base_cross_bilateral, image, albedo, normal,
                                                    depth)
                else:
                    image_denoised = image
                if train_config["masked_require"]:
                    image = dr.clamp(mask_dr(image_denoised, data['im_mask']), 0, 1)
                else:
                    image = dr.clamp(image_denoised,0,1)
                #smooth loss on BRDF
                loss_BRDF = train_config["BRDF_lambda"] *  ( #tv_loss(depth= opt[key]) +
                                             tv_loss(depth= TensorXf(dr.repeat(opt[key1].array,3),
                                                                      shape = (opt[key1].shape[0],opt[key1].shape[1],3)))
                                             +tv_loss(depth=opt[key])
                )
                if train_config['loss'] == "l2":
                    loss_dr = dr.sum(dr.sqr(image - data['cam_train_list'][idx])) / len(image) + loss_BRDF
                elif train_config['loss'] == "l1":  # l1 loss
                    loss_dr = dr.mean(dr.abs(image - data['cam_train_list'][idx]))+ loss_BRDF
                else:
                    loss = ssim_loss(image, data['cam_train_list'][idx], .5, .5)+ loss_BRDF
                dr.backward(loss_dr)
                loss_dr_accum += loss_dr[0]
                ref_num += 1
            losses.append(loss_dr_accum)
            opt.step()
            #%%
            # Insure surface material map at a valid range
            apply_transformation(params_uncom, opt,initial_to_world)
            opt[key] = dr.clamp(opt[key], 0, 1)
            opt[key1] = dr.clamp(opt[key1], 0, 1)
            opt[key4] = constraint_normal(opt[key4])
            opt[key5]=dr.clamp(opt[key5],0.2,2.5)
            opt[key6] = dr.clamp(opt[key6], 0, 1)
            params_uncom.update(opt)
            print(f"  --> iteration {it + 1:02d}: error={loss_dr_accum:6f}")
        print()
        print("-------------------------------------stage %d learning is ending------------------------------" % stage)
        print()
        params_uncom.update(opt)
    print('')
    print('Done')
    print()
    print("Optimization result shows...")
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('MSE(param)')
    plt.title('Parameter error plot')
    plt.show()
    return params_uncom, scene_uncom, opt

def relight_DPCS(train_config, data,opt,scene_uncom,params_uncom,params_path,relit_folder_path,cam_valid):
    """
    * \brief : Relight simulation for DPCS model and compute rendering quality metrics.
    * This function performs a test simulation by rendering relighted images using Mitsuba with the given
    * scene and parameter configuration.  optionally applies a denoising operator, and saves the resulting images. Then,
    * it computes quality metrics (PSNR, RMSE, SSIM) by comparing simulated images with camera validation data.
    * the function writes out BRDF maps (albedo, roughness, metallic, normal) to disk.
    * \params :
    *   train_config : A dictionary containing configurations for testing and simulation, including:
    *                  - num_test: Number of test images to render.
    *                  - batch_size: Batch size for each simulation (limited to a maximum of 4).
    *                  - Testspp: Samples per pixel for test rendering.
    *                  - is_metric: Flag indicating whether to perform metric validation.
    *                  - loss: The loss type used in training (used in naming the output files).
    *                  - num_train: Number of training samples (used in naming the output files).
    *                  - max_iters: Maximum iterations considered in training (used in naming the output files).
    *                  - spp: Samples per pixel used during training (used in naming the output files).
    *   data : A dictionary storing input and auxiliary data for simulation and validation, such as:
    *          - prj_valid_list: A list containing projector irradiance data for simulation.
    *          - im_mask: An image mask applied during the rendering process.
    *          - im_mask_torch: The mask in tensor format, used for metric computation.
    *          - backlight: camera captured projector black level light
    *   opt : updated optimizer parameters during training stage
    *   scene_uncom : The uncompiled scene representation used by Mitsuba to render the images.
    *   params_uncom : A container for uncompiled scene parameters
    *   params_path : The directory path where BRDF and albedo maps (e.g., roughness, metallic, normal) will be saved.
    *   relit_folder_path : The directory path where the relighting simulation output images will be stored.
    *   cam_valid : A tensor containing validation camera images used for quality metric computations.
    * \output : No return
    """
    # %% Test simulation
    num_test = train_config['num_test']
    batch_size = min(train_config['batch_size'], 4)
    if not os.path.exists(relit_folder_path):
        os.makedirs(relit_folder_path)
    with dr.suspend_grad():
        if train_config['is_metric']:
            print("render all simulation data set")
            for idx in range(num_test):
                seed_f = np.random.randint(2 ** 31)
                seed_b = np.random.randint(2 ** 31)
                params_uncom["Projector.irradiance.data"] = PRF(data['prj_valid_list'][idx], opt[key2])
                params_uncom.update()
                print("start render image_{:04d}.png".format(idx + 1))
                img = mi.render(scene_uncom, params_uncom, spp=train_config["Testspp"], seed = seed_f, seed_grad=seed_b)
                img_wb = White_Balance_Camera(opt[key5], img)
                img_warped = dr.clamp(img_wb,0,1)
                #img_warped = img_wb
                image = CRF(img_warped,opt[key3])+TensorXf(data["backlight"].squeeze().permute(1,2,0).numpy())
                if (train_config['denoiser_op'] == "cross_bilatera"):
                    with dr.suspend_grad():
                        aovs = mi.render(scene_uncom, params_uncom, integrator=aov_integrator, spp=train_config['Testspp'],
                                         spp_grad=train_config['Testspp'], seed=seed_f, seed_grad=seed_b)
                        albedo = aovs[:, :, 0:3]
                        normal = aovs[:, :, 3:6]
                        depth = aovs[:, :, 6:7]
                        albedo = dr.clamp(albedo, 0.0, 1.0)
                        max_depth = dr.max(depth)
                        depth /= max_depth
                    image_denoised = torch_ops.base_cross_bilateral_op(base_cross_bilateral, image, albedo, normal,
                                                                       depth)
                else:
                    image_denoised = image
                if train_config['masked_require']:
                    image = dr.clamp(mask_dr(image_denoised, data['im_mask']), 0, 1)
                else:
                    image = dr.clamp(image_denoised, 0, 1)
                image = mi.Bitmap(image)
                image.set_srgb_gamma(True)
                mi.util.write_bitmap(fullfile(relit_folder_path, 'img_{:04d}.png'.format(idx + 1)), image)

        ########################### Metric validation shows #################################################
        # ensure before read image, the image should be saved in the folder
        time.sleep(5)
        simu_test = readImgsMT(relit_folder_path)
        for i in range(num_test):
            img = cam_valid[i, :, :, :].permute((1, 2, 0)) * data['im_mask_torch'].unsqueeze(2).cpu()
            cam_valid[i, :, :, :] = torch.tensor(img).permute((2, 0, 1))
        for i in range(num_test):
            img = simu_test[i, :, :, :].permute((1, 2, 0)) * data['im_mask_torch'].unsqueeze(2).cpu()
            simu_test[i, :, :, :] = torch.tensor(img).permute((2, 0, 1))

        simu_psnr, simu_rmse, simu_ssim = computeMetrics(cam_valid[:num_test, :, :, :], simu_test[:num_test, :, :, :])
        print("valid_psnr:")
        print(simu_psnr)
        print()
        print("valid_ssim:")
        print(simu_ssim)
        print()
        print("valid_rmse:")
        print(simu_rmse)
        write_metrics(
            f"Mitsuba3_{train_config['loss']}_{data['num_train']}_{batch_size}_{train_config['max_iters']}_spp={train_config['spp']}",
            simu_psnr, simu_ssim, simu_rmse, os.path.dirname(os.path.dirname(relit_folder_path)))
        #write BRDF to params_path
        albedo=write_albedo(opt[key3],opt[key5],params_path,opt[key])
        roughness=write_BRDF(opt[key1], fullfile(params_path,"roughness.png"))
        metallic=write_BRDF(opt[key6], fullfile(params_path,"metallic.png"))
        map = mi.Bitmap(opt[key4])
        map.set_srgb_gamma(True)
        mi.util.write_bitmap(fullfile(params_path, "normal.png"), map)
        del simu_test,image,image_denoised,img_wb,img_warped,albedo,roughness,metallic,map,img

def Compensate_DPCS(train_config, data, opt, scene_uncom, params_uncom, cmp_test_path, im_mask, num_train, batch_size):
    integrator = dict({
        'type': "prb_reparam",
        'max_radiance': train_config['scale'] * 5,  # times 5 as min(w) = 0.2
        'max_depth': 4 #simple direct light mainly scene use less depth can be well to save time, for complex scene, it should be larger
        # e.g. like training stage use max_depth = 36
    })
    integrator = mi.load_dict(integrator)

    print()
    print(
        "------------------------------------------Start compensation:--------------------------------------------------------")
    print()
    if train_config["requried_compensation"]:
        cmp_folder_path = os.path.join(cmp_test_path,
                                       f"Mitsuba3_{train_config['loss']}_{num_train}_{batch_size}_{train_config['max_iters']}")
        if not os.path.exists(cmp_folder_path):
            os.makedirs(cmp_folder_path)
        for idx in range(0, 5):  # for every test image render first
            print()
            print(
                "----------------------------------------start compensate test image_{:04d}---------------------------------------".format(
                    idx + 1))
            print()
            pro_in = TensorXf(data['prj_ref'][2, :, :, :].permute(1, 2, 0).cpu().numpy())
            params_uncom["Projector.irradiance.data"] = PRF(pro_in, opt[key2])
            params_uncom.update()
            opt_cmp = mi.ad.Adam(.01)
            opt_cmp["Projector.irradiance.data"] = params_uncom["Projector.irradiance.data"]
            # for every rendered image start from that simulation to do compensation task
            losses = []
            for it in range(500):
                if (it > 400):
                    opt_cmp.set_learning_rate(.002)
                seed_f = np.random.randint(2 ** 31)
                seed_b = np.random.randint(2 ** 31)
                img = mi.render(scene_uncom, params_uncom, spp=train_config['spp'], seed=seed_f, seed_grad=seed_b,
                                integrator=integrator)
                img_wb = White_Balance_Camera(opt[key5], img)
                img_warped = dr.clamp(img_wb, 0, 1)
                image = CRF(img_warped, opt[key3])
                with dr.suspend_grad():
                    aovs = mi.render(scene_uncom, params_uncom, integrator=aov_integrator, spp=train_config['spp'],
                                     spp_grad=train_config['spp'], seed=seed_f, seed_grad=seed_b)
                    albedo = aovs[:, :, 0:3]
                    normal = aovs[:, :, 3:6]
                    depth = aovs[:, :, 6:7]
                    albedo = dr.clamp(albedo, 0.0, 1.0)
                    max_depth = dr.max(depth)
                    depth /= max_depth
                image_denoised = torch_ops.base_cross_bilateral_op(base_cross_bilateral, image, albedo, normal,
                                                                   depth)
                if train_config['masked_require']:
                    image = dr.clamp(mask_dr(image_denoised, im_mask), 0, 1)
                else:
                    image = dr.clamp(image_denoised, 0, 1)
                # l1_smooth_loss for compensation or l1 loss
                # l1_smooth may get less outlier for the compensated image I_p^*
                if train_config['compensation_loss'] == "l1":
                    loss = dr.mean(dr.abs(image - data['desire_img_list'][idx]))
                elif train_config['compensation_loss'] == "l1_smooth":
                    loss = Huber_loss(image, data['desire_img_list'][idx])
                dr.backward(loss)
                opt_cmp.step()
                opt_cmp["Projector.irradiance.data"] = dr.clamp(opt_cmp["Projector.irradiance.data"], 0, 1)
                params_uncom.update(opt_cmp)
                losses.append(loss[0])
                print(f"Iteration {it:02d}: error={loss[0]:6f}")
            tst = mi.Bitmap(image)
            tst.set_srgb_gamma(True)
            mi.util.write_bitmap(fullfile(cmp_folder_path, 'simu_img_{:04d}.png'.format(idx + 1)), tst)
            prj_irradiance = params_uncom["Projector.irradiance.data"].torch()
            inverse_gamma = {}
            inverse_gamma[key2] = 1 / opt[key2]
            prj_sRGB = mi.Bitmap(PRF_cmp(TensorXf(prj_irradiance), inverse_gamma[key2]))
            prj_sRGB.set_srgb_gamma(True)
            mi.util.write_bitmap(fullfile(cmp_folder_path, 'img_{:04d}.png'.format(idx + 1)), prj_sRGB)
            plt.plot(losses)
            plt.show()
            del prj_irradiance, prj_sRGB,image,img_wb,img,img_warped,image_denoised
