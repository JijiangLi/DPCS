
import os
from os.path import join as fullfile
import platform
import numpy as np
import cv2 as cv
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from skimage.filters import threshold_multiotsu
from utils.validation import readImgsMT

import pytorch_ssim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import yaml
import drjit as dr
import mitsuba as mi
from drjit.cuda.ad import UInt32
from drjit.cuda.ad import Int32
from mitsuba import TensorXf

#insure that the size is assign to your data
cam_size=(360,640,3)
prj_size=(600,600,3)
cmp_size = (600,600,3)
BRDF_size = (2400,3200, 3)

# check if each dataset has valid images and parameters
def checkDataList(dataset_root, data_list):
    for data_name in data_list:
        data_full_path = fullfile(dataset_root, data_name)
        assert os.path.exists(data_full_path), data_full_path + ' does not exist\n'

        param_file = fullfile(data_full_path, 'params/params.yml')
        assert os.path.exists(param_file), param_file + ' does not exist\n'









"""
* \brief : Calculate one-dimensional indices for a specific channel in a 3D tensor.
* This function is used to convert 3D indices (height, width, channel) of a tensor 
* into a flattened 1D array of indices. It's particularly useful for operations like 
* scatter where you need to specify the exact positions in a flattened array for each 
* channel of a 3D tensor.
* \params : 
*   height: The height of the tensor (number of rows).
*   width: The width of the tensor (number of columns).
*   depth: The depth of the tensor (number of channels, typically 3 for RGB).
*   channel: The specific channel (0 for Red, 1 for Green, 2 for Blue) for which 
*            the indices are being calculated.
* \output :
*   indices: A flattened 1D numpy array containing the calculated indices for the 
*            specified channel.
"""
def calculate_indices_for_channel(height, width, depth, channel):
    i_indices = np.arange(height, dtype=np.uint32).reshape(height, 1) * np.uint32(width * depth)
    j_indices = np.arange(width, dtype=np.uint32) * np.uint32(depth)
    indices = i_indices + j_indices + np.uint32(channel)
    return indices.flatten()

    # Calculate indices for each channel

height, width, depth = cam_size
c_indices_r = UInt32(calculate_indices_for_channel(height, width, depth, 0)) # Red channel
c_indices_g = UInt32(calculate_indices_for_channel(height, width, depth, 1)) # Green channel
c_indices_b = UInt32(calculate_indices_for_channel(height, width, depth, 2))  # Blue channel
ctst = TensorXf(np.random.rand(height, width, depth))
height, width, depth = prj_size
p_indices_r = UInt32(calculate_indices_for_channel(height, width, depth, 0)) # Red channel
p_indices_g = UInt32(calculate_indices_for_channel(height, width, depth, 1)) # Green channel
p_indices_b = UInt32(calculate_indices_for_channel(height, width, depth, 2))  # Blue channel
ptst = TensorXf(np.random.rand(height, width, depth))
height, width, depth = cmp_size
cmp_indices_r = UInt32(calculate_indices_for_channel(height, width, depth, 0)) # Red channel
cmp_indices_g = UInt32(calculate_indices_for_channel(height, width, depth, 1)) # Green channel
cmp_indices_b = UInt32(calculate_indices_for_channel(height, width, depth, 2))  # Blue channel
height, width, depth = BRDF_size
BRDF_indices_r = UInt32(calculate_indices_for_channel(height, width, depth, 0)) # Red channel
BRDF_indices_g = UInt32(calculate_indices_for_channel(height, width, depth, 1)) # Green channel
BRDF_indices_b = UInt32(calculate_indices_for_channel(height, width, depth, 2))  # Blue channel
BRDFtst = TensorXf(np.random.rand(height, width, depth))



"""
* \brief : Apply a response function to an irradiance tensor using Dr.Jit and Mitsuba.
* This function takes an irradiance tensor, applies a power function to each color 
* channel based on provided options, and scatters the results back into the tensor.
* It's used in rendering and image processing tasks where modifications to the 
* irradiance values based on certain parameters are required.
* \params : 
*   Irradiance_torch: A tensor representing the irradiance values to be processed.
*   opt: A dictionary containing options that dictate how the response function 
*        should be applied to each channel.
*   key: The key to access the relevant values in the 'opt' dictionary.
* \output :
*   Irradiance: The modified irradiance tensor after applying the response function 
*               and scattering the values to their respective channels.
"""

def White_Balance_Camera(WB, img):
    dr.scatter(ctst.array, WB[0], c_indices_r)
    dr.scatter(ctst.array, WB[1], c_indices_g)
    dr.scatter(ctst.array, WB[2], c_indices_b)
    return ctst * img
def White_Balance_Projector(WB, img):
    dr.scatter(ptst.array, WB[0], p_indices_r)
    dr.scatter(ptst.array, WB[1], p_indices_g)
    dr.scatter(ptst.array, WB[2], p_indices_b)
    return ptst * img
def White_Balance_BRDF(WB, img):
    dr.scatter(BRDFtst.array, WB[0], BRDF_indices_r)
    dr.scatter(BRDFtst.array, WB[1], BRDF_indices_g)
    dr.scatter(BRDFtst.array, WB[2], BRDF_indices_b)
    return BRDFtst * img
def CRF( Irradiance_torch ,gamma):
    tst=Irradiance_torch
    Irradiance=TensorXf(Irradiance_torch)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 0]), gamma[0]).array, c_indices_r)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 1]), gamma[1]).array, c_indices_g)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 2]), gamma[2]).array, c_indices_b)
    return Irradiance

def PRF( Irradiance_torch , gamma ):
    tst=Irradiance_torch
    Irradiance=TensorXf(Irradiance_torch)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 0]), gamma[0]).array, p_indices_r)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 1]), gamma[1]).array, p_indices_g)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 2]), gamma[2]).array, p_indices_b)
    return Irradiance

def PRF_cmp( Irradiance_torch , gamma):
    tst=Irradiance_torch
    Irradiance=TensorXf(Irradiance_torch)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 0]), gamma[0]).array, cmp_indices_r)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 1]), gamma[1]).array, cmp_indices_g)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 2]), gamma[2]).array, cmp_indices_b)
    return Irradiance

def PRF_BRDF( Irradiance_torch , gamma):
    tst=Irradiance_torch
    Irradiance=TensorXf(Irradiance_torch)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 0]), gamma[0]).array, BRDF_indices_r)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 1]), gamma[1]).array, BRDF_indices_g)
    dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 2]), gamma[2]).array, BRDF_indices_b)
    return Irradiance




import cv2
import numpy as np
import matplotlib.pyplot as plt


# %%
ssim_fun = pytorch_ssim.SSIM().cuda()
l1_fun = torch.nn.L1Loss()
@dr.wrap_ad(source='drjit', target='torch')
def ssim_loss(img1, img2, ssim_weight=0.5, l1_weight=0.5):
    image1 = img1.cuda().unsqueeze(0)
    image2 = img2.cuda().unsqueeze(0)
    # Compute the SSIM loss
    SSIM_loss = ssim_weight * (1 - ssim_fun(image1, image2))
    # Compute the L1 loss
    l1_loss = l1_weight * l1_fun(image1, image2)
    # Combine the losses
    loss = SSIM_loss + l1_loss
    return loss

def apply_transformation(params, opt , initial_to_world):
    opt['trans'] = dr.clamp(opt['trans'], -0.5, 0.5)
    opt['angle'] = dr.clamp(opt['angle'], -0.5, 0.5)
    trafo = mi.Transform4f.translate([opt['trans'].x, opt['trans'].y, opt['trans'].z]).rotate([0, 1, 0],
                                                                                              opt['angle'] * 100.0)
    params["sensor.to_world"] = trafo @ initial_to_world
    params.update()

def generate_error_heatmap(image1_path, image2_path, cmap='coolwarm', threshold=4):
    # Load the two images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same dimensions")

    # Calculate the absolute difference between the two images
    error_image = cv2.absdiff(image1, image2)

    # Convert the error image to grayscale
    error_image_gray = cv2.cvtColor(error_image, cv2.COLOR_BGR2GRAY)

    # Create a heatmap of the error image with adjusted color range
    plt.imshow(error_image_gray, cmap=cmap, interpolation='nearest', vmin=0, vmax=threshold)
    plt.colorbar()  # Add a color bar

    # Add labels to the axes
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.title('Error Heatmap')

    # Show the plot
    plt.show()


def write_metrics(model_name, psnr, ssim, rmse, output_path):
    import os
    file_path = os.path.join(output_path, "metrics.txt")
    if os.path.exists(file_path):
        mode = 'a'
    else:
        mode = 'w'
    with open(file_path, mode) as file:
        if mode == 'w':
            file.write(f"{'Model Name':^30} {'PSNR':^30} {'SSIM':^30} {'RMSE':^30}\n")
        file.write(f"{model_name:^30} {psnr:^30} {ssim:^30} {rmse:^30}\n")

import drjit as dr
@dr.wrap_ad(source='drjit', target='torch')
def total_variation_loss(img, TVLoss_weight=1):
    h_x, w_x, _ = img.shape
    h_tv = torch.pow((img[1:, :, :] - img[:-1, :, :]), 2).sum()
    w_tv = torch.pow((img[:, 1:, :] - img[:, :-1, :]), 2).sum()
    count_h = (h_x - 1) * w_x * img.shape[2]
    count_w = h_x * (w_x - 1) * img.shape[2]
    tv_loss = TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w)
    return tv_loss

def write_albedo(gamma,wb,folder,albedo):
    # For better display albedo should in sRGB space
    tst = albedo
    Wb_albedo = White_Balance_BRDF(wb,tst)
    sRGB_albedo = PRF_BRDF(Wb_albedo,gamma)
    sRGB_albedo = dr.clamp(sRGB_albedo,0,1)
    sRGB_albedo=mi.Bitmap(sRGB_albedo)
    sRGB_albedo.set_srgb_gamma(True)
    mi.util.write_bitmap(fullfile(folder, "albedo.png"), sRGB_albedo)
    return sRGB_albedo

def write_BRDF(BRDF,folder):
    # Note that BRDF map like roughness and metallic are in linear space
    # and save them without gamma correct just for linear value see code below
    tst = TensorXf(dr.repeat(BRDF.array,3),shape = (BRDF.shape[0],BRDF.shape[1],3))
    map= dr.clamp(tst,0,1)
    map = mi.Bitmap(map)
    map.set_srgb_gamma(True)
    mi.util.write_bitmap(folder, map)
    return map


import cv2
import numpy as np


def plot_montage(images):
    target_size = (512,512)
    processed_images = []
    for img in images:
        img = np.array(img)
        if len(img.shape) == 2:  # HW
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:  # HW1
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        resized_img = cv2.resize(img, target_size)
        processed_images.append(resized_img)

    num_images = len(processed_images)
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols  # 向上取整

    montage_height = num_rows * target_size[1]
    montage_width = num_cols * target_size[0]
    montage_image = np.ones((montage_height, montage_width, 3), dtype=np.float32)

    for idx, img in enumerate(processed_images):
        row = idx // num_cols
        col = idx % num_cols
        start_row = row * target_size[1]
        start_col = col * target_size[0]
        montage_image[start_row:start_row + target_size[1], start_col:start_col + target_size[0], :] = img

    return montage_image



def plot_montage(images, shape, corners):
    target_size = (512, 512)

    def extract_corner_region(image, corners):
        h, w = image.shape[:2]
        x_min = int((corners[0][0] + 1) * w / 2)
        y_min = int((corners[0][1] + 1) * h / 2)
        x_max = int((corners[2][0] + 1) * w / 2)
        y_max = int((corners[2][1] + 1) * h / 2)

    processed_images = []
    for img in images:
        if img.shape == shape:

            img = extract_corner_region(img, corners)

        if len(img.shape) == 2:  # HW
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:  # HW1
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        resized_img = cv2.resize(img, target_size)
        processed_images.append(resized_img)

    num_images = len(processed_images)
    if num_images == 0:
        return np.array([])
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols

    montage_height = num_rows * target_size[1]
    montage_width = num_cols * target_size[0]
    montage_image = np.ones((montage_height, montage_width, 3), dtype=np.float32)

    for idx, img in enumerate(processed_images):
        row = idx // num_cols
        col = idx % num_cols
        start_row = row * target_size[1]
        start_col = col * target_size[0]
        montage_image[start_row:start_row + target_size[1], start_col:start_col + target_size[0], :] = img

    return montage_image




# threshold surface image and get mask and mask bbox corners
def threshDeProCams(im, thresh=None, bias=None):
    # get rid of negative values
    im[im < 0] = 0

    # threshold im_diff with Otsu's method
    if im.ndim == 3:
        im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)  # !!very important, result of COLOR_RGB2GRAY is different from COLOR_BGR2GRAY
        if im.dtype == 'float32':
            im = np.uint8(im * 255)
            im_in_smooth = cv.GaussianBlur(im, ksize=(3, 3), sigmaX=1.5)
            if thresh is None:
                # Use Otus's method
                levels = 2
                thresh = threshold_multiotsu(im_in_smooth, levels)
                if bias is not None:
                    thresh += bias
                print('Otus Threshold:', thresh)
                im_mask = np.digitize(im_in_smooth, bins=thresh) > 0
            else:
                im_mask = im_in_smooth > thresh
    elif im.dtype == np.bool:  # if already a binary image
        im_mask = im

    # find the largest contour by area then convert it to convex hull
    # im_contours, contours, hierarchy = cv.findContours(np.uint8(im_mask), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # only works for OpenCV 3.x
    contours, hierarchy = cv.findContours(np.uint8(im_mask), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]  # works for OpenCV 3.x and 4.x
    max_contours = np.concatenate(contours)
    hulls = cv.convexHull(max_contours)

    im_roi = cv.fillConvexPoly(np.zeros_like(im_mask, dtype=np.uint8), hulls, True) > 0

    # also calculate the bounding box
    bbox = cv.boundingRect(max_contours)
    corners = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]]

    # normalize to (-1, 1) following pytorch grid_sample coordinate system
    h = im.shape[0]
    w = im.shape[1]

    for pt in corners:
        pt[0] = 2 * (pt[0] / w) - 1
        pt[1] = 2 * (pt[1] / h) - 1
    im_mask = im_mask.astype(np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((7,7), np.uint8)
    # Open & Close Operations
    im_mask = cv.morphologyEx(im_mask, cv.MORPH_OPEN, kernel_open, iterations=3)
    im_mask = cv.morphologyEx(im_mask, cv.MORPH_CLOSE, kernel_close, iterations=3)
    im_mask = im_mask > 0
    return im_mask, im_roi, corners

def mask_dr(img, mask):
    if len(mask.shape) == 2 and len(img.shape) == 3:
        mask = TensorXf(dr.repeat(mask.array, 3),shape = [mask.shape[0],mask.shape[1],3])
    return img * mask


def calculate_normalized_error(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("The two images must have the same shape")
    difference = np.abs(image1 - image2)
    normalized_difference = (difference - np.min(difference)) / (np.max(difference) - np.min(difference))

    return normalized_difference







def constraint_normal(normal_map):
    shape = normal_map.shape
    normal_map = dr.clamp(normal_map,0,1)
    xyz_map = 2 * normal_map - 1
    normalized_xyz_map = dr.normalize(dr.unravel(mi.Vector3f, xyz_map))
    normal = TensorXf(dr.ravel(normalized_xyz_map),shape = shape)
    return (normal + 1) / 2


def tv_loss(depth):
    # return spatial_gradient(data[None], order=2)[0, :, [0, 2]].abs().sum(1).mean()
    h_tv = dr.mean(dr.sqr(depth[..., 1:, :] - depth[..., :-1, :]))
    w_tv = dr.mean(dr.sqr(depth[..., :, 1:] - depth[..., :, :-1]))
    return h_tv + w_tv

#Dr.jit version Smooth_l1 loss or you can use l1,
def Huber_loss(y_pred, y_true,delta = 1):
    error = y_true - y_pred
    is_small_error = dr.abs(error) <= delta
    small_error_loss = 0.5 * dr.power(error,2)
    large_error_loss = delta * (dr.abs(error) - 0.5 * delta)
    return dr.mean(dr.select(is_small_error, small_error_loss, large_error_loss))

def compute_im_masks(cam_ref_path, data_root, train_config):
    """
    Compute and return the image mask and its CUDA tensor based on the input parameters.

    Parameters:
        cam_ref_path (str): Path to the reference camera images.
        data_root (str): Root directory of the dataset.
        train_config (dict): Training configuration dictionary, containing keys like:
            - "is_in_direct": (bool) whether to combine direct and indirect masks.
            - "Threshold": (float) threshold for masking when no checkerboard images are found.

    Returns:
        tuple: (im_mask, im_mask_torch) where:
            im_mask is the computed mask,
            im_mask_torch is the CUDA tensor of the mask.
    """
    im_diff = readImgsMT(cam_ref_path, index=[2]) - readImgsMT(cam_ref_path, index=[0])
    im_diff = im_diff.numpy().transpose((2, 3, 1, 0))
    cam_cb_path = fullfile(data_root, 'cam/raw/cb')
    # the same mask as DeProCams
    if os.path.exists(cam_cb_path):
            # find projector direct light mask
        im_cb = readImgsMT(cam_cb_path)
        im_cb = im_cb.numpy().transpose((2, 3, 1, 0))

        # find direct light mask using Nayar's TOG'06 method (also see Moreno 3DV'12)
        l1 = im_cb.max(axis=3)  # max image L+
        l2 = im_cb.min(axis=3)  # max image L-
        b = 0.9  # projector back light strength (for mask use a large b, for real direct/indirect separation, use a smaller b)
        im_direct = (l1 - l2) / (1 - b)  # direct light image
        im_indirect = 2 * (l2 - b * l1) / (1 - b * b)  # indirect (global) light image
        im_indirect = im_indirect.clip(0,1)
        im_mask_indirect, _, mask_corners = threshDeProCams(im_indirect)  # use thresholded as mask
        im_mask_indirect = torch.Tensor(im_mask_indirect).bool()

        im_direct = im_direct.clip(0, 1)
        im_mask_direct, _, mask_corners = threshDeProCams(im_direct)  # use thresholded as mask
        im_mask_direct = torch.Tensor(im_mask_direct).bool()

        if train_config["is_in_direct"]:
            im_mask = torch.logical_or(im_mask_direct,im_mask_indirect)
            _, _, mask_corners = threshDeProCams(im_indirect)
        else:
            im_mask = im_mask_direct
    else:  # without using extra shifted checkerboard images
        # find projector FOV mask
        im_diff = readImgsMT(cam_ref_path, index=[2]) - readImgsMT(cam_ref_path, index=[0])
        im_diff = im_diff.numpy().transpose((2, 3, 1, 0))
        im_mask, _, mask_corners =threshDeProCams(im_diff[..., 0], thresh=train_config["Threshold"])  # use thresholded surface image as mask
        im_mask = torch.Tensor(im_mask.astype("float"))

    im_mask_torch = im_mask.cuda()
    return im_mask,im_mask_torch