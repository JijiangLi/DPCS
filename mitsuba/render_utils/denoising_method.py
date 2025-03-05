# borrowed from https://github.com/CGLab-GIST/target-aware-denoising
# please refer to the original repository for more details
import drjit as dr
import mitsuba as mi
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from custom_ops.simple_regression import Regression
import torch_ops
from matplotlib import pyplot as plt
from custom_ops.cross_bilateral import GbufferCrossBilateral

#%%
def target_denoising(image_sRGB, cam_train_list, idx, regression):
    return torch_ops.weighted_simple_regression_op(regression, image_sRGB, cam_train_list[idx])

def cross_bilateral_denoising(image_sRGB, render_scene, scene_params, sensor, aov_integrator, train_option, seed_f, seed_b,base_cross_bilateral):
    with dr.suspend_grad():
        aovs = mi.render(render_scene, params=scene_params, sensor=sensor, integrator=aov_integrator, spp=train_option['spp'], seed=seed_f, seed_grad=seed_b)
        albedo = aovs[:, :, 0:3]
        normal = aovs[:, :, 3:6]
        depth = aovs[:, :, 6:7]
        albedo = dr.clamp(albedo, 0.0, 1.0)
        max_depth = dr.max(depth)
        depth /= max_depth
    return torch_ops.base_cross_bilateral_op(base_cross_bilateral, image_sRGB, albedo, normal, depth)

def oidn_denoising(image_sRGB, render_scene, scene_params, sensor, aov_integrator, train_option, seed_f, seed_b,base_cross_bilateral=None):
    with dr.suspend_grad():
        aovs = mi.render(render_scene, params=scene_params, sensor=sensor, integrator=aov_integrator, spp=train_option['spp'], seed=seed_f, seed_grad=seed_b)
        albedo = aovs[:, :, 0:3]
        normal = aovs[:, :, 3:6]
        depth = aovs[:, :, 6:7]
        albedo = dr.clamp(albedo, 0.0, 1.0)
        max_depth = dr.max(depth)
        depth /= max_depth
    return torch_ops.oidn_op(image_sRGB, albedo, normal)

denoising_methods = {
    "TARGET": target_denoising,
    "cross_bilatera": cross_bilateral_denoising,
    "OIDN": oidn_denoising
}