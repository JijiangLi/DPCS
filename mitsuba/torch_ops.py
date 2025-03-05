# This script is borrowed from https://github.com/CGLab-GIST/target-aware-denoising
# Please refer to the original repository for more details


#  Copyright (c) 2024 CGLab, GIST. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import drjit as dr
import torch
import numpy as np

# from nn_denoiser.oidn.oidn import OIDN
# denosier = OIDN()

@dr.wrap_ad(source='drjit', target='torch')
def rel_l2_loss_op(input, ref):
    denom =  torch.mean(ref,dim=2, keepdim=True)
    eps = 1e-2
    num = torch.square(input - ref)
    relse = num / (denom * denom + eps)
    return relse


@dr.wrap_ad(source='drjit', target='torch')
def torch_weighted_simple_regression_op(normalizedWgtSumFunc, input, input_ref): 
    wgtX, wgtXX, wgtXY, wgtY = normalizedWgtSumFunc(input.cuda(), input_ref.cuda())
    return wgtX, wgtXX, wgtXY, wgtY 

def weighted_simple_regression_op(normalizedWgtSumFunc, input, input_ref):
    wgtX, wgtXX, wgtXY, wgtY = torch_weighted_simple_regression_op(normalizedWgtSumFunc, input, input_ref)
    beta = (wgtXY - wgtX * wgtY) / dr.maximum(wgtXX - dr.sqr(wgtX),1e-7)
    alpha = wgtY - beta * wgtX
    output = alpha + beta * input_ref
    output = dr.maximum(0.0, output)
    return output

@dr.wrap_ad(source='drjit', target='torch')
def base_cross_bilateral_op(base_cross_bilateral_func, input, albedo, normal, depth):
    output = base_cross_bilateral_func(input.cuda(), albedo.cuda(), normal.cuda(), depth.cuda())
    return output

# @dr.wrap_ad(source='drjit', target='torch')
# def oidn_op(cNoisy, cAlbedo, cNormal):
#
#     cNoisy = cNoisy.unsqueeze(0).permute(0, 3, 1, 2)
#     cAlbedo = cAlbedo.unsqueeze(0).permute(0, 3, 1, 2)
#     cNormal = cNormal.unsqueeze(0).permute(0, 3, 1, 2)
#
#     cDenoised = denosier(cNoisy, cAlbedo, cNormal)
#     cDenoised = cDenoised.squeeze(0).permute(1, 2, 0)
#
#     return cDenoised
import torch.nn.functional as F
@dr.wrap_ad(source='drjit', target='torch')
def oidn_op(cNoisy, cAlbedo, cNormal):
    # Function to pad to multiple of alignment
    def pad_to_alignment(tensor, alignment):
        _, _, h, w = tensor.shape
        pad_h = (alignment - h % alignment) % alignment
        pad_w = (alignment - w % alignment) % alignment
        padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        return padded_tensor, pad_h, pad_w

    # Add batch and channel dimensions and permute
    cNoisy = cNoisy.unsqueeze(0).permute(0, 3, 1, 2)
    cAlbedo = cAlbedo.unsqueeze(0).permute(0, 3, 1, 2)
    cNormal = cNormal.unsqueeze(0).permute(0, 3, 1, 2)

    # Pad inputs
    alignment = 16
    cNoisy, pad_h_n, pad_w_n = pad_to_alignment(cNoisy, alignment)
    cAlbedo, pad_h_a, pad_w_a = pad_to_alignment(cAlbedo, alignment)
    cNormal, pad_h_m, pad_w_m = pad_to_alignment(cNormal, alignment)

    # Forward pass through the denoiser
    cDenoised = denosier(cNoisy, cAlbedo, cNormal)

    # Remove padding
    if pad_h_n > 0:
        cDenoised = cDenoised[:, :, :-pad_h_n, :]
    if pad_w_n > 0:
        cDenoised = cDenoised[:, :, :, :-pad_w_n]

    # Permute back to original shape and remove batch dimension
    cDenoised = cDenoised.squeeze(0).permute(1, 2, 0)

    return cDenoised

@dr.wrap_ad(source='drjit', target='torch')
def least_squares_op(least_squares, input, depth, albedo, normal):
    out = least_squares(input.cuda(),depth.cuda(), albedo.cuda(), normal.cuda())
    return out