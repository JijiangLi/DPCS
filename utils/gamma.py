
import os
from os.path import join as fullfile
import numpy as np
import drjit as dr
import mitsuba as mi
from drjit.cuda.ad import UInt32
# from drjit.cuda.ad import Int32
from mitsuba import TensorXf
def calculate_indices_for_channel(height, width, depth, channel):
    i_indices = np.arange(height, dtype=np.uint32).reshape(height, 1) * np.uint32(width * depth)
    j_indices = np.arange(width, dtype=np.uint32) * np.uint32(depth)
    indices = i_indices + j_indices + np.uint32(channel)
    return indices.flatten()
class gamma_mapping:
    def __init__(self, scene_param = None,camera_batch = None):
        height, width, depth = scene_param['cam_size']
        width  = width*camera_batch
        self.c_indices_r = UInt32(calculate_indices_for_channel(height, width, depth, 0))  # Red channel
        self.c_indices_g = UInt32(calculate_indices_for_channel(height, width, depth, 1))  # Green channel
        self.c_indices_b = UInt32(calculate_indices_for_channel(height, width, depth, 2))  # Blue channel
        self.ctst = TensorXf(np.random.rand(height, width, depth))
        height, width, depth = scene_param['prj_size']
        self.p_indices_r = UInt32(calculate_indices_for_channel(height, width, depth, 0))  # Red channel
        self.p_indices_g = UInt32(calculate_indices_for_channel(height, width, depth, 1))  # Green channel
        self.p_indices_b = UInt32(calculate_indices_for_channel(height, width, depth, 2))  # Blue channel
        self.ptst = TensorXf(np.random.rand(height, width, depth))
        height, width, depth = scene_param['texture_size']
        self.texture_indices_r = UInt32(calculate_indices_for_channel(height, width, depth, 0))  # Red channel
        self.texture_indices_g = UInt32(calculate_indices_for_channel(height, width, depth, 1))  # Green channel
        self.texture_indices_b = UInt32(calculate_indices_for_channel(height, width, depth, 2))  # Blue channel
        self.texturetst = TensorXf(np.random.rand(height, width, depth))
        height, width, depth = scene_param['cmp_size']
        self.cmp_indices_r = UInt32(calculate_indices_for_channel(height, width, depth, 0))  # Red channel
        self.cmp_indices_g = UInt32(calculate_indices_for_channel(height, width, depth, 1))  # Green channel
        self.cmp_indices_b = UInt32(calculate_indices_for_channel(height, width, depth, 2))  # Blue channel
        self.cmptst = TensorXf(np.random.rand(height, width, depth))

    def PRF( self, Irradiance, gamma):
        tst = Irradiance
        Irradiance = TensorXf(Irradiance)
        dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 0]), gamma[0]).array, self.p_indices_r)
        dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 1]), gamma[1]).array, self.p_indices_g)
        dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 2]), gamma[2]).array, self.p_indices_b)
        return Irradiance
    def CRF(self,Irradiance, gamma):
        tst = Irradiance
        Irradiance = TensorXf(Irradiance)
        dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 0]), gamma[0]).array, self.c_indices_r)
        dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 1]), gamma[1]).array, self.c_indices_g)
        dr.scatter(Irradiance.array, dr.power(TensorXf(tst[:, :, 2]), gamma[2]).array, self.c_indices_b)
        return Irradiance

    def White_Balance_Camera(self,WB, img):
        dr.scatter(self.ctst.array, WB[0], self.c_indices_r)
        dr.scatter(self.ctst.array, WB[1], self.c_indices_g)
        dr.scatter(self.ctst.array, WB[2], self.c_indices_b)
        return self.ctst * img

    def White_Balance_Projector(self,WB, img):
        dr.scatter(self.ptst.array, WB[0], self.p_indices_r)
        dr.scatter(self.ptst.array, WB[1], self.p_indices_g)
        dr.scatter(self.ptst.array, WB[2], self.p_indices_b)
        return self.ptst * img

    def White_Balance_BRDF(self,WB, img):
        dr.scatter(self.texturetst.array, WB[0], self.texture_indices_r)
        dr.scatter(self.texturetst.array, WB[1],self.texture_indices_g)
        dr.scatter(self.texturetst.array, WB[2], self.texture_indices_b)
        return self.texturetst * img

    def PRF_cmp(self,Irradiance, gamma):
        dr.scatter(Irradiance.array, dr.power(TensorXf(self.cmptst[:, :, 0]), gamma[0]).array, self.cmp_indices_r)
        dr.scatter(Irradiance.array, dr.power(TensorXf(self.cmptst[:, :, 1]), gamma[1]).array, self.cmp_indices_g)
        dr.scatter(Irradiance.array, dr.power(TensorXf(self.cmptst[:, :, 2]), gamma[2]).array, self.cmp_indices_b)
        return Irradiance

    def PRF_BRDF(self,Irradiance, gamma):
        dr.scatter(Irradiance.array, dr.power(TensorXf(self.texturetst[:, :, 0]), gamma[0]).array, self.texture_indices_r)
        dr.scatter(Irradiance.array, dr.power(TensorXf(self.texturetst[:, :, 1]), gamma[1]).array, self.texture_indices_g)
        dr.scatter(Irradiance.array, dr.power(TensorXf(self.texturetst[:, :, 2]), gamma[2]).array, self.texture_indices_b)
        return Irradiance

    def write_albedo(self, gamma, wb, folder, albedo):
        self.texturetst = albedo
        Wb_albedo = self.White_Balance_BRDF(wb, self.texturetst)
        sRGB_albedo = self.PRF_BRDF(Wb_albedo, gamma)
        sRGB_albedo = dr.clamp(sRGB_albedo, 0, 1)
        sRGB_albedo = mi.Bitmap(sRGB_albedo)
        sRGB_albedo.set_srgb_gamma(True)
        mi.util.write_bitmap(fullfile(folder, "albedo.png"), sRGB_albedo)
        return sRGB_albedo

    def write_BRDF(self,gamma, wb, BRDF, folder):
        tst = self.White_Balance_BRDF(wb, BRDF)
        # sRGB = response_function(wb.torch(), opt, CRF_key,wb.shape[0], wb.shape[1], 1)=
        wb = tst ** (1 / 2.4)
        wb = dr.clamp(wb, 0, 1)
        map = mi.Bitmap(wb)
        map.set_srgb_gamma(True)
        mi.util.write_bitmap(folder, map)
        return map


