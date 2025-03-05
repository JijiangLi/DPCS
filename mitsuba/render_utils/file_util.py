import os
import json
import numpy as np
import open3d as o3d
import mitsuba as mi
import drjit as dr
import torch

from utils.fov import intrinsic2fov_x


def XML_write(calib_data, filename, dataset_root,train_config = None):
    import xml.etree.ElementTree as ET
    from utils.fov import fov
    scale = train_config["scale"]
    try:
        spp = train_config["spp"]
        params_path = train_config["params_path"]
        data_list = train_config["data_list"]
    except:
        print("No scene config into XML.")
    max_radiance = scale * 5
    cam_fov_x = torch.squeeze(calib_data["camK"])[0, 0]
    prj_fov_x = torch.squeeze(calib_data["prjK"])[0, 0]
    cam_shift_x = (-(torch.squeeze(calib_data["camK"])[0, 2] - calib_data["cam_w"] / 2) / calib_data["cam_w"]).numpy()
    cam_shift_y = (-(torch.squeeze(calib_data["camK"])[1, 2] - calib_data["cam_h"] / 2) / calib_data["cam_h"]).numpy()
    prj_shift_x = (-(torch.squeeze(calib_data["prjK"])[0, 2] - calib_data["prj_w"] / 2) / calib_data["prj_w"]).numpy()
    prj_shift_y = (-(torch.squeeze(calib_data["prjK"])[1, 2] - calib_data["prj_h"] / 2) / calib_data["prj_h"]).numpy()
    cam_fov = intrinsic2fov_x(calib_data["cam_w"], cam_fov_x)
    max_depth = train_config['max_depth']

    # prj_fov = fov(fov(intrinsic2fov_x(calib_data["prj_w"], prj_fov_x)))
    prj_fov = intrinsic2fov_x(calib_data["prj_w"], prj_fov_x)
    tree = ET.parse(filename)
    root = tree.getroot()
    for sensor in root.findall('.//sensor'):
        fov = sensor.find('.//float[@name="fov"]')
        if fov is not None:
            fov.set('value', str(cam_fov))

    for pp_offset_x in root.findall(".//sensor/float[@name='principal_point_offset_x']"):
        pp_offset_x.set('value', str(cam_shift_x))
    for pp_offset_y in root.findall(".//sensor/float[@name='principal_point_offset_y']"):
        pp_offset_y.set('value', str(cam_shift_y))

    for emitter in root.findall('.//emitter'):
        tst = emitter.find('.//float[@name="fov"]')
        if tst is not None:
            tst.set('value', str(prj_fov))
        for pp_offset_x in emitter.findall(".//float[@name='principal_point_offset_x']"):
            pp_offset_x.set('value', str(prj_shift_x))

        for pp_offset_y in emitter.findall(".//float[@name='principal_point_offset_y']"):
            pp_offset_y.set('value', str(prj_shift_y))

    for obj in root.findall(".//shape[@type='obj']"):
        filename_element = obj.find(".//string[@name='filename']")
        if filename_element is not None:
            filename_element.set('value',
                                 os.path.join(params_path, f'{os.path.basename(data_list[0])}.obj'))
    defaults = [
        ('spp', str(spp)),
        ('res', '512'),
        ('max_depth', str(max_depth)),
        ('integrator', 'prb_reparam'),
        ('scale', str(scale)),
        ('max_radiance', str(max_radiance))
    ]
    for name, value in defaults:
        default_element = root.find(f".//default[@name='{name}']")
        if default_element is not None:
            default_element.set('value', value)
        else:
            new_default = ET.Element('default', name=name, value=value)
            root.insert(0, new_default)
    for normal_bsdf in root.findall(".//bsdf[@type='normalmap']"):
        principled_bsdf = normal_bsdf.find(".//bsdf[@type='principled']")
        if principled_bsdf is not None:
            base_color_texture = principled_bsdf.find(".//texture[@name='base_color']")
            if base_color_texture is not None:
                filename_element = base_color_texture.find(".//string[@name='filename']")
                if filename_element is not None:
                    if (os.path.exists(os.path.join(params_path, "base_color.png"))):
                        new_texture_filename = os.path.join(params_path, "base_color.png")
                    else:
                        new_texture_filename = f'./texture/white_image.png'
                    filename_element.set('value', new_texture_filename)
    tree.write('./scene/simu.xml')



