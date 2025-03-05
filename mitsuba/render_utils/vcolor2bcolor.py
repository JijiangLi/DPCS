import os
import argparse
import sys
import open3d as o3d
import numpy as np
import pymeshlab
import trimesh
model_path = sys.argv[1]
export_path = sys.argv[2]
mesh_name = sys.argv[3]

# Set file paths
blend_path = os.path.join(export_path, f"{mesh_name}.blend")
image_path = os.path.join(export_path, "base_color.png")
mesh_path = os.path.join(export_path, f"{mesh_name}.obj")


#%% Reconstruction using Screened Poission:
ms = pymeshlab.MeshSet()
ms.load_new_mesh(mesh_path)
ms.transfer_vertex_color_to_texture(textw = 3200 , texth = 2400, pullpush = False,textname ="base_color.png")
ms.save_current_mesh(mesh_path, save_textures=True)


# mesh = trimesh.load_mesh(mesh_filename)
# mesh.fix_normals()
# mesh.export(mesh_path)



