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
mesh_path = os.path.join(export_path, f"{mesh_name}.ply")


#%% Reconstruction using Screened Poission:
pcd = o3d.io.read_point_cloud(model_path)
points = np.asarray(pcd.points)
points /= 1000
print(points)
pcd.points = o3d.utility.Vector3dVector(points)
#pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=4.0)
o3d.io.write_point_cloud(os.path.join(export_path,"denoised_pointcloud.ply"), pcd)
ms = pymeshlab.MeshSet()
ms.load_new_mesh(os.path.join(export_path,"denoised_pointcloud.ply"))
ms.compute_normal_for_point_clouds(k=15,smoothiter=5, flipflag=True)
ms.surface_reconstruction_screened_poisson(depth=8, fulldepth=5, cgdepth=0, scale=1.1, samplespernode=1.5, pointweight=4.0,iters = 8)
#ms.invert_faces_orientation(forceflip=True)
ms.re_compute_face_normals()
ms.re_compute_vertex_normals()
ms.apply_normal_normalization_per_vertex()
ms.apply_normal_normalization_per_face()
# apply Laplacian smoothing
ms.hc_laplacian_smooth()
ms.hc_laplacian_smooth()
ms.hc_laplacian_smooth()
# ms.apply_coord_laplacian_smoothing()
# ms.apply_coord_laplacian_smoothing()
# ms.apply_coord_laplacian_smoothing()
mesh_filename = os.path.join(export_path, "reconstructed_mesh_rough.ply")
ms.save_current_mesh(mesh_filename, save_vertex_color=True)

mesh = trimesh.load_mesh(mesh_filename)
mesh.fix_normals()
mesh.export(mesh_path)



