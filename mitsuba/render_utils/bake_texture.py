import bpy
import os
import argparse
import sys
import subprocess
import time
import yaml
# Set up argument parsing
def parse_args():

    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Bake and export mesh in Blender")
    parser.add_argument("--model_path", type=str, help="Path to the input .ply model file")
    parser.add_argument("--export_path", type=str, help="Path to the export directory")
    parser.add_argument("--mesh_name",default='test', type=str, help="Name of the model")
    return parser.parse_args(argv)

# Get arguments from the command line
start_time = time.time()
args = parse_args()
model_path = args.model_path
export_path = args.export_path
mesh_name = args.mesh_name

# Set file paths
blend_path = os.path.join(export_path, f"{args.mesh_name}.blend")
image_path = os.path.join(export_path, "base_color.png")
mesh_path = os.path.join(export_path, f"{args.mesh_name}.ply")

python_executable = sys.executable
#%% Reconstruction using Screened Poission:
subprocess.run([
    python_executable, "mitsuba/render_utils/pcd2mesh.py", model_path, export_path, mesh_name
])


#%%


# Ensure export directory exists
if not os.path.exists(export_path):
    os.makedirs(export_path)

# Delete the default cube if it exists
if "Cube" in bpy.data.objects:
    bpy.data.objects["Cube"].select_set(True)
    bpy.ops.object.delete()

# Import the .ply file
bpy.ops.import_mesh.ply(filepath=mesh_path)
obj = bpy.context.selected_objects[0]

# Ensure the object is in object mode
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='OBJECT')



# Triangulate the mesh
bpy.ops.object.modifier_add(type='TRIANGULATE')
bpy.ops.object.modifier_apply(modifier="Triangulate")

# Enter edit mode for UV unwrapping
bpy.ops.object.mode_set(mode='EDIT')
#bpy.ops.uv.smart_project(angle_limit=90.0, island_margin=0, area_weight=1, correct_aspect=True, scale_to_bounds=True)
bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0)
bpy.ops.object.mode_set(mode='OBJECT')

# Export the PLY file with UV and normals
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
if(not os.path.exists(os.path.join(export_path, f"{args.mesh_name}.obj"))):
    # os.remove(mesh_path)
    bpy.ops.export_scene.obj(
        filepath=os.path.join(export_path, f"{args.mesh_name}.obj"),
        use_normals=True,  # Export normals
        use_materials = True,
        axis_forward='Y',  # Blender axis setting
        axis_up='Z',
    )

subprocess.run([
    python_executable, "mitsuba/render_utils/vcolor2bcolor.py", model_path, export_path, mesh_name
])
calib_yaml_path = os.path.join(export_path,"../calib/results/calibration.yml")
print(os.path.exists(calib_yaml_path))
params_yaml_path = os.path.join(export_path,"params.yml")
if os.path.exists(calib_yaml_path):
    subprocess.run([
        python_executable, "mitsuba/render_utils/calib2params.py", calib_yaml_path, params_yaml_path
    ])

source_file = os.path.join(export_path, "../recon/depth.mat")
destination_file = os.path.join(export_path, "depth.mat")
import shutil
shutil.copyfile(source_file, destination_file)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"time: {elapsed_time:.6f} seconds")
print("Baking, exporting, and saving the .blend file is complete!")
