
# DPCS: Path Tracing-Based Differentiable Projector-Camera Systems


## Introduction
Implementation of DPCS: Path Tracing-Based Differentiable Projector-Camera Systems.
We recommend running this code through [Nvidia-docker](https://hub.docker.com/repository/docker/jijiangli/dpcs/) on Ubuntu. 
Please refer to the detailed instruction for the installation of [Docker](https://docs.docker.com/engine/install/ubuntu/) and [Nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Also see the guidance for docker usage [below](#simple-way-to-get-started).

## Prerequisites
* PyTorch compatible GPU
* Python 3
* PyTorch
* opencv-python 
* visdom
* matplotlib
* opencv-contrib-python-headless==4.8.0.76
* color-science
* scikit-image
* DPCS forked version of [mitsuba3](https://github.com/JijiangLi/mitsuba3)
* [CUDA denoising methods](https://github.com/CGLab-GIST/target-aware-denoising)
* [lpips==0.1.4](https://github.com/richzhang/PerceptualSimilarity)
* json
* blender 3.6
* open3d


## Usage

### Simple way to get started
If you feel hard to get started from compiling of [mitsuba3](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html) and [CUDA denoising methods](https://github.com/CGLab-GIST/target-aware-denoising), we have offered a [docker image](https://hub.docker.com/repository/docker/jijiangli/dpcs/)
to help you get started quickly.

1. Download the [DPCS dataset](https://drive.google.com/file/d/10BITDSg3g0y9ajSKn5zqb_1OmO1Xoab5/view?usp=drive_link) and extract it to a folder e.g. `r"path/DPCS_dataset"`.
2. Pull the docker image.
```bash
     docker pull jijiangli/dpcs:latest
```
3. Run the docker image with the dataset and code mounted.
```bash
      git clone git clone https://github.com/JijiangLi/DPCS
      cp -r "path/DPCS_dataset" ./DPCS
      docker run -it --gpus all --workdir /home -v "./DPCS":/home dpcs:latest
```
4. Run the DPCS code to reproduce the relight results shows in the table 2 and Figure 5 of the main paper.
```bash
    cd DPCS/mitsuba
    python3 run_DPCS.py --dataset_root "DPCS/DPCS_dataset"
```

### On your local machine
1. Clone this repo:
```bash
    git clone https://github.com/JijiangLi/DPCS
    cd DPCS
```
   


2. Install required packages by typing.
```bash
   pip install -r requirements.txt
   cd ..
```

3. Compiling the modified Mitsuba3 renderer. Please follow the instructions in [mitsuba3](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html)

```bash
    git clone --recursive https://github.com/JijiangLi/mitsuba3
    cd mitsuba3
    git checkout mitsuba3-DPCS
    git submodule update --init --recursive
    mkdir build
    cd build
    cmake -GNinja ..
    ninja
    # back to your work dictionary
    cd ../..
```

4. Compiling the CUDA denoising methods. Please follow the instructions in [CUDA denoising methods](https://github.com/CGLab-GIST/target-aware-denoising)
```bash
   git clone https://github.com/CGLab-GIST/target-aware-denoising
   cd target-aware-denoising/example_code/custom_ops
   python setup.py install
   python setup_bilateral.py install
   python setup_simple.py install
   # back to your work dictionary
   cd ../../..
```

3. Download [DPCS dataset](https://drive.google.com/file/d/10BITDSg3g0y9ajSKn5zqb_1OmO1Xoab5/view?usp=drive_link) and extract it to a folder e.g. `r"path/DPCS_dataset"`.

4. Enter to the root of `DPCS/mitsuba`, and run DPCS to reproduce the relight results shown in the table 2 and Figure 5 of the main paper. 
```bash
    cd DPCS
    cd mitsuba
    python3 run_DPCS.py --dataset_root `r"path/DPCS_dataset"`
```
### Evaluation of all baseline methods and DPCS
For evaluation, we also offer code for the result after running `run_DPCS.py`, the code is in `DPCS_dataset/main.py`. 
After you run this code, you can get the results of the relighting in the `DPCS_dataset/metrics.excel` for all `data_name` of different methods inferred result in `DPCS_dataset/sets_up/`.
Note that this script works simply to calculate the metrics for all methods inferred images appear in `DPCS_dataset/sets_up/data_name/pred/relit/test`.
## Apply DPCS to your own setup

1. Calibrate the projector-camera system using the calibration software in [A Fast and Flexible Projector-Camera Calibration System](https://github.com/BingyaoHuang/single-shot-pro-cam-calib).
2. Capture a setup using the same software as 1.
3. Reconstruct a surface point cloud in the same software as 1., the pcd file will be in `'data_name/recon'`.
4. Reconstruction of the mesh with an initial ``base_color.png`` from point cloud in code,
   note that this code takes the input of the pcd and outputs a `mesh` with its texture as `initial base color` directly
   from pcd. This code may have some issues with UV for the `.obj` file because some version of Blender may not support UV in
   Python package. To solve this, in Blender GUI, you can export manually the mesh to `.obj` with UV map, and then rerun the following code,
   it will work.
```bash
  cd DPCS
   blender --background --python3 mitsuba/render_utils/bake_texture.py `
    -- --model_path DPCS_dataset\sets_up\data_name\recon\pointCloud_Set01.ply `
    --export_path DPCS_dataset\data_name\sets_up\data_name\params `
    --mesh_name data_name
```
5. Add your new set_up to the dataset config in `DPCS_dataset/sets_up/config.json` and may delete all dataset config in `DPCS_dataset`, 
   this can save time to only run with your own setup rather than all datasets in `DPCS_dataset`.
6. `[Optional]` If you want to use the compensation, you can set the `required_compensation` to `True` in the `train_config` of the `run_DPCS.py` file. 
7. Run the DPCS to your own setup. Do not forget to change the `config.json` to your own setup.
```bash
    cd DPCS/mitsuba
    python3 run_DPCS.py --dataset_root "DPCS/DPCS_dataset"
```
----
## Common issue and Tips
1. The directory structure of the data is consistent with those used in [CompenNeSt++](https://github.com/BingyaoHuang/CompenNeSt-plusplus) and [DeProCams](https://github.com/BingyaoHuang/DeProCams), which facilitates the reproduction of the baseline methods presented in the paper on the `DPCS_dataset`.
2. After running the code, you can find the relighting results in `DPCS_dataset/sets_up/data_name/pred/relit/test` folder.
3. If you run with config of `Compensation`, you can find the compensation results in `DPCS_dataset/sets_up/data_name/prj/cmp/test` folder.
4. For the different dataset appear in `DPCS_dataset`, you can find the detail config in `DPCS_dataset/sets_up/config.json`.
   E.g., whether to use a mask during training or what of the number of test patterns you use.
5. Initialize the BRDF, for `metallic` and `roughness`, initialize them to black,
   for the `normal`, init it as the `(0,0,1)` no change to the surface. 
   For `base color`, initialize it from the point cloud, or you can use a  white texture as initialization also.
   but init from pcd sometimes performs a little bit better. See more in `DPCS/render_utils/bake_texture.py`.
6. To test a  novel scene, for example, you want to modify the scene material or geometry,
   go to `DPCS/mitsuba/scene/simu.xml` to change the configuration, e.g., change geometry file or BRDF maps path.
   Then you can run the `run_DPCS.py` in the `relit_DPCS` function to test.
----


## Citation
      @ARTICLE{Li2025DPCS,
       author  = {Li, Jijiang and Deng, Qingyue and Ling, Haibin and Huang, Bingyao},
       journal = {IEEE Transactions on Visualization and Computer Graphics (in press)},
       title   = {DPCS: Path Tracing-Based Differentiable Projector-Camera Systems},
       year    = {2025},}


## Acknowledgments
- The PyTorch implementation of SSIM loss is modified from [Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim).
- The denoising algorithms are borrowed from this excellent work [CGLab-GIST/target-aware-denoising](https://github.com/CGLab-GIST/target-aware-denoising).
- The compensation is inspired by [Projector Compensation Framework using Differentiable Rendering](https://github.com/CGLab-GIST/pc-using-dr).
- The dataloader and file structure are adopted from [CompenNeSt++](https://github.com/BingyaoHuang/CompenNeSt-plusplus) and [DeProCams](https://github.com/BingyaoHuang/DeProCams).
- The calibration software used in this work is [A Fast and Flexible Projector-Camera Calibration System](https://github.com/BingyaoHuang/single-shot-pro-cam-calib).
- We thank the anonymous reviewers for valuable and inspiring comments and suggestions.

## Troubleshooting / Requests
If you have any questions, please feel free to open a GitHub issue or send an e-mail to [jijiangli@email.swu.edu.cn](jijiangli@email.swu.edu.cn).
