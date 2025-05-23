# borrowed from https://github.com/BingyaoHuang/CompenNeSt-plusplus
# please refer to the original repository for more details
import numpy as np
import cv2 as cv
import torch
from skimage.filters import threshold_multiotsu
# threshold surface image and get mask and mask bbox corners

# invert a sampling grid using cubic interpolation. Note that grid shape is 1xHxWx2 and values must be in (-1, 1)
def invertGrid(src_grid, dst_size):
    from scipy.interpolate import griddata
    import kornia
    _, h, w, _ = src_grid.shape
    dst_pts2d = src_grid.permute(3, 1, 2, 0).reshape(2, -1).numpy().transpose()  # data points
    src_pts2d = kornia.utils.create_meshgrid(h, w, normalized_coordinates=True).permute(0, 3, 1, 2).reshape(2, -1).numpy().transpose()

    # dense dst meshgrid, i.e., query points
    dst_dense_pts2d = kornia.utils.create_meshgrid(dst_size[0], dst_size[1], normalized_coordinates=True).permute(0, 3, 1, 2).squeeze().numpy()

    # ignore values that are out of (-1,1) range
    inlier_idx = (dst_pts2d[:, 0] >= -1) & (dst_pts2d[:, 0] <= 1) & (dst_pts2d[:, 1] >= -1) & (dst_pts2d[:, 1] <= 1)
    dst_pts2d = dst_pts2d[inlier_idx]
    src_pts2d = src_pts2d[inlier_idx]

    # interpolate using griddata
    dst_grid = griddata(dst_pts2d, src_pts2d, (dst_dense_pts2d[0], dst_dense_pts2d[1]), method='linear', fill_value=-2)
    return dst_grid


# compute the stereo rectification and inverse rectification grid
def getRectifyGrid(calib_data):
    camK = calib_data['camK'][0, 0:3, 0:3].numpy().astype(np.float64)
    if 'camKc' in calib_data:
        camKc = calib_data['camKc'][0, ...].numpy().astype(np.float64)
    else:
        camKc = np.zeros((1, 4), dtype=np.float64)
    prjK = calib_data['prjK'][0, 0:3, 0:3].numpy().astype(np.float64)
    if 'prjKc' in calib_data:
        prjKc = calib_data['prjKc'][0, ...].numpy().astype(np.float64)
    else:
        prjKc = np.zeros((1, 4), dtype=np.float64)
    R = calib_data['prjRT'][0, 0:3, 0:3].numpy().astype(np.float64)
    T = (calib_data['prjRT'][0, 0:3, 3] * calib_data['scale_T']).numpy().astype(np.float64)
    cam_w = calib_data['cam_w']
    cam_h = calib_data['cam_h']
    prj_w = calib_data['prj_w']
    prj_h = calib_data['prj_h']

    # resize prj intrinsics such that its image size is the same as camera
    prjK[0, 0] *= cam_w / prj_w
    prjK[0, 2] *= cam_w / prj_w
    prjK[1, 1] *= cam_h / prj_h
    prjK[1, 2] *= cam_h / prj_h

    # stereo rectification
    # check whether the camera is on the right (should be easy to change this code to account for vertical ProCams setups)
    if T[0] > 0:  # camera on the right
        R2, R1, P2, P1, Q, roi2, roi1 = cv.stereoRectify(prjK, prjKc, camK, camKc, (cam_w, cam_h), R.transpose(), -R.transpose() @ T, alpha=1,
                                                         newImageSize=(0, 0))
    else:  # camera on the left
        R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(camK, camKc, prjK, prjKc, (cam_w, cam_h), R, T, flags=0, alpha=1, newImageSize=(0, 0))

    # the homography that rectifies the original image, usually used in OpenCV (but not the optimal in our case, it fails sometimes)
    H1 = camK @ R1 @ np.linalg.inv(camK)

    # compute the scale and translation given the output image size
    corners = H1 @ np.array([[0, cam_w - 1, cam_w - 1, 0], [0, 0, cam_h - 1, cam_h - 1], [1, 1, 1, 1]])
    corners /= corners[-1, :]
    t = np.array([[1, 0, -corners.min(1)[0]], [0, 1, -corners.min(1)[1]], [0, 0, 1]])  # translation matrix, make sure warped corners are at (0,0)
    scale = max((corners.max(1)[0] - corners.min(1)[0]) / cam_w, (corners.max(1)[1] - corners.min(1)[1]) / cam_h)
    s = np.array([[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1]])

    # optimal
    H1_opt = s @ t @ H1
    R1_opt = np.linalg.inv(camK) @ H1_opt @ camK
    P1_opt = camK

    x_map, y_map = cv.initUndistortRectifyMap(camK, camKc, R1_opt, P1_opt, (cam_w, cam_h), cv.CV_32F)  # rectify grid
    x_invmap, y_invmap = cv.initUndistortRectifyMap(P1_opt, camKc, np.linalg.inv(R1_opt), camK, (cam_w, cam_h), cv.CV_32F)  # unrectify grid

    return torch.Tensor(np.stack((x_map, y_map), 2)).unsqueeze(0), torch.Tensor(np.stack((x_invmap, y_invmap), 2)).unsqueeze(0)
