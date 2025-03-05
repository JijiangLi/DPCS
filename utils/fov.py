
import torch
import torchvision.transforms as transforms
interpolation_method=transforms.InterpolationMode.BILINEAR
resize_transform = transforms.Resize((600,600),antialias=True,interpolation=interpolation_method)


def fov(alpha1):
    import math
    alpha1=math.radians(alpha1)
    alpha2=math.degrees(2*math.atan(2*math.tan(alpha1/2)))
    return alpha2


"""
* \brief : this is a function do shift of projector input in order to implement the 
* optics center shift of the projector  input and also convert it into linear sRGB space
* to do simulation of the projector intensity function gamma 2.4
* \params : 
*   cx: x shift pixel units
*   cy: y shift pixel units
*   filename: the input sRGB data we wanna use.
* \output :
*   temping: shifted input and gamma corrected linear rgb data after ccm
"""

import torch
import numpy as np
import cv2
import colour



def intrinsic2fov_x(width,fx):
    import math
    return math.degrees(2*math.atan(width/2.0/fx));

def intrinsic2fov_d(width,height,fx,fy):
    import math
    """
        Calculate the diagonal FOV based on image dimensions and focal lengths.

        Parameters:
        - w: Image width in pixels
        - h: Image height in pixels
        - fx: Focal length in pixels along the x-axis
        - fy: Focal length in pixels along the y-axis

        Returns:
        - Diagonal FOV in degrees
        """
    d = math.sqrt(width ** 2 + height ** 2)
    fov_d_rad = 2 * math.atan(d / (2 * math.sqrt(fx ** 2 + fy ** 2)))

    # Convert radians to degrees
    fov_d_deg = math.degrees(fov_d_rad)

    return fov_d_deg
