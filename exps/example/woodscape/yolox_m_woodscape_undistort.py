#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from yolox.exp import Exp as MyExp
"""

Experiment Reference Pipeline for YOLOX_m on WoodScape Dataset
results in results/p1_reference_yolox_m
        
"""


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.test_size = (640, 640)
        self.num_classes = 80 # default from pretrained model.
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
         
        self.nmsthre = 0.65 # default from pretrained model
   
        # Define yourself dataset path
        self.data_dir = os.environ.get("WOODSCAPE_COCO_DIR", "/app/data/woodscape_coco2")
        self.train_ann = "train.json"
        self.val_ann = "val.json"
   

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="images/val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)), # todo replace valtransform with cylindrical preproc
        )
        return dataset



import cv2
import numpy as np
from pyquaternion import Quaternion


## from https://plaut.github.io/fisheye_tutorial/
def get_mapping(calib, hfov=np.deg2rad(190), vfov=np.deg2rad(143)):
    """
    Compute the pixel mapping from a fisheye image to a cylindrical image
    :param calib: calibration in WoodScape format, as a dictionary
    :param hfov: horizontal field of view, in radians
    :param vfov: vertical field of view, in radians
    :return: horizontal and vertical mapping
    """
    # Prepare intrinsic and extrinsic matrices for the cylindrical image
    R = Quaternion(
        w=calib["extrinsic"]["quaternion"][3],
        x=calib["extrinsic"]["quaternion"][0],
        y=calib["extrinsic"]["quaternion"][1],
        z=calib["extrinsic"]["quaternion"][2],
    ).rotation_matrix.T
    rdf_to_flu = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    R = (
        R @ rdf_to_flu
    )  # Rotation from vehicle to camera includes FLU-to-RDF. Remove FLU-to-RDF from R.
    azimuth = np.arccos(
        R[2, 2] / np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2)
    )  # azimuth angle parallel to the ground
    if R[0, 2] < 0:
        azimuth = 2 * np.pi - azimuth
    tilt = -np.arccos(
        np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2)
    )  # elevation to the ground plane
    Ry = np.array(
        [
            [np.cos(azimuth), 0, np.sin(azimuth)],
            [0, 1, 0],
            [-np.sin(azimuth), 0, np.cos(azimuth)],
        ]
    ).T
    R = (
        R @ Ry
    )  # now forward axis is parallel to the ground, but in the direction of the camera (not vehicle's forward)
    f = calib["intrinsic"]["k1"]
    h, w = int(2 * f * np.tan(vfov / 2)), int(
        f * hfov
    )  # cylindrical image has a different size than the fisheye image
    K = np.array(
        [[f, 0, w / 2], [0, f, f * np.tan(vfov / 2 + tilt)], [0, 0, 1]],
        dtype=np.float32,
    )  # intrinsic matrix for the cylindrical projection
    K_inv = np.linalg.inv(K)
    # Create pixel grid and compute a ray for every pixel
    xv, yv = np.meshgrid(range(w), range(h), indexing="xy")
    p = np.stack([xv, yv, np.ones_like(xv)])  # pixel homogeneous coordinates
    p = p.transpose(1, 2, 0)[:, :, :, np.newaxis]
    r = K_inv @ p  # r is in cylindrical coordinates
    r /= r[
        :, :, [2], :
    ]  # r is now in cylindrical coordinates with unit cylindrical radius
    # Convert to Cartesian coordinates
    r[:, :, 2, :] = np.cos(r[:, :, 0, :])
    r[:, :, 0, :] = np.sin(r[:, :, 0, :])
    r[:, :, 1, :] = r[:, :, 1, :]
    r = R @ r  # extrinsic rotation from an upright cylinder to the camera axis
    theta = np.arccos(
        r[:, :, [2], :] / np.linalg.norm(r, axis=2, keepdims=True)
    )  # compute incident angle
    # project the ray onto the fisheye image according to the fisheye model and intrinsic calibration
    c_X = calib["intrinsic"]["cx_offset"] + calib["intrinsic"]["width"] / 2 - 0.5
    c_Y = calib["intrinsic"]["cy_offset"] + calib["intrinsic"]["height"] / 2 - 0.5
    k1, k2, k3, k4 = [calib["intrinsic"]["k%d" % i] for i in range(1, 5)]
    rho = k1 * theta + k2 * theta**2 + k3 * theta**3 + k4 * theta**4
    chi = np.linalg.norm(r[:, :, :2, :], axis=2, keepdims=True)
    u = np.true_divide(
        rho * r[:, :, [0], :], chi, out=np.zeros_like(chi), where=(chi != 0)
    )  # horizontal
    v = np.true_divide(
        rho * r[:, :, [1], :], chi, out=np.zeros_like(chi), where=(chi != 0)
    )  # vertical
    mapx = u[:, :, 0, 0] + c_X
    mapy = v[:, :, 0, 0] * calib["intrinsic"]["aspect_ratio"] + c_Y
    return mapx, mapy


def fisheye_to_cylindrical(image, calib):
    """
    Warp a fisheye image to a cylindrical image
    :param image: fisheye image, as a numpy array
    :param calib: calibration in WoodScape format, as a dictionary
    :return: cylindrical image
    """
    mapx, mapy = get_mapping(calib)
    return mapx, mapy, cv2.remap(
        image,
        mapx.astype(np.float32),
        mapy.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
    )


calib = {
    "extrinsic": {
        "quaternion": [
            0.5946970238045494,
            -0.5837953694518585,
            0.39063952590941586,
            -0.3910488170060691,
        ],
        "translation": [3.7484, 0.0, 0.6577999999999999],
    },
    "intrinsic": {
        "aspect_ratio": 1.0,
        "cx_offset": 3.942,
        "cy_offset": -3.093,
        "height": 966.0,
        "k1": 339.749,
        "k2": -31.988,
        "k3": 48.275,
        "k4": -7.201,
        "model": "radial_poly",
        "poly_order": 4,
        "width": 1280.0,
    },
    "name": "FV",
}
def cylindrical_to_fisheye(cylindrical_image, mapx, mapy, calib):
    """
    Approximate inverse warp from cylindrical back to fisheye.
    :param cylindrical_image: cylindrical image as numpy array
    :param mapx: x mapping from cylindrical to fisheye
    :param mapy: y mapping from cylindrical to fisheye
    :param calib: calibration dictionary (used for output size)
    :return: fisheye image
    """
    fisheye_h = int(calib["intrinsic"]["height"])
    fisheye_w = int(calib["intrinsic"]["width"])

    inv_mapx = np.full((fisheye_h, fisheye_w), -1, dtype=np.float32)
    inv_mapy = np.full((fisheye_h, fisheye_w), -1, dtype=np.float32)

    cyl_h, cyl_w = mapx.shape
    xv, yv = np.meshgrid(np.arange(cyl_w), np.arange(cyl_h), indexing="xy")

    x = np.rint(mapx).astype(np.int32)
    y = np.rint(mapy).astype(np.int32)
    valid = (x >= 0) & (x < fisheye_w) & (y >= 0) & (y < fisheye_h)

    inv_mapx[y[valid], x[valid]] = xv[valid].astype(np.float32)
    inv_mapy[y[valid], x[valid]] = yv[valid].astype(np.float32)

    return cv2.remap(
        cylindrical_image,
        inv_mapx,
        inv_mapy,
        interpolation=cv2.INTER_LINEAR,
    )

def scale_to_input_size(image, target_size):
    """
    Scale an image to the target size while maintaining aspect ratio and padding with zeros if necessary.
    :param image: input image as a numpy array
    :param target_size: tuple (width, height) for the target size
    :return: scaled and padded image
    """
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image


class CylindricalCOCODataset(COCODataset):
    def __init__(self, *args, calib=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.calib = calib

    def load_resized_img(self, index):
        img = self.load_image(index)
        _, _, img_cyl = fisheye_to_cylindrical(img, self.calib)
        r = min(self.img_size[0] / img_cyl.shape[0], self.img_size[1] / img_cyl.shape[1])
        resized_img = cv2.resize(
            img_cyl,
            (int(img_cyl.shape[1] * r), int(img_cyl.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img