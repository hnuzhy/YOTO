import os
import torch
import argparse
import time
import cv2
import yaml
import open3d as o3d
import numpy as np
from glob import glob

from utils_igev import StereoRectifyUtil
from glia.dl.pipelines import StereoInterface


def generate_pcd(left_image, disp, fx, fy, cx, cy, baseline, object_mask_binary=None, roi_mask_binary=None):
    height, width, _ = left_image.shape
    depth = fx * baseline / disp
    
    left_img_o3d = o3d.geometry.Image(left_image.astype(np.uint8))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    depth_img_o3d = o3d.geometry.Image(depth.astype(np.float32))
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        left_img_o3d,
        depth_img_o3d,
        depth_scale=1.0,
        depth_trunc=depth.max()+1,  # do not remove 3d point in any pixel
        # depth_trunc=4.0,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
    print(len(pcd.points), np.array(pcd.points).shape, height*width)
    
    if object_mask_binary is not None:
        # https://www.cnblogs.com/massquantity/p/8908859.html
        left_pts_idx = np.where(object_mask_binary.reshape(-1) > 0)[0]
        print("left points:", len(left_pts_idx))
        pcd = pcd.select_by_index(left_pts_idx)
        
    if roi_mask_binary is not None:
        left_pts_idx = np.where(roi_mask_binary.reshape(-1) > 0)[0]
        print("left points:", len(left_pts_idx))
        pcd = pcd.select_by_index(left_pts_idx)
        
    return pcd, depth


def init_igev_model():
    model_path = "./kingfisher/igev.trt"
    calib_file = "./kingfisher/calib_1106.yaml"
    im_height, im_width = 1024, 1280

    # IGEV Tensorrt模型初始化成功
    model = StereoInterface(backend="trt", model_path=model_path)
    model.cuda()
    
    # 加载相机参数文件
    with open(calib_file, "r", encoding="utf-8") as f:
        stereo_res = yaml.load(stream=f, Loader=yaml.FullLoader)
    ret_results = StereoRectifyUtil.rectify_params(stereo_res, im_height, im_width)
    ori_k1, R1, rect_k1, baseline, map_x_1, map_y_1, map_x_2, map_y_2 = ret_results
    
    model.set_camera(0, im_height, im_width,
        fx=rect_k1[0, 0], fy=rect_k1[1, 1], cx=rect_k1[0, 2], cy=rect_k1[1, 2])
    model.set_camera(1, im_height, im_width,
        fx=rect_k1[0, 0], fy=rect_k1[1, 1], cx=rect_k1[0, 2], cy=rect_k1[1, 2])
   
    param_list = [model, R1, rect_k1, baseline, map_x_1, map_y_1, map_x_2, map_y_2]
    return param_list

   
if __name__ == '__main__':
    
    print("infer scene point cloud using kingfisher and igev!!!")
