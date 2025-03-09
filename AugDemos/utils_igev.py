import cv2
import numpy as np
import open3d as o3d

class StereoRectifyUtil(object):
    @staticmethod
    def rectify_params(stereo_res: dict, height: int, width: int):
        cam1_K = np.array(stereo_res["cam1_k"])
        cam2_K = np.array(stereo_res["cam2_k"])

        cam1_dist = np.array(stereo_res["dist_1"]).reshape(-1)
        cam2_dist = np.array(stereo_res["dist_2"]).reshape(-1)
        R = np.array(stereo_res["R_l_r"])
        t = np.array(stereo_res["t_l_r"])
        
        # 注意后端不再考虑畸变，需前端确认已进行去畸变处理。
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            cameraMatrix1=cam1_K,
            distCoeffs1=cam1_dist,
            cameraMatrix2=cam2_K,
            distCoeffs2=cam2_dist,
            imageSize=(width, height),
            R=R,
            T=t,
            flags=1024,
            newImageSize=(0, 0),
        )

        rectified_image_size = (width, height)

        map_x_1, map_y_1 = cv2.initUndistortRectifyMap(
            cameraMatrix=cam1_K,
            distCoeffs=np.zeros((1, 5)),
            R=R1,
            newCameraMatrix=P1,
            size=rectified_image_size,
            m1type=cv2.CV_32FC1,
        )
        map_x_2, map_y_2 = cv2.initUndistortRectifyMap(
            cameraMatrix=cam2_K,
            distCoeffs=np.zeros((1, 5)),
            R=R2,
            newCameraMatrix=P2,
            size=rectified_image_size,
            m1type=cv2.CV_32FC1,
        )

        rect_cam_k = P2[:3, :3]
        baseline = -P2[0, 3] / P2[0, 0] * 0.001

        return (
            cam1_K,
            R1,
            rect_cam_k,
            baseline,
            map_x_1,
            map_y_1,
            map_x_2,
            map_y_2,
        )

    @staticmethod
    def rectify_img(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray):
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def visualize_rect_imgs(left_img: np.ndarray, right_img: np.ndarray, save_path: str):
        height, width, _ = left_img.shape
        img = np.hstack((left_img, right_img))
        iend = 8
        for i in range(1, iend + 1):
            h = int(height/iend * i)
            img = cv2.line(img, (0, h), (width*2, h), (0,0,255), 2)
        cv2.imwrite(save_path, img)

    @staticmethod
    def from_rectify_pcd_to_origin_depth(
        left_img: np.ndarray,
        src_pcd: o3d.geometry.PointCloud,
        source2target_rotation: np.ndarray,
        target_camera_intrinsic: np.ndarray,
        img_h: int,
        img_w: int,
        depth_scale: float = 1.0,
        depth_trunc: float = 4.0,
    ):
        # 1. get 3xN vertices coordinate of source pcd
        source_point_cloud = np.asarray(src_pcd.points).T

        # 2. point cloud coordinate transformation
        target_point_cloud = source2target_rotation.dot(source_point_cloud)

        assert (target_point_cloud[2] > 0).all()
        return target_point_cloud

        # 3. target point cloud reproject to image
        target_depth_map = np.zeros((img_h, img_w), dtype=np.float32)
        target_uv = np.around(np.dot(target_camera_intrinsic, target_point_cloud) / target_point_cloud[2, :])

        # 交换前两列
        target_uv[2, :] = target_uv[0, :]
        target_uv = target_uv[1:, :]

        val_locs = (
            (target_uv[0] < img_h)
            & (target_uv[0] >= 0)
            & (target_uv[1] < img_w)
            & (target_uv[1] >= 0)
        )

        # 图像平面内的有效x,y位置
        row_col_pos = target_uv.T[val_locs].astype(np.int32)
        # 图像平面内的有效z位置
        target_pt_zs = target_point_cloud[2][val_locs]

        # 按深度值从大到小排序后，依次投影到目标图像平面，最终同像素位置小深度值会替换大深度值
        z_sorted_idxs = np.argsort(target_pt_zs)
        z_sorted_idxs = z_sorted_idxs[::-1]

        row_col_pos = row_col_pos[z_sorted_idxs]
        target_depth_map[row_col_pos[:, 0], row_col_pos[:, 1]] = target_pt_zs[z_sorted_idxs]

        # 4. return target depth map and pcd
        height, width, _ = left_img.shape
        fx = target_camera_intrinsic[0, 0]
        fy = target_camera_intrinsic[1, 1]
        cx = target_camera_intrinsic[0, 2]
        cy = target_camera_intrinsic[1, 2]
        left_img_o3d = o3d.geometry.Image((left_img).astype(np.uint8))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        depth_img_o3d = o3d.geometry.Image(target_depth_map.astype(np.float32))
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            left_img_o3d,
            depth_img_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        tgt_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)

        return target_depth_map, tgt_pcd
