
import os
import cv2
import math
from math import cos, sin
import numpy as np
import torch
import json
import pickle
import open3d as o3d
import matplotlib.pyplot as plt

##################################################################
# (right) method-2: https://github.com/hnuzhy/DirectMHP/blob/main/exps/compare_img2pose.py#L91
def plot_3axis_Zaxis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50., limited=True, thickness=2):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    
    if tdx != None and tdy != None:
        face_x = tdx
        face_y = tdy
    else:
        height, width = img.shape[:2]
        face_x = width / 2
        face_y = height / 2

    # X-Axis (pointing to right) drawn in red
    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    
    # Y-Axis (pointing to down) drawn in green
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Plot head oritation line in black
    # scale_ratio = 5
    scale_ratio = 2
    base_len = math.sqrt((face_x - x3)**2 + (face_y - y3)**2)
    if face_x == x3:
        endx = tdx
        if face_y < y3:
            if limited:
                endy = tdy + (y3 - face_y) * scale_ratio
            else:
                endy = img.shape[0]
        else:
            if limited:
                endy = tdy - (face_y - y3) * scale_ratio
            else:
                endy = 0
    elif face_x > x3:
        if limited:
            endx = tdx - (face_x - x3) * scale_ratio
            endy = tdy - (face_y - y3) * scale_ratio
        else:
            endx = 0
            endy = tdy - (face_y - y3) / (face_x - x3) * tdx
    else:
        if limited:
            endx = tdx + (x3 - face_x) * scale_ratio
            endy = tdy + (y3 - face_y) * scale_ratio
        else:
            endx = img.shape[1]
            endy = tdy - (face_y - y3) / (face_x - x3) * (tdx - endx)
    # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,0,0), 2)
    # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (255,255,0), 2)
    # not plot the extend line
    # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,255,255), thickness)

    # X-Axis pointing to right. drawn in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),thickness)
    # Y-Axis pointing to down. drawn in green    
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,255,0),thickness)
    # Z-Axis (out of the screen) drawn in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),thickness)

    return img

##################################################################
# (right) method-3: https://github.com/hassony2/manopth/blob/master/manopth/rodrigues_layer.py#L43
# (right) method-3: https://github.com/MandyMo/HAMR/blob/master/src/util.py#L21
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat

def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat

##################################################################
def predicted_6dof_vis(pkl_file_path):

    with open(pkl_file_path, "rb") as fp:   # unpickling
        tracking_joints_poses= pickle.load(fp)

    point_cloud = o3d.geometry.PointCloud()
    hand_l_points_arr, hand_l_colors_arr = [], []
    hand_r_points_arr, hand_r_colors_arr = [], []
    render_image = np.ones((720, 900, 3)) * 255

    total_cnt = len(tracking_joints_poses)
    for idx, tracking_6dof in enumerate(tracking_joints_poses):
        [frame_id, is_right, euler_angles, kpt_2d, kpt_3d, _,_,_] = tracking_6dof
        [pitch, yaw, roll] = euler_angles
        color_value = 0.3+0.6*idx/total_cnt  # from 0.3 to 0.9
        if is_right:
            hand_r_points_arr.append(kpt_3d)
            hand_r_colors_arr.append((color_value, color_value , 0))  # o3d using RGB
            color_cv2 = (0, int(color_value*255), int(color_value*255))  # cv2 using BGR
        else:
            hand_l_points_arr.append(kpt_3d)
            hand_l_colors_arr.append((0, color_value, color_value))  # o3d using RGB
            color_cv2 = (int(color_value*255), int(color_value*255), 0)  # cv2 using BGR
        cv2.circle(render_image, (int(kpt_2d[0]), int(kpt_2d[1])), 5, color_cv2, -1)
    cv2.imwrite(pkl_file_path.replace(".pkl", ".jpg"), render_image)
    
    points_arr = np.array(hand_r_points_arr + hand_l_points_arr)
    colors_arr = np.array(hand_r_colors_arr + hand_l_colors_arr)
    print(points_arr.shape, colors_arr.shape)
    point_cloud.points = o3d.utility.Vector3dVector(points_arr * 10)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_arr)
    showed_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=point_cloud.get_center())
    o3d.visualization.draw_geometries([point_cloud, showed_axis])

##################################################################
def predicted_6dof_vis_v2(video_id_str):
    from config import video_info_dict
    from infer_trt import init_igev_model, infer_paired_image  # conda activate embodychain
    param_list = init_igev_model() 
   
    fig = plt.figure()
    ax = plt.axes(projection ='3d')  # syntax for 3-D projection
    X_left, Y_left, Z_left, C_left = [], [], [], []
    X_right, Y_right, Z_right, C_right = [], [], [], []
    X_left_2d, Y_left_2d, Z_left_2d, C_left_2d = [], [], [], []
    X_right_2d, Y_right_2d, Z_right_2d, C_right_2d = [], [], [], []
    X_left_kf, Y_left_kf, Z_left_kf, C_left_kf = [], [], [], []
    X_right_kf, Y_right_kf, Z_right_kf, C_right_kf = [], [], [], []
     
    eef_pose_dict = {"right": [], "left": []}

    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    task_name = video_id_str.split("_")[0]
    save_dir = "./results/" + task_name
    os.makedirs(save_dir, exist_ok=True)
    
    pkl_file_path = os.path.join(save_dir, video_info_dict[video_id_str]["cam_l_name"] + "_v2.pkl")
    with open(pkl_file_path, "rb") as fp:   # unpickling
        tracking_joints_poses= pickle.load(fp)

    render_image = np.ones((1024, 1280, 3)) * 255
    total_cnt = len(tracking_joints_poses)
    for idx, joint_pose in enumerate(tracking_joints_poses):
        [is_right, euler_angles, rot_mat, tracking_joint, tracking_joint_3d, kf_flag, gripper, other_info] = joint_pose
        color_value = 0.3+0.6*idx/total_cnt  # from 0.3 to 0.9
        if kf_flag:  # keyframe_flag
            print(other_info)
            [pitch, yaw, roll] = euler_angles
            [frame_lp, frame_rp, frame_id] = other_info
            src_point = [int(tracking_joint[0]), int(tracking_joint[1])]
            pixel_3d = infer_paired_image(param_list, frame_lp, frame_rp, src_point, is_right, gripper)  # about 0.3 seconds
            eef_pose = [pixel_3d, rot_mat.tolist(), gripper, euler_angles, frame_id]  # for gripper state. "1": open, "0": close
            if is_right:
                eef_pose_dict["right"].append(eef_pose)
                X_right_kf.append(pixel_3d[0])
                Y_right_kf.append(pixel_3d[1])
                Z_right_kf.append(pixel_3d[2])
                C_right_kf.append( (color_value, color_value, 0) )  # not cv2
            else:
                eef_pose_dict["left"].append(eef_pose)
                X_left_kf.append(pixel_3d[0])
                Y_left_kf.append(pixel_3d[1])
                Z_left_kf.append(pixel_3d[2])
                C_left_kf.append( (0, color_value, color_value) )  # not cv2
                
            color = (0, 255, 255) if is_right else (255, 255, 0)
            render_image = plot_3axis_Zaxis(render_image, yaw, pitch, roll, 
                tdx=tracking_joint[0], tdy=tracking_joint[1], size=60, thickness=6)
            cv2.circle(render_image, (int(tracking_joint[0]), int(tracking_joint[1])), 9, color, -1)
            cv2.circle(render_image, (int(tracking_joint[0]), int(tracking_joint[1])), 6, (0,0,0), -1)
            cv2.circle(render_image, (int(tracking_joint[0]), int(tracking_joint[1])), 3, (255,255,255), -1)
            
        if is_right:
            color_cv2 = (0, int(color_value*255), int(color_value*255))  # cv2 using BGR
            X_right.append(tracking_joint_3d[0])
            Y_right.append(tracking_joint_3d[1])
            Z_right.append(tracking_joint_3d[2])
            C_right.append( (color_value, color_value, 0) )  # not cv2
            X_right_2d.append(tracking_joint[0])
            Y_right_2d.append(tracking_joint[1])
            Z_right_2d.append(0)
            C_right_2d.append( (color_value, color_value, 0) )  # not cv2
        else:
            color_cv2 = (int(color_value*255), int(color_value*255), 0)  # cv2 using BGR
            X_left.append(tracking_joint_3d[0])
            Y_left.append(tracking_joint_3d[1])
            Z_left.append(tracking_joint_3d[2])
            C_left.append( (0, color_value, color_value) )  # not cv2
            X_left_2d.append(tracking_joint[0])
            Y_left_2d.append(tracking_joint[1])
            Z_left_2d.append(0)
            C_left_2d.append( (0, color_value, color_value) )  # not cv2
        
        cv2.circle(render_image, (int(tracking_joint[0]), int(tracking_joint[1])), 5, color_cv2, -1)

        
    cv2.imwrite(pkl_file_path.replace("_v2.pkl", "_v2.jpg"), render_image)
    
    with open(pkl_file_path.replace("_v2.pkl", "_v2_eep.json"), "w") as json_file:
        json.dump(eef_pose_dict, json_file)


    if "drawer" in video_id_str:  # for paper writing
        # ax.plot3D(X_left, Y_left, X_left, 'gray', linewidth=3, markersize=30)
        # ax.scatter3D(X_left, Y_left, X_left, c=C_left, s=30)  # cyan
        # ax.plot3D(X_right, Y_right, X_right, 'gray', linewidth=3, markersize=30)
        # ax.scatter3D(X_right, Y_right, X_right, c=C_right, s=30)  # yellow

        # ax.plot3D(X_left_2d, Y_left_2d, X_left_2d, 'gray', linewidth=2, markersize=20)
        # ax.scatter3D(X_left_2d, Y_left_2d, X_left_2d, c=C_left_2d, s=20)  # cyan
        # ax.plot3D(X_right_2d, Y_right_2d, X_right_2d, 'gray', linewidth=2, markersize=20)
        # ax.scatter3D(X_right_2d, Y_right_2d, X_right_2d, c=C_right_2d, s=20)  # yellow

        ax.plot3D(X_left_kf, Y_left_kf, X_left_kf, 'gray', linewidth=2, markersize=20)
        ax.scatter3D(X_left_kf, Y_left_kf, X_left_kf, c=C_left_kf, s=20)  # cyan
        ax.plot3D(X_right_kf, Y_right_kf, X_right_kf, 'gray', linewidth=2, markersize=20)
        ax.scatter3D(X_right_kf, Y_right_kf, X_right_kf, c=C_right_kf, s=20)  # yellow
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_title('3D Scatter Points') # syntax for plotting
        plt.show()
           
##################################################################
# conda activate embodychain
# python utils_self.py
if __name__ == '__main__':
    
    # predicted_6dof_vis("./results/drawer/Video_20241128090342905_drawer_v2.pkl")
    
    # predicted_6dof_vis_v2("drawer_04")
    # predicted_6dof_vis_v2("pouring_05")
    # predicted_6dof_vis_v2("uncover_02")
    # predicted_6dof_vis_v2("unscrew_01")
    # predicted_6dof_vis_v2("openbox_01")
    
    # predicted_6dof_vis_v2("invert_01")
    # predicted_6dof_vis_v2("redirect_02")
    # predicted_6dof_vis_v2("reorient_01")
    # predicted_6dof_vis_v2("dualpap_01")
    # predicted_6dof_vis_v2("insertpen_01")
    predicted_6dof_vis_v2("stacking_01")
    
    
    
    
    