import os
import cv2
import time
import json
import shutil
import open3d as o3d
import numpy as np
import copy
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from infer_trt import init_igev_model, generate_pcd
from utils_igev import StereoRectifyUtil


##################################################################
cfg_dict_init = {
    "arm1": {
        "gripper_port": '/dev/ttyUSB0',  # for Ubuntu, sudo chmod 666 /dev/ttyUSB0
        "robot_ip_add": "192.168.9.134",  # robot IP address "192.168.31.134"
        "handeye_para": np.array([
            [-0.14495955, -0.82672254,  0.54361435, -0.46128096],
            [-0.98826554,  0.09424301, -0.120206  , -0.55922609],
            [ 0.04814516, -0.55466034, -0.83068282,  0.77215299],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])  # camera calib in 2024-11-20
    },
    "arm2": {
        "gripper_port": '/dev/ttyUSB1',  # for Ubuntu, sudo chmod 666 /dev/ttyUSB1
        "robot_ip_add": "192.168.9.135",  # robot IP address "192.168.31.135"
        "handeye_para": np.array([
            [ 0.15464175,  0.82413502, -0.54487375,  0.48699159],
            [ 0.98677   , -0.10165977,  0.12629433, -0.77214491],
            [ 0.04869184, -0.55719545, -0.82895256,  0.76490652],
            [-0.        ,  0.        , -0.        ,  1.        ]
        ])  # camera calib in 2024-11-20 
    },
}
##################################################################
def preprocess_eef_to_act(json_path):
    act_list = []
    
    if os.path.exists(json_path): # async task with only one json file
        eef_pose_dict_new = json.load(open(json_path, "r"))
    else:  # sync task with only two json files
        eef_pose_dict_new_L = json.load(open(json_path.replace(".json", "_robotL.json"), "r"))
        eef_pose_dict_new_R = json.load(open(json_path.replace(".json", "_robotR.json"), "r"))
        eef_pose_dict_new = {}
        eef_pose_dict_new["L"] = eef_pose_dict_new_L["L"]
        eef_pose_dict_new["R"] = eef_pose_dict_new_R["R"]
    
    # [step_id] + robot_cur_state + [cur_gripper_state_L]
    eef_pose_list_L = eef_pose_dict_new["L"]
    assert eef_pose_list_L[0][0] == -1, "You must give the init left robot pose"
    eef_pose_list_R = eef_pose_dict_new["R"]
    assert eef_pose_list_R[0][0] == -1, "You must give the init right robot pose"
    max_step_id_l, max_step_id_r = eef_pose_list_L[-1][0], eef_pose_list_R[-1][0]
    max_step_id = max(max_step_id_l, max_step_id_r) + 1  # step_id starts from 0
    
    # format 1
    if max_step_id_l == max_step_id_r:  # sync task
        for eef_pose_l, eef_pose_r in zip(eef_pose_list_L[1:], eef_pose_list_R[1:]):  # remove init step
            act_list.append(eef_pose_l[1:])  # 7 dof = 3 location + 3 rotation + 1 gripper
            act_list.append(eef_pose_r[1:])  # 7 dof = 3 location + 3 rotation + 1 gripper
        return act_list, None, None
    else: # async task
        steps_list_L = [i[0] for i in eef_pose_list_L]
        steps_list_R = [i[0] for i in eef_pose_list_R]
        for step_id in range(max_step_id):
            if step_id in steps_list_L:
                for eef_pose_l in eef_pose_list_L:  # one step may contain multiple actions
                    if eef_pose_l[0] == step_id:
                        act_list.append(eef_pose_l[1:])  # 7 dof = 3 location + 3 rotation + 1 gripper
            if step_id in steps_list_R:
                for eef_pose_r in eef_pose_list_R:  # one step may contain multiple actions
                    if eef_pose_r[0] == step_id:
                        act_list.append(eef_pose_r[1:])  # 7 dof = 3 location + 3 rotation + 1 gripper
        return act_list, steps_list_L, steps_list_R
##################################################################
def preprocess_obs2pcd_eep2act_w_aug(param_list, img_l_path, img_r_path, lp_mask_path, 
                                     task_name, prefix_name, json_path, aug_times=50, save_res_dir=None):
    
    [model, R1, rect_k1, baseline, map_x_1, map_y_1, map_x_2, map_y_2] = param_list
    
    using_erode = False  # True or False, the default value is False
    if using_erode: kernel = np.ones((8, 8),np.uint8)
    fname = img_l_path.split("/")[-1].split(".")[-2]

    debug_flag_level = 3  # 0: no vis; 1: vis _objs.ply; 2: _paired.jpg; 3: _rect_objmask_b.png

    left_image = cv2.imread(img_l_path)
    right_image = cv2.imread(img_r_path)
    rect_left_img = StereoRectifyUtil.rectify_img(left_image, map_x_1, map_y_1)
    rect_right_img = StereoRectifyUtil.rectify_img(right_image, map_x_2, map_y_2)

    if debug_flag_level>=2 and save_res_dir is not None:
        img_path = os.path.join(save_res_dir, f"{fname}_paired.jpg")
        StereoRectifyUtil.visualize_rect_imgs(rect_left_img, rect_right_img, img_path)  # for vis debug

    rect_left_img = cv2.cvtColor(rect_left_img, cv2.COLOR_BGR2RGB)
    rect_right_img = cv2.cvtColor(rect_right_img, cv2.COLOR_BGR2RGB)
    model.update_camera_data(0, rect_left_img)
    model.update_camera_data(1, rect_right_img)

    start_time = time.time()
    disp_tensor = model.inference_disp([0, 1])
    end_time = time.time()
    print(f"Inference time: {end_time-start_time} s")
    
    disp_npy = disp_tensor.squeeze().cpu().numpy()
    fx=rect_k1[0, 0]
    fy=rect_k1[1, 1]
    cx=rect_k1[0, 2]
    cy=rect_k1[1, 2]

    
    if debug_flag_level>=1 and save_res_dir is not None:
        lp_rect_path = os.path.join(save_res_dir, f"{fname}_rect.jpg")
        cv2.imwrite(lp_rect_path, rect_left_img[:,:,::-1])
        print("saving the rectified left image file ...")
    
    if lp_mask_path:
        object_mask_rgb_ori = cv2.imread(lp_mask_path)  # un-rectify
        height, width, _ = rect_left_img.shape  # example [1024, 1280, 3]
        object_mask_rgb_ori = cv2.resize(object_mask_rgb_ori, (width, height), cv2.INTER_NEAREST)
        object_mask_rgb = StereoRectifyUtil.rectify_img(object_mask_rgb_ori, map_x_1, map_y_1)  # rectify
        
        # https://blog.csdn.net/weixin_45939019/article/details/104391620
        if using_erode:
            object_mask_rgb = cv2.erode(object_mask_rgb, kernel)
            
        # https://blog.csdn.net/a19990412/article/details/81172426
        # https://blog.csdn.net/JNingWei/article/details/77747959
        object_mask_gray = cv2.cvtColor(object_mask_rgb, cv2.COLOR_BGR2GRAY)
        ret, object_mask_binary = cv2.threshold(object_mask_gray, 1, 255, cv2.THRESH_BINARY)
        
        if debug_flag_level>=3 and save_res_dir is not None:
            lp_mask_binary_path = os.path.join(save_res_dir, f"{fname}_rect_objmask_b.png")
            cv2.imwrite(lp_mask_binary_path, object_mask_binary)
            print("saving the object_mask_binary file ...")

        pcd, depth = generate_pcd(rect_left_img, disp_npy, 
            fx, fy, cx, cy, baseline, object_mask_binary=object_mask_binary, roi_mask_binary=None)
        # pcd.rotate(R1)
        if debug_flag_level>=1 and save_res_dir is not None:
            o3d.io.write_point_cloud(os.path.join(save_res_dir, f"{fname}_objs.ply"), pcd)  # for vis debug
            print("saving the pcd file ...", f"{fname}_objs.ply")

        if task_name in ["drawer", "pouring"]:  # two tasks with two objects
            if task_name == "drawer": w_split_pixel = 520
            if task_name == "pouring": w_split_pixel = 665
            
            object_mask_rgb_ori_L = object_mask_rgb_ori.copy()
            object_mask_rgb_ori_L[:, w_split_pixel:, :] = object_mask_rgb_ori[0,0,0]
            object_mask_rgb_L = StereoRectifyUtil.rectify_img(object_mask_rgb_ori_L, map_x_1, map_y_1)  # rectify
            if using_erode:
                object_mask_rgb_L = cv2.erode(object_mask_rgb_L, kernel)
            object_mask_gray_L = cv2.cvtColor(object_mask_rgb_L, cv2.COLOR_BGR2GRAY)
            ret, object_mask_binary_L = cv2.threshold(object_mask_gray_L, 1, 255, cv2.THRESH_BINARY)
            if debug_flag_level>=3 and save_res_dir is not None:
                lp_mask_binary_path = os.path.join(save_res_dir, f"{fname}_rect_objmask_b_L.png")
                cv2.imwrite(lp_mask_binary_path, object_mask_binary_L)
                print("saving the object_mask_binary_L file ...")
            pcd_L, depth = generate_pcd(rect_left_img, disp_npy, 
                fx, fy, cx, cy, baseline, object_mask_binary=object_mask_binary_L, roi_mask_binary=None)
        
            object_mask_rgb_ori_R = object_mask_rgb_ori.copy()
            object_mask_rgb_ori_R[:, :w_split_pixel, :] = object_mask_rgb_ori[0,0,0]
            object_mask_rgb_R = StereoRectifyUtil.rectify_img(object_mask_rgb_ori_R, map_x_1, map_y_1)  # rectify
            if using_erode:
                object_mask_rgb_R = cv2.erode(object_mask_rgb_R, kernel)
            object_mask_gray_R = cv2.cvtColor(object_mask_rgb_R, cv2.COLOR_BGR2GRAY)
            ret, object_mask_binary_R = cv2.threshold(object_mask_gray_R, 1, 255, cv2.THRESH_BINARY)
            if debug_flag_level>=3 and save_res_dir is not None:
                lp_mask_binary_path = os.path.join(save_res_dir, f"{fname}_rect_objmask_b_R.png")
                cv2.imwrite(lp_mask_binary_path, object_mask_binary_R)
                print("saving the object_mask_binary_R file ...")
            pcd_R, depth = generate_pcd(rect_left_img, disp_npy, 
                fx, fy, cx, cy, baseline, object_mask_binary=object_mask_binary_R, roi_mask_binary=None)
            
    else:
        pcd, depth = generate_pcd(rect_left_img, disp_npy, 
            fx, fy, cx, cy, baseline, object_mask_binary=None, roi_mask_binary=None)
        # pcd.rotate(R1)
        if debug_flag_level>=1 and save_res_dir is not None:
            o3d.io.write_point_cloud(os.path.join(save_res_dir, f"{fname}_scene.ply"), pcd)  # for vis debug
            print("saving the pcd file ...", f"{fname}_scene.ply")
    
    # return pcd
    
    pcd_o3ds, acts = [], []
    
    # save the first init demostration
    pcd_o3ds.append(pcd)
    act_list, steps_list_L, steps_list_R = preprocess_eef_to_act(json_path)
    acts.append(act_list)

    # save the left augmented demostrations [pre-defined parameters]
    if task_name in ["drawer", "pouring"]:  # two tasks with two objects
        R_str, L_str = prefix_name.split("_")[1], prefix_name.split("_")[2]
        if task_name == "drawer":
            object_pos_id, drawer_pos_id = int(L_str.split("-")[1]), int(R_str.split("-")[1])
            drawer_delta_list = [  # 45cm x 30cm --> 20cm x 5cm
                [0.045, -0.045, 0.025, -0.025, 0.000, -0.000], 
                [0.045, -0.015, 0.025, -0.025, 0.000, -0.000],
                [0.015, -0.045, 0.025, -0.025, 0.000, -0.000],
            ]
            dxpR, dxnR, dypR, dynR, dzpR, dznR = drawer_delta_list[drawer_pos_id - 1]
            drawer_related_step_ids = [0, 1, 2, 3, 4, 6, 7, 8, 9]
            
            object_delta_list = [  # 45cm x 25cm --> 35cm x 15cm
                [0.075, -0.075, 0.100, -0.050, 0.000, -0.000], 
                [0.025, -0.075, 0.100, -0.050, 0.000, -0.000],
                [0.075, -0.025, 0.100, -0.050, 0.000, -0.000],
            ]
            dxpL, dxnL, dypL, dynL, dzpL, dznL = object_delta_list[object_pos_id - 1]
            object_related_step_ids = [5]
            
        if task_name == "pouring":
            bottle_pos_id, mugcup_pos_id = int(L_str.split("-")[1]), int(R_str.split("-")[1])
            mugcup_delta_list = [  # 30cm x 20cm --> 30cm x 10cm
                [0.075, -0.075, 0.075, -0.025, 0.000, -0.000], 
                [0.075, -0.000, 0.075, -0.025, 0.000, -0.000],
                [0.000, -0.075, 0.075, -0.025, 0.000, -0.000],
            ]
            dxpR, dxnR, dypR, dynR, dzpR, dznR = mugcup_delta_list[mugcup_pos_id - 1]
            mugcup_related_step_ids = [0]  # not including step_ids 3 and 6
            
            bottle_delta_list = [  # 20cm x 10cm --> 20cm x 10cm
                [0.050, -0.050, 0.000, -0.050, 0.000, -0.000], 
                [0.000, -0.050, 0.050, -0.000, 0.000, -0.000],
                [0.050, -0.000, 0.050, -0.000, 0.000, -0.000],
            ]
            dxpL, dxnL, dypL, dynL, dzpL, dznL = bottle_delta_list[bottle_pos_id - 1]
            bottle_related_step_ids = [1]  # not including step_ids 2, 4 and 5

    else:  # other three tasks with single object
        R_str = prefix_name.split("_")[1]
        if task_name == "unscrew":
            bottle_pos_id = int(R_str.split("-")[1])
            bottle_delta_list = [  # 20cm x 20cm --> 20cm x 20cm
                [0.050, -0.050, 0.000, -0.050, 0.000, -0.000], 
                [0.050, -0.050, 0.050, -0.050, 0.000, -0.000],
                [0.050, -0.050, 0.050, -0.000, 0.000, -0.000],
                [0.000, -0.050, 0.000, -0.050, 0.000, -0.000],
                [0.000, -0.050, 0.050, -0.050, 0.000, -0.000],
                [0.000, -0.050, 0.050, -0.000, 0.000, -0.000],
                [0.050, -0.000, 0.000, -0.050, 0.000, -0.000],
                [0.050, -0.000, 0.050, -0.050, 0.000, -0.000],
                [0.050, -0.000, 0.050, -0.000, 0.000, -0.000],
            ]
            dxpL, dxnL, dypL, dynL, dzpL, dznL = bottle_delta_list[bottle_pos_id - 1]
            bottle_related_step_ids = [0]  # not including step_ids 1, 4, 6 and 7
            
        if task_name == "uncover":
            lidbox_pos_id = int(R_str.split("-")[1])
            lidbox_delta_list = [  # 40cm x 20cm --> 20cm x 6cm
                [0.050, -0.050, 0.010, -0.010, 0.000, -0.000], 
                [0.050, -0.050, 0.010, -0.010, 0.000, -0.000],
                [0.050, -0.050, 0.010, -0.010, 0.000, -0.000],
                [0.000, -0.050, 0.010, -0.010, 0.000, -0.000],
                [0.000, -0.050, 0.010, -0.010, 0.000, -0.000],
                [0.000, -0.050, 0.010, -0.010, 0.000, -0.000],
                [0.050, -0.000, 0.010, -0.010, 0.000, -0.000],
                [0.050, -0.000, 0.010, -0.010, 0.000, -0.000],
                [0.050, -0.000, 0.010, -0.010, 0.000, -0.000],
            ]
            dxpL, dxnL, dypL, dynL, dzpL, dznL = lidbox_delta_list[lidbox_pos_id - 1]
            
        if task_name == "openbox":
            expbox_pos_id = int(R_str.split("-")[1])
            expbox_delta_list = [  # 20cm x 10cm --> 20cm x 10cm
                [0.050, -0.000, 0.025, -0.025, 0.000, -0.000],
                [0.050, -0.000, 0.000, -0.025, 0.000, -0.000],
                [0.050, -0.000, 0.025, -0.000, 0.000, -0.000],              
                [0.050, -0.050, 0.025, -0.025, 0.000, -0.000], 
                [0.050, -0.050, 0.000, -0.025, 0.000, -0.000],
                [0.050, -0.050, 0.025, -0.000, 0.000, -0.000],
                [0.000, -0.050, 0.025, -0.025, 0.000, -0.000],
                [0.000, -0.050, 0.000, -0.025, 0.000, -0.000],
                [0.000, -0.050, 0.025, -0.000, 0.000, -0.000],
            ]
            dxpL, dxnL, dypL, dynL, dzpL, dznL = expbox_delta_list[expbox_pos_id - 1]
    
    # save the left augmented demostrations [begin to augment]   
    for aug_id in tqdm(range(aug_times)):
        act_list_new = np.array(act_list.copy())
        
        if task_name in ["drawer", "pouring"]:  # two tasks with two objects
            pcd_L_copy, pcd_R_copy = copy.deepcopy(pcd_L), copy.deepcopy(pcd_R)       
            handeye_camera_L = cfg_dict_init["arm1"]["handeye_para"]
            pts_arr_L_camera = np.array(pcd_L_copy.points)  # (N, 3), for object / bottle
            pts_arr_L_robot = handeye_camera_L[:3, :3] @ pts_arr_L_camera.T  # (3, N)
            pts_arr_L_robot = pts_arr_L_robot.T + handeye_camera_L[:3, -1]  # (N, 3)
            
            handeye_camera_R = cfg_dict_init["arm2"]["handeye_para"]
            pts_arr_R_camera = np.array(pcd_R_copy.points)  # (N, 3), for drawer / mugcup
            pts_arr_R_robot = handeye_camera_R[:3, :3] @ pts_arr_R_camera.T  # (3, N)
            pts_arr_R_robot = pts_arr_R_robot.T + handeye_camera_R[:3, -1]  # (N, 3)
            
            if task_name == "drawer":
                x_offset = np.random.rand() * (dxpL - dxnL) + dxnL
                y_offset = np.random.rand() * (dypL - dynL) + dynL
                z_offset = np.random.rand() * (dzpL - dznL) + dznL
                object_pos_offset_L = np.array([x_offset, y_offset, z_offset])
                object_pos_offset_R = np.array([i*j for (i, j) in zip(object_pos_offset_L, [-1,-1,1])])  # note the z-axis
                pts_arr_L_robot += object_pos_offset_L
                for step_id in object_related_step_ids:
                    if step_id in steps_list_L: act_list_new[step_id][:3] += object_pos_offset_L
                    if step_id in steps_list_R: act_list_new[step_id][:3] += object_pos_offset_R
                
                # rotation augmentation around z-axis https://en.wikipedia.org/wiki/Rotation_matrix
                rot_offset_rad = (np.random.rand() * 30 - 15) * np.pi / 180
                sin_off, cos_off = np.sin(rot_offset_rad), np.cos(rot_offset_rad)
                rot_mat = np.array([[cos_off, -sin_off, 0], [sin_off, cos_off, 0], [0, 0, 1]])
                pt_robot_avg = pts_arr_L_robot.mean(axis=0)
                pts_arr_robot_avg = pts_arr_L_robot - pt_robot_avg
                pts_arr_robot_avg = rot_mat @ pts_arr_robot_avg.T
                pts_arr_L_robot = pts_arr_robot_avg.T + pt_robot_avg
                
                x_offset = np.random.rand() * (dxpR - dxnR) + dxnR
                y_offset = np.random.rand() * (dypR - dynR) + dynR
                z_offset = np.random.rand() * (dzpR - dznR) + dznR
                drawer_pos_offset_R = np.array([x_offset, y_offset, z_offset])
                drawer_pos_offset_L = np.array([i*j for (i, j) in zip(drawer_pos_offset_R, [-1,-1,1])])  # note the z-axis
                pts_arr_R_robot += drawer_pos_offset_R
                for step_id in drawer_related_step_ids:
                    if step_id in steps_list_L: act_list_new[step_id][:3] += drawer_pos_offset_L
                    if step_id in steps_list_R: act_list_new[step_id][:3] += drawer_pos_offset_R
                
            if task_name == "pouring":
                x_offset = np.random.rand() * (dxpL - dxnL) + dxnL
                y_offset = np.random.rand() * (dypL - dynL) + dynL
                z_offset = np.random.rand() * (dzpL - dznL) + dznL
                bottle_pos_offset_L = np.array([x_offset, y_offset, z_offset])
                bottle_pos_offset_R = np.array([i*j for (i, j) in zip(bottle_pos_offset_L, [-1,-1,1])])  # note the z-axis
                pts_arr_L_robot += bottle_pos_offset_L
                for step_id in bottle_related_step_ids:
                    if step_id in steps_list_L: act_list_new[step_id][:3] += bottle_pos_offset_L
                    if step_id in steps_list_R: act_list_new[step_id][:3] += bottle_pos_offset_R
                    
                x_offset = np.random.rand() * (dxpR - dxnR) + dxnR
                y_offset = np.random.rand() * (dypR - dynR) + dynR
                z_offset = np.random.rand() * (dzpR - dznR) + dznR
                mugcup_pos_offset_R = np.array([x_offset, y_offset, z_offset])
                mugcup_pos_offset_L = np.array([i*j for (i, j) in zip(mugcup_pos_offset_R, [-1,-1,1])])  # note the z-axis
                pts_arr_R_robot += mugcup_pos_offset_R
                for step_id in mugcup_related_step_ids:
                    if step_id in steps_list_L: act_list_new[step_id][:3] += mugcup_pos_offset_L
                    if step_id in steps_list_R: act_list_new[step_id][:3] += mugcup_pos_offset_R

            pts_arr_L_camera = handeye_camera_L[:3, :3].T @ (pts_arr_L_robot - handeye_camera_L[:3, -1]).T  # (3, N)
            pcd_L_copy.points = o3d.utility.Vector3dVector(pts_arr_L_camera.T)  # (N, 3)
            pts_arr_R_camera = handeye_camera_R[:3, :3].T @ (pts_arr_R_robot - handeye_camera_R[:3, -1]).T  # (3, N)
            pcd_R_copy.points = o3d.utility.Vector3dVector(pts_arr_R_camera.T)  # (N, 3)
            
            pcd_mixed = pcd_L_copy
            pcd_points = np.concatenate([np.array(pcd_L_copy.points), np.array(pcd_R_copy.points)], axis=0)
            pcd_mixed.points = o3d.utility.Vector3dVector(pcd_points)
            pcd_colors = np.concatenate([np.array(pcd_L_copy.colors), np.array(pcd_R_copy.colors)], axis=0)
            pcd_mixed.colors = o3d.utility.Vector3dVector(pcd_colors)
            pcd_o3ds.append(pcd_mixed)
            acts.append(act_list_new.tolist())
            
            if debug_flag_level>=3 and save_res_dir is not None:
                o3d.io.write_point_cloud(os.path.join(save_res_dir, f"{fname}_mixed_{aug_id}.ply"), pcd_mixed)  # for vis debug
                print("saving the pcd file ...", f"{fname}_mixed_{aug_id}.ply")
         
        else:  # other three tasks with single object
            pcd_copy = copy.deepcopy(pcd)
            handeye_camera_L = cfg_dict_init["arm1"]["handeye_para"]
            pts_arr_camera = np.array(pcd_copy.points)  # (N, 3), for bottle / lidbox / expbox
            pts_arr_robot = handeye_camera_L[:3, :3] @ pts_arr_camera.T  # (3, N)
            pts_arr_robot = pts_arr_robot.T + handeye_camera_L[:3, -1]  # (N, 3)
            
            x_offset = np.random.rand() * (dxpL - dxnL) + dxnL
            y_offset = np.random.rand() * (dypL - dynL) + dynL
            z_offset = np.random.rand() * (dzpL - dznL) + dznL
            if task_name == "unscrew":
                bottle_pos_offset_L = np.array([x_offset, y_offset, z_offset])
                bottle_pos_offset_R = np.array([i*j for (i, j) in zip(bottle_pos_offset_L, [-1,-1,1])])  # note the z-axis
                pts_arr_robot += bottle_pos_offset_L
                for step_id in bottle_related_step_ids:
                    if step_id in steps_list_L: act_list_new[step_id][:3] += bottle_pos_offset_L
                    if step_id in steps_list_R: act_list_new[step_id][:3] += bottle_pos_offset_R
            
            if task_name == "uncover" or task_name == "openbox": # lidbox_pos_offset / expbox_pos_offset
                box_pos_offset = np.array([x_offset, y_offset, z_offset])
                pts_arr_robot += box_pos_offset
                for step_id in range(len(act_list_new)//2):
                    act_list_new[2*step_id][0] += x_offset
                    act_list_new[2*step_id][1] += y_offset
                    act_list_new[2*step_id][2] += z_offset
                    act_list_new[2*step_id+1][0] -= x_offset
                    act_list_new[2*step_id+1][1] -= y_offset
                    act_list_new[2*step_id+1][2] += z_offset
                    
            pts_arr_camera = handeye_camera_L[:3, :3].T @ (pts_arr_robot - handeye_camera_L[:3, -1]).T  # (3, N)
            pcd_copy.points = o3d.utility.Vector3dVector(pts_arr_camera.T)  # (N, 3)
            
            pcd_o3ds.append(pcd_copy)
            acts.append(act_list_new.tolist())
            
            if debug_flag_level>=3 and save_res_dir is not None:
                o3d.io.write_point_cloud(os.path.join(save_res_dir, f"{fname}_aug_{aug_id}.ply"), pcd_copy)  # for vis debug
                print("saving the pcd file ...", f"{fname}_aug_{aug_id}.ply")
            
            
    return pcd_o3ds, acts
           
##################################################################
def keep_n_points(pc_point: np.ndarray, pc_color: np.ndarray, pc_num: int = 2048):
    n = pc_point.shape[0]
    replace = n < pc_num
    idx = np.arange(n)
    indices_preserve = idx[np.random.choice(n, pc_num, replace=replace)]
    pc_point = pc_point[indices_preserve]
    pc_color = pc_color[indices_preserve]
    return pc_point, pc_color
def keep_n_points_wo_color(pc_point: np.ndarray, pc_num: int = 2048):
    n = pc_point.shape[0]
    replace = n < pc_num
    idx = np.arange(n)
    indices_preserve = idx[np.random.choice(n, pc_num, replace=replace)]
    pc_point = pc_point[indices_preserve]
    return pc_point
##################################################################

if __name__ == '__main__':
    
    debug_flag = False  # True or False
    aug_times = 100  # 1 / 5 / 25 / 100 / 500
    left_pts_num = 4096  # 32768 / 16384 / 8192 / 4096 / 2048
    task_names = ["openbox", "uncover", "unscrew", "pouring", "drawer"]
    # task_names = ["unscrew", "pouring", "drawer"]
    # task_names = ["uncover"]

    param_list = init_igev_model() 
    for task_name in task_names:
        ##################################################################
        json_dir = f"./datasets/{task_name}/"
        mask_dir = f"./datasets/{task_name}_masks/"
        save_res_dir = f"./datasets/{task_name}_debug/" if debug_flag else None
        if save_res_dir is not None:
            if os.path.exists(save_res_dir):
                shutil.rmtree(save_res_dir)
            os.mkdir(save_res_dir)

        preprocessed_dataset_json = []
        save_dataset_json = f"./datasets/data_jsons/{task_name}_preprocessed_aug{aug_times}x.json"
        
        masks_list = os.listdir(mask_dir)
        masks_list.sort()
        debug_ids = np.random.choice(len(masks_list), 10) if debug_flag else []
        for sample_id, mask_name in enumerate(masks_list):
            if debug_flag and sample_id not in debug_ids: continue
            
            if sample_id % 75 == 0 and sample_id != 0:  # too many demonstrations, split them into multiple parts
                temp_id = sample_id // 75
                save_dataset_json_temp = save_dataset_json.replace("x.json", f"x_{temp_id}.json")
                print("Saving ...", save_dataset_json_temp)
                with open(save_dataset_json_temp, "w") as json_file:
                    json.dump(preprocessed_dataset_json, json_file)
                preprocessed_dataset_json = []
            
            prefix_name = mask_name.replace("_imgL-removebg-preview.png", "")
            mask_path = os.path.join(mask_dir, mask_name)
            left_img_name = mask_name.replace("imgL-removebg-preview.png", "imgL.jpg")
            left_img_path = os.path.join(json_dir, left_img_name)
            right_img_name = mask_name.replace("imgL-removebg-preview.png", "imgR.jpg")
            right_img_path = os.path.join(json_dir, right_img_name)
            
            json_name = mask_name.replace("_imgL-removebg-preview.png", ".json")
            json_path = os.path.join(json_dir, json_name)    

            print("Preprocessing One Demonstration/Sample:", sample_id, json_name)
            
            pcd_o3ds, acts = preprocess_obs2pcd_eep2act_w_aug(  # observation / actions
                param_list, left_img_path, right_img_path, mask_path, 
                task_name, prefix_name, json_path, aug_times=aug_times, save_res_dir=save_res_dir) 
            
            for pcd_o3d, act_list in zip(pcd_o3ds, acts):
                pts_arr_cam = np.array(pcd_o3d.points)  # (N, 3), o3d pcd --> npy arr points, without color
                if pts_arr_cam.shape[0] > left_pts_num:  # we do not need too many 3d points
                    pts_arr_cam = keep_n_points_wo_color(pts_arr_cam, left_pts_num)
                pts_arr_cam_list = np.round(pts_arr_cam, 6).tolist()  # we do not need too accurate 3d points
            
                preprocessed_dataset_json.append({"obs": pts_arr_cam_list, "actions": act_list})
        
        if sample_id < 75:            
            print("Saving ...", save_dataset_json)
            with open(save_dataset_json, "w") as json_file:
                json.dump(preprocessed_dataset_json, json_file)
        else:  # too many demonstrations, split them into multiple parts
            temp_id = sample_id // 75 + 1
            save_dataset_json_temp = save_dataset_json.replace("x.json", f"x_{temp_id}.json")
            print("Saving ...", save_dataset_json_temp)
            with open(save_dataset_json_temp, "w") as json_file:
                json.dump(preprocessed_dataset_json, json_file)
            
            
        ##################################################################
    #===============================================================
    # this version, we begin to apply the demonstration augmentation
    #===============================================================

