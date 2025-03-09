import os
import cv2
import time
import json
import shutil
import open3d as o3d
import numpy as np
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
task_cfg_info = {
    "drawer": {"steps_length": 10, "arm_order": ["R","R","L","R","L","L","L","L","L","R"]},
    "pouring": {"steps_length": 11, "arm_order": ["R","R","L","L","L","R","L","L","L","L","R"]},
    "unscrew": {"steps_length": 12, "arm_order": ["L","L","L","R","R","R","R","R","L","R","L","L"]},
    "uncover": {"steps_length": 24, "arm_order": ["L","R","L","R","L","R","L","R","L","R","L","R",
                                                  "L","R","L","R","L","R","L","R","L","R","L","R"]},
    "openbox": {"steps_length": 32, "arm_order": ["L","R","L","R","L","R","L","R","L","R","L","R",
                                                  "L","R","L","R","L","R","L","R","L","R","L","R",
                                                  "L","R","L","R","L","R","L","R"]},
}
##################################################################
def preprocess_obs_to_pcd(param_list, img_l_path, img_r_path, lp_mask_path, save_res_dir=None):
    
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
        object_mask_rgb = cv2.imread(lp_mask_path)  # un-rectify
        height, width, _ = rect_left_img.shape  # example [1024, 1280, 3]
        object_mask_rgb = cv2.resize(object_mask_rgb, (width, height), cv2.INTER_NEAREST)
        object_mask_rgb = StereoRectifyUtil.rectify_img(object_mask_rgb, map_x_1, map_y_1)  # rectify
        
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

    else:
        pcd, depth = generate_pcd(rect_left_img, disp_npy, 
            fx, fy, cx, cy, baseline, object_mask_binary=None, roi_mask_binary=None)
        # pcd.rotate(R1)
        if debug_flag_level>=1 and save_res_dir is not None:
            o3d.io.write_point_cloud(os.path.join(save_res_dir, f"{fname}_scene.ply"), pcd)  # for vis debug
            print("saving the pcd file ...", f"{fname}_scene.ply")
    
    return pcd
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
       
    return act_list
##################################################################
def robot2camera_6dof_pose(pose_4x4, arm_side):
    if arm_side == "L":
        handeye_param = cfg_dict_init["arm1"]["handeye_para"]
    if arm_side == "R":
        handeye_param = cfg_dict_init["arm2"]["handeye_para"]
    
    camara_pose_4x4 = np.eye(4)
    camara_pose_4x4[:3, :3] = handeye_param[:3, :3].T @ pose_4x4[:3, :3]
    camara_pose_4x4[:3, -1] = handeye_param[:3, :3].T @ (pose_4x4[:3, -1] - handeye_param[:3, -1])
    return camara_pose_4x4
##################################################################
def covert_act_list_robot2camera(act_list, task_name):
    real_horizon = task_cfg_info[task_name]["steps_length"]
    arm_order_list = task_cfg_info[task_name]["arm_order"]
    assert real_horizon == len(act_list), "the length of act_list is wrong!!!"
    
    new_act_list = []
    for i, act_vec in enumerate(act_list):
        gt_action_4x4 = np.eye(4)
        gt_action_4x4[:3, :3] = R.from_euler("xyz", act_vec[3:6], degrees=True).as_matrix()
        gt_action_4x4[:3, -1] = act_vec[0:3]
        arm_side = arm_order_list[i]
        camara_pose_4x4 = robot2camera_6dof_pose(gt_action_4x4, arm_side)
        new_act_vec = [0,0,0, 0,0,0, act_vec[-1]]
        new_act_vec[0:3] = camara_pose_4x4[:3, -1]
        new_act_vec[3:6] = R.from_matrix(camara_pose_4x4[:3, :3]).as_euler("xyz", degrees=True)
        new_act_list.append(new_act_vec)
    return new_act_list
##################################################################


if __name__ == '__main__':
    
    debug_flag = False  # True or False
    robot2camera_trans = True  # True or False (we have valided that set it as True will not be better)
    task_act_len = {"drawer": 10, "pouring": 11, "unscrew": 12, "uncover": 12*2, "openbox": 16*2}
    task_names = ["drawer", "pouring", "unscrew", "uncover", "openbox"]
    # task_names = ["drawer", "pouring", "unscrew"]
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
        if robot2camera_trans:
            save_dataset_json = f"./datasets/data_jsons/{task_name}_preprocessed_trans.json"
        else:
            save_dataset_json = f"./datasets/data_jsons/{task_name}_preprocessed.json"
        
        masks_list = os.listdir(mask_dir)
        masks_list.sort()
        debug_ids = np.random.choice(len(masks_list), 10) if debug_flag else []
        for smaple_id, mask_name in enumerate(masks_list):
            if debug_flag and smaple_id not in debug_ids: continue
            
            json_name = mask_name.replace("_imgL-removebg-preview.png", ".json")
            json_path = os.path.join(json_dir, json_name)    
            act_list = preprocess_eef_to_act(json_path)  # actions: in robot view
            assert len(act_list) == task_act_len[task_name], "You have obtained a wrong demonstration!!!"

            print("Preprocessing One Demonstration/Sample:", smaple_id, json_name)
            
            mask_path = os.path.join(mask_dir, mask_name)
            left_img_name = mask_name.replace("imgL-removebg-preview.png", "imgL.jpg")
            left_img_path = os.path.join(json_dir, left_img_name)
            right_img_name = mask_name.replace("imgL-removebg-preview.png", "imgR.jpg")
            right_img_path = os.path.join(json_dir, right_img_name)
            pcd_o3d = preprocess_obs_to_pcd(  # observation: in camera view
                param_list, left_img_path, right_img_path, mask_path, save_res_dir=save_res_dir) 
            pts_arr_cam = np.array(pcd_o3d.points)  # (N, 3), o3d pcd --> npy arr points, without color
            if pts_arr_cam.shape[0] > 32768:  # we do not need too many 3d points
                pts_arr_cam = keep_n_points_wo_color(pts_arr_cam, 32768)
            pts_arr_cam_list = np.round(pts_arr_cam, 6).tolist()  # we do not need too accurate 3d points
            
            if robot2camera_trans:
                act_list = covert_act_list_robot2camera(act_list, task_name)
                
            preprocessed_dataset_json.append({"obs": pts_arr_cam_list, "actions": act_list})
            
        with open(save_dataset_json, "w") as json_file:
            json.dump(preprocessed_dataset_json, json_file)
        
        ##################################################################    
    #===============================================================
    # this version, we do not apply any demonstration augmentation 
    #===============================================================
