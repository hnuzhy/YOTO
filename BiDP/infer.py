import os
import sys
import cv2
import torch
import hydra
import json
import ffmpeg
from typing import Dict, List
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from policies.utils.misc import get_agent

##################################################################
task_cfg_info = {
    "drawer": {"steps_length": 10, "arm_order": ["L","R",  "R","R","L","R","L","L","L","L","L","R"]},
    "pouring": {"steps_length": 11, "arm_order": ["L","R",  "R","R","L","L","L","R","L","L","L","L","R"]},
    "unscrew": {"steps_length": 12, "arm_order": ["L","R",  "L","L","L","R","R","R","R","R","L","R","L","L"]},
    "uncover": {"steps_length": 24, "arm_order": ["L","R",  "L","R","L","R","L","R","L","R","L","R","L","R",
                                                            "L","R","L","R","L","R","L","R","L","R","L","R"]},
    "openbox": {"steps_length": 32, "arm_order": ["L","R",  "L","R","L","R","L","R","L","R","L","R","L","R",
                                                            "L","R","L","R","L","R","L","R","L","R","L","R",
                                                            "L","R","L","R","L","R","L","R"]},
}
handeye_param_dict = [
    np.array([
        [-0.14495955, -0.82672254,  0.54361435, -0.46128096],
        [-0.98826554,  0.09424301, -0.120206  , -0.55922609],
        [ 0.04814516, -0.55466034, -0.83068282,  0.77215299],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ]),  # left robot arm, camera calib in 2024-11-20
    np.array([
        [ 0.15464175,  0.82413502, -0.54487375,  0.48699159],
        [ 0.98677   , -0.10165977,  0.12629433, -0.77214491],
        [ 0.04869184, -0.55719545, -0.82895256,  0.76490652],
        [-0.        ,  0.        , -0.        ,  1.        ]
    ])  # right robot arm, camera calib in 2024-11-20
]   
##################################################################
def get_init_robot_pose(cfg):
    init_eep_pos, init_eep_pos_4x4 = None, []
    
    if cfg.env.env_class == "drawer":
        robot_init_pose_L = [-0.0424427, -0.230288137, 0.494382357, 162.717697144, -0.241176724, 0.80037123]
        robot_init_pose_R = [0.30657307, -0.13161421, 0.220758319, -89.734809875, -1.474895716, 157.852661133]
    if cfg.env.env_class == "pouring":
        robot_init_pose_L = [-0.108424807, -0.166306722, 0.347436588, -84.848640442, -0.673811913, 179.939193726]
        robot_init_pose_R = [0.147416735, -0.128141205, 0.305548777, -60.362663269, -89.610832214, 148.666717529]
    if cfg.env.env_class == "unscrew":
        robot_init_pose_L = [-0.128552342, -0.180086863, 0.364960731, -89.310050964, -0.575453699, 179.958602905]
        robot_init_pose_R = [0.093864335, -0.171052892, 0.421423988, 179.400680542, 0.340320289, 89.97151947]
    if cfg.env.env_class == "uncover":
        robot_init_pose_L = [-0.133840928, -0.1393452, 0.35099278, 134.39755249, -0.280338049, 0.282890916]
        robot_init_pose_R = [0.133840928, -0.1393452, 0.35099278, 179.605224609, 45.672283173, 90.392219543]
    if cfg.env.env_class == "openbox":
        robot_init_pose_L = [-0.051750433, -0.186152327, 0.563311709, -179.186920166, 0.269218802, -89.381820679]
        robot_init_pose_R = [0.051750433, -0.186152327, 0.563311709, 179.11781311, -0.614165246, -178.604156494]

    for arm_id, init_pose in enumerate([robot_init_pose_L, robot_init_pose_R]):
        init_pose = np.array(init_pose)
        position_robot = init_pose[0:3]
        rot_mat_robot = R.from_euler("xyz", init_pose[3:], degrees=True).as_matrix()
        init_6dof_pose = np.eye(4)
        init_6dof_pose[:3, :3] = rot_mat_robot
        init_6dof_pose[:3, -1] = position_robot

        if cfg.data.dataset.is_transformed:  # robot pose --> camera pose
            handeye_param = handeye_param_dict[arm_id]
            camara_pose_4x4 = np.eye(4)
            camara_pose_4x4[:3, :3] = handeye_param[:3, :3].T @ init_6dof_pose[:3, :3]
            camara_pose_4x4[:3, -1] = handeye_param[:3, :3].T @ (init_6dof_pose[:3, -1] - handeye_param[:3, -1])
            init_6dof_pose = camara_pose_4x4

        # conver init_6dof_pose into equibot format init_13dof_pose
        ef_pose = init_6dof_pose[:3, [3, 0, 1]].transpose(1, 0)  # center, two column vectors.
        ef_pose_vec = ef_pose.reshape(-1)  # 3x3 --> 1x9, 9-dofs
        gravity_vec = np.array([0, 0, -1])  # step id
        close_vec = np.asarray([1,])  # always be one / open
        init_state = np.concatenate([ef_pose_vec, gravity_vec, close_vec], -1).astype(np.float32)
        
        if init_eep_pos is None:
            init_eep_pos = init_state  # shape (13,)
        else:
            init_eep_pos = np.stack([init_eep_pos, init_state], 0)  # shape (2, 13)
        init_eep_pos_4x4.append(init_6dof_pose)
    return init_eep_pos, init_eep_pos_4x4
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
def robot2camera_6dof_pose(pose_4x4, arm_side):
    if arm_side == "L": handeye_param = handeye_param_dict[0]
    if arm_side == "R": handeye_param = handeye_param_dict[1]
    
    camara_pose_4x4 = np.eye(4)
    camara_pose_4x4[:3, :3] = handeye_param[:3, :3].T @ pose_4x4[:3, :3]
    camara_pose_4x4[:3, -1] = handeye_param[:3, :3].T @ (pose_4x4[:3, -1] - handeye_param[:3, -1])
    return camara_pose_4x4
##################################################################
def visualize_results(obs: Dict, actions: List[Dict], 
    gt_actions=None, arm_order=None, is_transformed=False,
    color: List[float] = None, is_visual: bool = True):
    
    pts_objects, pts_platform = obs["pc"][0][0], obs["pc_platform"] 
    pts_arr = np.concatenate([pts_objects, pts_platform], 0)  # (N, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_arr)
    objects_color = np.array([0.8,0,0.8] * len(pts_objects)).reshape(-1, 3)  # objects pcd as magenta color
    platform_color = np.array([0,0.8,0] * len(pts_platform)).reshape(-1, 3)  # platform pcd as green color
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate([objects_color, platform_color], 0))


    frames = [pcd,]
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    # cmap_L, cmap_R = plt.cm.get_cmap("jet"), plt.cm.get_cmap("ocean")  # jet: blue-->red；ocean: green--> white
    color_L, color_R, color_L_gt, color_R_gt = [0,0,1], [1,0,0], [0,0,0.5], [0.5,0,0]  # blue, red
    
    for i, action in enumerate(actions):  # the initial two acts (L&R arm) + N predicted acts (L/R arm)
        if i == 0 or i == 1:  # the initial two acts
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
        else:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        if not is_transformed: vis_act = robot2camera_6dof_pose(action["ef_pose"], arm_order[i])
        else: vis_act = action["ef_pose"]
        frame.transform(vis_act)
        frame_pnts = frame.sample_points_uniformly(1024)
        if color is not None:  color_current = color
        # else: color_current = cmap(i / len(actions))[:3]
        else: color_current = color_L if arm_order[i]=="L" else color_R
        frame_pnts.paint_uniform_color(color_current)
        frames.append(frame_pnts)
    
    if gt_actions is not None:
        for i, gt_action in enumerate(gt_actions):  # N ground-truth acts (L/R arm)
            gt_action_4x4 = np.eye(4)
            gt_action_4x4[:3, :3] = R.from_euler("xyz", gt_action[3:6], degrees=True).as_matrix()
            gt_action_4x4[:3, -1] = gt_action[0:3]
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
            if not is_transformed: vis_act = robot2camera_6dof_pose(gt_action_4x4, arm_order[i+2])
            else: vis_act = gt_action_4x4
            frame.transform(vis_act)
            frame_pnts = frame.sample_points_uniformly(1024)
            color_current = color_L_gt if arm_order[i+2]=="L" else color_R_gt
            frame_pnts.paint_uniform_color(color_current)
            frames.append(frame_pnts)

    if is_visual:
        o3d.visualization.draw_geometries(frames)
    return frames
##################################################################


@hydra.main(config_path="policies/configs", config_name="basic")
def main(cfg):
    assert cfg.mode == "train"
    np.random.seed(cfg.seed)
    
    vis_saved_path = "/home/dexforce/zhouhuayi/projects/BiDP/test/"
    is_trans = "withtrans" if cfg.data.dataset.is_transformed else "notrans"
    is_aug = "noaug" if "noaug" in cfg.training.ckpt else "withaug"
    id_ood = "id" if cfg.data.one_real_data is None else "ood"
    testing_set_str = f"demo_{is_trans}_{is_aug}_{id_ood}"


    ##### step 1: init agent / BiDP
    cfg.data.dataset.num_training_steps = 100  # manually set for avoiding a bug
    agent = get_agent(cfg.agent.agent_name)(cfg)
    assert cfg.training.ckpt is not None, "You must give a ckpt path during the inference stage!!!"
    agent.load_snapshot(cfg.training.ckpt)
    is_transformed = cfg.data.dataset.is_transformed  # robot pose --> camera pose
    
    
    ##### step 2: init observation
    obs_dict = {}

    init_eep_pos, init_eep_pos_4x4 = get_init_robot_pose(cfg)
    obs_dict["state"] = np.expand_dims(init_eep_pos[0:1], 0)  # shape ((1, 13))
    
    if cfg.data.one_real_data is not None:
        pcd = o3d.io.read_point_cloud(cfg.data.one_real_data)
        pc, center = np.asarray(pcd.points), pcd.get_center()
        gt_act_list = None
    else:
        if not is_transformed:
            json_path = os.path.join(cfg.data.dataset.path, f"{cfg.env.env_class}_preprocessed.json")
        else:
            json_path = os.path.join(cfg.data.dataset.path, f"{cfg.env.env_class}_preprocessed_trans.json")
        print("[dataset loading...]", json_path)
        dataset_dict_list = json.load(open(json_path, "r"))
        print("[dataset loaded !!!]", len(dataset_dict_list))
        single_demo_dict = dataset_dict_list[0]
        pts_arr_cam_list = single_demo_dict["obs"]  # (N, 3)
        gt_act_list = single_demo_dict["actions"]  # (3 location + 3 orientation + 1 gripper)
        pc = np.asarray(pts_arr_cam_list)
    # pc = keep_n_points_wo_color(pc, pc_num=cfg.data.dataset.num_points)  # 2048, 3072, 4096
    pc = keep_n_points_wo_color(pc, pc_num=32768)  # 2048, 3072, 4096
    print(pc.shape)
    obs_dict["pc"] = [pc]


    ##### step 3: inference
    with torch.no_grad():
        return_dict, debug = False, False
        pred_actions = agent.act(obs_dict, return_dict, debug)
    
    init_position = obs_dict["state"][0][0, 0, :3]  # state value's shape is changed
    
    horizon = pred_actions.shape[0]
    print("Task name:", cfg.env.env_class, ", the corresponding horizon:", horizon)
    real_horizon = task_cfg_info[cfg.env.env_class]["steps_length"]  # some tasks are padded when training
    arm_order_list = task_cfg_info[cfg.env.env_class]["arm_order"]
    
    ret_actions = []
    for i in range(real_horizon):
        ret_i = {}
        ret_i["close"] = pred_actions[i, 0] > 0.5  # 0 close state / 1 open state
        ret_i["ef_pose"] = np.eye(4)
        ret_i["ef_pose"][:3, :3] = R.from_rotvec(pred_actions[i, 4:]).as_matrix()
        ret_i["ef_pose"][:3, -1] = pred_actions[:i+1, 1:4].sum(axis=0) + init_position
        ret_actions.append(ret_i)
    
    
    ##### step 4: visualization
    new_actions = [{"ef_pose": init_eep_pos_4x4[0], "close": 1}, {"ef_pose": init_eep_pos_4x4[1], "close": 1}]
    new_actions = new_actions + ret_actions
    print(len(ret_actions), ret_actions[0], ret_actions[0].keys(), "\n", [i['close'] for i in ret_actions])
    
    platform_pcd_path = os.path.join(vis_saved_path, "platform_pcd.ply")
    platform_pcd = o3d.io.read_point_cloud(platform_pcd_path)
    pcd_pf = keep_n_points_wo_color(np.array(platform_pcd.points), pc_num=32768)
    obs_dict["pc_platform"] = pcd_pf  # (N, 3)
   
    
    # frames = visualize_results(obs_dict, new_actions, gt_actions=gt_act_list, 
        # arm_order=arm_order_list, is_transformed=is_transformed, color=None, is_visual=True)
    frames = visualize_results(obs_dict, new_actions, gt_actions=gt_act_list, 
        arm_order=arm_order_list, is_transformed=is_transformed, color=None, is_visual=False)

    vis_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = frames[0]
    pcd.transform(vis_pose)
    vis.add_geometry(pcd)

    video_task_path = os.path.join(vis_saved_path, cfg.env.env_class)
    if not os.path.exists(video_task_path):
        os.mkdir(video_task_path)
    video_path = os.path.join(video_task_path, testing_set_str)
    if not os.path.exists(video_path):
        os.mkdir(video_path)
 
    np.save(os.path.join(video_path, "pred_actions.npy"), ret_actions)
    
    gt_act_len = len(gt_act_list) if gt_act_list is not None else 0
    for i in range(len(new_actions) + gt_act_len):
        frames[i+1].transform(vis_pose)
        vis.add_geometry(frames[i+1])
        vis.poll_events()
        vis.update_renderer()
        img_path = os.path.join(video_path, "temp_%04d.jpg" % i)
        vis.capture_screen_image(img_path)
        
        img_cv2 = cv2.imread(img_path)  # (1848, 1016)
        img_crop = img_cv2[200:800, 400:1400]  # (1000, 600) --> 5:3
        if i == 0: text_str = "L: init pose"; color = [255, 0, 0]
        elif i == 1: text_str = "R: init pose"; color = [0, 0, 255]
        elif i > 1 and i < len(arm_order_list):
            if arm_order_list[i] == "L": text_str = f"L: pred pose (keyframe {i-1})"; color = [255, 0, 0]
            if arm_order_list[i] == "R": text_str = f"R: pred pose (keyframe {i-1})"; color = [0, 0, 255]
        else:
            ir = i - real_horizon
            if arm_order_list[ir] == "L": text_str = f"L: gt pose (keyframe {ir-1})"; color = [255, 0, 0]
            if arm_order_list[ir] == "R": text_str = f"R: gt pose (keyframe {ir-1})"; color = [0, 0, 255]
        cv2.putText(img_crop, text_str, (5, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75, color=color, thickness=2, lineType=cv2.LINE_AA)    
        cv2.imwrite(img_path, img_crop)
        
    
    multi_frame_path = os.path.join(video_path, "*.jpg")
    saved_video_path = os.path.join(video_path, "BiDP_infer_demo1.mp4")
    ffmpeg.input(multi_frame_path, pattern_type="glob", framerate=2).output(saved_video_path).run(overwrite_output=True)


if __name__ == "__main__":
    main()
