import os
import glob
import time
import json
import ffmpeg
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

# from policies.utils.misc import rotate_around_z
from scipy.spatial.transform import Rotation as R


class BaseDataset(Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()
        
        self.mode = mode
        self.dof = cfg["dof"]
        self.num_eef = cfg["num_eef"]
        self.eef_dim = cfg["eef_dim"]
        self.num_points = cfg["num_points"]
        
        self.shuffle_pc = cfg["shuffle_pc"]
        self.obs_horizon = cfg["obs_horizon"]  # should be 2 of dual arms
        self.pred_horizon = cfg["pred_horizon"]  # should be a number in [10+2, 11+1, 12, 12*2, 16*2]
        
        self.task_name = cfg["task_name"]
        self.is_augmented = cfg["is_augmented"]
        self.is_transformed = cfg["is_transformed"]
        self.data_dir = os.path.join(cfg["path"])
        
        self.init_eep_pos = None  # update by load_and_process_json_files()
        self.demos_list = self.load_and_process_json_files()
            

    def load_and_process_json_files(self):

        if (not self.is_augmented) and (not self.is_transformed):  # drawer / pouring / unscrew / uncover / openbox
            json_path = os.path.join(self.data_dir, f"{self.task_name}_preprocessed.json")
            print("[dataset loading...]", json_path)
            dataset_dict_list = json.load(open(json_path, "r"))
        elif (not self.is_augmented) and self.is_transformed:
            json_path = os.path.join(self.data_dir, f"{self.task_name}_preprocessed_trans.json")
            print("[dataset loading...]", json_path)
            dataset_dict_list = json.load(open(json_path, "r"))
        elif self.is_augmented:
            dataset_dict_list = []
            if self.task_name == "drawer":
                for sub_id in range(4):
                    sub_id_str = str(sub_id + 1) + "_trans" if self.is_transformed else str(sub_id + 1)
                    json_file_name = f"drawer_preprocessed_aug100x_{sub_id_str}.json"
                    json_path_sub = os.path.join(self.data_dir, json_file_name)
                    print("[dataset loading...]", json_path_sub)
                    dataset_dict_list_sub = json.load(open(json_path_sub, "r"))
                    dataset_dict_list += dataset_dict_list_sub
            elif self.task_name == "pouring":
                for sub_id in range(3):
                    sub_id_str = str(sub_id + 1) + "_trans" if self.is_transformed else str(sub_id + 1)
                    json_file_name = f"pouring_preprocessed_aug100x_{sub_id_str}.json"
                    json_path_sub = os.path.join(self.data_dir, json_file_name)
                    print("[dataset loading...]", json_path_sub)
                    dataset_dict_list_sub = json.load(open(json_path_sub, "r"))
                    dataset_dict_list += dataset_dict_list_sub
            else:  # unscrew / uncover / openbox
                if self.is_transformed:
                    json_file_name = f"{self.task_name}_preprocessed_aug100x_trans.json"
                else:
                    json_file_name = f"{self.task_name}_preprocessed_aug100x.json"
                json_path = os.path.join(self.data_dir, json_file_name)
                print("[dataset loading...]", json_path)
                dataset_dict_list = json.load(open(json_path, "r"))
        print("[dataset loaded!]", self.task_name, len(dataset_dict_list))
        
        # dataset_dict_list --> {"obs": pts_arr_cam_list, "actions": act_list})
        
        if self.task_name == "drawer":
            robot_init_pose_L = [-0.0424427, -0.230288137, 0.494382357, 162.717697144, -0.241176724, 0.80037123]
            robot_init_pose_R = [0.30657307, -0.13161421, 0.220758319, -89.734809875, -1.474895716, 157.852661133]
        if self.task_name == "pouring":
            robot_init_pose_L = [-0.108424807, -0.166306722, 0.347436588, -84.848640442, -0.673811913, 179.939193726]
            robot_init_pose_R = [0.147416735, -0.128141205, 0.305548777, -60.362663269, -89.610832214, 148.666717529]
        if self.task_name == "unscrew":
            robot_init_pose_L = [-0.128552342, -0.180086863, 0.364960731, -89.310050964, -0.575453699, 179.958602905]
            robot_init_pose_R = [0.093864335, -0.171052892, 0.421423988, 179.400680542, 0.340320289, 89.97151947]
        if self.task_name == "uncover":
            robot_init_pose_L = [-0.133840928, -0.1393452, 0.35099278, 134.39755249, -0.280338049, 0.282890916]
            robot_init_pose_R = [0.133840928, -0.1393452, 0.35099278, 179.605224609, 45.672283173, 90.392219543]
        if self.task_name == "openbox":
            robot_init_pose_L = [-0.051750433, -0.186152327, 0.563311709, -179.186920166, 0.269218802, -89.381820679]
            robot_init_pose_R = [0.051750433, -0.186152327, 0.563311709, 179.11781311, -0.614165246, -178.604156494]
        
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

        for arm_id, init_pose in enumerate([robot_init_pose_L, robot_init_pose_R]):
            init_pose = np.array(init_pose)
            position_robot = init_pose[0:3]
            rot_mat_robot = R.from_euler("xyz", init_pose[3:], degrees=True).as_matrix()
            init_6dof_pose = np.eye(4)
            init_6dof_pose[:3, :3] = rot_mat_robot
            init_6dof_pose[:3, -1] = position_robot
            
            if self.is_transformed:  # robot pose --> camera pose
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
            
            if self.init_eep_pos is None:
                self.init_eep_pos = init_state  # shape (13,)
            else:
                self.init_eep_pos = np.stack([self.init_eep_pos, init_state], 0)  # shape (2, 13)

        return dataset_dict_list
        
        
    def __len__(self):
        return len(self.demos_list)

    def __getitem__(self, idx):
        single_demo_dict = self.demos_list[idx]
        pts_arr_cam_list = single_demo_dict["obs"]  # (N, 3)
        act_list = single_demo_dict["actions"]  # (3 location + 3 orientation + 1 gripper)
        
        ret = { "pc": [], "eef_pos": [], "action": [] }

        xyz = np.array(pts_arr_cam_list).astype(np.float32)
        if self.mode == "train" and self.shuffle_pc:  # (N, 3) --> (1024, 3)
            choice = np.random.choice(xyz.shape[0], self.num_points,
                replace=False if xyz.shape[0] >= self.num_points else True)
            xyz = xyz[choice, :]
        else:
            step = xyz.shape[0] // self.num_points
            xyz = xyz[::step, :][: self.num_points, :]
        
        # ret["pc"] = np.stack([xyz, xyz], 0)  # shape (2, 1024, 3)
        ret["pc"] = np.expand_dims(xyz, 0)  # shape (1, 1024, 3)
        
        # ret["eef_pos"] = self.init_eep_pos  # shape (2, 13)
        ret["eef_pos"] = np.expand_dims(self.init_eep_pos[0], 0)  # shape (1, 13)
        
        # assert self.pred_horizon == len(act_list), "you must make sure the self.pred_horizon = act_len !!!"
        
        gripper_vec, position_vec, rotation_vec = [], [], []
        for act_step in act_list:
            act_step = np.array(act_step)
            gripper_vec.append(act_step[-1])  # gripper
            position_robot = act_step[0:3]  # location
            position_vec.append(position_robot)
            rot_mat_robot = R.from_euler("xyz", act_step[3:6], degrees=True).as_matrix()  # orientation
            rotation_vec.append(rot_mat_robot)
        gripper_vec = np.array(gripper_vec).astype(np.float32)
        position_vec = np.array(position_vec).astype(np.float32)
        rotation_vec = np.array(rotation_vec).astype(np.float32)
        
        position_init = np.expand_dims(self.init_eep_pos[0][:3], 0)  # shape (1, 3)
        full_position_vec = np.concatenate([position_init, position_vec], 0)  # shape (1+pred_horizon, 3)
        
        if self.task_name == "drawer":  # extend pred_horizon 10 --> 12 with dummy actions
            gripper_vec = np.concatenate([gripper_vec, [gripper_vec[-1]], [gripper_vec[-1]]])
            full_position_vec = np.concatenate([full_position_vec, position_vec[-2:-1,:], position_vec[-2:-1,:]], 0)
            rotation_vec = np.concatenate([rotation_vec, rotation_vec[-2:-1,:], rotation_vec[-2:-1,:]], 0)
        if self.task_name == "pouring":  # extend pred_horizon 11 --> 12 with dummy actions
            gripper_vec = np.concatenate([gripper_vec, [gripper_vec[-1]]])
            full_position_vec = np.concatenate([full_position_vec, position_vec[-2:-1,:]], 0)
            rotation_vec = np.concatenate([rotation_vec, rotation_vec[-2:-1,:]], 0)
            
        new_actions = np.concatenate([
            np.expand_dims(gripper_vec, -1), 
            np.diff(full_position_vec, axis=0),
            R.from_matrix(rotation_vec).as_rotvec()], -1).astype(np.float32)
        ret["action"] = new_actions  # shape (pred_horizon, 7)
        
        assert len(ret["pc"]) == self.obs_horizon
        assert len(ret["eef_pos"]) == self.obs_horizon
        assert len(ret["action"]) == self.pred_horizon
        
        return ret

