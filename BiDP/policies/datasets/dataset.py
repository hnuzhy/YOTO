import os
import glob
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from policies.utils.misc import rotate_around_z


class BaseDataset(Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()

        self.mode = mode
        self.dof = cfg["dof"]
        self.num_eef = cfg["num_eef"]
        self.eef_dim = cfg["eef_dim"]
        self.num_points = cfg["num_points"]
        self.num_augment = cfg["num_augment"]
        self.aug_keep_original = cfg["aug_keep_original"]
        self.aug_scale_low = cfg["aug_scale_low"]
        self.aug_scale_high = cfg["aug_scale_high"]
        self.aug_scale_aspect_limit = cfg["aug_scale_aspect_limit"]
        self.aug_scale_pos = cfg["aug_scale_pos"]
        self.aug_scale_rot = cfg["aug_scale_rot"]
        self.aug_center = np.array(cfg["aug_center"])
        self.same_aug_per_sample = cfg["same_aug_per_sample"]
        self.aug_zero_z_offset = cfg["aug_zero_z_offset"]
        self.reduce_horizon_dim = cfg["reduce_horizon_dim"]
        self.shuffle_pc = cfg["shuffle_pc"]
        self.min_demo_length = cfg["min_demo_length"]
        if "latency" in cfg:
            self.state_latency = cfg["latency"]["state"]
            self.state_latency_random = cfg["latency"]["random"]
        else:
            self.state_latency = 0
        if "obs_horizon" in cfg:
            self.obs_horizon = cfg["obs_horizon"]
            self.pred_horizon = cfg["pred_horizon"]
        else:
            self.obs_horizon = 1
            self.pred_horizon = 1

        self.data_dir = os.path.join(cfg["path"], "*")

        self.file_names = sorted(glob.glob(self.data_dir))
        if "num_demos" in cfg:
            key_fn = lambda x: "_".join(x.split("/")[-1].split("_")[:-1])
            ep_list = list(sorted(set([key_fn(fn) for fn in self.file_names])))
            if cfg["num_demos"] < len(ep_list):
                print("[dataset.py] Filtering demos to {cfg['num_demos']} demos")
                filtered_ep_list = ep_list[: cfg["num_demos"]]
                self.file_names = [
                    f for f in self.file_names if key_fn(f) in filtered_ep_list
                ]
        else:
            print("[dataset.py] Using all demos")

        ep_length_dict = defaultdict(lambda: 0)
        ep_t_offset_dict = defaultdict(lambda: 1000)
        key_fn = lambda x: "_".join(x.split("/")[-1].split("_")[:-1])
        for fn in self.file_names:
            ep_t = int(fn.split("/")[-1].split(".")[0].split("_")[-1][1:])

            ep_length_dict[key_fn(fn)] += 1
            ep_t_offset_dict[key_fn(fn)] = min(ep_t_offset_dict[key_fn(fn)], ep_t)
        self.ep_length_dict = ep_length_dict
        self.ep_t_offset_dict = ep_t_offset_dict

        # delete episodes that are too short
        self.file_names = [
            fn
            for fn in self.file_names
            if ep_length_dict[key_fn(fn)] >= self.min_demo_length
        ]
        ep_t_offset_dict = {
            k: v
            for k, v in ep_t_offset_dict.items()
            if ep_length_dict[v] >= self.min_demo_length
        }
        ep_length_dict = {
            k: v for k, v in ep_length_dict.items() if v >= self.min_demo_length
        }

        # check if using two digit or four digit filenames
        if len(self.file_names[0].split("_")[-1].split(".")[0][1:]) == 4:
            self.use_four_digit_time = True
        else:
            self.use_four_digit_time = False

        # NOTE: pre-allocate memory for a single step data
        self._init_cache()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        start_t = time.time()
        fn = self.file_names[idx]
        key_fn = lambda x: "_".join(x.split("/")[-1].split("_")[:-1])
        offset_t = self.ep_t_offset_dict[key_fn(fn)]
        ep_t = int(fn.split("_")[-1][1:-4]) - offset_t
        start_t = ep_t - (self.obs_horizon - 1)
        end_t = ep_t + self.pred_horizon
        ep_t_list = np.arange(start_t, end_t)
        clipped_ep_t_list = np.clip(ep_t_list, 0, self.ep_length_dict[key_fn(fn)] - 1)
        ret = dict(pc=[], rgb=[], eef_pos=[], eef_rot=[], action=[], offset=[])
        if self.num_augment > 0:
            if self.same_aug_per_sample:
                aug_idx = np.random.randint(self.num_augment)
            else:
                aug_idx = idx * self.num_augment + np.random.randint(self.num_augment)
        else:
            aug_idx = None
        for t, clipped_t in zip(ep_t_list, clipped_ep_t_list):
            tt = time.time()
            if self.use_four_digit_time:
                fn_t = "_".join(
                    fn.split("_")[:-1] + [f"t{clipped_t + offset_t:04d}.npz"]
                )
            else:
                fn_t = "_".join(
                    fn.split("_")[:-1] + [f"t{clipped_t + offset_t:02d}.npz"]
                )
            if t == start_t and self.obs_horizon == 2 and self.state_latency > 0:
                # take into account observation latency for previous obs
                # current timestep is [start_t]
                # sample state from range [t + 1 - state_latency, ep_t]

                if self.state_latency_random:
                    state_t = t - np.random.randint(self.state_latency)
                else:
                    state_t = t - (self.state_latency - 1)

                clipped_state_t = max(0, state_t)

                if self.use_four_digit_time:
                    fn_state_t = "_".join(
                        fn.split("_")[:-1] + [f"t{clipped_state_t + offset_t:04d}.npz"]
                    )
                else:
                    fn_state_t = "_".join(
                        fn.split("_")[:-1] + [f"t{clipped_state_t + offset_t:02d}.npz"]
                    )

                data_t = self._process_data_from_file(
                    fn_state_t, ["pc", "rgb", "eef_pos"], aug_idx=aug_idx
                )
            else:
                keys = ["action", "eef_pos", "pc"]
                if t > ep_t:
                    keys = ["action"]
                data_t = self._process_data_from_file(fn_t, keys, aug_idx=aug_idx)
            if t >= ep_t:
                ret["action"].append(data_t["action"])

            for k in data_t.keys():
                if k == "action":
                    continue
                if k in ret:
                    ret[k].append(data_t[k])
        ret = {k: np.array(v) for k, v in ret.items() if len(v) > 0}

        assert len(ret["pc"]) == self.obs_horizon
        # assert len(ret["rgb"]) == self.obs_horizon
        assert len(ret["eef_pos"]) == self.obs_horizon
        assert len(ret["action"]) == self.pred_horizon

        if self.obs_horizon == 1 and self.pred_horizon == 1 and self.reduce_horizon_dim:
            ret = {k: v[0] for k, v in ret.items()}

        return ret

    def _init_cache(self, keys_to_keep=["pc", "eef_pos", "action"]):
        self.cache = dict()
        for fn in tqdm(self.file_names):
            # load the npz file and use the filename to create a new dictionary
            data = np.load(fn)
            self.cache[fn] = dict()
            for k in keys_to_keep:
                assert k in data.keys(), f"Key {k} not found in {fn}"
                self.cache[fn][k] = data[k].astype(np.float32)
            del data

    def _process_data_from_file(
        self, fn, keys=["pc", "eef_pos", "action"], aug_idx=None
    ):
        data = self.cache[fn]

        if "pc" in keys:
            xyz = data["pc"].astype(np.float32)
        if "eef_pos" in keys:
            eef_pos = data["eef_pos"].astype(np.float32)
            eef_pos = eef_pos.reshape(self.num_eef, -1)
            eef_pos = eef_pos[:, : self.eef_dim]
        if "pc" in keys or "offset" in keys:
            choice = np.random.choice(
                xyz.shape[0],
                self.num_points,
                replace=False if xyz.shape[0] >= self.num_points else True,
            )

        if "pc" in keys:
            if self.mode == "train" and self.shuffle_pc:
                xyz = xyz[choice, :]
            else:
                step = xyz.shape[0] // self.num_points
                xyz = xyz[::step, :][: self.num_points, :]
        if "offset" in keys:
            if "offset" in data:
                offset = data["offset"].astype(np.float32)
            if self.mode == "train" and self.shuffle_pc:
                if "offset" in data:
                    offset = offset[choice, :]
            else:
                if "offset" in data:
                    offset = offset[::step, :][: self.num_points, :]
        if "action" in keys:
            action = data["action"].astype(np.float32)

        if self.num_augment > 0:
            assert aug_idx is not None
            # augment point cloud, eef pos, and action
            # note that this scaling code hardcodes scaling constants
            if aug_idx == 0 and self.aug_keep_original:
                pass
            else:
                rs = np.random.RandomState(aug_idx)
                if self.aug_scale_aspect_limit > 1.0:
                    while True:
                        scale = (
                            rs.rand(3) * (self.aug_scale_high - self.aug_scale_low)
                            + self.aug_scale_low
                        )
                        if scale.max() / scale.min() < 1.33:
                            break
                else:
                    scale = np.full(
                        (3,),
                        rs.rand() * (self.aug_scale_high - self.aug_scale_low)
                        + self.aug_scale_low,
                    )

                if self.aug_scale_rot < 0:
                    rot = rs.rand() * np.pi * 2
                else:
                    rot = (rs.rand() * 2 - 1) * self.aug_scale_rot
                offset = rs.randn(3) * self.aug_scale_pos

                if self.aug_zero_z_offset:
                    # NOTE: disable translation about z axis
                    offset[2] = 0

                center = self.aug_center
                if "pc" in keys:
                    xyz = rotate_around_z(xyz, rot, center, scale).astype(np.float32)
                    xyz += offset[None]
                if "eef_pos" in keys:
                    eef_pos_shape = eef_pos.shape
                    if self.dof < 7:
                        eef_pos = eef_pos.reshape(-1, 3)
                        eef_pos = rotate_around_z(eef_pos, rot, center, scale)
                        eef_pos += offset[None]
                        eef_pos = eef_pos.reshape(eef_pos_shape).astype(np.float32)
                    else:
                        assert self.eef_dim in [13, 16]
                        eef_pos[:, 0:3] = (
                            rotate_around_z(eef_pos[:, 0:3], rot, center, scale)
                            + offset[None]
                        )
                        eef_pos[:, 3:6] = rotate_around_z(eef_pos[:, 3:6], rot)
                        eef_pos[:, 6:9] = rotate_around_z(eef_pos[:, 3:6], rot)
                        if self.eef_dim == 16:
                            eef_pos[:, 13:16] = (
                                rotate_around_z(eef_pos[:, 0:3], rot, center, scale)
                                + offset[None]
                            )
                if "action" in keys:
                    if self.dof == 3:
                        action = action.reshape(-1, 3)
                        action = rotate_around_z(action, rot, center, scale)
                    elif self.dof == 4:
                        action = action.reshape(-1, 4)
                        action[:, 1:] = rotate_around_z(
                            action[:, 1:], rot, center, scale
                        )
                    elif self.dof == 7:
                        action = action.reshape(-1, 7)
                        # scale and rotate position action
                        action[:, 1:4] = rotate_around_z(
                            action[:, 1:4], rot, center, scale
                        )
                        # rotate velocity action
                        action[:, 4:7] = rotate_around_z(action[:, 4:7], rot)
                    else:
                        raise ValueError(
                            f"Unexpected action shape {action.shape} and dof {self.dof}"
                        )

        if "action" in keys:
            action = action.flatten().astype(np.float32)

        ret = dict()
        if "pc" in keys:
            ret["pc"] = xyz
            assert ret["pc"].dtype == np.float32
        if "eef_pos" in keys:
            ret["eef_pos"] = eef_pos
            assert ret["eef_pos"].dtype == np.float32
        if "action" in keys:
            ret["action"] = action
            assert ret["action"].dtype == np.float32
        if "offset" in keys:
            ret["offset"] = offset
        del data

        return ret
