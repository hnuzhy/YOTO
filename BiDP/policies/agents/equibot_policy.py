import copy
import hydra
import torch
from torch import nn
import torch.nn.functional as F

from policies.vision.sim3_encoder import SIM3Vec4Latent
from policies.utils.diffusion.ema_model import EMAModel
from policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D


def vec2mat(vec):
    # converts non-orthogonal x, y vectors to a rotation matrix
    # input: (..., 2, 3)
    # output: (..., 3, 3)
    x, y = vec[..., [0], :], vec[..., [1], :]
    x_norm = F.normalize(x, dim=-1)
    y = y - x_norm * torch.sum(x_norm * y, dim=-1, keepdim=True)
    y_norm = F.normalize(y, dim=-1)
    z = torch.cross(x_norm, y_norm, dim=-1)
    mat = torch.cat([x_norm, y_norm, z], dim=-2).transpose(-2, -1)
    return mat


class EquiBotPolicy(nn.Module):
    def __init__(self, cfg, device="cpu"):
        nn.Module.__init__(self)
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.use_torch_compile = cfg.model.use_torch_compile
        self.device = device

        # |o|o|                             observations: 2
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        # | |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        self.pred_horizon = cfg.model.pred_horizon
        self.obs_horizon = cfg.model.obs_horizon
        self.action_horizon = cfg.model.ac_horizon

        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps

        if self.obs_mode.startswith("pc"):
            self.encoder = SIM3Vec4Latent(**cfg.model.encoder)
        else:
            self.encoder = None
        self.encoder_out_dim = cfg.model.encoder.c_dim

        self.num_eef = cfg.env.num_eef
        self.eef_dim = cfg.env.eef_dim
        self.dof = cfg.env.dof
        if cfg.model.obs_mode == "state":
            self.obs_dim = self.num_eef * (self.eef_dim // 3)
        elif cfg.model.obs_mode == "rgb":
            raise NotImplementedError()
        else:
            self.obs_dim = self.encoder_out_dim + self.num_eef * (self.eef_dim // 3)
        self.action_dim = (2 if self.dof > 4 else 1) * self.num_eef

        num_scalar_dims = (0 if self.dof == 3 else 1) * self.num_eef
        self.noise_pred_net = VecConditionalUnet1D(
            input_dim=self.action_dim,
            cond_dim=self.obs_dim * self.obs_horizon,
            scalar_cond_dim=(0 if self.dof == 3 else self.num_eef * self.obs_horizon),
            scalar_input_dim=num_scalar_dims,
            diffusion_step_embed_dim=self.obs_dim * self.obs_horizon,
            cond_predict_scale=True,
        )

        self.nets = nn.ModuleDict(
            {"encoder": self.encoder, "noise_pred_net": self.noise_pred_net}
        )
        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized SIM(3) DP Policy with {num_parameters} parameters")

    def _init_torch_compile(self):
        if self.use_torch_compile:
            self.encoder_handle = torch.compile(self.encoder)
            self.noise_pred_net_handle = torch.compile(self.noise_pred_net)
        else:
            self.encoder_handle = self.encoder
            self.noise_pred_net_handle = self.noise_pred_net

    def _convert_state_to_vec(self, state):
        # state format for 3d and 4d actions: eef_pos
        # state format for 7d actions: eef_pos, eef_rot_x, eef_rot_z, gravity_dir, gripper_pose, [optional] goal_pos
        # input: (B, H, E * eef_dim)
        # output: (B, H, ?, 3) [need norm] + (B, H, ?, 3) [does not need norm] + maybe (B, H, E)
        if self.dof == 3:
            return state.view(state.shape[0], state.shape[1], -1, 3), None, None
        elif self.dof == 4:
            state = state.view(state.shape[0], state.shape[1], self.num_eef, -1)
            assert state.shape[-1] in [4, 7]
            eef_pos = state[:, :, :, :3]
            scalar_state = state[:, :, :, 3]
            if state.shape[-1] == 7:
                goal_pos = state[:, :, :, -3:]
                vec_state_pos = torch.cat([eef_pos, goal_pos], dim=2)
            else:
                vec_state_pos = eef_pos
            return vec_state_pos, None, scalar_state
        else:
            state = state.view(state.shape[0], state.shape[1], self.num_eef, -1)
            assert state.shape[-1] in [13, 16]
            eef_pos = state[:, :, :, :3]
            dir1 = state[:, :, :, 3:6]
            dir2 = state[:, :, :, 6:9]
            gravity_dir = state[:, :, :, 9:12]
            gripper_pose = state[:, :, :, 12]

            if state.shape[-1] > 13:
                goal_pos = state[:, :, :, 13:16]
                vec_state_pos = torch.cat([eef_pos, goal_pos], dim=2)
            else:
                vec_state_pos = eef_pos
            vec_state_dir = torch.cat([dir1, dir2, gravity_dir], dim=2)
            scalar_state = gripper_pose
            return vec_state_pos, vec_state_dir, scalar_state

    def _convert_action_to_vec(self, ac, batch=None):
        # input: (B, H, E * dof); output: (B, ac_dim, 3, H) + maybe (B, E, H)
        # rotation actions are always treated as relative axis-angle rotation
        ac = ac.view(ac.shape[0], ac.shape[1], -1, self.dof)
        if self.dof in [4, 7]:
            gripper_ac = ac[:, :, :, 0]  # (B, H, E)
            eef_ac = ac[:, :, :, 1:]  # (B, H, E, 3)
            if self.dof == 7:
                eef_ac = eef_ac.reshape(
                    ac.shape[0], ac.shape[1], -1, 3
                )  # (B, H, E * 2, 3)
            return eef_ac.permute(0, 2, 3, 1), gripper_ac.permute(0, 2, 1)
        elif self.dof == 3:
            return ac.permute(0, 2, 3, 1), None
        else:
            raise ValueError(f"Cannot handle dof = {self.dof}")

    def _convert_action_to_scalar(self, eef_ac, gripper_ac=None, batch=None):
        # input: (B, ac_dim, 3, H) + maybe (B, E, H); output: (B, H, E * dof)
        if self.dof in [4, 7]:
            assert len(eef_ac.shape) == 4
            assert len(gripper_ac.shape) == 3
            assert eef_ac.shape[0] == gripper_ac.shape[0]
            assert eef_ac.shape[-1] == gripper_ac.shape[-1]
            assert eef_ac.shape[2] == 3
            eef_ac = eef_ac.reshape(
                eef_ac.shape[0], self.num_eef, 3 if self.dof == 4 else 6, -1
            )
            grip_ac = gripper_ac[:, :, None]
            scalar_ac = torch.cat([grip_ac, eef_ac], dim=2).permute(0, 3, 1, 2)
            scalar_ac = scalar_ac.reshape(scalar_ac.shape[0], scalar_ac.shape[1], -1)
        elif self.dof == 3:
            scalar_ac = eef_ac.reshape(eef_ac.shape[0], -1, eef_ac.shape[-1]).permute(
                0, 2, 1
            )
        else:
            raise ValueError(f"Cannot handle dof = {self.dof}")
        return scalar_ac

    def forward(self, obs, predict_action=True, debug=False):
        # assumes that observation has format:
        # - pc: [BS, obs_horizon, num_pts, 3]
        # - state: [BS, obs_horizon, obs_dim]
        # returns:
        # - action: [BS, pred_horizon, ac_dim]
        pc = obs["pc"]
        state = obs["state"]

        pc = self.pc_normalizer.normalize(pc)

        pc_shape = pc.shape
        batch_size = B = pc.shape[0]
        Ho = self.obs_horizon
        Hp = self.pred_horizon

        ema_nets = self.ema.averaged_model

        if self.obs_mode == "state":
            z_pos, z_dir, z_scalar = self._convert_state_to_vec(state)
            z_pos = self.state_normalizer.normalize(z_pos)
            if self.dof > 4:
                z = torch.cat([z_pos, z_dir], dim=-2)
            else:
                z = z_pos
        else:
            feat_dict = ema_nets["encoder"](pc, target_norm=self.pc_scale)
            center = (
                feat_dict["center"].reshape(B, Ho, 1, 3)[:, [-1]].repeat(1, Ho, 1, 1)
            )
            scale = feat_dict["scale"].reshape(B, Ho, 1, 1)[:, [-1]].repeat(1, Ho, 1, 1)
            z_pos, z_dir, z_scalar = self._convert_state_to_vec(state)
            z_pos = self.state_normalizer.normalize(z_pos)
            z_pos = (z_pos - center) / scale
            z = feat_dict["so3"]
            z = z.reshape(B, Ho, -1, 3)
            if self.dof == 7:
                z = torch.cat([z, z_pos, z_dir], dim=-2)
            else:
                z = torch.cat([z, z_pos], dim=-2)
        obs_cond_vec, obs_cond_scalar = z.reshape(B, -1, 3), (
            z_scalar.reshape(B, -1) if z_scalar is not None else None
        )

        initial_noise_scale = 0.0 if debug else 1.0
        if self.dof > 3:
            noisy_action = (
                torch.randn((B, Hp, self.action_dim, 3)).to(self.device)
                * initial_noise_scale,
                torch.randn((B, Hp, self.num_eef)).to(self.device)
                * initial_noise_scale,
            )
        else:
            noisy_action = (
                torch.randn((B, Hp, self.action_dim, 3)).to(self.device)
                * initial_noise_scale,
                None,
            )
        curr_action = noisy_action
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            # load from existing data statistics
            # predict noise
            noise_pred = ema_nets["noise_pred_net"](
                sample=curr_action[0],
                timestep=k,
                scalar_sample=curr_action[1],
                cond=obs_cond_vec,
                scalar_cond=obs_cond_scalar,
            )

            # inverse diffusion step
            new_action = [None, None]
            new_action[0] = self.noise_scheduler.step(
                model_output=noise_pred[0], timestep=k, sample=curr_action[0]
            ).prev_sample
            if noise_pred[1] is not None:
                new_action[1] = self.noise_scheduler.step(
                    model_output=noise_pred[1], timestep=k, sample=curr_action[1]
                ).prev_sample
            curr_action = tuple(new_action)

        if curr_action[1] is not None:
            action = self._convert_action_to_scalar(
                curr_action[0].permute(0, 2, 3, 1),
                curr_action[1].permute(0, 2, 1),
                batch=obs,
            )
        else:
            action = self._convert_action_to_scalar(
                curr_action[0].permute(0, 2, 3, 1), batch=obs
            )
        if self.obs_mode.startswith("pc"):
            E = self.num_eef
            if self.ac_mode == "abs":
                center = (
                    feat_dict["center"]
                    .reshape(B, Ho, 3)[:, [-1], None]
                    .repeat(1, Hp, 1, 1)
                )
            else:
                center = 0
            scale = (
                feat_dict["scale"].reshape(B, Ho, 1)[:, [-1], None].repeat(1, Hp, 1, 1)
            )
            action = action.reshape(B, Hp, E, self.dof)
            if self.dof == 4:
                action[..., 1:] = action[..., 1:] * scale + center
            elif self.dof == 3:
                action = action * scale + center
            elif self.dof == 7:
                action[..., 1:4] = action[..., 1:4] * scale + center
            else:
                raise ValueError(f"Dof {self.dof} not supported.")
            action = action.reshape(B, Hp, E * self.dof)

        ret = dict(ac=action)
        if debug:
            ret.update(
                dict(
                    obs_cond_vec=obs_cond_vec.detach().cpu().numpy(),
                )
            )
            if obs_cond_scalar is not None:
                ret.update(
                    dict(
                        obs_cond_scalar=(
                            obs_cond_scalar.detach().cpu().numpy()
                            if obs_cond_scalar is not None
                            else None
                        )
                    )
                )
            if self.obs_mode != "state":
                ret.update(
                    dict(
                        center=feat_dict["center"]
                        .detach()
                        .reshape(B, Ho, 3)
                        .cpu()
                        .numpy(),
                        scale=feat_dict["scale"].detach().reshape(B, Ho).cpu().numpy(),
                        feat_so3=feat_dict["so3"]
                        .detach()
                        .reshape(B, Ho, -1, 3)
                        .cpu()
                        .numpy(),
                    )
                )
        return ret

    def step_ema(self):
        self.ema.step(self.nets)
