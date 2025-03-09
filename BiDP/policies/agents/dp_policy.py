import copy
import hydra
import torch
from torch import nn

from policies.vision.pointnet_encoder import PointNetEncoder
from policies.utils.diffusion.ema_model import EMAModel
from policies.utils.diffusion.conditional_unet1d import ConditionalUnet1D
from policies.utils.diffusion.resnet_with_gn import get_resnet, replace_bn_with_gn


class DPPolicy(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim = cfg.model.hidden_dim
        self.obs_mode = cfg.model.obs_mode
        self.device = device

        # |o|o|                             observations: 2
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        self.pred_horizon = cfg.model.pred_horizon
        self.obs_horizon = cfg.model.obs_horizon
        self.action_horizon = cfg.model.ac_horizon

        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps

        self.num_eef = cfg.env.num_eef
        self.eef_dim = cfg.env.eef_dim
        self.dof = cfg.env.dof
        if cfg.model.obs_mode == "state":
            self.obs_dim = self.num_eef * self.eef_dim
        elif cfg.model.obs_mode == "rgb":
            self.obs_dim = 512 + self.num_eef * self.eef_dim
        else:
            self.obs_dim = hidden_dim + self.num_eef * self.eef_dim
        self.action_dim = self.dof * cfg.env.num_eef

        if self.obs_mode.startswith("pc"):
            self.encoder = PointNetEncoder(
                h_dim=hidden_dim,
                c_dim=hidden_dim,
                num_layers=cfg.model.encoder.backbone_args.num_layers,
            )
        elif self.obs_mode == "rgb":
            self.encoder = replace_bn_with_gn(get_resnet("resnet18"))
        else:
            self.encoder = nn.Identity()
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            diffusion_step_embed_dim=self.obs_dim * self.obs_horizon,
            global_cond_dim=self.obs_dim * self.obs_horizon,
        )

        self.nets = nn.ModuleDict(
            {"encoder": self.encoder, "noise_pred_net": self.noise_pred_net}
        )
        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized DP Policy with {num_parameters} parameters")

    def _init_torch_compile(self):
        if self.cfg.model.use_torch_compile:
            self.encoder_handle = torch.compile(self.encoder)
            self.noise_pred_net_handle = torch.compile(self.noise_pred_net)

    def forward(self, obs, predict_action=True, debug=False):
        # assumes that observation has format:
        # - pc: [BS, obs_horizon, num_pts, 3]
        # - state: [BS, obs_horizon, obs_dim]
        # returns:
        # - action: [BS, pred_horizon, ac_dim]
        pc = obs["pc"]
        state = obs["state"]

        if self.obs_mode.startswith("pc"):
            pc = self.pc_normalizer.normalize(pc)
        state = self.state_normalizer.normalize(state)

        pc_shape = pc.shape
        batch_size = pc.shape[0]

        ema_nets = self.ema.averaged_model

        if self.obs_mode == "state":
            z = state
        else:
            if self.obs_mode == "rgb":
                rgb = obs["rgb"]
                rgb_shape = rgb.shape
                flattened_rgb = rgb.reshape(
                    batch_size * self.obs_horizon, *rgb_shape[-3:]
                )
                z = ema_nets["encoder"](flattened_rgb.permute(0, 3, 1, 2))
            else:
                flattened_pc = pc.reshape(batch_size * self.obs_horizon, *pc_shape[-2:])
                z = ema_nets["encoder"](flattened_pc.permute(0, 2, 1))["global"]
            z = z.reshape(batch_size, self.obs_horizon, -1)
            z = torch.cat([z, state], dim=-1)
        obs_cond = z.reshape(batch_size, -1)  # (BS, obs_horizon * obs_dim)

        initial_noise_scale = 0.0 if debug else 1.0
        noisy_action = (
            torch.randn((batch_size, self.pred_horizon, self.action_dim)).to(
                self.device
            )
            * initial_noise_scale
        )
        curr_action = noisy_action
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = ema_nets["noise_pred_net"](
                sample=curr_action, timestep=k, global_cond=obs_cond
            )

            # inverse diffusion step
            curr_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=curr_action
            ).prev_sample

        ret = dict(ac=curr_action)
        return ret

    def step_ema(self):
        self.ema.step(self.nets)
