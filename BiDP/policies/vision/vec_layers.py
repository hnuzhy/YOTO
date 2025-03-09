import torch
from torch import nn
import torch.nn.functional as F
import math
import logging

"""
The shape convention should be
B,Channel,3,...
"""

EPS = 1e-7


def channel_equi_vec_normalize(x):
    # B,C,3,...
    assert x.ndim >= 3, "x shape [B,C,3,...]"
    x_dir = F.normalize(x, dim=2)
    x_norm = x.norm(dim=2, keepdim=True)
    x_normalized_norm = F.normalize(x_norm, dim=1)  # normalize across C
    y = x_dir * x_normalized_norm
    return y


class VecLayerNorm(nn.Module):
    def __init__(self, normalized_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_dim))

    def forward(self, x):
        x = channel_equi_vec_normalize(x)
        x = x * self.gamma.view(tuple([1, -1] + [1] * (len(x.shape) - 2)))
        return x


class VecLinear(nn.Module):
    r"""
    from pytorch Linear
    Can be SO3 or SE3
    Can have hybrid feature
    The input scalar feature must be invariant
    valid mode: V,h->V,h; V,h->V; V->V,h; V->V; V,h->h
    The output has two, Vec output, Scalar output; if scalar output is not set, return None
    """

    v_in: int
    v_out: int
    s_in: int
    s_out: int
    weight: torch.Tensor

    def __init__(
        self,
        v_in: int,
        v_out: int,
        s_in=0,
        s_out=0,
        mode="so3",
        scalar2vec_normalized_scale=True,
        vec2scalar_dir_learnable=True,
        cross=False,
        device=None,
        dtype=None,
    ) -> None:
        mode = mode.lower()
        assert mode in ["so3", "se3"], "mode must be so3 or se3"
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.v_in = v_in
        self.v_out = v_out
        self.s_in = s_in
        self.s_out = s_out

        assert self.s_out + self.v_out > 0, "vec, scalar output both zero"

        self.se3_flag = mode == "se3"
        if self.se3_flag:
            assert v_in > 1, "se3 layers must have at least two input layers"

        if self.v_out > 0:
            self.weight = nn.Parameter(
                torch.empty(
                    (v_out, v_in - 1 if self.se3_flag else v_in), **factory_kwargs
                )  # if use se3 mode, should constrain the weight to have sum 1.0
            )  # This is the main weight of the vector, due to historical reason, for old checkpoint, not rename this
            self.reset_parameters()

        if (
            self.s_in > 0 and self.v_out > 0
        ):  # if has scalar input, must have a path to fuse to vector
            self.sv_linear = nn.Linear(s_in, v_out)
            self.s2v_normalized_scale_flag = scalar2vec_normalized_scale

        if self.s_out > 0:  # if has scalar output, must has vector to scalar path
            self.vec2scalar_dir_learnable = vec2scalar_dir_learnable
            if self.vec2scalar_dir_learnable:
                self.vs_dir_linear = VecLinear(
                    v_in, v_in, mode="so3"
                )  # TODO: can just have 1 dir
            self.vs_linear = nn.Linear(v_in, s_out)
        if self.s_in > 0 and self.s_out > 0:  # when have s in and s out, has ss path
            self.ss_linear = nn.Linear(s_in, s_out)

        self.cross_flag = cross
        if self.v_out > 0 and self.cross_flag:
            self.v_out_cross = VecLinear(v_in, v_out, mode=mode, cross=False)
            self.v_out_cross_fc = VecLinear(v_out * 2, v_out, mode=mode, cross=False)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # ! warning, now the initialization will bias to the last channel with larger weight, need better init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.se3_flag:
            self.weight.data += 1.0 / self.v_in

    def forward(self, v_input: torch.Tensor, s_input=None):
        # B,C,3,...; B,C,...

        # First do Vector path if output vector
        v_shape = v_input.shape
        assert v_shape[2] == 3, "not vector neuron"
        if self.v_out > 0:
            if self.se3_flag:
                W = torch.cat(
                    [self.weight, 1.0 - self.weight.sum(-1, keepdim=True)], -1
                ).contiguous()
            else:
                W = self.weight
            v_output = F.linear(v_input.transpose(1, -1), W).transpose(
                -1, 1
            )  # B,C,3,...
        else:
            v_output = None

        # Optional Scalar path
        if self.s_in > 0:  # Note, if have s_input but not set hybrid processing, still
            assert s_input is not None, "missing scalar input"
            s_shape = s_input.shape
            assert v_shape[3:] == s_shape[2:]
            # must do scalar to vector fusion
            if self.v_out > 0:
                s2v_invariant_scale = self.sv_linear(
                    s_input.transpose(1, -1)
                ).transpose(-1, 1)
                if self.s2v_normalized_scale_flag:
                    s2v_invariant_scale = F.normalize(s2v_invariant_scale, dim=1)
                if self.se3_flag:  # need to scale the rotation part, exclude the center
                    v_new_mean = v_output.mean(dim=1, keepdim=True)
                    v_output = (v_output - v_new_mean) * s2v_invariant_scale.unsqueeze(
                        2
                    ) + v_new_mean
                else:
                    v_output = v_output * s2v_invariant_scale.unsqueeze(2)
                # now v_new done

        if self.v_out > 0 and self.cross_flag:
            # do cross production
            v_out_dual, _ = self.v_out_cross(v_input)
            if self.se3_flag:
                v_out_dual_o = v_out_dual.mean(dim=1, keepdim=True)
                v_output_o = v_output.mean(dim=1, keepdim=True)
                v_cross = torch.cross(
                    channel_equi_vec_normalize(v_out_dual - v_out_dual_o),
                    v_output - v_output_o,
                    dim=2,
                )
            else:
                v_cross = torch.cross(
                    channel_equi_vec_normalize(v_out_dual), v_output, dim=2
                )
            v_cross = v_cross + v_output
            v_output, _ = self.v_out_cross_fc(torch.cat([v_cross, v_output], dim=1))

        if self.s_out > 0:
            # must have the vector to scalar path
            v_sR = (
                v_input - v_input.mean(dim=1, keepdim=True)
                if self.se3_flag
                else v_input
            )
            if self.vec2scalar_dir_learnable:
                v_sR_dual_dir, _ = self.vs_dir_linear(v_sR)
                v_sR_dual_dir = F.normalize(v_sR_dual_dir, dim=2)
                s_from_v = F.normalize(
                    (v_sR * v_sR_dual_dir).sum(dim=2), dim=1
                )  # B,C,...
            else:
                # here can not directly use mean and then do the normalization
                # ! if non-learnable direction, we use the length directly!
                s_from_v = F.normalize(v_sR.norm(dim=2), dim=1)
                # v_sR_dual_dir = F.normalize(v_sR.mean(dim=1, keepdim=True), dim=2)

            s_from_v = self.vs_linear(s_from_v.transpose(-1, 1)).transpose(-1, 1)
            if self.s_in > 0:
                s_from_s = self.ss_linear(s_input.transpose(-1, 1)).transpose(-1, 1)
                s_output = s_from_s + s_from_v
            else:
                s_output = s_from_v
        else:
            s_output = None
        return v_output, s_output


class VecActivation(nn.Module):
    # input B,C,3,...
    # The normalization is in this layer
    # Order: 1.) centered [opt] 2.) normalization in norm [opt] 3.) act 4.) add center [opt]
    def __init__(
        self,
        in_features,
        act_func,
        shared_nonlinearity=False,
        mode="so3",
        cross=False,
        normalization_method=None,
        safe_bound=-1,
        act_mode="vec",
        pre_act_len_bound=-1.0,
    ) -> None:
        super().__init__()
        self.act_mode = act_mode
        assert self.act_mode in ["vec", "len"], "act_mode must be vec or len"
        mode = mode.lower()
        assert mode in ["so3", "se3"], "mode must be so3 or se3"
        self.se3_flag = mode == "se3"
        self.shared_nonlinearity_flag = shared_nonlinearity
        self.act_func = act_func

        if self.act_mode == "vec":
            nonlinear_out = 1 if self.shared_nonlinearity_flag else in_features
            self.lin_dir = VecLinear(in_features, nonlinear_out, mode=mode, cross=cross)
            if self.se3_flag:
                self.lin_ori = VecLinear(
                    in_features, nonlinear_out, mode=mode, cross=cross
                )
            self.safe_bound = safe_bound
            self.use_safe_activation = self.safe_bound > 0
            if self.use_safe_activation:
                self.tanh = torch.nn.Hardtanh(-safe_bound, safe_bound)

        self.use_normalization = normalization_method is not None
        if self.use_normalization:
            logging.debug(
                f"Note, the VecActivation layer use {normalization_method}, not scale equivariant"
            )
            self.nrm = normalization_method(in_features)

        self.pre_act_len_bound = pre_act_len_bound

    def forward(self, x):
        # B,C,3,...
        if self.act_mode == "len":
            v_len = x.norm(dim=2, keepdim=True) + EPS
            if self.pre_act_len_bound > 0:
                bounded_v_len = self.pre_act_len_bound * torch.tanh(
                    v_len / self.pre_act_len_bound
                )
                x = x / v_len * bounded_v_len
                v_len = bounded_v_len
            # logging.debug(f"len act before act, len mean {v_len.mean()} std {v_len.std()}")
            act_len = self.act_func(v_len)
            y = x / v_len * act_len
            return y
        else:
            assert x.shape[2] == 3, "not vector neuron"
            q = x
            k, _ = self.lin_dir(x)
            if self.se3_flag:
                o, _ = self.lin_ori(x)
                q = q - o
                k = k - o

            # normalization if set
            if self.use_normalization:
                q_dir = F.normalize(q, dim=2)
                q_len = q.norm(dim=2)  # note: the shape into BN is [B,C,...]
                q_len_normalized = self.nrm(q_len.reshape(x.shape[0], x.shape[1], -1))
                q_len_normalized = q_len_normalized.reshape(q_len.shape)
                q = q_dir * q_len_normalized.unsqueeze(2)

            if self.use_safe_activation:
                # another way to compute, if the activation function is scale equivariant
                qTk = (q * k).sum(dim=2, keepdim=True)
                kTk = (k * k).sum(dim=2, keepdim=True)
                factor = qTk / (kTk + EPS)
                factor = self.tanh(factor)
                q_acted = q + k * (self.act_func(factor) - factor)
            else:
                # actual non-linearity on the parallel component length
                k_dir = F.normalize(k, dim=2)
                q_para_len = (q * k_dir).sum(dim=2, keepdim=True)
                q_orthogonal = q - q_para_len * k_dir
                acted_len = self.act_func(q_para_len)
                q_acted = q_orthogonal + k_dir * acted_len

            if self.se3_flag:
                q_acted = q_acted + o
            return q_acted


class VecMeanPool(nn.Module):
    def __init__(self, pooling_dim=-1, **kwargs) -> None:  # Have dummy args here
        super().__init__()
        self.pooling_dim = pooling_dim

    def forward(self, x, return_weight=False):
        if return_weight:
            return x.mean(self.pooling_dim), None
        else:
            return x.mean(self.pooling_dim)


class VecMaxPool(nn.Module):
    def __init__(
        self,
        in_features,
        shared_nonlinearity=False,
        mode="so3",
        pooling_dim=-1,
        softmax_factor=-1.0,  # if positive, use softmax, else hard pool not learnable
        cross=False,
    ) -> None:
        super().__init__()

        mode = mode.lower()
        assert mode in ["so3", "se3"], "mode must be so3 or se3"
        self.se3_flag = mode == "se3"
        self.shared_nonlinearity_flag = shared_nonlinearity

        nonlinear_out = 1 if self.shared_nonlinearity_flag else in_features
        if self.se3_flag:
            self.lin_ori = VecLinear(in_features, nonlinear_out, mode=mode, cross=cross)

        assert (
            pooling_dim < 0 or pooling_dim >= 3
        ), "invalid pooling dim, the input should have [B,C,3,...] shape"
        self.pooling_dim = pooling_dim

        self.softmax_factor = softmax_factor
        if self.softmax_factor > 0.0:
            # add one layer here in the max pooling
            self.lin_k = VecLinear(in_features, in_features, mode=mode, cross=cross)

    def forward(self, x, return_weight=False):
        # B,C,3,...
        reshape_flag = False
        if x.ndim == 5:  # B,C,3,N,K
            B, C, _, N, K = x.shape
            x = x.permute(0, 3, 1, 2, 4).reshape(B * N, C, 3, K)
            reshape_flag = True
        else:
            B, C, _, N = x.shape

        assert not return_weight or x.ndim == 4, "now don't support 5dim return weight"
        assert x.shape[2] == 3, "not vector neuron"

        q = x
        # get k
        k = x.mean(dim=self.pooling_dim, keepdim=True)
        k, _ = self.lin_k(k)

        if self.se3_flag:
            o, _ = self.lin_ori(k)
            q = q - o
            k = k - o
        k_scale_inv = channel_equi_vec_normalize(k)
        if self.softmax_factor > 0.0:  # Soft max pool
            q_scale_inv = channel_equi_vec_normalize(q)
            sim3_invariant_w = (q_scale_inv * k_scale_inv).mean(dim=2, keepdim=True)
            pooling_w = torch.softmax(
                self.softmax_factor * sim3_invariant_w, dim=self.pooling_dim
            )
            out = (x * pooling_w).sum(self.pooling_dim)
            if reshape_flag:
                out = out.reshape(B, N, C, 3).permute(0, 2, 3, 1)
            if return_weight:
                return out, pooling_w
            else:
                return out
        else:  # hard max pool
            q_para_len = (q * k_scale_inv).sum(dim=2, keepdim=True)
            selection = torch.argmax(q_para_len, dim=self.pooling_dim, keepdim=True)
            _expand_args = [-1] * selection.ndim
            _expand_args[2] = 3
            selection = selection.expand(*_expand_args)
            selected_x = torch.gather(input=x, dim=self.pooling_dim, index=selection)
            selected_x = selected_x.squeeze(self.pooling_dim)
            if reshape_flag:
                selected_x = selected_x.reshape(B, N, C, 3).permute(0, 2, 3, 1)
            if return_weight:
                return selected_x, None
            else:
                return selected_x


class VecLinNormAct(nn.Module):
    # support vector scalar hybrid operation
    def __init__(
        self,
        v_in: int,
        v_out: int,
        act_func,
        s_in=0,
        s_out=0,
        mode="so3",
        shared_nonlinearity=False,
        vs_dir_learnable=True,
        normalization_method=None,
        cross=False,
        safe_bound=-1,
        act_mode="vec",
        pre_act_len_bound=-1.0,
        return_tuple=True,
    ) -> None:
        super().__init__()
        self.scalar_out_flag = s_out > 0
        self.lin = VecLinear(
            v_in,
            v_out,
            s_in,
            s_out,
            mode=mode,
            vec2scalar_dir_learnable=vs_dir_learnable,
            cross=cross,
        )
        self.act = VecActivation(
            v_out,
            act_func,
            shared_nonlinearity,
            mode,
            cross=cross,
            normalization_method=normalization_method,
            safe_bound=safe_bound,
            act_mode=act_mode,
            pre_act_len_bound=pre_act_len_bound,
        )
        self.act_func = act_func
        self.return_tuple = return_tuple

        self.use_s_normalization = normalization_method is not None and s_out > 0
        if self.use_s_normalization:
            self.s_nrm = normalization_method(s_out)
        return

    def forward(self, v, s=None):
        if isinstance(v, tuple) and len(v) > 1:
            v, s = v
        if self.scalar_out_flag:  # hybrid mode
            v_out, s_out = self.lin(v, s)
            v_act = self.act(v_out)
            if self.use_s_normalization:
                s_out = self.s_nrm(s_out)
            s_act = self.act_func(s_out)
            return v_act, s_act
        else:
            v_out, _ = self.lin(v, s)
            v_act = self.act(v_out)
            if self.return_tuple:
                return v_act, None
            else:
                return v_act


class VecResBlock(nn.Module):
    # Different from the original vnn code, the order changed, first linear and the activate, so the last layer has an act option; Note, the network will be different especially when applying max pool, in vnn original code, first do pooling and then do the activation, but here we first do activation and then do the pooling
    # * if set scalar out channels, return 2 values, else return 1 values, not elegant, but this is for running with old codes and models
    def __init__(
        self,
        v_in,
        v_out,
        v_hidden,
        act_func,
        s_in=0,
        s_out=0,
        s_hidden=0,
        mode="so3",
        last_activate=True,
        vs_dir_learnable=True,
        cross=False,
        normalization_method=None,
        safe_bound=-1.0,
    ) -> None:
        super().__init__()

        self.last_activate = last_activate
        self.act_func = act_func

        self.s_in_features = s_in
        self.s_out_features = s_out
        self.s_hidden_features = s_hidden

        self.fc0 = VecLinNormAct(
            v_in=v_in,
            v_out=v_hidden,
            s_in=s_in,
            s_out=s_hidden,
            act_func=act_func,
            shared_nonlinearity=False,
            mode=mode,
            vs_dir_learnable=vs_dir_learnable,
            cross=cross,
            normalization_method=normalization_method,
        )

        self.lin1 = VecLinear(
            v_in=v_hidden,
            v_out=v_out,
            s_in=s_hidden,
            s_out=s_out,
            mode=mode,
            vec2scalar_dir_learnable=vs_dir_learnable,
            cross=cross,
        )

        if self.last_activate:
            self.act2 = VecActivation(
                in_features=v_out,
                act_func=act_func,
                shared_nonlinearity=False,
                mode=mode,
                cross=cross,
                normalization_method=normalization_method,
                safe_bound=safe_bound,
            )

        self.shortcut = None if v_in == v_out else VecLinear(v_in, v_out, mode=mode)
        if (
            self.s_in_features > 0
            and self.s_out_features > 0
            and self.s_in_features != self.s_out_features
        ):
            self.s_shortcut = nn.Linear(
                self.s_in_features, self.s_out_features, bias=True
            )
        else:
            self.s_shortcut = None

        self.se3_flag = mode == "se3"
        if self.se3_flag:
            # ! this is because the short cut add another t!
            self.subtract = VecLinear(v_in, v_out, mode="se3")

    def forward(self, v, s=None):
        # strict shape x: [B,C,3,N]; [B,C,N]
        assert v.shape[2] == 3, "vec dim should be at dim [2]"
        if self.s_in_features == 0:
            s = None  # for more flexible usage, the behavior is determined by init, not passed in args

        v_net, s_net = self.fc0(v, s)
        dv, ds = self.lin1(v_net, s_net)

        if self.shortcut is not None:
            v_s, _ = self.shortcut(v)
        else:
            v_s = v
        v_out = v_s + dv
        if self.se3_flag:
            subtract, _ = self.subtract(v)
            v_out = v_out - subtract
        if self.last_activate:
            v_out = self.act2(v_out)

        if self.s_shortcut is not None:
            assert s is not None and ds is not None
            s_s = self.s_shortcut(s.transpose(-1, 1)).transpose(-1, 1)
            s_out = s_s + ds
        elif ds is not None:  # s_in == s_out or s_in = 0
            if s is None:
                s_out = ds
            else:
                s_out = s + ds
        else:
            s_out = None

        if s_out is not None and self.last_activate:
            s_out = self.act_func(s_out)
        return v_out, s_out
