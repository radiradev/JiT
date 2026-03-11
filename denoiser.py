import torch
import torch.nn as nn
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # soflow hyperparams
        self.lambda_fm = getattr(args, 'lambda_fm', 0.75)
        self.adaptive_loss_p = getattr(args, 'adaptive_loss_p', 0.5)

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels, r_value=None):
        if r_value is not None:
            return self._forward_soflow(x, labels, r_value)

        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    def _forward_soflow(self, x, labels, r_value):
        """SoFlow: flow matching + solution consistency loss."""
        b = x.size(0)
        labels_dropped = self.drop_labels(labels) if self.training else labels

        # split batch into FM and consistency samples
        fm_mask = torch.rand(b, device=x.device) < self.lambda_fm
        scm_mask = ~fm_mask

        t = self.sample_t(b, device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        # --- Flow matching loss on fm_mask samples ---
        z_t = t * x + (1 - t) * e
        v_target = (x - z_t) / (1 - t).clamp_min(self.t_eps)

        x_pred_t = self.net(z_t, t.flatten(), labels_dropped)
        v_pred = (x_pred_t - z_t) / (1 - t).clamp_min(self.t_eps)

        fm_per_sample = ((v_target - v_pred) ** 2).mean(dim=(1, 2, 3))

        # adaptive weighting for FM: weight = 1 / (per_sample_loss^p + eps)
        with torch.no_grad():
            fm_weights = 1.0 / (fm_per_sample.detach() ** self.adaptive_loss_p + 1e-6)
            fm_weights = fm_weights / fm_weights.mean()

        fm_loss = (fm_per_sample * fm_weights * fm_mask.float()).sum() / fm_mask.float().sum().clamp_min(1.0)

        # --- Solution consistency loss on scm_mask samples ---
        # l = t + (1 - t) * r  (a point closer to clean than t)
        l = t + (1 - t) * r_value
        l = l.clamp(max=1.0 - self.t_eps)

        z_l = l * x + (1 - l) * e  # same noise e, different interpolation point

        # teacher prediction at l (detached)
        with torch.no_grad():
            x_pred_l = self.net(z_l, l.flatten(), labels_dropped)

        # student prediction at t (already computed above as x_pred_t)
        scm_per_sample = ((x_pred_t - x_pred_l) ** 2).mean(dim=(1, 2, 3))

        # adaptive weighting for consistency
        with torch.no_grad():
            scm_weights = 1.0 / (scm_per_sample.detach() ** self.adaptive_loss_p + 1e-6)
            scm_weights = scm_weights / scm_weights.mean()

        scm_loss = (scm_per_sample * scm_weights * scm_mask.float()).sum() / scm_mask.float().sum().clamp_min(1.0)

        total_loss = fm_loss + scm_loss

        return total_loss, fm_loss.detach(), scm_loss.detach()

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
