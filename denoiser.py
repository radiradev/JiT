import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # soflow hyperparams
        self.lambda_fm = getattr(args, 'lambda_fm', 0.75)
        self.adaptive_loss_p = getattr(args, 'adaptive_loss_p', 0.5)

        # self-flow hyperparams
        self.self_flow = getattr(args, 'self_flow', False)
        self.repr_loss_weight = getattr(args, 'repr_loss_weight', 1.0)
        self.student_align_layer = getattr(args, 'student_align_layer', 3)
        self.teacher_align_layer = getattr(args, 'teacher_align_layer', 8)
        self.self_flow_ema_decay = getattr(args, 'self_flow_ema_decay', 0.999)

        # self-flow EMA teacher and projector (initialized lazily in _init_self_flow)
        self.teacher_net = None
        self.projector = None

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, r_value=None):
        if r_value is not None:
            return self._forward_soflow(x, r_value)

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten())
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    def _forward_soflow(self, x, r_value):
        """SoFlow: flow matching + solution consistency loss."""
        b = x.size(0)

        # split batch into FM and consistency samples
        fm_mask = torch.rand(b, device=x.device) < self.lambda_fm
        scm_mask = ~fm_mask

        t = self.sample_t(b, device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        # --- Flow matching loss on fm_mask samples ---
        z_t = t * x + (1 - t) * e
        v_target = (x - z_t) / (1 - t).clamp_min(self.t_eps)

        x_pred_t = self.net(z_t, t.flatten())
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
            x_pred_l = self.net(z_l, l.flatten())

        # student prediction at t (already computed above as x_pred_t)
        scm_per_sample = ((x_pred_t - x_pred_l) ** 2).mean(dim=(1, 2, 3))

        # adaptive weighting for consistency
        with torch.no_grad():
            scm_weights = 1.0 / (scm_per_sample.detach() ** self.adaptive_loss_p + 1e-6)
            scm_weights = scm_weights / scm_weights.mean()

        scm_loss = (scm_per_sample * scm_weights * scm_mask.float()).sum() / scm_mask.float().sum().clamp_min(1.0)

        total_loss = fm_loss + scm_loss

        return total_loss, fm_loss.detach(), scm_loss.detach()

    def _init_self_flow(self):
        """Lazily initialize self-flow teacher network and projector."""
        if self.teacher_net is None:
            self.teacher_net = copy.deepcopy(self.net)
            for p in self.teacher_net.parameters():
                p.requires_grad = False
            self.teacher_net.eval()

            hidden_size = self.net.hidden_size
            self.projector = nn.Sequential(
                nn.RMSNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            ).to(next(self.net.parameters()).device)

    @torch.no_grad()
    def update_self_flow_teacher(self):
        """Update EMA teacher for self-flow."""
        if self.teacher_net is None:
            return
        decay = self.self_flow_ema_decay
        for t_param, s_param in zip(self.teacher_net.parameters(), self.net.parameters()):
            t_param.detach().mul_(decay).add_(s_param, alpha=1 - decay)

    def _forward_self_flow(self, x):
        """Self-Flow: flow matching + representation alignment loss.

        Based on Chefer et al. (BFL) - Self-Supervised Flow Matching.
        Uses an EMA teacher model, dual-timestep scheduling with per-patch
        masking on the student input, and a representation alignment loss.

        Per-patch masking (Eq. 6-7 of the paper): the student input is
        constructed so that masked patches are interpolated at t_student
        while unmasked patches are at t_teacher_raw. The teacher sees the
        cleaner view at max(t_student, t_teacher_raw).
        """
        self._init_self_flow()

        b = x.size(0)
        device = x.device
        patch_size = self.net.patch_size

        # Dual-timestep scheduling
        t_student = self.sample_t(b, device=device)  # (B,)
        t_teacher_raw = self.sample_t(b, device=device)  # (B,)
        # Teacher always sees cleaner input: t_teacher >= both drawn times
        t_teacher = torch.maximum(t_student, t_teacher_raw)

        e = torch.randn_like(x) * self.noise_scale

        # Per-patch masking for student input (Eq. 6)
        # Draw two raw timesteps t, s and a mask M. Each student token gets
        # noised at t (if masked) or s (if unmasked). The teacher sees the
        # cleaner view at τ_min = min{t,s} (paper) = max in our 0→1 convention.
        _, _, H, W = x.shape
        grid_h, grid_w = H // patch_size, W // patch_size
        mask = torch.rand((b, grid_h, grid_w), device=device) < 0.5  # mask_ratio=0.5

        t_s_patch = t_student.view(b, 1, 1).expand(b, grid_h, grid_w)
        t_raw_patch = t_teacher_raw.view(b, 1, 1).expand(b, grid_h, grid_w)
        t_student_patch = torch.where(mask, t_s_patch, t_raw_patch)

        # Upsample per-patch times to pixel resolution
        t_student_pixel = t_student_patch.unsqueeze(1)  # (B, 1, grid_h, grid_w)
        t_student_pixel = t_student_pixel.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)

        # Student input: per-patch interpolation between noise and data
        z_student = t_student_pixel * x + (1 - t_student_pixel) * e

        # Student forward with per-patch timesteps and hidden states
        # Flatten per-patch times to (B, num_patches) for the model
        times_student_model = t_student_patch.reshape(b, -1)  # (B, grid_h * grid_w)

        x_pred_student, student_hiddens = self.net(
            z_student, times_student_model, return_hiddens=True
        )

        # Derive velocity prediction and target from model output
        # Use per-pixel times for the velocity conversion
        v_pred = (x_pred_student - z_student) / (1 - t_student_pixel).clamp_min(self.t_eps)
        v_target = (x - z_student) / (1 - t_student_pixel).clamp_min(self.t_eps)

        # Flow matching loss
        flow_loss = ((v_target - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()

        # Representation alignment loss
        repr_loss = torch.zeros((), device=device)
        if self.repr_loss_weight > 0:
            self.teacher_net.eval()

            # Teacher input at t_teacher (cleaner)
            t_t = t_teacher.view(-1, *([1] * (x.ndim - 1)))
            z_teacher = t_t * x + (1 - t_t) * e

            with torch.no_grad():
                _, teacher_hiddens = self.teacher_net(
                    z_teacher, t_teacher, return_hiddens=True
                )

            # Get representations from specified layers
            student_repr = student_hiddens[self.student_align_layer]
            teacher_repr = teacher_hiddens[self.teacher_align_layer]

            # Project student representation to match teacher
            student_proj = self.projector(student_repr)

            # Cosine similarity loss
            repr_loss = 1.0 - F.cosine_similarity(
                student_proj, teacher_repr.detach(), dim=-1
            ).mean()

        total_loss = flow_loss + self.repr_loss_weight * repr_loss

        return total_loss, flow_loss.detach(), repr_loss.detach()

    @torch.no_grad()
    def generate(self):
        bsz = 16
        device = next(self.net.parameters()).device
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
            z = stepper(z, t, t_next)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1])
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t):
        x_pred = self.net(z, t.flatten())
        v = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        return v

    @torch.no_grad()
    def _euler_step(self, z, t, t_next):
        v_pred = self._forward_sample(z, t)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next):
        v_pred_t = self._forward_sample(z, t)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next)

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
