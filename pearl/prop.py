from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _to_gray(image: torch.Tensor) -> torch.Tensor:
    if image.size(1) == 3:
        r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b
    return image[:, :1]


@torch.no_grad()
def _downsample_hw(x: torch.Tensor, sh: int, sw: int) -> torch.Tensor:
    sh = max(1, int(sh))
    sw = max(1, int(sw))
    if sh == 1 and sw == 1:
        return x
    return F.avg_pool2d(
        x, kernel_size=(sh, sw), stride=(sh, sw), ceil_mode=False, count_include_pad=False
    )


@torch.no_grad()
def _upsample(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size_hw, mode='bilinear', align_corners=False)


class TLP(nn.Module):
    def __init__(self, grid: int = 80, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self._target_grid = grid
        self._tau_S = 0.5
        self._diag_boost = 10.0
        self._ke_img = 5.0
        self._cg_iters = 25
        self.register_buffer('_S', None, persistent=False)

    @torch.no_grad()
    def bind_text(self, text_features: torch.Tensor):
        T = F.normalize(text_features, dim=-1)               # [C,D]
        S = T @ T.t()                                        # [C,C] cosine
        S = torch.softmax(S / max(self._tau_S, self.eps), dim=1)
        if self._diag_boost > 0:
            C = S.size(0)
            S = S + self._diag_boost * torch.eye(C, device=S.device, dtype=S.dtype)
            S = S / (S.sum(dim=1, keepdim=True) + self.eps)
        self._S = S

    @torch.no_grad()
    def _auto_scale(self, H: int, W: int) -> Tuple[int, int]:
        sh = int(max(1, round(H / float(self._target_grid))))
        sw = int(max(1, round(W / float(self._target_grid))))
        return sh, sw

    @torch.no_grad()
    def _semantic_agree(self, probs: torch.Tensor, S: Optional[torch.Tensor]) -> torch.Tensor:
        if S is None:
            return torch.zeros_like(probs[:, :1])
        Sp = torch.einsum('cc,bcij->bcij', S, probs)         # [B,C,H,W]
        agree = (probs * Sp).sum(dim=1, keepdim=True)        # [B,1,H,W]
        return agree.clamp(0.0, 1.0)

    @torch.no_grad()
    def _edge_weights(self, gray: torch.Tensor, probs: torch.Tensor, S: Optional[torch.Tensor]):
        gx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])  # [B,1,H,W-1]
        gy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])  # [B,1,H-1,W]
        gx = gx / (gx.mean() + self.eps)
        gy = gy / (gy.mean() + self.eps)
        w_h_img = torch.exp(-self._ke_img * gx)
        w_v_img = torch.exp(-self._ke_img * gy)

        if S is not None:
            p_cur = probs[:, :, :, 1:]                                # [B,C,H,W-1]
            p_prev = probs[:, :, :, :-1]
            Sp_prev = torch.einsum('cc,bcij->bcij', S, p_prev)        # [B,C,H,W-1]
            g_h = (p_cur * Sp_prev).sum(dim=1, keepdim=True)          # [B,1,H,W-1]

            p_cur = probs[:, :, 1:, :]
            p_prev = probs[:, :, :-1, :]
            Sp_prev = torch.einsum('cc,bcij->bcij', S, p_prev)
            g_v = (p_cur * Sp_prev).sum(dim=1, keepdim=True)          # [B,1,H-1,W]

            gain_h = 1.0 + g_h.clamp(0.0, 1.0)
            gain_v = 1.0 + g_v.clamp(0.0, 1.0)
            w_h = (w_h_img * gain_h).clamp_(0.0, 1.0)
            w_v = (w_v_img * gain_v).clamp_(0.0, 1.0)
        else:
            w_h, w_v = w_h_img, w_v_img

        w_h = torch.cat([torch.ones_like(w_h[:, :, :, :1]), w_h], dim=3)
        w_v = torch.cat([torch.ones_like(w_v[:, :, :1, :]), w_v], dim=2)
        return w_h, w_v

    @torch.no_grad()
    def _apply_laplacian(self, F: torch.Tensor, w_h: torch.Tensor, w_v: torch.Tensor) -> torch.Tensor:
        diff_h = F[:, :, :, 1:] - F[:, :, :, :-1]             # [B,C,H,W-1]
        wh = w_h[:, :, :, 1:]                                 # [B,1,H,W-1]
        out = torch.zeros_like(F)
        out[:, :, :, 1:]  += wh * diff_h
        out[:, :, :, :-1] -= wh * diff_h

        diff_v = F[:, :, 1:, :] - F[:, :, :-1, :]             # [B,C,H-1,W]
        wv = w_v[:, :, 1:, :]                                 # [B,1,H-1,W]
        out[:, :, 1:, :]  += wv * diff_v
        out[:, :, :-1, :] -= wv * diff_v
        return out

    @torch.no_grad()
    def _cg_solve(self, A_apply, B_init: torch.Tensor, X0: Optional[torch.Tensor] = None, iters: int = 25):
        X = B_init.clone() if X0 is None else X0.clone()
        R = B_init - A_apply(X)
        P = R.clone()
        rsold = (R * R).sum(dim=(1, 2, 3), keepdim=True)
        for _ in range(iters):
            AP = A_apply(P)
            denom = (P * AP).sum(dim=(1, 2, 3), keepdim=True) + 1e-12
            alpha = rsold / denom
            X = X + alpha * P
            R = R - alpha * AP
            rsnew = (R * R).sum(dim=(1, 2, 3), keepdim=True)
            beta = rsnew / (rsold + 1e-12)
            P = R + beta * P
            rsold = rsnew
        return X

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,      # [B,3,H,W]
        logits: torch.Tensor,     # [B,C,H,W]
        text_features: Optional[torch.Tensor] = None  # [C,D]
    ) -> torch.Tensor:
        B, C, H, W = logits.shape
        dtype_out = logits.dtype

        if text_features is not None:
            self.bind_text(text_features)
        S = None if self._S is None else self._S.to(logits.device).to(logits.dtype)

        sh, sw = self._auto_scale(H, W)
        if sh > 1 or sw > 1:
            img_s = _downsample_hw(image.float(), sh, sw)   # [B,3,Hs,Ws]
            log_s = _downsample_hw(logits.float(), sh, sw)  # [B,C,Hs,Ws]
            Hs, Ws = img_s.shape[-2:]
        else:
            img_s = image.float()
            log_s = logits.float()
            Hs, Ws = H, W

        probs = torch.softmax(log_s, dim=1)                 # [B,C,Hs,Ws]
        conf  = probs.max(dim=1, keepdim=True)[0]           # [B,1,Hs,Ws]

        agree = self._semantic_agree(probs, S)              # [B,1,Hs,Ws], ∈[0,1]
        Cw    = (conf.clamp_min(0.05) ** 2) * (1.0 + agree)

        gray = _to_gray(img_s)                              # [B,1,Hs,Ws]
        w_h, w_v = self._edge_weights(gray, probs, S)

        tau = 1

        def A_apply(X):
            return Cw * X + tau * self._apply_laplacian(X, w_h, w_v)

        B_term = Cw * log_s

        F_sol = self._cg_solve(A_apply, B_term, X0=log_s, iters=self._cg_iters)
        
        if sh > 1 or sw > 1:
            F_sol = _upsample(F_sol, (H, W)).to(dtype_out)
        else:
            F_sol = F_sol.to(dtype_out)
        return F_sol
