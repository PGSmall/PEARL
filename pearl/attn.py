import torch
import torch.nn.functional as F


def _polar_unitary_newton_schulz(M: torch.Tensor, iters: int = 5, eps: float = 1e-6) -> torch.Tensor:
    B, d, _ = M.shape
    I = torch.eye(d, device=M.device, dtype=M.dtype).unsqueeze(0).expand(B, -1, -1)
    A = M.transpose(1, 2) @ M
    Anorm = A.detach().reshape(B, -1).norm(dim=1).clamp_min(eps).view(B, 1, 1)
    A_tilde = (A / Anorm) + eps * I
    Y = A_tilde
    Z = I.clone()
    for _ in range(iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    inv_sqrt_A = Z / torch.sqrt(Anorm)
    R = M @ inv_sqrt_A
    return R


def pa_attn(
    q: torch.Tensor,     # [B*H, T, d]
    k: torch.Tensor,     # [B*H, T, d]
    v: torch.Tensor,     # [B*H, T, d]
    detach_cls: bool = True,
    *,
    use_kk: bool = True,
    alpha_kk: float = 0.08,
    align_solver: str = "polar",   # 'polar' | 'svd'
    polar_iters: int = 5,
    return_attn: bool = False,     # [B*H, T, T]
    with_attn: bool = False        # (out, attn)
):
    dtype_out = q.dtype
    q32 = q.float(); k32 = k.float(); v32 = v.float()

    BHT, T, d = q32.shape
    scale = d ** -0.5

    w = q32.norm(dim=2, keepdim=True)
    if detach_cls and T > 0:
        w[:, 0:1, :] = 0.0
    w = w / (w.sum(dim=1, keepdim=True) + 1e-6)

    q_mu = (w * q32).sum(dim=1, keepdim=True)
    k_mu = (w * k32).sum(dim=1, keepdim=True)
    q_c = q32 - q_mu
    k_c = k32 - k_mu

    M = torch.bmm(k_c.transpose(1, 2), q_c)
    if align_solver == "polar":
        R = _polar_unitary_newton_schulz(M, iters=polar_iters, eps=1e-6)
    elif align_solver == "svd":
        try:
            U, _, Vh = torch.linalg.svd(M, full_matrices=False)
            R = torch.bmm(U, Vh)
        except Exception:
            R = torch.eye(d, device=M.device, dtype=M.dtype).unsqueeze(0).repeat(BHT, 1, 1)
    else:
        raise ValueError(f"Unknown align_solver: {align_solver}")

    k_aligned = torch.bmm(k32, R)
    logits_pa = torch.bmm(q32 * scale, k_aligned.transpose(1, 2))

    if use_kk:
        logits_kk = torch.bmm(k_c, k_c.transpose(1, 2)) * scale
        logits_pa = logits_pa + alpha_kk * logits_kk

    attn = F.softmax(logits_pa, dim=-1)  # [B*H, T, T]

    if return_attn:
        return attn.to(dtype_out)

    out = torch.bmm(attn, v32)  # [B*H, T, d]
    if with_attn:
        return out.to(dtype_out), attn.to(dtype_out)
    return out.to(dtype_out)
