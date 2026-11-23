"""
Opt-in, read-only diagnostics for the LAT PGD inner loop.

Nothing in this module writes to any `.attack`, `.grad`, or optimizer state -- it only
reads (via `.detach().clone()` snapshots taken by the caller) and reports. Importing or
calling these functions must never change training behavior; `diagnostics=False` (the
default everywhere this is wired in) skips all of it.
"""
import csv
import os

import torch


_STAT_NAMES = ("mean", "median", "p90", "p99", "max")


def summarize_tensor(x: torch.Tensor) -> dict:
    """Distribution summary (mean/median/p90/p99/max) of a 1-D tensor of norms."""
    if x.numel() == 0:
        return {name: float("nan") for name in _STAT_NAMES}
    x = x.float()
    return {
        "mean": x.mean().item(),
        "median": x.median().item(),
        "p90": torch.quantile(x, 0.90).item(),
        "p99": torch.quantile(x, 0.99).item(),
        "max": x.max().item(),
    }


def build_diagnostic_row(
    *,
    epoch,
    pgd_iter: int,
    batch_start_idx,
    layer_label: str,
    epsilon: float,
    pre_step: torch.Tensor,
    post_step: torch.Tensor,
    post_proj: torch.Tensor,
    attack_mask: torch.Tensor,
) -> dict:
    """
    pre_step:  adv.attack right before adv_optim.step() (this iteration's starting point)
    post_step: adv.attack right after adv_optim.step(), before clip_attack() (raw, unprojected)
    post_proj: adv.attack right after clip_attack() (this iteration's retained perturbation)
    All three share shape [batch, seq, hidden_dim]; norms are taken over dim=-1 (hidden_dim),
    matching GDAdversary.clip_attack(), then restricted to attacked token positions.
    """
    seq_len = pre_step.shape[1]
    mask = attack_mask[:, :seq_len].bool()
    n_tokens = int(mask.sum().item())

    delta_norm = torch.norm(post_proj, dim=-1)[mask]
    pre_proj_norm = torch.norm(post_step, dim=-1)[mask]
    raw_update_norm = torch.norm(post_step - pre_step, dim=-1)[mask]
    retained_update_norm = torch.norm(post_proj - pre_step, dim=-1)[mask]

    tol = 1e-4 * max(epsilon, 1.0)
    if n_tokens > 0:
        frac_at_boundary = float((pre_proj_norm >= epsilon - tol).float().mean().item())
    else:
        frac_at_boundary = float("nan")

    row = {
        "epoch": epoch,
        "pgd_iter": pgd_iter,
        "batch_start_idx": batch_start_idx,
        "layer": layer_label,
        "epsilon": epsilon,
        "n_tokens": n_tokens,
        "frac_at_boundary": frac_at_boundary,
    }
    for name, tensor in (
        ("delta_norm", delta_norm),
        ("raw_update_norm", raw_update_norm),
        ("retained_update_norm", retained_update_norm),
    ):
        stats = summarize_tensor(tensor)
        for stat_name, val in stats.items():
            row[f"{name}_{stat_name}"] = val
            row[f"{name}_over_eps_{stat_name}"] = val / epsilon if epsilon else float("nan")
    return row


def print_diagnostic_row(row: dict) -> None:
    print(
        f"[diag] epoch={row['epoch']} iter={row['pgd_iter']} batch={row['batch_start_idx']} "
        f"layer={row['layer']} n={row['n_tokens']} "
        f"delta_norm(mean/p99/max)={row['delta_norm_mean']:.3f}/{row['delta_norm_p99']:.3f}/{row['delta_norm_max']:.3f} "
        f"raw_upd(mean/p99)={row['raw_update_norm_mean']:.4f}/{row['raw_update_norm_p99']:.4f} "
        f"retained_upd(mean/p99)={row['retained_update_norm_mean']:.4f}/{row['retained_update_norm_p99']:.4f} "
        f"frac_boundary={row['frac_at_boundary']:.3f}"
    )


def append_diagnostics_csv(path: str, rows: list) -> None:
    if not rows:
        return
    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
    fieldnames = list(rows[0].keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def capture_clean_activations(model, adversary_locations, input_ids) -> dict:
    """
    One no_grad forward pass of `model` on `input_ids`, capturing the OUTPUT activation
    of each submodule named in `adversary_locations` -- the exact tensor `GDAdversary.forward()`
    perturbs at that location (mlp output for numeric layers, embed_tokens output for
    "embedding"). Must be called before add_hooks()/deepspeed_add_hooks() attaches any
    GDAdversary to those same submodules, so the captured activations are genuinely clean.

    Returns {(layer, subcomponent): tensor}, matching adversary_locations' keys/order.
    """
    activations = {}
    handles = []

    def make_hook(key):
        def hook(module, inputs, output):
            activations[key] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    for layer, subcomponent in adversary_locations:
        submodule = model.get_submodule(layer).get_submodule(subcomponent)
        handles.append(submodule.register_forward_hook(make_hook((layer, subcomponent))))

    with torch.no_grad():
        model(input_ids=input_ids)

    for handle in handles:
        handle.remove()

    return activations


def compute_D_and_P_from_rows(rows: list, layer_label: str) -> tuple:
    """
    From the accumulated per-PGD-iteration diagnostic rows (all layers, one PGD run),
    extract for `layer_label`:
      D_stats[stat] = delta_norm_{stat} at the final PGD iteration
      P_stats[stat] = sum over all PGD iterations of retained_update_norm_{stat}
    D and P are exact at stat="mean" (final value / linearity-of-expectation sum,
    respectively, as established for the mean trajectory); at median/p90/p99/max they are
    the same documented approximations (final-iteration order statistic; sum-of-per-iteration
    order-statistics) used elsewhere in this diagnostic, not true pooled quantiles.
    """
    layer_rows = [r for r in rows if r["layer"] == layer_label]
    if not layer_rows:
        return {}, {}
    final_iter = max(r["pgd_iter"] for r in layer_rows)
    final_row = next(r for r in layer_rows if r["pgd_iter"] == final_iter)
    D_stats = {stat: final_row[f"delta_norm_{stat}"] for stat in _STAT_NAMES}
    P_stats = {stat: sum(r[f"retained_update_norm_{stat}"] for r in layer_rows) for stat in _STAT_NAMES}
    return D_stats, P_stats


def build_activation_norm_row(
    *,
    epoch,
    batch_start_idx,
    layer_label: str,
    token_stream: str,
    epsilon: float,
    h: torch.Tensor,
    attack_mask: torch.Tensor,
    D_stats: dict,
    P_stats: dict,
) -> dict:
    """
    h: the clean activation tensor captured by capture_clean_activations() for this layer
       (shape [batch, seq, hidden_dim] -- the exact tensor GDAdversary would perturb).
    Reports ||h||_2 (dim=-1, matching clip_attack's convention), eps/||h||, and D/||h||,
    P/||h|| as ratios of matched quantiles (e.g. D_p99/h_norm_p99) -- an approximation for
    anything beyond "mean" since no per-token D_i/P_i are available (see
    compute_D_and_P_from_rows), documented here rather than presented as an exact per-token
    ratio.
    """
    seq_len = h.shape[1]
    mask = attack_mask[:, :seq_len].bool()
    n_tokens = int(mask.sum().item())

    h_norm = torch.norm(h, dim=-1)[mask]
    h_stats = summarize_tensor(h_norm)

    row = {
        "epoch": epoch,
        "batch_start_idx": batch_start_idx,
        "layer": layer_label,
        "token_stream": token_stream,
        "epsilon": epsilon,
        "n_tokens": n_tokens,
    }
    for stat_name, h_val in h_stats.items():
        d_val = D_stats.get(stat_name, float("nan"))
        p_val = P_stats.get(stat_name, float("nan"))
        row[f"h_norm_{stat_name}"] = h_val
        row[f"eps_over_h_norm_{stat_name}"] = epsilon / h_val if h_val else float("nan")
        row[f"D_{stat_name}"] = d_val
        row[f"P_{stat_name}"] = p_val
        row[f"D_over_h_norm_{stat_name}"] = d_val / h_val if h_val else float("nan")
        row[f"P_over_h_norm_{stat_name}"] = p_val / h_val if h_val else float("nan")
    return row


def print_activation_norm_row(row: dict) -> None:
    print(
        f"[diag-h] epoch={row['epoch']} batch={row['batch_start_idx']} layer={row['layer']} "
        f"stream={row['token_stream']} n={row['n_tokens']} "
        f"h_norm(mean/p99/max)={row['h_norm_mean']:.3f}/{row['h_norm_p99']:.3f}/{row['h_norm_max']:.3f} "
        f"eps/h(mean)={row['eps_over_h_norm_mean']:.4f} "
        f"D/h(mean)={row['D_over_h_norm_mean']:.4f} P/h(mean)={row['P_over_h_norm_mean']:.4f}"
    )
