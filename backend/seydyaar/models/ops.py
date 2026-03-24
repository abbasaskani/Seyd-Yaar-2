from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from .scoring import score_current_m_s, score_waves_hs


def ops_feasibility(
    current_m_s: np.ndarray,
    waves_hs_m: np.ndarray,
    priors: Dict,
    gear_depth_m: float = 10.0,
    wind_m_s: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Operational feasibility Pops (0..1)."""
    soft_max = float(priors.get("waves_hs_soft_max_m", 1.5))
    s_w = score_waves_hs(waves_hs_m, soft_max_m=soft_max)

    opt = float(priors.get("current_opt_m_s", 0.4))
    sig = float(priors.get("current_sigma_m_s", 0.25))
    s_c = score_current_m_s(current_m_s, opt_m_s=opt, sigma_m_s=sig)

    d = float(gear_depth_m)
    w_waves = 0.55 + (10.0 - d) * 0.01
    w_curr = 0.45 + (d - 10.0) * 0.01
    w_waves = float(np.clip(w_waves, 0.40, 0.70))
    w_curr = float(np.clip(w_curr, 0.30, 0.60))

    parts = [w_waves * s_w, w_curr * s_c]
    if wind_m_s is not None:
        wind_soft = float(priors.get("wind_soft_max_m_s", 9.0))
        s_wind = 1.0 / (1.0 + np.exp((wind_m_s - wind_soft) / 1.2))
        parts.append(0.15 * s_wind)
    pops = np.sum(parts, axis=0) / (w_waves + w_curr + (0.15 if wind_m_s is not None else 0.0))
    return np.clip(pops, 0.0, 1.0)
