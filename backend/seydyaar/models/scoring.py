from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _logistic_penalty(x: np.ndarray, threshold: float, softness: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((x - threshold) / max(float(softness), 1e-6)))


def _robust_norm01(arr: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> np.ndarray:
    lo, hi = np.nanpercentile(arr, p_lo), np.nanpercentile(arr, p_hi)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _median3x3(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    pad = np.pad(a, ((1,1),(1,1)), mode="edge")
    stack = np.stack([
        pad[0:-2,0:-2], pad[0:-2,1:-1], pad[0:-2,2:],
        pad[1:-1,0:-2], pad[1:-1,1:-1], pad[1:-1,2:],
        pad[2:,0:-2],   pad[2:,1:-1],   pad[2:,2:],
    ], axis=0)
    return np.nanmedian(stack, axis=0).astype(np.float32)


def score_temp_c(sst_c: np.ndarray, opt_c: float, sigma_c: float) -> np.ndarray:
    return _gauss(sst_c, opt_c, sigma_c)


def score_chl_mg_m3(chl: np.ndarray, opt_mg_m3: float, sigma_log10: float) -> np.ndarray:
    chl = np.clip(chl, 1e-6, None)
    return _gauss(np.log10(chl), np.log10(opt_mg_m3), sigma_log10)


def score_current_m_s(spd: np.ndarray, opt_m_s: float, sigma_m_s: float) -> np.ndarray:
    return _gauss(spd, opt_m_s, sigma_m_s)


def score_waves_hs(hs_m: np.ndarray, soft_max_m: float = 1.5, softness: float = 0.35) -> np.ndarray:
    return _logistic_penalty(hs_m, soft_max_m, softness)


def score_salinity_psu(sss: np.ndarray, opt_psu: float, sigma_psu: float) -> np.ndarray:
    return _gauss(sss, opt_psu, sigma_psu)


def score_oxygen_mmol_m3(o2: np.ndarray, opt_mmol_m3: float, sigma_mmol_m3: float, lower_soft: float | None = None) -> np.ndarray:
    s = _gauss(o2, opt_mmol_m3, sigma_mmol_m3)
    if lower_soft is not None:
        s = np.minimum(s, 1.0 / (1.0 + np.exp((lower_soft - o2) / 1.0)))
    return np.clip(s, 0.0, 1.0)


def score_mld_m(mld: np.ndarray, opt_m: float, sigma_m: float) -> np.ndarray:
    return _gauss(mld, opt_m, sigma_m)


def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(arr.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def boa_front_score(temp_like: np.ndarray, chl_like: np.ndarray, ssh_like: np.ndarray,
                    w_temp: float = 0.4, w_chl: float = 0.4, w_ssh: float = 0.2) -> np.ndarray:
    tf = gradient_magnitude(_median3x3(temp_like))
    cf = gradient_magnitude(_median3x3(chl_like))
    sf = gradient_magnitude(_median3x3(ssh_like))
    s = w_temp * _robust_norm01(tf) + w_chl * _robust_norm01(cf) + w_ssh * _robust_norm01(sf)
    return np.clip(s, 0.0, 1.0)


def front_score(temp_front: np.ndarray, chl_front: np.ndarray, ssh_front: np.ndarray,
                w_temp: float = 0.5, w_chl: float = 0.25, w_ssh: float = 0.25) -> np.ndarray:
    s = w_temp * temp_front + w_chl * chl_front + w_ssh * ssh_front
    return _robust_norm01(s)


def eddy_score(ssh_m: np.ndarray, u_m_s: Optional[np.ndarray] = None, v_m_s: Optional[np.ndarray] = None) -> np.ndarray:
    ssh_anom = ssh_m - np.nanmedian(ssh_m)
    parts = [_robust_norm01(np.abs(ssh_anom))]
    if u_m_s is not None and v_m_s is not None:
        du_dy, du_dx = np.gradient(u_m_s.astype(np.float32))
        dv_dy, dv_dx = np.gradient(v_m_s.astype(np.float32))
        vort = dv_dx - du_dy
        shear = du_dx - dv_dy
        stretch = du_dx + dv_dy
        ow = stretch * stretch + shear * shear - vort * vort
        eke = 0.5 * (u_m_s.astype(np.float32)**2 + v_m_s.astype(np.float32)**2)
        parts.extend([_robust_norm01(np.maximum(-ow, 0.0)), _robust_norm01(np.abs(vort)), _robust_norm01(eke)])
    return np.clip(np.nanmean(np.stack(parts, axis=0), axis=0), 0.0, 1.0)


@dataclass
class HabitatInputs:
    sst_c: np.ndarray
    chl_mg_m3: np.ndarray
    current_m_s: np.ndarray
    waves_hs_m: np.ndarray
    ssh_m: np.ndarray
    salinity_psu: Optional[np.ndarray] = None
    oxygen_mmol_m3: Optional[np.ndarray] = None
    mld_m: Optional[np.ndarray] = None
    u_current_m_s: Optional[np.ndarray] = None
    v_current_m_s: Optional[np.ndarray] = None


def habitat_scoring(inputs: HabitatInputs, priors: Dict, weights: Dict) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    s_temp = score_temp_c(inputs.sst_c, priors["sst_opt_c"], priors["sst_sigma_c"])
    s_chl = score_chl_mg_m3(inputs.chl_mg_m3, priors["chl_opt_mg_m3"], priors["chl_sigma_log10"])
    s_cur = score_current_m_s(inputs.current_m_s, priors["current_opt_m_s"], priors["current_sigma_m_s"])

    tf = gradient_magnitude(inputs.sst_c)
    cf = gradient_magnitude(np.log10(np.clip(inputs.chl_mg_m3, 1e-6, None)))
    sf = gradient_magnitude(inputs.ssh_m)
    fw = priors.get("front_weights", {"temp":0.5,"chl":0.25,"ssh":0.25})
    s_front = front_score(tf, cf, sf, fw.get("temp",0.5), fw.get("chl",0.25), fw.get("ssh",0.25))
    s_boa = boa_front_score(inputs.sst_c, np.log10(np.clip(inputs.chl_mg_m3, 1e-6, None)), inputs.ssh_m,
                            fw.get("temp",0.4), fw.get("chl",0.4), fw.get("ssh",0.2))
    s_eddy = eddy_score(inputs.ssh_m, inputs.u_current_m_s, inputs.v_current_m_s)
    s_waves = score_waves_hs(inputs.waves_hs_m, priors.get("waves_hs_soft_max_m", 1.5))

    s_sal = np.ones_like(s_temp, dtype=np.float32)
    if inputs.salinity_psu is not None:
        s_sal = score_salinity_psu(inputs.salinity_psu, priors.get("sss_opt_psu", 35.05), priors.get("sss_sigma_psu", 0.325))
    s_o2 = np.ones_like(s_temp, dtype=np.float32)
    if inputs.oxygen_mmol_m3 is not None:
        s_o2 = score_oxygen_mmol_m3(inputs.oxygen_mmol_m3, priors.get("o2_opt_mmol_m3", 202.0), priors.get("o2_sigma_mmol_m3", 3.0), priors.get("o2_lower_soft_mmol_m3", None))
    s_mld = np.ones_like(s_temp, dtype=np.float32)
    if inputs.mld_m is not None:
        s_mld = score_mld_m(inputs.mld_m, priors.get("mld_opt_m", 92.0), priors.get("mld_sigma_m", 32.0))

    w = dict(weights)
    total = sum(max(v, 0.0) for v in w.values())
    if total <= 0:
        w = {"temp":1.0}
        total = 1.0
    for k in list(w.keys()):
        w[k] = max(float(w[k]), 0.0) / total

    phab = (
        w.get("temp",0.0) * s_temp +
        w.get("chl",0.0) * s_chl +
        w.get("front",0.0) * s_front +
        w.get("current",0.0) * s_cur +
        w.get("salinity",0.0) * s_sal +
        w.get("oxygen",0.0) * s_o2 +
        w.get("mld",0.0) * s_mld +
        w.get("eddy",0.0) * s_eddy
    )
    phab = np.clip(phab, 0.0, 1.0)
    phab_boaeddy = np.clip(
        0.55 * phab + 0.20 * s_boa + 0.15 * s_eddy + 0.05 * s_sal + 0.03 * s_o2 + 0.02 * s_mld,
        0.0, 1.0
    )

    comps = {
        "score_temp": s_temp,
        "score_chl": s_chl,
        "score_front": s_front,
        "score_current": s_cur,
        "score_waves": s_waves,
        "score_salinity": s_sal,
        "score_oxygen": s_o2,
        "score_mld": s_mld,
        "score_boa": s_boa,
        "score_eddy": s_eddy,
        "temp_front": tf,
        "chl_front": cf,
        "ssh_front": sf,
        "phab_boaeddy": phab_boaeddy,
    }
    return phab, comps
