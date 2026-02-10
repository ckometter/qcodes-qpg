"""
2D transport sweep using station.yaml + config section.

- Sweeps back gate and top gate
- Measures X and Y from both lock-ins (I and V)
- Uses station.yaml config for mapping, limits, ramp settings, settling, monitor

QCoDeS: 0.54.4
"""

import time
import yaml
import numpy as np

from qcodes.station import Station
from qcodes.dataset import (
    initialise_or_create_database_at,
    load_or_create_experiment,
    Measurement,
)

# Robust import for Instrument.close_all across QCoDeS layouts
try:
    from qcodes.instrument import Instrument
except Exception:  # pragma: no cover
    from qcodes.instrument.instrument import Instrument  # type: ignore


STATION_YAML = "transport.station.yaml"
DB_PATH = "ML_WSe2.db"
EXPERIMENT_NAME = "transport"
SAMPLE_NAME = "Device 20"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ramp_to(param, target: float, rate_V_per_s: float, step_delay_s: float) -> None:
    """Ramp param to target at ~rate_V_per_s with step_delay_s per step."""
    start = float(param())
    dv = target - start
    if dv == 0:
        return
    step = np.sign(dv) * rate_V_per_s * step_delay_s
    n = int(abs(dv) / abs(step)) + 1
    for v in np.linspace(start, target, n):
        param(v)
        time.sleep(step_delay_s)


def main() -> None:
    # ---------- Load YAML config ----------
    with open(STATION_YAML, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    cfg = y.get("config", {})
    gcfg = cfg.get("gates", {})
    lcfg = cfg.get("lockins", {})
    mcfg = cfg.get("monitor", {})

    # ---------- Connect instruments ----------
    Instrument.close_all()
    station = Station(config_file=STATION_YAML)

    uvolt = station.load_instrument("uvolt")
    lockin_I = station.load_instrument("lockin_I")
    lockin_V = station.load_instrument("lockin_V")

    # ---------- Gate mapping from config ----------
    # Example in station.yaml:
    # config:
    #   gates:
    #     back_gate: {channel: dac0, soft_limits_V: [...], ramp_rate_V_per_s: ..., step_delay_s: ...}
    #     top_gate:  {channel: dac1, ...}
    back = gcfg.get("back_gate", {})
    top = gcfg.get("top_gate", {})

    bg_chan = back.get("channel", "dac0")
    tg_chan = top.get("channel", "dac1")

    BG = getattr(uvolt, bg_chan).voltage
    TG = getattr(uvolt, tg_chan).voltage

    bg_soft = tuple(back.get("soft_limits_V", [-10.0, 10.0]))
    tg_soft = tuple(top.get("soft_limits_V", [-10.0, 10.0]))

    bg_rate = float(back.get("ramp_rate_V_per_s", 0.2))
    tg_rate = float(top.get("ramp_rate_V_per_s", 0.2))
    bg_step_delay = float(back.get("step_delay_s", 0.02))
    tg_step_delay = float(top.get("step_delay_s", 0.02))

    # Optional: apply QCoDeS parameter delays (only affects param.set calls)
    # Put these in station.yaml if you want:
    # config:
    #   gates:
    #     back_gate: {inter_delay_s: 0.0, post_delay_s: 0.0, ...}
    BG.inter_delay = float(back.get("inter_delay_s", 0.0))
    BG.post_delay = float(back.get("post_delay_s", 0.0))
    TG.inter_delay = float(top.get("inter_delay_s", 0.0))
    TG.post_delay = float(top.get("post_delay_s", 0.0))

    # ---------- Lock-in readout selection from config ----------
    # config:
    #   lockins:
    #     settle_factor_tau: 7
    settle_factor = float(lcfg.get("settle_factor_tau", 7.0))

    # SR830 driver usually has time_constant() in seconds.
    tau_I = float(lockin_I.time_constant())
    tau_V = float(lockin_V.time_constant())
    settle_s = settle_factor * max(tau_I, tau_V)

    # Always measure X,Y from both lockins (as you requested)
    I_X, I_Y = lockin_I.X, lockin_I.Y
    V_X, V_Y = lockin_V.X, lockin_V.Y

    # ---------- Define sweep grids (script-level intent) ----------
    # Keep ranges here; config is “defaults”, experiment decides actual sweep.
    BG_POINTS = np.linspace(bg_soft[0], bg_soft[1], 81)
    TG_POINTS = np.linspace(tg_soft[0], tg_soft[1], 81)

    # ---------- Database / experiment ----------
    initialise_or_create_database_at(DB_PATH)
    exp = load_or_create_experiment(EXPERIMENT_NAME, SAMPLE_NAME)

    meas = Measurement(exp=exp)
    meas.register_parameter(BG)
    meas.register_parameter(TG)
    meas.register_parameter(I_X, setpoints=(BG, TG))
    meas.register_parameter(I_Y, setpoints=(BG, TG))
    meas.register_parameter(V_X, setpoints=(BG, TG))
    meas.register_parameter(V_Y, setpoints=(BG, TG))

    # ---------- Monitor thresholds from config ----------
    # Example:
    # config:
    #   monitor:
    #     abort_if_I_R_gt_V: 1.0
    #     abort_if_V_R_gt_V: 2.0
    abort_I_R = mcfg.get("abort_if_I_R_gt_V", None)
    abort_V_R = mcfg.get("abort_if_V_R_gt_V", None)

    with meas.run() as datasaver:
        for vbg in BG_POINTS:
            vbg = clamp(float(vbg), bg_soft[0], bg_soft[1])
            ramp_to(BG, vbg, rate_V_per_s=bg_rate, step_delay_s=bg_step_delay)
            time.sleep(settle_s)

            for vtg in TG_POINTS:
                vtg = clamp(float(vtg), tg_soft[0], tg_soft[1])
                ramp_to(TG, vtg, rate_V_per_s=tg_rate, step_delay_s=tg_step_delay)
                time.sleep(settle_s)

                # Read lock-ins
                ix = float(I_X())
                iy = float(I_Y())
                vx = float(V_X())
                vy = float(V_Y())

                # Monitor (optional)
                if abort_I_R is not None and hasattr(lockin_I, "R"):
                    if float(lockin_I.R()) > float(abort_I_R):
                        raise RuntimeError("Abort: lockin_I.R above threshold")
                if abort_V_R is not None and hasattr(lockin_V, "R"):
                    if float(lockin_V.R()) > float(abort_V_R):
                        raise RuntimeError("Abort: lockin_V.R above threshold")

                datasaver.add_result(
                    (BG, vbg),
                    (TG, vtg),
                    (I_X, ix),
                    (I_Y, iy),
                    (V_X, vx),
                    (V_Y, vy),
                )

        print(f"Saved run_id: {datasaver.run_id}")


if __name__ == "__main__":
    main()
