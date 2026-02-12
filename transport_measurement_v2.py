import time
import numpy as np
import tqdm

from qcodes.station import Station
from qcodes.dataset import (
    initialise_or_create_database_at,
    load_or_create_experiment,
    Measurement,
)
from qcodes.parameters import Parameter

# QCoDeS 0.54.4: Instrument.close_all import can vary with layout
try:
    from qcodes.instrument import Instrument
except Exception:
    from qcodes.instrument.instrument import Instrument  # type: ignore


STATION_YAML = "stations/g15/transport.station.yaml"
DB_PATH = "databases/ML_WSe2.db"
EXPERIMENT_NAME = "transport"
SAMPLE_NAME = "Device 20"

# ---- set initial voltages for this run (edit) ----
BG_INIT = 10
TG_INIT = 6

# ---- what to do at the end (edit) ----
BG_FINAL = 0
TG_FINAL = 0

# ---- sweep grid ----
# Choose these intentionally in the script (experiment intent).
# Station already enforces limits, step, inter_delay, post_delay on BG/TG.
BG_POINTS = np.linspace(10, 0, 40)
TG_POINTS = np.linspace(6.0, 3, 20)


def main() -> None:
    # ---- connect ----
    Instrument.close_all()
    station = Station(config_file=STATION_YAML)

    uvolt = station.load_instrument("uvolt")
    lia_I = station.load_instrument("lia_I")
    lia_V = station.load_instrument("lia_V")

    # ---- semantic parameters (from alias in station.yaml) ----
    BG = uvolt.back_gate   # alias of dac0.voltage
    TG = uvolt.top_gate    # alias of dac1.voltage

    # ---- lock-in readouts ----
    I_X, I_Y = lia_I.X, lia_I.Y
    V_X, V_Y = lia_V.X, lia_V.Y

    # ---- lock-in settling ----
    # Station YAML can’t automatically do “wait 7*tau”, so we compute it here.
    # (time_constant() is in seconds for the SR830 driver)
    settle_factor = 3.0
    settle_s = settle_factor * max(float(lia_I.time_constant()), float(lia_V.time_constant()))

    # ---- database ----
    initialise_or_create_database_at(DB_PATH)
    exp = load_or_create_experiment(EXPERIMENT_NAME, SAMPLE_NAME)

    R = Parameter("R", label="Vx/Ix", unit="Ohm")

    # ---- measurement registration ----
    meas = Measurement(exp=exp)
    meas.register_parameter(TG)
    meas.register_parameter(BG)
    meas.register_parameter(I_X, setpoints=(TG, BG))
    meas.register_parameter(I_Y, setpoints=(TG, BG))
    meas.register_parameter(V_X, setpoints=(TG, BG))
    meas.register_parameter(V_Y, setpoints=(TG, BG))
    meas.register_parameter(R, setpoints=(TG, BG))

    # ---- hooks ----
    def before_run() -> None:
        # set initial voltages (respects limits/step/inter_delay/post_delay from station)
        BG(BG_INIT)
        TG(TG_INIT)
        time.sleep(settle_s)

    def after_run() -> None:
        # zero voltages at the end
        print("Rampind down")
        BG(BG_FINAL)
        TG(TG_FINAL)
        time.sleep(settle_s)

    # QCoDeS Measurement supports these hooks
    meas.add_before_run(before_run,())
    meas.add_after_run(after_run,())

    meas.write_period = 0.5

    # ---- run ----
    try:
        with meas.run() as datasaver:
            for vtg in tqdm.tqdm(TG_POINTS):
                TG(vtg)                 # stepping/delays/limits handled by station parameter config
                time.sleep(settle_s)

                for vbg in BG_POINTS:
                    BG(vbg)
                    time.sleep(settle_s)

                    ix = float(I_X())
                    vx = float(V_X())
                    r = vx / ix if abs(ix) > 1e-30 else float("nan")

                    datasaver.add_result(
                        (TG, float(vtg)),
                        (BG, float(vbg)),
                        (I_X, ix),
                        (I_Y, float(I_Y())),
                        (V_X, vx),
                        (V_Y, float(V_Y())),
                        (R, r)
                    )

            print(f"Saved run_id: {datasaver.run_id}")

    finally:
        # In case of Ctrl+C / exception: still try to zero
        try:
            after_run()
        except Exception:
            pass

if __name__ == "__main__":
    main()
