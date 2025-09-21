from pathlib import Path
from time import monotonic, sleep
import tqdm

import numpy as np

import qcodes as qc
from qcodes.dataset import (
    Measurement,
    initialise_or_create_database_at,
    load_by_guid,
    load_by_run_spec,
    load_or_create_experiment,
    plot_dataset,
)
from qcodes.dataset.descriptions.detect_shapes import detect_shape_of_measurement
from qcodes.instrument_drivers.mock_instruments import (
    DummyInstrument,
    DummyInstrumentWithMeasurement,
)
from qcodes.logger import start_all_logging

start_all_logging()

station = qc.Station(config_file=str(Path.cwd() / "g15.station.yaml"))
lia1 = station.load_instrument('lia1')
lia2 = station.load_instrument('lia2')
dmm1 = station.load_instrument('dmm1')
hp = station.load_instrument('hp')


Vcg_val = -28
Vt_0 = -15
Vt_f = -6
Vb_0 = -15
Vb_f = 15
Vt_npts = 50
Vb_npts = 100

Vcg_startup = [-10,-28]
Vt_startup = [-15,-15]
Vb_startup = [-15,-15]

# Vcg_startup = [-10,-10]
# Vt_startup = [-15,-15]
# Vb_startup = [-15,-15]

Vcg_sh = [-15,0] 
Vt_sh = [0,0]
Vb_sh = [0,0]

Vt_Range = np.linspace(Vt_0,Vt_f,Vt_npts)
Vb_Range = np.linspace(Vb_0,Vb_f,Vb_npts)
    
Vt_mesh, Vb_mesh = np.meshgrid(Vt_Range, Vb_Range)
    
mask = np.abs(Vt_mesh-(Vcg_val)) <= 100
    
Vt_masked = np.where(mask, Vt_mesh, np.nan)
Vb_masked = np.where(mask, Vb_mesh, np.nan)

def veryfirst():
    print("Starting the measurement")

def set_initial_conditions(v1, v2, v3):
    print("Ramping to intial conditions")
    for i in range(2):
        print(f"Ramping {dmm1} to {v1[i]}")
        dmm1.volt(v1[i])
        print(f"Ramping {hp.smu4} to {v2[i]}")
        hp.smu4.volt(v2[i])
        print(f"Ramping {hp.smu1} to {v3[i]}")
        hp.smu1.volt(v3[i])
    print("Ramping finished")
    input("\nPress Enter to to continue...")

def thelast(v1,v2,v3):
    print("Ramping down")
    for i in range(2):
        print(f"Ramping {dmm1} to {v1[i]}")
        dmm1.volt(v1[i])
        print(f"Ramping {hp.smu4} to {v2[i]}")
        hp.smu4.volt(v2[i])
        print(f"Ramping {hp.smu1} to {v3[i]}")
        hp.smu1.volt(v3[i])
    print("End of experiment")

initialise_or_create_database_at(Path.cwd() / "performing_meas.db")
exp = load_or_create_experiment(
    experiment_name="Vb vs Vt map",
    sample_name="Bilayer WSe2",
)

meas = Measurement(exp=exp, station=station, name="Four Probe Transport")
meas.register_parameter(dmm1.volt)
meas.register_parameter(hp.smu1.volt)
meas.register_parameter(hp.smu4.volt)
meas.register_parameter(lia1.X, setpoints=(hp.smu1.volt,hp.smu4.volt))  # now register the dependent oone
meas.register_parameter(lia1.Y, setpoints=(hp.smu1.volt,hp.smu4.volt))  # now register the dependent oone
meas.register_parameter(lia2.X, setpoints=(hp.smu1.volt,hp.smu4.volt))  # now register the dependent oone
meas.register_parameter(lia2.Y, setpoints=(hp.smu1.volt,hp.smu4.volt))  # now register the dependent oone
meas.register_parameter(hp.smu1.curr, setpoints=(hp.smu1.volt,hp.smu4.volt))  # now register the dependent oone
meas.register_parameter(hp.smu4.curr, setpoints=(hp.smu1.volt,hp.smu4.volt))  # now register the dependent oone

meas.add_before_run(veryfirst, ())  # add a set-up action
meas.add_before_run(set_initial_conditions, (Vcg_startup,Vt_startup,Vb_startup))  # add a set-up action
meas.add_after_run(thelast, (Vcg_sh,Vt_sh,Vb_sh))  # add a tear-down action

meas.write_period = 0.1


with meas.run() as datasaver:
    for j in tqdm.tqdm(range(Vt_npts)):
        for i in range(Vb_npts):
            v1 = Vt_masked[i,j]
            v2 = Vb_masked[i,j]
            if np.isnan(v1) or np.isnan(v2):
                continue
            hp.smu4.volt(v1)
            hp.smu1.volt(v2)
            sleep(1.5)
            i1 = hp.smu4.curr()
            i2 = hp.smu1.curr()
            z1 = lia1.X()
            z2 = lia1.Y()
            z3 = lia2.X()
            z4 = lia2.Y()
            datasaver.add_result((hp.smu1.volt, v2), (hp.smu4.volt, v1), (hp.smu1.curr, i1), (hp.smu4.curr, i2), (lia1.X, z1), (lia1.Y, z2), (lia2.X, z3), (lia2.Y, z4))
    dataset2D = datasaver.dataset

ax, cbax = plot_dataset(dataset2D)