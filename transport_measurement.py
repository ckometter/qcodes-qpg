from pathlib import Path
from time import monotonic, sleep
import tqdm

import numpy as np

from include.rotated_basis import RotatedBasis

import qcodes as qc
from qcodes.parameters import Parameter
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

station = qc.Station(config_file=str(Path.cwd() / "stations/g15/transport.station.yaml"))
lia1 = station.load_instrument('lia1')
lia2 = station.load_instrument('lia2')
#dmm1 = station.load_instrument('dmm1')
da = station.load_instrument('da')


Vcg_val = 0
Vt_0 = -3
Vt_f = 3
Vb_0 = -3
Vb_f = 3
Vt_npts = 100
Vb_npts = 100
v_preamp = 100

Vcg_startup = [0,0]
Vt_startup = [-2,-2]
Vb_startup = [-2,-2]

# Vcg_startup = [-10,-10]
# Vt_startup = [-15,-15]
# Vb_startup = [-15,-15]

Vcg_sh = [0,0] 
Vt_sh = [0,0]
Vb_sh = [0,0]

Vt_Range = np.linspace(Vt_0,Vt_f,Vt_npts)
Vb_Range = np.linspace(Vb_0,Vb_f,Vb_npts)
    
Vt_mesh, Vb_mesh = np.meshgrid(Vt_Range, Vb_Range)
    
mask = np.abs(Vt_mesh-(Vcg_val)) <= 100
    
Vt_masked = np.where(mask, Vt_mesh, np.nan)
Vb_masked = np.where(mask, Vb_mesh, np.nan)


n_0 = -1
n_f = 1
p_0 = -1
p_f = 1
n_npts = 75
p_npts = 75

n_Range = np.linspace(n_0,n_f,n_npts)
p_Range = np.linspace(p_0,p_f,p_npts)

transform = 'vs'
v_fixed = 0
delta = 0
rot = RotatedBasis(da.DAC0, da.DAC1, transform, delta, v_fixed)
n = rot.n
p = rot.p

R2 = Parameter(
    name="R2",
    label="Two Point Resistance",
    unit="Ohm",            # or '' for dimensionless ratio
    get_cmd=None         # it's not read from hardware; we pass values manually
)

R4 = Parameter(
    name="R4",
    label="Four Point Resistance",
    unit="Ohm",            # or '' for dimensionless ratio
    get_cmd=None         # it's not read from hardware; we pass values manually
)

def veryfirst():
    print("Starting the measurement")

def set_initial_conditions(v1, v2, v3):
    print("Ramping to intial conditions")
    for i in range(2):
        # print(f"Ramping {dmm1} to {v1[i]}")
        # dmm1.volt(v1[i])
        print(f"Ramping {da.DAC0} to {v2[i]}")
        da.DAC0.volt(v2[i])
        print(f"Ramping {da.DAC1} to {v3[i]}")
        da.DAC1.volt(v3[i])

    v1 = Vt_masked[0,0]
    v2 = Vb_masked[0,0]
    if np.isnan(v1) or np.isnan(v2):
        return
    da.DAC0.volt(v1)
    da.DAC1.volt(v2)
    print("Ramping finished")
    #input("\nPress Enter to to continue...")

def thelast(v1,v2,v3):
    print("Ramping down")
    for i in range(2):
        # print(f"Ramping {dmm1} to {v1[i]}")
        # dmm1.volt(v1[i])
        print(f"Ramping {da.DAC0} to {v2[i]}")
        da.DAC0.volt(v2[i])
        print(f"Ramping {da.DAC1} to {v3[i]}")
        da.DAC1.volt(v3[i])
    print("End of experiment")

initialise_or_create_database_at(Path.cwd() / "transport_meas.db")
exp = load_or_create_experiment(
    experiment_name="n vs p map",
    sample_name="Bilayer Graphene D15",
)

meas = Measurement(exp=exp, station=station, name="Four Probe Transport")
#meas.register_parameter(dmm1.volt)
meas.register_parameter(n)
meas.register_parameter(p)
meas.register_parameter(da.DAC0.volt, setpoints=(n,p))
meas.register_parameter(da.DAC1.volt, setpoints=(n,p))
meas.register_parameter(lia1.X, setpoints=(n,p))  # now register the dependent oone
meas.register_parameter(lia1.Y, setpoints=(n,p))  # now register the dependent oone
meas.register_parameter(lia2.X, setpoints=(n,p))  # now register the dependent oone
meas.register_parameter(lia2.Y, setpoints=(n,p))  # now register the dependent oone
meas.register_parameter(R2, setpoints=(n,p))  # now register the dependent oone
meas.register_parameter(R4, setpoints=(n,p))  # now register the dependent oone

meas.add_before_run(veryfirst, ())  # add a set-up action
meas.add_before_run(set_initial_conditions, (Vcg_startup,Vt_startup,Vb_startup))  # add a set-up action
meas.add_after_run(thelast, (Vcg_sh,Vt_sh,Vb_sh))  # add a tear-down action

meas.write_period = 0.1

with meas.run() as datasaver:
    for j in tqdm.tqdm(range(p_npts)):
        p_j = p_Range[j]
        sleep(1.8)
        for i in range(n_npts):
            n_i = n_Range[i]
            n(n_i)
            p(p_j)
            v1 = da.DAC0.volt.cache.get()
            v2 = da.DAC1.volt.cache.get()
            sleep(1.2)
            z1 = lia1.X()
            z2 = lia1.Y()
            z3 = lia2.X()
            z4 = lia2.Y()
            r2 = 0.016/z1
            r4 = z3/z1/v_preamp
            datasaver.add_result((n, n_i), (p, p_j), (da.DAC0.volt, v1), (da.DAC1.volt, v2), (lia1.X, z1), (lia1.Y, z2), (lia2.X, z3), (lia2.Y, z4), (R2, r2), (R4, r4))

    dataset2D = datasaver.dataset

ax, cbax = plot_dataset(dataset2D)