from pathlib import Path
from time import monotonic, sleep

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

def veryfirst():
    print("Starting the measurement")


def numbertwo(inst1, inst2):
    print(f"Doing stuff with the following two instruments: {inst1}, {inst2}")


def thelast():
    hp.smu1.volt(0)
    hp.smu4.volt(0)
    print("End of experiment")

initialise_or_create_database_at(Path.cwd() / "performing_meas.db")
exp = load_or_create_experiment(
    experiment_name="test",
    sample_name="no sample",
)

meas = Measurement(exp=exp, station=station, name="Four Probe Transport")
meas.register_parameter(dmm1.volt)
meas.register_parameter(hp.smu1.volt)
meas.register_parameter(hp.smu4.volt)
meas.register_parameter(lia1.X, setpoints=(hp.smu1.volt,hp.smu4.volt))  # now register the dependent oone
meas.register_parameter(lia1.Y, setpoints=(hp.smu1.volt,hp.smu4.volt))  # now register the dependent oone
meas.register_parameter(lia2.X, setpoints=(hp.smu1.volt,hp.smu4.volt))  # now register the dependent oone
meas.register_parameter(lia2.Y, setpoints=(hp.smu1.volt,hp.smu4.volt))  # now register the dependent oone

meas.add_before_run(veryfirst, ())  # add a set-up action
meas.add_before_run(numbertwo, (hp, hp))  # add a set-up action
meas.add_after_run(thelast, ())  # add a tear-down action

meas.write_period = 0.5


with meas.run() as datasaver:
    for v1 in np.linspace(0, -1, 5):
        for v2 in np.linspace(0, -1, 5):
            hp.smu4.volt(v1)
            hp.smu1.volt(v2)
            z1 = lia1.X()
            z2 = lia1.Y()
            z3 = lia2.X()
            z4 = lia2.Y()
            datasaver.add_result((hp.smu1.volt, v2), (hp.smu4.volt, v1), (lia1.X, z1), (lia1.Y, z2), (lia2.X, z3), (lia2.Y, z4))

    dataset2D = datasaver.dataset

ax, cbax = plot_dataset(dataset2D)