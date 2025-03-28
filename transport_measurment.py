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

station = qc.Station(config_file=Path.cwd() / "g15.station.yaml")

initialise_or_create_database_at(Path.cwd() / "performing_meas.db")
exp = load_or_create_experiment(
    experiment_name="performing_meas_using_parameters_and_dataset",
    sample_name="no sample",
)

meas = Measurement(exp=exp, station=station, name="exponential_decay")
meas.register_parameter(dac.ch1)  # register the first independent parameter
meas.register_parameter(dmm.v1, setpoints=(dac.ch1,))  # now register the dependent oone

meas.add_before_run(veryfirst, ())  # add a set-up action
meas.add_before_run(numbertwo, (dmm, dac))  # add another set-up action
meas.add_after_run(thelast, ())  # add a tear-down action

meas.write_period = 0.5

with meas.run() as datasaver:
    for set_v in np.linspace(0, 25, 10):
        dac.ch1.set(set_v)
        get_v = dmm.v1.get()
        datasaver.add_result((dac.ch1, set_v), (dmm.v1, get_v))

    dataset1D = datasaver.dataset  # convenient to have for data access and plotting