from pathlib import Path
from time import sleep
import tqdm
import yaml
with open("transistor_characterization.yml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
meas_settings = cfg[cfg['measurement']]
balancing_settings = cfg['balancing_settings']
lockin_settings = cfg['lockin_settings']
meas_parameters = cfg['meas_parameters']
awg_settings = cfg['awg_settings']

import numpy as np

from include.capacitance_bridge import CapacitanceBridgeSR830Lockin as bridge

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
from qcodes.logger import start_all_logging

start_all_logging()

station = qc.Station(config_file=str(Path.cwd() / "g15.station.yaml"))

awg = station.load_instrument('awg')
lia1 = station.load_instrument('lia1')
da = station.load_instrument('da')
# lia2 = station.load_instrument('lia2')
# dmm1 = station.load_instrument('dmm1')
# hp = station.load_instrument('hp')

Vt_0 = -2
Vb_0 = 0

Vg_0 = 0
Vg_f = -1
Vg_npts = 100

Vg_Range = np.linspace(Vg_0,Vg_f,Vg_npts)
    
#Vt_mesh, Vb_mesh = np.meshgrid(Vt_Range, Vb_Range)
    

def veryfirst():
    print("Starting the measurement")

def init_awg(awg, awg_settings):
    # vs_scale = 10**(-stngs['sample_atten']/20.0) * 250.0
    # refsc = 10**(-stngs['ref_atten']/20.0) * 250.0
    # ac_scale = (refsc / vs_scale)/float(stngs['chY1'])
    print("Initialize AWG")
    ac_scale = 10**(-(awg_settings['ref_atten'] - awg_settings['sample_atten'])/20.0)/float(awg_settings['ch1_v']/3.0)
    
    awg.channel1.enabled(True)
    awg.channel2.enabled(True)

    awg.channel1.frequency(awg_settings['frequency'])
    awg.channel2.frequency(awg_settings['frequency'])

    awg.channel1.phase(0)
    awg.channel2.phase(0)

    awg.channel1.amplitude(awg_settings['ch1_v'])
    awg.channel2.amplitude(awg_settings['ch2_v'])

    sleep(1)

def set_initial_conditions():
    print("Ramping to intial conditions")
    da.DAC0.volt(Vt_0)
    da.DAC1.volt(Vb_0)
    da.DAC3.volt(Vg_0)
    sleep(1.5)

def thelast():
    print("Ramping down")
    da.DAC0.volt(0)
    da.DAC1.volt(0)
    da.DAC3.volt(0)

initialise_or_create_database_at(Path.cwd() / "transistor_meas.db")
exp = load_or_create_experiment(
    experiment_name="test",
    sample_name="test",
)

meas = Measurement(exp=exp, station=station, name="Transistor characterization")

meas.register_parameter(da.DAC0.volt)
meas.register_parameter(da.DAC1.volt)
meas.register_parameter(da.DAC3.volt)
meas.register_parameter(da.ADC0.volt,setpoints=(da.DAC3.volt,))
meas.register_parameter(lia1.X, setpoints=(da.DAC3.volt,))  # now register the dependent oone
meas.register_parameter(lia1.Y, setpoints=(da.DAC3.volt,))  # now register the dependent oone
meas.register_parameter(lia1.R, setpoints=(da.DAC3.volt,))  # now register the dependent oone
meas.register_parameter(lia1.P, setpoints=(da.DAC3.volt,))  # now register the dependent oone


meas.add_before_run(veryfirst, ())  # add a set-up action
meas.add_before_run(init_awg, (awg, awg_settings))  # add a set-up action
meas.add_before_run(set_initial_conditions, ())  # add a set-up action
meas.add_after_run(thelast, ())  # add a tear-down action

meas.write_period = 0.1

with meas.run() as datasaver:
    #lia1.autorange(8)

    lia1.sensitivity(float(lockin_settings['sensitivity']))
    lia1.time_constant(float(lockin_settings['tc']))

    for j in tqdm.tqdm(range(Vg_npts)):
        v = Vg_Range[j]
        da.DAC3.volt(v)
        sleep(1.5)
        zx = lia1.X()
        zy = lia1.Y()
        zr = lia1.R()
        zp = lia1.P()
        z_dc = da.ADC0.volt()

        datasaver.add_result((da.DAC3.volt, v), (lia1.X, zx), (lia1.Y, zy), (lia1.R, zr), (lia1.P, zp), (da.ADC0.volt, z_dc))
    dataset2D = datasaver.dataset

ax, cbax = plot_dataset(dataset2D)