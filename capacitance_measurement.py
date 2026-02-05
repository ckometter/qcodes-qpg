from pathlib import Path
from time import sleep
import tqdm
import yaml
with open("capacitance_measurement.yml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    
meas_type = cfg["measurement"]
meas_settings = cfg[meas_type]
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
Vt_f = 2
Vb_0 = -2
Vb_f = 2
Vt_npts = 10
Vb_npts = 10
Vcg_val = 0

Vt_Range = np.linspace(Vt_0,Vt_f,Vt_npts)
Vb_Range = np.linspace(Vb_0,Vb_f,Vb_npts)
    
Vt_mesh, Vb_mesh = np.meshgrid(Vt_Range, Vb_Range)
    
mask = np.abs(Vt_mesh-(Vcg_val)) <= 100
    
Vt_masked = np.where(mask, Vt_mesh, np.nan)
Vb_masked = np.where(mask, Vb_mesh, np.nan)

#s1 = np.array((0.5, 0.5)).reshape(2, 1)
#s2 = np.array((-0.5, -0.5)).reshape(2, 1)

s1 = np.array((0.1, 0.1)).reshape(2, 1)
s2 = np.array((-0.1, -0.1)).reshape(2, 1)
def init_bridge(lck, acbox, cfg):
    stngs = cfg['balancing_settings']
    ref_ch = stngs['ref_ch']
    cb = bridge(lck=lck, acbox=acbox, time_const=float(stngs['balance_tc']),
                           iterations=stngs['iter'], tolerance=stngs['tolerance'],
                           s_in1=s1, s_in2=s2)

    return cb

def balance(cb, lck):
    ac_scale = 10**(-(awg_settings['ref_atten'] - awg_settings['sample_atten'])/20.0)/float(awg_settings['ch1_v'])
    v1_balance, v2_balance = function_select(meas_settings['fixed'])(balancing_settings['p0'], balancing_settings['n0'], meas_parameters['delta_var'], v_fixed)

    lck.time_constant(float(balancing_settings['balance_tc']))

    print(cb.balance())
    cs, ds = cb.capacitance(ac_scale)
    print("Via balance: Cs = {}, Ds = {}".format(cs, ds))
    c_, d_ = cb.offBalance(ac_scale)
    print("Scaling factors for offset: Ctg {} and Dtg {}".format(c_, d_))

    vb = cb.vb # this is the balance point. this is a vector with (x,y) -> amplitude sqrt(x**2+y**2), phase (atan y/x)
    magnitude = np.sqrt(vb[0]**2 + vb[1]**2)
    phase = np.degrees(np.arctan2(vb[1], vb[0])) * (-1.0)
    if phase < 0: phase = 360 + phase

    capacitance_params = {'Capacitance': cs, 'Dissipation': ds,
                          'offbalance c_': c_, 'offbalance d_': d_}
    
    return capacitance_params

def vb_fixed(p0, n0, delta, vb):
    """
    :param p0: polarizing field
    :param n0: charge carrier density
    :param delta: capacitor asymmetry
    :param vb: fixed voltage set on the bottom gate
    :return: (v_top, v_sample)
    """
    return vb - (n0 * delta - p0) / (1.0 - delta ** 2), vb - 0.5 * (n0 - p0) / (1.0 - delta)

def vt_fixed(p0, n0, delta, vt):
    """
    :param p0: polarizing field
    :param n0: charge carrier density
    :param delta: capacitor asymmetry
    :param vt: fixed voltage set on the top gate
    :return: (v_bot, v_sample)
    """
    return (n0 * delta - p0) / (1.0 - delta ** 2) + vt, vt - 0.5 * (n0 + p0) / (1.0 + delta)

def vs_fixed(p0, n0, delta, vs):
    """
    :param p0: polarizing field
    :param n0: charge carrier density
    :param delta: capacitor asymmetry
    :param vs: fixed voltage set on graphene sample
    :return: (v_top, v_bottom)
    """
    return vs + 0.5 * (n0 + p0) / (1.0 + delta), vs + 0.5 * (n0 - p0) / (1.0 - delta)

def function_select(s):
    """
    :param s: ('vb', 'vt', 'vs') selection based on which parameter is fixed
    :return: function f
    """
    if s == 'vb':
        f = vb_fixed

    elif s == 'vt':
        f = vt_fixed
    elif s == 'vs':
        f = vs_fixed
    return f

n_0 = -1
n_f = 1
p_0 = -1
p_f = 1
n_npts = 100
p_npts = 100
v_fixed = -0.415
delta = 0

n_Range = np.linspace(n_0,n_f,n_npts)
p_Range = np.linspace(p_0,p_f,p_npts)

voltage_calculator = function_select(meas_settings["fixed"])
v_fast, v_slow = voltage_calculator(n_Range, p_Range, delta, v_fixed)

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

initialise_or_create_database_at(Path.cwd() / "capacitance_meas.db")
exp = load_or_create_experiment(
    experiment_name="test",
    sample_name="test",
)

meas = Measurement(exp=exp, station=station, name="Capacitance measurement")

class RotatedBasis:
    def __init__(self, dac0, dac1, transform, delta, v_fixed):
        """
        dac0, dac1 : QCoDeS DAC parameters (e.g. da.DAC0.volt, da.DAC1.volt)
        transform  : function f(vlin1, vlin2) -> (v0, v1)
        """
        self.dac0 = dac0
        self.dac1 = dac1
        self.transform = transform
        self.delta = delta
        self.v_fixed = v_fixed
        self._n = 0
        self._p = 0

        self.n = Parameter(
            "n", unit="V", label="n",
            set_cmd=self._set_n, get_cmd=lambda: self._n
        )

        self.p = Parameter(
            "p", unit="V", label="p",
            set_cmd=self._set_p, get_cmd=lambda: self._p
        )

    def _update_dacs(self):
        v0, v1 = self.transform(self._p, self._n, self.delta, self.v_fixed)
        self.dac0.volt(v0)
        self.dac1.volt(v1)

    def _set_n(self, val):
        self._n = val
        self._update_dacs()

    def _set_p(self, val):
        self._p = val
        self._update_dacs()

rot = RotatedBasis(da.DAC0, da.DAC1, voltage_calculator, delta, v_fixed)

Cap = Parameter(
    name="Cap",
    label="Capacitance",
    unit="F",            # or '' for dimensionless ratio
    get_cmd=None         # it's not read from hardware; we pass values manually
)

Dis = Parameter(
    name="Dis",
    label="Dissipation",
    unit="Ohm",            # or '' for dimensionless ratio
    get_cmd=None         # it's not read from hardware; we pass values manually
)

n = rot.n
p = rot.p

meas.register_parameter(n)
meas.register_parameter(p)
meas.register_parameter(da.DAC0.volt,setpoints=(n,p))
meas.register_parameter(da.DAC1.volt,setpoints=(n,p))
meas.register_parameter(da.ADC0.volt,setpoints=(n,p))
meas.register_parameter(lia1.X, setpoints=(n,p))  # now register the dependent oone
meas.register_parameter(lia1.Y, setpoints=(n,p))  # now register the dependent oone
meas.register_parameter(Cap, setpoints=(n,p))  # now register the dependent oone
meas.register_parameter(Dis, setpoints=(n,p))

# meas.register_parameter(da.DAC0.volt)
# meas.register_parameter(da.DAC1.volt)
# #meas.register_parameter(lia1.X, setpoints=(da.DAC0.volt,da.DAC1.volt))  # now register the dependent oone
# #meas.register_parameter(lia1.Y, setpoints=(da.DAC0.volt,da.DAC1.volt))  # now register the dependent oone
# meas.register_parameter(Cap, setpoints=(da.DAC0.volt,da.DAC1.volt))  # now register the dependent oone
# meas.register_parameter(Dis, setpoints=(da.DAC0.volt,da.DAC1.volt))  # now register the dependent oone

def set_initial_conditions():
    print("Ramping to intial conditions")
    n(-1)
    p(-1)
    da.DAC3.volt(v_fixed)
    sleep(1.5)

def thelast():
    print("Ramping down")
    da.DAC0.volt(0)
    da.DAC1.volt(0)
    da.DAC3.volt(0)

meas.add_before_run(veryfirst, ())  # add a set-up action
meas.add_before_run(init_awg, (awg, awg_settings))  # add a set-up action
meas.add_before_run(set_initial_conditions, ())  # add a set-up action
meas.add_after_run(thelast, ())  # add a tear-down action

meas.write_period = 0.1

with meas.run() as datasaver:
    lia1.autorange(8)

    cb = init_bridge(lia1, awg, cfg)
    capacitance_params = balance(cb, lia1)
    cs = capacitance_params['Capacitance']
    ds = capacitance_params['Dissipation']
    c_ = capacitance_params['offbalance c_']
    d_ = capacitance_params['offbalance d_']

    lia1.sensitivity(float(lockin_settings['sensitivity']))
    lia1.time_constant(float(lockin_settings['tc']))

    for j in tqdm.tqdm(range(p_npts)):
        p_j = p_Range[j]
        sleep(3)
        for i in range(n_npts):
            n_i = n_Range[i]
            n(n_i)
            p(p_j)
            v1 = da.DAC0.volt.cache.get()
            v2 = da.DAC1.volt.cache.get()
            # v1 = Vt_masked[i,j]
            # v2 = Vb_masked[i,j]
            # if np.isnan(v1) or np.isnan(v2):
            #     continue
            sleep(0.5)
            #capacitance_params = balance(cb, lia1)
            #d_cap = capacitance_params['Capacitance']
            #d_dis = capacitance_params['Dissipation']
            zx = lia1.X()
            zy = lia1.Y()
            z_dc = da.ADC0.volt()

            d_cap = (c_ * zx + d_ * zy) + cs
            d_dis = (d_ * zx - c_ * zy) + ds

            #datasaver.add_result((da.DAC0.volt, v2), (da.DAC1.volt, v1), (lia1.X, zx), (lia1.Y, zy), (Cap, d_cap), (Dis, d_dis))
            datasaver.add_result((n, n_i), (p, p_j), (da.DAC0.volt, v1), (da.DAC1.volt, v2), (da.ADC0.volt, z_dc), (lia1.X, zx), (lia1.Y, zy),(Cap, d_cap), (Dis, d_dis))
    dataset2D = datasaver.dataset

ax, cbax = plot_dataset(dataset2D)