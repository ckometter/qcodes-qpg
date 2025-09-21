from pathlib import Path
from time import sleep
import tqdm
import yaml
with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
meas_settings = cfg[cfg['measurement']]
balancing_settings = cfg['balancing_settings']
lockin_settings = cfg['lockin_settings']
meas_parameters = cfg['meas_parameters']

import numpy as np

from include.capacitance_bridge import CapacitanceBridgeSR830Lockin as bridge

import qcodes as qc
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
# lia2 = station.load_instrument('lia2')
# dmm1 = station.load_instrument('dmm1')
# hp = station.load_instrument('hp')

s1 = np.array((0.5, 0.5)).reshape(2, 1)
s2 = np.array((-0.5, -0.5)).reshape(2, 1)
def init_bridge(lck, acbox, cfg):
    stngs = cfg['balancing_settings']
    ref_ch = stngs['ref_ch']
    cb = bridge(lck=lck, acbox=acbox, time_const=stngs['balance_tc'],
                           iterations=stngs['iter'], tolerance=stngs['tolerance'],
                           s_in1=s1, s_in2=s2, excitation_channel=ref_ch)

    return cb

cb = init_bridge(lia1, awg, cfg)
v_fixed = -0.46
def balance(cb, lck):
    ac_scale = 10**(-(stngs['ref_atten'] - stngs['sample_atten'])/20.0)/float(stngs['chY1']/3.0)
    v1_balance, v2_balance = function_select(meas_settings['fixed'])(balancing_settings['p0'], balancing_settings['n0'], meas_parameters['delta_var'], v_fixed)

    lck.time_constant(balancing_settings['balance_tc'])

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

def veryfirst():
    print("Starting the measurement")

def init_awg(acbox, stngs):
    # vs_scale = 10**(-stngs['sample_atten']/20.0) * 250.0
    # refsc = 10**(-stngs['ref_atten']/20.0) * 250.0
    # ac_scale = (refsc / vs_scale)/float(stngs['chY1'])
    ac_scale = 10**(-(stngs['ref_atten'] - stngs['sample_atten'])/20.0)/float(stngs['chY1']/3.0)
    awg.channel1
    acbox.set_voltage("X1", stngs['chX1'])
    acbox.set_voltage("X2", stngs['chX1'])
    acbox.set_voltage("Y1", stngs['chY1'])
    acbox.set_voltage("Y2", stngs['chY2'])
    acbox.set_frequency(stngs['frequency'])
    sleep(1)

def set_initial_conditions():
    print("Ramping to intial conditions")

def thelast():
    print("Ramping down")

initialise_or_create_database_at(Path.cwd() / "capacitance_meas.db")
exp = load_or_create_experiment(
    experiment_name="test",
    sample_name="test",
)

meas = Measurement(exp=exp, station=station, name="Capacitance measurement")

meas.add_before_run(veryfirst, ())  # add a set-up action
meas.add_before_run(set_initial_conditions, ())  # add a set-up action
meas.add_after_run(thelast, ())  # add a tear-down action

meas.write_period = 0.1

