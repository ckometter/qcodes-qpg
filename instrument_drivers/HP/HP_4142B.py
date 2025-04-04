from typing import TYPE_CHECKING

from functools import partial

import numpy as np

from qcodes.instrument import (
    ChannelList,
    Instrument,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.validators import Numbers
from qcodes import validators as vals

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter

from qcodes.parameters import (
    ArrayParameter,
    Parameter,
    ParameterWithSetpoints,
    ParamRawDataType,
    create_on_off_val_mapping,
)

class HP4142BChannel(InstrumentChannel):
    """
    Class to hold the two Keithley channels, i.e.
    SMU1 and SMU4.
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        channel: str,
        type: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The 'colloquial' name of the channel
            channel: The name used by the HP4142B, i.e. either
                'smu1' or 'smu2'
            **kwargs: Forwarded to base class.

        """

        if channel not in ["1", "4"]:
            raise ValueError('channel must be either "1" or "4"')
        
        if type not in ["SMU_A", "SMU_B"]:
            raise ValueError('type must be either "SMU_A" or "SMU_B"')

        super().__init__(parent, name, **kwargs)
        self.channel = channel
        self.voltage_output_range = 100
        self.current_compliance = 1e-6
        self.channel_type = type
        self.write(f"CN {channel}")

        self.volt = self.add_parameter(
            "volt",
            get_cmd=f"TV{channel}",
            get_parser=self._convert_from_ascii,
            set_cmd=f"DV{channel},{self._voltage_to_range_setting(self.voltage_output_range)},{{}},{str(self.current_compliance)},0",
            # note that the set_cmd is either the following format string
            #'smua.source.levelv={:.12f}' or 'smub.source.levelv={:.12f}'
            # depending on the value of `channel`
            label="Voltage",
            unit="V",
        )
        """Parameter volt"""

        self.curr = self.add_parameter(
            "curr",
            get_cmd=f"TI{channel}",
            get_parser=self._convert_from_ascii,
            # note that the set_cmd is either the following format string
            #'smua.source.levelv={:.12f}' or 'smub.source.levelv={:.12f}'
            # depending on the value of `channel`
            label="Current",
            unit="A",
        )
        """Parameter curr"""

        self.turnOn = self.add_parameter(
            "turnOn",
            get_cmd=f"*LRN? {channel}",
            get_parser=str,
            set_cmd=f"CN {channel}",
            # note that the set_cmd is either the following format string
            #'smua.source.levelv={:.12f}' or 'smub.source.levelv={:.12f}'
            # depending on the value of `channel`
            label="Channel Turn On",
        )
        """Parameter on"""

    def _voltage_to_range_setting(self, voltage='Auto'):

        voltage_range_setting = {'Auto':'0','0.2':'10','2':'11','20':'12','40':'13','100':'14','200':'15','500':'16','1000':'17'}
        if voltage == 'Auto':
            return '0'
        voltage_keys_dict = {
            'SMU_A': np.array([2,20,40,100,200]),
            'SMU_B': np.array([2,20,40,100]),
            'VS': np.array([2,20,40]),
            'VM': np.array([0.2,2,20,40])
            }
        voltage_keys = voltage_keys_dict[self.channel_type]
        vk = voltage_keys - abs(float(voltage))
        #vk.clip(0)
        idx = np.where(vk>=0)[0]
        if len(idx) == 0:
            # BEFORE: if we set a source to a too high voltage limit,
            # it just goes to auto range! This is not the behaviour that we want,
            # rather for safety it should go to the highest range or throw an error
            print('invalid voltage range! setting highest range')
            idx = [len(voltage_keys)-1]
        return voltage_range_setting[str(voltage_keys[idx[0]])]
    
    def _convert_from_ascii(self,ascii_data):
        #ASCII as defined below is assumed, i.e. with header: self.set_ascii_output()
        if len(ascii_data[:-1])%15 != 0:
            return([False, False, 0, 10, float('NaN'), 0, "unexpected ASCII data format"])

        ascii_error = ascii_data[0]  #status in HP docu, indicates status of measurement
        if ascii_error == 'N': #N - no error
            status = 'N: normal data, no error'
            error_code = 0
            is_measurement_data = True
        elif ascii_error == 'T': #T
            status = 'T: another channel reached compliance limit'
            error_code = 1
            is_measurement_data = True
        elif ascii_error == 'C': #C
            status = 'C: this channel reached compliance limit'
            error_code = 2
            is_measurement_data = True
        elif ascii_error == 'V': #V
            status = 'V: overflow'
            error_code = 3
            is_measurement_data = True
        elif ascii_error == 'X': #X
            status = 'X: SMU/HVU oscillating'
            error_code = 4
            is_measurement_data = True
        elif ascii_error == 'F': #F
            status = 'F: HVU output not settled'
            error_code = 5
            is_measurement_data = True
        elif ascii_error == 'G': #G
            status = 'G: check manual'
            error_code = 6
            is_measurement_data = True
        elif ascii_error == 'S': #S
            status = 'S: check manual'
            error_code = 7
            is_measurement_data = True
        elif ascii_error == 'W': #W
            status = 'W: sweep source - first or intermediate sweep step'
            error_code = 1
            is_measurement_data = False
        elif ascii_error == 'E': #E
            status = 'E: sweep source - final sweep step'
            error_code = 2
            is_measurement_data = False
                        
        channel = ascii_data[1]
        if channel == 'A':
            channel_number = 1
        elif channel == 'B':
            channel_number = 2
        elif channel == 'C':
            channel_number = 3
        elif channel == 'D':
            channel_number = 4
        elif channel == 'E':
            channel_number = 5
        elif channel == 'F':
            channel_number = 6
        elif channel == 'G':
            channel_number = 7
        elif channel == 'H':
            channel_number = 8
        elif channel == 'I':
            channel_number = 11
        elif channel == 'J':
            channel_number = 12
        elif channel == 'K':
            channel_number = 13
        elif channel == 'L':
            channel_number = 14
        elif channel == 'M':
            channel_number = 15
        elif channel == 'N':
            channel_number = 16
        elif channel == 'O':
            channel_number = 17
        elif channel == 'P':
            channel_number = 18
        elif channel == 'Q':
            channel_number = 21
        elif channel == 'R':
            channel_number = 22
        elif channel == 'S':
            channel_number = 23
        elif channel == 'T':
            channel_number = 24
        elif channel == 'U':
            channel_number = 25
        elif channel == 'V':
            channel_number = 26
        elif channel == 'W':
            channel_number = 27
        elif channel == 'X':
            channel_number = 28
        
        if ascii_data[2] == 'V':
            is_voltage_data = True #voltage data
        else: # 'I'
            is_voltage_data = False #current data
            
        value = ascii_data[3:len(ascii_data)]
        reading = float(value)
        
        #if not error_code==0: # debug
            #print('_convert_from_asci: %s'%status)
        #same output as with convert_from_binary but range_setting = 0 as it is not given in the ASCII output
        return(reading)

class HP4142B(VisaInstrument):
    """
    This is the qcodes driver for the Keithley2600 Source-Meter series,
    tested with Keithley2614B
    """

    default_terminator = "\n"

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
            address: VISA ressource address
            **kwargs: kwargs are forwarded to the base class.

        """
        super().__init__(name, address, **kwargs)

        # Add the channel to the instrument
        for ch in [("1","SMU_A"), ("4", "SMU_B")]:
            ch_name = f"smu{ch[0]}"
            channel = HP4142BChannel(self, ch_name, ch[0], ch[1])
            self.add_submodule(ch_name, channel)

        self.connect_message()