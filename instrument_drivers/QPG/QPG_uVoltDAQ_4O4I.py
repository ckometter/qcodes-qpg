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

class QPGUVoltDAQOutputChannel(InstrumentChannel):
    """
    Class for .
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        channel: str,
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

        if channel not in range(4):
            raise ValueError('channel must be either "0", "1", "2" or "3"')

        super().__init__(parent, name, **kwargs)
        self.channel = channel
        self.voltage_output_range = 10
        self.channel_type = type

        self.volt = self.add_parameter(
            "volt",
            get_cmd=f"GET_DAC,{self.channel}",
            get_parser=float,
            set_cmd=self._set_volt,
            # note that the set_cmd is either the following format string
            #'smua.source.levelv={:.12f}' or 'smub.source.levelv={:.12f}'
            # depending on the value of `channel`
            label="Voltage",
            unit="V",
        )
        """Parameter volt"""

    def _set_volt(self, v):
        _ = self.ask(f"SET,{self.channel},{v}")   # read & ignore/validate reply

class QPGUVoltDAQInputChannel(InstrumentChannel):
    """
    Class for .
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        channel: str,
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

        if channel not in range(4):
            raise ValueError('channel must be either "0", "1", "2" or "3"')

        super().__init__(parent, name, **kwargs)
        self.channel = channel
        self.voltage_output_range = 10
        self.channel_type = type

        self.volt = self.add_parameter(
            "volt",
            get_cmd=f"GET_ADC,{channel}",
            get_parser=float,
            # note that the set_cmd is either the following format string
            #'smua.source.levelv={:.12f}' or 'smub.source.levelv={:.12f}'
            # depending on the value of `channel`
            label="Voltage",
            unit="V",
        )
        """Parameter volt"""

class QPGUVoltDAQ4O4I(VisaInstrument):
    """
    This is the qcodes driver for the dac-adc,
    tested with dac-adc
    """

    default_terminator = "\r"

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

        self.ask(f"INITIALIZE")

        # Add the channel to the instrument
        for ch in range(4):
            ch_name = f"DAC{ch}"
            channel = QPGUVoltDAQOutputChannel(self, ch_name, ch)
            self.add_submodule(ch_name, channel)

        for ch in range(4):
            ch_name = f"ADC{ch}"
            channel = QPGUVoltDAQInputChannel(self, ch_name, ch)
            self.add_submodule(ch_name, channel)

        self.connect_message()