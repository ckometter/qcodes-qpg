from typing import TYPE_CHECKING

from qcodes.instrument import (
    ChannelList,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
)
from qcodes.validators import Numbers

if TYPE_CHECKING:
    from typing import Unpack

class DacChannel(InstrumentChannel):
    """
    One DAC output channel.
    """

    def __init__(
        self,
        parent: VisaInstrument,
        name: str,
        channel: int,
        vmin: float = -10.0,
        vmax: float = 10.0,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The 'colloquial' name of the channel
            channel: Channel number
            **kwargs: Forwarded to base class.

        """

        super().__init__(parent, name, **kwargs)

        self._ch = channel

        self.add_parameter(
            "voltage",
            get_cmd=f"GET_DAC,{self._ch}",
            get_parser=float,
            set_cmd=self._set_voltage,
            label=f"DAC {self._ch} voltage",
            unit="V",
            vals=Numbers(vmin, vmax)
        )

    def _set_voltage(self, v: float) -> None:
        inst = self.root_instrument
        inst.ask(f"SET,{self._ch},{v:.9e}")

class AdcChannel(InstrumentChannel):
    """
    One ADC input channel.
    """

    def __init__(
        self,
        parent: VisaInstrument,
        name: str,
        channel: int,
        vmin: float = -10.0,
        vmax: float = 10.0,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The 'colloquial' name of the channel
            channel: Channel number
            **kwargs: Forwarded to base class.

        """

        super().__init__(parent, name, **kwargs)
        self._ch = channel

        self.add_parameter(
            "voltage",
            get_cmd=f"GET_ADC,{self._ch}",
            get_parser=float,
            label=f"ADC {self._ch} voltage",
            unit="V",
        )

class QPGUVoltDAQ4O4I(VisaInstrument):
    """
    This is the qcodes driver for the uVoltDAQ 4O4I.
    """
    default_terminator = "\r\n"

    def __init__(
        self,
        name: str,
        address: str,
        timeout: float = 5.0,
        *,
        initialize: bool = False,
        vmin_dac: float = -10.0,
        vmax_dac: float = 10.0,
        vmin_adc: float = -10.0,
        vmax_adc: float = 10.0,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
            address: VISA ressource address
            **kwargs: kwargs are forwarded to the base class.

        """
        super().__init__(name, address, timeout=timeout, **kwargs)
        if initialize:
            self.ask("INITIALIZE")

        self.add_parameter(
            "idn",
            get_cmd="*IDN?",
        )

        # Create DAC channels
        dacs = ChannelList(self, "dacs", DacChannel, snapshotable=True)
        for ch in range(4):
            ch_name = f"dac{ch}"
            dac = DacChannel(self, name=ch_name, channel=ch, vmin=vmin_dac, vmax=vmax_dac)
            dacs.append(dac)
            self.add_submodule(ch_name, dac)
        self.add_submodule("dacs", dacs)
        
        # Create ADC channels
        adcs = ChannelList(self, "adcs", AdcChannel, snapshotable=True)
        for ch in range(4):
            ch_name = f"adc{ch}"
            adc = AdcChannel(self, name=ch_name, channel=ch, vmin=vmin_adc, vmax=vmax_adc)
            adcs.append(adc)
            self.add_submodule(ch_name, adc)
        self.add_submodule("adcs", adcs)

        self.connect_message()
