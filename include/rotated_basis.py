from qcodes.parameters import Parameter

class RotatedBasis:
    def __init__(self, dac0, dac1, transform, delta, v_fixed):
        """
        dac0, dac1 : QCoDeS DAC parameters (e.g. da.DAC0.volt, da.DAC1.volt)
        transform  : function f(vlin1, vlin2) -> (v0, v1)
        """
        self.dac0 = dac0
        self.dac1 = dac1
        self.transform = self._transform_select(transform)
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
        v0, v1 = self.transform(self._p, self._n)
        self.dac0.volt(v0)
        self.dac1.volt(v1)

    def _set_n(self, val):
        self._n = val
        self._update_dacs()

    def _set_p(self, val):
        self._p = val
        self._update_dacs()

    def _vb_fixed(self, p0, n0):
        """
        :param p0: polarizing field
        :param n0: charge carrier density
        :param delta: capacitor asymmetry
        :param vb: fixed voltage set on the bottom gate
        :return: (v_top, v_sample)
        """
        return self.v_fixed - (n0 * self.delta - p0) / (1.0 - self.delta ** 2), self.v_fixed - 0.5 * (n0 - p0) / (1.0 - self.delta)

    def _vt_fixed(self, p0, n0):
        """
        :param p0: polarizing field
        :param n0: charge carrier density
        :param delta: capacitor asymmetry
        :param vt: fixed voltage set on the top gate
        :return: (v_bot, v_sample)
        """
        return (n0 * self.delta - p0) / (1.0 - self.delta ** 2) + self.v_fixed, self.v_fixed - 0.5 * (n0 + p0) / (1.0 + self.delta)

    def _vs_fixed(self, p0, n0):
        """
        :param p0: polarizing field
        :param n0: charge carrier density
        :param delta: capacitor asymmetry
        :param vs: fixed voltage set on graphene sample
        :return: (v_top, v_bottom)
        """
        return self.v_fixed + 0.5 * (n0 + p0) / (1.0 + self.delta), self.v_fixed + 0.5 * (n0 - p0) / (1.0 - self.delta)

    def _transform_select(self, s):
        """
        :param s: ('vb', 'vt', 'vs') selection based on which parameter is fixed
        :return: function f
        """
        if s == 'vb':
            f = self._vb_fixed

        elif s == 'vt':
            f = self._vt_fixed
        elif s == 'vs':
            f = self._vs_fixed
        return f