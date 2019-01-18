from __future__ import print_function, division


class Solution(object):
    _x_scale=[1e-2, 1e-2, 1e-5, 1e-5, 1e-9, 1e-9, 1e-9]

    def __init__(self):
        self.params = [0] * 7
        self.failed = False

    @property
    def Ax(self):
        return self.params[0]

    @property
    def Ay(self):
        return self.params[1]

    def get_params(self, order):
        if order == 0:
            return self.params[:2]
        elif order == 1:
            return self.params[:4]
        elif order == 2:
            return self.params[:7]

        raise Exception("Unknown solution order requested")

    def set_params(self, params):
        self.params[:len(params)] = params

    def x_scale(self, order):
        if order == 0:
            return self._x_scale[:2]
        elif order == 1:
            return self._x_scale[:4]
        elif order == 2:
            return self._x_scale[:7]

        raise Exception("Unknown solution order requested")

    def phases(self, U, V):
        x, y, xx, xy, yy = self.params[2:]
        return (
            x * U +
            y * V +
            xx * U**2 +
            xy * U * V +
            yy * V**2
        )
