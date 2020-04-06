import numpy as np


class MotionHistoryImage:
    def __init__(self, tau, frame):
        self.tau = tau
        h, w = frame.shape
        self.MT = np.zeros((self.tau, h, w))

    def update_m_at_t(self, Bt, t):
        self.MT[t, (Bt == 1)] = self.tau
        bt_zero_idx = np.argwhere(Bt == 0)
        for idx in bt_zero_idx:
            new_value = self.MT[t-1, idx[0], idx[1]] - 1 if self.MT[t-1, idx[0], idx[1]] - 1 > 0 else 0
            self.MT[t, idx[0], idx[1]] = new_value


class MotionEnergyImage:
    def __init__(self, frame):
        h, w = frame.shape
        self.M = np.zeros((h, w))

    def update_m(self, Bt):
        self.M[(Bt == 1)] = 1

