from time import time
import numpy as np
import math
from tqdm import tqdm
from termcolor import colored

class SmallEnoughException(Exception):
    pass


class CheckConvCriterion:
    def __init__(self, phi, tol, max_wait=500):
        self.phi = phi
        self.tol = tol
        self.result = None
        self.next_milestone = None
        self.counter = 0
        self.err = None
        self.gerr = None
        self.tic = time()

        self.max_wait, self.wait = max_wait, 0

        self.logs_scat = []
        self.logs_harm = []
        self.logs_loss = []
        self.logs_grad = []
        self.logs_x = []

    def __call__(self, xk):
        # err, grad_xk = self.phi.joint(xk)
        err, grad_xk = self.phi.res
        gerr = np.linalg.norm(grad_xk, ord=float('inf'))
        err, gerr = float(err), float(gerr)
        self.err = err
        self.gerr = gerr
        self.counter += 1

        loss_scat, loss_harm = self.phi.loss_logs
        loss_scat, loss_harm = float(loss_scat), float(loss_harm)
        self.logs_scat.append(loss_scat)
        self.logs_harm.append(loss_harm)
        self.logs_loss.append(err)
        self.logs_grad.append(gerr)

        if self.next_milestone is None:
            self.next_milestone = 10 ** (np.floor(np.log10(gerr)))

        info_already_printed_p = False
        if not math.log2(self.counter) % 1:
            self.logs_x.append(xk)
            self.print_info_line('SAVED X')
            info_already_printed_p = True

        if err <= self.tol:
            self.result = xk
            raise SmallEnoughException()
        elif gerr <= self.next_milestone or self.wait >= self.max_wait:
            if not info_already_printed_p:
                self.print_info_line()
                info_already_printed_p = True
            if gerr <= self.next_milestone:
                self.next_milestone /= 10
            self.wait = 0
        else:
            self.wait += 1

    def print_info_line(self, msg=''):
        delta_t = time() - self.tic
        tqdm.write(colored(
            "{:6}it in {} ( {:.2f}it/s )".format(
                self.counter, self.hms_string(delta_t), self.counter / delta_t) \
            + " ........ " \
            + "{:.2E} (S: {:.2E}  H: {:.2E}) -- {:.2E}".format(
                self.err, self.logs_scat[-1], self.logs_harm[-1], self.gerr) \
            + "  " \
            + msg,
            'blue'))

    @staticmethod
    def hms_string(sec_elapsed):
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60.
        return "{}:{:>02}:{:>05.2f}".format(h, m, s)
