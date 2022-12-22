import sys
if __name__ == "__main__":
    sys.path.append("../pyscatwave")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.nn import Parameter
import torch.optim as optim
import loss as lF
from tqdm import tqdm
from utils import NanError
from numba import jit
from time import time
from global_const import Tensor
import math
from termcolor import colored

class OptimAbstract(nn.Module):
    def __init__(self, loss_type, detail=False):
        super(OptimAbstract, self).__init__()
        self.is_cuda = False
        self.register_parameter('x', None)
        self.detail = detail

        if callable(loss_type):
            self.criterion = loss_type
        elif loss_type == "MSE":
            self.criterion = lF.mse_loss
        elif loss_type == "L1":
            self.criterion = lF.l1_loss
        elif loss_type == "LogMSE":
            self.criterion = lF.LogMSELoss(1e-4)
        elif loss_type == "RelativeMSE":
            self.criterion = lF.RelativeMSELoss(1e-4)
        else:
            raise ValueError("Unknown loss type: {}".format(loss_type))

    def cpu(self):
        self.is_cuda = False
        return self

    def cuda(self):
        self.is_cuda = True
        return self

    def rand_init_value(self, x0):
        x = torch.randn(*x0.size(), dtype=Tensor.dtype)
        x.requires_grad = x0.requires_grad
        if self.is_cuda:
            x = x.cuda()
        return x

    def init_past_x(self, x0, past_x):
        if past_x is None:
            x = self.rand_init_value(x0)  # hack Pytorch autograd optimizer
            self.x = Parameter(x)
        else:
            print()
            x = past_x

            if self.is_cuda:
                x = x.cuda()
            else:
                x = x.cpu()

            if self.x is None:
                self.x = Parameter(x)
            else:
                self.x.data = x


class GradStepOptimizerFst(OptimAbstract):
    def opt_fun(self, **kwargs):
        raise NotImplementedError

    def __call__(self, x0, metric, niter=1000, print_freq=500,
                 past_x=None, past_logs=None, **opt_kwargs):

        self.init_past_x(x0, past_x)

        optimizer = self.opt_fun(self.parameters(), **opt_kwargs)

        # complex x0
        zero = torch.zeros_like(x0)
        x0 = torch.stack((x0, zero), dim=-1)
        x0 = Variable(x0)

        # initialize embeding of true signal and different loss to compare running loss
        phi_x0 = metric(x0).detach()
        loss0 = self.criterion(torch.zeros_like(phi_x0), phi_x0)

        x_save = self.x.data.clone()  # safe state in case NaNs appear

        # initialize logs and return value
        if past_logs is None:
            logs, logs_detail = [], []
        elif self.detail:
            logs, logs_detail = past_logs
        else:
            logs, = past_logs

        # iter
        past_epoch = len(logs)
        for n in tqdm(range(niter)):
            # complex vector
            x = torch.stack((self.x, torch.zeros_like(self.x)), dim=-1)

            # set gradients to 0
            optimizer.zero_grad()

            # compute embedding
            phi_x = metric(x)

            # compute loss
            loss = self.criterion(phi_x, phi_x0)
            loss = loss / loss0  # relative loss

            # check for NaNs
            if np.any(np.isnan(loss.data.cpu().numpy())):
                print("NaN found at iteration {}".format(n + 1))
                break
            try:
                loss.backward()
            except NanError as e:
                print(e)
                print("NaN found at iteration {}".format(n + 1))
                break

            # backup
            x_save = self.x.data.clone()

            # gradient descent step
            optimizer.step()

            # add detailed loss to log file
            loss_scalar = loss.data.cpu().numpy()[0]
            logs.append(loss_scalar)
            if self.detail:
                loss_detail = self.criterion(phi_x, phi_x0, detail=True) / loss0
                logs_detail.append(loss_detail.data.cpu().numpy()[0])

            # print loss
            if (n + past_epoch + 1) % print_freq == 0:
                tqdm.write("epoch {}: {}".format(n + past_epoch + 1, loss_scalar))

        logs = (logs, logs_detail) if self.detail else (logs,)

        return x_save, logs


class GradStepOptimizerScd(OptimAbstract):
    def opt_fun(self, **kwargs):
        raise NotImplementedError

    def __call__(self, x0, metric, niter=1000, print_freq=500,
                 past_x=None, past_logs=None,
                 milestones=[], gamma=0.1, **opt_kwargs):

        self.init_past_x(x0, past_x)

        optimizer = self.opt_fun(self.parameters(), **opt_kwargs)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
        # complex x0
        zero = torch.zeros_like(x0)
        x0 = torch.stack((x0, zero), dim=-1)
        x0 = Variable(x0)

        # initialize embeding of true signal and different loss to compare running loss
        fst_order = metric.compute_first_order(x0).detach()
        if metric.fst_order:
            phi_x0_fst, phi_x0_scd = metric(x0, fst_order)
            phi_x0_fst = phi_x0_fst.detach()
            phi_x0_scd = phi_x0_scd.detach()
            loss_fst0 = self.criterion(torch.zeros_like(phi_x0_fst), phi_x0_fst)
            loss_scd0 = self.criterion(torch.zeros_like(phi_x0_scd), phi_x0_scd)
            loss0 = loss_fst0 ** 2 + loss_scd0
        else:
            phi_x0_scd, = metric(x0, fst_order)
            phi_x0_scd = phi_x0_scd.detach()
            loss_fst0 = self.criterion(torch.zeros_like(fst_order), fst_order)
            loss_scd0 = self.criterion(torch.zeros_like(phi_x0_scd), phi_x0_scd)
            loss0 = loss_scd0

        # initialize logs and return value
        x_save = self.x.data.clone()  # safe state in case NaNs appear
        if past_logs is None:
            logs = []
            logs_fst = []
            logs_scd = []
        elif self.detail:
            logs, logs_fst, logs_scd = past_logs
        else:
            logs, = past_logs

        # iter
        past_epoch = len(logs)
        for n in tqdm(range(niter)):
            # update learning rate if milestone
            scheduler.step()

            # complex vector
            x = torch.stack((self.x, torch.zeros_like(self.x)), dim=-1)

            # set gradients to 0
            optimizer.zero_grad()

            # compute embedding
            phi_x = metric(x, fst_order)

            # compute loss
            if metric.fst_order:
                phi_x_fst, phi_x_scd = phi_x
                loss_fst = self.criterion(phi_x_fst, phi_x0_fst)
                loss_scd = self.criterion(phi_x_scd, phi_x0_scd)
                loss = loss_fst ** 2 + loss_scd
            else:
                phi_x_scd, = phi_x
                loss = self.criterion(phi_x_scd, phi_x0_scd)
            loss = loss / loss0  # relative loss

            # check for NaNs
            if np.any(np.isnan(loss.data.cpu().numpy())):
                print("NaN found at iteration {}".format(n + 1))
                break
            try:
                loss.backward()
            except NanError as e:
                print(e)
                print("NaN found at iteration {}".format(n + 1))
                break

            # backup
            x_save = self.x.data.clone()

            # gradient descent step
            optimizer.step()

            # add detailed loss to log file
            loss_scalar = float(loss.data.cpu().numpy())
            logs.append(loss_scalar)

            # print loss
            if (n + past_epoch + 1) % print_freq == 0:
                tqdm.write("epoch {}: {}".format(n + past_epoch + 1, loss_scalar))
                if self.detail:
                    phi_x_fst = metric.compute_first_order(x)
                    loss_fst = self.criterion(phi_x_fst, fst_order, detail=True) / loss_fst0
                    logs_fst.append(loss_fst.data.cpu().numpy()[0, :, :, 0, 0])
                    loss_scd = self.criterion(phi_x_scd, phi_x0_scd, detail=True)
                    loss_scd = loss_scd / loss_scd0
                    logs_scd.append(loss_scd.data.cpu().numpy()[0, :])

        logs = (logs, logs_fst, logs_scd) if self.detail else (logs,)

        return x_save, logs


# Class to use algorithms that require closures, like Conjugate Gradient and LBFGS
class GradStepOptimizerClosure(OptimAbstract):
    def opt_fun(self, **kwargs):
        '''Optimization function to be redefined by subclasses.'''
        raise NotImplementedError

    def __call__(self, x0, metric, maxiter=1000, tol=1e-12, gtol=1e-8, **opt_kwargs):
        '''
        opt_kwargs are passed to the optimizer

        Assumes that x0 is a torch tensor.
        '''

        # Init parameter x to optimize
        self.init_past_x(x0, None)

        # Embedding of initial point:
        # Convert to complex before embedding
        x0 = torch.stack((x0, torch.zeros_like(x0)), dim=-1)
        phi_x0 = metric(x0)
        loss0 = self.criterion(phi_x0, None)

        # Initialize optimizer
        optimizer = self.opt_fun(self.parameters(), **opt_kwargs)

        # Iterate
        prev_loss, curr_loss = loss0, loss0
        msg = ''
        converged_p = False
        niter = 0
        next_milestone_print = None  # Initialized from first gradient
        tic = time()
        #onet = Tensor(1)
        #if self.x.device.type == 'cuda':
        #    onet = onet.cuda()
        while not converged_p and niter < maxiter:
            # Define closure needed by some algorithms to reevaluate function
            def closure():
                # Create complex version of current x
                x = torch.stack((self.x, torch.zeros_like(self.x)), dim=-1)

                # Set stored gradients to zero before recomputing them
                optimizer.zero_grad()

                # Compute embedding:
                phi_x = metric(x)

                # Compute relative loss:
                loss = self.criterion(phi_x, phi_x0)
                loss = loss / loss0

                # Compute gradients of loss wrt model parameters
                # Need to retain_graph to recompute gradient after step
                loss.backward(retain_graph=True)
                return loss
            # Take optimization step
            optimizer.step(closure)

            # Store current loss as scalar
            prev_loss = curr_loss
            curr_loss = closure()
            #curr_loss = float(curr_loss.data.cpu().numpy())
            curr_grad_norm = torch.norm(
                grad((curr_loss), (self.x), retain_graph=True)[0],
                p=float('inf')).cpu().detach().numpy()

            if next_milestone_print is None:
                next_milestone_print = 10 ** (np.floor(np.log10(curr_grad_norm)))
                #if self.x.device.type == 'cuda':
                #    next_milestone_print = next_milestone_print.cuda()


            # Print evolution
            #tqdm.write('{:6}: loss {:.6E}, |grad| {:.6E}'.format(niter, curr_loss, curr_grad_norm))


            # Check convergence, print evolution if gradient norm reaches milestone
            # Use convergence criteria of scipy.minimize
            niter += 1
            if abs(curr_loss - prev_loss) / max(curr_loss, prev_loss, 1) < tol:
               #torch.max(torch.max (curr_loss, prev_loss), onet) < tol:
                converged_p = True
                msg = 'Convergence: relative loss below threshold'
                print(msg)
            elif curr_grad_norm <= gtol:
                converged_p = True
                msg = 'Convergence: gradient below threshold'
                print(msg)
            elif curr_grad_norm <= next_milestone_print:
                delta_t = time() - tic
                fmt_str = '{:6}it in {} ( {:.2f}it/s ) ........  {:.4E} -- {:.4E}'
                print(colored(fmt_str.format(niter, hms_string(delta_t),
                                             niter/delta_t, curr_loss,
                                             curr_grad_norm)))
                next_milestone_print /= 10
        return {'x': self.x, 'niter': niter, 'loss':curr_loss, 'msg': msg}

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


class GradientDescentFst(GradStepOptimizerFst):
    def opt_fun(self, parameters, **kwargs):
        return optim.SGD(parameters, **kwargs)

class GradientDescentScd(GradStepOptimizerScd):
    def opt_fun(self, parameters, **kwargs):
        return optim.SGD(parameters, **kwargs)

class AdamDescentFst(GradStepOptimizerFst):
    def opt_fun(self, parameters, **kwargs):
        return optim.Adam(parameters, **kwargs)

class AdamDescentScd(GradStepOptimizerScd):
    def opt_fun(self, parameters, **kwargs):
        return optim.Adam(parameters, **kwargs)

class LBFGSDescent(GradStepOptimizerClosure):
    def opt_fun(self, parameters, **kwargs):
        return optim.LBFGS(parameters, **kwargs)

class SGDDescent(GradStepOptimizerClosure):
    def opt_fun(self, parameters, **kwargs):
        return optim.SGD(parameters, lr=1, **kwargs)
