import os,gc
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import torch
from torch.autograd import Variable, grad

# ---- Reconstruct marks. At initiation, every point has the average value of the marks.----#
#---- Trying scipy L-BFGS ----#
def obj_fun(x,wph_ops,factr_ops,Sims,op_id):
    if x.grad is not None:
        x.grad.data.zero_()
    #global wph_ops
    wph_op = wph_ops[op_id]
    p = wph_op(x)
    diff = p-Sims[op_id]
    diff = diff * factr_ops[op_id]
    loss = torch.mul(diff,diff).sum()
    return loss

def grad_obj_fun(x_gpu,grad_err,wph_ops,factr_ops,Sims):
    loss = 0
    #global grad_err
    grad_err[:] = 0
    for op_id in range(len(wph_ops)):
        x_t = x_gpu.clone().requires_grad_(True)
        loss_t = obj_fun(x_t,wph_ops,factr_ops,Sims,op_id)
        grad_err_t, = grad([loss_t],[x_t], retain_graph=False)
        loss = loss + loss_t
        grad_err = grad_err + grad_err_t
        
    return loss, grad_err

from time import time
def fun_and_grad_conv(x,grad_err,wph_ops,factr_ops,Sims,size):
    x_float = torch.reshape(torch.tensor(x,dtype=torch.float),(1,1,size,size))
    x_gpu = x_float.cuda()
    loss, grad_err = grad_obj_fun(x_gpu,grad_err,wph_ops,factr_ops,Sims)
    return loss.cpu().item(), np.asarray(grad_err.reshape(size**2).cpu().numpy(), dtype=np.float64)

def callback_print(x):
    return

def call_lbfgs_routine(FOLOUT,labelname,im,wph_ops,Sims,N,Krec,nb_restarts,maxite,factr,factr_ops,\
                       maxcor=20,gtol=1e-14,ftol=1e-14,init='normal',toskip=True):
    grad_err = im.clone()
    size = N
    for krec in range(Krec):
        if init=='normal':
            print('init normal')
            x = torch.Tensor(1, 1, N, N).normal_()
        elif init=='normal00105':
            print('init normal00105')
            x = torch.Tensor(1, 1, N, N).normal_(std=0.01)+0.5
        elif "maxent" in init:
            print('load init from ' + init)
            xinit = sio.loadmat('./data/maxent/' + init + '.mat')
            x = torch.from_numpy(xinit['imgs'][:,:,krec]) #  .shape)
            #assert(false)
        elif init=='normalstdbarx':
            stdbarx = im.std()
            print('init normal with std barx ' + str(stdbarx))
            x = torch.Tensor(1, 1, N, N).normal_(std=stdbarx)
        else:
            assert(false)
        x0 = x.reshape(size**2).numpy()
        x0 = np.asarray(x0, dtype=np.float64)
        x_opt = None
        for start in range(nb_restarts+1):
            time0 = time()
            datname =  FOLOUT + '/' + labelname + '_krec' + str(krec) + '_start' + str(start) + '.pt'
            if os.path.isfile(datname) and toskip:
                print('skip', datname)
                continue
            else:
                print('save to',datname)
                
            if start==0:
                x_opt = x0
            elif x_opt is None:
                # load from previous saved file
                prename = FOLOUT + '/' + labelname + '_krec' + str(krec) + '_start' + str(start-1) + '.pt'
                print('load x_opt from',prename)
                saved_result = torch.load(prename)
                im_opt = saved_result['tensor_opt'].numpy()
                x_opt = im_opt.reshape(size**2)
                x_opt = np.asarray(x_opt,dtype=np.float64)
                
            res = opt.minimize(fun_and_grad_conv, x_opt,args=(grad_err,wph_ops,factr_ops,Sims,size),\
                               method='L-BFGS-B', jac=True, tol=None,\
                               callback=callback_print,\
                               options={'maxiter': maxite, 'gtol': gtol, 'ftol': ftol, 'maxcor': maxcor})
            final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']
            print('OPT fini avec:', final_loss,niter,msg)
            
            im_opt = np.reshape(x_opt, (size,size))
            tensor_opt = torch.tensor(im_opt, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            
            ret = dict()
            ret['tensor_opt'] = tensor_opt
            ret['normalized_loss'] = final_loss/(factr**2)
            torch.save(ret, datname)

            print('krec',krec,'strat', start, 'using time (sec):' , time()-time0)
            time0 = time()
