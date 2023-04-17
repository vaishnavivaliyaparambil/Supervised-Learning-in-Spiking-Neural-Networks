#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import torch as torch
from itertools import combinations 
from autograd import grad
import numpy as np
import random
import seaborn as sns

# import  os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
nPairs = 10
use_cuda = False
num_synapses_in = 15
num_neurons_out = 1
threshold = 20
tau_ce = 20.0 # in msec
tau_ci = 10.0
tau_m = 1.2
A = 400
sim = [700]
verbose = True
alpha_exc = 1.5
beta = 1.0
alpha_inh = 1.2
mu = 1e-8
cap = 0.01
mydevice = 'cpu'
N = 1000
cache_in = {}
cache_in['spiketrains'] = torch.zeros((num_synapses_in,N), dtype = int, device = mydevice)
cache_in['last_t'] = -1
cache_in['last_spike'] = torch.ones(num_synapses_in, dtype = int, device = mydevice)*-10000
cache_in['last_potential'] = torch.zeros(num_synapses_in, dtype = float, device = mydevice)
update_potential = []
cache_out = {}
cache_out['spiketrains'] = torch.zeros((num_neurons_out,N), dtype = int, device = mydevice)
cache_out['last_t'] = -1
cache_out['last_spike'] = torch.ones(num_neurons_out, dtype = int, device = mydevice)*-10000
cache_out['last_potential'] = -60+torch.zeros(num_neurons_out, dtype = float, device = mydevice)
tau = 150

def homo_poissonST(rate, bin_size, tmax, num_synapses_in):
    spikes_tensor = torch.zeros((num_synapses_in,tmax), device = mydevice)
    nbins = np.floor(tmax/bin_size).astype(int)
    prob_of_spike = rate * bin_size/tmax
    for i in range(num_synapses_in):
        spikes_tensor[i,:] = torch.rand((1,nbins)) < prob_of_spike
    return spikes_tensor*1.0   
def eta(s): 
    values = torch.tensor([0.0]).double()#
    s = s.clone().detach().double().requires_grad_(False)
    with torch.no_grad():
        heavy = torch.heaviside(s,values)
    eta_ans = -A*torch.exp(-s/tau_m)*heavy
    return eta_ans
def d_eta(s):
    values = torch.tensor([0.0]).double()#
    s = s.clone().detach().double().requires_grad_(False)
    with torch.no_grad():
        heavy = torch.heaviside (s,values)
#     print('torch.exp(-s/tau_m)',torch.exp(-s*10**(-3)/tau_m))
    eta_ = -A*torch.exp(-s/tau_m)*heavy
#     print('eta',eta_)
#     eta_.backward()
    return (A*s/tau_m)*torch.exp(-s/tau_m)*heavy
def eps_exc(s):
    s = s.clone().detach().double().requires_grad_(False)
#     print('s',s)
    #epsilon = torch.where(s>0,1/(alpha_exc*torch.sqrt(s))*torch.exp(-beta*(alpha_exc**2)/s)*torch.exp(-s/tau_ce),0)
    epsilon = 1/(alpha_exc*torch.sqrt(s))*torch.exp(-beta*(alpha_exc**2)/s)*torch.exp(-s/tau_ce)
    epsilon = torch.nan_to_num(epsilon, 0)
#     print('epsilon')
    return epsilon
def eps_inh(s):
    # s = torch.tensor(s, dtype = torch.float32, requires_grad = False) 
    s = s.clone().detach().double().requires_grad_(False)
    epsilon = torch.where(s>0,1/(alpha_inh*torch.sqrt(s))*torch.exp(-beta*(alpha_inh**2)/s)*torch.exp(-s/tau_ci),0)
    return epsilon
def d_eps_exc(s):
    s = s.clone().detach().double().requires_grad_(True)
    epsilon = 1/(alpha_exc*torch.sqrt(s))*torch.exp(-beta*(alpha_exc**2)/s)*torch.exp(-s/tau_ce)
#     epsilon = torch.where(s>0,1/(alpha_exc*torch.sqrt(s))*torch.exp(-beta*(alpha_exc**2)/s)*torch.exp(-s/tau_ce),0)
    epsilon = torch.nan_to_num(epsilon, 0)
    epsilon.backward()
    return s.grad.float()
def d_eps_inh(s):
    s = s.clone().detach().double().requires_grad_(False)
    epsilon = torch.where(s>0,1/(alpha_inh*torch.sqrt(s))*torch.exp(-beta*(alpha_inh**2)/s)*torch.exp(-s/tau_ci),0)
    epsilon.backward()
    return s.grad.float()
def eps_matrixe (spiketimes, rows, columns):
    epmatrix = torch.zeros((rows, columns), dtype = float, device = mydevice)  
    epmatrix[:,:] = eps_exc(spiketimes[:,:])
    epmatrix = torch.nan_to_num(epmatrix, 0).requires_grad_(False)
    return epmatrix  
def eps_matrixi (spiketimes, rows, columns):
    epmatrix = torch.zeros((rows, columns), dtype = float, device = mydevice)
    epmatrix[:,:] = eps_inh(spiketimes[:,:]) 
    epmatrix = torch.nan_to_num(epmatrix, 0)
    return epmatrix 
def eta_matrix(spiketimes, rows, columns ):
    etamatrix = torch.zeros((rows, columns), dtype = float, device = mydevice)
    etamatrix[:, :] = eta(spiketimes[:,0])
    return etamatrix 
def check_spikes(spiketrain, weights, t, num_neurons,i):
    if i == 2:
        spiketrain_win = torch.flip(spiketrain, dims =(1,))#
        indices = torch.nonzero(spiketrain_win)
        neurons, timestep = spiketrain_win.shape
        spiketimes_l = torch.arange(timestep, 0, -1)
        spiketimes_list = spiketimes_l.repeat(num_neurons_out,1)
        if use_cuda:
            spiketrain_win
        weights = weights[:,:, max(0, t+1-simulation_window_size) : t+1]
        # if cache['last_spike'] == 1 or cache['last_spike'] == t-1:
        # else:
        #     last_spike = t - torch.argmax(spiketrain_win[:, ::-1], axis=1)
        #     last_potential = torch.zeros(neurons)
        past_spiketimes_n= cache_out['spiketrains'][:,max(0,t+1-simulation_window_size):t+1]
        past_indices = torch.nonzero(past_spiketimes_n)
        past_spiketimes = torch.zeros(past_spiketimes_n.shape, device = mydevice)
        for d in past_indices:
            past_spiketimes[d[0],d[1]] = past_spiketimes_n.shape[1]-d[1]
        if use_cuda:
            weights
#         exc_index = num_synapses_in-int(0.2*num_synapses_in)
#         inh_index = int(0.2*num_synapses_in)
#         if inh_index<1:
#             epsilon_matrix_exc = eps_matrixe(spiketimes_list[:exc_index,:] , exc_index, timestep)
#             epsilon_matrix_inh = eps_matrixi(spiketimes_list[:inh_index,:] , inh_index, timestep)
#             epsilon_matrix = torch.cat((epsilon_matrix_exc,epsilon_matrix_inh), dim = 0)
#         else:
        epsilon_matrix = eps_matrixe(spiketimes_list , num_neurons_out, timestep)
        epsilon_matrix.unsqueeze(-1)
        epsilon_matrix_ex = epsilon_matrix.expand(num_neurons_out,num_synapses_in, timestep)
        if use_cuda:
            epsilon_matrix
        incoming_spikes = torch.mul(weights, spiketrain_win)
        if use_cuda:
            incoming_spikes
        potential_change_ = incoming_spikes * epsilon_matrix_ex # potential change has shape n
        potential_change_1 = torch.sum(potential_change_, axis = 2)
        potential_change_2 = torch.sum(potential_change_1, axis = 1)
        if use_cuda:
            potential_change_2
        
        total_potential = cache_out['last_potential']+torch.sum(eta(past_spiketimes), axis =1) + potential_change_2
        if use_cuda:
            total_potential
        pot_chng.append(total_potential)
        # print("Check",total_potential > threshold )
        spike_indices = torch.where((total_potential > torch.ones((num_neurons_out,),device = mydevice)*threshold) & (cache_out["last_potential"] < torch.ones((num_neurons_out,),device = mydevice) *threshold ))[0]


        ##Update cache
        spiking_neurons = spike_indices #torch.nonzero(sample_win[:,-1])
        # print('spiking_neurons:', spiking_neurons)
        if not len(spiking_neurons) == 0:
            cache_out['spiketrains'][spiking_neurons,t] = 1
        cache_out['last_t'] = t
        cache_out['last_potential'] = total_potential

        # print("SRM0 Time step", t)
        # print("Incoming current", potential_change_2)
        # print("Total potential", total_potential)
        # print("Last spike", cache_out['last_spike'])
        # print("")
    return total_potential, spiking_neurons

def spiketrainTospiketimes(spikeTrain):
    a, b = torch.where(spikeTrain == 1)
    spiketimes = spikeTrain.shape[1]-b
    return spiketimes
def spiketrainTospiketimes2(spikeTrain):
    b = torch.where(spikeTrain == 1)
    spiketimes = spikeTrain.shape[0]-b[0]
    return spiketimes
def spiketrainTospiketimes3(spikeTrain):
#     print('spikeTrain', spikeTrain)
    a, b = torch.where(spikeTrain == 1)
#     print('b',b)
    spiketimes = spikeTrain.shape[1]-b-1
    return spiketimes
def error_function(num_neurons, desiredSpikes, outputSpikes): 
    error_list = []
    for i in range(num_neurons_out):
        desired = desiredSpikes[i, :]
        desired = spiketrainTospiketimes(desiredSpikes)
        output = outputSpikes[i, :]
        output = spiketrainTospiketimes(outputSpikes)
        sigma1 = 0
        sigma2 = 0
        sigma3 = 0
        for v in combinations(desired, 2):
            sigma1 += (v[0]*v[1])/(v[0]+v[1])*torch.exp(-(v[0]+v[1])/tau).double()
        for u in combinations(output, 2):
            sigma2 += (u[0]*u[1]/(u[0]+u[1])*torch.exp(-(u[0]+u[1]))/tau).double()
        for v in desired:
            for u in output:
                sigma3 += (u*v/(u+v)*torch.exp(-(u+v)/tau)).double()
        err = (sigma1+sigma2+sigma3)
        error_list.append(err)

    return error_list

def delE_deltO(num_neurons, desiredSpikes, outputSpikes):
    grad_list = []
    for i in range(num_neurons_out):
        desired = desiredSpikes[i,:]
        desired = spiketrainTospiketimes(desiredSpikes)
        output = outputSpikes[i,:]
        output = spiketrainTospiketimes(outputSpikes)
        for u in output:
            sigma1 = 0
            sigma2 = 0
            for z in output: 
                if z != u:
                    sigma1 += (torch.exp(-(z+u)/tau)*z*((z-u)-u/tau*(z+u))/(z+u)**3).double()
            for v in desired:
                sigma2 += (torch.exp(-(v+u)/tau)*v*((v-u)-u/tau*(v+u))/(v+u)**3).double()
            grad_iter = 2*(sigma1 - sigma2)
            grad_list.append(grad_iter)
    grad_mat = torch.zeros((num_neurons_out, output.shape[0]), device = mydevice, dtype = torch.float64)
    #grad_mat2 = torch.zeros((num_neurons_out, N), device = mydevice)
    #print('grad_list', grad_list)
    for t,m in enumerate(output):
        grad_mat[:,t] = grad_list[t]
    #grad_mat2[:,output] = grad_list
    #print('grad_mat1', grad_mat)
    #print('grad_mat2', grad_mat2)
    return grad_mat
def add_entry(a,b,c,d):
    element_dict["inputST"] = a
    element_dict["outputST"] = b
    element_dict["referenceweights"] = c
    element_dict["initialweights"] = d
    return element_dict

tmax = 1.0
max_rate = 10
f_osc = 2
bin_size = 10**(-3)
time = torch.arange(0, tmax, bin_size)
rate = max_rate*(np.sin(2*np.pi*f_osc*time)+1)/2 #
def inhomogeneous_poisson(rate, bin_size, num_synapses_in):
    n_bins = len(rate)
    phase_shift = 1000*torch.rand(num_synapses_in,1)
#     print('phase shift', phase_shift)
    spikes_tensor = torch.zeros(num_synapses_in,n_bins)
    for i in range(num_synapses_in):
        md_rate = torch.roll(rate,shifts=int(phase_shift[i].item()), dims = 0)
        plt.plot(range(0,n_bins), md_rate, lw=2)
        spikes_tensor[i,:] = torch.rand(1,n_bins) < md_rate * bin_size
    return spikes_tensor*1.0 
def inhomogeneous_poisson_generator(n_trials, rate, bin_size):
    for i in range(n_trials):
        yield torch.nonzero(inhomogeneous_poisson(rate, bin_size, num_synapses_in)[3,:])
ST = inhomogeneous_poisson(rate, bin_size, num_synapses_in)
def raster_plot_multi(spike_times):
    for i, spt in enumerate(spike_times):
        plt.vlines(spt, i, i+1)
    plt.yticks([])
spike_times = torch.nonzero(ST[0,:])
print('spike_times', spike_times)
n_trials = 100
plt.figure(figsize=(8,2))
plt.plot(time*10**3, rate, lw=2)
plt.ylabel('rate (Hz)')
plt.xlabel('time (s)')
# spike_times = list(inhomogeneous_poisson_generator(n_trials, rate, bin_size))
# plt.twinx()
# raster_plot_multi(spike_times)
# def fixWeights(num_synapses_in, N):
#     ran_num = torch.rand(1)
#     print('ran_num', ran_num)
#     weights = 0.2*ran_num+torch.zeros((num_neurons_out, num_synapses_in, tmax))#
#     output_ST = torch.zeros((num_neurons_out,tmax), device = mydevice)
#     time = torch.arange(0,tmax)
#     pot_chng = []
#     for t in time:
#         ST_win = ST[:, max(0, t+1-simulation_window_size) : t+1]
#         total_potential_out, spiking_neurons_out = check_spikes(ST_win, weights, t, num_neurons_out, i = 2)
#         for i in spiking_neurons_out:
#             output_ST[i, t] = 1
#     spikes = torch.nonzero(output_ST)
#     print('spikes',len(spikes))
#     spiking_rate = spikes.shape[0]/output_ST.shape[1]
#     return ran_num, spiking_rate

referenceWeights_dict = {}
initialWeights_dict = {}
interrefSpikeTrain_dict = {}
outputSpikeTrain_dict = {}

N = 1000
nTrials = 10000
for hhh in sim:
    simulation_window_size = hhh
    for s in range(nPairs):
        print('Pair:',s)
        initialWeights = []
        past = []
        cache_out = {}
        cache_out['spiketrains'] = torch.zeros((num_neurons_out,N), dtype = int, device = mydevice)
        cache_out['last_t'] = -1
        cache_out['last_spike'] = torch.ones(num_neurons_out, dtype = int, device = mydevice)*-10000
        cache_out['last_potential'] = -60+torch.zeros(num_neurons_out, dtype = float, device = mydevice)
#         ran_num = (10.0 - 11.0) * torch.rand((num_synapses_in, 1), device = mydevice) + 11.0
#         ran_num_re = ran_num.expand(num_synapses_in,N)
#         refinterWeights = torch.zeros((num_neurons_out, num_synapses_in, N), dtype = torch.float64, device = mydevice)##, requires_grad = True)#
#         refinterWeights[0,:,:] = ran_num_re
    # inh_inter = int(0.8*num_synapses_in)
    #     # interWeights[:,inh_inter:,:] *= -1
    #     # inputWeightsmod = inputWeights
        s = 20
        filename3 = "/home/vaishnavi/multiple_opt/rmsprop/nPair{}Testnrefw.pt".format(s)
#         torch.save(refinterWeights,filename3)
        refinterWeights = torch.load(filename3)
        refinterWeightsmod = refinterWeights
#         inputSpikeTrain = torch.zeros((num_synapses_in,N), device = mydevice)
    # #     # inputSpikeTrain[:,torch.rand(N) < 1/4] = 1
#         interrefSpikeTrain = inhomogeneous_poisson(rate, bin_size, num_synapses_in)
#         refSpikeTrain = torch.zeros((num_neurons_out,N), device = mydevice)
        filename2 = "/home/vaishnavi/multiple_opt/rmsprop/nPair{}TestninputST.pt".format(s)
        interrefSpikeTrain = next(iter(torch.load(filename2)))
        filename4 = "/home/vaishnavi/multiple_opt/rmsprop/nPair{}TestnrefST.pt".format(s)
        refSpikeTrain = next(iter(torch.load(filename4)))
        time = torch.arange(0,N)
        pot_chng = []
#         for t in time:
#             interrefspiketrain_win = interrefSpikeTrain[:, max(0, t+1-simulation_window_size) : t+1]
#             total_potential_out, spiking_neurons_out = check_spikes(interrefspiketrain_win, refinterWeightsmod, t, num_neurons_out, i = 2)
#             for i in spiking_neurons_out:
#                 refSpikeTrain[i, t] = 1
        pot_chng_ref = pot_chng
        print('freq',len(torch.nonzero(refSpikeTrain)))
#         plt.plot(time.cpu().numpy(),pot_chng)
#         plt.title("refSpikeTrain")
#         plt.show()
        if len(torch.nonzero(refSpikeTrain)) < 15 or len(torch.nonzero(refSpikeTrain))>30 :
           continue
        else: 

#             inputSpT = {interrefSpikeTrain}
#             filename2 = "/home/vaishnavi/multiple_opt/normal/nPair{}TestninputST.pt".format(s)
#             torch.save(inputSpT, filename2)
#             refST = {refSpikeTrain}
#             filename4 = "/home/vaishnavi/multiple_opt/normal/nPair{}TestnrefST.pt".format(s)
#             torch.save(refST, filename4)
#             outputSpikeTrain_dict[s] = refSpikeTrain
            filename1 = "/home/vaishnavi/multiple_opt/normal/nPair{}Testnweights7.pt".format(s)
            ran_num2 = torch.load(filename1)[-1].T
# #             ran_num2 = (10.0 - 11.0) * torch.rand((num_synapses_in, 1), device = mydevice) + 11.0
# #             plus_minus = torch.randint(-1,2, size = (num_synapses_in, 1),device = mydevice)
# #             ran_num1 = ran_num2+plus_minus*0.1
            ran_num_re2 = ran_num2.expand(num_synapses_in,N)
            interWeights = torch.zeros((num_neurons_out, num_synapses_in, N), dtype = torch.float64, device = mydevice)##, requires_grad = True)#
            interWeights[0,:,:] = ran_num_re2
#             filename1 = "/home/vaishnavi/multiple_opt/normal/nPair{}Testinitweights1.pt".format(s)
#             torch.save(interWeights,filename1)
#             interWeights = torch.load(filename1)
            interWeightsupdate = interWeights
            weightupdate_list = []
            synaptic_wt_list = []
            update_list = []
            spikes_trials = []
            error_mat = []
            pot_chng_list = []
            m0 = 0
            v0 = 0
            for w in range(nTrials):
                print("Trial No",w)
                interWeightsmod = interWeightsupdate
                update_tens = torch.zeros((num_neurons_out, num_synapses_in, N), dtype = torch.float64, device = mydevice)
                time = torch.arange(0,N, device = 'cpu')
                cache_out = {}
                cache_out['spiketrains'] = torch.zeros((num_neurons_out,N), dtype = int, device = mydevice)
                cache_out['last_t'] = -1
                cache_out['last_spike'] = torch.ones(num_neurons_out, dtype = int, device = mydevice)*-10000
                cache_out['last_potential'] = -60+torch.zeros(num_neurons_out, dtype = float, device = mydevice)
                outputSpikeTrain = torch.zeros((num_neurons_out,N), device = mydevice)
                pot_chng = []
                ssd = 0
                ddd = 0
                for r in time:
        #             print('r',r)
                    interspiketrain_win = interrefSpikeTrain[:, max(0, r+1-simulation_window_size) : r+1]
                    total_potential_out_sim, spiking_neurons_out_sim = check_spikes(interspiketrain_win, interWeightsmod, r, num_neurons_out, i = 2)
                    for i in spiking_neurons_out_sim:
                        outputSpikeTrain[i, r] = 1
#                     print('ref', torch.nonzero(refSpikeTrain))
                    if outputSpikeTrain[0, r-10] == 1 or refSpikeTrain[0, r-10]==1 :

                        outputSpikes_full  = torch.nonzero(outputSpikeTrain[:,:r])[:,1]
#                         print('ouputSpikes_full', outputSpikes_full)
            #             desiredSpikes_full = torch.nonzero(refSpikeTrain[:,:r])[:,1]
                        spiketimesout_win = torch.nonzero(outputSpikeTrain[:,max(0,r+1-simulation_window_size):r+1])[:,1]
                        spiketimesdes_win = torch.nonzero(refSpikeTrain[:,max(0,r+1-simulation_window_size):r+1])[:,1]
            #             spiketimes = torch.cat((outputSpikes_full,desiredSpikes_full))
                        spiketimes = torch.cat((spiketimesout_win, spiketimesdes_win))
                        spiketimes = torch.unique(spiketimes)
            #             print('spiketimes_out', spiketimesout_win)
            #             print('spiketimes_des', spiketimesdes_win)
        #                     if torch.any(r-5 == spiketimes):
#                         print('r', r)
#                         print('spiketimes_out', spiketimesout_win)
#                         print('spiketimes_des', spiketimesdes_win)
                        
                        for h in range(len(spiketimesout_win)):
    #                         last_spiketime = spiketimesout_win[h]
    #                         print('last_spiketime',last_spiketime)
                            desiredSpikes = refSpikeTrain[:, max(0, r+1-simulation_window_size) : r+1]
                            outputSpikes = outputSpikeTrain[:, max(0, r+1-simulation_window_size) : r+1]
                            delE_delt = delE_deltO(num_neurons_out, desiredSpikes, outputSpikes)
    #                         print('delE_delt',delE_delt)
    #                         delindices = torch.nonzero(delE_delt)#
                            delE_dw = torch.zeros((outputSpikes.shape[1], num_synapses_in), dtype = torch.float64, device = mydevice)
                            
                            for j in range(num_synapses_in):
    #                             print('j',j)
                                interWeightsmod_win = interWeightsmod[:, j, max(0, r+1-simulation_window_size) : r+1]
                                interspike = spiketrainTospiketimes2(interrefSpikeTrain[j,max(0, r+1-simulation_window_size) : r+1])
    #                             print('ST',len(torch.nonzero(interrefSpikeTrain[j,:])))
    #                             print('interspike', interspike)
                                interspikeweights_win = interWeightsmod_win[0,interspike-1]
    #                             print('interspikeweights_win',interspikeweights_win)
                                outputspike = spiketrainTospiketimes3(outputSpikes)
                                delt_delw = torch.zeros((interspike.shape[0],outputspike.shape[0]), dtype = torch.float64, device = mydevice)
                                cross_prod = 0
                                deltwh_mat= torch.zeros((interWeightsmod_win.shape[1],len(outputspike)), device = mydevice)
            #                 # delEw_sum = torch.zeros((interWeightsmod_win.shape[1], num_neurons_out), device = mydevice)
                                delEw_sum = 0
                                outputspike = torch.flip(outputspike, dims = (0,))
    #                             print('outputspike',outputspike)
                                for m in range(len(outputspike)):
                                    tl = outputspike[m]
    #                                 print('tl',m,tl)
                                    eppsmatrix = torch.zeros((interspike.shape[0],num_neurons_out), dtype = torch.float64, device = mydevice)
                                    depsmatrix = torch.zeros((interspike.shape[0],num_neurons_out), dtype=torch.float64, device = mydevice)
                                    for z,k in enumerate(interspike): 
                                        t0 = (k-tl).double()
                                        eppsmatrix[z,:] = (eps_exc(t0)).double() 
                                        depsmatrix[z,:] = torch.nan_to_num(d_eps_exc(t0), nan = 0.0) #nan = 1 because i dont want weights to change during multiplication in the nextstep
            #                             depsmatrix[i,:] = d_eps_exc(eppsmatrix[i,:])
                                    term2 = torch.sum(interspikeweights_win *depsmatrix.T).double()#
    #                                 print('term2', term2)
                                    detamatrix = torch.zeros((outputspike.shape[0], num_neurons_out), dtype=torch.float64, device = mydevice)
                                    for z,k in enumerate(outputspike): 
    #                                     print('outputspike2',outputspike)
                                        t0 = k-tl
    #                                     print('t0',t0)
    #                                     print('d_eta(t0)',d_eta(t0))
                                        detamatrix[z,:] = d_eta(t0)
    #                                     print('z',z)
    #                                     print('detamatrix[z,:]',detamatrix[z,:])
                                    term3 = torch.sum(detamatrix, axis = 0).double() #
    #                                 print('term3', term3)
    #                                 print('CALC')
                                    if outputspike.nelement() == 1:
    #                                     print('outputspike',outputspike)
                                        if term2 != 0 or term3[0] != 0:
                                            delt_delw[:,m] = (eppsmatrix/(term2+term3))[:,0]
        #                                         print('epps_nonzero',torch.nonzero(eppsmatrix))
        #                                     if use_cuda:
        #                                         delt_delw
    #                                         print('deta_', detamatrix)
        #                                         print('delt_delwnonz',torch.nonzero(delt_delw)[:,m])
                                            cross_prod = torch.sum(detamatrix[0,0]*delt_delw[:,0], axis=0)
    #                                         print('CP', cross_prod)

                                    else:
                                        for f in range(len(outputspike)):
                                            if term2 != 0 or term3[0] != 0:
    #                                             print('outputspike',outputspike)
                                                delt_delw[:,f] = ((eppsmatrix+cross_prod)/(term2+term3))[:,0]
                                                if use_cuda:
                                                    delt_delw.cuda()
                                                cross_prod += torch.sum(detamatrix[f,0]*delt_delw[:,f], axis=0)  
                                    if interspike.nelement() != 0:
                                        delEw_mat = torch.matmul(delE_delt,delt_delw.T)
                                        delEw_sum += torch.sum(delEw_mat,axis=1).double()
    #                                 print('delEw_sum',delEw_sum)
                                update_tens[:,j,r+1:] = delEw_sum
                                    
    #                         jj = 0
                        #
#                         sns.heatmap(update_tens_cap[0,:,:].detach().cpu().numpy())
#                         plt.title('Update_Tensor(Capped)')
#                         plt.show()
                        
                        up_w = update_tens[:,:,-1].reshape(1, num_synapses_in).T
#                         upw_expand = up_w.expand(num_synapses_in,N)
#                         upw_expand2 = torch.zeros((num_neurons_out, num_synapses_in, N), dtype = torch.float64, device = mydevice)##, requires_grad = True)#
#                         upw_expand2[0,:,:] = upw_expand
                        update_tens_cap = torch.clamp(up_w ,min = -cap, max=cap)
                        interWeightsupdate = interWeightsmod - update_tens_cap
#                         sns.heatmap(update_tens_cap[0,:,:].detach().cpu().numpy())
#                         plt.title('Update_Tensor(Capped)')
#                         plt.show()
                        if interWeightsupdate.isnan().any():
                            break

    #                         if last_w.any()<0:
    #                             mu = -mu
                        last_weight = interWeightsupdate[0,:,-1].reshape(1,num_synapses_in)
    #                     print('last_weight', last_weight)
                        synaptic_wt_list.append(last_weight)
#                         print(synaptic_wt_list[w])
#                         interWeightsupdate[:,:,:] = last_weight.t()
#                         filename100 = "/home/vaishnavi/multiple_opt/normal/nPair{}Testnweights_updates8.pt".format(s)
#                         torch.save(synaptic_wt_list,filename100)
#                 print(synaptic_wt_list[w])           
#                 filename1 = "/home/vaishnavi/multiple_opt/normal/nPair{}Testnweights8.pt".format(s)
#                 weightup_dict = {}
#                 weightupdate_list.append(synaptic_wt_list[-1])
#                 weightup_dict[w] = weightupdate_list
#                 torch.save(weightupdate_list,filename1)
#                 filename2 = "/home/vaishnavi/multiple_opt/normal/nPair{}Testnerror8.pt".format(s)
#                 error_dict = {}
#                 error = error_function(num_neurons_out, refSpikeTrain[:,:], outputSpikeTrain[:,:])
#                 error_mat.append(error)
#                 error_dict[w] = error_mat
#                 torch.save(error_dict,filename2)
#         filename2 = "/home/vaishnavi/multiple_opt/normal/nPair{}Testnmu8.pt".format(s)
#         torch.save(mu,filename2)


# In[ ]:


import torch as torch
filename = "/home/vaishnavi/multiple_opt/adam/nPair20Testnweights2.pt"
ex = torch.load(filename)
ex


# In[ ]:




