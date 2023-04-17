#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import torch as torch
import numpy as np


# In[2]:


filename = ##"C:/Users/vaish/Desktop/multiple_opt/adam/nPair1Testnweights_updates.pt"##


# In[3]:


export = torch.load(filename)


# In[4]:


filename9 = ##"C:/Users/vaish/Desktop/multiple_opt/adam/nPair1Testnrefw.pt"##
refweights = torch.load(filename9)
refweight = []
for i in refweights[0]:
    refweight.append(i[0])
print(len(refweight))


# In[5]:


initweights = export


# In[6]:


num_synapses_in = 15
time = torch.arange(0,len(initweights),1).cuda()


# In[7]:


syn_mape_dict = {}
for i in range(num_synapses_in):
    syn_mape_list = []
    for n in range(len(initweights)):
        upd_weight = initweights[n]
        upd_synw = upd_weight[0][i]
        syn_mape = torch.abs(upd_synw - refweight[i])/refweight[i]
#         print(syn_mape)
        syn_mape_list.append(syn_mape)
    syn_mape_dict[i] = syn_mape_list


# In[8]:


avg_mape_list = []
for n in range(len(initweights)):
    avg_mape = 0
    for i in range(num_synapses_in):
        avg_mape += syn_mape_dict[i][n]/num_synapses_in
    avg_mape_list.append(avg_mape)


# In[9]:


fig = plt.figure(figsize =(20, 6))
plt.plot(time.cpu().detach().numpy(),avg_mape_list)
plt.xlabel("Update number")
plt.ylabel("Mean Absolute Percentage Error")
# plt.ylim(0,1.2)
# plt.xlim(0,20000)
plt.title('Avg')
plt.show()


# In[10]:


fig = plt.figure(figsize =(10, 6))
# print(time.shape)
for n in range(num_synapses_in):
        plt.plot(time.cpu().detach().numpy(),syn_mape_dict[n], label = n)
plt.xlabel("Update number")
plt.ylabel("Mean Absolute Percentage Error")
plt.title('Synapses')
plt.legend()
plt.show()


# In[ ]:




