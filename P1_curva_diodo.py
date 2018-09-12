# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:36:24 2018

@author: Marco
"""


#%%

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import threading
import datetime
import time
import matplotlib.pylab as pylab
from scipy import signal
from sys import stdout
import numpy.fft as fft
import os


from P1_funciones import play_rec
from P1_funciones import signalgen
from P1_funciones import sincroniza_con_trigger

params = {'legend.fontsize': 'large',
     #     'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'medium',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)


#%%


carpeta_salida = 'CurvaDiodo'
subcarpeta_salida = 'Curvas'
if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))     
    

# Genero matriz de señales: ejemplo de barrido en frecuencias en el canal 0
fs = 44100*8  
duracion = 2
muestras = int(fs*duracion)
input_channels = 2
output_channels = 1
amplitud = 1
frec_ini = 400
frec_fin = 400
pasos_frec = 1
delta_frec = (frec_fin-frec_ini)/(pasos_frec+1)
data_out = np.zeros([pasos_frec,muestras,output_channels])

for i in range(pasos_frec):
    parametros_signal = {}
    fs = fs
    amp = amplitud
    fr = frec_ini + i*delta_frec
    duration = duracion
    
    
    output_signal = signalgen('ramp',fr,amp,duration,fs)
    data_out[i,:,0] = output_signal
        
        


# Realiza medicion
offset_correlacion = 0#int(fs*(1))
steps_correlacion = 0#int(fs*(1))
data_in, retardos = play_rec(fs,input_channels,data_out,'si',offset_correlacion,steps_correlacion)

np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'data_out'),data_out)
np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'data_in'),data_in)

#%%

data_out = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'data_out.npy'))
data_in = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'data_in.npy'))

# Calibracion de los canales
data_in[:,:,0] = (data_in[:,:,0]-6.37*1e4)/(1.72*1e9)
data_in[:,:,1] = (data_in[:,:,1]-1.13*1e4)/(1.76*1e9)

resistencia = 1000
delay = 1
med = 0.1
caida_tot = data_in[0,int(fs*delay):int(fs*(delay+med)),0]
caida_res = data_in[0,int(fs*delay):int(fs*(delay+med)),1]
caida_diodo = caida_tot - caida_res
i_res = caida_res/resistencia


# Seno creciente y decreciente
ind_cre = np.diff(data_out[0,int(fs*delay):int(fs*(delay+med)),0]) > 0
ind_dec = np.diff(data_out[0,int(fs*delay):int(fs*(delay+med)),0]) < 0

caida_diodo_cre = caida_diodo[1:]
caida_diodo_cre = caida_diodo_cre[ind_cre]
caida_diodo_dec = caida_diodo[1:]
caida_diodo_dec = caida_diodo_dec[ind_dec]

i_res_cre = i_res[1:]
i_res_cre = i_res_cre[ind_cre]
i_res_dec = i_res[1:]
i_res_dec = i_res_dec[ind_dec]

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax.plot(-caida_diodo_cre,-i_res_cre*1000,'.',label='Flanco creciente',alpha=0.8)
ax.plot(-caida_diodo_dec,-i_res_dec*1000,'.',label='Flanco decreciente',alpha=0.8)
ax.legend()
ax.grid(linestyle='--')
ax.set_xlabel(u'Tensión diodo [V]')
ax.set_ylabel(u'Corriente diodo [mA]')
ax.set_ylim([-0.20e-2,1.20e-2])
ax.set_title('Curva del diodo utilizando una rampa')
figname = os.path.join(carpeta_salida,subcarpeta_salida, 'curva_diodo.png')
fig.savefig(figname, dpi=300)  
plt.close(fig)

## CURVA PROMEDIO
# Reshape
caida_diodo_dec = np.append(caida_diodo_dec,caida_diodo_dec[caida_diodo_dec.shape[0]-1])
caida_diodo_cre_r = np.reshape(caida_diodo_cre,[int(caida_diodo_cre.shape[0]/(fs/frec_ini/2)),int(fs/frec_ini/2)])
caida_diodo_dec_r = np.reshape(caida_diodo_dec,[int(caida_diodo_dec.shape[0]/(fs/frec_ini/2)),int(fs/frec_ini/2)])

i_res_dec = np.append(i_res_dec,caida_diodo_dec[i_res_dec.shape[0]-1])
i_res_cre_r = np.reshape(i_res_cre,[int(i_res_cre.shape[0]/(fs/frec_ini/2)),int(fs/frec_ini/2)])
i_res_dec_r = np.reshape(i_res_dec,[int(i_res_dec.shape[0]/(fs/frec_ini/2)),int(fs/frec_ini/2)])

caida_diodo_cre_m = np.mean(caida_diodo_cre_r,axis=0)
caida_diodo_dec_m = np.mean(caida_diodo_dec_r,axis=0)

i_res_cre_m = np.mean(i_res_cre_r,axis=0)
i_res_dec_m = np.mean(i_res_dec_r,axis=0)

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax.plot(-caida_diodo_cre_m,-i_res_cre_m*1000,'.',label='Flanco creciente',alpha=0.8)
ax.plot(-caida_diodo_dec_m,-i_res_dec_m*1000,'.',label='Flanco decreciente',alpha=0.8)
ax.legend()
ax.grid(linestyle='--')
ax.set_xlabel(u'Tensión diodo [V]')
ax.set_ylabel(u'Corriente diodo [mA]')
ax.set_ylim([-0.20e-2,1.20e-2])
ax.set_title('Curva del diodo promedio utilizando una rampa')
figname = os.path.join(carpeta_salida,subcarpeta_salida, 'curva_diodo_promedio.png')
fig.savefig(figname, dpi=300)  
plt.close(fig)


#%% AJuste de curvas



#%%

Is = 1.0*1e-12
Vt = 26.0*1e-3
n = 1.

Vtot = np.linspace(-1,1,1000)
Id = Is*(np.exp(Vtot/n/Vt)-1)

Rs = 1000
Vs = 1
Ir = Vs/Rs - Vtot/Rs

Vdiodo = Vtot - Ir*Rs


plt.plot(Vtot,Id)
plt.plot(Vdiodo,Ir)