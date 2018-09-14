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
# CALIBRACION PLACA MARCO PC CASA

windows_nivel = np.array([10,20,30,40,50,60,70,80,90,100])
tension_rms_v_ch0 = np.array([0.050, 0.142, 0.284, 0.441, 0.678, 0.884, 1.143, 1.484, 1.771, 2.280])
amplitud_v_ch0 = tension_rms_v_ch0*np.sqrt(2)
tension_rms_v_ch1 = np.array([0.050, 0.146, 0.291, 0.451, 0.693, 0.904, 1.170, 1.518, 1.812, 2.330])
amplitud_v_ch1 = tension_rms_v_ch1*np.sqrt(2)

amplitud_v_chs = np.array([amplitud_v_ch0,amplitud_v_ch1])


#plt.plot(windows_nivel,amplitud_v_ch0,'o')
#plt.plot(windows_nivel,amplitud_v_ch1,'o')    
#%%

# Genero matriz de señales: ejemplo de barrido en frecuencias en el canal 0
ind_nivel = 6
mic_level = 70    

dato = 'int16'    
fs = 44100*8  
duracion_trigger = 0.5
duracion_ruido = 5
muestras = int(fs*duracion_trigger) + int(fs*duracion_ruido)
input_channels = 2
output_channels = 2
amplitud_v_chs_out = [1.0,1.0] #V
amplitud_chs = []
for i in range(output_channels):
    amplitud_chs.append(amplitud_v_chs_out[i]/amplitud_v_chs[i,ind_nivel])
fr_trigger = 500
data_out = np.zeros([1,muestras,output_channels])


output_signal_ch0 = signalgen('sine',fr_trigger,amplitud_chs[0],duracion_trigger,fs)
output_signal_ch0 = np.append(output_signal_ch0,np.zeros(int(fs*duracion_ruido)))

output_signal_ch1 = signalgen('sine',fr_trigger,amplitud_chs[1],duracion_trigger,fs)
output_signal_ch1 = np.append(output_signal_ch1,np.zeros(int(fs*duracion_ruido)))
    
data_out[0,:,0] = output_signal_ch0
data_out[0,:,1] = output_signal_ch1
        

carpeta_salida = 'Ruido'
subcarpeta_salida = dato
if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))  


calibracion_CH0_seno = np.load(os.path.join('Calibracion',dato, 'Seno_CH0_wp'+ str(windows_nivel[ind_nivel]) +  '_wm'+str(mic_level)+'_'+dato+'_ajuste.npy'))
calibracion_CH1_seno = np.load(os.path.join('Calibracion',dato, 'Seno_CH1_wp'+ str(windows_nivel[ind_nivel]) +  '_wm'+str(mic_level)+'_'+dato+'_ajuste.npy'))


# Realiza medicion
offset_correlacion = 0#int(fs*(1))
steps_correlacion = 0#int(fs*(1))
data_in, retardos = play_rec(fs,input_channels,data_out,'si',offset_correlacion,steps_correlacion,dato=dato)

np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'data_out'),data_out)
np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'data_in'),data_in)

data_out = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'data_out.npy'))
data_in = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'data_in.npy'))

# Calibracion de los canales
data_in_cal = np.zeros([data_in.shape[0],data_in.shape[1],data_in.shape[2]])
data_in_cal[:,:,0] = (data_in[:,:,0]-calibracion_CH0_seno[1])/(calibracion_CH0_seno[0])
data_in_cal[:,:,1] = (data_in[:,:,1]-calibracion_CH1_seno[1])/(calibracion_CH1_seno[0])


delay = 3
med = 2
data_in_med = data_in[0,int(fs*delay):int(fs*(delay+med)),:]
data_in_cal_med = data_in_cal[0,int(fs*delay):int(fs*(delay+med)),:]


np.std(data_in_cal_med[:,0])
np.std(data_in_cal_med[:,1])

label0='CH0 - STD: ' + '{:6.3f}'.format(np.std(data_in_cal_med[:,0]*1000)) + ' mV -' + '{:6.1f}'.format(np.std(data_in_med[:,0])) + ' cuentas'
label1='CH1 - STD: ' + '{:6.3f}'.format(np.std(data_in_cal_med[:,1]*1000)) + ' mV -' + '{:6.1f}'.format(np.std(data_in_med[:,1])) + ' cuentas'

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax.hist(data_in_cal_med[:,0],bins=10,rwidth=0.9,label=label0,alpha=0.7,align ='mid')
ax.grid(linestyle='--')
ax.set_xlabel(u'Tensión [V]')
ax.set_ylabel(u'Frecuencia [cts.]')
ax.legend()
ax.set_title(u'Histograma de ruido en tensión del CH0')
figname = os.path.join(carpeta_salida,subcarpeta_salida, 'ruido_v_ch0.png')
fig.savefig(figname, dpi=300)  
plt.close(fig)


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax.hist(data_in_med[:,0],bins=10,rwidth=0.9,label=label0,alpha=0.7,align ='mid')
ax.grid(linestyle='--')
ax.set_xlabel(u'Cuentas')
ax.set_ylabel(u'Frecuencia [cts.]')
ax.legend()
ax.set_title(u'Histograma de ruido en cuentas del CH0')
figname = os.path.join(carpeta_salida,subcarpeta_salida, 'ruido_cuentas_ch0.png')
fig.savefig(figname, dpi=300)  
plt.close(fig)


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax.hist(data_in_cal_med[:,1],bins=10,rwidth=0.9,label=label1,alpha=0.7,align ='mid')
ax.grid(linestyle='--')
ax.set_xlabel(u'Tensión [V]')
ax.set_ylabel(u'Frecuencia [cts.]')
ax.legend()
ax.set_title(u'Histograma de ruido en tensión del CH1')
figname = os.path.join(carpeta_salida,subcarpeta_salida, 'ruido_v_ch1.png')
fig.savefig(figname, dpi=300)  
plt.close(fig)


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax.hist(data_in_med[:,1],bins=10,rwidth=0.9,label=label1,alpha=0.7,align ='mid')
ax.grid(linestyle='--')
ax.set_xlabel(u'Cuentas')
ax.set_ylabel(u'Frecuencia [cts.]')
ax.legend()
ax.set_title(u'Histograma de ruido en cuentas del CH1')
figname = os.path.join(carpeta_salida,subcarpeta_salida, 'ruido_cuentas_ch1.png')
fig.savefig(figname, dpi=300)  
plt.close(fig)