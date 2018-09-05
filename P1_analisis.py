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


from P1_funciones import play_rec
from P1_funciones import signalgen
from P1_funciones import sincroniza_con_trigger

params = {'legend.fontsize': 'medium',
     #     'figure.figsize': (15, 5),
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)


#%%
    


# Genero matriz de señales: ejemplo de barrido en frecuencias en el canal 0
fs = 44100*8  
duracion = 2
muestras = int(fs*duracion)
input_channels = 2
output_channels = 2
amplitud = 0.1
frec_ini = 200
frec_fin = 0
pasos = 1
delta_frec = (frec_fin-frec_ini)/(pasos+1)
data_out = np.zeros([pasos,muestras,output_channels])

for i in range(pasos):
    parametros_signal = {}
    fs = fs
    amp = amplitud
    fr = frec_ini + i*delta_frec
    duration = duracion
    type = 'sine'
    
    output_signal = signalgen(type,fr,amp,duration,fs)
    output_signal = output_signal*np.arange(muestras)/muestras
    #output_signal = amplitud*signal.chirp(np.arange(muestras)/fs,frec_fin,duracion,frec_ini)
    
    data_out[i,:,0] = output_signal


# Realiza medicion
offset_correlacion = 0#int(fs*(1))
steps_correlacion = 0#int(fs*(1))
data_in, retardos = play_rec(fs,input_channels,data_out,'si',offset_correlacion,steps_correlacion)

plt.plot(np.transpose(data_in[0,:,0]))


#%%

### Corrige retardo y grafica
#
#data_in, retardos = sincroniza_con_trigger(data_out, data_in) 

#%%
ch = 0
step = 0

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax1 = ax.twinx()
ax.plot(np.transpose(data_in[step,:,ch]),'-',color='r', label='señal adquirida',alpha=0.5)
ax1.plot(np.transpose(data_out[step,:,ch]),'-',color='b', label='señal enviada',alpha=0.5)
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax.hist(retardos/fs*1000, bins=100)
ax.set_title(u'Retardos')
ax.set_xlabel('Retardos [ms]')
ax.set_ylabel('Frecuencia')


#%%
### ANALISIS de la señal adquirida. Cheque que la señal adquirida corresponde a la enviada


ch_acq = 0
ch_send = 0
paso = 0

### Realiza la FFT de la señal enviada y adquirida
fft_send = abs(fft.fft(data_out[paso,:,ch_send]))**2/int(data_out.shape[1]/2+1)/fs
fft_send = fft_send[0:int(data_out.shape[1]/2+1)]
fft_send[1:] = 2*fft_send[1:]
fft_acq = abs(fft.fft(data_in[paso,:,ch_acq]))**2/int(data_in.shape[1]/2+1)/fs
fft_acq = fft_acq[0:int(data_in.shape[1]/2+1)]
fft_acq[1:] = 2*fft_acq[1:]

frec_send = np.linspace(0,int(data_out.shape[1]/2),int(data_out.shape[1]/2+1))
frec_send = frec_send*(fs/2+1)/int(data_out.shape[1]/2+1)
frec_acq = np.linspace(0,int(data_in.shape[1]/2),int(data_in.shape[1]/2+1))
frec_acq = frec_acq*(fs/2+1)/int(data_in.shape[1]/2+1)

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .12, .75, .8])
ax1 = ax.twinx()
ax.plot(frec_send,fft_send,'-' ,label='Frec enviada',alpha=0.7)
ax1.plot(frec_acq,fft_acq,'-',color='red', label=u'Señal adquirida',alpha=0.7)
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud [a.u.]')
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .12, .75, .8])
ax.plot(frec_acq,fft_acq/fft_send,'-',color='red', label=u'Señal adquirida',alpha=0.7)
ax.set_xlim([0,23000])
ax.set_ylim([0,1e10])
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud [a.u.]')
ax.legend(loc=1)
plt.show()
 

#%%

Is = 1.0*1e-12
Vt = 26.0*1e-3
n = 1.

Vd = np.linspace(-1,1,1000)
Id = Is*(np.exp(Vd/n/Vt)-1)

Rs = 100
Vs = 1
Ir = Vs/Rs - Vd/Rs


plt.plot(Vd,Id)
plt.plot(Vd,Ir)

#%%

# Respuesta emisor-receptor

fs = 44100*8  
duracion = 10
muestras = int(fs*duracion)
input_channels = 2
output_channels = 2
amplitud = 0.1

# Frecuencias bajas
frec_ini = 0
frec_fin = 23000
data_out1 = np.zeros([1,muestras,output_channels])
output_signal = amplitud*signal.chirp(np.arange(muestras)/fs,frec_fin,duracion,frec_ini)
#output_signal = amplitud*np.random.rand(muestras)
data_out1[0,:,0] = output_signal

offset_correlacion = int(fs*(5))
steps_correlacion = int(fs*(0.2))
data_in1, retardos1 = play_rec(fs,input_channels,data_out1,'si',offset_correlacion,steps_correlacion)


#plt.plot(np.transpose(data_out1[0,:,0]))


### Realiza la FFT de la señal enviada y adquirida
paso = 0
ch_send = 0
ch_acq = 0
frec_comp = 10000

fft_send1 = abs(fft.fft(data_out1[paso,:,ch_send]))**2/int(data_out1.shape[1]/2+1)/fs
fft_send1 = fft_send1[0:int(data_out1.shape[1]/2+1)]
fft_acq1 = abs(fft.fft(data_in1[paso,:,ch_acq]))**2/int(data_in1.shape[1]/2+1)/fs
fft_acq1 = fft_acq1[0:int(data_in1.shape[1]/2+1)]

frec_send1 = np.linspace(0,int(data_out1.shape[1]/2),int(data_out1.shape[1]/2+1))
frec_send1 = frec_send1*(fs/2+1)/int(data_out1.shape[1]/2+1)
frec_acq1 = np.linspace(0,int(data_in1.shape[1]/2),int(data_in1.shape[1]/2+1))
frec_acq1 = frec_acq1*(fs/2+1)/int(data_in1.shape[1]/2+1)

frec_ind = np.argmin(np.abs(frec_acq1-frec_comp))

#fft_norm1 = fft_acq1/fft_send1

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .12, .75, .8])
ax.semilogy(frec_acq1,fft_acq1/fft_acq1[frec_ind],'-',color='red', label=u'Señal adquirida',alpha=0.7)
ax.set_xlim([-2000,28000])
ax.set_ylim([1e-4,1e1])
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Potencia [db]')
ax.legend(loc=1)
ax.grid(linewidth=0.5,linestyle='--')
plt.show()

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .12, .75, .8])
ax.semilogy(frec_acq1,fft_acq1/fft_acq1[frec_ind],'-',color='red', label=u'Señal adquirida',alpha=0.7)
ax.set_xlim([-100,400])
ax.set_ylim([1e-3,1e1])
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Potencia [db]')
ax.legend(loc=1)
ax.grid(linewidth=0.5,linestyle='--')
plt.show()


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .12, .75, .8])
ax.semilogy(frec_acq1,fft_acq1/fft_acq1[frec_ind],'-',color='red', label=u'Señal adquirida',alpha=0.7)
ax.set_xlim([10000,30000])
ax.set_ylim([1e-3,1e1])
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Potencia [db]')
ax.legend(loc=1)
ax.grid(linewidth=0.5,linestyle='--')
plt.show()

