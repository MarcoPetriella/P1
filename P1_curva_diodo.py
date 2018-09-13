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
duracion =5
muestras = int(fs*duracion)
input_channels = 2
output_channels = 1
amplitud_V = 0.98 #V
valor_rms = 0.7 # en V. Depende del nivel de volumen de parlante.
amplitud = amplitud_V/valor_rms/np.sqrt(2)
frec_ini = 500
frec_fin = 500
pasos_frec = 1
delta_frec = (frec_fin-frec_ini)/(pasos_frec+1)
data_out = np.zeros([pasos_frec,muestras,output_channels])

for i in range(pasos_frec):
    parametros_signal = {}
    fs = fs
    amp = amplitud
    fr = frec_ini + i*delta_frec
    duration = duracion
    
    
    output_signal = signalgen('sine',fr,amp,duration,fs)
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

# Para nivel de sonido windows de 24/100
calibracion_CH0_rampa = [2.06*1e9, 6.02*1e4]
calibracion_CH0_seno = [2.03*1e9, 1.65*1e5]

calibracion_CH1_rampa = [2.10*1e9, 4.53*1e4]
calibracion_CH1_seno = [2.07*1e9, 1.41*1e5]


# Calibracion de los canales
data_in[:,:,0] = (data_in[:,:,0]-calibracion_CH0_seno[1])/(calibracion_CH0_seno[0])
data_in[:,:,1] = (data_in[:,:,1]-calibracion_CH1_seno[1])/(calibracion_CH1_seno[0])

resistencia = 150
delay = 2
med = 1
caida_tot = -data_in[0,int(fs*delay):int(fs*(delay+med)),0]
caida_res = -data_in[0,int(fs*delay):int(fs*(delay+med)),1]
caida_diodo = caida_tot - caida_res
i_res = caida_res/resistencia


# Seno creciente y decreciente
ind_cre = np.diff(data_out[0,int(fs*delay):int(fs*(delay+med)),0]) > 0
ind_dec = np.diff(data_out[0,int(fs*delay):int(fs*(delay+med)),0]) < 0

#ole = data_out[0,int(fs*delay):int(fs*(delay+med))-1,0]
#plt.plot(ole[ind_dec],'.')

caida_diodo_cre = caida_diodo[1:]
caida_diodo_cre = caida_diodo_cre[ind_cre]
caida_diodo_dec = caida_diodo[1:]
caida_diodo_dec = caida_diodo_dec[ind_dec]

i_res_cre = i_res[:-1]
i_res_cre = i_res_cre[ind_cre]
i_res_dec = i_res[:-1]
i_res_dec = i_res_dec[ind_dec]

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax.plot(caida_diodo_cre,i_res_cre*1000,'.',label='Flanco creciente',alpha=0.8)
ax.plot(caida_diodo_dec,i_res_dec*1000,'.',label='Flanco decreciente',alpha=0.8)
ax.legend()
ax.grid(linestyle='--')
ax.set_xlabel(u'Tensión diodo [V]')
ax.set_ylabel(u'Corriente diodo [mA]')
#ax.set_ylim([-0.20e-2,0.8e-2])
ax.set_title('Curva del diodo utilizando un seno')
figname = os.path.join(carpeta_salida,subcarpeta_salida, 'curva_diodo.png')
fig.savefig(figname, dpi=300)  
plt.close(fig)






#fig = plt.figure(figsize=(14, 7), dpi=250)
#ax = fig.add_axes([.12, .15, .75, .8])
#ax1 = ax.twinx()
#ax.plot(data_in[0,:,0] ,alpha=0.8)
#ax1.plot(data_in[0,:,1] ,color='red',alpha=0.8)
#
#
#ax1.plot(data_out[0,:,0],alpha=0.8)



#%% AJuste de curvas



#%%


Is = 1.0*1e-12
Vt = 26.0*1e-3
n = 1.0

vpico = 1

Vd = np.linspace(-vpico,vpico,1000)
Id = Is*(np.exp(Vd/n/Vt)-1)

Rs = 50

plt.plot(Vd,Id)

#Rs = 50
#for i in range(-10,10):
#    Vs = 0.1*i
#    Ir = Vs/Rs - Vd/Rs
#    plt.plot(Vd,Ir,color='blue')
   
Rs = 500    
for i in range(-10,10):
    Vs = 0.1*i
    Ir = Vs/Rs - Vd/Rs
    plt.plot(Vd,Ir,color='red')    
    
 

plt.ylim([-0.2e-2,0.05])




