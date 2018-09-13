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

carpeta_salida = 'Calibracion'
subcarpeta_salida = '1'
if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))     
    
#%%

# Genero matriz de señales: ejemplo de barrido en frecuencias en el canal 0
fs = 44100*8  
duracion = 0.5
muestras = int(fs*duracion)
input_channels = 2
output_channels = 2
amplitud = 1
valor_rms = 0.7 # en V. Depende del nivel de volumen de parlante: 0.7 para 60/100
amplitud_V = amplitud*valor_rms*np.sqrt(2) #V
frec_ini = 1000
frec_fin = 1000
pasos = 2
delta_frec = (frec_fin-frec_ini)/(pasos+1)
data_out = np.zeros([pasos,muestras,output_channels])

for i in range(pasos):
    parametros_signal = {}
    fs = fs
    amp = amplitud
    fr = frec_ini + i*delta_frec
    duration = duracion
    
    if i == 0:
        output_signal = signalgen('sine',fr,amp,duration,fs)
        data_out[i,:,0] = output_signal
        
        output_signal = signalgen('sine',fr,amp,duration,fs)
        data_out[i,:,1] = output_signal

    if i == 1:
        output_signal = signalgen('ramp',fr,amp,duration,fs)
        data_out[i,:,0] = output_signal
        
        output_signal = signalgen('ramp',fr,amp,duration,fs)
        data_out[i,:,1] = output_signal

# Realiza medicion
offset_correlacion = 0#int(fs*(1))
steps_correlacion = 0#int(fs*(1))
data_in, retardos = play_rec(fs,input_channels,data_out,'si',offset_correlacion,steps_correlacion)

np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'data_out'),data_out)
np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'data_in'),data_in)



#%%

# CAlibracion para nivel de parlante 70, y nivel de microfono de 80
# En este valor de nivel de parlante Amplitud = 1 ==> 1V de amplitud
# En este nivel de parlante la señal de amplitud 1 (1V) ocupa todo el rango de medición
# Esto vale para la placa de pc de escritorio de casa de Marco y windows 10

data_out = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'data_out.npy'))
data_in = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'data_in.npy'))

formas = ['Seno','Rampa']
canales = ['CH0','CH1']

for i in range(2):
    
    for j in range(2):
        

        ajuste = np.polyfit(data_out[i,int(fs*0.1):-int(fs*0.1),j]/amplitud*amplitud_V,data_in[i,int(fs*0.1):-int(fs*0.1),j],1)
        
        
        fig = plt.figure(figsize=(14, 7), dpi=250)
        ax = fig.add_axes([.12, .15, .75, .8])        
        ax.plot(data_out[i,int(fs*0.1):-int(fs*0.1),j]/amplitud*amplitud_V,data_in[i,int(fs*0.1):-int(fs*0.1),j],'--',color='blue',alpha=0.8,label='Señal')
        ax.plot(data_out[i,int(fs*0.1):-int(fs*0.1),j]/amplitud*amplitud_V,data_out[i,int(fs*0.1):-int(fs*0.1),j]/amplitud*amplitud_V*ajuste[0]+ajuste[1],'--',color='red',alpha=0.8,label='Ajuste')
        ax.grid(linestyle='--')
        ax.legend(loc=4)
        ax.text(0.1,0.8,'Ajuste: ax + b', transform=ax.transAxes)
        ax.text(0.1,0.75,'a: ' '{:6.2e}'.format(ajuste[0]) + ' [cuentas/V]', transform=ax.transAxes)
        ax.text(0.1,0.70,'b: ' '{:6.2e}'.format(ajuste[1]) + ' [cuentas]', transform=ax.transAxes)
        ax.set_xlabel('Señal enviada [V]')
        ax.set_ylabel('Señal recibida [cuentas]')
        ax.set_title(u'Señal enviada y adquirida en ' + canales[j] + ' utilizando función ' + formas[i] + '. Nivel de parlante en 60/100 y microfono 80/100' )
        figname = os.path.join(carpeta_salida,subcarpeta_salida, 'ajuste_'+canales[j]+ '_'+formas[i]+'.png')
        fig.savefig(figname, dpi=300)  
        plt.close(fig)

        
        
        fig = plt.figure(figsize=(14, 7), dpi=250)
        ax = fig.add_axes([.12, .15, .75, .8])        
        ax.plot(data_out[i,int(fs*0.1):-int(fs*0.1),j]/amplitud*amplitud_V,(data_in[i,int(fs*0.1):-int(fs*0.1),j]-ajuste[1])/ajuste[0],'--',color='red')       
        ax.set_xlabel('Señal enviada [V]')
        ax.set_ylabel('Señal recibida [V]')   
        ax.grid(linestyle='--')
        ax.set_title(u'Señal enviada y adquirida en ' + canales[j] + ' utilizando función ' + formas[i] + '. Nivel de parlante en 60/100 y microfono 80/100' )
        figname = os.path.join(carpeta_salida,subcarpeta_salida, 'conversion_'+canales[j]+ '_'+formas[i]+'.png')
        fig.savefig(figname, dpi=300)  
        plt.close(fig)        

        
        
#%% Medimos linealidad en amplitud
#Linealidad para seno

data_out = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'data_out.npy'))
data_in = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'data_in.npy'))


lin_sen_ch0 = data_in[0,:,0]/data_out[0,:,0]
lin_sen_ch1 = data_in[0,:,1]/data_out[0,:,1]
lin_ramp_ch0 = data_in[1,:,0]/data_out[1,:,0]
lin_ramp_ch1 = data_in[1,:,1]/data_out[1,:,1]

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax.plot(data_out[0,int(fs*0.1):-int(fs*0.1),0],data_in[0,int(fs*0.1):-int(fs*0.1),0],'.',label='Seno en CH0',markersize=1)
ax.plot(data_out[0,int(fs*0.1):-int(fs*0.1),1],data_in[0,int(fs*0.1):-int(fs*0.1),1],'.',label='Seno en CH1',markersize=1)
ax.plot(data_out[1,int(fs*0.1):-int(fs*0.1),0],data_in[1,int(fs*0.1):-int(fs*0.1),0],'.',label='Rampa en CH0',markersize=1)
ax.plot(data_out[1,int(fs*0.1):-int(fs*0.1),1],data_in[1,int(fs*0.1):-int(fs*0.1),1],'.',label='Rampa en CH1',markersize=1)
ax.set_xlabel('señal enviada')
ax.set_ylabel('señal recibida [u.a.]')
ax.legend(loc=2)
ax.grid(linestyle='--')
figname = os.path.join(carpeta_salida,subcarpeta_salida, 'calibracion.png')
fig.savefig(figname, dpi=300)  
plt.close(fig)