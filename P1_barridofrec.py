# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:36:24 2018
"""

#%%
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import threading
import numpy.fft as fft
import datetime
import time
import matplotlib.pylab as pylab
from scipy import signal

params = {'legend.fontsize': 'medium',
     #     'figure.figsize': (15, 5),
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

def play_rec(parametros):
    """
    Descripción:
    ------------
    Esta función permite utilizar la placa de audio de la pc como un generador de funciones / osciloscopio
    con dos canales de entrada y dos de salida. Para ello utiliza la libreria pyaudio y las opciones de write() y read()
    para escribir y leer los buffer de escritura y lectura. Para realizar el envio y adquisición simultánea de señales, utiliza
    un esquema de tipo productor-consumidor que se ejecutan en thread o hilos diferenntes. Para realizar la comunicación
    entre threads y evitar overrun o sobreescritura de los datos del buffer de lectura se utilizan dos variables de tipo block.
    El block1 se activa desde proceso productor y avisa al consumidor que el envio de la señal ha comenzado y que por lo tanto
    puede iniciar la adquisición.
    El block2 se activa desde el proceso consumidor y aviso al productor que la lesctura de los datos ha finalizado y por lo tanto
    puede comenzar un nuevo paso del barrido.
    Teniendo en cuenta que existe un retardo entre la señal enviada y adquirida, y que existe variabilidad en el retardo; se puede
    utilizar el canal 0 de entrada y salida para el envio y adquisicón de una señal de disparo que permita sincronizar las mediciones.
    Notar que cuando se pone output_channel = 1, en la segunda salida pone la misma señal que en el channel 1 de salida.

    Parámetros:
    -----------
    Para el ingreso de los parametros de adquisición se utiliza un diccionario.

    parametros = {}
    parametros['fs'] : int, frecuencia de sampleo de la placa de audio. Valor máximo 44100*8 Hz. [Hz]
    parametros['steps_frec'] : int, cantidad de pasos del barrido de frecuencias.
    parametros['duration_sec_send'] : float, tiempo de duración de la adquisición. [seg]
    parametros['input_channels'] : int, cantidad de canales de entrada.
    parametros['output_channels'] : int, cantidad de canales de salida.
    parametros['tipo_ch0'] : {'square', 'sin', 'ramp', 'constant'}, tipo de señal enviada en el canal 0.
    parametros['amplitud_ch0'] : float, amplitud de la señal del canal 0. [V]. Máximo valor 1 V.
    parametros['frec_ini_hz_ch0'] : float, frecuencia inicial del barrido del canal 0. [Hz]
    parametros['frec_fin_hz_ch0'] : float, frecuencia final del barrido del canal 0. [Hz]
    parametros['tipo_ch1'] : {'square', 'sin', 'ramp'}, tipo de señal enviada en el canal 1.
    parametros['amplitud_ch1'] : float, amplitud de la señal del canal 1. [V]. Máximo valor 1 V.
    parametros['frec_ini_hz_ch1'] : float, frecuencia inicial del barrido del canal 1. [Hz]
    parametros['frec_fin_hz_ch1'] : float, frecuencia final del barrido del canal 1. [Hz]

    Salida (returns):
    -----------------
    data_acq: numpy array, array de tamaño [steps_frec][muestras_por_pasos_input][input_channels]
    data_send: numpy array, array de tamaño [steps_frec][muestras_por_pasos_output][output_channels]
    frecs_send: numpy array, array de tamaño [steps_frec][output_channels]

    Las muestras_por_pasos está determinada por los tiempos de duración de la señal enviada y adquirida. El tiempo entre
    muestras es 1/fs.

    Ejemplo:
    --------

    parametros = {}
    parametros['fs'] = 44100
    parametros['steps_frec'] = 10
    parametros['duration_sec_send'] = 0.3
    parametros['input_channels'] = 2
    parametros['output_channels'] = 2
    parametros['tipo_ch0'] = 'square'
    parametros['amplitud_ch0'] = 0.1
    parametros['frec_ini_hz_ch0'] = 500
    parametros['frec_fin_hz_ch0'] = 500
    parametros['tipo_ch1'] = 'ramp'
    parametros['amplitud_ch1'] = 0.1
    parametros['frec_ini_hz_ch1'] = 500
    parametros['frec_fin_hz_ch1'] = 5000

    data_acq, data_send, frecs_send = play_rec(parametros)


    Autores: Leslie Cusato, Marco Petriella
    """

    # Cargo parametros comunes a los dos canales
    fs = parametros['fs']
    duration_sec_send = parametros['duration_sec_send']
    steps_frec = parametros['steps_frec']
    input_channels = parametros['input_channels']
    output_channels = parametros['output_channels']

    # Estos parametros son distintos par cada canal
    frec_ini_hz = []
    frec_fin_hz = []
    amplitud = []
    delta_frec_hz = []
    tipo = []
    for i in range(output_channels):

        frec_ini_hz.append(parametros['frec_ini_hz_ch' + str(i)])
        frec_fin_hz.append(parametros['frec_fin_hz_ch' + str(i)])
        amplitud.append(parametros['amplitud_ch' + str(i)])
        tipo.append(parametros['tipo_ch' + str(i)])

        if steps_frec == 1:
            delta_frec_hz.append(0.)
            frec_fin_hz[i] = frec_ini_hz[i]
        else:
            delta_frec_hz.append((frec_fin_hz[i]-frec_ini_hz[i])/(steps_frec-1)) # paso del barrido en Hz

    # Obligo a la duracion de la adquisicion > a la de salida
    duration_sec_acq = duration_sec_send + 0.1

    # Inicia pyaudio
    p = pyaudio.PyAudio()

    # Defino los buffers de lectura y escritura
    chunk_send = int(fs*duration_sec_send)
    chunk_acq = int(fs*duration_sec_acq)

    # Defino el stream del parlante
    stream_output = p.open(format=pyaudio.paFloat32,
                    channels = output_channels,
                    rate = fs,
                    output = True,

    )

    # Defino un buffer de lectura efectivo que tiene en cuenta el delay de la medición
    chunk_delay = int(fs*stream_output.get_output_latency())
    chunk_acq_eff = chunk_acq + chunk_delay

    # Defino el stream del microfono
    stream_input = p.open(format = pyaudio.paInt16,
                    channels = input_channels,
                    rate = fs,
                    input = True,
                    frames_per_buffer = chunk_acq_eff*p.get_sample_size(pyaudio.paInt16),
    )

    # Defino los semaforos para sincronizar la señal y la adquisicion
    lock1 = threading.Lock() # Este lock es para asegurar que la adquisicion este siempre dentro de la señal enviada
    lock2 = threading.Lock() # Este lock es para asegurar que no se envie una nueva señal antes de haber adquirido y guardado la anterior
    lock1.acquire() # Inicializa el lock, lo pone en cero.

    # Guardo los parametros de la señal de salida por canal, para usarlos en la funcion function_generator
    parametros_output_signal_chs = []
    for i in range(output_channels):
        para = {}
        para['fs'] = fs
        para['amplitud'] = amplitud[i]
        para['duracion'] = parametros['duration_sec_send']
        para['tipo'] = tipo[i]

        parametros_output_signal_chs.append(para)

    # Defino el thread que envia la señal
    data_send = np.zeros([steps_frec,chunk_send,output_channels],dtype=np.float32)  # aqui guardo la señal enviada
    frecs_send = np.zeros([steps_frec,output_channels])   # aqui guardo las frecuencias

    def producer(steps_frec, delta_frec):
        for i in range(steps_frec):

            # Genero las señales de salida para los canales
            samples = np.zeros([output_channels,4*chunk_send],dtype = np.float32)
            for j in range(output_channels):

                f = frec_ini_hz[j] + delta_frec_hz[j]*i

                parametros_output_signal = parametros_output_signal_chs[j]
                parametros_output_signal['frec'] = f
                samples[j,0:chunk_send] = function_generator(parametros_output_signal)

                # Guardo las señales de salida
                data_send[i,:,j] = samples[j,0:chunk_send]
                frecs_send[i,j] = f

            # Paso la salida a un array de una dimension
            samples_out = np.reshape(samples,4*chunk_send*output_channels,order='F')


            for j in range(output_channels):
                print ('Frecuencia ch'+ str(j) +': ' + str(frecs_send[i,j]) + ' Hz')

            print ('Empieza Productor: '+ str(i))

            # Se entera que se guardó el paso anterior (lock2), avisa que comienza el nuevo (lock1), y envia la señal
            lock2.acquire()
            lock1.release()
            stream_output.start_stream()
            stream_output.write(samples_out)
            stream_output.stop_stream()

        producer_exit[0] = True



    # Defino el thread que adquiere la señal
    data_acq = np.zeros([steps_frec,chunk_acq,input_channels],dtype=np.int16)  # aqui guardo la señal adquirida

    def consumer(steps_frec):
        for i in range(steps_frec):

            # Toma el lock, adquiere la señal y la guarda en el array
            lock1.acquire()
            stream_input.start_stream()
            stream_input.read(chunk_delay)
            data_i = stream_input.read(chunk_acq)
            stream_input.stop_stream()

            data_i = np.frombuffer(data_i, dtype=np.int16)

            for j in range(input_channels):
                data_acq[i,:,j] = -data_i[j::input_channels]

            print ('Termina Consumidor: '+ str(i))
            print ('')
            lock2.release() # Avisa al productor que terminó de escribir los datos y puede comenzar con el próximo step

        consumer_exit[0] = True

    # Variables de salida de los threads
    producer_exit = [False]
    consumer_exit = [False]

    # Inicio los threads
    t1 = threading.Thread(target=producer, args=[steps_frec,delta_frec_hz])
    t2 = threading.Thread(target=consumer, args=[steps_frec])
    t1.start()
    t2.start()


    while(not producer_exit[0] or not consumer_exit[0]):
        time.sleep(0.2)

    stream_input.close()
    stream_output.close()
    p.terminate()

    return data_acq, data_send, frecs_send


#%%

### ANALISIS de la señal adquirida

# Elijo la frecuencia
ind_frec = 0


### Muestra la serie temporal de las señales enviadas y adquiridas
t_send = np.linspace(0,np.size(data_send,1)-1,np.size(data_send,1))/fs
t_adq = np.linspace(0,np.size(data_acq,1)-1,np.size(data_acq,1))/fs

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.15, .15, .75, .8])
ax1 = ax.twinx()
ax.plot(t_send,data_send[ind_frec,:], label=u'Señal enviada: ' + str(frecs_send[ind_frec]) + ' Hz')
ax1.plot(t_adq,data_acq[ind_frec,:],color='red', label=u'Señal adquirida')
ax.set_xlabel('Tiempo [seg]')
ax.set_ylabel('Amplitud [a.u.]')
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()

### Realiza la FFT de la señal enviada y adquirida
fft_send = abs(fft.fft(data_send[ind_frec,:]))
fft_send = fft_send[0:int(chunk_send/2+1)]
fft_acq = abs(fft.fft(data_acq[ind_frec,:]))
fft_acq = fft_acq[0:int(chunk_acq/2+1)]

frec_send = np.linspace(0,int(chunk_send/2),int(chunk_send/2+1))
frec_send = frec_send/duration_sec_send
frec_acq = np.linspace(0,int(chunk_acq/2),int(chunk_acq/2+1))
frec_acq = frec_acq/duration_sec_acq

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.1, .1, .75, .8])
ax1 = ax.twinx()
ax.plot(frec_send,fft_send, label='Frec enviada: ' + str(frecs_send[ind_frec]) + ' Hz')
ax1.plot(frec_acq,fft_acq,color='red', label=u'Señal adquirida')
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud [a.u.]')
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()

#%%

## Estudo del retardo en caso que delta_frec = 0

i_comp = 5

retardos = np.array([])
for i in range(steps):

    data_acq_i = data_acq[i,:]
    corr = np.correlate(data_acq[i_comp,:] - np.mean(data_acq[i_comp,:]),data_acq_i - np.mean(data_acq_i),mode='full')
    pos_max = np.argmax(corr) - len(data_acq_i)
    retardos = np.append(retardos,pos_max/fs)



fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.15, .15, .8, .8])
ax.hist(1000*retardos,bins=1000, rwidth =0.99)
ax.set_xlabel(u'Retardo [ms]')
ax.set_ylabel('Frecuencia [eventos]')
ax.set_title(u'Histograma de retardo respecto a la i = ' + str(i_comp) + ' medición')
ax1.legend(loc=4)
plt.show()


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.15, .15, .8, .8])
ax.hist(retardos*frec_ini_hz,bins=100, rwidth =0.99)
ax.set_xlabel(u'Retardo relativo [periodo]')
ax.set_ylabel('Frecuencia [eventos]')
ax.set_title(u'Histograma de retardo relativo a la duración del período respecto a la i = ' + str(i_comp) + ' medición')
ax1.legend(loc=4)
plt.show()

#%%
plt.plot(np.transpose(data_acq))


plt.plot(np.mean(data_acq[2:,:],axis = 0))



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
