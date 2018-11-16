# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:39:59 2018

@author: Marco
"""

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
from matplotlib import cm
cmap = cm.get_cmap('jet')

from P1_funciones import play_rec_continuo

#%%
fs = 44100*2
frames_per_buffer = 4*256
chunks_buffer = 5000


callback_variables = []

def callback(callback_variables,*args):
    
    input_buffer = args[0]
    output_buffer = args[1]
    i = args[2]
    frames_per_buffer = args[3]
    chunks_buffer = args[4]
    fs = args[5]
    
    output_buffer_i = 0.1*np.sin(2.*np.pi*np.arange(i*frames_per_buffer,(i+1)*frames_per_buffer)*0/fs)
    
    return output_buffer_i



input_buffer, output_buffer = play_rec_continuo(fs,frames_per_buffer,chunks_buffer,callback,callback_variables)



input_buffer_vec = np.reshape(input_buffer,input_buffer.shape[0]*input_buffer.shape[1])
plt.plot(input_buffer_vec)


##################
fs = 44100*2
output_channels = 1
p = pyaudio.PyAudio()

stream_output = p.open(format=pyaudio.paFloat32,
                channels = output_channels,
                rate = fs,
                output = True,                  
)  


output_buffer = input_buffer
output_buffer = output_buffer.astype(np.float32)/np.max(output_buffer)/10
output_buffer_vec = np.reshape(output_buffer,output_buffer.shape[0]*output_buffer.shape[1])
plt.plot(output_buffer_vec)


stream_output.start_stream()
i = 0
while i < output_buffer.shape[0]:
    
    stream_output.write(output_buffer[i,:])
    
    i = i+1

stream_output.stop_stream() 
stream_output.close()
p.terminate()   


############################
import wave
import struct
import array
from scipy.io import wavfile
fs, data = wavfile.read('Pink - Floyd.wav')


chunk = 1024
filas = 5000
output_channels = 1
output_buffer = np.zeros([filas,chunk],dtype=np.float32)

for i in range(filas):
    output_buffer[i,:] = data[i*chunk:(i+1)*chunk,0]
    
#data_vec = np.reshape(output_buffer,output_buffer.shape[0]*output_buffer.shape[1])
#plt.plot(data[:,0])
#plt.plot(data_vec)


output_buffer.astype(np.float32)

p = pyaudio.PyAudio()
stream_output = p.open(format=pyaudio.paFloat32,
                channels = 1,
                rate = fs,
                output = True,                  
)  


stream_output.start_stream()
i = 0
while i < output_buffer.shape[0]:
    
    stream_output.write(output_buffer[i,:])
    
    i = i+1

stream_output.stop_stream() 
stream_output.close()
p.terminate()   

######################


import pyaudio
import wave
import sys

CHUNK = 1024


wf = wave.open('Pink - Floyd.wav', 'rb')

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

data = wf.readframes(CHUNK)

while data != '':
    stream.write(data)
    data = wf.readframes(CHUNK)

stream.stop_stream()
stream.close()

p.terminate()


#%%
todo = np.reshape(input_buffer,input_buffer.shape[0]*input_buffer.shape[1])
plt.plot(todo,alpha=0.5)


#%%
todo = np.reshape(input_buffer,input_buffer.shape[0]*input_buffer.shape[1])
senal =  0.1*np.sin(2.*np.pi*np.arange(256*2*1000)*(1000)/fs)

todo = todo/todo.max()
senal = senal/senal.max()

plt.plot(todo,alpha=0.5)
plt.plot(senal,alpha=0.5)


plt.plot(todo-senal)


#for i in range(input_buffer.shape[0]):
#    plt.axvline(i*input_buffer.shape[1],linestyle='--',color='red')


#%%

output_channels = 1

p = pyaudio.PyAudio()

stream_output = p.open(format=pyaudio.paFloat32,
                channels = output_channels,
                rate = fs,
                output = True,                  
    ) 

todo_out = todo/todo.max()
todo_out = todo_out.astype(np.float32)

stream_output.start_stream()
stream_output.write(todo_out)
stream_output.stop_stream()    

stream_output.close()
p.terminate() 

#%%
plt.plot(todo_out)