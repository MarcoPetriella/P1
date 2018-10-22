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
fs = 44100

input_buffer, output_buffer = play_rec_continuo(fs)


#%%
todo = np.reshape(input_buffer,input_buffer.shape[0]*input_buffer.shape[1])

plt.plot(input_buffer[1,:])

plt.plot(todo)
for i in range(input_buffer.shape[0]):
    plt.axvline(i*input_buffer.shape[1],linestyle='--',color='red')


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