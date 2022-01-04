
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as signal

from scipy.fft import *
from util import read_signal, graph, hamming_filter, first_peak


n0_axis, input_signal = read_signal("FILTER.mat", 5)
w0_axis, input_signal_fft = signal.freqz(input_signal, whole=True)
w0_axis = fftshift(fftfreq(512, 1))
w0_axis *= np.pi/w0_axis.max()



 # PLOT INPUT SIGNAL IN TIME AND FREQUENCY DOMAINS

plt.subplot(2, 1, 1)
plt.title("INPUT SIGNAL PLTOS")
plt.xlabel("[n]")
plt.ylabel("s[n] TIME DOMAIN")
plt.plot(n0_axis, input_signal, "blue")

plt.subplot(2, 1, 2)
plt.xlabel("ω")
plt.ylabel("|S(ω)| FREQUENCY DOMAIN")
plt.plot(w0_axis, np.abs(input_signal_fft), "red")

plt.show()

cut_off = 0.0125
order = 500

n1_axis, flt = hamming_filter(order, cut_off, domain="time")
w1_axis, flt_fft = hamming_filter(order, cut_off, domain="frequency")

plt.subplot(3, 1, 1)
plt.title("FILTER REPONSE PLOTS")
plt.xlabel("ω")
plt.ylabel("|H(ω)|")
plt.plot(w1_axis, np.abs(flt_fft), "blue")

plt.subplot(3, 1, 2)
plt.xlabel("ω")
plt.ylabel("|H(ω)dB|")
plt.plot(w1_axis, order/2*np.log10(abs(flt_fft)), "green")

plt.subplot(3, 1, 3)
plt.xlabel("ω")
plt.ylabel("<|H(ω)|")
plt.plot(w1_axis, np.angle(flt_fft), "orange")

plt.show()

plt.stem(signal.windows.get_window("hamming", 500))
plt.title("FILTER IMPULSE RESPONSE (TIME DOMAIN)")
plt.xlabel("[n]")
plt.ylabel("h[n]")
plt.show()


filtered_signal_fft = flt_fft * input_signal_fft
filtered_signal = ifft(flt_fft * input_signal_fft)

plt.subplot(2, 1, 1)
plt.title("FILTERED SIGNAL PLOTS")
plt.xlabel("ω")
plt.ylabel("|Y(ω)| FREQUENCY DOMAIN")
plt.plot(w1_axis, np.abs(filtered_signal_fft), "red")

plt.subplot(2, 1, 2)
plt.xlabel("[n]")
plt.ylabel("y[n] TIME DOMAIN")
plt.plot(n0_axis, filtered_signal.real[:501], "orange")


plt.show()
