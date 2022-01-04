
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as signal

from scipy.fft import *


def read_signal(filename: str, s_no: int):

	signal = np.transpose(sio.loadmat("FILTER.mat")["input_"+str(s_no)])
	n_axis = np.arange(0, len(signal))
	return n_axis, signal

def graph(values, title="", xlabel="", ylabel="", color="b"):

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(*values, color)
	plt.show()

def hamming_filter(order: int, cut_off: float, domain: str):

	n_axis = np.arange(0, order)
	flt = signal.firwin(order, cut_off, window="hamming")

	if domain == "time": return n_axis, flt
	elif domain == "frequency": return signal.freqz(flt)
	else: raise TypeError("Invalid Domain")

def first_peak(array):
	peak = 0
	for idx in range(0, len(array)):
		if peak < array[idx]: peak = idx
		elif array[idx] < peak: return idx