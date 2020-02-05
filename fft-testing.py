# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from numba import cuda


@cuda.jit
def add_kernel(x, y, out):
    tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
    ty = cuda.blockIdx.x  # Similarly, this is the unique block ID within the 1D grid

    block_size = cuda.blockDim.x  # number of threads per block
    grid_size = cuda.gridDim.x    # number of blocks in the grid
    
    start = tx + ty * block_size
    stride = block_size * grid_size

    # assuming x and y inputs are same length
    for i in range(start, x.shape[0], stride):
        out[i] = x[i] + y[i]


# +
import numpy as np

n = 100000000
x = np.arange(n).astype(np.float32)
y = 2 * x
out = np.empty_like(x)

threads_per_block = 128
blocks_per_grid = 30

add_kernel[blocks_per_grid, threads_per_block](x, y, out)
print(out[:10])
# -

import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

coeffs = sig.cheby1(4,0.001,[0.001,0.3],btype="bandpass")
len(coeffs[1])

# ?sig.butter
coeffs2= sig.butter(4,0.1)
coeffs2

x_range = np.arange(0,10,1/(10*np.pi))
x = np.sin(x_range) +np.sin(x_range*2-10)#+ np.random.normal(0,0.1,len(x_range)) +np.sin(x_range*10)
plt.plot(x)

fft = np.fft.fft(x)
x2 = np.fft.ifft(fft[:100])
x3 = np.fft.ifft(np.pad(fft,(0,len(fft)), mode="constant", constant_values = 0))
plt.plot(x2)

y = sig.lfilter(*coeffs,x)/2
plt.plot(y)

fft = np.fft.fft(x)
plt.plot(abs(fft))

len(coeffs[0])
len(fft)-len(coeffs[0])
#len(np.pad(coeffs[0],(0,len(fft)-len(coeffs[0])), mode="constant"))
filtfft = fft*np.pad(coeffs[0],(0,len(fft)-len(coeffs[0])), mode="constant")
filtfft =  filtfft * np.pad(
    coeffs[1],(0,len(fft)-len(coeffs[1])), mode="constant", constant_values = 0)
plt.plot(np.fft.ifft(filtfft))

np.fft.ifft(np.fft.fft(signal) * np.fft.fft(coeffs[0])/np.fft.fft(coeffs[1]))

sig.cheby(5,.5)

# ?sig.cheby1

test_imp = np.ones((6000))
plt.plot(abs(np.fft.fft(sig.lfilter(*coeffs, test_imp)))[:100])

coeffs = sig.butter(4,0.5)

impulse = np.zeros(200)
impulse [10] =1
impulse_resp = sig.lfilter(*coeffs, impulse)
plt.plot(impulse_resp)

plt.plot((np.absolute(np.fft.fft(impulse_resp)[:100])))

w, h = sig.freqz(*coeffs)
x = w * 200 * 1.0 / (2 * np.pi)
y = 20 * np.log10(abs(h))
plt.figure(figsize=(10,5))
plt.semilogx(x, y)
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.title('Frequency response')
plt.grid(which='both', linestyle='-', color='grey')
plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ["20", "50", "100", "200", "500", "1K", "2K", "5K", "10K", "20K"])
plt.show()

my_filter = np.zeros(200)
my_filter[10:50] = 1000

filt = np.fft.ifft(my_filter)

plt.plot(my_filter)


