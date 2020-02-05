import numpy as np
import matplotlib.pyplot as plt


def is_power2(number):
    if number == 0:
        return False
    if (number & (number - 1)) == 0:
        return True
    else:
        return False

def reverse_bits(n):
    n = (n >>  1) & 0x55555555 | (n << 0x01) & 0xaaaaaaaa
    n = (n >>  2) & 0x33333333 | (n <<  2) & 0xcccccccc
    n = (n >>  4) & 0x0f0f0f0f | (n <<  4) & 0xf0f0f0f0
    n = (n >>  8) & 0x00ff00ff | (n <<  8) & 0xff00ff00
    n = (n >> 16) & 0x0000ffff | (n << 16) & 0xffff0000
    return n

def sort_bit_reverse(signal):
    assert is_power2(len(signal)), "Length of `signal` must be a power of 2."
    result = np.copy(signal)
    shift_factor = 32 - np.log2(len(result))
    for i in range(1,len(result)):
        j=reverse_bits(i) >> int(shift_factor)
        if j!=i:
            result[i], result[j] = signal[j],signal[i]
    return result

def fft(signal):
    result = np.array(sort_bit_reverse(np.copy(signal)),dtype=complex)
    for s in range(1,1+int(np.log2(len(signal)))):
        m = 1<<s
        m2 = m>>1
        w=complex(1,0)
        wm = np.exp(-1j*(np.pi/m2))
        j=0
        while j< m2:
            k=j
            while k < len(signal):
                u = result[k]
                t = w*result[k+m2]
                result[k] =u+t
                result[k+m2]=u-t
                k+=m
            w*=wm
                
            j+=1
    return result

def _fft_rec_k(signal,k):
    if len(signal) == 1:
        return signal[0]
    W=np.exp(-1j*2*np.pi*k/len(signal))
    a,b = signal[:len(signal)//2], signal[len(signal)//2:]
    #print(f"a={a}|b={b}")
    return _fft_rec_k(a, k) + W * _fft_rec_k(b, k)

def fft_rec(signal):
    result = np.array(sort_bit_reverse(signal),dtype=complex)
    result = [_fft_rec_k(result,k) for k,_ in enumerate(signal)]
    return np.array(result)