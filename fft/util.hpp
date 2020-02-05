std::vector<double> spectral_power(std::vector<std::complex<double>> fourier_transform)
{
    std::vector<double> power(fourier_transform.size());
#pragma omp parallel for
    for (int i = 0; i < fourier_transform.size(); ++i)
    {
        power.at(i) = 2 * std::pow(fourier_transform[i].real(), 2);
    }
    return power;
}

bool is_power2(int number)
{
    if (number == 0)
        return false;
    if ((number & (number - 1)) == 0)
        return true;
    else
        return false;
}

// https://helloacm.com/how-to-reverse-bits-for-32-bit-unsigned-integer-in-cc/
uint32_t reverseBits(uint32_t n)
{
    n = (n >> 1) & 0x55555555 | (n << 1) & 0xaaaaaaaa;
    n = (n >> 2) & 0x33333333 | (n << 2) & 0xcccccccc;
    n = (n >> 4) & 0x0f0f0f0f | (n << 4) & 0xf0f0f0f0;
    n = (n >> 8) & 0x00ff00ff | (n << 8) & 0xff00ff00;
    n = (n >> 16) & 0x0000ffff | (n << 16) & 0xffff0000;
    return n;
}

std::vector<std::complex<double>> sort_bit_reversed(std::vector<std::complex<double>> signal)
{
    assert(("Length of `signal` must be a power of 2.", is_power2(signal.size())));
    //bit-reversed ordering
    std::vector<std::complex<double>> result(signal.size());
    const int shift_factor = 32 - std::log2(signal.size());
    #pragma omp parallel for
    for (int i = 0; i < signal.size(); i++)
    {
        int j = reverseBits(i) >> shift_factor;
        result[j] = signal[i];
        result[i] = signal[j];
            // std::swap(signal[j], signal[i]);
    }
    return result;
}