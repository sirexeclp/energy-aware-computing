#include "../matplotlib-cpp/matplotlibcpp.h"
#include <complex>
#include <cmath>
#include <iostream>
#include <assert.h>
#include <fftw3.h>
#include <string>
#include <map>
#include "util.hpp"
#include"../cnpy/cnpy.h"
namespace plt = matplotlibcpp;
using namespace std::complex_literals;
// std::vector<double> range(begin)

/* 
benchmark mit 
int n = 50 000;

g++ dft.cpp -o dft -I/usr/include/python3.7m -lpython3.7m
    267,92s user 0,49s system 100% cpu 4:27,82 total

g++ dft.cpp -o dft -I/usr/include/python3.7m -lpython3.7m -O3 
    78,26s user 0,48s system 100% cpu 1:18,02 total

g++ dft.cpp -o dft -I/usr/include/python3.7m -lpython3.7m -O3 -fopenmp
    141,00s user 0,60s system 695% cpu 20,346 total
*/

//TODO implement 12.2.2/ benchmark vs. naive aproach
std::vector<std::complex<double>> dft(std::vector<std::complex<double>> signal)
{
    std::vector<std::complex<double>> fourier_transform(signal.size());
    const std::complex<double> W = 2 * M_PI * 1i / (double)signal.size();
#pragma omp parallel for
    for (int index = 0; index < signal.size(); ++index)
    {
        std::complex<double> sum = 0;
        for (int k = 0; k < signal.size(); ++k)
        {
            sum += signal[k] * std::exp(W * (double)k * (double)index);
        }
        fourier_transform.at(index) = sum;
    }
    return fourier_transform;
}


std::vector<double> dift(std::vector<std::complex<double>> fourier_transform)
{
    std::vector<double> signal(fourier_transform.size());
#pragma omp parallel for
    for (int index = 0; index < fourier_transform.size(); ++index)
    {
        std::complex<double> sum = 0;
        for (int k = 0; k < fourier_transform.size(); ++k)
        {
            sum += fourier_transform[k] * std::exp(-2 * M_PI * 1i * (double)k * (double)index / (double)fourier_transform.size());
        }
        signal.at(index) = sum.real() / fourier_transform.size();
    }
    return signal;
}

std::vector<std::complex<double>> fft(std::vector<std::complex<double>> signal, bool inverse = false)
{
    assert(("Length of `signal` must be a power of 2.", is_power2(signal.size())));
    double sign = inverse ? -1. : 1.;
    std::vector<std::complex<double>> result(signal);
    //bit-reversed ordering
    result = sort_bit_reversed(signal);
    std::cerr << "ordering done" << std::endl;
    //actual fourier transform
    int max_ebene = std::log2(signal.size());
    for (int ebene = 1; ebene <= max_ebene; ebene++)
    {

        const double m = std::pow(2, ebene);
        const int m_i = m;
        std::complex<double> wm = std::exp(-2 * M_PI * 1i / m);


        for (int abschnitt = 0; abschnitt < signal.size(); abschnitt += m_i)
        {
            std::complex<double> w = 1.;
            for (int element = 0; element < m / 2; element++)
            {
                std::complex<double> u = result[abschnitt + element];
                std::complex<double> t = w * result[abschnitt + element + m / 2];
                result[abschnitt + element] = u + t;
                result[abschnitt + element + m / 2] = u - t;
                w = w * wm;
            }
        }
    }
    return result;
}

std::complex<double> _fft_r(std::vector<std::complex<double>>::iterator begin, std::vector<std::complex<double>>::iterator end,int k)
{  
    auto length = (end-begin);
    if(length==1)
        return *begin;
    
    std::complex<double> w = std::exp(-2 * M_PI * 1i *(double)k/(double)length);
    // if((end-begin) == 4)
    // {
    //     std::vector<std::complex<double>> tmp(begin,end);
    //     auto result = dft(tmp);
    //     std::copy(result.begin(),result.end(),begin);

    // }   
    auto even = _fft_r(begin,begin+(length/2),k);
    auto odd = _fft_r(begin+(length/2),end,k);
    return even + w * odd;
}


std::vector<std::complex<double>> fft_r(std::vector<std::complex<double>> signal, bool inverse = false)
{
    assert(("Length of `signal` must be a power of 2.", is_power2(signal.size())));
    double sign = inverse ? -1. : 1.;
    std::vector<std::complex<double>> result(signal.size());
    //bit-reversed ordering
    auto reversed = sort_bit_reversed(signal);
    //actual fourier transform
    #pragma omp parallel for
    for (int k = 0; k< signal.size(); k++)
        result[k] = _fft_r(reversed.begin(),reversed.end(),k);
    return result;
}

std::vector<std::complex<double>> fftw3(std::vector<std::complex<double>> signal)
{
    std::vector<std::complex<double>> result(signal.size());
    auto plan = fftw_plan_dft_1d(signal.size(), reinterpret_cast<fftw_complex *>(signal.data()), reinterpret_cast<fftw_complex *>(result.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    fftw_cleanup();
    return result;
}
std::vector<std::complex<double>> fftw3_t(std::vector<std::complex<double>> signal)
{
    int fftw_threads_init(void);
    std::vector<std::complex<double>> result(signal.size());
    auto plan = fftw_plan_dft_1d(signal.size(), reinterpret_cast<fftw_complex *>(signal.data()), reinterpret_cast<fftw_complex *>(result.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    fftw_cleanup();
    return result;
}

// std::vector<std::complex<double>> srft(std::vector<std::complex<double>> signal)
// { 
//     int n2 = 2*signal.size();
//     for (int ebene = 1; ebene <= std::log2(signal.size()); ebene++)
//     {
//         n2 = n2/2;
//         int n4 = n2/4;
//         std::complex<double> wm = std::exp(2 * M_PI * 1i / n2);
//         for (int abschnitt = 0; abschnitt < n4; abschnitt += m_i)
//         {
//     }

// }
// void do_nothing(std::vector<std::complex<double>> dummy)
// {
//     return;
// }
void help(char *argv[])
{
    std::cout << "usage: ./" << argv[0] << " [dft|fft|fftw] 2^signal_length [stdin|file|none] [stdout|file|plot_show|plot_save|none]" << std::endl;
}
int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        help(argv);
        return 1;
    }
    auto algorithm = std::string(argv[1]);
    int n = 1 << atoi(argv[2]);
    auto input_format = std::string(argv[3]);
    auto output_format = std::string(argv[4]);


    std::vector<double> x(n);
    std::vector<std::complex<double>> y(n), F;

    if (input_format == "none")
    {
    #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            x.at(i) = i;
            y.at(i) = sin(2 * M_PI * i / 360.0); //+sin(2*M_PI*i/36.0);
        }
    }else if(input_format =="stdin")
    {
        std::string ts;
        for (int i = 0; i < n; ++i)
        {
            x.at(i) = i;
            std::getline(std::cin, ts);
            std::istringstream(ts) >> y[i];
        }
    }else if(input_format=="file")
    {
        cnpy::NpyArray input_data = cnpy::npy_load("test_data.npy");
        auto begin = input_data.data<std::complex<double>>();
        auto size = input_data.num_bytes()/(input_data.word_size);
        std::cerr <<"word_size:" <<input_data.word_size << std::endl <<"input length:"<< size<<std::endl;
        assert(("Expected numpy array of type complex128 (word_size 16 byte)", input_data.word_size == 16));
        
        y = std::vector<std::complex<double>>(begin,begin+size);
        n = size;
        x.resize(size);
        for (int i = 0; i < size; ++i)
        {
            x.at(i) = i;
        }
    }
    else{
        std::cout << "invalid input_format" << std::endl;
         help(argv);
        return 2;
    }
    //     std::map<std::string, (*std::vector<std::complex<double>>)(std::vector<std::complex<double>>)> algorithm = {
    //     { "dft", &dft },
    //     { "fft", &fft },
    //     { "fftw", &fftw3 }
    // };
    //     auto fourier_transform = dft(y);
    //     auto inverse_f = dift(fourier_transform);
    //     fft(y);
    //         // Set the size of output image to 1200x780 pixels
    //     // plt::figure_size(1200, 780);
    //     // Plot line from given x and y data. Color is selected automatically.
    //    plt::plot(x,y,"bx");
    //    plt::plot(x,inverse_f);
    if (algorithm == "dft")
    {
        std::cerr << "running dft" << std::endl;
        F = dft(y);
    }
    else if (algorithm == "fft")
    {
        std::cerr << "running fft" << std::endl;
        F = fft(y);
    }
        else if (algorithm == "fftr")
    {
        std::cerr << "running fft_recursive" << std::endl;
        F = fft_r(y);
    }
    else
    {
        std::cerr << "running fftw" << std::endl;
        F = fftw3(y);
    }

    //output
    if(output_format == "stdout")
    {
        for(auto item:F)
            std::cout << item <<"\n";
    }
    else if (output_format == "plot_show")
    {  
        plt::xkcd();
        plt::plot(x, spectral_power(F));
        plt::show();
    } else if (output_format == "plot_save")
    {
        plt::xkcd();
        plt::plot(x, spectral_power(F));
        plt::save("test.png");
    }else if (output_format == "file")
    {
        std::string filename = "output.npy";
        cnpy::npy_save(filename, F.data(),{F.size()},"w");
    }
}
