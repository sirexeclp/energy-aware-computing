#include "../matplotlib-cpp/matplotlibcpp.h"
#include <complex>
#include <cmath>
#include <iostream>
namespace plt = matplotlibcpp;
using namespace std::complex_literals;
// std::vector<double> range(begin)

/* 
benchmark mit 
int n = 50000;

g++ dft.cpp -o dft -I/usr/include/python3.7m -lpython3.7m
    267,92s user 0,49s system 100% cpu 4:27,82 total

g++ dft.cpp -o dft -I/usr/include/python3.7m -lpython3.7m -O3 
    78,26s user 0,48s system 100% cpu 1:18,02 total

g++ dft.cpp -o dft -I/usr/include/python3.7m -lpython3.7m -O3 -fopenmp
    141,00s user 0,60s system 695% cpu 20,346 total
*/

std::vector<std::complex<double>> dft(std::vector<double> signal)
{
    std::vector<std::complex<double>> fourier_transform(signal.size());
    #pragma omp parallel for
    for(int index=0; index<signal.size(); ++index) {
        std::complex<double> sum = 0;
        for(int k=0; k<signal.size(); ++k) {
            sum += signal[k] * std::exp(2*M_PI*1i*(double)k*(double)index/(double)signal.size());
        }
        fourier_transform.at(index) = sum;
    }
    return fourier_transform;
}
std::vector<double> dift(std::vector<std::complex<double>> fourier_transform)
{
    std::vector<double> signal(fourier_transform.size());
    #pragma omp parallel for
    for(int index=0; index<fourier_transform.size(); ++index) {
        std::complex<double> sum = 0;
        for(int k=0; k<fourier_transform.size(); ++k) {
            sum += fourier_transform[k] * std::exp(-2*M_PI*1i*(double)k*(double)index/(double)fourier_transform.size());
        }
        signal.at(index) = sum.real()/fourier_transform.size();
    }
    return signal;
}

std::vector<double> spectral_power(std::vector<std::complex<double>> fourier_transform)
{
    std::vector<double> power(fourier_transform.size());
    #pragma omp parallel for
    for(int i=0; i<fourier_transform.size(); ++i) {
        power.at(i) = 2*std::pow(fourier_transform[i].real(),2);
    }
    return power;
}

int main() {
    // Prepare data.
    int n = 500;
    std::vector<double> x(n), y(n);
    std::vector<std::complex<double>> F(n);
    #pragma omp parallel for
    for(int i=0; i<n; ++i) {
        x.at(i) = i;
        y.at(i) = sin(2*M_PI*i/360.0);
    }

    auto fourier_transform = dft(y);
    auto inverse_f = dift(fourier_transform);

        // Set the size of output image to 1200x780 pixels
    // plt::figure_size(1200, 780);
    // Plot line from given x and y data. Color is selected automatically.
   plt::plot(x,y,"bx");
   plt::plot(x,inverse_f);
   
    // plt::plot(x, spectral_power(fourier_transform));
    // plt::save("test.png");
     plt::show();
}
