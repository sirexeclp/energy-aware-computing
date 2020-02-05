#include "../matplotlib-cpp/matplotlibcpp.h"
#include <complex>
#include <cmath>
#include <iostream>
namespace plt = matplotlibcpp;
using namespace std::complex_literals;

bool is_power2(int number)
{
    if (number == 0)
        return false;
    if ((number & (number - 1)) == 0)
        return true;
    else
        return false;
}