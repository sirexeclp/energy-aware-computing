INCLUDES = -I/usr/include/python3.7m
LIBS = -lpython3.7m  -lfftw3 -lcnpy -lz -L../cnpy/build

dft:dft.cpp util.hpp
	g++ $< -o $@ $(INCLUDES) $(LIBS) -O3 -fopenmp

test:dft
	./$<

clean:
	rm dft

validate: dft
	./$< fft 1 file file
	python3 validate.py

.PHONY: test clean validate