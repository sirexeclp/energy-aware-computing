INCLUDES = -I/usr/include/python3.7m
LIBS = -lpython3.7m

dft:dft.cpp
	g++ $< -o $@ $(INCLUDES) $(LIBS)
	#-O3 
	#-fopenmp

test:dft
	./$<

clean:
	rm dft

.PHONY: test clean