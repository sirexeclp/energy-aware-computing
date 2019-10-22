TASKS=a b c

colorfilter: colorfilter.c
	cc -fopenmp -O3 $< -o $@

$(TASKS): colorfilter apples.bmp
	./colorfilter $@ apples.bmp
	$(open output.bmp)
	