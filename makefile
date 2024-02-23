COMPILER = nvcc
CFLAGS = -g -G
COBJS = bmpfile.o
CEXES =  mandelbrot

all: ${CEXES}

mandelbrot: mandelbrot.cu ${COBJS}
	${COMPILER} ${CFLAGS} mandelbrot.cu ${COBJS} -o mandelbrot -lm

%.o: %.c %.h  makefile
	${COMPILER} ${CFLAGS} -lm $< -c

clean:
	rm -f *.o *~ ${CEXES}

run:
	./mandelbrot 1920 1080