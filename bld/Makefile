LIB = libdnn

OBJECTS = blas.o mat.o nn.o state.o

DEPS += libio libalpaca

override SRC_ROOT = ../../src

override CFLAGS += \
	-I../../src/include \
	-I../../src/include/$(LIB)

include ../Makefile.config
override SRC_ROOT = ../../src/$(LIBDNN_BACKEND)
