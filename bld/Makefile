LIB = libdnn

OBJECTS = mat.o nn.o state.o \
			$(LIBDNN_BACKEND)/nonlinear.o \
			$(LIBDNN_BACKEND)/blas.o 

$(info $(OBJECTS))

DEPS += libio libalpaca

override SRC_ROOT = ../../src

override CFLAGS += \
	-I../../src/include \
	-I../../src/include/$(LIB)
