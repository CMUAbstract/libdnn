LIB = libdnn

OBJECTS = nn.o state.o linalg.o buffer.o profile.o \
		$(LIBDNN_BACKEND)/nonlinear.o \
		$(LIBDNN_BACKEND)/blas.o 

$(info $(OBJECTS))
$(info $(CFLAGS))

DEPS = libio libalpaca libfixed libmat

override SRC_ROOT = ../../src

override CFLAGS += \
	-I../../src/include \
	-I../../src/include/$(LIB)

include $(MAKER_ROOT)/Makefile.$(TOOLCHAIN)
