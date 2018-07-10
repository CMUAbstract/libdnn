LIB = libdnn

OBJECTS = nn.o state.o linalg.o buffer.o profile.o cleanup.o misc.o \
		$(LIBDNN_BACKEND)/nonlinear.o \
		$(LIBDNN_BACKEND)/task_ds_zero.o $(LIBDNN_BACKEND)/task_ds_add.o \
		$(LIBDNN_BACKEND)/task_ds_mul.o $(LIBDNN_BACKEND)/task_ds_div.o \
		$(LIBDNN_BACKEND)/task_dm_add.o $(LIBDNN_BACKEND)/task_dm_mul.o \
		$(LIBDNN_BACKEND)/task_dm_conv.o $(LIBDNN_BACKEND)/task_sm_mul.o \
		$(LIBDNN_BACKEND)/task_svm_mul.o $(LIBDNN_BACKEND)/task_sm_conv.o

DEPS = libio libalpaca libfixed libmat

override SRC_ROOT = ../../src

override CFLAGS += \
	-I../../src/include \
	-I../../src/include/$(LIB)

include $(MAKER_ROOT)/Makefile.$(TOOLCHAIN)
