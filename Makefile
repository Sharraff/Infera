CC = gcc
ARCH := $(shell uname -m)
BUILD := build

BASE_CFLAGS := -Wall -Wextra -std=c11 -g -Iruntime -Idispatch -Ioperators -Ikernels

.PHONY: test_tensor clean

OBJS := \
	$(BUILD)/tensor_runtime.o \
	$(BUILD)/tensor_main.o \
	$(BUILD)/opps.o \
	$(BUILD)/dispatch.o \
	$(BUILD)/add_fp32.o \
	$(BUILD)/sub_fp32.o \
	$(BUILD)/div_fp32.o \
	$(BUILD)/mul_fp32.o \
	$(BUILD)/binary_fp16.o \
	$(BUILD)/binary_bf16.o

ifeq ($(ARCH),x86_64)
OBJS += $(BUILD)/binary_fp32_avx2.o
endif

test_tensor: $(OBJS)
	$(CC) $(BASE_CFLAGS) -o $@ $(OBJS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/tensor_runtime.o: runtime/tensor_runtime.c | $(BUILD)
	$(CC) $(BASE_CFLAGS) -c $< -o $@

$(BUILD)/tensor_main.o: runtime/tensor_main.c | $(BUILD)
	$(CC) $(BASE_CFLAGS) -c $< -o $@

$(BUILD)/opps.o: operators/opps.c | $(BUILD)
	$(CC) $(BASE_CFLAGS) -c $< -o $@

$(BUILD)/dispatch.o: dispatch/dispatch.c | $(BUILD)
	$(CC) $(BASE_CFLAGS) -c $< -o $@

$(BUILD)/add_fp32.o: kernels/fp32/add_fp32.c | $(BUILD)
	$(CC) $(BASE_CFLAGS) -c $< -o $@

$(BUILD)/sub_fp32.o: kernels/fp32/sub_fp32.c | $(BUILD)
	$(CC) $(BASE_CFLAGS) -c $< -o $@

$(BUILD)/div_fp32.o: kernels/fp32/div_fp32.c | $(BUILD)
	$(CC) $(BASE_CFLAGS) -c $< -o $@

$(BUILD)/mul_fp32.o: kernels/fp32/mul_fp32.c | $(BUILD)
	$(CC) $(BASE_CFLAGS) -c $< -o $@

$(BUILD)/binary_fp16.o: kernels/fp16/binary_fp16.c | $(BUILD)
	$(CC) $(BASE_CFLAGS) -c $< -o $@

$(BUILD)/binary_bf16.o: kernels/bf16/binary_bf16.c | $(BUILD)
	$(CC) $(BASE_CFLAGS) -c $< -o $@

$(BUILD)/binary_fp32_avx2.o: kernels/fp32/binary_fp32_avx2.c | $(BUILD)
	$(CC) $(BASE_CFLAGS) -mavx2 -mfma -c $< -o $@

clean:
	rm -rf $(BUILD) test_tensor
