CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g -Iruntime

.PHONY: test_tensor clean

test_tensor: runtime/tensor_runtime.c runtime/tensor_main.c
	$(CC) $(CFLAGS) -o $@ runtime/tensor_runtime.c runtime/tensor_main.c

clean:
	rm -f test_tensor
