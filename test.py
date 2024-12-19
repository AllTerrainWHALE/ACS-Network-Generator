# from src.cell import Cell

# c = 0

# c = Cell.setState(c, 1)
# c = Cell.setPheroA(c, 5)
# c = Cell.setPheroB(c, 4)

# print(Cell.getState(c))
# print(Cell.getPheroA(c))
# print(Cell.getPheroB(c))
# print()

# c = Cell.setPheroA(c, 0x7FFFFFFF)

# print(Cell.getState(c))
# print(Cell.getPheroA(c))
# print(Cell.getPheroB(c))
# print()
# print(0x7FFFFFFF)

from numba import cuda
import numpy as np

import os
os.environ["PATH"] += r";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\libnvvm"


@cuda.jit
def add_arrays(a, b, c):
    idx = cuda.grid(1)
    if idx < c.size:
        c[idx] = a[idx] + b[idx]

# Allocate data
n = 1000
a = np.ones(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)
c = np.zeros(n, dtype=np.float32)

# Copy to GPU
a_device = cuda.to_device(a)
b_device = cuda.to_device(b)
c_device = cuda.device_array_like(c)

# Configure the kernel
threads_per_block = 128
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Launch the kernel
add_arrays[blocks_per_grid, threads_per_block](a_device, b_device, c_device)

# Copy result back
c_result = c_device.copy_to_host()
print("Sum result:", c_result[:10])
