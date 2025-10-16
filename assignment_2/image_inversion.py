import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import cupy as cp
BLOCK_SIZES = [(8, 8), (16, 16), (32, 32)] 


input_image_path = 'image.jpg'
pil_image = Image.open(input_image_path).convert('L')
image_data = np.asarray(pil_image, dtype=np.uint8)


rows, cols = image_data.shape


invert_kernel = cp.RawKernel(r'''
extern "C" __global__
void invert(const unsigned char* input, unsigned char* output, int rows, int cols) {
    // Get 2D indices based on thread ID and block/grid dimensions
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    // Check bounds
    if (x < cols && y < rows) {
        int index = y * cols + x;
        output[index] = 255 - input[index];
    }
}
''', 'invert')


image_data_gpu = cp.asarray(image_data)
inverted_gpu = cp.empty_like(image_data_gpu)


for block_x, block_y in BLOCK_SIZES:
    grid_x = (cols + block_x - 1) // block_x
    grid_y = (rows + block_y - 1) // block_y
    
    grid_dims = (grid_x, grid_y)
    block_dims = (block_x, block_y)
    
    start_time_gpu = time.perf_counter()
    
    invert_kernel(
        grid_dims, 
        block_dims, 
        (image_data_gpu, inverted_gpu, rows, cols)
    )
    
    cp.cuda.Stream.null.synchronize()
    end_time_gpu = time.perf_counter()
    time_gpu = end_time_gpu - start_time_gpu
    
    key = f"({block_x},{block_y}) - Total Threads: {block_x * block_y}"
    print(f"Block Size {key}: {time_gpu:.6f} seconds")

# The last one is fastest because it maximizes occupancy. 
# This size creates 32 full warps ensuring 100% warp utilization and avoiding wasted cycles by not creating any partial warps.