import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import cupy as cp


# --- Load the image ---
input_image_path = 'image.jpg'
pil_image = Image.open(input_image_path).convert('RGB')
image_data = np.asarray(pil_image)


# --- CPU Inversion (NumPy) ---
print("\nRunning CPU inversion...")
start_time_cpu = time.perf_counter()
inverted_cpu = 255 - image_data
end_time_cpu = time.perf_counter()
time_cpu = end_time_cpu - start_time_cpu


# --- GPU Inversion (CuPy) ---


print("Running GPU inversion...")
image_data_gpu = cp.asarray(image_data)
start_time_gpu = time.perf_counter()
inverted_gpu = 255 - image_data_gpu
cp.cuda.Stream.null.synchronize()
end_time_gpu = time.perf_counter()
time_gpu = end_time_gpu - start_time_gpu
inverted_gpu_host = cp.asnumpy(inverted_gpu)


# --- Display Images and Compare Times ---
print("\n--- Performance Comparison ---")
print(f"NumPy (CPU) Time: {time_cpu:.6f} seconds")


print(f"CuPy (GPU) Time: {time_gpu:.6f} seconds")
speedup = time_cpu / time_gpu
print(f"GPU Speedup: {speedup:.2f}x")
is_identical = np.array_equal(inverted_cpu, inverted_gpu_host)
print(f"CPU and GPU outputs are identical: {is_identical}")


# Display the images
fig, axes = plt.subplots(1, 3, figsize=(18, 6))


axes[0].imshow(image_data)
axes[0].set_title("Original Image")
axes[0].axis('off')


axes[1].imshow(inverted_cpu)
axes[1].set_title("CPU Inverted (NumPy)")
axes[1].axis('off')


axes[2].imshow(inverted_gpu_host)
axes[2].set_title("GPU Inverted (CuPy)")
axes[2].axis('off')


plt.tight_layout()
plt.show()
