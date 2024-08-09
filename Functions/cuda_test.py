import tensorflow as tf
from gpuinfo import GPUInfo

# Testing GPU connection ------------------------------------------------------------------------------------------------------------------

def gpu_test():
    available_device=GPUInfo.check_empty()
    percent,memory=GPUInfo.gpu_usage()
    min_percent=percent.index(min([percent[i] for i in available_device]))
    min_memory=memory.index(min([memory[i] for i in available_device]))

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), '\n')
    print("Num GPUs Available:", available_device)
    print("GPUs percent usage:", percent)
    print("GPUs min percent:", min_percent)
    print("GPUs memory usage:", memory)
    print("GPUs min memory:", min_memory, "\n")
