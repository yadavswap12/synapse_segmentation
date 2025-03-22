"""
Script to check if tensorflow is using gpu.
"""

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

