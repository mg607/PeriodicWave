# Copyright (c) 2025 Max Geier, Massachusetts Institute of Technology, MA, USA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Logging functionalities. """

import jax
import jax.numpy as jnp
import platform
import os
import numpy as np
import logging
import jaxlib
import kfac_jax
import subprocess
import json
from sys import argv

def log_device_info(log_file="device_info.log"):
    """
    Logs important device and environment information when training a neural network with JAX.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:  # Only create if a directory path is provided
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format="%(asctime)s - %(message)s")
    
    # JAX-related information
    jax_version = jax.__version__
    jaxlib_version = jaxlib.__version__
    kfac_jax_version = kfac_jax.__version__
    available_devices = jax.devices()
    platform_name = jax.default_backend()
    precision = jnp.finfo(jnp.float32).dtype.name  # Default precision
    matmul_precision = jax.config.jax_default_matmul_precision
    jax_enable_x64 = jax.config.read("jax_enable_x64")  # Check if 64-bit precision is enabled
    
    # GPU details if available
    if "gpu" in platform_name:
        gpu_info = [device.device_kind for device in available_devices]
    else:
        gpu_info = "No GPU detected."
    
    # Python and system information
    python_version = platform.python_version()
    system_name = platform.system()
    system_version = platform.version()
    processor_info = platform.processor()
    numpy_version = np.__version__
    
    # Environment variables (e.g., CUDA paths)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    
    # NVIDIA driver version and GPU model
    try:
        nvidia_driver_version = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], 
            text=True
        ).strip()
        gpu_models = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
            text=True
        ).strip().split("\n")
    except (FileNotFoundError, subprocess.CalledProcessError):
        nvidia_driver_version = "NVIDIA driver not found or not installed."
        gpu_models = ["No GPU detected."]
    
    # Log the collected information
    logging.info(f"JAX version: {jax_version}")
    logging.info(f"jax_enable_x64: {jax_enable_x64}")
    logging.info(f"JAXlib version: {jaxlib_version}")
    logging.info(f"KFAC-JAX version: {kfac_jax_version}")
    logging.info(f"Available devices: {[str(device) for device in available_devices]}")
    logging.info(f"Platform: {platform_name}")
    logging.info(f"Default precision: {precision}")
    logging.info(f"Default matmul precision: {matmul_precision}")
    logging.info(f"GPU information: {gpu_info}")
    logging.info(f"Python version: {python_version}")
    logging.info(f"System: {system_name} {system_version}")
    logging.info(f"Processor: {processor_info}")
    logging.info(f"Numpy version: {numpy_version}")
    logging.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    logging.info(f"NVIDIA driver version: {nvidia_driver_version}")
    logging.info(f"GPU models: {', '.join(gpu_models)}")
    
    print(f"Device and environment info logged to {log_file}")

def save_config_dict_as_json(config_dict, file_path):
    """
    Saves a ConfigDict object to a JSON file using its `to_json` method.
    
    Args:
        config_dict (ConfigDict): The ConfigDict object to save.
        file_path (str): The path to the JSON file.
    """
    with open(file_path, 'w') as json_file:
        json_file.write(config_dict.to_json_best_effort(indent=4))
    print(f"Configuration saved as JSON to {file_path}")

def load_config_dict_from_json(file_path, config_class):
    """
    Loads a ConfigDict object from a JSON file.
    
    Args:
        file_path (str): The path to the JSON file.
        config_class (type): The class of the ConfigDict (e.g., `ConfigDict`).
        
    Returns:
        ConfigDict: The loaded ConfigDict object.
    """
    with open(file_path, 'r') as json_file:
        json_content = json_file.read()
    config_dict = config_class.from_json(json_content)
    print(f"Configuration loaded from {file_path}")
    return config_dict

# Example usage
if __name__ == "__main__":
    program_name = argv[0]
    logfile_name = argv[1]
    log_device_info(logfile_name + '.log')