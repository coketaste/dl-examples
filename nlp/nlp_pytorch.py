# Databricks notebook source
import sys
sys.version

# COMMAND ----------

# torch-2.0.1+cpu.cxx11.abi-cp39-cp39-linux_x86_64.whl
%pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# COMMAND ----------

import torch
import torch.nn as nn

# Import pprint, module we use for making our print statements prettier
import pprint
pp = pprint.PrettyPrinter()

# COMMAND ----------

# Part 1: Tensors
# Tensors are PyTorch's most basic building block. Each tensor is a multi-dimensional matrix; for example, a 256x256 square image might be represented by a 3x256x256 tensor, where the first dimension represents color. Here is how to create a tensor:

# COMMAND ----------

list_of_lists = [
  [1, 2, 3],
  [4, 5, 6],
]
print(list_of_lists)

# COMMAND ----------

# Initializing a tensor
data = torch.tensor([
                     [0, 1],    
                     [2, 3],
                     [4, 5]
                    ])
print(data)

# Each tensor has a data type: the major data types you'll need to worry about are floats (torch.float32) and integers (torch.int). You can specify the data type explicitly when you create the tensor:

# COMMAND ----------

# Initializing a tensor with an explicit data type
# Notice the dots after the numbers, which specify that they're floats
data = torch.tensor([
                     [0, 1],    
                     [2, 3],
                     [4, 5]
                    ], dtype=torch.float32)
print(data)

# COMMAND ----------

# Initializing a tensor with an explicit data type
# Notice the dots after the numbers, which specify that they're floats
data = torch.tensor([
                     [0.11111111, 1],    
                     [2, 3],
                     [4, 5]
                    ], dtype=torch.float32)
print(data)

# COMMAND ----------

# Initializing a tensor with an explicit data type
# Notice the dots after the numbers, which specify that they're floats
data = torch.tensor([
                     [0.11111111, 1],    
                     [2, 3],
                     [4, 5]
                    ])
print(data)

# COMMAND ----------

# Utility functions also exist to create tensors with given shapes and contents:
zeros = torch.zeros(2, 5)  # a tensor of all zeros
print(zeros) 

ones = torch.ones(3, 4)   # a tensor of all ones
print(ones)

rr = torch.arange(1, 10) # range from [1, 10) 
print(rr)
print(rr + 2)
print(rr * 2)



# COMMAND ----------

a = torch.tensor([[1, 2], [2, 3], [4, 5]])      # (3, 2)
b = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # (2, 4)  

print("A is", a)
print("B is", b)
print("The product is", a.matmul(b)) # (3, 4)
print("The other product is", a @ b) # +, -, *, @

# COMMAND ----------

v = torch.tensor([1, 2, 3])
v.shape

# COMMAND ----------

torch.tensor([[1, 2, 3], [4, 5, 6]]) @ v  #(2, 3) @ (3)  -> (2)

# COMMAND ----------

# The shape of a matrix (which can be accessed by .shape) is defined as the dimensions of the matrix. Here's some examples:
matr_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(matr_2d.shape)
print(matr_2d)

# COMMAND ----------

matr_3d = torch.tensor([[[1, 2, 3, 4], [-2, 5, 6, 9]], [[5, 6, 7, 2], [8, 9, 10, 4]], [[-3, 2, 2, 1], [4, 6, 5, 9]]])
print(matr_3d)
print(matr_3d.shape)

# COMMAND ----------

# Reshaping tensors can be used to make batch operations easier (more on that later), but be careful that the data is reshaped in the order you expect:
rr = torch.arange(1, 16)
print("The shape is currently", rr.shape)
print("The contents are currently", rr)
print()
rr = rr.view(5, 3)
print("After reshaping, the shape is currently", rr.shape)
print("The contents are currently", rr)

# COMMAND ----------

# Finally, you can also inter-convert tensors with NumPy arrays:
import numpy as np

# numpy.ndarray --> torch.Tensor:
arr = np.array([[1, 0, 5]])
data = torch.tensor(arr)
print("This is a torch.tensor", data)

# torch.Tensor --> numpy.ndarray:
new_arr = data.numpy()
print("This is a np.ndarray", new_arr)

# COMMAND ----------

# One of the reasons why we use tensors is vectorized operations: operations that be conducted in parallel over a particular dimension of a tensor.
data = torch.arange(1, 36, dtype=torch.float32).reshape(5, 7)
print("Data is:", data)

# We can perform operations like *sum* over each row...
print("Taking the sum over columns:")
print(data.sum(dim=0))

# or over each column.
print("Taking thep sum over rows:")
print(data.sum(dim=1))

# Other operations are available:
print("Taking the stdev over rows:")
print(data.std(dim=1))

print(data.sum())
