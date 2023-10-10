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

# MAGIC %md
# MAGIC # Part 1: Tensors
# MAGIC
# MAGIC Tensors are PyTorch's most basic building block. Each tensor is a multi-dimensional matrix; for example, a 256x256 square image might be represented by a 3x256x256 tensor, where the first dimension represents color. Here is how to create a tensor:

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

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Quiz
# MAGIC
# MAGIC Write code that creates a `torch.tensor` with the following contents:
# MAGIC $\begin{bmatrix} 1 & 2.2 & 9.6 \\ 4 & -7.2 & 6.3 \end{bmatrix}$
# MAGIC
# MAGIC Now compute the average of each row (`.mean()`) and each column.
# MAGIC
# MAGIC What's the shape of the results?

# COMMAND ----------

data = torch.tensor([[1, 2.2, 9.6], [4, -7.2, 6.3]])

row_avg = data.mean(dim=1)
col_avg = data.mean(dim=0)

print(row_avg.shape)
print(row_avg)

print(col_avg.shape)
print(col_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Indexing
# MAGIC
# MAGIC You can access arbitrary elements of a tensor using the [] operator.

# COMMAND ----------

# Initialize an example tensor
x = torch.Tensor([
                  [[1, 2], [3, 4]],
                  [[5, 6], [7, 8]], 
                  [[9, 10], [11, 12]] 
                 ])
print(x)
print(x.shape)

# COMMAND ----------

# Access the 0th element, which is the first row
x[0] # Equivalent to x[0, :]

# COMMAND ----------

x[:, 0]

# COMMAND ----------



# COMMAND ----------

matr = torch.arange(1, 16).view(5, 3)
print(matr)

# COMMAND ----------

""" 
tensor([[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12],
        [13, 14, 15]])
"""
print(matr[0])      # 1st row, tensor([1, 2, 3])
print(matr[0,:])    # 1st row, tensor([1, 2, 3])
print(matr[:,0])    # 1st column, tensor([ 1,  4,  7, 10, 13])
print(matr[0:3])    # 0-2 rows
print(matr[:, 0:2]) # 0-1 coluns
print(matr[0:3, 0:2])   # 0-2 rows, 0-1 columns
print(matr[0][2])   # item located at 0 row and 3rd column
print(matr[0:3, 2]) # items located at 0-2 rows and 3rd column
print(matr[0:3][2]) # items located at 3rd row and 0-2 columns
print(matr[0:3])
print(matr[[0, 2, 4]])


# COMMAND ----------

# We can also index into multiple dimensions with :.
# Get the top left element of each element in our tensor
print(x[:, 0, 0])
print(x[:, :, :])

# Print x again to see our tensor
print(x)

# Let's access the 0th and 1st elements, each twice
i = torch.tensor([0, 0, 1, 1])
print(i)
print(x[i])

# COMMAND ----------

# Let's access the 0th elements of the 1st and 2nd elements
i = torch.tensor([1, 2])
j = torch.tensor([0])
x[i, j]

# COMMAND ----------

# We can get a Python scalar value from a tensor with item().
x[0, 0, 0]
x[0, 0, 0].item()### Exercise:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise:
# MAGIC
# MAGIC Write code that creates a `torch.tensor` with the following contents:
# MAGIC $\begin{bmatrix} 1 & 2.2 & 9.6 \\ 4 & -7.2 & 6.3 \end{bmatrix}$
# MAGIC
# MAGIC How do you get the first column? The first row?

# COMMAND ----------

test = torch.tensor([[1, 2.2, 9.6], [4, -7.2, 6.3]])
print(test)
print(test[:,0])    # 1st column
print(test[0])      # 1st row

# COMMAND ----------

# MAGIC %md
# MAGIC ## Autograd
# MAGIC
# MAGIC Pytorch is well-known for its automatic differentiation feature. We can call the backward() method to ask PyTorch to calculate the gradients, which are then stored in the grad attribute.

# COMMAND ----------

# Create an example tensor
# requires_grad parameter tells PyTorch to store gradients
x = torch.tensor([2.], requires_grad=True)

# Print the gradient if it is calculated
# Currently None since x is a scalar
pp.pprint(x.grad)
