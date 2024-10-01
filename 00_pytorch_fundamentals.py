# %% Import and test GPU
from os import times
from numpy._core.records import array
from numpy._core.umath import numpy
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

# %% Introduction to tensors
# %% Scalar
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)
print(scalar.item())

# %% Vector
vector = torch.tensor([7, 7])
vector
vector.ndim
vector.size()

# %% Matrix
matrix = torch.tensor([[7, 8], [9, 10]])
matrix
matrix.ndim
matrix.shape

# %% Tensor
tensor = torch.tensor([[[1, 2, 3, 0],
                        [4, 5, 6, 0],
                        [7, 8, 9, 0]]])
tensor
tensor.ndim
tensor.shape
tensor[0]

# %% Random tensors
# %% Create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
random_tensor

# %% Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(3, 224, 224))
random_image_size_tensor.shape, random_image_size_tensor.ndim

# %% Zeros and Ones
zeros = torch.zeros(size=(3, 4))
ones = torch.ones(size=(3, 4))
zeros, ones

# %% Creating a range of tensors
one_to_ten = torch.arange(1, 11)
one_to_ten

# %% Creating tensors-like
ten_zeros = torch.zeros_like(one_to_ten)
ten_zeros

# %% Tensor datatypes
# default is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
    dtype=None, # data type of the tensor element
    device=None, # "cpu" or "cuda"
    requires_grad=False) # whether or not to track gradients with this tensors operators
# can also be other length
float_16_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float16)
float_16_tensor_2 = float_32_tensor.type(torch.float16)
float_32_tensor.dtype, float_16_tensor.dtype, float_16_tensor_2.dtype

# %% Getting information from tensors - tensor attributes
some_tensor = torch.rand(3, 4)
print(some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Size of tensor: {some_tensor.size()}")
print(f"Device of tensor: {some_tensor.device}")

# %% Tensor operations
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)
print(tensor * 10)
print(tensor / 10)
print(tensor * 10 - 10)
print(torch.add(tensor, 10))
print(torch.mul(tensor, 10))
print(torch.div(tensor, 10))

# %% Matrix multiplication
# elementwise
print(tensor * tensor)
# matrix multiplication
print(tensor.matmul(tensor))
torch.matmul(torch.rand(3, 10), torch.rand(10, 3))
# transpose
torch.matmul(torch.rand(3, 5), torch.rand(4, 5).T) # .T is transpose

# %% Tensor aggregation - min, max, mean, sum, etc.
x = torch.arange(0, 100, 10)
print(x)
print(torch.min(x), x.min())
print(torch.max(x), x.max())
# mean() requires float datatype
print(torch.mean(x.type(torch.float32)), x.type(torch.float32).mean())
print(torch.sum(x), x.sum())

# %% Positional min and max
x = torch.arange(1, 100, 10)
print(x)
print(x.argmin())
print(x.argmax())

# %% Reshaping, stacking, squeezing, unsqueezing and permuting tensors
x = torch.arange(1., 10.)
print(x, x.shape)
# reshapes input to shape (if compatible), can also use torch.Tensor.reshape().
x_reshaped = x.reshape(9, 1)
print(x_reshaped)
# returns a view of the original tensor in a different shape but shares the same data as the original tensor.
z = x.view(1, 9)
print(z, z.shape)
# changes z also changes x since share same memory
z[:, 0] = 5
print(z, x)
# stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim = 1)
x_stacked
# squeeze - squeezes input to remove all the dimenions with value 1.
x_squeezed = x_reshaped.squeeze()
print(x_squeezed, x_squeezed.shape)
# unsqueeze - Returns input with a dimension value of 1 added at dim.
x_unsqueezed = x_squeezed.unsqueeze(dim=1)
print(x_unsqueezed, x_unsqueezed.shape)
# permute - Returns a view of the original input with its dimensions permuted (rearranged) to dims.
x_original = torch.rand(size=(224, 224, 3))
print(f"Original shape: {x_original.shape}")
print(f"Permuted shape: {x_original.permute(dims=(2, 0, 1)).shape}")

# %% Indexing
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)
print(x[0], x[0][0], x[0][0][0])
print(x[:, 0], x[:, :, 1])
print(x[0, 0, :])
print(x[0, 2, 2], x[0, :, 2])

# %% PyTorch tensors and NumPy
import torch
import numpy as np
array = np.arange(1.0, 8.0)
# numpy to torch
tensor = torch.from_numpy(array) # notice numpy default dtype is float64
array, tensor
# torch tensor to numpy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
tensor, numpy_tensor

# %% Reproducbility (trying to take random out of random)
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)
print(random_tensor_A == random_tensor_B)

# set random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)

# %% Running tensors and PyTorch objects on the GPUs (and making faster computations)
# check if GPU is available
print(torch.cuda.is_available())
# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# count the number of GPUs
print(torch.cuda.device_count())

# %% Putting tensors on the GPU
# default device is CPU
tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)
# move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu, tensor_on_gpu.device)
# if tensor is on GPU, cannot transfer it to numpy
# so need to move tensor back to CPU
tensor_back_on_cpu = tensor_on_gpu.to("cpu").numpy()
print(tensor_back_on_cpu, tensor_back_on_cpu.device)
