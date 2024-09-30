# %% Import and test GPU
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
