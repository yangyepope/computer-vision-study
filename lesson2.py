import torch
import numpy as np


data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)


x_one = torch.ones_like(x_data)
print(x_one)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)