import torch
print(torch.cuda.is_available())      # True -> GPU OK
print(torch.cuda.get_device_name(0))  # GeForce GTX 1650
