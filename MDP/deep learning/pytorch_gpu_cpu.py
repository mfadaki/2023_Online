import time
import torch

# In Windows that Cuda or GPA is available directly
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("device is running on GPU")
else:
    device = torch.device('cpu')
    print("device is running on CPU")

# No of available GPUs
torch.cuda.device_count()


device = torch.device('cuda:0')  # GPU #0
print(device)
torch.cuda.is_available()  # Not working with Mac

device = torch.device('mps')  # GPU
print(device)

device = torch.device('cpu')  # CPU
print(device)

x = torch.rand((70000, 70000), dtype=torch. float32)
y = torch.rand((70000, 70000), dtype=torch.float32)
x = x.to(device)
y = y.to(device)

tic = time.time()
x * y
toc = time.time()
print(toc-tic)
print(f"Total time is {toc - tic:0.4f} seconds")
