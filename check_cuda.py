import torch

print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA是否可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')

if torch.cuda.is_available():
    print(f'当前GPU: {torch.cuda.get_device_name(0)}')
else:
    print('当前GPU: N/A')