import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch

# 1. 加载 .npz
npz_path = "Cifar100_clustered/iteration_0/train/0.npz"
data = np.load(npz_path, allow_pickle=True)

# 2. 查看里面的键（可选）
print("keys:", data.files)  # 应该包含 ['x','y']

# 3. 自动提取 x / y
def get_xy(npz):
    files = npz.files
    # 优先x,y键
    if 'x' in files and 'y' in files:
        return npz['x'], npz['y']
    # 尝试嵌套data字典
    if 'data' in files:
        raw = npz['data']
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            try:
                obj = raw.item()
                if isinstance(obj, dict) and 'x' in obj and 'y' in obj:
                    return obj['x'], obj['y']
            except Exception:
                pass
    # 使用前两个数组作为fallback
    if len(files) >= 2:
        return npz[files[0]], npz[files[1]]
    raise KeyError(f"无法从npz中提取x,y，keys: {files}")

x_np, y_np = get_xy(data)

# 4. 转为 torch.Tensor
#    如果你的模型接收 (C,H,W) 格式，可先 permute，或直接在 Dataset 中处理
x_tensor = torch.tensor(x_np, dtype=torch.float32).permute(0,3,1,2)
y_tensor = torch.tensor(y_np, dtype=torch.long)

# 5. 构造 DataLoader
dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 6. 测试
for xb, yb in loader:
    print(xb.shape, yb.shape)
    break