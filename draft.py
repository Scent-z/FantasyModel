import torch

# dim_index = torch.arange(0, 8, 2, device="cuda").float()
# print(dim_index)
# a = torch.arange(8)
# print(a)


end = 100
dim = 8
rope_base = 1e6

freqs= 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
attn_factor = 1.0 

print(freqs.shape)
print(freqs)

# 外积计算所有位置的角度
t = torch.arange(end, device=freqs.device)

print(t.shape)

freqs = torch.outer(t, freqs).float()  # 每个位置、每个维度组对应的旋转角度  角度 = 位置 × 频率 

print(freqs.shape)

# x' = x cosθ - y sinθ  y' = x sinθ + y cosθ
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # 复制两次，本项目选取前后配对而不是相邻配对，更方便更工程化
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

print(freqs_cos.shape)
print(freqs_sin.shape)

x = torch.tensor([1,2,3,4,5,6,7,8])

print(x)
print(torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1))