import torch
A = torch.randn([12, 9, 64])
B = torch.randn([12, 9, 64])
Ar = A.repeat(1, 1, 9).view(12, 81, 64)
Br = B.repeat(1, 9, 1)
C = torch.cat((Ar, Br), dim=2)
D = torch.cat([A.unsqueeze(2).expand(-1, -1, 9, -1),
               B.unsqueeze(1).expand(-1, 9, -1, -1)], dim=-1).view(12, 81, 128)
print ((C-D).abs().max().item())


K=torch.randn([1,3,258,258])
# N=K.repeats(a)
1
C = torch.cat((K, K), dim=0)
1