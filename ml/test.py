import torch
import torch.nn.functional as F
class Test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1,5)
    def forward(self, data):
        print(data.size())
        out = self.layer(data)
        out = F.softmax(out, dim=1)
        return out
if __name__ == "__main__":
    model = Test()
    t = torch.tensor([[1], [3], [5]], dtype= torch.float32)
    #t = torch.randn(5,1)
    model.eval()
    o = model(t)
    print(o)