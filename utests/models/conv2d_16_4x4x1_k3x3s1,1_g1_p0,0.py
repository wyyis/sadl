import os
import sys
import torch


class Pad0Module(torch.nn.Module):
    def __init__(self):
        super(Pad0Module, self).__init__()
        self.conv3x3_pad00 = torch.nn.Conv2d( in_channels=16, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(0,0) )
        
    def forward(self, x):
        y = self.conv3x3_pad00(x)
        return y

input_size = (1, 16, 4, 4)
output_size = (1, 1, 2, 2)

model = Pad0Module()
input = torch.randn(*input_size)
model.eval()
output = model(input)


# Output onnx model path
onnx_model_path = os.path.join(os.path.dirname(sys.argv[0]), "conv2d_16_4x4x1_k3x3s1,1_g1_p0,0.onnx")
# Convert to onnx
torch.onnx.export(model, input, onnx_model_path, opset_version=10)
