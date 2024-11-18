import os
import sys
import torch


#  more7:  8 in 16 out <- pass
#  more8: 16 in  8 out <- fail
#  more9: 16 in  4 out <- fail
# more10: 16 in  1 out <- fail
# more11: 10x10 -> 10x8
# more12: 10x10 -> 10x8

class Pad0Module(torch.nn.Module):
    def __init__(self):
        super(Pad0Module, self).__init__()
        self.conv1x3_pad00 = torch.nn.Conv2d( in_channels=16, out_channels=1, kernel_size=(3,1), stride=(1,1), padding=(0,0) )
        
    def forward(self, x):
        y = self.conv1x3_pad00(x)
        return y

input_size = (1, 16, 4, 4)
output_size = (1, 1, 2, 4)

model = Pad0Module()
input = torch.randn(*input_size)
model.eval()
output = model(input)


# Output onnx model path
onnx_model_path = os.path.join(os.path.dirname(sys.argv[0]), "conv2d_16_4x4x1_k3x1s1,1_g1_p0,0.onnx")
# Convert to onnx
torch.onnx.export(model, input, onnx_model_path, opset_version=10)
