import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import os

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3)
        self.norm2 = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        return x + out


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        
        self.pad = nn.ReflectionPad2d(in_channels)
        self.conv = nn.Conv2d(in_channels, 64, 7)
        self.norm = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.down1_conv = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down1_norm = nn.InstanceNorm2d(128)
        self.down1_relu = nn.ReLU(inplace=True)
        
        self.down2_conv = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.down2_norm = nn.InstanceNorm2d(256)
        self.down2_relu = nn.ReLU(inplace=True)

        self.residual_blocks = []
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(256))
        self.residual_blocks = nn.ModuleList(self.residual_blocks)

        self.up1_upsample = nn.Upsample(scale_factor=2)
        self.up1_conv = nn.Conv2d(256, 128, 3, padding=1)
        self.up1_norm = nn.InstanceNorm2d(128)
        self.up1_relu = nn.ReLU(inplace=True)
        
        self.up2_upsample = nn.Upsample(scale_factor=2)
        self.up2_conv = nn.Conv2d(128, 64, 3, padding=1)
        self.up2_norm = nn.InstanceNorm2d(64)
        self.up2_relu = nn.ReLU(inplace=True)

        self.out_pad = nn.ReflectionPad2d(in_channels)
        self.out_conv = nn.Conv2d(64, in_channels, 7)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.down1_conv(x)
        x = self.down1_norm(x)
        x = self.down1_relu(x)

        x = self.down2_conv(x)
        x = self.down2_norm(x)
        x = self.down2_relu(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.up1_upsample(x)
        x = self.up1_conv(x)
        x = self.up1_norm(x)
        x = self.up1_relu(x)

        x = self.up2_upsample(x)
        x = self.up2_conv(x)
        x = self.up2_norm(x)
        x = self.up2_relu(x)

        x = self.out_pad(x)
        x = self.out_conv(x)
        x = self.tanh(x)

        return x


cuda = torch.cuda.is_available()

G_AB = GeneratorResNet(3, num_residual_blocks=9)
model_path = '/kaggle/input/testdatasetmangaacolo2/G_AB(2).pth'
G_AB.load_state_dict(torch.load(model_path))

if cuda:
    G_AB = G_AB.cuda()

G_AB.eval()

generate_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_image(image_path, output_dir):
    img = Image.open(image_path).convert("RGB")

    img_tensor = generate_transforms(img)
    img_tensor = torch.unsqueeze(img_tensor, 0).type(torch.cuda.FloatTensor if cuda else torch.Tensor)
    fake_img = G_AB(img_tensor).detach().cpu()

    fake_img = fake_img.squeeze().permute(1, 2, 0).numpy()
    fake_img = (fake_img + 1) * 127.5 
    fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(fake_img)

    output_image.show()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    _, name = os.path.split(image_path)
    output_image.save(os.path.join(output_dir, name))

image_path = '/kaggle/working/bw_images/bw_10.png'

output_dir = './generated_images'

predict_image(image_path, output_dir)

print(f"Generated image saved to {output_dir}")
