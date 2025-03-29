import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import itertools
import torchvision.transforms as transforms

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


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.zero_pad = nn.ZeroPad2d((1,0,1,0))
        self.conv5 = nn.Conv2d(512, 1, 4, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.zero_pad(x)
        x = self.conv5(x)

        return x


criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()


G_AB = GeneratorResNet(3, num_residual_blocks=9)
D_B = Discriminator(3)

G_BA = GeneratorResNet(3, num_residual_blocks=9)
D_A = Discriminator(3)

cuda = torch.cuda.is_available()
print(f'cuda: {cuda}')
if cuda:
    G_AB = G_AB.cuda()
    D_B = D_B.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    
    criterion_GAN = criterion_GAN.cuda()
    criterion_cycle = criterion_cycle.cuda()
    criterion_identity = criterion_identity.cuda()


lr = 0.0002
b1 = 0.5
b2 = 0.999

optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
)

optimizer_D_A = torch.optim.Adam(
    D_A.parameters(), lr=lr, betas=(b1, b2)
)

optimizer_D_B = torch.optim.Adam(
    D_B.parameters(), lr=lr, betas=(b1, b2)
)

n_epoches = 100
decay_epoch = 20

lambda_func = lambda epoch: 1 - max(0, epoch-decay_epoch)/(n_epoches-decay_epoch)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_func)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_func)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_func)

class ImageDataset(Dataset):
    def __init__(self, data_dira,data_dirb, transforms=None):
        A_dir = data_dira
        B_dir = data_dirb
        
        self.files_A = [os.path.join(A_dir, name) for name in sorted(os.listdir(A_dir))[:450]]
        self.files_B = [os.path.join(B_dir, name) for name in sorted(os.listdir(B_dir))[:450]]
        
        self.transforms = transforms
        
    def __len__(self):
        return len(self.files_A)
    
    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = self.files_B[index]
        
        img_A = Image.open(file_A)
        img_B = Image.open(file_B)
        img_A = img_A.convert("RGB")
        img_B = img_B.convert("RGB")
        
        if self.transforms is not None:
            img_A = self.transforms(img_A)
            img_B = self.transforms(img_B)
        
        return img_A, img_B

data_dira = '/kaggle/working/bw_images'
data_dirb = '/kaggle/working/color_images'
transforms_ = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 5

trainloader = DataLoader(
    ImageDataset(data_dira,data_dirb, transforms=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 3
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

def sample_images(real_A, real_B, figsize=(6, 6)):
    assert real_A.size() == real_B.size(), 'The image size for two domains must be the same'
    
    G_AB.eval()
    G_BA.eval()
    
    real_A = real_A.type(Tensor)
    fake_B = G_AB(real_A).detach()
    real_B = real_B.type(Tensor)
    fake_A = G_BA(real_B).detach()
    
    image_to_show = fake_B[0].cpu()  
    image_to_show = image_to_show.permute(1, 2, 0).numpy() 
    image_to_show = (image_to_show - np.min(image_to_show)) * 255 / (np.max(image_to_show) - np.min(image_to_show))
    image_to_show = np.clip(image_to_show, 0, 255).astype(np.uint8) 

    plt.figure(figsize=figsize)
    plt.imshow(image_to_show)
    plt.axis('off')
    plt.show()
    

real_A, real_B = next(iter(trainloader))
sample_images(real_A, real_B)

for epoch in range(n_epoches):
    for i, (real_A, real_B) in enumerate(trainloader):
        real_A, real_B = real_A.type(Tensor), real_B.type(Tensor)
        
        out_shape = [real_A.size(0), 1, real_A.size(2)//16, real_A.size(3)//16]

        valid = torch.ones(out_shape).type(Tensor)
        fake = torch.zeros(out_shape).type(Tensor)
        
        """Train Generators"""
        
        G_AB.train()
        G_BA.train()
        
        optimizer_G.zero_grad()
        
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        
        loss_id_A = criterion_identity(fake_B, real_A)
        loss_id_B = criterion_identity(fake_A, real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2
        
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid) 
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        
        loss_G = 5.0*loss_identity + loss_GAN + 10.0*loss_cycle
        
        loss_G.backward()
        optimizer_G.step()
        
        """Train Discriminator A"""
        optimizer_D_A.zero_grad()
        
        loss_real = criterion_GAN(D_A(real_A), valid)
        loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
        loss_D_A = (loss_real + loss_fake) / 2
        
        loss_D_A.backward()
        optimizer_D_A.step()
        
        """Train Discriminator B"""
        optimizer_D_B.zero_grad()
        
        loss_real = criterion_GAN(D_B(real_B), valid)
        loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
        loss_D_B = (loss_real + loss_fake) / 2
        
        loss_D_B.backward()
        optimizer_D_B.step()
    
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    
    if (epoch+1) % 10 == 0:
        test_real_A, test_real_B = next(iter(trainloader))
        sample_images(test_real_A, test_real_B)

        loss_D = (loss_D_A + loss_D_B) / 2
        print(f'[Epoch {epoch+1}/{n_epoches}]')
        print(f'[G loss: {loss_G.item()} | identity: {loss_identity.item()} GAN: {loss_GAN.item()} cycle: {loss_cycle.item()}]')
        print(f'[D loss: {loss_D.item()} | D_A: {loss_D_A.item()} D_B: {loss_D_B.item()}]')


save_model_dir = './saved_models'
os.makedirs(save_model_dir, exist_ok=True)

torch.save(G_AB.state_dict(), os.path.join(save_model_dir, 'G_AB.pth'))
torch.save(G_BA.state_dict(), os.path.join(save_model_dir, 'G_BA.pth'))

torch.save(D_A.state_dict(), os.path.join(save_model_dir, 'D_A.pth'))
torch.save(D_B.state_dict(), os.path.join(save_model_dir, 'D_B.pth'))

torch.save(optimizer_G.state_dict(), os.path.join(save_model_dir, 'optimizer_G.pth'))
torch.save(optimizer_D_A.state_dict(), os.path.join(save_model_dir, 'optimizer_D_A.pth'))
torch.save(optimizer_D_B.state_dict(), os.path.join(save_model_dir, 'optimizer_D_B.pth'))

torch.save(lr_scheduler_G.state_dict(), os.path.join(save_model_dir, 'lr_scheduler_G.pth'))
torch.save(lr_scheduler_D_A.state_dict(), os.path.join(save_model_dir, 'lr_scheduler_D_A.pth'))
torch.save(lr_scheduler_D_B.state_dict(), os.path.join(save_model_dir, 'lr_scheduler_D_B.pth'))

print("Models and optimizers saved.")
