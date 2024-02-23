import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.transforms import CenterCrop


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.decoder4 = self.conv_block(1024, 512)
        self.decoder3 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)

        # Up-sampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)


        dec4 = self.decoder4(self.upsample(bottleneck))
        dec3 = self.decoder3(self.upsample(dec4)[:, :, :min(dec4.size(2), enc3.size(2)), :min(dec4.size(3), enc3.size(3))])
        dec2 = self.decoder2(self.upsample(dec3)[:, :, :min(dec3.size(2), enc2.size(2)), :min(dec3.size(3), enc2.size(3))])
        dec1 = self.decoder1(self.upsample(dec2)[:, :, :min(dec2.size(2), enc1.size(2)), :min(dec2.size(3), enc1.size(3))])

        output = self.final_conv(dec1)

        return output

class UNetwithskip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.decoder4 = self.conv_block(1024, 512)
        self.decoder3 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)

        # Up-sampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)


        dec4 = self.decoder4(self.upsample(bottleneck))
        dec3 = self.decoder3(self.upsample(dec4 + enc3)[:, :, :min(dec4.size(2), enc3.size(2)), :min(dec4.size(3), enc3.size(3))])
        dec2 = self.decoder2(self.upsample(dec3  +enc2)[:, :, :min(dec3.size(2), enc2.size(2)), :min(dec3.size(3), enc2.size(3))])
        dec1 = self.decoder1(self.upsample(dec2 + enc1)[:, :, :min(dec2.size(2), enc1.size(2)), :min(dec2.size(3), enc1.size(3))])

        output = self.final_conv(dec1)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(TransformerBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = x + F.interpolate(residual, size=x.size()[2:], mode='nearest')
        return x

class UNetTransformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetTransformer, self).__init__()
        
        # Encoder
        self.encoder1 = TransformerBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = TransformerBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = TransformerBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = TransformerBlock(256, 512)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = TransformerBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = TransformerBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = TransformerBlock(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)

        # Output layer
        output = self.out_conv(dec1)
        return output


class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class Unetwithprompts(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 32)
        self.encoder2 = self.conv_block(32 , 64)
        self.encoder3 = self.conv_block(64 , 128)
        self.encoder4 = self.conv_block(128 , 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256 , 512)

        # Decoder
        self.decoder4 = self.conv_block(512 , 256)
        self.decoder3 = self.conv_block(256 , 128)
        self.decoder2 = self.conv_block(128 , 64)
        self.decoder1 = self.conv_block(64 ,  32)

        self.prompt1 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 32)
        self.prompt2 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 64)
        self.prompt3 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 128)
        self.prompt4 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 256)


        # Up-sampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        prompt4 = self.prompt4(self.upsample(bottleneck))
        dec4 = self.decoder4(torch.cat([dec4 , prompt4]) , axis = 1)

        prompt3 = self.prompt3(self.upsample(bottleneck))

        dec3 = self.decoder3(torch.cat[(self.upsample(dec4)[:, :, :min(dec4.size(2), enc3.size(2)), :min(dec4.size(3), enc3.size(3))])])
        dec2 = self.decoder2(self.upsample(dec3)[:, :, :min(dec3.size(2), enc2.size(2)), :min(dec3.size(3), enc2.size(3))])
        dec1 = self.decoder1(self.upsample(dec2)[:, :, :min(dec2.size(2), enc1.size(2)), :min(dec2.size(3), enc1.size(3))])

        output = self.final_conv(dec1)

        return output

class unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.MaxPool2d(2)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  
        )
        self.upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        enc = self.encoder(x)
        bottle = self.bottleneck(enc)

        up = self.upconv(bottle)   
        enc = self.crop(up, enc)
        x = torch.cat([up, enc], dim=1) #skip conn
        dec = self.decoder(x)

        out = self.out(dec)
        return torch.sigmoid(out)
    
    def crop(self, up, enc):
        _,_,H,W = up.shape
        enc = CenterCrop([H, W])(enc)
        return enc


class UNetwithprompts(nn.Module):
    def __init__(self, in_channels, out_channels, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(UNetwithprompts, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.decoder4 = self.conv_block(1024, 512)
        self.decoder3 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)

        # Up-sampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # PromptGenBlock for each decoder level
        self.prompt_gen_block4 = PromptGenBlock(prompt_dim, prompt_len, prompt_size, lin_dim)
        self.prompt_gen_block3 = PromptGenBlock(prompt_dim, prompt_len, prompt_size, lin_dim)
        self.prompt_gen_block2 = PromptGenBlock(prompt_dim, prompt_len, prompt_size, lin_dim)
        self.prompt_gen_block1 = PromptGenBlock(prompt_dim, prompt_len, prompt_size, lin_dim)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder
        dec4 = self.decoder4(self.upsample(bottleneck))
        dec3 = self.decoder3(self.upsample(dec4)[:, :, :min(dec4.size(2), enc3.size(2)), :min(dec4.size(3), enc3.size(3))])
        dec2 = self.decoder2(self.upsample(dec3)[:, :, :min(dec3.size(2), enc2.size(2)), :min(dec3.size(3), enc2.size(3))])
        dec1 = self.decoder1(self.upsample(dec2)[:, :, :min(dec2.size(2), enc1.size(2)), :min(dec2.size(3), enc1.size(3))])

        # Use PromptGenBlock at each level in the decoder
        dec4_param = self.prompt_gen_block4(dec4)
        out_dec_level4 = torch.cat([dec4, dec4_param], 1)

        dec3_param = self.prompt_gen_block3(out_dec_level4)
        out_dec_level3 = torch.cat([out_dec_level4, dec3_param], 1)

        dec2_param = self.prompt_gen_block2(out_dec_level3)
        out_dec_level2 = torch.cat([out_dec_level3, dec2_param], 1)

        dec1_param = self.prompt_gen_block1(out_dec_level2)
        out_dec_level1 = torch.cat([out_dec_level2, dec1_param], 1)

        output = self.final_conv(out_dec_level1)

        return output
    
    def conv_block(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
