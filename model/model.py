# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:29:53 2021

@author: LEGION
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:54:13 2021

@author: LEGION
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(encoder1, self).__init__()
        self.conv1 = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        return y

class encoder2(nn.Module):
    def __init__(self,out_channel):
        super(encoder2, self).__init__()
        self.conv1 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class encoder3(nn.Module):
    def __init__(self,out_channel):
        super(encoder3, self).__init__()
        self.conv1 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias= False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 2,bias = False,dilation = 2)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 4,bias = False,dilation = 4)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class encoder4(nn.Module):
    def __init__(self,out_channel):
        super(encoder4, self).__init__()
        self.conv1 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 3,bias= False,dilation = 3)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 4,bias = False,dilation = 4)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 5,bias = False,dilation = 5)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class decoder1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decoder1, self).__init__()
        self.conv1 = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class decoder2(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decoder2, self).__init__()
        self.conv1 = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class decoder3(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decoder3, self).__init__()
        self.conv1 = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class decoder4(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decoder4, self).__init__()
        self.conv1 = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        return y

class down1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(down1, self).__init__()
        self.down = nn.Conv3d(in_channel,out_channel,kernel_size = 2,stride = 2,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.down(x)
        y = self.bn1(y)
        y = F.relu(y)
        return y

class down2(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(down2, self).__init__()
        self.down = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1 ,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.down(x)
        y = self.bn1(y)
        y = F.relu(y)
        return y

class up(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channel, out_channel, kernel_size = 2,stride = 2)
        self.bn1 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.up(x)
        y = self.bn1(y)
        y = F.relu(y)
        return y

class mapping(nn.Module):
    def __init__(self,n_channel,n_output,scale_factor = (8, 8, 8)):
        super(mapping, self).__init__()
        self.conv = nn.Conv3d(n_channel,n_output,kernel_size = 1,stride = 1)
        self.upsample = nn.Upsample(scale_factor= scale_factor, mode = 'trilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.conv(x)
        y = self.upsample(y)
        return y


#????????????resize???????????????256*256*32
class resunet3d(nn.Module):
    def __init__(self , n_channel = 1, n_output = 10,training = True):
        super(resunet3d, self).__init__()
        self.training = training
        self.drop_rate = 0.2
        self.en1 = encoder1(n_channel,16)
        self.en2 = encoder2(32)
        self.en3 = encoder3(64)
        self.en4 = encoder4(128)
        self.de1 = decoder1(128,256)
        self.de2 = decoder2(192,128)
        self.de3 = decoder3(96,64)
        self.de4 = decoder4(48,32)
        self.down1 = down1(16,32)
        self.down2 = down1(32,64)
        self.down3 = down1(64,128)
        self.down4 = down2(128,256)
        self.up2 = up(256,128)
        self.up3 = up(128,64)
        self.up4 = up(64,32)
        self.map1 = mapping(256,n_output,scale_factor=(8, 8 ,8))
        self.map2 = mapping(128,n_output,scale_factor=(4, 4, 4))
        self.map3 = mapping(64,n_output,scale_factor=(2, 2, 2))
        self.map4 = mapping(32,n_output,scale_factor=(1 ,1 ,1))

    def forward(self,x) :
        l1=self.en1(x)+x
        s1=self.down1(l1)
        l2=self.en2(s1)+s1
        l2=F.dropout(l2,self.drop_rate,self.training)
        s2=self.down2(l2)
        l3=self.en3(s2)+s2
        l3=F.dropout(l3,self.drop_rate,self.training)
        s3=self.down3(l3)
        l4=self.en4(s3)+s3
        l4=F.dropout(l4,self.drop_rate,self.training)
        s4=self.down4(l4)
        out=self.de1(l4)+s4
        out=F.dropout(out,self.drop_rate,self.training)
        output1=self.map1(out)
        s6=self.up2(out)
        out=self.de2(torch.cat([s6, l3], dim=1))+s6
        out=F.dropout(out,self.drop_rate,self.training)
        output2=self.map2(out)
        s7=self.up3(out)
        out=self.de3(torch.cat([s7, l2], dim=1))+s7
        out=F.dropout(out,self.drop_rate,self.training)
        output3=self.map3(out)
        s8=self.up4(out)
        out=self.de4(torch.cat([s8, l1], dim=1))+s8
        output4=self.map4(out)
        # print(output1.shape)
        # print(output2.shape)
        # print(output3.shape)
        # print(output4.shape)
        if self.training is True:
            #return output1, output2, output3, output4
            return output4
        else:
            return output4

# #???3Dunet???????????????????????????encoder?????????????????????????????????decoder????????????dropout?????????conv+bn+relu??????????????????????????????????????????
# #output(B*C*D*H*W)
#
# # model = resunet3d(training = True)
# # from torchsummary import summary
# # summary(model, input_size=[(1, 64 , 64 , 64)], device="cpu")
# # loss???la dice+ce
#
# # def to_one_hot_3d(tensor, n_classes=3):  # shape = [batch, s, h, w]
# #     n,c, s, h, w = tensor.size()
# #     one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
# #     return one_hot
#
# # img = torch.LongTensor(2, 3, 1, 28, 28).random_()
# # #tensor([[6],
# # #        [0],
# # #        [3],
# # #        [2]])
# # result = to_one_hot_3d(img)



# import torch
# import torch.nn as nn
# import torch.functional as F
# # import torchsnooper
# # from loss_func import Dice_loss
#
#
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                                                   kernel_size=kernel_size, stride=stride,
                                                   padding=padding, output_padding=output_padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv3d_transpose(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x


class U_Net_3D(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=2)

        self.encoder_1_1 = ConvBlock(1, 32)
        self.encoder_1_2 = ConvBlock(32, 64)
        self.encoder_2_1 = ConvBlock(64, 64)
        self.encoder_2_2 = ConvBlock(64, 128)
        self.encoder_3_1 = ConvBlock(128, 128)
        self.encoder_3_2 = ConvBlock(128, 256)
        self.encoder_4_1 = ConvBlock(256, 256)
        self.encoder_4_2 = ConvBlock(256, 512)

        self.deconv_4 = ConvTranspose(512, 512)
        self.deconv_3 = ConvTranspose(256, 256)
        self.deconv_2 = ConvTranspose(128, 128)

        self.decoder_3_1 = ConvBlock(768, 256)
        self.decoder_3_2 = ConvBlock(256, 256)
        self.decoder_2_1 = ConvBlock(384, 128)
        self.decoder_2_2 = ConvBlock(128, 128)
        self.decoder_1_1 = ConvBlock(192, 64)
        self.decoder_1_2 = ConvBlock(64, 64)
        self.final = ConvBlock(64, 1)

    # @torchsnooper.snoop()
    def forward(self, x):
        encoder_1_out = self.encoder_1_2(self.encoder_1_1(x))  # [n 64 64 64 64]
        encoder_2_out = self.encoder_2_2(self.encoder_2_1(self.max_pool(encoder_1_out)))  # [n 128 32 32 32]
        encoder_3_out = self.encoder_3_2(self.encoder_3_1(self.max_pool(encoder_2_out)))  # [n 256 16 16 16]
        encoder_4_out = self.encoder_4_2(self.encoder_4_1(self.max_pool(encoder_3_out)))  # [n 512 8 8 8]

        decoder_4_out = self.deconv_4(encoder_4_out)  # [n 512 16 16 16]
        concatenated_data_3 = torch.cat((encoder_3_out, decoder_4_out), dim=1)  # [n 768 16 16 16]
        decoder_3_out = self.decoder_3_2(self.decoder_3_1(concatenated_data_3))  # [n 256 16 16 16]

        decoder_3_out = self.deconv_3(decoder_3_out)  # [n 256 32 32 32]
        concatenated_data_2 = torch.cat((encoder_2_out, decoder_3_out), dim=1)  # [n 384 32 32 32]
        decoder_2_out = self.decoder_2_2(self.decoder_2_1(concatenated_data_2))  # [n 128 32 32 32]

        decoder_2_out = self.deconv_2(decoder_2_out)  # [n 128 64 64 64]
        concatenated_data_1 = torch.cat((encoder_1_out, decoder_2_out), dim=1)  # [n 192 64 64 64]
        decoder_1_out = self.decoder_1_2(self.decoder_1_1(concatenated_data_1))  # [n 64 64 64 64]

        final_data = self.final(decoder_1_out)  # [n 1 64 64 64]

        return final_data


class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=2)

        self.encoder_1_1 = ConvBlock(1, 16)
        self.encoder_1_2 = ConvBlock(16, 16)
        self.encoder_2_1 = ConvBlock(16, 32)
        self.encoder_2_2 = ConvBlock(32, 32)
        self.encoder_3_1 = ConvBlock(32, 64)
        self.encoder_3_2 = ConvBlock(64, 64)
        self.encoder_4_1 = ConvBlock(64, 128)
        self.encoder_4_2 = ConvBlock(128, 128)

        self.deconv_4 = ConvTranspose(128, 128)
        self.deconv_3 = ConvTranspose(64, 64)
        self.deconv_2 = ConvTranspose(32, 32)

        self.decoder_3_1 = ConvBlock(192, 64)
        self.decoder_3_2 = ConvBlock(64, 64)
        self.decoder_2_1 = ConvBlock(96, 32)
        self.decoder_2_2 = ConvBlock(32, 32)

        self.final = nn.Conv3d(in_channels=48, out_channels=1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # @torchsnooper.snoop()
    def forward(self, x):
        encoder_1_out = self.encoder_1_2(self.encoder_1_1(x))
        encoder_2_out = self.encoder_2_2(self.encoder_2_1(self.max_pool(encoder_1_out)))
        encoder_3_out = self.encoder_3_2(self.encoder_3_1(self.max_pool(encoder_2_out)))
        encoder_4_out = self.encoder_4_2(self.encoder_4_1(self.max_pool(encoder_3_out)))

        decoder_4_out = self.deconv_4(encoder_4_out)
        concatenated_data_3 = torch.cat((encoder_3_out, decoder_4_out), dim=1)
        decoder_3_out = self.decoder_3_2(self.decoder_3_1(concatenated_data_3))

        decoder_3_out = self.deconv_3(decoder_3_out)
        concatenated_data_2 = torch.cat((encoder_2_out, decoder_3_out), dim=1)
        decoder_2_out = self.decoder_2_2(self.decoder_2_1(concatenated_data_2))

        decoder_2_out = self.deconv_2(decoder_2_out)
        concatenated_data_1 = torch.cat((encoder_1_out, decoder_2_out), dim=1)

        final_data = self.final(concatenated_data_1)

        return final_data