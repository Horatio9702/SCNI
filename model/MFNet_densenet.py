import torch
import torch.nn as nn
import torch.nn.functional as f
from model.backbone.densenet_169 import *


__all__ = ['MFNet']

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

class MFNet(nn.Module):
    def __init__(self):
        super(MFNet, self).__init__()
        # ------------------------  1st directive filter  ---------------------------- #
        self.conv1_1 = nn.Conv2d(1280, 320, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_3 = nn.Conv2d(320, 80, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_5 = nn.Conv2d(80, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_7 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_9 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))

        # ------------------------  2nd directive filter  ---------------------------- #
        self.conv2_1 = nn.Conv2d(1280, 320, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_3 = nn.Conv2d(320, 80, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_5 = nn.Conv2d(80, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_7 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_9 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))

        # ---------------------------  saliency decoder  ------------------------------ #
        self.side3_1_2 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1))
        self.side3_2_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.side3_3_2 = nn.ModuleList()
        self.side3_3_2.append(MHSA(64, 64, 64, heads=4))
        self.side3_3_2 = nn.Sequential(*self.side3_3_2)
        self.sidebn3_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

        self.side4_1_2 = nn.Conv2d(512, 128, kernel_size=(3, 3), padding=(1, 1))
        self.side4_2_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.side4_3_2 = nn.ModuleList()
        self.side4_3_2.append(MHSA(64, 32, 32, heads=4))
        self.side4_3_2 = nn.Sequential(*self.side4_3_2)
        self.sidebn4_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

        self.side5_1_2 = nn.Conv2d(1280, 128, kernel_size=(3, 3), padding=(1, 1))
        self.side5_2_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.side5_3_2 = nn.ModuleList()
        self.side5_3_2.append(MHSA(64, 16, 16, heads=4))
        self.side5_3_2 = nn.Sequential(*self.side5_3_2)
        self.sidebn5_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        # self.side5_3_2 = nn.Conv2d(64, 1, kernel_size=(3, 3), padding=(1, 1))

        self.side3cat2 = nn.Conv2d(192, 64, kernel_size=(3, 3), padding=(1, 1))
        self.side4cat2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.side3out2 = nn.Conv2d(64, 1, kernel_size=(3, 3), padding=(1, 1))
        self.side4out2 = nn.Conv2d(64, 1, kernel_size=(3, 3), padding=(1, 1))

        # -----------------------------  others  -------------------------------- #
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self._initialize_weights()

        # ---------------------------  shared encoder  ----------------------------- #
        self.densenet = densenet169(pretrained=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # ---------------------------  shared encoder  -------------- ---------------- #
        x3, x4, x5 = self.densenet(x)

        # ------------------------  1st directive filter  ---------------------------- #
        sal1 = (self.conv1_9(self.upsample(self.conv1_7(
                                 self.upsample(self.conv1_5(
                                    self.upsample(self.conv1_3(
                                        self.upsample(self.conv1_1(x5))))))))))

        # ------------------------  2nd directive filter  ---------------------------- #
        sal2 = (self.conv2_9(self.upsample(self.conv2_7(
                                self.upsample(self.conv2_5(
                                    self.upsample(self.conv2_3(
                                        self.upsample(self.conv2_1(x5))))))))))

        # ---------------------------  saliency decoder  ------------------------------ #
        h_side3_2 = self.sidebn3_2(self.side3_3_2(self.side3_2_2(self.side3_1_2(x3))))
        h_side4_2 = self.sidebn4_2(self.upsample(self.side4_3_2(self.side4_2_2(self.side4_1_2(x4)))))
        h_side5_2 = self.sidebn5_2(self.upsample(self.upsample(self.side5_3_2(self.side5_2_2(self.side5_1_2(x5))))))

        sal3 = self.side3out2(self.side3cat2(torch.cat((h_side5_2, h_side4_2, h_side3_2), 1)))
        sal3 = f.interpolate(sal3, scale_factor=4, mode='bilinear', align_corners=False)

        return sal1.sigmoid(), sal2.sigmoid(), sal3.sigmoid()
