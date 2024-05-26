import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bias=False,
                 kernel_size=3, down_sample=None, dropout_rate=0.):
        super(BasicBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = planes
        self.down_sample = down_sample
        self.stride = stride
        self.bias = bias
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=self.kernel_size,
                               stride=self.stride, padding=1, bias=self.bias)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.0, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=self.kernel_size,
                               stride=1, padding=1, bias=self.bias)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.0, track_running_stats=False)
        self.dropout = nn.Dropout(self.dropout_rate)



    def decom(self, ratio_LR):

        a, b, c, d = self.conv1.weight.shape  # (out_planes, in_planes, k, k)
        dim1, dim2 = a * c, b * d
        rank = int(min(dim1, dim2) * ratio_LR)

        W = self.conv1.weight.data.reshape(dim1, dim2)
        U, S, V = torch.svd(W)
        sqrtS = torch.diag(torch.sqrt(S[:rank]))
        new_U, new_V = torch.matmul(U[:, :rank], sqrtS), torch.matmul(V[:, :rank], sqrtS).T
        self.conv1 = FactorizedConv(
            in_channels=self.in_planes,
            out_channels=self.out_planes,
            kernel_size=c,
            low_rank_ratio=ratio_LR,
            stride=self.stride,
            bias=self.bias
        )
        self.conv1.conv[0].weight.data = new_V.reshape(self.in_planes, c, 1, rank).permute(3, 0, 2, 1)
        self.conv1.conv[1].weight.data = new_U.reshape(self.out_planes, c, rank, 1).permute(0, 2, 1, 3)

        a, b, c, d = self.conv2.weight.shape
        dim1, dim2 = a * c, b * d
        rank = int(min(dim1, dim2) * ratio_LR)

        W = self.conv2.weight.data.reshape(dim1, dim2)
        U, S, V = torch.svd(W)
        sqrtS = torch.diag(torch.sqrt(S[:rank]))
        new_U, new_V = torch.matmul(U[:, :rank], sqrtS), torch.matmul(V[:, :rank], sqrtS).T
        self.conv2 = FactorizedConv(
            in_channels=self.out_planes,
            out_channels=self.out_planes,
            kernel_size=c,
            low_rank_ratio=ratio_LR,
            stride=self.stride,
            bias=self.bias
        )
        self.conv2.conv[0].weight.data = new_V.reshape(self.out_planes, c, 1, rank).permute(3, 0, 2, 1)
        self.conv2.conv[1].weight.data = new_U.reshape(self.out_planes, c, rank, 1).permute(0, 2, 1, 3)
        # print("Done")

    def recover(self):
        W1 = self.conv1.recover()
        W2 = self.conv2.recover()
        self.conv1 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=self.kernel_size,
                               stride=self.stride, padding=1, bias=False)
        self.conv1.weight.data = W1
        self.conv2 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=self.kernel_size,
                               stride=1, padding=1, bias=False)
        self.conv2.weight.data = W2

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bias=False, down_sample=None):
        super(Bottleneck, self).__init__()
        self.in_planes = in_planes
        self.out_planes = planes
        self.down_sample = down_sample
        self.stride = stride
        self.bias = bias

        self.conv1 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, bias=self.bias)
        self.bn1 = nn.BatchNorm2d(self.out_planes, momentum=0.0, track_running_stats=False)
        self.conv2 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=self.stride,
                               padding=1, bias=self.bias)
        self.bn2 = nn.BatchNorm2d(self.out_planes, momentum=0.0, track_running_stats=False)
        self.conv3 = nn.Conv2d(self.out_planes, self.out_planes*4, kernel_size=1, bias=self.bias)
        self.bn3 = nn.BatchNorm2d(self.out_planes*4, momentum=0.0, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)


    def decom(self, ratio_LR):
        a, b, c, d = self.conv1.weight.shape  # (out_planes, in_planes, k, k)
        dim1, dim2 = a * c, b * d
        rank = int(min(dim1, dim2) * ratio_LR)
        W = self.conv1.weight.data.reshape(dim1, dim2)
        U, S, V = torch.svd(W)
        sqrtS = torch.diag(torch.sqrt(S[:rank]))
        new_U, new_V = torch.matmul(U[:, :rank], sqrtS), torch.matmul(V[:, :rank], sqrtS).T
        self.conv1 = FactorizedConv(
            in_channels=self.in_planes,
            out_channels=self.out_planes,
            kernel_size=c,
            low_rank_ratio=ratio_LR,
            bias=self.bias
        )
        self.conv1.conv[0].weight.data = new_V.reshape(self.in_planes, c, 1, rank).permute(3, 0, 2, 1)
        self.conv1.conv[1].weight.data = new_U.reshape(self.out_planes, c, rank, 1).permute(0, 2, 1, 3)

        a, b, c, d = self.conv2.weight.shape
        dim1, dim2 = a * c, b * d
        rank = int(min(dim1, dim2) * ratio_LR)
        W = self.conv2.weight.data.reshape(dim1, dim2)
        U, S, V = torch.svd(W)
        sqrtS = torch.diag(torch.sqrt(S[:rank]))
        new_U, new_V = torch.matmul(U[:, :rank], sqrtS), torch.matmul(V[:, :rank], sqrtS).T
        self.conv2 = FactorizedConv(
            in_channels=self.out_planes,
            out_channels=self.out_planes,
            kernel_size=c,
            low_rank_ratio=ratio_LR,
            stride=self.stride,
            bias=self.bias
        )
        self.conv2.conv[0].weight.data = new_V.reshape(self.out_planes, c, 1, rank).permute(3, 0, 2, 1)
        self.conv2.conv[1].weight.data = new_U.reshape(self.out_planes, c, rank, 1).permute(0, 2, 1, 3)

        a, b, c, d = self.conv3.weight.shape
        dim1, dim2 = a * c, b * d
        rank = int(min(dim1, dim2) * ratio_LR)
        W = self.conv3.weight.data.reshape(dim1, dim2)
        U, S, V = torch.svd(W)
        sqrtS = torch.diag(torch.sqrt(S[:rank]))
        new_U, new_V = torch.matmul(U[:, :rank], sqrtS), torch.matmul(V[:, :rank], sqrtS).T
        self.conv3 = FactorizedConv(
            in_channels=self.out_planes,
            out_channels=self.out_planes*4,
            kernel_size=c,
            low_rank_ratio=ratio_LR,
            bias=self.bias
        )
        self.conv3.conv[0].weight.data = new_V.reshape(self.out_planes, c, 1, rank).permute(3, 0, 2, 1)
        self.conv3.conv[1].weight.data = new_U.reshape(self.out_planes*4, c, rank, 1).permute(0, 2, 1, 3)



    def recover(self):
        W1 = self.conv1.recover()
        W2 = self.conv2.recover()
        W3 = self.conv3.recover()
        self.conv1 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, bias=self.bias)
        self.conv1.weight.data = W1
        self.conv2 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3,
                               stride=self.stride, padding=1, bias=self.bias)
        self.conv2.weight.data = W2
        self.conv3 = nn.Conv2d(self.out_planes, self.out_planes*4, kernel_size=1, bias=self.bias)
        self.conv3.weight.data = W3


    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


