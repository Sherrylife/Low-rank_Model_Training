import torch.nn as nn
import torch

class FactorizedLinear(nn.Module):
    def __init__(self, input_size, output_size, low_rank_ratio=0.2, bias=True):
        """
        The original 2-D linear layer with tensor shape
        (output_size, input_size)
        will be factorized into two 2-D linear layers. with
        tensor shape : (output_size, r) and (r, input_size),
        where r = min(output_size, input_size) * low_rank_ratio
        :param input_size:
        :param output_size:
        :param low_rank_ratio:
        :param bias:
        """
        super(FactorizedLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ratio_LR = low_rank_ratio
        self.bias = bias
        self.r = int(
            min(self.input_size, self.output_size) * self.ratio_LR
        )

        assert self.r > 0, "the low rank ratio is too small"

        modules = [
            nn.Linear(in_features=self.input_size, out_features=self.r, bias=self.bias),
            nn.Linear(in_features=self.r, out_features=self.output_size, bias=self.bias)
        ]

        self.linear = nn.Sequential(*modules)

    def recover(self):
        linear1 = self.linear[1]
        U = linear1.weight.data
        linear2 = self.linear[0]
        VT = linear2.weight.data
        W = torch.matmul(U, VT)
        return W

    def frobenius_loss(self):
        # note: use `weight` instead of `weight.data`
        linear1 = self.linear[1]
        U = linear1.weight
        linear2 = self.linear[0]
        VT = linear2.weight
        W = torch.matmul(U, VT)
        loss = torch.norm(W,  p='fro')**2
        return loss

    def L2_loss(self):
        linear1 = self.linear[1]
        U = linear1.weight
        linear2 = self.linear[0]
        VT = linear2.weight
        loss = torch.norm(U, p='fro')**2 + torch.norm(VT, p='fro')**2
        return loss

    def kronecker_loss(self):
        linear1 = self.linear[1]
        U = linear1.weight
        linear2 = self.linear[0]
        VT = linear2.weight
        loss = (torch.norm(U, p='fro')**2) * (torch.norm(VT, p='fro')**2)
        return loss

    def forward(self, x):
        return self.linear(x)

class FactorizedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 padding=0, low_rank_ratio=0.2, stride=1, bias=False):
        """
        The original 2-D convolutional filter with tensor shape
        (out_channels, in_channels, kernel_size, kernel_size)
        will be factorized into two 1-D convolutional filter, with
        tensor shape: (out_channels, r, kernel_size, 1) and
        (r, in_channels, 1, kernel_size), where
        r = min(out_channels * kernel_size, in_channels * kernel_size) * low_rank_ratio
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param dilation:
        :param low_rank_ratio:
        :param padding:
        :param stride:
        :param bias:
        """

        # TODO: test `dilation` and `padding`
        super(FactorizedConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.r = int(
            min(in_channels * self.kernel_size, out_channels * self.kernel_size) * low_rank_ratio
        )

        assert self.r > 0, "the low rank ratio is too small"

        modules = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.r,
                kernel_size=(1, self.kernel_size),
                padding=(0, self.padding),
                stride=(1, self.stride),
                dilation=(1, self.dilation),
                bias=self.bias,
            ),
            nn.Conv2d(
                in_channels=self.r,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size, 1),
                padding=(self.padding, 0),
                stride=(self.stride, 1),
                dilation=(self.dilation, 1),
                bias=self.bias,
            )
        ]
        self.conv = nn.Sequential(*modules)

    def recover(self):
        # It seems a wrong way
        # conv1 = self.conv[0] # (rank, in_planes, 1, 3)
        # a, b, c, d = conv1.weight.shape
        # dim1, dim2 = b * d, a * c
        # VT = conv1.weight.data.reshape(dim1, dim2)
        # conv2 = self.conv[1] # (out_planes, rank, 3, 1)
        # a, b, c, d = conv2.weight.shape
        # dim1, dim2 = b * d, a * c
        # U = conv2.weight.data.reshape(dim1, dim2)
        # W = torch.matmul(VT, U).reshape(self.out_channels, self.in_channels, 3, 3)

        conv1 = self.conv[0] # (rank, in_planes, 1, 3)
        conv1.weight.data = conv1.weight.data.permute(1, 3, 2, 0)
        a, b, c, d = conv1.weight.shape
        dim1, dim2 = a * b, c * d
        VT = conv1.weight.data.reshape(dim1, dim2)
        conv2 = self.conv[1] # (out_planes, rank, 3, 1)
        conv2.weight.data = conv2.weight.data.permute(0, 2, 1, 3)
        a, b, c, d = conv2.weight.shape
        dim1, dim2 = a * b, c * d
        U = conv2.weight.data.reshape(dim1, dim2)
        W = torch.matmul(U, VT.T).reshape(self.out_channels, 3, self.in_channels, 3,).permute(0, 2, 1, 3)
        return W

    def frobenius_loss(self):
        conv1 = self.conv[0]
        conv2 = self.conv[1]

        temp_VT = conv1.weight.permute(1, 3, 2, 0)
        a, b, c, d = temp_VT.data.shape
        dim1, dim2 = a * b, c * d
        VT = torch.reshape(temp_VT, (dim1, dim2))

        temp_UT = conv2.weight.permute(0, 2, 1, 3)
        a, b, c, d = temp_UT.data.shape
        dim1, dim2 = a * b, c * d
        U = torch.reshape(temp_UT, (dim1, dim2))

        loss = torch.norm(torch.matmul(U, torch.transpose(VT, 0, 1)), p='fro')**2
        return loss


    def L2_loss(self):
        conv1 = self.conv[0]
        conv2 = self.conv[1]
        loss = torch.norm(conv1.weight, p='fro')**2 + torch.norm(conv2.weight, p='fro')**2
        return loss

    def kronecker_loss(self):
        conv1 = self.conv[0]
        conv2 = self.conv[1]
        loss = (torch.norm(conv1.weight, p='fro')**2) * (torch.norm(conv2.weight, p='fro')**2)
        return loss

    def forward(self, x):
        return self.conv(x)