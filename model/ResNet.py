import torch
import torch.nn as nn
from model.cnn_base import BasicBlock, Bottleneck
from model.low_rank_base import *
from utils.experiment_config import *


class LowRankBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, low_rank_ratio=0.2,
                 bias=False, down_sample=None):
        super(LowRankBottleneck, self).__init__()
        self.in_planes = in_planes
        self.out_planes = planes
        self.low_rank_ratio = low_rank_ratio
        self.stride = stride
        self.bias = bias

        self.conv1 = FactorizedConv(
            in_channels=self.in_planes,
            out_channels=self.out_planes,
            kernel_size=1,
            bias=self.bias
        )
        self.bn1 = nn.BatchNorm2d(self.out_planes, momentum=0., track_running_stats=False)
        self.conv2 = FactorizedConv(
            in_channels=self.out_planes,
            out_channels=self.out_planes,
            kernel_size=3,
            stride=self.stride,
            bias=self.bias,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(self.out_planes, momentum=0., track_running_stats=False)
        self.conv3 = FactorizedConv(
            in_channels=self.out_planes,
            out_channels=self.out_planes*4,
            kernel_size=1,
            bias=self.bias
        )
        self.bn3 = nn.BatchNorm2d(self.out_planes*4, momentum=0., track_running_stats=False)
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
            out_channels=self.out_planes * 4,
            kernel_size=c,
            low_rank_ratio=ratio_LR,
            bias=self.bias
        )
        self.conv3.conv[0].weight.data = new_V.reshape(self.out_planes, c, 1, rank).permute(3, 0, 2, 1)
        self.conv3.conv[1].weight.data = new_U.reshape(self.out_planes * 4, c, rank, 1).permute(0, 2, 1, 3)

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

    def cal_smallest_svdvals(self):
        """
        calculate the smallest singular value of a residual block
        """

        temp_VT = self.conv1.conv[0].weight.data.clone().permute(1, 3, 2, 0)
        a, b, c, d = temp_VT.shape
        dim1, dim2 = a * b, c * d
        V1 = torch.reshape(temp_VT, (dim1, dim2))

        temp_UT = self.conv1.conv[1].weight.data.clone().permute(0, 2, 1, 3)
        a, b, c, d = temp_UT.shape
        dim1, dim2 = a * b, c * d
        U1 = torch.reshape(temp_UT, (dim1, dim2))

        S1 = torch.linalg.svdvals(V1)
        S2 = torch.linalg.svdvals(U1)

        temp_VT = self.conv2.conv[0].weight.data.clone().permute(1, 3, 2, 0)
        a, b, c, d = temp_VT.shape
        dim1, dim2 = a * b, c * d
        V2 = torch.reshape(temp_VT, (dim1, dim2))

        temp_UT = self.conv2.conv[1].weight.data.clone().permute(0, 2, 1, 3)
        a, b, c, d = temp_UT.shape
        dim1, dim2 = a * b, c * d
        U2 = torch.reshape(temp_UT, (dim1, dim2))

        S3 = torch.linalg.svdvals(V2)
        S4 = torch.linalg.svdvals(U2)

        temp_VT = self.conv3.conv[0].weight.data.clone().permute(1, 3, 2, 0)
        a, b, c, d = temp_VT.shape
        dim1, dim2 = a * b, c * d
        V3 = torch.reshape(temp_VT, (dim1, dim2))

        temp_UT = self.conv3.conv[1].weight.data.clone().permute(0, 2, 1, 3)
        a, b, c, d = temp_UT.shape
        dim1, dim2 = a * b, c * d
        U3 = torch.reshape(temp_UT, (dim1, dim2))

        S5 = torch.linalg.svdvals(V3)
        S6 = torch.linalg.svdvals(U3)

        S = torch.cat([S1, S2, S3, S4, S5, S6], dim=0)
        return torch.min(S).item()

    def frobenius_loss(self):
        loss1 = self.conv1.frobenius_loss()
        loss2 = self.conv2.frobenius_loss()
        loss3 = self.conv3.frobenius_loss()
        return loss1+loss2+loss3

    def kronecker_loss(self):
        loss1 = self.conv1.kronecker_loss()
        loss2 = self.conv2.kronecker_loss()
        loss3 = self.conv3.kronecker_loss()

        return loss1+loss2+loss3

    def L2_loss(self):
        loss1 = self.conv1.L2_loss()
        loss2 = self.conv2.L2_loss()
        loss3 = self.conv3.L2_loss()

        return loss1+loss2+loss3


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





class LowRankBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, low_rank_ratio=0.2,
                 bias=False, kernel_size=3, down_sample=None, dropout_rate=0.):
        super(LowRankBasicBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = planes
        self.low_rank_ratio = low_rank_ratio
        self.stride = stride
        self.kernel_size = kernel_size
        self.bias = bias
        self.dropout_rate = dropout_rate

        # self.conv1 = DecomBlock(in_planes, planes, n_basis, stride=stride, bias=False)
        self.conv1 = FactorizedConv(
            in_channels=self.in_planes,
            out_channels=self.out_planes,
            kernel_size=self.kernel_size,
            low_rank_ratio=self.low_rank_ratio,
            stride=self.stride,
            bias=self.bias,
            padding=1,
        )

        self.bn1 = nn.BatchNorm2d(self.out_planes, momentum=0.0, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

        # self.conv2 = DecomBlock(planes, planes, n_basis, stride=1, bias=False)
        self.conv2 = FactorizedConv(
            in_channels=self.out_planes,
            out_channels=self.out_planes,
            kernel_size=self.kernel_size,
            low_rank_ratio=self.low_rank_ratio,
            stride=1,
            bias=self.bias,
            padding=1,
        )

        self.bn2 = nn.BatchNorm2d(self.out_planes, momentum=0.0, track_running_stats=False)
        self.down_sample = down_sample
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
            bias=self.bias,
            padding=1,
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
            stride=1,
            bias=self.bias,
            padding=1,
        )
        self.conv2.conv[0].weight.data = new_V.reshape(self.out_planes, c, 1, rank).permute(3, 0, 2, 1)
        self.conv2.conv[1].weight.data = new_U.reshape(self.out_planes, c, rank, 1).permute(0, 2, 1, 3)
        # print("Done")


    def cal_smallest_svdvals(self):
        """
        calculate the smallest singular value of a residual block
        """

        temp_VT = self.conv1.conv[0].weight.data.clone().permute(1, 3, 2, 0)
        a, b, c, d = temp_VT.shape
        dim1, dim2 = a * b, c * d
        V1 = torch.reshape(temp_VT, (dim1, dim2))

        temp_UT = self.conv1.conv[1].weight.data.clone().permute(0, 2, 1, 3)
        a, b, c, d = temp_UT.shape
        dim1, dim2 = a * b, c * d
        U1 = torch.reshape(temp_UT, (dim1, dim2))

        S1 = torch.linalg.svdvals(V1)
        S2 = torch.linalg.svdvals(U1)

        temp_VT = self.conv2.conv[0].weight.data.clone().permute(1, 3, 2, 0)
        a, b, c, d = temp_VT.shape
        dim1, dim2 = a * b, c * d
        V2 = torch.reshape(temp_VT, (dim1, dim2))

        temp_UT = self.conv2.conv[1].weight.data.clone().permute(0, 2, 1, 3)
        a, b, c, d = temp_UT.shape
        dim1, dim2 = a * b, c * d
        U2 = torch.reshape(temp_UT, (dim1, dim2))

        S3 = torch.linalg.svdvals(V2)
        S4 = torch.linalg.svdvals(U2)

        S = torch.cat([S1, S2, S3, S4], dim=0)
        return torch.min(S).item()

    def recover(self):
        W1 = self.conv1.recover()
        W2 = self.conv2.recover()
        self.conv1 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.conv1.weight.data = W1
        self.conv2 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight.data = W2

    def frobenius_loss(self):
        loss1 = self.conv1.frobenius_loss()
        loss2 = self.conv2.frobenius_loss()
        return loss1+loss2

    def kronecker_loss(self):
        loss1 = self.conv1.kronecker_loss()
        loss2 = self.conv2.kronecker_loss()
        return loss1+loss2

    def L2_loss(self):
        loss1 = self.conv1.L2_loss()
        loss2 = self.conv2.L2_loss()
        return loss1+loss2

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


class HyperResNet(nn.Module):

    def __init__(self, dataset_name, input_size, hidden_size, output_size, num_blocks,
                 default_block, ratio_LR, decom_rule, dropout_rate=0., args=None):
        super(HyperResNet, self).__init__()
        """
        decom_rule is a 2-tuple like (block_index, layer_index).
        For resnet18, block_index is selected from [0,1,2,3] and layer_index is selected from [0,1].
        Example: If we only want to decompose layers starting form the 8-th layer for resnet18, 
                 then we set decom_rule = (1, 1);
                 If we want to decompose all layer(expept head and tail layer), we can set 
                 decom_rule = (-1, 0);
                 If we don't want to decompose any layer, we can set 
                 decom_rule = (4, 0).
        """
        self.args = args
        self.device = args['device']
        self.dataset_name = dataset_name
        self.intput_size = input_size
        self.in_planes = hidden_size[0] # a tmp variable for generating layers
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.feature_num = hidden_size[-1]
        self.output_size = output_size
        self.decom_rule = decom_rule
        self.ratio_LR = ratio_LR
        self.dropout_rate = dropout_rate
        self.default_block = default_block

        self.head = nn.Sequential(
            nn.Conv2d(self.intput_size, self.hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0., track_running_stats=False),
        )
        self.relu = nn.ReLU(inplace=True)

        # initialization the hybrid model
        strides = [1, 2, 2, 2]
        all_layers, common_layers, personalized_layers = [], [], []
        common_layers.append(self.head)
        for block_idx in range(4):
            if block_idx < self.decom_rule[0]:
                layer = self._make_original_layer(
                    block=self.default_block,
                    planes=hidden_size[block_idx],
                    blocks=num_blocks[block_idx],
                    stride=strides[block_idx]
                )
                all_layers.append(layer)
                common_layers.append(layer)
            elif block_idx == self.decom_rule[0]:
                layer = self._make_hybrid_layer(
                    original_block=self.default_block,
                    low_rank_block=LowRankBasicBlock,
                    planes=hidden_size[block_idx],
                    blocks=num_blocks[block_idx],
                    stride=strides[block_idx],
                    start_decom_idx=self.decom_rule[1],
                )
                all_layers.append(layer)
                for layer_idx in range(self.decom_rule[1]):
                    common_layers.append(layer[layer_idx])
                for layer_idx in range(self.decom_rule[1], self.num_blocks[block_idx]):
                    personalized_layers.append(layer[layer_idx])

            else:
                """ block_idx > self.decom_rule[0] """
                layer = self._make_low_rank_layer(
                    block=LowRankBasicBlock,
                    planes=hidden_size[block_idx],
                    blocks=num_blocks[block_idx],
                    stride=strides[block_idx]
                )
                all_layers.append(layer)
                personalized_layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tail = nn.Linear(self.feature_num, self.output_size)
        personalized_layers.append(self.tail)

        self.body = nn.Sequential(*all_layers)
        self.common = nn.Sequential(*common_layers)
        self.personalized = nn.Sequential(*personalized_layers)

        self._init_parameters()

    def _init_parameters(self):
        # initialization for the hybrid model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_low_rank_layer(self, block, planes, blocks, stride=1):
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=False)
            )
        else:
            down_sample = None

        layers = []
        layers.append(block(self.in_planes, planes, stride=stride, down_sample=down_sample,
                            dropout_rate=self.dropout_rate))

        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def _make_hybrid_layer(self, original_block, low_rank_block, planes, blocks,
                           stride=1, start_decom_idx=0):
        """
        :param start_decom_idx: range from [0, blocks-1]
        """

        block = low_rank_block if start_decom_idx == 0 else original_block
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=False)
            )
        else:
            down_sample = None

        layers = []

        layers.append(
                block(self.in_planes, planes, stride, down_sample=down_sample, dropout_rate=self.dropout_rate))

        self.in_planes = planes * block.expansion

        for idx in range(1, blocks):
            block = original_block if idx < start_decom_idx else low_rank_block
            layers.append(
                block(self.in_planes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)


    def _make_original_layer(self, block, planes, blocks, stride=1):

        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=False)
            )
        else:
            down_sample = None

        layers = []
        layers.append(
            block(self.in_planes, planes, stride, down_sample=down_sample, dropout_rate=self.dropout_rate))

        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def recover_low_rank_layer(self):
        # for idx in range(self.decom_rule[0], 4):
        #     meta_block = self.body.pop(idx)
        #     for j in range(layer_index):
        #         meta_block[j].recover()
        #     meta_block_params = copy.deepcopy(meta_block.state_dict())
        #     self.body.insert(idx, meta_block)
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length-1:
                if isinstance(block, LowRankBasicBlock):
                    block.recover()
                else:
                    for j in range(len(block)):
                        block[j].recover()


    def decom_original_layer(self, ratio_LR=0.2):
        # for i, idx in enumerate(block_index):
        #     large_block = self.body.pop(idx)
        #     for j in range(layer_index):
        #         large_block[j].decom(self.ratio_LR)
        #     # large_block_params = copy.deepcopy(large_block.state_dict())
        #     self.body.insert(idx, large_block)
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length-1:
                if isinstance(block, LowRankBasicBlock):
                    block.decom(ratio_LR=ratio_LR)
                else:
                    for j in range(len(block)):
                        block[j].decom(ratio_LR=ratio_LR)

    def frobenius_decay(self):
        loss = torch.tensor(0.).to(self.device)
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length-1:
                if isinstance(block, LowRankBasicBlock):
                    loss += block.frobenius_loss()
                else:
                    for j in range(len(block)):
                        loss += block[j].frobenius_loss()
        return loss

    def kronecker_decay(self):
        loss = torch.tensor(0.).to(self.device)
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length-1:
                if isinstance(block, LowRankBasicBlock):
                    loss += block.kronecker_loss()
                else:
                    for j in range(len(block)):
                        loss += block[j].kronecker_loss()
        return loss

    def L2_decay(self):
        loss = torch.tensor(0.).to(self.device)
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length-1:
                if isinstance(block, LowRankBasicBlock):
                    loss += block.L2_loss()
                else:
                    for j in range(len(block)):
                        loss += block[j].L2_loss()
        return loss

    def cal_smallest_svdvals(self):
        """
        calculate the smallest singular value of each residual block.
        For example, if the model is resnet18, then there are 8 residual blocks.
        """
        smallest_svdvals = []
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length-1:
                if isinstance(block, LowRankBasicBlock):
                    smallest_svdvals.append(block.cal_smallest_svdvals())
                else:
                    for j in range(len(block)):
                        smallest_svdvals.append(block[j].cal_smallest_svdvals())
        return smallest_svdvals

    def forward(self, x, ):
        x = self.head(x)
        x = self.relu(x)
        # if self.dataset_name == 'tinyImagenet':
        #     x = self.maxpool(x)
        for idx, layer in enumerate(self.body):
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.tail(x)

        return x


class OriginalResNet(nn.Module):
    def __init__(self, dataset_name, input_size, hidden_size, output_size, num_blocks,
                 default_block, dropout_rate=0.):
        super(OriginalResNet, self).__init__()
        """
        decom_rule is a 2-tuple like (block_index, layer_index).
        For resnet18, block_index is selected from [0,1,2,3] and layer_index is selected from [0,1].
        Example: If we only want to decompose layers starting form the 8-th layer for resnet18, 
                 then we set decom_rule = (1, 1);
                 If we want to decompose all layer(expept head and tail layer), we can set 
                 decom_rule = (-1, 0);
                 If we don't want to decompose any layer, we can set 
                 decom_rule = (4, 0).
        """
        self.dataset_name = dataset_name
        self.intput_size = input_size
        self.in_planes = hidden_size[0] # a tmp variable for generating layers
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.feature_num = hidden_size[-1]
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.default_block = default_block

        if self.dataset_name in ['tinyImageNet', 'ImageNet']:
            self.head = nn.Sequential(
                    nn.Conv2d(self.input_size, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(self.in_planes, momentum=0.0),
                )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.head = nn.Sequential(
                nn.Conv2d(self.intput_size, self.hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64, momentum=0., track_running_stats=False),
            )
        self.relu = nn.ReLU(inplace=True)
        strides = [1, 2, 2, 2]
        all_layers = []
        for idx in range(4):
            all_layers.append(self._make_original_layer(
                block=self.default_block,
                planes=self.hidden_size[idx],
                blocks=num_blocks[idx],
                stride=strides[idx],
            ))
        self.body = nn.Sequential(*all_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tail = nn.Linear(self.feature_num, self.output_size)

        self._init_parameters()

    def _init_parameters(self):
        # initialization for model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_original_layer(self, block, planes, blocks, stride=1):
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=False)
            )
        else:
            down_sample = None
        layers = []
        layers.append(
            block(self.in_planes, planes, stride, down_sample=down_sample, dropout_rate=self.dropout_rate))

        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x, ):
        x = self.head(x)
        x = self.relu(x)
        if self.dataset_name in ['tinyImageNet', 'ImageNet']:
            x = self.maxpool(x)
        for idx, layer in enumerate(self.body):
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.tail(x)

        return x




def LowRankResNet18(ratio_LR=1, decom_rule=[1, 1], args=None):
    dataset_name = args['dataset']
    assert dataset_name in ['cifar10', 'cifar100', 'svhn', 'tinyImageNet', 'ImageNet']

    input_size = IMAGE_RESNET_CONFIG['input_size']
    hidden_size = IMAGE_RESNET_CONFIG['hidden_size']
    output_size = IMAGE_RESNET_CONFIG['output_size'][str(dataset_name)]

    model = HyperResNet(
        dataset_name=dataset_name,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        default_block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        ratio_LR=ratio_LR,
        decom_rule=decom_rule,
        args=args,
    )

    return model

def ResNet18(args=None):
    dataset_name = args['dataset']
    assert dataset_name in ['cifar10', 'cifar100', 'svhn', 'tinyImageNet', 'ImageNet']
    input_size = IMAGE_RESNET_CONFIG['input_size']
    hidden_size = IMAGE_RESNET_CONFIG['hidden_size']
    output_size = IMAGE_RESNET_CONFIG['output_size'][str(dataset_name)]
    model = OriginalResNet(
        dataset_name=dataset_name,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        default_block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
    )
    return model

def ResNet101(args=None):
    dataset_name = args['dataset']
    assert dataset_name in ['cifar10', 'cifar100', 'svhn', 'tinyImageNet', 'ImageNet']
    input_size = IMAGE_RESNET_CONFIG['input_size']
    hidden_size = IMAGE_RESNET_CONFIG['hidden_size']
    output_size = IMAGE_RESNET_CONFIG['output_size'][str(dataset_name)]
    model = OriginalResNet(
        dataset_name=dataset_name,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        default_block=BasicBlock,
        num_blocks=[3, 4, 23, 3],
    )
    return model

def LowRankResNet101(ratio_LR=1, decom_rule=[1, 1], args=None):
    dataset_name = args['dataset']
    assert dataset_name in ['cifar10', 'cifar100', 'svhn', 'tinyImageNet', 'ImageNet']

    input_size = IMAGE_RESNET_CONFIG['input_size']
    hidden_size = IMAGE_RESNET_CONFIG['hidden_size']
    output_size = IMAGE_RESNET_CONFIG['output_size'][str(dataset_name)]

    model = HyperResNet(
        dataset_name=dataset_name,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        default_block=BasicBlock,
        num_blocks=[3, 4, 23, 3],
        ratio_LR=ratio_LR,
        decom_rule=decom_rule,
        args=args,
    )
    return model