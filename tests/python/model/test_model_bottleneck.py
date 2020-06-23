import pytest
import torch

import mnm
from mnm.model import Conv2d, BatchNorm

from utils import check, randn # pylint: disable=E0401


class MNMBottleNeck(mnm.Model):
    expansion = 4

    # pylint: disable=attribute-defined-outside-init
    def build(self, in_planes, planes, stride=1):
        self.bn1 = BatchNorm(in_planes)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv2 = Conv2d(planes,
                            planes,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            bias=False)
        self.bn3 = BatchNorm(planes)
        self.conv3 = Conv2d(planes,
                            self.expansion * planes,
                            kernel_size=1,
                            bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Conv2d(in_planes,
                                   self.expansion * planes,
                                   kernel_size=1,
                                   stride=stride,
                                   bias=False)
        else:
            self.shortcut = None

    # pylint: enable=attribute-defined-outside-init

    @mnm.model.trace
    def forward(self, x):
        out = mnm.relu(self.bn1(x))
        if self.shortcut is None:
            shortcut = x
        else:
            shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(mnm.relu(self.bn2(out)))
        out = self.conv3(mnm.relu(self.bn3(out)))
        out = mnm.add(out, shortcut)
        return out


class TorchPreActBottleneck(torch.nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(TorchPreActBottleneck, self).__init__()
        from torch import nn  # pylint: disable=import-outside-toplevel
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False))

    def forward(self, x):  # pylint: disable=arguments-differ
        from torch.nn import functional as F  # pylint: disable=import-outside-toplevel
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


def t2m_param(param):
    return mnm.ndarray(param.detach().numpy(), ctx="cuda")  # pylint: disable=unexpected-keyword-arg


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("config", [
    ((64, 64, 1), (16, 64, 32, 32)),
    ((256, 64, 1), (16, 256, 32, 32)),
    ((256, 64, 1), (16, 256, 32, 32)),
    ((256, 128, 2), (16, 256, 32, 32)),
    ((512, 128, 1), (16, 512, 16, 16)),
    ((512, 128, 1), (16, 512, 16, 16)),
    ((512, 128, 1), (16, 512, 16, 16)),
    ((512, 256, 2), (16, 512, 16, 16)),
])
@pytest.mark.parametrize("is_train", [True, False])
def test_bottleneck(config, is_train):
    m_block = MNMBottleNeck(*config[0])
    t_block = TorchPreActBottleneck(*config[0])
    # set the parameters to be exactly the same
    # pylint: disable=attribute-defined-outside-init,invalid-name
    m_block.bn1.w = t2m_param(t_block.bn1.weight)
    m_block.bn1.b = t2m_param(t_block.bn1.bias)
    m_block.bn1.running_mean = t2m_param(t_block.bn1.running_mean)
    m_block.bn1.running_var = t2m_param(t_block.bn1.running_var)
    m_block.conv1.w = t2m_param(t_block.conv1.weight)
    m_block.bn2.w = t2m_param(t_block.bn2.weight)
    m_block.bn2.b = t2m_param(t_block.bn2.bias)
    m_block.bn2.running_mean = t2m_param(t_block.bn2.running_mean)
    m_block.bn2.running_var = t2m_param(t_block.bn2.running_var)
    m_block.conv2.w = t2m_param(t_block.conv2.weight)
    m_block.bn3.w = t2m_param(t_block.bn3.weight)
    m_block.bn3.b = t2m_param(t_block.bn3.bias)
    m_block.bn3.running_mean = t2m_param(t_block.bn3.running_mean)
    m_block.bn3.running_var = t2m_param(t_block.bn3.running_var)
    m_block.conv3.w = t2m_param(t_block.conv3.weight)
    if m_block.shortcut is not None:
        m_block.shortcut.w = t2m_param(t_block.shortcut[0].weight)
    # pylint: enable=attribute-defined-outside-init,invalid-name
    # set train/eval mode
    if is_train:
        m_block.train_mode()
        t_block.train()
    else:
        m_block.infer_mode()
        t_block.eval()
    # run the model
    m_x, t_x = randn(config[1], requires_grad=is_train)
    t_y = t_block(t_x)
    m_y = m_block(m_x)
    if is_train:
        m_dy, t_dy = randn(m_y.shape, std=m_y.asnumpy().std() * 0.0001)
        t_y.backward(t_dy)
        m_y.backward(m_dy)
    # check outputs
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    # check model parameters
    mapping = {
        "bn1.w": t_block.bn1.weight,
        "bn1.b": t_block.bn1.bias,
        "bn1.running_mean": t_block.bn1.running_mean,
        "bn1.running_var": t_block.bn1.running_var,
        "conv1.w": t_block.conv1.weight,
        "bn2.w": t_block.bn2.weight,
        "bn2.b": t_block.bn2.bias,
        "bn2.running_mean": t_block.bn2.running_mean,
        "bn2.running_var": t_block.bn2.running_var,
        "conv2.w": t_block.conv2.weight,
        "bn3.w": t_block.bn3.weight,
        "bn3.b": t_block.bn3.bias,
        "bn3.running_mean": t_block.bn3.running_mean,
        "bn3.running_var": t_block.bn3.running_var,
        "conv3.w": t_block.conv3.weight,
    }
    if m_block.shortcut is not None:
        mapping["shortcut.w"] = t_block.shortcut[0].weight
    m_dict = m_block.state()
    for m_name, t_v in mapping.items():
        m_v = m_dict[m_name]
        check(m_v, t_v, rtol=1e-4, atol=1e-4)
    if not is_train:
        return
    for m_name, t_v in mapping.items():
        m_v = m_dict[m_name]
        if "running_" not in m_name:
            print(m_name)
            check(m_v.grad, t_v.grad, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])