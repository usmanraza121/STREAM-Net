"""Temporal Shift Convolutional Attention Network (TS-CAN).
Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
NeurIPS, 2020
Xin Liu, Josh Fromm, Shwetak Patel, Daniel McDuff
"""
import os
from calflops import calculate_flops
import time
import torch
import torch.nn as nn
from torchsummary import summary
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = nn.Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

# ==============================
def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, groups=groups,bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class PConv(nn.Module):
    def __init__(self, channels, reduction=16):
        super(PConv, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,groups= 2,   padding=0 ) # group= 2 
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, groups= 2, padding=0) # group= 2 
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x
        
class SSFEM(nn.Module):

    def __init__(self, inplanes, outplanes, dilat, downsample=None, stride=1, t=1, scales=4, se=True, norm_layer=None):
        super(SSFEM, self).__init__()
        if inplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = inplanes * t
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales,
                                            groups=(bottleneck_planes // scales),dilation=dilat[i],
                                            padding=1*dilat[i]) for i in range(scales)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales)])
        self.conv3 = conv1x1(bottleneck_planes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = PConv(outplanes) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)
            # print('SE module added')

        if self.downsample is not None:
            identity = self.downsample(identity)
            # print('identity module')

        out += identity
        out = self.relu(out)

        return out


class WTSM1(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(WTSM1, self).__init__()
        self.n_segment = n_segment  #n_segment = 10
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x1 = x
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, -1, :fold] = x[:, 0, :fold] # wrap left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]  # wrap right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # no shift for final fold
        y = out.view(nt, c, h, w)
        return out.view(nt, c, h, w)

class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)


class TSCAN(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20, img_size=36, 
                 sampling_dropout_rate=0.3, n_dropout_samples=30 ):
        """Definition of TS_CAN.
        Args:
          in_channels: the number of input channel. Default: 3
          frame_depth: the number of frame (window size) used in temport shift. Default: 20
          img_size: height/width of each frame. Default: 36.
        Returns:
          TS_CAN model.
        """
        super(TSCAN, self).__init__()
        self.sampling_dropout_rate = sampling_dropout_rate  #adding for MC_dropout
        self.n_dropout_samples = n_dropout_samples          #adding for MC_dropout

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # WSTM layers WTSM1()
        # self.TSM_1 = WTSM1()
        # self.TSM_2 = WTSM1()
        # self.TSM_3 = WTSM1()
        # self.TSM_4 = WTSM1()
        # SSFEM-----------
        # self.ssfem = SSFEM(64, 64, [1, 2, 4, 8])
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(
            self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(
            self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        elif img_size == 128:
            self.final_dense_1 = nn.Linear(57600, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)
    
    #================MC_dropout==========================================#
    def enable_dropout(self):
        """
        Turns on the dropout layers, sets the dropout rate to self.sampling_dropout_rate
        """
        dropout_rates1=[0.125, 0.125, 0.125, 0.25] # dropout/2
        dropout_rates2=[0.06, 0.06, 0.06, 0.16] # dropout/3
        idx = 0
        for module in self.modules():
            if module.__class__.__name__.startswith('Dropout'):
                # Turn on dropout
                module.train()
                # print('Module:', module)
                # Set dropout rate
                module.p = self.sampling_dropout_rate #uncomment for specified dropout value
                # module.p = dropout_rates2[idx] # uncomment for dropout/2 and dropout/3
                # print('Module:', module)
                idx+=1


    def mc_sample(self, data):
        """
        Takes n Monte Carlo samples of the network by repeatedly
        applying a dropout mask and making a prediction using
        that mask.

        Args:
            data: data point to forward
        """
        samples = torch.empty((data.shape[0], self.n_dropout_samples))
        
        for i in range(self.n_dropout_samples):
            # forward push
            output2 = self(data)
            # predictions = self.softmax(output2)
            predictions = output2

            # Save results
            # samples[:, i] = predictions
            samples[:, i] = predictions.squeeze()
            # print('samples=', samples.shape)
        
        # Calculate mean and variance over the samples, return results
        sample_mean = samples.mean(dim=1).type_as(data)
        sample_var = samples.var(dim=1).type_as(data)
        
        return samples, sample_mean, sample_var
    def mc_sample2(self, data):
    # 
        pred = self(data)
        return pred


    def forward(self, inputs, params=None):
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        # diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        # d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        # d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        # d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        # r6 = self.ssfem(r6)     # SSFEM 

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return out


class MTTS_CAN(nn.Module):
    """MTTS_CAN is the multi-task (respiration) version of TS-CAN"""

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20):
        super(MTTS_CAN, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(
            self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(
            self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4_y = nn.Dropout(self.dropout_rate2)
        self.dropout_4_r = nn.Dropout(self.dropout_rate2)

        # Dense layers
        self.final_dense_1_y = nn.Linear(16384, self.nb_dense, bias=True)
        self.final_dense_2_y = nn.Linear(self.nb_dense, 1, bias=True)
        self.final_dense_1_r = nn.Linear(16384, self.nb_dense, bias=True)
        self.final_dense_2_r = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)

        d10 = torch.tanh(self.final_dense_1_y(d9))
        d11 = self.dropout_4_y(d10)
        out_y = self.final_dense_2_y(d11)

        d10 = torch.tanh(self.final_dense_1_r(d9))
        d11 = self.dropout_4_r(d10)
        out_r = self.final_dense_2_r(d11)

        return out_y, out_r

def count_par(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import torch
import time

def measure_inference_time(model, input_tensor, num_runs=100):
    model.eval()
    model.cuda()
    input_tensor = input_tensor.cuda()

    # Warm-up (to stabilize GPU performance)
    for _ in range(10):
        _ = model(input_tensor)

    # Timing
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(num_runs):
            starter.record()
            _ = model(input_tensor)
            ender.record()

            # Wait for GPU to finish
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # Time in milliseconds
            timings.append(curr_time)

    avg_time = sum(timings) / num_runs
    return avg_time  # in milliseconds
import torch
import numpy as np  # for mean and std

def measure_inference_time2(model, input_tensor, num_runs=100):
    model.eval()
    model.cuda()
    input_tensor = input_tensor.cuda()

    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(num_runs):
            starter.record()
            _ = model(input_tensor)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # milliseconds
            timings.append(curr_time)

    avg_time = np.mean(timings)
    std_time = np.std(timings)

    return avg_time, std_time

def measure_memory_usage(model, input_tensor):
    model.eval()
    model.cuda()
    input_tensor = input_tensor.cuda()

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = model(input_tensor)

    mem_used = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
    return mem_used


# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#     net = TSCAN().cuda()
#     total_par = count_par(net)
#     print(f'total parameters= {total_par}')
      
#     # print(net)
#     print('\n:')
#     t1 = torch.rand(720, 6, 36, 36).cuda()  # 720 frames, 6 channels, 36x36 pixels
#     x2 = net(t1)
#     print("input: ", t1.shape)
#     print('out:',x2.shape)
    # print(summary(net, (6, 36, 36), device='cuda' ))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model = TSCAN()
    batch_size = 720
    input_shape = (batch_size, 6, 36, 36)
    flops, macs, params = calculate_flops(model=model, 
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

    input_shape = torch.randn(batch_size, 6, 36, 36)  
    
    time_ms = measure_inference_time(model, input_shape)
    memory_mb = measure_memory_usage(model, input_shape)

    print(f"Inference Time: {time_ms:.2f} ms | GPU Memory Usage: {memory_mb:.2f} MB")

    avg_time, std_time = measure_inference_time2(model, input_shape)
    print(f"Average Inference Time: {avg_time:.4f} ms")
    print(f"Standard Deviation: {std_time:.4f} ms")