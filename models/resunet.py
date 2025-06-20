import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils


DEFAULT_BETA = 0.90  
DEFAULT_THRESHOLD = 1.0 
DEFAULT_SPIKE_GRAD = surrogate.fast_sigmoid()

class SNNResBlock(nn.Module):
    def __init__(self, cin, cout, stride, kernel_size=3, beta=DEFAULT_BETA):
        super().__init__()
        padding = kernel_size // 2

        self.bn1_block = nn.BatchNorm2d(cin)
        self.lif1_block = snn.Leaky(beta=beta, threshold=DEFAULT_THRESHOLD, spike_grad=DEFAULT_SPIKE_GRAD, init_hidden=True)
        self.conv1_block = nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
        self.bn2_block = nn.BatchNorm2d(cout)
        self.lif2_block = snn.Leaky(beta=beta, threshold=DEFAULT_THRESHOLD, spike_grad=DEFAULT_SPIKE_GRAD, init_hidden=True)
        self.conv2_block = nn.Conv2d(cout, cout, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

        self.conv_res = nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn_res = nn.BatchNorm2d(cout)

    def forward(self, x_current_in): 
        res_path = x_current_in
        
        main_path_out = self.bn1_block(x_current_in)
        main_path_out_spk = self.lif1_block(main_path_out)
        main_path_out_cur = self.conv1_block(main_path_out_spk)
        
        main_path_out = self.bn2_block(main_path_out_cur)
        main_path_out_spk = self.lif2_block(main_path_out)
        main_path_out_current = self.conv2_block(main_path_out_spk)

        res_path_current = self.bn_res(self.conv_res(res_path))
        
        final_current = main_path_out_current + res_path_current
        return final_current

def Upsample(channels):
    return nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2) 

class SNNResUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dim=32, conv_kernel_size=3, beta=DEFAULT_BETA):
        super().__init__()
        c = dim
        self.beta = beta 

        self.input_conv1 = nn.Conv2d(in_channels, c, kernel_size=3, padding=1, bias=False)
        self.input_bn1 = nn.BatchNorm2d(c)
        self.input_lif1 = snn.Leaky(beta=self.beta, threshold=DEFAULT_THRESHOLD, spike_grad=DEFAULT_SPIKE_GRAD, init_hidden=True)
        self.input_conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

        self.input_skip_conv = nn.Conv2d(in_channels, c, kernel_size=conv_kernel_size, padding=conv_kernel_size//2, bias=True)
        
        self.residual_conv_1 = SNNResBlock(c, 2*c, stride=2, kernel_size=conv_kernel_size, beta=self.beta)
        self.residual_conv_2 = SNNResBlock(2*c, 4*c, stride=2, kernel_size=conv_kernel_size, beta=self.beta)
        self.bridge = SNNResBlock(4*c, 8*c, stride=2, kernel_size=conv_kernel_size, beta=self.beta)
        
        self.upsample_1 = Upsample(8*c)
        self.up_residual_conv1 = SNNResBlock(8*c + 4*c, 4*c, stride=1, kernel_size=conv_kernel_size, beta=self.beta)
        self.upsample_2 = Upsample(4*c)
        self.up_residual_conv2 = SNNResBlock(4*c + 2*c, 2*c, stride=1, kernel_size=conv_kernel_size, beta=self.beta)
        self.upsample_3 = Upsample(2*c)
        self.up_residual_conv3 = SNNResBlock(2*c + c, c, stride=1, kernel_size=conv_kernel_size, beta=self.beta)
        
        self.output_layer_conv = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x_in, time): 
        utils.reset(self)
        sum_of_outputs_over_time = None 
        num_steps = time

        for step in range(num_steps):
            main_path_input_cur = self.input_conv1(x_in)
            main_path_input_cur = self.input_bn1(main_path_input_cur)
            main_path_input_spk = self.input_lif1(main_path_input_cur)
            main_path_input_cur = self.input_conv2(main_path_input_spk)
            skip_path_input_cur = self.input_skip_conv(x_in)
            x1_current = main_path_input_cur + skip_path_input_cur

            x2_current = self.residual_conv_1(x1_current)
            x3_current = self.residual_conv_2(x2_current)
            x4_bridge_current = self.bridge(x3_current)
            
            x4_up_current = self.upsample_1(x4_bridge_current)
            x5_cat_current = torch.cat([x4_up_current, x3_current], dim=1)
            x6_current = self.up_residual_conv1(x5_cat_current)
            
            x6_up_current = self.upsample_2(x6_current)
            x7_cat_current = torch.cat([x6_up_current, x2_current], dim=1)
            x8_current = self.up_residual_conv2(x7_cat_current)
            
            x8_up_current = self.upsample_3(x8_current)
            x9_cat_current = torch.cat([x8_up_current, x1_current], dim=1)
            x10_current = self.up_residual_conv3(x9_cat_current)
            
            current_output_this_step = self.output_layer_conv(x10_current)
            
            #summing part
            if sum_of_outputs_over_time is None:
                sum_of_outputs_over_time = current_output_this_step
            else:
                sum_of_outputs_over_time += current_output_this_step
        
        if num_steps > 0:
            averaged_output = sum_of_outputs_over_time / num_steps
        else: 
            batch_size = x_in.shape[0]
            spatial_dims = x_in.shape[2:] 
            out_c = self.output_layer_conv.out_channels
            averaged_output = torch.zeros(batch_size, out_c, *spatial_dims, device=x_in.device)

        final_output = torch.sigmoid(averaged_output) #sig working
        return final_output
