import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

index = [0, 2, 4, 6, 8, 
10, 12, 14, 16, 
18, 20, 21, 22, 23, 24, 26, 
28, 29, 30, 31, 32, 33, 34, 
36, 38, 39, 40, 41, 42, 44, 
46, 47, 48, 49, 50, 51, 52, 
54, 56, 57, 58, 59, 60, 62, 
64, 66, 68, 70, 
72, 74, 76, 78, 80]

class CostVolume(nn.Module):
    def __init__(self, dim, max_displacement, weight_kernel_size=1):
        super(CostVolume, self).__init__()
        self.dim = dim
        self.max_disp = max_displacement
        
        k = 2 * max_displacement + 1
        weight = torch.zeros(dim * len(index), 1, k, k)
        i = 0
        for y in range(k):
            for x in range(k):
                if y * k + x in index:
                    weight[i:i+dim, :, y, x] = 1.
                    i += dim
        self.conv1 = nn.Conv2d(dim * len(index), dim * len(index), k, padding=max_displacement, groups=dim * len(index), bias=False)
        self.conv1.weight = nn.Parameter(torch.Tensor(weight))
        self.conv1.requires_grad = False

        self.conv2 = nn.Conv2d(dim * len(index), len(index), weight_kernel_size, padding=(weight_kernel_size-1)//2, groups=len(index), bias=False)
        weight = torch.ones(len(index), dim, weight_kernel_size, weight_kernel_size)
        weight /= dim * (weight_kernel_size ** 2)
        self.conv2.weight = nn.Parameter(torch.Tensor(weight))
        self.conv2.requires_grad = False


    def forward(self, in1, in2):
        n, c, h, w = in2.size()

        in1 = in1.repeat(1, len(index), 1, 1)
        in2 = in2.repeat(1, len(index), 1, 1)

        in1 = self.conv1(in1)
        out = in1 * in2
        out = self.conv2(out)
        return out
        

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = convrelu(96, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out


class FastFlowNet(nn.Module):
    def __init__(self, groups=3):
        super(FastFlowNet, self).__init__()
        self.groups = groups
        self.pconv1_1 = convrelu(3, 16, 3, 2)
        self.pconv1_2 = convrelu(16, 16, 3, 1)
        self.pconv2_1 = convrelu(16, 32, 3, 2)
        self.pconv2_2 = convrelu(32, 32, 3, 1)
        self.pconv2_3 = convrelu(32, 32, 3, 1)
        self.pconv3_1 = convrelu(32, 64, 3, 2)
        self.pconv3_2 = convrelu(64, 64, 3, 1)
        self.pconv3_3 = convrelu(64, 64, 3, 1)


        self.rconv2 = convrelu(32, 32, 3, 1)
        self.rconv3 = convrelu(64, 32, 3, 1)
        self.rconv4 = convrelu(64, 32, 3, 1)
        self.rconv5 = convrelu(64, 32, 3, 1)
        self.rconv6 = convrelu(64, 32, 3, 1)

        self.up3 = deconv(2, 2)
        self.up4 = deconv(2, 2)
        self.up5 = deconv(2, 2)
        self.up6 = deconv(2, 2)

        self.decoder2 = Decoder(87, groups)
        self.decoder3 = Decoder(87, groups)
        self.decoder4 = Decoder(87, groups)
        self.decoder5 = Decoder(87, groups)
        self.decoder6 = Decoder(87, groups)
        
        self.corr64 = CostVolume(64, 4)
        self.corr32 = CostVolume(32, 4)

    def warp(self, x, flo):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1).to(x)
        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H-1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)        
        output = F.grid_sample(x, vgrid, mode='bilinear', align_corners=False)
        return output

    def extract_feature_pyramid(self, img):
        """
        입력 이미지에서 특징 피라미드 추출
        
        Args:
            img (torch.Tensor): 입력 이미지 [B, C, H, W]
            
        Returns:
            list: 다양한 해상도의 특징 맵 리스트
        """
        f1 = self.pconv1_2(self.pconv1_1(img))
        f2 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f1)))
        f3 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f2)))
        f4 = F.avg_pool2d(f3, kernel_size=(2, 2), stride=(2, 2))
        f5 = F.avg_pool2d(f4, kernel_size=(2, 2), stride=(2, 2))
        f6 = F.avg_pool2d(f5, kernel_size=(2, 2), stride=(2, 2))
        
        return [f1, f2, f3, f4, f5, f6]

    def compute_flow(self, feature_pyramid1, feature_pyramid2):
        """
        두 특징 피라미드 간의 광학 흐름 계산
        
        Args:
            feature_pyramid1 (list): 첫 번째 이미지의 특징 피라미드
            feature_pyramid2 (list): 두 번째 이미지의 특징 피라미드
            
        Returns:
            list: 다양한 해상도의 광학 흐름 리스트
        """
        f12, f22, f13, f23, f14, f24, f15, f25, f16, f26 = (
            feature_pyramid1[1], feature_pyramid2[1], 
            feature_pyramid1[2], feature_pyramid2[2],
            feature_pyramid1[3], feature_pyramid2[3],
            feature_pyramid1[4], feature_pyramid2[4],
            feature_pyramid1[5], feature_pyramid2[5]
        )
        
        flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
        cv6 = self.corr64(f16, f26)
        r16 = self.rconv6(f16)
        cat6 = torch.cat([cv6, r16, flow7_up], 1)
        flow6 = self.decoder6(cat6)

        flow6_up = self.up6(flow6)
        f25_w = self.warp(f25, flow6_up*0.625)
        cv5 = self.corr64(f15, f25_w)
        r15 = self.rconv5(f15)
        cat5 = torch.cat([cv5, r15, flow6_up], 1)
        flow5 = self.decoder5(cat5) + flow6_up

        flow5_up = self.up5(flow5)
        f24_w = self.warp(f24, flow5_up*1.25)
        cv4 = self.corr64(f14, f24_w)
        r14 = self.rconv4(f14)
        cat4 = torch.cat([cv4, r14, flow5_up], 1)
        flow4 = self.decoder4(cat4) + flow5_up

        flow4_up = self.up4(flow4)
        f23_w = self.warp(f23, flow4_up*2.5)
        cv3 = self.corr64(f13, f23_w)
        r13 = self.rconv3(f13)
        cat3 = torch.cat([cv3, r13, flow4_up], 1)
        flow3 = self.decoder3(cat3) + flow4_up

        flow3_up = self.up3(flow3)
        f22_w = self.warp(f22, flow3_up*5.0)
        cv2 = self.corr32(f12, f22_w)
        r12 = self.rconv2(f12)
        cat2 = torch.cat([cv2, r12, flow3_up], 1)
        flow2 = self.decoder2(cat2) + flow3_up
        
        flows = [flow2, flow3, flow4, flow5, flow6]
        
        # 입력 이미지 크기로 확대하고 스케일링
        flows_upscaled = []
        for flow in flows:
            # 모든 flow를 원본 크기로 확대 (flow2는 4배, flow3는 8배, 등)
            scale_factor = 4  # 모든 flow에 동일한 scale_factor 적용
            flow_upscaled = F.interpolate(flow, scale_factor=scale_factor, mode='bilinear', align_corners=False) * 20.0
            flows_upscaled.append(flow_upscaled)
        return flows_upscaled

    def forward(self, img1, img2):
        """
        두 이미지 간의 광학 흐름 계산
        
        Args:
            img1 (torch.Tensor): 첫 번째 이미지 [B, C, H, W]
            img2 (torch.Tensor): 두 번째 이미지 [B, C, H, W]
            
        Returns:
            tuple: (flows, feature_pyramid1, feature_pyramid2)
                - flows (list): 다양한 해상도의 광학 흐름 리스트
                - feature_pyramid1 (list): 첫 번째 이미지의 특징 피라미드
                - feature_pyramid2 (list): 두 번째 이미지의 특징 피라미드
        """
        # 특징 피라미드 추출
        feature_pyramid1 = self.extract_feature_pyramid(img1)
        feature_pyramid2 = self.extract_feature_pyramid(img2)
        
        # 광학 흐름 계산
        flows = self.compute_flow(feature_pyramid1, feature_pyramid2)
        
        return flows, feature_pyramid1, feature_pyramid2
    
    def forward_backward_flow(self, img1, img2):
        """
        두 이미지 간의 양방향 광학 흐름 계산
        
        Args:
            img1 (torch.Tensor): 첫 번째 이미지 [B, C, H, W]
            img2 (torch.Tensor): 두 번째 이미지 [B, C, H, W]
            
        Returns:
            tuple: (forward_flows, backward_flows, feature_pyramid1, feature_pyramid2)
                - forward_flows (list): 순방향 광학 흐름 리스트
                - backward_flows (list): 역방향 광학 흐름 리스트
                - feature_pyramid1 (list): 첫 번째 이미지의 특징 피라미드
                - feature_pyramid2 (list): 두 번째 이미지의 특징 피라미드
        """
        # 특징 피라미드 추출
        feature_pyramid1 = self.extract_feature_pyramid(img1)
        feature_pyramid2 = self.extract_feature_pyramid(img2)
        
        # 순방향 광학 흐름 계산 (img1 -> img2)
        forward_flows = self.compute_flow(feature_pyramid1, feature_pyramid2)
        
        # 역방향 광학 흐름 계산 (img2 -> img1)
        backward_flows = self.compute_flow(feature_pyramid2, feature_pyramid1)
        
        return forward_flows, backward_flows, feature_pyramid1, feature_pyramid2
    
    def infer_occlusion(self, flow_forward, flow_backward, method='wang'):
        """
        광학 흐름에서 가려짐 마스크 추정
        
        Args:
            flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
            flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
            method (str): 가려짐 추정 방식 (default: 'wang')
            
        Returns:
            torch.Tensor: 가려짐 마스크 [B, 1, H, W], 1=가려짐 없음, 0=가려짐
        """
        # 입력 흐름의 크기 확인
        flow_magnitude = torch.sqrt(flow_forward[:, 0]**2 + flow_forward[:, 1]**2).mean()
        
        # 흐름이 너무 크면 스케일 조정 (FastFlowNet에서는 흐름에 20.0을 곱하기 때문에)
        scale_factor = 1.0
        if flow_magnitude > 10.0:
            scale_factor = 5.0 / flow_magnitude
            # 스케일 조정된 흐름 복사본 생성
            flow_forward_scaled = flow_forward * scale_factor
            flow_backward_scaled = flow_backward * scale_factor
        else:
            flow_forward_scaled = flow_forward
            flow_backward_scaled = flow_backward
        
        # Occlusion 추정 시 더 적합한 값 사용
        occ_weights = {
            'fb_abs': 10.0,           # 더 낮은 값으로 조정
            'forward_collision': 10.0, # 더 낮은 값으로 조정
            'backward_zero': 10.0      # 더 낮은 값으로 조정
        }
        
        occ_thresholds = {
            'fb_abs': 4.0,             # 더 높은 임계값으로 조정
            'forward_collision': 0.3,  # 약간 낮게 조정
            'backward_zero': 0.2       # 약간 낮게 조정
        }
        
        occ_clip_max = {
            'fb_abs': 10.0,            # 유지
            'forward_collision': 5.0   # 유지
        }
        
        # 실험: 'wang' 방식이 잘 작동하지 않으면 'forward_backward' 방식 시도
        if method == 'wang':
            try:
                occlusion_mask = utils.estimate_occlusion_mask(
                    flow_forward_scaled, 
                    flow_backward_scaled,
                    method=method,
                    occ_weights=occ_weights,
                    occ_thresholds=occ_thresholds,
                    occ_clip_max=occ_clip_max
                )
                
                # 마스크 확인: 모두 0이거나 모두 1인 경우 문제 있음
                if torch.mean(occlusion_mask) < 0.01 or torch.mean(occlusion_mask) > 0.99:
                    # 다른 방법 시도
                    method = 'forward_backward'
                    occlusion_mask = utils.estimate_occlusion_mask(
                        flow_forward_scaled, 
                        flow_backward_scaled,
                        method=method,
                        occ_weights=occ_weights,
                        occ_thresholds=occ_thresholds,
                        occ_clip_max=occ_clip_max
                    )
            except Exception as e:
                # 오류 발생 시 'forward_backward' 방식으로 폴백
                print(f"Wang 방식 occlusion 계산 오류, forward_backward 방식으로 대체: {str(e)}")
                method = 'forward_backward'
                occlusion_mask = utils.estimate_occlusion_mask(
                    flow_forward_scaled, 
                    flow_backward_scaled,
                    method=method,
                    occ_weights=occ_weights,
                    occ_thresholds=occ_thresholds,
                    occ_clip_max=occ_clip_max
                )
        else:
            occlusion_mask = utils.estimate_occlusion_mask(
                flow_forward_scaled, 
                flow_backward_scaled,
                method=method,
                occ_weights=occ_weights,
                occ_thresholds=occ_thresholds,
                occ_clip_max=occ_clip_max
            )
        
        return occlusion_mask
    
    def warp_image(self, img, flow):
        """
        이미지를 광학 흐름으로 와핑
        
        Args:
            img (torch.Tensor): 와핑할 이미지 [B, C, H, W]
            flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
            
        Returns:
            torch.Tensor: 와핑된 이미지 [B, C, H, W]
        """
        return utils.warp_image(img, flow)


# 모델 테스트 코드
if __name__ == "__main__":
    # 테스트용 이미지 쌍 생성 (배치 크기 2, 3채널, 192x256)
    img1 = torch.randn(2, 3, 192, 256)
    img2 = torch.randn(2, 3, 192, 256)
    
    # 모델 초기화
    model = FastFlowNet(groups=3)
    
    # GPU 사용 가능한 경우 모델 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # 모델 순전파
    with torch.no_grad():
        flows, fp1, fp2 = model(img1, img2)
    
    # 결과 확인
    print(f"Flow 피라미드 레벨 수: {len(flows)}")
    for i, flow in enumerate(flows):
        print(f"레벨 {i} 흐름 크기: {flow.shape}")
    
    # 특징 피라미드 크기 출력
    print(f"\n특징 피라미드 레벨 수: {len(fp1)}")
    for i, (feat1, feat2) in enumerate(zip(fp1, fp2)):
        print(f"레벨 {i} 특징 맵 크기: {feat1.shape}, {feat2.shape}")
    
    # 양방향 흐름 테스트
    with torch.no_grad():
        forward_flows, backward_flows, _, _ = model.forward_backward_flow(img1, img2)
        occlusion_mask = model.infer_occlusion(forward_flows[0], backward_flows[0])
        warped_img2 = model.warp_image(img2, forward_flows[0])
    
    print(f"\n양방향 흐름 테스트:")
    print(f"순방향 흐름 크기: {forward_flows[0].shape}")
    print(f"역방향 흐름 크기: {backward_flows[0].shape}")
    print(f"가려짐 마스크 크기: {occlusion_mask.shape}")
    print(f"와핑된 이미지 크기: {warped_img2.shape}")
    
    # 모델 파라미터 수 계산
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n모델 파라미터 수: {param_count:,}")
    
    print("\n테스트 성공!")