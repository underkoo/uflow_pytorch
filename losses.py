import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class OcclusionMask(nn.Module):
    """
    광학 흐름에서 가려짐(occlusion) 영역을 탐지하는 모듈
    """
    def __init__(self, method='forward_backward', alpha=0.01, beta=0.5):
        """
        Args:
            method (str): 가려짐 탐지 방법
                - 'forward_backward': 순방향 및 역방향 흐름의 일관성 확인
                - 'divergence': 발산 기반 방법
            alpha (float): forward_backward 방식의 매개변수
            beta (float): forward_backward 방식의 매개변수
        """
        super(OcclusionMask, self).__init__()
        self.method = method
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, flow_forward, flow_backward):
        """
        가려짐 마스크 계산
        
        Args:
            flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
            flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
            
        Returns:
            torch.Tensor: 이진 가려짐 마스크 [B, 1, H, W], 1=가려짐 없음, 0=가려짐
        """
        return utils.estimate_occlusion_mask(
            flow_forward, 
            flow_backward,
            method=self.method,
            alpha=self.alpha,
            beta=self.beta
        )
    
    def _warp_flow(self, flow, ref_flow):
        """
        광학 흐름을 사용하여 다른 흐름 와핑
        
        Args:
            flow (torch.Tensor): 와핑할 흐름 [B, 2, H, W]
            ref_flow (torch.Tensor): 참조 흐름 [B, 2, H, W]
            
        Returns:
            torch.Tensor: 와핑된 흐름 [B, 2, H, W]
        """
        B, _, H, W = flow.shape
        
        # 그리드 생성
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H-1, H, device=flow.device),
            torch.linspace(0, W-1, W, device=flow.device)
        )
        grid = torch.stack([grid_x, grid_y]).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # 순방향 흐름으로 와핑할 위치 계산
        warped_grid = grid + ref_flow
        
        # 정규화된 좌표로 변환 [-1, 1] 범위
        grid_normalized = torch.zeros_like(warped_grid)
        grid_normalized[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        grid_normalized[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        
        # grid_sample에 맞게 형태 변경
        grid_normalized = grid_normalized.permute(0, 2, 3, 1)
        
        # 흐름 와핑
        warped_flow = F.grid_sample(
            flow, grid_normalized, 
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        
        return warped_flow
    
    def _spatial_gradient(self, tensor, direction):
        """
        공간적 기울기 계산
        
        Args:
            tensor (torch.Tensor): 기울기를 계산할 텐서 [B, C, H, W]
            direction (str): 기울기 방향 ('x' 또는 'y')
            
        Returns:
            torch.Tensor: 기울기 [B, C, H, W]
        """
        B, C, H, W = tensor.shape
        
        if direction == 'x':
            kernel = torch.tensor([[-1, 0, 1]], dtype=tensor.dtype, device=tensor.device)
            kernel = kernel.view(1, 1, 1, 3).repeat(C, 1, 1, 1)
            padding = (1, 1, 0, 0)
        elif direction == 'y':
            kernel = torch.tensor([[-1], [0], [1]], dtype=tensor.dtype, device=tensor.device)
            kernel = kernel.view(1, 1, 3, 1).repeat(C, 1, 1, 1)
            padding = (0, 0, 1, 1)
        else:
            raise ValueError(f"Unknown gradient direction: {direction}")
        
        # 컨볼루션 적용
        gradient = F.conv2d(
            F.pad(tensor, padding, mode='replicate'), 
            kernel, 
            groups=C
        )
        
        return gradient


class PhotometricLoss(nn.Module):
    """
    Photometric Loss (강건한 L1/Charbonnier + SSIM)
    
    와핑된 이미지와 타겟 이미지 간의 차이를 측정
    """
    def __init__(self, epsilon=0.01, use_occlusion_mask=True, alpha=0.85, ssim_window_size=11):
        """
        Args:
            epsilon (float): Charbonnier 손실 함수의 안정성 파라미터
            use_occlusion_mask (bool): 가려짐 마스크 사용 여부
            alpha (float): L1과 SSIM 손실 간의 가중치 (alpha * L1 + (1-alpha) * SSIM)
            ssim_window_size (int): SSIM 계산에 사용할 윈도우 크기 (홀수여야 함)
        """
        super(PhotometricLoss, self).__init__()
        assert ssim_window_size % 2 == 1, "SSIM 윈도우 크기는 홀수여야 합니다"
        
        self.epsilon = epsilon
        self.use_occlusion_mask = use_occlusion_mask
        self.alpha = alpha
        self.ssim_window_size = ssim_window_size
    
    def forward(self, img1, img2_warped, occlusion_mask=None, valid_mask=None):
        """
        Photometric 손실 계산
        
        Args:
            img1 (torch.Tensor): 타겟 이미지 [B, C, H, W]
            img2_warped (torch.Tensor): 와핑된 이미지 [B, C, H, W]
            occlusion_mask (torch.Tensor, optional): 가려짐 마스크 [B, 1, H, W], 1=가려짐 없음, 0=가려짐
            valid_mask (torch.Tensor, optional): 유효 영역 마스크 [B, 1, H, W], 1=유효, 0=무효
            
        Returns:
            torch.Tensor: Photometric 손실 [B, 1, H, W]
        """
        # 이미지 차이 계산 (L1/Charbonnier 손실)
        diff = img1 - img2_warped
        l1_loss = torch.sqrt(diff**2 + self.epsilon**2)
        
        # SSIM 손실 계산
        ssim_loss = 1 - self._compute_ssim(img1, img2_warped)
        
        # L1과 SSIM 손실 조합
        loss = self.alpha * l1_loss + (1 - self.alpha) * ssim_loss
        
        # 마스크 적용
        if self.use_occlusion_mask and occlusion_mask is not None:
            loss = loss * occlusion_mask
        
        if valid_mask is not None:
            loss = loss * valid_mask
            
        return loss
    
    def _compute_ssim(self, x, y):
        """
        Structural Similarity (SSIM) 계산
        
        Args:
            x (torch.Tensor): 첫 번째 이미지 [B, C, H, W]
            y (torch.Tensor): 두 번째 이미지 [B, C, H, W]
            
        Returns:
            torch.Tensor: SSIM 유사도 맵 [B, 1, H, W] (1=완전 유사, 0=완전 다름)
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 채널별 계산 후 평균
        if x.shape[1] > 1:
            ssim_map = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
            for c in range(x.shape[1]):
                ssim_map += self._compute_ssim_single(x[:, c:c+1], y[:, c:c+1])
            ssim_map /= x.shape[1]
            return ssim_map
        else:
            return self._compute_ssim_single(x, y)
    
    def _compute_ssim_single(self, x, y):
        """
        단일 채널에 대한 SSIM 계산
        
        Args:
            x (torch.Tensor): 첫 번째 이미지 [B, 1, H, W]
            y (torch.Tensor): 두 번째 이미지 [B, 1, H, W]
            
        Returns:
            torch.Tensor: SSIM 유사도 맵 [B, 1, H, W]
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 가우시안 커널 생성
        kernel_size = self.ssim_window_size
        kernel_sigma = 1.5
        kernel = self._create_gaussian_kernel(kernel_size, kernel_sigma, x.device)
        
        # 패딩
        pad = (kernel_size - 1) // 2
        
        # 지역적 평균 및 분산 계산을 위한 컨볼루션
        x_padded = F.pad(x, [pad, pad, pad, pad], mode='reflect')
        y_padded = F.pad(y, [pad, pad, pad, pad], mode='reflect')
        
        # 로컬 평균
        mu_x = F.conv2d(x_padded, kernel, groups=1)
        mu_y = F.conv2d(y_padded, kernel, groups=1)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # 로컬 분산 계산을 위한 컨볼루션
        x_sq_padded = F.pad(x ** 2, [pad, pad, pad, pad], mode='reflect')
        y_sq_padded = F.pad(y ** 2, [pad, pad, pad, pad], mode='reflect')
        xy_padded = F.pad(x * y, [pad, pad, pad, pad], mode='reflect')
        
        sigma_x_sq = F.conv2d(x_sq_padded, kernel, groups=1) - mu_x_sq
        sigma_y_sq = F.conv2d(y_sq_padded, kernel, groups=1) - mu_y_sq
        sigma_xy = F.conv2d(xy_padded, kernel, groups=1) - mu_xy
        
        # 수치 안정성을 위한 상수
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # SSIM 공식
        ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim = ssim_n / ssim_d
        
        return ssim
    
    def _create_gaussian_kernel(self, kernel_size, sigma, device):
        """
        가우시안 커널 생성
        
        Args:
            kernel_size (int): 커널 크기
            sigma (float): 가우시안 시그마
            device (torch.device): 텐서 디바이스
            
        Returns:
            torch.Tensor: 가우시안 커널 [1, 1, kernel_size, kernel_size]
        """
        coords = torch.arange(kernel_size, device=device).float()  # float 타입으로 변환
        coords -= (kernel_size - 1) / 2
        
        g = coords ** 2
        g = torch.exp(-(g[:, None] + g[None, :]) / (2 * sigma ** 2))
        g /= g.sum()
        
        return g.view(1, 1, kernel_size, kernel_size)


class CensusLoss(nn.Module):
    """
    Census 변환 기반 손실 함수
    
    지역적 패턴 변화에 강인한 손실 함수
    """
    def __init__(self, patch_size=3, use_occlusion_mask=True):
        """
        Args:
            patch_size (int): Census 변환 패치 크기 (홀수여야 함)
            use_occlusion_mask (bool): 가려짐 마스크 사용 여부
        """
        super(CensusLoss, self).__init__()
        assert patch_size % 2 == 1, "패치 크기는 홀수여야 합니다"
        self.patch_size = patch_size
        self.use_occlusion_mask = use_occlusion_mask
        
        # 패치 중앙 인덱스
        self.half_patch = patch_size // 2
    
    def forward(self, img1, img2_warped, occlusion_mask=None, valid_mask=None):
        """
        Census 손실 계산
        
        Args:
            img1 (torch.Tensor): 타겟 이미지 [B, C, H, W]
            img2_warped (torch.Tensor): 와핑된 이미지 [B, C, H, W]
            occlusion_mask (torch.Tensor, optional): 가려짐 마스크 [B, 1, H, W], 1=가려짐 없음, 0=가려짐
            valid_mask (torch.Tensor, optional): 유효 영역 마스크 [B, 1, H, W], 1=유효, 0=무효
            
        Returns:
            torch.Tensor: Census 손실 [B, 1, H, W]
        """
        B, C, H, W = img1.shape
        
        # 그레이스케일 변환 (이미지가 다중 채널인 경우)
        if C > 1:
            img1_gray = 0.299 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
            img2_warped_gray = 0.299 * img2_warped[:, 0:1] + 0.587 * img2_warped[:, 1:2] + 0.114 * img2_warped[:, 2:3]
        else:
            img1_gray = img1
            img2_warped_gray = img2_warped
        
        # 유효 영역 패딩 계산
        pad = self.half_patch
        
        # 손실 저장을 위한 텐서
        loss = torch.zeros((B, 1, H, W), device=img1.device)
        
        # 패치 기반 Census 변환 및 해밍 거리 계산
        census1 = self._compute_census_transform(img1_gray)
        census2 = self._compute_census_transform(img2_warped_gray)
        
        # 해밍 거리 계산 (XOR 비트 개수 세기)
        hamming_dist = torch.zeros_like(census1[:, 0:1])
        
        for i in range(census1.size(1)):
            hamming_dist += (census1[:, i:i+1] != census2[:, i:i+1]).float()
        
        # 정규화 (최대 거리는 (patch_size^2 - 1)로 나누기)
        norm_factor = float(self.patch_size * self.patch_size - 1)
        loss = hamming_dist / norm_factor
        
        # 마스크 적용
        if self.use_occlusion_mask and occlusion_mask is not None:
            loss = loss * occlusion_mask
        
        if valid_mask is not None:
            loss = loss * valid_mask
            
        return loss
    
    def _compute_census_transform(self, img):
        """
        Census 변환 계산
        
        Args:
            img (torch.Tensor): 입력 이미지 [B, 1, H, W]
            
        Returns:
            torch.Tensor: Census 변환 [B, (patch_size^2-1), H, W]
        """
        B, C, H, W = img.shape
        
        # 패치 추출을 위한 패딩
        pad = self.half_patch
        img_padded = F.pad(img, (pad, pad, pad, pad), mode='replicate')
        
        # 중앙 픽셀
        center = img
        
        # Census 비트 저장 텐서
        census = torch.zeros((B, self.patch_size * self.patch_size - 1, H, W), device=img.device)
        
        # 패치 내의 각 위치에서 중앙과 비교
        idx = 0
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                if i == self.half_patch and j == self.half_patch:
                    # 중앙 픽셀은 건너뛰기
                    continue
                
                # 인접 픽셀 추출
                neighbor = img_padded[:, :, i:i+H, j:j+W]
                
                # Census 비트 계산 (1: 이웃 > 중앙, 0: 이웃 <= 중앙)
                census[:, idx:idx+1] = (neighbor > center).float()
                idx += 1
        
        return census


class SmoothnessLoss(nn.Module):
    """
    광학 흐름 평활화 손실
    
    흐름 필드의 부드러움을 촉진
    """
    def __init__(self, edge_aware=True, second_order=False):
        """
        Args:
            edge_aware (bool): 이미지 에지를 고려한 평활화 사용
            second_order (bool): 2차 도함수(라플라시안) 사용 여부
        """
        super(SmoothnessLoss, self).__init__()
        self.edge_aware = edge_aware
        self.second_order = second_order
    
    def forward(self, flow, image=None, valid_mask=None):
        """
        평활화 손실 계산
        
        Args:
            flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
            image (torch.Tensor, optional): 입력 이미지 [B, C, H, W], edge_aware=True일 때 사용
            valid_mask (torch.Tensor, optional): 유효 영역 마스크 [B, 1, H, W]
            
        Returns:
            torch.Tensor: 평활화 손실 [B, 1, H, W]
        """
        # 흐름의 기울기 계산
        flow_dx = self._gradient(flow, 'x')
        flow_dy = self._gradient(flow, 'y')
        
        # 2차 평활화 (라플라시안)
        if self.second_order:
            flow_ddx = self._gradient(flow_dx, 'x')
            flow_ddy = self._gradient(flow_dy, 'y')
            flow_dxy = self._gradient(flow_dx, 'y')
            
            # 기울기의 L1 노름 계산
            smoothness = torch.abs(flow_ddx) + torch.abs(flow_ddy) + torch.abs(flow_dxy)
        else:
            # 기울기의 L1 노름 계산
            smoothness = torch.abs(flow_dx) + torch.abs(flow_dy)
        
        # 채널 평균 계산
        smoothness = torch.mean(smoothness, dim=1, keepdim=True)
        
        # 에지 인식 가중치 적용
        if self.edge_aware and image is not None:
            # 이미지 기울기 계산
            image_dx = self._gradient(image, 'x')
            image_dy = self._gradient(image, 'y')
            
            # 기울기의 L1 노름
            image_dx_mag = torch.mean(torch.abs(image_dx), dim=1, keepdim=True)
            image_dy_mag = torch.mean(torch.abs(image_dy), dim=1, keepdim=True)
            
            # 이미지 에지에서 가중치 감소
            weights_x = torch.exp(-image_dx_mag)
            weights_y = torch.exp(-image_dy_mag)
            
            # 에지 인식 평활화
            if self.second_order:
                smoothness = (
                    torch.abs(flow_ddx) * weights_x + 
                    torch.abs(flow_ddy) * weights_y + 
                    torch.abs(flow_dxy) * weights_x * weights_y
                )
                smoothness = torch.mean(smoothness, dim=1, keepdim=True)
            else:
                smoothness_x = torch.abs(flow_dx) * weights_x
                smoothness_y = torch.abs(flow_dy) * weights_y
                smoothness = torch.mean(smoothness_x + smoothness_y, dim=1, keepdim=True)
        
        # 마스크 적용
        if valid_mask is not None:
            smoothness = smoothness * valid_mask
            
        return smoothness
    
    def _gradient(self, tensor, direction):
        """
        텐서의 기울기 계산
        
        Args:
            tensor (torch.Tensor): 입력 텐서 [B, C, H, W]
            direction (str): 기울기 방향 ('x' 또는 'y')
            
        Returns:
            torch.Tensor: 기울기 [B, C, H, W]
        """
        B, C, H, W = tensor.shape
        
        if direction == 'x':
            # x 방향 기울기 (패딩 사용)
            tensor_pad = F.pad(tensor, (0, 1, 0, 0), mode='replicate')
            gradient = tensor_pad[:, :, :, 1:] - tensor_pad[:, :, :, :-1]
        elif direction == 'y':
            # y 방향 기울기 (패딩 사용)
            tensor_pad = F.pad(tensor, (0, 0, 0, 1), mode='replicate')
            gradient = tensor_pad[:, :, 1:, :] - tensor_pad[:, :, :-1, :]
        else:
            raise ValueError(f"Unknown gradient direction: {direction}")
            
        return gradient


class UFlowLoss(nn.Module):
    """
    UFlow 모델의 전체 손실 함수
    
    여러 손실 함수를 결합
    """
    def __init__(self,
                 photometric_weight=1.0,
                 census_weight=1.0,
                 smoothness_weight=0.1,
                 ssim_weight=0.85,
                 window_size=11,
                 occlusion_method='forward_backward',
                 use_occlusion=True,
                 use_valid_mask=True,
                 second_order_smoothness=False,
                 edge_aware_smoothness=True,
                 stop_gradient=True,
                 bidirectional=False):
        """
        Args:
            photometric_weight (float): Photometric 손실 가중치
            census_weight (float): Census 손실 가중치
            smoothness_weight (float): 평활화 손실 가중치
            ssim_weight (float): SSIM 손실 가중치
            window_size (int): SSIM 계산에 사용할 윈도우 크기
            occlusion_method (str): 가려짐 탐지 방법
            use_occlusion (bool): 가려짐 마스크 사용 여부
            use_valid_mask (bool): 유효 영역 마스크 사용 여부
            second_order_smoothness (bool): 2차 평활화 사용 여부
            edge_aware_smoothness (bool): 에지 인식 평활화 사용 여부
            stop_gradient (bool): 역전파 중지 플래그
            bidirectional (bool): 양방향 손실 계산 여부
        """
        super(UFlowLoss, self).__init__()
        
        self.photometric_weight = photometric_weight
        self.census_weight = census_weight
        self.smoothness_weight = smoothness_weight
        self.ssim_weight = ssim_weight
        self.window_size = window_size
        self.use_occlusion = use_occlusion
        self.use_valid_mask = use_valid_mask
        self.stop_gradient = stop_gradient
        self.bidirectional = bidirectional
        
        self.occlusion_mask = OcclusionMask(method=occlusion_method)
        self.photometric_loss = PhotometricLoss(use_occlusion_mask=use_occlusion, alpha=ssim_weight, ssim_window_size=window_size)
        self.census_loss = CensusLoss(use_occlusion_mask=use_occlusion)
        self.smoothness_loss = SmoothnessLoss(edge_aware=edge_aware_smoothness, second_order=second_order_smoothness)
    
    def forward(self, 
                img1, 
                img2, 
                flow_forward, 
                flow_backward=None, 
                valid_mask=None):
        """
        전체 UFlow 손실 계산
        
        Args:
            img1 (torch.Tensor): 첫 번째 이미지 [B, C, H, W]
            img2 (torch.Tensor): 두 번째 이미지 [B, C, H, W]
            flow_forward (torch.Tensor): 순방향 광학 흐름 (img1 -> img2) [B, 2, H, W]
            flow_backward (torch.Tensor, optional): 역방향 광학 흐름 (img2 -> img1) [B, 2, H, W]
            valid_mask (torch.Tensor, optional): 유효 영역 마스크 [B, 1, H, W]
            
        Returns:
            dict: 각 손실 값들과 총 손실
                - 'total_loss': 전체 손실
                - 'photometric_loss': Photometric 손실
                - 'census_loss': Census 손실
                - 'smoothness_loss': 평활화 손실
                - 'occlusion_mask': 가려짐 마스크 (있는 경우)
        """
        # 이미지 정규화 (0-1 범위로 가정)
        img1_norm = img1
        img2_norm = img2
        
        # 유효 마스크
        if valid_mask is None and self.use_valid_mask:
            # 기본 유효 마스크는 모든 픽셀이 유효
            valid_mask = torch.ones((img1.shape[0], 1, img1.shape[2], img1.shape[3]), device=img1.device)
            
            # stop-gradient 적용
            if self.stop_gradient:
                valid_mask = valid_mask.detach()
        
        # 순방향 손실 계산
        forward_loss_dict = self._compute_unidirectional_loss(
            img1_norm, img2_norm, flow_forward, flow_backward, valid_mask, "forward"
        )
        
        # 양방향 손실 계산 (필요한 경우)
        if self.bidirectional and flow_backward is not None:
            backward_loss_dict = self._compute_unidirectional_loss(
                img2_norm, img1_norm, flow_backward, flow_forward, valid_mask, "backward"
            )
            
            # 순방향 및 역방향 손실 결합
            loss_dict = {}
            for key in forward_loss_dict:
                if key.endswith('loss') or key == 'total_loss':
                    loss_dict[key] = (forward_loss_dict[key] + backward_loss_dict[key]) * 0.5
                else:
                    # 마스크 등의 정보는 그대로 유지
                    loss_dict[key] = forward_loss_dict[key]
                    loss_dict[f'backward_{key}'] = backward_loss_dict[key]
            
            return loss_dict
        else:
            # 단방향(순방향) 손실만 반환
            return forward_loss_dict
    
    def _compute_unidirectional_loss(self, img_src, img_tgt, flow, flow_opposite=None, valid_mask=None, direction="forward"):
        """
        단방향 손실 계산
        
        Args:
            img_src (torch.Tensor): 소스 이미지 [B, C, H, W]
            img_tgt (torch.Tensor): 타겟 이미지 [B, C, H, W]
            flow (torch.Tensor): 소스에서 타겟으로의 광학 흐름 [B, 2, H, W]
            flow_opposite (torch.Tensor): 타겟에서 소스로의 광학 흐름 [B, 2, H, W]
            valid_mask (torch.Tensor): 유효 영역 마스크 [B, 1, H, W]
            direction (str): 흐름 방향 ("forward" 또는 "backward")
            
        Returns:
            dict: 손실 딕셔너리
        """
        # 가려짐 마스크 계산
        occlusion_mask = None
        if self.use_occlusion and flow_opposite is not None:
            occlusion_mask = self.occlusion_mask(flow, flow_opposite)
            
            # stop-gradient 적용 - occlusion 마스크의 그래디언트가 네트워크에 영향을 미치지 않도록 함
            if self.stop_gradient:
                occlusion_mask = occlusion_mask.detach()
        
        # 와핑된 이미지 계산
        img_tgt_warped = self._warp_image(img_tgt, flow)
        
        # stop-gradient 적용 - 와핑된 이미지 그래디언트가 흐름 네트워크로 직접 전파되지 않도록 함
        if self.stop_gradient:
            img_tgt_warped = img_tgt_warped.detach()
        
        # 각 손실 함수 계산
        photometric = self.photometric_loss(img_src, img_tgt_warped, occlusion_mask, valid_mask)
        census = self.census_loss(img_src, img_tgt_warped, occlusion_mask, valid_mask)
        smoothness = self.smoothness_loss(flow, img_src, valid_mask)
        
        # 손실 가중치 계산 (영역별 다른 가중치 적용 가능)
        weights = torch.ones_like(valid_mask)
        if self.stop_gradient:
            weights = weights.detach()
        
        # 손실 평균 계산
        if valid_mask is not None and torch.sum(valid_mask) > 0:
            weighted_mask = weights * valid_mask
            if self.use_occlusion and occlusion_mask is not None:
                weighted_mask = weighted_mask * occlusion_mask
            
            norm_factor = torch.sum(weighted_mask) + 1e-8
            flow_consistency_loss = torch.sum(photometric * weighted_mask) / norm_factor
        else:
            flow_consistency_loss = torch.mean(photometric)
        
        # 총 손실 (가중치 적용)
        total_loss = (
            self.photometric_weight * flow_consistency_loss +
            self.census_weight * torch.mean(census) +
            self.smoothness_weight * torch.mean(smoothness)
        )
        
        # 결과 딕셔너리 반환
        loss_dict = {
            'total_loss': total_loss,
            'photometric_loss': flow_consistency_loss,
            'census_loss': torch.mean(census),
            'smoothness_loss': torch.mean(smoothness)
        }
        
        if occlusion_mask is not None:
            loss_dict['occlusion_mask'] = occlusion_mask
            
        return loss_dict
    
    def _warp_image(self, img, flow):
        """
        광학 흐름을 사용하여 이미지 와핑
        
        Args:
            img (torch.Tensor): 와핑할 이미지 [B, C, H, W]
            flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
            
        Returns:
            torch.Tensor: 와핑된 이미지 [B, C, H, W]
        """
        # utils.py의 warp_image 함수 사용
        return utils.warp_image(img, flow)


class SequenceLoss(nn.Module):
    """
    시퀀스 기반 손실 (Sequence Loss)
    
    연속 프레임에서의 시간적 일관성을 강제하는 손실 함수
    t1→t2 흐름과 t2→t3 흐름을 조합하여 얻은 t1→t3 합성 흐름이
    직접 계산된 t1→t3 흐름과 일치해야 한다는 제약 조건 적용
    """
    def __init__(self, 
                 alpha=0.01, 
                 use_occlusion_mask=True, 
                 use_valid_mask=True,
                 stop_gradient=True,
                 distance='robust_l1',
                 epsilon=0.01):
        """
        Args:
            alpha (float): 시퀀스 손실 가중치
            use_occlusion_mask (bool): 가려짐 마스크 사용 여부
            use_valid_mask (bool): 유효 영역 마스크 사용 여부
            stop_gradient (bool): 그래디언트 흐름 제어 사용 여부
            distance (str): 거리 측정 방식 ('l1', 'l2', 'robust_l1')
            epsilon (float): robust L1 손실의 안정성 파라미터
        """
        super(SequenceLoss, self).__init__()
        
        self.alpha = alpha
        self.use_occlusion_mask = use_occlusion_mask
        self.use_valid_mask = use_valid_mask
        self.stop_gradient = stop_gradient
        self.distance = distance
        self.epsilon = epsilon
        
        self.occlusion_mask = OcclusionMask(method='forward_backward')
    
    def forward(self, 
                flow_t1_t2, 
                flow_t2_t3, 
                flow_t1_t3, 
                flow_t3_t1=None, 
                valid_mask=None):
        """
        시퀀스 손실 계산
        
        Args:
            flow_t1_t2 (torch.Tensor): t1→t2 방향 흐름 [B, 2, H, W]
            flow_t2_t3 (torch.Tensor): t2→t3 방향 흐름 [B, 2, H, W]
            flow_t1_t3 (torch.Tensor): t1→t3 방향 흐름 (직접 계산) [B, 2, H, W]
            flow_t3_t1 (torch.Tensor, optional): t3→t1 방향 흐름 (가려짐 마스크 계산용) [B, 2, H, W]
            valid_mask (torch.Tensor, optional): 유효 영역 마스크
            
        Returns:
            dict: 시퀀스 손실 값들
                - 'sequence_loss': 전체 시퀀스 손실
                - 'flow_consistency_loss': 흐름 일관성 손실
                - 'occlusion_mask': 가려짐 마스크 (있는 경우)
        """
        # 유효 마스크 확인
        if valid_mask is None and self.use_valid_mask:
            # 기본 유효 마스크는 모든 픽셀이 유효
            valid_mask = torch.ones((flow_t1_t2.shape[0], 1, flow_t1_t2.shape[2], flow_t1_t2.shape[3]), device=flow_t1_t2.device)
            
            # stop-gradient 적용
            if self.stop_gradient:
                valid_mask = valid_mask.detach()
        
        # 가려짐 마스크 계산
        occlusion_mask = None
        if self.use_occlusion_mask and flow_t3_t1 is not None:
            occlusion_mask = self.occlusion_mask(flow_t1_t3, flow_t3_t1)
            
            # stop-gradient 적용 - occlusion 마스크의 그래디언트가 네트워크에 영향을 미치지 않도록 함
            if self.stop_gradient:
                occlusion_mask = occlusion_mask.detach()
        
        # flow_t1_t2를 사용하여 flow_t2_t3 와핑
        flow_t2_t3_warped = self._warp_flow(flow_t2_t3, flow_t1_t2)
        
        # stop-gradient 적용
        if self.stop_gradient:
            flow_t2_t3_warped = flow_t2_t3_warped.detach()
        
        # 합성 흐름 계산: flow_t1_t2 + warped_flow_t2_t3
        flow_t1_t3_composed = flow_t1_t2 + flow_t2_t3_warped
        
        # 직접 계산된 흐름과 합성 흐름 간의 거리 계산
        if self.distance == 'l1':
            # L1 거리
            flow_diff = torch.abs(flow_t1_t3 - flow_t1_t3_composed)
            flow_dist = torch.sum(flow_diff, dim=1, keepdim=True)  # 채널 축으로 합산
        elif self.distance == 'l2':
            # L2 거리
            flow_diff = flow_t1_t3 - flow_t1_t3_composed
            flow_dist = torch.sqrt(torch.sum(flow_diff ** 2, dim=1, keepdim=True))
        else:
            # robust L1 (Charbonnier)
            flow_diff = flow_t1_t3 - flow_t1_t3_composed
            flow_dist = torch.sqrt(torch.sum(flow_diff ** 2 + self.epsilon ** 2, dim=1, keepdim=True))
        
        # 마스크 적용
        if self.use_occlusion_mask and occlusion_mask is not None:
            flow_dist = flow_dist * occlusion_mask
        
        if valid_mask is not None:
            flow_dist = flow_dist * valid_mask
        
        # 손실 평균 계산
        if valid_mask is not None and torch.sum(valid_mask) > 0:
            weights = torch.ones_like(valid_mask)
            if self.stop_gradient:
                weights = weights.detach()
            
            weighted_mask = weights * valid_mask
            if self.use_occlusion_mask and occlusion_mask is not None:
                weighted_mask = weighted_mask * occlusion_mask
            
            norm_factor = torch.sum(weighted_mask) + 1e-8
            flow_consistency_loss = torch.sum(flow_dist * weighted_mask) / norm_factor
        else:
            flow_consistency_loss = torch.mean(flow_dist)
        
        # 총 손실 (가중치 적용)
        total_loss = self.alpha * flow_consistency_loss
        
        # 결과 딕셔너리 반환
        loss_dict = {
            'sequence_loss': total_loss,
            'flow_consistency_loss': flow_consistency_loss
        }
        
        if occlusion_mask is not None:
            loss_dict['occlusion_mask'] = occlusion_mask
        
        return loss_dict
    
    def _warp_flow(self, flow, ref_flow):
        """
        참조 흐름을 따라 흐름 필드를 와핑
        
        Args:
            flow (torch.Tensor): 와핑할 흐름 [B, 2, H, W]
            ref_flow (torch.Tensor): 참조 흐름 [B, 2, H, W]
            
        Returns:
            torch.Tensor: 와핑된 흐름 [B, 2, H, W]
        """
        # utils.py의 warp_flow 함수 사용
        return utils.warp_flow(flow, ref_flow)


class MultiScaleUFlowLoss(nn.Module):
    """
    다중 스케일 UFlow 손실
    
    여러 해상도에서 손실을 계산하고 결합
    """
    def __init__(
        self,
        photometric_weight=1.0,
        census_weight=1.0,
        smoothness_weight=1.0,
        ssim_weight=0.85,
        window_size=7,
        occlusion_method='forward_backward',
        edge_weighting=True,
        stop_gradient=False,
        bidirectional=False,
        scale_weights=None
    ):
        """
        Args:
            photometric_weight (float): 포토메트릭 손실 가중치
            census_weight (float): Census 손실 가중치
            smoothness_weight (float): 평활화 손실 가중치
            ssim_weight (float): SSIM 손실 가중치 (0 ~ 1)
            window_size (int): Census 및 SSIM 계산을 위한 창 크기
            occlusion_method (str): 가려짐 마스크 계산 방법
            edge_weighting (bool): 이미지 엣지에 따른 평활화 가중치 적용 여부
            stop_gradient (bool): 역전파 중지 플래그
            bidirectional (bool): 양방향 손실 계산 여부
            scale_weights (list, optional): 각 스케일에 대한 가중치 리스트
        """
        super(MultiScaleUFlowLoss, self).__init__()
        
        self.base_loss = UFlowLoss(
            photometric_weight=photometric_weight,
            census_weight=census_weight,
            smoothness_weight=smoothness_weight,
            ssim_weight=ssim_weight,
            window_size=window_size,
            occlusion_method=occlusion_method,
            edge_aware_smoothness=edge_weighting,
            stop_gradient=stop_gradient,
            bidirectional=bidirectional
        )
        
        self.scale_weights = scale_weights
    
    def _create_image_pyramid(self, image, num_scales):
        """
        이미지 피라미드 생성

        Args:
            image (torch.Tensor): 입력 이미지 [B, C, H, W]
            num_scales (int): 피라미드 레벨 수

        Returns:
            list: 이미지 피라미드 리스트
        """
        pyramid = [image]
        
        for scale in range(1, num_scales):
            # 이미지 크기 계산
            height = image.shape[2] // (2 ** scale)
            width = image.shape[3] // (2 ** scale)
            
            # 다운샘플링
            scaled_image = F.interpolate(
                image, size=(height, width), mode='bilinear', align_corners=False
            )
            
            pyramid.append(scaled_image)
        
        return pyramid
    
    def _create_mask_pyramid(self, mask, num_scales):
        """
        마스크 피라미드 생성

        Args:
            mask (torch.Tensor): 입력 마스크 [B, 1, H, W]
            num_scales (int): 피라미드 레벨 수

        Returns:
            list: 마스크 피라미드 리스트
        """
        pyramid = [mask]
        
        for scale in range(1, num_scales):
            # 마스크 크기 계산
            height = mask.shape[2] // (2 ** scale)
            width = mask.shape[3] // (2 ** scale)
            
            # 다운샘플링
            scaled_mask = F.interpolate(
                mask, size=(height, width), mode='nearest'
            )
            
            pyramid.append(scaled_mask)
        
        return pyramid
    
    def forward(self, image1, image2, flow_pyramids_forward, flow_pyramids_backward=None, valid_mask=None):
        """
        다중 스케일 UFlow 총 손실 계산

        Args:
            image1 (torch.Tensor): 첫 번째 이미지 [B, C, H, W]
            image2 (torch.Tensor): 두 번째 이미지 [B, C, H, W]
            flow_pyramids_forward (list): 순방향 광학 흐름 피라미드 리스트
            flow_pyramids_backward (list, optional): 역방향 광학 흐름 피라미드 리스트
            valid_mask (torch.Tensor, optional): 유효 영역 마스크 [B, 1, H, W]

        Returns:
            dict: 각 스케일 및 손실 구성 요소에 대한 총 손실 값
        """
        num_scales = len(flow_pyramids_forward)
        
        # 자동 스케일 가중치 계산
        if self.scale_weights is None:
            scale_weights = [1.0 / (2 ** scale) for scale in range(num_scales)]
            scale_weights_sum = sum(scale_weights)
            scale_weights = [w / scale_weights_sum for w in scale_weights]
        else:
            scale_weights = self.scale_weights
        
        # 이미지 피라미드 생성
        image1_pyramid = self._create_image_pyramid(image1, num_scales)
        image2_pyramid = self._create_image_pyramid(image2, num_scales)
        
        # 유효 마스크 피라미드 생성 (있는 경우)
        valid_mask_pyramid = None
        if valid_mask is not None:
            valid_mask_pyramid = self._create_mask_pyramid(valid_mask, num_scales)
        
        # 역방향 흐름 피라미드 확인
        if flow_pyramids_backward is not None:
            backward_flow_exists = True
        else:
            backward_flow_exists = False
        
        # 모든 스케일에 대한 손실 계산
        total_loss = 0.0
        all_losses = {}
        
        for scale in range(num_scales):
            # 현재 스케일의 이미지 및 흐름
            img1_scaled = image1_pyramid[scale]
            img2_scaled = image2_pyramid[scale]
            flow_forward_scaled = flow_pyramids_forward[scale]
            
            # 현재 스케일의 유효 마스크 (있는 경우)
            scale_valid_mask = None
            if valid_mask_pyramid is not None:
                scale_valid_mask = valid_mask_pyramid[scale]
            
            # 역방향 흐름 (있는 경우)
            flow_backward_scaled = None
            if backward_flow_exists:
                flow_backward_scaled = flow_pyramids_backward[scale]
            
            # 현재 스케일의 손실 계산
            scale_losses = self.base_loss(
                img1_scaled, img2_scaled, flow_forward_scaled, flow_backward_scaled, scale_valid_mask
            )
            
            # 스케일 가중치 적용
            weighted_loss = scale_weights[scale] * scale_losses['total_loss']
            total_loss += weighted_loss
            
            # 손실 저장
            all_losses[f'scale_{scale}'] = scale_losses
        
        # 총 손실 저장
        all_losses['total_loss'] = total_loss
        
        return all_losses


class FullSequentialLoss(nn.Module):
    """
    UFlow의 전체 손실 함수 (다중 스케일 + 시퀀스 손실)
    
    다중 스케일 손실과 시퀀스 손실을 결합
    """
    def __init__(
        self,
        photometric_weight=1.0,
        census_weight=1.0,
        smoothness_weight=1.0,
        sequence_weight=0.2,
        occlusion_method='forward_backward',
        use_occlusion=True,
        use_valid_mask=True,
        second_order_smoothness=False,
        edge_aware_smoothness=True,
        stop_gradient=True,
        bidirectional=False,
        scale_weights=None
    ):
        """
        Args:
            photometric_weight (float): Photometric 손실 가중치
            census_weight (float): Census 손실 가중치
            smoothness_weight (float): 평활화 손실 가중치
            sequence_weight (float): 시퀀스 손실 가중치
            occlusion_method (str): 가려짐 탐지 방법
            use_occlusion (bool): 가려짐 마스크 사용 여부
            use_valid_mask (bool): 유효 영역 마스크 사용 여부
            second_order_smoothness (bool): 2차 평활화 사용 여부
            edge_aware_smoothness (bool): 에지 인식 평활화 사용 여부
            stop_gradient (bool): 그래디언트 흐름 제어 사용 여부
            bidirectional (bool): 양방향 손실 계산 여부
            scale_weights (list, optional): 각 스케일에 대한 가중치 리스트
        """
        super(FullSequentialLoss, self).__init__()
        
        # 다중 스케일 손실
        self.multiscale_loss = MultiScaleUFlowLoss(
            photometric_weight=photometric_weight,
            census_weight=census_weight,
            smoothness_weight=smoothness_weight,
            ssim_weight=0.85,
            window_size=7,
            occlusion_method=occlusion_method,
            edge_weighting=edge_aware_smoothness,
            stop_gradient=stop_gradient,
            bidirectional=bidirectional,
            scale_weights=scale_weights
        )
        
        # 시퀀스 손실
        self.sequence_loss = SequenceLoss(
            alpha=1.0,  # 여기서는 1.0으로 설정하고 아래에서 sequence_weight로 조정
            use_occlusion_mask=use_occlusion,
            use_valid_mask=use_valid_mask,
            stop_gradient=stop_gradient,
            distance='robust_l1'
        )
        
        # 가중치
        self.sequence_weight = sequence_weight
    
    def forward(self, 
                images, 
                flow_pyramids,
                flow_t1_t3=None, 
                flow_t3_t1=None, 
                valid_mask=None):
        """
        전체 손실 계산
        
        Args:
            images (list): 연속 3개 이미지 리스트 [img_t1, img_t2, img_t3]
            flow_pyramids (list): 방향별 흐름 피라미드 리스트 [t1->t2, t2->t3, t2->t1, t3->t2]
            flow_t1_t3 (torch.Tensor, optional): t1→t3 방향 흐름 (직접 계산)
            flow_t3_t1 (torch.Tensor, optional): t3→t1 방향 흐름 (직접 계산)
            valid_mask (torch.Tensor, optional): 유효 영역 마스크
            
        Returns:
            dict: 각 스케일 및 손실 구성 요소에 대한 총 손실 값
        """
        if len(images) < 3:
            raise ValueError("시퀀스 손실 계산을 위해 최소 3개의 연속 이미지가 필요합니다.")
        
        img_t1, img_t2, img_t3 = images[0], images[1], images[2]
        
        if len(flow_pyramids) < 2:
            raise ValueError("시퀀스 손실 계산을 위해 최소 t1→t2, t2→t3 흐름 피라미드가 필요합니다.")
        
        flow_pyramids_t1_t2 = flow_pyramids[0]
        flow_pyramids_t2_t3 = flow_pyramids[1]
        
        flow_pyramids_t2_t1 = None
        flow_pyramids_t3_t2 = None
        
        if len(flow_pyramids) >= 4:
            flow_pyramids_t2_t1 = flow_pyramids[2]
            flow_pyramids_t3_t2 = flow_pyramids[3]
        
        # 다중 스케일 손실 계산 (t1→t2)
        multiscale_losses_t1_t2 = self.multiscale_loss(
            img_t1, img_t2, flow_pyramids_t1_t2, flow_pyramids_t2_t1, valid_mask
        )
        
        # 다중 스케일 손실 계산 (t2→t3)
        multiscale_losses_t2_t3 = self.multiscale_loss(
            img_t2, img_t3, flow_pyramids_t2_t3, flow_pyramids_t3_t2, valid_mask
        )
        
        # 다중 스케일 손실 평균
        multiscale_total_loss = (
            multiscale_losses_t1_t2['total_loss'] + 
            multiscale_losses_t2_t3['total_loss']
        ) / 2.0
        
        # t1→t3 흐름이 제공되지 않은 경우, 시퀀스 손실은 계산하지 않음
        sequence_total_loss = 0.0
        
        if flow_t1_t3 is not None:
            # 가장 높은 해상도의 흐름 추출
            flow_t1_t2 = flow_pyramids_t1_t2[0]
            flow_t2_t3 = flow_pyramids_t2_t3[0]
            
            # 시퀀스 손실 계산
            sequence_losses = self.sequence_loss(
                flow_t1_t2, flow_t2_t3, flow_t1_t3, flow_t3_t1, valid_mask
            )
            
            sequence_total_loss = sequence_losses['sequence_loss']
        
        # 총 손실 계산
        total_loss = multiscale_total_loss + self.sequence_weight * sequence_total_loss
        
        # 결과 딕셔너리 생성
        loss_dict = {
            'total_loss': total_loss,
            'multiscale_loss': multiscale_total_loss,
            'sequence_loss': sequence_total_loss * self.sequence_weight
        }
        
        # 다중 스케일 손실 결과 추가
        for key, value in multiscale_losses_t1_t2.items():
            if key != 'total_loss' and isinstance(value, torch.Tensor) and value.numel() == 1:
                loss_dict[f't1_t2_{key}'] = value
        
        for key, value in multiscale_losses_t2_t3.items():
            if key != 'total_loss' and isinstance(value, torch.Tensor) and value.numel() == 1:
                loss_dict[f't2_t3_{key}'] = value
        
        return loss_dict


# 테스트 코드
if __name__ == "__main__":
    # 테스트 데이터 생성
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    
    # 이미지와 다중 스케일 흐름 피라미드 생성
    img1 = torch.rand(batch_size, channels, height, width)
    img2 = torch.rand(batch_size, channels, height, width)
    img3 = torch.rand(batch_size, channels, height, width)
    
    # 다양한 해상도의 흐름 피라미드 생성 (4개 레벨)
    flow_pyramids_t1_t2 = []
    flow_pyramids_t2_t3 = []
    flow_pyramids_t2_t1 = []
    flow_pyramids_t3_t2 = []
    
    # 레벨 0: 원본 크기 (256x256)
    flow_pyramids_t1_t2.append(torch.randn(batch_size, 2, height, width) * 5.0)
    flow_pyramids_t2_t3.append(torch.randn(batch_size, 2, height, width) * 5.0)
    flow_pyramids_t2_t1.append(torch.randn(batch_size, 2, height, width) * 5.0)
    flow_pyramids_t3_t2.append(torch.randn(batch_size, 2, height, width) * 5.0)
    
    # 레벨 1: 원본 크기 1/2 (128x128)
    flow_pyramids_t1_t2.append(torch.randn(batch_size, 2, height//2, width//2) * 2.5)
    flow_pyramids_t2_t3.append(torch.randn(batch_size, 2, height//2, width//2) * 2.5)
    flow_pyramids_t2_t1.append(torch.randn(batch_size, 2, height//2, width//2) * 2.5)
    flow_pyramids_t3_t2.append(torch.randn(batch_size, 2, height//2, width//2) * 2.5)
    
    # t1→t3, t3→t1 직접 흐름 (원본 크기)
    flow_t1_t3 = torch.randn(batch_size, 2, height, width) * 8.0
    flow_t3_t1 = torch.randn(batch_size, 2, height, width) * 8.0
    
    # GPU 사용 가능한 경우 데이터 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1 = img1.to(device)
    img2 = img2.to(device)
    img3 = img3.to(device)
    flow_pyramids_t1_t2 = [flow.to(device) for flow in flow_pyramids_t1_t2]
    flow_pyramids_t2_t3 = [flow.to(device) for flow in flow_pyramids_t2_t3]
    flow_pyramids_t2_t1 = [flow.to(device) for flow in flow_pyramids_t2_t1]
    flow_pyramids_t3_t2 = [flow.to(device) for flow in flow_pyramids_t3_t2]
    flow_t1_t3 = flow_t1_t3.to(device)
    flow_t3_t1 = flow_t3_t1.to(device)
    
    print("\n1. 시퀀스 손실 테스트")
    sequence_loss = SequenceLoss(alpha=0.2, stop_gradient=True)
    sequence_loss = sequence_loss.to(device)
    
    losses_seq = sequence_loss(
        flow_pyramids_t1_t2[0],  # 가장 높은 해상도의 t1→t2 흐름
        flow_pyramids_t2_t3[0],  # 가장 높은 해상도의 t2→t3 흐름
        flow_t1_t3,              # 직접 계산된 t1→t3 흐름
        flow_t3_t1               # 직접 계산된 t3→t1 흐름
    )
    
    print("시퀀스 손실 결과:")
    for key, value in losses_seq.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"  {key}: {value.item():.6f}")
    
    print("\n2. 전체 시퀀셜 손실 테스트")
    full_loss = FullSequentialLoss(
        photometric_weight=1.0,
        census_weight=1.0,
        smoothness_weight=0.1,
        sequence_weight=0.2,
        stop_gradient=True,
        bidirectional=False
    )
    full_loss = full_loss.to(device)
    
    losses_full = full_loss(
        [img1, img2, img3],
        [flow_pyramids_t1_t2, flow_pyramids_t2_t3, flow_pyramids_t2_t1, flow_pyramids_t3_t2],
        flow_t1_t3,
        flow_t3_t1
    )
    
    print("전체 시퀀셜 손실 결과:")
    for key, value in sorted(losses_full.items()):
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"  {key}: {value.item():.6f}")
    
    print("\n테스트 성공!") 