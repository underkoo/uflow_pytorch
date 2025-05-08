import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

def abs_robust_loss(diff, eps=0.01, q=0.4):
  """The so-called robust loss used by DDFlow."""
  return torch.pow((torch.abs(diff) + eps), q)
  
class OcclusionMask(nn.Module):
    """
    광학 흐름에서 가려짐(occlusion) 영역을 탐지하는 모듈
    """
    def __init__(self, method='wang', 
                 alpha=0.01, beta=0.5,
                 occ_weights=None,
                 occ_thresholds=None,
                 occ_clip_max=None):
        """
        Args:
            method (str): 가려짐 탐지 방법
                - 'forward_backward': 순방향 및 역방향 흐름의 일관성 확인
                - 'brox': Brox의 방법
                - 'wang': Wang et al.의 방법 (기본값)
                - 'wang4': Wang 방법 4배 다운샘플링 적용
                - 'uflow': UFlow 논문의 고급 복합 방법
            alpha (float): forward_backward 방식의 매개변수
            beta (float): forward_backward 방식의 매개변수
            occ_weights (dict): uflow 방식의 가중치 매개변수
            occ_thresholds (dict): uflow 방식의 임계값 매개변수
            occ_clip_max (dict): uflow 방식의 최대 클리핑 값
        """
        super(OcclusionMask, self).__init__()
        self.method = method
        self.alpha = alpha
        self.beta = beta
        
        # UFlow 방식 매개변수
        self.occ_weights = occ_weights if occ_weights is not None else {
            'fb_abs': 1000.0,
            'forward_collision': 1000.0,
            'backward_zero': 1000.0
        }
        
        self.occ_thresholds = occ_thresholds if occ_thresholds is not None else {
            'fb_abs': 1.5,
            'forward_collision': 0.4,
            'backward_zero': 0.25
        }
        
        self.occ_clip_max = occ_clip_max if occ_clip_max is not None else {
            'fb_abs': 10.0,
            'forward_collision': 5.0
        }
    
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
            beta=self.beta,
            occ_weights=self.occ_weights,
            occ_thresholds=self.occ_thresholds,
            occ_clip_max=self.occ_clip_max
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
    def __init__(self, patch_size=7, use_occlusion_mask=True):
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
        
        # 패치 기반 Census 변환 및 해밍 거리 계산
        census1 = self._compute_census_transform(img1, self.patch_size)
        census2 = self._compute_census_transform(img2_warped, self.patch_size)
        
        hamming_bhw1 = self._soft_hamming(census1, census2)

        loss = abs_robust_loss(hamming_bhw1)
        
        # 마스크 적용
        if self.use_occlusion_mask and occlusion_mask is not None:
            loss = loss * occlusion_mask
        
        if valid_mask is not None:
            loss = loss * valid_mask
            
        return loss
    
    def _compute_census_transform(self, image, patch_size):
        """PyTorch 구현의 Census transform"""
        # RGB to grayscale 변환
        if image.shape[1] == 3:
            intensities = (0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]) * 255
        else:
            intensities = image * 255
        # Identity kernel 생성 (PyTorch 컨볼루션에 맞게 채널 수 조정)
        kernel = torch.eye(patch_size * patch_size).view(
            1, 1, patch_size, patch_size, patch_size * patch_size
        ).permute(4, 0, 1, 2, 3).reshape(
            patch_size * patch_size, 1, patch_size, patch_size
        ).to(image.device)
        
        # 컨볼루션 수행
        neighbors = F.conv2d(intensities, kernel, padding='same')
        
        # 차이 계산 및 정규화
        diff = neighbors - intensities.repeat(1, patch_size * patch_size, 1, 1)
        diff_norm = diff / torch.sqrt(0.81 + diff**2)
        
        return diff_norm

    def _soft_hamming(self, a_bhwk, b_bhwk, thresh=.1):
        sq_dist_bhwk = torch.square(a_bhwk - b_bhwk)
        soft_thresh_dist_bhwk = sq_dist_bhwk / (thresh + sq_dist_bhwk)
        return torch.sum(soft_thresh_dist_bhwk, dim=1, keepdim=True)

class SmoothnessLoss(nn.Module):
    """
    평활화 손실 (Smoothness Loss)
    
    광학 흐름의 공간적 평활함을 촉진하는 손실 함수
    """
    def __init__(self, edge_aware=True, second_order=False, edge_constant=150.0):
        """
        Args:
            edge_aware (bool): 이미지 에지를 고려한 평활화 여부
            second_order (bool): 이차 미분 사용 여부 (일차 미분보다 더 강한 평활화)
            edge_constant (float): 에지 가중치 계산에 사용되는 상수
        """
        super(SmoothnessLoss, self).__init__()
        self.edge_aware = edge_aware
        self.second_order = second_order
        self.edge_constant = edge_constant
    
    def forward(self, flow, image=None, valid_mask=None):
        """
        평활화 손실 계산
        
        Args:
            flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
            image (torch.Tensor, optional): 참조 이미지 [B, C, H, W], edge_aware=True일 때 사용
            valid_mask (torch.Tensor, optional): 유효 영역 마스크 [B, 1, H, W]
            
        Returns:
            torch.Tensor: 평활화 손실 (스칼라)
        """
        # 일차 또는 이차 미분 계산
        if not self.second_order:
            # 일차 미분
            flow_dx = self._gradient(flow, 'x')
            flow_dy = self._gradient(flow, 'y')
            
            # 평활화 손실 계산
            if self.edge_aware and image is not None:
                # 이미지 그라디언트 계산
                img_dx = self._gradient(image, 'x')
                img_dy = self._gradient(image, 'y')
                
                # 이미지 에지 가중치 (exp(-edge_constant * gradient))
                weights_x = torch.exp(-self.edge_constant * torch.mean(torch.abs(img_dx), dim=1, keepdim=True))
                weights_y = torch.exp(-self.edge_constant * torch.mean(torch.abs(img_dy), dim=1, keepdim=True))
                
                # 에지 가중치 적용
                weighted_flow_dx = flow_dx * weights_x
                weighted_flow_dy = flow_dy * weights_y
                
                loss = torch.mean(torch.abs(weighted_flow_dx)) + torch.mean(torch.abs(weighted_flow_dy))
            else:
                # 기본 평활화 손실
                loss = torch.mean(torch.abs(flow_dx)) + torch.mean(torch.abs(flow_dy))
        else:
            # 이차 미분 (라플라시안)
            flow_lap = self._gradient(self._gradient(flow, 'x'), 'x') + self._gradient(self._gradient(flow, 'y'), 'y')
        
            # 평활화 손실 계산
            if self.edge_aware and image is not None:
                # 이미지 라플라시안
                img_lap = self._gradient(self._gradient(image, 'x'), 'x') + self._gradient(self._gradient(image, 'y'), 'y')
            
                # 이미지 에지 가중치 (exp(-edge_constant * gradient))
                weights = torch.exp(-self.edge_constant * torch.mean(torch.abs(img_lap), dim=1, keepdim=True))
                
                # 에지 가중치 적용
                weighted_flow_lap = flow_lap * weights
                
                loss = torch.mean(torch.abs(weighted_flow_lap))
            else:
                # 기본 이차 평활화 손실
                loss = torch.mean(torch.abs(flow_lap))
        
        # 유효 마스크 적용 (선택적)
        if valid_mask is not None:
            valid_pixels = torch.sum(valid_mask) + 1e-8
            loss = torch.sum(loss * valid_mask) / valid_pixels
            
        return loss
    
    def _gradient(self, tensor, direction):
        """
        텐서의 공간 그라디언트 계산
        
        Args:
            tensor (torch.Tensor): 그라디언트를 계산할 텐서 [B, C, H, W]
            direction (str): 그라디언트 방향 ('x' 또는 'y')
            
        Returns:
            torch.Tensor: 그라디언트 [B, C, H, W]
        """
        B, C, H, W = tensor.shape
        
        if direction == 'x':
            # x 방향 그라디언트
            tensor_pad = F.pad(tensor, (1, 1, 0, 0), mode='replicate')
            grad = tensor_pad[:, :, :, 2:] - tensor_pad[:, :, :, :-2]
            grad = grad / 2.0
        elif direction == 'y':
            # y 방향 그라디언트
            tensor_pad = F.pad(tensor, (0, 0, 1, 1), mode='replicate')
            grad = tensor_pad[:, :, 2:, :] - tensor_pad[:, :, :-2, :]
            grad = grad / 2.0
        else:
            raise ValueError(f"알 수 없는 그라디언트 방향: {direction}")
            
        return grad


class UFlowLoss(nn.Module):
    """
    UFlow 손실 함수
    
    논문에서 설명된 다양한 손실 함수들의 조합
    - Photometric Loss (L1 + SSIM)
    - Census Loss (지역적 패턴 유지)
    - Smoothness Loss (흐름 필드의 부드러움)
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
                 edge_constant=150.0,
                 stop_gradient=True,
                 bidirectional=False):
        """
        Args:
            photometric_weight (float): Photometric 손실 가중치
            census_weight (float): Census 손실 가중치
            smoothness_weight (float): 평활화 손실 가중치
            ssim_weight (float): Photometric 손실 내에서 SSIM 비중
            window_size (int): SSIM 계산에 사용할 윈도우 크기
            occlusion_method (str): 가려짐 탐지 방법
            use_occlusion (bool): 가려짐 마스크 사용 여부
            use_valid_mask (bool): 유효 영역 마스크 사용 여부
            second_order_smoothness (bool): 2차 도함수 기반 평활화 사용 여부
            edge_aware_smoothness (bool): 에지 인식 평활화 여부
            edge_constant (float): 에지 가중치 계산에 사용되는 상수
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
        self.smoothness_loss = SmoothnessLoss(edge_aware=edge_aware_smoothness, second_order=second_order_smoothness, edge_constant=edge_constant)
    
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
        # 광학 흐름 디태치 (TensorFlow의 stop_gradient와 같은 역할)
        flow_for_occlusion = flow.detach() if self.stop_gradient else flow
        flow_opposite_for_occlusion = flow_opposite.detach() if self.stop_gradient and flow_opposite is not None else flow_opposite
        
        # 가려짐 마스크 계산
        occlusion_mask = None
        if self.use_occlusion and flow_opposite is not None:
            occlusion_mask = self.occlusion_mask(flow_for_occlusion, flow_opposite_for_occlusion)
            
            # stop-gradient 적용 - occlusion 마스크의 그래디언트가 네트워크에 영향을 미치지 않도록 함
            if self.stop_gradient:
                occlusion_mask = occlusion_mask.detach()
        
        # 와핑된 이미지 계산 (flow 그래디언트는 유지)
        img_tgt_warped = self._warp_image(img_tgt, flow)
        
        # stop-gradient 적용 - 와핑된 이미지 그래디언트가 흐름 네트워크로 직접 전파되지 않도록 함
        # 이미지만 디태치하여 흐름에 대한 그래디언트는 보존
        if self.stop_gradient:
            # 원본 흐름에서 와핑된 이미지의 그래디언트를 분리
            img_tgt_warped = img_tgt_warped.detach()
        
        # 각 손실 함수 계산
        # 이미지와 마스크에 대한 그래디언트만 계산, 흐름에 대한 그래디언트는 photometric과 census에서 제외
        photometric = self.photometric_loss(img_src, img_tgt_warped, occlusion_mask, valid_mask)
        census = self.census_loss(img_src, img_tgt_warped, occlusion_mask, valid_mask)
        
        # smoothness 손실은 흐름에 대한 그래디언트를 포함
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
            
            # 정규화 팩터를 그래디언트로부터 분리
            norm_factor = torch.sum(weighted_mask) + 1e-8
            if self.stop_gradient:
                norm_factor = norm_factor.detach()
                
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


class MultiScaleUFlowLoss(nn.Module):
    """
    다중 스케일 UFlow 손실 함수
    
    각 스케일에서 손실을 계산하고 가중 평균을 구함
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
        edge_constant=150.0,
        stop_gradient=True,
        bidirectional=False,
        scale_weights=None
    ):
        """
        Args:
            photometric_weight (float): Photometric 손실 가중치
            census_weight (float): Census 손실 가중치
            smoothness_weight (float): 평활화 손실 가중치
            ssim_weight (float): Photometric 손실 내에서 SSIM 비중
            window_size (int): SSIM 계산에 사용할 윈도우 크기
            occlusion_method (str): 가려짐 탐지 방법
            edge_weighting (bool): 에지를 고려한 평활화 여부
            edge_constant (float): 에지 가중치 계산에 사용되는 상수
            stop_gradient (bool): 역전파 중지 플래그
            bidirectional (bool): 양방향 손실 계산 여부
            scale_weights (list, optional): 각 스케일의 가중치
        """
        super(MultiScaleUFlowLoss, self).__init__()
        
        self.photometric_weight = photometric_weight
        self.census_weight = census_weight
        self.smoothness_weight = smoothness_weight
        self.ssim_weight = ssim_weight
        self.window_size = window_size
        self.stop_gradient = stop_gradient
        self.bidirectional = bidirectional
        self.scale_weights = scale_weights
        self.edge_constant = edge_constant
        
        # 스케일별 손실 함수
        self.uflow_losses = nn.ModuleList([
            UFlowLoss(
            photometric_weight=photometric_weight,
            census_weight=census_weight,
            smoothness_weight=smoothness_weight,
            ssim_weight=ssim_weight,
            window_size=window_size,
            occlusion_method=occlusion_method,
                use_occlusion=True,
                use_valid_mask=True,
                second_order_smoothness=False,
            edge_aware_smoothness=edge_weighting,
                edge_constant=edge_constant,
            stop_gradient=stop_gradient,
            bidirectional=bidirectional
        )
            for _ in range(5)  # 5개의 피라미드 레벨
        ])
    
    def _create_image_pyramid(self, image, num_scales):
        """
        이미지 피라미드 생성

        Args:
            image (torch.Tensor): 입력 이미지 [B, C, H, W]
            num_scales (int): 피라미드 레벨 수

        Returns:
            list: 다양한 크기의 이미지 리스트
        """
        pyramid = [image]
        
        for i in range(1, num_scales):
            if self.stop_gradient:
                # 이전 스케일의 그래디언트를 분리하여 새 스케일 계산
                prev_image = pyramid[-1].detach()
            else:
                prev_image = pyramid[-1]
                
            # 평균 풀링으로 다운샘플링
            down_image = F.avg_pool2d(prev_image, kernel_size=2, stride=2)
            pyramid.append(down_image)
        
        return pyramid
    
    def _create_mask_pyramid(self, mask, num_scales):
        """
        마스크 피라미드 생성

        Args:
            mask (torch.Tensor): 입력 마스크 [B, 1, H, W]
            num_scales (int): 피라미드 레벨 수

        Returns:
            list: 다양한 크기의 마스크 리스트
        """
        if mask is None:
            return None
            
        pyramid = [mask]
        
        for i in range(1, num_scales):
            if self.stop_gradient:
                # 이전 스케일의 그래디언트를 분리하여 새 스케일 계산
                prev_mask = pyramid[-1].detach()
            else:
                prev_mask = pyramid[-1]
                
            # 최대 풀링으로 다운샘플링 (마스크에서는 유효성을 보존하기 위해)
            down_mask = F.max_pool2d(prev_mask, kernel_size=2, stride=2)
            pyramid.append(down_mask)
        
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
        backward_flow_exists = flow_pyramids_backward is not None
        
        # 모든 스케일에 대한 손실 계산
        total_loss = 0.0
        all_losses = {}
        all_losses['total_loss'] = 0.0
        all_losses['photometric_loss'] = 0.0
        all_losses['census_loss'] = 0.0
        all_losses['smoothness_loss'] = 0.0
        
        for scale in range(num_scales):
            # 현재 스케일의 이미지와 흐름
            img1_scale = image1_pyramid[scale]
            img2_scale = image2_pyramid[scale]
            flow_forward_scale = flow_pyramids_forward[scale]
            
            # 현재 스케일의 역방향 흐름 (있는 경우)
            flow_backward_scale = None
            if backward_flow_exists:
                flow_backward_scale = flow_pyramids_backward[scale]
            
            # 현재 스케일의 유효 마스크 (있는 경우)
            mask_scale = None
            if valid_mask_pyramid is not None:
                mask_scale = valid_mask_pyramid[scale]
            
            # 현재 스케일에서 손실 계산
            scale_loss_dict = self.uflow_losses[scale](
                img1_scale, 
                img2_scale, 
                flow_forward_scale, 
                flow_backward_scale, 
                mask_scale
            )
            
            # 스케일 가중치 적용
            weight = scale_weights[scale]
            scale_total_loss = scale_loss_dict['total_loss'] * weight
            
            # 전체 손실에 누적
            total_loss += scale_total_loss
            
            # 각 손실 구성 요소를 누적
            for loss_name in ['photometric_loss', 'census_loss', 'smoothness_loss']:
                if loss_name in scale_loss_dict:
                    all_losses[loss_name] += scale_loss_dict[loss_name] * weight
            
            # 현재 스케일의 손실을 저장
            all_losses[f'scale_{scale}_total_loss'] = scale_loss_dict['total_loss']
            all_losses[f'scale_{scale}_weight'] = weight
            
            # 가려짐 마스크가 있으면 저장
            if 'occlusion_mask' in scale_loss_dict:
                all_losses[f'scale_{scale}_occlusion_mask'] = scale_loss_dict['occlusion_mask']
        
        # 총 손실 저장
        all_losses['total_loss'] = total_loss
        
        return all_losses


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
    
    # 다양한 해상도의 흐름 피라미드 생성 (2개 레벨)
    flow_pyramids_forward = []
    flow_pyramids_backward = []
    
    # 레벨 0: 원본 크기 (256x256)
    flow_pyramids_forward.append(torch.randn(batch_size, 2, height, width) * 5.0)
    flow_pyramids_backward.append(torch.randn(batch_size, 2, height, width) * 5.0)
    
    # 레벨 1: 원본 크기 1/2 (128x128)
    flow_pyramids_forward.append(torch.randn(batch_size, 2, height//2, width//2) * 2.5)
    flow_pyramids_backward.append(torch.randn(batch_size, 2, height//2, width//2) * 2.5)
    
    # GPU 사용 가능한 경우 데이터 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1 = img1.to(device)
    img2 = img2.to(device)
    flow_pyramids_forward = [flow.to(device) for flow in flow_pyramids_forward]
    flow_pyramids_backward = [flow.to(device) for flow in flow_pyramids_backward]
    
    print("\n" + "-"*60)
    print("다중 스케일 손실 테스트")
    print("-"*60)
    
    multiscale_loss = MultiScaleUFlowLoss(
        photometric_weight=1.0,
        census_weight=1.0,
        smoothness_weight=0.1,
        stop_gradient=True,
        bidirectional=True
    )
    multiscale_loss = multiscale_loss.to(device)
    
    losses_ms = multiscale_loss(img1, img2, flow_pyramids_forward, flow_pyramids_backward)
    
    print("다중 스케일 손실 결과:")
    for key, value in losses_ms.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"  {key}: {value.item():.6f}")
    
    print("\n" + "-"*60)
    print("테스트 성공!")
    print("-"*60) 