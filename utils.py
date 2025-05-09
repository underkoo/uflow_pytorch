import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def flow_to_warp(flow):
    """
    광학 흐름을 와핑 그리드로 변환
    
    Args:
        flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
        
    Returns:
        torch.Tensor: 와핑 그리드 [B, 2, H, W]
    """
    B, _, H, W = flow.shape
    
    # 기본 그리드 생성
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H-1, H, device=flow.device),
        torch.linspace(0, W-1, W, device=flow.device)
    )
    grid = torch.stack([grid_x, grid_y]).unsqueeze(0).repeat(B, 1, 1, 1)
    
    # 디버깅용
    print(f"[DEBUG-WARP] 기본 그리드 x: min={grid[:, 0].min().item()}, max={grid[:, 0].max().item()}")
    print(f"[DEBUG-WARP] 기본 그리드 y: min={grid[:, 1].min().item()}, max={grid[:, 1].max().item()}")
    print(f"[DEBUG-WARP] 흐름 x: min={flow[:, 0].min().item()}, max={flow[:, 0].max().item()}")
    
    # 그리드에 흐름 추가
    warp = grid + flow
    
    print(f"[DEBUG-WARP] 왼쪽 경계 기본 x: {grid[0, 0, H//2, :5].cpu().numpy().flatten()}")
    print(f"[DEBUG-WARP] 왼쪽 경계 warp x: {warp[0, 0, H//2, :5].cpu().numpy().flatten()}")
    print(f"[DEBUG-WARP] 오른쪽 경계 기본 x: {grid[0, 0, H//2, -5:].cpu().numpy().flatten()}")
    print(f"[DEBUG-WARP] 오른쪽 경계 warp x: {warp[0, 0, H//2, -5:].cpu().numpy().flatten()}")
    
    return warp


def coords_grid(batch, h, w, device):
    """
    배치 크기에 맞는 좌표 그리드 생성
    
    Args:
        batch (int): 배치 크기
        h (int): 높이
        w (int): 너비
        device (torch.device): 텐서 장치
        
    Returns:
        torch.Tensor: 좌표 그리드 [B, 2, H, W]
    """
    y, x = torch.meshgrid(
        torch.linspace(0, h-1, h, device=device),
        torch.linspace(0, w-1, w, device=device)
    )
    coords = torch.stack([x, y], dim=0).unsqueeze(0).repeat(batch, 1, 1, 1)
    return coords


def normalize_coords(coords, h, w):
    """
    좌표를 [-1, 1] 범위로 정규화
    
    Args:
        coords (torch.Tensor): 좌표 [B, 2, H, W]
        h (int): 높이
        w (int): 너비
        
    Returns:
        torch.Tensor: 정규화된 좌표 [B, 2, H, W]
    """
    # 정규화된 좌표로 변환 [-1, 1] 범위
    coords_normalized = torch.zeros_like(coords)
    coords_normalized[:, 0] = 2.0 * coords[:, 0] / (w - 1) - 1.0
    coords_normalized[:, 1] = 2.0 * coords[:, 1] / (h - 1) - 1.0
    
    return coords_normalized


def create_gaussian_kernel(kernel_size, sigma, device):
    """
    가우시안 커널 생성
    
    Args:
        kernel_size (int): 커널 크기
        sigma (float): 가우시안 시그마
        device (torch.device): 텐서 디바이스
        
    Returns:
        torch.Tensor: 가우시안 커널 [1, 1, kernel_size, kernel_size]
    """
    # kernel_size는 홀수여야 함
    if kernel_size % 2 == 0:
        raise ValueError("커널 크기는 홀수여야 합니다")
        
    coords = torch.arange(kernel_size, device=device).float()
    coords -= (kernel_size - 1) / 2
    
    g = coords ** 2
    g = torch.exp(-(g[:, None] + g[None, :]) / (2 * sigma ** 2))
    g /= g.sum()
    
    return g.view(1, 1, kernel_size, kernel_size)


def warp_image(image, flow):
    """
    이미지를 광학 흐름을 사용하여 와핑
    
    Args:
        image (torch.Tensor): 이미지 [B, C, H, W]
        flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
        
    Returns:
        torch.Tensor: 와핑된 이미지 [B, C, H, W]
    """
    return warp_features(image, flow, mode='bilinear')


def warp_features(features, flow, mode='bilinear'):
    """
    특징 맵을 광학 흐름을 사용하여 와핑
    
    Args:
        features (torch.Tensor): 특징 맵 [B, C, H, W]
        flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
        mode (str): 보간 모드 ('nearest', 'bilinear')
        
    Returns:
        torch.Tensor: 와핑된 특징 맵 [B, C, H, W]
    """
    batch_size, channels, height, width = features.shape
    device = features.device
    
    # 좌표 그리드 생성 ([-1, 1] 범위) - 이제 [B, H, W, 2] 형태
    grid = create_grid(batch_size, height, width, device)
    
    # 흐름 스케일링 ([-1, 1] 범위로 정규화)
    scaled_flow = torch.zeros_like(flow)
    scaled_flow[:, 0] = flow[:, 0] / (width - 1) * 2  # x 방향
    scaled_flow[:, 1] = flow[:, 1] / (height - 1) * 2  # y 방향
    
    # 샘플링 좌표 계산
    # grid는 이미 [B, H, W, 2] 형태이므로 추가 permute 불필요
    sample_grid = grid + scaled_flow.permute(0, 2, 3, 1)
    
    # 그리드 샘플링으로 와핑
    warped_features = F.grid_sample(
        features, 
        sample_grid, 
        mode=mode, 
        padding_mode='zeros', 
        align_corners=False
    )
    
    return warped_features


def compute_range_map(flow, downsampling_factor=1, reduce_downsampling_bias=True, resize_output=True):
    """
    광학 흐름에서 범위 맵 계산 (각 대상 픽셀이 소스 픽셀에 의해 얼마나 잘 커버되는지)
    
    Args:
        flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
        downsampling_factor (int): 다운샘플링 비율
        reduce_downsampling_bias (bool): 다운샘플링 편향 감소 여부
        resize_output (bool): 출력을 원본 크기로 조정할지 여부
        
    Returns:
        torch.Tensor: 범위 맵 [B, 1, H, W], 값이 클수록 해당 픽셀의 커버리지가 높음
    """
    device = flow.device
    batch_size, _, height, width = flow.shape
    
    if downsampling_factor > 1:
        # 흐름 다운샘플링
        small_height = height // downsampling_factor
        small_width = width // downsampling_factor
        
        # 다운샘플링 전 흐름 조정
        flow_small = F.interpolate(
            flow, 
            size=(small_height, small_width), 
            mode='bilinear', 
            align_corners=False
        ) / downsampling_factor
        
        height, width = small_height, small_width
    else:
        flow_small = flow
    
    # 타겟 자리표 생성 (정규화된 좌표계)
    # torch 1.10 이상에서는 indexing='ij' 인자가 필요함
    try:
        y_t, x_t = torch.meshgrid(
            torch.linspace(0, height - 1, height, device=device),
            torch.linspace(0, width - 1, width, device=device),
            indexing='ij'
        )
    except TypeError:
        # 이전 버전 PyTorch에서는 indexing 인자 없음
        y_t, x_t = torch.meshgrid(
            torch.linspace(0, height - 1, height, device=device),
            torch.linspace(0, width - 1, width, device=device)
        )
    
    # 타겟 자리표를 배치 차원으로 확장
    y_t = y_t.reshape(1, height, width, 1).repeat(batch_size, 1, 1, 1)
    x_t = x_t.reshape(1, height, width, 1).repeat(batch_size, 1, 1, 1)
    
    # 흐름을 사용하여 소스 자리표 계산
    flow_u = flow_small[:, 0].permute(0, 2, 1).reshape(batch_size, width, height, 1).permute(0, 2, 1, 3)
    flow_v = flow_small[:, 1].permute(0, 1, 2).reshape(batch_size, height, width, 1)
    
    x_s = x_t + flow_u
    y_s = y_t + flow_v
    
    # 정수 자리표 계산
    x_s_floor = torch.floor(x_s)
    y_s_floor = torch.floor(y_s)
    
    # 가중치 계산 (bilinear 보간용)
    alpha_x = x_s - x_s_floor
    alpha_y = y_s - y_s_floor
    
    # 비선형 보간을 위한 가중치
    w_0 = (1 - alpha_x) * (1 - alpha_y)  # top-left
    w_1 = alpha_x * (1 - alpha_y)        # top-right
    w_2 = (1 - alpha_x) * alpha_y        # bottom-left
    w_3 = alpha_x * alpha_y              # bottom-right
    
    # 정수 자리표를 정수로 변환하고 비트 쉬프트로 해시화
    # 이 부분은 PyTorch에서 scatter_add를 사용하기 위한 준비
    x_s_i = x_s_floor.long()
    y_s_i = y_s_floor.long()
    
    # 범위 맵 초기화
    range_map = torch.zeros((batch_size, height, width), device=device)
    
    # 유효 마스크 (이미지 경계 내부 확인)
    valid_0 = (x_s_i >= 0) & (x_s_i < width) & (y_s_i >= 0) & (y_s_i < height)
    valid_1 = (x_s_i + 1 >= 0) & (x_s_i + 1 < width) & (y_s_i >= 0) & (y_s_i < height)
    valid_2 = (x_s_i >= 0) & (x_s_i < width) & (y_s_i + 1 >= 0) & (y_s_i + 1 < height)
    valid_3 = (x_s_i + 1 >= 0) & (x_s_i + 1 < width) & (y_s_i + 1 >= 0) & (y_s_i + 1 < height)
    
    # 각 배치에 대해 반복하여 scatter_add로 값 누적
    for b in range(batch_size):
        # top-left
        idx_0 = y_s_i[b][valid_0[b]] * width + x_s_i[b][valid_0[b]]
        val_0 = w_0[b][valid_0[b]]
        range_map[b].reshape(-1).scatter_add_(0, idx_0.squeeze(-1), val_0.squeeze(-1))
        
        # top-right
        idx_1 = y_s_i[b][valid_1[b]] * width + (x_s_i[b][valid_1[b]] + 1)
        val_1 = w_1[b][valid_1[b]]
        range_map[b].reshape(-1).scatter_add_(0, idx_1.squeeze(-1), val_1.squeeze(-1))
        
        # bottom-left
        idx_2 = (y_s_i[b][valid_2[b]] + 1) * width + x_s_i[b][valid_2[b]]
        val_2 = w_2[b][valid_2[b]]
        range_map[b].reshape(-1).scatter_add_(0, idx_2.squeeze(-1), val_2.squeeze(-1))
        
        # bottom-right
        idx_3 = (y_s_i[b][valid_3[b]] + 1) * width + (x_s_i[b][valid_3[b]] + 1)
        val_3 = w_3[b][valid_3[b]]
        range_map[b].reshape(-1).scatter_add_(0, idx_3.squeeze(-1), val_3.squeeze(-1))
    
    # 다운샘플링 편향 감소
    if reduce_downsampling_bias and downsampling_factor > 1:
        range_map = range_map * (downsampling_factor ** 2)
    
    # 채널 차원 추가
    range_map = range_map.unsqueeze(1)
    
    # 원본 크기로 복원
    if resize_output and downsampling_factor > 1:
        range_map = F.interpolate(
            range_map, 
            size=(flow.shape[2], flow.shape[3]), 
            mode='bilinear', 
            align_corners=False
        )
    
    return range_map


def estimate_occlusion_mask(flow_forward, flow_backward, method='forward_backward', **kwargs):
    """
    광학 흐름에서 가려짐 마스크 추정
    
    Args:
        flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
        flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
        method (str): 가려짐 추정 방식
            - 'forward_backward': 단순 순방향-역방향 일관성 확인
            - 'brox': Brox 방식의 일관성 확인 
            - 'wang': Wang et al.의 방법 (역방향 범위 맵 기반)
            - 'wang4': Wang 방법의 변형: 4배 다운샘플링
            - 'uflow': UFlow 논문의 복합 방법 (세 가지 요소 결합)
        
    Returns:
        torch.Tensor: 가려짐 마스크 [B, 1, H, W], 1=가려짐 없음, 0=가려짐
    """
    # 그래디언트 전파를 막기 위해 흐름 분리
    with torch.no_grad():
        flow_forward_detached = flow_forward.detach()
        flow_backward_detached = flow_backward.detach()
    
    if method == 'forward_backward':
        alpha = kwargs.get('alpha', 0.01)
        beta = kwargs.get('beta', 0.5)
        
        # 역방향 흐름 와핑
        warped_backward_flow = warp_features(flow_backward_detached, flow_forward_detached)
        
        # 흐름 일관성 오차 계산
        forward_mag = torch.sum(flow_forward_detached ** 2, dim=1, keepdim=True) + 1e-6
        backward_mag = torch.sum(warped_backward_flow ** 2, dim=1, keepdim=True) + 1e-6
        
        # 흐름 합 계산 (이상적으로는 0)
        flow_sum = flow_forward_detached + warped_backward_flow
        flow_sum_mag = torch.sum(flow_sum ** 2, dim=1, keepdim=True)
        
        # 일관성 확인
        threshold = alpha * (forward_mag + backward_mag) + beta
        occlusion_mask = (flow_sum_mag < threshold).float()
        
    elif method == 'brox':
        # 역방향 흐름 와핑
        warped_backward_flow = warp_features(flow_backward_detached, flow_forward_detached)
        
        # 순방향-역방향 제곱 차이 계산
        flow_sum = flow_forward_detached + warped_backward_flow
        fb_sq_diff = torch.sum(flow_sum ** 2, dim=1, keepdim=True)
        
        # 제곱 합 계산
        fb_sum_sq = torch.sum(flow_forward_detached ** 2 + warped_backward_flow ** 2, dim=1, keepdim=True)
        
        # Brox 기준으로 가려짐 판별
        occlusion_mask = (fb_sq_diff <= 0.01 * fb_sum_sq + 0.5).float()
        
    elif method == 'wang':
        # Wang et al.의 방법: 역방향 흐름의 범위 맵 사용
        range_map_backward = compute_range_map(
            flow_backward_detached, 
            downsampling_factor=1, 
            reduce_downsampling_bias=False, 
            resize_output=False
        )
        
        # TensorFlow 구현과 일치: 낮은 값이 가려짐을 나타냄
        # 범위 맵 값이 0에 가까울수록 해당 위치로 오는 흐름이 없음 = 가려짐
        occlusion_mask = torch.clamp(range_map_backward, 0.0, 1.0)
        
    elif method == 'wang4':
        # Wang 방법의 변형: 4배 다운샘플링
        range_map_backward = compute_range_map(
            flow_backward_detached, 
            downsampling_factor=4, 
            reduce_downsampling_bias=True, 
            resize_output=True
        )
        
        # TensorFlow 구현과 일치: 낮은 값이 가려짐을 나타냄
        occlusion_mask = torch.clamp(range_map_backward, 0.0, 1.0)
        
    elif method == 'uflow':
        # UFlow 논문의 복합 방법 설정값
        occ_weights = kwargs.get('occ_weights', {
            'fb_abs': 1000.0,
            'forward_collision': 1000.0,
            'backward_zero': 1000.0
        })
        
        occ_thresholds = kwargs.get('occ_thresholds', {
            'fb_abs': 1.5,
            'forward_collision': 0.4,
            'backward_zero': 0.25
        })
        
        occ_clip_max = kwargs.get('occ_clip_max', {
            'fb_abs': 10.0,
            'forward_collision': 5.0
        })
        
        # 사용자가 지정한 활성 요소 (기본값은 전부 활성화)
        occ_active = kwargs.get('occ_active', {
            'fb_abs': True,
            'forward_collision': True,
            'backward_zero': True
        })
        
        batch_size, _, height, width = flow_forward.shape
        device = flow_forward.device
        occlusion_scores = {}
        
        # 1. 순방향-역방향 일관성 (fb_abs)
        if 'fb_abs' in occ_weights and occ_active.get('fb_abs', True):
            # 역방향 흐름 와핑
            warped_backward_flow = warp_features(flow_backward_detached, flow_forward_detached)
            
            # 순방향-역방향 차이 계산
            fb_diff = flow_forward_detached + warped_backward_flow
            fb_sq_diff = torch.sum(fb_diff ** 2, dim=1, keepdim=True) ** 0.5
            
            # 클리핑 적용
            occlusion_scores['fb_abs'] = torch.clamp(
                fb_sq_diff, 0.0, occ_clip_max['fb_abs'])
        
        # 2. 순방향 충돌 (forward_collision)
        if 'forward_collision' in occ_weights and occ_active.get('forward_collision', True):
            # 순방향 흐름의 범위 맵 계산
            range_map_forward = compute_range_map(
                flow_forward_detached, 
                downsampling_factor=1, 
                reduce_downsampling_bias=True, 
                resize_output=True
            )
            
            # 순방향 범위 맵을 순방향 흐름으로 와핑
            fwd_range_map_warped = warp_features(range_map_forward, flow_forward_detached)
            
            # [1, max-1] 범위로 리스케일
            occlusion_scores['forward_collision'] = torch.clamp(
                fwd_range_map_warped, 1.0, occ_clip_max['forward_collision']) - 1.0
        
        # 3. 역방향 제로 (backward_zero)
        if 'backward_zero' in occ_weights and occ_active.get('backward_zero', True):
            # 역방향 흐름의 범위 맵 계산
            range_map_backward = compute_range_map(
                flow_backward_detached, 
                downsampling_factor=4, 
                reduce_downsampling_bias=True, 
                resize_output=True
            )
            
            # 범위 맵이 거의 0인 지역이 가려짐 가능성 높음
            # 0에 가까울수록 높은 점수 (가려짐 가능성)
            occlusion_scores['backward_zero'] = (
                1.0 - torch.clamp(range_map_backward, 0.0, 1.0))
        
        # 가려짐 로짓 계산
        occlusion_logits = torch.zeros((batch_size, 1, height, width), device=device)
        
        for k, v in occlusion_scores.items():
            occlusion_logits += (v - occ_thresholds[k]) * occ_weights[k]
        
        # 시그모이드 적용하여 최종 마스크 계산 (1=가려짐 없음, 0=가려짐)
        # TensorFlow 구현과 일치하도록 수정
        occlusion_mask = torch.sigmoid(-occlusion_logits)  # 높은 로짓 = 가려짐 가능성 높음
    
    else:
        raise ValueError(f"알 수 없는 가려짐 추정 방식: {method}")
        
    return occlusion_mask


def spatial_gradient(tensor, direction):
    """
    텐서의 공간적 그래디언트 계산
    
    Args:
        tensor (torch.Tensor): 그래디언트를 계산할 텐서 [B, C, H, W]
        direction (str): 그래디언트 방향 ('x' 또는 'y')
        
    Returns:
        torch.Tensor: 그래디언트 [B, C, H, W]
    """
    B, C, H, W = tensor.shape
    
    if direction == 'x':
        # x 방향 그래디언트 (패딩 사용)
        tensor_pad = F.pad(tensor, (1, 1, 0, 0), mode='replicate')
        gradient = (tensor_pad[:, :, :, 2:] - tensor_pad[:, :, :, :-2]) / 2.0
    elif direction == 'y':
        # y 방향 그래디언트 (패딩 사용)
        tensor_pad = F.pad(tensor, (0, 0, 1, 1), mode='replicate')
        gradient = (tensor_pad[:, :, 2:, :] - tensor_pad[:, :, :-2, :]) / 2.0
    else:
        raise ValueError(f"Unknown gradient direction: {direction}")
        
    return gradient


def create_image_pyramid(img, num_levels, with_original=True):
    """
    이미지 피라미드 생성
    
    Args:
        img (torch.Tensor): 입력 이미지 [B, C, H, W]
        num_levels (int): 피라미드 레벨 수
        with_original (bool): 원본 이미지를 피라미드에 포함할지 여부
        
    Returns:
        list: 다양한 크기의 이미지 리스트
    """
    pyramid = []
    
    if with_original:
        pyramid.append(img)
    
    current = img
    for _ in range(num_levels - 1 if with_original else num_levels):
        current = F.avg_pool2d(current, kernel_size=2, stride=2)
        pyramid.append(current)
    
    return pyramid


def upsample_flow(flow, target_size=None, scale_factor=None):
    """
    광학 흐름을 업샘플링하고 스케일 적용
    
    Args:
        flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
        target_size (tuple, optional): 목표 크기 (H', W')
        scale_factor (float, optional): 스케일 인자
        
    Returns:
        torch.Tensor: 업샘플링된 광학 흐름 [B, 2, H', W']
    """
    if target_size is None and scale_factor is None:
        raise ValueError("target_size 또는 scale_factor 중 하나는 제공되어야 합니다.")
    
    # 업샘플링 전 크기
    original_h, original_w = flow.shape[2:4]
    
    # 목표 크기 계산
    if target_size is not None:
        target_h, target_w = target_size
        h_scale = target_h / original_h
        w_scale = target_w / original_w
    else:
        target_h = int(original_h * scale_factor)
        target_w = int(original_w * scale_factor)
        h_scale = w_scale = scale_factor
    
    # 흐름 업샘플링
    upsampled_flow = F.interpolate(
        flow, 
        size=(target_h, target_w), 
        mode='bilinear', 
        align_corners=False
    )
    
    # 흐름 값 조정
    upsampled_flow[:, 0] *= w_scale  # x 방향 흐름
    upsampled_flow[:, 1] *= h_scale  # y 방향 흐름
    
    return upsampled_flow


def compute_cost_volume(features1, features2, max_displacement=4):
    """
    두 특징 맵 간의 비용 볼륨(cost volume) 계산
    
    Args:
        features1 (torch.Tensor): 첫 번째 특징 맵 [B, C, H, W]
        features2 (torch.Tensor): 두 번째 특징 맵 [B, C, H, W]
        max_displacement (int): 최대 변위
        
    Returns:
        torch.Tensor: 비용 볼륨 [B, (2*max_displacement+1)^2, H, W]
    """
    B, C, H, W = features1.shape
    
    # 특징 정규화
    features1_norm = F.normalize(features1, p=2, dim=1)
    features2_norm = F.normalize(features2, p=2, dim=1)
    
    # 패딩
    pad = max_displacement
    features2_padded = F.pad(features2_norm, [pad, pad, pad, pad])
    
    # 비용 볼륨 계산
    cost_list = []
    for i in range(2 * pad + 1):
        for j in range(2 * pad + 1):
            features2_shifted = features2_padded[:, :, i:i+H, j:j+W]
            correlation = torch.mean(features1_norm * features2_shifted, dim=1, keepdim=True)
            cost_list.append(correlation)
    
    # 결합
    cost_volume = torch.cat(cost_list, dim=1)
    
    return cost_volume


def compute_ssim(img1, img2, window_size=7, sigma=1.5, c1=0.01**2, c2=0.03**2):
    """
    두 이미지 간의 구조적 유사성 지수(SSIM) 계산
    
    Args:
        img1 (torch.Tensor): 첫 번째 이미지 [B, C, H, W]
        img2 (torch.Tensor): 두 번째 이미지 [B, C, H, W]
        window_size (int): SSIM 계산용 가우시안 창 크기
        sigma (float): 가우시안 표준 편차
        c1 (float): 분모 0 방지 상수
        c2 (float): 분모 0 방지 상수
        
    Returns:
        torch.Tensor: SSIM 유사성 맵 [B, 1, H, W]
    """
    device = img1.device
    
    # 가우시안 창 생성
    window = generate_gaussian_kernel(window_size, sigma, device)
    
    # 패딩 적용
    pad = window_size // 2
    
    # 각 채널 처리
    batch_size, channels, height, width = img1.shape
    ssim_map = torch.zeros((batch_size, 1, height, width), device=device)
    
    for c in range(channels):
        # 각 채널 추출
        img1_c = img1[:, c:c+1, :, :]
        img2_c = img2[:, c:c+1, :, :]
        
        # 평균 및 분산 계산을 위한 컨볼루션 적용
        mu1 = F.conv2d(F.pad(img1_c, [pad]*4, mode='reflect'), window, groups=1)
        mu2 = F.conv2d(F.pad(img2_c, [pad]*4, mode='reflect'), window, groups=1)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # 분산 및 공분산 계산
        sigma1_sq = F.conv2d(F.pad(img1_c**2, [pad]*4, mode='reflect'), window, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(F.pad(img2_c**2, [pad]*4, mode='reflect'), window, groups=1) - mu2_sq
        sigma12 = F.conv2d(F.pad(img1_c * img2_c, [pad]*4, mode='reflect'), window, groups=1) - mu1_mu2
        
        # SSIM 계산
        cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        ssim_map_c = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
        
        # 채널별 SSIM 합산
        ssim_map += ssim_map_c / channels
    
    return ssim_map


def compute_flow_smooth_loss(flow, image=None, order=1, boundary_penalty=0.1):
    """
    광학 흐름의 평활도(smoothness) 손실 계산
    
    Args:
        flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
        image (torch.Tensor, optional): 참조 이미지 [B, C, H, W]
        order (int): 손실 계산 차수 (1=일차 도함수, 2=이차 도함수)
        boundary_penalty (float): 경계 페널티 가중치
        
    Returns:
        torch.Tensor: 평활도 손실 스칼라
    """
    B, _, H, W = flow.shape
    
    if order == 1:
        # 그래디언트 계산
        flow_dx, flow_dy = compute_flow_gradients(flow)
        
        # 이미지 에지 가중치
        weights = torch.ones_like(flow_dx)
        
        if image is not None:
            # 이미지 그래디언트 계산
            img_dx, img_dy = compute_image_gradients(image)
            img_grad_mag = torch.sqrt(img_dx ** 2 + img_dy ** 2).mean(dim=1, keepdim=True)
            
            # 이미지 에지에서는 평활도 페널티 감소
            weights = torch.exp(-img_grad_mag / 0.1)
        
        # 평활도 손실 계산
        loss_x = (flow_dx ** 2 * weights).mean()
        loss_y = (flow_dy ** 2 * weights).mean()
        
        # 경계 페널티 (선택 사항)
        if boundary_penalty > 0:
            boundary_mask_x = torch.ones_like(flow_dx)
            boundary_mask_y = torch.ones_like(flow_dy)
            
            boundary_mask_x[:, :, :, 0] = boundary_penalty
            boundary_mask_x[:, :, :, -1] = boundary_penalty
            boundary_mask_y[:, :, 0, :] = boundary_penalty
            boundary_mask_y[:, :, -1, :] = boundary_penalty
            
            loss_x = (flow_dx ** 2 * weights * boundary_mask_x).mean()
            loss_y = (flow_dy ** 2 * weights * boundary_mask_y).mean()
        
        loss = loss_x + loss_y
    
    elif order == 2:
        # 이차 도함수 (라플라시안) 계산
        padded_flow = F.pad(flow, [1, 1, 1, 1], mode='replicate')
        
        laplacian_x = (padded_flow[:, 0:1, 1:-1, 2:] + padded_flow[:, 0:1, 1:-1, 0:-2] + 
                      padded_flow[:, 0:1, 2:, 1:-1] + padded_flow[:, 0:1, 0:-2, 1:-1] - 
                      4 * padded_flow[:, 0:1, 1:-1, 1:-1])
        
        laplacian_y = (padded_flow[:, 1:2, 1:-1, 2:] + padded_flow[:, 1:2, 1:-1, 0:-2] + 
                      padded_flow[:, 1:2, 2:, 1:-1] + padded_flow[:, 1:2, 0:-2, 1:-1] - 
                      4 * padded_flow[:, 1:2, 1:-1, 1:-1])
        
        # 이미지 에지 가중치
        weights = torch.ones_like(laplacian_x)
        
        if image is not None:
            img_dx, img_dy = compute_image_gradients(image)
            img_grad_mag = torch.sqrt(img_dx ** 2 + img_dy ** 2).mean(dim=1, keepdim=True)
            weights = torch.exp(-img_grad_mag / 0.1)
        
        loss = ((laplacian_x ** 2 + laplacian_y ** 2) * weights).mean()
    
    else:
        raise ValueError(f"지원되지 않는 차수: {order}, 1 또는 2만 지원됩니다.")
    
    return loss


def charbonnier_loss(x, alpha=0.25, epsilon=1e-9):
    """
    Charbonnier 손실 함수
    
    Args:
        x (torch.Tensor): 입력 텐서
        alpha (float): Charbonnier 매개변수
        epsilon (float): 수치 안정성을 위한 작은 상수
        
    Returns:
        torch.Tensor: Charbonnier 손실
    """
    return torch.mean((x ** 2 + epsilon) ** alpha)


def create_flow_pyramid(flow, num_levels):
    """
    광학 흐름 피라미드 생성
    
    Args:
        flow (torch.Tensor): 원본 광학 흐름 [B, 2, H, W]
        num_levels (int): 피라미드 레벨 수
        
    Returns:
        list: 광학 흐름 피라미드 (최고 해상도에서 최저 해상도 순)
    """
    flow_pyramid = [flow]
    
    for level in range(1, num_levels):
        # 이전 레벨의 흐름
        prev_flow = flow_pyramid[-1]
        
        # 다운샘플링
        curr_flow = F.avg_pool2d(prev_flow, kernel_size=2, stride=2)
        
        # 흐름 크기 조정
        curr_flow = curr_flow / 2.0
        
        flow_pyramid.append(curr_flow)
    
    return flow_pyramid


def compute_photometric_loss(image1, image2, flow, ssim_weight=0.85, window_size=7, occlusion_mask=None, valid_mask=None):
    """
    광학 흐름에 기반한 두 이미지 간의 포토메트릭 손실 계산
    
    Args:
        image1 (torch.Tensor): 첫 번째 이미지 [B, C, H, W]
        image2 (torch.Tensor): 두 번째 이미지 [B, C, H, W]
        flow (torch.Tensor): 이미지 1에서 이미지 2로의 광학 흐름 [B, 2, H, W]
        ssim_weight (float): SSIM 손실의 가중치 (0 ~ 1 사이의 값)
        window_size (int): SSIM 계산을 위한 윈도우 크기
        occlusion_mask (torch.Tensor, optional): 가려짐 마스크 [B, 1, H, W], 1=가려짐 없음, 0=가려짐
        valid_mask (torch.Tensor, optional): 유효 픽셀 마스크 [B, 1, H, W], 1=유효, 0=무효
        
    Returns:
        torch.Tensor: 포토메트릭 손실
    """
    # 이미지 2를 와핑
    warped_image2 = warp_image(image2, flow)
    
    # L1 손실 계산
    l1_loss = torch.abs(image1 - warped_image2)
    
    # 채널 차원에 대해 평균
    l1_loss = torch.mean(l1_loss, dim=1, keepdim=True)
    
    # SSIM 손실 계산 (1 - SSIM: 유사도가 낮을수록 손실이 높음)
    if ssim_weight > 0:
        ssim_map = compute_ssim(image1, warped_image2, window_size=window_size)
        ssim_loss = (1.0 - ssim_map) / 2.0  # 범위: [0, 1]
        
        # 합성 손실 (가중 평균)
        loss = (1.0 - ssim_weight) * l1_loss + ssim_weight * ssim_loss
    else:
        loss = l1_loss
    
    # 마스크 적용
    if occlusion_mask is not None:
        loss = loss * occlusion_mask
    
    if valid_mask is not None:
        loss = loss * valid_mask
    
    # 유효 픽셀에 대한 평균 계산
    valid_pixels = torch.ones_like(loss)
    if occlusion_mask is not None:
        valid_pixels = valid_pixels * occlusion_mask
    if valid_mask is not None:
        valid_pixels = valid_pixels * valid_mask
    
    return torch.sum(loss) / (torch.sum(valid_pixels) + 1e-8)


def visualize_flow(flow, max_flow=None):
    """
    광학 흐름을 시각화하는 함수
    
    Args:
        flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
        max_flow (float, optional): 최대 흐름 크기
        
    Returns:
        torch.Tensor: 시각화된 흐름 [B, 3, H, W]
    """
    B, _, H, W = flow.shape
    flow_np = flow.detach().cpu().numpy()
    
    # HSV 색상 맵 생성
    flow_vis_list = []
    
    for b in range(B):
        # 광학 흐름 분리
        u = flow_np[b, 0]
        v = flow_np[b, 1]
        
        # 각도와 크기 계산
        rad = np.sqrt(u ** 2 + v ** 2)
        
        if max_flow is None:
            max_flow = np.max(rad) if np.max(rad) > 0 else 1
        
        # 정규화
        rad = np.minimum(rad / max_flow, 1)
        
        # 각도 계산
        ang = np.arctan2(v, u) / np.pi
        
        # HSV 색상
        h = (ang + 1) / 2
        s = rad
        v = np.ones_like(rad)
        
        # HSV -> RGB 변환
        hsv = np.stack([h, s, v], axis=2)
        flow_vis = hsv_to_rgb(hsv)
        
        # 텐서로 변환
        flow_vis = torch.from_numpy(flow_vis).permute(2, 0, 1).float()
        flow_vis_list.append(flow_vis)
    
    return torch.stack(flow_vis_list)


def hsv_to_rgb(hsv):
    """
    HSV 색상 공간을 RGB로 변환
    
    Args:
        hsv (numpy.ndarray): HSV 이미지 [H, W, 3]
        
    Returns:
        numpy.ndarray: RGB 이미지 [H, W, 3]
    """
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    c = v * s
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = v - c
    
    zeros = np.zeros_like(h)
    
    mask = h < 1/6
    r = np.where(mask, c, zeros)
    g = np.where(mask, x, zeros)
    b = np.where(mask, zeros, zeros)
    
    mask = (1/6 <= h) & (h < 2/6)
    r += np.where(mask, x, zeros)
    g += np.where(mask, c, zeros)
    b += np.where(mask, zeros, zeros)
    
    mask = (2/6 <= h) & (h < 3/6)
    r += np.where(mask, zeros, zeros)
    g += np.where(mask, c, zeros)
    b += np.where(mask, x, zeros)
    
    mask = (3/6 <= h) & (h < 4/6)
    r += np.where(mask, zeros, zeros)
    g += np.where(mask, x, zeros)
    b += np.where(mask, c, zeros)
    
    mask = (4/6 <= h) & (h < 5/6)
    r += np.where(mask, x, zeros)
    g += np.where(mask, zeros, zeros)
    b += np.where(mask, c, zeros)
    
    mask = (5/6 <= h)
    r += np.where(mask, c, zeros)
    g += np.where(mask, zeros, zeros)
    b += np.where(mask, x, zeros)
    
    rgb = np.stack([r + m, g + m, b + m], axis=2)
    
    return rgb


def create_grid(batch_size, height, width, device):
    """
    정규화된 좌표 그리드 생성 ([-1, 1] 범위)
    
    Args:
        batch_size (int): 배치 크기
        height (int): 높이
        width (int): 너비
        device (torch.device): 텐서를 저장할 장치
        
    Returns:
        torch.Tensor: 좌표 그리드 [B, H, W, 2] 형태로 변경
    """
    # 그리드 좌표 생성 ([-1, 1] 범위)
    x_coords = torch.linspace(-1.0, 1.0, width, device=device)
    y_coords = torch.linspace(-1.0, 1.0, height, device=device)
    
    # 2D 메쉬그리드 생성
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # [H, W, 2] 형태로 쌓기
    grid = torch.stack((x_grid, y_grid), dim=-1)
    
    # 배치 차원 추가 [B, H, W, 2]
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    return grid


def compute_flow_gradients(flow):
    """
    광학 흐름의 그라디언트 계산

    Args:
        flow (torch.Tensor): 광학 흐름 [B, 2, H, W]

    Returns:
        tuple: (dx, dy) 각각 그라디언트 텐서 [B, 2, H, W]
    """
    # 패딩 적용
    padded_flow = F.pad(flow, pad=(1, 1, 1, 1), mode='replicate')
    
    # 그라디언트 계산 (중앙 차분)
    dx = 0.5 * (padded_flow[:, :, 1:-1, 2:] - padded_flow[:, :, 1:-1, :-2])
    dy = 0.5 * (padded_flow[:, :, 2:, 1:-1] - padded_flow[:, :, :-2, 1:-1])
    
    return dx, dy


def compute_image_gradients(image):
    """
    이미지의 그라디언트 계산

    Args:
        image (torch.Tensor): 이미지 [B, C, H, W]

    Returns:
        tuple: (dx, dy) 각각 그라디언트 텐서 [B, C, H, W]
    """
    # 패딩 적용
    padded_image = F.pad(image, pad=(1, 1, 1, 1), mode='replicate')
    
    # 그라디언트 계산 (소벨 필터 근사)
    dx = 0.5 * (padded_image[:, :, 1:-1, 2:] - padded_image[:, :, 1:-1, :-2])
    dy = 0.5 * (padded_image[:, :, 2:, 1:-1] - padded_image[:, :, :-2, 1:-1])
    
    return dx, dy


def shift_tensor(tensor, offset_x, offset_y):
    """
    텐서를 수평 및 수직으로 이동

    Args:
        tensor (torch.Tensor): 이동할 텐서 [B, C, H, W]
        offset_x (int): 수평 이동량
        offset_y (int): 수직 이동량

    Returns:
        torch.Tensor: 이동된 텐서 [B, C, H, W]
    """
    # 이동 후 크기
    batch_size, channels, height, width = tensor.shape
    
    # 패딩 계산
    pad_x = max(abs(offset_x), 0)
    pad_y = max(abs(offset_y), 0)
    
    # 패딩 적용
    padded = F.pad(tensor, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)
    
    # 오프셋에 따른 슬라이싱
    x_start = pad_x + offset_x if offset_x < 0 else pad_x
    y_start = pad_y + offset_y if offset_y < 0 else pad_y
    
    # 이동된 텐서 추출
    shifted = padded[:, :, y_start:y_start + height, x_start:x_start + width]
    
    return shifted


def warp_flow(flow_ab, flow_bc):
    """
    두 개의 연속된 광학 흐름을 결합하여 새로운 흐름 생성 (A → C)

    Args:
        flow_ab (torch.Tensor): A에서 B로의 광학 흐름 [B, 2, H, W]
        flow_bc (torch.Tensor): B에서 C로의 광학 흐름 [B, 2, H, W]

    Returns:
        torch.Tensor: A에서 C로의 합성된 광학 흐름 [B, 2, H, W]
    """
    # B에서 C로의 흐름을 A에서 B로의 흐름 경로를 따라 와핑
    warped_flow_bc = warp_features(flow_bc, flow_ab)
    
    # 두 흐름 결합 (A→B→C)
    return flow_ab + warped_flow_bc


def compute_flow_warp_loss(flow_ab, flow_bc, flow_ac, occlusion_mask=None, valid_mask=None):
    """
    실제 흐름과 합성된 흐름 간의 워핑 손실 계산

    Args:
        flow_ab (torch.Tensor): A에서 B로의 광학 흐름 [B, 2, H, W]
        flow_bc (torch.Tensor): B에서 C로의 광학 흐름 [B, 2, H, W]
        flow_ac (torch.Tensor): A에서 C로의 광학 흐름 [B, 2, H, W]
        occlusion_mask (torch.Tensor, optional): 가려짐 마스크 [B, 1, H, W]
        valid_mask (torch.Tensor, optional): 유효 픽셀 마스크 [B, 1, H, W]

    Returns:
        torch.Tensor: 흐름 워핑 손실
    """
    # A→B→C 흐름 합성
    warped_flow = warp_flow(flow_ab, flow_bc)
    
    # 합성 흐름과 실제 흐름 간의 L1 차이
    loss = torch.abs(warped_flow - flow_ac)
    
    # 채널 차원에 대해 평균
    loss = torch.mean(loss, dim=1, keepdim=True)
    
    # 마스크 적용
    if occlusion_mask is not None:
        loss = loss * occlusion_mask
    
    if valid_mask is not None:
        loss = loss * valid_mask
    
    # 유효 픽셀에 대한 평균 계산
    valid_pixels = torch.ones_like(loss)
    if occlusion_mask is not None:
        valid_pixels = valid_pixels * occlusion_mask
    if valid_mask is not None:
        valid_pixels = valid_pixels * valid_mask
    
    return torch.sum(loss) / (torch.sum(valid_pixels) + 1e-8)


def compute_flow_consistency_loss(flow_forward, flow_backward, occlusion_mask=None, valid_mask=None):
    """
    순방향 및 역방향 흐름 간의 일관성 손실 계산

    Args:
        flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
        flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
        occlusion_mask (torch.Tensor, optional): 가려짐 마스크 [B, 1, H, W]
        valid_mask (torch.Tensor, optional): 유효 픽셀 마스크 [B, 1, H, W]

    Returns:
        torch.Tensor: 흐름 일관성 손실
    """
    # 역방향 흐름 와핑
    warped_backward_flow = warp_features(flow_backward, flow_forward)
    
    # 순방향 흐름 + 와핑된 역방향 흐름 (이상적으로는 0)
    flow_sum = flow_forward + warped_backward_flow
    
    # L1 손실 계산
    loss = torch.abs(flow_sum)
    
    # 채널 차원에 대해 평균
    loss = torch.mean(loss, dim=1, keepdim=True)
    
    # 마스크 적용
    if occlusion_mask is not None:
        loss = loss * occlusion_mask
    
    if valid_mask is not None:
        loss = loss * valid_mask
    
    # 유효 픽셀에 대한 평균 계산
    valid_pixels = torch.ones_like(loss)
    if occlusion_mask is not None:
        valid_pixels = valid_pixels * occlusion_mask
    if valid_mask is not None:
        valid_pixels = valid_pixels * valid_mask
    
    return torch.sum(loss) / (torch.sum(valid_pixels) + 1e-8)


def compute_valid_mask(flow, border_ratio=0.1):
    """
    유효한 광학 흐름 영역을 위한 마스크 계산
    
    Args:
        flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
        border_ratio (float): 경계 버퍼 비율 (0~1)
        
    Returns:
        torch.Tensor: 유효 마스크 [B, 1, H, W]
    """
    batch_size, _, height, width = flow.shape
    device = flow.device
    
    # 경계 크기
    border_x = int(width * border_ratio)
    border_y = int(height * border_ratio)
    
    # 모든 픽셀 유효로 초기화
    valid_mask = torch.ones((batch_size, 1, height, width), device=device)
    
    # 경계 부분 마스킹
    if border_x > 0 and border_y > 0:
        valid_mask[:, :, :border_y, :] = 0.0  # 상단
        valid_mask[:, :, height - border_y:, :] = 0.0  # 하단
        valid_mask[:, :, :, :border_x] = 0.0  # 좌측
        valid_mask[:, :, :, width - border_x:] = 0.0  # 우측
    
    # 광학 흐름을 고려한 추가 마스킹
    grid = create_grid(batch_size, height, width, device)
    
    # 흐름 스케일링 ([-1, 1] 범위로 정규화)
    scaled_flow = torch.zeros_like(flow)
    scaled_flow[:, 0] = flow[:, 0] / (width - 1) * 2  # x 방향
    scaled_flow[:, 1] = flow[:, 1] / (height - 1) * 2  # y 방향
    
    # 샘플링 좌표 계산
    sample_grid = grid + scaled_flow.permute(0, 2, 3, 1)
    
    # 좌표가 유효 범위 내에 있는지 확인 ([-1, 1])
    valid_x = (sample_grid[:, :, :, 0] >= -1.0) & (sample_grid[:, :, :, 0] <= 1.0)
    valid_y = (sample_grid[:, :, :, 1] >= -1.0) & (sample_grid[:, :, :, 1] <= 1.0)
    
    # 유효한 좌표 마스크 생성
    flow_valid = (valid_x & valid_y).unsqueeze(1).float()
    
    # 최종 유효 마스크 조합
    return valid_mask * flow_valid


def generate_gaussian_kernel(size=7, sigma=1.0, device='cpu'):
    """
    가우시안 커널 생성
    
    Args:
        size (int): 커널 크기 (반드시 홀수)
        sigma (float): 가우시안 시그마
        device (str or torch.device): 텐서를 저장할 장치
        
    Returns:
        torch.Tensor: 정규화된 가우시안 커널 [1, 1, size, size]
    """
    if size % 2 == 0:
        raise ValueError("커널 크기는 홀수여야 합니다.")
    
    # 가우시안 커널을 위한 좌표 생성
    coords = torch.arange(size, device=device).float() - (size - 1) / 2
    coords = coords.to(torch.float)  # 타입 변환 추가
    
    # x, y 좌표 그리드 생성
    x = coords.repeat(size, 1)
    y = x.t()
    
    # 2D 가우시안 계산
    kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
    
    # 정규화
    kernel = kernel / kernel.sum()
    
    # [1, 1, H, W] 형태로 반환
    return kernel.unsqueeze(0).unsqueeze(0)


# forward-backward 일관성 검사 관련 함수 추가

def compute_fb_squared_diff(flow_forward, flow_backward):
    """
    순방향 및 역방향 흐름 사이의 제곱 차이 계산
    
    Args:
        flow_forward (torch.Tensor): 순방향 흐름 [B, 2, H, W]
        flow_backward (torch.Tensor): 역방향 흐름 [B, 2, H, W]
        
    Returns:
        torch.Tensor: 제곱 차이 [B, 1, H, W]
    """
    # 역방향 흐름 와핑
    flow_backward_warped = warp_flow(flow_backward, flow_forward)
    
    # 역방향 흐름 반전 (좌표계 변환)
    flow_backward_warped_rev = -flow_backward_warped
    
    # 흐름 벡터 간 차이 계산
    diff = flow_forward - flow_backward_warped_rev
    
    # 제곱 차이 합 계산
    sq_diff = torch.sum(diff**2, dim=1, keepdim=True)
    
    return sq_diff


def compute_fb_sum_squared(flow_forward, flow_backward):
    """
    순방향 및 역방향 흐름 벡터 크기의 제곱 합 계산
    
    Args:
        flow_forward (torch.Tensor): 순방향 흐름 [B, 2, H, W]
        flow_backward (torch.Tensor): 역방향 흐름 [B, 2, H, W]
        
    Returns:
        torch.Tensor: 제곱 합 [B, 1, H, W]
    """
    # 역방향 흐름 와핑
    flow_backward_warped = warp_flow(flow_backward, flow_forward)
    
    # 흐름 벡터 크기의 제곱 계산
    forward_sq = torch.sum(flow_forward**2, dim=1, keepdim=True)
    backward_sq = torch.sum(flow_backward_warped**2, dim=1, keepdim=True)
    
    # 제곱 합 계산 (0으로 나누는 것을 방지하기 위해 작은 상수 추가)
    sum_sq = forward_sq + backward_sq + 1e-6
    
    return sum_sq


def mask_invalid(coords):
    """
    이미지 범위를 벗어난 좌표를 마스킹합니다.
    
    Args:
        coords (torch.Tensor): 이미지 좌표 [B, 2, H, W]
        
    Returns:
        torch.Tensor: 유효한 좌표를 나타내는 마스크 [B, 1, H, W] (유효=1, 무효=0)
    """
    if len(coords.shape) != 4:
        raise NotImplementedError("4D 텐서만 지원됩니다.")
        
    max_height = float(coords.shape[-2] - 1)
    max_width = float(coords.shape[-1] - 1)
    
    # x, y 좌표가 이미지 범위 내에 있는지 확인
    mask = torch.logical_and(
        torch.logical_and(coords[:, 0] >= 0.0, coords[:, 0] <= max_width),
        torch.logical_and(coords[:, 1] >= 0.0, coords[:, 1] <= max_height)
    )
    
    # 마스크를 float32로 변환하고 채널 차원 추가
    mask = mask.float().unsqueeze(1)
    
    return mask


# 테스트 코드
if __name__ == "__main__":
    import torch
    
    # 테스트 데이터 생성
    batch_size = 2
    height = 128
    width = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n----- mask_invalid 테스트 -----")
    
    # 1. 정상적인 좌표 테스트
    coords_normal = torch.zeros(batch_size, 2, height, width, device=device)
    coords_normal[:, 0] = torch.linspace(0, width-1, width, device=device).view(1, 1, -1)  # x 좌표
    coords_normal[:, 1] = torch.linspace(0, height-1, height, device=device).view(1, -1, 1)  # y 좌표
    
    mask_normal = mask_invalid(coords_normal)
    print(f"1. 정상 좌표 마스크 비율: {mask_normal.mean().item():.4f} (1=유효, 0=무효)")
    print(f"   - 최소 좌표: ({coords_normal[:, 0].min().item():.1f}, {coords_normal[:, 1].min().item():.1f})")
    print(f"   - 최대 좌표: ({coords_normal[:, 0].max().item():.1f}, {coords_normal[:, 1].max().item():.1f})")
    
    # 2. 경계값 테스트
    coords_boundary = torch.zeros(batch_size, 2, height, width, device=device)
    coords_boundary[:, 0] = torch.linspace(-1, width, width, device=device).view(1, 1, -1)  # x 좌표
    coords_boundary[:, 1] = torch.linspace(-1, height, height, device=device).view(1, -1, 1)  # y 좌표
    
    mask_boundary = mask_invalid(coords_boundary)
    print(f"\n2. 경계값 좌표 마스크 비율: {mask_boundary.mean().item():.4f}")
    print(f"   - 최소 좌표: ({coords_boundary[:, 0].min().item():.1f}, {coords_boundary[:, 1].min().item():.1f})")
    print(f"   - 최대 좌표: ({coords_boundary[:, 0].max().item():.1f}, {coords_boundary[:, 1].max().item():.1f})")
    
    # 3. 랜덤 좌표 테스트
    coords_random = torch.rand(batch_size, 2, height, width, device=device) * (width * 1.5) - (width * 0.25)
    mask_random = mask_invalid(coords_random)
    print(f"\n3. 랜덤 좌표 마스크 비율: {mask_random.mean().item():.4f}")
    print(f"   - 최소 좌표: ({coords_random[:, 0].min().item():.1f}, {coords_random[:, 1].min().item():.1f})")
    print(f"   - 최대 좌표: ({coords_random[:, 0].max().item():.1f}, {coords_random[:, 1].max().item():.1f})")
    
    # 4. 시각화를 위한 특정 패턴 테스트
    coords_pattern = torch.zeros(batch_size, 2, height, width, device=device)
    # 체스 패턴처럼 유효/무효 영역 생성
    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == 0:
                coords_pattern[:, 0, i, j] = j  # 유효한 x 좌표
                coords_pattern[:, 1, i, j] = i  # 유효한 y 좌표
            else:
                coords_pattern[:, 0, i, j] = width + 10  # 무효한 x 좌표
                coords_pattern[:, 1, i, j] = height + 10  # 무효한 y 좌표
    
    mask_pattern = mask_invalid(coords_pattern)
    print(f"\n4. 패턴 좌표 마스크 비율: {mask_pattern.mean().item():.4f}")
    print(f"   - 최소 좌표: ({coords_pattern[:, 0].min().item():.1f}, {coords_pattern[:, 1].min().item():.1f})")
    print(f"   - 최대 좌표: ({coords_pattern[:, 0].max().item():.1f}, {coords_pattern[:, 1].max().item():.1f})")
    
    # 5. 예외 처리 테스트
    try:
        # 3D 텐서로 테스트
        coords_3d = torch.zeros(batch_size, height, width, device=device)
        mask_3d = mask_invalid(coords_3d)
    except NotImplementedError as e:
        print(f"\n5. 3D 텐서 예외 처리 성공: {str(e)}")
    
    print("\n모든 mask_invalid 테스트가 완료되었습니다.") 