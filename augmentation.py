import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import math
import numpy as np


class PhotometricAugmentation:
    """
    이미지 쌍에 대한 색상 변환(photometric) 증강 적용
    """
    def __init__(self,
                 augment_color_swap=True,
                 augment_hue_shift=True,
                 augment_saturation=False,
                 augment_brightness=False,
                 augment_contrast=False,
                 augment_gaussian_noise=False,
                 augment_brightness_individual=False,
                 augment_contrast_individual=False,
                 max_delta_hue=0.5,
                 min_bound_saturation=0.8,
                 max_bound_saturation=1.2,
                 max_delta_brightness=0.1,
                 min_bound_contrast=0.8,
                 max_bound_contrast=1.2,
                 min_bound_gaussian_noise=0.0,
                 max_bound_gaussian_noise=0.02,
                 max_delta_brightness_individual=0.02,
                 min_bound_contrast_individual=0.95,
                 max_bound_contrast_individual=1.05):
        """
        Args:
            augment_color_swap (bool): 색상 채널 순서 변경 여부
            augment_hue_shift (bool): Hue 값 변경 여부
            augment_saturation (bool): 채도 변경 여부
            augment_brightness (bool): 밝기 변경 여부 (두 이미지 동일하게)
            augment_contrast (bool): 대비 변경 여부 (두 이미지 동일하게)
            augment_gaussian_noise (bool): 가우시안 노이즈 추가 여부
            augment_brightness_individual (bool): 각 이미지마다 독립적으로 밝기 변경 여부
            augment_contrast_individual (bool): 각 이미지마다 독립적으로 대비 변경 여부
            max_delta_hue (float): Hue 변화의 최대 범위 [-max_delta, max_delta]
            min_bound_saturation (float): 채도 변화의 최소값
            max_bound_saturation (float): 채도 변화의 최대값
            max_delta_brightness (float): 밝기 변화의 최대 범위 [-max_delta, max_delta]
            min_bound_contrast (float): 대비 변화의 최소값
            max_bound_contrast (float): 대비 변화의 최대값
            min_bound_gaussian_noise (float): 가우시안 노이즈 표준편차의 최소값
            max_bound_gaussian_noise (float): 가우시안 노이즈 표준편차의 최대값
            max_delta_brightness_individual (float): 개별 이미지 밝기 변화 최대 범위
            min_bound_contrast_individual (float): 개별 이미지 대비 변화 최소값
            max_bound_contrast_individual (float): 개별 이미지 대비 변화 최대값
        """
        self.augment_color_swap = augment_color_swap
        self.augment_hue_shift = augment_hue_shift
        self.augment_saturation = augment_saturation
        self.augment_brightness = augment_brightness
        self.augment_contrast = augment_contrast
        self.augment_gaussian_noise = augment_gaussian_noise
        self.augment_brightness_individual = augment_brightness_individual
        self.augment_contrast_individual = augment_contrast_individual
        
        self.max_delta_hue = max_delta_hue
        self.min_bound_saturation = min_bound_saturation
        self.max_bound_saturation = max_bound_saturation
        self.max_delta_brightness = max_delta_brightness
        self.min_bound_contrast = min_bound_contrast
        self.max_bound_contrast = max_bound_contrast
        self.min_bound_gaussian_noise = min_bound_gaussian_noise
        self.max_bound_gaussian_noise = max_bound_gaussian_noise
        self.max_delta_brightness_individual = max_delta_brightness_individual
        self.min_bound_contrast_individual = min_bound_contrast_individual
        self.max_bound_contrast_individual = max_bound_contrast_individual
    
    def __call__(self, images):
        """
        이미지 쌍에 색상 증강 적용
        
        Args:
            images (List[torch.Tensor]): 이미지 쌍 [B, C, H, W] 또는 [C, H, W]
            
        Returns:
            List[torch.Tensor]: 증강된 이미지 쌍
        """
        # 입력이 배치인지 단일 이미지인지 확인
        is_batch = images[0].dim() == 4
        device = images[0].device
        
        # 단일 이미지의 경우, 배치 차원 추가
        if not is_batch:
            images = [img.unsqueeze(0) for img in images]
        
        # 색상 채널 순서 변경
        if self.augment_color_swap and random.random() < 0.5:
            # 0, 1, 2 채널 중 랜덤 순서로 변경
            channel_perm = torch.randperm(3)
            images = [img[:, channel_perm] for img in images]
        
        # 이미지 쌍을 하나의 배치로 결합
        combined = torch.cat(images, dim=0)
        
        # Hue 변경
        if self.augment_hue_shift and random.random() < 0.5:
            hue_factor = random.uniform(-self.max_delta_hue, self.max_delta_hue)
            combined = TF.adjust_hue(combined, hue_factor)
        
        # 채도 변경
        if self.augment_saturation and random.random() < 0.5:
            saturation_factor = random.uniform(
                self.min_bound_saturation, self.max_bound_saturation)
            combined = TF.adjust_saturation(combined, saturation_factor)
        
        # 밝기 변경 (두 이미지 동일하게)
        if self.augment_brightness and random.random() < 0.5:
            brightness_factor = random.uniform(
                1.0 - self.max_delta_brightness, 1.0 + self.max_delta_brightness)
            combined = TF.adjust_brightness(combined, brightness_factor)
        
        # 대비 변경 (두 이미지 동일하게)
        if self.augment_contrast and random.random() < 0.5:
            contrast_factor = random.uniform(
                self.min_bound_contrast, self.max_bound_contrast)
            combined = TF.adjust_contrast(combined, contrast_factor)
        
        # 가우시안 노이즈 추가
        if self.augment_gaussian_noise and random.random() < 0.5:
            sigma = random.uniform(
                self.min_bound_gaussian_noise, self.max_bound_gaussian_noise)
            noise = torch.randn_like(combined) * sigma
            combined = combined + noise
        
        # 배치 다시 분리
        b = images[0].shape[0]
        images = [combined[:b], combined[b:]]
        
        # 개별 이미지마다 밝기 및 대비 변경
        if self.augment_contrast_individual and random.random() < 0.5:
            contrast_factor1 = random.uniform(
                self.min_bound_contrast_individual, 
                self.max_bound_contrast_individual)
            contrast_factor2 = random.uniform(
                self.min_bound_contrast_individual, 
                self.max_bound_contrast_individual)
            images[0] = TF.adjust_contrast(images[0], contrast_factor1)
            images[1] = TF.adjust_contrast(images[1], contrast_factor2)
        
        if self.augment_brightness_individual and random.random() < 0.5:
            brightness_factor1 = random.uniform(
                1.0 - self.max_delta_brightness_individual, 
                1.0 + self.max_delta_brightness_individual)
            brightness_factor2 = random.uniform(
                1.0 - self.max_delta_brightness_individual, 
                1.0 + self.max_delta_brightness_individual)
            images[0] = TF.adjust_brightness(images[0], brightness_factor1)
            images[1] = TF.adjust_brightness(images[1], brightness_factor2)
        
        # 값을 [0, 1] 범위로 클리핑
        images = [torch.clamp(img, 0.0, 1.0) for img in images]
        
        # 필요하면 배치 차원 제거
        if not is_batch:
            images = [img.squeeze(0) for img in images]
        
        return images


class GeometricAugmentation:
    """
    이미지 쌍에 대한 기하학적(geometric) 증강 적용
    광학 흐름과 호환되도록 구현
    """
    def __init__(self,
                 augment_flip_left_right=False,
                 augment_flip_up_down=False,
                 augment_rotation=False,
                 augment_relative_rotation=False,
                 max_rotation_deg=15,
                 max_relative_rotation_deg=3):
        """
        Args:
            augment_flip_left_right (bool): 좌우 반전 여부
            augment_flip_up_down (bool): 상하 반전 여부
            augment_rotation (bool): 회전 적용 여부 (두 이미지 동일 각도)
            augment_relative_rotation (bool): 두 번째 이미지에 상대적 회전 적용 여부
            max_rotation_deg (float): 최대 회전 각도(도)
            max_relative_rotation_deg (float): 두 번째 이미지에 적용할 최대 상대 회전 각도(도)
        """
        self.augment_flip_left_right = augment_flip_left_right
        self.augment_flip_up_down = augment_flip_up_down
        self.augment_rotation = augment_rotation
        self.augment_relative_rotation = augment_relative_rotation
        self.max_rotation_deg = max_rotation_deg
        self.max_relative_rotation_deg = max_relative_rotation_deg
    
    def __call__(self, images, flows=None):
        """
        이미지 쌍과 광학 흐름에 기하학적 증강 적용
        
        Args:
            images (List[torch.Tensor]): 이미지 쌍 [B, C, H, W] 또는 [C, H, W]
            flows (List[torch.Tensor], optional): 광학 흐름 쌍 [B, 2, H, W] 또는 [2, H, W]
            
        Returns:
            tuple: (증강된 이미지 쌍, 변환된 광학 흐름 쌍)
        """
        # 입력이 배치인지 단일 이미지인지 확인
        is_batch = images[0].dim() == 4
        device = images[0].device
        
        # 단일 이미지의 경우, 배치 차원 추가
        if not is_batch:
            images = [img.unsqueeze(0) for img in images]
            if flows is not None:
                flows = [flow.unsqueeze(0) if flow is not None else None for flow in flows]
        
        # 좌우 반전
        if self.augment_flip_left_right and random.random() < 0.5:
            images = [TF.hflip(img) for img in images]
            if flows is not None:
                # 광학 흐름의 x 방향 반전
                flows = [
                    torch.cat([-flow[:, 0:1], flow[:, 1:2]], dim=1) if flow is not None else None
                    for flow in flows
                ]
        
        # 상하 반전
        if self.augment_flip_up_down and random.random() < 0.5:
            images = [TF.vflip(img) for img in images]
            if flows is not None:
                # 광학 흐름의 y 방향 반전
                flows = [
                    torch.cat([flow[:, 0:1], -flow[:, 1:2]], dim=1) if flow is not None else None
                    for flow in flows
                ]
        
        # 회전 변환 (두 이미지 동일 각도)
        if self.augment_rotation and random.random() < 0.5:
            angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            
            # 회전 변환 행렬 계산
            b, _, h, w = images[0].shape
            center = (w / 2, h / 2)
            angle_rad = math.radians(angle)
            
            # 이미지 회전
            images = [TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                     for img in images]
            
            # 광학 흐름 회전
            if flows is not None:
                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                rotation_matrix = torch.tensor([
                    [cos_a, -sin_a],
                    [sin_a, cos_a]
                ], device=device)
                
                # 각 배치와 각 픽셀에 회전 변환 적용
                flows_rotated = []
                for flow in flows:
                    if flow is None:
                        flows_rotated.append(None)
                        continue
                    
                    # 흐름 벡터 회전
                    b, _, h, w = flow.shape
                    flow_reshaped = flow.permute(0, 2, 3, 1).reshape(b * h * w, 2)
                    flow_rotated = torch.matmul(flow_reshaped, rotation_matrix.T)
                    flow_result = flow_rotated.reshape(b, h, w, 2).permute(0, 3, 1, 2)
                    
                    flows_rotated.append(flow_result)
                
                flows = flows_rotated
        
        # 두 번째 이미지에 상대적 회전 적용
        if self.augment_relative_rotation and random.random() < 0.5 and len(images) > 1:
            rel_angle = random.uniform(-self.max_relative_rotation_deg, 
                                      self.max_relative_rotation_deg)
            
            # 두 번째 이미지만 회전
            images[1] = TF.rotate(images[1], rel_angle, 
                                interpolation=TF.InterpolationMode.BILINEAR)
            
            # 광학 흐름에 상대 회전 적용 (첫 번째 흐름만 해당)
            if flows is not None and flows[0] is not None:
                rel_angle_rad = math.radians(rel_angle)
                cos_a, sin_a = math.cos(rel_angle_rad), math.sin(rel_angle_rad)
                rel_rotation_matrix = torch.tensor([
                    [cos_a, -sin_a],
                    [sin_a, cos_a]
                ], device=device)
                
                # 첫 번째 흐름 벡터 회전
                b, _, h, w = flows[0].shape
                flow_reshaped = flows[0].permute(0, 2, 3, 1).reshape(b * h * w, 2)
                flow_rotated = torch.matmul(flow_reshaped, rel_rotation_matrix.T)
                flows[0] = flow_rotated.reshape(b, h, w, 2).permute(0, 3, 1, 2)
        
        # 필요하면 배치 차원 제거
        if not is_batch:
            images = [img.squeeze(0) for img in images]
            if flows is not None:
                flows = [flow.squeeze(0) if flow is not None else None for flow in flows]
        
        return images, flows


class FlowAugmentation:
    """
    광학 흐름을 위한 통합 증강 파이프라인
    """
    def __init__(self,
                 # 색상 변환(photometric) 증강 옵션
                 augment_color_swap=True,
                 augment_hue_shift=True,
                 augment_saturation=False,
                 augment_brightness=False,
                 augment_contrast=False,
                 augment_gaussian_noise=False,
                 augment_brightness_individual=False,
                 augment_contrast_individual=False,
                 max_delta_hue=0.5,
                 min_bound_saturation=0.8,
                 max_bound_saturation=1.2,
                 max_delta_brightness=0.1,
                 min_bound_contrast=0.8,
                 max_bound_contrast=1.2,
                 min_bound_gaussian_noise=0.0,
                 max_bound_gaussian_noise=0.02,
                 max_delta_brightness_individual=0.02,
                 min_bound_contrast_individual=0.95,
                 max_bound_contrast_individual=1.05,
                 
                 # 기하학적(geometric) 증강 옵션
                 augment_flip_left_right=False,
                 augment_flip_up_down=False,
                 augment_rotation=False,
                 augment_relative_rotation=False,
                 max_rotation_deg=15,
                 max_relative_rotation_deg=3):
        """
        Args:
            # 색상 변환(photometric) 증강 옵션
            augment_color_swap (bool): 색상 채널 순서 변경 여부
            augment_hue_shift (bool): Hue 값 변경 여부
            augment_saturation (bool): 채도 변경 여부
            augment_brightness (bool): 밝기 변경 여부 (두 이미지 동일하게)
            augment_contrast (bool): 대비 변경 여부 (두 이미지 동일하게)
            augment_gaussian_noise (bool): 가우시안 노이즈 추가 여부
            augment_brightness_individual (bool): 각 이미지마다 독립적으로 밝기 변경 여부
            augment_contrast_individual (bool): 각 이미지마다 독립적으로 대비 변경 여부
            max_delta_hue (float): Hue 변화의 최대 범위 [-max_delta, max_delta]
            min_bound_saturation (float): 채도 변화의 최소값
            max_bound_saturation (float): 채도 변화의 최대값
            max_delta_brightness (float): 밝기 변화의 최대 범위 [-max_delta, max_delta]
            min_bound_contrast (float): 대비 변화의 최소값
            max_bound_contrast (float): 대비 변화의 최대값
            min_bound_gaussian_noise (float): 가우시안 노이즈 표준편차의 최소값
            max_bound_gaussian_noise (float): 가우시안 노이즈 표준편차의 최대값
            max_delta_brightness_individual (float): 개별 이미지 밝기 변화 최대 범위
            min_bound_contrast_individual (float): 개별 이미지 대비 변화 최소값
            max_bound_contrast_individual (float): 개별 이미지 대비 변화 최대값
            
            # 기하학적(geometric) 증강 옵션
            augment_flip_left_right (bool): 좌우 반전 여부
            augment_flip_up_down (bool): 상하 반전 여부
            augment_rotation (bool): 회전 적용 여부 (두 이미지 동일 각도)
            augment_relative_rotation (bool): 두 번째 이미지에 상대적 회전 적용 여부
            max_rotation_deg (float): 최대 회전 각도(도)
            max_relative_rotation_deg (float): 두 번째 이미지에 적용할 최대 상대 회전 각도(도)
        """
        # 색상 변환(photometric) 증강
        self.photometric_aug = PhotometricAugmentation(
            augment_color_swap=augment_color_swap,
            augment_hue_shift=augment_hue_shift,
            augment_saturation=augment_saturation,
            augment_brightness=augment_brightness,
            augment_contrast=augment_contrast,
            augment_gaussian_noise=augment_gaussian_noise,
            augment_brightness_individual=augment_brightness_individual,
            augment_contrast_individual=augment_contrast_individual,
            max_delta_hue=max_delta_hue,
            min_bound_saturation=min_bound_saturation,
            max_bound_saturation=max_bound_saturation,
            max_delta_brightness=max_delta_brightness,
            min_bound_contrast=min_bound_contrast,
            max_bound_contrast=max_bound_contrast,
            min_bound_gaussian_noise=min_bound_gaussian_noise,
            max_bound_gaussian_noise=max_bound_gaussian_noise,
            max_delta_brightness_individual=max_delta_brightness_individual,
            min_bound_contrast_individual=min_bound_contrast_individual,
            max_bound_contrast_individual=max_bound_contrast_individual
        )
        
        # 기하학적(geometric) 증강
        self.geometric_aug = GeometricAugmentation(
            augment_flip_left_right=augment_flip_left_right,
            augment_flip_up_down=augment_flip_up_down,
            augment_rotation=augment_rotation,
            augment_relative_rotation=augment_relative_rotation,
            max_rotation_deg=max_rotation_deg,
            max_relative_rotation_deg=max_relative_rotation_deg
        )
    
    def __call__(self, images, flows=None):
        """
        이미지와 광학 흐름에 증강 적용
        
        Args:
            images (List[torch.Tensor]): 이미지 리스트 [B, C, H, W] 또는 [C, H, W]
            flows (List[torch.Tensor], optional): 광학 흐름 리스트 [B, 2, H, W] 또는 [2, H, W]
            
        Returns:
            tuple: (증강된 이미지 리스트, 변환된 광학 흐름 리스트)
        """
        # 기하학적 변환 적용 (이미지와 흐름 모두에 적용)
        images, flows = self.geometric_aug(images, flows)
        
        # 색상 변환 적용 (이미지에만 적용)
        images = self.photometric_aug(images)
        
        return images, flows


def apply_augmentation(images, flows=None, use_photometric=True, use_geometric=True):
    """
    이미지와 광학 흐름에 증강 적용
    
    Args:
        images (List[torch.Tensor]): 이미지 리스트 [B, C, H, W] 또는 [C, H, W]
        flows (List[torch.Tensor], optional): 광학 흐름 리스트 [B, 2, H, W] 또는 [2, H, W]
        use_photometric (bool): 색상 변환 증강 적용 여부
        use_geometric (bool): 기하학적 변환 증강 적용 여부
        
    Returns:
        tuple: (증강된 이미지 리스트, 변환된 광학 흐름 리스트)
    """
    # 입력 이미지 형태 기록
    original_shapes = [img.shape for img in images]
    original_dims = [img.dim() for img in images]
    
    aug = FlowAugmentation(
        # 기하학적 변환 옵션 (기본값 False)
        augment_flip_left_right=use_geometric,
        augment_flip_up_down=use_geometric,
        augment_rotation=use_geometric,
        
        # 색상 변환 옵션 (기본값 활성화)
        augment_color_swap=use_photometric,
        augment_hue_shift=use_photometric,
        augment_saturation=use_photometric,
        augment_brightness=use_photometric,
        augment_contrast=use_photometric,
        augment_gaussian_noise=use_photometric
    )
    
    # 증강 적용
    augmented_images, augmented_flows = aug(images, flows)
    
    # 형태 확인 및 수정 (원래 형태 유지)
    for i in range(min(len(augmented_images), len(original_dims))):
        if augmented_images[i].dim() != original_dims[i]:
            # 차원이 변경된 경우 처리
            if original_dims[i] == 3 and augmented_images[i].dim() == 4:
                augmented_images[i] = augmented_images[i].squeeze(0)  # 배치 차원 제거
            elif original_dims[i] == 4 and augmented_images[i].dim() == 3:
                augmented_images[i] = augmented_images[i].unsqueeze(0)  # 배치 차원 추가
    
    return augmented_images, augmented_flows


# 테스트 함수
def test_augmentation(height=196, width=256):
    """
    증강 파이프라인 테스트
    """
    # 테스트 이미지 생성
    img1 = torch.rand(3, height, width)
    img2 = torch.rand(3, height, width)
    
    # 테스트 광학 흐름 생성
    flow = torch.zeros(2, height, width)
    for y in range(height):
        for x in range(width):
            flow[0, y, x] = (x - width/2) / 10  # x 방향 흐름
            flow[1, y, x] = (y - height/2) / 10  # y 방향 흐름
    
    # 증강 적용
    images_aug, flows_aug = apply_augmentation([img1, img2], [flow, None])
    
    print(f"원본 이미지 1 크기: {img1.shape}")
    print(f"원본 이미지 2 크기: {img2.shape}")
    print(f"원본 흐름 크기: {flow.shape}")
    print(f"증강된 이미지 1 크기: {images_aug[0].shape}")
    print(f"증강된 이미지 2 크기: {images_aug[1].shape}")
    print(f"증강된 흐름 크기: {flows_aug[0].shape if flows_aug[0] is not None else None}")
    
    return images_aug, flows_aug


if __name__ == "__main__":
    images_aug, flows_aug = test_augmentation()
    print("증강 테스트 완료!") 