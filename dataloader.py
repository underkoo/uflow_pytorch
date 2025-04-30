import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

try:
    # 새로 구현한 augmentation 모듈 임포트
    import augmentation as aug
    AUGMENTATION_AVAILABLE = True
except ImportError:
    print("Warning: augmentation 모듈을 로드할 수 없습니다. 데이터 증강이 비활성화됩니다.")
    AUGMENTATION_AVAILABLE = False


def preprocess_raw_bayer(raw_image, target_height=196, target_width=256):
    """
    Bayer RAW 이미지를 RGB로 변환하는 함수
    
    Args:
        raw_image (torch.Tensor): Bayer 패턴의 RAW 이미지 [B, 1, H, W]
        target_height (int): 출력 이미지의 높이 (기본값: 196)
        target_width (int): 출력 이미지의 너비 (기본값: 256)
        
    Returns:
        torch.Tensor: RGB 이미지 [B, 3, target_height, target_width]
    """
    # print(raw_image.shape)
    # Space2Depth: Bayer RAW를 4채널로 변환
    b, c, h, w = raw_image.shape
    bayer_channels = F.pixel_unshuffle(raw_image, 2)
    
    # GRBG 패턴에서 RGB 추출
    # GRBG 배열은 일반적으로:
    # G R
    # B G
    G1 = bayer_channels[:, 0]  # G 채널 (좌상단)
    R = bayer_channels[:, 1]  # R 채널 (우상단)
    B = bayer_channels[:, 2]  # B 채널 (좌하단)
    G2 = bayer_channels[:, 3]   # G 채널 (우하단)
    
    # G 채널은 두 개의 G 값 평균
    G = (G1 + G2) / 2
    
    # RGB 채널 구성 [B, 3, H/2, W/2]
    rgb_image = torch.stack([R, G, B], dim=1)
    
    # 목표 크기로 다운샘플링
    rgb_image = F.interpolate(
        rgb_image, 
        size=(target_height, target_width), 
        mode='bilinear', 
        align_corners=False
    )
    
    
    return rgb_image


class MultiFrameRawDataset(Dataset):
    """
    Dataset for loading multi-frame bayer raw data stored as .npy files
    랜덤하게 두 프레임을 선택하는 방식
    """
    def __init__(self, data_dir, target_height=196, target_width=256, 
                convert_to_rgb=True, use_augmentation=False, use_photometric=True, use_geometric=False,
                exclude_ev_minus=True):
        """
        Args:
            data_dir (str): .npy 파일이 있는 디렉토리
            transform (callable, optional): 각 프레임에 적용할 변환 함수
            target_height (int): 다운샘플링 후 이미지 높이
            target_width (int): 다운샘플링 후 이미지 너비
            convert_to_rgb (bool): Bayer RAW를 RGB로 변환할지 여부
            use_augmentation (bool): 데이터 증강 사용 여부
            use_photometric (bool): 색상 변환 증강 적용 여부
            use_geometric (bool): 기하학적 변환 증강 적용 여부
            exclude_ev_minus (bool): 'ev minus' 프레임(인덱스 1, 2, 3) 제외 여부
        """
        self.data_dir = Path(data_dir)
        self.file_list = sorted([f for f in self.data_dir.glob('*.npy')])
        self.target_height = target_height
        self.target_width = target_width
        self.convert_to_rgb = convert_to_rgb
        self.exclude_ev_minus = exclude_ev_minus
        
        # 데이터 증강 옵션
        self.use_augmentation = use_augmentation and AUGMENTATION_AVAILABLE
        self.use_photometric = use_photometric
        self.use_geometric = use_geometric
        
        # 파일 목록 확인
        print(f"Found {len(self.file_list)} .npy files in '{data_dir}'")
        
        # 파일 별로 seq len 다를 수 있음
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 파일 로드
        file_path = self.file_list[idx]
        multi_frames = np.load(str(file_path))
        
        num_frames = multi_frames.shape[0]
        
        # 'ev minus' 프레임 제외 설정
        available_indices = list(range(num_frames))
        
        # 10개 이상의 프레임이 있고, exclude_ev_minus가 활성화된 경우 1, 2, 3번 인덱스 제외
        if self.exclude_ev_minus and num_frames >= 10:
            ev_minus_indices = [1, 2, 3]  # 'ev minus' 프레임 인덱스
            available_indices = [i for i in available_indices if i not in ev_minus_indices]
        
        # 사용 가능한 인덱스에서 랜덤하게 2개 선택
        frame_idx1, frame_idx2 = random.sample(available_indices, 2)
        
        # 선택된 프레임 가져오기
        frame1 = multi_frames[frame_idx1]
        frame2 = multi_frames[frame_idx2]
        
        # Convert to torch tensors and normalize to [0, 1]
        frame1 = torch.from_numpy(frame1).float() / 65535.0
        frame2 = torch.from_numpy(frame2).float() / 65535.0
        
        # add channel dimension
        frame1 = frame1.unsqueeze(0)
        frame2 = frame2.unsqueeze(0)
        
        frames = torch.stack([frame1, frame2], dim=0)

        frames = preprocess_raw_bayer(frames, self.target_height, self.target_width)

        [frame1, frame2] = frames
        
        # 데이터 증강 적용
        if self.use_augmentation:
            # 증강 적용 (광학 흐름은 아직 없으므로 None 전달)
            [frame1, frame2], _ = aug.apply_augmentation(
                [frame1, frame2], 
                flows=None, 
                use_photometric=self.use_photometric, 
                use_geometric=self.use_geometric
            )
        
        # 프레임 인덱스도 함께 반환 (디버깅/시각화 목적)
        return {
            'frame1': frame1, 
            'frame2': frame2, 
            'file_idx': idx,
            'frame_idx1': frame_idx1, 
            'frame_idx2': frame_idx2
        }


def create_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=4, 
                      target_height=196, target_width=256, convert_to_rgb=True,
                      use_augmentation=False, use_photometric=True, use_geometric=False,
                      exclude_ev_minus=True):
    """
    MultiFrameRawDataset에 대한 데이터로더 생성
    
    Args:
        data_dir (str): .npy 파일이 있는 디렉토리
        batch_size (int): 배치 크기
        shuffle (bool): 데이터 셔플 여부
        num_workers (int): 데이터 로딩에 사용할 워커 수
        target_height (int): 다운샘플링 후 이미지 높이
        target_width (int): 다운샘플링 후 이미지 너비
        convert_to_rgb (bool): Bayer RAW를 RGB로 변환할지 여부
        use_augmentation (bool): 데이터 증강 사용 여부
        use_photometric (bool): 색상 변환 증강 적용 여부
        use_geometric (bool): 기하학적 변환 증강 적용 여부
        exclude_ev_minus (bool): 'ev minus' 프레임(인덱스 1, 2, 3) 제외 여부
        
    Returns:
        torch.utils.data.DataLoader: 데이터로더
    """
    # 데이터셋 생성
    dataset = MultiFrameRawDataset(
        data_dir=data_dir,
        target_height=target_height,
        target_width=target_width,
        convert_to_rgb=convert_to_rgb,
        use_augmentation=use_augmentation,
        use_photometric=use_photometric,
        use_geometric=use_geometric,
        exclude_ev_minus=exclude_ev_minus
    )
    
    # 데이터로더 생성
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def test_dataloader(data_dir, batch_size=4, use_augmentation=False, exclude_ev_minus=True):
    """
    데이터로더 테스트 및 시각화
    
    Args:
        data_dir (str): .npy 파일이 있는 디렉토리
        batch_size (int): 배치 크기
        use_augmentation (bool): 데이터 증강 사용 여부
        exclude_ev_minus (bool): 'ev minus' 프레임(인덱스 1, 2, 3) 제외 여부
    """
    # 데이터로더 생성
    dataloader = create_dataloader(
        data_dir, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        convert_to_rgb=True,
        use_augmentation=use_augmentation,
        use_photometric=True,      # 색상 변환 증강 활성화
        use_geometric=False,       # 기하학적 변환 비활성화 (크기 고정이므로)
        exclude_ev_minus=exclude_ev_minus
    )
    
    # 데이터 로드
    print(f"Loading batch from dataloader...")
    batch = next(iter(dataloader))
    
    # 배치 정보 출력
    frame1 = batch['frame1']
    frame2 = batch['frame2']
    frame_idx1 = batch['frame_idx1']
    frame_idx2 = batch['frame_idx2']
    
    print(f"Batch size: {frame1.shape}")
    print(f"Frame 1 indices: {frame_idx1}")
    print(f"Frame 2 indices: {frame_idx2}")
    
    # 이미지 시각화
    plt.figure(figsize=(10, 8))
    for i in range(min(batch_size, 4)):  # 최대 4개까지 표시
        # 프레임 1
        plt.subplot(4, 2, i*2+1)
        img = frame1[i].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title(f"Frame 1 (idx {frame_idx1[i]})")
        plt.axis('off')
        
        # 프레임 2
        plt.subplot(4, 2, i*2+2)
        img = frame2[i].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title(f"Frame 2 (idx {frame_idx2[i]})")
        plt.axis('off')
    
    aug_status = "enabled" if use_augmentation else "disabled"
    plt.suptitle(f"Data Samples (Augmentation: {aug_status})")
    plt.tight_layout()
    plt.savefig("dataloader_test.png")
    plt.show()

# 메인 실행 코드
if __name__ == "__main__":
    # 테스트할 데이터 디렉토리
    data_dir = "C:\\Users\\under\\Documents\\output_arrays"
    
    # 일반 데이터로더 테스트
    print("\n=== Testing standard dataloader (no augmentation) ===")
    test_dataloader(data_dir, use_augmentation=False)
    
    # 증강이 적용된 데이터로더 테스트
    print("\n=== Testing standard dataloader (with augmentation) ===")
    test_dataloader(data_dir, use_augmentation=True)