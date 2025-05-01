import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import argparse

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
    def __init__(self, file_list_path=None, data_dir=None, target_height=196, target_width=256, 
                convert_to_rgb=True, use_augmentation=False, use_photometric=True, use_geometric=False,
                exclude_ev_minus=True, apply_pregamma=False, pregamma_value=2.0):
        """
        Args:
            file_list_path (str): 파일 경로 목록이 저장된 텍스트 파일
            data_dir (str, optional): .npy 파일이 있는 디렉토리 (file_list_path가 None인 경우)
            target_height (int): 다운샘플링 후 이미지 높이
            target_width (int): 다운샘플링 후 이미지 너비
            convert_to_rgb (bool): Bayer RAW를 RGB로 변환할지 여부
            use_augmentation (bool): 데이터 증강 사용 여부
            use_photometric (bool): 색상 변환 증강 적용 여부
            use_geometric (bool): 기하학적 변환 증강 적용 여부
            exclude_ev_minus (bool): 'ev minus' 프레임(인덱스 1, 2, 3) 제외 여부
            apply_pregamma (bool): 어두운 이미지 보정을 위한 pre-gamma 적용 여부
            pregamma_value (float): gamma 보정 값 (클수록 어두운 영역이 밝아짐)
        """
        self.target_height = target_height
        self.target_width = target_width
        self.convert_to_rgb = convert_to_rgb
        self.exclude_ev_minus = exclude_ev_minus
        self.apply_pregamma = apply_pregamma
        self.pregamma_value = pregamma_value
        
        # 데이터 증강 옵션
        self.use_augmentation = use_augmentation and AUGMENTATION_AVAILABLE
        self.use_photometric = use_photometric
        self.use_geometric = use_geometric
        
        # 파일 목록 로드 방식 선택
        self.file_list = []
        
        if file_list_path is not None:
            # 텍스트 파일에서 경로 목록 읽기
            with open(file_list_path, 'r') as f:
                lines = f.readlines()
                self.file_list = [Path(line.strip()) for line in lines if line.strip()]
            
            print(f"Loaded {len(self.file_list)} file paths from '{file_list_path}'")
        elif data_dir is not None:
            # 디렉토리에서 npy 파일 찾기
            self.data_dir = Path(data_dir)
            self.file_list = sorted([f for f in self.data_dir.glob('*.npy')])
            print(f"Found {len(self.file_list)} .npy files in '{data_dir}'")
        else:
            raise ValueError("file_list_path 또는 data_dir 중 하나를 제공해야 합니다.")
        
        if self.apply_pregamma:
            print(f"Pre-gamma 보정 활성화됨 (gamma = {self.pregamma_value})")
        
        # 파일 별로 seq len 다를 수 있음
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 파일 로드
        file_path = self.file_list[idx]
        
        try:
            multi_frames = np.load(str(file_path))
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            # 로드 실패 시 다른 인덱스 시도
            idx = (idx + 1) % len(self.file_list)
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
        frame1 = torch.from_numpy(frame1.astype(np.float32) / 4095.0)
        frame2 = torch.from_numpy(frame2.astype(np.float32) / 4095.0)
        
        # pre-gamma 보정 적용 (어두운 이미지를 더 밝게 변환)
        if self.apply_pregamma:
            # I' = I^(1/gamma), gamma > 1이면 어두운 영역이 더 밝아짐
            frame1 = torch.pow(frame1, 1.0 / self.pregamma_value)
            frame2 = torch.pow(frame2, 1.0 / self.pregamma_value)
        
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


def create_dataloader(file_list_path=None, data_dir=None, batch_size=32, shuffle=True, num_workers=4, 
                      target_height=196, target_width=256, convert_to_rgb=True,
                      use_augmentation=False, use_photometric=True, use_geometric=False,
                      exclude_ev_minus=True, apply_pregamma=False, pregamma_value=2.0):
    """
    MultiFrameRawDataset에 대한 데이터로더 생성
    
    Args:
        file_list_path (str, optional): 파일 경로 목록이 저장된 텍스트 파일
        data_dir (str, optional): .npy 파일이 있는 디렉토리
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
        apply_pregamma (bool): 어두운 이미지 보정을 위한 pre-gamma 적용 여부
        pregamma_value (float): gamma 보정 값 (기본값: 2.0)
        
    Returns:
        torch.utils.data.DataLoader: 데이터로더
    """
    # 데이터셋 생성
    dataset = MultiFrameRawDataset(
        file_list_path=file_list_path,
        data_dir=data_dir,
        target_height=target_height,
        target_width=target_width,
        convert_to_rgb=convert_to_rgb,
        use_augmentation=use_augmentation,
        use_photometric=use_photometric,
        use_geometric=use_geometric,
        exclude_ev_minus=exclude_ev_minus,
        apply_pregamma=apply_pregamma,
        pregamma_value=pregamma_value
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

def test_dataloader(file_list_path=None, data_dir=None, batch_size=4, use_augmentation=False, exclude_ev_minus=True,
                   apply_pregamma=False, pregamma_value=2.0):
    """
    데이터로더 테스트 및 시각화
    
    Args:
        file_list_path (str, optional): 파일 경로 목록이 저장된 텍스트 파일
        data_dir (str, optional): .npy 파일이 있는 디렉토리
        batch_size (int): 배치 크기
        use_augmentation (bool): 데이터 증강 사용 여부
        exclude_ev_minus (bool): 'ev minus' 프레임(인덱스 1, 2, 3) 제외 여부
        apply_pregamma (bool): pre-gamma 보정 적용 여부
        pregamma_value (float): gamma 보정 값 (기본값: 2.0)
    """
    # 데이터 소스 확인
    if file_list_path is None and data_dir is None:
        raise ValueError("file_list_path 또는 data_dir 중 하나를 제공해야 합니다.")
    
    # 사용하는 데이터 소스 출력
    if file_list_path is not None:
        print(f"파일 목록을 사용: {file_list_path}")
    else:
        print(f"디렉토리를 사용: {data_dir}")
    
    # 데이터로더 생성
    dataloader = create_dataloader(
        file_list_path=file_list_path,
        data_dir=data_dir, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        convert_to_rgb=True,
        use_augmentation=use_augmentation,
        use_photometric=True,      # 색상 변환 증강 활성화
        use_geometric=False,       # 기하학적 변환 비활성화 (크기 고정이므로)
        exclude_ev_minus=exclude_ev_minus,
        apply_pregamma=apply_pregamma,
        pregamma_value=pregamma_value
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
    print(f"Frame indices: {frame_idx1}, {frame_idx2}")
    
    # matplotlib 시각화
    try:
        import matplotlib.pyplot as plt
        
        # 배치의 첫 번째 이미지만 시각화
        img1 = frame1[0].permute(1, 2, 0).cpu().numpy()
        img2 = frame2[0].permute(1, 2, 0).cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(img1)
        axes[0].set_title(f'Frame 1 (idx: {frame_idx1[0]})')
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].set_title(f'Frame 2 (idx: {frame_idx2[0]})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")
        
    return dataloader, batch

# 메인 실행 코드
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='데이터로더 테스트')
    parser.add_argument('--data_dir', type=str, default=None, help='데이터 디렉토리 경로')
    parser.add_argument('--file_list', type=str, default=None, help='파일 경로 목록 텍스트 파일')
    parser.add_argument('--batch_size', type=int, default=4, help='배치 크기')
    parser.add_argument('--augmentation', action='store_true', help='데이터 증강 사용')
    parser.add_argument('--exclude_ev_minus', action='store_true', default=True, help='EV Minus 프레임 제외')
    parser.add_argument('--apply_pregamma', action='store_true', help='pre-gamma 보정 적용')
    parser.add_argument('--pregamma_value', type=float, default=2.0, help='gamma 보정 값')
    
    args = parser.parse_args()
    
    # 데이터 소스 확인
    if args.data_dir is None and args.file_list is None:
        # 기본값 설정
        args.data_dir = "C:\\Users\\under\\Documents\\output_arrays"
        print(f"데이터 소스가 지정되지 않아 기본 디렉토리를 사용합니다: {args.data_dir}")
    
    # 일반 데이터로더 테스트
    print("\n=== 기본 데이터로더 테스트 ===")
    test_dataloader(
        file_list_path=args.file_list,
        data_dir=args.data_dir, 
        batch_size=args.batch_size,
        use_augmentation=args.augmentation,
        exclude_ev_minus=args.exclude_ev_minus,
        apply_pregamma=args.apply_pregamma,
        pregamma_value=args.pregamma_value
    )
    
    # 추가 테스트 (사용자가 원할 경우 활성화)
    """
    # 증강이 적용된 데이터로더 테스트
    print("\n=== 증강이 적용된 데이터로더 테스트 ===")
    test_dataloader(
        file_list_path=args.file_list,
        data_dir=args.data_dir, 
        use_augmentation=True
    )
    
    # pre-gamma 보정이 적용된 데이터로더 테스트
    print("\n=== pre-gamma 보정이 적용된 데이터로더 테스트 (gamma=2.0) ===")
    test_dataloader(
        file_list_path=args.file_list,
        data_dir=args.data_dir, 
        use_augmentation=False, 
        apply_pregamma=True, 
        pregamma_value=2.0
    )
    """