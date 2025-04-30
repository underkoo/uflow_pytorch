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
        raw_image (torch.Tensor): Bayer 패턴의 RAW 이미지 [B, H, W] 또는 [H, W]
        target_height (int): 출력 이미지의 높이 (기본값: 196)
        target_width (int): 출력 이미지의 너비 (기본값: 256)
        
    Returns:
        torch.Tensor: RGB 이미지 [B, 3, target_height, target_width] 또는 [3, target_height, target_width]
    """
    # 입력이 배치인지 단일 이미지인지 확인
    is_batch = raw_image.dim() == 3
    
    if not is_batch:
        # 단일 이미지를 배치로 변환 [H, W] -> [1, H, W]
        raw_image = raw_image.unsqueeze(0)
    
    # Space2Depth: Bayer RAW를 4채널로 변환
    b, h, w = raw_image.shape
    if h % 2 != 0 or w % 2 != 0:
        # 높이와 너비가 짝수가 아니면 패딩 추가
        pad_h = (h % 2 != 0)
        pad_w = (w % 2 != 0)
        raw_image = F.pad(raw_image, (0, pad_w, 0, pad_h))
        h, w = raw_image.shape[1], raw_image.shape[2]
    
    # [B, H, W] -> [B, H/2, 2, W/2, 2]
    reshaped = raw_image.reshape(b, h//2, 2, w//2, 2)
    # [B, H/2, 2, W/2, 2] -> [B, H/2, W/2, 2, 2] -> [B, H/2, W/2, 4]
    bayer_channels = reshaped.permute(0, 1, 3, 2, 4).reshape(b, h//2, w//2, 4)
    # [B, H/2, W/2, 4] -> [B, 4, H/2, W/2]
    bayer_channels = bayer_channels.permute(0, 3, 1, 2)
    
    # RGGB 패턴에서 RGB 추출
    # RGGB 배열은 일반적으로:
    # R G
    # G B
    R = bayer_channels[:, 0]  # R 채널 (좌상단)
    G1 = bayer_channels[:, 1]  # G 채널 (우상단)
    G2 = bayer_channels[:, 2]  # G 채널 (좌하단)
    B = bayer_channels[:, 3]   # B 채널 (우하단)
    
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
    
    # 배치가 아닌 경우 배치 차원 제거
    if not is_batch:
        rgb_image = rgb_image.squeeze(0)
    
    return rgb_image


class MultiFrameRawDataset(Dataset):
    """
    Dataset for loading multi-frame bayer raw data stored as .npy files
    랜덤하게 두 프레임을 선택하는 방식
    """
    def __init__(self, data_dir, transform=None, target_height=196, target_width=256, 
                convert_to_rgb=True, use_augmentation=False, use_photometric=True, use_geometric=False):
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
        """
        self.data_dir = Path(data_dir)
        self.file_list = sorted([f for f in self.data_dir.glob('*.npy')])
        self.transform = transform
        self.target_height = target_height
        self.target_width = target_width
        self.convert_to_rgb = convert_to_rgb
        
        # 데이터 증강 옵션
        self.use_augmentation = use_augmentation and AUGMENTATION_AVAILABLE
        self.use_photometric = use_photometric
        self.use_geometric = use_geometric
        
        # 파일 목록 확인
        print(f"Found {len(self.file_list)} .npy files in '{data_dir}'")
        
        # 첫 번째 파일로 테스트하여 프레임 개수 확인
        if len(self.file_list) > 0:
            test_data = np.load(str(self.file_list[0]))
            print(f"First file loaded: {test_data.shape}")
            print(f"Each npy file contains {test_data.shape[0]} frames")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 파일 로드
        file_path = self.file_list[idx]
        multi_frames = np.load(str(file_path))
        
        # 확인: 차원이 3차원(batch,height,width)인지 4차원(batch,height,width,channel)인지
        has_channel = len(multi_frames.shape) == 4
        
        num_frames = multi_frames.shape[0]
        
        # 두 개의 서로 다른 프레임 인덱스 무작위 선택
        if num_frames < 2:
            # 만약 프레임이 1개뿐이라면 같은 프레임을 사용 (예외 처리)
            frame_idx1, frame_idx2 = 0, 0
        else:
            frame_idx1, frame_idx2 = random.sample(range(num_frames), 2)
        
        # 선택된 프레임 가져오기
        frame1 = multi_frames[frame_idx1]
        frame2 = multi_frames[frame_idx2]
        
        # Convert to torch tensors and normalize to [0, 1]
        frame1 = torch.from_numpy(frame1).float() / 65535.0
        frame2 = torch.from_numpy(frame2).float() / 65535.0
        
        # Bayer RAW를 RGB로 변환 및 다운샘플링 (사용자 지정 크기로)
        if not has_channel and self.convert_to_rgb:
            frame1 = preprocess_raw_bayer(frame1, self.target_height, self.target_width)
            frame2 = preprocess_raw_bayer(frame2, self.target_height, self.target_width)
        else:
            # 기존 처리 방식으로 계속
            if has_channel:
                # 채널이 있는 경우 (B,H,W,C): 채널 차원을 앞으로 이동
                frame1 = frame1.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
                frame2 = frame2.permute(2, 0, 1)
            else:
                # 채널이 없는 경우 (B,H,W): 채널 차원 추가 (그레이스케일 이미지로 처리)
                frame1 = frame1.unsqueeze(0)  # (H,W) -> (1,H,W)
                frame2 = frame2.unsqueeze(0)
                
            # 크기 조정 (다운샘플링)
            frame1 = F.interpolate(
                frame1.unsqueeze(0), 
                size=(self.target_height, self.target_width), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            frame2 = F.interpolate(
                frame2.unsqueeze(0), 
                size=(self.target_height, self.target_width), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Transform 적용
        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
        
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


class SequentialFrameDataset(Dataset):
    """
    시퀀스 기반 손실 계산을 위한 연속 프레임 데이터셋
    연속된 n개의 프레임을 선택하는 방식
    """
    def __init__(self, data_dir, seq_len=3, stride=1, transform=None, normalize=True, 
                 target_height=196, target_width=256, convert_to_rgb=True,
                 use_augmentation=False, use_photometric=True, use_geometric=False):
        """
        Args:
            data_dir (str): .npy 파일이 있는 디렉토리
            seq_len (int): 시퀀스 길이 (연속 프레임 수, 기본값: 3)
            stride (int): 시퀀스 샘플링 스트라이드 (기본값: 1)
            transform (callable, optional): 각 프레임에 적용할 변환 함수
            normalize (bool): 이미지 값을 [0,1] 범위로 정규화할지 여부
            target_height (int): 다운샘플링 후 이미지 높이
            target_width (int): 다운샘플링 후 이미지 너비
            convert_to_rgb (bool): Bayer RAW를 RGB로 변환할지 여부
            use_augmentation (bool): 데이터 증강 사용 여부
            use_photometric (bool): 색상 변환 증강 적용 여부
            use_geometric (bool): 기하학적 변환 증강 적용 여부
        """
        self.data_dir = Path(data_dir)
        self.file_list = sorted([f for f in self.data_dir.glob('*.npy')])
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.normalize = normalize
        self.target_height = target_height
        self.target_width = target_width
        self.convert_to_rgb = convert_to_rgb
        
        # 데이터 증강 옵션
        self.use_augmentation = use_augmentation and AUGMENTATION_AVAILABLE
        self.use_photometric = use_photometric
        self.use_geometric = use_geometric
        
        # 첫 번째 파일로 데이터 형식 확인
        self.has_channel = False
        if len(self.file_list) > 0:
            try:
                test_data = np.load(str(self.file_list[0]))
                # 차원이 4차원이면 채널 있음 (batch, height, width, channel)
                self.has_channel = len(test_data.shape) == 4
                print(f"Data format: {'With channels (B,H,W,C)' if self.has_channel else 'Without channels (B,H,W)'}")
            except Exception as e:
                print(f"Error loading first file: {e}")
        
        # 각 파일에서 가능한 시퀀스의 시작 위치 목록 생성
        self.sequences = []
        
        for file_idx, file_path in enumerate(self.file_list):
            # 파일 로드
            try:
                multi_frames = np.load(str(file_path))
                num_frames = multi_frames.shape[0]
                
                # 가능한 시퀀스 시작점 계산
                for start_idx in range(0, num_frames - seq_len + 1, stride):
                    self.sequences.append({
                        'file_idx': file_idx,
                        'start_idx': start_idx
                    })
            except Exception as e:
                print(f"Error loading file '{file_path}': {e}")
        
        print(f"Total {len(self.sequences)} sequences generated.")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # 시퀀스 정보 가져오기
        seq_info = self.sequences[idx]
        file_idx = seq_info['file_idx']
        start_idx = seq_info['start_idx']
        file_path = self.file_list[file_idx]
        
        # 파일 로드
        multi_frames = np.load(str(file_path))
        
        # 연속된 프레임 선택
        frames = []
        frame_indices = []
        
        for i in range(self.seq_len):
            frame_idx = start_idx + i
            frame = multi_frames[frame_idx]
            
            # 텐서로 변환 및 정규화
            frame = torch.from_numpy(frame).float()
            if self.normalize:
                # 최대값 확인 (8비트 또는 16비트)
                max_val = 65535.0
                frame = frame / max_val
            
            # Bayer RAW를 RGB로 변환 및 다운샘플링
            if not self.has_channel and self.convert_to_rgb:
                frame = preprocess_raw_bayer(frame, self.target_height, self.target_width)
            else:
                # 기존 처리 방식으로 계속
                if self.has_channel:
                    # 채널이 있는 경우 (B,H,W,C): 채널 차원을 앞으로 이동
                    frame = frame.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
                else:
                    # 채널이 없는 경우 (B,H,W): 채널 차원 추가 (그레이스케일 이미지로 처리)
                    frame = frame.unsqueeze(0)  # (H,W) -> (1,H,W)
                
                # 크기 조정 (다운샘플링)
                frame = F.interpolate(
                    frame.unsqueeze(0), 
                    size=(self.target_height, self.target_width), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # 변환 적용
            if self.transform:
                frame = self.transform(frame)
            
            frames.append(frame)
            frame_indices.append(frame_idx)
        
        # 데이터 증강 적용 - 각 프레임을 개별적으로 처리
        if self.use_augmentation:
            augmented_frames = []
            for frame in frames:
                # 각 프레임을 하나씩 증강 (증강 시 리스트로 전달하고 첫 번째 결과만 사용)
                aug_frame, _ = aug.apply_augmentation(
                    [frame], 
                    flows=None, 
                    use_photometric=self.use_photometric, 
                    use_geometric=self.use_geometric
                )
                augmented_frames.append(aug_frame[0])  # 첫 번째 결과만 사용
            frames = augmented_frames
        
        # 프레임 리스트를 단일 텐서로 변환 [seq_len, C, H, W]
        frames_tensor = torch.stack(frames, dim=0)
        
        # 반환
        return {
            'frames': frames_tensor,
            'file_idx': seq_info['file_idx'],
            'start_idx': seq_info['start_idx'],
            'frame_indices': torch.tensor(frame_indices)  # 텐서로 변환하여 배치 처리 가능하게 함
        }


def sequential_collate_fn(batch):
    """
    시퀀셜 데이터셋의 배치를 처리하는 collate 함수
    
    Args:
        batch (list): DataLoader가 반환한 배치 항목 리스트
        
    Returns:
        dict: 배치 처리된 데이터
    """
    # 각 항목 초기화
    frames_list = []
    file_indices = []
    start_indices = []
    frame_indices = []
    
    # 배치의 각 항목 처리
    for item in batch:
        frames_list.append(item['frames'])
        file_indices.append(item['file_idx'])
        start_indices.append(item['start_idx'])
        frame_indices.append(item['frame_indices'])
    
    # 텐서로 변환
    # frames_list는 이미 각 항목이 [seq_len, C, H, W] 형태의 텐서
    frames_batch = torch.stack(frames_list, dim=0)  # [B, seq_len, C, H, W]
    frame_indices_batch = torch.stack(frame_indices, dim=0)  # [B, seq_len]
    
    # 배치 데이터 반환
    return {
        'frames': frames_batch,
        'file_idx': torch.tensor(file_indices),
        'start_idx': torch.tensor(start_indices),
        'frame_indices': frame_indices_batch
    }


def create_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=4, 
                      target_height=196, target_width=256, convert_to_rgb=True,
                      use_augmentation=False, use_photometric=True, use_geometric=False):
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
        
    Returns:
        torch.utils.data.DataLoader: 데이터로더
    """
    # 데이터셋 생성
    dataset = MultiFrameRawDataset(
        data_dir=data_dir,
        transform=None,  # 기본 transform은 사용하지 않음
        target_height=target_height,
        target_width=target_width,
        convert_to_rgb=convert_to_rgb,
        use_augmentation=use_augmentation,
        use_photometric=use_photometric,
        use_geometric=use_geometric
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


def create_sequential_dataloader(data_dir, seq_len=3, batch_size=32, shuffle=True, num_workers=4, stride=1,
                               target_height=196, target_width=256, convert_to_rgb=True,
                               use_augmentation=False, use_photometric=True, use_geometric=False):
    """
    SequentialFrameDataset에 대한 데이터로더 생성
    
    Args:
        data_dir (str): .npy 파일이 있는 디렉토리
        seq_len (int): 시퀀스 길이 (연속 프레임 수)
        batch_size (int): 배치 크기
        shuffle (bool): 데이터 셔플 여부
        num_workers (int): 데이터 로딩에 사용할 워커 수
        stride (int): 시퀀스 샘플링 스트라이드
        target_height (int): 다운샘플링 후 이미지 높이
        target_width (int): 다운샘플링 후 이미지 너비
        convert_to_rgb (bool): Bayer RAW를 RGB로 변환할지 여부
        use_augmentation (bool): 데이터 증강 사용 여부
        use_photometric (bool): 색상 변환 증강 적용 여부
        use_geometric (bool): 기하학적 변환 증강 적용 여부
        
    Returns:
        torch.utils.data.DataLoader: 데이터로더
    """
    # 데이터셋 생성
    dataset = SequentialFrameDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        stride=stride,
        transform=None,  # 기본 transform은 사용하지 않음
        normalize=True,
        target_height=target_height,
        target_width=target_width,
        convert_to_rgb=convert_to_rgb,
        use_augmentation=use_augmentation,
        use_photometric=use_photometric,
        use_geometric=use_geometric
    )
    
    # 데이터로더 생성
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=sequential_collate_fn  # 사용자 정의 collate 함수 사용
    )
    
    return dataloader


def test_dataloader(data_dir, batch_size=4, use_augmentation=False):
    """
    데이터로더 테스트 및 시각화
    
    Args:
        data_dir (str): .npy 파일이 있는 디렉토리
        batch_size (int): 배치 크기
        use_augmentation (bool): 데이터 증강 사용 여부
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
        use_geometric=False        # 기하학적 변환 비활성화 (크기 고정이므로)
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


def test_sequential_dataloader(data_dir, seq_len=3, batch_size=2, use_augmentation=False):
    """
    시퀀셜 데이터로더 테스트 및 시각화
    
    Args:
        data_dir (str): .npy 파일이 있는 디렉토리
        seq_len (int): 시퀀스 길이
        batch_size (int): 배치 크기
        use_augmentation (bool): 데이터 증강 사용 여부
    """
    # 데이터로더 생성
    dataloader = create_sequential_dataloader(
        data_dir, 
        seq_len=seq_len, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        convert_to_rgb=True,
        use_augmentation=use_augmentation,
        use_photometric=True,      # 색상 변환 증강 활성화
        use_geometric=False        # 기하학적 변환 비활성화 (크기 고정이므로)
    )
    
    # 데이터 로드
    print(f"Loading batch from sequential dataloader...")
    batch = next(iter(dataloader))
    
    # 배치 정보 출력
    frames = batch['frames']  # [batch_size, seq_len, C, H, W]
    frame_indices = batch['frame_indices']  # [batch_size, seq_len]
    
    # frames 구조 확인
    print(f"Frames type: {type(frames)}")
    print(f"Frames tensor shape: {frames.shape}")
    print(f"Frame indices shape: {frame_indices.shape}")
    print(f"Frame indices: {frame_indices}")
    
    # 배치의 시퀀스 시각화 (최대 2개의 배치만 표시)
    plt.figure(figsize=(12, 4 * min(batch_size, 2)))
    
    # 각 배치와 시퀀스에 대해 시각화
    for b in range(min(batch_size, 2)):  # 최대 2개 배치만 표시
        for s in range(frames.size(1)):  # 시퀀스 길이
            plt.subplot(min(batch_size, 2), frames.size(1), b*frames.size(1) + s + 1)
            # [C, H, W] -> [H, W, C]로 변환
            img = frames[b, s].permute(1, 2, 0).cpu().numpy()
            plt.imshow(img)
            plt.title(f"Batch {b}, Frame {frame_indices[b][s].item()}")
            plt.axis('off')
    
    aug_status = "enabled" if use_augmentation else "disabled"
    plt.suptitle(f"Sequential Data Samples (Augmentation: {aug_status})")
    plt.tight_layout()
    plt.savefig("sequential_dataloader_test.png")
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
    
    # 시퀀셜 데이터로더 테스트
    print("\n=== Testing sequential dataloader (no augmentation) ===")
    test_sequential_dataloader(data_dir, seq_len=3, use_augmentation=False)
    
    # 증강이 적용된 시퀀셜 데이터로더 테스트
    print("\n=== Testing sequential dataloader (with augmentation) ===")
    test_sequential_dataloader(data_dir, seq_len=3, use_augmentation=True) 