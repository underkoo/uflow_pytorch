import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F


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
    def __init__(self, data_dir, transform=None, target_height=196, target_width=256, convert_to_rgb=True):
        """
        Args:
            data_dir (str): .npy 파일이 있는 디렉토리
            transform (callable, optional): 각 프레임에 적용할 변환 함수
            target_height (int): 다운샘플링 후 이미지 높이
            target_width (int): 다운샘플링 후 이미지 너비
            convert_to_rgb (bool): Bayer RAW를 RGB로 변환할지 여부
        """
        self.data_dir = Path(data_dir)
        self.file_list = sorted([f for f in self.data_dir.glob('*.npy')])
        self.transform = transform
        self.target_height = target_height
        self.target_width = target_width
        self.convert_to_rgb = convert_to_rgb
        
        # 파일 목록 확인
        print(f"{len(self.file_list)}개의 .npy 파일을 '{data_dir}'에서 찾았습니다")
        
        # 첫 번째 파일로 테스트하여 프레임 개수 확인
        if len(self.file_list) > 0:
            test_data = np.load(str(self.file_list[0]))
            print(f"첫 번째 파일 로드 완료: {test_data.shape}")
            print(f"각 npy 파일은 {test_data.shape[0]}개의 프레임을 포함")
    
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
        
        # Apply transforms if specified
        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
        
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
                 target_height=196, target_width=256, convert_to_rgb=True):
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
        
        # 파일 목록 확인
        print(f"{len(self.file_list)}개의 .npy 파일을 '{data_dir}'에서 찾았습니다")
        
        # 사용 가능한 시퀀스 인덱스 미리 계산
        self.sequences = []
        for file_idx, file_path in enumerate(self.file_list):
            frames = np.load(str(file_path))
            num_frames = frames.shape[0]
            
            # 가능한 시퀀스 시작점 계산
            for start_idx in range(0, num_frames - seq_len + 1, stride):
                if start_idx + seq_len <= num_frames:
                    self.sequences.append((file_idx, start_idx))
        
        print(f"총 {len(self.sequences)}개의 시퀀스를 생성했습니다 (seq_len={seq_len}, stride={stride})")
        
        # 첫 번째 파일로 테스트하여 데이터 형식 확인
        if len(self.file_list) > 0:
            test_data = np.load(str(self.file_list[0]))
            self.has_channel = len(test_data.shape) == 4
            print(f"데이터 형식: {'채널 있음 (B,H,W,C)' if self.has_channel else '채널 없음 (B,H,W)'}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # 시퀀스 정보 가져오기
        file_idx, start_idx = self.sequences[idx]
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
                max_val = 255.0 if frame.max() <= 255 else 65535.0
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
        
        # 결과 반환
        result = {
            'frames': frames,  # 프레임 리스트 [frame_t-1, frame_t, frame_t+1, ...]
            'file_idx': file_idx,
            'frame_indices': torch.tensor(frame_indices)  # 텐서로 변환하여 배치 처리 가능하게 함
        }
        
        # 편의를 위해 첫 3개 프레임에 대해 별도 이름 제공 (시퀀스 손실 계산용)
        if self.seq_len >= 3:
            result['frame_tm1'] = frames[0]  # t-1 프레임
            result['frame_t'] = frames[1]    # t 프레임
            result['frame_tp1'] = frames[2]  # t+1 프레임
        
        return result


def create_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=4, 
                      target_height=196, target_width=256, convert_to_rgb=True):
    """
    멀티프레임 raw 데이터셋용 데이터로더 생성
    
    Args:
        data_dir (str): .npy 파일이 있는 디렉토리
        batch_size (int): 배치 크기
        shuffle (bool): 데이터셋 셔플 여부
        num_workers (int): 데이터 로딩용 워커 수
        target_height (int): 다운샘플링 후 이미지 높이
        target_width (int): 다운샘플링 후 이미지 너비
        convert_to_rgb (bool): Bayer RAW를 RGB로 변환할지 여부
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = MultiFrameRawDataset(
        data_dir, 
        target_height=target_height, 
        target_width=target_width, 
        convert_to_rgb=convert_to_rgb
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def create_sequential_dataloader(data_dir, seq_len=3, batch_size=32, shuffle=True, num_workers=4, stride=1,
                                target_height=196, target_width=256, convert_to_rgb=True):
    """
    시퀀스 기반 손실 계산을 위한 연속 프레임 데이터로더 생성
    
    Args:
        data_dir (str): .npy 파일이 있는 디렉토리
        seq_len (int): 시퀀스 길이 (연속 프레임 수)
        batch_size (int): 배치 크기
        shuffle (bool): 데이터셋 셔플 여부
        num_workers (int): 데이터 로딩용 워커 수
        stride (int): 시퀀스 샘플링 스트라이드
        target_height (int): 다운샘플링 후 이미지 높이
        target_width (int): 다운샘플링 후 이미지 너비
        convert_to_rgb (bool): Bayer RAW를 RGB로 변환할지 여부
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = SequentialFrameDataset(
        data_dir, 
        seq_len=seq_len, 
        stride=stride,
        target_height=target_height,
        target_width=target_width,
        convert_to_rgb=convert_to_rgb
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# 테스트 함수
def test_dataloader(data_dir, batch_size=4):
    """
    데이터로더 테스트 및 샘플 시각화
    
    Args:
        data_dir (str): .npy 파일이 있는 디렉토리
        batch_size (int): 테스트용 배치 크기
    """
    # 데이터 로더 생성
    dataloader = create_dataloader(
        data_dir, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    if len(dataloader) == 0:
        print("데이터로더가 비어 있습니다. 디렉토리를 확인하세요.")
        return
    
    # 첫 번째 배치 가져오기
    batch = next(iter(dataloader))
    frame1 = batch['frame1']
    frame2 = batch['frame2']
    file_indices = batch['file_idx']
    frame_indices1 = batch['frame_idx1']
    frame_indices2 = batch['frame_idx2']
    
    # 배치 정보 출력
    print(f"Batch size: {batch_size}")
    print(f"Frame1 shape: {frame1.shape}")
    print(f"Frame2 shape: {frame2.shape}")
    
    # 샘플 시각화
    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 5*batch_size))
    
    # 배치 크기가 1인 경우 axes 차원 처리
    if batch_size == 1:
        axes = axes.reshape(1, 2)
    
    for i in range(min(batch_size, len(frame1))):
        # 프레임 정보 출력
        file_idx = file_indices[i].item()
        frame_idx1 = frame_indices1[i].item()
        frame_idx2 = frame_indices2[i].item()
        
        # frame1 시각화
        img1 = frame1[i].permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        if img1.shape[2] == 1:  # 흑백 이미지인 경우
            axes[i, 0].imshow(img1.squeeze(), cmap='gray')
        else:
            axes[i, 0].imshow(np.clip(img1, 0, 1))  # 값이 0-1 범위를 넘어갈 경우 클립
        axes[i, 0].set_title(f"File {file_idx}, Frame {frame_idx1}")
        axes[i, 0].axis('off')
        
        # frame2 시각화
        img2 = frame2[i].permute(1, 2, 0).numpy()
        if img2.shape[2] == 1:  # 흑백 이미지인 경우
            axes[i, 1].imshow(img2.squeeze(), cmap='gray')
        else:
            axes[i, 1].imshow(np.clip(img2, 0, 1))
        axes[i, 1].set_title(f"File {file_idx}, Frame {frame_idx2}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataloader_test.png')
    plt.show()
    
    print("데이터 로더 테스트 완료! 이미지가 'dataloader_test.png'로 저장되었습니다.")
    
    return dataloader


def test_sequential_dataloader(data_dir, seq_len=3, batch_size=2):
    """
    시퀀스 데이터로더 테스트 및 샘플 시각화
    
    Args:
        data_dir (str): .npy 파일이 있는 디렉토리
        seq_len (int): 시퀀스 길이
        batch_size (int): 테스트용 배치 크기
    """
    # 데이터 로더 생성
    dataloader = create_sequential_dataloader(
        data_dir, 
        seq_len=seq_len,
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    if len(dataloader) == 0:
        print("데이터로더가 비어 있습니다. 디렉토리를 확인하세요.")
        return
    
    # 첫 번째 배치 가져오기
    batch = next(iter(dataloader))
    frames_list = batch['frames']  # 리스트 형태 [seq_len][batch_size, C, H, W]
    file_indices = batch['file_idx']  # [batch_size]
    frame_indices = batch['frame_indices']  # [batch_size, seq_len]
    
    # 배치 정보 출력
    print(f"Batch size: {batch_size}")
    print(f"시퀀스 길이: {seq_len}")
    print(f"첫 번째 프레임 shape: {frames_list[0].shape}")
    print(f"frame_indices shape: {frame_indices.shape}")
    
    # 샘플 시각화
    fig, axes = plt.subplots(batch_size, seq_len, figsize=(4*seq_len, 4*batch_size))
    
    # 배치 크기가 1인 경우 axes 차원 처리
    if batch_size == 1:
        axes = axes.reshape(1, seq_len)
    
    for i in range(min(batch_size, frames_list[0].size(0))):
        # 프레임 정보 출력
        file_idx = file_indices[i].item()
        
        # 각 시퀀스 프레임 시각화
        for j in range(seq_len):
            frame = frames_list[j][i]  # 시퀀스의 j번째, 배치의 i번째
            frame_idx = frame_indices[i, j].item()  # [batch, seq] 형태로 접근
            
            # 시각화
            img = frame.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
            if img.shape[2] == 1:  # 흑백 이미지인 경우
                axes[i, j].imshow(img.squeeze(), cmap='gray')
            else:
                axes[i, j].imshow(np.clip(img, 0, 1))
            axes[i, j].set_title(f"File {file_idx}, Frame {frame_idx}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('sequential_dataloader_test.png')
    plt.show()
    
    print("시퀀스 데이터로더 테스트 완료! 이미지가 'sequential_dataloader_test.png'로 저장되었습니다.")
    
    return dataloader


# 메인 실행 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dataloader for multi-frame raw data')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with .npy files')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--mode', type=str, default='sequential', choices=['random', 'sequential'], 
                        help='Dataloader mode: random pairs or sequential frames')
    parser.add_argument('--seq_len', type=int, default=3, help='Sequence length for sequential mode')
    
    args = parser.parse_args()
    
    if args.mode == 'random':
        test_dataloader(args.data_dir, args.batch_size)
    else:
        test_sequential_dataloader(args.data_dir, args.seq_len, args.batch_size) 