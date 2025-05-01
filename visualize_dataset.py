#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

# 데이터로더 모듈 임포트
from dataloader import preprocess_raw_bayer


def create_output_directory(output_dir):
    """출력 디렉토리 생성"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리 생성: {output_dir}")


def read_file_list(file_list_path):
    """파일 목록 텍스트 파일 읽기"""
    with open(file_list_path, 'r') as f:
        lines = f.readlines()
        file_paths = [Path(line.strip()) for line in lines if line.strip()]
    
    print(f"{len(file_paths)}개의 파일 경로를 {file_list_path}에서 로드했습니다.")
    return file_paths


def visualize_frame(frame, frame_idx, cmap=None, title=None):
    """단일 프레임 시각화"""
    if isinstance(frame, torch.Tensor):
        # 텐서를 넘파이 배열로 변환
        if frame.dim() == 3 and frame.shape[0] == 3:  # [C, H, W] 형태
            frame = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, C] 형태로 변환
        else:
            frame = frame.cpu().numpy()
    
    plt.imshow(frame, cmap=cmap)
    if title:
        plt.title(title)
    else:
        plt.title(f"Frame {frame_idx}")
    plt.axis('off')


def visualize_file(file_path, output_dir, target_height=192, target_width=256, apply_pregamma=False, pregamma_value=2.0):
    """
    단일 파일 내의 모든 프레임 시각화
    
    Args:
        file_path: .npy 파일 경로
        output_dir: 결과물 저장 디렉토리
        target_height: 이미지 높이
        target_width: 이미지 너비
        apply_pregamma: pre-gamma 보정 적용 여부
        pregamma_value: gamma 값
    """
    try:
        # 파일명 추출
        file_name = file_path.stem
        
        # .npy 파일 로드
        multi_frames = np.load(str(file_path))
        num_frames = multi_frames.shape[0]
        
        # RAW 이미지와 RGB 이미지용 시각화 준비
        num_rows = 2  # RAW와 RGB 행
        num_cols = min(num_frames, 8)  # 한 행에 최대 8개 프레임
        
        # 전체 프레임 수가 많은 경우 여러 이미지로 분할
        num_pages = (num_frames + num_cols - 1) // num_cols
        
        for page in range(num_pages):
            fig, axes = plt.subplots(num_rows, min(num_cols, num_frames - page * num_cols), 
                                     figsize=(min(num_cols, num_frames - page * num_cols) * 3, 6))
            
            # 단일 축일 경우 배열로 변환
            if num_rows == 1 and min(num_cols, num_frames - page * num_cols) == 1:
                axes = np.array([[axes]])
            elif num_rows == 1:
                axes = axes.reshape(1, -1)
            elif min(num_cols, num_frames - page * num_cols) == 1:
                axes = axes.reshape(-1, 1)
            
            # 현재 페이지의 프레임 시각화
            for i in range(min(num_cols, num_frames - page * num_cols)):
                frame_idx = page * num_cols + i
                
                # RAW 이미지 시각화 (첫 번째 행)
                raw_frame = multi_frames[frame_idx]
                # 정규화
                raw_norm = raw_frame.astype(np.float32) / 1023.0
                
                # RAW 이미지 표시
                axes[0, i].imshow(raw_norm, cmap='gray')
                axes[0, i].set_title(f"RAW {frame_idx}")
                axes[0, i].axis('off')
                
                # RGB 변환 이미지 시각화 (두 번째 행)
                # 텐서로 변환
                frame_tensor = torch.from_numpy(raw_norm).unsqueeze(0)  # [1, H, W]
                
                # pre-gamma 보정 적용
                if apply_pregamma:
                    frame_tensor = torch.pow(frame_tensor, 1.0 / pregamma_value)
                
                # 배치 형태로 준비 (preprocess_raw_bayer 함수를 위해)
                frame_batch = frame_tensor.unsqueeze(0)  # [1, 1, H, W]
                
                # RGB로 변환
                rgb_frame = preprocess_raw_bayer(frame_batch, target_height, target_width)
                rgb_frame = rgb_frame[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3] 형태로 변환
                
                # RGB 이미지 표시
                axes[1, i].imshow(rgb_frame)
                axes[1, i].set_title(f"RGB {frame_idx}")
                axes[1, i].axis('off')
            
            # 여백 설정 및 제목 추가
            plt.tight_layout()
            fig.suptitle(f"File: {file_name} (Frames {page * num_cols} - {min((page + 1) * num_cols - 1, num_frames - 1)})", 
                         fontsize=16, y=1.02)
            
            # 파일 저장
            output_path = os.path.join(output_dir, f"{file_name}_page_{page+1}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            
        print(f"파일 시각화 완료: {file_path} -> {output_dir}/{file_name}_page_*.png")
        return True
    
    except Exception as e:
        print(f"파일 시각화 중 오류 발생 {file_path}: {e}")
        return False


def visualize_dataset(file_list_path, output_dir, target_height=192, target_width=256, 
                     apply_pregamma=False, pregamma_value=2.0):
    """
    데이터셋 전체 시각화
    
    Args:
        file_list_path: 파일 목록이 있는 텍스트 파일 경로
        output_dir: 결과물 저장 디렉토리
        target_height: 이미지 높이
        target_width: 이미지 너비
        apply_pregamma: pre-gamma 보정 적용 여부
        pregamma_value: gamma 값
    """
    # 출력 디렉토리 생성
    create_output_directory(output_dir)
    
    # 파일 목록 로드
    file_paths = read_file_list(file_list_path)
    
    # 각 파일 시각화
    success_count = 0
    error_count = 0
    
    for file_path in tqdm(file_paths, desc="파일 시각화 진행 중"):
        success = visualize_file(
            file_path, 
            output_dir, 
            target_height=target_height, 
            target_width=target_width,
            apply_pregamma=apply_pregamma, 
            pregamma_value=pregamma_value
        )
        
        if success:
            success_count += 1
        else:
            error_count += 1
    
    print(f"\n시각화 완료: 성공 {success_count}, 오류 {error_count}")
    print(f"결과물이 {output_dir} 디렉토리에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='데이터셋 시각화 도구')
    parser.add_argument('--file_list', type=str, default='validation.txt', help='파일 경로 목록 텍스트 파일')
    parser.add_argument('--output_dir', type=str, default='visualization_output', help='결과물 저장 디렉토리')
    parser.add_argument('--target_height', type=int, default=192, help='처리 후 이미지 높이')
    parser.add_argument('--target_width', type=int, default=256, help='처리 후 이미지 너비')
    parser.add_argument('--apply_pregamma', action='store_true', help='pre-gamma 보정 적용')
    parser.add_argument('--pregamma_value', type=float, default=2.0, help='gamma 보정 값 (기본값: 2.0)')
    
    args = parser.parse_args()
    
    # 데이터셋 시각화
    visualize_dataset(
        args.file_list,
        args.output_dir,
        target_height=args.target_height,
        target_width=args.target_width,
        apply_pregamma=args.apply_pregamma,
        pregamma_value=args.pregamma_value
    )


if __name__ == "__main__":
    main() 