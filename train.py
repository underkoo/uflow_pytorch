#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import utils

from models import UFlow
import losses
from dataloader import create_dataloader


class DebugLogger:
    """
    디버깅 로그를 파일에 저장하는 유틸리티 클래스
    """
    def __init__(self, log_dir, enabled=False):
        self.enabled = enabled
        self.log_dir = log_dir
        
        if enabled:
            # 디버그 로그 디렉토리 생성
            self.debug_dir = os.path.join(log_dir, 'debug_logs')
            os.makedirs(self.debug_dir, exist_ok=True)
            
            # 로그 파일 경로
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(self.debug_dir, f'debug_{timestamp}.log')
            
            # 로거 설정
            self.logger = logging.getLogger('debug')
            self.logger.setLevel(logging.DEBUG)
            
            # 파일 핸들러
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # 포맷터
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # 로거에 핸들러 추가
            self.logger.addHandler(file_handler)
            
            # 콘솔 핸들러 (ERROR 이상만 표시)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            self.log_info(f"디버그 로그 초기화 완료. 로그 파일: {self.log_file}")
            
            # 특징 시각화를 위한 디렉토리 생성
            self.features_dir = os.path.join(log_dir, 'features')
            os.makedirs(self.features_dir, exist_ok=True)
            self.log_info(f"특징 시각화 디렉토리 생성: {self.features_dir}")
    
    def log_debug(self, message):
        """디버그 레벨 로그 (파일에만 저장)"""
        if self.enabled:
            self.logger.debug(message)
    
    def log_info(self, message):
        """정보 레벨 로그 (파일에만 저장)"""
        if self.enabled:
            self.logger.info(message)
    
    def log_warning(self, message):
        """경고 레벨 로그 (파일에 저장 + 중요 경고는 화면에 출력)"""
        if self.enabled:
            self.logger.warning(message)
    
    def log_error(self, message):
        """오류 레벨 로그 (파일에 저장 + 화면에 출력)"""
        if self.enabled:
            self.logger.error(message)
    
    def log_critical(self, message):
        """심각한 오류 로그 (파일에 저장 + 화면에 출력)"""
        if self.enabled:
            self.logger.critical(message)
    
    def log_model_stats(self, step, img1, img2, flows):
        """모델 출력 통계 기록"""
        if not self.enabled:
            return
            
        self.log_info(f"\n[스텝 {step}] 모델 출력 통계")
        
        # 이미지 통계
        self.log_info(f"이미지 1 범위: {img1.min():.4f} ~ {img1.max():.4f}, 평균: {img1.mean():.4f}")
        self.log_info(f"이미지 2 범위: {img2.min():.4f} ~ {img2.max():.4f}, 평균: {img2.mean():.4f}")
        
        # 흐름 통계
        for i, flow in enumerate(flows):
            self.log_info(f"피라미드 레벨 {i} 흐름 크기: {flow.shape}")
            self.log_info(f"  범위: {flow.min():.4f} ~ {flow.max():.4f}, 평균 변위: {flow.abs().mean():.4f}")
            
            # 흐름 크기 (픽셀 변위)
            flow_mag = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
            self.log_info(f"  변위 크기 - 최소: {flow_mag.min():.4f}, 최대: {flow_mag.max():.4f}, 평균: {flow_mag.mean():.4f}")
            
            # 문제가 있는지 확인
            if torch.isnan(flow).any():
                self.log_error(f"[심각] 레벨 {i} 흐름에 NaN 값이 있습니다!")
            if torch.isinf(flow).any():
                self.log_error(f"[심각] 레벨 {i} 흐름에 Inf 값이 있습니다!")
            
            # 모델이 학습 중인지 확인 (흐름 값이 모두 0에 가까운 경우 의심)
            if flow.abs().mean() < 1e-4:
                self.log_warning(f"[경고] 레벨 {i} 흐름이 거의 0입니다. 모델이 제대로 학습되지 않을 수 있습니다.")
    
    def log_feature_stats(self, step, features1, features2):
        """특징 피라미드 통계 기록"""
        if not self.enabled:
            return
            
        self.log_info(f"\n[스텝 {step}] 특징 피라미드 통계")
        
        # 특징 통계
        for i, (feat1, feat2) in enumerate(zip(features1, features2)):
            self.log_info(f"피라미드 레벨 {i} 특징 크기: {feat1.shape}")
            self.log_info(f"  특징1 범위: {feat1.min():.4f} ~ {feat1.max():.4f}, 평균: {feat1.mean():.4f}, 표준편차: {feat1.std():.4f}")
            self.log_info(f"  특징2 범위: {feat2.min():.4f} ~ {feat2.max():.4f}, 평균: {feat2.mean():.4f}, 표준편차: {feat2.std():.4f}")
            
            # 활성화 정보 (L1 norm)
            feat1_norm = feat1.abs().mean().item()
            feat2_norm = feat2.abs().mean().item()
            self.log_info(f"  특징1 L1 norm: {feat1_norm:.6f}, 특징2 L1 norm: {feat2_norm:.6f}")
            
            # 특징 간 차이
            feat_diff = (feat1 - feat2).abs().mean().item()
            self.log_info(f"  특징 간 평균 절대 차이: {feat_diff:.6f}")
            
            # 문제가 있는지 확인
            if torch.isnan(feat1).any() or torch.isnan(feat2).any():
                self.log_error(f"[심각] 레벨 {i} 특징에 NaN 값이 있습니다!")
            if torch.isinf(feat1).any() or torch.isinf(feat2).any():
                self.log_error(f"[심각] 레벨 {i} 특징에 Inf 값이 있습니다!")
            
            # 특징이 의미 있는지 확인 (모든 값이 거의 0인 경우)
            if feat1_norm < 1e-4 or feat2_norm < 1e-4:
                self.log_warning(f"[경고] 레벨 {i} 특징이 거의 0입니다. 특징 추출기가 제대로 학습되지 않을 수 있습니다.")
            
            # 특징이 거의 동일한지 확인 (거의 변화가 없는 경우)
            if feat_diff < 1e-6:
                self.log_warning(f"[경고] 레벨 {i} 특징 간 차이가 거의 없습니다. 두 이미지가 동일하게 인식될 수 있습니다.")
    
    def log_gradient_stats(self, step, param_stats):
        """그래디언트 통계 기록"""
        if not self.enabled:
            return
            
        self.log_info(f"\n[스텝 {step}] 그래디언트 통계")
        
        # Top-5 가장 큰 그래디언트 비율을 가진 레이어 출력
        param_stats.sort(key=lambda x: x['ratio'], reverse=True)
        for i, stat in enumerate(param_stats[:5]):
            self.log_info(f"{i+1}. {stat['name']}: 그래디언트 {stat['grad_norm']:.6f}, 파라미터 {stat['param_norm']:.6f}, 비율 {stat['ratio']:.6f}")
        
        # 그래디언트가 0인 레이어 수 확인
        zero_grads = sum(1 for stat in param_stats if stat['grad_norm'] < 1e-8)
        self.log_info(f"전체 레이어 수: {len(param_stats)}, 그래디언트가 0인 레이어 수: {zero_grads}")
        
        # 위험 신호: 대부분의 레이어가 그래디언트 0
        if zero_grads > len(param_stats) * 0.5:
            self.log_warning(f"[경고] 전체 {len(param_stats)}개 중 {zero_grads}개 레이어의 그래디언트가 0입니다. 학습이 제대로 진행되지 않을 수 있습니다.")
    
    def log_loss_info(self, step, losses):
        """손실 정보 기록"""
        if not self.enabled:
            return
            
        self.log_info(f"\n[스텝 {step}] 손실 통계")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.log_info(f"{key}: {value.item():.6f}")
                
                # NaN/Inf 체크
                if torch.isnan(value) or torch.isinf(value):
                    self.log_error(f"[심각] {key} 손실이 {value.item()}입니다!")
    
    def log_gradient_flow_check(self, step, flow_norm, grad_norm, ratio):
        """그래디언트 흐름 체크 결과 기록"""
        if not self.enabled:
            return
            
        self.log_info(f"\n[스텝 {step}] 그래디언트 흐름 체크 결과:")
        self.log_info(f"  흐름 norm: {flow_norm:.6f}")
        self.log_info(f"  그래디언트 norm: {grad_norm:.6f}")
        self.log_info(f"  비율: {ratio:.6f}")
        
        if grad_norm < 1e-6:
            self.log_warning("  [경고] 그래디언트가 너무 작습니다! 그래디언트 흐름에 문제가 있을 수 있습니다.")
    
    def save_feature_visualization(self, step, img1, img2, features1, features2, flows):
        """특징 피라미드 및 광학 흐름 시각화 저장"""
        if not self.enabled or not self.features_dir:
            return
            
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # GUI 없이 이미지 저장
            import numpy as np
            import cv2
            
            # 현재 랭크 확인 (분산 환경에서 랭크 0에서만 저장)
            is_master = True
            local_rank = 0
            
            # 스텝별 디렉토리 생성
            step_dir = os.path.join(self.features_dir, f'step_{step:06d}')
            os.makedirs(step_dir, exist_ok=True)
            
            # 배치에서 첫 번째 샘플만 시각화
            idx = 0
            
            # 입력 이미지 시각화
            img1_np = img1[idx].detach().cpu().permute(1, 2, 0).numpy()
            img2_np = img2[idx].detach().cpu().permute(1, 2, 0).numpy()
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(np.clip(img1_np, 0, 1))
            plt.title('Image 1')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(np.clip(img2_np, 0, 1))
            plt.title('Image 2')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(step_dir, 'input_images.png'))
            plt.close()
            
            # 특징 피라미드 시각화
            for level, (feat1, feat2) in enumerate(zip(features1, features2)):
                # 각 레벨의 특징 맵 시각화
                self._visualize_feature_maps(
                    feat1[idx], feat2[idx], 
                    os.path.join(step_dir, f'features_level_{level}.png'),
                    title=f'Feature Level {level}'
                )
                
                # 특징 맵 통계 기록 (로그 파일)
                feat_stats_path = os.path.join(step_dir, f'features_level_{level}_stats.txt')
                with open(feat_stats_path, 'w') as f:
                    f.write(f"Feature Level {level} Statistics:\n")
                    f.write(f"Shape: {feat1[idx].shape}\n\n")
                    
                    # 채널별 통계 (최대 10개 채널만)
                    num_channels = min(10, feat1[idx].shape[0])
                    f.write("Channel-wise Statistics (First 10 channels):\n")
                    
                    for c in range(num_channels):
                        f1_chan = feat1[idx][c]
                        f2_chan = feat2[idx][c]
                        
                        f.write(f"\nChannel {c}:\n")
                        f.write(f"  Image 1 - Min: {f1_chan.min().item():.6f}, Max: {f1_chan.max().item():.6f}, ")
                        f.write(f"Mean: {f1_chan.mean().item():.6f}, Std: {f1_chan.std().item():.6f}\n")
                        
                        f.write(f"  Image 2 - Min: {f2_chan.min().item():.6f}, Max: {f2_chan.max().item():.6f}, ")
                        f.write(f"Mean: {f2_chan.mean().item():.6f}, Std: {f2_chan.std().item():.6f}\n")
                    
                    # 전체 통계
                    f.write("\nOverall Statistics:\n")
                    f.write(f"  Image 1 - Min: {feat1[idx].min().item():.6f}, Max: {feat1[idx].max().item():.6f}, ")
                    f.write(f"Mean: {feat1[idx].mean().item():.6f}, Std: {feat1[idx].std().item():.6f}\n")
                    
                    f.write(f"  Image 2 - Min: {feat2[idx].min().item():.6f}, Max: {feat2[idx].max().item():.6f}, ")
                    f.write(f"Mean: {feat2[idx].mean().item():.6f}, Std: {feat2[idx].std().item():.6f}\n")
                    
                    # 특징 간 차이
                    feat_diff = (feat1[idx] - feat2[idx]).abs()
                    f.write("\nFeature Difference:\n")
                    f.write(f"  Mean Absolute Difference: {feat_diff.mean().item():.6f}\n")
                    f.write(f"  Max Absolute Difference: {feat_diff.max().item():.6f}\n")
                    f.write(f"  L1 Norm: {feat_diff.sum().item():.6f}\n")
            
            # 광학 흐름 피라미드 시각화
            for level, flow in enumerate(flows):
                flow_np = flow[idx].detach().cpu().permute(1, 2, 0).numpy()
                
                # 광학 흐름을 색상으로 변환
                flow_rgb = self._flow_to_color(flow_np)
                
                # 흐름 크기 (픽셀 변위) 계산
                flow_mag = np.sqrt(flow_np[..., 0]**2 + flow_np[..., 1]**2)
                flow_mag_norm = flow_mag / (flow_mag.max() + 1e-8)
                
                # 저장
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.imshow(flow_rgb)
                plt.title(f'Flow Level {level} (Color)')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(flow_mag_norm, cmap='inferno')
                plt.title(f'Flow Level {level} (Magnitude)')
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(step_dir, f'flow_level_{level}.png'))
                plt.close()
                
                # 히스토그램 시각화 (수평/수직 성분 분포)
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.hist(flow_np[..., 0].flatten(), bins=50, alpha=0.7, label='x-flow')
                plt.title(f'Flow Level {level} - X Component Histogram')
                plt.xlabel('Flow Magnitude (pixels)')
                plt.ylabel('Frequency')
                plt.axvline(x=0, color='r', linestyle='--')
                
                plt.subplot(1, 2, 2)
                plt.hist(flow_np[..., 1].flatten(), bins=50, alpha=0.7, label='y-flow')
                plt.title(f'Flow Level {level} - Y Component Histogram')
                plt.xlabel('Flow Magnitude (pixels)')
                plt.ylabel('Frequency')
                plt.axvline(x=0, color='r', linestyle='--')
                
                plt.tight_layout()
                plt.savefig(os.path.join(step_dir, f'flow_level_{level}_histogram.png'))
                plt.close()
            
            # 와핑된 이미지 시각화
            import utils
            warped_img2 = utils.warp_image(img2, flows[0])
            warped_img2_np = warped_img2[idx].detach().cpu().permute(1, 2, 0).numpy()
            
            error = np.abs(img1_np - warped_img2_np)
            error_norm = np.clip(error / (error.max() + 1e-8), 0, 1)
            
            plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            plt.imshow(np.clip(img1_np, 0, 1))
            plt.title('Image 1')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(np.clip(warped_img2_np, 0, 1))
            plt.title('Warped Image 2')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(error_norm)
            plt.title('Warping Error')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(step_dir, 'warping_result.png'))
            plt.close()
            
            # 오류 히스토그램
            plt.figure(figsize=(10, 6))
            plt.hist(error.flatten(), bins=50, alpha=0.7)
            plt.title('Warping Error Histogram')
            plt.xlabel('Absolute Error')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(step_dir, 'warping_error_histogram.png'))
            plt.close()
            
            # 요약 리포트 저장
            report_path = os.path.join(step_dir, 'feature_summary.txt')
            with open(report_path, 'w') as f:
                f.write(f"Feature Visualization Summary for Step {step}\n")
                f.write(f"====================================\n\n")
                
                # 피라미드 레벨 수
                f.write(f"Number of feature pyramid levels: {len(features1)}\n")
                f.write(f"Number of flow pyramid levels: {len(flows)}\n\n")
                
                # 각 레벨 크기 요약
                f.write("Feature Pyramid Level Sizes:\n")
                for i, (feat1, feat2) in enumerate(zip(features1, features2)):
                    feat = feat1[idx]
                    f.write(f"  Level {i}: {feat.shape[0]} channels, {feat.shape[1]}x{feat.shape[2]} resolution\n")
                
                f.write("\nFlow Pyramid Level Sizes:\n")
                for i, flow in enumerate(flows):
                    f.write(f"  Level {i}: {flow.shape[1]} channels, {flow.shape[2]}x{flow.shape[3]} resolution\n")
                
                # 입력 이미지 정보
                f.write("\nInput Image Information:\n")
                f.write(f"  Image 1 - Min: {img1_np.min():.4f}, Max: {img1_np.max():.4f}, Mean: {img1_np.mean():.4f}\n")
                f.write(f"  Image 2 - Min: {img2_np.min():.4f}, Max: {img2_np.max():.4f}, Mean: {img2_np.mean():.4f}\n")
                
                # 와핑 오류 정보
                f.write("\nWarping Error Information:\n")
                f.write(f"  Mean Error: {error.mean():.6f}\n")
                f.write(f"  Max Error: {error.max():.6f}\n")
                f.write(f"  Median Error: {np.median(error):.6f}\n")
            
            self.log_info(f"특징 및 흐름 시각화가 {step_dir}에 저장되었습니다.")
            
            return True
            
        except Exception as e:
            self.log_error(f"특징 시각화 중 오류 발생: {str(e)}")
            import traceback
            self.log_error(traceback.format_exc())
            return False
    
    def _visualize_feature_maps(self, feat1, feat2, save_path, title='Feature Maps', max_channels=16):
        """특징 맵 시각화 및 저장"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 채널 수 제한 (너무 많으면 시각화가 복잡해짐)
        C = min(feat1.shape[0], max_channels)
        
        # 특징 맵을 [C, H, W] -> [H, W, C] 변환 후 numpy 배열로 변환
        feat1_np = feat1[:C].detach().cpu().numpy()
        feat2_np = feat2[:C].detach().cpu().numpy()
        
        # 배치 정규화를 통한 시각화 (각 특징 맵마다 개별 정규화)
        feat1_viz = np.zeros_like(feat1_np)
        feat2_viz = np.zeros_like(feat2_np)
        
        for c in range(C):
            # 최소-최대 정규화
            f1_min, f1_max = feat1_np[c].min(), feat1_np[c].max()
            f2_min, f2_max = feat2_np[c].min(), feat2_np[c].max()
            
            # 범위가 0인 경우 (상수) 처리
            if f1_max - f1_min > 1e-6:
                feat1_viz[c] = (feat1_np[c] - f1_min) / (f1_max - f1_min)
            else:
                feat1_viz[c] = 0.5 * np.ones_like(feat1_np[c])
                
            if f2_max - f2_min > 1e-6:
                feat2_viz[c] = (feat2_np[c] - f2_min) / (f2_max - f2_min)
            else:
                feat2_viz[c] = 0.5 * np.ones_like(feat2_np[c])
        
        # 그리드 계산
        grid_size = int(np.ceil(np.sqrt(C * 2)))  # 두 이미지의 특징 맵을 함께 표시
        n_rows = int(np.ceil(C * 2 / grid_size))
        
        # 전체 특징 맵 시각화
        plt.figure(figsize=(grid_size * 2, n_rows * 2))
        plt.suptitle(title, fontsize=16)
        
        for i in range(C):
            # 이미지 1의 특징 맵
            plt.subplot(n_rows, grid_size, i * 2 + 1)
            plt.imshow(feat1_viz[i], cmap='viridis')
            plt.axis('off')
            plt.title(f'Img1 Ch{i}')
            
            # 이미지 2의 특징 맵
            plt.subplot(n_rows, grid_size, i * 2 + 2)
            plt.imshow(feat2_viz[i], cmap='viridis')
            plt.axis('off')
            plt.title(f'Img2 Ch{i}')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        # 특징 맵 차이 시각화 추가 (두 이미지 간의 특징 차이)
        diff_path = save_path.replace('.png', '_diff.png')
        feat_diff_viz = np.zeros_like(feat1_np)
        
        plt.figure(figsize=(int(np.ceil(np.sqrt(C))) * 3, int(np.ceil(C / np.ceil(np.sqrt(C)))) * 3))
        plt.suptitle(f'{title} - Differences', fontsize=16)
        
        for c in range(C):
            # 차이 계산 (절대값)
            diff = np.abs(feat1_np[c] - feat2_np[c])
            
            # 정규화
            diff_min, diff_max = diff.min(), diff.max()
            if diff_max - diff_min > 1e-6:
                feat_diff_viz[c] = (diff - diff_min) / (diff_max - diff_min)
            else:
                feat_diff_viz[c] = 0.0 * np.ones_like(diff)
            
            # 차이 시각화
            plt.subplot(int(np.ceil(C / np.ceil(np.sqrt(C)))), int(np.ceil(np.sqrt(C))), c + 1)
            plt.imshow(feat_diff_viz[c], cmap='hot')  # 'hot' 컬러맵으로 차이 강조
            plt.axis('off')
            plt.title(f'Diff Ch{c}')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(diff_path, dpi=150)
        plt.close()
        
        # 특징 맵 활성화 통계 시각화 (채널별 평균 활성화 강도)
        stats_path = save_path.replace('.png', '_stats.png')
        
        # 채널별 평균 활성화 계산
        feat1_mean = np.mean(np.abs(feat1_np), axis=(1, 2))
        feat2_mean = np.mean(np.abs(feat2_np), axis=(1, 2))
        
        # 채널별 통계 시각화
        plt.figure(figsize=(12, 8))
        
        # 활성화 평균
        plt.subplot(2, 1, 1)
        x = np.arange(C)
        width = 0.35
        plt.bar(x - width/2, feat1_mean, width, label='Image 1')
        plt.bar(x + width/2, feat2_mean, width, label='Image 2')
        plt.xlabel('Channel')
        plt.ylabel('Mean Absolute Activation')
        plt.title('Channel-wise Mean Absolute Activation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 이미지 간 활성화 차이
        plt.subplot(2, 1, 2)
        act_diff = np.abs(feat1_mean - feat2_mean)
        plt.bar(x, act_diff, color='orange')
        plt.xlabel('Channel')
        plt.ylabel('Activation Difference')
        plt.title('Absolute Difference in Channel Activations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(stats_path, dpi=150)
        plt.close()
        
        # 채널 유사도 시각화
        similarity_path = save_path.replace('.png', '_similarity.png')
        plt.figure(figsize=(10, 8))
        
        try:
            # 채널 간 코사인 유사도 계산
            from scipy.spatial.distance import cosine
            
            # 모든 채널 쌍에 대한 유사도 계산
            similarity_matrix = np.zeros((C, C))
            for i in range(C):
                for j in range(C):
                    f1_flat = feat1_np[i].flatten()
                    f2_flat = feat2_np[j].flatten()
                    
                    # 0 벡터가 아닌 경우에만 코사인 유사도 계산
                    if np.sum(np.abs(f1_flat)) > 1e-6 and np.sum(np.abs(f2_flat)) > 1e-6:
                        similarity_matrix[i, j] = 1 - cosine(f1_flat, f2_flat)  # 코사인 거리를 유사도로 변환
                    else:
                        similarity_matrix[i, j] = 0  # 0 벡터의 경우 유사도 0으로 설정
        except ImportError:
            # SciPy가 설치되지 않은 경우, 간단한 대체 유사도 사용
            similarity_matrix = np.zeros((C, C))
            for i in range(C):
                for j in range(C):
                    f1_flat = feat1_np[i].flatten()
                    f2_flat = feat2_np[j].flatten()
                    
                    # 간단한 L2 정규화된 내적 (유사도)
                    f1_norm = np.linalg.norm(f1_flat)
                    f2_norm = np.linalg.norm(f2_flat)
                    
                    if f1_norm > 1e-6 and f2_norm > 1e-6:
                        similarity_matrix[i, j] = np.dot(f1_flat, f2_flat) / (f1_norm * f2_norm)
                    else:
                        similarity_matrix[i, j] = 0
        
        # 유사도 행렬 시각화
        plt.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.xlabel('Image 2 Channel')
        plt.ylabel('Image 1 Channel')
        plt.title('Cross-image Channel Similarity')
        
        # 채널 인덱스 라벨 설정
        tick_positions = np.arange(C)
        plt.xticks(tick_positions, [str(i) for i in range(C)])
        plt.yticks(tick_positions, [str(i) for i in range(C)])
        
        plt.tight_layout()
        plt.savefig(similarity_path, dpi=150)
        plt.close()

    def _flow_to_color(self, flow):
        """광학 흐름을 RGB 색상으로 변환"""
        try:
            import cv2
            import numpy as np
            
            # 광학 흐름 시각화를 위한 간단한 함수
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[..., 1] = 255
            
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return rgb
        except:
            # cv2 없는 경우 간단한 시각화
            viz = np.zeros_like(flow)
            flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            viz[..., 0] = np.clip(flow[..., 0] / (flow_mag.max() + 1e-8) * 0.5 + 0.5, 0, 1)
            viz[..., 1] = np.clip(flow[..., 1] / (flow_mag.max() + 1e-8) * 0.5 + 0.5, 0, 1)
            viz[..., 2] = np.clip(flow_mag / (flow_mag.max() + 1e-8), 0, 1)
            return viz


class UFlowLightningModule(pl.LightningModule):
    """
    UFlow 모델을 훈련하기 위한 PyTorch Lightning 모듈
    """
    def __init__(self, 
                 # 모델 매개변수
                 num_channels: int = 3,
                 num_levels: int = 5,
                 feature_channels: int = 32,
                 use_cost_volume: bool = True, 
                 max_displacement: int = 4,
                 use_feature_warp: bool = True,
                 context_channels: int = 32,
                 flow_refinement_channels: int = 128,
                 dropout_rate: float = 0.25,
                 channel_multiplier: float = 1.0,
                 leaky_relu_alpha: float = 0.1,
                 shared_flow_decoder: bool = False,
                 
                 # 손실 함수 매개변수
                 photometric_weight: float = 0.0,
                 census_weight: float = 1.0, 
                 smoothness_weight: float = 2.0,
                 use_occlusion: bool = True,
                 use_valid_mask: bool = True,
                 use_stop_gradient: bool = True,
                 use_bidirectional: bool = True,
                 
                 # 훈련 매개변수
                 lr: float = 1e-4,
                 lr_decay_rate: float = 0.5,
                 lr_decay_steps: int = 50000,
                 weight_decay: float = 0.0,
                 train_batch_size: int = 4,
                 val_batch_size: int = 1,
                 
                 # 디버깅 매개변수
                 debug: bool = False,
                 vis_interval: int = 50,
                 debug_feature_interval: int = 200):
        """
        Args:
            # 모델 매개변수
            num_channels: 입력 이미지의 채널 수
            num_levels: 피라미드 레벨 수
            feature_channels: 기본 특징 채널 수
            use_cost_volume: 비용 볼륨 사용 여부
            max_displacement: 최대 변위 거리
            use_feature_warp: 특징 와핑 사용 여부
            context_channels: 문맥 특징의 채널 수
            flow_refinement_channels: 흐름 정제 네트워크의 채널 수
            dropout_rate: 드롭아웃 비율
            channel_multiplier: 채널 수 배수
            leaky_relu_alpha: LeakyReLU의 음수 기울기
            shared_flow_decoder: 공유 흐름 디코더 사용 여부
            
            # 손실 함수 매개변수
            photometric_weight: 포토메트릭 손실 가중치
            census_weight: Census 손실 가중치
            smoothness_weight: 평활화 손실 가중치
            use_occlusion: 가려짐 마스크 사용 여부
            use_valid_mask: 유효 영역 마스크 사용 여부
            use_stop_gradient: 그래디언트 흐름 제어 사용 여부
            use_bidirectional: 양방향 손실 계산 여부
            
            # 훈련 매개변수
            lr: 초기 학습률
            lr_decay_rate: 학습률 감소 비율
            lr_decay_steps: 학습률 감소 단계
            weight_decay: 가중치 감쇠
            train_batch_size: 훈련 배치 크기
            val_batch_size: 검증 배치 크기
            
            # 디버깅 매개변수
            debug: 디버깅 정보 출력 활성화
            vis_interval: 시각화 저장 간격 (단계 수)
            debug_feature_interval: 특징 시각화 저장 간격 (단계 수)
        """
        super(UFlowLightningModule, self).__init__()
        
        # 하이퍼파라미터 저장
        self.save_hyperparameters()
        
        # 모델 초기화
        self.model = UFlow(
            num_channels=num_channels,
            num_levels=num_levels,
            feature_channels=feature_channels,
            use_cost_volume=use_cost_volume,
            max_displacement=max_displacement,
            use_feature_warp=use_feature_warp,
            context_channels=context_channels,
            flow_refinement_channels=flow_refinement_channels,
            leaky_relu_alpha=leaky_relu_alpha,
            dropout_rate=dropout_rate,
            channel_multiplier=channel_multiplier,
            shared_flow_decoder=shared_flow_decoder
        )
        
        # 손실 함수 초기화
        self.criterion = losses.MultiScaleUFlowLoss(
            photometric_weight=photometric_weight,
            census_weight=census_weight,
            smoothness_weight=smoothness_weight,
            ssim_weight=0.0,
            window_size=7,
            occlusion_method='wang',
            edge_weighting=True,
            stop_gradient=use_stop_gradient,
            bidirectional=use_bidirectional
        )
        
        # 훈련/검증용 지표
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # 훈련 매개변수
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.weight_decay = weight_decay
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        
        # 디버깅 매개변수
        self.debug = debug
        self.vis_interval = vis_interval
        self.debug_feature_interval = debug_feature_interval
        
        # 디버그 로거 초기화
        self.debug_logger = None
    
    def forward(self, img1, img2):
        """모델 순전파"""
        return self.model(img1, img2)
    
    def configure_optimizers(self):
        """옵티마이저 및 학습률 스케줄러 설정"""
        # 학습률 낮추기 - 시작 학습률을 1e-5로 감소
        initial_lr = self.lr
        print(f"[옵티마이저 설정] 초기 학습률: {initial_lr}")
        
        optimizer = optim.Adam(
            self.parameters(), 
            lr=initial_lr, 
            weight_decay=self.weight_decay
        )
        
        # 학습률 스케줄러
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.lr_decay_steps, 
            gamma=self.lr_decay_rate
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step'  # 'epoch' 또는 'step'
            }
        }
    
    def training_step(self, batch, batch_idx):
        """훈련 단계"""
        # 데이터 추출
        img_t1 = batch['frame1']
        img_t2 = batch['frame2']
        
        # 순방향 및 역방향 흐름 계산
        forward_flows, backward_flows, features1, features2 = self.model.forward_backward_flow(img_t1, img_t2)
        
        # 현재 스텝
        global_step = self.global_step
        
        # 디버깅 모드에서 지정된 간격마다 모델 출력 및 그래디언트 흐름 체크
        if self.debug and self.debug_logger is not None and global_step % self.debug_feature_interval == 0:
            # 모델 출력 통계 확인
            self.debug_logger.log_model_stats(global_step, img_t1, img_t2, forward_flows)
            
            # 특징 피라미드 통계 기록
            self.debug_logger.log_feature_stats(global_step, features1, features2)
            
            # 특징 시각화 저장
            self.debug_logger.save_feature_visualization(global_step, img_t1, img_t2, features1, features2, forward_flows)
            
            # 그래디언트 흐름 체크
            self._check_gradient_flow(img_t1, img_t2, forward_flows, backward_flows)
        
        # 손실 계산
        losses = self.criterion(img_t1, img_t2, forward_flows, backward_flows)
        
        # 총 손실
        total_loss = losses['total_loss']
        
        # 로깅
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # 개별 손실 로깅
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.log(f'train_{key}', value, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        # 디버깅 모드에서 손실 정보를 1000 스텝마다 로깅
        if self.debug and self.debug_logger is not None and global_step % 1000 == 0:
            self.debug_logger.log_loss_info(global_step, losses)
        
        # 시각화 (vis_interval 스텝마다)
        if global_step % self.vis_interval == 0:
            self._save_visualizations(img_t1, img_t2, forward_flows, losses)
        
        # 결과 저장
        self.training_step_outputs.append(total_loss.detach())
        
        return total_loss
    
    def on_train_epoch_end(self):
        """훈련 에폭 종료 시 호출"""
        # 평균 손실 계산
        if self.training_step_outputs:
            epoch_loss = torch.stack(self.training_step_outputs).mean()
            self.log('train_epoch_loss', epoch_loss, prog_bar=True, logger=True, sync_dist=True)
            self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        """검증 단계"""
        # 데이터 추출
        img_t1 = batch['frame1']
        img_t2 = batch['frame2']
        
        # 모델을 eval 모드로 전환하고 순방향 및 역방향 흐름 계산
        self.model.eval()
        with torch.no_grad():
            forward_flows, backward_flows, _, _ = self.model.forward_backward_flow(img_t1, img_t2)
            
            # 손실 계산
            losses = self.criterion(img_t1, img_t2, forward_flows, backward_flows)
            
            # 총 손실
            total_loss = losses['total_loss']
        
        # 로깅
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # 개별 손실 로깅
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.log(f'val_{key}', value, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        # 결과 저장
        self.validation_step_outputs.append(total_loss.detach())
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """검증 에폭 종료 시 호출"""
        # 평균 손실 계산
        if self.validation_step_outputs:
            epoch_loss = torch.stack(self.validation_step_outputs).mean()
            self.log('val_epoch_loss', epoch_loss, prog_bar=True, logger=True, sync_dist=True)
            self.validation_step_outputs.clear()

    def loss_gradient_check(self, img1, img2, flow_forward, flow_backward):
        """
        손실 함수의 그래디언트 흐름을 확인하는 메서드
        디버깅 목적으로 사용
        
        Args:
            img1 (torch.Tensor): 첫 번째 이미지
            img2 (torch.Tensor): 두 번째 이미지
            flow_forward (list): 순방향 광학 흐름 리스트
            flow_backward (list): 역방향 광학 흐름 리스트
        """
        # 그래디언트 확인을 위해 새로운 텐서 생성
        flow_check = flow_forward[0].clone().detach().requires_grad_(True)
        
        # 단일 스케일 손실 계산 (가장 높은 해상도)
        with torch.enable_grad():
            # 임시 손실 함수
            photo_loss = losses.PhotometricLoss(alpha=0.0)  # SSIM 없이 순수 L1만 사용
            
            # 이미지 와핑
            warped_img = utils.warp_image(img2, flow_check)
            
            # L1 손실 계산
            loss = torch.mean(torch.abs(warped_img - img1))
            
            # 그래디언트 계산
            loss.backward()
            
            # 그래디언트 확인
            if flow_check.grad is None:
                print("[심각] 손실 함수의 그래디언트가 None입니다!")
            else:
                flow_grad_norm = flow_check.grad.norm().item()
                flow_norm = flow_check.norm().item()
                ratio = flow_grad_norm / (flow_norm + 1e-8)
                
                print(f"[그래디언트 체크] 단순 L1 손실 그래디언트 테스트:")
                print(f"  흐름 norm: {flow_norm:.6f}")
                print(f"  그래디언트 norm: {flow_grad_norm:.6f}")
                print(f"  비율: {ratio:.6f}")
                
                if flow_grad_norm < 1e-6:
                    print("  [경고] 그래디언트가 너무 작습니다! 그래디언트 흐름에 문제가 있을 수 있습니다.")
                    
                if torch.isnan(flow_check.grad).any() or torch.isinf(flow_check.grad).any():
                    print("  [심각] 그래디언트에 NaN 또는 Inf 값이 있습니다!")

    def check_model_outputs(self, img1, img2, flows):
        """
        모델 출력 통계를 확인하는 메서드
        디버깅 목적으로 사용
        
        Args:
            img1 (torch.Tensor): 첫 번째 이미지
            img2 (torch.Tensor): 두 번째 이미지
            flows (list): 광학 흐름 피라미드
        """
        print("\n[모델 출력 통계]")
        
        # 이미지 통계
        print(f"이미지 1 범위: {img1.min():.4f} ~ {img1.max():.4f}, 평균: {img1.mean():.4f}")
        print(f"이미지 2 범위: {img2.min():.4f} ~ {img2.max():.4f}, 평균: {img2.mean():.4f}")
        
        # 흐름 통계
        for i, flow in enumerate(flows):
            print(f"피라미드 레벨 {i} 흐름 크기: {flow.shape}")
            print(f"  범위: {flow.min():.4f} ~ {flow.max():.4f}, 평균 변위: {flow.abs().mean():.4f}")
            
            # 흐름 크기 (픽셀 변위)
            flow_mag = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
            print(f"  변위 크기 - 최소: {flow_mag.min():.4f}, 최대: {flow_mag.max():.4f}, 평균: {flow_mag.mean():.4f}")
            
            # 문제가 있는지 확인
            if torch.isnan(flow).any():
                print(f"  [심각] 레벨 {i} 흐름에 NaN 값이 있습니다!")
            if torch.isinf(flow).any():
                print(f"  [심각] 레벨 {i} 흐름에 Inf 값이 있습니다!")
            
            # 모델이 학습 중인지 확인 (흐름 값이 모두 0에 가까운 경우 의심)
            if flow.abs().mean() < 1e-4:
                print(f"  [경고] 레벨 {i} 흐름이 거의 0입니다. 모델이 제대로 학습되지 않을 수 있습니다.")

    def on_fit_start(self):
        """훈련 시작 시 호출되는 메서드 - 디버그 로거 초기화"""
        # 디버그 로거 초기화
        if self.debug:
            log_dir = self.logger.log_dir if self.logger is not None else './logs'
            self.debug_logger = DebugLogger(log_dir, enabled=self.debug)
            print(f"[디버그] 디버그 로거가 초기화되었습니다. 경로: {log_dir}")
            if self.debug_logger is not None:
                self.debug_logger.log_info("UFlow 훈련 시작")
        else:
            self.debug_logger = DebugLogger(None, enabled=False)
            print("[디버그] 디버그 모드가 비활성화되어 있습니다.")
    
    def _check_gradient_flow(self, img1, img2, forward_flows, backward_flows):
        """
        그래디언트 흐름을 확인하는 메서드 (디버깅용)
        
        Args:
            img1: 첫 번째 이미지
            img2: 두 번째 이미지
            forward_flows: 순방향 광학 흐름 리스트
            backward_flows: 역방향 광학 흐름 리스트
        """
        # 그래디언트 확인을 위해 새로운 텐서 생성
        flow_check = forward_flows[0].clone().detach().requires_grad_(True)
        
        # 단일 스케일 손실 계산 (가장 높은 해상도)
        with torch.enable_grad():
            try:
                # 이미지 와핑
                warped_img = utils.warp_image(img2, flow_check)
                
                # L1 손실 계산
                loss = torch.mean(torch.abs(warped_img - img1))
                
                # 그래디언트 계산
                loss.backward()
                
                # 그래디언트 확인
                if flow_check.grad is None:
                    self.debug_logger.log_error("[심각] 손실 함수의 그래디언트가 None입니다!")
                else:
                    flow_grad_norm = flow_check.grad.norm().item()
                    flow_norm = flow_check.norm().item()
                    ratio = flow_grad_norm / (flow_norm + 1e-8)
                    
                    # 그래디언트 흐름 체크 결과 기록
                    self.debug_logger.log_gradient_flow_check(
                        self.global_step, flow_norm, flow_grad_norm, ratio
                    )
                    
                    # 그래디언트에 NaN/Inf 확인
                    if torch.isnan(flow_check.grad).any() or torch.isinf(flow_check.grad).any():
                        self.debug_logger.log_error("[심각] 그래디언트에 NaN 또는 Inf 값이 있습니다!")
            except Exception as e:
                self.debug_logger.log_error(f"그래디언트 흐름 확인 중 오류 발생: {str(e)}")

    def _save_visualizations(self, img_t1, img_t2, forward_flows, losses):
        """
        현재 상태 시각화를 저장하는 메서드
        여러 GPU 프로세스 간 충돌 방지를 위해 rank 0에서만 저장
        
        Args:
            img_t1: 첫 번째 이미지
            img_t2: 두 번째 이미지
            forward_flows: 순방향 광학 흐름 리스트
            losses: 손실 딕셔너리
        """
        # 현재 프로세스의 랭크 확인
        local_rank = 0
        is_master = True
        
        # 분산 환경에서 실행 중인지 확인
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'strategy'):
            if hasattr(self.trainer.strategy, 'local_rank'):
                local_rank = self.trainer.strategy.local_rank
                is_master = local_rank == 0
            elif hasattr(self.trainer.strategy, 'root_device'):
                local_rank = self.trainer.strategy.root_device.index
                is_master = local_rank == 0
        
        # 마스터 프로세스(rank 0)에서만 시각화 저장
        if not is_master:
            if self.debug:
                self.debug_logger.log_info(f"GPU {local_rank}: 시각화 건너뜀 (마스터 프로세스가 아님)")
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # GUI 없이 이미지 저장
            import numpy as np
            import os
            
            # 디렉토리 생성
            vis_dir = os.path.join(self.logger.log_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 배치에서 첫 번째 이미지만 시각화
            idx = 0
            img1 = img_t1[idx].detach().cpu().permute(1, 2, 0).numpy()
            img2 = img_t2[idx].detach().cpu().permute(1, 2, 0).numpy()
            flow = forward_flows[0][idx].detach().cpu().permute(1, 2, 0).numpy()
            
            # 와핑된 이미지 계산
            warped_img2 = utils.warp_image(img_t2, forward_flows[0])
            warped_img2 = warped_img2[idx].detach().cpu().permute(1, 2, 0).numpy()
            
            # 옵션: 가려짐 마스크 시각화 (있는 경우)
            occlusion_mask = None
            if 'scale_0_occlusion_mask' in losses:
                occlusion_mask = losses['scale_0_occlusion_mask'][idx, 0].detach().cpu().numpy()
            
            # 광학 흐름 시각화 함수
            def flow_to_color(flow):
                try:
                    import cv2
                    # 광학 흐름 시각화를 위한 간단한 함수
                    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
                    hsv[..., 1] = 255
                    
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    return rgb
                except:
                    # cv2 없는 경우 간단한 시각화
                    viz = np.zeros_like(flow)
                    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    viz[..., 0] = np.clip(flow[..., 0] / (flow_mag.max() + 1e-8) * 0.5 + 0.5, 0, 1)
                    viz[..., 1] = np.clip(flow[..., 1] / (flow_mag.max() + 1e-8) * 0.5 + 0.5, 0, 1)
                    viz[..., 2] = np.clip(flow_mag / (flow_mag.max() + 1e-8), 0, 1)
                    return viz
            
            # 광학 흐름 시각화
            flow_viz = flow_to_color(flow)
            
            # 시각화 생성
            plt.figure(figsize=(15, 10))
            
            # 원본 이미지 및 와핑된 이미지
            plt.subplot(2, 3, 1)
            plt.imshow(np.clip(img1, 0, 1))
            plt.title('Image 1')
            plt.axis('off')
            
            plt.subplot(2, 3, 2)
            plt.imshow(np.clip(img2, 0, 1))
            plt.title('Image 2')
            plt.axis('off')
            
            plt.subplot(2, 3, 3)
            plt.imshow(np.clip(warped_img2, 0, 1))
            plt.title('Warped Image 2')
            plt.axis('off')
            
            # 와핑 결과 시각적 평가
            error = np.abs(img1 - warped_img2)
            error = np.clip(error / (error.max() + 1e-8), 0, 1)
            
            plt.subplot(2, 3, 4)
            plt.imshow(flow_viz)
            plt.title('Optical Flow')
            plt.axis('off')
            
            plt.subplot(2, 3, 5)
            plt.imshow(error)
            plt.title('Warping Error')
            plt.axis('off')
            
            if occlusion_mask is not None:
                plt.subplot(2, 3, 6)
                plt.imshow(occlusion_mask, cmap='viridis')
                plt.title('Occlusion Mask')
                plt.axis('off')
            
            # 시각화 저장 (안전하게 저장하기 위해 파일 이름에 프로세스 랭크 정보 추가)
            plt.tight_layout()
            
            # 파일 경로 생성
            viz_filename = f'step_{self.global_step:06d}.png'
            loss_filename = f'step_{self.global_step:06d}_loss.txt'
            
            # 시각화 이미지 저장
            plt.savefig(os.path.join(vis_dir, viz_filename))
            plt.close()
            
            # 손실값 기록
            with open(os.path.join(vis_dir, loss_filename), 'w') as f:
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        f.write(f"{key}: {value.item():.6f}\n")
            
            # 디버그 모드에서만 로그 출력
            if self.debug:
                self.debug_logger.log_info(f"GPU {local_rank}(마스터): 시각화 저장 완료: {vis_dir}/{viz_filename}")
                
                # 디버그 모드에서 특징 피라미드 시각화 저장 (모델에서 features1, features2를 가져올 수 있을 때)
                if hasattr(self, 'model') and hasattr(self.model, 'forward_backward_flow'):
                    # 현재 이미지로 특징 피라미드 계산
                    with torch.no_grad():
                        _, _, features1, features2 = self.model.forward_backward_flow(img_t1, img_t2)
                        
                        # 특징 피라미드 시각화 저장 디렉토리 생성
                        features_dir = os.path.join(vis_dir, 'features')
                        os.makedirs(features_dir, exist_ok=True)
                        
                        # 특징 피라미드 시각화 호출
                        self.debug_logger.save_feature_visualization(
                            self.global_step, 
                            img_t1, 
                            img_t2, 
                            features1, 
                            features2, 
                            forward_flows
                        )
                        
                        self.debug_logger.log_info(f"GPU {local_rank}(마스터): 특징 시각화 저장 완료: {features_dir}")
            
        except Exception as e:
            if self.debug:
                self.debug_logger.log_error(f"GPU {local_rank}: 시각화 생성 중 오류 발생: {str(e)}")
            else:
                # 디버그 모드가 아닐 때는 조용히 오류 처리
                pass


def parse_args():
    parser = argparse.ArgumentParser(description='UFlow 훈련 스크립트')
    
    # 데이터 관련 인자
    parser.add_argument('--data_dir', type=str, default=None, help='데이터셋 디렉토리 (하위 호환성 유지)')
    parser.add_argument('--val_data_dir', type=str, default=None, help='검증 데이터셋 디렉토리 (하위 호환성 유지)')
    parser.add_argument('--train_list_path', type=str, default="train.txt", help='훈련 데이터 경로 목록 파일 (train.txt)')
    parser.add_argument('--val_list_path', type=str, default="validation.txt", help='검증 데이터 경로 목록 파일 (validation.txt)')
    parser.add_argument('--target_height', type=int, default=192, help='처리 후 이미지 높이')
    parser.add_argument('--target_width', type=int, default=256, help='처리 후 이미지 너비')
    parser.add_argument('--convert_to_rgb', action='store_true', default=True, help='Bayer RAW를 RGB로 변환')
    parser.add_argument('--exclude_ev_minus', action='store_true', default=True, help='ev minus 프레임(인덱스 1, 2, 3) 제외')
    parser.add_argument('--apply_pregamma', action='store_true', default=True, help='어두운 이미지 보정을 위한 pre-gamma 적용')
    parser.add_argument('--pregamma_value', type=float, default=2.0, help='Pre-gamma 보정 값 (기본값: 2.0)')
    
    # 데이터 증강 관련 인자
    parser.add_argument('--use_augmentation', action='store_true', default=True, help='데이터 증강 사용')
    parser.add_argument('--use_photometric', action='store_true', default=True, help='색상 변환 증강 적용')
    parser.add_argument('--use_geometric', action='store_true', default=False, help='기하학적 변환 증강 적용')
    parser.add_argument('--val_augmentation', action='store_true', default=False, help='검증 데이터에도 증강 적용')
    
    # 모델 관련 인자
    parser.add_argument('--num_channels', type=int, default=3, help='입력 이미지 채널 수')
    parser.add_argument('--num_levels', type=int, default=5, help='피라미드 레벨 수')
    parser.add_argument('--feature_channels', type=int, default=32, help='기본 특징 채널 수')
    parser.add_argument('--use_cost_volume', action='store_false', default=True, help='비용 볼륨 사용 안함')
    parser.add_argument('--max_displacement', type=int, default=4, help='최대 변위 거리')
    parser.add_argument('--use_feature_warp', action='store_true', help='특징 와핑 사용')
    parser.add_argument('--context_channels', type=int, default=32, help='문맥 특징 채널 수')
    parser.add_argument('--flow_refinement_channels', type=int, default=128, help='흐름 정제 채널 수')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='드롭아웃 비율')
    parser.add_argument('--channel_multiplier', type=float, default=1.0, help='채널 수 배수')
    parser.add_argument('--leaky_relu_alpha', type=float, default=0.1, help='LeakyReLU 기울기')
    parser.add_argument('--shared_flow_decoder', action='store_true', help='공유 흐름 디코더 사용')
    
    # 손실 함수 관련 인자
    parser.add_argument('--photometric_weight', type=float, default=1.0, help='포토메트릭 손실 가중치')
    parser.add_argument('--census_weight', type=float, default=1.0, help='센서스 손실 가중치')
    parser.add_argument('--smoothness_weight', type=float, default=2.0, help='평활화 손실 가중치')
    parser.add_argument('--use_occlusion', action='store_false', dest='use_occlusion', help='가려짐 마스크 사용 안함')
    parser.add_argument('--use_valid_mask', action='store_false', dest='use_valid_mask', help='유효 마스크 사용 안함')
    parser.add_argument('--use_stop_gradient', action='store_false', dest='use_stop_gradient', help='그래디언트 흐름 제어 사용 안함')
    parser.add_argument('--use_bidirectional', action='store_false', dest='use_bidirectional', help='양방향 손실 계산 안함')
    
    # 훈련 관련 인자
    parser.add_argument('--train_batch_size', type=int, default=8, help='훈련 배치 크기')
    parser.add_argument('--val_batch_size', type=int, default=8, help='검증 배치 크기')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로더 워커 수')
    parser.add_argument('--lr', type=float, default=1e-4, help='초기 학습률')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='학습률 감소 비율')
    parser.add_argument('--lr_decay_steps', type=int, default=50000, help='학습률 감소 단계')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='가중치 감쇠')
    parser.add_argument('--epochs', type=int, default=1000, help='훈련 에폭 수')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='검증 체크 간격 (에폭의 비율 또는 단계 수)')
    
    # 디버깅 및 시각화 관련 인자
    parser.add_argument('--debug', action='store_true', help='디버깅 정보 출력 활성화')
    parser.add_argument('--vis_interval', type=int, default=50, help='시각화 저장 간격 (단계 수)')
    parser.add_argument('--debug_feature_interval', type=int, default=200, help='특징 시각화 저장 간격 (단계 수)')
    
    # 기타 인자
    parser.add_argument('--checkpoint_dir', type=str, default='/group-volume/sdp-aiip-night/dongmin/models/mpi_training/uflow/', help='체크포인트 저장 디렉토리')
    parser.add_argument('--log_dir', type=str, default='logs', help='로그 저장 디렉토리')
    parser.add_argument('--resume', type=str, default=None, help='체크포인트에서 훈련 재개')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--gpu', type=int, nargs='+', default=None, help='사용할 GPU ID')
    parser.add_argument('--precision', type=str, default='32', help='계산 정밀도 (16, 32, bf16)')
    
    return parser.parse_args()


def seed_everything(seed):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 인자 파싱
    args = parse_args()
    
    # 재현성을 위한 시드 설정
    seed_everything(args.seed)
    
    # 모델 체크포인트 콜백
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='uflow-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    # 학습률 모니터 콜백
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # 로거 설정
    logger = TensorBoardLogger(save_dir=args.log_dir, name='uflow')
    
    # 훈련 데이터 로더 생성
    # 새 방식(file_list_path)과 기존 방식(data_dir) 중 하나 선택
    if args.train_list_path is not None:
        print(f"훈련 데이터 경로 목록 사용: {args.train_list_path}")
        train_dataloader = create_dataloader(
            file_list_path=args.train_list_path,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            target_height=args.target_height,
            target_width=args.target_width,
            convert_to_rgb=args.convert_to_rgb,
            use_augmentation=args.use_augmentation,
            use_photometric=args.use_photometric,
            use_geometric=args.use_geometric,
            exclude_ev_minus=args.exclude_ev_minus,
            apply_pregamma=args.apply_pregamma,
            pregamma_value=args.pregamma_value
        )
    else:
        print(f"훈련 데이터 디렉토리 사용: {args.data_dir}")
        train_dataloader = create_dataloader(
            data_dir=args.data_dir,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            target_height=args.target_height,
            target_width=args.target_width,
            convert_to_rgb=args.convert_to_rgb,
            use_augmentation=args.use_augmentation,
            use_photometric=args.use_photometric,
            use_geometric=args.use_geometric,
            exclude_ev_minus=args.exclude_ev_minus,
            apply_pregamma=args.apply_pregamma,
            pregamma_value=args.pregamma_value
        )
    
    # 검증 데이터 로더
    val_dataloader = None
    
    # 새 방식(file_list_path)과 기존 방식(data_dir) 중 하나 선택
    if args.val_list_path is not None:
        print(f"검증 데이터 경로 목록 사용: {args.val_list_path}")
        val_dataloader = create_dataloader(
            file_list_path=args.val_list_path,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            target_height=args.target_height,
            target_width=args.target_width,
            convert_to_rgb=args.convert_to_rgb,
            use_augmentation=args.val_augmentation,
            use_photometric=args.use_photometric,
            use_geometric=args.use_geometric,
            exclude_ev_minus=args.exclude_ev_minus,
            apply_pregamma=args.apply_pregamma,
            pregamma_value=args.pregamma_value
        )
    elif args.val_data_dir is not None:
        print(f"검증 데이터 디렉토리 사용: {args.val_data_dir}")
        val_dataloader = create_dataloader(
            data_dir=args.val_data_dir,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            target_height=args.target_height,
            target_width=args.target_width,
            convert_to_rgb=args.convert_to_rgb,
            use_augmentation=args.val_augmentation,
            use_photometric=args.use_photometric,
            use_geometric=args.use_geometric,
            exclude_ev_minus=args.exclude_ev_minus,
            apply_pregamma=args.apply_pregamma,
            pregamma_value=args.pregamma_value
        )
    
    # 모델 초기화
    model = UFlowLightningModule(
        # 모델 매개변수
        num_channels=args.num_channels,
        num_levels=args.num_levels,
        feature_channels=args.feature_channels,
        use_cost_volume=args.use_cost_volume,
        max_displacement=args.max_displacement,
        use_feature_warp=args.use_feature_warp,
        context_channels=args.context_channels,
        flow_refinement_channels=args.flow_refinement_channels,
        dropout_rate=args.dropout_rate,
        channel_multiplier=args.channel_multiplier,
        leaky_relu_alpha=args.leaky_relu_alpha,
        shared_flow_decoder=args.shared_flow_decoder,
        
        # 손실 함수 매개변수
        photometric_weight=args.photometric_weight,
        census_weight=args.census_weight,
        smoothness_weight=args.smoothness_weight,
        use_occlusion=args.use_occlusion,
        use_valid_mask=args.use_valid_mask,
        use_stop_gradient=args.use_stop_gradient,
        use_bidirectional=args.use_bidirectional,
        
        # 훈련 매개변수
        lr=args.lr,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_steps=args.lr_decay_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        
        # 디버깅 매개변수
        debug=args.debug,
        vis_interval=args.vis_interval,
        debug_feature_interval=args.debug_feature_interval
    )
    
    # 훈련 설정
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        accelerator='gpu' if args.gpu is not None else 'auto',
        devices=args.gpu if args.gpu is not None else 'auto',
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        strategy='ddp_find_unused_parameters_true'
    )
    
    # 훈련 시작 (체크포인트에서 재개할 경우 ckpt_path 사용)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.resume)


if __name__ == '__main__':
    main() 