#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from models import UFlow
import losses
from dataloader import create_dataloader


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
                 vis_interval: int = 500):
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
    
    def forward(self, img1, img2):
        """모델 순전파"""
        return self.model(img1, img2)
    
    def configure_optimizers(self):
        """옵티마이저 및 학습률 스케줄러 설정"""
        # 학습률 낮추기 - 시작 학습률을 1e-5로 감소
        initial_lr = 1e-5
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
    
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """옵티마이저 스텝 실행 전 호출 - 그래디언트 클리핑 적용"""
        # 1.0 임계값으로 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # 디버깅 모드에서 100 스텝마다 그래디언트 노름 확인
        if self.debug and self.global_step % 100 == 0:
            grad_norms = []
            for param in self.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
            
            if grad_norms:
                avg_norm = sum(grad_norms) / len(grad_norms)
                max_norm = max(grad_norms)
                print(f"[그래디언트 클리핑] 평균 노름: {avg_norm:.6f}, 최대 노름: {max_norm:.6f}")
            else:
                print("[그래디언트 클리핑] 그래디언트가 없습니다!")
    
    def training_step(self, batch, batch_idx):
        """훈련 단계"""
        # 데이터 추출
        img_t1 = batch['frame1']
        img_t2 = batch['frame2']
        
        # 순방향 및 역방향 흐름 계산
        forward_flows, backward_flows, features1, features2 = self.model.forward_backward_flow(img_t1, img_t2)
        
        # 디버깅 모드에서 가끔씩 그래디언트 흐름 체크 (1000 스텝마다)
        if self.debug and self.global_step % 1000 == 0:
            # 모델 출력 통계 확인
            self.check_model_outputs(img_t1, img_t2, forward_flows)
            # 그래디언트 흐름 체크
            self.loss_gradient_check(img_t1, img_t2, forward_flows, backward_flows)
        
        # 손실 계산
        losses = self.criterion(img_t1, img_t2, forward_flows, backward_flows)
        
        # 총 손실
        total_loss = losses['total_loss']
        
        # 로깅
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # 개별 손실 로깅
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.log(f'train_{key}', value, on_step=False, on_epoch=True, logger=True)
        
        # 현재 스텝
        global_step = self.global_step
        
        # 디버깅 모드에서 그래디언트 및 파라미터 통계 확인 (500 스텝마다)
        if self.debug and global_step % 500 == 0:
            param_stats = []
            print("\n[파라미터 및 그래디언트 통계]")
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    param_norm = param.norm().item()
                    param_stats.append({
                        'name': name,
                        'grad_norm': grad_norm,
                        'param_norm': param_norm,
                        'ratio': grad_norm / (param_norm + 1e-8)
                    })
                    
                    # NaN 또는 Inf 체크
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"[심각] {name}에 NaN 또는 Inf 값이 있습니다!")
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"[심각] {name}.grad에 NaN 또는 Inf 값이 있습니다!")
                
            # Top-5 가장 큰 그래디언트 비율을 가진 레이어 출력
            param_stats.sort(key=lambda x: x['ratio'], reverse=True)
            print("\n[Top-5 그래디언트/파라미터 비율]")
            for i, stat in enumerate(param_stats[:5]):
                print(f"{i+1}. {stat['name']}: 그래디언트 {stat['grad_norm']:.6f}, 파라미터 {stat['param_norm']:.6f}, 비율 {stat['ratio']:.6f}")
            
            # 그래디언트가 0인 레이어 수 확인
            zero_grads = sum(1 for stat in param_stats if stat['grad_norm'] < 1e-8)
            print(f"\n전체 레이어 수: {len(param_stats)}, 그래디언트가 0인 레이어 수: {zero_grads}")
            
            # 손실 통계 출력
            print("\n[손실 통계]")
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    print(f"{key}: {value.item():.6f}")
        
        # 시각화 (vis_interval 스텝마다)
        if global_step % self.vis_interval == 0:
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
                import utils
                warped_img2 = utils.warp_image(img_t2, forward_flows[0])
                warped_img2 = warped_img2[idx].detach().cpu().permute(1, 2, 0).numpy()
                
                # 옵션: 가려짐 마스크 시각화 (있는 경우)
                occlusion_mask = None
                if 'scale_0_occlusion_mask' in losses:
                    occlusion_mask = losses['scale_0_occlusion_mask'][idx, 0].detach().cpu().numpy()
                
                # 광학 흐름 시각화 함수
                def flow_to_color(flow):
                    # 광학 흐름 시각화를 위한 간단한 함수
                    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
                    hsv[..., 1] = 255
                    
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    return rgb
                
                try:
                    import cv2
                    flow_viz = flow_to_color(flow)
                except:
                    # cv2 없는 경우 간단한 시각화
                    flow_viz = np.zeros_like(img1)
                    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    flow_viz[..., 0] = flow[..., 0] / (flow_mag.max() + 1e-8) * 0.5 + 0.5
                    flow_viz[..., 1] = flow[..., 1] / (flow_mag.max() + 1e-8) * 0.5 + 0.5
                
                # 시각화 생성
                plt.figure(figsize=(15, 10))
                
                # 원본 이미지 및 와핑된 이미지
                plt.subplot(2, 3, 1)
                plt.imshow(img1)
                plt.title('Image 1')
                plt.axis('off')
                
                plt.subplot(2, 3, 2)
                plt.imshow(img2)
                plt.title('Image 2')
                plt.axis('off')
                
                plt.subplot(2, 3, 3)
                plt.imshow(warped_img2)
                plt.title('Warped Image 2')
                plt.axis('off')
                
                # 와핑 결과 시각적 평가
                error = np.abs(img1 - warped_img2)
                error = error / error.max()
                
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
                
                # 시각화 저장
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'step_{global_step:06d}.png'))
                plt.close()
                
                # 손실값 기록
                with open(os.path.join(vis_dir, f'step_{global_step:06d}_loss.txt'), 'w') as f:
                    for key, value in losses.items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            f.write(f"{key}: {value.item():.6f}\n")
                            
                if self.debug:
                    print(f"시각화 저장 완료: {vis_dir}/step_{global_step:06d}.png")
                
            except Exception as e:
                print(f"시각화 생성 중 오류 발생: {e}")
        
        # 결과 저장
        self.training_step_outputs.append(total_loss.detach())
        
        return total_loss
    
    def on_train_epoch_end(self):
        """훈련 에폭 종료 시 호출"""
        # 평균 손실 계산
        if self.training_step_outputs:
            epoch_loss = torch.stack(self.training_step_outputs).mean()
            self.log('train_epoch_loss', epoch_loss, prog_bar=True, logger=True)
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
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # 개별 손실 로깅
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.log(f'val_{key}', value, on_step=False, on_epoch=True, logger=True)
        
        # 결과 저장
        self.validation_step_outputs.append(total_loss.detach())
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """검증 에폭 종료 시 호출"""
        # 평균 손실 계산
        if self.validation_step_outputs:
            epoch_loss = torch.stack(self.validation_step_outputs).mean()
            self.log('val_epoch_loss', epoch_loss, prog_bar=True, logger=True)
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


def parse_args():
    parser = argparse.ArgumentParser(description='UFlow 훈련 스크립트')
    
    # 데이터 관련 인자
    parser.add_argument('--data_dir', type=str, default='/data-sets/sdp-aiip-night/dataset/S25/MPI_20250426_RG/training/', help='데이터셋 디렉토리')
    parser.add_argument('--val_data_dir', type=str, default=None, help='검증 데이터셋 디렉토리 (기본값: None, 훈련 데이터 일부 사용)')
    parser.add_argument('--target_height', type=int, default=192, help='처리 후 이미지 높이')
    parser.add_argument('--target_width', type=int, default=256, help='처리 후 이미지 너비')
    parser.add_argument('--convert_to_rgb', action='store_true', default=True, help='Bayer RAW를 RGB로 변환')
    parser.add_argument('--exclude_ev_minus', action='store_true', default=True, help='ev minus 프레임(인덱스 1, 2, 3) 제외')
    
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
    parser.add_argument('--photometric_weight', type=float, default=0.0, help='포토메트릭 손실 가중치')
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
    
    # 데이터 로더 생성
    train_dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        target_height=args.target_height,
        target_width=args.target_width,
        convert_to_rgb=args.convert_to_rgb,
        use_augmentation=True,
        use_photometric=True,
        use_geometric=False,
        exclude_ev_minus=args.exclude_ev_minus
    )
    
    # 검증 데이터 로더
    if args.val_data_dir:
        val_dataloader = create_dataloader(
            data_dir=args.val_data_dir,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            target_height=args.target_height,
            target_width=args.target_width,
            convert_to_rgb=args.convert_to_rgb,
            use_augmentation=False,
            exclude_ev_minus=args.exclude_ev_minus
        )
    else:
        # 훈련 데이터의 일부를 검증에 사용
        val_dataloader = None
    
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
        vis_interval=args.vis_interval
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