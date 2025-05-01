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
                 photometric_weight: float = 1.0,
                 census_weight: float = 1.0, 
                 smoothness_weight: float = 0.1,
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
                 val_batch_size: int = 1):
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
            ssim_weight=0.85,
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
    
    def forward(self, img1, img2):
        """모델 순전파"""
        return self.model(img1, img2)
    
    def configure_optimizers(self):
        """옵티마이저 및 학습률 스케줄러 설정"""
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.lr, 
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
        forward_flows, backward_flows, _, _ = self.model.forward_backward_flow(img_t1, img_t2)
        
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
    parser.add_argument('--photometric_weight', type=float, default=1.0, help='포토메트릭 손실 가중치')
    parser.add_argument('--census_weight', type=float, default=1.0, help='센서스 손실 가중치')
    parser.add_argument('--smoothness_weight', type=float, default=0.1, help='평활화 손실 가중치')
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
        val_batch_size=args.val_batch_size
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