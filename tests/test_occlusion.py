import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses import UFlowLoss
import utils

from utils import estimate_occlusion_mask
from PWCNet import PWCFlow

def create_complex_test_flow(batch_size, height, width, flow_type='rotation'):
    """
    다양한 복잡한 패턴의 광학 흐름 생성
    
    Args:
        batch_size (int): 배치 크기
        height (int): 높이
        width (int): 너비
        flow_type (str): 흐름 패턴 유형
            - 'rotation': 회전 운동
            - 'zoom': 확대/축소
            - 'discontinuous': 불연속적인 움직임
    """
    # 기본 그리드 생성
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    grid = torch.stack([x, y]).float()
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # 중심점 계산
    center_x = width / 2
    center_y = height / 2
    
    # 중심점으로부터의 상대 좌표
    dx = grid[:, 0] - center_x
    dy = grid[:, 1] - center_y
    
    flow_forward = torch.zeros_like(grid)
    flow_backward = torch.zeros_like(grid)
    
    if flow_type == 'rotation':
        # 회전 각도 (라디안)
        theta = np.pi / 6  # 30도
        
        # 순방향 회전
        flow_forward[:, 0] = dx * np.cos(theta) + dy * np.sin(theta) - dx
        flow_forward[:, 1] = -dx * np.sin(theta) + dy * np.cos(theta) - dy
        
        # 역방향 회전 (약간의 오차 포함)
        theta_back = -theta * 0.95  # 5% 오차
        flow_backward[:, 0] = dx * np.cos(theta_back) + dy * np.sin(theta_back) - dx
        flow_backward[:, 1] = -dx * np.sin(theta_back) + dy * np.cos(theta_back) - dy
        
    elif flow_type == 'zoom':
        # 확대/축소 계수
        scale_forward = 1.5
        scale_backward = 1.0 / 1.45  # 약간의 오차 포함
        
        # 순방향 확대
        flow_forward[:, 0] = dx * (scale_forward - 1)
        flow_forward[:, 1] = dy * (scale_forward - 1)
        
        # 역방향 축소
        flow_backward[:, 0] = dx * (scale_backward - 1)
        flow_backward[:, 1] = dy * (scale_backward - 1)
        
    elif flow_type == 'discontinuous':
        # 이미지를 4개의 영역으로 나누고 각각 다른 움직임 적용
        mask1 = (grid[:, 0] < width/2) & (grid[:, 1] < height/2)
        mask2 = (grid[:, 0] >= width/2) & (grid[:, 1] < height/2)
        mask3 = (grid[:, 0] < width/2) & (grid[:, 1] >= height/2)
        mask4 = (grid[:, 0] >= width/2) & (grid[:, 1] >= height/2)
        
        # 각 영역별 다른 움직임 설정
        # 영역 1: 오른쪽으로 이동
        flow_forward[:, 0][mask1] = 10.0
        flow_backward[:, 0][mask1] = -9.5
        
        # 영역 2: 왼쪽으로 이동
        flow_forward[:, 0][mask2] = -10.0
        flow_backward[:, 0][mask2] = 9.5
        
        # 영역 3: 위로 이동
        flow_forward[:, 1][mask3] = -10.0
        flow_backward[:, 1][mask3] = 9.5
        
        # 영역 4: 아래로 이동
        flow_forward[:, 1][mask4] = 10.0
        flow_backward[:, 1][mask4] = -9.5
    
    # 노이즈 추가 (더 적은 노이즈)
    noise = torch.randn_like(flow_forward) * 0.2
    flow_forward += noise
    flow_backward += noise
    
    return flow_forward, flow_backward

def analyze_occlusion_mask(occlusion_mask, flow_forward, flow_backward):
    """
    Occlusion mask의 통계적 분석
    
    Args:
        occlusion_mask (torch.Tensor): [B, 1, H, W]
        flow_forward (torch.Tensor): [B, 2, H, W]
        flow_backward (torch.Tensor): [B, 2, H, W]
    """
    # Occlusion 비율
    occluded_ratio = (occlusion_mask < 0.5).float().mean().item()
    print(f"가려진 영역 비율: {occluded_ratio:.2%}")
    
    # Flow consistency 분석
    flow_forward_magnitude = torch.sqrt(flow_forward[:, 0]**2 + flow_forward[:, 1]**2)
    flow_backward_magnitude = torch.sqrt(flow_backward[:, 0]**2 + flow_backward[:, 1]**2)
    
    # 가려진 영역과 가려지지 않은 영역에서의 flow 크기 비교
    occluded_flow_mag = (flow_forward_magnitude * (1 - occlusion_mask[:, 0])).sum() / ((1 - occlusion_mask[:, 0]).sum() + 1e-6)
    visible_flow_mag = (flow_forward_magnitude * occlusion_mask[:, 0]).sum() / (occlusion_mask[:, 0].sum() + 1e-6)
    
    print(f"가려진 영역의 평균 flow 크기: {occluded_flow_mag.item():.4f}")
    print(f"가려지지 않은 영역의 평균 flow 크기: {visible_flow_mag.item():.4f}")
    
    # Forward-backward consistency
    warped_backward = flow_backward.clone()  # 실제로는 flow로 와핑해야 하지만, 간단한 분석을 위해 생략
    fb_consistency = torch.norm(flow_forward + warped_backward, dim=1, keepdim=True)
    avg_fb_consistency = fb_consistency.mean().item()
    
    print(f"평균 forward-backward consistency 오차: {avg_fb_consistency:.4f}")

def compare_occlusion_methods(flow_forward, flow_backward):
    """
    다양한 occlusion estimation 방법 비교
    
    Args:
        flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
        flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
    """
    methods = ['wang', 'brox', 'uflow']
    
    plt.figure(figsize=(15, 5))
    
    print("\n각 방법별 분석:")
    print("=" * 30)
    
    for idx, method in enumerate(methods, 1):
        print(f"\n[{method} 방법]")
        occlusion_mask = estimate_occlusion_mask(
            flow_forward, 
            flow_backward,
            method=method
        )
        
        # 통계적 분석
        analyze_occlusion_mask(occlusion_mask, flow_forward, flow_backward)
        
        plt.subplot(1, 3, idx)
        plt.imshow(occlusion_mask[0, 0].cpu().numpy(), cmap='RdYlBu')
        plt.title(f'Occlusion Mask ({method})')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def visualize_occlusion_mask(flow_forward, flow_backward, method='wang'):
    """
    Occlusion mask를 시각화하는 함수
    
    Args:
        flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
        flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
        method (str): 가려짐 추정 방식
    """
    # Occlusion mask 계산
    occlusion_mask = estimate_occlusion_mask(
        flow_forward, 
        flow_backward,
        method=method
    )
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 순방향 흐름 벡터장
    plt.subplot(131)
    flow_magnitude = torch.sqrt(flow_forward[:, 0]**2 + flow_forward[:, 1]**2)
    plt.imshow(flow_magnitude[0].cpu().numpy())
    plt.quiver(flow_forward[0, 0].cpu().numpy()[::16, ::16],
              flow_forward[0, 1].cpu().numpy()[::16, ::16],
              scale=50)
    plt.title('Forward Flow Field')
    plt.colorbar()
    
    # 역방향 흐름 벡터장
    plt.subplot(132)
    flow_magnitude = torch.sqrt(flow_backward[:, 0]**2 + flow_backward[:, 1]**2)
    plt.imshow(flow_magnitude[0].cpu().numpy())
    plt.quiver(flow_backward[0, 0].cpu().numpy()[::16, ::16],
              flow_backward[0, 1].cpu().numpy()[::16, ::16],
              scale=50)
    plt.title('Backward Flow Field')
    plt.colorbar()
    
    # Occlusion mask
    plt.subplot(133)
    plt.imshow(occlusion_mask[0, 0].cpu().numpy(), cmap='RdYlBu')
    plt.title('Occlusion Mask (1=visible, 0=occluded)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def test_learning_effectiveness(flow_forward, flow_backward, target_flow, method='wang'):
    """
    가려진 영역과 가려지지 않은 영역에서의 학습 효과를 비교하는 함수
    
    Args:
        flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
        flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
        target_flow (torch.Tensor): 목표 광학 흐름 [B, 2, H, W]
        method (str): 가려짐 추정 방식
    """
    # Occlusion mask 계산
    occlusion_mask = estimate_occlusion_mask(
        flow_forward, 
        flow_backward,
        method=method
    )
    
    # MSE loss 계산
    mse_loss = torch.nn.MSELoss(reduction='none')(flow_forward, target_flow)
    mse_loss = mse_loss.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # 가려진 영역과 가려지지 않은 영역에서의 loss 계산
    occluded_loss = (mse_loss * (1 - occlusion_mask)).sum() / ((1 - occlusion_mask).sum() + 1e-6)
    visible_loss = (mse_loss * occlusion_mask).sum() / (occlusion_mask.sum() + 1e-6)
    
    print(f"Occluded regions loss: {occluded_loss.item():.4f}")
    print(f"Visible regions loss: {visible_loss.item():.4f}")
    
    # 0으로 나누기 방지
    if occluded_loss.item() > 1e-6:
        print(f"Loss ratio (visible/occluded): {visible_loss.item()/occluded_loss.item():.4f}")
    else:
        print("Loss ratio cannot be computed (occluded loss is too close to zero)")
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # MSE Loss
    plt.subplot(131)
    plt.imshow(mse_loss[0, 0].cpu().numpy())
    plt.title('MSE Loss')
    plt.colorbar()
    
    # Occlusion Mask
    plt.subplot(132)
    plt.imshow(occlusion_mask[0, 0].cpu().numpy(), cmap='RdYlBu')
    plt.title('Occlusion Mask')
    plt.colorbar()
    
    # Loss with Mask
    plt.subplot(133)
    plt.imshow((mse_loss * occlusion_mask)[0, 0].cpu().numpy())
    plt.title('Loss in Visible Regions')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def verify_occlusion_mask_effectiveness(flow_forward, flow_backward, target_flow, method='wang'):
    """
    Occlusion mask가 loss 계산에 미치는 영향을 검증하는 함수
    
    Args:
        flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
        flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
        target_flow (torch.Tensor): 목표 광학 흐름 [B, 2, H, W]
        method (str): 가려짐 추정 방식
    """
    # 1. Occlusion mask 계산
    occlusion_mask = estimate_occlusion_mask(
        flow_forward, 
        flow_backward,
        method=method
    )
    
    # 2. MSE loss 계산 (mask 적용 전)
    mse_loss = torch.nn.MSELoss(reduction='none')(flow_forward, target_flow)
    mse_loss = mse_loss.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # 3. Mask 적용 후 loss 계산
    masked_loss = mse_loss * occlusion_mask
    
    # 4. 통계 분석
    print("\nOcclusion Mask Loss 검증:")
    print("=" * 30)
    
    # 전체 loss 통계
    print(f"전체 평균 loss: {mse_loss.mean().item():.4f}")
    print(f"Mask 적용 후 평균 loss: {masked_loss.mean().item():.4f}")
    
    # 가려진/가려지지 않은 영역의 loss 분포
    occluded_loss = mse_loss * (1 - occlusion_mask)
    visible_loss = mse_loss * occlusion_mask
    
    print("\n가려진 영역:")
    print(f"평균 loss: {occluded_loss.mean().item():.4f}")
    print(f"최대 loss: {occluded_loss.max().item():.4f}")
    print(f"표준편차: {occluded_loss.std().item():.4f}")
    
    print("\n가려지지 않은 영역:")
    print(f"평균 loss: {visible_loss.mean().item():.4f}")
    print(f"최대 loss: {visible_loss.max().item():.4f}")
    print(f"표준편차: {visible_loss.std().item():.4f}")
    
    # 5. 의도적인 가려짐 테스트
    print("\n의도적인 가려짐 테스트:")
    # 중앙 영역을 의도적으로 가려짐으로 설정
    center_mask = torch.ones_like(occlusion_mask)
    h, w = center_mask.shape[-2:]
    center_mask[:, :, h//4:3*h//4, w//4:3*w//4] = 0
    
    # 가려진 영역에 큰 오차 추가
    synthetic_target = target_flow.clone()
    synthetic_target[:, :, h//4:3*h//4, w//4:3*w//4] += 10.0
    
    # Loss 계산
    synthetic_loss = torch.nn.MSELoss(reduction='none')(flow_forward, synthetic_target)
    synthetic_loss = synthetic_loss.mean(dim=1, keepdim=True)
    
    # Mask 적용 전/후 비교
    print(f"Mask 적용 전 중앙 영역 loss: {synthetic_loss[:, :, h//4:3*h//4, w//4:3*w//4].mean().item():.4f}")
    print(f"Mask 적용 후 중앙 영역 loss: {(synthetic_loss * center_mask)[:, :, h//4:3*h//4, w//4:3*w//4].mean().item():.4f}")
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 원본 loss
    plt.subplot(131)
    plt.imshow(mse_loss[0, 0].cpu().numpy())
    plt.title('Original Loss')
    plt.colorbar()
    
    # Masked loss
    plt.subplot(132)
    plt.imshow(masked_loss[0, 0].cpu().numpy())
    plt.title('Masked Loss')
    plt.colorbar()
    
    # Synthetic test
    plt.subplot(133)
    plt.imshow((synthetic_loss * center_mask)[0, 0].cpu().numpy())
    plt.title('Synthetic Test Loss')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def test_gradient_flow_with_mask(flow_forward, flow_backward, target_flow, method='wang'):
    """
    Occlusion mask가 gradient flow에 미치는 영향을 테스트하는 함수
    
    Args:
        flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
        flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
        target_flow (torch.Tensor): 목표 광학 흐름 [B, 2, H, W]
        method (str): 가려짐 추정 방식
    """
    # 1. Occlusion mask 계산
    occlusion_mask = estimate_occlusion_mask(
        flow_forward, 
        flow_backward,
        method=method
    )
    
    # 2. 학습 가능한 파라미터 생성 (실제 학습과 유사한 환경)
    flow_pred = torch.nn.Parameter(flow_forward.clone())
    optimizer = torch.optim.Adam([flow_pred], lr=0.01)
    
    # 3. 학습 루프
    print("\nGradient Flow 테스트:")
    print("=" * 30)
    
    # Loss 계산 함수들
    def compute_masked_loss(pred, target, mask):
        # 1. 단순 마스킹
        mse = torch.nn.MSELoss(reduction='none')(pred, target)
        masked_mse = mse * mask
        return masked_mse.mean()
    
    def compute_normalized_masked_loss(pred, target, mask):
        # 2. 정규화된 마스킹
        mse = torch.nn.MSELoss(reduction='none')(pred, target)
        masked_mse = mse * mask
        # 가려지지 않은 영역의 수로 정규화
        return masked_mse.sum() / (mask.sum() + 1e-6)
    
    def compute_balanced_masked_loss(pred, target, mask):
        # 3. 균형잡힌 마스킹
        mse = torch.nn.MSELoss(reduction='none')(pred, target)
        masked_mse = mse * mask
        unmasked_mse = mse * (1 - mask)
        # 가려진 영역과 가려지지 않은 영역의 loss를 균형있게 조합
        return 0.5 * (masked_mse.sum() / (mask.sum() + 1e-6) + 
                     unmasked_mse.sum() / ((1 - mask).sum() + 1e-6))
    
    # 각 loss 함수별 학습 진행
    loss_functions = {
        '단순 마스킹': compute_masked_loss,
        '정규화된 마스킹': compute_normalized_masked_loss,
        '균형잡힌 마스킹': compute_balanced_masked_loss
    }
    
    for loss_name, loss_fn in loss_functions.items():
        print(f"\n[{loss_name}]")
        flow_pred = torch.nn.Parameter(flow_forward.clone())
        optimizer = torch.optim.Adam([flow_pred], lr=0.01)
        
        # 학습 진행
        for step in range(10):
            optimizer.zero_grad()
            loss = loss_fn(flow_pred, target_flow, occlusion_mask)
            loss.backward()
            
            # Gradient 통계
            grad_norm = flow_pred.grad.norm().item()
            grad_mean = flow_pred.grad.mean().item()
            grad_std = flow_pred.grad.std().item()
            
            print(f"Step {step+1}:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Gradient Norm: {grad_norm:.6f}")
            print(f"  Gradient Mean: {grad_mean:.6f}")
            print(f"  Gradient Std: {grad_std:.6f}")
            
            optimizer.step()
        
        # 최종 결과 시각화
        plt.figure(figsize=(15, 5))
        
        # 원본 flow
        plt.subplot(131)
        plt.imshow(torch.sqrt(flow_forward[0, 0]**2 + flow_forward[0, 1]**2).cpu().numpy())
        plt.title('Original Flow')
        plt.colorbar()
        
        # 학습된 flow
        plt.subplot(132)
        plt.imshow(torch.sqrt(flow_pred[0, 0]**2 + flow_pred[0, 1]**2).detach().cpu().numpy())
        plt.title('Learned Flow')
        plt.colorbar()
        
        # Occlusion mask
        plt.subplot(133)
        plt.imshow(occlusion_mask[0, 0].cpu().numpy(), cmap='RdYlBu')
        plt.title('Occlusion Mask')
        plt.colorbar()
        
        plt.suptitle(f'Flow Learning with {loss_name}')
        plt.tight_layout()
        plt.show()

def test_loss_normalization(flow_forward, flow_backward, method='wang'):
    """
    Loss 정규화 전후를 비교하는 테스트
    
    Args:
        flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
        flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
        method (str): 가려짐 탐지 방법
    """
    # 가려짐 마스크 계산
    occlusion_mask = estimate_occlusion_mask(flow_forward, flow_backward, method=method)
    
    # 테스트용 타겟 흐름 생성 (원본 흐름에 노이즈 추가)
    target_flow = flow_forward + torch.randn_like(flow_forward) * 0.1
    
    # 1. 단순 마스킹 (정규화 없음)
    mse = torch.nn.MSELoss(reduction='none')(flow_forward, target_flow)
    masked_mse = mse * occlusion_mask
    simple_loss = masked_mse.mean()
    
    # 2. 정규화된 마스킹 (가려지지 않은 픽셀 수로 정규화)
    norm_factor = occlusion_mask.sum() + 1e-16
    normalized_loss = masked_mse.sum() / norm_factor
    
    # 3. 가중치가 적용된 마스킹 (uflow_tensorflow 방식)
    weighted_mask = occlusion_mask / norm_factor
    weighted_loss = (mse * weighted_mask).sum()
    
    # 결과 출력
    print(f"\nLoss 정규화 비교 (방법: {method}):")
    print(f"가려진 영역 비율: {(1 - occlusion_mask.mean()).item():.2%}")
    print(f"1. 단순 마스킹 Loss: {simple_loss.item():.6f}")
    print(f"2. 정규화된 마스킹 Loss: {normalized_loss.item():.6f}")
    print(f"3. 가중치가 적용된 마스킹 Loss: {weighted_loss.item():.6f}")
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(masked_mse[0, 0].detach().cpu().numpy())
    plt.title('단순 마스킹 Loss')
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow((masked_mse / norm_factor)[0, 0].detach().cpu().numpy())
    plt.title('정규화된 마스킹 Loss')
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow((mse * weighted_mask)[0, 0].detach().cpu().numpy())
    plt.title('가중치가 적용된 마스킹 Loss')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def create_test_data(batch_size=1, height=256, width=256):
    """테스트용 데이터 생성"""
    # 이미지 생성
    img1 = torch.rand(batch_size, 3, height, width)
    img2 = torch.rand(batch_size, 3, height, width)
    
    # 광학 흐름 생성 (회전 패턴)
    y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    
    # 회전하는 흐름 필드
    flow_forward = torch.stack([
        -y * torch.sin(theta),
        x * torch.cos(theta)
    ], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1) * 10.0
    
    # 역방향 흐름 (약간의 불일치 추가)
    flow_backward = -flow_forward + torch.randn_like(flow_forward) * 2.0
    
    return img1, img2, flow_forward, flow_backward

def visualize_results(img1, img2, flow_forward, flow_backward, occlusion_mask, loss_dict):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 이미지 표시
    axes[0, 0].imshow(img1[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title('Image 1')
    axes[0, 1].imshow(img2[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title('Image 2')
    
    # 흐름 시각화
    flow_mag = torch.sqrt(flow_forward[0, 0]**2 + flow_forward[0, 1]**2)
    axes[0, 2].imshow(flow_mag.cpu().numpy())
    axes[0, 2].set_title('Flow Magnitude')
    
    # 가려짐 마스크 시각화
    axes[1, 0].imshow(occlusion_mask[0, 0].cpu().numpy())
    axes[1, 0].set_title('Occlusion Mask')
    
    # 손실 값 표시
    loss_text = '\n'.join([f'{k}: {v.item():.6f}' for k, v in loss_dict.items() if isinstance(v, torch.Tensor) and v.numel() == 1])
    axes[1, 1].text(0.1, 0.5, loss_text, fontsize=10)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Loss Values')
    
    # 흐름 방향 시각화
    flow_dir = torch.atan2(flow_forward[0, 1], flow_forward[0, 0])
    axes[1, 2].imshow(flow_dir.cpu().numpy())
    axes[1, 2].set_title('Flow Direction')
    
    plt.tight_layout()
    return fig

def test_occlusion_mask():
    """Occlusion mask 테스트"""
    # 테스트 데이터 생성
    img1, img2, flow_forward, flow_backward = create_test_data()
    
    # UFlowLoss 인스턴스 생성
    uflow_loss = UFlowLoss(
        photometric_weight=1.0,
        census_weight=1.0,
        smoothness_weight=0.1,
        occlusion_method='wang',
        use_occlusion=True,
        use_valid_mask=True,
        stop_gradient=True
    )
    
    # 손실 계산
    loss_dict = uflow_loss(img1, img2, flow_forward, flow_backward)
    
    # 결과 시각화
    fig = visualize_results(
        img1, img2, flow_forward, flow_backward,
        loss_dict['occlusion_mask'], loss_dict
    )
    
    # 결과 출력
    print("\nTest Results:")
    print("-" * 50)
    print("Flow Statistics:")
    print(f"Max flow magnitude: {torch.max(torch.sqrt(flow_forward[0, 0]**2 + flow_forward[0, 1]**2)).item():.2f}")
    print(f"Mean flow magnitude: {torch.mean(torch.sqrt(flow_forward[0, 0]**2 + flow_forward[0, 1]**2)).item():.2f}")
    print("\nOcclusion Mask Statistics:")
    print(f"Mean occlusion value: {torch.mean(loss_dict['occlusion_mask']).item():.4f}")
    print(f"Occlusion ratio: {(1 - torch.mean(loss_dict['occlusion_mask'])).item():.4f}")
    print("\nLoss Values:")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            print(f"{k}: {v.item():.6f}")
    
    return fig

def verify_mask_direction(batch_size=1, height=128, width=128):
    """
    Occlusion mask의 방향을 검증하는 테스트
    
    1. 명확한 가려짐이 있는 flow 패턴 생성
    2. 각 방법별 mask 생성 및 방향 확인
    3. Loss 계산 시 mask 적용 방향 검증
    """
    print("\nOcclusion Mask Direction Test")
    print("=" * 50)
    
    # 1. 명확한 가려짐이 있는 flow 패턴 생성
    # 왼쪽에서 오른쪽으로 이동하는 물체를 시뮬레이션
    flow_forward = torch.zeros(batch_size, 2, height, width)
    flow_forward[:, 0, :, :-20] = 20.0  # x 방향으로 20픽셀 이동
    
    # 역방향 flow는 정확히 반대 방향 (약간의 불일치 추가)
    flow_backward = torch.zeros_like(flow_forward)
    flow_backward[:, 0, :, 20:] = -19.5  # 약간의 오차 포함
    
    # 2. 각 방법별 mask 생성 및 방향 확인
    methods = ['wang', 'brox', 'uflow']
    
    for method in methods:
        print(f"\n[{method} 방법]")
        mask = utils.estimate_occlusion_mask(flow_forward, flow_backward, method=method)
        
        # 가려진 영역 (이론상 물체가 이동한 후의 위치)
        occluded_region = mask[0, 0, :, -20:].mean().item()
        visible_region = mask[0, 0, :, :-20].mean().item()
        
        print(f"예상 가려짐 영역의 평균 mask 값: {occluded_region:.4f}")
        print(f"예상 비가려짐 영역의 평균 mask 값: {visible_region:.4f}")
        
        # 3. Loss 계산 검증
        # 인위적인 에러가 있는 flow 생성
        test_flow = flow_forward.clone()
        test_flow[:, 0, :, -20:] += 10.0  # 가려진 영역에 큰 에러 추가
        
        # Loss 계산
        mse = torch.nn.MSELoss(reduction='none')(test_flow, flow_forward)
        mse = mse.mean(dim=1, keepdim=True)
        
        # Mask 적용 전/후 loss
        unmasked_loss = mse.mean().item()
        masked_loss = (mse * mask).mean().item()
        
        print(f"Mask 적용 전 평균 loss: {unmasked_loss:.4f}")
        print(f"Mask 적용 후 평균 loss: {masked_loss:.4f}")
        
        # 시각화
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(mask[0, 0].cpu().numpy(), cmap='RdYlBu')
        plt.title(f'Occlusion Mask ({method})\n1=visible, 0=occluded')
        plt.colorbar()
        
        plt.subplot(132)
        plt.imshow(mse[0, 0].cpu().numpy())
        plt.title('Original MSE Loss')
        plt.colorbar()
        
        plt.subplot(133)
        plt.imshow((mse * mask)[0, 0].cpu().numpy())
        plt.title('Masked MSE Loss')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()

def verify_gradient_flow(batch_size=1, height=128, width=128):
    """
    Gradient flow를 검증하는 테스트
    
    1. 학습 가능한 flow 생성
    2. Forward/Backward pass 수행
    3. Gradient 분석
    """
    print("\nGradient Flow Test")
    print("=" * 50)
    
    # 1. 테스트 데이터 생성
    # 학습 가능한 flow
    flow_pred = torch.zeros(batch_size, 2, height, width, requires_grad=True)
    flow_pred.data[:, 0, :, :-20] = 20.0  # x 방향으로 20픽셀 이동
    
    # 타겟 flow (ground truth)
    flow_target = flow_pred.data.clone()
    flow_target[:, 0, :, :-20] = 25.0  # 의도적으로 다른 값 설정
    
    # 역방향 flow
    flow_backward = torch.zeros_like(flow_pred.data)
    flow_backward[:, 0, :, 20:] = -19.5
    
    # 2. Loss 계산을 위한 컴포넌트들
    uflow_loss = UFlowLoss(
        photometric_weight=1.0,
        census_weight=1.0,
        smoothness_weight=0.1,
        use_occlusion=True,  # occlusion mask 사용
        occlusion_method='wang'
    )
    
    # 이미지 생성 (간단한 패턴)
    img1 = torch.zeros(batch_size, 3, height, width)
    img2 = torch.zeros(batch_size, 3, height, width)
    # 이미지에 패턴 추가
    img1[:, 0, :, ::4] = 1.0  # 빨간색 수직 줄무늬
    img2 = utils.warp_image(img1, flow_target)  # 타겟 flow로 와핑
    
    optimizer = torch.optim.Adam([flow_pred], lr=0.01)
    
    print("\n학습 진행 상황:")
    for step in range(5):  # 5 스텝만 테스트
        optimizer.zero_grad()
        
        # Forward pass
        loss_dict = uflow_loss(img1, img2, flow_pred, flow_backward)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient 분석
        grad = flow_pred.grad
        grad_visible = grad[:, :, :, :-20]  # 가려지지 않은 영역
        grad_occluded = grad[:, :, :, -20:]  # 가려진 영역
        
        print(f"\nStep {step + 1}")
        print(f"Total Loss: {total_loss.item():.6f}")
        print(f"Gradient 통계:")
        print(f"- 전체 평균 크기: {grad.abs().mean().item():.6f}")
        print(f"- 가려지지 않은 영역 평균 크기: {grad_visible.abs().mean().item():.6f}")
        print(f"- 가려진 영역 평균 크기: {grad_occluded.abs().mean().item():.6f}")
        print(f"- gradient가 0이 아닌 비율: {(grad.abs() > 1e-6).float().mean().item():.2%}")
        
        if 'occlusion_mask' in loss_dict:
            mask = loss_dict['occlusion_mask']
            print(f"Occlusion mask 평균값: {mask.mean().item():.4f}")
        
        optimizer.step()
        
        # 시각화 (첫 번째와 마지막 스텝만)
        if step in [0, 4]:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(grad[0, 0].detach().cpu().numpy())
            plt.title(f'Gradient Map (Step {step + 1})')
            plt.colorbar()
            
            plt.subplot(132)
            if 'occlusion_mask' in loss_dict:
                plt.imshow(mask[0, 0].detach().cpu().numpy(), cmap='RdYlBu')
                plt.title('Occlusion Mask\n1=visible, 0=occluded')
                plt.colorbar()
            
            plt.subplot(133)
            flow_error = (flow_pred - flow_target).abs().mean(dim=1)
            plt.imshow(flow_error[0].detach().cpu().numpy())
            plt.title('Flow Error')
            plt.colorbar()
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    verify_gradient_flow() 