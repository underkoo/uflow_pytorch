import torch
import torch.nn as nn
import numpy as np
from losses import SmoothnessLoss

def test_smoothness_loss():
    """
    Smoothness Loss 구현 검증 테스트
    """
    print("\n" + "-"*60)
    print("Smoothness Loss 테스트")
    print("-"*60)
    
    # 테스트 데이터 생성
    batch_size = 2
    channels = 3
    height = 64
    width = 64
    
    # 이미지와 플로우 생성
    image = torch.rand(batch_size, channels, height, width)
    flow = torch.randn(batch_size, 2, height, width) * 0.1
    
    # GPU 사용 가능한 경우 데이터 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    flow = flow.to(device)
    
    # 1. 기본 평활화 손실 테스트
    print("\n1. 기본 평활화 손실 테스트")
    smoothness_loss = SmoothnessLoss(edge_aware=False, second_order=False)
    smoothness_loss = smoothness_loss.to(device)
    
    loss = smoothness_loss(flow)
    print(f"기본 평활화 손실: {loss.item():.6f}")
    
    # 2. 에지 인식 평활화 손실 테스트 (지수 가중치)
    print("\n2. 에지 인식 평활화 손실 테스트 (지수 가중치)")
    smoothness_loss = SmoothnessLoss(
        edge_aware=True, 
        second_order=False,
        edge_weighting='exponential'
    )
    smoothness_loss = smoothness_loss.to(device)
    
    loss = smoothness_loss(flow, image)
    print(f"에지 인식 평활화 손실 (지수): {loss.item():.6f}")
    
    # 3. 에지 인식 평활화 손실 테스트 (가우시안 가중치)
    print("\n3. 에지 인식 평활화 손실 테스트 (가우시안 가중치)")
    smoothness_loss = SmoothnessLoss(
        edge_aware=True, 
        second_order=False,
        edge_weighting='gaussian'
    )
    smoothness_loss = smoothness_loss.to(device)
    
    loss = smoothness_loss(flow, image)
    print(f"에지 인식 평활화 손실 (가우시안): {loss.item():.6f}")
    
    # 4. 이차 미분 평활화 손실 테스트
    print("\n4. 이차 미분 평활화 손실 테스트")
    smoothness_loss = SmoothnessLoss(
        edge_aware=True, 
        second_order=True,
        edge_weighting='exponential'
    )
    smoothness_loss = smoothness_loss.to(device)
    
    # 더 큰 변화를 가진 플로우 생성
    flow_strong = torch.randn(batch_size, 2, height, width) * 1.0
    flow_strong = flow_strong.to(device)
    
    loss = smoothness_loss(flow_strong, image)
    print(f"이차 미분 평활화 손실: {loss.item():.6f}")
    
    # 5. 그라디언트 계산 검증
    print("\n5. 그라디언트 계산 검증")
    # 테스트 이미지 생성 (x, y 방향 모두 변화가 있는 그라디언트)
    test_image = torch.zeros(1, 1, 4, 4, device=device)
    # x 방향과 y 방향 모두 변화가 있는 그라디언트 생성
    for i in range(4):
        for j in range(4):
            test_image[0, 0, i, j] = (i + j) * 0.3
    
    print("테스트 이미지:")
    print(test_image[0, 0].cpu().numpy())
    
    # x 방향 그라디언트 계산
    gx = smoothness_loss._gradient(test_image, 'x')
    print("\nx 방향 그라디언트:")
    print(gx[0, 0].cpu().numpy())
    
    # y 방향 그라디언트 계산
    gy = smoothness_loss._gradient(test_image, 'y')
    print("\ny 방향 그라디언트:")
    print(gy[0, 0].cpu().numpy())
    
    # 이차 미분 검증
    gxx = smoothness_loss._gradient(gx, 'x')
    print("\nx 방향 이차 미분:")
    print(gxx[0, 0].cpu().numpy())
    
    # y 방향 이차 미분도 검증
    gyy = smoothness_loss._gradient(gy, 'y')
    print("\ny 방향 이차 미분:")
    print(gyy[0, 0].cpu().numpy())
    
    print("\n" + "-"*60)
    print("테스트 완료!")
    print("-"*60)

if __name__ == "__main__":
    test_smoothness_loss() 