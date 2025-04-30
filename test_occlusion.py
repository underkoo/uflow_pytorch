import torch
import utils
import matplotlib.pyplot as plt
import numpy as np

def test_occlusion_estimation():
    """
    다양한 가려짐 추정 방식을 시각적으로 비교하는 테스트
    """
    # 테스트용 흐름 생성
    batch_size = 1
    height, width = 64, 64
    
    print("===== 광학 흐름 생성 =====")
    # 순방향 흐름 (오른쪽으로 4픽셀 이동)
    flow_forward = torch.zeros(batch_size, 2, height, width)
    flow_forward[:, 0] = 4.0  # x 방향으로 4픽셀
    
    # 역방향 흐름 (왼쪽으로 4픽셀 이동)
    flow_backward = torch.zeros(batch_size, 2, height, width)
    flow_backward[:, 0] = -4.0  # x 방향으로 -4픽셀
    
    # 순방향 흐름의 일부를 다르게 설정하여 가려짐 생성
    flow_forward[:, :, 20:40, 20:40] = 10.0
    
    # wang 방식을 위해 역방향 흐름도 가운데 영역 수정
    # 가려짐으로 인식되도록 화면 밖으로 향하는 큰 값 설정
    flow_backward[:, 0, 20:40, 20:40] = -100.0  # 왼쪽 방향으로 크게
    flow_backward[:, 1, 20:40, 20:40] = -100.0  # 위쪽 방향으로 크게
    
    # 직접 범위 맵 확인
    print("\n===== 범위 맵 계산 결과 =====")
    range_map = utils.compute_range_map(
        flow_backward, 
        downsampling_factor=1, 
        reduce_downsampling_bias=False, 
        resize_output=False
    )
    
    print(f"범위 맵 형태: {range_map.shape}")
    print(f"범위 맵 값 범위: {range_map.min().item():.2f} ~ {range_map.max().item():.2f}")
    print(f"왼쪽 경계 (0~3열) 값: {range_map[0, 0, 32, :4].numpy().flatten()}")
    print(f"오른쪽 경계 (60~63열) 값: {range_map[0, 0, 32, 60:].numpy().flatten()}")
    print(f"가운데 영역 (30,30) 값: {range_map[0, 0, 30, 30].item():.2f}")
    
    # 모든 가려짐 추정 방식 테스트
    print("\n===== 가려짐 마스크 추정 결과 =====")
    methods = ['forward_backward', 'brox', 'wang', 'wang4', 'uflow']
    masks = {}
    
    for method in methods:
        print(f"\n{method} 방식:")
        masks[method] = utils.estimate_occlusion_mask(flow_forward, flow_backward, method=method)
        print(f"  마스크 값 범위: {masks[method].min().item():.2f} ~ {masks[method].max().item():.2f}")
        print(f"  왼쪽 경계 (0~3열) 값: {masks[method][0, 0, 32, :4].numpy().flatten()}")
        print(f"  오른쪽 경계 (60~63열) 값: {masks[method][0, 0, 32, 60:].numpy().flatten()}")
        print(f"  중앙 영역 (30,30) 값: {masks[method][0, 0, 30, 30].item():.2f}")
        print(f"  특수 영역 (24,24) 값: {masks[method][0, 0, 24, 24].item():.2f}")
    
    # 결과 시각화
    plt.figure(figsize=(18, 12))
    
    # 입력 흐름 시각화
    plt.subplot(3, 3, 1)
    plt.imshow(flow_forward[0, 0].numpy())
    plt.title(f'Flow Forward (x direction)')
    plt.colorbar()
    
    plt.subplot(3, 3, 2)
    plt.imshow(flow_backward[0, 0].numpy())
    plt.title(f'Flow Backward (x direction)')
    plt.colorbar()
    
    plt.subplot(3, 3, 3)
    plt.imshow(range_map[0, 0].numpy())
    plt.title(f'Range Map (Backward Flow)')
    plt.colorbar()
    
    # 가려짐 마스크 시각화
    for i, method in enumerate(methods):
        plt.subplot(3, 3, i+4)
        plt.imshow(masks[method][0, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title(f'Method: {method}')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('occlusion_comparison.png')
    print("\n테스트 완료! occlusion_comparison.png 파일을 확인하세요.")

if __name__ == "__main__":
    test_occlusion_estimation() 