import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class PWCFeaturePyramid(nn.Module):
    """
    PWC-Net 특징 피라미드 모듈
    
    입력 이미지에서 다중 스케일 특징 맵을 추출하는 역할
    """
    def __init__(self, 
                 num_channels=3, 
                 num_levels=5, 
                 feature_channels=32, 
                 level1_num_layers=3,
                 leaky_relu_alpha=0.1,
                 channel_multiplier=1.0):
        """
        Args:
            num_channels (int): 입력 이미지의 채널 수
            num_levels (int): 피라미드 레벨 수
            feature_channels (int): 첫 번째 레벨의 특징 채널 수
            level1_num_layers (int): 첫 번째 레벨의 컨볼루션 레이어 수
            leaky_relu_alpha (float): LeakyReLU의 음수 기울기
            channel_multiplier (float): 채널 수 배수
        """
        super(PWCFeaturePyramid, self).__init__()
        
        self.num_levels = num_levels
        self.channel_multiplier = channel_multiplier
        self.leaky_relu_alpha = leaky_relu_alpha
        
        # 각 레벨별 특징 추출 레이어 정의
        self.layers = nn.ModuleList()
        
        # 첫 번째 레벨 (최고 해상도)
        level1_layers = []
        in_channels = num_channels
        out_channels = int(feature_channels * channel_multiplier)
        
        # 첫 번째 레벨의 컨볼루션 레이어
        for i in range(level1_num_layers):
            level1_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            level1_layers.append(nn.LeakyReLU(leaky_relu_alpha))
            in_channels = out_channels
        
        self.layers.append(nn.Sequential(*level1_layers))
        
        # 레벨 2부터 num_levels까지
        for level in range(1, num_levels):
            layers = []
            in_channels = int(feature_channels * channel_multiplier)
            out_channels = int(feature_channels * 2 * channel_multiplier)
            
            # 스트라이드 2 컨볼루션으로 다운샘플링
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.LeakyReLU(leaky_relu_alpha))
            
            # 추가 컨볼루션 레이어
            for i in range(level1_num_layers - 1):
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.LeakyReLU(leaky_relu_alpha))
            
            self.layers.append(nn.Sequential(*layers))
            feature_channels *= 2
    
    def forward(self, x):
        """
        특징 피라미드 추출
        
        Args:
            x (torch.Tensor): 입력 이미지 [B, C, H, W]
            
        Returns:
            list: 다양한 해상도의 특징 맵 리스트 [level1, level2, ..., levelN]
        """
        features = []
        
        # 각 레벨 순차적으로 처리
        for i in range(self.num_levels):
            if i == 0:
                feature = self.layers[i](x)
            else:
                feature = self.layers[i](features[-1])
            features.append(feature)
        
        return features


class CostVolumeLayer(nn.Module):
    """
    Cost Volume 계산 모듈
    
    두 특징 맵 간의 유사도를 계산하는 역할
    """
    def __init__(self, max_displacement=4):
        """
        Args:
            max_displacement (int): 최대 변위 (탐색 범위)
        """
        super(CostVolumeLayer, self).__init__()
        self.max_displacement = max_displacement
    
    def forward(self, features1, features2):
        """
        Args:
            features1 (torch.Tensor): 첫 번째 특징 맵 [B, C, H, W]
            features2 (torch.Tensor): 두 번째 특징 맵 [B, C, H, W]
            
        Returns:
            torch.Tensor: Cost volume [B, (2*max_displacement+1)^2, H, W]
        """
        B, C, H, W = features1.shape
        
        # 최대 변위 검사
        if self.max_displacement <= 0 or self.max_displacement >= min(H, W):
            raise ValueError(f'Max displacement {self.max_displacement} is too large for feature map size {H}x{W}')
        
        # utils.py의 함수 사용
        return utils.compute_cost_volume(features1, features2, self.max_displacement)


class FlowEstimator(nn.Module):
    """
    광학 흐름 추정 네트워크
    
    비용 볼륨과 특징을 기반으로 광학 흐름을 추정하는 역할
    """
    def __init__(self, in_channels, hidden_channels=128, leaky_relu_alpha=0.1):
        """
        Args:
            in_channels (int): 입력 채널 수 (비용 볼륨 + 특징 + 옵션 채널)
            hidden_channels (int): 은닉층 채널 수
            leaky_relu_alpha (float): LeakyReLU의 음수 기울기
        """
        super(FlowEstimator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 2 * hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels + 3 * hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels + 4 * hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.flow = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(leaky_relu_alpha)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 특징 [B, C, H, W]
            
        Returns:
            tuple: (flow, context)
                - flow (torch.Tensor): 추정된 광학 흐름 [B, 2, H, W]
                - context (torch.Tensor): 다음 레벨에서 사용할 문맥 특징 [B, hidden_channels, H, W]
        """
        x1 = self.leaky_relu(self.conv1(x))
        x2 = self.leaky_relu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.leaky_relu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.leaky_relu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.leaky_relu(self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1)))
        
        flow = self.flow(x5)
        
        return flow, x5


class FlowRefinement(nn.Module):
    """
    광학 흐름 정제 네트워크
    
    추정된 초기 광학 흐름을 미세 조정하는 역할
    """
    def __init__(self, in_channels, leaky_relu_alpha=0.1):
        """
        Args:
            in_channels (int): 입력 채널 수 (일반적으로 context + flow)
            leaky_relu_alpha (float): LeakyReLU의 음수 기울기
        """
        super(FlowRefinement, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(leaky_relu_alpha)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 특징 (context + flow) [B, C, H, W]
            
        Returns:
            torch.Tensor: 정제된 흐름 잔차 [B, 2, H, W]
        """
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        flow_residual = self.conv6(x)
        
        return flow_residual


class PWCFlow(nn.Module):
    """
    PWC 광학 흐름 네트워크
    
    두 특징 피라미드 간의 광학 흐름을 계산하는 모듈
    """
    def __init__(self,
                 num_levels=5,
                 feature_channels=32,
                 use_cost_volume=True,
                 max_displacement=4,
                 use_feature_warp=True,
                 context_channels=32,
                 flow_refinement_channels=128, 
                 leaky_relu_alpha=0.1,
                 dropout_rate=0.25,
                 channel_multiplier=1.0,
                 shared_flow_decoder=False):
        """
        Args:
            num_levels (int): 피라미드 레벨 수
            feature_channels (int): 기본 특징 채널 수
            use_cost_volume (bool): 비용 볼륨 사용 여부
            max_displacement (int): 최대 변위 거리
            use_feature_warp (bool): 특징 와핑 사용 여부
            context_channels (int): 문맥 특징의 채널 수
            flow_refinement_channels (int): 흐름 정제 네트워크의 채널 수
            leaky_relu_alpha (float): LeakyReLU의 음수 기울기
            dropout_rate (float): 드롭아웃 비율
            channel_multiplier (float): 채널 수 배수
            shared_flow_decoder (bool): 모든 레벨에서 공유 흐름 디코더 사용 여부
        """
        super(PWCFlow, self).__init__()
        
        self.num_levels = num_levels
        self.use_cost_volume = use_cost_volume
        self.max_displacement = max_displacement
        self.use_feature_warp = use_feature_warp
        self.context_channels = context_channels
        self.flow_refinement_channels = flow_refinement_channels
        self.dropout_rate = dropout_rate
        self.leaky_relu_alpha = leaky_relu_alpha
        self.shared_flow_decoder = shared_flow_decoder
        
        # 비용 볼륨 계산 모듈
        if use_cost_volume:
            self.cost_volume = CostVolumeLayer(max_displacement=max_displacement)
        else:
            # 비용 볼륨 대체 컨볼루션
            self.cost_volume_surrogate_convs = nn.ModuleList()
            for level in range(num_levels):
                # 각 레벨별 실제 특징 채널 수 계산
                level_channels = int(feature_channels * (2 ** level) * channel_multiplier)
                in_channels = level_channels * 2  # 두 특징 맵이 연결되므로 2배
                
                # 출력 크기가 특징 맵 크기와 동일하도록 적절한 패딩 설정
                self.cost_volume_surrogate_convs.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            level_channels,  # 출력은 원래 특징 맵과 같은 채널 수
                            kernel_size=3,
                            padding=1,
                            stride=1
                        ),
                        nn.LeakyReLU(negative_slope=leaky_relu_alpha),
                        nn.Conv2d(
                            level_channels,
                            (2 * max_displacement + 1) ** 2,  # cost volume과 같은 채널 수 출력
                            kernel_size=1,
                            stride=1
                        )
                    )
                )
        
        # 각 레벨별 흐름 추정기
        self.flow_estimators = nn.ModuleList()
        for level in range(num_levels - 1, -1, -1):
            # 해당 레벨의 특징 채널 수
            curr_channels = int(feature_channels * (2 ** level) * channel_multiplier)
            
            # 입력 채널 수 계산
            # 기본: 특징 + 비용 볼륨
            if use_cost_volume:
                input_channels = curr_channels + (2 * max_displacement + 1) ** 2
            else:
                input_channels = curr_channels * 2
                
            # 이전 레벨의 업샘플링된 흐름과 문맥 추가
            if level < num_levels - 1:
                input_channels += 2  # 업샘플링된 흐름
                if context_channels > 0:
                    input_channels += context_channels  # 업샘플링된 문맥
            
            # 흐름 추정기 생성
            self.flow_estimators.append(
                FlowEstimator(
                    in_channels=input_channels,
                    hidden_channels=flow_refinement_channels,
                    leaky_relu_alpha=leaky_relu_alpha
                )
            )
        
        # 흐름 정제 네트워크 (레벨 0의 결과를 정제)
        self.flow_refinement = FlowRefinement(
            in_channels=flow_refinement_channels + 2,  # 문맥 + 흐름
            leaky_relu_alpha=leaky_relu_alpha
        )
        
        # 문맥 업샘플러 (상위 레벨에서 하위 레벨로 문맥 정보 전달)
        if context_channels > 0:
            self.context_upsamplers = nn.ModuleList()
            for _ in range(num_levels - 1):
                self.context_upsamplers.append(
                    nn.ConvTranspose2d(
                        flow_refinement_channels,
                        context_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    )
                )
                
        # 공유 흐름 디코더를 위한 1x1 컨볼루션
        if self.shared_flow_decoder:
            self.conv_1x1_shared_decoder = self._build_1x1_shared_decoder()
            
    def _build_1x1_shared_decoder(self):
        """
        공유 흐름 디코더를 위한 1x1 컨볼루션 레이어 생성
        
        Returns:
            nn.ModuleList: 1x1 컨볼루션 레이어 목록
        """
        result = nn.ModuleList([nn.Identity()])  # 레벨 0에는 1x1 컨볼루션 적용 안 함
        
        for _ in range(1, self.num_levels):
            result.append(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(1, 1),
                    stride=1
                )
            )
        
        return result
    
    def forward(self, feature_pyramid1, feature_pyramid2):
        """
        두 특징 피라미드 간의 광학 흐름 계산
        
        Args:
            feature_pyramid1 (list): 첫 번째 이미지의 특징 피라미드
            feature_pyramid2 (list): 두 번째 이미지의 특징 피라미드
            
        Returns:
            list: 다양한 해상도의 광학 흐름 리스트
        """
        flows = []
        
        # 상위 레벨에서 하위 레벨로 흐름 계산
        flow_up = None
        context_up = None
        
        # 가장 높은 레벨(가장 낮은 해상도)부터 시작
        for level, (features1, features2) in reversed(list(enumerate(zip(feature_pyramid1, feature_pyramid2)))):
            # 상위 레벨의 흐름이 있는 경우 크기 조정 계산
            flow_scale = 1.0
            if flow_up is not None:
                flow_scale = features1.shape[2] / flow_up.shape[2]
                
            # 특징 와핑 사용
            if flow_up is not None and self.use_feature_warp:
                # 현재 레벨에 맞게 흐름 크기 조정
                flow_resized = utils.upsample_flow(flow_up, 
                                              target_size=(features1.shape[2], features1.shape[3])) * flow_scale
                
                # 와핑 그리드 생성
                warped_features2 = utils.warp_features(features2, flow_resized)
            else:
                warped_features2 = features2
            
            # 비용 볼륨 또는 대체 방법 사용
            if self.use_cost_volume:
                # 비용 볼륨 계산 - 정규화는 CostVolumeLayer 내부에서 수행
                cost_volume = self.cost_volume(features1, warped_features2)
            else:
                # 비용 볼륨 대체: 특징 연결 후 컨볼루션
                concat_features = torch.cat([features1, warped_features2], dim=1)
                cost_volume = self.cost_volume_surrogate_convs[level](concat_features)
            
            # 비용 볼륨에 활성화 함수 적용
            cost_volume = F.leaky_relu(cost_volume, negative_slope=self.leaky_relu_alpha)
            
            # shared_flow_decoder 사용 시 1x1 컨볼루션 적용
            if self.shared_flow_decoder:
                # 현재 레벨의 특징 맵에 1x1 컨볼루션 적용
                conv_1x1 = self.conv_1x1_shared_decoder[level]
                features1 = conv_1x1(features1)
            
            # 흐름 추정 입력 준비
            if flow_up is None:
                # 첫 레벨: 비용 볼륨 + 특징만 사용
                flow_input = torch.cat([cost_volume, features1], dim=1)
            else:
                # 하위 레벨: 상위 레벨에서 업샘플링된 흐름 추가
                flow_resized = utils.upsample_flow(flow_up, 
                                              target_size=(features1.shape[2], features1.shape[3])) * flow_scale
                
                if context_up is None:
                    flow_input = torch.cat([flow_resized, cost_volume, features1], dim=1)
                else:
                    # 업샘플링된 문맥도 추가
                    context_resized = F.interpolate(
                        context_up, size=(features1.shape[2], features1.shape[3]),
                        mode='bilinear', align_corners=False
                    )
                    flow_input = torch.cat([context_resized, flow_resized, cost_volume, features1], dim=1)
            
            # 흐름 추정기 적용
            flow_estimator = self.flow_estimators[self.num_levels - 1 - level]
            flow, context = flow_estimator(flow_input)
            
            # 학습 중 드롭아웃 적용
            if self.training and self.dropout_rate > 0:
                flow_mask = torch.rand(1, device=flow.device) > self.dropout_rate
                context_mask = torch.rand(1, device=context.device) > self.dropout_rate
                
                if not flow_mask:
                    flow = flow * 0
                if not context_mask:
                    context = context * 0
            
            # 상위 레벨의 흐름이 있으면 누적
            if flow_up is not None:
                flow = flow + flow_resized
            
            # 결과 저장
            flows.insert(0, flow)
            
            # 다음 레벨을 위한 준비
            if level > 0:
                # 하위 레벨을 위한 흐름 업샘플링
                flow_up = flow
                
                # 문맥 업샘플링 (있는 경우)
                if self.context_channels > 0:
                    context_up = self.context_upsamplers[self.num_levels - 1 - level](context)
        
        return flows
    
    def infer_occlusion(self, flow_forward, flow_backward, method='wang'):
        """
        광학 흐름에서 가려짐 마스크 추정
        
        Args:
            flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
            flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
            method (str): 가려짐 추정 방식 (default: 'wang')
            
        Returns:
            torch.Tensor: 가려짐 마스크 [B, 1, H, W], 1=가려짐 없음, 0=가려짐
        """
        occ_weights = {
            'fb_abs': 1000.0,
            'forward_collision': 1000.0,
            'backward_zero': 1000.0
        }
        
        occ_thresholds = {
            'fb_abs': 1.5,
            'forward_collision': 0.4,
            'backward_zero': 0.25
        }
        
        occ_clip_max = {
            'fb_abs': 10.0,
            'forward_collision': 5.0
        }
        
        return utils.estimate_occlusion_mask(
            flow_forward, 
            flow_backward,
            method=method,
            occ_weights=occ_weights,
            occ_thresholds=occ_thresholds,
            occ_clip_max=occ_clip_max
        )


class UFlow(nn.Module):
    """
    UFlow 모델
    
    PWCFeaturePyramid와 PWCFlow를 결합하는 최상위 모듈
    """
    def __init__(self, 
                 num_channels=3, 
                 num_levels=5, 
                 feature_channels=32,
                 use_cost_volume=True, 
                 max_displacement=4, 
                 use_feature_warp=True,
                 context_channels=32,
                 flow_refinement_channels=128,
                 leaky_relu_alpha=0.1,
                 dropout_rate=0.25,
                 channel_multiplier=1.0,
                 shared_flow_decoder=False):
        """
        Args:
            num_channels (int): 입력 이미지의 채널 수
            num_levels (int): 피라미드 레벨 수
            feature_channels (int): 기본 특징 채널 수
            use_cost_volume (bool): 비용 볼륨 사용 여부
            max_displacement (int): 최대 변위 거리
            use_feature_warp (bool): 특징 와핑 사용 여부
            context_channels (int): 문맥 특징의 채널 수
            flow_refinement_channels (int): 흐름 정제 네트워크의 채널 수
            leaky_relu_alpha (float): LeakyReLU의 음수 기울기
            dropout_rate (float): 드롭아웃 비율
            channel_multiplier (float): 채널 수 배수
            shared_flow_decoder (bool): 공유 흐름 디코더 사용 여부
        """
        super(UFlow, self).__init__()
        
        # 특징 피라미드 모듈
        self.feature_pyramid = PWCFeaturePyramid(
            num_channels=num_channels,
            num_levels=num_levels,
            feature_channels=feature_channels,
            leaky_relu_alpha=leaky_relu_alpha,
            channel_multiplier=channel_multiplier
        )
        
        # 광학 흐름 모듈
        self.flow_network = PWCFlow(
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
    
    def forward(self, img1, img2):
        """
        두 이미지 간의 광학 흐름 계산
        
        Args:
            img1 (torch.Tensor): 첫 번째 이미지 [B, C, H, W]
            img2 (torch.Tensor): 두 번째 이미지 [B, C, H, W]
            
        Returns:
            tuple: (flows, feature_pyramid1, feature_pyramid2)
                - flows (list): 다양한 해상도의 광학 흐름 리스트
                - feature_pyramid1 (list): 첫 번째 이미지의 특징 피라미드
                - feature_pyramid2 (list): 두 번째 이미지의 특징 피라미드
        """
        # 특징 피라미드 추출
        feature_pyramid1 = self.feature_pyramid(img1)
        feature_pyramid2 = self.feature_pyramid(img2)
        
        # 광학 흐름 계산
        flows = self.flow_network(feature_pyramid1, feature_pyramid2)
        
        return flows, feature_pyramid1, feature_pyramid2
    
    def forward_backward_flow(self, img1, img2):
        """
        두 이미지 간의 양방향 광학 흐름 계산
        
        Args:
            img1 (torch.Tensor): 첫 번째 이미지 [B, C, H, W]
            img2 (torch.Tensor): 두 번째 이미지 [B, C, H, W]
            
        Returns:
            tuple: (forward_flows, backward_flows, feature_pyramid1, feature_pyramid2)
                - forward_flows (list): 순방향 광학 흐름 리스트
                - backward_flows (list): 역방향 광학 흐름 리스트
                - feature_pyramid1 (list): 첫 번째 이미지의 특징 피라미드
                - feature_pyramid2 (list): 두 번째 이미지의 특징 피라미드
        """
        # 특징 피라미드 추출
        feature_pyramid1 = self.feature_pyramid(img1)
        feature_pyramid2 = self.feature_pyramid(img2)
        
        # 순방향 광학 흐름 계산 (img1 -> img2)
        forward_flows = self.flow_network(feature_pyramid1, feature_pyramid2)
        
        # 역방향 광학 흐름 계산 (img2 -> img1)
        backward_flows = self.flow_network(feature_pyramid2, feature_pyramid1)
        
        return forward_flows, backward_flows, feature_pyramid1, feature_pyramid2
    
    def infer_occlusion(self, flow_forward, flow_backward, method='wang'):
        """
        광학 흐름에서 가려짐 마스크 추정
        
        Args:
            flow_forward (torch.Tensor): 순방향 광학 흐름 [B, 2, H, W]
            flow_backward (torch.Tensor): 역방향 광학 흐름 [B, 2, H, W]
            method (str): 가려짐 추정 방식 (default: 'wang')
            
        Returns:
            torch.Tensor: 가려짐 마스크 [B, 1, H, W], 1=가려짐 없음, 0=가려짐
        """
        return self.flow_network.infer_occlusion(flow_forward, flow_backward, method)
    
    def warp_image(self, img, flow):
        """
        이미지를 광학 흐름으로 와핑
        
        Args:
            img (torch.Tensor): 와핑할 이미지 [B, C, H, W]
            flow (torch.Tensor): 광학 흐름 [B, 2, H, W]
            
        Returns:
            torch.Tensor: 와핑된 이미지 [B, C, H, W]
        """
        return utils.warp_image(img, flow)


# 모델 테스트 코드
if __name__ == "__main__":
    # 테스트용 이미지 쌍 생성 (배치 크기 2, 3채널, 256x256)
    img1 = torch.randn(2, 3, 192, 256)
    img2 = torch.randn(2, 3, 192, 256)
    
    # 모델 초기화
    model = UFlow(
        num_channels=3,
        num_levels=5,
        feature_channels=32,
        use_cost_volume=True
    )
    
    # GPU 사용 가능한 경우 모델 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # 모델 순전파
    with torch.no_grad():
        flows, fp1, fp2 = model(img1, img2)
    
    # 결과 확인
    print(f"Flow 피라미드 레벨 수: {len(flows)}")
    for i, flow in enumerate(flows):
        print(f"레벨 {i} 흐름 크기: {flow.shape}")
    
    # 양방향 흐름 테스트
    with torch.no_grad():
        forward_flows, backward_flows, _, _ = model.forward_backward_flow(img1, img2)
        occlusion_mask = model.infer_occlusion(forward_flows[0], backward_flows[0])
        warped_img2 = model.warp_image(img2, forward_flows[0])
    
    print(f"\n양방향 흐름 테스트:")
    print(f"순방향 흐름 크기: {forward_flows[0].shape}")
    print(f"역방향 흐름 크기: {backward_flows[0].shape}")
    print(f"가려짐 마스크 크기: {occlusion_mask.shape}")
    print(f"와핑된 이미지 크기: {warped_img2.shape}")
    
    # 모델 파라미터 수 계산
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n모델 파라미터 수: {param_count:,}")
    
    print("\n테스트 성공!") 