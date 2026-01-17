import torch
import torch.nn as nn

class HigherOrderStatisticsPooling(nn.Module):
    
    def __init__(self, input_dim):
        super(HigherOrderStatisticsPooling, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (Batch, Hidden, Time) 혹은 (Batch, Time, Hidden)
        # 마지막 차원(-1)을 기준으로 통계량 산출
        dim = -1
        eps = 1e-6

        # 1. 기초 통계량 (Mean, Std)
        mu = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim, unbiased=True) + eps
        
        # 표준화된 잔차 (Standardized Residuals) 계산
        # x - mu 연산을 위해 mu와 std의 차원을 일시적으로 확장(unsqueeze)
        delta = x - mu.unsqueeze(dim)
        z = delta / std.unsqueeze(dim)

        # 2. 왜도 (Skewness): 분포의 비대칭성
        # [Robot DT]: 센서 데이터가 한쪽으로 편향되는 'Bias' 현상 감지
        skew = torch.mean(z ** 3, dim=dim)

        # 3. 첨도 (Kurtosis): 분포의 뾰족함과 꼬리 두께
        # [Voice MOS]: '퍽' 하는 팝 노이즈나 순간적인 피크치 감지
        # [Robot DT]: 로봇의 순간적인 충격(Jerk)이나 결함 징후 포착
        kurt = torch.mean(z ** 4, dim=dim)

        # 4. 모든 특징 결합 (Feature Fusion)
        # 결과 차원: (Batch, Hidden * 4) -> 각 특징점마다 4개의 통계적 지표를 가짐
        return torch.cat([mu, std, skew, kurt], dim=-1)

# ==============================================================================
# [공학적 기대]
# ==============================================================================
# 1. 추상화의 깊이: 
#    '모델의 충실도(Fidelity)' 측면에서, 왜도와 첨도의 추가는 
#    단순 요약을 넘어 신호의 '비선형적 특성'을 보존하는 고차원적 추상화 방식임.
#
# 2. 일반화 성능의 근거: 
#    어텐션은 특정 시점의 값에 집착할 수 있지만, 고차 모멘트 풀링은 데이터의 
#    '확률적 구조' 자체를 피처로 사용하므로 환경 변화에 훨씬 강건(Robust)함.
# ==============================================================================