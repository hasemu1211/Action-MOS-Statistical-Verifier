# Action-MOS-Statistical-Verifier

📈 Statistical Action-MOS: Robust Behavior Verifier
"Beyond Success: Quantifying the 'Quality' of Robot Motion through Higher-Order Statistics"

👤 About Me: A Statistical Perspective on Robotics
저는 통계학적 통찰을 바탕으로 생성된 시퀀스 데이터의 품질을 평가하고 검증하는 연구를 진행하고 있습니다.

저의 연구는 음성 합성 분야의 MOS(Mean Opinion Score) 예측에서 시작되었습니다. 대규모 자기지도학습(SSL) 모델의 임베딩 데이터에서 평균(Mean)을 넘어선 **고차 통계량(Higher-Order Statistics)**을 추출하여, 시스템의 강건성(Robustness)과 품질을 판별하는 기술을 보유하고 있습니다.

현재는 이 메커니즘을 로봇 디지털 트윈(Unity) 환경으로 전이하여, 로봇의 궤적(Trajectory)과 행동(Action)이 얼마나 자연스럽고 안전한지를 평가하는 'Action-MOS' 검증기를 구축하고 있습니다.

🎯 My Project Goal: Action-MOS Verifier
로봇이 목표에 도달했는가(Success Rate)라는 이진 결과보다, **"얼마나 신뢰할 수 있는 방식으로 움직였는가"**를 통계적으로 증명하는 것이 저의 프로젝트 목적입니다.

1. Advanced Statistical Pooling (ASP)
하샘표 통계 풀링 기법을 통해 데이터의 품질을 다각도로 분석합니다.

1차 & 2차 모멘트 (Mean, Std): 동작의 중심 경향과 전반적인 변동성(Jitter) 측정.

3차 & 4차 모멘트 (Skewness, Kurtosis): 데이터 분포의 비대칭성과 꼬리 두께를 분석하여, 예기치 못한 이상치(Outlier)나 충격(Jerk)을 정밀하게 감지.

2. Reward Model & Test-time Verifier
검증기(Verifier): 로봇이 생성한 여러 경로 후보 중, 통계적으로 가장 안정적인 경로를 Action-MOS 점수로 필터링합니다.

보상 모델(Reward Model): 단순 거리 기반 보상이 아닌, '통계적으로 부드러운(Smooth)' 동작에 높은 보상을 주는 리워드 시스템을 설계하여 시뮬레이션의 Sim-to-Real 강건성을 확보합니다.

🛠 Tech Stack & Core Assets
Core Logic: HigherOrderStatisticsPooling.py (PyTorch based Feature Extractor)

Backbone Candidates: VC-1, WavLM (Cross-domain Representation)

Environment: Unity 3D Digital Twin, ROS2 (In-progress)

Expertise: Time-series Data Analysis, Quality Prediction, OOD Robustness
