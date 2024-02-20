# Image-Similarity-Project

팀원: 안현준, 유성민, 김백운, 김재원

1. 프로젝트  개요:
    - 프로젝트의 주제 선택 동기: 모델이 입은 옷이 궁금하다.
    - 주제 관련  도메인 소개
        - 어떤 문제를 해결 하려는지 설명: 모델이 입은 옷과 수집한 옷과 비교하여 비슷한 옷을 찾을 수 있는지
2. 데이터 수집:
    - 사용한 데이터에 대한 정보
	    - 수집 방법: Selenium
	    - 데이터 크기, 종류 등 정보: 900개의 상의 사진
    - 데이터 수집 및 전 처리 방법: 제품명, 브랜드, 가격, 상세 주소, 이미지 주소 수집

```python
import selenium
...
```

3. 사용한 딥러닝 모델 설명
	    - 사용한 딥러닝 모델의 구조, 기법 등에 대한 간단한 소개.  
   모델의 사진 중 상의 사진만 추출하기 위해 Segmentation 모델인 `SegFormer` 모델 사용
   <image src='https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/segformer_architecture.png'>

   사진으로부터 Feature를 추출하기 위해 `ResNet` 모델 사용
   <image src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdWvmSt%2Fbtq8HUxeGbt%2FRYjh295Vsf1UTixT1xsKNk%2Fimg.png'>
   모델의 사진과 수집한 사진들의 유사성을 확인하기 위해 코사인 유사도 계산

   ```python
   import pytorch
   ...
   ```   
	   
5. 모델 학습 결과:
	 - 학습 과정 설명
		- 데이터 분할 및 학습, 검증, 테스트 데이터 세트 설명: 미리 학습된 모델 사용
		- 학습 알고리즘 및 최적화 방법 설명
	- 학습 결과
		- 모델의 성능 지표 (정확도, 손실 등) 및 평가 결과
		- 결과를 통해 얻은 통찰 혹은 해석

![비교](/image/image.png)
[0.861, 0.855, 0.854, 0.852, 0.848, 0.846, 0.846, 0.842, 0.84, 0.831]

	- 모델 개선 노력:
		- 모델 성능 향상을 위해 어떤 시도를 했는지
  	수집된 사진 배경이 흰색이 많아 입력값의 배경을 검은색에서 흰색으로 수정
  
  
		
6. 결론
	- 미래 작업 방향 및 업그레이드 계획: 특징 추출 개선, 유사도 계산 속도
	- 회고
