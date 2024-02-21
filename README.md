# Image-Similarity-Project

### 팀원: 안현준, 유성민, 김백운, 김재원 

# **1. 프로젝트 개요**

## 주제 선택 동기

<image src='https://img.seoul.co.kr/img/upload/2016/09/20/SSI_20160920150331_O2.jpg' width='30%'>

인터넷으로 마음에 드는 옷을 구매하거나, 다른 사람이 입고 있는 옷이 궁금한 경우가 있을 것이다.   
제품의 정보를 모르거나 품절, 가격의 문제로 구매할 수 없을 때  
효과적으로 대체 품목을 추천해 주는 서비스를 제공하는 프로젝트를 진행하였다.


## 주제 관련 도메인 소개 

어떤 문제를 해결하려는지 설명: 모델이 입은 옷과 수집한 옷을 비교하여 비슷한 옷을 찾아줄 수 있는지. 


# **2. 데이터 수집**
   - 사용한 데이터에 대한 정보: 패션 쇼핑 웹사이트에서 크롤링하여 의류 이미지를 수집
     - 수집 방법: Selenium, request
     - 데이터 크기, 종류 등 정보: 1800개의 상의 이미지 (jpg, png)
 - 데이터 수집 및 전 처리 방법: 딥러닝을 사용하기 위해 다운로드한 이미지 파일을 크기 조정 및 전처리


```python
def crawling(page=1, category='001'):
    '''
    Init signature:
    crawling(
        page=1,
        category='001'
    ) -> 'DataFrame'
    Docstring:     
    category: '001' = 상의, '002' = 아우터, ...
    '''

    for i in range(1, page+1):
    
        query['page'] = i
        query_url = urllib.parse.urlencode(query, doseq=True)
        base_url = 'https://www.musinsa.com/categories/item/' + category # 001: 상의, 002: 아우터, ...
        url = base_url + '?' + query_url
        
        res = requests.get(url=url)
        dom = BeautifulSoup(res.text, 'html.parser')
        info = dom.select('#searchList > li div.li_inner')
        img = [(_.select('img')[0]['data-original'][:-7] + '500.jpg') if _.select('img')[0]['data-original'].endswith('jpg') else (_.select('img')[0]['data-original'][:-7] + '500.png') for _ in info]
        href = [_.select('a')[0]['href'][2:] for _ in info]
        title = [_.select('a')[0]['title'] for _ in info]
        brand = [_.select('p.item_title > a')[0].text for _ in info]
        price = [_.select('p.price')[0].find('del').decompose() if _.select('p.price')[0].find('del') != None else 0 for _ in info]
        price = [_.select('p.price')[0].text.split()[0] for _ in info]
        df = pd.concat([df, pd.DataFrame([img, href, title, brand, price]).T], axis=0)

    df.rename(columns={0:'이미지', 1:'주소', 2:'제품명', 3:'브랜드', 4:'최종가격'}, inplace=True)
    return df
```

## 크롤링한 데이터

![크롤링](/image/crawling_df.jpeg)

이미지, 주소, 제품명, 브랜드, 최종가격을 수집하였다.  
실시간 또는 주기적으로 수집이 이루어질 경우, 중복 데이터가 발생할 수 있다.  
이를 해결할 수 있는 다양한 방법들이 있다.  

![랭킹](/image/ranking_price.jpeg)
<image src='https://image.msscdn.net/images/goods_img/20230330/3192375/3192375_16843739731826_500.jpg' width='30%'>

가격별 상위, 하위 5개 데이터입니다.




## 수집한 데이터 RGB 분포

![EDA](/image/EDA.png)

수집한 이미지들의 R, G, B들의 분포를 확인했습니다.  

## RGB에 대한 기술통계량

![RGB_EDA](/image/RGB_EDA.jpeg)

각 이미지의 shape과 R,G,B 값들에 대한 통계량을 구했습니다.  
이런 여러가지 값들로부터 유사한 이미지를 찾을 수 있을 것 입니다.  

# **3. 사용한 딥러닝 모델 설명**
   - 사용한 딥러닝 모델의 구조, 기법 등에 대한 간단한 소개.
     	사용한 모델 : `Segformer`, `ResNet`

   - Segformer  
     transformer 기반의 세그멘테이션 모델
     의상 분할을 위해 이미지의 픽셀 수준에서 의류 영역을 식별하고 분할

   - ResNet  
     CNN(컨폴루션 뉴럴 네트워크) 구조이며, 이미지 특징을 추출하기 위해 사용

### Segformer  
모델의 이미지 중 상의 이미지만 추출하기 위해 Segmentation 모델인 `SegFormer` 모델 사용  

SegFormer 모델은 HONG KONG University와 NVIDIA에서 2021년 10월에 발표한 모델입니다.  

SegFormer의 주요 특징은  
1. 다양한 Scale의 특징들을 활용할 수 있다.
2. 학습 때 사용한 이미지의 해상도와 다른 크기의 이미지를 사용해도 성능 감소가 크지 않다.
3. 간단한 구조의 Decoder와 Encoder의 여러 계층에서 얻어낸 특징들을 통합하여 사용한다.

<image src='https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/segformer_architecture.png'>

SegFormer 구조  
1. 입력 이미지에 대해 4 × 4크기의 패치로 나눕니다.   
2. 계층적 구조의 Encoder에 넣어 원본 이미지 크기의 {1/4, 1/8, 1/16, 1/32}의 특징맵을 얻어냅니다.  
3. Encoder에서 얻어낸 모든 특징맵을 활용해 Decoder를 통해 최종 결과를 출력합니다.

### 학습 데이터  

ATR(Active Template Regression) dataset for clothes segmentation으로 학습된 SegFormer B2 모델을 이용했습니다.  

```
17,706개의 패션 이미지와 마스크 쌍의 데이터

background     0
hat            1
hair           2 
sunglass       3
upper-clothes  4
skirt          5
pants          6
dress          7
belt           8
left-shoe      9
right-shoe     10
face           11
left-leg       12
right-leg      13
left-arm       14
right-arm      15
bag            16
scarf          17
```

<image src='https://github.com/hugozanini/segformer-clothes-tfjs/raw/main/git-media/segformer-demo.gif?raw=true'>

> SegFormer 예시 (https://github.com/hugozanini/segformer-clothes-tfjs)

## 상의 옷 분리하기

![seg](/image/seg.png)

학습된 `SegFormer` 모델을 불러온 후 이미지를 적용한 결과입니다.

이후 추출된 상의 이미지으로부터 Feature를 추출하기 위해 `ResNet` 모델 사용했습니다.  

### ResNet50  

ResNet 모델은 마이크로소프트에서 개발한 알고리즘입니다.  

<image src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbki0hr%2Fbtrc8gb9QG9%2F9Zt5ob4zKIf1TtXYqEGtC0%2Fimg.png'>

> Deep Residual Learning for Image Recognition 인용 수

ResNet의 주요 특징은 `Gradient Vanishing` 문제를 해결하기 위해 제안되었다.    

<image src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FAeBv9%2Fbtrc3qmA8Nj%2FhurnDTuSkIK2ocWdmeyJH1%2Fimg.png'>

> ResNet 모델에 사용되는 기본적인 구조 (Residual Block)

$F(x) + x$ 를 최소화 하는 것이 목적  
위 구조가 쌓인 것이 Residual Network (ResNet)  


<image src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdWvmSt%2Fbtq8HUxeGbt%2FRYjh295Vsf1UTixT1xsKNk%2Fimg.png'>

ResNet50 구조는 layer마다 다른 residual block 형태가 반복되어 학습되는 과정을 거친다.

각 이미지가 가진 특징을 추출하고,  
모델의 이미지과 수집한 이미지들의 유사성을 확인하기 위해 `코사인 유사도` 계산을 수행했습니다.  

> 코사인 유사도: 벡터의 내적을 통해 두 벡터의 닮음을 계산한다.  

```python
# 이미지를 불러오고 특징 벡터로 변환하는 함수
def image_to_feature_vector(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_vector = model(image_tensor)
    return feature_vector

# 이미지 유사도를 계산하는 함수
def compute_image_similarity(image_feature1, image_feature2):
    return F.cosine_similarity(image_feature1, image_feature2).item()


# 상의 이미지의 경로 설정
upper_clothes_image_path = r'./upper_clothes.png'

# 미리 훈련된 ResNet 모델 불러오기
resnet_model = models.resnet50(weights=True)  # 변경
resnet_model.fc = nn.Identity()
resnet_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model.to(device)

# 상의 이미지의 특징 벡터 추출
upper_clothes_feature_vector = image_to_feature_vector(upper_clothes_image_path, resnet_model, device)

# 크롤링한 상의 이미지들과의 유사도 계산
folder_path = r'./상의_이미지/'

image_similarity_scores = []

for image_file in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_file)
    image_feature_vector = image_to_feature_vector(image_path, resnet_model, device)
    similarity = compute_image_similarity(upper_clothes_feature_vector, image_feature_vector)
    image_similarity_scores.append((image_path, similarity))

# 유사도를 기준으로 상위 10개 이미지를 선택
top_10_similar_images = sorted(image_similarity_scores, key=lambda x: x[1], reverse=True)[:10]
```   
	   
# **4. 모델 학습 결과**

## 학습 과정 설명
- 데이터 분할 및 학습, 검증, 테스트 데이터 세트 설명
- 학습 알고리즘 및 최적화 방법 설명
	Segformer와 ResNet 모델을 PyTorch 프레임워크를 사용하여 학습되어 있는 모델을 사용.

## 학습 결과
- 모델의 성능 지표 (정확도, 손실 등) 및 평가 결과
- 결과를 통해 얻은 통찰 혹은 해석

![비교이미지](/image/image1.png)
![비교](/image/image.png)
___
![RGB비교](/image/RGB_비교.png)


색깔 또는 패턴을 잘 찾는 것으로 보인다.  
극단값을 제외하면 비교된 이미지의 RGB 분포와 비슷한 모양을 보인다.  

![비교이미지](/image/image3.png)
![비교](/image/image4.png)

후드티나 옷에 있는 로고들도 잘 찾는 것으로 보인다.  

## 모델 개선 노력
- 모델 성능 향상을 위해 어떤 시도를 했는가?  
	수집된 이미지 배경이 흰색이 많아 입력 값의 배경을 검은색에서 흰색으로 수정했다.  
	다양한 이미지를 계속 비교하며 유사성을 찾으려해보았다.  

# **5. 결론**	
- 미래 작업 방향 및 업그레이드 계획  
	 1. 특징 추출 개선, 유사도 계산 속도  이미지 정확도 성능 개선과 이미지뿐만 아닌 다른 콘텐츠(ex. 가격 범위)도 추가할 예정.
	 2. 상의 뿐만 아닌 하의, 신발 등등 다른 카테고리 추가예정.
	 3. 하나의 카테고리안에서 좀 더 세분화하게 분류할 예정(ex. 상의 - 스트라이프 유무, 후드 유무, 로고 유무 등등..)	
		

- 회고
	- 안현준: 나의 첫 프로젝트가 완성되었다.
		프로젝트의 방향성과 추구하는 목적으로 마음 맞는 팀원들이 결성되었고,
		순조롭게 잘 마무리 되었다.
		평소 따로 공부해야지 하면서 미루고 미웠던 크롤링도 이번 프로젝트로 인해서 다시 공부하고 적용한 것이 뿌듯하였다.
		프로젝트를 하면서 느낀점은, 프로젝트가 어떻게 흘러가는지 대략적으로 알 수 있었다.
		하지만, 모르는 것들이 너무많았다. 
		딥러닝의 ‘딥’자 정도는 할 수 있는 그날까지…
 
	- 유성민: 이미지 검색을 더 빠르게 할 수 있는 방법에 대해 고민해보고, 크롤링을 자동화하여 데이터를 수집하는 것과 사용자로부터 입력값을 받으면 결과를 제공할 수 있도록 설계하는 것이 중요할 것 같다.
	- 김백운: 프로젝트를 통해 많은 것을 배우고 성장할 수 있었다. 초기 목표를 90% 이상 달성하여 만족스럽고, 특히 어려움을 해결하는 과정에서 팀원과 협력 능력을 향상될 수 있었다. 이전보다 적극적으로 참여하고 새로운 모델들을 경험했지만, 딥러닝 모델에 대한 이해가 부족하다는 점을 느낌. 
	- 김재원:  처음에 스냅사진에서 상의 하의를 구별하여 추출하는 부분이 어렵고 막막했는데, 팀원들과 같이 공부하고 찾아가면서 해결하는 과정이 너무 재미있었고 문제가 해결될때마다 짜릿했다. 아직 부족하지만 완성하여 너무 뿌듯하고 팀원들도 너무 잘 만나서 첫 프로젝트를 잘 마무리 할 수 있었던 것 같다!
	
