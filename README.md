
# Image-Similarity-Project

### 팀원: 안현준, 유성민, 김백운, 김재원 

**1. 프로젝트 개요:**

- 프로젝트의 주제 선택 동기 : 인터넷으로 마음에 드는 옷을 구매하거나, 다른 사람이 입고 있는 옷이 궁금한 경우가 있을 것이다.   
제품의 정보를 모르거나 품절, 가격의 문제로 구매할 수 없을 때 
효과적으로 대체 품목을 추천해 주는 서비스를 제공하는 프로젝트를 진행하였다.



- 주제 관련 도메인 소개 

어떤 문제를 해결하려는지 설명: 모델이 입은 옷과 수집한 옷을 비교하여 비슷한 옷을 찾아줄 수 있는지. 


**2. 데이터 수집:**
   - 사용한 데이터에 대한 정보: 패션 쇼핑 웹사이트에서 크롤링하여 의류 이미지를 수집
     - 수집 방법: Selenium, request
     - 데이터 크기, 종류 등 정보: 1800개의 상의 사진 (jpg, png)
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

## 수집한 데이터 RGB 분포

![EDA](/image/EDA.png)


## RGB에 대한 기술통계량

![RGB_EDA](/image/RGB_EDA.jpeg)

**3. 사용한 딥러닝 모델 설명**
   - 사용한 딥러닝 모델의 구조, 기법 등에 대한 간단한 소개.
     	사용한 모델 : `Segformer`, `ResNet`

   - Segformer
     transformer 기반의 세그멘테이션 모델
     의상 분할을 위해 이미지의 픽셀 수준에서 의류 영역을 식별하고 분할

   - ResNet
     CNN(컨폴루션 뉴럴 네트워크) 구조이며, 이미지 특징을 추출하기 위해 사용

   모델의 사진 중 상의 사진만 추출하기 위해 Segmentation 모델인 `SegFormer` 모델 사용
   <image src='https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/segformer_architecture.png'>

   사진으로부터 Feature를 추출하기 위해 `ResNet` 모델 사용
   <image src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdWvmSt%2Fbtq8HUxeGbt%2FRYjh295Vsf1UTixT1xsKNk%2Fimg.png'>
   모델의 사진과 수집한 사진들의 유사성을 확인하기 위해 코사인 유사도 계산

   ![seg](/image/seg.png)

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

# ResNet18 모델
# resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=True)
# resnet_model.fc = nn.Identity()
# resnet_model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# resnet_model.to(device)

# 상의 이미지의 특징 벡터 추출
upper_clothes_feature_vector = image_to_feature_vector(upper_clothes_image_path, resnet_model, device)

# 크롤링한 상의 이미지들과의 유사도 계산
folder_path = r'./상의_사진/'

image_similarity_scores = []

for image_file in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_file)
    image_feature_vector = image_to_feature_vector(image_path, resnet_model, device)
    similarity = compute_image_similarity(upper_clothes_feature_vector, image_feature_vector)
    image_similarity_scores.append((image_path, similarity))

# 유사도를 기준으로 상위 10개 이미지를 선택
top_10_similar_images = sorted(image_similarity_scores, key=lambda x: x[1], reverse=True)[:10]
```   
	   
**4. 모델 학습 결과:**
	 - 학습 과정 설명
		- 데이터 분할 및 학습, 검증, 테스트 데이터 세트 설명
		- 학습 알고리즘 및 최적화 방법 설명
		Segformer와 ResNet 모델을 PyTorch 프레임워크를 사용하여 학습되어 있는 모델을 사용.

	- 학습 결과
		- 모델의 성능 지표 (정확도, 손실 등) 및 평가 결과
		- 결과를 통해 얻은 통찰 혹은 해석

![비교](/image/image.png)
![비교사진](/image/image1.png)


![RGB비교](/image/RGB_비교.png)

- 모델 개선 노력:
	- 모델 성능 향상을 위해 어떤 시도를 했는가?

수집된 사진 배경이 흰색이 많아 입력값의 배경을 검은색에서 흰색으로 수정
 다양한 이미지를 계속 비교하며 유사성을 찾으려해보았다.

  
  
		
**5. 결론**
	
- 미래 작업 방향 및 업그레이드 계획:

 특징 추출 개선, 유사도 계산 속도  이미지 정확도 성능 개선과 이미지뿐만 아닌 다른 콘텐츠도 추가할 예정이다.  

- 회고

		- 안현준:     
		- 유성민: 이미지 검색을 더 빠르게 할 수 있는 방법에 대해 고민을 해보아야겠다.  
		- 김백운:  
		- 김재원:  
   
   
