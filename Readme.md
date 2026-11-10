# 🎵 Word2Vec 기반 감정 분석 및 음악 추천 시스템

## 📌 프로젝트 소개

사용자가 입력한 한국어 문장을 자연어 처리(NLP) 기술로 분석하여 감정을 분류하고, 해당 감정에 맞는 음악을 추천하는 프로젝트입니다.

Word2Vec 기반 문장 임베딩과 코사인 유사도를 활용하여 감정을 분석하며, TF-IDF 가중치 및 머신러닝 분류 모델을 추가해 성능을 비교했습니다.

### 프로젝트 목표

- 한국어 문장의 감정을 5가지 범주로 분류
- Word2Vec을 이용한 의미 기반 유사도 검색 구현
- TF-IDF 가중치를 활용한 감정 분석 성능 개선
- 머신러닝 모델별 성능 비교 및 오류 분석
- 감정에 따른 음악 추천 기능 구현

## 🛠 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python |
| NLP | Kiwi, Word2Vec, TF-IDF |
| Machine Learning | Gensim, scikit-learn |
| Data Processing | Pandas, NumPy |
| Similarity | Cosine Similarity |
| Web Framework | Streamlit |
| Version Control | Git, GitHub |
| Development | PyCharm |

## 🧠 감정 분석 시스템

사용자의 입력 문장을 다음 5가지 감정으로 분류합니다.

| 감정 | 예시 |
|---|---|
| 😊 행복 | 오늘 정말 기분 좋은 일이 있었어 |
| 😢 슬픔 | 기대했던 일이 잘되지 않아서 속상해 |
| 😡 분노 | 내 물건을 허락 없이 사용해서 화가 나 |
| 😫 피로 | 하루 종일 일해서 너무 지쳤어 |
| 😌 평온 | 오늘은 마음이 편안하고 차분해 |

### 감정 분석 처리 과정

1. 사용자가 한국어 문장을 입력합니다.
2. Kiwi 형태소 분석기로 문장을 토큰화합니다.
3. Word2Vec 모델을 이용해 단어를 벡터로 변환합니다.
4. 단어 벡터를 평균 내어 문장 벡터를 생성합니다.
5. 각 감정의 대표 벡터와 코사인 유사도를 계산합니다.
6. 가장 유사도가 높은 감정을 최종 결과로 출력합니다.
7. 분석된 감정에 맞는 음악을 추천합니다.

## 📊 데이터셋 구성

5가지 감정에 대해 총 1,500개의 한국어 문장 데이터를 구성했습니다.

| 감정 | 학습 데이터 | 테스트 데이터 |
|---|---:|---:|
| 행복 | 240 | 60 |
| 슬픔 | 240 | 60 |
| 분노 | 240 | 60 |
| 피로 | 240 | 60 |
| 평온 | 240 | 60 |
| **합계** | **1,200** | **300** |

초기 감정 문장을 기반으로 표현을 변형하여 데이터를 확장했습니다.

유사 문장이 학습 데이터와 테스트 데이터에 동시에 포함되는 문제를 줄이기 위해 유사 문장 그룹 단위로 데이터를 분리했습니다.

## 🔬 모델 구현 및 비교

### Model 1. Word2Vec + Cosine Similarity

- Kiwi 기반 한국어 형태소 분석
- Word2Vec Skip-gram 학습
- 단어 임베딩의 단순 평균으로 문장 벡터 생성
- 감정별 대표 벡터와 코사인 유사도 비교

### Model 2. TF-IDF + Word2Vec

- 기존 Word2Vec 임베딩 활용
- TF-IDF를 이용해 단어별 가중치 계산
- 가중 평균으로 문장 벡터 생성
- 감정별 대표 벡터와 코사인 유사도 비교

### Model 3. TF-IDF + Logistic Regression

- Kiwi 기반 형태소 분석
- TF-IDF 특징 벡터 생성
- Logistic Regression을 이용한 지도학습 감정 분류

## 📈 모델 성능 비교

총 300개의 합성 테스트 문장으로 세 모델을 평가했습니다.

| 모델 | 정확도 | 정답 문장 |
|---|---:|---:|
| Word2Vec | 70.00% | 210 / 300 |
| **TF-IDF + Word2Vec** | **73.00%** | **219 / 300** |
| TF-IDF + Logistic Regression | 66.67% | 200 / 300 |

### 실험 결과

TF-IDF 가중 평균을 적용한 Word2Vec 모델이 기존 모델보다 **3.00%p 높은 정확도**를 기록했습니다.

감정별 분석에서는 슬픔의 재현율이 63.3%에서 80.0%로 개선되었지만, 행복의 재현율은 41.7%에서 35.0%로 감소했습니다.

따라서 전체 정확도뿐 아니라 감정별 오분류를 함께 분석해야 한다는 점을 확인했습니다.

### 평가의 한계

현재 데이터는 원본 문장의 변형으로 생성된 합성 데이터이며, 테스트 데이터도 모델 개선 과정에서 반복적으로 사용되었습니다.

따라서 현재 정확도를 실제 사용자 입력에 대한 일반화 성능으로 해석하기는 어렵습니다. 향후 독립적인 평가 데이터를 구성하여 추가 검증할 계획입니다.

## 🎵 음악 추천 기능

감정 분석 결과를 바탕으로 해당 감정에 맞는 음악을 추천합니다.

음악 추천 기능과 감정 분석 모델은 분리하여 구성했으며, 향후 추천 정확도 및 개인화 기능을 개선할 계획입니다.

## 📂 주요 프로젝트 파일

| 파일 | 설명 |
|---|---|
| `app.py` | Streamlit 애플리케이션 |
| `train_word2vec.py` | Word2Vec 학습 및 형태소 분석 |
| `emotion_engine.py` | 기본 Word2Vec 감정 분석 |
| `emotion_engine_tfidf.py` | TF-IDF 가중 Word2Vec 감정 분석 |
| `emotion_engine_logistic.py` | Logistic Regression 감정 분류 |
| `compare_models.py` | 두 Word2Vec 모델 비교 |
| `compare_three_models.py` | 세 가지 모델 성능 비교 |
| `evaluate_model.py` | 모델 성능 평가 |
| `prepare_data.py` | 학습 및 테스트 데이터 분리 |
| `music_data.py` | 음악 추천 데이터 |

## 🚀 실행 방법

### 1. 저장소 복제

```bash
git clone https://github.com/kdh0202/Word2Vec_emotion_model.git
cd Word2Vec_emotion_model
```

### 2. 가상환경 생성 및 활성화

Windows PowerShell 기준:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 4. Word2Vec 모델 학습

```bash
python train_word2vec.py
```

### 5. 모델 성능 비교

```bash
python compare_three_models.py
```

### 6. Streamlit 앱 실행

```bash
streamlit run app.py
```

실행 전 필요한 데이터 파일과 모델 파일이 준비되어 있어야 합니다.

## 🔧 향후 개선 계획

- 감정별 학습 문장 다양성 확대
- 독립적인 실제 문장 평가 데이터 구축
- 행복과 피로 감정의 오분류 개선
- Word2Vec 모델 하이퍼파라미터 비교
- 다양한 한국어 감정 분류 모델과 성능 비교
- Streamlit 사용자 인터페이스 개선
- 감정별 음악 추천 품질 개선

## 💡 프로젝트를 통해 학습한 내용

- 한국어 텍스트 전처리 및 형태소 분석
- Word2Vec 임베딩 학습 및 문장 벡터 생성
- 코사인 유사도를 이용한 감정 분류
- TF-IDF 기반 특징 가중치 적용
- 지도학습과 비지도 임베딩 기반 분류 방식 비교
- 혼동행렬, Precision, Recall, F1-score를 활용한 모델 평가
- Git 및 GitHub 기반 프로젝트 버전 관리