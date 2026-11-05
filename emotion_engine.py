
from pathlib import Path
import csv

import numpy as np
from gensim.models import Word2Vec

from train_word2vec import tokenize


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_word2vec.model"
DATA_PATH = BASE_DIR / "data" / "emotion_dataset.csv"

EMOTIONS = ["행복", "슬픔", "분노", "피로", "평온"]

EMOJI = {
    "행복": "😊",
    "슬픔": "😢",
    "분노": "😡",
    "피로": "😫",
    "평온": "😌",
}

DESCRIPTIONS = {
    "행복": "기분 좋은 순간을 충분히 즐겨 보세요.",
    "슬픔": "속상한 마음을 천천히 돌아봐도 괜찮아요.",
    "분노": "잠시 숨을 고르고 마음을 정리해 보세요.",
    "피로": "잠깐 쉬면서 에너지를 회복해 보세요.",
    "평온": "지금의 차분한 기분을 이어가 보세요.",
}


# Word2Vec 모델 불러오기
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "학습 모델이 없습니다. train_word2vec.py를 먼저 실행해 주세요."
    )

model = Word2Vec.load(str(MODEL_PATH))


def sentence_vector(text, debug=False):
    """문장을 단어 벡터의 평균으로 변환한다."""

    tokens = tokenize(text)

    known_tokens = [
        word for word in tokens
        if word in model.wv
    ]

    if debug:
        print("입력 문장:", text)
        print("분리된 단어:", tokens)
        print("학습된 단어:", known_tokens)

    if not known_tokens:
        return None

    vectors = [model.wv[word] for word in known_tokens]
    vector = np.mean(vectors, axis=0)

    norm = np.linalg.norm(vector)

    if norm == 0:
        return None

    return vector


def cosine_score(vector_a, vector_b):
    """두 벡터의 코사인 유사도를 계산한다."""

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b) / (norm_a * norm_b)
    )


def build_emotion_vectors():
    """CSV 문장으로 감정별 대표 벡터를 생성한다."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {DATA_PATH}"
        )

    emotion_vectors = {
        emotion: [] for emotion in EMOTIONS
    }

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            emotion = row["emotion"].strip()
            text = row["text"].strip()

            if emotion not in emotion_vectors:
                continue

            vector = sentence_vector(text)

            if vector is not None:
                emotion_vectors[emotion].append(vector)

    centroids = {}

    for emotion, vectors in emotion_vectors.items():

        if not vectors:
            raise ValueError(
                f"{emotion} 감정의 유효한 문장이 없습니다."
            )

        centroid = np.mean(vectors, axis=0)

        # 대표 벡터 정규화
        norm = np.linalg.norm(centroid)

        if norm > 0:
            centroid = centroid / norm

        centroids[emotion] = centroid

        print(
            f"{emotion} 대표 벡터 생성 완료: "
            f"{len(vectors)}개 문장"
        )

    return centroids


# 프로그램 시작 시 한 번만 생성
EMOTION_VECTORS = build_emotion_vectors()


def analyze_emotion(text):
    """입력 문장의 감정을 분석한다."""

    if not text or not text.strip():
        return {
            "emotion": "판단 어려움",
            "emoji": "🤔",
            "score": 0.0,
            "scores": {},
            "description": "감정을 분석할 문장을 입력해 주세요.",
        }

    user_vector = sentence_vector(text, debug=True)

    if user_vector is None:
        return {
            "emotion": "판단 어려움",
            "emoji": "🤔",
            "score": 0.0,
            "scores": {},
            "description": (
                "입력 문장에서 학습된 단어를 찾지 못했어요. "
                "다른 표현으로 입력해 주세요."
            ),
        }

    scores = {}

    for emotion, reference_vector in EMOTION_VECTORS.items():

        similarity = cosine_score(
            user_vector,
            reference_vector
        )

        scores[emotion] = similarity

    best_emotion = max(scores, key=scores.get)
    best_score = scores[best_emotion]

    print("\n감정별 유사도")

    for emotion, score in scores.items():
        print(f"{emotion}: {score:.4f}")

    print("\n최종 분석 결과:", best_emotion)

    return {
        "emotion": best_emotion,
        "emoji": EMOJI.get(best_emotion, "🤔"),
        "score": best_score,
        "scores": scores,
        "description": DESCRIPTIONS.get(
            best_emotion,
            "입력 문장을 바탕으로 감정을 추정했어요."
        ),
    }
