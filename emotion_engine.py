
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from train_word2vec import EMOTION_DATA, tokenize


# 모델 파일 경로
MODEL_PATH = Path(__file__).with_name("emotion_word2vec.model")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "학습 모델이 없습니다. train_word2vec.py를 먼저 실행해 주세요."
    )

model = Word2Vec.load(str(MODEL_PATH))


# 감정별 이모지
EMOJI = {
    "행복": "😊",
    "슬픔": "😢",
    "분노": "😡",
    "피로": "😫",
    "평온": "😌",
}

# 감정별 안내 메시지
DESCRIPTIONS = {
    "행복": "기분 좋은 순간을 충분히 즐겨 보세요.",
    "슬픔": "속상한 마음을 천천히 돌아봐도 괜찮아요.",
    "분노": "잠시 숨을 고르고 마음을 정리해 보세요.",
    "피로": "잠깐 쉬면서 에너지를 회복해 보세요.",
    "평온": "지금의 차분한 기분을 이어가 보세요.",
}


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

    if np.linalg.norm(vector) == 0:
        return None

    return vector


def analyze_emotion(text):
    """입력 문장을 분석하고 감정별 유사도를 반환한다."""

    if not text or not text.strip():
        return {
            "emotion": "판단 어려움",
            "emoji": "🤔",
            "score": 0.0,
            "scores": {},
            "description": "감정을 분석할 문장을 입력해 주세요.",
        }

    # 사용자 입력 문장을 벡터로 변환
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

    # 감정별 예시 문장과 입력 문장의 유사도 계산
    for emotion, examples in EMOTION_DATA.items():
        example_scores = []

        for example in examples:
            reference_vector = sentence_vector(example)

            if reference_vector is None:
                continue

            similarity = cosine_similarity(
                user_vector.reshape(1, -1),
                reference_vector.reshape(1, -1),
            )[0, 0]

            example_scores.append(float(similarity))

        if example_scores:
            # 해당 감정 예시들의 평균 유사도
            scores[emotion] = float(np.mean(example_scores))
        else:
            scores[emotion] = 0.0

    if not scores or all(value == 0 for value in scores.values()):
        return {
            "emotion": "판단 어려움",
            "emoji": "🤔",
            "score": 0.0,
            "scores": scores,
            "description": "감정을 분석하기 어려워요. 다른 문장을 입력해 주세요.",
        }

    # 유사도가 가장 높은 감정을 선택
    best_emotion = max(scores, key=scores.get)
    best_score = scores[best_emotion]

    print("\n감정별 유사도")
    for emotion, score in scores.items():
        print(f"{emotion}: {score:.4f}")

    print("최종 분석 결과:", best_emotion)

    return {
        "emotion": best_emotion,
        "emoji": EMOJI.get(best_emotion, "🤔"),
        "score": best_score,
        "scores": scores,
        "description": DESCRIPTIONS.get(
            best_emotion,
            "입력 문장을 바탕으로 감정을 추정했어요.",
        ),
    }
