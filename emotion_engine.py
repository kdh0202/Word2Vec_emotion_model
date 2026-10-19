
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from train_word2vec import EMOTION_DATA, tokenize

MODEL_PATH = "emotion_word2vec.model"

model = Word2Vec.load(MODEL_PATH)

EMOJI = {
    "행복": "😊",
    "슬픔": "😢",
    "분노": "😡",
    "피로": "😫",
    "평온": "😌",
}


def sentence_vector(text):
    """문장 속 단어 벡터의 평균을 계산한다."""
    tokens = tokenize(text)

    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if not vectors:
        return None

    return np.mean(vectors, axis=0)


def analyze_emotion(text):
    user_vector = sentence_vector(text)

    # 학습 어휘와 겹치는 단어가 없을 때
    if user_vector is None or np.linalg.norm(user_vector) == 0:
        return {
            "emotion": "판단 어려움",
            "emoji": "🤔",
            "score": 0.0,
            "scores": {},
            "description": (
                "입력 문장에서 학습한 단어를 찾지 못했어요. "
                "다른 표현으로 입력해 주세요."
            ),
        }

    scores = {}

    for emotion, examples in EMOTION_DATA.items():
        reference_vectors = []

        for example in examples:
            vector = sentence_vector(example)

            if vector is not None:
                reference_vectors.append(vector)

        if not reference_vectors:
            scores[emotion] = 0.0
            continue

        # 감정별 기준 문장 벡터의 평균
        reference_vector = np.mean(
            reference_vectors,
            axis=0
        )

        score = cosine_similarity(
            user_vector.reshape(1, -1),
            reference_vector.reshape(1, -1)
        )[0][0]

        scores[emotion] = float(score)

    best_emotion = max(scores, key=scores.get)
    best_score = scores[best_emotion]

    descriptions = {
        "행복": "기분 좋은 순간을 즐겨 보세요.",
        "슬픔": "속상한 마음을 천천히 돌아봐도 괜찮아요.",
        "분노": "잠시 숨을 고르고 마음을 정리해 보세요.",
        "피로": "잠깐 쉬면서 에너지를 회복해 보세요.",
        "평온": "지금의 차분한 기분을 이어가 보세요.",
    }

    return {
        "emotion": best_emotion,
        "emoji": EMOJI[best_emotion],
        "score": best_score,
        "scores": scores,
        "description": descriptions[best_emotion],
    }
