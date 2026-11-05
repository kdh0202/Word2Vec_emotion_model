from pathlib import Path
import csv
from collections import Counter

from gensim.models import Word2Vec


# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parent

# CSV 데이터 경로
DATA_PATH = BASE_DIR / "data" / "emotion_dataset.csv"

# 모델 저장 경로
MODEL_PATH = BASE_DIR / "emotion_word2vec.model"


# 감정 종류
EMOTIONS = ["행복", "슬픔", "분노", "피로", "평온"]


def tokenize(text):
    """
    문장부호를 제거하고 공백 기준으로 단어를 분리한다.
    """
    import re

    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def load_data():
    """
    CSV 파일에서 문장과 감정 데이터를 불러온다.
    """

    sentences = []
    emotion_counts = Counter()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {DATA_PATH}"
        )

    with open(DATA_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        # CSV 열 확인
        if not {"text", "emotion"}.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV 파일에는 text와 emotion 열이 필요합니다."
            )

        for row in reader:
            text = row["text"].strip()
            emotion = row["emotion"].strip()

            if not text:
                continue

            if emotion not in EMOTIONS:
                raise ValueError(
                    f"잘못된 감정 라벨: {emotion}"
                )

            tokens = tokenize(text)

            if tokens:
                sentences.append(tokens)
                emotion_counts[emotion] += 1

    print("전체 학습 문장 수:", len(sentences))

    print("\n감정별 데이터 수:")
    for emotion in EMOTIONS:
        print(f"{emotion}: {emotion_counts[emotion]}개")

    if not sentences:
        raise ValueError("학습할 데이터가 없습니다.")

    if len(emotion_counts) != len(EMOTIONS):
        raise ValueError("일부 감정 데이터가 누락되었습니다.")

    return sentences


def train_model():
    """
    CSV 데이터를 이용해 Word2Vec 모델을 학습한다.
    """

    print("CSV 데이터 불러오는 중...")

    sentences = load_data()

    print("\nWord2Vec 학습 시작...")

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        epochs=100,
        seed=42
    )

    # 기존 모델 파일과 같은 위치에 저장
    model.save(str(MODEL_PATH))

    print("\nWord2Vec 모델 학습 완료!")

    print("저장 위치:", MODEL_PATH)
    print("벡터 크기:", model.vector_size)
    print("학습된 단어 수:", len(model.wv))

    print("\n학습된 단어 목록:")
    print(model.wv.index_to_key)


if __name__ == "__main__":
    train_model()