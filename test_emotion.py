
from emotion_engine import analyze_emotion

test_sentences = [
    "오늘 친구들과 놀아서 정말 행복해",
    "친구와 헤어져서 너무 슬퍼",
    "약속을 안 지켜서 정말 화가 나",
    "오늘 일이 너무 많아서 피곤해",
    "조용한 곳에서 쉬니까 마음이 편안해",
]

for sentence in test_sentences:
    print("\n" + "=" * 40)
    print("테스트 문장:", sentence)

    result = analyze_emotion(sentence)

    print("예측 감정:", result["emotion"])
    print("이모지:", result["emoji"])
    print("최고 유사도:", round(result["score"], 4))
