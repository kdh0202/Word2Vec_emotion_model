
import streamlit as st

from emotion_engine import analyze_emotion
from music_data import recommend_songs

st.set_page_config(
    page_title="Word2Vec 마음 플레이리스트",
    page_icon="🎧",
)

st.title("🎧 마음 플레이리스트")
st.write(
    "오늘의 감정을 입력하면 Word2Vec 기반 유사도 분석으로 "
    "감정을 추정하고 노래를 추천해요."
)

st.info(
    "학습 데이터가 적은 실험용 모델입니다. "
    "결과는 실제 감정이나 감정일 확률을 확정하는 값이 아닙니다."
)

user_text = st.text_area(
    "오늘 어떤 일이 있었나요?",
    placeholder="예: 오늘 열심히 했는데 결과가 좋지 않아 속상해요.",
    height=120,
)

purpose = st.radio(
    "노래 추천 방식",
    ["현재 감정에 어울리는 노래", "기분 전환"],
    horizontal=True,
)

if st.button("감정 분석하기", type="primary"):
    if not user_text.strip():
        st.warning("먼저 감정을 입력해 주세요.")
    else:
        result = analyze_emotion(user_text)

        st.session_state["result"] = result
        st.session_state["purpose"] = purpose

if "result" in st.session_state:
    result = st.session_state["result"]

    st.divider()
    st.subheader(
        f"분석 결과: {result['emoji']} {result['emotion']}"
    )
    st.write(result["description"])

    if result["scores"]:
        st.caption(
            f"가장 높은 코사인 유사도: {result['score']:.3f}"
        )

        st.write("### 감정별 유사도 비교")

        for emotion, score in result["scores"].items():
            st.write(f"{emotion}: {score:.3f}")
            st.progress(max(0.0, min(1.0, score)))

    if result["emotion"] != "판단 어려움":
        st.write("### 🎵 추천 노래")

        songs = recommend_songs(
            result["emotion"],
            st.session_state["purpose"],
        )

        for title, artist, reason in songs:
            with st.container(border=True):
                st.markdown(f"**{title} — {artist}**")
                st.write(reason)

                query = f"{title} {artist}".replace(" ", "+")
                url = (
                    "https://www.youtube.com/results"
                    f"?search_query={query}"
                )
                st.markdown(
                    f"[YouTube에서 노래 검색하기]({url})"
                )
