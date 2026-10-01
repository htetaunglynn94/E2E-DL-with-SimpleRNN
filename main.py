
import re
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config( page_title="IMDB Sentiment Analyzer",
                    page_icon="🎬",
                    layout="wide",
                    initial_sidebar_state="collapsed")


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
/* Main application */
.stApp {
    background: linear-gradient(135deg, #0b1020 0%, #111827 55%, #172033 100%);
    color: #f8fafc;
}

/* Main container */
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {
    visibility: hidden;
}

/* Hero section */
.hero {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
}

.hero-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}

.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero p {
    color: #a5b4c8;
    font-size: 1.05rem;
    margin: 0;
}

/* Cards */
.card {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.5rem;
}

.muted {
    color: #94a3b8;
    font-size: 0.9rem;
}

/* Input */
.stTextArea textarea {
    background: #111827 !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
}

.stTextArea textarea:focus {
    border: 1px solid #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

/* Buttons */
.stButton > button {
    border: none;
    border-radius: 12px;
    background: linear-gradient(90deg, #0284c7, #6366f1);
    color: white;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.65rem 1.2rem;
    min-height: 3rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59,130,246,0.3);
    color: white;
}

/* Result */
.result {
    text-align: center;
    padding: 1.5rem;
    border-radius: 16px;
    margin: 0.5rem 0 1rem;
}

.positive {
    background: rgba(16,185,129,0.10);
    border: 1px solid rgba(16,185,129,0.45);
}

.negative {
    background: rgba(244,63,94,0.10);
    border: 1px solid rgba(244,63,94,0.45);
}

.result-icon {
    font-size: 2.5rem;
}

.result-label {
    font-size: 2rem;
    font-weight: 800;
    margin-top: 0.4rem;
}

.positive-text {
    color: #34d399;
}

.negative-text {
    color: #fb7185;
}

/* Metrics */
.metric {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(148,163,184,0.15);
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
}

.metric-value {
    color: #f8fafc;
    font-size: 1.5rem;
    font-weight: 800;
}

.metric-label {
    color: #94a3b8;
    font-size: 0.85rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    padding: 1.5rem 0 0;
}

/* Example review buttons */
div[data-testid="stHorizontalBlock"] button {
    min-height: 2.5rem;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------
@st.cache_resource
def load_resources():
    word_index = imdb.get_word_index()
    model = load_model("./model/simple_rnn_imdb.h5")
    return word_index, model


word_index, model = load_resources()


# --------------------------------------------------
# CLEAR ALL IN TEXT BOX
# --------------------------------------------------
def clear_review():
    st.session_state.review_input = ""
    st.session_state.pop("prediction_result", None)


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------
MAX_LEN = 500

def preprocess_text(text):
    """
    Convert text into a padded sequence compatible
    with the trained SimpleRNN model.
    """
    vocab_size = model.layers[0].input_dim

    words = re.sub(r"[^\w\s]", "", text.lower()).split()

    encoded_review = []

    for word in words:
        index = word_index.get(word, 0)

        if index == 0 or index + 3 >= vocab_size:
            encoded_review.append(2)
        else:
            encoded_review.append(index + 3)

    encoded_review = [1] + encoded_review

    padded_review = sequence.pad_sequences([encoded_review],
                                           maxlen=500,
                                           padding="pre",
                                           truncating="pre",
                                           value=0)

    return padded_review


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def predict_sentiment(text):
    preprocessed_input = preprocess_text(text)

    vocab_size = model.layers[0].input_dim

    if preprocessed_input.min() < 0 or preprocessed_input.max() >= vocab_size:
        raise ValueError(
            f"Invalid token index: {preprocessed_input.max()}, "
            f"vocabulary size: {vocab_size}")

    prediction = float(model.predict(preprocessed_input, verbose=0)[0][0])
    sentiment = "Positive" if prediction >= 0.5 else "Negative"

    return sentiment, prediction


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-icon">🎬 🍿</div>
    <h1>IMDB Sentiment Analyzer</h1>
    <p>Discover the sentiment behind every movie review
       using SimpleRNN deep learning.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("""
    <div class="card">
        <div class="card-title">✍️ Enter Your Movie Review</div>
        <div class="muted">
            Write or paste your review below to analyze
            its sentiment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Try an example:**")

    ex1, ex2 = st.columns(2)

    # Positive example
    with ex1:
        if st.button("😊 Positive example", use_container_width=True):
            st.session_state.review_input = (
                "I absolutely loved this movie! "
                "The performances were outstanding, "
                "the storyline was captivating, and "
                "every scene was beautifully directed. "
                "It was an amazing experience, and "
                "I would definitely watch it again."
            )
            st.session_state.pop("prediction_result", None)

    # Negative example
    with ex2:
        if st.button("😞 Negative example", use_container_width=True):
            st.session_state.review_input = (
                "This movie was terrible and boring. "
                "The acting was poor, the story was predictable, "
                "and the ending was disappointing. "
                "I regret wasting my time watching it."
            )
            st.session_state.pop("prediction_result", None)

    # Review input
    review = st.text_area(
        "Movie review",
        key="review_input",
        height=230,
        placeholder="Type your movie review here...",
        label_visibility="collapsed"
    )

    # Word and character count
    word_count = len(review.split())
    st.caption(
        f"📝 {len(review)} characters · {word_count} words"
    )

    # Analyze and Clear buttons
    col1, col2 = st.columns([3, 1])

    with col1:
        analyze = st.button(
            "🔍 Analyze Sentiment",
            use_container_width=True,
            type="primary"
        )

    with col2:
        st.button(
            "🗑️ Clear",
            use_container_width=True,
            on_click=clear_review
        )

    # Prediction
    if analyze:
        if not review.strip():
            st.warning("Please enter a movie review first.")
        else:
            with st.spinner("Analyzing your review..."):
                sentiment, score = predict_sentiment(review)

            st.session_state.prediction_result = {
                "sentiment": sentiment,
                "score": score
            }


# --------------------------------------------------
# ANALYSIS RESULT
# --------------------------------------------------
with right:
    st.markdown("""
    <div class="card">
        <div class="card-title">📊 Analysis Result</div>
        <div class="muted">
            The model's prediction will appear here.
        </div>
    </div>
    """, unsafe_allow_html=True)

    result = st.session_state.get("prediction_result")

    if result:
        sentiment = result["sentiment"]
        score = result["score"]

        if sentiment == "Positive":
            css_class = "positive"
            text_class = "positive-text"
            icon = "😊"
        else:
            css_class = "negative"
            text_class = "negative-text"
            icon = "😞"

        negative_score = 1 - score

        st.markdown(f"""
        <div class="result {css_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-label {text_class}">
                {sentiment}
            </div>
            <div class="muted">Predicted sentiment</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        m1, m2 = st.columns(2)

        with m1:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-value">{score:.2%}</div>
                <div class="metric-label">Positive probability</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric">
            </div>
            """, unsafe_allow_html=True)

        # Probability distribution
        st.markdown("#### Probability distribution")

        st.markdown(f"**Positive:** {score:.2%}")
        st.progress(float(score))

        st.markdown(f"**Negative:** {negative_score:.2%}")
        st.progress(float(negative_score))

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem 1rem;">
            <div style="font-size:3rem;">🧠</div>
            <h3 style="color:#cbd5e1;">
                Waiting for your review
            </h3>
            <p class="muted">
                Enter a movie review and click Analyze
                Sentiment to see the prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    Built by Htet Aung Lynn using Streamlit, TensorFlow & SimpleRNN
</div>
""", unsafe_allow_html=True)