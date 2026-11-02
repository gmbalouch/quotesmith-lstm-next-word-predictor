import pandas as pd
import streamlit as st

from predictor import NextWordPredictor

st.set_page_config(page_title="Quote Next-Word Predictor", page_icon="✍️", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def get_predictor() -> NextWordPredictor:
    return NextWordPredictor()


predictor = get_predictor()

EXAMPLES = ["Life is", "I am going to", "The only thing we have to fear is", "Love is not"]

if "text" not in st.session_state:
    st.session_state.text = ""
if "generated" not in st.session_state:
    st.session_state.generated = None


def set_text(value: str):
    st.session_state.text = value
    st.session_state.generated = None


def append_word(word: str):
    st.session_state.text = (st.session_state.text.strip() + " " + word).strip()
    st.session_state.generated = None


# ---------------- Header ----------------
st.title("✍️ Quote Next-Word Predictor")
st.caption("An LSTM neural network trained on a dataset of quotes. Type the start of a sentence and see what it predicts.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    n_suggestions = st.slider("Suggestions to show", 3, 10, 5)
    st.subheader("Sentence generation")
    n_words = st.slider("Words to generate", 3, 30, 12)
    temperature = st.slider(
        "Creativity", 0.0, 1.5, 0.8, 0.1,
        help="0 = always pick the most likely word (repetitive). Higher = more varied, but less sensible.",
    )
    sample_k = st.slider("Choose from top-N words", 2, 50, 10)
    st.divider()
    st.caption(f"Vocabulary: {predictor.num_words:,} words · Context: last {predictor.max_len} words")

# ---------------- Input ----------------
st.text_input("Start typing a sentence", key="text", placeholder="e.g. Life is")

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, on_click=set_text, args=(ex,), width="stretch", key=f"ex_{ex}")

text = st.session_state.text.strip()
tab_next, tab_gen, tab_about = st.tabs(["Next word", "Generate sentence", "About"])

# ---------------- Tab 1: next word ----------------
with tab_next:
    if not text:
        st.info("Type something above, or pick an example.")
    else:
        preds, enc = predictor.top_k(text, n_suggestions)
        if enc.unknown:
            st.warning("Words the model doesn't know (ignored): " + ", ".join(f"`{w}`" for w in dict.fromkeys(enc.unknown)))
        if not preds:
            st.error("None of these words are in the model's vocabulary. Try different words.")
        else:
            st.subheader("Most likely next words")
            st.caption("Click a word to add it to your sentence.")
            bcols = st.columns(min(len(preds), 5))
            for i, (word, _) in enumerate(preds):
                bcols[i % len(bcols)].button(word, on_click=append_word, args=(word,), key=f"sug_{i}_{word}", width="stretch")
            df = pd.DataFrame(preds, columns=["Word", "Probability"])
            st.dataframe(
                df, hide_index=True, width="stretch",
                column_config={"Probability": st.column_config.ProgressColumn(
                    "Probability", min_value=0.0, max_value=1.0, format="percent")},
            )

# ---------------- Tab 2: generate ----------------
with tab_gen:
    if not text:
        st.info("Type the beginning of a sentence above first.")
    else:
        if st.button("Generate", type="primary"):
            with st.spinner("Generating…"):
                result, enc = predictor.generate(text, n_words, temperature, sample_k)
            st.session_state.generated = result if enc.ids else None
            if not enc.ids:
                st.error("None of these words are in the model's vocabulary.")
        if st.session_state.generated:
            st.markdown(f"> {st.session_state.generated}")
            st.caption("Press Generate again for a different result (unless Creativity is 0).")

# ---------------- Tab 3: about ----------------
with tab_about:
    st.markdown(
        """
**How it works** — text is lowercased, punctuation is removed, and words are converted to numbers.
An embedding layer feeds an LSTM, and a softmax layer gives a probability for every word in the vocabulary.

**Limitations** — this is a small model (~1.7M parameters) trained on quotes only.
It only knows words from its training vocabulary, and it is not a general chatbot. Expect quote-like, sometimes odd, output.
        """
    )
