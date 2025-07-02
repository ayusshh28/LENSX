import streamlit as st
import joblib
import pandas as pd
from streamlit_lottie import st_lottie
import requests

spam_model=joblib.load("spam_detect.pkl")
language_model=joblib.load("lang_detect.pkl")
news_model=joblib.load("news_category.pkl")
review_model=joblib.load("review.pkl")


# ---------------------------
# Load Lottie animation from URL or local JSON
# ---------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ---------------------------
# Page config & custom CSS
# ---------------------------
st.set_page_config(
    page_title="LENS EXPERT – NLP Suite",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    :root {
        --violet: #6C63FF;
        --violet-dark: #5248e8;
        --gradient: linear-gradient(135deg, #6C63FF 0%, #B388FF 100%);
        --shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }

    html, body {
        font-family: 'Poppins', sans-serif;
        background: #eef1f7;
    }

    .stTabs [role="tab"] {
        font-size=30px;
        padding: 12px 20px;
    }
    .lensx-box {
        padding: 10px 20px;
        border-left: 6px solid #2563eb;
        background-color: #f1f5f9;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin-bottom:40px;
    }

    .lensx-title {
        font-size: 54px;
        font-weight: 900;
        color: #1e3a8a;
        text-shadow: 2px 2px #dbeafe;
        
    }

    .lensx-subtitle {
        font-size: 20px;
        font-weight: 500;
        color: #1e3a8a;

        
    }

    .stButton > button[kind="primary"] {
        background: var(--gradient);
        color: #fff;
        border: none;
        padding: 0.7rem 2.2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.1);
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }

    .css-1y4p8pa {
        border: 2px dashed var(--violet);
        border-radius: 1rem;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #ffffff;
        box-shadow: var(--shadow);
        padding: 1.5rem;
        border-radius: 0 1rem 1rem 0;
    }

    [data-testid="stSidebar"] .st-expanderHeader {
        font-weight: bold;
        font-size: 1.1rem;
    }
            

    </style>
""", unsafe_allow_html=True)





# --- Sidebar style ---
st.markdown("""
    <style>
        /* Sidebar width */
        section[data-testid="stSidebar"] {
            width: 360px !important;
            padding: 20px 15px;
        }

        /* Make sidebar scrollable */
        section[data-testid="stSidebar"] > div:first-child {
            overflow-y: auto;
            max-height: 100vh;
        }

        /* Increase default sidebar font size */
        .sidebar-content, .st-emotion-cache-1xw8zd0, .st-emotion-cache-1v0mbdj {
            font-size: 17px !important;
        }

        /* Style for expander headers */
        .stSidebar .st-expanderHeader, .st-expanderHeader {
            font-size: 18px !important;
            font-weight: 600;
        }

        /* Input box styling */
        .stSidebar .stTextInput input, 
        .stSidebar .stTextArea textarea {
            font-size: 16px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)




# ---------------------------
# Load assets & models
# ---------------------------
lottie_nlp = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_qp1q7mct.json")


# ---------------------------
# Header Section: Lottie + Branding
# ---------------------------
col1, col2 = st.columns([2, 3])
with col1:
    st_lottie(lottie_nlp, height=140, speed=1)
with col2:
    st.markdown("""
        <div class="lensx-box">
            <div class="lensx-title">LensX </div>
        <span class="lensx-subtitle"> (Lens Expert - NLP Suite) </span>  
        <div style='font-size: 1rem; color: #444; margin-top: 0.4rem; font-weight:bold'>
            <em>From Spam to Sentiment – Know It All with <strong style='color: #1e3a8a'>LensX</strong></em>
        </div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Tabs layout
# ---------------------------





spam_tab, lang_tab, review_tab, news_tab = st.tabs([
    "🚨 Spam Classifier",
    "🌐 Language Detection",
    "🍽️ Review Sentiment",
    "📰 News Classification",
])

# === 1. Spam Classifier ===
with spam_tab:
    st.subheader("🔍 Detect whether a message is Spam or Not Spam")
    msg = st.text_input("Enter a message")
    if st.button("Classify"):
        prediction = spam_model.predict([msg])
        if prediction[0]==0:

            st.error("🚫 Spam Alert! This message appears to be spam.")
            st.markdown("### 🧨 Message flagged as SPAM!")
            st.markdown("Be cautious while interacting with this message.")
          
        
        else:
            st.success("✅ This message is clean and not spam.")
            st.balloons()



    st.divider()
    uploaded_spam = st.file_uploader("Batch classify messages (CSV/TXT)", type=["csv", "txt"], key="spam_upload")
    if uploaded_spam:
        try:
            lines = uploaded_spam.read().decode("utf-8").splitlines()
            df_spam = pd.DataFrame(lines, columns=['Msg'])
            df_spam["Prediction"] = pd.Series(spam_model.predict(df_spam['Msg'])).map({0: "Spam", 1: "Not Spam"})
            df_spam.index += 1
            st.dataframe(df_spam, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {e}")

# === 2. Language Detection ===
with lang_tab:
    st.subheader("🌍 Identify the language of a text snippet")
    text_input = st.text_input("Enter text to detect language", key="lang_input")
    if st.button("Detect", key="lang_detect_btn") and text_input:
        lang_pred = language_model.predict([text_input])[0]
        st.success(f"Detected Language: **{lang_pred}**")

    st.divider()
    uploaded_lang = st.file_uploader("Batch detect languages (CSV/TXT)", type=["csv", "txt"], key="lang_upload")
    if uploaded_lang:
        try:
            lines = uploaded_lang.read().decode("utf-8").splitlines()
            df_lang = pd.DataFrame(lines, columns=["Text"])
            df_lang["Language"] = pd.Series(language_model.predict(df_lang["Text"]))
            df_lang.index += 1
            st.dataframe(df_lang, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {e}")

# === 3. Review Sentiment ===
with review_tab:
    st.subheader("🍽️ Analyse restaurant review sentiment")
    review_text = st.text_input("Write a restaurant review", key="review_input")
    if st.button("Analyse", key="review_btn") and review_text:
        review_pred = review_model.predict([review_text])[0]
        if review_pred:
            st.success("😊 Positive Review")
            st.balloons()
        else:
            st.error("😞 Negative Review")

    st.divider()
    uploaded_reviews = st.file_uploader("Batch analyse reviews (CSV/TXT)", type=["csv", "txt"], key="review_upload")
    if uploaded_reviews:
        try:
            lines = uploaded_reviews.read().decode("utf-8").splitlines()
            df_reviews = pd.DataFrame(lines, columns=["Review"])
            df_reviews["Sentiment"] = pd.Series(review_model.predict(df_reviews["Review"])).map({0: "Negative", 1: "Positive"})
            df_reviews.index += 1
            st.dataframe(df_reviews, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")

with news_tab:
    st.markdown("""
    <div style="padding: 20px; background-color: #fff3cd; border-left: 6px solid #ffa500; font-size: 16px;">
        ⚠️ <strong>News Classifier is currently under maintenance.</strong><br>
        We're working hard to improve it. Please check back later!
    </div>
    """, unsafe_allow_html=True)


# ---------------------------
# Sidebar info
# ---------------------------
# --- Sidebar content ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 10px;">
  <h3 style="font-weight: 600; font-size: 20px; line-height: 1.6;">
    <span style="font-size: 24px;">👋</span>
    <span style="
      background: linear-gradient(90deg, #FF6F61, #FF8C42);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: bold;
    ">Hello</span>
    <span style="color: #2F2F2F;">User, Welcome to</span>
    <span style="color: #1e3a8a; font-weight: bold;">LensX</span>
  </h3>
</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("🧑‍💻 About the Project", expanded=False):
    st.write("LENSX (LENS Expert) is an interactive NLP suite built with Streamlit.")
    st.write("It showcases classification in 4 key domains: Spam, Language, Review Sentiment, and News.")
    st.markdown("💡 **Technologies**: Python, Streamlit, Pandas, Scikit-learn")
    st.markdown("🔍 **ML Models**: Trained on real-world text datasets")
    st.markdown("📂 **Demo Purpose**: Educational / Exploratory")

with st.sidebar.expander("📞 Contact & Links", expanded=False):
    st.write("📧 Email: support@lensxpert.com")
    st.write("📍 Location: India, Remote")
    st.markdown("🔗 **Live App**: [lens-x.streamlit.app](https://lens-x.streamlit.app)")

with st.sidebar.expander("💬 Feedback"):
    st.write("Have suggestions? We’d love to hear from you.")
    st.text_input("Your email")
    st.text_area("Your feedback")

with st.sidebar.expander("✨ Credits"):
    st.markdown("- 👨‍💼 Developed by Ayush\n- 🤖 NLP Models: Sklearn\n- 🎨 UI by Streamlit + Lottie")

with st.sidebar.expander("📌 Version"):
    st.markdown("**App Version**: 1.0.1\n**Last Updated**: June 30, 2025")


