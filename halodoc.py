import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Analisis Sentimen Halodoc",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/ulasan_halodoc_stemmed.csv", encoding="utf-8")

    df["Stemmed"] = df["Stemmed"].apply(
        lambda x: " ".join(ast.literal_eval(x)) if isinstance(x, str) else x
    )
    return df

df = load_data()

X_text = df["Stemmed"]
y = df["label"]

# =========================
# TRAIN MODEL
# =========================
@st.cache_resource
def train_models(X_text, y):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ----- TANPA SMOTE -----
    svm_no_smote = SVC(kernel="linear")
    svm_no_smote.fit(X_train, y_train)
    pred_no = svm_no_smote.predict(X_test)
    acc_no = accuracy_score(y_test, pred_no)

    # ----- DENGAN SMOTE -----
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    svm_smote = SVC(kernel="linear")
    svm_smote.fit(X_train_res, y_train_res)
    pred_smote = svm_smote.predict(X_test)
    acc_smote = accuracy_score(y_test, pred_smote)

    return vectorizer, svm_no_smote, svm_smote, acc_no, acc_smote, y_train, y_train_res

vectorizer, svm_no_smote, svm_smote, acc_no, acc_smote, y_train, y_train_res = train_models(X_text, y)

# =========================
# HEADER
# =========================
st.markdown(
    "<h1 style='text-align:center;'>Analisis Sentimen Ulasan Pengguna Halodoc</h1>",
    unsafe_allow_html=True
)
st.write("Perbandingan Model **SVM tanpa SMOTE** dan **SVM dengan SMOTE**")

# =========================
# AKURASI
# =========================
col1, col2 = st.columns(2)
col1.metric("Akurasi SVM Tanpa SMOTE", f"{acc_no*100:.2f}%")
col2.metric("Akurasi SVM Dengan SMOTE", f"{acc_smote*100:.2f}%")


st.subheader("📊 Distribusi Sentimen Data Latih")

col1, col2 = st.columns(2)

# ===== TANPA SMOTE =====
with col1:
    st.markdown("**Tanpa SMOTE (Data Asli)**")
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    pd.Series(y_train).value_counts().plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Sentimen")
    ax1.set_ylabel("Jumlah")
    st.pyplot(fig1)

# ===== DENGAN SMOTE =====
with col2:
    st.markdown("**Dengan SMOTE (Data Seimbang)**")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    pd.Series(y_train_res).value_counts().plot(kind="bar", ax=ax2)
    ax2.set_xlabel("Sentimen")
    ax2.set_ylabel("Jumlah")
    st.pyplot(fig2)


# =========================
# INPUT ULASAN
# =========================
st.subheader("✍️ Prediksi Ulasan Baru")

input_text = st.text_area("Masukkan ulasan pengguna")

model_choice = st.radio(
    "Pilih Model",
    ("SVM Tanpa SMOTE", "SVM Dengan SMOTE")
)

if st.button("Prediksi Sentimen"):
    if input_text.strip() == "":
        st.warning("Ulasan tidak boleh kosong!")
    else:
        vec = vectorizer.transform([input_text])

        threshold = 0.25  # ambang netral

        if model_choice == "SVM Tanpa SMOTE":
            decision = svm_no_smote.decision_function(vec)
            pred_raw = svm_no_smote.predict(vec)[0]
        else:
            decision = svm_smote.decision_function(vec)
            pred_raw = svm_smote.predict(vec)[0]

        # ---- LOGIKA NETRAL ----
        margin = decision.ravel()[0]

        if abs(margin) < threshold:

            pred = "netral"
        else:
            pred = pred_raw.lower()

        # ---- OUTPUT ----
        if pred == "positif":
            st.success("Sentimen: **Positif** 😊")
        elif pred == "negatif":
            st.error("Sentimen: **Negatif** 😠")
        else:
            st.info("Sentimen: **Netral** 😐")
