import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader

st.set_page_config(page_title="Study Buddy", page_icon="📚", layout="wide")

# ---------------- Sidebar: Konfigurasi ----------------
st.sidebar.title("📚 Study Buddy")
st.sidebar.markdown("Chatbot belajar dari dokumenmu sendiri, ditenagai **Gemini API + RAG (LangChain)**.")

api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    st.error("GOOGLE_API_KEY belum dikonfigurasi. Tambahkan ke file .env.")
    st.stop()

genai.configure(api_key=api_key)

st.sidebar.subheader("⚙️ Konfigurasi Bot")
style = st.sidebar.selectbox("Gaya Bahasa", ["Santai", "Formal"])
domain = st.sidebar.text_input(
    "Domain Pengetahuan (opsional)",
    placeholder="misal: Biologi, Sejarah, Pemrograman",
)
temperature = st.sidebar.slider("Temperature (kreativitas jawaban)", 0.0, 1.0, 0.3, 0.05)
top_k = st.sidebar.slider("Top-K konteks yang diambil", 1, 8, 4)

if api_key:
    genai.configure(api_key=api_key)

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = 0

# ---------------- Header ----------------
st.title("📚 Study Buddy")
st.caption("Upload materi belajarmu (PDF/teks), lalu tanya apa saja soal isinya. Bot hanya jawab berdasarkan dokumenmu.")

# ---------------- Input Dokumen ----------------
tab1, tab2 = st.tabs(["📄 Upload PDF", "✏️ Paste Teks"])

with tab1:
    pdf_file = st.file_uploader("Upload file PDF", type=["pdf"])
    if pdf_file is not None:
        reader = PdfReader(pdf_file)
        text = "\n".join([(page.extract_text() or "") for page in reader.pages])
        st.session_state.doc_text = text
        st.success(f"PDF berhasil dibaca ({len(text.split())} kata).")

with tab2:
    pasted = st.text_area("Paste teks materi di sini", height=200, placeholder="Tempel catatan, artikel, atau materi kuliah...")
    if st.button("Gunakan teks ini"):
        st.session_state.doc_text = pasted
        st.success(f"Teks tersimpan ({len(pasted.split())} kata).")

process = st.button("🔎 Proses & Index Dokumen", type="primary", disabled=not api_key)

if process:
    if not st.session_state.doc_text.strip():
        st.warning("Upload PDF atau paste teks dulu, ya.")
    else:
        with st.spinner("Memproses dan meng-index dokumen..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

            chunks = splitter.split_text(st.session_state.doc_text)

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key
            )

            vectorstore = Chroma(
                collection_name="study_buddy",
                embedding_function=embeddings,
                persist_directory="./chroma_db"
            )

            vectorstore.add_texts(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.doc_chunks = len(chunks)
            st.session_state.messages = []

        st.success(
            f"Dokumen ter-index jadi {len(chunks)} bagian. Siap ditanya!"
        )

if st.session_state.vectorstore is not None:
    st.info(f"✅ Dokumen aktif: {st.session_state.doc_chunks} bagian ter-index. Silakan tanya di bawah.")

st.divider()

# ---------------- Persona / System Prompt ----------------
def build_system_prompt():
    gaya = "santai dan ramah, seperti teman belajar" if style == "Santai" else "formal dan terstruktur, seperti tutor profesional"
    domain_line = f" Fokus domain: {domain}." if domain else ""
    return (
        "Kamu adalah Study Buddy, asisten belajar AI.\n"
        f"Gaya bicaramu {gaya}.{domain_line}\n"
        "Jawab HANYA berdasarkan konteks dokumen yang diberikan di bawah.\n"
        "Jika jawaban tidak ada di dalam konteks, katakan dengan jujur bahwa informasi itu "
        "tidak ditemukan di dokumen — jangan mengarang jawaban.\n"
        "Selalu gunakan Bahasa Indonesia."
    )

# ---------------- Riwayat Chat ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Tanya sesuatu soal dokumenmu...")

if question:
    if not api_key:
        st.error("Masukkan Gemini API Key dulu di sidebar.")
    elif st.session_state.vectorstore is None:
        st.error("Proses dokumen dulu sebelum bertanya.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])

        # Memory: sertakan beberapa riwayat percakapan terakhir sebagai konteks tambahan
        history_msgs = st.session_state.messages[-7:-1]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_msgs]) or "(belum ada riwayat)"

        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=api_key, temperature=temperature)

        prompt = ChatPromptTemplate.from_messages([
            ("system", build_system_prompt()),
            ("human",
             "Riwayat percakapan sebelumnya:\n{history}\n\n"
             "Konteks dari dokumen:\n{context}\n\n"
             "Pertanyaan pengguna: {question}"),
        ])

        chain = prompt | llm | StrOutputParser()

        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                try:
                    answer = chain.invoke({"history": history_text, "context": context, "question": question})
                except Exception as e:
                    error_text = str(e)
                    if "429" in error_text or "ResourceExhausted" in error_text or "quota" in error_text.lower():
                        answer = (
                            "⏳ Gemini API sedang mencapai batas quota free tier. "
                            "Coba lagi setelah quota tersedia atau gunakan project/API key dengan quota yang berbeda."
                        )
                    else:
                        answer = f"⚠️ Terjadi error saat memproses jawaban: {error_text}"
                    st.warning(answer)
                else:
                    st.markdown(answer)
                    with st.expander("📎 Lihat bagian dokumen yang dipakai sebagai konteks"):
                        for i, d in enumerate(docs, 1):
                            st.caption(f"Bagian {i}")
                            st.text(d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""))

        st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()

# ---------------- Fitur Tambahan: Function Calling ----------------
st.subheader("🧮 Analisis Cepat Dokumen")
st.caption("Fitur tambahan memakai Function Calling: Gemini memanggil fungsi Python secara langsung untuk menghitung statistik dokumen (bukan menebak-nebak).")


def get_document_stats():
    """Menghitung statistik dasar dari dokumen yang sedang dipelajari: jumlah kata,
    jumlah karakter, dan estimasi waktu baca dalam menit (asumsi 200 kata/menit)."""
    text = st.session_state.doc_text
    word_count = len(text.split())
    char_count = len(text)
    reading_time = round(word_count / 200, 1)
    return {
        "jumlah_kata": word_count,
        "jumlah_karakter": char_count,
        "estimasi_waktu_baca_menit": reading_time,
    }


analyze_disabled = not st.session_state.doc_text.strip() or not api_key
if st.button("📊 Analisis Dokumen", disabled=analyze_disabled):
    with st.spinner("Menganalisis..."):
        try:
            model = genai.GenerativeModel(
                model_name="gemini-flash-latest",
                tools=[get_document_stats],
            )
            chat = model.start_chat(enable_automatic_function_calling=True)
            response = chat.send_message(
                "Analisis dokumen yang sedang aku pelajari. Panggil fungsi yang tersedia untuk "
                "mendapatkan datanya, lalu jelaskan hasilnya dengan bahasa yang mudah dipahami."
            )
            st.info(response.text)
        except Exception as e:
            error_text = str(e)
            if "429" in error_text or "ResourceExhausted" in error_text or "quota" in error_text.lower():
                st.warning(
                    "⏳ Lagi kena limit request Gemini API (free tier cuma 5 request/menit). "
                    "Tunggu sekitar 20-30 detik dulu, terus coba lagi ya."
                )
            else:
                st.error(f"⚠️ Terjadi error: {error_text}")

if analyze_disabled:
    st.caption("⚠️ Masukkan API Key dan proses dokumen dulu untuk memakai fitur ini.")
