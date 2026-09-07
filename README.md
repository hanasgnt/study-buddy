# 📚 Study Buddy

Chatbot AI berbasis **Gemini API + RAG (LangChain)** yang membantu belajar dari dokumenmu sendiri. Upload PDF atau paste teks materi, lalu tanya apa saja — bot akan menjawab berdasarkan isi dokumen tersebut, bukan mengarang.

Final Project — *LLM-Based Tools and Gemini API Integration for Data Scientists* (Hacktiv8).

## Use Case

Education bot / personal study assistant. Cocok untuk mahasiswa/pelajar yang ingin tanya-jawab cepat soal materi kuliah, artikel, atau catatan tanpa harus baca ulang seluruh dokumen.

## Fitur

- **Upload PDF atau paste teks** sebagai sumber materi.
- **RAG (Retrieval-Augmented Generation)**: dokumen di-chunk, di-embed, disimpan di vector database (Chroma), lalu bagian paling relevan diambil untuk menjawab pertanyaan — mengurangi halusinasi.
- **Konfigurasi parameter kreatif**:
  - Gaya bahasa: Santai / Formal
  - Domain pengetahuan opsional (misal: Biologi, Sejarah)
  - Temperature (kontrol kreativitas jawaban)
  - Top-K jumlah konteks yang diambil
- **Memory percakapan**: riwayat chat sebelumnya ikut jadi konteks jawaban.
- **Transparansi sumber**: jawaban bisa dicek dari potongan dokumen mana ia diambil.
- **Function Calling**: fitur "Analisis Dokumen" memakai Gemini function calling untuk memanggil fungsi Python langsung (hitung jumlah kata, karakter, estimasi waktu baca).

## Arsitektur

```
User Question
     │
     ▼
Embed (Gemini Embeddings)
     │
     ▼
Vector Search (Chroma) ──► ambil Top-K chunk relevan
     │
     ▼
Prompt (persona + konteks + riwayat) ──► Gemini LLM
     │
     ▼
Jawaban ke User
```

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI & deployment
- [Google Gemini API](https://ai.google.dev/) — LLM & embeddings
- [LangChain](https://www.langchain.com/) — orkestrasi RAG
- [Chroma](https://www.trychroma.com/) — vector database
- [pypdf](https://pypdf.readthedocs.io/) — ekstraksi teks PDF

## Instalasi Lokal

```bash
git clone <url-repo-ini>
cd study-buddy
pip install -r requirements.txt
streamlit run app.py
```

Masukkan Gemini API Key di sidebar (dapatkan gratis di [Google AI Studio](https://aistudio.google.com/app/apikey)).

## Deploy ke Streamlit Cloud

1. Push repo ini ke GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io/), hubungkan ke repo.
3. Set `app.py` sebagai entry point.
4. (Opsional) Simpan API Key sebagai secret, atau biarkan user input sendiri di sidebar seperti pada app ini.

## Cara Pakai

1. Masukkan Gemini API Key di sidebar.
2. Upload PDF atau paste teks materi.
3. Klik **Proses & Index Dokumen**.
4. Tanya apa saja soal isi dokumen di kolom chat.
5. (Opsional) Klik **Analisis Dokumen** untuk lihat statistik cepat.

## Struktur Repo

```
study-buddy/
├── app.py              # Aplikasi utama Streamlit
├── requirements.txt    # Dependencies
├── .env.example         # Contoh environment variable
└── README.md
```
