# SQuAD RAG Chatbot

A retrieval-augmented Q&A chatbot built on the SQuAD dataset. Live demo: [squad-rag-chatbot.streamlit.app](https://squad-rag-chatbot.streamlit.app)

Ask it a question and it'll search a knowledge base of ~18,900 Wikipedia passages (from SQuAD), pull back the most relevant ones, and generate an answer grounded in that text rather than just answering from the model's own training data.

## How it works

1. **Ingestion** - SQuAD's train and validation splits get combined and deduplicated down to unique passages (a lot of questions in SQuAD share the same source paragraph).
2. **Embedding** - each passage gets turned into a vector using `sentence-transformers` (`all-MiniLM-L6-v2`), running locally, no API needed for this part.
3. **Retrieval** - a Chroma vector store holds all the embeddings. A question gets embedded the same way and compared against them to pull the top 3 closest matches.
4. **Generation** - those 3 passages plus the question get sent to Groq (running `openai/gpt-oss-120b`), which writes the actual answer, instructed to only use what was retrieved.

## Stack

- Embeddings: `sentence-transformers`, local, free, no rate limits
- Vector store: Chroma, also local
- Generation: Groq's free tier, no card required, fast
- Dataset: SQuAD (`rajpurkar/squad`), comes with ground truth answers so I could actually measure accuracy instead of just eyeballing it
- Frontend: Streamlit

## Results

Ran it against 100 random questions from SQuAD's validation set:

- Retrieval hit rate: 70% (the right passage showed up in the top 3)
- Answer accuracy: 44% (generated answer matched the ground truth)

## The interesting bug I found

The gap between those two numbers isn't just noise. Some of it is my accuracy check being strict (case-insensitive substring match, so a correctly worded but differently phrased answer gets marked wrong). But the real finding: when retrieval pulls back irrelevant passages, which happens on any question outside SQuAD's topics, the instruction telling the model to "only use the provided passages" can actually make it worse than if it had just used its own knowledge. I tested this directly, asked the deployed app what Japan's capital is, and it got it wrong, even though the underlying model obviously knows that, because it was forced to reason from irrelevant retrieved text instead.

Fix for a real version of this: add a relevance threshold on retrieval, only pass chunks to the model if they're actually similar enough to the question, otherwise skip grounding and let it answer normally or say it doesn't know.

## Running it yourself

```
git clone https://github.com/nidhy-abr/squad-rag-chatbot.git
cd squad-rag-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key
```

Then:
```
streamlit run app.py
```

First run builds the vector store from scratch, takes a few minutes since it's embedding ~18,900 passages.

## Files

- `app.py` - Streamlit UI
- `setup_vectorstore.py` - builds Chroma on first run
- `retrieve.py` - retrieval
- `generate.py` - prompt + Groq call
- `evaluate.py` - evaluation script
