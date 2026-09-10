from groq import Groq
from dotenv import load_dotenv
import os
from retrieve import retrieve_passages

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(question, k=3):
    passages = retrieve_passages(question, k=k)
    context_text = "\n\n".join(passages)

    prompt = f"""Answer the question using only the information in the passages below. If the passages do not contain the answer, say "I cannot find the answer in the provided passages."

Passages:
{context_text}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",        
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    question = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
    answer = generate_answer(question)
    print(answer)
