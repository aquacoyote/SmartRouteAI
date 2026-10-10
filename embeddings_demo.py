from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
model = SentenceTransformer("all-MiniLM-L6-v2")

questions = [ "What is Python?" ,
 "Explain what Python is." ,
 "How do I cook biryani?" ,
]

embeddings = model.encode(questions)

for i in range(len(questions)):
    for j in range(i+1,len(questions)):
        similiarity = cos_sim(
            embeddings[i],
            embeddings[j]
        ).item()

        print(f"\nQuestion A: {questions[i]}")
        print(f"Question B: {questions[j]}")
        print(f"Similiarity : {similiarity:.3f}")


