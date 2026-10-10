cache = {}

def answer_question(question):
    if question in cache:
        print("CACHE HIT! Reusing previous answer.")
        return cache[question]

    print("CACHE MISS! Generating a new answer.")

    answer = f"This is an answer to: {question}"

    cache[question] = answer

    return answer

question = input("Enter question : ")
print(answer_question(question))
print(answer_question(question))
print(answer_question(question))
