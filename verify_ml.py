import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from utils import TextSummarizer, MCQGenerator, CodingRecommender

def test_summarizer():
    print("\n--- Testing TextSummarizer ---")
    summarizer = TextSummarizer()
    text = "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks."
    
    summary = summarizer.summarize(text, num_sentences=2)
    print("Original Length:", len(text.split('.')))
    print("Summary Length:", len(summary))
    print("Summary:", summary)
    
    if len(summary) > 0 and len(summary) <= 2:
        print("[PASS] Summarizer working.")
    else:
        print("[FAIL] Summarizer output unexpected.")

def test_mcq_generator():
    print("\n--- Testing MCQGenerator (ML Template Matching) ---")
    mcq_gen = MCQGenerator()
    text = "The CPU is the central processing unit of a computer. It is responsible for executing instructions. The speed of the CPU determines the overall performance of the system. RAM is used for temporary storage."
    
    questions = mcq_gen.generate_mcqs(text, num_questions=3)
    print("Generated Questions:", len(questions))
    
    for q in questions:
        print(f"Q: {q['question']}")
        print(f"   (Answer: {q['answer']})")
        
    if len(questions) > 0:
        print("[PASS] MCQ Generator with Template Matching working.")
    else:
        print("[FAIL] MCQ Generator failed.")

def test_coding_recommender():
    print("\n--- Testing CodingRecommender ---")
    recommender = CodingRecommender()
    topic = "arrays"
    suggestions = recommender.suggest_problems(topic)
    
    print(f"Topic: {topic}")
    print("Suggestions:", [p['title'] for p in suggestions])
    
    if len(suggestions) > 0:
        print("[PASS] Coding Recommender working.")
    else:
        print("[FAIL] Coding Recommender failed.")

def test_advanced_features():
    print("\n--- Testing Advanced Templates ---\n")
    from utils import MCQGenerator, CodingRecommender
    
    # 1. Test MCQs & Short Questions
    mcq_gen = MCQGenerator()
    text = "The quick sort algorithm is efficient. Recursion is defined as a function calling itself."
    print("Testing MCQ Templates:")
    mcqs = mcq_gen.generate_mcqs(text, 2)
    for q in mcqs:
        print(f"[{q.get('type', q.get('subtopic', 'General'))}] {q['question']}")
        
    print("\nTesting Short Questions:")
    sqs = mcq_gen.generate_short_questions(text, 2)
    for q in sqs:
        print(f"Q: {q['question']}")

    # 2. Test Coding Challenges
    print("\nTesting Coding Challenges:")
    coder = CodingRecommender()
    levels = ["easy", "medium", "hard", "algorithmic"]
    for lvl in levels:
        chall = coder.generate_challenge("Recursion", difficulty=lvl)
        print(f"[{lvl.upper()}] {chall['desc']}")

if __name__ == "__main__":
    try:
        # test_pipeline()
        test_advanced_features()
    except Exception as e:
        print(f"Global Error: {e}")
