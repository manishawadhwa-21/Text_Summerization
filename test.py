from transformers import pipeline

def generate_summary(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summary[0]['summary_text']

# Example Text
input_text = """
Your long input text goes here. Provide a large chunk of text that you want to summarize.
"""

# Generate Summary
summary = generate_summary(input_text)

# Print the Summary
print("Original Text:")
print(input_text)
print("\nSummary:")
print(summary)
