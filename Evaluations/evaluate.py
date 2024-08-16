import os
import markdown
from bs4 import BeautifulSoup
from bertopic import BERTopic

# Function to convert Markdown to plain text
def markdown_to_text(markdown_text):
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, features="html.parser")
    return soup.get_text()

# Path to the folder containing Markdown files
folder_path = 'privacy-policy-historical-master'

# Read all Markdown files
documents = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.md'):  # assuming Markdown files
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    markdown_text = f.read()
                    text = markdown_to_text(markdown_text)
                    if isinstance(text, str):  # Ensure it's a string
                        documents.append(text)
                    else:
                        print(f"Warning: Document from {file_path} is not a string.")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

# Debugging: Check document types
for i, doc in enumerate(documents):
    if not isinstance(doc, str):
        print(f"Document at index {i} is of type {type(doc)} and not a string.")

# Initialize BERTopic
topic_model = BERTopic()

# Fit BERTopic model
try:
    topics, probs = topic_model.fit_transform(documents)
    # Print topics
    print(topic_model.get_topic_info())
    # Inspect specific topics
    for topic_num in range(len(topic_model.get_topics())):
        print(f"Topic {topic_num}: {topic_model.get_topic(topic_num)}")
except Exception as e:
    print(f"Error during BERTopic fitting: {e}")
