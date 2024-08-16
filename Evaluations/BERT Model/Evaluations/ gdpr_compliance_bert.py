

import os
import re
import glob
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Load and preprocess the Markdown files from nested folders
def load_policies_from_md(root_directory):
    policies = []
    for folder_path in glob.glob(os.path.join(root_directory, '**'), recursive=True):
        if os.path.isdir(folder_path):
            for filepath in glob.glob(os.path.join(folder_path, '*.md')):
                with open(filepath, 'r', encoding='utf-8') as file:
                    policy_text = file.read()
                    policy_text = preprocess_text(policy_text)
                    policies.append(policy_text)
    return policies

def preprocess_text(text):
    # Remove markdown headers, links, and special characters
    text = re.sub(r'#+\s', '', text)  # Remove headers
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove links
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

# Load the policies
policies = load_policies_from_md('privacy-policy-historical-master')

# Step 2: Define GDPR-related phrases and generate pseudo-labels
gdpr_phrases = {
    "data_processing_purposes": ["purpose of data processing", "how we use your data"],
    "data_subject_rights": ["right to access", "right to rectification", "right to erasure"],
    "data_protection_officer": ["data protection officer", "contact information"],
    "lawful_basis_for_processing": ["lawful basis for processing", "legal basis", "consent", "contract", "legal obligation"],
    "data_retention_period": ["data retention period", "how long we keep your data"],
    "data_transfer": ["data transfer", "third parties", "outside the EU"],
    "security_measures": ["security measures", "how we protect your data"],
    "cookies_and_tracking": ["cookies", "tracking technologies"],
    "children_privacy": ["children's privacy", "children's data"],
    "data_breach_notification": ["data breach notification", "how we handle data breaches"],
    "automated_decision_making": ["automated decision-making", "profiling"]
}

def generate_pseudo_labels(policy_text, gdpr_phrases):
    labels = []
    for criterion, phrases in gdpr_phrases.items():
        if all(phrase in policy_text.lower() for phrase in phrases):
            labels.append(1)
        else:
            labels.append(0)
    return labels

# Generate pseudo-labels
pseudo_labels = [generate_pseudo_labels(policy, gdpr_phrases) for policy in policies]

# Step 3: Split dataset into training and testing sets
texts = policies
labels = pseudo_labels

texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 4: Prepare the dataset for BERT
class PrivacyPolicyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs, torch.tensor(label, dtype=torch.float)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = PrivacyPolicyDataset(texts_train, labels_train, tokenizer)
test_dataset = PrivacyPolicyDataset(texts_test, labels_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Step 5: Train the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(gdpr_phrases))

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):  # Number of epochs
    for batch in train_loader:
        inputs, labels = batch
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Step 6: Evaluate the BERT model
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).tolist()
        predictions.extend(preds)
        true_labels.extend(labels.tolist())

true_labels = [int(label) for label in true_labels]
predictions = [int(pred) for pred in predictions]

accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
