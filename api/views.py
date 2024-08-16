from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import json
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
logger = logging.getLogger(__name__)
from transformers import BertTokenizer, BertModel
import torch

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def calculate_bert_similarity(text1, text2):
    if not text1.strip() or not text2.strip():
        logger.error("One or both texts are empty.")
        return 0.0

    # Tokenize the texts
    inputs = tokenizer([text1, text2], return_tensors='pt', truncation=True, padding=True)

    # Forward pass through BERT
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the [CLS] token embeddings for similarity
    embeddings = outputs.last_hidden_state[:, 0, :]  # Get the embeddings for [CLS] token
    similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()

    return similarity

def check_compliance_with_bert(text):
    criteria = [
        "data processing purposes",
        "data subject rights",
        "data protection officer",
        "lawful basis for processing",
        "data retention period",
        "data transfer",
        "security measures",
        "cookies and tracking",
        "children's privacy",
        "data breach notification",
        "automated decision-making"
    ]

    results = {}
    for criterion in criteria:
        similarity = calculate_bert_similarity(text, criterion)
        results[criterion] = {
            "similarity": similarity,
            "compliant": similarity > 0.6
        }
    
    is_compliant = all(result['compliant'] for result in results.values())
    results["is_compliant"] = is_compliant
    
    return results


@csrf_exempt
def check_similarity_bert(request):
    if request.method == 'POST':
        try:
            if 'file1' not in request.FILES or 'file2' not in request.FILES:
                return JsonResponse({'error': 'Both files are required.'}, status=400)

            file1 = request.FILES['file1']
            file2 = request.FILES['file2']
            
            file1_path = default_storage.save(file1.name, file1)
            file2_path = default_storage.save(file2.name, file2)
            
            file1_full_path = os.path.join(settings.MEDIA_ROOT, file1_path)
            file2_full_path = os.path.join(settings.MEDIA_ROOT, file2_path)
            
            text1 = read_markdown_file(file1_full_path)
            text2 = read_markdown_file(file2_full_path)
            
            if not text1 or not text2:
                return JsonResponse({'error': 'One or both files are empty or failed to read.'}, status=400)
            
            similarity = calculate_bert_similarity(text1, text2)
            
            return JsonResponse({'similarity': similarity})
        
        except Exception as e:
            logger.error(f"Error checking similarity with BERT: {str(e)}")
            return JsonResponse({'error': 'Failed to check similarity with BERT.'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def check_gdpr_compliance_bert(request):
    if request.method == 'POST':
        try:
            if 'file' not in request.FILES:
                logger.error("No file in request")
                return JsonResponse({'error': 'File is required.'}, status=400)

            file = request.FILES['file']
            file_path = default_storage.save(file.name, file)
            file_full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            
            logger.debug(f"File saved to: {file_full_path}")

            text = read_markdown_file(file_full_path)
            
            if not text:
                logger.error(f"File content is empty or failed to read: {file_full_path}")
                return JsonResponse({'error': 'The file is empty or failed to read.'}, status=400)
            
            compliance_results = check_compliance_with_bert(text)
            
            return JsonResponse(compliance_results)
        
        except Exception as e:
            logger.error(f"Error checking GDPR compliance with BERT: {str(e)}")
            return JsonResponse({'error': 'Failed to check GDPR compliance with BERT.'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def register(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            username = data['username']
            password = data['password']

            if User.objects.filter(username=username).exists():
                return JsonResponse({'error': 'Username already taken'}, status=400)

            user = User.objects.create_user(username=username, password=password)
            user.save()

            return JsonResponse({'message': 'User registered successfully'}, status=201)
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return JsonResponse({'error': 'Failed to register user'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def login(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            username = data['username']
            password = data['password']

            user = authenticate(username=username, password=password)
            if user is not None:
                return JsonResponse({'message': 'Login successful', 'token': 'dummy-token'}, status=200)
            else:
                return JsonResponse({'error': 'Invalid username or password'}, status=400)
        except Exception as e:
            logger.error(f"Error logging in user: {str(e)}")
            return JsonResponse({'error': 'Failed to login user'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        try:
            logger.debug(f"Files received: {request.FILES}")

            if 'file' in request.FILES:
                file = request.FILES['file']
                file_name = default_storage.save(file.name, file)
                logger.info(f"File '{file.name}' saved as '{file_name}'")
                
                file_url = default_storage.url(file_name)
                
                return JsonResponse({
                    'file_name': file_name,
                    'file_url': file_url
                })
            else:
                logger.warning('No file in request')
                return JsonResponse({'error': 'No file in request'}, status=400)
        
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            return JsonResponse({'error': 'Failed to upload file.'}, status=500)
    
    else:
        logger.warning('Invalid request method')
        return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def check_similarity(request):
    if request.method == 'POST':
        try:
            if 'file1' not in request.FILES or 'file2' not in request.FILES:
                return JsonResponse({'error': 'Both files are required.'}, status=400)

            file1 = request.FILES['file1']
            file2 = request.FILES['file2']
            
            file1_path = default_storage.save(file1.name, file1)
            file2_path = default_storage.save(file2.name, file2)
            
            file1_full_path = os.path.join(settings.MEDIA_ROOT, file1_path)
            file2_full_path = os.path.join(settings.MEDIA_ROOT, file2_path)
            
            text1 = read_markdown_file(file1_full_path)
            text2 = read_markdown_file(file2_full_path)
            
            if not text1 or not text2:
                return JsonResponse({'error': 'One or both files are empty or failed to read.'}, status=400)
            
            sections1, sections2 = split_into_sections(text1, text2)
            similarity = calculate_cosine_similarity(text1, text2)
            
            similar_texts, dissimilar_texts = categorize_text_sections(sections1, sections2)
            
            top_similar_texts = sorted(similar_texts, key=lambda x: x['similarity'], reverse=True)[:3]
            top_dissimilar_texts = sorted(dissimilar_texts, key=lambda x: x['similarity'])[:3]

            return JsonResponse({
                'similarity': similarity,
                'similar_texts': top_similar_texts,
                'dissimilar_texts': top_dissimilar_texts
            })
        
        except Exception as e:
            logger.error(f"Error checking similarity: {str(e)}")
            return JsonResponse({'error': 'Failed to check similarity.'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

def read_markdown_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        if not text.strip():
            logger.warning(f"No text extracted from {file_path}.")
        return text
    except Exception as e:
        logger.error(f"Error reading Markdown file: {str(e)}")
        return ""

def calculate_cosine_similarity(text1, text2):
    if not text1.strip() or not text2.strip():
        logger.error("One or both texts are empty.")
        return 0.0

    documents = [text1, text2]
    tfidf = TfidfVectorizer(analyzer='word', stop_words='english').fit_transform(documents)
    
    if tfidf.shape[1] == 0:
        logger.error("TF-IDF vectorizer returned an empty vocabulary.")
        return 0.0

    similarity_matrix = cosine_similarity(tfidf[0:1], tfidf)
    return similarity_matrix[0][1]

def split_into_sections(text1, text2):
    sections1 = text1.split('\n\n')
    sections2 = text2.split('\n\n')
    sections1 = [section.strip() for section in sections1 if section.strip()]
    sections2 = [section.strip() for section in sections2 if section.strip()]
    return sections1, sections2

def categorize_text_sections(sections1, sections2):
    similar_texts = []
    dissimilar_texts = []
    threshold = 0.5
    
    for i, section1 in enumerate(sections1):
        for j, section2 in enumerate(sections2):
            similarity = calculate_cosine_similarity(section1, section2)
            if similarity > threshold:
                similar_texts.append({
                    'section1_index': i,
                    'section2_index': j,
                    'similarity': similarity,
                    'section1': section1,
                    'section2': section2
                })
            else:
                dissimilar_texts.append({
                    'section1_index': i,
                    'section2_index': j,
                    'similarity': similarity,
                    'section1': section1,
                    'section2': section2
                })
    
    return similar_texts, dissimilar_texts

@csrf_exempt
def check_gdpr_compliance(request):
    if request.method == 'POST':
        try:
            if 'file' not in request.FILES:
                return JsonResponse({'error': 'File is required.'}, status=400)

            file = request.FILES['file']
            file_path = default_storage.save(file.name, file)
            file_full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            
            text = read_markdown_file(file_full_path)
            
            if not text:
                return JsonResponse({'error': 'The file is empty or failed to read.'}, status=400)
            
            compliance_results = check_compliance(text)
            
            return JsonResponse(compliance_results)
        
        except Exception as e:
            logger.error(f"Error checking GDPR compliance: {str(e)}")
            return JsonResponse({'error': 'Failed to check GDPR compliance.'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

def check_compliance(text):
    # Define criteria with associated keywords or phrases
    criteria = {
        "data_processing_purposes": {
            "keywords": ["purpose of data processing", "how we use your data"],
            "description": "The policy should clearly state the purposes for which personal data is processed."
        },
        "data_subject_rights": {
            "keywords": ["right to access", "right to rectification", "right to erasure"],
            "description": "The policy should inform users about their rights regarding their personal data."
        },
        "data_protection_officer": {
            "keywords": ["data protection officer", "contact information"],
            "description": "The policy should provide contact details for the data protection officer or equivalent."
        },
        "lawful_basis_for_processing": {
            "keywords": ["lawful basis for processing", "legal basis", "consent", "contract", "legal obligation"],
            "description": "The policy should specify the lawful basis for processing personal data."
        },
        "data_retention_period": {
            "keywords": ["data retention period", "how long we keep your data"],
            "description": "The policy should indicate how long personal data will be retained."
        },
        "data_transfer": {
            "keywords": ["data transfer", "third parties", "outside the EU"],
            "description": "The policy should detail any transfers of data to third parties or outside the EU."
        },
        "security_measures": {
            "keywords": ["security measures", "how we protect your data"],
            "description": "The policy should outline the measures in place to protect personal data."
        },
        "cookies_and_tracking": {
            "keywords": ["cookies", "tracking technologies"],
            "description": "The policy should explain the use of cookies and other tracking technologies."
        },
        "children_privacy": {
            "keywords": ["children's privacy", "children's data"],
            "description": "The policy should address how it handles data of children, if applicable."
        },
        "data_breach_notification": {
            "keywords": ["data breach notification", "how we handle data breaches"],
            "description": "The policy should describe how data breaches are handled and notified."
        },
        "automated_decision_making": {
            "keywords": ["automated decision-making", "profiling"],
            "description": "The policy should inform users about any automated decision-making or profiling."
        },
    }

    results = {}
    for criterion, details in criteria.items():
        # Check if any of the keywords are in the text
        found = any(keyword.lower() in text.lower() for keyword in details['keywords'])
        results[criterion] = {
            "description": details['description'],
            "compliant": found,
            "missing_info": not found
        }
    
    # Determine overall compliance
    is_compliant = all(result['compliant'] for result in results.values())
    results["is_compliant"] = is_compliant
    
    return results