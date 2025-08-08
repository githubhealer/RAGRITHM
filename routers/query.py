from fastapi import APIRouter
import re
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import os

router = APIRouter(prefix="/query", tags=["query"])

class QueryRequest(BaseModel):
    text: str

class KeywordResponse(BaseModel):
    keywords: Dict[str, Any]
    embeddings: Optional[List[float]] = None

def extract_keywords(text: str) -> Dict[str, Any]:
    """
    Extract keywords from text input like '46M, knee surgery, Pune, 3-month policy'
    Returns categorized keywords
    """
    keywords = {
        "age": None,
        "gender": None,
        "medical_conditions": [],
        "procedures": [],
        "locations": [],
        "policy_duration": None,
        "general_keywords": []
    }
    
    # Clean and split text
    text = text.strip()
    parts = [part.strip() for part in text.split(',')]
    
    # Patterns for different types of information
    age_pattern = r'(\d+)\s*[Yy](?:ears?)?(?:\s+old)?|(\d+)M|(\d+)F'
    gender_pattern = r'\b(?:male|female|M|F)\b'
    policy_duration_pattern = r'(\d+)\s*-?\s*(?:month|year|day)s?\s*policy'
    
    # Medical procedure keywords (expandable list)
    medical_procedures = [
        'surgery', 'operation', 'procedure', 'treatment', 'therapy',
        'knee surgery', 'heart surgery', 'bypass', 'transplant',
        'chemotherapy', 'radiation', 'dialysis', 'rehabilitation',
        'appendectomy', 'tonsillectomy', 'cataract surgery', 'hip replacement',
        'angioplasty', 'stent', 'biopsy', 'endoscopy', 'colonoscopy'
    ]
    
    # Medical condition keywords (expandable list)
    medical_conditions = [
        'diabetes', 'hypertension', 'heart disease', 'cancer',
        'arthritis', 'asthma', 'kidney disease', 'liver disease',
        'stroke', 'pneumonia', 'bronchitis', 'migraine', 'epilepsy',
        'depression', 'anxiety', 'osteoporosis', 'thyroid', 'cholesterol'
    ]
    
    # Common Indian cities (expandable list)
    indian_cities = [
        'mumbai', 'delhi', 'bangalore', 'pune', 'chennai', 'kolkata',
        'hyderabad', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur',
        'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam',
        'surat', 'agra', 'vadodara', 'nashik', 'faridabad', 'meerut',
        'gurgaon', 'noida', 'ghaziabad', 'chandigarh', 'coimbatore'
    ]
    
    for part in parts:
        part_lower = part.lower()
        
        # Extract age
        age_match = re.search(age_pattern, part, re.IGNORECASE)
        if age_match:
            age = age_match.group(1) or age_match.group(2) or age_match.group(3)
            keywords["age"] = int(age)
            
            # Extract gender from age pattern
            if 'M' in part:
                keywords["gender"] = "Male"
            elif 'F' in part:
                keywords["gender"] = "Female"
        
        # Extract gender separately
        gender_match = re.search(gender_pattern, part, re.IGNORECASE)
        if gender_match and not keywords["gender"]:
            gender = gender_match.group().upper()
            keywords["gender"] = "Male" if gender in ['M', 'MALE'] else "Female"
        
        # Extract policy duration
        policy_match = re.search(policy_duration_pattern, part, re.IGNORECASE)
        if policy_match:
            keywords["policy_duration"] = part.strip()
        
        # Check for medical procedures
        for procedure in medical_procedures:
            if procedure.lower() in part_lower:
                if procedure not in keywords["procedures"]:
                    keywords["procedures"].append(procedure)
        
        # Check for medical conditions
        for condition in medical_conditions:
            if condition.lower() in part_lower:
                if condition not in keywords["medical_conditions"]:
                    keywords["medical_conditions"].append(condition)
        
        # Check for locations (Indian cities)
        for city in indian_cities:
            if city.lower() in part_lower:
                if city.title() not in keywords["locations"]:
                    keywords["locations"].append(city.title())
        
        # Add as general keyword if it doesn't fit other categories
        if (not any([
            re.search(age_pattern, part, re.IGNORECASE),
            re.search(gender_pattern, part, re.IGNORECASE),
            re.search(policy_duration_pattern, part, re.IGNORECASE),
            any(proc.lower() in part_lower for proc in medical_procedures),
            any(cond.lower() in part_lower for cond in medical_conditions),
            any(city.lower() in part_lower for city in indian_cities)
        ]) and len(part.strip()) > 1):
            keywords["general_keywords"].append(part.strip())
    
    return keywords

def generate_embeddings(text: str) -> Optional[List[float]]:
    """
    Generate embeddings for the input text using Gemini embeddings
    """
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel
        
        vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GOOGLE_CLOUD_REGION"))
        model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

@router.post("/extract-keywords", response_model=KeywordResponse)
async def extract_keywords_endpoint(request: QueryRequest):
    """
    Extract keywords and generate embeddings from input text
    Example input: "46M, knee surgery, Pune, 3-month policy"
    """
    # Extract keywords
    keywords = extract_keywords(request.text)
    
    # Generate embeddings for the input text
    embeddings = generate_embeddings(request.text)
    
    return KeywordResponse(
        keywords=keywords,
        embeddings=embeddings[0:3] if embeddings else None
    )