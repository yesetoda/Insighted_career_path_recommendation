# from fastapi import FastAPI, HTTPException, UploadFile, File
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from typing import List, Dict, Optional
# import pandas as pd
# import numpy as np
# from catboost import CatBoostClassifier, Pool
# import io
# import logging
# from datetime import datetime


# career_labels = [
#     "Software Engineer", "Data Scientist", "Civil Engineer", "Mechanical Engineer",
#     "Electrical Engineer", "Chemical Engineer", "Biomedical Engineer", "Environmental Scientist",
#     "Architect", "Graphic Designer", "Web Developer", "Network Administrator",
#     "Database Administrator", "Cybersecurity Analyst", "AI Researcher", "Product Manager",
#     "Business Analyst", "Marketing Specialist", "Sales Representative", "Financial Analyst",
#     "Accountant", "Actuary", "Economist", "Statistician", "Operations Manager",
#     "Human Resources Manager", "Public Relations Specialist", "Content Writer", "Editor",
#     "Journalist", "Photographer", "Videographer", "Social Media Manager", "UX/UI Designer",
#     "Game Developer", "Mobile App Developer", "Cloud Solutions Architect", "DevOps Engineer",
#     "Quality Assurance Tester", "Research Scientist", "Lab Technician", "Pharmacist",
#     "Nurse", "Physician", "Dentist", "Veterinarian", "Physical Therapist",
#     "Occupational Therapist", "Speech-Language Pathologist", "Psychologist",
#     "Social Worker", "Counselor", "Teacher", "Professor", "Education Administrator",
#     "Librarian", "Historian", "Anthropologist", "Sociologist", "Political Scientist",
#     "International Relations Specialist", "Lawyer", "Paralegal", "Judge",
#     "Mediator", "Forensic Scientist", "Crime Analyst", "Firefighter", "Police Officer",
#     "Emergency Medical Technician", "Military Officer", "Logistics Coordinator",
#     "Supply Chain Manager", "Real Estate Agent", "Urban Planner", "Insurance Underwriter",
#     "Investment Banker", "Stockbroker", "Retail Manager", "Hospitality Manager", "Chef",
#     "Event Planner", "Tour Guide", "Travel Consultant", "Fashion Designer",
#     "Textile Engineer", "Interior Designer", "Jewelry Designer", "Cosmetologist",
#     "Fitness Trainer", "Sports Coach", "Athletic Director", "Professional Athlete",
#     "Music Producer", "Sound Engineer", "Composer", "Actor", "Dancer",
#     "Theater Director", "Film Director", "Screenwriter", "Animator", "Illustrator",
#     "Art Director", "Curator", "Museum Educator", "Art Restorer",
#     "Cultural Anthropologist", "Philosopher", "Theologian", "Ethicist", "Linguist",
#     "Translator", "Interpreter", "Foreign Service Officer", "Diplomat",
#     "Nonprofit Administrator", "Grant Writer", "Fundraising Manager",
#     "Community Organizer", "Environmental Policy Analyst", "Agricultural Scientist",
#     "Food Scientist", "Nutritional Consultant", "Veterinary Technician",
#     "Wildlife Biologist", "Marine Biologist", "Geologist", "Meteorologist",
#     "Astrophysicist", "Oceanographer", "Climate Scientist", "Renewable Energy Consultant",
#     "Mining Engineer", "Petroleum Engineer", "Urban Ecologist", "Fashion Merchandiser",
#     "Retail Buyer", "E-commerce Specialist", "Digital Marketing Analyst", "SEO Specialist",
#     "Brand Strategist", "Market Research Analyst", "Customer Service Manager",
#     "Corporate Trainer", "Business Consultant", "Change Management Specialist",
#     "Risk Manager", "Compliance Officer", "Healthcare Administrator",
#     "Medical Records Technician", "Health Educator", "Public Health Advisor",
#     "Epidemiologist", "Health Policy Analyst", "Sports Psychologist",
#     "Rehabilitation Counselor", "Clinical Research Coordinator", "Pharmaceutical Sales Representative",
#     "Toxicologist", "Microbiologist", "Geneticist", "Biochemist", "Biophysicist",
#     "Biotechnologist", "Clinical Lab Scientist", "Health Information Manager",
#     "Digital Content Creator", "Podcast Producer", "Online Course Developer"
# ]

# # ——— Setup —
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(
#     title="Career Path Recommendation API",
#     description="Recommend career paths based on student profiles and scores",
#     version="1.0.0",
# )

# # ——— Load model —
# career_model: Optional[CatBoostClassifier] = None
# logger.info("Loading CatBoost model...")
# try:
#     career_model = CatBoostClassifier()
#     career_model.load_model("career_recommender.cbm")
#     expected_features = career_model.feature_names_
#     logger.info(f"Model loaded. Expected features: {expected_features}")
# except Exception as e:
#     logger.error(f"Error loading model: {e}")
#     expected_features = []

# # ——— Feature lists —
# CATEGORICAL_FEATURES = [
#     "Gender", "Health Issue", "Career Interest", "Father's Education",
#     "Mother's Education", "Parental Involvement", "Home Internet Access",
#     "Electricity Access", "School Type", "School Location", "Field Choice",
# ]
# NUMERIC_SUBJECTS = [
#     "Grade 12 - Civics and Ethical Education Test Score",
#     "Grade 12 - Affan Oromoo Test Score",
#     "Grade 12 - English Test Score",
#     "Grade 12 - HPE Test Score",
#     "Grade 12 - ICT Test Score",
# ]
# ALL_EXPECTED = expected_features  # align with model

# # ——— Schemas —
# class CareerRequest(BaseModel):
#     gender: str
#     date_of_birth: Optional[str] = None
#     age: Optional[int] = None
#     health_issue: str
#     career_interest: Optional[str] = None
#     fathers_education: Optional[str] = None
#     mothers_education: Optional[str] = None
#     parental_involvement: str
#     home_internet_access: str
#     electricity_access: str
#     school_type: str
#     school_location: str
#     field_choice: str
#     scores: Dict[str, float]



# class CareerPrediction(BaseModel):
#     career: str
#     probability: float

# class CareerResponse(BaseModel):
#     recommended_career: str
#     confidence: float
#     top_10_careers: List[CareerPrediction]

# class PredictionRecord(BaseModel):
#     index: int
#     recommended_career: str
#     confidence: float
#     top_10_careers: List[CareerPrediction]


# # Modified prediction logic
# def get_top_predictions(probabilities: np.ndarray, n: int = 10) -> List[CareerPrediction]:
#     """Get top N predictions with career names"""
#     sorted_indices = np.argsort(-probabilities)
#     return [
#         CareerPrediction(
#             career=career_labels[i],
#             probability=round(float(probabilities[i]), 4)
#         )
#         for i in sorted_indices[:n]
#     ]

# # ——— Health check —
# @app.get("/health")
# async def health():
#     return {"status": "ok", "model_loaded": career_model is not None}

# # ——— Required features —
# @app.get("/required-features")
# async def required_features():
#     return {"expected_features": ALL_EXPECTED}

# # ——— Preprocessing —
# def preprocess(df: pd.DataFrame) -> pd.DataFrame:
#     # Convert date_of_birth to age
#     if "date_of_birth" in df.columns and df["date_of_birth"].notna().any():
#         year = datetime.now().year
#         df["dob_year"] = pd.to_datetime(df["date_of_birth"], errors="coerce").dt.year
#         df["age"] = year - df["dob_year"]
#         df.drop(columns=["date_of_birth", "dob_year"], inplace=True)
#     if "age" not in df.columns or df["age"].isnull().all():
#         raise ValueError("Provide 'age' or valid 'date_of_birth'.")

#     # Fill categorical
#     for col in CATEGORICAL_FEATURES:
#         key = col.lower().replace(' ', '_')
#         df[col] = df.get(key, 'Unknown').astype(str).fillna('Unknown')

#     # Fill numeric
#     for subj in NUMERIC_SUBJECTS:
#         key = subj.lower().replace(' ', '_')
#         if key in df.columns:
#             df[subj] = pd.to_numeric(df[key], errors='coerce').fillna(df[key].median())
#         else:
#             df[subj] = 0.0

#     # Align DataFrame with expected features
#     X = pd.DataFrame({feat: df.get(feat, pd.Series(['Unknown'] * len(df)) if feat in CATEGORICAL_FEATURES else pd.Series([0.0] * len(df))) for feat in ALL_EXPECTED})
#     return X

# # ——— Single prediction —
# @app.post("/predict_career_single", response_model=CareerResponse)
# async def predict_career_single(req: CareerRequest):
#     if career_model is None:
#         raise HTTPException(503, "Model not loaded.")
    
#     try:
#         data = req.dict(exclude_none=True)
#         merged = {**{k: v for k, v in data.items() if k != 'scores'}, **data.get('scores', {})}
#         df = pd.DataFrame([merged])
        
#         processed = preprocess(df)
#         cat_idx = [i for i, f in enumerate(processed.columns) if f in CATEGORICAL_FEATURES]
#         pool = Pool(data=processed, cat_features=cat_idx)
        
#         probs = career_model.predict_proba(pool)[0]
#         top_predictions = get_top_predictions(probs)
        
#         return CareerResponse(
#             recommended_career=top_predictions[0].career,
#             confidence=top_predictions[0].probability,
#             top_10_careers=top_predictions
#         )
        
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         raise HTTPException(500, "Prediction failed.")


# # ——— Batch CSV prediction —
# @app.post("/predict_career_csv", response_model=List[PredictionRecord])
# async def predict_career_csv(file: UploadFile = File(...)):
#     if career_model is None:
#         raise HTTPException(503, "Model not loaded.")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.StringIO(content.decode('utf-8')))
#         df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        
#         if 'field_choice' not in df.columns:
#             raise HTTPException(400, "Missing 'field_choice' column.")
            
#         processed = preprocess(df)
#         cat_idx = [i for i, f in enumerate(processed.columns) if f in CATEGORICAL_FEATURES]
#         pool = Pool(data=processed, cat_features=cat_idx)
        
#         all_probs = career_model.predict_proba(pool)
#         results = []
        
#         for i, probs in enumerate(all_probs):
#             top_predictions = get_top_predictions(probs)
#             results.append(PredictionRecord(
#                 index=i,
#                 recommended_career=top_predictions[0].career,
#                 confidence=top_predictions[0].probability,
#                 top_10_careers=top_predictions
#             ))
            
#         return results
        
#     except Exception as e:
#         logger.error(f"Batch prediction error: {e}")
#         raise HTTPException(500, "Batch prediction failed.")
#
# # ——— Run server —
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import io
import logging
from datetime import datetime

# --- Logging Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- FastAPI Initialization ---
app = FastAPI(
    title="Career Path Recommendation API",
    description="Recommend career paths based on student profiles and scores",
    version="1.0.0",
)

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables ---
career_model: Optional[CatBoostClassifier] = None
expected_features: List[str] = []


career_labels = [
    "Software Engineer", "Data Scientist", "Civil Engineer", "Mechanical Engineer",
    "Electrical Engineer", "Chemical Engineer", "Biomedical Engineer", "Environmental Scientist",
    "Architect", "Graphic Designer", "Web Developer", "Network Administrator",
    "Database Administrator", "Cybersecurity Analyst", "AI Researcher", "Product Manager",
    "Business Analyst", "Marketing Specialist", "Sales Representative", "Financial Analyst",
    "Accountant", "Actuary", "Economist", "Statistician", "Operations Manager",
    "Human Resources Manager", "Public Relations Specialist", "Content Writer", "Editor",
    "Journalist", "Photographer", "Videographer", "Social Media Manager", "UX/UI Designer",
    "Game Developer", "Mobile App Developer", "Cloud Solutions Architect", "DevOps Engineer",
    "Quality Assurance Tester", "Research Scientist", "Lab Technician", "Pharmacist",
    "Nurse", "Physician", "Dentist", "Veterinarian", "Physical Therapist",
    "Occupational Therapist", "Speech-Language Pathologist", "Psychologist",
    "Social Worker", "Counselor", "Teacher", "Professor", "Education Administrator",
    "Librarian", "Historian", "Anthropologist", "Sociologist", "Political Scientist",
    "International Relations Specialist", "Lawyer", "Paralegal", "Judge",
    "Mediator", "Forensic Scientist", "Crime Analyst", "Firefighter", "Police Officer",
    "Emergency Medical Technician", "Military Officer", "Logistics Coordinator",
    "Supply Chain Manager", "Real Estate Agent", "Urban Planner", "Insurance Underwriter",
    "Investment Banker", "Stockbroker", "Retail Manager", "Hospitality Manager", "Chef",
    "Event Planner", "Tour Guide", "Travel Consultant", "Fashion Designer",
    "Textile Engineer", "Interior Designer", "Jewelry Designer", "Cosmetologist",
    "Fitness Trainer", "Sports Coach", "Athletic Director", "Professional Athlete",
    "Music Producer", "Sound Engineer", "Composer", "Actor", "Dancer",
    "Theater Director", "Film Director", "Screenwriter", "Animator", "Illustrator",
    "Art Director", "Curator", "Museum Educator", "Art Restorer",
    "Cultural Anthropologist", "Philosopher", "Theologian", "Ethicist", "Linguist",
    "Translator", "Interpreter", "Foreign Service Officer", "Diplomat",
    "Nonprofit Administrator", "Grant Writer", "Fundraising Manager",
    "Community Organizer", "Environmental Policy Analyst", "Agricultural Scientist",
    "Food Scientist", "Nutritional Consultant", "Veterinary Technician",
    "Wildlife Biologist", "Marine Biologist", "Geologist", "Meteorologist",
    "Astrophysicist", "Oceanographer", "Climate Scientist", "Renewable Energy Consultant",
    "Mining Engineer", "Petroleum Engineer", "Urban Ecologist", "Fashion Merchandiser",
    "Retail Buyer", "E-commerce Specialist", "Digital Marketing Analyst", "SEO Specialist",
    "Brand Strategist", "Market Research Analyst", "Customer Service Manager",
    "Corporate Trainer", "Business Consultant", "Change Management Specialist",
    "Risk Manager", "Compliance Officer", "Healthcare Administrator",
    "Medical Records Technician", "Health Educator", "Public Health Advisor",
    "Epidemiologist", "Health Policy Analyst", "Sports Psychologist",
    "Rehabilitation Counselor", "Clinical Research Coordinator", "Pharmaceutical Sales Representative",
    "Toxicologist", "Microbiologist", "Geneticist", "Biochemist", "Biophysicist",
    "Biotechnologist", "Clinical Lab Scientist", "Health Information Manager",
    "Digital Content Creator", "Podcast Producer", "Online Course Developer"
]


# Feature specifications
CATEGORICAL_FEATURES = [
    "Gender", "Health Issue", "Career Interest", "Father's Education",
    "Mother's Education", "Parental Involvement", "Home Internet Access",
    "Electricity Access", "School Type", "School Location", "Field Choice",
]
NUMERIC_SUBJECTS = [
    "Grade 12 - Civics and Ethical Education Test Score",
    "Grade 12 - Affan Oromoo Test Score",
    "Grade 12 - English Test Score",
    "Grade 12 - HPE Test Score",
    "Grade 12 - ICT Test Score",
]

# --- Pydantic Schemas ---
class CareerPrediction(BaseModel):
    career: str = Field(..., description="Predicted career name")
    probability: float = Field(..., description="Prediction probability (0-1)")

class CareerResponse(BaseModel):
    recommended_career: str = Field(..., description="Top recommended career")
    confidence: float = Field(..., description="Confidence score of top career")
    top_10_careers: List[CareerPrediction] = Field(..., description="Top 10 career predictions with probabilities")

class CareerRequest(BaseModel):
    gender: str = Field(..., example="Male")
    date_of_birth: Optional[str] = Field(None, example="2005-04-15", description="YYYY-MM-DD")
    age: Optional[int] = Field(None, ge=10, le=100, description="Age in years if date_of_birth not provided")
    health_issue: str = Field(..., example="None")
    career_interest: Optional[str] = Field(None)
    fathers_education: Optional[str] = Field(None)
    mothers_education: Optional[str] = Field(None)
    parental_involvement: str = Field(...)
    home_internet_access: str = Field(...)
    electricity_access: str = Field(...)
    school_type: str = Field(...)
    school_location: str = Field(...)
    field_choice: str = Field(...)
    scores: Dict[str, float] = Field(..., description="Mapping of subject to test score")

class PredictionRecord(BaseModel):
    index: int
    recommended_career: str
    confidence: float
    top_10_careers: List[CareerPrediction]

# --- Utility Functions ---
def load_model(path: str = "career_recommender.cbm") -> None:
    global career_model, expected_features
    try:
        model = CatBoostClassifier()
        model.load_model(path)
        career_model = model
        expected_features = model.feature_names_
        logger.info(f"Loaded model with features: {expected_features}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading error: {e}")

@app.on_event("startup")
async def startup_event():
    load_model()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Convert DOB to age
    if 'date_of_birth' in df.columns and df['date_of_birth'].notna().any():
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        df['age'] = datetime.now().year - df['date_of_birth'].dt.year
        df = df.drop(columns=['date_of_birth'])
    if 'age' not in df.columns or df['age'].isnull().all():
        raise ValueError("Provide a valid 'age' or 'date_of_birth'.")

    # Fill categorical and numeric features
    for col in CATEGORICAL_FEATURES:
        key = col.lower().replace(' ', '_')
        df[col] = df.get(key, 'Unknown').fillna('Unknown').astype(str)

    for subj in NUMERIC_SUBJECTS:
        key = subj.lower().replace(' ', '_')
        df[subj] = pd.to_numeric(df.get(key, np.nan), errors='coerce').fillna(0.0)

    # Align to expected features
    aligned = {}
    for feat in expected_features:
        if feat in CATEGORICAL_FEATURES:
            aligned[feat] = df[feat]
        else:
            aligned[feat] = df.get(feat, 0.0)
    return pd.DataFrame(aligned)


def get_top_predictions(probs: np.ndarray, n: int = 10) -> List[CareerPrediction]:
    idx_sorted = np.argsort(-probs)
    return [CareerPrediction(career=career_labels[i], probability=round(float(probs[i]), 4)) for i in idx_sorted[:n]]

# --- Exception Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# --- Endpoints ---
@app.get("/health", summary="Health check", tags=["System"])
async def health():
    return {"status": "ok", "model_loaded": career_model is not None}

@app.get("/required-features", summary="List required features", tags=["System"])
async def required_features():
    return {"expected_features": expected_features}

@app.post(
    "/predict_career_single",
    response_model=CareerResponse,
    summary="Predict a single career path",
    tags=["Prediction"],
)
async def predict_career_single(req: CareerRequest):
    if career_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    data = req.dict(exclude_none=True)
    merged = {**{k: v for k, v in data.items() if k != 'scores'}, **data.get('scores', {})}
    df = pd.DataFrame([merged])
    processed = preprocess(df)
    cat_idx = [i for i, f in enumerate(processed.columns) if f in CATEGORICAL_FEATURES]
    pool = Pool(data=processed, cat_features=cat_idx)
    probs = career_model.predict_proba(pool)[0]
    top10 = get_top_predictions(probs)
    return CareerResponse(
        recommended_career=top10[0].career,
        confidence=top10[0].probability,
        top_10_careers=top10
    )

@app.post(
    "/predict_career_csv",
    response_model=List[PredictionRecord],
    summary="Batch career prediction via CSV",
    tags=["Prediction"],
)
async def predict_career_csv(file: UploadFile = File(...)):
    if career_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    if 'field_choice' not in df.columns:
        raise HTTPException(status_code=400, detail="Missing 'field_choice' column.")
    processed = preprocess(df)
    cat_idx = [i for i, f in enumerate(processed.columns) if f in CATEGORICAL_FEATURES]
    pool = Pool(data=processed, cat_features=cat_idx)
    all_probs = career_model.predict_proba(pool)
    records = []
    for idx, probs in enumerate(all_probs):
        top10 = get_top_predictions(probs)
        records.append(PredictionRecord(
            index=idx,
            recommended_career=top10[0].career,
            confidence=top10[0].probability,
            top_10_careers=top10
        ))
    return records

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

