from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi import UploadFile, File
from pathlib import Path

from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, Query
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel, EmailStr, field_validator
from fastapi import Response

from fastapi import BackgroundTasks

import tempfile
import PyPDF2
import docx2txt
import re
from collections import Counter
import httpx
import json  # Add this with your other imports

import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from typing import List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum
from bson.binary import Binary
import pymongo
from jose import JWTError, jwt
from passlib.context import CryptContext
import re
import os
import certifi
from fastapi.responses import JSONResponse





app = FastAPI(
    title="SkillMatch API",
    version="1.0",
    description="API for job seekers to find matching jobs and courses",
    docs_url="/docs",
    redoc_url=None  # Disable redoc for production
)


# Absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
templates_dir = os.path.join(current_dir, "templates")

# Mount the static directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=templates_dir)



# Configure upload directory (add this with your other config)
UPLOAD_DIR = "uploads/cvs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print(app.routes)


# Constants
ALLOWED_EXTENSIONS = ['.pdf', '.doc', '.docx']
MIN_MATCH_SCORE = 0.1  # Show jobs with at least 20% match
MAX_RECOMMENDATIONS = 50  # Limit number of recommendations

# Common English stop words to filter out
STOP_WORDS = {
    'the', 'and', 'for', 'with', 'this', 'that', 'have', 'has', 'had', 
    'from', 'your', 'you', 'are', 'were', 'been', 'they', 'their', 'will',
    # ... keep the rest of your stop words list ...
}

# Configure templates
templates = Jinja2Templates(directory=templates_dir)



# Configure CORS based on environment
allowed_origins = [
    "http://localhost:3000",
    "https://your-production-frontend.com"
] if os.getenv("ENVIRONMENT") == "production" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Add this line
)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"


# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY must be set in environment variables")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Database Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI must be set in environment variables")

try:
    client = MongoClient(
        MONGODB_URI,
        tls=True,
        tlsCAFile=certifi.where(),
        connectTimeoutMS=10000,
        socketTimeoutMS=30000
    )
    db = client.get_database()  # Gets database from connection string
    print("Connection successful to:", MONGODB_URI)  # Verify which URI is used
except Exception as err:
    raise RuntimeError(f"MongoDB connection failed: {err}")



# Collections
users = db.users
jobs = db.jobs
courses = db.courses



from pymongo import MongoClient
import certifi
from urllib.parse import quote_plus

def get_mongo_client():
    try:
        # Escape special characters in password
        password = "Test1234"
        escaped_password = quote_plus(password)
        
        uri = f"mongodb+srv://snitdan17:{escaped_password}@cluster0.uhl3bge.mongodb.net/?retryWrites=true&w=majority"
        
        client = MongoClient(uri,
                           tls=True,
                           tlsCAFile=certifi.where(),
                           connectTimeoutMS=30000,
                           socketTimeoutMS=30000)
        
        # Verify connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas!")
        return client
        
    except Exception as e:
        print("Failed to connect to MongoDB Atlas:", str(e))
        raise

# Initialize connection
try:
    client = get_mongo_client()
    db = client["skillmatch_db"]
    jobs = db["jobs"]
    
    # Test collection access
    print("Collections:", db.list_collection_names())
    print("Jobs count:", jobs.count_documents({}))
except Exception as e:
    print("Database initialization failed:", str(e))
    raise

# Security Utilities
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
    pbkdf2_sha256__rounds=29000
)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login",
    scheme_name="JWT"
)

# Helper functions
def uuid_to_bin(uuid):
    return Binary.from_uuid(uuid)

def bin_to_uuid(binary):
    return UUID(bytes=binary)

def validate_password(password: str):
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain an uppercase letter")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain a lowercase letter")
    if not re.search(r"\d", password):
        raise ValueError("Password must contain a digit")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise ValueError("Password must contain a special character")
    return password



def extract_and_match_skills(cv_text, threshold=3):
    """
    Extract meaningful words from CV and match against skills from all jobs
    threshold = minimum length of word to consider
    """
    # Get all unique skills from all posted jobs
    all_job_skills = set()
    for job in jobs.find({}, {"skills_required": 1}):
        all_job_skills.update(skill.lower() for skill in job.get("skills_required", []))
    
    # Extract potential skills from CV text
    words = re.findall(r'\b[a-zA-Z]{'+str(threshold)+r',}\b', cv_text.lower())
    word_counts = Counter(words)
    
    # Match against job skills and get top matches
    matched_skills = [
        skill.title() for skill in all_job_skills 
        if skill in word_counts
    ]
    
    # Also include frequently mentioned words that might be skills
    frequent_words = [
        word.title() for word, count in word_counts.most_common(20) 
        if len(word) >= threshold and word not in STOP_WORDS
    ]
    
    return list(set(matched_skills + frequent_words))

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class ExperienceLevel(str, Enum):
    ENTRY = "Entry Level"
    MID = "Mid Level"
    SENIOR = "Senior Level"

class EmploymentType(str, Enum):
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"
    REMOTE = "Remote"
    HYBRID = "Hybrid"

class UserSignup(BaseModel):
    email: EmailStr
    password: str

    @field_validator('password')  # Changed from @validator
    def validate_password(cls, v):
        return validate_password(v)

    

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserProfile(BaseModel):
    full_name: str
    additional_skills: List[str]


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str
    additional_skills: List[str]
    cv_extracted_skills: List[str] = []  # Add this new field


class Job(BaseModel):
    id: str
    title: str
    company: str
    location: str
    description: str
    skills_required: List[str] = []
    experience_level: str
    employment_type: str
    apply_url: str
    posted_at: datetime


class MatchResult(BaseModel):
    job: Job
    match_score: float
    missing_skills: List[str]
    recommended_courses: List[dict]

# Auth utilities
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({
        "exp": expire,
        "iss": "skillmatch-api",
        "aud": "skillmatch-web",
        "iat": datetime.utcnow()
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

from fastapi import Request
from fastapi.responses import RedirectResponse
from urllib.parse import quote_plus

async def get_current_user(request: Request):
    # Try to get token from cookie first
    token = request.cookies.get("access_token")
    
    # If not in cookie, try Authorization header
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove 'Bearer ' prefix
    
    if not token:
        # Redirect to login with the current path as next parameter
        current_path = quote_plus(str(request.url.path))
        return RedirectResponse(url=f"/login?next={current_path}")
    
    try:
        # Verify token (remove 'Bearer ' if present)
        if token.startswith("Bearer "):
            token = token[7:]
            
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            audience="skillmatch-web",
            issuer="skillmatch-api"
        )
        user_id = payload.get("sub")
        if not user_id:
            raise JWTError("Invalid token")
            
    except JWTError as e:
        current_path = quote_plus(str(request.url.path))
        return RedirectResponse(url=f"/login?next={current_path}")
    
    user = users.find_one({"_id": uuid_to_bin(UUID(user_id))})
    if not user:
        current_path = quote_plus(str(request.url.path))
        return RedirectResponse(url=f"/login?next={current_path}")
    
    return user


# Initialize with sample data
def init_data():
    # Only initialize in development
    if os.getenv("ENVIRONMENT") == "production":
        print("Skipping data initialization in production")
        return
        
    # Clear existing data
    print("Initializing sample data...")
    users.delete_many({})
    jobs.delete_many({})
    courses.delete_many({})

    
    # Sample users
    users.insert_many([
        {
            "_id": uuid_to_bin(uuid4()),
            "email": "user1@example.com",
            "hashed_password": pwd_context.hash("Password123!"),
            "full_name": "John Doe",
            "cv_url": "https://example.com/cvs/john_doe.pdf",
            "additional_skills": ["Python", "FastAPI", "REST APIs", "Docker"]
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "email": "user2@example.com",
            "hashed_password": pwd_context.hash("Password123!"),
            "full_name": "Jane Smith",
            "cv_url": "https://example.com/cvs/jane_smith.pdf",
            "additional_skills": ["Machine Learning", "Python", "Pandas", "SQL"]
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "email": "user3@example.com",
            "hashed_password": pwd_context.hash("Password123!"),
            "full_name": "Bob Johnson",
            "cv_url": "https://example.com/cvs/bob_johnson.pdf",
            "additional_skills": ["JavaScript", "React", "Node.js", "HTML/CSS"]
        }
    ])
    
    # Sample jobs (10 jobs)
    jobs.insert_many([
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Python Developer",
            "active": True,
            "company": "TechCorp",
            "location": "Remote",
            "description": "Looking for Python developers with FastAPI experience",
            "skills_required": ["Python", "FastAPI", "REST APIs"],
            "experience_level": "Mid Level",
            "employment_type": "Full-time",
            "apply_url": "https://example.com/jobs/123",
            "posted_at": datetime.now()
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Data Scientist",
            "company": "DataWorks",
            "location": "New York",
            "description": "Seeking data scientists with machine learning skills",
            "skills_required": ["Python", "Machine Learning", "Pandas", "SQL"],
            "experience_level": "Senior Level",
            "employment_type": "Full-time",
            "apply_url": "https://example.com/jobs/456",
            "posted_at": datetime.now()
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Frontend Developer",
            "company": "WebSolutions",
            "location": "San Francisco",
            "description": "Frontend developer with React experience needed",
            "skills_required": ["JavaScript", "React", "HTML/CSS"],
            "experience_level": "Mid Level",
            "employment_type": "Full-time",
            "apply_url": "https://example.com/jobs/789",
            "posted_at": datetime.now()
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "DevOps Engineer",
            "company": "CloudTech",
            "location": "Remote",
            "description": "DevOps engineer with Kubernetes and AWS experience",
            "skills_required": ["Docker", "Kubernetes", "AWS", "CI/CD"],
            "experience_level": "Senior Level",
            "employment_type": "Full-time",
            "apply_url": "https://example.com/jobs/101",
            "posted_at": datetime.now()
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Backend Developer",
            "company": "API Masters",
            "location": "Boston",
            "description": "Backend developer with Node.js experience",
            "skills_required": ["JavaScript", "Node.js", "Express", "MongoDB"],
            "experience_level": "Entry Level",
            "employment_type": "Full-time",
            "apply_url": "https://example.com/jobs/112",
            "posted_at": datetime.now()
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Full Stack Developer",
            "company": "Digital Creations",
            "location": "Chicago",
            "description": "Full stack developer with React and Node.js experience",
            "skills_required": ["JavaScript", "React", "Node.js", "Express"],
            "experience_level": "Mid Level",
            "employment_type": "Full-time",
            "apply_url": "https://example.com/jobs/131",
            "posted_at": datetime.now()
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Data Engineer",
            "company": "Big Data Inc",
            "location": "Austin",
            "description": "Data engineer with ETL pipeline experience",
            "skills_required": ["Python", "SQL", "ETL", "Data Warehousing"],
            "experience_level": "Mid Level",
            "employment_type": "Full-time",
            "apply_url": "https://example.com/jobs/415",
            "posted_at": datetime.now()
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "UX Designer",
            "company": "Design Hub",
            "location": "Remote",
            "description": "UX designer with Figma and prototyping experience",
            "skills_required": ["Figma", "UI/UX Design", "Prototyping", "User Research"],
            "experience_level": "Mid Level",
            "employment_type": "Part-time",
            "apply_url": "https://example.com/jobs/161",
            "posted_at": datetime.now()
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Mobile Developer",
            "company": "App World",
            "location": "Seattle",
            "description": "Mobile developer with React Native experience",
            "skills_required": ["JavaScript", "React Native", "Mobile Development"],
            "experience_level": "Entry Level",
            "employment_type": "Full-time",
            "apply_url": "https://example.com/jobs/718",
            "posted_at": datetime.now()
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "QA Engineer",
            "company": "Quality Assurance Co",
            "location": "Remote",
            "description": "QA engineer with automated testing experience",
            "skills_required": ["Testing", "Automation", "Selenium", "JIRA"],
            "experience_level": "Mid Level",
            "employment_type": "Full-time",
            "apply_url": "https://example.com/jobs/192",
            "posted_at": datetime.now()
        }
    ])
    
    # Sample courses (10 courses)
    courses.insert_many([
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Advanced Python Programming",
            "provider": "Udemy",
            "skills_covered": ["Python", "OOP", "Decorators"],
            "url": "https://example.com/courses/python-advanced",
            "duration": "8 weeks"
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Machine Learning Fundamentals",
            "provider": "Coursera",
            "skills_covered": ["Machine Learning", "Python", "Scikit-learn"],
            "url": "https://example.com/courses/ml-fundamentals",
            "duration": "10 weeks"
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "React for Beginners",
            "provider": "Udemy",
            "skills_covered": ["JavaScript", "React", "Frontend Development"],
            "url": "https://example.com/courses/react-beginners",
            "duration": "6 weeks"
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Docker and Kubernetes Mastery",
            "provider": "Pluralsight",
            "skills_covered": ["Docker", "Kubernetes", "Containerization"],
            "url": "https://example.com/courses/docker-k8s",
            "duration": "5 weeks"
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Node.js: The Complete Guide",
            "provider": "Udemy",
            "skills_covered": ["JavaScript", "Node.js", "Express"],
            "url": "https://example.com/courses/node-complete",
            "duration": "12 weeks"
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Data Science with Python",
            "provider": "DataCamp",
            "skills_covered": ["Python", "Pandas", "NumPy", "Data Analysis"],
            "url": "https://example.com/courses/data-science-python",
            "duration": "8 weeks"
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "AWS Certified Solutions Architect",
            "provider": "A Cloud Guru",
            "skills_covered": ["AWS", "Cloud Computing", "Infrastructure"],
            "url": "https://example.com/courses/aws-solutions-architect",
            "duration": "10 weeks"
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "UX Design Principles",
            "provider": "Interaction Design Foundation",
            "skills_covered": ["UI/UX Design", "Figma", "User Research"],
            "url": "https://example.com/courses/ux-principles",
            "duration": "6 weeks"
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Mobile App Development with React Native",
            "provider": "Udacity",
            "skills_covered": ["JavaScript", "React Native", "Mobile Development"],
            "url": "https://example.com/courses/react-native",
            "duration": "12 weeks"
        },
        {
            "_id": uuid_to_bin(uuid4()),
            "title": "Automated Testing with Selenium",
            "provider": "Test Automation University",
            "skills_covered": ["Testing", "Automation", "Selenium"],
            "url": "https://example.com/courses/selenium",
            "duration": "4 weeks"
        }
    ])



# Initialize data only in development
if os.getenv("ENVIRONMENT") != "production":
    init_data()

@app.get("/")
async def home(request: Request):
    print("Requesting home page")
    return templates.TemplateResponse("home.html", {"request": request})




@app.get("/debug-paths")
async def debug_paths():
    return {
        "working_directory": os.getcwd(),
        "static_files_exist": os.path.exists("static/css/style.css"),
        "templates_exist": os.path.exists("templates/base.html")
    }
async def root() -> JSONResponse:
    """Root endpoint for health checks and API discovery"""
    # Base response structure
    response_data = {
        "service": "SkillMatch API",
        "version": "1.0",
        "status": {
            "api": "operational",
            "database": "available" if db.command('ping').get('ok') == 1 else "unavailable"
        },
        "docs": "/docs",
        "contact": "support@skillmatch.com"
    }

    # Add debug information in non-production environments
    if os.getenv("ENVIRONMENT", "development") != "production":
        response_data.update({
            "environment": os.getenv("ENVIRONMENT"),
            "endpoints": {
                "auth": ["/signup", "/auth/login"],
                "user": ["/users/me"],
                "jobs": ["/jobs", "/jobs/recommendations"],
                "courses": ["/courses"]
            }
        })



    return JSONResponse(
        content=response_data,
        headers={
            "X-API-Version": "1.0",
            "Cache-Control": "no-cache" if os.getenv("ENVIRONMENT") != "production" else "no-store"
        }
    )



jobs_collection = db["jobs"]



from fastapi.responses import RedirectResponse
from fastapi import status

@app.get("/signup", response_class=HTMLResponse)
async def show_signup_page(request: Request):
    return templates.TemplateResponse("sign_in.html", {"request": request})



@app.post("/signup", response_class=HTMLResponse)
async def handle_signup(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    try:
        user = UserSignup(email=email, password=password)
        
        if users.find_one({"email": user.email}):
            return templates.TemplateResponse(
                "sign_in.html",
                {
                    "request": request,
                    "error": {
                        "message": "Please fix these password requirements:",
                        "type": "password",
                        "style": "error"
                    }
                },
                status_code=400
            )
                
        user_id = uuid4()
        users.insert_one({
            "_id": uuid_to_bin(user_id),
            "email": user.email,
            "hashed_password": pwd_context.hash(user.password),
            "full_name": "",
            "cv_url": "",
            "additional_skills": []
        })
        
        # Change this to redirect to login instead of profile
        return RedirectResponse(url="/login", status_code=303)
    
    except ValueError as e:
        return templates.TemplateResponse(
            "sign_in.html",
            {"request": request, "error": str(e)},
            status_code=400
        )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("log_in.html", {"request": request})





@app.post("/auth/login")
async def handle_login(
    response: Response,
    username: str = Form(...),
    password: str = Form(...)
):
    user = users.find_one({"email": username})
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    access_token = create_access_token({"sub": str(bin_to_uuid(user["_id"]))})
    
    response = RedirectResponse(url="/profile", status_code=303)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        secure=False,  # Set to True in production (requires HTTPS)
        samesite='lax',
        max_age=3600,  # 1 hour
        path="/"  # Important for all routes
    )
    return response



@app.post("/users/me/profile", response_model=UserResponse)
async def update_user_profile(
    background_tasks: BackgroundTasks,
    full_name: str = Form(...),
    cv_file: UploadFile = File(None),  # Make optional if user might not always upload
    additional_skills: str = Form(""),  # Default empty string
    current_user: dict = Depends(get_current_user)
):
    # Process manual skills
    skills_list = [skill.strip() for skill in additional_skills.split(',') if skill.strip()]
    extracted_skills = []
    cv_update = {}

    if cv_file:
        # Validate file type
        file_ext = Path(cv_file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only PDF and Word documents allowed")
        
        # Process CV in background
        cv_content = await cv_file.read()
        cv_update = {
            "cv_file": Binary(cv_content),
            "cv_filename": cv_file.filename,
            "cv_content_type": cv_file.content_type,
            "cv_last_updated": datetime.utcnow()
        }
        
        # Add background task for CV processing
        background_tasks.add_task(
            process_cv_and_update_skills,
            cv_content=cv_content,
            file_ext=file_ext,
            user_id=current_user["_id"],
            existing_skills=skills_list
        )
    else:
        # If no CV uploaded but has existing CV skills, preserve them
        if current_user.get("cv_extracted_skills"):
            extracted_skills = current_user["cv_extracted_skills"]

    # Combine skills (manual + preserved CV skills if no new CV)
    combined_skills = list(set(skills_list + extracted_skills))
    
    # Update user profile
    update_data = {
        "full_name": full_name,
        "additional_skills": combined_skills,
        **cv_update
    }
    
    users.update_one(
        {"_id": current_user["_id"]},
        {"$set": update_data}
    )
    
    return RedirectResponse("/profile?success=true", status_code=303)

async def process_cv_and_update_skills(
    cv_content: bytes,
    file_ext: str,
    user_id: Any,
    existing_skills: List[str]
):
    """Background task to process CV and update skills"""
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(cv_content)
            temp_file_path = temp_file.name
        
        extracted_text = ""
        if file_ext == '.pdf':
            with open(temp_file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                extracted_text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif file_ext in ['.doc', '.docx']:
            extracted_text = docx2txt.process(temp_file_path)
        
        # Enhanced skill extraction with context awareness
        extracted_skills = extract_and_match_skills(extracted_text)
        
        # Combine with existing skills
        combined_skills = list(set(existing_skills + extracted_skills))
        
        # Update user with extracted skills
        users.update_one(
            {"_id": user_id},
            {"$set": {
                "cv_extracted_skills": extracted_skills,
                "additional_skills": combined_skills
            }}
        )
        
    except Exception as e:
        print(f"CV processing error: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

@app.get("/debug/skill-match")
async def debug_skill_match(current_user: dict = Depends(get_current_user)):
    # Get first 3 jobs and check matching
    jobs_sample = await jobs.find({"active": True}).limit(3).to_list(None)
    
    user_skills = set()
    if current_user.get("additional_skills"):
        user_skills.update(s.lower().strip() for s in current_user["additional_skills"])
    if current_user.get("cv_extracted_skills"):
        user_skills.update(s.lower().strip() for s in current_user["cv_extracted_skills"])
    
    results = []
    for job in jobs_sample:
        required_skills = set(s.lower().strip() for s in job.get("skills_required", []))
        common_skills = user_skills & required_skills
        results.append({
            "job_title": job["title"],
            "required_skills": list(required_skills),
            "user_skills": list(user_skills),
            "common_skills": list(common_skills),
            "match_percentage": len(common_skills)/len(required_skills) if required_skills else 0
        })
    
    return {
        "user_email": current_user["email"],
        "total_user_skills": len(user_skills),
        "matches": results
    }


@app.get("/jobs/recommendations")
async def get_job_recommendations(
    current_user: dict = Depends(get_current_user),
    min_match: float = Query(0.1, ge=0, le=1),
    limit: int = Query(10, le=100)
):
    try:
        # Normalize all skills to lowercase
        user_skills = set()
        if current_user.get("additional_skills"):
            user_skills.update(s.lower().strip() for s in current_user["additional_skills"])
        if current_user.get("cv_extracted_skills"):
            user_skills.update(s.lower().strip() for s in current_user["cv_extracted_skills"])
        
        if not user_skills:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_skills",
                    "message": "Please add skills to your profile",
                    "solutions": [
                        "Upload your CV to auto-detect skills",
                        "Add at least 3 skills manually"
                    ]
                }
            )

        # Get all active jobs
        jobs_list = list(jobs.find({"active": True}))
        
        recommendations = []
        for job in jobs_list:
            # Normalize job skills to lowercase
            required_skills = set(s.lower().strip() for s in job.get("skills_required", []))
            if not required_skills:
                continue
                
            common_skills = user_skills & required_skills
            match_score = len(common_skills) / len(required_skills)
            
            if match_score >= min_match:
                recommendations.append({
                    "job_id": str(job["_id"]),
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                    "match_score": round(match_score, 2),
                    "matching_skills": list(common_skills),
                    "missing_skills": list(required_skills - user_skills)
                })

        recommendations.sort(key=lambda x: x["match_score"], reverse=True)
        return recommendations[:limit]
        
    except Exception as e:
        print(f"Error in recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/debug/jobs")
async def debug_jobs():
    try:
        # Comprehensive projection to exclude all binary and sensitive fields
        projection = {
            "_id": 0,  # We'll manually include it as string
            "cv_file": 0,
            "cv_filename": 0,
            "cv_content_type": 0,
            "cv_last_updated": 0,
            "hashed_password": 0,
            "cv_extracted_skills": 0,
            "additional_skills": 0,
            "cv_url": 0
        }
        
        # Get sample jobs with our projection
        all_jobs = list(jobs.find({}, projection).limit(5))
        active_jobs = list(jobs.find({"active": True}, projection).limit(5))
        
        # Manually include _id as string
        def process_job(job):
            if "_id" in job:
                job["id"] = str(job["_id"])
                del job["_id"]
            return job
        
        return {
            "total_jobs": jobs.count_documents({}),
            "active_jobs_count": jobs.count_documents({"active": True}),
            "sample_jobs": [process_job(job) for job in all_jobs[:2]],
            "active_jobs_sample": [process_job(job) for job in active_jobs[:2]]
        }
    except Exception as e:
        return {"error": str(e)}

# Keep your existing download endpoint
@app.get("/download/cv")
async def download_cv(current_user: dict = Depends(get_current_user)):
    if "cv_file" not in current_user:
        raise HTTPException(status_code=404, detail="CV not found")
    
    return Response(
        content=current_user["cv_file"],
        media_type=current_user.get("cv_content_type", "application/octet-stream"),
        headers={
            "Content-Disposition": f"attachment; filename={current_user['cv_filename']}"
        }
    )


@app.get("/jobs")
async def get_all_jobs(
    request: Request,  # Add request parameter
    search: Optional[str] = None,
    experience: Optional[str] = None,
    employment: Optional[str] = None,
    skills: Optional[str] = None
):
    try:
        query = {}  # Start with empty query
        
        # Add debug logging
        print(f"Received request with params - search: {search}, experience: {experience}, employment: {employment}, skills: {skills}")
        
        # Build query based on parameters
        if search:
            query["$or"] = [
                {"title": {"$regex": search, "$options": "i"}},
                {"company": {"$regex": search, "$options": "i"}},
                {"description": {"$regex": search, "$options": "i"}}
            ]
        
        if experience:
            experience_levels = [x.strip() for x in experience.split(',')]
            query["experience_level"] = {"$in": experience_levels}
        
        if employment:
            employment_types = [x.strip() for x in employment.split(',')]
            query["employment_type"] = {"$in": employment_types}
        
        if skills:
            skills_list = [x.strip() for x in skills.split(',')]
            query["skills_required"] = {"$all": skills_list}
        
        # Get jobs from database
        jobs_list = list(jobs.find(query))
        print(f"Found {len(jobs_list)} jobs matching query")
        
        # Transform jobs data
        transformed_jobs = []
        for job in jobs_list:
            transformed = {
                "id": str(job["_id"]),
                "title": job.get("title", "No Title"),
                "company": job.get("company", "No Company"),
                "location": job.get("location", "Remote"),
                "description": job.get("description", "No description available"),
                "skills_required": job.get("skills_required", []),
                "experience_level": job.get("experience_level", "Not specified"),
                "employment_type": job.get("employment_type", "Not specified"),
                "apply_url": job.get("apply_url", ""),
                "posted_at": job.get("posted_at", datetime.now()).isoformat()
            }
            transformed_jobs.append(transformed)
        
        return transformed_jobs
        
    except Exception as e:
        print(f"Error in /jobs endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading jobs")

@app.get("/courses", response_model=List[dict])
async def get_all_courses():
    return [
        {
            "id": str(bin_to_uuid(course["_id"])),
            "title": course["title"],
            "provider": course["provider"],
            "skills_covered": course["skills_covered"],
            "url": course["url"],
            "duration": course["duration"]
        }
        for course in courses.find()
    ]



@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request):
    user = await get_current_user(request)
    if isinstance(user, RedirectResponse):
        return user
    return templates.TemplateResponse("profile_building.html", {"request": request, "user": user})

@app.get("/job_search", response_class=HTMLResponse)
async def job_search(request: Request):
    user = await get_current_user(request)
    if isinstance(user, RedirectResponse):
        return user
    return templates.TemplateResponse("job_search.html", {"request": request, "user": user})

@app.get("/interview_prep", response_class=HTMLResponse)
async def interview_prep(request: Request):
    user = await get_current_user(request)
    if isinstance(user, RedirectResponse):
        return user
    return templates.TemplateResponse("interview_prep.html", {"request": request, "user": user})



@app.post("/generate_questions")
async def generate_questions(payload: dict):
    job_title = payload.get("job_title", "").strip()
    if not job_title:
        return JSONResponse(status_code=400, content={"error": "Missing job title."})

    prompt = f"""
    Generate 5 concise and relevant interview questions for a {job_title} role.
    Format them as a numbered list without any additional commentary.
    """

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:  # Increased timeout
            response = await client.post(
                DEEPSEEK_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()  # Raises exception for 4XX/5XX responses
            
            content = response.json()
            if "choices" not in content:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Unexpected response format from API"}
                )
                
            raw_questions = content["choices"][0]["message"]["content"]
            # Parse the numbered list into individual questions
            questions = [
                q.strip() 
                for q in raw_questions.split("\n") 
                if q.strip() and q[0].isdigit()
            ]
            
            return {"questions": questions}
            
    except httpx.ReadTimeout:
        return JSONResponse(
            status_code=504,
            content={"error": "The request timed out. Please try again."}
        )
    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=e.response.status_code,
            content={"error": f"API request failed: {str(e)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"}
        )
    



@app.post("/get_feedback")
async def get_feedback(payload: dict):
    question = payload.get("question", "").strip()
    answer = payload.get("answer", "").strip()

    if not question or not answer:
        return JSONResponse(
            status_code=400,
            content={"error": "Both question and answer are required."}
        )

    prompt = f"""
    You are an interview coach analyzing this response:
    
    Question: {question}
    Candidate's Answer: {answer}
    
    Provide detailed feedback in this exact JSON format (no other text):
    {{
      "score": (float between 0 and 10 with 1 decimal place),
      "strengths": ["specific strength 1", "specific strength 2"],
      "improvements": ["specific improvement 1", "specific improvement 2"],
      "suggested_answer": "a well-structured sample answer"
    }}
    
    Important: Return ONLY the JSON object, no additional text or markdown.
    """

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                DEEPSEEK_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"}
                }
            )
            
            response.raise_for_status()
            content = response.json()
            
            # Debug logging
            print("Raw API response:", content)
            
            if "choices" not in content or not content["choices"]:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Unexpected response format from API"}
                )
                
            feedback_str = content["choices"][0]["message"]["content"]
            
            # Clean the response in case it has markdown or other wrappers
            feedback_str = feedback_str.replace("```json", "").replace("```", "").strip()
            
            # Parse the JSON
            feedback = json.loads(feedback_str)
            
            # Validate the response structure
            required_keys = ["score", "strengths", "improvements", "suggested_answer"]
            if not all(key in feedback for key in required_keys):
                return JSONResponse(
                    status_code=500,
                    content={"error": "Incomplete feedback received from API"}
                )
                
            return feedback
            
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Could not parse the feedback response"}
        )
    except httpx.ReadTimeout:
        return JSONResponse(
            status_code=504,
            content={"error": "The feedback analysis took too long. Please try again with a shorter answer."}
        )
    except httpx.HTTPStatusError as e:
        print(f"API request failed: {str(e)}")
        return JSONResponse(
            status_code=e.response.status_code,
            content={"error": f"API request failed: {str(e)}"}
        )
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"}
        )


@app.get("/routes")
async def list_routes():
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": list(route.methods)
        })
    return routes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )