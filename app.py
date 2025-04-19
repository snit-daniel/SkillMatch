from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request


from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel, EmailStr, field_validator


import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from typing import List, Optional
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



# Get the absolute path to the SkillMatch directory
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
templates_dir = os.path.join(current_dir, "templates")

print(f"Static directory: {static_dir}")
print(f"Templates directory: {templates_dir}")

# Configure static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configure templates
templates = Jinja2Templates(directory=templates_dir)



# Configure CORS based on environment
allowed_origins = [
    "http://localhost:3000",
    "https://your-production-frontend.com"
] if os.getenv("ENVIRONMENT") == "production" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # This should allow POST
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)

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
    cv_url: str
    additional_skills: List[str]

class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str
    cv_url: str
    additional_skills: List[str]

class Job(BaseModel):
    id: UUID
    title: str
    company: str
    location: str
    description: str
    skills_required: List[str]
    experience_level: ExperienceLevel
    employment_type: EmploymentType
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

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            audience="skillmatch-web",
            issuer="skillmatch-api"
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError as e:
        raise credentials_exception
    
    user = users.find_one({"_id": uuid_to_bin(UUID(user_id))})
    if not user:
        raise credentials_exception
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
    return templates.TemplateResponse("home.html", {"request": request})


# # API Endpoints
# @app.get(
#     "/",
#     include_in_schema=True,
#     summary="API Status",
#     description="Provides API health status and metadata",
#     tags=["Service Status"]
# )

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



@app.post("/auth/login", response_class=HTMLResponse)
async def handle_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    user = users.find_one({"email": username})
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return templates.TemplateResponse(
            "log_in.html",
            {
                "request": request,
                "error": {
                    "message": "The email or password you entered is incorrect",
                    "type": "auth",
                    "style": "error"
                }
            },
            status_code=401
        )
    
    access_token = create_access_token({"sub": str(bin_to_uuid(user["_id"]))})
    
    # Change this to redirect to profile instead of dashboard
    response = RedirectResponse(url="/profile", status_code=303)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=3600
    )
    return response

# User profile endpoints
@app.get("/users/me", response_model=UserResponse)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        id=bin_to_uuid(current_user["_id"]),
        email=current_user["email"],
        full_name=current_user["full_name"],
        cv_url=current_user["cv_url"],
        additional_skills=current_user["additional_skills"]
    )

@app.post("/users/me/profile", response_model=UserResponse)
async def update_user_profile(
    profile: UserProfile,
    current_user: dict = Depends(get_current_user)
):
    users.update_one(
        {"_id": current_user["_id"]},
        {"$set": {
            "full_name": profile.full_name,
            "cv_url": profile.cv_url,
            "additional_skills": profile.additional_skills
        }}
    )
    updated_user = users.find_one({"_id": current_user["_id"]})
    return UserResponse(
        id=bin_to_uuid(updated_user["_id"]),
        email=updated_user["email"],
        full_name=updated_user["full_name"],
        cv_url=updated_user["cv_url"],
        additional_skills=updated_user["additional_skills"]
    )

# Job matching endpoint
@app.get("/jobs/recommendations", response_model=List[MatchResult])
async def get_job_recommendations(current_user: dict = Depends(get_current_user)):
    if not current_user["additional_skills"]:
        raise HTTPException(
            status_code=400,
            detail="Please complete your profile with skills to get recommendations"
        )
    
    user_skills = set(current_user["additional_skills"])
    recommendations = []
    
    for job in jobs.find():
        required_skills = set(job.get("skills_required", []))
        common_skills = user_skills & required_skills
        match_score = len(common_skills) / len(required_skills) if required_skills else 0
        
        if match_score > 0.3:  # Only show jobs with at least 30% match
            missing_skills = list(required_skills - user_skills)
            
            # Find courses that cover missing skills
            recommended_courses = []
            if missing_skills:
                for course in courses.find({"skills_covered": {"$in": missing_skills}}).limit(2):
                    recommended_courses.append({
                        "title": course["title"],
                        "provider": course["provider"],
                        "url": course["url"],
                        "covered_skills": list(set(course["skills_covered"]) & set(missing_skills))
                    })
            
            recommendations.append({
                "job": Job(
                    id=bin_to_uuid(job["_id"]),
                    **{k: v for k, v in job.items() if k != "_id"}
                ),
                "match_score": round(match_score, 2),
                "missing_skills": missing_skills,
                "recommended_courses": recommended_courses
            })
    
    return sorted(recommendations, key=lambda x: x["match_score"], reverse=True)

# Public endpoints
@app.get("/jobs", response_model=List[Job])
async def get_all_jobs():
    return [
        Job(
            id=bin_to_uuid(job["_id"]),
            **{k: v for k, v in job.items() if k != "_id"}
        )
        for job in jobs.find()
    ]

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



# Add these new endpoints to serve your frontend pages



@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request):
    return templates.TemplateResponse("profile_building.html", {"request": request})

@app.get("/job_search", response_class=HTMLResponse)
async def job_search(request: Request):
    return templates.TemplateResponse("job_search.html", {"request": request})

@app.get("/interview_prep", response_class=HTMLResponse)
async def interview_prep(request: Request):
    return templates.TemplateResponse("interview_prep.html", {"request": request})


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