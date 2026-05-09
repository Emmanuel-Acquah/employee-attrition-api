"""
Employee Attrition Prediction API
Team 1 — Data Analytics Internship 2026

FastAPI application that serves the trained ML model as a REST API.
Run with: uvicorn app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import joblib
import os

# ══════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predict employee attrition risk using machine learning. Built by Team 1.",
    version="1.0.0"
)

# ══════════════════════════════════════════════
# LOAD MODEL (or use fallback demo mode)
# ══════════════════════════════════════════════

MODEL_PATH = "model/attrition_model_final.pkl"
SCALER_PATH = "model/attrition_scaler.pkl"
FEATURES_PATH = "model/attrition_features.pkl"

model = None
scaler = None
feature_names = None
DEMO_MODE = False

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("✅ Model loaded from saved files.")
else:
    DEMO_MODE = True
    print("⚠️  Model files not found — running in DEMO MODE.")
    print("   To use your real model, place .pkl files in the model/ folder.")
    print("   The API will return simulated predictions for now.\n")


# ══════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════

class EmployeeInput(BaseModel):
    """Input schema — an employee's HR data."""
    Age: int = Field(..., ge=18, le=65, description="Employee age")
    BusinessTravel: str = Field(..., description="Travel frequency: Non-Travel, Travel_Rarely, Travel_Frequently")
    Department: str = Field(..., description="Department: Sales, Research & Development, Human Resources")
    DistanceFromHome: int = Field(..., ge=0, description="Distance from home in km")
    Education: int = Field(..., ge=1, le=5, description="Education level 1-5")
    EducationField: str = Field(..., description="Field of education")
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4, description="Environment satisfaction 1-4")
    Gender: str = Field(..., description="Male or Female")
    JobInvolvement: int = Field(..., ge=1, le=4, description="Job involvement 1-4")
    JobLevel: int = Field(..., ge=1, le=5, description="Job level 1-5")
    JobRole: str = Field(..., description="Job role title")
    JobSatisfaction: int = Field(..., ge=1, le=4, description="Job satisfaction 1-4")
    MaritalStatus: str = Field(..., description="Single, Married, or Divorced")
    MonthlyIncome: float = Field(..., ge=0, description="Monthly income in USD")
    NumCompaniesWorked: int = Field(..., ge=0, description="Number of companies worked at")
    OverTime: str = Field(..., description="Yes or No")
    PercentSalaryHike: float = Field(..., ge=0, description="Percent salary hike")
    PerformanceRating: int = Field(..., ge=1, le=4, description="Performance rating 1-4")
    RelationshipSatisfaction: int = Field(..., ge=1, le=4, description="Relationship satisfaction 1-4")
    StockOptionLevel: int = Field(..., ge=0, le=3, description="Stock option level 0-3")
    TotalWorkingYears: int = Field(..., ge=0, description="Total working years")
    TrainingTimesLastYear: int = Field(..., ge=0, description="Training times last year")
    WorkLifeBalance: int = Field(..., ge=1, le=4, description="Work-life balance 1-4")
    YearsAtCompany: int = Field(..., ge=0, description="Years at company")
    YearsInCurrentRole: int = Field(..., ge=0, description="Years in current role")
    YearsSinceLastPromotion: int = Field(..., ge=0, description="Years since last promotion")
    YearsWithCurrManager: int = Field(..., ge=0, description="Years with current manager")
    DailyRate: float = Field(default=800, ge=0, description="Daily rate")
    HourlyRate: float = Field(default=65, ge=0, description="Hourly rate")
    MonthlyRate: float = Field(default=14000, ge=0, description="Monthly rate")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 28,
                "BusinessTravel": "Travel_Frequently",
                "Department": "Sales",
                "DistanceFromHome": 15,
                "Education": 3,
                "EducationField": "Marketing",
                "EnvironmentSatisfaction": 2,
                "Gender": "Male",
                "JobInvolvement": 3,
                "JobLevel": 1,
                "JobRole": "Sales Representative",
                "JobSatisfaction": 1,
                "MaritalStatus": "Single",
                "MonthlyIncome": 2500,
                "NumCompaniesWorked": 3,
                "OverTime": "Yes",
                "PercentSalaryHike": 11,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 2,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 5,
                "TrainingTimesLastYear": 2,
                "WorkLifeBalance": 2,
                "YearsAtCompany": 1,
                "YearsInCurrentRole": 0,
                "YearsSinceLastPromotion": 0,
                "YearsWithCurrManager": 0,
                "DailyRate": 800,
                "HourlyRate": 65,
                "MonthlyRate": 14000
            }
        }


class PredictionResponse(BaseModel):
    """Output schema — the prediction result."""
    attrition_risk_percent: float
    risk_level: str
    risk_color: str
    prediction: str
    confidence: float
    top_risk_factors: list
    recommendation: str
    demo_mode: bool = False


# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════

def classify_risk(probability: float) -> tuple:
    """Assign risk level, color, and recommendation."""
    if probability >= 0.75:
        return "Critical", "#e74c3c", "URGENT: Immediate retention intervention needed — schedule 1-on-1, review compensation, and address overtime."
    elif probability >= 0.50:
        return "High", "#f39c12", "HIGH PRIORITY: Review workload and satisfaction. Consider salary adjustment and career development plan."
    elif probability >= 0.30:
        return "Medium", "#3498db", "MONITOR: Keep an eye on this employee. Schedule regular check-ins and ensure work-life balance."
    else:
        return "Low", "#2ecc71", "HEALTHY: Employee appears stable. Continue standard engagement and development practices."


def identify_risk_factors(emp: EmployeeInput) -> list:
    """Identify the top risk factors for this employee."""
    factors = []
    if emp.OverTime == "Yes":
        factors.append({"factor": "Works Overtime", "severity": "high",
                        "detail": "Overtime workers have 3x higher attrition (30.5% vs 10.4%)"})
    if emp.MonthlyIncome < 3000:
        factors.append({"factor": "Low Income (<$3,000)", "severity": "high",
                        "detail": f"Monthly income ${emp.MonthlyIncome:,.0f} is below the $3,000 risk threshold"})
    if emp.JobSatisfaction <= 2:
        factors.append({"factor": "Low Job Satisfaction", "severity": "high",
                        "detail": f"Satisfaction score {emp.JobSatisfaction}/4 — low satisfaction doubles attrition risk"})
    if emp.YearsAtCompany <= 2:
        factors.append({"factor": "New Employee (<2 years)", "severity": "medium",
                        "detail": f"Only {emp.YearsAtCompany} year(s) at company — 30% of new hires leave within 2 years"})
    if emp.Age < 30:
        factors.append({"factor": "Young Worker (<30)", "severity": "medium",
                        "detail": f"Age {emp.Age} — employees 18-25 have 35.8% attrition rate"})
    if emp.WorkLifeBalance == 1:
        factors.append({"factor": "Poor Work-Life Balance", "severity": "medium",
                        "detail": "WLB score 1/4 — poor balance leads to 31.3% attrition"})
    if emp.EnvironmentSatisfaction <= 2:
        factors.append({"factor": "Low Environment Satisfaction", "severity": "low",
                        "detail": f"Environment satisfaction {emp.EnvironmentSatisfaction}/4"})
    if emp.DistanceFromHome > 20:
        factors.append({"factor": "Long Commute", "severity": "low",
                        "detail": f"Distance from home: {emp.DistanceFromHome} km"})
    if emp.NumCompaniesWorked > 5:
        factors.append({"factor": "Job Hopper History", "severity": "low",
                        "detail": f"Worked at {emp.NumCompaniesWorked} companies — may indicate flight risk"})
    if emp.YearsSinceLastPromotion > 5:
        factors.append({"factor": "Promotion Stagnation", "severity": "medium",
                        "detail": f"{emp.YearsSinceLastPromotion} years since last promotion"})
    if emp.StockOptionLevel == 0:
        factors.append({"factor": "No Stock Options", "severity": "low",
                        "detail": "No equity stake reduces retention incentive"})

    return factors[:5]  # Return top 5


def demo_predict(emp: EmployeeInput) -> float:
    """Generate a realistic demo prediction when model files aren't available."""
    score = 0.10  # Base risk

    if emp.OverTime == "Yes": score += 0.20
    if emp.MonthlyIncome < 3000: score += 0.18
    if emp.JobSatisfaction <= 2: score += 0.12
    if emp.YearsAtCompany <= 2: score += 0.10
    if emp.Age < 30: score += 0.08
    if emp.WorkLifeBalance == 1: score += 0.08
    if emp.EnvironmentSatisfaction <= 2: score += 0.05
    if emp.DistanceFromHome > 20: score += 0.04
    if emp.NumCompaniesWorked > 5: score += 0.04
    if emp.YearsSinceLastPromotion > 5: score += 0.05

    # Protective factors
    if emp.MonthlyIncome > 10000: score -= 0.10
    if emp.YearsAtCompany > 10: score -= 0.08
    if emp.JobSatisfaction == 4: score -= 0.08
    if emp.StockOptionLevel >= 2: score -= 0.05

    return max(0.02, min(0.98, score))


def engineer_features(emp_dict: dict) -> dict:
    """Add engineered features matching the training notebook."""
    d = emp_dict.copy()
    d['IncomePerYearWorked'] = d['MonthlyIncome'] / (d['TotalWorkingYears'] + 1)
    d['SatisfactionComposite'] = (
        d['JobSatisfaction'] + d['EnvironmentSatisfaction'] +
        d['RelationshipSatisfaction'] + d['WorkLifeBalance']
    ) / 4
    d['PromotionStagnation'] = d['YearsSinceLastPromotion'] / (d['YearsAtCompany'] + 1)
    d['RoleTenureRatio'] = d['YearsInCurrentRole'] / (d['YearsAtCompany'] + 1)
    d['IsNewEmployee'] = int(d['YearsAtCompany'] <= 2)
    d['IsYoungWorker'] = int(d['Age'] < 30)
    d['IsLowIncome'] = int(d['MonthlyIncome'] < 3000)
    ot_val = 1 if d.get('OverTime', 0) == 'Yes' or d.get('OverTime', 0) == 1 else 0
    d['CrownMetricRisk'] = ot_val + d['IsLowIncome'] + int(d['JobSatisfaction'] <= 2) + d['IsNewEmployee']
    d['YearsWithManager_Ratio'] = d['YearsWithCurrManager'] / (d['YearsAtCompany'] + 1)
    d['CareerVelocity'] = d['JobLevel'] / (d['TotalWorkingYears'] + 1)
    return d


# ══════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend UI."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "demo_mode": DEMO_MODE
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_attrition(employee: EmployeeInput):
    """
    Predict attrition risk for a single employee.

    Returns risk probability, risk level, top risk factors, and HR recommendation.
    """
    try:
        if DEMO_MODE:
            probability = demo_predict(employee)
        else:
            # Convert to dict, encode categoricals, engineer features
            emp_dict = employee.dict()

            # Label encode categorical columns (must match training encoding)
            categorical_mappings = {
                'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 2, 'Travel_Frequently': 1},
                'Department': {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2},
                'EducationField': {'Human Resources': 0, 'Life Sciences': 1, 'Marketing': 2,
                                   'Medical': 3, 'Other': 4, 'Technical Degree': 5},
                'Gender': {'Female': 0, 'Male': 1},
                'JobRole': {'Healthcare Representative': 0, 'Human Resources': 1,
                            'Laboratory Technician': 2, 'Manager': 3,
                            'Manufacturing Director': 4, 'Research Director': 5,
                            'Research Scientist': 6, 'Sales Executive': 7,
                            'Sales Representative': 8},
                'MaritalStatus': {'Divorced': 0, 'Married': 1, 'Single': 2},
                'OverTime': {'No': 0, 'Yes': 1}
            }

            for col, mapping in categorical_mappings.items():
                if col in emp_dict and isinstance(emp_dict[col], str):
                    emp_dict[col] = mapping.get(emp_dict[col], 0)

            # Add engineered features
            emp_dict = engineer_features(emp_dict)

            # Build feature array in correct order
            features_array = np.array([[emp_dict.get(f, 0) for f in feature_names]])
            features_scaled = scaler.transform(features_array)
            probability = float(model.predict_proba(features_scaled)[0][1])

        # Classify risk
        risk_level, risk_color, recommendation = classify_risk(probability)
        risk_factors = identify_risk_factors(employee)

        return PredictionResponse(
            attrition_risk_percent=round(probability * 100, 1),
            risk_level=risk_level,
            risk_color=risk_color,
            prediction="Will likely leave" if probability >= 0.5 else "Will likely stay",
            confidence=round(max(probability, 1 - probability) * 100, 1),
            top_risk_factors=risk_factors,
            recommendation=recommendation,
            demo_mode=DEMO_MODE
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(employees: list[EmployeeInput]):
    """Predict attrition risk for multiple employees at once."""
    results = []
    for emp in employees:
        result = await predict_attrition(emp)
        results.append(result)
    return {"predictions": results, "total": len(results)}


# ── Serve static files ──
app.mount("/static", StaticFiles(directory="static"), name="static")
