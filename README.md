# Employee Attrition Prediction API

**Team 1 — Data Analytics Internship 2026**

A FastAPI web application that serves the trained ML model as a REST API with a beautiful web interface.

---

## Quick Start

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Add your trained model

Copy the 3 `.pkl` files from the ML notebook into the `model/` folder:

```
model/
  attrition_model_final.pkl
  attrition_scaler.pkl
  attrition_features.pkl
```

> If you skip this step, the API runs in **Demo Mode** using rule-based predictions.

### Step 3: Run the server

```bash
uvicorn app:app --reload
```

### Step 4: Open the app

Go to: **http://127.0.0.1:8000**

---

## API Endpoints

| Method | Endpoint       | Description                              |
|--------|----------------|------------------------------------------|
| GET    | `/`            | Web interface (frontend)                 |
| GET    | `/health`      | Health check — confirms model is loaded  |
| POST   | `/predict`     | Predict attrition for one employee       |
| POST   | `/predict/batch` | Predict for multiple employees         |
| GET    | `/docs`        | Auto-generated Swagger API docs          |

---

## Example API Call (Python)

```python
import requests

employee = {
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

response = requests.post("http://127.0.0.1:8000/predict", json=employee)
print(response.json())
```

---

## Project Structure

```
attrition_api/
├── app.py               # FastAPI application
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── model/               # Place .pkl files here
│   └── README.txt
└── static/
    └── index.html       # Web frontend
```

---

## Tech Stack

- **FastAPI** — Modern Python web framework
- **scikit-learn** — ML model serving
- **Pydantic** — Data validation
- **Uvicorn** — ASGI server
- **HTML/CSS/JS** — Frontend interface

---

*Team 1 — Employee Attrition Analysis | Data Analytics Internship 2026*
