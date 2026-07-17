# AgriPlan AI FastAPI Backend

## Folder structure

```text
backend/
├── app.py
├── recommendation.py
├── requirements.txt
└── model/
    ├── processed_data.csv
    ├── ppo_farmer_model.zip
    └── vec_normalize.pkl
```

## 1. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install packages

```bash
pip install -r requirements.txt
```

## 3. Start FastAPI

Run this command inside the `backend` folder:

```bash
uvicorn app:app --reload
```

Open:

- API: `http://127.0.0.1:8000`
- Swagger documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## 4. Test request

```json
{
  "rainfall": 850,
  "expected_yield": 3.8,
  "market_price": 2400,
  "savings": 150000,
  "previous_crop": "Wheat",
  "current_year": 0
}
```

## 5. Connect the frontend

In `frontend/script.js`, change:

```js
const USE_DEMO_MODE = false;
```

The frontend already uses:

```js
const API_URL = "http://127.0.0.1:8000/recommend";
```

## Important model note

The backend preserves the six crop actions in the first-appearance order of the
training CSV:

```text
Cotton, Maize, Onion, Potato, Wheat, Banana
```

The observation contains six values:

```text
planning progress, savings, rainfall, market price, yield, previous crop
```

The backend then applies the uploaded `VecNormalize` statistics before PPO
prediction.
