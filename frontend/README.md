# AgriPlan AI Frontend

Responsive frontend for the Farmer Crop and Income Planning PPO project.

## Run

Open `index.html` in your browser or use VS Code Live Server.

## Demo mode

`script.js` currently contains:

```js
const USE_DEMO_MODE = true;
```

This allows the frontend to work without a backend.

## Connect to FastAPI

Change:

```js
const USE_DEMO_MODE = false;
```

The expected endpoint is:

```text
POST http://127.0.0.1:8000/recommend
```

The JavaScript sends:

```json
{
  "rainfall": 850,
  "expected_yield": 3.8,
  "market_price": 2400,
  "savings": 150000,
  "previous_crop": "None",
  "current_year": 0
}
```

Financial result values are divided by 100 only for display using:

```js
const DISPLAY_SCALE = 100;
```
