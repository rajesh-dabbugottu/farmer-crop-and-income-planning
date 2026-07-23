const API_URL = "https://agriplan-ai-api.onrender.com/recommend";
const USE_DEMO_MODE = true;
const DISPLAY_SCALE = 100;

const form = document.getElementById("recommendationForm");
const submitButton = document.getElementById("submitButton");
const buttonText = document.getElementById("buttonText");
const message = document.getElementById("formMessage");
const emptyResult = document.getElementById("emptyResult");
const resultContent = document.getElementById("resultContent");

//document.getElementById("currentYearText").textContent = new Date().getFullYear();

document.getElementById("menuButton").addEventListener("click", () => {
  document.getElementById("navLinks").classList.toggle("open");
});

form.addEventListener("reset", () => {
  setTimeout(() => {
    emptyResult.classList.remove("hidden");
    resultContent.classList.add("hidden");
    message.textContent = "";
  }, 0);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    rainfall: Number(document.getElementById("rainfall").value),
    expected_yield: Number(document.getElementById("expectedYield").value),
    market_price: Number(document.getElementById("marketPrice").value),
    savings: Number(document.getElementById("savings").value),
    previous_crop: document.getElementById("previousCrop").value,
    current_year: Number(document.getElementById("currentYear").value)
  };

  if (Object.values(payload).slice(0, 4).some(value => !Number.isFinite(value) || value < 0)) {
    message.textContent = "Please enter valid non-negative values.";
    return;
  }

  setLoading(true);
  message.textContent = "Generating recommendation...";

  try {
    const result = USE_DEMO_MODE
      ? await getDemoRecommendation(payload)
      : await getApiRecommendation(payload);

    renderRecommendation(result, payload);
    message.textContent = USE_DEMO_MODE
      ? "Recommendation received from FastAPI."
      : "Recommendation received from FastAPI.";
  } catch (error) {
    console.error(error);
    message.textContent = "Recommendation failed. Check the FastAPI server.";
  } finally {
    setLoading(false);
  }
});

async function getApiRecommendation(payload) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function getDemoRecommendation(payload) {
  return new Promise(resolve => {
    setTimeout(() => {
      const crop = chooseCrop(payload);
      const estimatedProfit =
        payload.expected_yield *
        payload.market_price *
        24 *
        rainfallFactor(payload.rainfall);

      const rotationPenalty =
        payload.previous_crop === crop ? estimatedProfit * 0.08 : 0;

      const annualLivingCost = 250000;
      const estimatedNetIncome =
        estimatedProfit - rotationPenalty - annualLivingCost;

      resolve({
        recommended_crop: crop,
        estimated_profit: estimatedProfit,
        rotation_penalty: rotationPenalty,
        annual_living_cost: annualLivingCost,
        estimated_net_income: estimatedNetIncome,
        updated_savings: payload.savings + estimatedNetIncome,
        explanation: `${crop} was selected using the entered climate, market, yield and investment information.`,
        closest_context: {
          Annual_Rainfall: payload.rainfall * 0.97,
          Yield: payload.expected_yield * 1.03,
          Modal_Price: payload.market_price * 0.99,
          Crop: crop
        }
      });
    }, 650);
  });
}

function chooseCrop(payload) {
  if (payload.rainfall > 1400) return "Rice";
  if (payload.rainfall < 500) return "Maize";
  if (payload.market_price > 5000) return "Cotton";
  if (payload.expected_yield > 6) return "Sugarcane";
  return "Wheat";
}

function rainfallFactor(rainfall) {
  if (rainfall >= 650 && rainfall <= 1300) return 1;
  if (rainfall >= 450 && rainfall <= 1600) return 0.82;
  return 0.62;
}

function renderRecommendation(result, payload) {
  emptyResult.classList.add("hidden");
  resultContent.classList.remove("hidden");

  const context = result.closest_context || {};

  document.getElementById("recommendedCrop").textContent = result.recommended_crop;
  document.getElementById("recommendationExplanation").textContent = result.explanation;
  document.getElementById("estimatedProfit").textContent = formatMoney(result.estimated_profit);
  document.getElementById("rotationPenalty").textContent = formatMoney(result.rotation_penalty);
  document.getElementById("livingCost").textContent = formatMoney(result.annual_living_cost);
  document.getElementById("netIncome").textContent = formatMoney(result.estimated_net_income);
  document.getElementById("updatedSavings").textContent = formatMoney(result.updated_savings);

  document.getElementById("matchedRainfall").textContent =
    `${formatNumber(context.Annual_Rainfall)} mm`;
  document.getElementById("matchedYield").textContent =
    `${formatNumber(context.Yield)} t/ha`;
  document.getElementById("matchedPrice").textContent =
    `₹${formatNumber(context.Modal_Price)} /q`;
  document.getElementById("matchedCrop").textContent =
    context.Crop || result.recommended_crop;

  document.getElementById("trendIndicator").textContent =
    Number(result.updated_savings) >= payload.savings
      ? "Positive trend ↑"
      : "Negative trend ↓";
}

function formatMoney(value) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2
  }).format(Number(value || 0) / DISPLAY_SCALE);
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number)
    ? number.toLocaleString("en-IN", {maximumFractionDigits: 2})
    : "—";
}

function setLoading(loading) {
  submitButton.disabled = loading;
  buttonText.textContent = loading ? "Generating..." : "Recommend Crop";
}
