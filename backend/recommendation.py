from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = MODEL_DIR / "processed_data.csv"
MODEL_PATH = MODEL_DIR / "ppo_farmer_model.zip"
VEC_NORMALIZE_PATH = MODEL_DIR / "vec_normalize.pkl"

ANNUAL_LIVING_COST = 250_000.0
ROTATION_PENALTY_RATE = 0.10
MAX_PLANNING_YEARS = 30


class InferenceEnvironment(gym.Env):
    """Minimal environment used only to restore VecNormalize statistics."""

    metadata = {"render_modes": []}

    def __init__(self, number_of_actions: int):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(number_of_actions)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        observation = np.zeros(6, dtype=np.float32)
        return observation, 0.0, True, False, {}


@dataclass(frozen=True)
class FeatureRange:
    minimum: float
    maximum: float

    def scale(self, value: float) -> float:
        if not np.isfinite(value):
            return 0.0

        if self.maximum <= self.minimum:
            return 0.0

        scaled = 2.0 * ((value - self.minimum) / (self.maximum - self.minimum)) - 1.0
        return float(np.clip(scaled, -1.0, 1.0))


class CropRecommendationService:
    """Loads the dataset, VecNormalize statistics and trained PPO policy once."""

    def __init__(self) -> None:
        self._validate_files()

        self.data = pd.read_csv(DATA_PATH)
        self._validate_dataset()

        # This preserves the exact first-appearance crop order in the training CSV.
        self.crops = self.data["Crop"].astype(str).drop_duplicates().tolist()
        self.crop_to_index = {crop: index for index, crop in enumerate(self.crops)}

        self.rainfall_range = self._robust_range("Annual_Rainfall")
        self.yield_range = self._robust_range("Yield")
        self.price_range = self._robust_range("Modal_Price")

        # Profit is used as a practical scale for user savings.
        profit_abs = pd.to_numeric(self.data["Profit"], errors="coerce").abs()
        savings_max = float(profit_abs.quantile(0.95))
        if not np.isfinite(savings_max) or savings_max <= 0:
            savings_max = 1_000_000.0
        self.savings_range = FeatureRange(0.0, savings_max)

        dummy_env = DummyVecEnv(
            [lambda: InferenceEnvironment(len(self.crops))]
        )

        self.vec_normalize = VecNormalize.load(
            str(VEC_NORMALIZE_PATH),
            dummy_env,
        )
        self.vec_normalize.training = False
        self.vec_normalize.norm_reward = False

        self.model = PPO.load(
            str(MODEL_PATH),
            device="cpu",
        )

    def _validate_files(self) -> None:
        missing = [
            str(path.name)
            for path in (DATA_PATH, MODEL_PATH, VEC_NORMALIZE_PATH)
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing required backend files: {', '.join(missing)}"
            )

    def _validate_dataset(self) -> None:
        required_columns = {
            "Crop",
            "Annual_Rainfall",
            "Yield",
            "Modal_Price",
            "Profit",
        }
        missing = required_columns.difference(self.data.columns)

        if missing:
            raise ValueError(
                f"Dataset is missing columns: {', '.join(sorted(missing))}"
            )

        numeric_columns = [
            "Annual_Rainfall",
            "Yield",
            "Modal_Price",
            "Profit",
        ]
        for column in numeric_columns:
            self.data[column] = pd.to_numeric(
                self.data[column],
                errors="coerce",
            )

        self.data = self.data.dropna(
            subset=["Crop", *numeric_columns]
        ).reset_index(drop=True)

        if self.data.empty:
            raise ValueError("The processed dataset contains no usable rows.")

    def _robust_range(self, column: str) -> FeatureRange:
        values = self.data[column].replace([np.inf, -np.inf], np.nan).dropna()
        minimum = float(values.quantile(0.01))
        maximum = float(values.quantile(0.99))

        if maximum <= minimum:
            minimum = float(values.min())
            maximum = float(values.max())

        return FeatureRange(minimum, maximum)

    def _previous_crop_value(self, previous_crop: str) -> float:
        if previous_crop == "None" or previous_crop not in self.crop_to_index:
            return -1.0

        if len(self.crops) == 1:
            return 0.0

        index = self.crop_to_index[previous_crop]
        return float(2.0 * index / (len(self.crops) - 1) - 1.0)

    def _build_observation(
        self,
        *,
        rainfall: float,
        expected_yield: float,
        market_price: float,
        savings: float,
        previous_crop: str,
        current_year: int,
    ) -> np.ndarray:
        progress = 2.0 * (
            min(current_year, MAX_PLANNING_YEARS - 1)
            / (MAX_PLANNING_YEARS - 1)
        ) - 1.0

        observation = np.array(
            [
                progress,
                self.savings_range.scale(savings),
                self.rainfall_range.scale(rainfall),
                self.price_range.scale(market_price),
                self.yield_range.scale(expected_yield),
                self._previous_crop_value(previous_crop),
            ],
            dtype=np.float32,
        )

        return observation.reshape(1, -1)

    def _distance_series(
        self,
        frame: pd.DataFrame,
        rainfall: float,
        expected_yield: float,
        market_price: float,
    ) -> pd.Series:
        rainfall_scale = max(
            self.rainfall_range.maximum - self.rainfall_range.minimum,
            1.0,
        )
        yield_scale = max(
            self.yield_range.maximum - self.yield_range.minimum,
            1.0,
        )
        price_scale = max(
            self.price_range.maximum - self.price_range.minimum,
            1.0,
        )

        return np.sqrt(
            ((frame["Annual_Rainfall"] - rainfall) / rainfall_scale) ** 2
            + ((frame["Yield"] - expected_yield) / yield_scale) ** 2
            + ((frame["Modal_Price"] - market_price) / price_scale) ** 2
        )

    def _closest_row(
        self,
        crop: str,
        rainfall: float,
        expected_yield: float,
        market_price: float,
    ) -> tuple[pd.Series, float]:
        selected = self.data[self.data["Crop"] == crop].copy()

        if selected.empty:
            selected = self.data.copy()

        distances = self._distance_series(
            selected,
            rainfall,
            expected_yield,
            market_price,
        )
        closest_index = distances.idxmin()
        return selected.loc[closest_index], float(distances.loc[closest_index])

    def recommend(
        self,
        *,
        rainfall: float,
        expected_yield: float,
        market_price: float,
        savings: float,
        previous_crop: str,
        current_year: int,
    ) -> dict:
        raw_observation = self._build_observation(
            rainfall=rainfall,
            expected_yield=expected_yield,
            market_price=market_price,
            savings=savings,
            previous_crop=previous_crop,
            current_year=current_year,
        )

        normalized_observation = self.vec_normalize.normalize_obs(
            raw_observation.copy()
        )

        action, _ = self.model.predict(
            normalized_observation,
            deterministic=True,
        )
        action_index = int(np.asarray(action).reshape(-1)[0])

        if action_index < 0 or action_index >= len(self.crops):
            raise ValueError(
                f"PPO returned invalid action index {action_index}."
            )

        recommended_crop = self.crops[action_index]
        closest_row, match_distance = self._closest_row(
            recommended_crop,
            rainfall,
            expected_yield,
            market_price,
        )

        estimated_profit = float(closest_row["Profit"])
        rotation_penalty = (
            abs(estimated_profit) * ROTATION_PENALTY_RATE
            if previous_crop == recommended_crop
            else 0.0
        )
        estimated_net_income = (
            estimated_profit
            - rotation_penalty
            - ANNUAL_LIVING_COST
        )
        updated_savings = savings + estimated_net_income

        rotation_text = (
            " A rotation penalty was applied because it matches the previous crop."
            if rotation_penalty > 0
            else " No rotation penalty was applied."
        )

        return {
            "recommended_crop": recommended_crop,
            "action_index": action_index,
            "estimated_profit": round(estimated_profit, 2),
            "rotation_penalty": round(rotation_penalty, 2),
            "annual_living_cost": round(ANNUAL_LIVING_COST, 2),
            "estimated_net_income": round(estimated_net_income, 2),
            "updated_savings": round(updated_savings, 2),
            "explanation": (
                f"The trained PPO policy selected {recommended_crop} "
                f"for the supplied climate, market and financial state."
                f"{rotation_text}"
            ),
            "closest_context": {
                "Crop": str(closest_row["Crop"]),
                "Crop_Year": self._safe_value(
                    closest_row.get("Crop_Year")
                ),
                "Season": self._safe_value(
                    closest_row.get("Season")
                ),
                "State": self._safe_value(
                    closest_row.get("State")
                ),
                "Annual_Rainfall": round(
                    float(closest_row["Annual_Rainfall"]),
                    4,
                ),
                "Yield": round(float(closest_row["Yield"]), 4),
                "Modal_Price": round(
                    float(closest_row["Modal_Price"]),
                    4,
                ),
                "Profit": round(float(closest_row["Profit"]), 2),
                "Match_Distance": round(match_distance, 6),
            },
        }

    @staticmethod
    def _safe_value(value):
        if value is None or pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value
