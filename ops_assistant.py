#!/usr/bin/env python3
"""
Operations Demand Assistant (Prototype)
--------------------------------------
A lightweight dispatch assistant designed for bike-share operations managers.
Translates plain-English queries into model-driven demand forecasts,
quantifies stockout risks, and calculates recommended fleet staging buffers.

Usage:
  python ops_assistant.py "How many bikes will we need tomorrow at 8 AM if it rains?"
  python ops_assistant.py "Compare demand on a clear working day vs a rainy working day at 5 PM"
  python ops_assistant.py --interactive
"""

import os
import re
import sys
import joblib
import numpy as np
import pandas as pd


MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bike_demand_models.joblib")


class BikeOpsAdvisor:
    """Operations demand forecasting and staging advisor."""

    def __init__(self, model_path=MODEL_FILE):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                "Please run modeling_and_evaluation.py first to train models."
            )

        artifacts = joblib.load(model_path)
        self.feature_cols = artifacts["feature_cols"]
        self.model_cnt = artifacts["model_cnt"]
        self.model_cas = artifacts["model_cas"]
        self.model_reg = artifacts["model_reg"]
        self.model_q90 = artifacts["model_q90"]
        self.model_q95 = artifacts["model_q95"]

        # Default baseline conditions (Fall, mature 2012 scale)
        self.default_temp_c = 20.0
        self.default_hum_pct = 60.0
        self.default_wind_kmh = 12.0
        self.default_weather = 1  # 1: Clear, 2: Mist/Overcast, 3: Light Rain/Snow, 4: Heavy Rain
        self.default_season = 3   # Fall
        self.default_year = 1     # Mature year index (2012)

    def parse_hour(self, text, fallback=8):
        """Extract hour of day from natural language text."""
        t = text.lower()

        # Named transit periods
        if any(w in t for w in ["morning rush", "morning commute", "breakfast rush"]):
            return 8
        if any(w in t for w in ["evening rush", "evening commute", "rush hour", "after work"]):
            return 17
        if any(w in t for w in ["noon", "midday", "lunch", "lunchtime"]):
            return 12
        if "midnight" in t:
            return 0
        if any(w in t for w in ["deep night", "late night", "wee hours", "overnight"]):
            return 2
        if "night" in t:
            return 21
        if "afternoon" in t:
            return 14
        if "morning" in t:
            return 9

        # 12-hour AM/PM formats (e.g., "8 am", "8am", "5:30 pm", "5pm")
        match_ampm = re.search(r"\b(\d{1,2})(?::\d{2})?\s*(am|pm)\b", t)
        if match_ampm:
            h = int(match_ampm.group(1))
            meridiem = match_ampm.group(2)
            if meridiem == "pm" and h < 12:
                h += 12
            elif meridiem == "am" and h == 12:
                h = 0
            return h

        # 24-hour military formats (e.g., "17:00", "08:30")
        match_24 = re.search(r"\b(\d{1,2}):\d{2}\b", t)
        if match_24:
            return int(match_24.group(1))

        # Explicit integer hour (e.g., "at 8", "at 17")
        match_at = re.search(r"\bat\s+(\d{1,2})\b", t)
        if match_at:
            h = int(match_at.group(1))
            if 0 <= h <= 23:
                return h

        return fallback

    def parse_day(self, text, fallback_weekday=2, fallback_workingday=1):
        """Extract day of week and workingday flag from query."""
        t = text.lower()
        calendar_map = {
            "monday": (1, 1),
            "tuesday": (2, 1),
            "wednesday": (3, 1),
            "thursday": (4, 1),
            "friday": (5, 1),
            "saturday": (6, 0),
            "sunday": (0, 0),
            "weekend": (6, 0),
            "weekday": (2, 1),
            "workday": (2, 1),
            "working day": (2, 1),
            "workingday": (2, 1),
            "holiday": (1, 0),
        }
        for keyword, val in calendar_map.items():
            if keyword in t:
                return val

        if "tomorrow" in t:
            # Assume weekday dispatch planning by default
            return (2, 1)

        return (fallback_weekday, fallback_workingday)

    def parse_weather(self, text):
        """Extract meteorological conditions and temperatures from query."""
        t = text.lower()

        # Weather category
        weathersit = self.default_weather
        if any(w in t for w in ["heavy rain", "downpour", "thunderstorm", "blizzard", "torrential"]):
            weathersit = 4
        elif any(w in t for w in ["rain", "rainy", "snow", "shower", "storm", "wet"]):
            weathersit = 3
        elif any(w in t for w in ["mist", "cloudy", "overcast", "fog"]):
            weathersit = 2
        elif any(w in t for w in ["clear", "sunny", "fair", "sun"]):
            weathersit = 1

        # Temperature
        temp_c = self.default_temp_c
        temp_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*c|c|deg|degrees)", t)
        if temp_match:
            temp_c = float(temp_match.group(1))
        elif any(w in t for w in ["freezing", "snow", "frigid"]):
            temp_c = 2.0
        elif any(w in t for w in ["cold", "chilly"]):
            temp_c = 8.0
        elif any(w in t for w in ["mild", "moderate"]):
            temp_c = 17.0
        elif "warm" in t:
            temp_c = 24.0
        elif any(w in t for w in ["hot", "scorching"]):
            temp_c = 32.0

        # Humidity
        hum_pct = self.default_hum_pct
        hum_match = re.search(r"(\d+)\s*(?:%|percent)\s*hum", t)
        if hum_match:
            hum_pct = float(hum_match.group(1))
        elif any(w in t for w in ["humid", "muggy", "sticky"]):
            hum_pct = 85.0
        elif "dry" in t:
            hum_pct = 35.0

        # Wind speed
        wind_kmh = self.default_wind_kmh
        wind_match = re.search(r"(\d+)\s*(?:km/h|kph|mph|kmh)\s*wind", t)
        if wind_match:
            wind_kmh = float(wind_match.group(1))
        elif any(w in t for w in ["gale", "high wind", "very windy"]):
            wind_kmh = 35.0
        elif any(w in t for w in ["windy", "breezy"]):
            wind_kmh = 24.0
        elif "calm" in t:
            wind_kmh = 5.0

        return weathersit, temp_c, hum_pct, wind_kmh

    def _build_feature_row(self, hr, weekday, workingday, weathersit, temp_c, hum_pct, wind_kmh, mnth=9, season=3):
        """Construct feature vector aligned with model training schema."""
        temp = np.clip(temp_c / 41.0, 0.0, 1.0)
        atemp_c = temp_c + (2.0 if temp_c > 15 else -1.0)
        atemp = np.clip(atemp_c / 50.0, 0.0, 1.0)
        hum = np.clip(hum_pct / 100.0, 0.0, 1.0)
        windspeed = np.clip(wind_kmh / 67.0, 0.0, 1.0)
        holiday = 1 if (workingday == 0 and weekday not in [0, 6]) else 0

        row = {
            "season": season,
            "yr": self.default_year,
            "mnth": mnth,
            "hr": hr,
            "holiday": holiday,
            "weekday": weekday,
            "workingday": workingday,
            "weathersit": weathersit,
            "temp": temp,
            "atemp": atemp,
            "hum": hum,
            "windspeed": windspeed,
            "temp_c": temp_c,
            "atemp_c": atemp_c,
            "temp_diff": atemp_c - temp_c,
            "hum_pct": hum_pct,
            "windspeed_kmh": wind_kmh,
            "temp_sq": temp ** 2,
            "hum_x_temp": hum * temp,
            "sin_hr": np.sin(2 * np.pi * hr / 24),
            "cos_hr": np.cos(2 * np.pi * hr / 24),
            "sin_mnth": np.sin(2 * np.pi * mnth / 12),
            "cos_mnth": np.cos(2 * np.pi * mnth / 12),
            "sin_weekday": np.sin(2 * np.pi * weekday / 7),
            "cos_weekday": np.cos(2 * np.pi * weekday / 7),
            "is_commute_morning": int(workingday == 1 and hr in [7, 8, 9]),
            "is_commute_evening": int(workingday == 1 and hr in [16, 17, 18, 19]),
            "is_commute_peak": int(workingday == 1 and hr in [7, 8, 9, 16, 17, 18, 19]),
            "is_weekend_afternoon": int(workingday == 0 and (11 <= hr <= 17)),
            "is_night": int(hr <= 5),
            "is_bad_weather": int(weathersit in [3, 4]),
            "workingday_x_hr": workingday * hr,
            "temp_x_workingday": temp * workingday,
            "temp_x_weekend": temp * (1 - workingday)
        }
        return pd.DataFrame([row])[self.feature_cols]

    def forecast(self, hr, weekday, workingday, weathersit, temp_c, hum_pct, wind_kmh):
        """
        Run inference across aggregate, decomposed, and quantile models.
        Returns coherent expected demand, segment breakdown, and staging buffers.
        """
        features = self._build_feature_row(hr, weekday, workingday, weathersit, temp_c, hum_pct, wind_kmh)

        # Raw model predictions
        pred_cas = float(np.maximum(self.model_cas.predict(features)[0], 0))
        pred_reg = float(np.maximum(self.model_reg.predict(features)[0], 0))
        pred_q90 = float(np.maximum(self.model_q90.predict(features)[0], 0))
        pred_q95 = float(np.maximum(self.model_q95.predict(features)[0], 0))

        # Consistent total: additive sum of casual + registered guarantees exact 100% breakdown
        expected_total = pred_cas + pred_reg

        # Operational safety sizing:
        # Peak commute hours carry directional stockout risk, requiring a larger reserve buffer
        is_peak_commute = (workingday == 1 and hr in [7, 8, 9, 16, 17, 18, 19])
        peak_buffer_ratio = 1.30 if is_peak_commute else 1.15
        
        # Staging capacity targets
        staging_90 = max(pred_q90, expected_total * (1.20 if is_peak_commute else 1.10))
        staging_95 = max(pred_q95, expected_total * peak_buffer_ratio)
        safety_buffer = staging_95 - expected_total

        return {
            "expected_total": int(round(expected_total)),
            "expected_registered": int(round(pred_reg)),
            "expected_casual": int(round(pred_cas)),
            "reg_share_pct": round((pred_reg / (expected_total + 1e-6)) * 100, 1),
            "cas_share_pct": round((pred_cas / (expected_total + 1e-6)) * 100, 1),
            "service_level_90": int(round(staging_90)),
            "service_level_95": int(round(staging_95)),
            "buffer_bikes": int(round(safety_buffer)),
            "buffer_pct": round((safety_buffer / (expected_total + 1e-6)) * 100, 1),
            "is_peak_commute": is_peak_commute,
            "hr": hr,
            "workingday": workingday,
            "weathersit": weathersit,
            "temp_c": temp_c,
            "hum_pct": hum_pct,
            "wind_kmh": wind_kmh
        }

    def _generate_ops_recommendation(self, res):
        """Craft practical, human dispatch guidance for operations teams."""
        hr = res["hr"]
        is_workday = res["workingday"] == 1
        weather = res["weathersit"]
        expected = res["expected_total"]
        staging = res["service_level_95"]

        if is_workday and hr in [7, 8, 9]:
            if weather in [3, 4]:
                return (
                    f"Morning commute with precipitation. Commuters still ride ({res['expected_registered']} trips), "
                    f"but expect slower trip turnarounds. Pre-stage {staging} bikes at residential origin docks by 07:15. "
                    "Station rebalancing vans at major metro transfers to clear receiving docks as bikes arrive."
                )
            return (
                f"High-volume morning rush. Residential docks will drain rapidly between 07:45 and 08:45. "
                f"Target staging capacity of {staging} bikes across residential corridors. Run early shuttle runs "
                "to downtown commercial hubs to prevent dock lockout upon arrival."
            )

        if is_workday and hr in [16, 17, 18, 19]:
            if weather in [3, 4]:
                return (
                    f"Rain during evening rush suppresses demand by ~30%, but registered commuters ({res['expected_registered']} rides) "
                    f"still require reliable bikes. Pre-stage {staging} bikes in downtown office cores by 16:30. "
                    "Avoid over-allocating vans to recreational stations."
                )
            return (
                f"Peak system demand window. Downtown business districts will experience severe depletion. "
                f"Ensure {staging} bikes are available at downtown hubs before 17:00. Activate user incentive credits "
                "(Bike Angels) for inbound reverse-commute trips."
            )

        if not is_workday and (11 <= hr <= 17):
            if weather in [3, 4]:
                return (
                    f"Rain substantially suppresses leisure riding ({res['expected_casual']} casual riders). "
                    "Keep fleet staged at baseline levels. Good window to pull bikes into depots for preventive maintenance."
                )
            return (
                f"Peak weekend recreational demand. Casual riders account for {res['cas_share_pct']}% of trips. "
                f"Focus staging ({staging} bikes) around waterfront paths, parks, and tourist hubs rather than business districts."
            )

        if hr <= 5 or hr >= 22:
            return (
                f"Off-peak / overnight window ({expected} rides/hr). Minimal rider activity. "
                "Ideal time for battery swapping, mechanical safety checks, and nocturnal depot-to-residential rebalancing."
            )

        return (
            f"Moderate midday activity ({expected} rides/hr). Baseline fleet staging of {staging} bikes is adequate. "
            "Maintain routine roving van sweeps across high-turnover corridors."
        )

    def answer_query(self, query):
        """Process plain-language query and return formatted operational report."""
        q_lower = query.lower()

        # Handle comparison queries (e.g., "compare A vs B")
        if any(w in q_lower for w in [" vs ", " versus ", "compare "]):
            return self._handle_comparison_query(query)

        # Single scenario resolution
        hr = self.parse_hour(query)
        weekday, workingday = self.parse_day(query)
        weathersit, temp_c, hum_pct, wind_kmh = self.parse_weather(query)

        res = self.forecast(hr, weekday, workingday, weathersit, temp_c, hum_pct, wind_kmh)
        rec = self._generate_ops_recommendation(res)

        weather_names = {
            1: "Clear / Partly Cloudy",
            2: "Mist / Overcast",
            3: "Light Rain / Snow",
            4: "Heavy Rain / Storm"
        }
        day_label = "Weekday (Working Day)" if workingday == 1 else "Weekend / Holiday"

        output = f"""
--------------------------------------------------------------------------------
DISPATCH DEMAND ADVISOR | Capital Bikeshare Operations
--------------------------------------------------------------------------------
Query: "{query}"

Parsed Conditions:
  • Target Window:       {hr:02d}:00 ({'Morning Peak' if hr in [7,8,9] else 'Evening Peak' if hr in [16,17,18,19] else 'Off-Peak'})
  • Day Type:            {day_label}
  • Weather Forecast:    {weather_names[weathersit]} ({temp_c:.1f}°C, {hum_pct:.0f}% hum, {wind_kmh:.1f} km/h wind)

Demand Forecast:
  • Total Expected Trips:       {res['expected_total']} rides/hr
    - Registered Commuters:     {res['expected_registered']} ({res['reg_share_pct']}%)
    - Casual Users / Tourists:  {res['expected_casual']} ({res['cas_share_pct']}%)

Fleet Staging & Buffer Targets:
  • Baseline Point Forecast:    {res['expected_total']} bikes
  • 90% Service Level Target:   {res['service_level_90']} bikes (+{res['service_level_90'] - res['expected_total']} buffer)
  • 95% Service Level Target:   {res['service_level_95']} bikes (+{res['buffer_bikes']} buffer / +{res['buffer_pct']}% reserve)

Operational Dispatch Guidance:
  {rec}
--------------------------------------------------------------------------------"""
        return output.strip()

    def _handle_comparison_query(self, query):
        """Handles comparative sensitivity queries between two operating conditions."""
        clean_q = re.sub(r"^compare\s+", "", query, flags=re.IGNORECASE)
        parts = re.split(r"\bvs\b|\bversus\b", clean_q, flags=re.IGNORECASE)

        part_a = parts[0].strip() if len(parts) > 0 else query
        part_b = parts[1].strip() if len(parts) > 1 else query

        global_hr = self.parse_hour(query, fallback=17)
        global_wk, global_wd = self.parse_day(query, fallback_weekday=2, fallback_workingday=1)

        # Scenario A
        hr_a = self.parse_hour(part_a, fallback=global_hr)
        wk_a, wd_a = self.parse_day(part_a, fallback_weekday=global_wk, fallback_workingday=global_wd)
        w_a, t_a, h_a, wnd_a = self.parse_weather(part_a)
        res_a = self.forecast(hr_a, wk_a, wd_a, w_a, t_a, h_a, wnd_a)

        # Scenario B
        hr_b = self.parse_hour(part_b, fallback=global_hr)
        wk_b, wd_b = self.parse_day(part_b, fallback_weekday=global_wk, fallback_workingday=global_wd)
        w_b, t_b, h_b, wnd_b = self.parse_weather(part_b)
        res_b = self.forecast(hr_b, wk_b, wd_b, w_b, t_b, h_b, wnd_b)

        delta = res_b["expected_total"] - res_a["expected_total"]
        pct_change = (delta / (res_a["expected_total"] + 1e-6)) * 100
        weather_labels = {1: "Clear", 2: "Mist", 3: "Rain/Snow", 4: "Storm"}

        output = f"""
--------------------------------------------------------------------------------
SCENARIO COMPARISON & SENSITIVITY ANALYSIS
--------------------------------------------------------------------------------
Query: "{query}"

Scenario 1: "{part_a}"
  • Setting:                 {hr_a:02d}:00 | {'Weekday' if wd_a == 1 else 'Weekend'} | {weather_labels[w_a]} | {t_a:.1f}°C
  • Expected Rides:          {res_a['expected_total']} trips/hr (Registered: {res_a['expected_registered']}, Casual: {res_a['expected_casual']})
  • 95% Staging Target:      {res_a['service_level_95']} bikes

Scenario 2: "{part_b}"
  • Setting:                 {hr_b:02d}:00 | {'Weekday' if wd_b == 1 else 'Weekend'} | {weather_labels[w_b]} | {t_b:.1f}°C
  • Expected Rides:          {res_b['expected_total']} trips/hr (Registered: {res_b['expected_registered']}, Casual: {res_b['expected_casual']})
  • 95% Staging Target:      {res_b['service_level_95']} bikes

Net Impact (Scenario 2 vs Scenario 1):
  • Demand Variance:         {delta:+d} bikes/hr ({pct_change:+.1f}%)
  • Fleet Buffer Variance:   {res_b['service_level_95'] - res_a['service_level_95']:+d} bikes

Dispatch Takeaway:
  {'Rain/cold sharply reduces demand. Scale back active van rebalancing runs and stage for lower turnover.' if pct_change < -25 else 'Significant demand surge. Pre-position extra staging inventory to avoid acute origin stockouts.' if pct_change > 25 else 'Demand is comparable. Standard operational buffers apply.'}
--------------------------------------------------------------------------------"""
        return output.strip()


def run_interactive_mode(advisor):
    """Interactive query REPL for operations managers."""
    print("\nBike-Share Operations Assistant | Interactive REPL")
    print("Type your plain-English demand or staging question below (or 'exit' to quit).\n")
    while True:
        try:
            query = input("Ops Query > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting Operations Assistant. Have a good shift!")
                break
            print("\n" + advisor.answer_query(query) + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    advisor = BikeOpsAdvisor()

    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ["--interactive", "-i"]:
            run_interactive_mode(advisor)
            return

        query = " ".join(sys.argv[1:])
        print(advisor.answer_query(query))
    else:
        # Default demonstration suite
        sample_questions = [
            "How many bikes will we need tomorrow at 8 AM if it rains?",
            "What is the peak fleet requirement on Friday at 5 PM if the weather is clear and 22 degrees?",
            "Compare demand on a clear working day vs a rainy working day at 5 PM",
            "How many bikes should be available on Sunday at 2 PM for casual riders?",
            "Are we at risk of bike shortages at 2 AM on Tuesday?"
        ]
        print("DEMO RUN: Operational Scenarios Evaluated by Demand Advisor\n")
        for q in sample_questions:
            print(advisor.answer_query(q))
            print()


if __name__ == "__main__":
    main()
