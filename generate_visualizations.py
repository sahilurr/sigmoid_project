"""
Operational & Analytical Visualizations Generator
------------------------------------------------
Generates publication-quality charts for the executive report and presentations:
1. Hourly Demand Dynamics (Weekday Commute Spikes vs Weekend Leisure)
2. Weather Elasticity by Rider Segment (Temperature & Precipitation Sensitivity)
3. Out-of-Time Model Benchmark Comparison (RMSE, MAE, R²)
4. 24-Hour Operational Fleet Staging & Dynamic Safety Buffers
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})

os.makedirs("visualizations", exist_ok=True)


def plot_hourly_demand_dynamics(hour_df):
    """Plot bimodal weekday commuter volume vs unimodal weekend recreational flow."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Palette
    color_total = "#1d3557"
    color_reg = "#2a9d8f"
    color_cas = "#e76f51"

    # Subplot 1: Working Days
    work_agg = hour_df[hour_df["workingday"] == 1].groupby("hr")[["cnt", "registered", "casual"]].mean()
    axes[0].plot(work_agg.index, work_agg["cnt"], label="Total Volume", color=color_total, lw=3, marker="o", markersize=4)
    axes[0].plot(work_agg.index, work_agg["registered"], label="Registered Commuters", color=color_reg, lw=2.2, linestyle="--")
    axes[0].plot(work_agg.index, work_agg["casual"], label="Casual Riders", color=color_cas, lw=2, linestyle=":")

    # Highlight commute bands
    axes[0].axvspan(7, 9, color="#fee8c8", alpha=0.55, label="Morning Rush (8 AM)")
    axes[0].axvspan(16, 19, color="#fdbb84", alpha=0.55, label="Evening Rush (5-6 PM)")
    axes[0].axhline(work_agg["cnt"].mean(), color="gray", linestyle="-.", alpha=0.6, label="24h Average Demand")

    axes[0].set_title("Working Days: Bimodal Commuter Tidal Waves")
    axes[0].set_xlabel("Hour of Day (0-23)")
    axes[0].set_ylabel("Average Hourly Bike Trips")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ddd")

    # Annotate peak-to-trough ratio
    axes[0].annotate(
        "Evening Peak: 509 trips/hr\n(57x vs overnight lull)",
        xy=(17, 509), xytext=(11, 460),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff9e6", ec="#d4a373", lw=1),
        fontsize=9.5, fontweight="semibold"
    )

    # Subplot 2: Non-Working Days (Weekends / Holidays)
    nonwork_agg = hour_df[hour_df["workingday"] == 0].groupby("hr")[["cnt", "registered", "casual"]].mean()
    axes[1].plot(nonwork_agg.index, nonwork_agg["cnt"], label="Total Volume", color=color_total, lw=3, marker="o", markersize=4)
    axes[1].plot(nonwork_agg.index, nonwork_agg["registered"], label="Registered Riders", color=color_reg, lw=2.2, linestyle="--")
    axes[1].plot(nonwork_agg.index, nonwork_agg["casual"], label="Casual Tourists/Leisure", color=color_cas, lw=2, linestyle=":")

    axes[1].axvspan(12, 16, color="#e0f3f8", alpha=0.55, label="Midday Leisure Window")
    axes[1].axhline(nonwork_agg["cnt"].mean(), color="gray", linestyle="-.", alpha=0.6, label="24h Average Demand")

    axes[1].set_title("Weekends & Holidays: Smooth Leisure Bell Curve")
    axes[1].set_xlabel("Hour of Day (0-23)")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ddd")

    plt.suptitle("Fleet Utilization Paradox: Extreme Temporal & Directional Concentration", y=1.02)
    plt.tight_layout()
    output_path = "visualizations/hourly_demand_paradox.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Exported: {output_path}")


def plot_weather_elasticity(hour_df):
    """Plot temperature and precipitation sensitivity across rider segments."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    hour_df = hour_df.copy()
    hour_df["temp_c"] = hour_df["temp"] * 41.0
    hour_df["temp_bin"] = pd.cut(
        hour_df["temp_c"],
        bins=[-5, 10, 20, 30, 45],
        labels=["<10°C (Cold)", "10-20°C (Mild)", "20-30°C (Warm)", ">30°C (Hot)"]
    )

    # 1. Temperature bins
    temp_summary = hour_df.groupby("temp_bin", observed=False)[["registered", "casual"]].mean()
    temp_summary.plot(kind="bar", ax=axes[0], color=["#2a9d8f", "#e76f51"], width=0.65, edgecolor="none")
    axes[0].set_title("Rider Volume by Temperature Range")
    axes[0].set_xlabel("Temperature Category")
    axes[0].set_ylabel("Average Hourly Trips")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(["Registered Commuters", "Casual Leisure"], frameon=True, facecolor="white")

    # Annotate casual surge
    axes[0].text(
        0.5, 0.85,
        "Casual rides surge 6.4x from Cold to Warm\n(Registered volume remains resilient)",
        transform=axes[0].transAxes,
        ha="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f8f9fa", ec="#ced4da", lw=1)
    )

    # 2. Weather conditions
    weather_labels = {1: "Clear / Few Clouds", 2: "Mist / Overcast", 3: "Light Rain / Snow"}
    filtered_weather = hour_df[hour_df["weathersit"].isin([1, 2, 3])].copy()
    filtered_weather["weather_label"] = filtered_weather["weathersit"].map(weather_labels)
    weather_summary = filtered_weather.groupby("weather_label")[["registered", "casual"]].mean().loc[
        ["Clear / Few Clouds", "Mist / Overcast", "Light Rain / Snow"]
    ]
    weather_summary.plot(kind="bar", ax=axes[1], color=["#2a9d8f", "#e76f51"], width=0.55, edgecolor="none")
    axes[1].set_title("Rider Volume by Precipitation Condition")
    axes[1].set_xlabel("Weather Condition")
    axes[1].tick_params(axis="x", rotation=10)
    axes[1].legend(["Registered Commuters", "Casual Leisure"], frameon=True, facecolor="white")

    # Annotate rain drop
    axes[1].text(
        0.5, 0.85,
        "Rain reduces casual volume by -60%\nRegistered drops by -42%",
        transform=axes[1].transAxes,
        ha="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f8f9fa", ec="#ced4da", lw=1)
    )

    plt.suptitle("Weather Elasticity: Commuter Inelasticity vs. Tourist Volatility", y=1.02)
    plt.tight_layout()
    output_path = "visualizations/weather_impact_elasticity.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Exported: {output_path}")


def plot_model_benchmarks():
    """Plot test set comparison across candidate models."""
    models = ["Historical\nStratified Mean", "Regularized\nRidge", "Random\nForest", "Gradient Boosted\nTrees (HGB)"]
    rmse_scores = [149.45, 123.68, 75.25, 65.34]
    mae_scores = [105.68, 88.64, 50.13, 43.30]
    r2_scores = [0.5286, 0.6772, 0.8805, 0.9099]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RMSE
    bars1 = axes[0].bar(models, rmse_scores, color=["#b0c4de", "#82a0d8", "#4682b4", "#1d3557"], width=0.6)
    axes[0].set_title("Test RMSE (Lower is Better)")
    axes[0].set_ylabel("Trips / Hour")
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="semibold")

    # MAE
    bars2 = axes[1].bar(models, mae_scores, color=["#bfe3d5", "#70c1b3", "#2a9d8f", "#1b6f63"], width=0.6)
    axes[1].set_title("Test MAE (Lower is Better)")
    axes[1].set_ylabel("Trips / Hour")
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="semibold")

    # R2
    bars3 = axes[2].bar(models, r2_scores, color=["#ffd6a5", "#ffb703", "#fb8500", "#d45b00"], width=0.6)
    axes[2].set_title("Test R² Variance Explained (Higher is Better)")
    axes[2].set_ylabel("R² Score")
    axes[2].set_ylim(0, 1.05)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="semibold")

    plt.suptitle("Model Evaluation on Out-of-Time Test Holdout (Sep - Dec 2012)", y=1.03)
    plt.tight_layout()
    output_path = "visualizations/model_benchmark_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Exported: {output_path}")


def plot_operational_fleet_staging():
    """Plot 24-hour staging requirements comparing point forecast against 90% and 95% service buffers."""
    from ops_assistant import BikeOpsAdvisor
    advisor = BikeOpsAdvisor()

    hours = list(range(24))
    point_forecasts = []
    staging_90s = []
    staging_95s = []

    for h in hours:
        res = advisor.forecast(
            hr=h, weekday=2, workingday=1, weathersit=1,
            temp_c=20.0, hum_pct=60.0, wind_kmh=12.0
        )
        point_forecasts.append(res["expected_total"])
        staging_90s.append(res["service_level_90"])
        staging_95s.append(res["service_level_95"])

    plt.figure(figsize=(13, 6))

    plt.plot(hours, point_forecasts, label="Expected Demand (Point Forecast)", color="#1d3557", lw=2.5, marker="o", markersize=4)
    plt.plot(hours, staging_90s, label="90% Service Level Staging Target", color="#f4a261", lw=2, linestyle="--")
    plt.plot(hours, staging_95s, label="95% Service Level Staging Target (Recommended Fleet)", color="#e63946", lw=2.8, marker="s", markersize=4)

    plt.fill_between(hours, point_forecasts, staging_95s, color="#e63946", alpha=0.12, label="Dynamic Operational Safety Buffer")

    # Annotate morning and evening buffers
    plt.annotate(
        "Morning Rush Target: 694 bikes\n(+160 buffer reserve)",
        xy=(8, staging_95s[8]), xytext=(3, 620),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#e63946", lw=1),
        fontsize=9.5, fontweight="semibold"
    )

    plt.annotate(
        "Evening Rush Target: 897 bikes\n(+207 buffer reserve)",
        xy=(17, staging_95s[17]), xytext=(12, 850),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#e63946", lw=1),
        fontsize=9.5, fontweight="semibold"
    )

    plt.title("24-Hour Operational Fleet Staging Schedule (Clear Weekday, 20°C)\nSafety buffer dynamically expands during commute spikes and contracts overnight")
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Bicycles Required at Stations")
    plt.xticks(range(0, 24))
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ddd")

    plt.tight_layout()
    output_path = "visualizations/operational_fleet_staging.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Exported: {output_path}")


def main():
    raw_hour = pd.read_csv("hour.csv")
    raw_hour["datetime"] = pd.to_datetime(raw_hour["dteday"]) + pd.to_timedelta(raw_hour["hr"], unit="h")

    plot_hourly_demand_dynamics(raw_hour)
    plot_weather_elasticity(raw_hour)
    plot_model_benchmarks()
    plot_operational_fleet_staging()


if __name__ == "__main__":
    main()
