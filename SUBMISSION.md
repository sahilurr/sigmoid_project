# Bike-Share Fleet Optimization & Demand Intelligence
**Candidate Case Study Submission: Technical Report & Strategic Findings**  
*Dataset: Capital Bikeshare Telemetry (2011–2012)*  

---

## Executive Summary: Corridor Conversation with the CEO

> **The CEO's Question:**  
> *"Riders keep complaining they can’t find a bike when they need one. Meanwhile I see bikes sitting unused all over the city. Should we just buy more bikes?"*

### The Direct Answer: Emphatically No.
Buying more bikes will inflate capital expenditure (\$1.5M–\$2.0M) and ongoing maintenance costs without solving the rider stockouts. In fact, injecting more bikes into the network will worsen destination dock congestion.

The data reveals that this is **not an aggregate capacity deficit, but an acute spatial-temporal rebalancing bottleneck**:

1. **Extreme Temporal Volatility (57:1 Peak-to-Trough Ratio):**
   Across the 2-year system history (17,379 recorded hours), system-wide median demand is only **142 bikes/hour**. However, on regular workdays, demand surges to **509 bikes/hour** during evening rush (peaking at **977 bikes/hour**), while deep-night demand drops to just **8.9 bikes/hour** and midday demand sits at **185 bikes/hour**. Outside of 4 peak commute hours, the vast majority of our fleet sits idle.
2. **The Unidirectional Commuter Tidal Wave:**
   Registered commuters drive **86.8% of weekday rides**. Commuters take bikes from residential stations in the morning (draining docks to zero) and flood commercial downtown corridors (leaving zero open docks to return a bike). In the evening, the exact reverse occurs. Buying 2,000 more bikes simply adds 2,000 more idle bicycles locked into downtown docks at 10:00 AM.
3. **The Strategic Remedy:**
   Shift capital away from fleet expansion and into **predictive pre-staging**, **dynamic off-peak van rebalancing**, and **user incentive credits ("Bike Angels")** to encourage voluntary reverse-commute trips.

---

## 1. Documented Assumptions

To ground this analysis, we explicitly document our core modeling and operational assumptions:

### 1.1 Data & Measurement Assumptions
1. **Unobserved Latent Demand:** The recorded metric `cnt` reflects *completed rentals*, not total attempted demand. When a station is empty, frustrated users walk away without being logged. Thus, `cnt` underestimates peak demand at depleted stations.
2. **System Boundary:** The dataset represents the Washington D.C. Capital Bikeshare network between Jan 1, 2011 and Dec 31, 2012. We assume macro transit habits are stable across this period, with organic growth captured via temporal trend features.
3. **Severe Weather Censoring:** The 165 missing hourly timestamps (0.94% of the calendar) are assumed to be exogenous system shutdowns (e.g., Hurricane Sandy in October 2012) rather than random sensor drops.

### 1.2 Operational & Economic Assumptions
1. **Fleet Capex vs Rebalancing Opex:** 
   - Purchasing an additional commercial bike + dock capacity costs ~\$1,000–\$1,200/unit upfront plus maintenance.
   - Motorized van rebalancing costs ~\$5.50–\$6.50 per bike moved.
   - In-app rider incentives ("Bike Angels" ride credits) cost ~\$1.50–\$2.20 per bike moved, delivering comparable rebalancing at a 60% lower cost.
2. **Turnaround & Buffer Capacity:** Bicycles in transit have an average trip duration of 15–25 minutes. A single bike cannot serve more than 2 trips in a single hour. Therefore, operational staging must include a safety buffer above the expected hourly demand.
3. **Target Service Level:** Operations targets a **95% service level** during peak commute windows (ensuring 95% of peak hours experience zero stockouts at origin hubs).

---

## 2. Data Hygiene, Quality Checks & Feature Understanding

### 2.1 Data Hygiene Audit
We verified data consistency across `day.csv` (731 rows) and `hour.csv` (17,379 rows):
- **Missing Values & Duplicates:** Zero null cells and zero duplicate rows across all tables.
- **Accounting Consistency:** Verified the fundamental identity $cnt = casual + registered$ across 100% of rows (`True` for all 17,379 entries).
- **Temporal Continuity:** Spanning 2011-01-01 to 2012-12-31 (17,544 total calendar hours), 165 hours (0.94%) were missing, corresponding to severe weather shutdowns and early overnight telemetry pauses.

### 2.2 Physical De-Normalization of Weather Attributes
The raw UCI dataset normalizes meteorological factors. We converted them back into real-world physical units to ensure operational interpretability:
- **Temperature:** Rescaled from $[0, 1]$ via $\text{temp} \times 41.0^\circ\text{C}$ (range: $0.8^\circ\text{C}$ to $41.0^\circ\text{C}$).
- **Feels-Like Temperature:** Rescaled via $\text{atemp} \times 50.0^\circ\text{C}$ (range: $0.0^\circ\text{C}$ to $50.0^\circ\text{C}$).
- **Relative Humidity:** Rescaled via $\text{hum} \times 100\%$ (range: $0\%$ to $100\%$).
- **Wind Speed:** Rescaled via $\text{windspeed} \times 67.0\text{ km/h}$ (range: $0.0\text{ km/h}$ to $57.0\text{ km/h}$).

### 2.3 Rider Segment Understanding & Elasticity
The dataset reveals two fundamentally different user bases sharing the same physical assets:

| Dimension | Registered Commuters | Casual Leisure Riders |
| :--- | :--- | :--- |
| **Share of Total Volume** | **81.2%** (2,672,662 rides) | **18.8%** (620,017 rides) |
| **Weekday Share** | **86.8%** | **13.2%** |
| **Weekend Share** | **68.3%** | **31.7%** |
| **Hourly Curve** | **Bimodal spikes** at 8:00 AM and 5:00–6:00 PM | **Unimodal bell curve** peaking 12:00–4:00 PM |
| **Weather Elasticity** | **Inelastic:** Commuters ride in cold & light rain | **Highly Elastic:** Surges **6.4x** from cold ($<10^\circ\text{C}$) to warm ($20\text{–}30^\circ\text{C}$); drops **-60.4%** in rain |
| **Operational Impact** | Drives acute directional dock depletion | Drives weekend recreational station depletion |

### 2.4 Engineered Features (34 Predictors)
To provide the machine learning models with strong inductive signals, we engineered:
1. **Continuous Cyclical Encodings:** $\sin$ and $\cos$ transformations for hour of the day ($\frac{2\pi \cdot hr}{24}$), day of the week ($\frac{2\pi \cdot weekday}{7}$), and month ($\frac{2\pi \cdot mnth}{12}$) to preserve boundary continuity (e.g., transitioning smoothly from 23:00 to 00:00).
2. **Domain-Specific Segment Flags:**
   - `is_commute_morning`: Weekdays from 7:00 AM to 9:00 AM.
   - `is_commute_evening`: Weekdays from 4:00 PM to 7:00 PM.
   - `is_commute_peak`: Union of morning and evening spikes.
   - `is_weekend_afternoon`: Non-workdays between 11:00 AM and 5:00 PM.
   - `is_night`: Low-demand window between 12:00 AM and 5:00 AM.
   - `is_bad_weather`: Adverse weather categories 3 (rain/snow) and 4 (heavy storm).
3. **Cross-Feature Interaction Terms:**
   - `workingday * hr`: Captures the differing diurnal rhythm of workdays vs non-workdays.
   - `temp * workingday` & `temp * weekend`: Separates temperature sensitivity for commuters vs tourists.
   - `temp_diff` ($atemp - temp$): Captures humidity and wind chill divergence.

---

## 3. Testable Hypothesis & Design of Experiment (DoE)

### 3.1 Hypothesis Formulation
> **Primary Hypothesis:**  
> *"Weekday rider stockouts during rush hours are driven by directional dock depletion rather than fleet shortage. Implementing predictive pre-staging combined with dynamic user incentive credits ('Bike Angels') will decrease peak-hour zero-bike stockout minutes by at least **35%** across target corridors without adding any bicycles to the fleet."*

### 3.2 Experimental Methodology: Cluster-Level Switchback Design
* **Why Traditional A/B Testing Fails:** A standard user-level randomized A/B test violates the **Stable Unit Treatment Value Assumption (SUTVA)** due to spatial network interference. If a "Treatment" user receives an incentive and moves a bike, they directly change bike availability for a "Control" user at the destination station.
* **Switchback Structure:**
  - We select **4 high-volume commute corridor clusters** (e.g., Corridor A: Capitol Hill $\leftrightarrow$ Downtown K Street; Corridor B: Dupont Circle $\leftrightarrow$ Federal Triangle).
  - We execute an **8-week trial alternating 2-week time blocks**:
    - **Treatment Blocks (Weeks 1–2, 5–6):** Pre-staging vans position buffer bikes at residential stations by 07:15 AM; mobile app activates "Bike Angels" bonus points for users riding reverse-commute routes.
    - **Control Blocks (Weeks 3–4, 7–8):** Standard passive operations (reactive rebalancing only).

### 3.3 Metrics Framework

| Level | Metric Name | Definition & Target | Statistical Power / MDE |
| :--- | :--- | :--- | :--- |
| **Primary** | **Zero-Bike Dockout Minutes** | % of peak commute minutes (7:30–9:00 AM, 4:30–6:30 PM) where an origin station has $\le 1$ available bike. **Target: $\ge 35\%$ reduction.** | MDE $= 35\%$, $\alpha = 0.05$, Statistical Power $> 95\%$ |
| **Secondary** | **Completed Peak Rides** | Total completed trips during peak commute windows. **Target: $+8\%\text{ to }12\%$ uplift** (capturing unmet latent demand). | Evaluated via two-sided Welch's t-test |
| **Secondary** | **Cost per Rebalanced Bike** | Total intervention cost divided by net bikes redistributed. **Target: $<\$2.20$** (vs $\$5.80$ diesel van baseline). | Unit economic efficiency |
| **Guardrail** | **Full-Dock Lockout Rate** | % of peak minutes where destination docks have 0 open spaces. **Tolerance: $\le 2\%$ increase.** | Ensures rebalancing does not overwhelm destination docks |
| **Guardrail** | **Bike Maintenance Tickets** | Rate of reported mechanical defects per 1,000 rides. **Tolerance: No statistically significant rise.** | Monitors fleet wear and tear |

---

## 4. Demand Modeling & Operational Fleet Sizing

### 4.1 Methodology & Out-of-Time Holdout Split
Time-series models must never be evaluated with random shuffling (`train_test_split(shuffle=True)`), which causes severe look-ahead leakage. We enforce a strict **chronological out-of-time validation split**:
- **Training Set:** Jan 1, 2011 to Aug 31, 2012 (14,491 hours, 20 months).
- **Holdout Test Set:** Sep 1, 2012 to Dec 31, 2012 (2,888 hours, 4 months of mature system operation across autumn and winter).

### 4.2 Benchmark Model Results

| Model Architecture | Test RMSE | Test MAE | Test $R^2$ | Test MAPE | Operational Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Historical Stratified Mean** | 149.45 | 105.68 | 0.5286 | 52.7% | Heuristic baseline lookup (Workingday $\times$ Hour $\times$ Month) |
| **Regularized Ridge Regression** | 123.68 | 88.64 | 0.6772 | 193.9% | Linear baseline with cyclical & interaction features |
| **Random Forest (100 trees)** | 75.25 | 50.13 | 0.8805 | 37.8% | Non-linear ensemble baseline |
| **HistGradientBoosting (Total Demand)** | **65.34** | **43.30** | **0.9099** | **60.1%** | **Primary demand forecaster** |
| **Decomposed Model (Casual + Reg)** | 69.29 | 46.76 | 0.8987 | 70.6% | Segment-specific demand forecasting |

### 4.3 Why Point Forecasts Fail Operations: Quantile Fleet Sizing
In operations, predicting average demand ($\mathbb{E}[Y|X]$) leaves 50% of peak commuters stranded:
- If expected demand at 5:00 PM is 500 bikes, and dispatch stages 500 bikes, any demand fluctuation above the mean results in zero bikes at origin racks and angry riders.
- **The Solution:** We fit **Quantile Gradient Boosters** at $\alpha = 0.90$ and $\alpha = 0.95$ to calculate the required **operational safety buffer**:
  $$\text{Fleet Sizing Target}(t) = \max\left(\hat{Q}_{0.95}(t),\; \hat{D}(t) \times (1 + \text{Buffer})\right)$$

**Dynamic Operational Sizing Across the Day (Clear Weekday, 20°C):**
- **Morning Rush (8:00 AM):** Expected demand = **534 bikes/hr**; 95% staging requirement = **694 bikes** ($+160$ reserve buffer, $+30\%$).
- **Evening Peak (5:00 PM):** Expected demand = **690 bikes/hr**; 95% staging requirement = **897 bikes** ($+207$ reserve buffer, $+30\%$).
- **Midday Lull (1:00 PM):** Expected demand = **210 bikes/hr**; staging requirement = **242 bikes** ($+32$ buffer).
- **Overnight (2:00 AM):** Expected demand = **12 bikes/hr**; staging requirement = **14 bikes**. Over **95% of the fleet can be safely serviced or held in depots**.

---

## 5. Operations Assistant Prototype (`ops_assistant.py`)

To operationalize the models, we built a lightweight Natural Language dispatch assistant that field managers can run from their terminal or integrate into Slack/dispatch systems.

### 5.1 Architecture & Workflow
```
Plain English Query ──> [Regex NLP Parser] ──> [34-Feature Vector] ──> [ML Inference] ──> [Quantile Sizing] ──> [Dispatch Card]
```
1. **NLP Parsing:** Extracts target hour, day classification, weather category, and temperature from plain text.
2. **Feature Generation:** Translates inputs into normalized weather and cyclical temporal encodings.
3. **Dual Model Inference:** Runs casual, registered, and 95% quantile models.
4. **Coherent Breakdown:** Total demand is modeled as the additive sum of casual + registered trips, guaranteeing exact mathematical consistency ($96.1\% + 3.9\% = 100.0\%$).
5. **Actionable Dispatch Guidance:** Maps conditions to concrete field directives (e.g., origin pre-staging, transit hub clearances, maintenance windows).

### 5.2 Live Usage & Output Examples

#### Example 1: Morning Commute with Rain
```bash
python ops_assistant.py "How many bikes will we need tomorrow at 8 AM if it rains?"
```
```text
--------------------------------------------------------------------------------
DISPATCH DEMAND ADVISOR | Capital Bikeshare Operations
--------------------------------------------------------------------------------
Query: "How many bikes will we need tomorrow at 8 AM if it rains?"

Parsed Conditions:
  • Target Window:       08:00 (Morning Peak)
  • Day Type:            Weekday (Working Day)
  • Weather Forecast:    Light Rain / Snow (20.0°C, 60% hum, 12.0 km/h wind)

Demand Forecast:
  • Total Expected Trips:       490 rides/hr
    - Registered Commuters:     471 (96.1%)
    - Casual Users / Tourists:  19 (3.9%)

Fleet Staging & Buffer Targets:
  • Baseline Point Forecast:    490 bikes
  • 90% Service Level Target:   635 bikes (+145 buffer)
  • 95% Service Level Target:   671 bikes (+181 buffer / +36.9% reserve)

Operational Dispatch Guidance:
  Morning commute with precipitation. Commuters still ride (471 trips), but expect
  slower trip turnarounds. Pre-stage 671 bikes at residential origin docks by 07:15.
  Station rebalancing vans at major metro transfers to clear receiving docks as bikes arrive.
--------------------------------------------------------------------------------
```

#### Example 2: Sensitivity Comparison (Clear vs Rainy Evening Rush)
```bash
python ops_assistant.py "Compare demand on a clear working day vs a rainy working day at 5 PM"
```
```text
--------------------------------------------------------------------------------
SCENARIO COMPARISON & SENSITIVITY ANALYSIS
--------------------------------------------------------------------------------
Query: "Compare demand on a clear working day vs a rainy working day at 5 PM"

Scenario 1: "demand on a clear working day"
  • Setting:                 17:00 | Weekday | Clear | 20.0°C
  • Expected Rides:          698 trips/hr (Registered: 634, Casual: 64)
  • 95% Staging Target:      907 bikes

Scenario 2: "a rainy working day at 5 PM"
  • Setting:                 17:00 | Weekday | Rain/Snow | 20.0°C
  • Expected Rides:          513 trips/hr (Registered: 478, Casual: 35)
  • 95% Staging Target:      666 bikes

Net Impact (Scenario 2 vs Scenario 1):
  • Demand Variance:         -185 bikes/hr (-26.5%)
  • Fleet Buffer Variance:   -241 bikes

Dispatch Takeaway:
  Rain/cold sharply reduces demand. Scale back active van rebalancing runs and stage for lower turnover.
--------------------------------------------------------------------------------
```


## 6. How to Reproduce & Run the Pipeline

All scripts are configured to run seamlessly inside the local virtual environment (`.venv`):

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run data hygiene audit and feature correlation analysis
python eda_and_feature_engineering.py

# 3. Train all models, benchmark on out-of-time test set, and export artifacts
python modeling_and_evaluation.py

# 4. Generate all operational and analytical visualizations
python generate_visualizations.py

# 5. Query the Operations Assistant prototype
python ops_assistant.py "How many bikes will we need tomorrow at 8 AM if it rains?"

# 6. (Optional) Run interactive REPL mode for continuous Q&A
python ops_assistant.py --interactive
```


