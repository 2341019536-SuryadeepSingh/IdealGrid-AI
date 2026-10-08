# ⚡ IdealGrid AI

AI-powered regional electricity demand forecasting, anomaly intelligence, and analytical generation-balance insights.

## Important data-unit decision 

The regional generation dataset contains **daily generation values in MU/day**, while the demand dataset is **hourly demand in MW**. The dashboard therefore converts daily generation to a **daily-average MW equivalent**:

`average_MW = daily_generation_MU × 1000 / 24`

This makes the two datasets comparable at an analytical level without pretending that the daily generation number is an instantaneous hourly supply measurement.

### How the dashboard labels it

- **Avg Generation Equivalent (MW)** — daily generation converted to an average MW equivalent.
- **Generation coverage** — forecast demand divided by that average-generation benchmark.
- **Gap** — average-generation equivalent minus forecast demand.
- **Analytical balancing** — surplus/deficit recommendations are planning indicators, **not physical dispatch commands**.

## Dashboard

- **Historical Replay** automatically loads lag-24, lag-168, daily generation converted to average MW equivalent, and actual demand.
- **Future / Manual Scenario** lets you enter lag-24, lag-168, generation-average benchmark, and optional actual demand for all five regions.
- **Grid Optimization** suggests analytical surplus-to-deficit balancing based on the benchmark.
- **Model Performance** shows MAE, RMSE, R² and MAPE for the demand forecasting models.
- **Data Explorer** shows hourly demand against the generation-average benchmark.

## ML pipeline

### Demand forecasting

Five regional XGBoost regression models use:

- `lag_24` — demand 24 hours earlier
- `lag_168` — demand 168 hours earlier (7 days for hourly data)
- `hour`
- `sin_month`
- `cos_month`

The models use a chronological 80/20 train-test split.

### Anomaly analysis

When actual demand is available, the dashboard calculates the prediction residual and applies the trained Isolation Forest using:

- generation coverage / utilization ratio
- prediction residual
- generation-demand gap

An anomaly is therefore a **post-hoc diagnostic**; the system does not claim to know a future anomaly before actual demand is observed.

## Run

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The included model files are already trained. Run `python train.py` again only if you change the source datasets or model configuration.
