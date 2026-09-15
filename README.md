# HMM Market Regime Detection

A Hidden Markov Model, built from scratch that detects
Bull/Bear regimes in S&P 500 data and backtests a simple trading strategy
based on those regimes.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python test_hmm_synthetic.py   
python main.py  
```

## API service

Install the package and start the API locally:

```bash
pip install -e ".[dev]"
uvicorn api.main:app --reload
```

Create a model before calling prediction endpoints:

```bash
PYTHONPATH=src:. python jobs/train.py --model-dir models
```

The service exposes `/api/v1/health`, `/api/v1/ready`, `/api/v1/model`,
`/api/v1/predict`, `/api/v1/predict/batch`, and `/api/v1/drift/check`.
Prediction calls use the loaded model parameters. They update only the real-time
filtered probability state; retraining is an explicit scheduled operation.

For Redis Streams ingestion, publish a JSON payload such as
`{"features": [0.01, 0.02], "timestamp": "2026-09-15T12:00:00Z"}` to
`market-observations`. Set `REDIS_URL` before starting the API. The scheduler
can also be run directly with `PYTHONPATH=src:. python jobs/scheduler.py`.
Prometheus metrics are available at `/metrics` and Prometheus runs on port 9090
in Compose.

Run the API and Redis with Docker:

```bash
docker compose up --build
```

## What it does

1. Trains the HMM on data before 2018
2. Tests it on 2018–2024 data it never saw
3. Backtests a simple strategy: hold stocks when the model says "Bull", stay out when it says "Bear"
4. Compares that strategy to just buying and holding
5. Saves 5 plots to `./plots/`
