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

## What it does

1. Trains the HMM on data before 2018
2. Tests it on 2018–2024 data it never saw
3. Backtests a simple strategy: hold stocks when the model says "Bull", stay out when it says "Bear"
4. Compares that strategy to just buying and holding
5. Saves 5 plots to `./plots/`
