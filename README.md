# Real-Time High-Frequency Crypto Prediction & Meta-Labeling Engine

An event-driven, multi-threaded quantitative trading engine for **BTC/USDT** that ingests live tick-by-tick trade and Level-2 order book streams via Binance WebSockets. The system engineers real-time market microstructure features and deploys a two-stage **LightGBM** machine learning architecture (Primary Direction + Secondary Meta-Labeling Bet Sizing) with automated walk-forward retraining.

---

## System Architecture & Methodology

### 1. Market Microstructure & Feature Engineering
Raw tick data is aggregated into 1-second bars alongside live Level-2 depth snapshots to capture order flow toxicity and liquidity imbalances over a 300-second rolling window:

* **Volume-Synchronized Probability of Informed Trading (VPIN):** Partitions rolling trade volume into $N = 10$ equal volume buckets to measure order flow toxicity:
  $$\text{VPIN} = \frac{\sum_{\tau=1}^{N} \vert{}V_{\tau}^B - V_{\tau}^S\vert{}}{\sum_{\tau=1}^{N} (V_{\tau}^B + V_{\tau}^S)}$$
* **Order Flow Imbalance (OFI):** Tracks net liquidity changes at the best bid and best ask across sequential order book updates.
* **Weighted Average Price (WAP) & Order Book Imbalance (OBI):** Computes micro-price weighted by inverse top-of-book depth and normalizes total bid/ask book skewness via a 20-period rolling Z-score (`obi_z_score`).
* **Higher-Order Momentum Derivatives:** Computes 14-period RSI alongside its 1st derivative (speed, $\Delta \text{RSI}_t$) and 2nd derivative (acceleration, $\Delta^2 \text{RSI}_t$) over 5-second and 15-second horizons, plus an interaction term coupling 60-second realized volatility with 15-second RSI acceleration.
* **Prediction Entropy (Meta-Feature):** Tracks the Shannon entropy $H(p) = -\sum p_i \ln(p_i)$ of the model's out-of-fold and live probability distributions over a 50-prediction rolling window to quantify regime uncertainty.

### 2. Dynamic Quantile Labeling
Rather than relying on fixed return thresholds that break down across volatility regimes, 15-second forward returns ($r_{t, t+15}$) are dynamically partitioned using rolling historical quantiles:
* **UP (`1`):** $r_{t, t+15} > Q_{0.75}$
* **DOWN (`0`):** $r_{t, t+15} < Q_{0.25}$
* **NEUTRAL (`2`):** $Q_{0.25} \le r_{t, t+15} \le Q_{0.75}$

### 3. Two-Stage Meta-Labeling Pipeline
Inspired by López de Prado's meta-labeling framework, execution is decoupled into two distinct gradient-boosted decision tree (`LightGBM`) classifiers:
1. **Primary Directional Model:** A 3-class balanced classifier (`DOWN`, `UP`, `NEUTRAL`) trained with early stopping and 5-fold stratified cross-validation to generate out-of-fold probability entropy features.
2. **Secondary Bet-Sizing Model:** A binary classifier trained exclusively on actionable directional signals ($\hat{y} \neq 2$) to predict the probability that the primary model's directional call is correct ($P(y = \hat{y})$). Paper trades are only executed when both the primary model signals a directional move and the secondary meta-model confirms high conviction.

### 4. Asynchronous Live Execution & Walk-Forward Retraining
* **Concurrency:** Runs two daemon threads (`websocket-client`) consuming `@trade` and `@depth` streams concurrently, synchronized via thread locks (`threading.Lock`).
* **Online Walk-Forward Loop:** Warm-starts on historical tick data (4-hour rolling training window), executes live paper trading with 15-second forward evaluation for 15 minutes, appends newly collected live ticks to the historical buffer (pruned to a 24-hour rolling cap), and retrains both models in-place.

---

## Repository Structure

```text
├── src/
│   └── live_predictor.py      # Main execution engine, WebSocket handlers, and ML pipeline
├── data/
│   └── sample_trades.csv      # Sample schema for historical Binance aggTrades/trades CSV
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Data Setup
The engine expects a historical Binance raw trades CSV file (`BTCUSDT-trades-recent.csv`) in the working directory to bootstrap the initial 4-hour training window before transitioning to live WebSocket streams. The CSV expects standard Binance raw trade columns (no header):
`trade_id, price, quantity, quote_qty, timestamp (microseconds), is_buyer_the_maker, is_best_match`

### Running the Engine
```bash
python src/live_predictor.py
```

### Sample Live Output
During the live prediction phase, resolved predictions are evaluated every 2 seconds against their 15-second forward horizon and logged with real-time PnL and benchmark alpha tracking:
```text
 Pred: UP🔼       | Conf: 64.2% | Bet: ✅ | Accurate: ✅ | PnL: $   1.85 | Port: $100,012.40
 Pred: NEUTRAL🟡  | Conf: 51.0% | Bet: ⛔ | Accurate: ✅ | PnL: $   0.00 | Port: $100,012.40
 Pred: DOWN🔻     | Conf: 48.7% | Bet: ⛔ | Accurate: ❌ | PnL: $  -0.62 | Port: $100,012.40
```
