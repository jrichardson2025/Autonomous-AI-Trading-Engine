# Autonomous AI Trading Engine: Multi-Asset Execution Pipeline

## 📌 Project Overview
This repository contains a production-ready algorithmic trading engine designed to autonomously harvest live market data, execute neural network inference, and manage complex order routing via the Alpaca API. 

Unlike standard research models, this project focuses heavily on **Machine Learning Engineering (MLE)** and live deployment mechanics. The engine is configured for multi-asset deployment, currently executing dynamically across traditional equities (**SPY**) and digital asset ETFs (**IBIT**), managing distinct volatility profiles simultaneously.

## 🏗️ Architectural Pipeline

### 1. Live Data Harvesting & Feature Engineering
* Utilizes the Alpaca SIP feed to aggregate high-frequency 1-minute bars into 5-minute structural periods across multiple ticker symbols.
* Dynamically calculates technical features in real-time to match the exact schema the Neural Network was trained on.

### 2. Deep Learning Inference (The Brain)
* Ingests the live feature state into a pre-trained Keras Deep Learning architecture.
* Outputs a probabilistic edge signal to trigger execution only when statistical thresholds align with localized low-volatility states.

### 3. Institutional Execution & Routing
* **Complex OTO Orders:** Submits One-Triggers-Other (OTO) market orders that natively link stop-losses to entry orders at the broker level, ensuring capital protection even in the event of a local server crash.
* **Latency Mitigation Loops:** Implements aggressive status-checking loops when canceling orders to prevent API race conditions (e.g., trying to sell shares that the broker hasn't verified as unlocked).

### 4. Dynamic Risk Management (The Ratchet)
* Replaces static stop-losses with an actively managed, ATR-driven (Average True Range) trailing stop system.
* Calculates local peak prices and autonomously ratchets the synthetic stop upward to lock in profits, eventually forcing a market exit when momentum decays.

## 🧰 Technology Stack
* **Language:** Python
* **Brokerage/API:** Alpaca Trade API (`alpaca-trade-api`)
* **AI/ML:** TensorFlow / Keras (Inference)
* **Execution:** `schedule` (Cron-style absolute clock syncing), `python-dotenv` (Cybersecurity/Key Management)

## ⚙️ Deployment Notes
This script is designed for live server deployment. It requires a `.env` file containing `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` for secure authentication. The pre-trained Keras models must be present in the root directory prior to engine initialization.
