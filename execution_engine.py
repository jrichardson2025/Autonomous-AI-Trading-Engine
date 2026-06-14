import os
import time
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

import schedule
from alpaca_trade_api.rest import REST, TimeFrame
from keras.models import load_model
from keras.layers import Dense

# --- GOOGLE COLAB BUG FILTER ---
original_dense_init = Dense.__init__
def safe_dense_init(self, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, **kwargs)
Dense.__init__ = safe_dense_init
# -------------------------------

print("Initializing Institutional Master Engine (Stationary V2)...")

# ==========================================
# 1. ENVIRONMENT & API SETUP
# ==========================================
API_KEY = os.environ.get('APCA_API_KEY_ID')
API_SECRET = os.environ.get('APCA_API_SECRET_KEY')
api = REST(key_id=API_KEY, secret_key=API_SECRET, base_url='https://paper-api.alpaca.markets') 

# Load the trained V2 Neural Network AND the Scaler
try:
    spy_model = load_model('spy_reversion_model_v2.keras')
    spy_scaler = joblib.load('spy_scaler.pkl')
except Exception as e:
    print(f"FATAL BOOT ERROR: Could not load the V2 Brain or Scaler. Check your file names. Error: {e}")
    exit()

# ==========================================
# 2. GLOBAL PARAMETERS & THRESHOLDS
# ==========================================
# Sizing & AI Thresholds
SPY_POSITION_SIZE = 100000
SPY_SIGNAL_THRESHOLD = 0.55     
SPY_MAX_ATR_PCT = 0.003         # Volatility cap (~0.3%)

# The Dynamic Trailing Stop Multipliers
SURVIVAL_MULT = 2.5      
ACTIVATION_MULT = 1.0    
PROFIT_LOCK_MULT = 0.5   

# ==========================================
# 3. LIVE DATA HARVESTING (STATIONARY MATH)
# ==========================================
def fetch_5m_features(symbol, feed_type='sip'):
    # We pull 2000 minutes to ensure 100% accurate rolling moving averages
    bars = api.get_bars(symbol, TimeFrame.Minute, limit=2000, feed=feed_type).df
    df = bars.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    
    # Universal Base Techs
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_High'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- STATIONARY PROPORTIONS (V2 Brain exclusively reads these) ---
    df['Dist_SMA'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['ATR_Pct'] = df['ATR_14'] / df['Close']
    df['Pct_B'] = (df['Close'] - df['BB_Lower']) / (df['BB_High'] - df['BB_Lower'])

    return df.dropna()

# ==========================================
# 4. THE EXECUTION LOGIC
# ==========================================
def execute_engine(symbol):
    print(f"\n--- Running 5-Minute Cycle for {symbol} ---")
    
    try:
        position = api.get_position(symbol)
        if float(position.qty) > 0:
            open_orders = api.list_orders(status='open', symbols=[symbol])
            pending_sells = [o for o in open_orders if o.side == 'sell' and o.type == 'market']
            
            if pending_sells:
                print(f">>> BROKER LAG: Market Sell is pending for {position.qty} shares. Waiting for Alpaca to clear the trade.")
            else:
                print(f"Currently holding {position.qty} shares of {symbol}. Active Stop Loss is protecting capital.")
            return 
    except Exception:
        pass 

    df = fetch_5m_features(symbol, feed_type='sip')
    
    if df.empty or len(df) < 10:
        print(f"> Data feed thin for {symbol}. Skipping cycle.")
        return
        
    current_candle = df.iloc[-1]
    
    # 1. Extract the exact 10-candle sequence
    sequence = df[['Dist_SMA', 'ATR_Pct', 'Pct_B', 'RSI']].tail(10)
    
    # 2. Filter it through the Lens (Scaler) - .values prevents Pandas warnings
    scaled_sequence = spy_scaler.transform(sequence.values)
    features_input = np.array([scaled_sequence])
    
    # 3. Query the V2 Brain
    buy_prob = spy_model(features_input, training=False).numpy()[0][0]
    
    # 4. Live Sizing & Stop Math
    stop_distance = current_candle['ATR_14'] * SURVIVAL_MULT
    shares_to_buy = int(SPY_POSITION_SIZE / current_candle['Close'])  # Dynamically sizes to your $100k limit
    
    print(f"AI Signal: {buy_prob:.4f} | V2 Threshold: {SPY_SIGNAL_THRESHOLD:.4f} | ATR Pct: {current_candle['ATR_Pct']:.5f}")

    # 5. The Execution
    if buy_prob > SPY_SIGNAL_THRESHOLD and current_candle['ATR_Pct'] < SPY_MAX_ATR_PCT:
        # Calculate the stop price based on the current candle close
        initial_stop = round(current_candle['Close'] - stop_distance, 2)
        
        print(f">>> SIGNAL CONFIRMED. Submitting Complex OTO Order for {shares_to_buy} shares of {symbol}. Survival Stop: ${initial_stop}")
        
        try:
            # The OTO (One-Triggers-Other) bundles the buy and the stop-loss into one instant transaction
            api.submit_order(
                symbol=symbol,
                qty=shares_to_buy,
                side='buy',
                type='market',
                time_in_force='day',
                order_class='oto',
                stop_loss={'stop_price': initial_stop}
            )
            print(f">>> Phase 1 Active: Broker accepted OTO order. Shares and Stop Loss are natively linked.")
        except Exception as e:
            print(f">>> BROKER REJECTION: {e}")
            
    else:
        print("No signal or volatility too high. Standing by.")

def manage_open_positions():
    try:
        positions = api.list_positions()
        for pos in positions:
            if pos.symbol != 'SPY':
                continue
            symbol = pos.symbol
            qty = int(pos.qty)
            entry_price = float(pos.avg_entry_price)
            current_price = float(pos.current_price)
            
            # Use our institutional data fetcher for mathematically perfect ATR
            df = fetch_5m_features(symbol, feed_type='sip')
            if df.empty: continue
            current_atr = df.iloc[-1]['ATR_14']
            
            # Check the recent 1-minute tape for the absolute peak price
            recent_1m = api.get_bars(symbol, TimeFrame.Minute, limit=5, feed='sip').df
            peak_price = current_price
            if not recent_1m.empty:
                peak_price = max(current_price, float(recent_1m['high'].max()))
            
            activation_target = entry_price + (current_atr * ACTIVATION_MULT)
            hard_floor_trigger = entry_price + (current_atr * 1.5)
            desired_stop = 0.0
            
            if peak_price >= activation_target:
                synthetic_stop = peak_price - (current_atr * PROFIT_LOCK_MULT)
                desired_stop = max(desired_stop, synthetic_stop)
                
            if peak_price >= hard_floor_trigger:
                desired_stop = max(desired_stop, entry_price + (current_atr * 0.2))
            
            open_orders = api.list_orders(status='open', symbols=[symbol])
            stop_order = next((o for o in open_orders if o.side == 'sell' and o.type == 'stop'), None)
            
            if stop_order and desired_stop > 0:
                current_stop_price = float(stop_order.stop_price)
                if desired_stop > (current_stop_price + 0.02) and desired_stop < current_price:
                    new_stop = round(desired_stop, 2)
                    print(f">>> RATCHET ENGAGED: Upgrading {symbol} stop loss from ${current_stop_price} to ${new_stop}")
                    api.replace_order(order_id=stop_order.id, stop_price=new_stop)
                    
                elif desired_stop >= current_price:
                    print(f">>> FORCE EXIT: Closing position to lock gains.")
                    
                    # 1. Manually kill the stop order
                    api.cancel_order(stop_order.id)
                    
                    # 2. Interrogate the broker until the shares are verified free (up to 10 seconds)
                    shares_freed = False
                    for _ in range(10):
                        time.sleep(1)
                        # Ask Alpaca if the order is actually gone yet
                        check_orders = api.list_orders(status='open', symbols=[symbol])
                        if not check_orders: 
                            shares_freed = True
                            break
                            
                    # 3. Pull the trigger only when the broker confirms the shares are unlocked
                    if shares_freed:
                        api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                        print(f">>> FORCE EXIT COMPLETE: Position liquidated and gains secured.")
                    else:
                        print(f">>> FATAL BROKER LAG: Alpaca refused to release the shares in time. Aborting exit.")
                    
    except Exception as e:
        print(f"Position Management Error: {e}")

def run_market_cycle():
    clock = api.get_clock()
    if clock.is_open:
        manage_open_positions()
        time.sleep(3) 
        execute_engine('SPY')  # <--- THIS ONE SAYS 'SPY'
    else:
        print("Market is closed. Sleeping...")

# Force the script to map perfectly to Wall Street 5-minute intervals
intervals = [f"{i:02d}" for i in range(0, 60, 5)]
for minute_mark in intervals:
    schedule.every().hour.at(f"{minute_mark}:05").do(run_market_cycle)

print("\nEngines armed. Synced to absolute clock intervals. Waiting for the tape...")
while True:
    schedule.run_pending()
    time.sleep(1)