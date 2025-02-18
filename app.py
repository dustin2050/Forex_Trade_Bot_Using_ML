from flask import Flask, request, render_template, redirect, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import bcrypt
import time
import os
import threading

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.secret_key = 'secret_key'

db = SQLAlchemy(app)

# Load the trained model
model_path = os.path.join(os.getcwd(), "model", "xgb_model.pkl")
model = joblib.load(model_path)

# MetaTrader 5 Credentials
MT5_LOGIN = ---------
MT5_PASSWORD = "----------"
MT5_SERVER = "MetaQuotes-Demo"

# Database model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

# Connect to MetaTrader 5
def connect_to_mt5():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"Failed to connect to MetaTrader 5: {mt5.last_error()}")
        return False
    print("Connected to MetaTrader 5")
    return True

# Fetch real-time data
def fetch_data(symbol, timeframe, bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"Error fetching data for {symbol}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Technical Indicator Calculations
def calculate_ema(df, period, column="close"):
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_bbands(df, period=20):
    sma = df['close'].rolling(window=period).mean()
    stddev = df['close'].rolling(window=period).std()
    upper_band = sma + (2 * stddev)
    lower_band = sma - (2 * stddev)
    return pd.DataFrame({"BB_MIDDLE": sma, "BB_UPPER": upper_band, "BB_LOWER": lower_band})

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = calculate_ema(df, fast_period)
    slow_ema = calculate_ema(df, slow_period)
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    return pd.DataFrame({"MACD": macd, "SIGNAL_LINE": signal_line})

def calculate_rsi(df, period=14, column="close"):
    delta = df[column].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Generate features for prediction
def prepare_features(df):
    df["9EMA"] = calculate_ema(df, 9).bfill()
    df["20EMA"] = calculate_ema(df, 20).bfill()
    df["50EMA"] = calculate_ema(df, 50).bfill()
    df["200SMA"] = df["close"].rolling(window=200).mean().bfill()
    df["ATR"] = calculate_atr(df).bfill()
    bbands_df = calculate_bbands(df)
    macd_df = calculate_macd(df)
    df["RSI"] = calculate_rsi(df).bfill()
    df = pd.concat([df, bbands_df, macd_df], axis=1)
    df["Bollinger_Bands_Below_Lower_BB"] = np.where(df["close"] < df["BB_LOWER"], 1, 0)
    df["Bollinger_Bands_Above_Upper_BB"] = np.where(df["close"] > df["BB_UPPER"], 1, 0)
    df['9EMA_above_20EMA'] = np.where(df['9EMA'] > df['20EMA'], 1, 0)
    df['9EMA_cross_20EMA'] = df['9EMA_above_20EMA'].diff().fillna(0)
    df['50EMA_above_200SMA'] = np.where(df['50EMA'] > df['200SMA'], 1, 0)
    df['50EMA_cross_200SMA'] = df['50EMA_above_200SMA'].diff().fillna(0)
    return df

# Make predictions
def make_prediction(df):
    df = prepare_features(df)
    X_live = df[["9EMA", "20EMA", "50EMA", "200SMA", "ATR", "RSI", "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "MACD",
                 "Bollinger_Bands_Below_Lower_BB", "Bollinger_Bands_Above_Upper_BB", "9EMA_above_20EMA",
                 "9EMA_cross_20EMA", "50EMA_above_200SMA", "50EMA_cross_200SMA"]].tail(1)
    print(f"Features for Prediction: {X_live}")
    prediction = model.predict(X_live)
    print(f"Prediction: {prediction[0]}")
    return prediction[0]

# List to store trade results
trade_results = []

def execute_trade(symbol, trade_type, volume, stop_loss, take_profit):
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"Error: Symbol info for {symbol} not available")
        return
    point = symbol_info.point
    price = mt5.symbol_info_tick(symbol).ask if trade_type == "buy" else mt5.symbol_info_tick(symbol).bid
    stop_loss = round(price - stop_loss if trade_type == "buy" else price + stop_loss, 5)
    take_profit = round(price + take_profit if trade_type == "buy" else price - take_profit, 5)
    min_stop_dist = symbol_info.trade_stops_level * point
    if abs(price - stop_loss) < min_stop_dist or abs(price - take_profit) < min_stop_dist:
        print(f"Error: SL or TP does not meet minimum stop distance ({min_stop_dist}).")
        return
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if trade_type == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 20,
        "magic": 123456,
        "comment": "Trade executed via bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    trade_result = {
        "symbol": symbol,
        "type": trade_type,
        "volume": volume,
        "price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "status": "Success" if result.retcode == mt5.TRADE_RETCODE_DONE else f"Failed: {result.comment}"
    }
    trade_results.append(trade_result)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Trade failed: {result.comment}")
    else:
        print(f"Trade successful: {result}")


# Calculate stop loss and take profit
def calculate_sl_tp(entry_price, atr, risk=1, reward=3, symbol="EURUSD"):
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return None, None
    point = symbol_info.point
    min_stop_dist = max(symbol_info.trade_stops_level * point, 10 * point)
    stop_loss = max(risk * atr, min_stop_dist)
    take_profit = max(reward * atr, min_stop_dist)
    return round(stop_loss, 5), round(take_profit, 5)

# Run bot
bot_thread = None

def run_bot():
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M1
    volume = 0.1
    if not connect_to_mt5():
        return
    try:
        while True:
            df = fetch_data(symbol, timeframe)
            if df.empty:
                continue
            prediction = make_prediction(df)
            entry_price = df["close"].iloc[-1]
            atr = df["ATR"].iloc[-1]
            stop_loss, take_profit = calculate_sl_tp(entry_price, atr, symbol=symbol)
            if stop_loss is None or take_profit is None:
                continue
            if prediction == 1:
                execute_trade(symbol, "buy", volume, stop_loss, take_profit)
            elif prediction == 0:
                execute_trade(symbol, "sell", volume, stop_loss, take_profit)
            time.sleep(60)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        mt5.shutdown()
@app.route('/bot')
def bot():
    return render_template('bot.html', trade_results=trade_results)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')



@app.route('/start-bot', methods=['POST'])
def start_bot():
    global bot_thread
    if bot_thread is None or not bot_thread.is_alive():
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        return jsonify({"status": "Bot started successfully."})
    else:
        return jsonify({"status": "Bot is already running."})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid user')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html', user=user)
    
    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)


