import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import logging, warnings, time, datetime, json, sys
from collections import deque
from threading import Thread, Lock
import websocket
from scipy.stats import entropy

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- START: ALL HELPER FUNCTIONS ---

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    return 100 - (100 / (1 + rs))

def _vpin_calculation_on_window(window_df):
    total_volume = window_df['quantity'].sum()
    if total_volume == 0: return 0.5
    num_buckets = 10
    volume_per_bucket = total_volume / num_buckets
    if volume_per_bucket == 0: return 0.5
    signed_volume = window_df['quantity'] * np.where(window_df['is_buyer_the_maker'], -1, 1)
    cum_volume = window_df['quantity'].cumsum()
    bucket_boundaries = np.linspace(0, total_volume, num_buckets + 1)
    bucket_assignments = np.searchsorted(bucket_boundaries, cum_volume, side='right')
    bucket_imbalance = signed_volume.groupby(bucket_assignments).sum()
    if len(bucket_imbalance) == 0: return 0.5
    vpin = bucket_imbalance.abs().sum() / total_volume
    return vpin

def add_feature_derivatives(df, feature_name, periods=[5, 15]):
    if feature_name in df.columns:
        for p in periods:
            df[f'{feature_name}_spd_{p}s'] = df[feature_name].diff(p)
            df[f'{feature_name}_acc_{p}s'] = df[f'{feature_name}_spd_{p}s'].diff(p)

# --- END: HELPER FUNCTIONS ---

class MetaFeatureTracker:
    def __init__(self, maxlen=1000):
        self.predictions = deque(maxlen=maxlen)
    
    def add_resolved_prediction(self, proba_dist):
        self.predictions.append({'entropy': entropy(proba_dist + 1e-9)})

    def get_features(self):
        if len(self.predictions) < 50: return {'prediction_entropy': 1.0}
        data = pd.DataFrame(list(self.predictions))
        rolling_entropy = data['entropy'].rolling(window=50, min_periods=20).mean().iloc[-1]
        return {'prediction_entropy': rolling_entropy if pd.notna(rolling_entropy) else 1.0}
    
    def reset(self):
        self.predictions.clear()


class AdvancedPredictor:
    def __init__(self, historical_data_file='BTCUSDT-trades-recent.csv'):
        self.symbol = 'BTCUSDT'
        self.historical_data_file = historical_data_file
        
        self.live_data = deque(maxlen=5000)
        self.order_book_features_live = deque(maxlen=5000)
        self.data_lock = Lock()
        self.bids, self.asks = {}, {}
        
        self.prev_best_bid, self.prev_best_ask = 0, 0

        self.direction_model, self.scaler = None, StandardScaler()
        self.bet_sizing_model, self.bet_scaler = None, StandardScaler()
        
        self.feature_names = []
        
        self.last_price, self.is_running = 0, False
        self.ws_trades, self.ws_trades_thread = None, None
        self.ws_depth, self.ws_depth_thread = None, None
        
        self.TRAINING_DATA_WINDOW_HOURS = 4
        self.PREDICTION_HORIZON_SECONDS = 15
        self.RETRAIN_INTERVAL_MINUTES = 15
        self.FEATURE_WINDOW_SECONDS = 300
        
        self.upper_quantile, self.lower_quantile = 0.75, 0.25

        self.initial_portfolio, self.current_portfolio = 100000.0, 100000.0
        self.TRADING_FEE_PCT, self.FIXED_TRADE_VALUE = 0.0, 1000
        self.pending_predictions = deque()
        self.output_log_queue = deque()
        self.resolved_predictions_log = []
        self.last_prediction_time = 0
        self.is_model_ready = False
        self.is_predicting_enabled = False
        self.meta_tracker = MetaFeatureTracker()
        self.label_thresholds = {'upper': 0, 'lower': 0}

    def reset_session_state(self):
        self.pending_predictions.clear()
        self.output_log_queue.clear()
        self.resolved_predictions_log.clear()
        self.meta_tracker.reset()

    def load_historical_data(self):
        try:
            logger.info(f"🔄 Loading historical data from '{self.historical_data_file}'...")
            dtypes = {'price': 'float64', 'quantity': 'float64', 'timestamp': 'int64', 'is_buyer_the_maker': 'bool'}
            df = pd.read_csv(self.historical_data_file, header=None,
                             names=['trade_id', 'price', 'quantity', 'quote_qty', 'timestamp', 'is_buyer_the_maker', 'is_best_match'],
                             usecols=dtypes.keys(), dtype=dtypes)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
            resampled = df.set_index('timestamp').resample('1S').agg({
                'price': 'ohlc', 'quantity': 'sum', 'is_buyer_the_maker': lambda x: x.mode()[0] if not x.empty else False
            }).ffill()
            resampled.columns = ['open', 'high', 'low', 'close', 'quantity', 'is_buyer_the_maker']
            resampled.reset_index(inplace=True)
            logger.info(f"🚀 Loaded and resampled to {len(resampled)} data points.")
            return resampled
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            return None
    
    def _generate_historical_meta_features(self, X, y, base_features_df):
        logger.info("...generating historical prediction entropy using cross-validation...")
        n_splits = 5
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2 or np.min(counts) < n_splits:
            logger.warning(f"Too few samples in a class for meta-features. Counts: {counts}. Skipping.")
            return pd.DataFrame({'prediction_entropy': 1.0}, index=base_features_df.index)

        oof_probas = np.full((len(y), 3), np.nan, dtype=float)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val, y_train_fold = X[train_idx], X[val_idx], y[train_idx]
            temp_scaler = StandardScaler().fit(X_train)
            X_train_s, X_val_s = temp_scaler.transform(X_train), temp_scaler.transform(X_val)
            temp_model = lgb.LGBMClassifier(objective='multiclass', class_weight='balanced', verbosity=-1)
            temp_model.fit(X_train_s, y_train_fold)
            oof_probas[val_idx] = temp_model.predict_proba(X_val_s)

        prediction_entropy = [entropy(p + 1e-9) for p in oof_probas]
        return pd.DataFrame({'prediction_entropy': prediction_entropy}, index=base_features_df.index).fillna(method='ffill').fillna(0)

    def _calculate_order_book_features(self):
        # This function NO LONGER acquires a lock. The caller is responsible.
        if not self.bids or not self.asks: return None
        best_bid_price, best_ask_price = max(self.bids.keys()), min(self.asks.keys())
        if best_bid_price >= best_ask_price: return None
        
        best_bid_vol, best_ask_vol = self.bids[best_bid_price], self.asks[best_ask_price]
        if (best_bid_vol + best_ask_vol) == 0: return None
        
        wap = (best_bid_price * best_ask_vol + best_ask_price * best_bid_vol) / (best_bid_vol + best_ask_vol)
        obi = (sum(self.bids.values()) - sum(self.asks.values())) / (sum(self.bids.values()) + sum(self.asks.values()))
        
        ofi = 0
        if self.prev_best_bid > 0 and self.prev_best_ask > 0:
            if best_bid_price < self.prev_best_bid: ofi = -self.bids.get(self.prev_best_bid, 0)
            elif best_bid_price > self.prev_best_bid: ofi = best_bid_vol
            else: ofi = best_bid_vol - self.bids.get(self.prev_best_bid, 0)
            
            if best_ask_price > self.prev_best_ask: ofi += self.asks.get(self.prev_best_ask, 0)
            elif best_ask_price < self.prev_best_ask: ofi -= best_ask_vol
            else: ofi += self.asks.get(self.prev_best_ask, 0) - best_ask_vol
            
        self.prev_best_bid, self.prev_best_ask = best_bid_price, best_ask_price
        
        return {'wap': wap, 'obi': obi, 'spread': best_ask_price - best_bid_price, 'ofi': ofi}

    def prepare_training_data(self, df_history):
        logger.info("🛠️ Preparing training data with QUANTILE labeling...")
        df = df_history.copy().set_index('timestamp')

        logger.info("Calculating base features...")
        df['log_returns'] = np.log(df['close']).diff()
        df['rsi_14'] = calculate_rsi(df['close'], period=14)
        df['realized_volatility_60s'] = df['log_returns'].rolling('60s').std()
        add_feature_derivatives(df, 'rsi_14', periods=[5, 15])
        df['vol_x_rsi_accel'] = df['realized_volatility_60s'] * df['rsi_14_acc_15s']
        
        logger.info("...calculating VPIN...")
        vpin_values = [_vpin_calculation_on_window(df.iloc[i - self.FEATURE_WINDOW_SECONDS + 1 : i + 1]) if i >= self.FEATURE_WINDOW_SECONDS - 1 else np.nan for i in range(len(df))]
        df['vpin'] = vpin_values
        
        base_feature_names = [
            'realized_volatility_60s', 'rsi_14', 'rsi_14_spd_5s', 'rsi_14_acc_5s',
            'rsi_14_spd_15s', 'rsi_14_acc_15s', 'vpin', 'vol_x_rsi_accel'
        ]
        
        logger.info("Calculating labels with return quantiles...")
        horizon_returns = df['close'].pct_change(self.PREDICTION_HORIZON_SECONDS).shift(-self.PREDICTION_HORIZON_SECONDS)
        df['horizon_returns'] = horizon_returns
        
        df.dropna(subset=base_feature_names + ['horizon_returns'], inplace=True)
        
        self.label_thresholds['upper'] = df['horizon_returns'].quantile(self.upper_quantile)
        self.label_thresholds['lower'] = df['horizon_returns'].quantile(self.lower_quantile)
        logger.info(f"Label thresholds determined: Lower={self.label_thresholds['lower']:.6f}, Upper={self.label_thresholds['upper']:.6f}")
        
        conditions = [
            df['horizon_returns'] > self.label_thresholds['upper'],
            df['horizon_returns'] < self.label_thresholds['lower']
        ]
        choices = [1, 0]
        df['label'] = np.select(conditions, choices, default=2)
        
        features_df = df[base_feature_names]
        X_base, y_base = features_df.values, df['label'].values.astype(int)

        meta_features_df = self._generate_historical_meta_features(X_base, y_base, features_df)
        
        new_features = ['wap', 'obi', 'spread', 'ofi', 'trade_imbalance', 'obi_z_score']
        for feature in new_features: meta_features_df[feature] = 0.0

        self.feature_names = base_feature_names + list(meta_features_df.columns)
        
        full_features_df = pd.concat([features_df, meta_features_df], axis=1).dropna()
        X_final, y_final = full_features_df[self.feature_names].values, df.loc[full_features_df.index, 'label'].values
        
        logger.info(f"Created {len(X_final)} training samples with {len(self.feature_names)} total features.")
        return X_final, y_final.astype(int)

    def train_primary_model(self, X_train, y_train, X_test, y_test):
        logger.info("🤖 Training primary model...")
        self.scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = self.scaler.transform(X_train), self.scaler.transform(X_test)
        
        self.direction_model = lgb.LGBMClassifier(
            objective='multiclass', class_weight='balanced', n_estimators=300, 
            learning_rate=0.05, num_leaves=31, random_state=42, 
            colsample_bytree=0.8, subsample=0.8, verbosity=-1
        )
        
        self.direction_model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)],
                                 callbacks=[lgb.early_stopping(25, verbose=False)])
        
        y_pred_labels = self.direction_model.predict(X_test_s)
        logger.info(f"✅ Primary model trained. Accuracy: {accuracy_score(y_test, y_pred_labels):.2%}")
        logger.info("\n" + classification_report(y_test, y_pred_labels, zero_division=0, digits=3, target_names=['DOWN', 'UP', 'NEUTRAL']))
        return self.direction_model, self.scaler

    def train_bet_sizing_model(self, X_primary_test, y_primary_test):
        logger.info("🤖 Training secondary bet-sizing model...")
        if self.direction_model is None: return

        primary_preds = self.direction_model.predict(self.scaler.transform(X_primary_test))
        trade_mask = (primary_preds != 2)
        if not np.any(trade_mask):
            logger.warning("No trade predictions to train bet-sizing model.")
            return

        secondary_X = X_primary_test[trade_mask]
        secondary_y = (primary_preds[trade_mask] == y_primary_test[trade_mask]).astype(int)

        unique_classes, counts = np.unique(secondary_y, return_counts=True)
        if len(unique_classes) < 2 or np.min(counts) < 2:
            logger.warning(f"Not enough samples for bet-sizing model. Counts: {counts}. Skipping.")
            self.bet_sizing_model = None
            return

        X_train_sec, X_test_sec, y_train_sec, y_test_sec = train_test_split(secondary_X, secondary_y, test_size=0.3, stratify=secondary_y)
        self.bet_scaler = StandardScaler().fit(X_train_sec)
        
        self.bet_sizing_model = lgb.LGBMClassifier(objective='binary', class_weight='balanced', verbosity=-1)
        self.bet_sizing_model.fit(self.bet_scaler.transform(X_train_sec), y_train_sec)
        
        auc = roc_auc_score(y_test_sec, self.bet_sizing_model.predict_proba(self.bet_scaler.transform(X_test_sec))[:, 1])
        logger.info(f"✅ Bet-sizing model trained. Test AUC: {auc:.3f}")

    def run_training_pipeline(self, df_history):
        training_window_df = df_history.tail(self.TRAINING_DATA_WINDOW_HOURS * 3600).copy()
        logger.info(f"Using a rolling window of {len(training_window_df)} data points for training.")

        X, y = self.prepare_training_data(training_window_df)
        if len(X) < 500:
            logger.error(f"Failed to generate enough training samples ({len(X)}). Need at least 500.")
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        
        self.train_primary_model(X_train, y_train, X_test, y_test)
        self.train_bet_sizing_model(X_test, y_test)
        
        self.is_model_ready = True
        return True

    def create_features(self, df_window):
        if len(df_window) < self.FEATURE_WINDOW_SECONDS: return None, None
        
        df = df_window.copy().set_index('timestamp')
        
        # --- Create all features on the fly ---
        df['log_returns'] = np.log(df['close']).diff()
        df['rsi_14'] = calculate_rsi(df['close'], period=14)
        df['realized_volatility_60s'] = df['log_returns'].rolling('60s').std()
        add_feature_derivatives(df, 'rsi_14', periods=[5, 15])
        df['vpin'] = _vpin_calculation_on_window(df)
        df['vol_x_rsi_accel'] = df['realized_volatility_60s'] * df['rsi_14_acc_15s']
        
        live_meta_features = self.meta_tracker.get_features()
        for key, value in live_meta_features.items(): df[key] = value

        with self.data_lock:
            order_book_features = self._calculate_order_book_features()
        if order_book_features:
            for key, value in order_book_features.items(): df[key] = value
        
        buy_vol = df['quantity'][df['is_buyer_the_maker'] == False].sum()
        sell_vol = df['quantity'][df['is_buyer_the_maker'] == True].sum()
        df['trade_imbalance'] = (buy_vol - sell_vol) / (buy_vol + sell_vol) if (buy_vol + sell_vol) > 0 else 0
        
        if len(self.order_book_features_live) > 20:
            ob_df = pd.DataFrame(list(self.order_book_features_live))
            obi_mean = ob_df['obi'].rolling(window=20).mean().iloc[-1]
            obi_std = ob_df['obi'].rolling(window=20).std().iloc[-1]
            df['obi_z_score'] = (df['obi'] - obi_mean) / (obi_std + 1e-9)
        else:
             df['obi_z_score'] = 0.0

        if not self.feature_names: return None, None
        for col in self.feature_names:
            if col not in df.columns: df[col] = 0

        features = df[self.feature_names].iloc[-1].fillna(0).values.reshape(1, -1)
        return features, df.index[-1]

    def process_new_prediction(self):
        with self.data_lock:
            if len(self.live_data) < self.FEATURE_WINDOW_SECONDS: return
            live_df = pd.DataFrame(list(self.live_data))
        
        features, timestamp = self.create_features(live_df)
        if features is None: return

        scaled_features = self.scaler.transform(features)
        pred_proba = self.direction_model.predict_proba(scaled_features)[0]
        pred_num = np.argmax(pred_proba)

        place_bet = False
        if pred_num != 2 and self.bet_sizing_model:
            bet_features_scaled = self.bet_scaler.transform(features)
            bet_prediction = self.bet_sizing_model.predict(bet_features_scaled)[0]
            if bet_prediction == 1: place_bet = True

        self.pending_predictions.append({
            'pred_time': timestamp, 'eval_time': datetime.datetime.now() + datetime.timedelta(seconds=self.PREDICTION_HORIZON_SECONDS),
            'pred_price': self.last_price, 'prediction': pred_num, 'confidence': np.max(pred_proba), 
            'place_bet': place_bet, 'proba_dist': pred_proba
        })

    def evaluate_resolved_predictions(self):
        now = datetime.datetime.now()
        if not self.pending_predictions or self.pending_predictions[0]['eval_time'] > now: return

        resolved = self.pending_predictions.popleft()
        price_change_pct = (self.last_price - resolved['pred_price']) / resolved['pred_price']
        
        actual_direction = 1 if price_change_pct > self.label_thresholds['upper'] else (0 if price_change_pct < self.label_thresholds['lower'] else 2)
        is_correct = resolved['prediction'] == actual_direction
        
        pnl_direction = 0
        if resolved['prediction'] == 1: pnl_direction = 1
        elif resolved['prediction'] == 0: pnl_direction = -1
        
        pnl = 0.0
        if pnl_direction != 0:
            pnl = price_change_pct * self.FIXED_TRADE_VALUE * pnl_direction
            pnl -= self.TRADING_FEE_PCT / 100 * self.FIXED_TRADE_VALUE
        
        is_profitable = pnl > 0

        if resolved['place_bet'] and pnl_direction != 0:
            self.current_portfolio += pnl
        
        if 'proba_dist' in resolved: self.meta_tracker.add_resolved_prediction(resolved['proba_dist'])
            
        log_entry = {'prediction_num': resolved['prediction'], 'is_correct': is_correct, 'pnl': pnl, 
                     'placed_bet': resolved['place_bet'], 'pred_price': resolved['pred_price'], 'is_profitable': is_profitable}
        self.resolved_predictions_log.append(log_entry)

        pred_labels = {0: 'DOWN🔻', 1: 'UP🔼', 2: 'NEUTRAL🟡'}
        bet_icon = "✅" if resolved['place_bet'] else "⛔"
        accuracy_icon = "✅" if is_correct else "❌"

        log_msg = (f"[{resolved['pred_time']:%H:%M:%S}] Pred: {pred_labels[resolved['prediction']]:<10} | "
                   f"Conf: {resolved['confidence']:.1%} | Bet: {bet_icon} | "
                   f"Accurate: {accuracy_icon} | PnL: ${pnl: >7.2f} | Port: ${self.current_portfolio:,.2f}")
        self.output_log_queue.append(log_msg)

    def print_final_summary(self):
        print("\n" + "=" * 80 + "\nSESSION FINISHED: PERFORMANCE SUMMARY\n" + "=" * 80)
        net_pnl = self.current_portfolio - self.initial_portfolio
        print(f"Initial Portfolio: ${self.initial_portfolio:,.2f}\nFinal Portfolio  : ${self.current_portfolio:,.2f}")
        strategy_return = (net_pnl / self.initial_portfolio) * 100 if self.initial_portfolio > 0 else 0
        print(f"Strategy Return  : {strategy_return:.4f}%")
        
        if self.resolved_predictions_log:
            first_log = self.resolved_predictions_log[0]
            if 'pred_price' in first_log and first_log['pred_price'] > 0 and self.last_price > 0:
                 start_price = first_log['pred_price']
                 market_return = ((self.last_price - start_price) / start_price) * 100
                 print(f"Buy & Hold Return: {market_return:.4f}% (Benchmark)")
                 if self.initial_portfolio > 0: print(f"Alpha (Outperformance): {(strategy_return - market_return):.4f}%")
        print("-" * 80)
        if not self.resolved_predictions_log:
            print("No predictions resolved during this session."); return

        logs = self.resolved_predictions_log
        
        print("--- Label-Based Accuracy Breakdown (vs. Quantile Thresholds) ---")
        total_preds, total_correct = len(logs), sum(l['is_correct'] for l in logs)
        if total_preds > 0: print(f"Overall Accuracy : {total_correct / total_preds:.2%} ({total_correct}/{total_preds})")
        bet_logs = [l for l in logs if l['placed_bet']]
        if bet_logs: print(f"Bet Accuracy     : {sum(l['is_correct'] for l in bet_logs) / len(bet_logs):.2%} ({sum(l['is_correct'] for l in bet_logs)}/{len(bet_logs)})")
        
        for pred_num, label in {0: 'DOWN 🔻', 1: 'UP 🔼', 2: 'NEUTRAL 🟡'}.items():
            group = [l['is_correct'] for l in logs if l['prediction_num'] == pred_num]
            if group: print(f"  - {label} Accuracy : {sum(group) / len(group):.2%} ({sum(group)}/{len(group)})")
            else: print(f"  - {label} Accuracy : N/A (0 predictions)")
        print("-" * 80)

        print("--- Profitability-Based Accuracy Breakdown (vs. PnL > 0) ---")
        total_profitable = sum(l['is_profitable'] for l in logs if l['prediction_num'] != 2)
        total_directional_preds = sum(1 for l in logs if l['prediction_num'] != 2)
        if total_directional_preds > 0: print(f"Overall Profitability : {total_profitable / total_directional_preds:.2%} ({total_profitable}/{total_directional_preds})")
        
        bet_profitable_logs = [l for l in bet_logs if l['prediction_num'] != 2]
        if bet_profitable_logs: print(f"Bet Profitability     : {sum(l['is_profitable'] for l in bet_profitable_logs) / len(bet_profitable_logs):.2%} ({sum(l['is_profitable'] for l in bet_profitable_logs)}/{len(bet_profitable_logs)})")

        for pred_num, label in {0: 'DOWN 🔻', 1: 'UP 🔼'}.items():
            group = [l['is_profitable'] for l in logs if l['prediction_num'] == pred_num]
            if group: print(f"  - {label} Profitability : {sum(group) / len(group):.2%} ({sum(group)}/{len(group)})")
            else: print(f"  - {label} Profitability : N/A (0 predictions)")
        print("=" * 80)

    def start_live_feed(self):
        self.is_running = True
        
        def on_message(ws, message_type, message):
            try:
                data = json.loads(message)
                with self.data_lock:
                    if message_type == 'trade':
                        if data.get('e') != 'trade': return
                        self.last_price = float(data['p'])
                        ts = datetime.datetime.fromtimestamp(data['T'] / 1000).replace(microsecond=0)
                        if not self.live_data or self.live_data[-1]['timestamp'] < ts:
                            self.live_data.append({'timestamp': ts, 'open': self.last_price, 'high': self.last_price, 'low': self.last_price, 
                                                   'close': self.last_price, 'quantity': float(data['q']), 'is_buyer_the_maker': data['m']})
                        else:
                            self.live_data[-1].update({'high': max(self.live_data[-1]['high'], self.last_price), 'low': min(self.live_data[-1]['low'], self.last_price),
                                                       'close': self.last_price, 'quantity': self.live_data[-1]['quantity'] + float(data['q'])})
                    
                    elif message_type == 'depth':
                        for bid in data.get('b', []):
                            price, qty = float(bid[0]), float(bid[1])
                            if qty == 0: self.bids.pop(price, None)
                            else: self.bids[price] = qty
                        for ask in data.get('a', []):
                            price, qty = float(ask[0]), float(ask[1])
                            if qty == 0: self.asks.pop(price, None)
                            else: self.asks[price] = qty
                        ob_features = self._calculate_order_book_features()
                        if ob_features: self.order_book_features_live.append(ob_features)

                if message_type == 'trade' and self.is_model_ready and self.is_predicting_enabled:
                    current_time = time.time()
                    if current_time - self.last_prediction_time >= 2.0:
                        self.last_prediction_time = current_time
                        self.process_new_prediction()
                    self.evaluate_resolved_predictions()
            except Exception as e:
                logger.error(f"Error in on_message ({message_type}): {e}", exc_info=True)

        def on_error(ws, error): logger.error(f"WebSocket Error for {ws.url}: {error}")
        def on_close(ws, code, msg): logger.warning(f"WebSocket Closed for {ws.url}: Code={code}, Msg={msg}")
        
        trade_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@trade"
        self.ws_trades = websocket.WebSocketApp(trade_url, on_message=lambda ws, msg: on_message(ws, 'trade', msg), on_open=lambda ws: logger.info("Trade WebSocket connected."), on_error=on_error, on_close=on_close)
        self.ws_trades_thread = Thread(target=self.ws_trades.run_forever, daemon=True)
        self.ws_trades_thread.start()

        depth_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@depth"
        self.ws_depth = websocket.WebSocketApp(depth_url, on_message=lambda ws, msg: on_message(ws, 'depth', msg), on_open=lambda ws: logger.info("Order Book WebSocket connected."), on_error=on_error, on_close=on_close)
        self.ws_depth_thread = Thread(target=self.ws_depth.run_forever, daemon=True)
        self.ws_depth_thread.start()

    def stop(self):
        self.is_running = False
        if self.ws_trades: self.ws_trades.close()
        if self.ws_depth: self.ws_depth.close()

if __name__ == '__main__':
    predictor = AdvancedPredictor()
    try:
        historical_data = predictor.load_historical_data()
        if historical_data is None: sys.exit(1)

        logger.info("Starting data feeds for initial collection...")
        predictor.start_live_feed()
        
        warmup_seconds = predictor.FEATURE_WINDOW_SECONDS + 10
        logger.info(f"Entering {warmup_seconds}-second warm-up period to collect initial data...")
        time.sleep(warmup_seconds)
        logger.info("Warm-up complete.")

        main_loop_running = True
        while main_loop_running:
            predictor.reset_session_state()
            predictor.is_predicting_enabled = False
            print("\n" + "=" * 80)
            logger.info("Starting new training pipeline...")
            
            with predictor.data_lock:
                live_data_df = pd.DataFrame(list(predictor.live_data))
            
            training_df = pd.concat([historical_data, live_data_df], ignore_index=True)
            training_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)

            if not predictor.run_training_pipeline(training_df):
                logger.error("Training pipeline failed. Retrying in 5 minutes.")
                time.sleep(300)
                continue
            
            predictor.is_predicting_enabled = True
            logger.info(f"Live prediction started. Will retrain in {predictor.RETRAIN_INTERVAL_MINUTES} minutes.")
            end_time = time.time() + predictor.RETRAIN_INTERVAL_MINUTES * 60
            
            while time.time() < end_time:
                try:
                    if predictor.output_log_queue: print(predictor.output_log_queue.popleft())
                    if not main_loop_running: break
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    main_loop_running = False
                    break
            
            if not main_loop_running: break
            
            predictor.print_final_summary()

            with predictor.data_lock:
                if not predictor.live_data:
                    logger.warning("No new live data collected to append for retraining.")
                    continue
                new_data = pd.DataFrame(list(predictor.live_data))
                new_data.drop_duplicates(subset=['timestamp'], inplace=True)
            
            logger.info(f"Appending {len(new_data)} new records for next training cycle.")
            historical_data = pd.concat([historical_data, new_data], ignore_index=True)
            historical_data.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            
            max_time = historical_data['timestamp'].max()
            cutoff_time = max_time - datetime.timedelta(hours=24)
            historical_data = historical_data[historical_data['timestamp'] > cutoff_time]
            logger.info(f"Historical data pruned. Now contains {len(historical_data)} records.")

    except KeyboardInterrupt:
        logger.info("⏹️ User interrupted. Shutting down.")
    finally:
        predictor.stop()
        logger.info("--- Final Summary on Script Exit ---")
        predictor.print_final_summary()
