import ccxt
import os
from time import sleep
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from datetime import datetime
import pickle
import csv
from dotenv import load_dotenv
from skopt.space import Real, Integer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import pandas as pd
from finta import TA

class TradingBot:
    def __init__(self, symbol, leverage, amount, take_profit_percentage, stop_loss_percentage):
        load_dotenv()
        self.symbol = symbol
        self.leverage = leverage
        self.amount = amount
        self.take_profit_percentage = take_profit_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.exchange = ccxt.kucoinfutures({
            'apiKey': os.getenv('API_KEY'),
            'secret': os.getenv('SECRET_KEY'),
            'password': os.getenv('PASSPHRASE'),
            'enableRateLimit': True
        })
        
        # Initialize these attributes
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def calculate_atr(self, data, period=14):
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['ATR'] = TA.ATR(df, period)
        return df['ATR'].iloc[-1]

    def save_data_to_csv(self, data, filename):
        header = ["timestamp", "open", "high", "low", "close"]
        with open(filename, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if csv_file.tell() == 0:
                writer.writerow(header)
            for d in data:
                writer.writerow(d)

    def fetch_data_and_preprocess(self, timeframe='4h', limit=1000):
        try:
            data = []
            since = None
            while True:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, since=since, limit=limit)
                if len(ohlcv) == 0:
                    break
                since = ohlcv[-1][0] + 1
                data.extend(list(ohlcv_point) for ohlcv_point in ohlcv)
                features = np.array([d[0:6] for d in data if len(d) == 6])  # Use all 6 columns
                if np.isnan(features).any():
                    mask = ~np.isnan(features).any(axis=1)
                    data = [d for d, m in zip(data, mask) if m]
                    features = features[mask]
                scaler = MinMaxScaler()
                scaled_features = scaler.fit_transform(features)
                z_scores = np.abs(stats.zscore(scaled_features))
                threshold = 3
                outlier_mask = (z_scores < threshold).all(axis=1)
                data = [d for d, o in zip(data, outlier_mask) if o]
                scaled_features = scaled_features[outlier_mask]
                self.save_data_to_csv(data, 'RF_vra.csv')  # Updated filename to reflect Random Forest
                mean_values = np.mean(scaled_features, axis=0)
                std_deviation = np.std(scaled_features, axis=0)
                min_values = np.min(scaled_features, axis=0)
                max_values = np.max(scaled_features, axis=0)
                feature_ranges = max_values - min_values
                for i in range(scaled_features.shape[1]):
                    percentage_removed = (np.sum(~outlier_mask) / len(data)) * 100
                    #print(f"Percentage of Data Removed as Outliers: {percentage_removed:.2f}%")
                return data, scaled_features, scaler
        except Exception as e:
            print(f"An error occurred while fetching and preprocessing data: {e}")
            return None, None

    def train_models(self, features, target):
        try:
            # Assign values to attributes
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(features, target, test_size=0.2, random_state=42)
            random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
            random_forest.fit(self.X_train, self.y_train)
            y_pred = random_forest.predict(self.X_val)
            classification_rep = classification_report(self.y_val, y_pred)
            print(classification_rep) #print("Classification Report for Random Forest Model:")
            print("Random Forest Model Training Completed")  
            accuracy_train = random_forest.score(self.X_train, self.y_train) # Print accuracy on training set
            print(f"Accuracy on Training Set: {accuracy_train:.4f}")   
            accuracy_val = random_forest.score(self.X_val, self.y_val) # Print accuracy on validation set
            print(f"Accuracy on Validation Set: {accuracy_val:.4f}")            
            print("Random Forest Model:", random_forest) # Print the models for debugging            
            with open('15mincheck15minochlvmodel.pkl', 'wb') as f:
                pickle.dump(random_forest, f)
            return random_forest
        except Exception as e:
            print(f"An error occurred while training models: {e}")
            return None

    def predict_market_direction(self, data, rf_model, scaler):
        try:
            features = np.array([d[0:6] for d in data if len(d) == 6])
            if not features.any():
                print("No valid data available for prediction.")
                return None
            print("Shapes before scaling:")
            print("X_train:", self.X_train.shape)
            print("X_val:", self.X_val.shape)
            print("features:", features.shape)

            scaled_features = scaler.transform(features)

            print("Shapes after scaling:")
            print("X_train:", self.X_train.shape)
            print("X_val:", self.X_val.shape)
            print("scaled_features:", scaled_features.shape)

            rf_accuracy_train = rf_model.score(self.X_train, self.y_train)
            rf_accuracy_val = rf_model.score(self.X_val, self.y_val)
            print(f"Accuracy of Random Forest Model on Training Set: {rf_accuracy_train:.4f}")
            print(f"Accuracy of Random Forest Model on Validation Set: {rf_accuracy_val:.4f}")
            rf_prediction = rf_model.predict(scaled_features)
            print("Random Forest Model Prediction on Validation Set:")
            print(rf_prediction)
            final_prediction = rf_prediction[-1]
            print("Final Prediction (0 for Bullish, 1 for Bearish):")
            print(final_prediction)
            return final_prediction
        except Exception as e:
            print(f"An error occurred while predicting market direction: {e}")
            return None

    def create_order_with_percentage_levels(self, side, entry_price):
        try:
            take_profit_price = entry_price * (1 + self.take_profit_percentage / 100)
            stop_loss_price = entry_price * (1 - self.stop_loss_percentage / 100)
            main_order = self.exchange.create_order(
                self.symbol,
                type='limit',
                side=side,
                amount=self.amount,
                price=entry_price,
                params={
                    'postOnly': True,
                    'timeInForce': 'GTC',
                    'leverage': self.leverage
                }
            )
            print("Main Order Created:", main_order)
            stop_loss_order = self.exchange.create_order(
                self.symbol,
                type='limit',
                side='sell' if side == 'buy' else 'buy',
                amount=self.amount,
                price=stop_loss_price
            )
            print("Stop-Loss Order Created:", stop_loss_order)
            take_profit_order = self.exchange.create_order(
                self.symbol,
                type='limit',
                side='sell' if side == 'buy' else 'buy',
                amount=self.amount,
                price=take_profit_price
            )
            print("Take-Profit Order Created:", take_profit_order)
            return main_order, stop_loss_order, take_profit_order
        except Exception as e:
            print(f"Error creating orders with percentage-based levels: {e}")
            return None, None, None

    def main_trading_loop(self):
        rf_model = None  # Initialize rf_model outside the loop

        while True:
            try:
                loop_start_time = datetime.now()

                while True:
                    current_time = datetime.now()
                    elapsed_time = current_time - loop_start_time

                    if elapsed_time.seconds < 60:
                        print(f"Waiting for {60 - elapsed_time.seconds} seconds before creating orders...")
                        sleep(60)

                    data, scaled_features, scaler = self.fetch_data_and_preprocess()

                    if data is not None:
                        target = np.array([1 if d[4] < d[1] else 0 for d in data])

                        # Train the random forest model
                        rf_model = self.train_models(scaled_features, target)

                        if rf_model is not None:
                            print("Random Forest model was successfully trained.")
                            ticker = self.exchange.fetch_ticker(self.symbol)
                            bid, ask = ticker['bid'], ticker['ask']
                            midpoint = (bid + ask) / 2
                            current_time = datetime.now()
                            market_direction = self.predict_market_direction(data, rf_model, scaler)
                            if market_direction is not None:
                                print("The market is ---> {}".format(market_direction))
                                print(current_time.strftime("%B %d, %Y %I:%M %p"))

                                if market_direction == 0:  # Bullish
                                    suggested_limit_price = midpoint - 0.01
                                else:  # Bearish
                                    suggested_limit_price = midpoint + 0.01

                                # Creating orders with a single level
                                self.create_order_with_percentage_levels('buy' if market_direction == 0 else 'sell', suggested_limit_price)

                                tp_price = suggested_limit_price * (1 + self.take_profit_percentage / 100)
                                sl_price = suggested_limit_price * (1 - self.stop_loss_percentage / 100)
                                print(f"Take-Profit Price: {tp_price}, Stop-Loss Price: {sl_price}")
                                print("\n" + "-"*50)
                                start_time = current_time
                            else:
                                print("Error: Prediction failed.")
                                sleep(160)
                        else:
                            print("Error: Training the random forest model failed.")
                            sleep(160)

            except Exception as e:
                print(f"An error occurred in the main trading loop: {e}")
                sleep(60)
                continue

if __name__ == "__main__":
    trading_bot = TradingBot(
        symbol='BTC/USDT:USDT',
        leverage=10,
        amount=0.1,
        take_profit_percentage=1.35,
        stop_loss_percentage=1.35
    )
    trading_bot.main_trading_loop()
