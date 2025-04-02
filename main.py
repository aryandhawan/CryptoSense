import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os

# Load API key securely
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API Key is missing! Add it to a .env file.")


class CryptoPredictor:
    def __init__(self, api_key, coin, currency):
        self.api_key = api_key
        self.coin = coin
        self.currency = currency
        self.headers = {'x-cg-demo-api-key': api_key}
        self.model = None
        self.data = None

    def fetch_historical_data(self):
        """Fetches last 6 months of crypto data and prepares it for training."""
        print("\nFetching historical data for prediction...\n")

        url = f'https://api.coingecko.com/api/v3/coins/{self.coin}/market_chart?vs_currency={self.currency}&days=180'
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            print("Error fetching data:", response.json())
            return

        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["Timestamp", "Price"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df.rename(columns={"Timestamp": "Date"}, inplace=True)

        df['Daily_return'] = df['Price'].pct_change() * 100
        df['MA_7'] = df['Price'].rolling(window=7).mean()
        df['MA_30'] = df['Price'].rolling(window=30).mean()
        df['Volatility'] = df['Daily_return'].rolling(window=7).std()
        df['Future_price'] = df['Price'].shift(-3)
        df['Target'] = (df['Future_price'] > df['Price']).astype(int)

        df.dropna(inplace=True)
        self.data = df

    def train_model(self):
        """Trains a RandomForest model using historical data."""
        if self.data is None:
            print("No historical data found. Fetching now...")
            self.fetch_historical_data()
            if self.data is None:
                return

        features = ['MA_7', 'MA_30', 'Volatility', 'Daily_return']
        X = self.data[features]
        y = self.data['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Market Prediction Model Accuracy: {accuracy:.2f}")

    def predict_market(self):
        """Predicts whether the market will go up or down."""
        if self.model is None:
            print("Model not trained. Training now...")
            self.train_model()
            if self.model is None:
                return

        features = ['MA_7', 'MA_30', 'Volatility', 'Daily_return']
        latest_data = self.data[features].iloc[-1:].values
        prediction = self.model.predict(latest_data)[0]

        if prediction == 1:
            print("AI Prediction: The price is likely to increase in the next few days.")
        else:
            print("AI Prediction: The price is likely to decrease in the next few days.")

    def fetch_market_info(self):
        """Fetches market trends and optionally predicts market movement."""
        market_capitalization = input('Include market capitalization? (yes/no): ').lower()
        change = input('Include 24h change? (yes/no): ').lower()
        market_prediction = input('Run AI market prediction? (yes/no): ').lower()

        url = f'https://api.coingecko.com/api/v3/simple/price?ids={self.coin}&vs_currencies={self.currency}'
        if market_capitalization == 'yes':
            url += '&include_market_cap=true'

        request = requests.get(url, headers=self.headers)
        data = request.json()

        data_price = data.get(self.coin, {}).get(self.currency, 'N/A')
        market_cap = data.get(self.coin, {}).get(f'{self.currency}_market_cap', 'N/A')

        change_data = 'N/A'
        if change == 'yes':
            url_change = f'https://api.coingecko.com/api/v3/coins/{self.coin}'
            request_change = requests.get(url_change, headers=self.headers)
            data_change = request_change.json()
            change_data = data_change.get("market_data", {}).get("price_change_percentage_24h", 'N/A')

        print("\n📊 **Current Market Trends**:")
        print(f"💰 {self.coin.capitalize()} Price in {self.currency.upper()}: {data_price}")
        if market_capitalization == 'yes':
            print(f"🏦 Market Cap: {market_cap}")
        if change == 'yes':
            print(f"📉 24h Change: {change_data}%")

        if market_prediction == 'yes':
            print("\n🔍 Running Market Prediction...")
            self.fetch_historical_data()
            self.train_model()
            self.predict_market()
        else:
            print("\n✅ Market trends displayed. No prediction made.")


predictor = CryptoPredictor(api_key=API_KEY, coin="bitcoin", currency="inr")
predictor.fetch_market_info()
