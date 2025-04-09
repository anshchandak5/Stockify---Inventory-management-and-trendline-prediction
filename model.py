import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
import optuna
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import json
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Custom JSON encoder to handle Pandas Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)

class DemandPredictor:
    def __init__(self):
        self.model = None
        self.prophet_model = None
        self.sarima_model = None
        self.scalers = {}
        self.label_encoders = {}
        self.last_trained = None
        self.model_path = "demand_model.keras"
        self.dl_model_path = "demand_model_dl.keras"
        self.scalers_path = "demand_model_scalers.joblib"
        self.encoders_path = "demand_model_encoders.joblib"
        self.prophet_path = "demand_model_prophet.json"
        self.prophet_history_path = "demand_model_prophet_history.csv"
        self.sarima_path = "demand_model_sarima.joblib"
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        # Define hyperparameters to optimize
        n_layers = trial.suggest_int('n_layers', 1, 3)
        n_units = trial.suggest_int('n_units', 32, 256)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # Build model with suggested hyperparameters
        model = Sequential()
        model.add(Dense(n_units, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(dropout_rate))
        
        for _ in range(n_layers - 1):
            n_units = trial.suggest_int(f'n_units_layer_{_}', 16, 128)
            model.add(Dense(n_units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(1))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,  # Reduced epochs for optimization
            batch_size=32,
            verbose=0
        )
        
        # Return validation loss
        return history.history['val_loss'][-1]
        
    def preprocess_data(self, df, training=True):
        data = df.copy()
        
        # Ensure date is datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data['day'] = data['date'].dt.day
            data['month'] = data['date'].dt.month
            data['year'] = data['date'].dt.year
        
        # Adapt to different column names in store-uploaded data
        if 'daily_sales' in data.columns and 'actual_demand' not in data.columns:
            data['actual_demand'] = data['daily_sales']
        
        if 'daily_revenue' in data.columns and 'revenue' not in data.columns:
            data['revenue'] = data['daily_revenue']
        
        if 'stock_level' in data.columns and 'stock' not in data.columns:
            data['stock'] = data['stock_level']
        
        if 'promotion_active' in data.columns and 'is_promotional_period' not in data.columns:
            data['is_promotional_period'] = data['promotion_active']
        
        # Add missing columns with default values if needed
        default_columns = {
            'store_size': 'Medium',
            'product_id': 'DEFAULT',
            'subcategory': 'DEFAULT',
            'income_level': 'Medium',
            'credit_limit': 5000,
            'discount': 0,
            'base_price': 100,
            'competitor_price': 100,
            'seasonal_factor': 1.0,
            'festival_boost': 0,
            'storage_capacity': 1000,
            'stock_days_remaining': 30,
            'is_urban': True
        }
        
        for col, default_value in default_columns.items():
            if col not in data.columns:
                data[col] = default_value
        
        # Define column types
        categorical_cols = ['store_id', 'area', 'store_size', 'product_id', 
                          'category', 'subcategory', 'income_level']
        
        numerical_cols = ['credit_limit', 'customer_footfall', 'discount',
                         'stock', 'margin', 'base_price', 'competitor_price', 
                         'seasonal_factor', 'festival_boost', 'storage_capacity',
                         'stock_days_remaining', 'day', 'month', 'year']
        
        boolean_cols = ['is_urban', 'is_promotional_period']
        
        # Process categorical columns
        for col in categorical_cols:
            if col in data.columns:
                if training:
                    self.label_encoders[col] = LabelEncoder()
                    data[col] = self.label_encoders[col].fit_transform(data[col])
                else:
                    # Handle new categories not seen during training
                    if col in self.label_encoders:
                        try:
                            data[col] = self.label_encoders[col].transform(data[col])
                        except ValueError:
                            # For unseen categories, assign a default value
                            data[col] = 0
        
        # Process numerical columns
        for col in numerical_cols:
            if col in data.columns:
                if training:
                    self.scalers[col] = StandardScaler()
                    data[col] = self.scalers[col].fit_transform(data[col].values.reshape(-1, 1))
                else:
                    if col in self.scalers:
                        data[col] = self.scalers[col].transform(data[col].values.reshape(-1, 1))
        
        # Process boolean columns
        for col in boolean_cols:
            if col in data.columns:
                data[col] = data[col].astype(int)
        
        # Drop date column for model input
        if 'date' in data.columns:
            data = data.drop('date', axis=1)
            
        return data
    
    def build_model(self, input_shape, trial=None):
        if trial:
            n_layers = trial.suggest_int('n_layers', 1, 3)
            n_units = trial.suggest_int('n_units', 32, 256)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        else:
            n_layers = 2
            n_units = 128
            dropout_rate = 0.2
        
        model = Sequential()
        model.add(Dense(n_units, activation='relu', input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        for _ in range(n_layers - 1):
            model.add(Dense(n_units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_prophet(self, df, store_id=None, category=None):
        prophet_data = df.copy()
        
        # Adapt to different column names
        if 'date' in prophet_data.columns:
            prophet_data['ds'] = prophet_data['date']
        
        if 'actual_demand' in prophet_data.columns:
            prophet_data['y'] = prophet_data['actual_demand']
        elif 'daily_sales' in prophet_data.columns:
            prophet_data['y'] = prophet_data['daily_sales']
        
        if store_id:
            prophet_data = prophet_data[prophet_data['store_id'] == store_id]
        if category:
            prophet_data = prophet_data[prophet_data['category'] == category]
        
        # Add default values for missing columns
        if 'seasonal_factor' not in prophet_data.columns:
            prophet_data['seasonal_factor'] = 1.0
        if 'festival_boost' not in prophet_data.columns:
            prophet_data['festival_boost'] = 0
        
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        self.prophet_model.add_regressor('seasonal_factor')
        self.prophet_model.add_regressor('festival_boost')
        
        self.prophet_model.fit(prophet_data[['ds', 'y', 'seasonal_factor', 'festival_boost']])
    
    def train_sarima(self, df, store_id=None, category=None):
        sarima_data = df.copy()
        
        # Ensure date is datetime
        if 'date' in sarima_data.columns:
            sarima_data['date'] = pd.to_datetime(sarima_data['date'])
        
        if store_id:
            sarima_data = sarima_data[sarima_data['store_id'] == store_id]
        if category:
            sarima_data = sarima_data[sarima_data['category'] == category]
        
        # Use appropriate column for demand
        target_col = 'actual_demand' if 'actual_demand' in sarima_data.columns else 'daily_sales'
        
        # Set date as index and resample
        sarima_data = sarima_data.set_index('date')
        sarima_data = sarima_data[target_col].resample('D').mean().ffill()
        
        self.sarima_model = SARIMAX(
            sarima_data,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        ).fit(disp=False)
    
    def train(self, df, epochs=50, use_optuna=False):
        processed_data = self.preprocess_data(df, training=True)
        
        # Determine target column
        target_col = 'actual_demand' if 'actual_demand' in processed_data.columns else 'daily_sales'
        
        X = processed_data.drop(target_col, axis=1)
        y = processed_data[target_col]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if use_optuna:
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=10)
            self.model = self.build_model((X_train.shape[1],), study.best_trial)
        else:
            self.model = self.build_model((X_train.shape[1],))
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        print("Training Prophet model...")
        self.train_prophet(df)
        
        print("Training SARIMA model...")
        self.train_sarima(df)
        
        self.last_trained = datetime.now()
        
        return history
    
    def predict(self, df):
        processed_data = self.preprocess_data(df, training=False)
        
        # Determine target column
        target_col = 'actual_demand' if 'actual_demand' in processed_data.columns else 'daily_sales'
        
        # Initialize predictions arrays
        num_samples = len(df)
        dl_preds = np.zeros(num_samples)
        prophet_preds = np.zeros(num_samples)
        sarima_preds = np.zeros(num_samples)
        
        # Count available models
        available_models = 0
        
        # Deep Learning predictions
        if self.model is not None:
            try:
                dl_preds = self.model.predict(processed_data.drop(target_col, axis=1) if target_col in processed_data.columns else processed_data).flatten()
                available_models += 1
            except Exception as e:
                print(f"Error making DL predictions: {str(e)}")
        
        # Get Prophet predictions
        if self.prophet_model is not None:
            try:
                future_prophet = pd.DataFrame({
                    'ds': df['date'],
                    'seasonal_factor': df['seasonal_factor'] if 'seasonal_factor' in df.columns else 1.0,
                    'festival_boost': df['festival_boost'] if 'festival_boost' in df.columns else 0
                })
                prophet_preds = self.prophet_model.predict(future_prophet)['yhat'].values
                available_models += 1
            except Exception as e:
                print(f"Error making Prophet predictions: {str(e)}")
        
        # Get SARIMA predictions
        if self.sarima_model is not None:
            try:
                sarima_preds = self.sarima_model.forecast(num_samples).values
                available_models += 1
            except Exception as e:
                print(f"Error making SARIMA predictions: {str(e)}")
        
        # Ensemble predictions (average of available models)
        if available_models > 0:
            predictions = (dl_preds + prophet_preds + sarima_preds) / available_models
        else:
            # Fallback to simple linear regression if no models are available
            print("No trained models available. Using simple linear regression as fallback.")
            
            # If we have historical data with the target column, fit a simple model
            if target_col in processed_data.columns:
                # Create a simple time-based feature
                X = np.array(range(len(processed_data))).reshape(-1, 1)
                y = processed_data[target_col].values
                
                # Fit a simple linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict using the simple model
                predictions = model.predict(X)
            else:
                # If no target column is available, return zeros
                predictions = np.zeros(num_samples)
        
        return predictions
    
    def save_models(self):
        """Save all models and preprocessing objects"""
        self.last_trained = datetime.now()
        
        # Save deep learning model
        self.model.save(self.dl_model_path)
        
        # Save scalers and encoders
        joblib.dump(self.scalers, self.scalers_path)
        joblib.dump(self.label_encoders, self.encoders_path)
        
        # Save Prophet model components instead of using to_json
        if self.prophet_model is not None:
            # Save Prophet model parameters
            prophet_params = {
                'growth': self.prophet_model.growth,
                'n_changepoints': self.prophet_model.n_changepoints,
                'yearly_seasonality': self.prophet_model.yearly_seasonality,
                'weekly_seasonality': self.prophet_model.weekly_seasonality,
                'daily_seasonality': self.prophet_model.daily_seasonality,
                'seasonality_mode': self.prophet_model.seasonality_mode,
                'seasonality_prior_scale': self.prophet_model.seasonality_prior_scale,
                'changepoint_prior_scale': self.prophet_model.changepoint_prior_scale,
                'holidays_prior_scale': self.prophet_model.holidays_prior_scale,
                'mcmc_samples': self.prophet_model.mcmc_samples,
                'interval_width': self.prophet_model.interval_width,
                'uncertainty_samples': self.prophet_model.uncertainty_samples,
            }
            
            with open(self.prophet_path, 'w') as f:
                json.dump(prophet_params, f, cls=CustomJSONEncoder)
            
            # Save Prophet model history
            if hasattr(self.prophet_model, 'history') and self.prophet_model.history is not None:
                self.prophet_model.history.to_csv(self.prophet_history_path)
        
        # Save SARIMA model
        if self.sarima_model is not None:
            joblib.dump(self.sarima_model, self.sarima_path)
        
        print(f"All models saved successfully. Last trained: {self.last_trained}")
    
    def load_models(self):
        """Load all saved models and preprocessing objects"""
        try:
            # Load deep learning model
            if os.path.exists(self.dl_model_path):
                self.model = load_model(self.dl_model_path)
            
            # Load scalers and encoders
            if os.path.exists(self.scalers_path):
                self.scalers = joblib.load(self.scalers_path)
            
            if os.path.exists(self.encoders_path):
                self.label_encoders = joblib.load(self.encoders_path)
            
            # Load Prophet model - recreate instead of using from_json
            if os.path.exists(self.prophet_path) and os.path.exists(self.prophet_history_path):
                # Load parameters
                with open(self.prophet_path, 'r') as f:
                    prophet_params = json.load(f)
                
                # Create a new Prophet model with the saved parameters
                self.prophet_model = Prophet(
                    growth=prophet_params.get('growth', 'linear'),
                    changepoints=None,  # Will be set after fitting
                    n_changepoints=prophet_params.get('n_changepoints', 25),
                    yearly_seasonality=prophet_params.get('yearly_seasonality', True),
                    weekly_seasonality=prophet_params.get('weekly_seasonality', True),
                    daily_seasonality=prophet_params.get('daily_seasonality', False),
                    seasonality_mode=prophet_params.get('seasonality_mode', 'additive'),
                    seasonality_prior_scale=prophet_params.get('seasonality_prior_scale', 10.0),
                    changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.05),
                    holidays_prior_scale=prophet_params.get('holidays_prior_scale', 10.0),
                    mcmc_samples=prophet_params.get('mcmc_samples', 0),
                    interval_width=prophet_params.get('interval_width', 0.8),
                    uncertainty_samples=prophet_params.get('uncertainty_samples', 1000)
                )
                
                # Add regressors if they were in the original model
                self.prophet_model.add_regressor('seasonal_factor')
                self.prophet_model.add_regressor('festival_boost')
                
                # Load history and refit the model
                history_df = pd.read_csv(self.prophet_history_path)
                # Convert string dates to datetime
                if 'ds' in history_df.columns:
                    history_df['ds'] = pd.to_datetime(history_df['ds'])
                
                # Only fit if we have the required columns
                required_cols = ['ds', 'y']
                if all(col in history_df.columns for col in required_cols):
                    # Add regressor columns if missing
                    for reg in ['seasonal_factor', 'festival_boost']:
                        if reg not in history_df.columns:
                            history_df[reg] = 1.0
                    
                    # Fit the model with historical data
                    try:
                        self.prophet_model.fit(history_df[['ds', 'y', 'seasonal_factor', 'festival_boost']])
                    except Exception as fit_error:
                        print(f"Warning: Could not fit Prophet model with history: {str(fit_error)}")
            
            # Load SARIMA model
            if os.path.exists(self.sarima_path):
                self.sarima_model = joblib.load(self.sarima_path)
            
            print("All models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def update_with_new_data(self, new_data, full_retrain=False):
        """
        Update models with new data
        
        Parameters:
        -----------
        new_data : pandas.DataFrame
            New data to update models with
        full_retrain : bool
            Whether to perform a full retraining or incremental update
        """
        try:
            if full_retrain:
                # Full retraining with new data
                self.train(new_data, epochs=30)
            else:
                # Incremental update for Prophet model
                self.train_prophet(new_data)
                
                # Incremental update for SARIMA model
                self.train_sarima(new_data)
                
                # For deep learning model, we could do fine-tuning
                # but for simplicity, we'll just use the existing model
            
            # Save updated models
            self.save_models()
            
            self.last_trained = datetime.now()
            return True, f"Models updated successfully at {self.last_trained}"
        
        except Exception as e:
            return False, f"Error updating models: {str(e)}"
    
    def update_city_zone_models(self, aggregated_data, zone=None, city=None):
        """
        Update models for city or zone-wide predictions
        
        Parameters:
        -----------
        aggregated_data : pandas.DataFrame
            Aggregated data for the city or zone
        zone : str, optional
            Zone name for zone-specific models
        city : str, optional
            City name for city-specific models
        """
        try:
            model_suffix = ""
            if zone:
                model_suffix = f"_{zone}"
            elif city:
                model_suffix = f"_{city}"
            
            # Create a new predictor for this zone/city
            zone_predictor = DemandPredictor()
            
            # Set custom paths for zone/city models
            zone_predictor.dl_model_path = f"demand_model_dl{model_suffix}.keras"
            zone_predictor.scalers_path = f"demand_model_scalers{model_suffix}.joblib"
            zone_predictor.encoders_path = f"demand_model_encoders{model_suffix}.joblib"
            zone_predictor.prophet_path = f"demand_model_prophet{model_suffix}.json"
            zone_predictor.prophet_history_path = f"demand_model_prophet_history{model_suffix}.csv"
            zone_predictor.sarima_path = f"demand_model_sarima{model_suffix}.joblib"
            
            # Train models with aggregated data
            zone_predictor.train(aggregated_data, epochs=30)
            
            # Save zone/city models
            zone_predictor.save_models()
            
            return True, f"Zone/city models updated successfully for {zone or city}"
        
        except Exception as e:
            return False, f"Error updating zone/city models: {str(e)}"

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv("retail_data.csv")
    
    print("Training models...")
    predictor = DemandPredictor()
    predictor.train(df, use_optuna=True)
    
    print("Saving models...")
    predictor.save_models()
