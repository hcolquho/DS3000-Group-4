import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

class MentalHealthModelTrainer:
    """
    Trains and evaluates multiple regression models to predict mental health scores
    from music listening habits and lifestyle factors.
    """
    
    def __init__(self, data_path='Data/mxmh_survey_results.csv'):
        """Initialize trainer with data path"""
        self.data_path = data_path
        self.df = None
        self.feature_cols = None
        self.genre_cols = None
        self.target_conditions = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
        self.trained_models = {}
        self.performance_metrics = {}
        
    def load_and_preprocess_data(self):
        """Load CSV and preprocess the data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Identify genre columns
        self.genre_cols = [col for col in self.df.columns if 'Frequency [' in col]
        
        # Map frequency to numerical values
        frequency_map = {
            'Never': 0,
            'Rarely': 1,
            'Sometimes': 2,
            'Very frequently': 3
        }
        
        # Encode genre frequencies
        for col in self.genre_cols:
            self.df[col] = self.df[col].map(frequency_map)
        
        # Select relevant features
        self.feature_cols = ['Age', 'Hours per day', 'BPM'] + self.genre_cols
        
        # Add binary features
        binary_features = ['While working', 'Instrumentalist', 'Composer', 
                          'Exploratory', 'Foreign languages']
        
        for col in binary_features:
            if col in self.df.columns:
                # Convert Yes/No to 1/0
                self.df[col] = self.df[col].map({'Yes': 1, 'No': 0})
                self.feature_cols.append(col)
        
        # Remove rows with missing target values
        for condition in self.target_conditions:
            self.df = self.df[self.df[condition].notna()]
        
        # Fill missing feature values with median
        for col in self.feature_cols:
            if self.df[col].isna().any():
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        print(f"Data loaded: {len(self.df)} samples, {len(self.feature_cols)} features")
        print(f"Features: {self.feature_cols}")
        
    def get_model_configs(self):
        """Define model configurations and hyperparameter grids"""
        return {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'CatBoost': {
                'model': CatBoostRegressor(random_state=42, verbose=0),
                'params': {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1, 3, 5]
                }
            },
            'MLP': {
                'model': MLPRegressor(random_state=42, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01]
                }
            }
        }
    
    def train_models_for_condition(self, condition_name):
        """
        Train multiple model types for a single mental health condition
        
        Args:
            condition_name: Name of target condition (e.g., 'Anxiety')
            
        Returns:
            Dictionary with trained models and metrics
        """
        print(f"\n{'='*60}")
        print(f"Training models for {condition_name}")
        print(f"{'='*60}")
        
        # Prepare data
        X = self.df[self.feature_cols].values
        y = self.df[condition_name].values
        
        # Train-test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        model_configs = self.get_model_configs()
        condition_results = {}
        best_score = -np.inf
        best_model_name = None
        
        # Train each model type
        for model_name, config in model_configs.items():
            print(f"\n--- Training {model_name} ---")
            
            # Hyperparameter tuning with cross-validation
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=15,  # Number of parameter combinations to try
                cv=5,  # 5-fold cross-validation
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            
            # Best model from CV
            best_model = search.best_estimator_
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"Best params: {search.best_params_}")
            print(f"Test R² Score: {r2:.4f}")
            print(f"Test RMSE: {rmse:.4f}")
            print(f"Test MAE: {mae:.4f}")
            
            # Store results
            condition_results[model_name] = {
                'model': best_model,
                'best_params': search.best_params_,
                'cv_score': -search.best_score_,  # Convert back to positive MSE
                'test_r2': r2,
                'test_rmse': rmse,
                'test_mae': mae
            }
            
            # Track best model
            if r2 > best_score:
                best_score = r2
                best_model_name = model_name
        
        print(f"\n🏆 Best model for {condition_name}: {best_model_name} (R² = {best_score:.4f})")
        
        return {
            'all_models': condition_results,
            'best_model': condition_results[best_model_name]['model'],
            'best_model_name': best_model_name
        }
    
    def train_all_conditions(self):
        """Train models for all four mental health conditions"""
        print("Starting training for all conditions...")
        
        for condition in self.target_conditions:
            results = self.train_models_for_condition(condition)
            self.trained_models[condition] = results['best_model']
            self.performance_metrics[condition] = results['all_models']
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of all trained models"""
        print("\n📊 FINAL MODEL SUMMARY")
        print("="*60)
        
        for condition in self.target_conditions:
            print(f"\n{condition}:")
            for model_name, metrics in self.performance_metrics[condition].items():
                print(f"  {model_name:15s} | R²: {metrics['test_r2']:.4f} | "
                      f"RMSE: {metrics['test_rmse']:.4f} | MAE: {metrics['test_mae']:.4f}")
    
    def save_models(self, output_dir='models/'):
        """Save trained models to disk"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for condition, model in self.trained_models.items():
            filename = f"{output_dir}{condition.lower()}_model.pkl"
            joblib.dump(model, filename)
            print(f"Saved {condition} model to {filename}")
    
    def get_model_dict(self):
        """Return dictionary of trained models for use with optimizer"""
        return {
            condition.lower(): model 
            for condition, model in self.trained_models.items()
        }


# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = MentalHealthModelTrainer('Data/mxmh_survey_results.csv')
    
    # Load and preprocess data
    trainer.load_and_preprocess_data()
    
    # Train models for all conditions
    trainer.train_all_conditions()
    
    # Save models
    trainer.save_models()
    
    # Get model dictionary for optimization
    model_dict = trainer.get_model_dict()
    print("\nModel dictionary ready for optimization!")
    print(f"Available models: {list(model_dict.keys())}")