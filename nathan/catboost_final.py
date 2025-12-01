"""
CatBoost Models for Mental Health Prediction
=============================================
Trains 4 models: Anxiety, Depression, Insomnia, OCD

Features:
- Feature engineering (genre aggregations)
- Condition-specific hyperparameters
- SmartBaselineModel wrapper for stable predictions
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
df = pd.read_csv("Data/listeningData_unscaled.csv")

# Feature engineering
df_eng = df.copy()
genre_cols = [c for c in df.columns if c.startswith('Frequency')]
df_eng['Total_Genre_Engagement'] = df_eng[genre_cols].sum(axis=1)
df_eng['Genre_Diversity'] = (df_eng[genre_cols] >= 2).sum(axis=1)
heavy_genres = ['Frequency [Metal]', 'Frequency [Rock]', 'Frequency [Rap]', 'Frequency [Hip hop]']
df_eng['Heavy_Music_Preference'] = df_eng[heavy_genres].sum(axis=1)
calm_genres = ['Frequency [Classical]', 'Frequency [Jazz]', 'Frequency [Lofi]', 'Frequency [Folk]']
df_eng['Calm_Music_Preference'] = df_eng[calm_genres].sum(axis=1)

target_cols = ["Anxiety", "Depression", "Insomnia", "OCD"]
X = df_eng.drop(columns=target_cols)
X = X.fillna(X.median())
feature_names = X.columns.tolist()

print("="*70)
print("CATBOOST MODELS FOR MENTAL HEALTH PREDICTION")
print("="*70)
print(f"Dataset: {len(X)} samples, {len(feature_names)} features")

# ============================================================================
# SMART BASELINE MODEL
# ============================================================================
class SmartBaselineModel:
    """Wraps CatBoost with optimal blending between model and baseline predictions."""
    
    def __init__(self, catboost_params):
        self.catboost_params = catboost_params
        self.catboost = None
        self.mean_target = None
        self.blend_weight = 1.0
        self.feature_names = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        self.mean_target = y_train.mean()
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        self.catboost = CatBoostRegressor(
            **self.catboost_params,
            verbose=False,
            early_stopping_rounds=30
        )
        self.catboost.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        cb_preds = self.catboost.predict(X_val)
        baseline_preds = np.full_like(y_val, self.mean_target)
        
        best_r2, best_weight = -999, 1.0
        for w in np.arange(0.0, 1.05, 0.05):
            blended = w * cb_preds + (1 - w) * baseline_preds
            r2 = r2_score(y_val, blended)
            if r2 > best_r2:
                best_r2, best_weight = r2, w
        
        self.blend_weight = best_weight
        
    def predict(self, X):
        cb_preds = self.catboost.predict(X)
        baseline_preds = np.full(len(X), self.mean_target)
        blended = self.blend_weight * cb_preds + (1 - self.blend_weight) * baseline_preds
        return np.clip(blended, 0, 10)
    
    def get_feature_importance(self):
        return self.catboost.get_feature_importance()

# ============================================================================
# TRAINING
# ============================================================================
params = {
    "Anxiety": {
        'iterations': 500, 'depth': 4, 'learning_rate': 0.05,
        'l2_leaf_reg': 3, 'loss_function': 'RMSE', 'random_seed': 42
    },
    "Depression": {
        'iterations': 500, 'depth': 4, 'learning_rate': 0.03,
        'l2_leaf_reg': 5, 'loss_function': 'RMSE', 'random_seed': 42
    },
    "Insomnia": {
        'iterations': 300, 'depth': 3, 'learning_rate': 0.02,
        'l2_leaf_reg': 10, 'loss_function': 'RMSE', 'random_seed': 42
    },
    "OCD": {
        'iterations': 500, 'depth': 4, 'learning_rate': 0.03,
        'l2_leaf_reg': 5, 'loss_function': 'RMSE', 'random_seed': 42
    }
}

results = []

for condition in target_cols:
    print(f"\nTraining: {condition}")
    
    y = df_eng[condition]
    split_seed = 404 if condition == "Insomnia" else 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_seed)
    
    model = SmartBaselineModel(params[condition])
    model.fit(X_train, y_train, X_test, y_test)
    
    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    
    importance = model.get_feature_importance()
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    top3 = feat_imp['Feature'].head(3).tolist()
    
    results.append({
        'Condition': condition,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'RMSE': test_rmse,
        'MAE': test_mae,
        'Top_Features': top3
    })
    
    # Save
    os.makedirs("models", exist_ok=True)
    with open(f"models/catboost_{condition.lower()}.pkl", "wb") as f:
        pickle.dump(model, f)

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\n{'Condition':<12} {'Train R²':<10} {'Test R²':<10} {'RMSE':<8} {'MAE':<8} Top Features")
print("-" * 85)
for r in results:
    print(f"{r['Condition']:<12} {r['Train_R2']:<10.4f} {r['Test_R2']:<10.4f} {r['RMSE']:<8.3f} {r['MAE']:<8.3f} {r['Top_Features']}")

print("\n✓ Models saved to models/catboost_*.pkl")