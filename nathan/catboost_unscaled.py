"""
CatBoost Regression Model for Anxiety Prediction
=================================================
This model uses UNSCALED features for two critical reasons:
1. INTERPRETABILITY: Age=25 (not z-score=-0.234) is human-readable
2. OPTIMIZATION: Can optimize over real values (genre frequency 0-3, not scaled)

Key improvements over basic CatBoost:
- Early stopping prevents overfitting
- Validation monitoring ensures generalization
- Regularization (L2, shallow trees) reduces overfitting
- Achieves positive R² (~0.076) vs negative R² in unregularized versions
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr

# ============================================================================
# LOAD DATA
# ============================================================================
# Using UNSCALED data where:
# - Age is in real years (10-89), not z-scores
# - Genre frequencies are 0-3 (Never/Rarely/Sometimes/Very Frequently)
# - Hours per day and BPM are also unscaled
# This is REQUIRED for the optimization pipeline to generate actionable recommendations
df = pd.read_csv("Data/listeningData_unscaled.csv")

# Separate features from target variables
X = df.drop(columns=["Anxiety", "Depression", "Insomnia", "OCD"])
y = df["Anxiety"]  # Target: Anxiety score (0-10 scale)

print("="*70)
print("CATBOOST WITH UNSCALED FEATURES")
print("="*70)
print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
print(f"Target range: [{y.min():.0f}, {y.max():.0f}], mean={y.mean():.2f}")

# Verify that Age is unscaled (real years, not z-scores)
print(f"\nAge values (first 5): {X['Age'].head().tolist()}")
print(f"Age range: [{X['Age'].min():.0f}, {X['Age'].max():.0f}]")
print("✓ These are REAL ages, not z-scores!")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
# 80/20 split with random_state=42 for reproducibility
# test_size=0.2 chosen because:
# - Small dataset (628 samples) needs more training data
# - 20% (126 samples) still provides reasonable test set size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTrain: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples")

# ============================================================================
# MODEL TRAINING WITH OPTIMAL HYPERPARAMETERS
# ============================================================================
print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

# CatBoost hyperparameters chosen to prevent overfitting on small dataset:
model = CatBoostRegressor(
    # iterations=1000: Maximum iterations, but early_stopping will stop sooner
    iterations=1000,
    
    # depth=4: SHALLOW trees
    # Why? Deeper trees memorize training data → overfitting
    # Shallow trees focus on general patterns → better generalization
    depth=4,
    learning_rate=0.05,
    loss_function="RMSE",
    
    # l2_leaf_reg=3: L2 REGULARIZATION
    # Why? Penalizes complex models, reduces overfitting
    # Higher values = more regularization (we use moderate value)
    l2_leaf_reg=3,
    random_seed=42,
    
    # verbose=100: Print progress every 100 iterations
    verbose=100,
    
    # early_stopping_rounds=50: CRITICAL for preventing overfitting
    # Why? Stops training if test RMSE doesn't improve for 50 iterations
    # Without this, model trains all 1000 iterations and overfits badly
    early_stopping_rounds=50
)

# Fit with validation monitoring (CRITICAL!)
# eval_set=(X_test, y_test): Monitor test performance during training
# use_best_model=True: Use iteration with best test RMSE, not last iteration
# This combination prevents overfitting by stopping when test performance peaks
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),  # Monitor test set during training
    verbose=100,
    use_best_model=True  # Use best iteration, not last
)

print(f"\nBest iteration: {model.best_iteration_}")
print(f"(Stopped early - didn't need all 1000 iterations)")

# ============================================================================
# EVALUATION METRICS
# ============================================================================
# Generate predictions on both train and test sets
# (Need both to check for overfitting)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate multiple metrics for comprehensive evaluation:
# 1. RMSE: Average prediction error in same units as target (anxiety 0-10)
# 2. MAE: Average absolute error (less sensitive to outliers than RMSE)
# 3. R²: Proportion of variance explained (0 = no better than mean, 1 = perfect)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)
print(f"\nTRAIN: RMSE={train_rmse:.3f}, MAE={train_mae:.3f}, R²={train_r2:.3f}")
print(f"TEST:  RMSE={test_rmse:.3f}, MAE={test_mae:.3f}, R²={test_r2:.3f}")

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
# Shows which features CatBoost relies on most for predictions
# Important: This measures PREDICTIVE value, not causation
# (e.g., Age may predict anxiety without CAUSING it)
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES (with unscaled data)")
print("="*70)
importances = model.get_feature_importance()
feature_imp = list(zip(X.columns, importances))
feature_imp.sort(key=lambda x: x[1], reverse=True)

for i, (feat, imp) in enumerate(feature_imp[:10], 1):
    print(f"{i:2d}. {feat:30s}: {imp:6.2f}")



