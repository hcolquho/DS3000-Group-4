# Given the 4 models each predicting mental health score in one of OCD, Insomnia, Depression,
# and Anxiety, this script tries to finds the best profile using Bayesian Optimisation.
# The best profile is the one that minismises the projected mental health score for the model
# corresponding to that condition that the individual would like to improve.
# Output will be formatted to suggest up to 3 genres the individual can try based on the best profile found
# as well up to 3 that they should avoid.

import os
import numpy as np
import pandas as pd
import optuna
import joblib
import xgboost as xgb


# Please make sure to have the required libraries installed
class MusicProfileOptimiser:

    def __init__(self, df: pd.DataFrame, model_dict: dict, condition: str):
        self.df = df
        self.condition = condition
        self.model_dict = model_dict
        self.model = self._load_model(model_dict, condition)
        self.genres, self.fixed_features = self._discover_features()
        self.feature_names = self.fixed_features + self.genres

    # ✅ FIXED MODEL LOADER
    def _load_model(self, model_dict, condition):

        key = condition.lower()

        if key not in model_dict:
            raise KeyError(f"Model for '{condition}' missing from model_dict")

        model_or_path = model_dict[key]

        # Already loaded model
        if not isinstance(model_or_path, str):
            print(f"Loaded in-memory model for {condition}")
            return model_or_path

        # Resolve path
        model_path = os.path.abspath(model_or_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        ext = os.path.splitext(model_path)[1].lower()

        # ✅ FIXED: if/elif indentation bug removed
        if ext in [".json", ".bst"]:
            model = xgb.XGBClassifier()
            model.load_model(model_path)

        elif ext in [".pkl", ".joblib"]:
            model = joblib.load(model_path)

        else:
            raise ValueError(f"Unsupported model format: {ext}")

        print(f"Loaded XGBoost model from: {model_path}")
        return model


    def _discover_features(self):
        possible_genres = [
            'Frequency [Classical]','Frequency [Country]','Frequency [EDM]','Frequency [Folk]',
            'Frequency [Hip-hop]','Frequency [Jazz]','Frequency [Pop]',
            'Frequency [Latin]','Frequency [Lofi]','Frequency [Metal]','Frequency [R&B]','Frequency [Rap]','Frequency [Rock]','Frequency [Video Game Music]','Frequency [Other]'
        ]
        colmap = {c.lower(): c for c in self.df.columns}
        genres = [colmap[g] for g in possible_genres if g in colmap]
        fixed = [c for c in self.df.columns if c not in genres]
        return genres, fixed


    # ✅ IMPORTANT FIX — CLASSIFIER-SAFE OBJECTIVE
    def predict(self, profile: dict):

        x_df = pd.DataFrame([[profile[f] for f in self.feature_names]], columns=self.feature_names)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_df)[0]
            return float(np.max(probs))
        
        return float(self.model.predict(x_df)[0])



    # ✅ OPTUNA BAYESIAN OPTIMISATION
    def optimise_genres(self, user_profile: dict, n_trials=40):

        def objective(trial):
            full = user_profile.copy()

            for g in self.genres:
                full[g] = trial.suggest_float(g, 0.0, 1.0)

            return self.predict(full)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params, study.best_value, study


    def recommend(self, original_profile: dict, optimized_genres: dict, top_genres=3):

        diffs = {g: optimized_genres[g] - original_profile[g] for g in self.genres}

        inc = sorted(diffs.items(), key=lambda x: -x[1])[:top_genres]
        dec = sorted(diffs.items(), key=lambda x: x[1])[:top_genres]

        return [g for g, _ in inc], [g for g, _ in dec]


    def run(self, user_profile: dict, n_calls=40):

        original_score = self.predict(user_profile)

        best_genres, improved_score, _ = self.optimise_genres(
            user_profile, n_trials=n_calls
        )

        optimised_profile = user_profile.copy()
        optimised_profile.update(best_genres)

        try_more, avoid = self.recommend(user_profile, best_genres)
        return {
            "original_score": original_score,
            "improved_score": improved_score,
            "try_more": try_more,
            "avoid": avoid,
            "optimised_profile": optimised_profile
        }
    
if __name__ == "__main__":
    print("This module provides the MusicProfileOptimiser class for optimising music listening profiles.")
    print("Import and instantiate MusicProfileOptimiser with your data and models to use it.")