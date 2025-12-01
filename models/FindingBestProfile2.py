# Given the 4 models each predicting mental health score in one of OCD, Insomnia, Depression,
# and Anxiety, this script tries to finds the best genre profile using the optuna Bayesian Optimisation library.
# The best profile is the one that minismises the projected mental health score for the model
# corresponding to that condition that the individual would like to improve.
# The output will be formatted to suggest up to 3 genres the individual can try based on the best profile found
# as well up to 3 that they should avoid
# The Optuna bayesian optimisation will only adjust the genre features, keeping other features fixed as per the user's profile.
# To perform this, the user profile is to be passed as a pandas Series.
# The condition we wish to minimise for is also to be specified, which is mapped to a dictionary containing the 4 models
# Model can be either XGBoost, Catboost, or MLP Classifers

import optuna
import numpy as np
import pandas as pd


# Method to extract genre features from user profile, which are variables to be optimised
def extract_genre_features(profile: pd.Series):
    """
    Identify genre frequency columns in the profile.

    By convention these columns are named like "Frequency [GenreName]".
    Only these columns will be considered for optimisation unless the
    optimisation function is explicitly extended.
    """
    return [col for col in profile.index if col.startswith("Frequency [") and col.endswith("]")]



# Create objective function for Optuna study
def create_objective(user_profile: pd.Series, model, genre_features):

    # Objective function to minimise predicted mental health score
    def objective(trial):
        profile_copy = user_profile.copy() # Create a copy of user profile

        # Sample genre frequencies in the range 1.0 to 4.0
        # We enforce that exactly one genre is set to the maximum value (4.0)
        # per trial. To do this we: (a) ask Optuna to pick one genre as the
        # "max_genre" via `suggest_categorical`, (b) set that genre to 4.0,
        # and (c) suggest continuous values in [1.0, 3.9999] for the remaining
        # genres so they cannot also be 4.0.
        # This yields a valid search space where only one feature can be 4.0.
        max_genre = trial.suggest_categorical("max_genre", genre_features)

        for gf in genre_features:
            if gf == max_genre:
                # Force the chosen genre to the absolute maximum 4.0
                profile_copy[gf] = 4.0
            else:
                # Other genres are allowed between 1.0 and just-under-4.0
                # (use 3.9999 to avoid floating equality to 4.0)
                profile_copy[gf] = trial.suggest_float(gf, 1.0, 3.9999)

        # Model expects input as 2D shape
        input_df = profile_copy.to_frame().T

        # Predict — supports XGB, CatBoost, MLPClassifier (predict_proba or predict)
        # We return a continuous score that Optuna minimises. By default we
        # take the model's predicted probability for the positive/high class
        # if `predict_proba` is available, otherwise fall back to `predict`.
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(input_df)[0][1]  # Probability of "high mental health score"
        else:
            pred = model.predict(input_df)[0]

        return float(pred)

    return objective



# Method to run the optimisation for a given user profile and condition
def optimise_genre_profile(
    user_profile: pd.Series,
    condition: str,
    model_dict: dict,
    n_trials: int = 50,
):

    if condition not in model_dict:
        raise ValueError(f"Condition '{condition}' not found in model_dict keys.")

    model = model_dict[condition]

    # Identify which columns are genre frequency features
    genre_features = extract_genre_features(user_profile)

    # Create the Optuna objective with the constrained search space
    objective = create_objective(user_profile, model, genre_features)

    # Run the study: we minimise the model's returned probability of predicted high
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Retrieve best trial values
    best_params = study.best_params

    # Construct best profile: best_params contains the sampled floats for
    # each non-max genre plus the chosen 'max_genre'. If 'max_genre' is
    # present it indicates which genre to set to 4.0.
    best_profile = user_profile.copy()
    # If Optuna recorded a chosen max_genre, set that genre to 4.0
    max_genre_choice = best_params.get("max_genre")
    for gf in genre_features:
        if gf in best_params:
            best_profile[gf] = best_params[gf]
    if max_genre_choice is not None:
        best_profile[max_genre_choice] = 4.0

    # Sort genres by high and low values
    sorted_genres = sorted(
        [(gf, best_profile[gf]) for gf in genre_features],
        key=lambda x: x[1],
        reverse=True,
    )

    try_genres = [g for g, _ in sorted_genres[:3]]
    avoid_genres = [g for g, _ in sorted_genres[-3:]]

    return best_profile, try_genres, avoid_genres

