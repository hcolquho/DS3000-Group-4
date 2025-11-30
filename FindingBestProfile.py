# Given the 4 models each predicting mental health score in one of OCD, Insomnia, Depression,
# and Anxiety, this script tries to finds the best profile using Bayesian Optimisation.
# The best profile is the one that minismises the projected mental health score for the model
# corresponding to that condition that the individual would like to improve.
# Output will be formatted to suggest up to 3 genres the individual can try based on the best profile found
# as well up to 3 that they should avoid.

# Please make sure to have the required libraries installed
import numpy as np
import pandas as pd 
from catboost import CatBoostRegressor 
from skopt import gp_minimize # Bayesian optimisation function (fitting Gaussian process to observations)
from skopt.space import Real # To define the search space for optimisation

class MusicProfileOptimiser:

    # Initialise optimiser object with dataframe, model dictionary and condition
    def __init__(self, df: pd.DataFrame, model_dict: dict, condition: str):
        self.df = df # DataFrame with user data in the same format as training data
        self.condition = condition
        self.model_dict = model_dict
        self.model = self._load_model(model_dict, condition) # Load model for specified condition to improve
        self.genres, self.fixed_features = self._discover_features() # Identify person's genre and fixed features
        self.feature_names = self.fixed_features + self.genres # Order of features for model input

    # Load model for specified condition
    def _load_model(self, model_dict, condition):
        key = condition.lower() # Ensure case-insensitive matching
        if key not in model_dict:
            raise KeyError(f"Oops. Model for predicting '{condition}' missing. Cannot recommend genres to user.") # Error handling for missing model
        else:
            print(f"Loaded model for condition: {condition}") # Confirmation message
        return model_dict[key] # Return the requested model object from the catboost model dict
            
    # Identify genre and fixed features from user DataFrame
    def _discover_features(self):

        # Identify genre features
        possible_genres = ['classical','country','edm','folk','gospel','hip-hop','jazz','pop','latin','lo-fi',
                           'metal','r&b','rap','rock','video game music','other']
        colmap = {c.lower(): c for c in self.df.columns} # Map lowercased column names to original
        genres = [colmap[g] for g in possible_genres if g in colmap] # Existing genre columns

        # Fixed features = all non-genre numerical/binary columns
        fixed = [c for c in self.df.columns if c not in genres] # Non-genre features are fixed
        return genres, fixed # Return lists of genre and fixed features
    

    # Predict the desired mental health score for a given user profile
    def predict(self, profile: dict):
        x = np.array([profile[f] for f in self.feature_names], dtype=float).reshape(1, -1) # Prepare input array
        return float(self.model.predict(x)[0]) # Return predicted score as float

    # Use Bayesian optimisation technique to find best genre profile
    def optimise_genres(self, user_profile: dict, n_calls=40): 

        # Vary only genre features. Fixed features constant from user_profile
        def objective(genre_vals):
            full = user_profile.copy() # Start with original profile
            for g, v in zip(self.genres, genre_vals): # Update genre values
                full[g] = v # Set genre to current optimization value
            return self.predict(full) # Return predicted score for full profile

        # Define search space for genres (0 to 1), representing listening frequency
        space = [Real(0.0, 1.0) for _ in self.genres] 

        # Run Bayesian optimisation to minimise the objective function
        res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

        # Extract best genre values and corresponding score
        best_genres = {g: float(v) for g, v in zip(self.genres, res.x)}
        return best_genres, float(res.fun), res

    # Produce final recommendations based on differences between original and optimised profiles
    def recommend(self, original_profile: dict, optimized_genres: dict, top_genres=3): # up to 3 genres each to recommend/avoid
        diffs = {g: optimized_genres[g] - original_profile[g] for g in self.genres} # Calculate differences
        inc = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:top_genres] # Top recommended genre lstening frequency increases
        dec = sorted(diffs.items(), key=lambda x: x[1])[:top_genres] # Top recommended grenre listening frequency decreases
        return [g for g, _ in inc], [g for g, _ in dec] # Return genre lists

    # Main method to run the full grenre optimisation and recommendation process
    def run(self, user_profile: dict, n_calls=40):
        # Predict original score
        original_score = self.predict(user_profile)

        # Optimise genre for better predicted score
        best_genres, improved_score, _ = self.optimise_genres(user_profile, n_calls=n_calls)

        # Build optimised profile
        optimised_profile = user_profile.copy() # start with original profile
        optimised_profile.update(best_genres) # update with best genre values

        # Recommendations
        try_more, avoid = self.recommend(user_profile, best_genres)

        # Return results (condition to improve, orginal score, improved score, genres to try, genres to avoid) as dictionary
        return {
            "MH Condition to improve": self.condition,
            "Original Predicted Score": original_score,
            "Improved Score": improved_score,
            "Genres to Try": try_more,
            "Genres to Avoid": avoid
        }
