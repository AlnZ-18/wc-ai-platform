import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, classification_report, log_loss
from xgboost import XGBClassifier

def load_processed_data():
    print("Loading processed training dataset...")
    df = pd.read_csv("datasets/processed/model_training_ready.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

def train_models():
    df = load_processed_data()
    
    # Chronological train/test split (highly professional sports forecasting practice)
    # Split: train on matches before 2025-06-01, test on matches after
    split_date = pd.to_datetime("2025-06-01")
    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]
    
    if len(test_df) == 0:
        # Fallback to standard split if dates don't span far enough
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
        
    print(f"Train set size: {len(train_df)} matches")
    print(f"Test set size: {len(test_df)} matches")
    
    # ----------------------------------------------------
    # 1. Train Double Poisson Regression Models for Goals
    # ----------------------------------------------------
    print("\nTraining Double Poisson Goal Expectancy Models...")
    
    # Features for home goals prediction
    home_goal_features = [
        "home_elo", "away_elo", "fifa_points_diff",
        "squad_rating_diff", "squad_fwd_diff", "squad_def_diff", "neutral"
    ]
    
    # Home goals model
    poisson_home = PoissonRegressor(alpha=1e-5, max_iter=300)
    poisson_home.fit(train_df[home_goal_features], train_df["home_score"])
    
    # Away goals model
    # We swap terms perspective so features represent the away perspective
    away_goal_features = [
        "away_elo", "home_elo", "fifa_points_diff",
        "squad_rating_diff", "squad_fwd_diff", "squad_def_diff", "neutral"
    ]
    # For away team, features like points diff and squad diff are negated to match away perspective
    train_away_features = train_df[away_goal_features].copy()
    train_away_features["fifa_points_diff"] = -train_away_features["fifa_points_diff"]
    train_away_features["squad_rating_diff"] = -train_away_features["squad_rating_diff"]
    train_away_features["squad_fwd_diff"] = -train_away_features["squad_fwd_diff"]
    train_away_features["squad_def_diff"] = -train_away_features["squad_def_diff"]
    
    poisson_away = PoissonRegressor(alpha=1e-5, max_iter=300)
    poisson_away.fit(train_away_features, train_df["away_score"])
    
    # Evaluate Poisson model predictions
    test_away_features = test_df[away_goal_features].copy()
    test_away_features["fifa_points_diff"] = -test_away_features["fifa_points_diff"]
    test_away_features["squad_rating_diff"] = -test_away_features["squad_rating_diff"]
    test_away_features["squad_fwd_diff"] = -test_away_features["squad_fwd_diff"]
    test_away_features["squad_def_diff"] = -test_away_features["squad_def_diff"]
    
    pred_xg_home = poisson_home.predict(test_df[home_goal_features])
    pred_xg_away = poisson_away.predict(test_away_features)
    
    print(f"Poisson Model - Mean Squared Error (Home Goals): {np.mean((test_df['home_score'] - pred_xg_home)**2):.4f}")
    print(f"Poisson Model - Mean Squared Error (Away Goals): {np.mean((test_df['away_score'] - pred_xg_away)**2):.4f}")
    
    # ----------------------------------------------------
    # 2. Train XGBoost Outcome Model (W / D / L)
    # ----------------------------------------------------
    print("\nTraining XGBoost Outcome Model...")
    
    # Complete feature set for match outcome prediction
    outcome_features = [
        "home_elo", "away_elo", "fifa_rank_diff", "fifa_points_diff", "neutral",
        "home_form_goals_scored", "home_form_goals_conceded", "home_form_win_rate",
        "away_form_goals_scored", "away_form_goals_conceded", "away_form_win_rate",
        "h2h_win_rate", "h2h_goals_diff",
        "squad_rating_diff", "squad_fwd_diff", "squad_def_diff", "squad_mid_diff"
    ]
    
    # Target outcome: map [-1, 0, 1] to [2, 0, 1] for XGBoost requirements
    # 0 = Draw, 1 = Home Win, 2 = Away Win
    y_train = train_df["outcome"].map({0: 0, 1: 1, -1: 2})
    y_test = test_df["outcome"].map({0: 0, 1: 1, -1: 2})
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    
    xgb_model.fit(
        train_df[outcome_features], y_train,
        eval_set=[(test_df[outcome_features], y_test)],
        verbose=False
    )
    
    # Evaluate XGBoost Model
    preds = xgb_model.predict(test_df[outcome_features])
    probs = xgb_model.predict_proba(test_df[outcome_features])
    
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Log Loss: {log_loss(y_test, probs):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Draw", "Home Win", "Away Win"]))
    
    # Feature Importance
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop Feature Importances (XGBoost):")
    for f in range(min(10, len(outcome_features))):
        print(f"{f + 1}. {outcome_features[indices[f]]}: {importances[indices[f]]:.4f}")
        
    # ----------------------------------------------------
    # 3. Save Models and Metadata
    # ----------------------------------------------------
    print("\nSerializing and saving trained models...")
    os.makedirs("backend/ml/models", exist_ok=True)
    
    # Save Poisson models
    with open("backend/ml/models/poisson_home.pkl", "wb") as f:
        pickle.dump(poisson_home, f)
    with open("backend/ml/models/poisson_away.pkl", "wb") as f:
        pickle.dump(poisson_away, f)
        
    # Save XGBoost model
    with open("backend/ml/models/xgboost_outcome.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
        
    # Save feature columns metadata for inference matching
    metadata = {
        "home_goal_features": home_goal_features,
        "away_goal_features": away_goal_features,
        "outcome_features": outcome_features
    }
    with open("backend/ml/models/model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
        
    print("All models and metadata successfully saved in backend/ml/models/")

if __name__ == "__main__":
    train_models()
