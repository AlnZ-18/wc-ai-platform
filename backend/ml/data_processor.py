import os
import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    print("Loading raw datasets...")
    results = pd.read_csv("datasets/raw/international_results.csv")
    rankings = pd.read_csv("datasets/raw/fifa_rankings.csv")
    players = pd.read_csv("datasets/raw/player_ratings_fc26.csv")
    
    results["date"] = pd.to_datetime(results["date"])
    rankings["date"] = pd.to_datetime(rankings["date"])
    
    return results, rankings, players

def compute_elo(results):
    print("Computing dynamic Elo ratings...")
    # Initialize all teams at 1500 Elo
    elo_ratings = {}
    
    # Store history of Elos for mapping features
    elo_history = []
    
    # Sort results chronologically
    results = results.sort_values("date").reset_index(drop=True)
    
    for idx, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        
        # Get current Elo or default to 1500
        elo_h = elo_ratings.get(home, 1500.0)
        elo_a = elo_ratings.get(away, 1500.0)
        
        # Append before-match Elo history to avoid data leakage
        elo_history.append({
            "match_idx": idx,
            "home_elo": elo_h,
            "away_elo": elo_a
        })
        
        # Calculate expected outcomes
        expected_h = 1.0 / (1.0 + 10.0 ** ((elo_a - elo_h) / 400.0))
        expected_a = 1.0 / (1.0 + 10.0 ** ((elo_h - elo_a) / 400.0))
        
        # Actual outcomes
        h_goals = row["home_score"]
        a_goals = row["away_score"]
        
        if h_goals > a_goals:
            score_h, score_a = 1.0, 0.0
        elif h_goals < a_goals:
            score_h, score_a = 0.0, 1.0
        else:
            score_h, score_a = 0.5, 0.5
            
        # Determine K-factor based on tournament
        tourney = str(row["tournament"]).lower()
        if "friendly" in tourney:
            k = 30
        elif "world cup" in tourney and "qual" not in tourney:
            k = 60
        else:
            k = 45 # Qualifiers, continental tournaments, nations league
            
        # Goal difference multiplier to reward big wins
        goal_diff = abs(h_goals - a_goals)
        g_multiplier = 1.0
        if goal_diff == 2:
            g_multiplier = 1.5
        elif goal_diff >= 3:
            g_multiplier = 1.75 + (goal_diff - 3) / 8.0
            
        # Update Elos
        elo_ratings[home] = elo_h + k * g_multiplier * (score_h - expected_h)
        elo_ratings[away] = elo_a + k * g_multiplier * (score_a - expected_a)
        
    elo_df = pd.DataFrame(elo_history)
    results = results.join(elo_df.set_index("match_idx"))
    return results, elo_ratings

def compute_rolling_features(results):
    print("Computing rolling form features...")
    # Sort chronologically
    results = results.sort_values("date").reset_index(drop=True)
    
    # Store features
    recent_features = []
    
    # Track match history per team
    team_history = {}
    
    # Track head-to-head match history between specific pairs
    h2h_history = {}
    
    for idx, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        
        # 1. Fetch rolling stats for Home team (prior to this match)
        home_stats = team_history.get(home, [])
        if len(home_stats) > 0:
            recent_h = home_stats[-5:] # last 5 matches
            goals_scored_h = np.mean([x["goals_scored"] for x in recent_h])
            goals_conceded_h = np.mean([x["goals_conceded"] for x in recent_h])
            win_rate_h = np.mean([x["win"] for x in recent_h])
        else:
            goals_scored_h = 1.2 # Default baseline
            goals_conceded_h = 1.2
            win_rate_h = 0.33
            
        # 2. Fetch rolling stats for Away team (prior to this match)
        away_stats = team_history.get(away, [])
        if len(away_stats) > 0:
            recent_a = away_stats[-5:]
            goals_scored_a = np.mean([x["goals_scored"] for x in recent_a])
            goals_conceded_a = np.mean([x["goals_conceded"] for x in recent_a])
            win_rate_a = np.mean([x["win"] for x in recent_a])
        else:
            goals_scored_a = 1.0
            goals_conceded_a = 1.2
            win_rate_a = 0.33
            
        # 3. Head to Head statistics (prior to this match)
        pair = tuple(sorted([home, away]))
        h2h_matches = h2h_history.get(pair, [])
        if len(h2h_matches) > 0:
            # We need to map perspective to home team
            h2h_win_rate = np.mean([
                1.0 if (m["winner"] == home) else (0.5 if m["winner"] == "Draw" else 0.0)
                for m in h2h_matches
            ])
            h2h_goals_diff = np.mean([
                (m["home_score"] - m["away_score"]) if m["home"] == home else (m["away_score"] - m["home_score"])
                for m in h2h_matches
            ])
        else:
            h2h_win_rate = 0.5
            h2h_goals_diff = 0.0
            
        recent_features.append({
            "match_idx": idx,
            "home_form_goals_scored": goals_scored_h,
            "home_form_goals_conceded": goals_conceded_h,
            "home_form_win_rate": win_rate_h,
            "away_form_goals_scored": goals_scored_a,
            "away_form_goals_conceded": goals_conceded_a,
            "away_form_win_rate": win_rate_a,
            "h2h_win_rate": h2h_win_rate,
            "h2h_goals_diff": h2h_goals_diff
        })
        
        # 4. Update histories *after* computing the features for the current match
        # Home team perspective
        h_score = row["home_score"]
        a_score = row["away_score"]
        
        team_history.setdefault(home, []).append({
            "goals_scored": h_score,
            "goals_conceded": a_score,
            "win": 1.0 if h_score > a_score else (0.5 if h_score == a_score else 0.0)
        })
        
        # Away team perspective
        team_history.setdefault(away, []).append({
            "goals_scored": a_score,
            "goals_conceded": h_score,
            "win": 1.0 if a_score > h_score else (0.5 if h_score == a_score else 0.0)
        })
        
        # H2H perspective
        h2h_history.setdefault(pair, []).append({
            "home": home,
            "away": away,
            "home_score": h_score,
            "away_score": a_score,
            "winner": home if h_score > a_score else (away if a_score > h_score else "Draw")
        })
        
    features_df = pd.DataFrame(recent_features)
    results = results.join(features_df.set_index("match_idx"))
    return results

def get_closest_ranking(team, date, rankings):
    # Filter for the team and get dates before or equal to match date
    team_ranks = rankings[(rankings["country"] == team) & (rankings["date"] <= date)]
    if team_ranks.empty:
        # Fallback to absolute earliest or default rank
        all_team_ranks = rankings[rankings["country"] == team]
        if all_team_ranks.empty:
            return 50, 1400.0 # Default rank, points
        return all_team_ranks["rank"].iloc[0], all_team_ranks["points"].iloc[0]
    
    # Sort by date descending and get the closest one
    closest = team_ranks.sort_values("date", ascending=False).iloc[0]
    return closest["rank"], closest["points"]

def merge_rankings(results, rankings):
    print("Merging FIFA rankings...")
    rank_features = []
    
    for idx, row in results.iterrows():
        h_rank, h_points = get_closest_ranking(row["home_team"], row["date"], rankings)
        a_rank, a_points = get_closest_ranking(row["away_team"], row["date"], rankings)
        
        rank_features.append({
            "match_idx": idx,
            "home_fifa_rank": h_rank,
            "home_fifa_points": h_points,
            "away_fifa_rank": a_rank,
            "away_fifa_points": a_points,
            "fifa_rank_diff": h_rank - a_rank,
            "fifa_points_diff": h_points - a_points
        })
        
    rank_df = pd.DataFrame(rank_features)
    results = results.join(rank_df.set_index("match_idx"))
    return results

def aggregate_player_features(players):
    print("Aggregating player attributes per country...")
    # Group players by country and compute squad averages
    aggregated = players.groupby("country").agg(
        squad_rating_avg=("overall_rating", "mean"),
        squad_stamina_avg=("stamina", "mean"),
        squad_injury_risk_avg=("injury_risk", "mean"),
        squad_goals_total=("goals_last_season", "sum"),
        squad_assists_total=("assists_last_season", "mean")
    ).reset_index()
    
    # Aggregated positions strength
    pos_strength = players.groupby(["country", "position"])["overall_rating"].mean().unstack(fill_value=70).reset_index()
    pos_strength = pos_strength.rename(columns={
        "GK": "squad_gk_rating",
        "DEF": "squad_def_rating",
        "MID": "squad_mid_rating",
        "FWD": "squad_fwd_rating"
    })
    
    squad_features = pd.merge(aggregated, pos_strength, on="country", how="left")
    return squad_features

def merge_player_features(results, squad_features):
    print("Merging player aggregated features into matches...")
    # Add home squad features
    home_squad = squad_features.rename(columns={col: f"home_{col}" for col in squad_features.columns if col != "country"})
    results = pd.merge(results, home_squad, left_on="home_team", right_on="country", how="left").drop(columns=["country"])
    
    # Add away squad features
    away_squad = squad_features.rename(columns={col: f"away_{col}" for col in squad_features.columns if col != "country"})
    results = pd.merge(results, away_squad, left_on="away_team", right_on="country", how="left").drop(columns=["country"])
    
    # Fill any NaNs with reasonable baselines
    results = results.fillna({
        "home_squad_rating_avg": 75.0, "away_squad_rating_avg": 75.0,
        "home_squad_stamina_avg": 80.0, "away_squad_stamina_avg": 80.0,
        "home_squad_injury_risk_avg": 20.0, "away_squad_injury_risk_avg": 20.0,
        "home_squad_goals_total": 10.0, "away_squad_goals_total": 10.0,
        "home_squad_gk_rating": 75.0, "away_squad_gk_rating": 75.0,
        "home_squad_def_rating": 75.0, "away_squad_def_rating": 75.0,
        "home_squad_mid_rating": 75.0, "away_squad_mid_rating": 75.0,
        "home_squad_fwd_rating": 75.0, "away_squad_fwd_rating": 75.0,
    })
    
    # Create diff features
    results["squad_rating_diff"] = results["home_squad_rating_avg"] - results["away_squad_rating_avg"]
    results["squad_fwd_diff"] = results["home_squad_fwd_rating"] - results["away_squad_fwd_rating"]
    results["squad_def_diff"] = results["home_squad_def_rating"] - results["away_squad_def_rating"]
    results["squad_mid_diff"] = results["home_squad_mid_rating"] - results["away_squad_mid_rating"]
    
    return results

def save_processed_data(results):
    print("Saving processed dataset to datasets/processed/...")
    # Add target outcome column: 1 for Home win, 0 for Draw, -1 for Away win
    conditions = [
        results["home_score"] > results["away_score"],
        results["home_score"] == results["away_score"],
        results["home_score"] < results["away_score"]
    ]
    choices = [1, 0, -1] # 1 = Home Win, 0 = Draw, -1 = Away Win
    results["outcome"] = np.select(conditions, choices, default=0)
    
    filepath = "datasets/processed/model_training_ready.csv"
    results.to_csv(filepath, index=False)
    print(f"Successfully processed {len(results)} matches. File saved to {filepath}.")

if __name__ == "__main__":
    results, rankings, players = load_data()
    
    # Build ML pipeline features step-by-step
    results, final_elos = compute_elo(results)
    results = compute_rolling_features(results)
    results = merge_rankings(results, rankings)
    
    squad_features = aggregate_player_features(players)
    results = merge_player_features(results, squad_features)
    
    # Save training ready file
    save_processed_data(results)
    
    # Also save the final Elo ratings for deployment prediction usage later
    elo_df = pd.DataFrame(list(final_elos.items()), columns=["country", "elo"])
    elo_df.to_csv("datasets/processed/latest_team_elos.csv", index=False)
    print("Latest Elo ratings saved to datasets/processed/latest_team_elos.csv.")
