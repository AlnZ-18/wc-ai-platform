import os
import pickle
import random
import json
import pandas as pd
import numpy as np

# Define groups for the 48-team 2026 World Cup
GROUPS = {
    "Group A": ["USA", "Mexico", "Canada", "New Zealand"],
    "Group B": ["Argentina", "Uruguay", "Egypt", "Iran"],
    "Group C": ["France", "Denmark", "Senegal", "Saudi Arabia"],
    "Group D": ["Brazil", "Switzerland", "Morocco", "Iraq"],
    "Group E": ["England", "Ukraine", "Tunisia", "Japan"],
    "Group F": ["Spain", "Austria", "Poland", "South Korea"],
    "Group G": ["Portugal", "Croatia", "Chile", "South Africa"],
    "Group H": ["Belgium", "Colombia", "Algeria", "Australia"],
    "Group I": ["Netherlands", "Turkey", "Peru", "Uzbekistan"],
    "Group J": ["Italy", "Sweden", "Nigeria", "Qatar"],
    "Group K": ["Germany", "Ecuador", "Cameroon", "Jamaica"],
    "Group L": ["Costa Rica", "Panama", "Ghana", "Wales"]
}

class TournamentSimulator:
    def __init__(self):
        print("Initializing Tournament Simulator...")
        self.load_models()
        self.load_team_features()
        
    def load_models(self):
        with open("backend/ml/models/poisson_home.pkl", "rb") as f:
            self.poisson_home = pickle.load(f)
        with open("backend/ml/models/poisson_away.pkl", "rb") as f:
            self.poisson_away = pickle.load(f)
        with open("backend/ml/models/xgboost_outcome.pkl", "rb") as f:
            self.xgboost_outcome = pickle.load(f)
        with open("backend/ml/models/model_metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
            
    def load_team_features(self):
        # 1. Load Elos
        elo_df = pd.read_csv("datasets/processed/latest_team_elos.csv")
        self.team_elos = dict(zip(elo_df["country"], elo_df["elo"]))
        
        # 2. Load latest FIFA Rankings
        rankings = pd.read_csv("datasets/raw/fifa_rankings.csv")
        rankings["date"] = pd.to_datetime(rankings["date"])
        latest_date = rankings["date"].max()
        latest_ranks = rankings[rankings["date"] == latest_date]
        self.fifa_ranks = dict(zip(latest_ranks["country"], latest_ranks["rank"]))
        self.fifa_points = dict(zip(latest_ranks["country"], latest_ranks["points"]))
        
        # 3. Load processed training data to extract latest form statistics per team
        df = pd.read_csv("datasets/processed/model_training_ready.csv")
        df["date"] = pd.to_datetime(df["date"])
        
        self.team_form = {}
        for team in self.team_elos.keys():
            # Get latest matches for the team
            team_matches = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date")
            if not team_matches.empty:
                latest_match = team_matches.iloc[-1]
                if latest_match["home_team"] == team:
                    goals_scored = latest_match["home_form_goals_scored"]
                    goals_conceded = latest_match["home_form_goals_conceded"]
                    win_rate = latest_match["home_form_win_rate"]
                else:
                    goals_scored = latest_match["away_form_goals_scored"]
                    goals_conceded = latest_match["away_form_goals_conceded"]
                    win_rate = latest_match["away_form_win_rate"]
            else:
                goals_scored, goals_conceded, win_rate = 1.2, 1.2, 0.33
                
            self.team_form[team] = {
                "goals_scored": goals_scored,
                "goals_conceded": goals_conceded,
                "win_rate": win_rate
            }
            
        # 4. Load player features
        players = pd.read_csv("datasets/raw/player_ratings_fc26.csv")
        # Reuse processor logic to aggregate
        from data_processor import aggregate_player_features
        squad_features = aggregate_player_features(players)
        self.squad_features = squad_features.set_index("country").to_dict(orient="index")
        
    def get_match_features(self, home, away):
        # Fetch ELO
        h_elo = self.team_elos.get(home, 1500.0)
        a_elo = self.team_elos.get(away, 1500.0)
        
        # Fetch Rankings
        h_rank = self.fifa_ranks.get(home, 50)
        a_rank = self.fifa_ranks.get(away, 50)
        h_points = self.fifa_points.get(home, 1400.0)
        a_points = self.fifa_points.get(away, 1400.0)
        
        # Fetch Form
        h_form = self.team_form.get(home, {"goals_scored": 1.2, "goals_conceded": 1.2, "win_rate": 0.33})
        a_form = self.team_form.get(away, {"goals_scored": 1.2, "goals_conceded": 1.2, "win_rate": 0.33})
        
        # Fetch Squad ratings
        h_squad = self.squad_features.get(home, {
            "squad_rating_avg": 75.0, "squad_fwd_rating": 75.0, "squad_def_rating": 75.0, "squad_mid_rating": 75.0
        })
        a_squad = self.squad_features.get(away, {
            "squad_rating_avg": 75.0, "squad_fwd_rating": 75.0, "squad_def_rating": 75.0, "squad_mid_rating": 75.0
        })
        
        features = {
            "home_elo": h_elo,
            "away_elo": a_elo,
            "fifa_rank_diff": h_rank - a_rank,
            "fifa_points_diff": h_points - a_points,
            "neutral": 1.0, # All World Cup matches are neutral venues
            
            "home_form_goals_scored": h_form["goals_scored"],
            "home_form_goals_conceded": h_form["goals_conceded"],
            "home_form_win_rate": h_form["win_rate"],
            
            "away_form_goals_scored": a_form["goals_scored"],
            "away_form_goals_conceded": a_form["goals_conceded"],
            "away_form_win_rate": a_form["win_rate"],
            
            "h2h_win_rate": 0.5, # Baseline default
            "h2h_goals_diff": 0.0,
            
            "squad_rating_diff": h_squad["squad_rating_avg"] - a_squad["squad_rating_avg"],
            "squad_fwd_diff": h_squad["squad_fwd_rating"] - a_squad["squad_fwd_rating"],
            "squad_def_diff": h_squad["squad_def_rating"] - a_squad["squad_def_rating"],
            "squad_mid_diff": h_squad["squad_mid_rating"] - a_squad["squad_mid_rating"]
        }
        
        # Create a 1-row DataFrame
        df = pd.DataFrame([features])
        return df[self.metadata["outcome_features"]]
        
    def simulate_match(self, home, away, is_knockout=False):
        features = self.get_match_features(home, away)
        
        # Predict outcome probabilities using XGBoost
        # Class mapping: 0 = Draw, 1 = Home Win, 2 = Away Win
        probs = self.xgboost_outcome.predict_proba(features)[0]
        
        draw_prob = probs[0]
        home_prob = probs[1]
        away_prob = probs[2]
        
        # Simulate outcome
        outcome = random.choices(
            population=["Draw", home, away],
            weights=[draw_prob, home_prob, away_prob]
        )[0]
        
        # Predict scores using Poisson regression (Expected goals)
        # Poisson features
        # Home goals features
        h_elo = self.team_elos.get(home, 1500.0)
        a_elo = self.team_elos.get(away, 1500.0)
        h_points = self.fifa_points.get(home, 1400.0)
        a_points = self.fifa_points.get(away, 1400.0)
        h_squad = self.squad_features.get(home, {"squad_rating_avg": 75.0, "squad_fwd_rating": 75.0, "squad_def_rating": 75.0})
        a_squad = self.squad_features.get(away, {"squad_rating_avg": 75.0, "squad_fwd_rating": 75.0, "squad_def_rating": 75.0})
        
        # Home goals features: ["home_elo", "away_elo", "fifa_points_diff", "squad_rating_diff", "squad_fwd_diff", "squad_def_diff", "neutral"]
        home_feats = pd.DataFrame([{
            "home_elo": h_elo, "away_elo": a_elo, "fifa_points_diff": h_points - a_points,
            "squad_rating_diff": h_squad["squad_rating_avg"] - a_squad["squad_rating_avg"],
            "squad_fwd_diff": h_squad["squad_fwd_rating"] - a_squad["squad_fwd_rating"],
            "squad_def_diff": h_squad["squad_def_rating"] - a_squad["squad_def_rating"],
            "neutral": 1.0
        }])
        
        # Away goals features: ["away_elo", "home_elo", "fifa_points_diff", "squad_rating_diff", "squad_fwd_diff", "squad_def_diff", "neutral"]
        # In train.py we negated these for the away perspective
        away_feats = pd.DataFrame([{
            "away_elo": a_elo, "home_elo": h_elo, "fifa_points_diff": -(h_points - a_points),
            "squad_rating_diff": -(h_squad["squad_rating_avg"] - a_squad["squad_rating_avg"]),
            "squad_fwd_diff": -(h_squad["squad_fwd_rating"] - a_squad["squad_fwd_rating"]),
            "squad_def_diff": -(h_squad["squad_def_rating"] - a_squad["squad_def_rating"]),
            "neutral": 1.0
        }])
        
        xg_home = self.poisson_home.predict(home_feats[self.metadata["home_goal_features"]])[0]
        xg_away = self.poisson_away.predict(away_feats[self.metadata["away_goal_features"]])[0]
        
        # Sample score based on xG (Poisson distribution)
        home_goals = np.random.poisson(xg_home)
        away_goals = np.random.poisson(xg_away)
        
        # Re-adjust scores to match outcome to ensure logical coherence
        if outcome == home and home_goals <= away_goals:
            home_goals = away_goals + random.randint(1, 2)
        elif outcome == away and away_goals <= home_goals:
            away_goals = home_goals + random.randint(1, 2)
        elif outcome == "Draw" and home_goals != away_goals:
            home_goals = away_goals
            
        # Tie breaker for knockout stage
        winner = outcome
        if is_knockout and outcome == "Draw":
            # Weighted random choice based on overall squad quality to simulate penalty shootout
            h_rating = h_squad.get("squad_rating_avg", 75.0)
            a_rating = a_squad.get("squad_rating_avg", 75.0)
            winner = random.choices([home, away], weights=[h_rating, a_rating])[0]
            
        return {
            "winner": winner,
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
            "probs": {
                "home": float(home_prob),
                "draw": float(draw_prob),
                "away": float(away_prob)
            }
        }

    def simulate_group_stage(self):
        advancing_teams = []
        group_standings = {}
        third_places = []
        
        for g_name, teams in GROUPS.items():
            standings = {t: {"points": 0, "goals_for": 0, "goals_against": 0, "goals_diff": 0, "name": t} for t in teams}
            
            # Sim round-robin matches (each plays 3 matches)
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    t1, t2 = teams[i], teams[j]
                    res = self.simulate_match(t1, t2, is_knockout=False)
                    
                    # Update goals
                    standings[t1]["goals_for"] += res["home_goals"]
                    standings[t1]["goals_against"] += res["away_goals"]
                    standings[t2]["goals_for"] += res["away_goals"]
                    standings[t2]["goals_against"] += res["home_goals"]
                    
                    # Update points
                    if res["winner"] == t1:
                        standings[t1]["points"] += 3
                    elif res["winner"] == t2:
                        standings[t2]["points"] += 3
                    else:
                        standings[t1]["points"] += 1
                        standings[t2]["points"] += 1
                        
            # Compute goal difference
            for t in teams:
                standings[t]["goals_diff"] = standings[t]["goals_for"] - standings[t]["goals_against"]
                
            # Sort standings by points, then goal diff, then goals for
            sorted_standings = sorted(
                standings.values(),
                key=lambda x: (x["points"], x["goals_diff"], x["goals_for"]),
                reverse=True
            )
            
            group_standings[g_name] = sorted_standings
            
            # Top 2 teams advance
            advancing_teams.append(sorted_standings[0]["name"])
            advancing_teams.append(sorted_standings[1]["name"])
            
            # 3rd place team goes to wild-card pool
            third_places.append(sorted_standings[2])
            
        # Select best 8 third-place teams (actual FIFA 2026 rules!)
        sorted_third = sorted(
            third_places,
            key=lambda x: (x["points"], x["goals_diff"], x["goals_for"]),
            reverse=True
        )
        for i in range(8):
            advancing_teams.append(sorted_third[i]["name"])
            
        return advancing_teams, group_standings

    def simulate_knockout_round(self, teams):
        next_round = []
        matches_played = []
        
        # Play matches in pairs
        for i in range(0, len(teams), 2):
            t1 = teams[i]
            t2 = teams[i+1]
            res = self.simulate_match(t1, t2, is_knockout=True)
            next_round.append(res["winner"])
            matches_played.append({
                "home": t1,
                "away": t2,
                "home_goals": res["home_goals"],
                "away_goals": res["away_goals"],
                "winner": res["winner"],
                "probs": res["probs"]
            })
            
        return next_round, matches_played

    def simulate_tournament(self):
        # 1. Group Stage
        advancing, group_standings = self.simulate_group_stage()
        
        # Shuffle advancing teams slightly for realistic bracket pairings (Round of 32 has 32 teams)
        random.shuffle(advancing)
        
        # 2. Round of 32
        r32_teams, r32_matches = self.simulate_knockout_round(advancing)
        
        # 3. Round of 16
        r16_teams, r16_matches = self.simulate_knockout_round(r32_teams)
        
        # 4. Quarter-finals
        qf_teams, qf_matches = self.simulate_knockout_round(r16_teams)
        
        # 5. Semi-finals
        sf_teams, sf_matches = self.simulate_knockout_round(qf_teams)
        
        # 6. Final
        winner_team, final_match = self.simulate_knockout_round(sf_teams)
        
        return {
            "winner": winner_team[0],
            "final": final_match[0],
            "semis": sf_matches,
            "quarters": qf_matches,
            "r16": r16_matches,
            "r32": r32_matches,
            "group_standings": group_standings
        }

    def run_monte_carlo(self, num_simulations=1000):
        print(f"Running {num_simulations} Monte Carlo tournament simulations...")
        
        # Accumulators
        win_counts = {}
        final_counts = {}
        semi_counts = {}
        quarter_counts = {}
        
        for i in range(num_simulations):
            if (i+1) % 100 == 0:
                print(f"Completed {i+1}/{num_simulations} simulations...")
            res = self.simulate_tournament()
            
            # Accumulate Winner
            w = res["winner"]
            win_counts[w] = win_counts.get(w, 0) + 1
            
            # Accumulate Finalists
            f1, f2 = res["final"]["home"], res["final"]["away"]
            final_counts[f1] = final_counts.get(f1, 0) + 1
            final_counts[f2] = final_counts.get(f2, 0) + 1
            
            # Accumulate Semi-finalists
            for m in res["semis"]:
                s1, s2 = m["home"], m["away"]
                semi_counts[s1] = semi_counts.get(s1, 0) + 1
                semi_counts[s2] = semi_counts.get(s2, 0) + 1
                
            # Accumulate Quarter-finalists
            for m in res["quarters"]:
                q1, q2 = m["home"], m["away"]
                quarter_counts[q1] = quarter_counts.get(q1, 0) + 1
                quarter_counts[q2] = quarter_counts.get(q2, 0) + 1
                
        # Convert to percentages
        aggregate_results = []
        for team in self.team_elos.keys():
            wins = win_counts.get(team, 0)
            finals = final_counts.get(team, 0)
            semis = semi_counts.get(team, 0)
            quarters = quarter_counts.get(team, 0)
            
            # Only include teams that achieved at least a quarter final in any sim to keep report clean
            if semis > 0 or wins > 0:
                aggregate_results.append({
                    "team": team,
                    "win_probability": round((wins / num_simulations) * 100, 2),
                    "final_probability": round((finals / num_simulations) * 100, 2),
                    "semi_probability": round((semis / num_simulations) * 100, 2),
                    "quarter_probability": round((quarters / num_simulations) * 100, 2),
                    "elo": int(self.team_elos.get(team, 1500.0)),
                    "fifa_rank": int(self.fifa_ranks.get(team, 50))
                })
                
        # Sort by win probability descending
        aggregate_results = sorted(aggregate_results, key=lambda x: x["win_probability"], reverse=True)
        
        # Save a single fully-detailed tournament run structure as a "sample path" for frontend animations
        sample_path = self.simulate_tournament()
        
        output_payload = {
            "num_simulations": num_simulations,
            "probabilities": aggregate_results,
            "sample_tournament_path": sample_path
        }
        
        filepath = "datasets/processed/tournament_simulation_results.json"
        with open(filepath, "w") as f:
            json.dump(output_payload, f, indent=2)
            
        print("\n=== TOP 10 PREDICTED WINNERS ===")
        for idx, item in enumerate(aggregate_results[:10]):
            print(f"{idx+1}. {item['team']}: {item['win_probability']}% chance to win (Rank: {item['fifa_rank']})")

if __name__ == "__main__":
    simulator = TournamentSimulator()
    simulator.run_monte_carlo(num_simulations=1000)
