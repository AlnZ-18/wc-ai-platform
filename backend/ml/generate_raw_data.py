import os
import csv
import random
from datetime import datetime, timedelta

def create_directories():
    os.makedirs("datasets/raw", exist_ok=True)
    os.makedirs("datasets/processed", exist_ok=True)
    os.makedirs("backend/ml/models", exist_ok=True)

# List of top teams, their base strength (for match score generation), base ranking, and confederation
TEAMS = {
    "Argentina": {"strength": 92, "rank": 1, "points": 1860, "confederation": "CONMEBOL"},
    "France": {"strength": 91, "rank": 2, "points": 1845, "confederation": "UEFA"},
    "Brazil": {"strength": 90, "rank": 5, "points": 1790, "confederation": "CONMEBOL"},
    "England": {"strength": 89, "rank": 4, "points": 1800, "confederation": "UEFA"},
    "Spain": {"strength": 88, "rank": 3, "points": 1810, "confederation": "UEFA"},
    "Portugal": {"strength": 87, "rank": 7, "points": 1750, "confederation": "UEFA"},
    "Netherlands": {"strength": 86, "rank": 8, "points": 1740, "confederation": "UEFA"},
    "Italy": {"strength": 85, "rank": 9, "points": 1725, "confederation": "UEFA"},
    "Germany": {"strength": 86, "rank": 11, "points": 1705, "confederation": "UEFA"},
    "Belgium": {"strength": 85, "rank": 6, "points": 1770, "confederation": "UEFA"},
    "Croatia": {"strength": 84, "rank": 10, "points": 1720, "confederation": "UEFA"},
    "Uruguay": {"strength": 85, "rank": 14, "points": 1665, "confederation": "CONMEBOL"},
    "Colombia": {"strength": 84, "rank": 12, "points": 1690, "confederation": "CONMEBOL"},
    "Morocco": {"strength": 83, "rank": 13, "points": 1675, "confederation": "CAF"},
    "USA": {"strength": 82, "rank": 16, "points": 1650, "confederation": "CONCACAF"},
    "Mexico": {"strength": 81, "rank": 15, "points": 1655, "confederation": "CONCACAF"},
    "Japan": {"strength": 83, "rank": 17, "points": 1630, "confederation": "AFC"},
    "Senegal": {"strength": 82, "rank": 18, "points": 1625, "confederation": "CAF"},
    "Iran": {"strength": 79, "rank": 20, "points": 1610, "confederation": "AFC"},
    "Denmark": {"strength": 81, "rank": 21, "points": 1600, "confederation": "UEFA"},
    "South Korea": {"strength": 80, "rank": 22, "points": 1590, "confederation": "AFC"},
    "Australia": {"strength": 78, "rank": 23, "points": 1580, "confederation": "AFC"},
    "Ukraine": {"strength": 80, "rank": 24, "points": 1575, "confederation": "UEFA"},
    "Austria": {"strength": 81, "rank": 25, "points": 1570, "confederation": "UEFA"},
    "Switzerland": {"strength": 81, "rank": 19, "points": 1615, "confederation": "UEFA"},
    "Turkey": {"strength": 80, "rank": 26, "points": 1560, "confederation": "UEFA"},
    "Ecuador": {"strength": 80, "rank": 27, "points": 1550, "confederation": "CONMEBOL"},
    "Poland": {"strength": 79, "rank": 28, "points": 1540, "confederation": "UEFA"},
    "Sweden": {"strength": 79, "rank": 29, "points": 1530, "confederation": "UEFA"},
    "Wales": {"strength": 78, "rank": 30, "points": 1520, "confederation": "UEFA"},
    "Egypt": {"strength": 79, "rank": 31, "points": 1515, "confederation": "CAF"},
    "Nigeria": {"strength": 80, "rank": 32, "points": 1510, "confederation": "CAF"},
    "Canada": {"strength": 78, "rank": 35, "points": 1495, "confederation": "CONCACAF"},
    "Saudi Arabia": {"strength": 76, "rank": 53, "points": 1440, "confederation": "AFC"},
    "Cameroon": {"strength": 77, "rank": 49, "points": 1460, "confederation": "CAF"},
    "Ghana": {"strength": 76, "rank": 64, "points": 1395, "confederation": "CAF"},
    "Tunisia": {"strength": 77, "rank": 41, "points": 1480, "confederation": "CAF"},
    "Chile": {"strength": 78, "rank": 40, "points": 1485, "confederation": "CONMEBOL"},
    "Peru": {"strength": 77, "rank": 42, "points": 1475, "confederation": "CONMEBOL"},
    "Algeria": {"strength": 78, "rank": 44, "points": 1470, "confederation": "CAF"},
    "Costa Rica": {"strength": 75, "rank": 54, "points": 1435, "confederation": "CONCACAF"},
    "Panama": {"strength": 76, "rank": 43, "points": 1472, "confederation": "CONCACAF"},
    "Qatar": {"strength": 75, "rank": 38, "points": 1490, "confederation": "AFC"},
    "Jamaica": {"strength": 75, "rank": 55, "points": 1430, "confederation": "CONCACAF"},
    "New Zealand": {"strength": 70, "rank": 95, "points": 1220, "confederation": "OFC"},
    "South Africa": {"strength": 74, "rank": 59, "points": 1410, "confederation": "CAF"},
    "Iraq": {"strength": 74, "rank": 58, "points": 1415, "confederation": "AFC"},
    "Uzbekistan": {"strength": 75, "rank": 60, "points": 1405, "confederation": "AFC"}
}

def generate_fifa_rankings():
    print("Generating fifa_rankings.csv...")
    filepath = "datasets/raw/fifa_rankings.csv"
    dates = []
    # Generate historical rankings over several dates from 2018 to 2026
    start_date = datetime(2018, 6, 1)
    for i in range(16):  # roughly 2 rankings per year
        dates.append(start_date + timedelta(days=180 * i))
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["country", "rank", "points", "confederation", "date"])
        for date in dates:
            # Sort teams with slight random perturbations to simulate ranking updates
            sorted_teams = sorted(
                TEAMS.keys(),
                key=lambda x: TEAMS[x]["points"] + random.randint(-40, 40),
                reverse=True
            )
            for idx, country in enumerate(sorted_teams):
                rank = idx + 1
                base_points = TEAMS[country]["points"]
                points = base_points + random.randint(-15, 15)
                conf = TEAMS[country]["confederation"]
                writer.writerow([country, rank, points, conf, date.strftime("%Y-%m-%d")])

def generate_player_ratings():
    print("Generating player_ratings_fc26.csv...")
    filepath = "datasets/raw/player_ratings_fc26.csv"
    
    # Roster templates for top teams
    players_data = []
    
    positions = ["GK", "DEF", "MID", "FWD"]
    
    # Real stars for top teams, others filled dynamically
    star_players = {
        "Argentina": [("Lionel Messi", "FWD", 90, 80, 88, 88, 35, 75, 15, 20, 18),
                      ("Lautaro Martinez", "FWD", 87, 80, 86, 72, 45, 85, 10, 24, 6),
                      ("Alexis Mac Allister", "MID", 85, 72, 78, 84, 78, 86, 12, 8, 10)],
        "France": [("Kylian Mbappe", "FWD", 91, 97, 90, 80, 36, 88, 10, 32, 10),
                   ("Antoine Griezmann", "FWD", 88, 79, 84, 87, 58, 85, 12, 15, 16),
                   ("William Saliba", "DEF", 86, 82, 40, 68, 87, 85, 10, 2, 1)],
        "Brazil": [("Vinicius Junior", "FWD", 90, 95, 84, 80, 38, 86, 12, 22, 11),
                   ("Rodrygo", "FWD", 86, 88, 81, 80, 42, 82, 15, 16, 9),
                   ("Bruno Guimaraes", "MID", 85, 72, 76, 83, 80, 88, 12, 6, 8)],
        "England": [("Harry Kane", "FWD", 90, 69, 93, 84, 45, 83, 10, 36, 7),
                    ("Jude Bellingham", "MID", 90, 80, 85, 83, 82, 90, 10, 20, 15),
                    ("Bukayo Saka", "FWD", 87, 86, 81, 80, 45, 84, 15, 16, 12)]
    }
    
    for country, details in TEAMS.items():
        base_strength = details["strength"]
        # Generate 23 players per country
        squad = []
        # 1. Add stars if defined
        if country in star_players:
            for p in star_players[country]:
                squad.append(p)
        
        # 2. Fill the rest of the 23-player squad
        gk_added = 0
        def_added = 0
        mid_added = 0
        fwd_added = 0
        
        # Count existing stars
        for p in squad:
            pos = p[1]
            if pos == "GK": gk_added += 1
            elif pos == "DEF": def_added += 1
            elif pos == "MID": mid_added += 1
            elif pos == "FWD": fwd_added += 1
            
        total_needed = 23 - len(squad)
        for i in range(total_needed):
            p_idx = len(squad) + 1
            # Determine position distribution
            if gk_added < 3:
                pos = "GK"
                gk_added += 1
            elif def_added < 8:
                pos = "DEF"
                def_added += 1
            elif mid_added < 8:
                pos = "MID"
                mid_added += 1
            else:
                pos = "FWD"
                fwd_added += 1
                
            name = f"Player {p_idx}"
            rating = base_strength + random.randint(-5, 4)
            # Clip rating
            rating = max(65, min(95, rating))
            
            # Attributes based on rating
            pace = rating + random.randint(-15, 10) if pos in ["FWD", "MID"] else rating - 20
            shooting = rating + random.randint(-10, 5) if pos == "FWD" else rating - 25
            passing = rating + random.randint(-5, 8) if pos == "MID" else rating - 15
            defending = rating + random.randint(-5, 8) if pos == "DEF" else rating - 30
            stamina = random.randint(70, 92)
            injury_risk = random.randint(5, 45)
            
            goals = random.randint(0, 15) if pos == "FWD" else random.randint(0, 5)
            assists = random.randint(0, 10) if pos in ["FWD", "MID"] else random.randint(0, 2)
            
            squad.append((name, pos, rating, pace, shooting, passing, defending, stamina, injury_risk, goals, assists))
            
        for p in squad:
            players_data.append([
                p[0], country, p[1], p[2], max(30, min(99, p[3])), max(30, min(99, p[4])),
                max(30, min(99, p[5])), max(30, min(99, p[6])), p[7], p[8], p[9], p[10]
            ])
            
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "country", "position", "overall_rating", "pace", "shooting", "passing", "defending", "stamina", "injury_risk", "goals_last_season", "assists_last_season"])
        writer.writerows(players_data)

def generate_international_results():
    print("Generating international_results.csv...")
    filepath = "datasets/raw/international_results.csv"
    
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2026, 5, 1)
    current_date = start_date
    
    matches = []
    
    # We will simulate ~1800 matches over the years
    tournaments = ["Friendly", "Copa America", "UEFA Euro", "World Cup Qualification", "Nations League", "AFC Asian Cup", "Africa Cup of Nations"]
    
    countries = list(TEAMS.keys())
    
    while current_date < end_date:
        # Generate a batch of matches for this date
        num_matches = random.randint(10, 25)
        for _ in range(num_matches):
            home = random.choice(countries)
            away = random.choice(countries)
            while away == home:
                away = random.choice(countries)
                
            # Simulate goals using a simple strength-based Poisson process
            h_strength = TEAMS[home]["strength"]
            a_strength = TEAMS[away]["strength"]
            
            # Neutral venue flag (50% for standard international friendlies/tournaments)
            neutral = random.choice([True, False])
            
            # Base home advantage
            home_adv = 0.3 if not neutral else 0.0
            
            # Lambda (expected goals) calculations
            lambda_home = max(0.5, (h_strength - a_strength) * 0.05 + 1.2 + home_adv)
            lambda_away = max(0.5, (a_strength - h_strength) * 0.05 + 1.0 - home_adv)
            
            home_score = random.choices(
                population=[0, 1, 2, 3, 4, 5],
                weights=[
                    max(0.05, 1 - (lambda_home*0.2)), 
                    max(0.1, lambda_home*0.3), 
                    max(0.1, lambda_home*0.25), 
                    max(0.05, lambda_home*0.15), 
                    0.05, 
                    0.01
                ]
            )[0]
            
            away_score = random.choices(
                population=[0, 1, 2, 3, 4, 5],
                weights=[
                    max(0.05, 1 - (lambda_away*0.2)), 
                    max(0.1, lambda_away*0.3), 
                    max(0.1, lambda_away*0.25), 
                    max(0.05, lambda_away*0.15), 
                    0.05, 
                    0.01
                ]
            )[0]
            
            tourney = random.choice(tournaments)
            matches.append([
                current_date.strftime("%Y-%m-%d"),
                home,
                away,
                home_score,
                away_score,
                tourney,
                int(neutral)
            ])
            
        # Jump ahead 10-25 days
        current_date += timedelta(days=random.randint(10, 25))
        
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"])
        writer.writerows(matches)
    print(f"Successfully generated {len(matches)} historical matches.")

if __name__ == "__main__":
    create_directories()
    generate_fifa_rankings()
    generate_player_ratings()
    generate_international_results()
    print("All raw datasets generated successfully in datasets/raw/")
