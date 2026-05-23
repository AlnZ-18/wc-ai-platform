from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "World Cup AI Platform Backend Running"
    }

@app.get("/matches")
def matches():
    return {
        "matches": [
            {
                "home": "Argentina",
                "away": "France",
                "prediction": "Argentina"
            }
        ]
    }