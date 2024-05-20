from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn
import json

app = FastAPI()

country_hash_map = {
    "AL": "Albania",
    "AD": "Andorra",
    "AM": "Armenia",
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "BA": "Bosnia and Herzegovina",
    "BY": "Belarus",
    "CH": "Switzerland",
    "CZ": "Czech Republic",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "FI": "Finland",
    "GB": "United Kingdom",
    "GE": "Georgia",
    "GR": "Greece",
    "HR": "Croatia",
    "HU": "Hungary",
    "IE": "Ireland",
    "IS": "Iceland",
    "IT": "Italy",
    "LI": "Liechtenstein",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "MD": "Moldova",
    "MK": "Macedonia",
    "ME": "Montenegro",
    "NO": "Norway",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "RS": "Serbia",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "SE": "Sweden",
    "TR": "Turkey",
    "UA": "Ukraine",
    "XK": "Kosovo",
    "NL": "Netherlands",
    "ES": "Spain",
    "FR": "France",
    "CY": "Cyprus",
}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for security as needed)
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/country", response_class=Response)
async def get_country_data(country_id: str):
    if country_id not in country_hash_map:
        raise HTTPException(status_code=404, detail="Country code not found!")
    
    country_name = country_hash_map[country_id]
    file_path = f"Tahminler/{country_name}.csv"
    
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Convert DataFrame to JSON string
        json_data = df.to_json(orient="records")
        # Include country name in the JSON response
        json_response = {"country_name": country_name, "data": json_data}
        # Encode the JSON response as bytes
        encoded_response = json.dumps(json_response).encode("utf-8")
        # Set content type to JSON and return encoded JSON data with country name
        return Response(content=encoded_response, media_type="application/json")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
