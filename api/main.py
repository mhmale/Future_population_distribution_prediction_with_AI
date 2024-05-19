from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

country_hash_map = {
    "AL" : "Albania",
    "AD" : "Andorra",
    "AM" : "Armenia",
    "AT" : "Austria",
    "BE" : "Belgium",
    "BG" : "Bulgaria",
    "BA" : "Bosnia and Herzegovina",
    "BY" : "Belarus",
    "CH" : "Switzerland",
    "CZ" : "Czech Republic",
    "DE" : "Germany",
    "DK" : "Denmark",
    "EE" : "Estonia",
    "FI" : "Finland",
    "GB" : "United Kingdom",
    "GE" : "Georgia",
    "GR" : "Greece",
    "HR" : "Croatia",
    "HU" : "Hungary",
    "IE" : "Ireland",
    "IS" : "Iceland",
    "IT" : "Italy",
    "LI" : "Liechtenstein",
    "LT" : "Lithuania",
    "LU" : "Luxembourg",
    "LV" : "Latvia",
    "MD" : "Moldova",
    "MK" : "Macedonia",
    "ME" : "Montenegro",
    "NO" : "Norway",
    "PL" : "Poland",
    "PT" : "Portugal", #SÜÜÜÜÜÜÜÜÜÜÜ
    "RO" : "Romania",
    "RS" : "Serbia",
    "SK" : "Slovakia",
    "SI" : "Slovenia",
    "SE" : "Sweden",
    "TR" : "Turkey",
    "UA" : "Ukraine",
    "XK" : "Kosovo",
    "NL" : "Netherlands",
    "ES" : "Spain",
    "FR" : "France",
    "CY" : "Cyprus",
}



# CORS middleware ekleme
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm origin'lere izin vermek (daha güvenli bir yapı için uygun originleri belirleyin)
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/country")
async def creat_item(country_id: str):
    if country_id not in country_hash_map:
        raise Exception("Country code not found !")
    
    country_name = country_hash_map[country_id]
    return{"country_name" : country_name}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

