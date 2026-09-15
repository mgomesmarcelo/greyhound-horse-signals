import datetime
import os
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI(title="Sinais API")

# Allow CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Token secreto lido do .env — nunca hardcoded no codigo
_FEED_TOKEN = os.getenv("API_FEED_TOKEN", "")
if not _FEED_TOKEN:
    raise RuntimeError("API_FEED_TOKEN nao definido no .env. A API nao pode subir sem um token.")

# Mapeamento para futuras estratégias e seus respectivos blocos no BFB
MAPA_PROVIDER_BLOCO = {
    # "Nome_da_Estrategia": "Nome_do_Bloco",
}

# Bloco padrão para todas as estratégias atuais de galgos
BLOCO_PADRAO_ATUAL_GALGOS = "layy"

def mapear_provider_galgos(estrategia: str) -> str:
    return MAPA_PROVIDER_BLOCO.get(str(estrategia), BLOCO_PADRAO_ATUAL_GALGOS)

def get_daily_file(sport: str) -> Path:
    today_str = datetime.date.today().isoformat()
    file_path = PROJECT_ROOT / "data" / "daily_tips" / sport / f"{today_str}_CONSOLIDADO.csv"
    return file_path

@app.get("/", response_class=HTMLResponse)
def root_menu():
    return """
    <html style="background-color: #202124; color: #e8eaed; font-family: monospace; font-size: 15px;">
        <head><title>Sinais API</title></head>
        <body style="padding: 20px;">
            <pre>API online.</pre>
        </body>
    </html>
    """

# Rotas antigas — retornam 404 para não revelar que existiram
@app.get("/sinais_galgos.csv")
def get_sinais_galgos_legacy():
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/sinais_cavalos.csv")
def get_sinais_cavalos_legacy():
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/feed/{token}/galgos.csv")
def get_sinais_galgos(token: str):
    if token != _FEED_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    file_path = get_daily_file("greyhounds")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="O arquivo de sinais consolidados para Galgos de hoje ainda não foi gerado.")
    try:
        df = pd.read_csv(file_path, dtype={"MarketId": str})
        if "Provider" in df.columns:
            df["Provider"] = df["Provider"].apply(mapear_provider_galgos)
        else:
            df["Provider"] = BLOCO_PADRAO_ATUAL_GALGOS
        csv_content = df.to_csv(index=False)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=sinais_galgos.csv"}
        )
    except Exception:
        return FileResponse(path=file_path, media_type="text/csv", filename="sinais_galgos.csv")

@app.get("/feed/{token}/cavalos.csv")
def get_sinais_cavalos(token: str):
    if token != _FEED_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    file_path = get_daily_file("horses")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="O arquivo de sinais consolidados para Cavalos de hoje ainda não foi gerado.")
    return FileResponse(path=file_path, media_type="text/csv", filename="sinais_cavalos.csv")

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

