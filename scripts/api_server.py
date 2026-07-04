import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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

def get_daily_file(sport: str) -> Path:
    today_str = datetime.date.today().isoformat()
    file_path = PROJECT_ROOT / "data" / "daily_tips" / sport / f"{today_str}_CONSOLIDADO.csv"
    return file_path

@app.get("/", response_class=HTMLResponse)
def root_menu(request: Request):
    base_url = str(request.base_url)
    return f"""
    <html style="background-color: #202124; color: #e8eaed; font-family: monospace; font-size: 15px;">
        <head><title>Sinais API</title></head>
        <body style="padding: 20px;">
<pre>
{{
    <span style="color: #9cdcfe;">"sinais_galgos"</span>: <a href="/sinais_galgos.csv" style="color: #ce9178; text-decoration: underline;">"{base_url}sinais_galgos.csv"</a>,
    <span style="color: #9cdcfe;">"sinais_cavalos"</span>: <a href="/sinais_cavalos.csv" style="color: #ce9178; text-decoration: underline;">"{base_url}sinais_cavalos.csv"</a>
}}
</pre>
        </body>
    </html>
    """

@app.get("/sinais_galgos.csv")
def get_sinais_galgos():
    file_path = get_daily_file("greyhounds")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="O arquivo de sinais consolidados para Galgos de hoje ainda não foi gerado.")
    return FileResponse(path=file_path, media_type="text/csv", filename="sinais_galgos.csv")

@app.get("/sinais_cavalos.csv")
def get_sinais_cavalos():
    file_path = get_daily_file("horses")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="O arquivo de sinais consolidados para Cavalos de hoje ainda não foi gerado.")
    return FileResponse(path=file_path, media_type="text/csv", filename="sinais_cavalos.csv")

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
