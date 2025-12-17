# Projeto Unificado

Suite de raspagem, limpeza e análise de corridas de cavalos e galgos, com dashboards Streamlit e rotinas diárias para geração de sinais de aposta.

## Visão geral
- Coleta dados brutos da Betfair, Timeform e Sporting Life usando Selenium.
- Padroniza e versiona arquivos em `data/` (CSV e Parquet) para histórico e consumo analítico.
- Gera sinais configuráveis (estratégias LAY/BACK, mercados WIN/PLACE) e indicadores auxiliares.
- Disponibiliza dashboards interativos em Streamlit para inspeção rápida das corridas e dos sinais.

## Pré-requisitos
- Python 3.9 ou superior.
- Google Chrome (ou Chromium) instalado; o `webdriver-manager` baixa o driver compatível automaticamente.
- Sistema operacional com suporte a Selenium (Windows, macOS ou Linux).

### Instalação rápida
1. `python -m venv .venv`
2. `.\.venv\Scripts\activate` (Windows) ou `source .venv/bin/activate` (Linux/macOS)
3. `pip install -r requirements.txt` ou `pip install -e .`
4. (Opcional) crie um `.env` na raiz para armazenar credenciais e chaves utilizadas pelos scrapers.

## Estrutura do projeto
- `src/core/`: helpers compartilhados (configuração de diretórios e caminhos).
- `src/horses/`: scrapers, análises e utilitários específicos de corridas de cavalos.
- `src/greyhounds/`: scrapers, análises e utilitários específicos de corridas de galgos.
- `scripts/`: pontos de entrada em linha de comando (raspagem diária, geração de sinais, dashboards); subpastas separam cavalos (`scripts/horses/`) e galgos.
- `data/`: armazenamento local (bruto e processado). A estrutura base é criada automaticamente:

```
data/
  horses/
    Result/
    signals/
    timeform_top3/
    TimeformForecast/
    betfair_top3/
    sportinglife_top3/
  greyhounds/
    Result/
    signals/
    race_links/
    timeform_top3/
    TimeformForecast/
    processed/
      Result/
      signals/
      race_links/
      timeform_top3/
      TimeformForecast/
```

## Fluxo de trabalho recomendado

### Cavalos (`horses`)
- **Raspagem diária:** `python -m src.horses.run_daily`  
  Executa em sequência os scrapers da Betfair, Sporting Life (atualização + backfill) e Timeform.
- **Limpeza opcional:** `python scripts/clean_horse_results.py [--force]`  
  Remove duplicados e corrige colunas de resultados.
- **Geração de sinais:**  
  `python scripts/horses/generate_horse_signals.py --source both --market both --strategy both --provider timeform`
- **Dashboard Streamlit:**  
  `python scripts/horses/run_horses_streamlit.py --port 8502 --address 0.0.0.0`

### Galgos (`greyhounds`)
- **Raspagem diária:** `python -m src.greyhounds.run_daily`  
  Busca o índice da Betfair e atualizações Timeform.
- **Limpeza opcional:** `python scripts/greyhounds/clean_greyhound_results.py [--force]`
- **Geração de sinais:**  
  `python scripts/greyhounds/generate_greyhound_signals.py --source both --market both --rule both --entry_type both`
- **Dashboard Streamlit:**  
  `python scripts/greyhounds/run_greyhounds_streamlit.py --port 8501 --address 0.0.0.0`

### Pipeline completo
- `python -m scripts.run_all_daily` executa `run_daily` de cavalos e galgos em sequência.

## Dashboards Streamlit
- `scripts/horses/run_horses_streamlit.py`: filtros por pista, intervalo de datas, fonte (Timeform/Sporting Life) e estratégia; gráficos de ROI, drawdown e distribuição de odds.
- `scripts/greyhounds/run_greyhounds_streamlit.py`: visão consolidada de sinais por regra, mercado e fonte; ranking de pistas/categorias, análise de desempenho e curvas acumuladas.
- Parâmetros úteis:
  - `--port`: porta HTTP (ex.: `--port 8502`).
  - `--address`: endereço para bind (use `0.0.0.0` ao publicar na rede).

## Scripts auxiliares
- `python scripts/download_betfair_prices.py [opções]`  
  Sincroniza históricos públicos Betfair SP para cavalos e galgos. Principais flags:
  - `--dry-run` lista arquivos faltantes sem baixar.
  - `--delay 2` ajusta a pausa entre downloads (default 1.5s).
  - `--horses-start-date YYYY-MM-DD` / `--greyhounds-start-date YYYY-MM-DD` força a data inicial.
  - `--max-downloads N` limita a quantidade por execução.
- `python scripts/greyhounds/convert_greyhound_history.py --dataset <nome> [--force]`  
  Converte CSV históricos de galgos para Parquet (datasets: `signals`, `timeform_top3`, `timeform_forecast`, `race_links`, `betfair_result`).
- `python scripts/validate_data_layout.py`  
  Validação leve dos nomes de arquivos e colunas esperadas em `data/`.
- `python scripts/greyhounds/analyze_track_aliases.py`  
  Gera relatório de aliases de pistas de galgos em `data/greyhounds/reports/track_alias_report.json` a partir dos CSVs brutos (`data/greyhounds/Result`).

## Configurações
- Ajuste `settings.LOG_LEVEL` em `src/horses/config.py` ou `src/greyhounds/config.py` para controlar a verbosidade dos logs.
- As mesmas configurações permitem ligar/desligar `SELENIUM_HEADLESS` e alterar timeouts ou diretórios base.
- As rotinas utilizam `utf-8-sig` por padrão nas leituras/escritas CSV; mantenha o encoding ao criar arquivos externos.

## Boas práticas
- Execute os scrapers em horários espaçados para evitar bloqueios nas fontes externas.
- Revise os diretórios `data/processed/` após processamentos antigos usando `convert_greyhound_history.py`.
- Mantenha o Chrome atualizado para garantir compatibilidade do Selenium.
