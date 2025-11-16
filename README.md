# Projeto Unificado

## Scripts principais

### Cavalos
- `python -m src.horses.scrape_betfair_index`
- `python -m src.horses.scrape_betfair_races`
- `python -m src.horses.scrape_timeform_update`
- `python scripts/generate_horse_signals.py --source both --market both --strategy both --provider timeform`
- `python scripts/clean_horse_results.py [--force]`
- `python scripts/run_horses_streamlit.py --port 8502 --address 0.0.0.0`

#### Fluxo recomendado (horses)
1. `python -m src.horses.run_daily`
2. `python scripts/clean_horse_results.py`
3. `python scripts/generate_horse_signals.py --source both --market both --strategy both --provider timeform`
4. `python scripts/run_horses_streamlit.py`

### Galgos
- `python -m src.greyhounds.scrape_betfair_index`
- `python -m src.greyhounds.scrape_timeform_update`
- `python -m src.greyhounds.run_daily`
- `python scripts/generate_greyhound_signals.py --source both --market both --rule both --entry_type both`
- `python scripts/clean_greyhound_results.py [--force]`
- `python scripts/run_greyhounds_streamlit.py --port 8501 --address 0.0.0.0`

#### Fluxo recomendado (daily)
1. `python -m src.greyhounds.run_daily`
2. `python scripts/clean_greyhound_results.py`
3. `python scripts/generate_greyhound_signals.py --source both --market both --rule both --entry_type both`
4. `python scripts/run_greyhounds_streamlit.py`

### Pipeline completo
- `python -m scripts.run_all_daily`

### Download automático Betfair SP
- `python scripts/download_betfair_prices.py [opções]`
  - Sincroniza automaticamente os arquivos públicos da Betfair SP para `data/horses/Result` e `data/greyhounds/Result`, respeitando os nomes originais (Win/Place, UK/IRE etc.).
  - Executa em duas fases: detecta a data mínima existente localmente (considerando toda a árvore `data/horses` ou `data/greyhounds`) e baixa apenas o que está faltando até a data atual.
  - Principais flags:
    - `--dry-run`: lista os arquivos faltantes sem baixar (recomendado na primeira execução).
    - `--delay 2`: impõe espera em segundos entre downloads (default 1.5s) para evitar bloqueios.
    - `--horses-start-date YYYY-MM-DD` / `--greyhounds-start-date YYYY-MM-DD`: sobrescrevem a data mínima detectada. Ex.: baixar galgos desde 2002 → `python scripts/download_betfair_prices.py --greyhounds-start-date 2002-01-01`.
    - `--max-downloads N`: limita a quantidade de arquivos (útil para testes rápidos).

## Estrutura de dados (data/)
`
data/
â”œâ”€â”€ horses/
â”‚   â”œâ”€â”€ betfair_top3/
â”‚   â””â”€â”€ TimeformForecast/
â””â”€â”€ greyhounds/
    â”œâ”€â”€ race_links/
    â”œâ”€â”€ TimeformForecast/
    â””â”€â”€ timeform_top3/
`

## Observacoes
- O Selenium necessita de Chrome/Chromedriver compatíveis (webdriver-manager cuida disso).
- Ajuste `settings.LOG_LEVEL` em `src/horses/config.py` ou `src/greyhounds/config.py` para controlar verbosidade.
- Use `.env` para variáveis sensíveis (lidas via python-dotenv, se necessário).