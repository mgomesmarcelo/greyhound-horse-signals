# Projeto Unificado

Suite de raspagem, limpeza e análise de corridas de cavalos e galgos, com dashboards Streamlit, rotinas diárias para geração de sinais de aposta e integração direta com o **BF Bot Manager**.

## Visão geral
- Coleta dados brutos da Betfair, Timeform e Sporting Life usando Selenium.
- Padroniza e versiona arquivos em `data/` (CSV e Parquet) para histórico e consumo analítico.
- Gera sinais configuráveis (estratégias LAY/BACK, mercados WIN/PLACE) e indicadores auxiliares.
- **Exportação automatizada de dicas** configuradas por estratégias de *Value Ratio* e *Rank* diretamente para o formato CSV aceito pelo BF Bot Manager.
- Disponibiliza dashboards interativos em Streamlit para inspeção rápida das corridas e dos sinais.
- Possui estrutura para rodar como servidor em ambiente de nuvem (VPS) via API.

## Pré-requisitos
- Python 3.9 ou superior.
- Google Chrome (ou Chromium) instalado; o `webdriver-manager` baixa o driver compatível automaticamente.
- Sistema operacional com suporte a Selenium (Windows, macOS ou Linux).

### Instalação rápida
1. `python -m venv .venv`
2. `.\.venv\Scripts\activate` (Windows) ou `source .venv/bin/activate` (Linux/macOS)
3. `pip install -r requirements.txt`
4. Configure as chaves de acesso essenciais no `.env` (opcional).
5. Defina e personalize suas estratégias de aposta no arquivo `config/bfb_tips_config.yaml`.

## Estrutura do projeto
- `config/`: Arquivos de configuração de estratégias (ex: `bfb_tips_config.yaml`).
- `src/core/`: Helpers compartilhados (configuração de diretórios e caminhos).
- `src/horses/` e `src/greyhounds/`: Scrapers, análises e utilitários específicos.
- `scripts/`: Pontos de entrada em linha de comando (raspagem diária, geração de sinais, dashboards).
- `scripts/vps_setup/`: Scripts em lotes (`.bat`) para automação de inicialização na VPS.
- `data/`: Armazenamento local (bruto e processado). A estrutura base é criada automaticamente.

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
  Busca o índice da Betfair e as atualizações do Timeform (utilizando o novo `betfair_race.py`).
- **Limpeza opcional:** `python scripts/greyhounds/clean_greyhound_results.py [--force]`
- **Geração de Sinais para BF Bot Manager:**  
  `python scripts/greyhounds/gerar_dicas_bfb.py`  
  Lê os forecasts raspados, aplica as regras definidas no `config/bfb_tips_config.yaml` (Filtros de Pista, Categoria, Ratio, etc) e salva os arquivos CSV na pasta de output pronta para o BFB ler. Também registra a auditoria na pasta de logs.
- **Dashboard Streamlit:**  
  `python scripts/greyhounds/run_greyhounds_streamlit.py --port 8501 --address 0.0.0.0` (acessa o `streamlit_app.py`)

### Pipeline completo
- `python -m scripts.run_all_daily` executa a esteira `run_daily` tanto de cavalos quanto galgos em sequência.

## Infraestrutura e VPS (API)
- **Servidor API:** `python scripts/api_server.py`  
  Inicia o servidor backend FastAPI responsável pela integração sistêmica na nuvem.
- **Automação VPS (`scripts/vps_setup/`):**
  - `start_api.bat`: Inicia o servidor web.
  - `start_dashboard_galgos.bat` / `start_dashboard_cavalos.bat`: Ativam o ambiente virtual e hospedam os dashboards em background.
  - `run_automacao_diaria.bat`: Automação da esteira agendada.

## Scripts auxiliares
- **Segurança da Banca:** `python scripts/greyhounds/stop_loss.py`  
  Script dedicado para gerenciar as métricas de segurança de parada (Stop Loss).
- **Dados Históricos:** `python scripts/download_betfair_prices.py [opções]`  
  Sincroniza os históricos públicos do Betfair SP.
- **Otimização:** `python scripts/greyhounds/convert_greyhound_history.py --dataset <nome> [--force]`  
  Converte CSVs históricos enormes para o formato Parquet.

## Configurações e Boas Práticas
- **Estratégias BFB:** Modifique o `config/bfb_tips_config.yaml` sempre que quiser afinar seus thresholds (como `value_ratio_min`, ranges de Odds e regras de horário).
- **Logs:** Ajuste `settings.LOG_LEVEL` em `src/horses/config.py` ou `src/greyhounds/config.py` para controlar a verbosidade.
- Execute os scrapers em horários mais espaçados para evitar bloqueios (rate limits) e mantenha sempre o Chrome atualizado para não quebrar o Selenium.
