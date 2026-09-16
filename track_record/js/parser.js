/**
 * BetParser - Modulo de processamento e normalizacao de dados CSV de apostas.
 * Projetado para dashboards de apostas Lay em corridas de galgos.
 */

const BetParser = {
    /**
     * Mapeamento de abreviacoes de meses em ingles para indices numericos (0 a 11).
     */
    _monthMap: {
        jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5,
        jul: 6, aug: 7, sep: 8, oct: 9, nov: 10, dec: 11
    },

    /**
     * Analisa o texto CSV utilizando PapaParse e retorna uma lista de apostas normalizadas.
     * @param {string} csvText - Conteudo bruto do arquivo CSV.
     * @returns {Array<Object>} Lista de objetos de aposta normalizados.
     */
    parseCSV(csvText) {
        if (!csvText || typeof csvText !== 'string' || !csvText.trim()) {
            return [];
        }

        if (typeof Papa === 'undefined') {
            console.error('PapaParse (Papa) nao foi carregado no escopo global.');
            return [];
        }

        try {
            const results = Papa.parse(csvText, {
                header: true,
                skipEmptyLines: true,
                dynamicTyping: false
            });

            if (results.errors && results.errors.length > 0) {
                console.warn('Avisos ao processar CSV com PapaParse:', results.errors);
            }

            if (!Array.isArray(results.data)) {
                return [];
            }

            const bets = [];
            for (let i = 0; i < results.data.length; i++) {
                const normalized = this.normalizeBet(results.data[i]);
                if (normalized !== null) {
                    bets.push(normalized);
                }
            }

            return bets;
        } catch (error) {
            console.error('Erro ao executar parseCSV:', error);
            return [];
        }
    },

    /**
     * Converte uma string de Profit/Loss para numero de ponto flutuante.
     * Trata valores positivos (ex: "1.00" -> 1.00) e valores negativos entre parenteses (ex: "(6.40)" -> -6.40).
     * @param {string|number} value - Valor bruto de lucro/prejuizo.
     * @returns {number} Valor numerico de Profit/Loss.
     */
    parseProfitLoss(value) {
        if (value === null || value === undefined) {
            return 0;
        }

        if (typeof value === 'number') {
            return isNaN(value) ? 0 : value;
        }

        const str = String(value).trim();
        if (!str) {
            return 0;
        }

        // Verifica formato com parenteses para valores negativos: "(6.40)" ou "( 6.40 )"
        const parenMatch = str.match(/^\((.*)\)$/);
        if (parenMatch) {
            const cleaned = parenMatch[1].replace(/[^\d.-]/g, '');
            const parsed = parseFloat(cleaned);
            if (isNaN(parsed) || parsed === 0) {
                return 0;
            }
            return -Math.abs(parsed);
        }

        // Trata valores regulares com possiveis simbolos de moeda ou espacos
        const cleaned = str.replace(/[^\d.-]/g, '');
        const parsed = parseFloat(cleaned);
        return isNaN(parsed) ? 0 : parsed;
    },

    /**
     * Converte uma string de data no formato "DD-MMM-YY HH:mm" (ex: "31-Aug-26 21:22") para um objeto Date.
     * Trata anos de 2 digitos considerando 2000 + YY (ex: 26 -> 2026).
     * @param {string} dateStr - String com a data da aposta.
     * @returns {Date|null} Objeto Date correspondente ou null se invalido.
     */
    parseDate(dateStr) {
        if (!dateStr || typeof dateStr !== 'string') {
            return null;
        }

        const str = dateStr.trim();
        if (!str) {
            return null;
        }

        // Formato esperado: "31-Aug-26 21:22" ou "31-Aug-2026 21:22:00"
        const regex = /^(\d{1,2})[-/]([A-Za-z]{3})[-/](\d{2,4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?/;
        const match = str.match(regex);

        if (!match) {
            const fallback = new Date(str);
            return isNaN(fallback.getTime()) ? null : fallback;
        }

        const day = parseInt(match[1], 10);
        const monthKey = match[2].toLowerCase();
        let year = parseInt(match[3], 10);
        const hours = parseInt(match[4], 10);
        const minutes = parseInt(match[5], 10);
        const seconds = match[6] ? parseInt(match[6], 10) : 0;

        if (year < 100) {
            year += 2000;
        }

        const month = this._monthMap[monthKey];
        if (month === undefined) {
            return null;
        }

        const parsedDate = new Date(year, month, day, hours, minutes, seconds);
        return isNaN(parsedDate.getTime()) ? null : parsedDate;
    },

    /**
     * Extrai o nome da pista (track) a partir da string do mercado.
     * Exemplo: "GB / Romford 31st Aug / 12:01 A3 400m" -> "Romford"
     * @param {string} market - String descritiva do mercado.
     * @returns {string} Nome da pista extraido ou string vazia.
     */
    extractTrack(market) {
        if (!market || typeof market !== 'string') {
            return '';
        }

        const trimmed = market.trim();
        if (!trimmed) {
            return '';
        }

        // Divide pelas barras separadoras do mercado
        const parts = trimmed.split('/');
        if (parts.length >= 2) {
            // A segunda parte contem a pista e o dia/mes (ex: "Romford 31st Aug" ou "Star Pelaw 31st Aug")
            const trackDatePart = parts[1].trim();
            // Remove sufixos de data como "31st Aug", "1st Sep", "2nd Oct", "3rd Nov", "4th Dec", "15 Aug"
            const trackClean = trackDatePart.replace(/\s+\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+.*$/i, '').trim();
            if (trackClean) {
                return trackClean;
            }
        }

        // Estrategia alternativa via regex caso o formato nao siga divisor '/' padrao
        const match = trimmed.match(/(?:[A-Z]{2}\s*\/\s*)?(.*?)\s+\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+/i);
        if (match && match[1]) {
            return match[1].replace(/^[A-Z]{2}\s*\/\s*/i, '').trim();
        }

        return trimmed;
    },

    /**
     * Extrai a distancia da corrida em metros a partir da string do mercado.
     * Exemplo: "GB / Romford 31st Aug / 12:01 A3 400m" -> 400
     * @param {string} market - String descritiva do mercado.
     * @returns {number} Distancia em metros como inteiro ou 0 se nao encontrada.
     */
    extractDistance(market) {
        if (!market || typeof market !== 'string') {
            return 0;
        }

        const trimmed = market.trim();
        if (!trimmed) {
            return 0;
        }

        // Procura por numero seguido da letra 'm' no final da string
        const match = trimmed.match(/(\d+)\s*m\s*$/i);
        if (match) {
            return parseInt(match[1], 10);
        }

        // Fallback para capturar distancia antes de 'm' caso haja espacos ou caracteres extras
        const fallbackMatch = trimmed.match(/(\d+)\s*m\b/i);
        if (fallbackMatch) {
            return parseInt(fallbackMatch[1], 10);
        }

        return 0;
    },

    /**
     * Normaliza uma linha bruta retornada pelo PapaParse em um objeto padronizado de aposta.
     * Pula linhas onde o Bet ID esteja vazio ou ausente.
     * @param {Object} rawRow - Objeto de linha bruta retornado pelo PapaParse.
     * @returns {Object|null} Objeto normalizado da aposta ou null se linha invalida.
     */
    normalizeBet(rawRow) {
        if (!rawRow || typeof rawRow !== 'object') {
            return null;
        }

        // Obtencao do Bet ID com suporte a variacoes de chave
        const betId = rawRow['Bet ID'] || rawRow['betId'] || rawRow['BetId'] || rawRow['bet_id'];
        if (betId === undefined || betId === null || String(betId).trim() === '') {
            return null;
        }

        const market = rawRow['Market'] || rawRow['market'] || '';
        const selection = rawRow['Selection'] || rawRow['selection'] || '';
        const bidType = rawRow['Bid type'] || rawRow['bidType'] || rawRow['Bid Type'] || '';
        const betPlacedRaw = rawRow['Bet placed'] || rawRow['datePlaced'] || rawRow['Bet Placed'] || '';
        const persistence = rawRow['Persistence'] || rawRow['persistence'] || '';

        const oddsReqRaw = rawRow['Odds req.'] || rawRow['oddsRequested'] || rawRow['Odds req'] || '0';
        const stakeRaw = rawRow['Stake (\u20ac)'] || rawRow['Stake (€)'] || rawRow['Stake'] || rawRow['stake'] || '0';
        const liabilityRaw = rawRow['Liability (\u20ac)'] || rawRow['Liability (€)'] || rawRow['Liability'] || rawRow['liability'] || '0';
        const avgOddsRaw = rawRow['Avg. odds matched'] || rawRow['avgOddsMatched'] || rawRow['Avg odds matched'] || '0';
        const profitLossRaw = rawRow['Profit/Loss (\u20ac)'] || rawRow['Profit/Loss (€)'] || rawRow['Profit/Loss'] || rawRow['profitLoss'] || '0';

        const parsedOddsReq = parseFloat(oddsReqRaw);
        const parsedStake = parseFloat(stakeRaw);
        const parsedLiability = parseFloat(liabilityRaw);
        const parsedAvgOdds = parseFloat(avgOddsRaw);
        const parsedProfitLoss = this.parseProfitLoss(profitLossRaw);

        const bet = {
            market: String(market).trim(),
            selection: String(selection).trim(),
            bidType: String(bidType).trim(),
            betId: String(betId).trim(),
            datePlaced: this.parseDate(betPlacedRaw),
            persistence: String(persistence).trim(),
            oddsRequested: isNaN(parsedOddsReq) ? 0 : parsedOddsReq,
            stake: isNaN(parsedStake) ? 0 : parsedStake,
            liability: isNaN(parsedLiability) ? 0 : parsedLiability,
            avgOddsMatched: isNaN(parsedAvgOdds) ? 0 : parsedAvgOdds,
            profitLoss: parsedProfitLoss,
            track: this.extractTrack(market),
            distance: this.extractDistance(market),
            isWin: false
        };

        // Aplica comissao de 2% sobre o lucro de apostas ganhas
        if (bet.profitLoss > 0) {
            bet.profitLoss = parseFloat((bet.profitLoss * 0.98).toFixed(2));
        }

        bet.isWin = bet.profitLoss > 0;

        return bet;
    },

    /**
     * Combina dois arrays de apostas, removendo duplicatas pelo Bet ID.
     * Retorna a lista unificada ordenada por datePlaced em ordem decrescente (mais recente primeiro).
     * @param {Array<Object>} existingBets - Lista de apostas existentes.
     * @param {Array<Object>} newBets - Lista de novas apostas a serem mescladas.
     * @returns {Array<Object>} Lista mesclada e ordenada.
     */
    mergeData(existingBets, newBets) {
        const existing = Array.isArray(existingBets) ? existingBets : [];
        const incoming = Array.isArray(newBets) ? newBets : [];

        const betMap = new Map();

        // Adiciona apostas existentes
        for (let i = 0; i < existing.length; i++) {
            const item = existing[i];
            if (item && item.betId) {
                betMap.set(String(item.betId), item);
            }
        }

        // Adiciona ou atualiza com as novas apostas
        for (let i = 0; i < incoming.length; i++) {
            const item = incoming[i];
            if (item && item.betId) {
                betMap.set(String(item.betId), item);
            }
        }

        const merged = Array.from(betMap.values());

        // Ordenacao decrescente por datePlaced (mais recente primeiro)
        merged.sort((a, b) => {
            const timeA = (a.datePlaced instanceof Date && !isNaN(a.datePlaced.getTime())) ? a.datePlaced.getTime() : 0;
            const timeB = (b.datePlaced instanceof Date && !isNaN(b.datePlaced.getTime())) ? b.datePlaced.getTime() : 0;
            return timeB - timeA;
        });

        return merged;
    }
};

// Exportacao para ambientes Node/CommonJS e navegadores
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BetParser;
}
if (typeof window !== 'undefined') {
    window.BetParser = BetParser;
}
