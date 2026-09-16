/**
 * Modulo de Metricas para o Dashboard de Apostas (BetMetrics)
 * Responsavel pelo processamento e calculo de todos os indicadores estatisticos e analiticos.
 */

const BetMetrics = {
    /**
     * Executa todos os calculos de metricas e retorna um objeto consolidado.
     * @param {Array<Object>} bets - Lista de apostas normalizadas.
     * @returns {Object} Objeto contendo todos os resultados analiticos.
     */
    calculateAll(bets) {
        return {
            kpis: this.calculateKPIs(bets),
            monthlyPL: this.calculateMonthlyPL(bets),
            dailyPL: this.calculateDailyPL(bets),
            cumulativePL: this.calculateCumulativePL(bets),
            byOddsRange: this.calculateByOddsRange(bets),
            byTrack: this.calculateByTrack(bets),
            byHour: this.calculateByHour(bets),
            winRateByOdds: this.calculateWinRateByOdds(bets),
            plDistribution: this.calculatePLDistribution(bets),
            cumulativePLBetByBet: this.calculateCumulativePLBetByBet(bets)
        };
    },

    /**
     * Calcula os principais indicadores de desempenho (KPIs).
     * @param {Array<Object>} bets - Lista de apostas normalizadas.
     * @returns {Object} KPIs calculados.
     */
    calculateKPIs(bets) {
        if (!Array.isArray(bets) || bets.length === 0) {
            return {
                totalBets: 0,
                winRate: 0,
                totalPL: 0,
                totalUnits: 0,
                roi: 0,
                avgPLPerBet: 0,
                maxWin: 0,
                maxLoss: 0
            };
        }

        const totalBets = bets.length;
        const winCount = bets.filter(b => Boolean(b.isWin)).length;
        const winRate = parseFloat(((winCount / totalBets) * 100).toFixed(2));

        const totalPLRaw = bets.reduce((acc, b) => acc + (Number(b.profitLoss) || 0), 0);
        const totalPL = parseFloat(totalPLRaw.toFixed(2));

        // Calcula Saldo de Unidades
        const totalUnitsRaw = bets.reduce((acc, b) => {
            const stake = Number(b.stake) || 1; // previne divisao por zero
            return acc + ((Number(b.profitLoss) || 0) / stake);
        }, 0);
        const totalUnits = parseFloat(totalUnitsRaw.toFixed(2));

        const totalLiability = bets.reduce((acc, b) => acc + (Number(b.liability) || 0), 0);
        const roi = totalLiability > 0 
            ? parseFloat(((totalPL / totalLiability) * 100).toFixed(2)) 
            : 0;

        const avgPLPerBet = parseFloat((totalPL / totalBets).toFixed(2));

        const plValues = bets.map(b => Number(b.profitLoss) || 0);
        const positiveWins = plValues.filter(v => v > 0);
        const negativeLosses = plValues.filter(v => v < 0);

        const maxWin = positiveWins.length > 0 
            ? parseFloat(Math.max(...positiveWins).toFixed(2)) 
            : 0;

        const maxLoss = negativeLosses.length > 0 
            ? parseFloat(Math.min(...negativeLosses).toFixed(2)) 
            : 0;

        // Calcular Pico Maximo e Drawdown (Maior Queda) do P&L Acumulado
        const sortedBets = [...bets].sort((a, b) => {
            const timeA = (a.datePlaced instanceof Date && !isNaN(a.datePlaced.getTime())) ? a.datePlaced.getTime() : 0;
            const timeB = (b.datePlaced instanceof Date && !isNaN(b.datePlaced.getTime())) ? b.datePlaced.getTime() : 0;
            return timeA - timeB;
        });

        let runningTotal = 0;
        let maxCumulative = 0;
        let minCumulative = 0;
        sortedBets.forEach(b => {
            runningTotal += (Number(b.profitLoss) || 0);
            if (runningTotal > maxCumulative) maxCumulative = runningTotal;
            if (runningTotal < minCumulative) minCumulative = runningTotal;
        });

        return {
            totalBets,
            winRate,
            totalPL,
            totalUnits,
            roi,
            avgPLPerBet,
            maxWin,
            maxLoss,
            maxCumulativePL: parseFloat(maxCumulative.toFixed(2)),
            minCumulativePL: parseFloat(minCumulative.toFixed(2))
        };
    },

    /**
     * Agrupa as apostas por data (YYYY-MM-DD) e calcula o P&L e total de apostas do dia.
     * @param {Array<Object>} bets - Lista de apostas normalizadas.
     * @returns {Array<Object>} Lista de P&L diario ordenada cronologicamente.
     */
    calculateDailyPL(bets) {
        if (!Array.isArray(bets) || bets.length === 0) {
            return [];
        }

        const dailyMap = {};

        bets.forEach(b => {
            if (!b.datePlaced || !(b.datePlaced instanceof Date) || isNaN(b.datePlaced.getTime())) {
                return;
            }

            const yyyy = b.datePlaced.getFullYear();
            const mm = String(b.datePlaced.getMonth() + 1).padStart(2, '0');
            const dd = String(b.datePlaced.getDate()).padStart(2, '0');
            const dateKey = `${yyyy}-${mm}-${dd}`;
            const label = `${dd}/${mm}`;

            if (!dailyMap[dateKey]) {
                dailyMap[dateKey] = {
                    date: dateKey,
                    label: label,
                    pl: 0,
                    bets: 0
                };
            }

            dailyMap[dateKey].pl += (Number(b.profitLoss) || 0);
            dailyMap[dateKey].bets += 1;
        });

        return Object.values(dailyMap)
            .sort((a, b) => a.date.localeCompare(b.date))
            .map(item => ({
                date: item.date,
                label: item.label,
                pl: parseFloat(item.pl.toFixed(2)),
                bets: item.bets
            }));
    },

    /**
     * Agrupa as apostas por mês (YYYY-MM) e calcula P&L, ROI e Win Rate mensais.
     * @param {Array<Object>} bets - Lista de apostas normalizadas.
     * @returns {Array<Object>} Lista de P&L mensal ordenada cronologicamente.
     */
    calculateMonthlyPL(bets) {
        if (!Array.isArray(bets) || bets.length === 0) {
            return [];
        }

        const monthlyMap = {};
        const monthNames = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"];

        bets.forEach(b => {
            if (!b.datePlaced || !(b.datePlaced instanceof Date) || isNaN(b.datePlaced.getTime())) {
                return;
            }

            const yyyy = b.datePlaced.getFullYear();
            const mm = String(b.datePlaced.getMonth() + 1).padStart(2, '0');
            const dateKey = `${yyyy}-${mm}`;
            const label = `${monthNames[b.datePlaced.getMonth()]}/${yyyy}`;

            if (!monthlyMap[dateKey]) {
                monthlyMap[dateKey] = {
                    date: dateKey,
                    label: label,
                    pl: 0,
                    bets: 0,
                    wins: 0,
                    liability: 0
                };
            }

            monthlyMap[dateKey].pl += (Number(b.profitLoss) || 0);
            monthlyMap[dateKey].bets += 1;
            if (b.isWin) {
                monthlyMap[dateKey].wins += 1;
            }
            monthlyMap[dateKey].liability += (Number(b.liability) || 0);
        });

        return Object.values(monthlyMap)
            .sort((a, b) => a.date.localeCompare(b.date))
            .map(item => {
                const winRate = item.bets > 0 ? ((item.wins / item.bets) * 100) : 0;
                const roi = item.liability > 0 ? ((item.pl / item.liability) * 100) : 0;
                
                return {
                    date: item.date,
                    label: item.label,
                    pl: parseFloat(item.pl.toFixed(2)),
                    bets: item.bets,
                    winRate: parseFloat(winRate.toFixed(2)),
                    roi: parseFloat(roi.toFixed(2))
                };
            });
    },

    /**
     * Calcula o P&L acumulado diario ao longo do tempo.
     * @param {Array<Object>} bets - Lista de apostas normalizadas.
     * @returns {Array<Object>} Lista com a evolucao do P&L acumulado.
     */
    calculateCumulativePL(bets) {
        const dailyPL = this.calculateDailyPL(bets);
        let runningTotal = 0;

        return dailyPL.map(item => {
            runningTotal += item.pl;
            return {
                date: item.date,
                label: item.label,
                cumulative: parseFloat(runningTotal.toFixed(2))
            };
        });
    },

    /**
     * Calcula o P&L acumulado aposta por aposta (cronologicamente).
     * @param {Array<Object>} bets - Lista de apostas normalizadas.
     * @returns {Array<Object>} Lista com a evolucao do P&L acumulado por aposta.
     */
    calculateCumulativePLBetByBet(bets) {
        if (!Array.isArray(bets) || bets.length === 0) {
            return [];
        }

        // Ordenar apostas da mais antiga para a mais recente
        const sortedBets = [...bets].sort((a, b) => {
            const timeA = (a.datePlaced instanceof Date && !isNaN(a.datePlaced.getTime())) ? a.datePlaced.getTime() : 0;
            const timeB = (b.datePlaced instanceof Date && !isNaN(b.datePlaced.getTime())) ? b.datePlaced.getTime() : 0;
            return timeA - timeB;
        });

        let runningTotal = 0;
        return sortedBets.map((bet, index) => {
            runningTotal += (Number(bet.profitLoss) || 0);
            return {
                label: `Aposta ${index + 1}`,
                cumulative: parseFloat(runningTotal.toFixed(2))
            };
        });
    },

    /**
     * Agrupa as apostas por faixas de odds correspondidas (avgOddsMatched).
     * Faixas: 2-5, 5-10, 10-20, 20-30, 30-50, 50-100, 100+
     * @param {Array<Object>} bets - Lista de apostas normalizadas.
     * @returns {Array<Object>} Estatisticas por faixa de odds.
     */
    calculateByOddsRange(bets) {
        const rangesConfig = [
            { range: '2-5', min: 0, max: 5 },
            { range: '5-10', min: 5, max: 10 },
            { range: '10-20', min: 10, max: 20 },
            { range: '20-30', min: 20, max: 30 },
            { range: '30-50', min: 30, max: 50 },
            { range: '50-100', min: 50, max: 100 },
            { range: '100+', min: 100, max: Infinity }
        ];

        const rangeStats = {};
        rangesConfig.forEach(r => {
            rangeStats[r.range] = {
                range: r.range,
                total: 0,
                wins: 0,
                losses: 0,
                pl: 0
            };
        });

        if (Array.isArray(bets)) {
            bets.forEach(b => {
                const odds = Number(b.avgOddsMatched) || Number(b.oddsRequested) || 0;
                let targetRange = '100+';

                if (odds < 5) {
                    targetRange = '2-5';
                } else if (odds < 10) {
                    targetRange = '5-10';
                } else if (odds < 20) {
                    targetRange = '10-20';
                } else if (odds < 30) {
                    targetRange = '20-30';
                } else if (odds < 50) {
                    targetRange = '30-50';
                } else if (odds < 100) {
                    targetRange = '50-100';
                }

                const stat = rangeStats[targetRange];
                stat.total += 1;
                if (b.isWin) {
                    stat.wins += 1;
                } else {
                    stat.losses += 1;
                }
                stat.pl += (Number(b.profitLoss) || 0);
            });
        }

        return rangesConfig.map(r => {
            const stat = rangeStats[r.range];
            return {
                range: stat.range,
                total: stat.total,
                wins: stat.wins,
                losses: stat.losses,
                pl: parseFloat(stat.pl.toFixed(2)),
                winRate: stat.total > 0 ? parseFloat(((stat.wins / stat.total) * 100).toFixed(2)) : 0
            };
        });
    },

    /**
     * Agrupa as apostas por pista (track) e ordena por volume de apostas decrescente.
     * @param {Array<Object>} bets - Lista de apostas normalizadas.
     * @returns {Array<Object>} Estatisticas por pista.
     */
    calculateByTrack(bets) {
        if (!Array.isArray(bets) || bets.length === 0) {
            return [];
        }

        const trackMap = {};

        bets.forEach(b => {
            const trackName = (b.track && String(b.track).trim()) || 'Desconhecido';
            if (!trackMap[trackName]) {
                trackMap[trackName] = {
                    track: trackName,
                    total: 0,
                    wins: 0,
                    pl: 0
                };
            }

            const stat = trackMap[trackName];
            stat.total += 1;
            if (b.isWin) {
                stat.wins += 1;
            }
            stat.pl += (Number(b.profitLoss) || 0);
        });

        return Object.values(trackMap)
            .map(stat => ({
                track: stat.track,
                total: stat.total,
                wins: stat.wins,
                pl: parseFloat(stat.pl.toFixed(2)),
                winRate: stat.total > 0 ? parseFloat(((stat.wins / stat.total) * 100).toFixed(2)) : 0
            }))
            .sort((a, b) => b.total - a.total);
    },

    /**
     * Agrupa as apostas por hora do dia (0-23) a partir de datePlaced.
     * Inclui todas as horas entre o menor e o maior horario presente nos dados.
     * @param {Array<Object>} bets - Lista de apostas normalizadas.
     * @returns {Array<Object>} Estatisticas por hora ordenadas por horario.
     */
    calculateByHour(bets) {
        if (!Array.isArray(bets) || bets.length === 0) {
            return [];
        }

        const validBets = bets.filter(b => b.datePlaced && b.datePlaced instanceof Date && !isNaN(b.datePlaced.getTime()));
        if (validBets.length === 0) {
            return [];
        }

        const hourMap = {};
        let minHour = 23;
        let maxHour = 0;

        validBets.forEach(b => {
            const hour = b.datePlaced.getHours();
            if (hour < minHour) minHour = hour;
            if (hour > maxHour) maxHour = hour;

            if (!hourMap[hour]) {
                hourMap[hour] = {
                    total: 0,
                    wins: 0,
                    pl: 0
                };
            }

            hourMap[hour].total += 1;
            if (b.isWin) {
                hourMap[hour].wins += 1;
            }
            hourMap[hour].pl += (Number(b.profitLoss) || 0);
        });

        const result = [];
        for (let h = minHour; h <= maxHour; h++) {
            const stat = hourMap[h] || { total: 0, wins: 0, pl: 0 };
            result.push({
                hour: h,
                label: `${h}h`,
                total: stat.total,
                wins: stat.wins,
                pl: parseFloat(stat.pl.toFixed(2))
            });
        }

        return result;
    },

    /**
     * Retorna a taxa de acerto (Win Rate) para cada faixa de odds.
     * @param {Array<Object>} bets - Lista de apostas normalizadas.
     * @returns {Array<Object>} Taxa de acerto por faixa de odds.
     */
    calculateWinRateByOdds(bets) {
        const byOddsRange = this.calculateByOddsRange(bets);
        return byOddsRange.map(item => ({
            range: item.range,
            winRate: item.winRate
        }));
    },

    /**
     * Cria a distribuicao de frequencia (histograma) dos valores de P&L em ~10 intervalos.
     * @param {Array<Object>} bets - Lista de apostas normalizadas.
     * @returns {Array<Object>} Intervalos de P&L com suas respectivas contagens.
     */
    calculatePLDistribution(bets) {
        if (!Array.isArray(bets) || bets.length === 0) {
            return [];
        }

        const values = bets.map(b => Number(b.profitLoss) || 0);
        const minPL = Math.min(...values);
        const maxPL = Math.max(...values);

        // Caso todos os valores sejam identicos
        if (minPL === maxPL) {
            const label = parseFloat(minPL.toFixed(2)).toString();
            return [{
                range: label,
                count: values.length
            }];
        }

        const numBuckets = 10;
        const bucketWidth = (maxPL - minPL) / numBuckets;

        const formatNumber = (num) => {
            const rounded = parseFloat(num.toFixed(2));
            return rounded.toString();
        };

        const buckets = Array.from({ length: numBuckets }, (_, i) => {
            const start = minPL + (i * bucketWidth);
            const end = minPL + ((i + 1) * bucketWidth);
            return {
                start,
                end,
                range: `${formatNumber(start)} a ${formatNumber(end)}`,
                count: 0
            };
        });

        values.forEach(val => {
            let index = Math.floor((val - minPL) / bucketWidth);
            if (index >= numBuckets) {
                index = numBuckets - 1;
            }
            if (index < 0) {
                index = 0;
            }
            buckets[index].count += 1;
        });

        return buckets.map(b => ({
            range: b.range,
            count: b.count
        }));
    }
};

// Suporte para ambiente Node.js / CommonJS para testes unitarios, se aplicavel
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BetMetrics;
}
