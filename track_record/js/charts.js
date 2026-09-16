/**
 * Modulo de criacao e gerenciamento dos graficos do dashboard de apostas.
 * Utiliza a biblioteca Chart.js para renderizacao dos graficos com tema escuro.
 */

const BetCharts = {
    // Armazenamento das instancias ativas dos graficos
    charts: {},

    // Esquema de cores do tema escuro
    colors: {
        green: '#238636',
        greenLight: 'rgba(35, 134, 54, 0.3)',
        red: '#da3633',
        redLight: 'rgba(218, 54, 51, 0.3)',
        blue: '#58a6ff',
        blueLight: 'rgba(88, 166, 255, 0.3)',
        yellow: '#d29922',
        purple: '#bc8cff',
        gray: '#8b949e',
        white: '#e6edf3',
        cardBg: '#161b22',
        gridColor: 'rgba(48, 54, 61, 0.6)'
    },

    /**
     * Retorna a configuracao padrao compartilhada entre todos os graficos.
     * @param {Object} options Configuracoes adicionais customizadas
     * @returns {Object} Configuracao base do Chart.js
     */
    getDefaultOptions(options = {}) {
        const {
            title = '',
            showLegend = false,
            indexAxis = 'x',
            yAxisFormatter = null,
            xAxisFormatter = null,
            tooltipFormatter = null,
            yMin = undefined,
            yMax = undefined
        } = options;

        return {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: indexAxis,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: showLegend,
                    labels: {
                        color: this.colors.gray,
                        font: {
                            family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                            size: 12
                        },
                        boxWidth: 12,
                        padding: 15
                    }
                },
                title: {
                    display: Boolean(title),
                    text: title,
                    color: this.colors.white,
                    font: {
                        family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                        size: 14,
                        weight: '600'
                    },
                    padding: {
                        top: 5,
                        bottom: 15
                    }
                },
                tooltip: {
                    backgroundColor: this.colors.cardBg,
                    titleColor: this.colors.white,
                    bodyColor: this.colors.white,
                    borderColor: this.colors.gridColor,
                    borderWidth: 1,
                    padding: 10,
                    cornerRadius: 6,
                    titleFont: {
                        family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                        size: 12,
                        weight: 'bold'
                    },
                    bodyFont: {
                        family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                        size: 12
                    },
                    callbacks: tooltipFormatter ? { label: tooltipFormatter } : {}
                }
            },
            scales: {
                x: {
                    grid: {
                        color: this.colors.gridColor,
                        borderColor: this.colors.gridColor
                    },
                    ticks: {
                        color: this.colors.gray,
                        font: {
                            family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                            size: 11
                        },
                        ...(xAxisFormatter ? { callback: xAxisFormatter } : {})
                    }
                },
                y: {
                    min: yMin,
                    max: yMax,
                    grid: {
                        color: this.colors.gridColor,
                        borderColor: this.colors.gridColor
                    },
                    ticks: {
                        color: this.colors.gray,
                        font: {
                            family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                            size: 11
                        },
                        ...(yAxisFormatter ? { callback: yAxisFormatter } : {})
                    }
                }
            }
        };
    },

    /**
     * Inicializa todos os graficos a partir do objeto de metricas.
     * @param {Object} metrics Objeto retornado por BetMetrics.calculateAll()
     */
    init(metrics) {
        if (!metrics) return;

        this.destroyAll();

        if (metrics.dailyPL) {
            this.createDailyPLChart(metrics.dailyPL);
        }

        if (metrics.monthlyPL) {
            this.createMonthlyPLChart(metrics.monthlyPL);
        }

        if (metrics.cumulativePL) {
            this.createCumulativePLChart(metrics.cumulativePL);
        }

        if (metrics.byOddsRange) {
            this.createOddsRangeChart(metrics.byOddsRange);
        }

        if (metrics.byTrack) {
            this.createTrackChart(metrics.byTrack);
        }

        if (metrics.byHour) {
            this.createHourlyChart(metrics.byHour);
        }

        if (metrics.winRateByOdds) {
            this.createWinRateByOddsChart(metrics.winRateByOdds);
        } else if (metrics.byOddsRange) {
            this.createWinRateByOddsChart(metrics.byOddsRange);
        }

        if (metrics.plDistribution) {
            this.createPLDistributionChart(metrics.plDistribution);
        }
    },

    /**
     * Destroi todas as instancias de graficos ativas para evitar vazamentos de memoria.
     */
    destroyAll() {
        Object.keys(this.charts).forEach(key => {
            if (this.charts[key] && typeof this.charts[key].destroy === 'function') {
                this.charts[key].destroy();
            }
        });
        this.charts = {};
    },

    /**
     * Grafico 1: P&L Diario (Grafico de barras)
     * Canvas ID: dailyPLChart
     * @param {Array|Object} dailyPL Dados de P&L por dia
     */
    createDailyPLChart(dailyPL) {
        const canvas = document.getElementById('dailyPLChart');
        if (!canvas) return;

        if (this.charts.dailyPL) {
            this.charts.dailyPL.destroy();
        }

        let labels = [];
        let dataValues = [];

        if (Array.isArray(dailyPL)) {
            labels = dailyPL.map(item => item.label || item.date || item.day || '');
            dataValues = dailyPL.map(item => {
                if (typeof item.pl === 'number') return item.pl;
                if (typeof item.value === 'number') return item.value;
                if (typeof item.profit === 'number') return item.profit;
                return Number(item) || 0;
            });
        } else if (dailyPL && typeof dailyPL === 'object') {
            labels = dailyPL.labels || Object.keys(dailyPL);
            dataValues = dailyPL.data || dailyPL.values || Object.values(dailyPL);
        }

        const backgroundColors = dataValues.map(val => val >= 0 ? this.colors.green : this.colors.red);
        const borderColors = dataValues.map(val => val >= 0 ? this.colors.green : this.colors.red);

        const options = this.getDefaultOptions({
            title: 'P&L Diario',
            yAxisFormatter: (val) => '€ ' + Number(val).toFixed(2),
            tooltipFormatter: (context) => {
                const val = context.raw;
                return `P&L: ${val >= 0 ? '+' : ''}€ ${Number(val).toFixed(2)}`;
            }
        });

        this.charts.dailyPL = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'P&L Diario',
                    data: dataValues,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: options
        });

        return this.charts.dailyPL;
    },

    /**
     * Grafico de P&L Mensal (Grafico de barras com ROI e Win Rate no tooltip)
     * Canvas ID: monthlyPLChart
     * @param {Array} monthlyPL Dados de P&L por mes
     */
    createMonthlyPLChart(monthlyPL) {
        const canvas = document.getElementById('monthlyPLChart');
        if (!canvas) return;

        if (this.charts.monthlyPL) {
            this.charts.monthlyPL.destroy();
        }

        let labels = [];
        let dataValues = [];
        let rois = [];
        let winRates = [];

        if (Array.isArray(monthlyPL)) {
            labels = monthlyPL.map(item => item.label || '');
            dataValues = monthlyPL.map(item => item.pl || 0);
            rois = monthlyPL.map(item => item.roi || 0);
            winRates = monthlyPL.map(item => item.winRate || 0);
        }

        const backgroundColors = dataValues.map(val => val >= 0 ? this.colors.green : this.colors.red);
        const borderColors = dataValues.map(val => val >= 0 ? this.colors.green : this.colors.red);

        const options = this.getDefaultOptions({
            title: 'P&L Mensal',
            yAxisFormatter: (val) => '€ ' + Number(val).toFixed(2),
            tooltipFormatter: (context) => {
                const index = context.dataIndex;
                const val = context.raw;
                const roi = rois[index];
                const winRate = winRates[index];
                
                // Retorna um array para ter multiplas linhas no tooltip
                return [
                    `P&L: ${val >= 0 ? '+' : ''}€ ${Number(val).toFixed(2)}`,
                    `ROI: ${roi}%`,
                    `Win Rate: ${winRate}%`
                ];
            }
        });

        this.charts.monthlyPL = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'P&L Mensal',
                    data: dataValues,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: options
        });

        return this.charts.monthlyPL;
    },

    /**
     * Grafico 2: P&L Acumulado (Grafico de linha suave)
     * Canvas ID: cumulativePLChart
     * @param {Array|Object} cumulativePL Dados de evolucao acumulada de P&L
     */
    createCumulativePLChart(cumulativePL) {
        const canvas = document.getElementById('cumulativePLChart');
        if (!canvas) return;

        if (this.charts.cumulativePL) {
            this.charts.cumulativePL.destroy();
        }

        let labels = [];
        let dataValues = [];

        if (Array.isArray(cumulativePL)) {
            labels = cumulativePL.map(item => item.label || item.date || item.day || '');
            dataValues = cumulativePL.map(item => {
                if (typeof item.cumulative === 'number') return item.cumulative;
                if (typeof item.cumulativePL === 'number') return item.cumulativePL;
                if (typeof item.pl === 'number') return item.pl;
                if (typeof item.value === 'number') return item.value;
                return Number(item) || 0;
            });
        } else if (cumulativePL && typeof cumulativePL === 'object') {
            labels = cumulativePL.labels || Object.keys(cumulativePL);
            dataValues = cumulativePL.data || cumulativePL.values || Object.values(cumulativePL);
        }

        const options = this.getDefaultOptions({
            title: 'P&L Acumulado',
            yAxisFormatter: (val) => '€ ' + Number(val).toFixed(2),
            tooltipFormatter: (context) => {
                const val = context.raw;
                return `Acumulado: ${val >= 0 ? '+' : ''}€ ${Number(val).toFixed(2)}`;
            }
        });

        this.charts.cumulativePL = new Chart(canvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'P&L Acumulado',
                    data: dataValues,
                    borderColor: this.colors.blue,
                    backgroundColor: this.colors.blueLight,
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: dataValues.length > 50 ? 0 : 3,
                    pointHoverRadius: 5,
                    pointBackgroundColor: this.colors.blue,
                    pointBorderColor: this.colors.cardBg
                }]
            },
            options: options
        });

        return this.charts.cumulativePL;
    },

    /**
     * Grafico 3: Apostas por Faixa de Odds (Grafico de barras agrupadas)
     * Canvas ID: oddsRangeChart
     * @param {Array|Object} byOddsRange Dados de apostas agrupadas por faixa de odds
     */
    createOddsRangeChart(byOddsRange) {
        const canvas = document.getElementById('oddsRangeChart');
        if (!canvas) return;

        if (this.charts.oddsRange) {
            this.charts.oddsRange.destroy();
        }

        let labels = [];
        let totalBets = [];
        let wonBets = [];

        if (Array.isArray(byOddsRange)) {
            labels = byOddsRange.map(item => item.range || item.label || '');
            totalBets = byOddsRange.map(item => item.total !== undefined ? item.total : (item.count !== undefined ? item.count : 0));
            wonBets = byOddsRange.map(item => item.won !== undefined ? item.won : (item.wins !== undefined ? item.wins : 0));
        } else if (byOddsRange && typeof byOddsRange === 'object') {
            labels = byOddsRange.labels || Object.keys(byOddsRange);
            if (byOddsRange.total && byOddsRange.won) {
                totalBets = byOddsRange.total;
                wonBets = byOddsRange.won;
            } else {
                labels.forEach(key => {
                    const item = byOddsRange[key];
                    totalBets.push(item.total !== undefined ? item.total : (item.count || 0));
                    wonBets.push(item.won !== undefined ? item.won : (item.wins || 0));
                });
            }
        }

        const options = this.getDefaultOptions({
            title: 'Apostas por Faixa de Odds',
            showLegend: true,
            tooltipFormatter: (context) => {
                const label = context.dataset.label || '';
                const val = context.raw;
                return `${label}: ${val}`;
            }
        });

        this.charts.oddsRange = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Total de Apostas',
                        data: totalBets,
                        backgroundColor: this.colors.blue,
                        borderRadius: 4,
                        borderWidth: 0
                    },
                    {
                        label: 'Apostas Ganhas',
                        data: wonBets,
                        backgroundColor: this.colors.green,
                        borderRadius: 4,
                        borderWidth: 0
                    }
                ]
            },
            options: options
        });

        return this.charts.oddsRange;
    },

    /**
     * Grafico 4: Performance por Pista (Grafico de barras horizontal)
     * Exibe o top 15 pistas ordenadas pelo total de apostas.
     * Canvas ID: trackChart
     * @param {Array|Object} byTrack Dados de desempenho por pista
     */
    createTrackChart(byTrack) {
        const canvas = document.getElementById('trackChart');
        if (!canvas) return;

        if (this.charts.track) {
            this.charts.track.destroy();
        }

        let items = [];

        if (Array.isArray(byTrack)) {
            items = [...byTrack];
        } else if (byTrack && typeof byTrack === 'object') {
            items = Object.keys(byTrack).map(trackName => ({
                track: trackName,
                ...byTrack[trackName]
            }));
        }

        // Ordenar pelo total de apostas e selecionar top 15
        items.sort((a, b) => {
            const totalA = a.total !== undefined ? a.total : (a.count || 0);
            const totalB = b.total !== undefined ? b.total : (b.count || 0);
            return totalB - totalA;
        });

        const topTracks = items.slice(0, 15);
        const labels = topTracks.map(item => item.track || item.name || item.label || '');
        const dataValues = topTracks.map(item => {
            if (typeof item.pl === 'number') return item.pl;
            if (typeof item.profit === 'number') return item.profit;
            if (typeof item.value === 'number') return item.value;
            return 0;
        });

        const backgroundColors = dataValues.map(val => val >= 0 ? this.colors.green : this.colors.red);
        const borderColors = dataValues.map(val => val >= 0 ? this.colors.green : this.colors.red);

        const options = this.getDefaultOptions({
            title: 'Performance por Pista',
            indexAxis: 'y',
            xAxisFormatter: (val) => '€ ' + Number(val).toFixed(2),
            tooltipFormatter: (context) => {
                const val = context.raw;
                return `P&L: ${val >= 0 ? '+' : ''}€ ${Number(val).toFixed(2)}`;
            }
        });

        // Reverte interação para o padrão do Chart.js especificamente neste gráfico
        options.interaction = {
            mode: 'nearest',
            intersect: true
        };

        this.charts.track = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'P&L por Pista',
                    data: dataValues,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: options
        });

        return this.charts.track;
    },

    /**
     * Grafico 5: Atividade por Hora do Dia (Grafico de barras)
     * Canvas ID: hourlyChart
     * @param {Array|Object} byHour Dados de volume de apostas por hora
     */
    createHourlyChart(byHour) {
        const canvas = document.getElementById('hourlyChart');
        if (!canvas) return;

        if (this.charts.hourly) {
            this.charts.hourly.destroy();
        }

        let labels = [];
        let dataValues = [];

        if (Array.isArray(byHour)) {
            labels = byHour.map(item => {
                if (item.hour !== undefined) {
                    return typeof item.hour === 'number' ? `${item.hour}h` : `${item.hour}`;
                }
                return item.label || '';
            });
            dataValues = byHour.map(item => {
                if (typeof item.count === 'number') return item.count;
                if (typeof item.total === 'number') return item.total;
                if (typeof item.bets === 'number') return item.bets;
                if (typeof item.value === 'number') return item.value;
                return Number(item) || 0;
            });
        } else if (byHour && typeof byHour === 'object') {
            const keys = Object.keys(byHour);
            labels = keys.map(k => k.endsWith('h') ? k : `${k}h`);
            dataValues = keys.map(k => {
                const val = byHour[k];
                if (typeof val === 'object' && val !== null) {
                    return val.count !== undefined ? val.count : (val.total || 0);
                }
                return Number(val) || 0;
            });
        }

        const options = this.getDefaultOptions({
            title: 'Atividade por Hora do Dia',
            tooltipFormatter: (context) => {
                const val = context.raw;
                return `Apostas: ${val}`;
            }
        });

        this.charts.hourly = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Numero de Apostas',
                    data: dataValues,
                    backgroundColor: this.colors.blue,
                    borderRadius: 4,
                    borderWidth: 0
                }]
            },
            options: options
        });

        return this.charts.hourly;
    },

    /**
     * Grafico 6: Taxa de Acerto por Faixa de Odds (Grafico de linha com pontos)
     * Canvas ID: winRateOddsChart
     * @param {Array|Object} winRateByOdds Dados de taxa de acerto por faixa de odds
     */
    createWinRateByOddsChart(winRateByOdds) {
        const canvas = document.getElementById('winRateOddsChart');
        if (!canvas) return;

        if (this.charts.winRateOdds) {
            this.charts.winRateOdds.destroy();
        }

        let labels = [];
        let dataValues = [];

        if (Array.isArray(winRateByOdds)) {
            labels = winRateByOdds.map(item => item.range || item.label || '');
            dataValues = winRateByOdds.map(item => {
                if (typeof item.winRate === 'number') return Number(item.winRate.toFixed(1));
                if (typeof item.rate === 'number') return Number(item.rate.toFixed(1));
                if (item.total && item.won !== undefined) {
                    return Number(((item.won / item.total) * 100).toFixed(1));
                }
                return 0;
            });
        } else if (winRateByOdds && typeof winRateByOdds === 'object') {
            labels = winRateByOdds.labels || Object.keys(winRateByOdds);
            if (winRateByOdds.data || winRateByOdds.values) {
                dataValues = winRateByOdds.data || winRateByOdds.values;
            } else {
                dataValues = labels.map(k => {
                    const item = winRateByOdds[k];
                    if (typeof item === 'number') return Number(item.toFixed(1));
                    if (item && typeof item.winRate === 'number') return Number(item.winRate.toFixed(1));
                    if (item && item.total && item.won !== undefined) {
                        return Number(((item.won / item.total) * 100).toFixed(1));
                    }
                    return 0;
                });
            }
        }

        const options = this.getDefaultOptions({
            title: 'Taxa de Acerto por Faixa de Odds',
            yMin: 0,
            yMax: 100,
            yAxisFormatter: (val) => `${val}%`,
            tooltipFormatter: (context) => {
                const val = context.raw;
                return `Taxa de Acerto: ${val}%`;
            }
        });

        this.charts.winRateOdds = new Chart(canvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Taxa de Acerto (%)',
                    data: dataValues,
                    borderColor: this.colors.yellow,
                    backgroundColor: this.colors.yellow,
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: this.colors.yellow,
                    pointBorderColor: this.colors.cardBg,
                    pointBorderWidth: 2
                }]
            },
            options: options
        });

        return this.charts.winRateOdds;
    },

    /**
     * Grafico 7: Distribuicao de Resultados (Grafico de barras)
     * Canvas ID: plDistChart
     * @param {Array|Object} plDistribution Dados da distribuicao de P&L em intervalos
     */
    createPLDistributionChart(plDistribution) {
        const canvas = document.getElementById('plDistChart');
        if (!canvas) return;

        if (this.charts.plDistribution) {
            this.charts.plDistribution.destroy();
        }

        let labels = [];
        let dataValues = [];

        if (Array.isArray(plDistribution)) {
            labels = plDistribution.map(item => item.range || item.bucket || item.label || '');
            dataValues = plDistribution.map(item => {
                if (typeof item.count === 'number') return item.count;
                if (typeof item.total === 'number') return item.total;
                if (typeof item.value === 'number') return item.value;
                return Number(item) || 0;
            });
        } else if (plDistribution && typeof plDistribution === 'object') {
            labels = plDistribution.labels || Object.keys(plDistribution);
            dataValues = plDistribution.data || plDistribution.values || Object.values(plDistribution);
        }

        const options = this.getDefaultOptions({
            title: 'Distribuicao de Resultados',
            tooltipFormatter: (context) => {
                const val = context.raw;
                return `Quantidade: ${val}`;
            }
        });

        this.charts.plDistribution = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Numero de Apostas',
                    data: dataValues,
                    backgroundColor: this.colors.purple,
                    borderRadius: 4,
                    borderWidth: 0
                }]
            },
            options: options
        });

        return this.charts.plDistribution;
    }
};
