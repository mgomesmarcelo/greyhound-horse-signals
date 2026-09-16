const App = {
    bets: [],
    filteredBets: [],
    currentSort: { column: 'datePlaced', direction: 'desc' },
    cumulativeMode: 'daily', // 'daily' ou 'betByBet'

    // Inicializa a aplicação
    init() {
        this.loadData();

        // Configura listeners para os filtros
        const filterTrack = document.getElementById('filterTrack');
        const filterDateFrom = document.getElementById('filterDateFrom');
        const filterDateTo = document.getElementById('filterDateTo');
        
        if (filterTrack) filterTrack.addEventListener('change', () => this.applyFilters());

        const filterQuickDate = document.getElementById('filterQuickDate');
        if (filterQuickDate) {
            filterQuickDate.addEventListener('change', (e) => this.handleQuickDateChange(e.target.value));
        }

        if (filterDateFrom) {
            filterDateFrom.addEventListener('change', () => {
                if (filterQuickDate) filterQuickDate.value = 'custom';
            });
        }
        if (filterDateTo) {
            filterDateTo.addEventListener('change', () => {
                if (filterQuickDate) filterQuickDate.value = 'custom';
            });
        }

        const btnApplyFilters = document.getElementById('btnApplyFilters');
        if (btnApplyFilters) btnApplyFilters.addEventListener('click', () => this.applyFilters());

        const btnClearFilters = document.getElementById('btnClearFilters');
        if (btnClearFilters) {
            btnClearFilters.addEventListener('click', () => {
                if (filterTrack) filterTrack.value = '';
                if (filterQuickDate) filterQuickDate.value = 'all';
                if (filterDateFrom) filterDateFrom.value = '';
                if (filterDateTo) filterDateTo.value = '';
                this.filteredBets = [...this.bets];
                this.updateDashboard(this.bets);
            });
        }

        // Configura listener para pesquisa na tabela
        const searchInput = document.getElementById('tableSearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => this.searchTable(e.target.value));
        }

        // Configura listeners para ordenação da tabela
        const tableHeaders = document.querySelectorAll('#betsTable th[data-sort]');
        tableHeaders.forEach(th => {
            th.addEventListener('click', () => {
                const column = th.getAttribute('data-sort');
                this.sortTable(column);
            });
        });

        // Configura listener para alternar modo do grafico cumulativo
        const btnToggleCumulative = document.getElementById('btnToggleCumulative');
        if (btnToggleCumulative) {
            btnToggleCumulative.addEventListener('click', () => {
                this.cumulativeMode = this.cumulativeMode === 'daily' ? 'betByBet' : 'daily';
                btnToggleCumulative.textContent = this.cumulativeMode === 'daily' ? 'Ver Bet a Bet' : 'Ver por Dia';
                this.updateDashboard(this.filteredBets, true); // true indica que é apenas atualização parcial (graficos)
            });
        }
    },

    // Carrega dados da API do servidor
    async loadData() {
        this.showLoading();
        try {
            const response = await fetch('/api/bets');
            if (!response.ok) throw new Error('Não foi possível carregar os dados do servidor.');
            
            const rawBets = await response.json();
            
            if (!rawBets || rawBets.length === 0) {
                this.showEmptyState();
                return;
            }

            // Converte strings de volta para objetos Date
            this.bets = rawBets.map(bet => ({
                ...bet,
                datePlaced: new Date(bet.datePlaced)
            }));
            
            this.filteredBets = [...this.bets];
            this.updateDashboard(this.bets);
        } catch (error) {
            console.error('Erro ao buscar os dados da API:', error);
            this.showEmptyState();
        } finally {
            this.hideLoading();
        }
    },

    // Aplica os filtros na lista de apostas
    applyFilters() {
        const filterTrack = document.getElementById('filterTrack')?.value || '';
        const filterDateFrom = document.getElementById('filterDateFrom')?.value;
        const filterDateTo = document.getElementById('filterDateTo')?.value;

        this.filteredBets = this.bets.filter(bet => {
            // Filtro de Pista
            if (filterTrack && bet.track !== filterTrack) {
                return false;
            }

            // Filtros de Data
            if (filterDateFrom) {
                const fromDate = new Date(filterDateFrom);
                if (bet.datePlaced < fromDate) return false;
            }
            if (filterDateTo) {
                const toDate = new Date(filterDateTo);
                // Adiciona 1 dia para incluir o dia inteiro de 'to'
                toDate.setDate(toDate.getDate() + 1);
                if (bet.datePlaced >= toDate) return false;
            }

            return true;
        });

        this.updateDashboard(this.filteredBets);
    },

    // Manipula a troca de data rápida
    handleQuickDateChange(preset) {
        const dateFrom = document.getElementById('filterDateFrom');
        const dateTo = document.getElementById('filterDateTo');
        if (!dateFrom || !dateTo) return;

        const today = new Date();
        let from, to;

        const formatDate = (d) => {
            const yyyy = d.getFullYear();
            const mm = String(d.getMonth() + 1).padStart(2, '0');
            const dd = String(d.getDate()).padStart(2, '0');
            return `${yyyy}-${mm}-${dd}`;
        };

        switch(preset) {
            case 'thisMonth':
                from = new Date(today.getFullYear(), today.getMonth(), 1);
                to = new Date(today.getFullYear(), today.getMonth() + 1, 0);
                break;
            case 'lastMonth':
                from = new Date(today.getFullYear(), today.getMonth() - 1, 1);
                to = new Date(today.getFullYear(), today.getMonth(), 0);
                break;
            case 'last3Months':
                from = new Date();
                from.setMonth(from.getMonth() - 3);
                to = today;
                break;
            case 'last6Months':
                from = new Date();
                from.setMonth(from.getMonth() - 6);
                to = today;
                break;
            case 'thisYear':
                from = new Date(today.getFullYear(), 0, 1);
                to = new Date(today.getFullYear(), 11, 31);
                break;
            case 'all':
            case 'custom':
            default:
                dateFrom.value = '';
                dateTo.value = '';
                if (preset === 'all') this.applyFilters();
                return;
        }

        dateFrom.value = formatDate(from);
        dateTo.value = formatDate(to);
        
        this.applyFilters();
    },

    // Atualiza todo o dashboard
    updateDashboard(betsToRender, skipTable = false) {
        if (!betsToRender || betsToRender.length === 0) {
            this.showEmptyState();
            return;
        }

        this.hideEmptyState();

        // Calcula métricas
        const metrics = BetMetrics.calculateAll(betsToRender);
        
        // Se estamos no modo betByBet, substituimos a metrica original para o grafico renderizar a nova
        if (this.cumulativeMode === 'betByBet' && metrics.cumulativePLBetByBet) {
            metrics.cumulativePL = metrics.cumulativePLBetByBet;
        }

        // Atualiza UI
        this.renderKPIs(metrics.kpis);
        
        // Atualiza Gráficos (limpa os antigos primeiro)
        if (typeof BetCharts !== 'undefined' && BetCharts.destroyAll) {
            BetCharts.destroyAll();
        }
        if (typeof BetCharts !== 'undefined') {
            BetCharts.init(metrics);
        }
        
        // Atualiza Tabela apenas se não for especificado para pular
        if (!skipTable) {
            this.renderTable(betsToRender);
            // Preenche o filtro de pistas usando TODAS as apostas, para manter as opções consistentes
            this.populateTrackFilter();
        }
    },

    // Preenche o dropdown de selecao de pista
    populateTrackFilter() {
        const filterTrack = document.getElementById('filterTrack');
        if (!filterTrack) return;

        // Guarda o valor atual
        const currentValue = filterTrack.value;
        
        // Pega todas as pistas unicas
        const tracks = [...new Set(this.bets.map(b => b.track).filter(t => t))].sort();
        
        // Reconstroi as opcoes
        filterTrack.innerHTML = '<option value="">Todas as Pistas</option>';
        tracks.forEach(track => {
            const option = document.createElement('option');
            option.value = track;
            option.textContent = track;
            filterTrack.appendChild(option);
        });

        // Restaura valor anterior se ainda existir
        if (tracks.includes(currentValue)) {
            filterTrack.value = currentValue;
        }
    },

    // Atualiza as metricas (KPIs) na tela
    renderKPIs(kpis) {
        const elTotalBets = document.getElementById('kpiTotalBets');
        const elWinRate = document.getElementById('kpiWinRate');
        const elTotalPL = document.getElementById('kpiTotalPL');
        const elTotalUnits = document.getElementById('kpiTotalUnits');
        const elROI = document.getElementById('kpiROI');
        const elAvgPL = document.getElementById('kpiAvgPL');
        const elMaxWin = document.getElementById('kpiMaxWin');
        const elMaxLoss = document.getElementById('kpiMaxLoss');
        const elMaxCum = document.getElementById('kpiMaxCum');
        const elMinCum = document.getElementById('kpiMinCum');

        if (elTotalBets) elTotalBets.textContent = kpis.totalBets;
        if (elWinRate) elWinRate.textContent = kpis.winRate + '%';
        
        if (elTotalPL) {
            elTotalPL.textContent = this.formatCurrency(kpis.totalPL);
            elTotalPL.className = 'kpi-value ' + (kpis.totalPL >= 0 ? 'kpi-positive' : 'kpi-negative');
        }

        if (elTotalUnits) {
            const sign = kpis.totalUnits >= 0 ? '+' : '';
            elTotalUnits.textContent = `${sign}${kpis.totalUnits} u`;
            elTotalUnits.className = 'kpi-value ' + (kpis.totalUnits >= 0 ? 'kpi-positive' : 'kpi-negative');
        }
        
        if (elROI) elROI.textContent = kpis.roi + '%';
        if (elAvgPL) elAvgPL.textContent = this.formatCurrency(kpis.avgPLPerBet);
        if (elMaxWin) elMaxWin.textContent = this.formatCurrency(kpis.maxWin);
        if (elMaxLoss) elMaxLoss.textContent = this.formatCurrency(kpis.maxLoss);
        if (elMaxCum) elMaxCum.textContent = this.formatCurrency(kpis.maxCumulativePL);
        if (elMinCum) elMinCum.textContent = this.formatCurrency(kpis.minCumulativePL);
    },

    // Renderiza a tabela de apostas (todas)
    renderTable(betsToRender) {
        const tbody = document.querySelector('#betsTable tbody');
        if (!tbody) return;

        tbody.innerHTML = '';

        for (let i = 0; i < betsToRender.length; i++) {
            const bet = betsToRender[i];
            const tr = document.createElement('tr');
            
            // Classe para facilitar estilizacao da linha baseada em lucro/prejuizo
            tr.className = bet.profitLoss >= 0 ? 'row-positive' : 'row-negative';

            tr.innerHTML = `
                <td>${this.formatDate(bet.datePlaced)}</td>
                <td>${bet.track || ''}</td>
                <td>${bet.selection}</td>
                <td>${bet.avgOddsMatched ? bet.avgOddsMatched.toFixed(2) : '0.00'}</td>
                <td>${bet.stake ? bet.stake.toFixed(2) : '0.00'}</td>
                <td>${bet.liability ? bet.liability.toFixed(2) : '0.00'}</td>
                <td class="${bet.profitLoss >= 0 ? 'text-green' : 'text-red'}">
                    ${this.formatCurrency(bet.profitLoss)}
                </td>
            `;
            tbody.appendChild(tr);
        }
    },

    // Ordena a tabela por uma determinada coluna
    sortTable(column) {
        if (this.currentSort.column === column) {
            this.currentSort.direction = this.currentSort.direction === 'asc' ? 'desc' : 'asc';
        } else {
            this.currentSort.column = column;
            this.currentSort.direction = 'desc'; // padrão inicial
        }

        const modifier = this.currentSort.direction === 'asc' ? 1 : -1;

        this.filteredBets.sort((a, b) => {
            let valA = a[column];
            let valB = b[column];

            if (typeof valA === 'string') valA = valA.toLowerCase();
            if (typeof valB === 'string') valB = valB.toLowerCase();

            if (valA < valB) return -1 * modifier;
            if (valA > valB) return 1 * modifier;
            return 0;
        });

        // Após ordenar, re-renderiza a tabela
        this.renderTable(this.filteredBets);
    },

    // Realiza busca nas linhas da tabela (client-side simple search)
    searchTable(query) {
        const lowerQuery = query.toLowerCase();
        const trs = document.querySelectorAll('#betsTable tbody tr');
        
        trs.forEach(tr => {
            const textContent = tr.textContent.toLowerCase();
            tr.style.display = textContent.includes(lowerQuery) ? '' : 'none';
        });
    },

    // Formata valores financeiros em Euro
    formatCurrency(value) {
        const sign = value >= 0 ? '+' : '-';
        const absVal = Math.abs(value).toFixed(2);
        return `${sign}€${absVal}`;
    },

    // Formata a data (DD/MM/YYYY HH:MM)
    formatDate(date) {
        if (!date || isNaN(date.getTime())) return '';
        const pad = n => n.toString().padStart(2, '0');
        
        const dd = pad(date.getDate());
        const mm = pad(date.getMonth() + 1);
        const yyyy = date.getFullYear();
        const hh = pad(date.getHours());
        const mins = pad(date.getMinutes());

        return `${dd}/${mm}/${yyyy} ${hh}:${mins}`;
    },

    // Mostra indicador de carregamento
    showLoading() {
        const loader = document.getElementById('loading');
        if (loader) loader.style.display = 'flex';
    },

    // Oculta indicador de carregamento
    hideLoading() {
        const loader = document.getElementById('loading');
        if (loader) loader.style.display = 'none';
    },

    // Mostra estado inicial sem dados
    showEmptyState() {
        const emptyState = document.getElementById('emptyState');
        const dashboard = document.getElementById('dashboardContent');
        
        if (emptyState) emptyState.style.display = 'block';
        if (dashboard) dashboard.style.display = 'none';
    },

    // Oculta estado inicial e exibe o dashboard
    hideEmptyState() {
        const emptyState = document.getElementById('emptyState');
        const dashboard = document.getElementById('dashboardContent');
        
        if (emptyState) emptyState.style.display = 'none';
        if (dashboard) dashboard.style.display = 'block';
    }
};

// Inicia a aplicação quando o DOM estiver pronto
document.addEventListener('DOMContentLoaded', () => App.init());
