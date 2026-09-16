const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const Papa = require('papaparse');

// Disponibiliza o Papa globalmente para que o parser.js (feito para o navegador) funcione no Node.js
global.Papa = Papa;

const BetParser = require('./js/parser');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());

// Serve todos os arquivos estáticos da pasta atual (HTML, CSS, JS)
app.use(express.static(__dirname));

// Rota da API para buscar e unificar todos os CSVs da pasta data/
app.get('/api/bets', (req, res) => {
    const dataDir = path.join(__dirname, 'data');
    let allBets = [];
    
    try {
        if (fs.existsSync(dataDir)) {
            const files = fs.readdirSync(dataDir);
            
            files.forEach(file => {
                if (file.toLowerCase().endsWith('.csv')) {
                    const filePath = path.join(dataDir, file);
                    const csvText = fs.readFileSync(filePath, 'utf8');
                    
                    // Utiliza o parser já existente para normalizar os dados
                    const parsedBets = BetParser.parseCSV(csvText);
                    
                    // Mescla os dados sem duplicatas
                    allBets = BetParser.mergeData(allBets, parsedBets);
                }
            });
        }
        
        res.json(allBets);
    } catch (error) {
        console.error('Erro ao ler a pasta data:', error);
        res.status(500).json({ error: 'Erro interno ao processar os dados.' });
    }
});

app.listen(PORT, () => {
    console.log(`Servidor rodando na porta ${PORT}`);
    console.log(`Acesse: http://localhost:${PORT}`);
});
