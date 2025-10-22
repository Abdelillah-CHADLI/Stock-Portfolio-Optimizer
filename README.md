# Stock Portfolio Optimizer

A portfolio optimization system using advanced search algorithms including A*, Greedy Search, CSP, and Simulated Annealing. 

Built with Flask (backend) and React (frontend), providing an interactive web interface for simulating and comparing different optimization strategies.

## 📘 Background

This project was originally developed as a second-year group project under the module **Introduction to Artificial Intelligence** at the University of Boumerdes (UMBB).

**Team members (original version):**
- Mohamed Abdelillah Chadli
- Madjd Baghdadi
- Kossai Baha
- Yassir Cherdouh
- Ali Habbeche
- Mehdi Bouzoul

**Improved and extended version:**

The system was later enhanced and refactored by **Mohamed Abdelillah Chadli** — who added a modern frontend (React app) and connected it to the Flask-based backend, offering real-time visualization and easier interaction with the algorithms.

## Features

- Portfolio configuration with customizable parameters:
  - Investment budget
  - Diversification limits
  - Risk tolerance levels
- Stock selection from historical data
- Multiple optimization algorithms available
- Performance metrics tracking (ROI, risk, diversification)
- Interactive graphs and charts
- Exportable trading history (CSV format)

## Project Structure

```
Stock-Portfolio-Optimizer/
├── backend/
│   ├── app.py                          # Flask backend server
│   ├── backend/PortfolioOptimizer.py   # Core optimization logic
│   └── ...
│
├── frontend/
│   ├── src/
│   │   └── app.js                      # React main component
│   └── package.json
│
├── my_data/
│   └── stocks.json                     # Stock price and metadata
│
└── README.md
```

**Backend:**
- Python (Flask + CORS)
- Core logic implemented in `PortfolioOptimizer`
- Exposes `/api/stocks` and `/api/optimize` endpoints

**Frontend:**
- React + TailwindCSS + Recharts + Lucide Icons
- Connects to backend using REST API
- Provides a dashboard for visualization and control

## Optimization Algorithms

The system implements four different optimization strategies:

| Algorithm | Description |
|-----------|-------------|
| **A*** | Informed search algorithm that finds optimal solutions using admissible heuristics |
| **Greedy Search** | Fast approximation algorithm that makes locally optimal choices |
| **CSP** | Constraint Satisfaction Problem solver that ensures all portfolio requirements are met |
| **Simulated Annealing** | Probabilistic optimization method that can avoid local optima |

Each algorithm offers different trade-offs between execution speed and solution quality. The choice depends on your specific needs - quick results vs optimal solutions.

## 🖥️ Running the Application

### 1. Backend Setup (Flask)

```bash
cd backend
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

By default, it runs on:
```
http://localhost:5000
```

### 2. Frontend Setup (React)

```bash
cd frontend
npm install
npm start
```

Runs on:
```
http://localhost:3000
```

If necessary, create a `.env` file in `frontend/` with:
```
REACT_APP_API_URL=http://localhost:5000
```

## 📊 Example Workflow

1. Select desired stocks (e.g., AAPL, AMZN, JNJ).
2. Configure:
   - Budget = 10,000 USD
   - Diversification = 50% max per stock
   - Risk tolerance = Moderate
3. Choose algorithm (e.g., A*, Greedy, CSP, SA).
4. Run optimization → visualize results:
   - Portfolio growth chart
   - ROI and risk metrics
   - Allocation pie chart
   - Trading actions history

## 📈 Sample Output

- **Final Portfolio Value:** $11,320.45
- **ROI:** +13.2%
- **Risk:** 4.5%
- **Diversified Assets:** 5

*(Values vary depending on algorithm and dataset.)*

## ⚙️ Future Improvements

- Integrate live market data APIs (e.g., Yahoo Finance)
- Add Genetic Algorithm (GA) and Particle Swarm Optimization (PSO)
- Backend multiprocessing for faster optimization
- Enhanced risk modeling (Sharpe ratio, volatility tracking)
- Cloud deployment (Render / Vercel)

## 🧾 License

This project is for educational and research purposes only. Not intended for financial or investment use.