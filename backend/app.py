from flask import Flask, request, jsonify
from flask_cors import CORS
from backend import PortfolioOptimizer
import traceback
import json


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize optimizer
optimizer = PortfolioOptimizer("../my_data/stocks.json")

@app.route('/api/stocks', methods=['GET'])
def get_available_stocks():
    """Get list of available stocks from actual data file"""
    try:
        with open("../my_data/stocks.json", 'r') as f:
            data = json.load(f)
        
        # Extract unique tickers from the data
        stocks = []
        for ticker in data.keys():
            stocks.append({
                "ticker": ticker,
                "name": ticker,
                "category": "Unknown"
            })
        
        return jsonify(stocks)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_portfolio():
    """
    Main optimization endpoint
    
    Request Body:
    {
        "algorithm": "astar" | "greedy" | "csp" | "sa",
        "user_config": {
            "budget": 10000,
            "diversification": 50,
            "risk_tolerance": "moderate"
        },
        "stocks": ["AAPL", "AMZN", "JNJ"],
        "start_date": "2018-01-02",
        "end_date": "2018-01-15"
    }
    
    Response:
    {
        "success": true,
        "metrics": {...},
        "portfolio": {...},
        "history": [...],
        "actions": [...]
    }
    """
    try:
        data = request.json
        
        # Validate input
        required = ['algorithm', 'user_config', 'stocks', 'start_date', 'end_date']
        if not all(k in data for k in required):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Load stocks
        stock_configs = [
            {
                "ticker": ticker,
                "name": ticker,
                "start": data['start_date'],
                "end": data['end_date'],
                "category": "Unknown"
            }
            for ticker in data['stocks']
        ]
        
        optimizer.load_stocks(stock_configs)
        
        # Run optimization
        result = optimizer.optimize(
            algorithm=data['algorithm'],
            user_config=data['user_config'],
            start_date=data['start_date'],
            end_date=data['end_date']
        )
        
        if not result['success']:
            return jsonify({"error": result.get('error', 'Optimization failed')}), 500
        
        # Format response
        response = {
            "success": True,
            "algorithm": data['algorithm'],
            "metrics": result['metrics'],
            "portfolio": result['final_portfolio'].holdings,
            "history": _generate_history(result),
            "actions": _generate_actions(result)
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

def _generate_history(result):
    """Generate portfolio value history from optimization result"""
    history = []
    
    if 'path' in result:
        # For A* and Greedy
        budget = optimizer.config.get('budget', 10000) if hasattr(optimizer, 'config') else 10000
        history.append({"date": result['path'][0][1] if result['path'] else "2018-01-02", 
                       "value": budget, "roi": 0})
        
        # Track value over time
        for i, (action, date) in enumerate(result['path'][:10]):  # Sample 10 points
            current_value = budget * (1 + (result['metrics']['roi_percent'] / 100) * (i / len(result['path'])))
            history.append({
                "date": date,
                "value": round(current_value, 2),
                "roi": round((current_value - budget) / budget * 100, 2)
            })
    
    elif 'history' in result:
        # For CSP multi-day
        for date, solution, portfolio_state in result['history']:
            prices = optimizer.market.get_all_prices(date)
            value = portfolio_state.get_total_value(prices)
            roi = ((value - portfolio_state.budget) / portfolio_state.budget) * 100
            history.append({
                "date": date,
                "value": round(value, 2),
                "roi": round(roi, 2)
            })
    
    # Ensure we always return something
    if not history:
        history = [{"date": "2018-01-02", "value": 10000, "roi": 0}]
    
    return history

def _generate_actions(result):
    """Generate trading actions list from optimization result"""
    actions = []
    
    if 'path' in result:
        # For A* and Greedy
        for action_data, date in result['path']:
            action_type, ticker, param = action_data
            
            if action_type in ['buy', 'sell']:
                price = optimizer.market.get_price(ticker, date) if ticker else 0
                actions.append({
                    "date": date,
                    "action": action_type.upper(),
                    "ticker": ticker,
                    "shares": param,
                    "price": round(price, 2)
                })
    
    elif 'history' in result:
        # For CSP
        for date, solution, portfolio_state in result['history']:
            prices = optimizer.market.get_all_prices(date)
            for ticker, change in solution.items():
                if change != 0:
                    actions.append({
                        "date": date,
                        "action": "BUY" if change > 0 else "SELL",
                        "ticker": ticker,
                        "shares": abs(change),
                        "price": round(prices.get(ticker, 0), 2)
                    })
    
    return actions

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)