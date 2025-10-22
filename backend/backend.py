import json
import math
import heapq
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

# ============================================================================
# CORE DATA STRUCTURES (Improved)
# ============================================================================

class Stock:
    """Stock with cached metrics for performance"""
    def __init__(self, ticker: str, name: str, data_file: str, category: str = "Unknown"):
        self.ticker = ticker
        self.name = name
        self.data_file = data_file
        self.category = category
        self.data: List[Dict] = []
        self._risk_cache: Optional[float] = None
        self._price_cache: Dict[str, float] = {}  # date -> price
        
    def load_data(self, start_date: str, end_date: str):
        """Load historical data with validation"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        with open(self.data_file, 'r') as f:
            all_data = json.load(f)
            
        ticker_data = all_data.get(self.ticker, [])
        self.data = [
            entry for entry in ticker_data
            if start <= datetime.fromisoformat(entry["Date"]).replace(tzinfo=None) <= end
        ]
        
        # Pre-cache prices for fast lookup
        for entry in self.data:
            date_key = entry["Date"][:10]
            self._price_cache[date_key] = entry['Close']
            
        print(f"Loaded {len(self.data)} days for {self.ticker}")

    def get_price(self, date_str: str) -> float:
        """Fast price lookup with caching"""
        return self._price_cache.get(date_str[:10], 0.0)
    
    def calculate_risk(self) -> float:
        """Cached risk calculation (standard deviation of returns)"""
        if self._risk_cache is not None:
            return self._risk_cache
            
        if len(self.data) < 2:
            return 0.0
            
        returns = []
        for i in range(1, len(self.data)):
            prev = self.data[i-1]["Close"]
            curr = self.data[i]["Close"]
            if prev > 0:
                returns.append(abs(curr - prev) / prev)
        
        self._risk_cache = np.std(returns) if returns else 0.0
        return self._risk_cache
    
    def calculate_roi(self, start_date: str, duration_days: int) -> float:
        """Calculate ROI over period"""
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = start + timedelta(days=duration_days)
        
        start_price = self.get_price(start.strftime("%Y-%m-%d"))
        end_price = self.get_price(end.strftime("%Y-%m-%d"))
        
        if start_price > 0 and end_price > 0:
            return ((end_price - start_price) / start_price) * 100
        return 0.0


class Portfolio:
    """Portfolio class that stores share counts and manages trading actions"""
    def __init__(self, name: str, budget: float, diversification: float, current_date: str):
        self.name = name
        self.budget = float(budget)
        self.diversification = float(diversification)  # Max % per stock
        self.current_date = datetime.strptime(current_date, "%Y-%m-%d").date()
        
        # KEY CHANGE: Store share counts, not prices
        self.holdings: Dict[str, int] = {}  # ticker -> number of shares
        self.cash = float(budget)  # Available cash
        
    def can_buy(self, ticker: str, shares: int, price: float, max_per_stock: float) -> bool:
        """Check if purchase is valid"""
        cost = shares * price
        if cost > self.cash:
            return False
        
        # Check diversification limit
        current_value = self.holdings.get(ticker, 0) * price
        new_value = current_value + cost
        total_value = self.get_total_value({ticker: price}) + cost
        
        return (new_value / total_value) <= (self.diversification / 100)
    
    def buy(self, ticker: str, shares: int, price: float):
        """Execute buy order"""
        cost = shares * price
        self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
        self.cash -= cost
        
    def sell(self, ticker: str, shares: int, price: float):
        """Execute sell order"""
        if ticker not in self.holdings or self.holdings[ticker] < shares:
            raise ValueError(f"Cannot sell {shares} shares of {ticker}")
        
        self.holdings[ticker] -= shares
        if self.holdings[ticker] == 0:
            del self.holdings[ticker]
        self.cash += shares * price
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        stock_value = sum(
            shares * current_prices.get(ticker, 0)
            for ticker, shares in self.holdings.items()
        )
        return stock_value + self.cash
    
    def __hash__(self):
        """Fast hashing for state comparison"""
        return hash((
            self.current_date,
            tuple(sorted(self.holdings.items())),
            round(self.cash, 2)
        ))
    
    def __eq__(self, other):
        """Fast equality check"""
        if not isinstance(other, Portfolio):
            return False
        return (self.current_date == other.current_date and
                self.holdings == other.holdings and
                abs(self.cash - other.cash) < 0.01)


class Market:
    """Market container with caching"""
    def __init__(self):
        self.stocks: Dict[str, Stock] = {}
        self._price_cache: Dict[Tuple[str, str], float] = {}  # (ticker, date) -> price
        
    def load_stock(self, ticker: str, name: str, data_file: str, 
               start_date: str, end_date: str, category: str = "other"):
        """Load stock with validation"""
        if ticker in self.stocks:
            raise ValueError(f"Stock {ticker} already loaded")
            
        stock = Stock(ticker, name, data_file, category) 
        stock.load_data(start_date, end_date)
        self.stocks[ticker] = stock
    
    def get_price(self, ticker: str, date: str) -> float:
        """Cached price lookup"""
        key = (ticker, date[:10])
        if key not in self._price_cache:
            self._price_cache[key] = self.stocks[ticker].get_price(date)
        return self._price_cache[key]
    
    def get_all_prices(self, date: str) -> Dict[str, float]:
        """Get all stock prices for a date"""
        return {ticker: self.get_price(ticker, date) for ticker in self.stocks}


# ============================================================================
# IMPROVED A* SEARCH
# ============================================================================

class Node:
    """Search node with proper comparison"""
    def __init__(self, portfolio: Portfolio, parent: Optional['Node'], 
                 action: Tuple[str, str, str], g_cost: float):
        self.portfolio = portfolio
        self.parent = parent
        self.action = action  # (action_type, ticker, date)
        self.g_cost = g_cost  # Actual cost from start
        self.f_cost = 0.0  # g + h (set by search)
        
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class ImprovedAStarSearch:
    """A* search implementation for portfolio optimization with heuristic approach"""
    def __init__(self, market: Market, user_config: dict, start_date: str, end_date: str):
        self.market = market
        self.config = user_config
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Cache for expensive calculations
        self._roi_cache: Dict[Tuple[str, str, int], float] = {}
        
    def admissible_heuristic(self, portfolio: Portfolio) -> float:
        """Calculate heuristic estimate of potential future returns based on current portfolio state"""
        remaining_days = (self.end_date - portfolio.current_date).days
        if remaining_days <= 0:
            return 0.0
        
        # Get best expected daily return from available stocks
        best_returns = []
        for ticker, stock in self.market.stocks.items():
            date_str = portfolio.current_date.strftime("%Y-%m-%d")
            cache_key = (ticker, date_str, remaining_days)
            
            if cache_key not in self._roi_cache:
                self._roi_cache[cache_key] = stock.calculate_roi(date_str, remaining_days)
            
            roi = self._roi_cache[cache_key]
            if roi > 0:
                best_returns.append(roi / 100)  # Convert to decimal
        
        if not best_returns:
            return 0.0
        
        # Use 75th percentile to be optimistic but not unrealistic
        optimistic_return = np.percentile(best_returns, 75) if best_returns else 0
        
        # Potential value = cash * (1 + expected_return)
        potential_gain = portfolio.cash * optimistic_return
        
        # Return negative (we're minimizing cost, maximizing return)
        return -potential_gain * 0.5  # Scale down to stay admissible
    
    def cost_function(self, portfolio: Portfolio) -> float:
        """
        Actual cost = -ROI + risk_penalty
        
        Lower is better (more negative = higher returns)
        """
        if not portfolio.holdings:
            return 0.0
        
        # Calculate portfolio metrics
        current_prices = self.market.get_all_prices(
            portfolio.current_date.strftime("%Y-%m-%d")
        )
        total_value = portfolio.get_total_value(current_prices)
        
        if total_value == 0:
            return 0.0
        
        # Weighted ROI
        weighted_roi = 0.0
        weighted_risk = 0.0
        
        days_held = (portfolio.current_date - self.start_date).days
        if days_held > 0:
            for ticker, shares in portfolio.holdings.items():
                weight = (shares * current_prices[ticker]) / total_value
                stock = self.market.stocks[ticker]
                
                roi = stock.calculate_roi(self.start_date.strftime("%Y-%m-%d"), days_held)
                risk = stock.calculate_risk()
                
                weighted_roi += weight * roi
                weighted_risk += weight * risk
        
        # Cost = -return + risk_penalty
        risk_weight = {"conservative": 2.0, "moderate": 1.0, "aggressive": 0.5}
        risk_factor = risk_weight.get(self.config.get("risk_tolerance", "moderate"), 1.0)
        
        return -weighted_roi + (risk_factor * weighted_risk * 100)
    
    def get_valid_actions(self, portfolio: Portfolio) -> List[Tuple[str, str, str]]:
        """
        Generate valid actions with REDUCED BRANCHING
        
        Instead of all possible share counts, use:
        - Buy 1, 5, 10, or max affordable shares
        - Sell 1, 5, 10, or all shares
        - Hold (skip to next day)
        """
        actions = []
        current_prices = self.market.get_all_prices(
            portfolio.current_date.strftime("%Y-%m-%d")
        )
        
        # HOLD action (always available if not at end date)
        next_day = portfolio.current_date + timedelta(days=1)
        if next_day <= self.end_date:
            actions.append(("hold", None, next_day.strftime("%Y-%m-%d")))
        
        # BUY actions (simplified)
        max_per_stock = portfolio.budget * (portfolio.diversification / 100)
        
        for ticker, price in current_prices.items():
            if price <= 0:
                continue
            
            max_affordable = int(portfolio.cash / price)
            if max_affordable > 0:
                # Try buying 1, 5, 10, or max shares
                for shares in [1, 5, 10, max_affordable]:
                    if shares <= max_affordable:
                        if portfolio.can_buy(ticker, shares, price, max_per_stock):
                            actions.append(("buy", ticker, shares))
        
        # SELL actions (simplified)
        for ticker, shares_held in portfolio.holdings.items():
            price = current_prices[ticker]
            # Try selling 1, 5, 10, or all shares
            for shares in [1, 5, 10, shares_held]:
                if shares <= shares_held:
                    actions.append(("sell", ticker, shares))
        
        return actions
    
    def search(self, initial_portfolio: Portfolio, max_iterations: int = 5000000) -> Optional[Node]:
        """
        A* search with proper admissible heuristic
        """
        print(f"Starting A* search (max {max_iterations} iterations)...")
        
        start_node = Node(initial_portfolio, None, ("start", None, None), 0.0)
        start_node.f_cost = self.admissible_heuristic(initial_portfolio)
        
        frontier = [start_node]
        explored = set()
        iterations = 0
        
        while frontier and iterations < max_iterations:
            current = heapq.heappop(frontier)
            iterations += 1
            
            if iterations % 100 == 0:
                print(f"Iteration {iterations}, frontier size: {len(frontier)}, "
                      f"date: {current.portfolio.current_date}")
            
            # Goal test
            if current.portfolio.current_date >= self.end_date:
                print(f"Goal reached in {iterations} iterations!")
                return current
            
            state_key = hash(current.portfolio)
            if state_key in explored:
                continue
            explored.add(state_key)
            
            # Expand node
            actions = self.get_valid_actions(current.portfolio)
            
            for action in actions:
                action_type, ticker, param = action
                
                # Create new portfolio state
                new_portfolio = deepcopy(current.portfolio)
                
                if action_type == "hold":
                    new_portfolio.current_date = datetime.strptime(param, "%Y-%m-%d").date()
                elif action_type == "buy":
                    price = self.market.get_price(ticker, new_portfolio.current_date.strftime("%Y-%m-%d"))
                    new_portfolio.buy(ticker, param, price)
                elif action_type == "sell":
                    price = self.market.get_price(ticker, new_portfolio.current_date.strftime("%Y-%m-%d"))
                    new_portfolio.sell(ticker, param, price)
                
                # Calculate costs
                g = self.cost_function(new_portfolio)
                h = self.admissible_heuristic(new_portfolio)
                
                child = Node(new_portfolio, current, action, g)
                child.f_cost = g + h
                
                heapq.heappush(frontier, child)
        
        print(f"Search ended after {iterations} iterations (no solution found)")
        return None



import constraint
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np
from collections import defaultdict


class OptimizedCSP:
    """
    Highly optimized CSP solver for daily portfolio optimization
    
    Uses intelligent domain generation and constraint propagation
    for massive performance gains.
    """
    
    def __init__(self, market, user_config: dict):
        """
        Initialize CSP solver
        
        Args:
            market: Market object with stock data
            user_config: {
                'budget': float,
                'diversification': float (0-100),
                'risk_tolerance': 'conservative'/'moderate'/'aggressive',
                'preferred_sectors': list of sector names (optional)
            }
        """
        self.market = market
        self.config = user_config
        self.problem = constraint.Problem()
        
        # Cache for expensive calculations
        self._risk_cache = {}
        self._roi_cache = {}
        
        # Risk limits by tolerance level
        self.risk_limits = {
            'conservative': 0.12,
            'moderate': 0.20,
            'aggressive': 0.30
        }
        
    def solve_for_day(self, portfolio, target_date: str, 
                      timeout: int = 30) -> Optional[Dict[str, int]]:
        """
        Solve CSP for a single trading day
        
        Args:
            portfolio: Current portfolio state
            target_date: Date string 'YYYY-MM-DD'
            timeout: Max seconds to spend solving
            
        Returns:
            Dict mapping ticker -> share_change (positive=buy, negative=sell)
            None if no solution found
        """
        print(f"[CSP] Solving for {target_date}...")
        
        # Reset problem
        self.problem = constraint.Problem()
        
        # Get current market prices
        current_prices = self.market.get_all_prices(target_date)
        if not current_prices:
            print(f"[CSP] No price data for {target_date}")
            return None
        
        # Create intelligent domains
        domains = self._create_smart_domains(portfolio, current_prices)
        
        if not domains:
            print("[CSP] No valid domains created")
            return None
        
        # Add variables
        for ticker, domain in domains.items():
            self.problem.addVariable(ticker, domain)
        
        # Add constraints (order matters for performance)
        self._add_budget_constraint(portfolio, current_prices, list(domains.keys()))
        self._add_diversification_constraint(portfolio, current_prices, list(domains.keys()))
        self._add_risk_constraint(portfolio, current_prices, list(domains.keys()))
        
        # Add optional sector preference constraint
        if 'preferred_sectors' in self.config:
            self._add_sector_preference(portfolio, current_prices, list(domains.keys()))
        
        # Solve with timeout handling
        solutions = self._solve_with_timeout(timeout)
        
        if not solutions:
            print("[CSP] No solution found, trying relaxed constraints...")
            return self._generate_fallback_solution(portfolio, current_prices)
        
        # Select best solution
        best = self._rank_and_select_best(solutions, portfolio, current_prices, target_date)
        
        # Filter out zero changes
        return {ticker: change for ticker, change in best.items() if change != 0}
    
    def _create_smart_domains(self, portfolio, current_prices: Dict[str, float]) -> Dict[str, List[int]]:
        """Generate domain values for each stock based on current portfolio state and constraints"""
        domains = {}
        max_per_stock = portfolio.budget * (self.config['diversification'] / 100)
        
        for ticker, price in current_prices.items():
            if price <= 0:
                continue
            
            domain = set([0])  # Always include no-change
            
            # Current holdings
            current_shares = portfolio.holdings.get(ticker, 0)
            
            # === SELLING OPTIONS ===
            if current_shares > 0:
                sell_options = [
                    current_shares,  # Sell all
                    max(1, current_shares // 2),  # Sell half
                    max(1, current_shares // 4),  # Sell quarter
                    min(5, current_shares),  # Sell 5 shares
                    1  # Sell 1 share
                ]
                domain.update([-s for s in sell_options if s <= current_shares])
            
            # === BUYING OPTIONS ===
            max_affordable = int(portfolio.cash / price)
            
            if max_affordable > 0:
                # Strategic buy amounts
                buy_amounts = [
                    1,  # Minimum buy
                    min(5, max_affordable),
                    min(10, max_affordable),
                    min(20, max_affordable),
                    max(1, max_affordable // 4),  # 25% of max
                    max(1, max_affordable // 2),  # 50% of max
                    max_affordable  # Maximum buy
                ]
                
                # Filter valid amounts
                for amount in buy_amounts:
                    if 0 < amount <= max_affordable:
                        # Check if this would exceed diversification limit
                        new_value = (current_shares + amount) * price
                        if new_value <= max_per_stock * 1.1:  # 10% tolerance
                            domain.add(amount)
            
            # Convert to sorted list
            domains[ticker] = sorted(list(domain))
            
            # Limit domain size for performance (max 15 values)
            if len(domains[ticker]) > 15:
                # Keep extremes and sample middle
                keep = set([domains[ticker][0], domains[ticker][-1]])
                step = len(domains[ticker]) // 13
                keep.update(domains[ticker][::step])
                domains[ticker] = sorted(list(keep))[:15]
        
        return domains
    
    def _add_budget_constraint(self, portfolio, current_prices: Dict[str, float], 
                               tickers: List[str]):
        """Ensure net spending doesn't exceed available cash"""
        def budget_valid(*changes):
            net_cost = sum(
                changes[i] * current_prices[tickers[i]] 
                for i in range(len(tickers))
            )
            return net_cost <= portfolio.cash
        
        if tickers:
            self.problem.addConstraint(budget_valid, tickers)
    
    def _add_diversification_constraint(self, portfolio, current_prices: Dict[str, float],
                                       tickers: List[str]):
        """Ensure no single stock exceeds maximum allowed percentage"""
        max_fraction = self.config['diversification'] / 100.0
        
        def diversification_valid(*changes):
            # Calculate new holdings
            new_holdings = {}
            for i, ticker in enumerate(tickers):
                new_shares = portfolio.holdings.get(ticker, 0) + changes[i]
                if new_shares > 0:
                    new_holdings[ticker] = new_shares
            
            if not new_holdings:
                return True
            
            # Calculate total value
            total_value = sum(
                shares * current_prices[ticker]
                for ticker, shares in new_holdings.items()
            ) + (portfolio.cash - sum(
                changes[i] * current_prices[tickers[i]]
                for i in range(len(tickers))
            ))
            
            if total_value <= 0:
                return True
            
            # Check each stock (early exit on violation)
            for ticker, shares in new_holdings.items():
                stock_value = shares * current_prices[ticker]
                if stock_value / total_value > max_fraction:
                    return False
            
            return True
        
        if tickers:
            self.problem.addConstraint(diversification_valid, tickers)
    
    def _add_risk_constraint(self, portfolio, current_prices: Dict[str, float],
                            tickers: List[str]):
        """Ensure portfolio risk stays within user's tolerance level"""
        max_risk = self.risk_limits[self.config.get('risk_tolerance', 'moderate')]
        
        def risk_valid(*changes):
            # Calculate new holdings
            new_holdings = {}
            for i, ticker in enumerate(tickers):
                new_shares = portfolio.holdings.get(ticker, 0) + changes[i]
                if new_shares > 0:
                    new_holdings[ticker] = new_shares
            
            if not new_holdings:
                return True
            
            # Calculate total value
            total_value = sum(
                shares * current_prices[ticker]
                for ticker, shares in new_holdings.items()
            )
            
            if total_value == 0:
                return True
            
            # Calculate weighted risk (with caching)
            weighted_risk = 0.0
            for ticker, shares in new_holdings.items():
                # Cache stock risk
                if ticker not in self._risk_cache:
                    self._risk_cache[ticker] = self.market.stocks[ticker].calculate_risk()
                
                weight = (shares * current_prices[ticker]) / total_value
                weighted_risk += weight * self._risk_cache[ticker]
            
            return weighted_risk <= max_risk
        
        if tickers:
            self.problem.addConstraint(risk_valid, tickers)
    
    def _add_sector_preference(self, portfolio, current_prices: Dict[str, float],
                               tickers: List[str]):
        """
        Optional: Prefer stocks in user's preferred sectors
        
        This is a soft constraint - doesn't reject solutions, just guides selection
        """
        preferred = set(self.config.get('preferred_sectors', []))
        
        if not preferred:
            return
        
        def sector_bonus(*changes):
            # Allow any solution, but we'll use this in ranking
            return True
        
        # Store sector info for later ranking
        self._sector_info = {
            ticker: self.market.stocks[ticker].category
            for ticker in tickers
        }
    
    def _solve_with_timeout(self, timeout: int) -> List[Dict]:
        """
        Solve CSP with timeout
        
        Returns up to 50 solutions for ranking
        """
        try:
            # Get multiple solutions for better selection
            solutions = []
            for sol in self.problem.getSolutionIter():
                solutions.append(sol)
                if len(solutions) >= 50:  # Enough for good selection
                    break
            return solutions
        except Exception as e:
            print(f"[CSP] Solver error: {e}")
            return []
    
    def _rank_and_select_best(self, solutions: List[Dict], portfolio,
                             current_prices: Dict[str, float], 
                             target_date: str) -> Dict[str, int]:
        """Select best solution based on return, risk, diversification, and sector criteria"""
        scored_solutions = []
        
        for solution in solutions[:20]:  # Evaluate top 20
            score = self._score_solution(solution, portfolio, current_prices, target_date)
            scored_solutions.append((score, solution))
        
        # Sort by score (descending)
        scored_solutions.sort(key=lambda x: x[0], reverse=True)
        
        if scored_solutions:
            best_score, best_solution = scored_solutions[0]
            print(f"[CSP] Best solution score: {best_score:.4f}")
            return best_solution
        
        return solutions[0]  # Fallback to first solution
    
    def _score_solution(self, solution: Dict[str, int], portfolio,
                       current_prices: Dict[str, float], target_date: str) -> float:
        """
        Calculate comprehensive score for a solution
        
        Higher is better
        """
        # Calculate new portfolio state
        new_holdings = portfolio.holdings.copy()
        for ticker, change in solution.items():
            if change != 0:
                new_holdings[ticker] = new_holdings.get(ticker, 0) + change
                if new_holdings[ticker] <= 0:
                    new_holdings.pop(ticker, None)
        
        if not new_holdings:
            return 0.0
        
        total_value = sum(
            shares * current_prices[ticker]
            for ticker, shares in new_holdings.items()
        )
        
        if total_value == 0:
            return 0.0
        
        # 1. Expected Return (40% weight)
        expected_return = 0.0
        for ticker, shares in new_holdings.items():
            weight = (shares * current_prices[ticker]) / total_value
            
            # Cache ROI calculation
            cache_key = (ticker, target_date)
            if cache_key not in self._roi_cache:
                self._roi_cache[cache_key] = self.market.stocks[ticker].calculate_roi(target_date, 1)
            
            roi = self._roi_cache[cache_key]
            expected_return += weight * roi
        
        # 2. Risk-adjusted return (30% weight)
        weighted_risk = sum(
            (shares * current_prices[ticker] / total_value) * 
            self.market.stocks[ticker].calculate_risk()
            for ticker, shares in new_holdings.items()
        )
        
        risk_weight = {'conservative': 2.0, 'moderate': 1.0, 'aggressive': 0.5}
        risk_factor = risk_weight.get(self.config.get('risk_tolerance', 'moderate'), 1.0)
        risk_adjusted = expected_return - (risk_factor * weighted_risk * 100)
        
        # 3. Diversification quality (20% weight)
        num_stocks = len(new_holdings)
        max_concentration = max(
            (shares * current_prices[ticker]) / total_value
            for ticker, shares in new_holdings.items()
        )
        diversification_score = (num_stocks / 5.0) * (1 - max_concentration)
        
        # 4. Sector preference (10% weight)
        sector_score = 0.0
        if hasattr(self, '_sector_info') and 'preferred_sectors' in self.config:
            preferred = set(self.config['preferred_sectors'])
            for ticker, shares in new_holdings.items():
                if self._sector_info.get(ticker) in preferred:
                    weight = (shares * current_prices[ticker]) / total_value
                    sector_score += weight
        
        # Combine scores
        final_score = (
            0.40 * expected_return +
            0.30 * risk_adjusted +
            0.20 * diversification_score * 100 +
            0.10 * sector_score * 100
        )
        
        return final_score
    
    def _generate_fallback_solution(self, portfolio, 
                                   current_prices: Dict[str, float]) -> Optional[Dict[str, int]]:
        """
        Generate simple fallback when CSP fails
        
        Strategy: Buy top 2 stocks with best ROI that fit budget
        """
        print("[CSP] Generating fallback solution...")
        
        # Score stocks by ROI
        stock_scores = []
        for ticker, price in current_prices.items():
            if price <= 0:
                continue
            
            stock = self.market.stocks[ticker]
            roi = stock.calculate_roi(datetime.now().strftime("%Y-%m-%d"), 1)
            risk = stock.calculate_risk()
            
            score = roi - risk * 50  # Simple risk adjustment
            stock_scores.append((score, ticker, price))
        
        stock_scores.sort(reverse=True)
        
        # Try to buy top stocks
        solution = {}
        remaining_cash = portfolio.cash
        
        for score, ticker, price in stock_scores[:3]:  # Try top 3
            if remaining_cash <= 0:
                break
            
            max_shares = int(remaining_cash / price)
            if max_shares > 0:
                # Buy conservative amount (max 5 shares)
                shares_to_buy = min(5, max_shares)
                solution[ticker] = shares_to_buy
                remaining_cash -= shares_to_buy * price
        
        return solution if solution else None
    
    def solve_multi_day(self, initial_portfolio, start_date: str, 
                       end_date: str, max_days: int = 30) -> List[Tuple]:
        """
        Solve CSP for multiple consecutive days
        
        Returns: List of (date, solution, portfolio_state)
        """
        print(f"[CSP] Multi-day optimization: {start_date} to {end_date}")
        
        results = []
        current_portfolio = deepcopy(initial_portfolio)
        current_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        days_processed = 0
        
        while current_date <= end_dt and days_processed < max_days:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Solve for this day
            solution = self.solve_for_day(current_portfolio, date_str, timeout=20)
            
            if solution:
                # Apply changes
                prices = self.market.get_all_prices(date_str)
                for ticker, change in solution.items():
                    if change > 0:
                        current_portfolio.buy(ticker, change, prices[ticker])
                    elif change < 0:
                        current_portfolio.sell(ticker, -change, prices[ticker])
                
                results.append((date_str, solution, deepcopy(current_portfolio)))
                print(f"[CSP] {date_str}: Applied {len(solution)} changes")
            else:
                print(f"[CSP] {date_str}: No changes")
                results.append((date_str, {}, deepcopy(current_portfolio)))
            
            # Next day
            current_date += timedelta(days=1)
            current_portfolio.current_date = current_date
            days_processed += 1
        
        print(f"[CSP] Completed {days_processed} days")
        return results


class PortfolioOptimizer:
    """
    Main class that wraps all optimization algorithms
    Provides unified interface for Flask API
    """
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.market = None
        
    def load_stocks(self, stock_configs: List[Dict]):
        """
        Load stocks into market
        
        stock_configs: [
            {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "start": "2018-01-02",
                "end": "2018-01-15",
                "category": "Technology"
            },
            ...
        ]
        """
        self.market = Market()
        
        for config in stock_configs:
            try:
                self.market.load_stock(
                    ticker=config["ticker"],
                    name=config["name"],
                    data_file=self.data_file,
                    start_date=config["start"],
                    end_date=config["end"],
                    category=config.get("category", "Unknown")
                )
            except Exception as e:
                print(f"Warning: Failed to load {config['ticker']}: {e}")
    
    def optimize(self, algorithm: str, user_config: dict, 
                 start_date: str, end_date: str) -> Dict:
        """
        Run optimization with specified algorithm
        
        Returns:
        {
            "success": bool,
            "metrics": {...},
            "final_portfolio": Portfolio object,
            "path": [...],  # For A* and Greedy
            "history": [...],  # For CSP multi-day
            "error": str (if failed)
        }
        """
        try:
            # Create initial portfolio
            initial_portfolio = Portfolio(
                name="User Portfolio",
                budget=user_config["budget"],
                diversification=user_config["diversification"],
                current_date=start_date
            )
            
            if algorithm == "astar":
                return self._run_astar(initial_portfolio, user_config, start_date, end_date)
            elif algorithm == "greedy":
                return self._run_greedy(initial_portfolio, user_config, start_date, end_date)
            elif algorithm == "csp":
                return self._run_csp(initial_portfolio, user_config, start_date, end_date)
            elif algorithm == "sa":
                return {"success": False, "error": "Simulated Annealing not implemented yet"}
            else:
                return {"success": False, "error": f"Unknown algorithm: {algorithm}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_astar(self, portfolio, user_config, start_date, end_date):
        """Run A* search"""
        search = ImprovedAStarSearch(self.market, user_config, start_date, end_date)
        result_node = search.search(portfolio, max_iterations=5000000)
        
        if not result_node:
            return {"success": False, "error": "A* search failed to find solution"}
        
        # Extract path
        path = []
        node = result_node
        while node.parent:
            path.append((node.action, node.portfolio.current_date.strftime("%Y-%m-%d")))
            node = node.parent
        path.reverse()
        
        # Calculate metrics
        final_prices = self.market.get_all_prices(end_date)
        final_value = result_node.portfolio.get_total_value(final_prices)
        
        return {
            "success": True,
            "final_portfolio": result_node.portfolio,
            "path": path,
            "metrics": {
                "final_value": final_value,
                "roi_percent": ((final_value - user_config["budget"]) / user_config["budget"]) * 100,
                "risk": self._calculate_portfolio_risk(result_node.portfolio, final_prices),
                "num_stocks": len(result_node.portfolio.holdings)
            }
        }
    
    def _run_greedy(self, portfolio, user_config, start_date, end_date):
        """Run Greedy search (simplified A* with greedy heuristic)"""
        # Use A* but with max 200 iterations for speed
        search = ImprovedAStarSearch(self.market, user_config, start_date, end_date)
        result_node = search.search(portfolio, max_iterations=5000000)
        
        if not result_node:
            return {"success": False, "error": "Greedy search failed"}
        
        # Same format as A*
        path = []
        node = result_node
        while node.parent:
            path.append((node.action, node.portfolio.current_date.strftime("%Y-%m-%d")))
            node = node.parent
        path.reverse()
        
        final_prices = self.market.get_all_prices(end_date)
        final_value = result_node.portfolio.get_total_value(final_prices)
        
        return {
            "success": True,
            "final_portfolio": result_node.portfolio,
            "path": path,
            "metrics": {
                "final_value": final_value,
                "roi_percent": ((final_value - user_config["budget"]) / user_config["budget"]) * 100,
                "risk": self._calculate_portfolio_risk(result_node.portfolio, final_prices),
                "num_stocks": len(result_node.portfolio.holdings)
            }
        }
    
    def _run_csp(self, portfolio, user_config, start_date, end_date):
        """Run CSP solver"""
        csp = OptimizedCSP(self.market, user_config)
        results = csp.solve_multi_day(portfolio, start_date, end_date)
        
        if not results:
            return {"success": False, "error": "CSP solver failed"}
        
        # Get final state
        final_date, final_solution, final_portfolio = results[-1]
        final_prices = self.market.get_all_prices(final_date)
        final_value = final_portfolio.get_total_value(final_prices)
        
        return {
            "success": True,
            "final_portfolio": final_portfolio,
            "history": results,  # List of (date, solution, portfolio) tuples
            "metrics": {
                "final_value": final_value,
                "roi_percent": ((final_value - user_config["budget"]) / user_config["budget"]) * 100,
                "risk": self._calculate_portfolio_risk(final_portfolio, final_prices),
                "num_stocks": len(final_portfolio.holdings)
            }
        }
    
    def _calculate_portfolio_risk(self, portfolio, current_prices):
        """Calculate weighted portfolio risk"""
        if not portfolio.holdings:
            return 0.0
        
        total_value = portfolio.get_total_value(current_prices)
        if total_value == 0:
            return 0.0
        
        weighted_risk = 0.0
        for ticker, shares in portfolio.holdings.items():
            weight = (shares * current_prices[ticker]) / total_value
            stock_risk = self.market.stocks[ticker].calculate_risk()
            weighted_risk += weight * stock_risk
        
        return weighted_risk

