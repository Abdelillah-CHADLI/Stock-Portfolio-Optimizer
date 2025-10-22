import React, { useState } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  TrendingUp,
  DollarSign,
  AlertTriangle,
  Target,
  Play,
  Settings,
  Download,
} from "lucide-react";

const PortfolioDashboard = () => {
  const [activeTab, setActiveTab] = useState("setup");
  const [config, setConfig] = useState({
    budget: 10000,
    diversification: 50,
    riskTolerance: "moderate",
    startDate: "2018-01-02",
    endDate: "2018-01-15",
    algorithm: "astar",
  });

  const [selectedStocks, setSelectedStocks] = useState([
    {
      ticker: "AAPL",
      name: "Apple Inc.",
      category: "Technology",
      selected: true,
    },
    {
      ticker: "AMZN",
      name: "Amazon.com",
      category: "Technology",
      selected: true,
    },
    {
      ticker: "JNJ",
      name: "Johnson & Johnson",
      category: "Healthcare",
      selected: true,
    },
    {
      ticker: "TSLA",
      name: "Tesla Inc.",
      category: "Automotive",
      selected: false,
    },
    {
      ticker: "MSFT",
      name: "Microsoft",
      category: "Technology",
      selected: false,
    },
  ]);

  const [results, setResults] = useState(null);
  const [isOptimizing, setIsOptimizing] = useState(false);

  const runOptimization = async () => {
    setIsOptimizing(true);

    try {
      const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";
      const response = await fetch(`${API_URL}/api/optimize`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          algorithm: config.algorithm,
          user_config: {
            budget: config.budget,
            diversification: config.diversification,
            risk_tolerance: config.riskTolerance,
          },
          stocks: selectedStocks.filter((s) => s.selected).map((s) => s.ticker),
          start_date: config.startDate,
          end_date: config.endDate,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        setResults(data);
        setActiveTab("results");
      } else {
        alert("Optimization failed: " + data.error);
      }
    } catch (error) {
      console.error("Optimization failed:", error);
      alert("Failed to connect to backend. Make sure the server is running.");
    } finally {
      setIsOptimizing(false);
    }
  };

  const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
            Portfolio Optimization System
          </h1>
          <p className="text-slate-400">
            AI-Powered Investment Strategy Using A*, CSP, and Heuristic Search
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex gap-2 mb-6 bg-slate-800/50 p-2 rounded-lg backdrop-blur">
          {["setup", "stocks", "results"].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 rounded-lg font-medium transition-all ${
                activeTab === tab
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-slate-400 hover:bg-slate-700"
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Setup Tab */}
        {activeTab === "setup" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Settings className="w-6 h-6" />
                Configuration
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Investment Budget ($)
                  </label>
                  <input
                    type="number"
                    value={config.budget}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        budget: parseFloat(e.target.value),
                      })
                    }
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Diversification Limit (% per stock)
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={config.diversification}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        diversification: parseInt(e.target.value),
                      })
                    }
                    className="w-full"
                  />
                  <div className="text-right text-blue-400 font-bold">
                    {config.diversification}%
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Risk Tolerance
                  </label>
                  <select
                    value={config.riskTolerance}
                    onChange={(e) =>
                      setConfig({ ...config, riskTolerance: e.target.value })
                    }
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                  >
                    <option value="conservative">
                      Conservative - Lower risk, stable returns
                    </option>
                    <option value="moderate">
                      Moderate - Balanced risk/return
                    </option>
                    <option value="aggressive">
                      Aggressive - Higher risk, potential high returns
                    </option>
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Start Date
                    </label>
                    <input
                      type="date"
                      value={config.startDate}
                      onChange={(e) =>
                        setConfig({ ...config, startDate: e.target.value })
                      }
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      End Date
                    </label>
                    <input
                      type="date"
                      value={config.endDate}
                      onChange={(e) =>
                        setConfig({ ...config, endDate: e.target.value })
                      }
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <h2 className="text-2xl font-bold mb-4">Algorithm Selection</h2>

              <div className="space-y-3">
                {[
                  {
                    id: "astar",
                    name: "A* Search",
                    desc: "Optimal solution with admissible heuristic",
                    time: "Slow",
                    quality: "Optimal",
                  },
                  {
                    id: "greedy",
                    name: "Greedy Search",
                    desc: "Fast heuristic approach",
                    time: "Fast",
                    quality: "Good",
                  },
                  {
                    id: "csp",
                    name: "CSP Solver",
                    desc: "Constraint satisfaction with daily optimization",
                    time: "Medium",
                    quality: "Optimal",
                  },
                  {
                    id: "sa",
                    name: "Simulated Annealing",
                    desc: "Probabilistic optimization",
                    time: "Medium",
                    quality: "Good",
                  },
                ].map((algo) => (
                  <div
                    key={algo.id}
                    onClick={() => setConfig({ ...config, algorithm: algo.id })}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      config.algorithm === algo.id
                        ? "border-blue-500 bg-blue-500/20"
                        : "border-slate-600 hover:border-slate-500"
                    }`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="font-bold text-lg">{algo.name}</div>
                      <div className="flex gap-2">
                        <span
                          className={`px-2 py-1 rounded text-xs ${
                            algo.time === "Fast"
                              ? "bg-green-500/20 text-green-400"
                              : algo.time === "Medium"
                              ? "bg-yellow-500/20 text-yellow-400"
                              : "bg-orange-500/20 text-orange-400"
                          }`}
                        >
                          {algo.time}
                        </span>
                        <span
                          className={`px-2 py-1 rounded text-xs ${
                            algo.quality === "Optimal"
                              ? "bg-purple-500/20 text-purple-400"
                              : "bg-blue-500/20 text-blue-400"
                          }`}
                        >
                          {algo.quality}
                        </span>
                      </div>
                    </div>
                    <div className="text-sm text-slate-400">{algo.desc}</div>
                  </div>
                ))}
              </div>

              <button
                onClick={runOptimization}
                disabled={isOptimizing}
                className="w-full mt-6 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-700 hover:to-emerald-700 disabled:from-slate-600 disabled:to-slate-600 text-white font-bold py-4 rounded-lg flex items-center justify-center gap-2 transition-all shadow-lg"
              >
                {isOptimizing ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Optimizing Portfolio...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Run Optimization
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Stocks Tab */}
        {activeTab === "stocks" && (
          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
            <h2 className="text-2xl font-bold mb-4">
              Select Stocks for Portfolio
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {selectedStocks.map((stock) => (
                <div
                  key={stock.ticker}
                  onClick={() =>
                    setSelectedStocks((prev) =>
                      prev.map((s) =>
                        s.ticker === stock.ticker
                          ? { ...s, selected: !s.selected }
                          : s
                      )
                    )
                  }
                  className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                    stock.selected
                      ? "border-emerald-500 bg-emerald-500/20"
                      : "border-slate-600 hover:border-slate-500"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-bold text-xl">{stock.ticker}</div>
                    <div
                      className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                        stock.selected
                          ? "bg-emerald-500 border-emerald-500"
                          : "border-slate-500"
                      }`}
                    >
                      {stock.selected && (
                        <span className="text-white text-xs">✓</span>
                      )}
                    </div>
                  </div>
                  <div className="text-sm text-slate-300">{stock.name}</div>
                  <div className="text-xs text-slate-400 mt-1 px-2 py-1 bg-slate-700 rounded inline-block">
                    {stock.category}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 p-4 bg-blue-500/20 border border-blue-500/50 rounded-lg flex justify-between items-center">
              <div className="text-sm">
                <span className="font-bold text-lg">
                  {selectedStocks.filter((s) => s.selected).length}
                </span>{" "}
                stocks selected
              </div>
              <button
                onClick={() => setActiveTab("setup")}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-all"
              >
                Continue to Setup
              </button>
            </div>
          </div>
        )}

        {/* Results Tab */}
        {activeTab === "results" && results && (
          <div className="space-y-6">
            {/* Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl p-6 shadow-lg">
                <div className="flex items-center justify-between mb-2">
                  <DollarSign className="w-8 h-8" />
                  <span className="text-xs bg-white/20 px-2 py-1 rounded">
                    Final
                  </span>
                </div>
                <div className="text-3xl font-bold">
                  ${results.metrics.final_value.toFixed(2)}
                </div>
                <div className="text-sm text-blue-200">Portfolio Value</div>
              </div>

              <div className="bg-gradient-to-br from-emerald-600 to-emerald-700 rounded-xl p-6 shadow-lg">
                <div className="flex items-center justify-between mb-2">
                  <TrendingUp className="w-8 h-8" />
                  <span className="text-xs bg-white/20 px-2 py-1 rounded">
                    ROI
                  </span>
                </div>
                <div className="text-3xl font-bold">
                  +{results.metrics.roi_percent.toFixed(1)}%
                </div>
                <div className="text-sm text-emerald-200">
                  Return on Investment
                </div>
              </div>

              <div className="bg-gradient-to-br from-orange-600 to-orange-700 rounded-xl p-6 shadow-lg">
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-8 h-8" />
                  <span className="text-xs bg-white/20 px-2 py-1 rounded">
                    Risk
                  </span>
                </div>
                <div className="text-3xl font-bold">
                  {(results.metrics.risk * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-orange-200">Portfolio Risk</div>
              </div>

              <div className="bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl p-6 shadow-lg">
                <div className="flex items-center justify-between mb-2">
                  <Target className="w-8 h-8" />
                  <span className="text-xs bg-white/20 px-2 py-1 rounded">
                    Stocks
                  </span>
                </div>
                <div className="text-3xl font-bold">
                  {results.metrics.num_stocks}
                </div>
                <div className="text-sm text-purple-200">
                  Diversified Assets
                </div>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
                <h3 className="text-xl font-bold mb-4">
                  Portfolio Growth Over Time
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={results.history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="date" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1e293b",
                        border: "1px solid #475569",
                        borderRadius: "8px",
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="#3b82f6"
                      strokeWidth={3}
                      dot={{ fill: "#3b82f6", r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
                <h3 className="text-xl font-bold mb-4">Portfolio Allocation</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={Object.entries(results.portfolio).map(
                        ([ticker, shares]) => ({
                          name: ticker,
                          value: shares,
                        })
                      )}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) =>
                        `${name} ${(percent * 100).toFixed(0)}%`
                      }
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {Object.keys(results.portfolio).map((_, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={COLORS[index % COLORS.length]}
                        />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1e293b",
                        border: "1px solid #475569",
                        borderRadius: "8px",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Trading History */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-bold">Trading Actions History</h3>
                <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-all">
                  <Download className="w-4 h-4" />
                  Export CSV
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="text-left py-3 px-4">Date</th>
                      <th className="text-left py-3 px-4">Action</th>
                      <th className="text-left py-3 px-4">Ticker</th>
                      <th className="text-right py-3 px-4">Shares</th>
                      <th className="text-right py-3 px-4">Price</th>
                      <th className="text-right py-3 px-4">Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.actions.map((action, idx) => (
                      <tr
                        key={idx}
                        className="border-b border-slate-700/50 hover:bg-slate-700/30"
                      >
                        <td className="py-3 px-4">{action.date}</td>
                        <td className="py-3 px-4">
                          <span
                            className={`px-2 py-1 rounded text-xs font-bold ${
                              action.action === "BUY"
                                ? "bg-emerald-500/20 text-emerald-400"
                                : "bg-red-500/20 text-red-400"
                            }`}
                          >
                            {action.action}
                          </span>
                        </td>
                        <td className="py-3 px-4 font-bold">{action.ticker}</td>
                        <td className="py-3 px-4 text-right">
                          {action.shares}
                        </td>
                        <td className="py-3 px-4 text-right">
                          ${action.price.toFixed(2)}
                        </td>
                        <td className="py-3 px-4 text-right font-bold">
                          ${(action.shares * action.price).toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Algorithm Info */}
            <div className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 border border-blue-500/50 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-2">
                Algorithm: {results.algorithm.toUpperCase()}
              </h3>
              <p className="text-slate-300">
                {results.algorithm === "astar" &&
                  "Used A* search algorithm with admissible heuristic to find the optimal portfolio configuration."}
                {results.algorithm === "greedy" &&
                  "Used greedy heuristic search to quickly find a good portfolio configuration."}
                {results.algorithm === "csp" &&
                  "Used constraint satisfaction problem solver to optimize portfolio daily while respecting all constraints."}
                {results.algorithm === "sa" &&
                  "Used simulated annealing to probabilistically explore the solution space and find an optimal portfolio."}
              </p>
            </div>
          </div>
        )}

        {activeTab === "results" && !results && (
          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-12 border border-slate-700 text-center">
            <Target className="w-16 h-16 mx-auto mb-4 text-slate-500" />
            <h3 className="text-2xl font-bold mb-2">No Results Yet</h3>
            <p className="text-slate-400 mb-6">
              Run an optimization to see results here
            </p>
            <button
              onClick={() => setActiveTab("setup")}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-all"
            >
              Go to Setup
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default PortfolioDashboard;
