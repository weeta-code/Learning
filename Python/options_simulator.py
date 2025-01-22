import tkinter as tk
from tkinter import ttk
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# Black-Scholes Model Class pulled from documentation
class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to expiration
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility

    def d1(self):
        return (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * math.sqrt(self.T)

    def call_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.S * norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(d2)

    def put_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

class OptionsTradingSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Options Trading Simulator")
        self.root.geometry("1000x800")

        # Variables
        self.symbol = tk.StringVar(value="AAPL")
        self.start_date = tk.StringVar(value="2023-01-01")
        self.end_date = tk.StringVar(value="2023-12-31")
        self.strike_price = tk.DoubleVar(value=150.0)

        self.setup_ui()

    def setup_ui(self):
        # Frame for inputs
        input_frame = tk.Frame(self.root, padx=20, pady=10)
        input_frame.pack(fill=tk.X)

        tk.Label(input_frame, text="Stock Symbol:").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(input_frame, textvariable=self.symbol, width=20).grid(row=0, column=1, sticky="w", pady=5)

        tk.Label(input_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(input_frame, textvariable=self.start_date, width=20).grid(row=1, column=1, sticky="w", pady=5)

        tk.Label(input_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(input_frame, textvariable=self.end_date, width=20).grid(row=2, column=1, sticky="w", pady=5)

        tk.Label(input_frame, text="Strike Price:").grid(row=3, column=0, sticky="w", pady=5)
        tk.Entry(input_frame, textvariable=self.strike_price, width=20).grid(row=3, column=1, sticky="w", pady=5)

        tk.Button(input_frame, text="Run Simulation", command=self.run_simulation).grid(row=4, column=0, columnspan=2, pady=10)

        self.canvas_frame = tk.Frame(self.root, padx=20, pady=10)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame)
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def run_simulation(self):
        symbol = self.symbol.get()
        start_date = self.start_date.get()
        end_date = self.end_date.get()
        strike_price = self.strike_price.get()

        data = yf.download(symbol, start=start_date, end=end_date)

        # Technical Indicators
        data["9EMA"] = data["Close"].ewm(span=9).mean()
        data["200EMA"] = data["Close"].ewm(span=200).mean()

        # Sigs
        data["Signal"] = (data["9EMA"] > data["20EMA"]).astype(int)

        # Black-Scholes Model
        current_price = float(data["Close"].iloc[-1])
        bs_model = BlackScholesModel(
            S=current_price,
            K=strike_price,
            T=30 / 365,  # Assuming 30 days to expiration
            r=0.01,  # Risk-free rate
            sigma=0.2  # Volatility
        )
        call_price = bs_model.call_price()
        put_price = bs_model.put_price()

        # Compare multiple strike prices
        strike_prices = [current_price * (1 + i / 10) for i in range(-3, 4)]  # ±30%
        call_prices = [BlackScholesModel(current_price, K, 30 / 365, 0.01, 0.2).call_price() for K in strike_prices]
        put_prices = [BlackScholesModel(current_price, K, 30 / 365, 0.01, 0.2).put_price() for K in strike_prices]

        # Plot results
        self.plot_results(data, call_price, put_price, strike_prices, call_prices, put_prices)

    def plot_results(self, data, call_price, put_price, strike_prices, call_prices, put_prices):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(2, 1, figsize=(10, 12))

        # Plot EMA crossover
        ax[0].plot(data.index, data["Close"], label="Close Price", color="blue")
        ax[0].plot(data.index, data["9EMA"], label="9 EMA", linestyle="--", color="orange")
        ax[0].plot(data.index, data["20EMA"], label="20 EMA", linestyle="--", color="green")
        ax[0].set_title(
            f"{self.symbol.get()} | Call: ${call_price:.2f}, Put: ${put_price:.2f} | Strike Price: {self.strike_price.get()}"
        )
        ax[0].legend()

        # Plot strike price comparison
        ax[1].bar(strike_prices, call_prices, width=2, label="Call Prices", alpha=0.7, color="blue")
        ax[1].bar(strike_prices, put_prices, width=2, label="Put Prices", alpha=0.7, color="red")
        ax[1].set_title("Options Prices for Different Strike Prices")
        ax[1].set_xlabel("Strike Price")
        ax[1].set_ylabel("Option Price")
        ax[1].legend()

        # Embed the plot 
        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = OptionsTradingSimulator(root)
    root.mainloop()
