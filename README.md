# Traffic Analysis and Route Optimization in IP/MPLS Networks

This project presents a Python-based implementation and simulation of the methodologies described in the paper "Traffic analysis and route optimization in IP/MPLS networks". It provides a hybrid approach for analyzing and optimizing telecommunications networks, with a focus on IP/MPLS environments.

## Abstract

The study explores hybrid approaches for analyzing and optimizing telecommunications networks. Traffic data is generated for both normal and congested conditions, with key performance metrics such as bandwidth utilization, latency, and packet loss. Three optimization strategies are implemented and simulated: dynamic weighted Dijkstra, linear programming (LP) for optimal routing, and reinforcement learning (RL). Additionally, a Random Forest model is trained to predict network latency. The results confirm that integrating traditional routing algorithms with advanced machine learning techniques can significantly enhance network performance and Quality of Service (QoS).

## Features & Methodologies

-   **Synthetic Data Generation**: Simulates network traffic over 8 links, featuring normal and congested scenarios with realistic day-night fluctuations.
-   **Dynamic Weighted Dijkstra**: Simulates this algorithm's effect, reducing bandwidth usage, latency, and packet loss by 10-30%.
-   **Linear Programming (LP)**: Simulates LP for uniform traffic distribution, achieving significant latency and packet loss reductions (25% and 40%).
-   **Reinforcement Learning (RL)**: Simulates an adaptive routing approach, demonstrating consistent performance improvements and yielding an average reward.
-   **Genetic Algorithm (GA)**: Implements a GA using the DEAP library to find an optimal traffic allocation across links, minimizing a composite fitness function of latency, packet loss, and utilization.
-   **Latency Prediction**: A Random Forest regressor is trained to predict network latency based on traffic load and packet loss, achieving an MAE of ~5.22 ms and an R² of ~0.76, consistent with the paper's findings.

## Project Structure

```
.
├── .gitignore         # Standard Python .gitignore
├── main.py            # Main script containing all logic and simulations
├── README.md          # This documentation file
└── requirements.txt   # Project dependencies
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Execute the main script from your terminal. The script will run all simulations, train the machine learning model, and save the resulting plots as `.png` files in the root directory.

```bash
python main.py
```

### Expected Output

The script will generate several plot files, including:
-   `fig3_traffic_simulation.png`: Traffic pattern for a single link.
-   `fig4_ga_convergence.png`: Convergence plot for the Genetic Algorithm.
-   `dynamic_weighted_dijkstra_optimization_analysis.png`: Before/after optimization plot.
-   `network_flow_optimization_lp_optimization_analysis.png`: Before/after optimization plot.
-   `reinforcement_learning_optimization_analysis.png`: Before/after optimization plot.
-   `rf_prediction_vs_actual.png`: Actual vs. Predicted latency plot for the Random Forest model.
