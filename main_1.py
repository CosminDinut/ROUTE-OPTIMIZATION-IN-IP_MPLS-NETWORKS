import pandas as pd
import numpy as np

# =============================================================================
# STEP 1: SYNTHETIC DATA GENERATION (Network Simulation)
# =============================================================================
def generate_synthetic_network_data(num_days=7, num_links=8):
    """
    Generates network traffic data for a number of links,
    simulating normal and congested scenarios.
    """
    print("STEP 1: Generating synthetic network data...")
    
    time_intervals = pd.date_range(start='2023-01-01', periods=num_days * 24 * 6, freq='10min')
    n_points = len(time_intervals)
    
    data = []

    for link_id in range(1, num_links + 1):
        time_component = np.linspace(0, num_days * 2 * np.pi, n_points)
        day_night_cycle = (np.sin(time_component) + 1.5) / 2.5

        # Normal Scenario
        base_traffic_normal = 300
        traffic_normal = (base_traffic_normal * day_night_cycle) + np.random.normal(0, 60, n_points)
        traffic_normal = np.clip(traffic_normal, 100, 500)
        utilization_normal = np.clip((traffic_normal / 1000) * 100 * 0.4, 35, 45)
        latency_normal = np.clip(5 + (traffic_normal / 100) + np.random.normal(0, 1.5, n_points), 5, 15)
        packet_loss_normal = np.clip(0.1 + (traffic_normal / 500)**2 + np.random.normal(0, 0.2, n_points), 0.1, 1.5)

        # Congested Scenario
        base_traffic_congested = 1100
        traffic_congested = (base_traffic_congested * day_night_cycle) + np.random.normal(0, 150, n_points)
        traffic_congested = np.clip(traffic_congested, 600, 1600)
        utilization_congested = np.clip((traffic_congested / 1600) * 100, 85, 95)
        latency_congested = np.clip(20 + (traffic_congested / 50) + np.random.normal(0, 5, n_points), 20, 50)
        packet_loss_congested = np.clip(5 + (traffic_congested / 200)**2 + np.random.normal(0, 1, n_points), 5, 15)

        for i in range(n_points):
            data.append([time_intervals[i], link_id, 'Normal', traffic_normal[i], latency_normal[i], utilization_normal[i], packet_loss_normal[i]])
            data.append([time_intervals[i], link_id, 'Congested', traffic_congested[i], latency_congested[i], utilization_congested[i], packet_loss_congested[i]])
            
    df = pd.DataFrame(data, columns=['Timestamp', 'LinkID', 'Scenario', 'Traffic (kbps)', 'Latency (ms)', 'Utilization (%)', 'Packet Loss (%)'])
    print("Data generation complete.\n")
    return df

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == '__main__':
    # 1. Generate data
    network_df = generate_synthetic_network_data()
    
    # Display statistics to verify against Tables 1 and 2
    print("\nVerifying data against the paper's tables:")
    print("--- Normal Scenario (Link 1) Statistics ---")
    print(network_df[(network_df['Scenario'] == 'Normal') & (network_df['LinkID'] == 1)].describe())
    print("\n--- Congested Scenario (Link 5) Statistics ---")
    print(network_df[(network_df['Scenario'] == 'Congested') & (network_df['LinkID'] == 5)].describe())

    print("\nScript finished successfully.")