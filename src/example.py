import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_sample_data():
    """Create some sample MLB-like data for demonstration."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'player_name': [f'Player_{i}' for i in range(n_samples)],
        'batting_average': np.random.normal(0.250, 0.050, n_samples),
        'home_runs': np.random.poisson(15, n_samples),
        'rbi': np.random.poisson(45, n_samples)
    }
    
    return pd.DataFrame(data)

def plot_sample_analysis(df):
    """Create some sample visualizations."""
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Distribution of batting averages
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='batting_average', bins=20)
    plt.title('Distribution of Batting Averages')
    
    # Plot 2: Home Runs vs RBI
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='home_runs', y='rbi')
    plt.title('Home Runs vs RBI')
    
    plt.tight_layout()
    plt.savefig('data/sample_analysis.png')
    plt.close()

def main():
    # Create sample data
    df = create_sample_data()
    
    # Save sample data
    df.to_csv('data/sample_data.csv', index=False)
    
    # Create visualizations
    plot_sample_analysis(df)
    
    # Print some basic statistics
    print("\nSample Data Statistics:")
    print(df.describe())

if __name__ == "__main__":
    main() 