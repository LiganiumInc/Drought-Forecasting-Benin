import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import math

class TaylorDiagram:
    """
    Taylor Diagram class for comparing multiple model performances
    """
    
    def __init__(self, ref_std, fig=None, rect=111, label='_'):
        """
        Initialize Taylor Diagram
        
        Parameters:
        ref_std: Standard deviation of reference data (observations)
        fig: matplotlib figure object
        rect: subplot parameters
        label: Reference label
        """
        
        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF
        
        self.ref_std = ref_std
        tr = PolarAxes.PolarTransform()
        
        # Correlation labels and positions
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        tlocs = np.arccos(rlocs)  # Conversion to polar angle
        gl1 = GF.FixedLocator(tlocs)  # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        
        # Standard deviation axis extent (in units of reference std)
        self.smin = 0
        self.smax = 1.6 * self.ref_std
        
        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi/2, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)
        
        if fig is None:
            fig = plt.figure()
            
        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        
        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")
        
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation")
        
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        
        ax.axis["bottom"].set_visible(False)
        
        # Add correlation grid lines (blue)
        ax.grid(True, color='blue', alpha=0.3, linestyle='--', linewidth=0.8)
        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates
        
        # Add reference point at (0°, ref_std) - following Taylor's convention
        l, = self.ax.plot([0], [self.ref_std], 'ko', ls='', ms=10, label=label, markerfacecolor='red', markeredgecolor='black')
        self.samplePoints = [l]
        
        # Add reference standard deviation arc (bold)
        angles = np.linspace(0, np.pi/2, 100)
        self.ax.plot(angles, [self.ref_std] * len(angles), 'k--', linewidth=2, alpha=0.6)

    def add_sample(self, std, corr, *args, **kwargs):
        """
        Add sample (model) to the Taylor diagram
        
        Parameters:
        std: Standard deviation of model predictions
        corr: Correlation coefficient between predictions and observations
        """
        
        # Ensure inputs are scalars
        std = float(std)
        corr = float(corr)
        
        # Validate correlation range
        if corr > 1.0:
            corr = 1.0
        elif corr < -1.0:
            corr = -1.0
        
        # Calculate polar coordinates
        theta = np.arccos(np.clip(corr, -1, 1))  # Ensure corr is in valid range for arccos
        
        l, = self.ax.plot(theta, std, *args, **kwargs)
        self.samplePoints.append(l)
        
        return l
        
    def add_grid(self, *args, **kwargs):
        """Add a grid"""
        self._ax.grid(*args, **kwargs)
        
    def add_contours(self, levels=5, **kwargs):
        """
        Add constant RMSD contours
        
        Parameters:
        levels: RMSD levels to plot
        kwargs: keyword arguments for plt.contour
        """
        
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                           np.linspace(0, np.pi/2))
        
        # Compute RMSD
        rms = np.sqrt(self.ref_std**2 + rs**2 - 2*self.ref_std*rs*np.cos(ts))
        
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        
        return contours


def calculate_taylor_stats(y_true, y_pred):
    """
    Calculate Taylor diagram statistics following Karl E. Taylor's original formulas
    
    Parameters:
    y_true: True values (reference field)
    y_pred: Predicted values (test field)
    
    Returns:
    dict: Dictionary containing std, correlation, and centered rmsd
    """
    
    # Ensure inputs are 1D arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        raise ValueError("No valid data points after removing NaN values")
    
    N = len(y_true_clean)
    
    # Calculate means
    f_bar = np.mean(y_pred_clean)
    r_bar = np.mean(y_true_clean)
    
    # Standard deviations (following Taylor's formula)
    sigma_f = np.sqrt(np.sum((y_pred_clean - f_bar)**2) / (N - 1))
    sigma_r = np.sqrt(np.sum((y_true_clean - r_bar)**2) / (N - 1))
    
    # Pattern correlation coefficient (following Taylor's formula)
    numerator = np.sum((y_pred_clean - f_bar) * (y_true_clean - r_bar))
    denominator = (N - 1) * sigma_f * sigma_r
    
    if denominator == 0:
        correlation = 0.0
    else:
        correlation = numerator / denominator
    
    # Ensure correlation is a scalar and within valid range
    if isinstance(correlation, (np.ndarray, list, tuple)):
        correlation = float(correlation)
    
    correlation = np.clip(correlation, -1.0, 1.0)
    
    # Centered RMS difference (following Taylor's formula: E'² = σf² + σr² - 2σfσrR)
    centered_rmsd_squared = sigma_f**2 + sigma_r**2 - 2*sigma_f*sigma_r*correlation
    centered_rmsd = np.sqrt(max(0, centered_rmsd_squared))  # Ensure non-negative
    
    # Standard RMSD (includes bias) for comparison
    standard_rmsd = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))
    
    return {
        'std': float(sigma_f),
        'correlation': float(correlation),
        'rmsd': float(standard_rmsd),  # Keep for comparison
        'centered_rmsd': float(centered_rmsd),  # This is what Taylor diagram uses
        'std_ratio': float(sigma_f / sigma_r) if sigma_r != 0 else 0.0,
        'ref_std': float(sigma_r)  # Reference standard deviation
    }


def plot_taylor_diagram(models_data, y_test, title="Taylor Diagram - Model Comparison"):
    """
    Create Taylor diagram for multiple models
    
    Parameters:
    models_data: Dictionary with model names as keys and y_pred arrays as values
    y_test: True test values
    title: Plot title
    
    Returns:
    fig: matplotlib figure object
    """
    
    # Ensure y_test is a 1D array
    y_test = np.array(y_test).flatten()
    
    # Calculate reference standard deviation
    ref_std = np.std(y_test, ddof=1)
    
    if ref_std == 0:
        raise ValueError("Reference data has zero standard deviation")
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    
    # Initialize Taylor diagram
    dia = TaylorDiagram(ref_std, fig=fig, label="Observations")
    
    # Colors and markers for different models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', '*', 'h']
    
    # Store statistics for analysis
    stats_summary = {}
    
    # Plot each model
    for i, (model_name, y_pred) in enumerate(models_data.items()):
        try:
            # Ensure y_pred is properly formatted
            y_pred = np.array(y_pred).flatten()
            
            # Check if shapes match
            if len(y_pred) != len(y_test):
                print(f"Warning: {model_name} prediction length ({len(y_pred)}) doesn't match test length ({len(y_test)})")
                # Truncate to minimum length
                min_len = min(len(y_pred), len(y_test))
                y_pred = y_pred[:min_len]
                y_test_temp = y_test[:min_len]
            else:
                y_test_temp = y_test
            
            stats = calculate_taylor_stats(y_test_temp, y_pred)
            stats_summary[model_name] = stats
            
            # Add model to Taylor diagram
            dia.add_sample(stats['std'], stats['correlation'], 
                          marker=markers[i % len(markers)], 
                          color=colors[i % len(colors)], 
                          markersize=8, 
                          label=model_name,
                          markeredgecolor='black',
                          markeredgewidth=0.5)
                          
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue
    
    # Add RMSD contours
    try:
        contours = dia.add_contours(colors='magenta', alpha=0.6)
        plt.clabel(contours, inline=True, fontsize=10, fmt='cRMSD: %.2f')
    except Exception as e:
        print(f"Warning: Could not add RMSD contours: {str(e)}")
    
    # Add legend
    plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left')
    
    # Add title
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, stats_summary


def generate_sample_data():
    """
    Generate sample data for demonstration
    This simulates the results you would have from your ML models
    """
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic time series observations
    time = np.linspace(0, 4*np.pi, n_samples)
    y_test = np.sin(time) + 0.5*np.sin(3*time) + 0.1*np.random.normal(0, 1, n_samples)
    
    # Generate synthetic predictions for different models
    # Each model has different characteristics (correlation, std, bias)
    models_predictions = {
        "Linear Regression": y_test + 0.2*np.random.normal(0, 0.3, n_samples),
        "Ridge": y_test + 0.1*np.random.normal(0, 0.25, n_samples),
        "Random Forest": y_test + 0.15*np.random.normal(0, 0.2, n_samples) + 0.1,
        "XGBoost": y_test + 0.1*np.random.normal(0, 0.15, n_samples) - 0.05,
        "LightGBM": y_test + 0.12*np.random.normal(0, 0.18, n_samples),
        "SVR": y_test + 0.25*np.random.normal(0, 0.4, n_samples) * 0.8,
        "Conv1D": y_test + 0.08*np.random.normal(0, 0.12, n_samples) + 0.02,
        "LSTM": y_test + 0.06*np.random.normal(0, 0.1, n_samples),
        "GRU": y_test + 0.07*np.random.normal(0, 0.11, n_samples),
        "Conv1D-LSTM": y_test + 0.05*np.random.normal(0, 0.08, n_samples)
    }
    
    return y_test, models_predictions


def print_statistics_summary(stats_summary):
    """
    Print detailed statistics for all models
    """
    
    print("="*95)
    print("TAYLOR DIAGRAM STATISTICS SUMMARY (Based on Karl E. Taylor's Original Formulas)")
    print("="*95)
    print("Key Concepts:")
    print("• Pattern Correlation: Measures how well the spatial/temporal patterns match")
    print("• Standard Deviation: Amplitude of variations in the field") 
    print("• Centered RMSD: Pattern error with bias removed (E'² = σf² + σr² - 2σfσrR)")
    print("• Note: Means are subtracted out - diagram shows pattern accuracy, NOT bias")
    print("="*95)
    print(f"{'Model':<15} {'Correlation':<12} {'Std Ratio':<12} {'Centered RMSD':<14} {'Standard RMSD':<14}")
    print("-"*95)
    
    # Sort models by correlation (descending)
    sorted_models = sorted(stats_summary.items(), 
                          key=lambda x: x[1]['correlation'], 
                          reverse=True)
    
    for model_name, stats in sorted_models:
        print(f"{model_name:<15} {stats['correlation']:<12.4f} "
              f"{stats['std_ratio']:<12.4f} {stats['centered_rmsd']:<14.4f} "
              f"{stats['rmsd']:<14.4f}")
    
    print("="*95)
    print("Interpretation (Following Taylor's Original Work):")
    print("• Correlation: Higher = better pattern matching (closer to 1.0)")
    print("• Std Ratio: Closer to 1.0 = correct amplitude of variations") 
    print("• Centered RMSD: Lower = better pattern accuracy (bias excluded)")
    print("• Standard RMSD: Lower = better overall accuracy (includes bias)")
    print("• Best models: Closest to reference point on Taylor diagram")
    print("• Reference point: Perfect correlation (1.0), matching std dev, zero centered RMSD")
    print("="*95)


# Example usage
if __name__ == "__main__":
    # Generate sample data (replace this with your actual data)
    y_test, models_predictions = generate_sample_data()
    
    # Create Taylor diagram
    fig, stats_summary = plot_taylor_diagram(models_predictions, y_test, 
                                           "Time Series Forecasting Models - Taylor Diagram")
    
    # Print statistics summary
    print_statistics_summary(stats_summary)
    
    # Show plot
    plt.show()
    
    
# HOW TO USE WITH YOUR ACTUAL DATA:
# 
# 1. Replace the generate_sample_data() function with your actual data
# 2. Prepare your data in this format:
#
# y_test = your_actual_test_values  # numpy array
# models_predictions = {
#     "Linear Regression": your_linear_regression_predictions,
#     "Ridge": your_ridge_predictions,
#     "Random Forest": your_random_forest_predictions,
#     "XGBoost": your_xgboost_predictions,
#     "LightGBM": your_lightgbm_predictions,
#     "SVR": your_svr_predictions,
#     "Conv1D": your_conv1d_predictions,
#     "LSTM": your_lstm_predictions,
#     "GRU": your_gru_predictions,
#     "Conv1D-LSTM": your_conv1d_lstm_predictions
# }
#
# 3. Then call:
# fig, stats_summary = plot_taylor_diagram(models_predictions, y_test)
# print_statistics_summary(stats_summary)