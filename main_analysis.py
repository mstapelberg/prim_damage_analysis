# main_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# goal here is to get the quantifiable metrics from cascade 
# 1. recombaination rate
# 2. total displacement of defects
# 3. their number of direction changes, tortuisity, mean free path
# 4. their velocity along <111> and momentum
# 5. the cluster size distribution and crowdion/dumbbel fractions 

# I need to then combine or compare these metrics to the following PEL metrics:
# 1. migration energy barrier standard deviation 
# 2. Dimensionality Reduced + Clustered grouped barriers ? Does local chemical environment play a role?
# 3. Are the local chemical environemnts with the highest variance, present in the cascades? 

def calc_tortuosity(points: np.ndarray) -> float:
    # Convert to NumPy array for convenience
    points = np.array(points)

    # Compute distances between consecutive points
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)

    # Total path length
    L_path = np.sum(segment_lengths)

    # Straight-line distance between start and end
    L_straight = np.linalg.norm(points[-1] - points[0])

    # Tortuosity
    tortuosity = L_path / L_straight

    return tortuosity

class RadiationResistanceAnalyzer:
    """
    Comprehensive analysis of radiation resistance metrics across compositions and temperatures.
    """
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.load_data()
        
    def load_data(self):
        """Load the pipeline results and model coefficients."""
        # Load aggregated metrics
        self.metrics_df = pd.read_csv(self.results_dir / "aggregated_metrics.csv")
        
        # Load RRI model coefficients
        with open(self.results_dir / "rri_model_info.json", 'r') as f:
            self.rri_coefs = json.load(f)
            
        # Load detailed per-run data
        with open(self.results_dir / "per_run_detailed.json", 'r') as f:
            self.detailed_data = json.load(f)
    
    def composition_comparison(self):
        """
        Compare radiation resistance across different alloy compositions.
        Returns summary statistics and rankings.
        """
        # Group by composition
        comp_groups = self.metrics_df.groupby('composition')
        
        # Calculate mean RRI scores for each composition
        rri_summary = comp_groups.agg({
            'RRI_phys': ['mean', 'std', 'count'],
            'RRI_ml': ['mean', 'std', 'count'],
            'recombination_rate_defects_per_ps': ['mean', 'std'],
            'crowdion_fraction': ['mean', 'std'],
            'dumbbell_mean_tortuosity': ['mean', 'std'],
            'temperature_K': ['mean', 'std']
        }).round(4)
        
        # Rank compositions by RRI_phys (higher = better)
        rri_ranking = comp_groups['RRI_phys'].mean().sort_values(ascending=False)
        
        return {
            'summary_stats': rri_summary,
            'rri_ranking': rri_ranking,
            'best_composition': rri_ranking.index[0],
            'worst_composition': rri_ranking.index[-1]
        }
    
    def temperature_effects(self):
        """
        Quantify the effects of temperature on radiation resistance.
        """
        # Filter out runs with missing temperature data
        temp_data = self.metrics_df.dropna(subset=['temperature_K', 'RRI_phys'])
        
        if len(temp_data) < 3:
            return {"error": "Insufficient temperature data for analysis"}
        
        # Calculate correlation between temperature and RRI
        temp_corr_phys = temp_data['temperature_K'].corr(temp_data['RRI_phys'])
        temp_corr_ml = temp_data['temperature_K'].corr(temp_data['RRI_ml'])
        
        # Group by temperature ranges
        temp_data['temp_range'] = pd.cut(temp_data['temperature_K'], 
                                       bins=3, labels=['Low', 'Medium', 'High'])
        
        temp_effects = temp_data.groupby('temp_range').agg({
            'RRI_phys': ['mean', 'std', 'count'],
            'RRI_ml': ['mean', 'std', 'count'],
            'recombination_rate_defects_per_ps': ['mean', 'std'],
            'temperature_K': ['min', 'max']
        }).round(4)
        
        # Linear regression for temperature dependence
        from scipy import stats
        slope_phys, intercept_phys, r_value_phys, p_value_phys, std_err_phys = stats.linregress(
            temp_data['temperature_K'], temp_data['RRI_phys'])
        slope_ml, intercept_ml, r_value_ml, p_value_ml, std_err_ml = stats.linregress(
            temp_data['temperature_K'], temp_data['RRI_ml'])
        
        return {
            'correlations': {
                'RRI_phys': temp_corr_phys,
                'RRI_ml': temp_corr_ml
            },
            'temperature_ranges': temp_effects,
            'linear_fits': {
                'RRI_phys': {
                    'slope': slope_phys,
                    'intercept': intercept_phys,
                    'r_squared': r_value_phys**2,
                    'p_value': p_value_phys,
                    'std_error': std_err_phys
                },
                'RRI_ml': {
                    'slope': slope_ml,
                    'intercept': intercept_ml,
                    'r_squared': r_value_ml**2,
                    'p_value': p_value_ml,
                    'std_error': std_err_ml
                }
            }
        }
    
    def feature_importance_analysis(self):
        """
        Analyze which features are most important for radiation resistance.
        """
        # Get the RRI model coefficients
        coefs = self.rri_coefs.copy()
        intercept = coefs.pop('intercept')
        
        # Sort features by absolute coefficient value
        feature_importance = pd.DataFrame([
            {'feature': k, 'coefficient': v, 'abs_coefficient': abs(v)}
            for k, v in coefs.items()
        ]).sort_values('abs_coefficient', ascending=False)
        
        # Calculate relative importance (normalized)
        feature_importance['relative_importance'] = (
            feature_importance['abs_coefficient'] / 
            feature_importance['abs_coefficient'].sum()
        )
        
        return {
            'feature_importance': feature_importance,
            'most_important': feature_importance.iloc[0]['feature'],
            'least_important': feature_importance.iloc[-1]['feature']
        }
    
    def composition_temperature_interaction(self):
        """
        Analyze how temperature effects vary by composition.
        """
        # Filter data with both composition and temperature
        data = self.metrics_df.dropna(subset=['composition', 'temperature_K', 'RRI_phys'])
        
        if len(data) < 6:
            return {"error": "Insufficient data for interaction analysis"}
        
        # Calculate temperature sensitivity for each composition
        comp_temp_effects = {}
        
        for comp in data['composition'].unique():
            comp_data = data[data['composition'] == comp]
            if len(comp_data) < 2:
                continue
                
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                comp_data['temperature_K'], comp_data['RRI_phys'])
            
            comp_temp_effects[comp] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'n_points': len(comp_data),
                'temp_range': (comp_data['temperature_K'].min(), comp_data['temperature_K'].max())
            }
        
        return comp_temp_effects
    
    def robust_z_score(self, data):
        """Calculate robust z-score using median and MAD."""
        data = np.asarray(data, dtype=float)
        med = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - med))
        if not np.isfinite(mad) or mad == 0:
            return np.zeros_like(data, dtype=float)
        return (data - med) / (1.4826 * mad)
    
    def generate_grouped_metrics_plots(self, save_dir="plots"):
        """
        Generate separate plots for metrics with similar scales using raw values.
        Groups metrics by their typical value ranges.
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Define metric groups by scale
        metric_groups = {
            'recombination_rate': {
                'metrics': ['recombination_rate_defects_per_ps'],
                'labels': ['Recombination Rate (defects/ps)'],
                'colors': ['#2A33C3'],
                'title': 'Recombination Rate Comparison'
            },
            'fractions': {
                'metrics': ['crowdion_fraction', 'dumbbell_fraction'],
                'labels': ['Crowdion Fraction', 'Dumbbell Fraction'],
                'colors': ['#A35D00', '#0B7285'],
                'title': 'Defect Type Fractions'
            },
            'displacements': {
                'metrics': ['crowdion_mean_net_displacement_A', 'dumbbell_mean_net_displacement_A'],
                'labels': ['Crowdion Net Displacement (Å)', 'Dumbbell Net Displacement (Å)'],
                'colors': ['#8F2D56', '#6E8B00'],
                'title': 'Net Displacements'
            },
            'tortuosity': {
                'metrics': ['dumbbell_mean_tortuosity'],
                'labels': ['Dumbbell Tortuosity'],
                'colors': ['#D2691E'],
                'title': 'Dumbbell Tortuosity'
            },
            'cluster_size': {
                'metrics': ['median_cluster_size_atoms'],
                'labels': ['Median Cluster Size (atoms)'],
                'colors': ['#2A33C3'],
                'title': 'Cluster Size'
            }
        }
        
        # Filter data for the two target temperatures
        target_temps = [873, 300]  # K
        temp_tolerance = 50  # K tolerance for temperature matching
        
        for group_name, group_info in metric_groups.items():
            metrics = group_info['metrics']
            labels = group_info['labels']
            colors = group_info['colors']
            title = group_info['title']
            
            # Create 2-panel plot for this group
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            for panel_idx, target_temp in enumerate(target_temps):
                ax = axes[panel_idx]
                
                # Filter data for this temperature range
                temp_mask = (self.metrics_df['temperature_K'] >= target_temp - temp_tolerance) & \
                           (self.metrics_df['temperature_K'] <= target_temp + temp_tolerance)
                temp_data = self.metrics_df[temp_mask].copy()
                
                if len(temp_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {target_temp}K ± {temp_tolerance}K', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{target_temp}K', fontsize=16, fontweight='bold')
                    continue
                
                # Get unique compositions
                compositions = sorted(temp_data['composition'].dropna().unique())
                
                if len(compositions) == 0:
                    ax.text(0.5, 0.5, f'No composition data for {target_temp}K', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{target_temp}K', fontsize=16, fontweight='bold')
                    continue
                
                # Group by composition and calculate mean values
                comp_values = {}
                for comp in compositions:
                    comp_mask = temp_data['composition'] == comp
                    comp_values[comp] = {}
                    for metric in metrics:
                        if metric in temp_data.columns:
                            comp_data = temp_data[comp_mask][metric].dropna()
                            if len(comp_data) > 0:
                                comp_values[comp][metric] = comp_data.mean()
                            else:
                                comp_values[comp][metric] = 0
                        else:
                            comp_values[comp][metric] = 0
                
                # Create grouped bar chart
                x = np.arange(len(compositions))
                width = 0.8 / len(metrics)  # Adjust width based on number of metrics
                
                for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
                    y_values = [comp_values[comp].get(metric, 0) for comp in compositions]
                    ax.bar(x + i * width, y_values, width, 
                          label=label, color=color, alpha=0.7)
                
                # Customize the plot
                ax.set_xlabel('Composition', fontsize=14)
                ax.set_ylabel('Value', fontsize=14)
                ax.set_title(f'{target_temp}K', fontsize=16, fontweight='bold')
                ax.set_xticks(x + width * (len(metrics) - 1) / 2)  # Center the x-ticks
                ax.set_xticklabels(compositions, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add legend only to the top panel
                if panel_idx == 0:
                    ax.legend(fontsize=12)
            
            plt.suptitle(title, fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path / f'{group_name}_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{title} plot saved to {save_path}/{group_name}_comparison.png")
    
    def generate_statistical_spread_plots(self, save_dir="plots"):
        """
        Generate plots showing statistical spread (q10, q25, q50, q75, q90, mean) 
        for each composition and temperature.
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Define metric groups by scale
        metric_groups = {
            'recombination_rate': {
                'metrics': ['recombination_rate_defects_per_ps'],
                'labels': ['Recombination Rate (defects/ps)'],
                'title': 'Recombination Rate Statistical Spread'
            },
            'fractions': {
                'metrics': ['crowdion_fraction', 'dumbbell_fraction'],
                'labels': ['Crowdion Fraction', 'Dumbbell Fraction'],
                'title': 'Defect Type Fractions Statistical Spread'
            },
            'displacements': {
                'metrics': ['crowdion_mean_net_displacement_A', 'dumbbell_mean_net_displacement_A'],
                'labels': ['Crowdion Net Displacement (Å)', 'Dumbbell Net Displacement (Å)'],
                'title': 'Net Displacements Statistical Spread'
            },
            'tortuosity': {
                'metrics': ['dumbbell_mean_tortuosity'],
                'labels': ['Dumbbell Tortuosity'],
                'title': 'Dumbbell Tortuosity Statistical Spread'
            },
            'cluster_size': {
                'metrics': ['median_cluster_size_atoms'],
                'labels': ['Median Cluster Size (atoms)'],
                'title': 'Cluster Size Statistical Spread'
            }
        }
        
        # Filter data for the two target temperatures
        target_temps = [873, 300]  # K
        temp_tolerance = 50  # K tolerance for temperature matching
        
        for group_name, group_info in metric_groups.items():
            metrics = group_info['metrics']
            labels = group_info['labels']
            title = group_info['title']
            
            # Create 2-panel plot for this group
            fig, axes = plt.subplots(2, 1, figsize=(16, 12))
            
            for panel_idx, target_temp in enumerate(target_temps):
                ax = axes[panel_idx]
                
                # Filter data for this temperature range
                temp_mask = (self.metrics_df['temperature_K'] >= target_temp - temp_tolerance) & \
                           (self.metrics_df['temperature_K'] <= target_temp + temp_tolerance)
                temp_data = self.metrics_df[temp_mask].copy()
                
                if len(temp_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {target_temp}K ± {temp_tolerance}K', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{target_temp}K', fontsize=16, fontweight='bold')
                    continue
                
                # Get unique compositions
                compositions = sorted(temp_data['composition'].dropna().unique())
                
                if len(compositions) == 0:
                    ax.text(0.5, 0.5, f'No composition data for {target_temp}K', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{target_temp}K', fontsize=16, fontweight='bold')
                    continue
                
                # Calculate statistics for each composition and metric
                x_pos = 0
                all_positions = []
                all_labels = []
                
                for comp in compositions:
                    comp_mask = temp_data['composition'] == comp
                    comp_data = temp_data[comp_mask]
                    
                    for i, metric in enumerate(metrics):
                        if metric not in comp_data.columns:
                            continue
                            
                        values = comp_data[metric].dropna()
                        if len(values) == 0:
                            continue
                        
                        # Calculate statistics
                        stats = {
                            'q10': np.percentile(values, 10),
                            'q25': np.percentile(values, 25),
                            'q50': np.percentile(values, 50),  # median
                            'q75': np.percentile(values, 75),
                            'q90': np.percentile(values, 90),
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
                        
                        # Create box plot data
                        box_data = [stats['q10'], stats['q25'], stats['q50'], 
                                  stats['q75'], stats['q90']]
                        
                        # Plot box plot
                        bp = ax.boxplot([values], positions=[x_pos], widths=0.6, 
                                       patch_artist=True, showfliers=True)
                        
                        # Color the box
                        bp['boxes'][0].set_facecolor(plt.cm.tab10(i % 10))
                        bp['boxes'][0].set_alpha(0.7)
                        
                        # Add mean as a diamond
                        ax.scatter(x_pos, stats['mean'], marker='D', s=100, 
                                 color='red', zorder=5, label='Mean' if x_pos == 0 else "")
                        
                        # Add sample size text
                        ax.text(x_pos, ax.get_ylim()[1] * 0.95, f'n={len(values)}', 
                               ha='center', va='top', fontsize=8)
                        
                        all_positions.append(x_pos)
                        all_labels.append(f'{comp}\n{labels[i]}')
                        x_pos += 1
                
                # Customize the plot
                ax.set_xlabel('Composition & Metric', fontsize=14)
                ax.set_ylabel('Value', fontsize=14)
                ax.set_title(f'{target_temp}K', fontsize=16, fontweight='bold')
                ax.set_xticks(all_positions)
                ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add legend only to the top panel
                if panel_idx == 0:
                    ax.legend(fontsize=12)
            
            plt.suptitle(title, fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path / f'{group_name}_statistical_spread.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{title} plot saved to {save_path}/{group_name}_statistical_spread.png")
    
    def generate_metrics_comparison_plot(self, save_dir="plots", normalization='minmax'):
        """
        Generate 2-panel chart comparing key metrics at 300K vs 873K.
        
        Parameters:
        - normalization: 'minmax', 'zscore', 'raw', or 'relative'
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Define the metrics to analyze
        metrics = [
            'recombination_rate_defects_per_ps',
            'dumbbell_mean_tortuosity', 
            'crowdion_fraction',
            'dumbbell_mean_net_displacement_A',
            'crowdion_mean_net_displacement_A',
            'median_cluster_size_atoms'
        ]
        
        metric_labels = [
            'Recombination Rate',
            'Dumbbell Tortuosity',
            'Crowdion Fraction', 
            'Dumbbell Net Displacement',
            'Crowdion Net Displacement',
            'Max Cluster Size'
        ]
        
        # Color scheme
        colors = ['#2A33C3', '#A35D00', '#0B7285', '#8F2D56', '#6E8B00', '#D2691E']
        
        # Filter data for the two target temperatures
        target_temps = [873, 300]  # K
        temp_tolerance = 50  # K tolerance for temperature matching
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # Calculate normalization factors across all data
        if normalization == 'minmax':
            norm_factors = {}
            for metric in metrics:
                if metric in self.metrics_df.columns:
                    data = self.metrics_df[metric].dropna()
                    if len(data) > 0:
                        norm_factors[metric] = {'min': data.min(), 'max': data.max()}
                    else:
                        norm_factors[metric] = {'min': 0, 'max': 1}
                else:
                    norm_factors[metric] = {'min': 0, 'max': 1}
        elif normalization == 'zscore':
            norm_factors = {}
            for metric in metrics:
                if metric in self.metrics_df.columns:
                    data = self.metrics_df[metric].dropna()
                    if len(data) > 0:
                        norm_factors[metric] = {'mean': data.mean(), 'std': data.std()}
                    else:
                        norm_factors[metric] = {'mean': 0, 'std': 1}
                else:
                    norm_factors[metric] = {'mean': 0, 'std': 1}
        
        for panel_idx, target_temp in enumerate(target_temps):
            ax = axes[panel_idx]
            
            # Filter data for this temperature range
            temp_mask = (self.metrics_df['temperature_K'] >= target_temp - temp_tolerance) & \
                       (self.metrics_df['temperature_K'] <= target_temp + temp_tolerance)
            temp_data = self.metrics_df[temp_mask].copy()
            
            if len(temp_data) == 0:
                ax.text(0.5, 0.5, f'No data for {target_temp}K ± {temp_tolerance}K', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{target_temp}K', fontsize=16, fontweight='bold')
                continue
            
            # Get unique compositions
            compositions = sorted(temp_data['composition'].dropna().unique())
            
            if len(compositions) == 0:
                ax.text(0.5, 0.5, f'No composition data for {target_temp}K', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{target_temp}K', fontsize=16, fontweight='bold')
                continue
            
            # Group by composition and calculate mean values
            comp_values = {}
            for comp in compositions:
                comp_mask = temp_data['composition'] == comp
                comp_values[comp] = {}
                for i, metric in enumerate(metrics):
                    if metric in temp_data.columns:
                        comp_data = temp_data[comp_mask][metric].dropna()
                        if len(comp_data) > 0:
                            mean_val = comp_data.mean()
                            
                            # Apply normalization
                            if normalization == 'minmax':
                                min_val, max_val = norm_factors[metric]['min'], norm_factors[metric]['max']
                                if max_val > min_val:
                                    norm_val = (mean_val - min_val) / (max_val - min_val)
                                else:
                                    norm_val = 0.5
                            elif normalization == 'zscore':
                                mean_norm, std_norm = norm_factors[metric]['mean'], norm_factors[metric]['std']
                                if std_norm > 0:
                                    norm_val = (mean_val - mean_norm) / std_norm
                                else:
                                    norm_val = 0
                            elif normalization == 'relative':
                                # Normalize by the mean across all compositions at this temperature
                                all_comp_data = temp_data[metric].dropna()
                                if len(all_comp_data) > 0 and all_comp_data.mean() != 0:
                                    norm_val = mean_val / all_comp_data.mean()
                                else:
                                    norm_val = 1.0
                            else:  # raw values
                                norm_val = mean_val
                            
                            comp_values[comp][metric] = norm_val
                        else:
                            comp_values[comp][metric] = 0
                    else:
                        comp_values[comp][metric] = 0
            
            # Create grouped bar chart
            x = np.arange(len(compositions))
            width = 0.12  # Width of each bar group
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                y_values = [comp_values[comp].get(metric, 0) for comp in compositions]
                ax.bar(x + i * width, y_values, width, 
                      label=label, color=colors[i], alpha=0.6)
            
            # Customize the plot
            ax.set_xlabel('Composition', fontsize=14)
            
            # Set y-axis label based on normalization
            if normalization == 'minmax':
                ax.set_ylabel('Normalized Value (0-1)', fontsize=14)
            elif normalization == 'zscore':
                ax.set_ylabel('Z-Score', fontsize=14)
            elif normalization == 'relative':
                ax.set_ylabel('Relative to Mean', fontsize=14)
            else:
                ax.set_ylabel('Raw Value', fontsize=14)
            
            ax.set_title(f'{target_temp}K', fontsize=16, fontweight='bold')
            ax.set_xticks(x + width * 2.5)  # Center the x-ticks
            ax.set_xticklabels(compositions, rotation=45, ha='right')
            
            # Add reference lines
            if normalization == 'zscore':
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            elif normalization == 'relative':
                ax.axhline(y=1, color='black', linestyle='-', alpha=0.3)
            elif normalization == 'minmax':
                ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add legend only to the top panel
            if panel_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path / f'metrics_comparison_2panel_{normalization}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"2-panel metrics comparison plot ({normalization}) saved to {save_path}/metrics_comparison_2panel_{normalization}.png")
    
    def load_cluster_data_from_csvs(self):
        """
        Load cluster size data from individual CSV files for each run.
        Returns a dictionary with composition -> temperature -> cluster_sizes
        """
        cluster_data = {}
        
        # Load detailed per-run data to get file paths
        with open(self.results_dir / "per_run_detailed.json", 'r') as f:
            detailed_data = json.load(f)
        
        for run_dir, run_data in detailed_data.items():
            composition = run_data.get('composition')
            temperature = run_data.get('temperature_K')
            
            if not composition or not temperature:
                continue
            
            if composition not in cluster_data:
                cluster_data[composition] = {}
            if temperature not in cluster_data[composition]:
                cluster_data[composition][temperature] = []
            
            # Look for cluster CSV file in the run directory
            cluster_csv_path = Path(run_dir) / "custom_clusters_per_frame.csv"
            
            if cluster_csv_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(cluster_csv_path)
                    if 'size' in df.columns:
                        cluster_sizes = df['size'].dropna().tolist()
                        cluster_data[composition][temperature].extend(cluster_sizes)
                except Exception as e:
                    print(f"Warning: Could not read cluster data from {cluster_csv_path}: {e}")
        
        return cluster_data
    
    def generate_cluster_size_distributions(self, save_dir="plots"):
        """
        Generate cluster size distribution plots for each composition at 300K and 873K.
        2 rows × 3 columns layout with overlaid temperature distributions.
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        print("Loading cluster data from individual CSV files...")
        cluster_data = self.load_cluster_data_from_csvs()
        
        # Get unique compositions
        compositions = sorted(cluster_data.keys())
        
        if len(compositions) == 0:
            print("No composition data found for cluster size analysis")
            return
        
        # Create 2x3 subplot layout
        n_compositions = len(compositions)
        n_cols = 3
        n_rows = (n_compositions + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Temperature colors
        temp_colors = {'300K': '#2A33C3', '873K': '#A35D00'}
        temp_tolerance = 50  # K tolerance for temperature matching
        
        for i, comp in enumerate(compositions):
            ax = axes_flat[i]
            
            if comp not in cluster_data:
                ax.text(0.5, 0.5, f'No cluster data for {comp}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(comp, fontsize=14, fontweight='bold')
                continue
            
            # Plot distributions for each temperature
            for temp_label, target_temp in [('300K', 300), ('873K', 873)]:
                # Find temperatures within tolerance
                matching_temps = []
                for temp in cluster_data[comp].keys():
                    if abs(temp - target_temp) <= temp_tolerance:
                        matching_temps.append(temp)
                
                if not matching_temps:
                    continue
                
                # Combine cluster sizes from all matching temperatures
                all_cluster_sizes = []
                for temp in matching_temps:
                    all_cluster_sizes.extend(cluster_data[comp][temp])
                
                if len(all_cluster_sizes) > 0:
                    # Create histogram
                    ax.hist(all_cluster_sizes, bins=20, alpha=0.6, 
                           color=temp_colors[temp_label], 
                           label=f'{temp_label} (n={len(all_cluster_sizes)})',
                           density=True, edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            ax.set_title(comp, fontsize=14, fontweight='bold')
            ax.set_xlabel('Cluster Size (atoms)', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = []
            for temp_label, target_temp in [('300K', 300), ('873K', 873)]:
                matching_temps = [temp for temp in cluster_data[comp].keys() 
                                if abs(temp - target_temp) <= temp_tolerance]
                
                if matching_temps:
                    all_sizes = []
                    for temp in matching_temps:
                        all_sizes.extend(cluster_data[comp][temp])
                    
                    if all_sizes:
                        mean_size = np.mean(all_sizes)
                        std_size = np.std(all_sizes)
                        stats_text.append(f'{temp_label}: μ={mean_size:.2f}±{std_size:.2f}')
            
            if stats_text:
                ax.text(0.02, 0.98, '\n'.join(stats_text), 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(compositions), len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path / 'cluster_size_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cluster size distribution plot saved to {save_path}/cluster_size_distributions.png")
    
    def generate_plots(self, save_dir="plots"):
        """
        Generate comprehensive visualization plots.
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Generate the main 2-panel comparison plot
        self.generate_metrics_comparison_plot(save_dir)
        
        # Set up the plotting style for other plots
        plt.style.use('seaborn-v0_8')
        fig_size = (12, 8)
        
        # 1. RRI comparison across compositions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        comp_data = self.metrics_df.groupby('composition').agg({
            'RRI_phys': ['mean', 'std'],
            'RRI_ml': ['mean', 'std']
        })
        
        # RRI_phys plot
        comp_data['RRI_phys']['mean'].plot(kind='bar', yerr=comp_data['RRI_phys']['std'], 
                                          ax=ax1, capsize=5)
        ax1.set_title('RRI_phys by Composition')
        ax1.set_ylabel('RRI_phys Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # RRI_ml plot
        comp_data['RRI_ml']['mean'].plot(kind='bar', yerr=comp_data['RRI_ml']['std'], 
                                        ax=ax2, capsize=5)
        ax2.set_title('RRI_ml by Composition')
        ax2.set_ylabel('RRI_ml Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path / 'rri_by_composition.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Temperature effects
        temp_data = self.metrics_df.dropna(subset=['temperature_K', 'RRI_phys'])
        
        if len(temp_data) > 0:
            fig, ax = plt.subplots(figsize=fig_size)
            
            # Scatter plot with composition colors
            compositions = temp_data['composition'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(compositions)))
            
            for i, comp in enumerate(compositions):
                comp_subset = temp_data[temp_data['composition'] == comp]
                ax.scatter(comp_subset['temperature_K'], comp_subset['RRI_phys'], 
                          label=comp, color=colors[i], alpha=0.7, s=50)
            
            # Add trend line
            z = np.polyfit(temp_data['temperature_K'], temp_data['RRI_phys'], 1)
            p = np.poly1d(z)
            ax.plot(temp_data['temperature_K'], p(temp_data['temperature_K']), 
                   "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel('RRI_phys Score')
            ax.set_title('Temperature Effects on Radiation Resistance')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path / 'temperature_effects.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Feature importance
        feature_imp = self.feature_importance_analysis()['feature_importance']
        
        fig, ax = plt.subplots(figsize=fig_size)
        feature_imp.plot(x='feature', y='relative_importance', kind='bar', ax=ax)
        ax.set_title('Feature Importance for Radiation Resistance')
        ax.set_ylabel('Relative Importance')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Correlation heatmap
        numeric_cols = self.metrics_df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.metrics_df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax, fmt='.2f')
        ax.set_title('Correlation Matrix of Radiation Resistance Metrics')
        
        plt.tight_layout()
        plt.savefig(save_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {save_path}/")
    
    def generate_report(self, output_file="radiation_resistance_analysis.txt"):
        """
        Generate a comprehensive text report of the analysis.
        """
        with open(output_file, 'w') as f:
            f.write("RADIATION RESISTANCE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Composition comparison
            f.write("1. COMPOSITION COMPARISON\n")
            f.write("-" * 30 + "\n")
            comp_analysis = self.composition_comparison()
            f.write(f"Best composition: {comp_analysis['best_composition']}\n")
            f.write(f"Worst composition: {comp_analysis['worst_composition']}\n\n")
            
            f.write("RRI Rankings (higher = better):\n")
            for i, (comp, score) in enumerate(comp_analysis['rri_ranking'].items(), 1):
                f.write(f"{i}. {comp}: {score:.4f}\n")
            f.write("\n")
            
            # Temperature effects
            f.write("2. TEMPERATURE EFFECTS\n")
            f.write("-" * 30 + "\n")
            temp_analysis = self.temperature_effects()
            
            if 'error' not in temp_analysis:
                f.write(f"Temperature correlation with RRI_phys: {temp_analysis['correlations']['RRI_phys']:.4f}\n")
                f.write(f"Temperature correlation with RRI_ml: {temp_analysis['correlations']['RRI_ml']:.4f}\n\n")
                
                f.write("Linear fit for RRI_phys vs Temperature:\n")
                fit = temp_analysis['linear_fits']['RRI_phys']
                f.write(f"  Slope: {fit['slope']:.6f} RRI/K\n")
                f.write(f"  R²: {fit['r_squared']:.4f}\n")
                f.write(f"  P-value: {fit['p_value']:.4f}\n\n")
            else:
                f.write(f"Error: {temp_analysis['error']}\n\n")
            
            # Feature importance
            f.write("3. FEATURE IMPORTANCE\n")
            f.write("-" * 30 + "\n")
            feat_analysis = self.feature_importance_analysis()
            f.write("Most important features (by coefficient magnitude):\n")
            for _, row in feat_analysis['feature_importance'].iterrows():
                f.write(f"  {row['feature']}: {row['coefficient']:.4f} (importance: {row['relative_importance']:.1%})\n")
            f.write("\n")
            
            # Summary statistics
            f.write("4. SUMMARY STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total runs analyzed: {len(self.metrics_df)}\n")
            f.write(f"Compositions: {', '.join(self.metrics_df['composition'].dropna().unique())}\n")
            f.write(f"Temperature range: {self.metrics_df['temperature_K'].min():.0f} - {self.metrics_df['temperature_K'].max():.0f} K\n")
            f.write(f"RRI_phys range: {self.metrics_df['RRI_phys'].min():.4f} - {self.metrics_df['RRI_phys'].max():.4f}\n")
            f.write(f"RRI_ml range: {self.metrics_df['RRI_ml'].min():.4f} - {self.metrics_df['RRI_ml'].max():.4f}\n")
        
        print(f"Report saved to {output_file}")

def main():
    """
    Run the complete radiation resistance analysis.
    """
    print("Starting Radiation Resistance Analysis...")
    
    # Initialize analyzer
    analyzer = RadiationResistanceAnalyzer()
    
    # Generate the main 2-panel metrics comparison plots with different normalizations
    print("1. Generating 2-panel metrics comparison plots...")
    print("   - Min-max normalized (0-1 scale)")
    analyzer.generate_metrics_comparison_plot(normalization='minmax')
    print("   - Relative to mean (1.0 = average)")
    analyzer.generate_metrics_comparison_plot(normalization='relative')
    print("   - Raw values")
    analyzer.generate_metrics_comparison_plot(normalization='raw')
    
    print("   - Grouped raw value plots (by scale)")
    analyzer.generate_grouped_metrics_plots()
    
    print("   - Statistical spread plots (q10, q25, q50, q75, q90, mean)")
    analyzer.generate_statistical_spread_plots()
    
    print("   - Cluster size distributions")
    analyzer.generate_cluster_size_distributions()
    
    # Run additional analyses
    print("2. Analyzing composition effects...")
    comp_results = analyzer.composition_comparison()
    print(f"   Best composition: {comp_results['best_composition']}")
    
    print("3. Analyzing temperature effects...")
    temp_results = analyzer.temperature_effects()
    if 'error' not in temp_results:
        print(f"   Temperature correlation: {temp_results['correlations']['RRI_phys']:.4f}")
    
    print("4. Analyzing feature importance...")
    feat_results = analyzer.feature_importance_analysis()
    print(f"   Most important feature: {feat_results['most_important']}")
    
    print("5. Generating additional plots...")
    analyzer.generate_plots()
    
    print("6. Generating report...")
    analyzer.generate_report()
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()