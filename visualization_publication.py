"""
Publication-Quality Visualization Module for Cancer Drug Synergy Pipeline
===========================================================================

Generates high-quality, journal-ready figures with professional aesthetics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

# Professional color palettes
COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Green
    'warning': '#D62828',      # Red
    'neutral': '#6C757D',      # Gray
    'cell_lines': ['#E63946', '#457B9D', '#2A9D8F'],
    'mechanisms': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', 
                   '#8338EC', '#3A86FF', '#FB5607', '#FF006E', '#06FFA5'],
    'synergy': ['#003049', '#D62828', '#F77F00', '#FCBF49', '#EAE2B7']
}


class PublicationVisualizer:
    """
    Generate publication-quality figures for cancer drug synergy analysis.

    Produces multi-panel figures with professional styling suitable for
    high-impact scientific journals.
    """

    def __init__(self, results_dir: Path, figures_dir: Path, logger=None):
        """
        Initialize the visualizer.

        Args:
            results_dir: Directory containing results CSV files
            figures_dir: Directory to save figures
            logger: Optional logger instance
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        # Load results
        self.screening_results = self._load_screening_results()

    def _load_screening_results(self) -> Dict[str, pd.DataFrame]:
        """Load all screening results from CSV files."""
        results = {}
        for csv_file in self.results_dir.glob('*_screening_results.csv'):
            cell_line = csv_file.stem.replace('_screening_results', '')
            results[cell_line] = pd.read_csv(csv_file)
            if self.logger:
                self.logger.info(f"Loaded {len(results[cell_line])} results for {cell_line}")
        return results

    def _log(self, message: str):
        """Log message if logger available."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def create_all_figures(self, history: Dict = None):
        """
        Generate all publication figures.

        Args:
            history: Training history dictionary (optional)
        """
        self._log("\n" + "="*80)
        self._log("GENERATING PUBLICATION-QUALITY FIGURES")
        self._log("="*80)

        # Figure 1: Training dynamics (if history provided)
        if history:
            self.plot_training_dynamics(history)

        # Figure 2: Synergy landscape overview
        self.plot_synergy_landscape()

        # Figure 3: Top combinations comparison
        self.plot_top_combinations_comparison()

        # Figure 4: Mechanism-based analysis
        self.plot_mechanism_analysis()

        # Figure 5: Drug interaction heatmaps
        self.plot_drug_interaction_heatmaps()

        # Figure 6: Cell line comparison
        self.plot_cell_line_comparison()

        # Figure 7: Distribution analysis
        self.plot_distribution_analysis()

        # Figure 8: Network visualization
        self.plot_drug_network()

        # Figure 9: Statistical summary
        self.plot_statistical_summary()

        self._log("\n" + "="*80)
        self._log(f"[OK] All figures saved to {self.figures_dir}")
        self._log("="*80 + "\n")

    def plot_training_dynamics(self, history: Dict):
        """
        Figure 1: Comprehensive training dynamics visualization.

        Multi-panel figure showing loss curves, R² progression, and convergence.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

        epochs = np.arange(1, len(history['train_loss']) + 1)

        # A: Training and validation loss
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(epochs, history['train_loss'], 'o-', 
                color=COLORS['primary'], linewidth=2.5, markersize=6,
                label='Training Loss', alpha=0.8)
        ax1.plot(epochs, history['val_loss'], 's-', 
                color=COLORS['secondary'], linewidth=2.5, markersize=6,
                label='Validation Loss', alpha=0.8)
        ax1.fill_between(epochs, history['train_loss'], alpha=0.2, color=COLORS['primary'])
        ax1.fill_between(epochs, history['val_loss'], alpha=0.2, color=COLORS['secondary'])
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('A. Training and Validation Loss', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # B: R² Score progression
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(epochs, history['val_r2'], 'D-', 
                color=COLORS['accent'], linewidth=2.5, markersize=6)
        ax2.fill_between(epochs, history['val_r2'], alpha=0.3, color=COLORS['accent'])
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax2.set_title('B. Model R² Score', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # C: Synergy loss component
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, history['train_synergy_loss'], 'o-', 
                color=COLORS['success'], linewidth=2, markersize=5, alpha=0.7)
        ax3.plot(epochs, history['val_synergy_loss'], 's-', 
                color=COLORS['warning'], linewidth=2, markersize=5, alpha=0.7)
        ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Synergy Loss', fontsize=11, fontweight='bold')
        ax3.set_title('C. Synergy Prediction Loss', fontsize=13, fontweight='bold', pad=12)
        ax3.legend(['Train', 'Val'], fontsize=9)
        ax3.grid(True, alpha=0.3)

        # D: MAE progression
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, history['val_mae'], '^-', 
                color=COLORS['primary'], linewidth=2, markersize=5)
        ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax4.set_ylabel('MAE', fontsize=11, fontweight='bold')
        ax4.set_title('D. Mean Absolute Error', fontsize=13, fontweight='bold', pad=12)
        ax4.grid(True, alpha=0.3)

        # E: MSE progression
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(epochs, history['val_mse'], 'v-', 
                color=COLORS['secondary'], linewidth=2, markersize=5)
        ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax5.set_ylabel('MSE', fontsize=11, fontweight='bold')
        ax5.set_title('E. Mean Squared Error', fontsize=13, fontweight='bold', pad=12)
        ax5.grid(True, alpha=0.3)

        # F: Learning rate (if available)
        ax6 = fig.add_subplot(gs[2, 0])
        if 'learning_rate' in history:
            ax6.plot(epochs, history['learning_rate'], 'o-', 
                    color=COLORS['accent'], linewidth=2, markersize=4)
            ax6.set_yscale('log')
        else:
            ax6.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                    ha='center', va='center', fontsize=12, color='gray')
        ax6.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
        ax6.set_title('F. Learning Rate Schedule', fontsize=13, fontweight='bold', pad=12)
        ax6.grid(True, alpha=0.3)

        # G: Loss improvement per epoch
        ax7 = fig.add_subplot(gs[2, 1])
        val_loss_diff = np.diff(history['val_loss'])
        ax7.bar(epochs[1:], -val_loss_diff, 
               color=np.where(val_loss_diff < 0, COLORS['success'], COLORS['warning']),
               alpha=0.7, edgecolor='black', linewidth=0.5)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax7.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Loss Improvement', fontsize=11, fontweight='bold')
        ax7.set_title('G. Epoch-to-Epoch Improvement', fontsize=13, fontweight='bold', pad=12)
        ax7.grid(True, alpha=0.3, axis='y')

        # H: Final metrics summary
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        metrics_text = f"""
        FINAL PERFORMANCE
        ─────────────────────
        Best Val Loss: {min(history['val_loss']):.4f}
        Final R²: {history['val_r2'][-1]:.4f}
        Final MAE: {history['val_mae'][-1]:.4f}
        Final MSE: {history['val_mse'][-1]:.4f}

        Total Epochs: {len(epochs)}
        Convergence: {"✓" if val_loss_diff[-1] > -0.001 else "..."}
        """
        ax8.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))

        plt.suptitle('Training Dynamics and Model Convergence', 
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(self.figures_dir / 'Figure1_Training_Dynamics.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        self._log("[OK] Figure 1: Training Dynamics")

    def plot_synergy_landscape(self):
        """
        Figure 2: Synergy landscape overview across all cell lines.

        Violin plots, box plots, and density distributions.
        """
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Prepare data
        all_data = []
        for cell_line, df in self.screening_results.items():
            temp_df = df.copy()
            temp_df['cell_line'] = cell_line
            all_data.append(temp_df)
        combined_df = pd.concat(all_data, ignore_index=True)

        # A: Violin plot of synergy distributions
        ax1 = fig.add_subplot(gs[0, :2])
        parts = ax1.violinplot([self.screening_results[cl]['predicted_synergy'] 
                               for cl in self.screening_results.keys()],
                              positions=range(len(self.screening_results)),
                              showmeans=True, showmedians=True, widths=0.7)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLORS['cell_lines'][i % len(COLORS['cell_lines'])])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)

        ax1.set_xticks(range(len(self.screening_results)))
        ax1.set_xticklabels(self.screening_results.keys(), fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Synergy Score', fontsize=13, fontweight='bold')
        ax1.set_title('A. Synergy Score Distributions Across Cell Lines', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # B: Kernel density estimation
        ax2 = fig.add_subplot(gs[0, 2])
        for i, (cell_line, df) in enumerate(self.screening_results.items()):
            df['predicted_synergy'].plot(kind='density', ax=ax2, 
                                        color=COLORS['cell_lines'][i],
                                        linewidth=2.5, alpha=0.7, label=cell_line)
        ax2.set_xlabel('Predicted Synergy', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax2.set_title('B. Probability Density', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # C: Box plot with statistical annotations
        ax3 = fig.add_subplot(gs[1, 0])
        box_data = [self.screening_results[cl]['predicted_synergy'].values 
                   for cl in self.screening_results.keys()]
        bp = ax3.boxplot(box_data, labels=list(self.screening_results.keys()),
                        patch_artist=True, notch=True, showfliers=True)

        for i, (patch, color) in enumerate(zip(bp['boxes'], COLORS['cell_lines'])):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

        for whisker in bp['whiskers']:
            whisker.set(linewidth=1.5, linestyle='--')
        for cap in bp['caps']:
            cap.set(linewidth=1.5)
        for median in bp['medians']:
            median.set(color='red', linewidth=2)

        ax3.set_ylabel('Synergy Score', fontsize=12, fontweight='bold')
        ax3.set_title('C. Statistical Distribution', fontsize=14, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # D: Cumulative distribution
        ax4 = fig.add_subplot(gs[1, 1])
        for i, (cell_line, df) in enumerate(self.screening_results.items()):
            sorted_synergy = np.sort(df['predicted_synergy'])
            cumulative = np.arange(1, len(sorted_synergy) + 1) / len(sorted_synergy)
            ax4.plot(sorted_synergy, cumulative, linewidth=2.5, 
                    color=COLORS['cell_lines'][i], label=cell_line, alpha=0.8)

        ax4.set_xlabel('Synergy Score', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax4.set_title('D. Cumulative Distribution', fontsize=14, fontweight='bold', pad=15)
        ax4.legend(frameon=True, fancybox=True)
        ax4.grid(True, alpha=0.3)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        # E: Hexbin plot (all combinations)
        ax5 = fig.add_subplot(gs[1, 2])
        synergy_values = combined_df['predicted_synergy'].values
        indices = np.arange(len(synergy_values))
        hexbin = ax5.hexbin(indices, synergy_values, gridsize=30, cmap='YlOrRd', 
                           mincnt=1, edgecolors='black', linewidths=0.2)
        ax5.set_xlabel('Combination Index', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Synergy Score', fontsize=12, fontweight='bold')
        ax5.set_title('E. Synergy Landscape', fontsize=14, fontweight='bold', pad=15)
        cb = plt.colorbar(hexbin, ax=ax5)
        cb.set_label('Count', fontsize=11, fontweight='bold')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)

        plt.suptitle('Synergy Score Landscape Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(self.figures_dir / 'Figure2_Synergy_Landscape.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        self._log("[OK] Figure 2: Synergy Landscape")

    def plot_top_combinations_comparison(self):
        """
        Figure 3: Top drug combinations comparison across cell lines.

        Horizontal bar charts with annotations and rankings.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(len(self.screening_results), 1, figure=fig, hspace=0.4)

        for i, (cell_line, df) in enumerate(self.screening_results.items()):
            ax = fig.add_subplot(gs[i, 0])

            # Get top 15 combinations
            top_15 = df.nlargest(15, 'predicted_synergy')

            # Create drug pair labels
            drug_pairs = [f"{row['drug1']} + {row['drug2']}" 
                         for _, row in top_15.iterrows()]
            synergies = top_15['predicted_synergy'].values

            # Create gradient colors
            colors_grad = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(synergies)))

            # Horizontal bar chart
            y_pos = np.arange(len(drug_pairs))
            bars = ax.barh(y_pos, synergies, color=colors_grad, 
                          edgecolor='black', linewidth=1.2, alpha=0.85)

            # Add value labels
            for j, (bar, val) in enumerate(zip(bars, synergies)):
                ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

            # Add rank badges
            for j, bar in enumerate(bars):
                rank_color = '#FFD700' if j == 0 else '#C0C0C0' if j == 1 else '#CD7F32' if j == 2 else '#E8E8E8'
                circle = plt.Circle((synergies[j]/2, bar.get_y() + bar.get_height()/2), 
                                   0.15, color=rank_color, ec='black', linewidth=1.5, zorder=10)
                ax.add_patch(circle)
                ax.text(synergies[j]/2, bar.get_y() + bar.get_height()/2, 
                       str(j+1), ha='center', va='center', fontsize=8, 
                       fontweight='bold', zorder=11)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(drug_pairs, fontsize=9)
            ax.set_xlabel('Predicted Synergy Score', fontsize=12, fontweight='bold')
            ax.set_title(f'{chr(65+i)}. Top 15 Drug Combinations for {cell_line}', 
                        fontsize=13, fontweight='bold', pad=10, loc='left')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(0, max(synergies) * 1.15)

        plt.suptitle('Top Drug Combination Rankings Across Cell Lines',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(self.figures_dir / 'Figure3_Top_Combinations.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        self._log("[OK] Figure 3: Top Combinations Comparison")

    def plot_mechanism_analysis(self):
        """
        Figure 4: Drug mechanism-based synergy analysis.

        Analyzes synergy patterns by drug mechanism of action.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Prepare mechanism data
        for cell_line, df in self.screening_results.items():
            # A: Mechanism pair frequency
            ax1 = fig.add_subplot(gs[0, 0])

            # Count mechanism pairs in top combinations
            top_50 = df.nlargest(50, 'predicted_synergy')
            mechanism_pairs = top_50['drug1_mechanism'] + ' × ' + top_50['drug2_mechanism']
            pair_counts = mechanism_pairs.value_counts().head(10)

            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(pair_counts)))
            bars = ax1.barh(range(len(pair_counts)), pair_counts.values, 
                           color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

            ax1.set_yticks(range(len(pair_counts)))
            ax1.set_yticklabels(pair_counts.index, fontsize=9)
            ax1.set_xlabel('Frequency in Top 50', fontsize=12, fontweight='bold')
            ax1.set_title('A. Most Frequent Mechanism Pairs', 
                         fontsize=14, fontweight='bold', pad=15)
            ax1.invert_yaxis()
            ax1.grid(axis='x', alpha=0.3)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, pair_counts.values)):
                ax1.text(count + 0.3, bar.get_y() + bar.get_height()/2, 
                        str(int(count)), va='center', fontsize=10, fontweight='bold')

            # B: Average synergy by mechanism
            ax2 = fig.add_subplot(gs[0, 1])

            # Calculate average synergy for each drug mechanism
            drug1_avg = df.groupby('drug1_mechanism')['predicted_synergy'].mean().sort_values(ascending=False).head(10)
            drug2_avg = df.groupby('drug2_mechanism')['predicted_synergy'].mean().sort_values(ascending=False).head(10)

            x_pos1 = np.arange(len(drug1_avg))
            x_pos2 = x_pos1 + 0.4
            width = 0.35

            bars1 = ax2.bar(x_pos1, drug1_avg.values, width, 
                           label='As Drug 1', color=COLORS['primary'], 
                           edgecolor='black', linewidth=1, alpha=0.8)
            bars2 = ax2.bar(x_pos2, drug2_avg.values, width,
                           label='As Drug 2', color=COLORS['secondary'],
                           edgecolor='black', linewidth=1, alpha=0.8)

            ax2.set_xticks(x_pos1 + width / 2)
            ax2.set_xticklabels(drug1_avg.index, rotation=45, ha='right', fontsize=9)
            ax2.set_ylabel('Average Synergy Score', fontsize=12, fontweight='bold')
            ax2.set_title('B. Average Synergy by Mechanism', 
                         fontsize=14, fontweight='bold', pad=15)
            ax2.legend(frameon=True, fancybox=True, shadow=True)
            ax2.grid(axis='y', alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            # C: Mechanism synergy heatmap (interaction matrix)
            ax3 = fig.add_subplot(gs[1, :])

            # Create mechanism interaction matrix
            mechanisms1 = df['drug1_mechanism'].unique()[:12]
            mechanisms2 = df['drug2_mechanism'].unique()[:12]

            synergy_matrix = np.zeros((len(mechanisms1), len(mechanisms2)))
            count_matrix = np.zeros((len(mechanisms1), len(mechanisms2)))

            for idx, row in df.iterrows():
                if row['drug1_mechanism'] in mechanisms1 and row['drug2_mechanism'] in mechanisms2:
                    i = list(mechanisms1).index(row['drug1_mechanism'])
                    j = list(mechanisms2).index(row['drug2_mechanism'])
                    synergy_matrix[i, j] += row['predicted_synergy']
                    count_matrix[i, j] += 1

            # Average synergy
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_synergy_matrix = np.where(count_matrix > 0, 
                                              synergy_matrix / count_matrix, 0)

            im = ax3.imshow(avg_synergy_matrix, cmap='RdYlGn', aspect='auto', 
                           vmin=0, vmax=avg_synergy_matrix.max())

            ax3.set_xticks(range(len(mechanisms2)))
            ax3.set_yticks(range(len(mechanisms1)))
            ax3.set_xticklabels(mechanisms2, rotation=45, ha='right', fontsize=8)
            ax3.set_yticklabels(mechanisms1, fontsize=8)
            ax3.set_xlabel('Drug 2 Mechanism', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Drug 1 Mechanism', fontsize=12, fontweight='bold')
            ax3.set_title('C. Mechanism Interaction Matrix (Average Synergy)',
                         fontsize=14, fontweight='bold', pad=15)

            # Add text annotations
            for i in range(len(mechanisms1)):
                for j in range(len(mechanisms2)):
                    if count_matrix[i, j] > 0:
                        text_color = 'white' if avg_synergy_matrix[i, j] > avg_synergy_matrix.max()/2 else 'black'
                        ax3.text(j, i, f'{avg_synergy_matrix[i, j]:.2f}',
                                ha='center', va='center', color=text_color, 
                                fontsize=7, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            cbar.set_label('Average Synergy Score', fontsize=11, fontweight='bold')

            break  # Only process first cell line for mechanism analysis

        plt.suptitle('Drug Mechanism Analysis',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(self.figures_dir / 'Figure4_Mechanism_Analysis.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        self._log("[OK] Figure 4: Mechanism Analysis")

    def plot_drug_interaction_heatmaps(self):
        """
        Figure 5: Drug-drug interaction heatmaps for each cell line.

        Shows synergy scores in a matrix format.
        """
        fig = plt.figure(figsize=(20, 6 * len(self.screening_results)))

        for idx, (cell_line, df) in enumerate(self.screening_results.items()):
            ax = plt.subplot(len(self.screening_results), 1, idx + 1)

            # Get unique drugs
            all_drugs = sorted(set(df['drug1'].unique()) | set(df['drug2'].unique()))
            n_drugs = len(all_drugs)

            # Create synergy matrix
            synergy_matrix = np.zeros((n_drugs, n_drugs))

            for _, row in df.iterrows():
                i = all_drugs.index(row['drug1'])
                j = all_drugs.index(row['drug2'])
                synergy_matrix[i, j] = row['predicted_synergy']
                synergy_matrix[j, i] = row['predicted_synergy']  # Symmetric

            # Plot heatmap
            im = ax.imshow(synergy_matrix, cmap='YlOrRd', aspect='auto',
                          interpolation='nearest')

            ax.set_xticks(range(n_drugs))
            ax.set_yticks(range(n_drugs))
            ax.set_xticklabels(all_drugs, rotation=90, fontsize=7)
            ax.set_yticklabels(all_drugs, fontsize=7)
            ax.set_title(f'{chr(65+idx)}. Drug Interaction Heatmap: {cell_line}',
                        fontsize=14, fontweight='bold', pad=15, loc='left')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Synergy Score', fontsize=10, fontweight='bold')

            # Add grid
            ax.set_xticks(np.arange(n_drugs) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_drugs) - 0.5, minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        plt.suptitle('Drug-Drug Interaction Heatmaps',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(self.figures_dir / 'Figure5_Drug_Interaction_Heatmaps.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        self._log("[OK] Figure 5: Drug Interaction Heatmaps")

    def plot_cell_line_comparison(self):
        """
        Figure 6: Comprehensive cell line comparison.

        Multi-panel comparison of synergy profiles across cell lines.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # A: Overlapping top combinations (Venn-style comparison)
        ax1 = fig.add_subplot(gs[0, 0])

        # Get top 20 from each cell line
        top_sets = {}
        for cell_line, df in self.screening_results.items():
            top_20 = df.nlargest(20, 'predicted_synergy')
            combinations = set(top_20['drug1'] + ' + ' + top_20['drug2'])
            top_sets[cell_line] = combinations

        # Calculate overlaps
        cell_lines = list(top_sets.keys())
        overlap_matrix = np.zeros((len(cell_lines), len(cell_lines)))

        for i, cl1 in enumerate(cell_lines):
            for j, cl2 in enumerate(cell_lines):
                overlap = len(top_sets[cl1] & top_sets[cl2])
                overlap_matrix[i, j] = overlap

        im1 = ax1.imshow(overlap_matrix, cmap='Blues', aspect='auto')
        ax1.set_xticks(range(len(cell_lines)))
        ax1.set_yticks(range(len(cell_lines)))
        ax1.set_xticklabels(cell_lines, fontsize=11, fontweight='bold')
        ax1.set_yticklabels(cell_lines, fontsize=11, fontweight='bold')
        ax1.set_title('A. Top 20 Combination Overlap',
                     fontsize=14, fontweight='bold', pad=15)

        # Add text annotations
        for i in range(len(cell_lines)):
            for j in range(len(cell_lines)):
                text_color = 'white' if overlap_matrix[i, j] > overlap_matrix.max()/2 else 'black'
                ax1.text(j, i, f'{int(overlap_matrix[i, j])}',
                        ha='center', va='center', color=text_color,
                        fontsize=12, fontweight='bold')

        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Overlap Count', fontsize=10, fontweight='bold')

        # B: Statistical comparison
        ax2 = fig.add_subplot(gs[0, 1])

        stats_data = []
        for cell_line, df in self.screening_results.items():
            stats_data.append({
                'Cell Line': cell_line,
                'Mean': df['predicted_synergy'].mean(),
                'Median': df['predicted_synergy'].median(),
                'Std': df['predicted_synergy'].std(),
                'Max': df['predicted_synergy'].max(),
                'Q75': df['predicted_synergy'].quantile(0.75)
            })

        stats_df = pd.DataFrame(stats_data)

        x = np.arange(len(cell_lines))
        width = 0.15

        bars1 = ax2.bar(x - 2*width, stats_df['Mean'], width, label='Mean',
                       color=COLORS['primary'], alpha=0.8, edgecolor='black')
        bars2 = ax2.bar(x - width, stats_df['Median'], width, label='Median',
                       color=COLORS['secondary'], alpha=0.8, edgecolor='black')
        bars3 = ax2.bar(x, stats_df['Max'], width, label='Max',
                       color=COLORS['accent'], alpha=0.8, edgecolor='black')
        bars4 = ax2.bar(x + width, stats_df['Q75'], width, label='Q75',
                       color=COLORS['success'], alpha=0.8, edgecolor='black')

        ax2.set_xticks(x)
        ax2.set_xticklabels(cell_lines, fontsize=11, fontweight='bold')
        ax2.set_ylabel('Synergy Score', fontsize=12, fontweight='bold')
        ax2.set_title('B. Statistical Metrics Comparison',
                     fontsize=14, fontweight='bold', pad=15)
        ax2.legend(frameon=True, fancybox=True, shadow=True, ncol=2)
        ax2.grid(axis='y', alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # C: Range and variability
        ax3 = fig.add_subplot(gs[1, 0])

        for i, (cell_line, df) in enumerate(self.screening_results.items()):
            synergies = df['predicted_synergy'].values
            mean_val = np.mean(synergies)
            std_val = np.std(synergies)

            ax3.errorbar(i, mean_val, yerr=std_val, fmt='o', markersize=12,
                        capsize=8, capthick=2, elinewidth=2,
                        color=COLORS['cell_lines'][i], 
                        markeredgecolor='black', markeredgewidth=1.5,
                        label=cell_line)

        ax3.set_xticks(range(len(cell_lines)))
        ax3.set_xticklabels(cell_lines, fontsize=11, fontweight='bold')
        ax3.set_ylabel('Synergy Score (Mean ± SD)', fontsize=12, fontweight='bold')
        ax3.set_title('C. Mean Synergy with Standard Deviation',
                     fontsize=14, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # D: Top drug frequency comparison
        ax4 = fig.add_subplot(gs[1, 1])

        # Count top drug appearances
        drug_counts = {}
        for cell_line, df in self.screening_results.items():
            top_30 = df.nlargest(30, 'predicted_synergy')
            for drug in pd.concat([top_30['drug1'], top_30['drug2']]):
                if drug not in drug_counts:
                    drug_counts[drug] = {cl: 0 for cl in cell_lines}
                drug_counts[drug][cell_line] += 1

        # Get top 10 most frequent drugs
        top_drugs = sorted(drug_counts.items(), 
                          key=lambda x: sum(x[1].values()), reverse=True)[:10]

        drug_names = [d[0] for d in top_drugs]
        drug_data = np.array([[d[1][cl] for cl in cell_lines] for d in top_drugs])

        x = np.arange(len(drug_names))
        width = 0.25

        for i, cell_line in enumerate(cell_lines):
            ax4.bar(x + i * width, drug_data[:, i], width, 
                   label=cell_line, color=COLORS['cell_lines'][i],
                   alpha=0.8, edgecolor='black', linewidth=0.8)

        ax4.set_xticks(x + width)
        ax4.set_xticklabels(drug_names, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('Frequency in Top 30', fontsize=12, fontweight='bold')
        ax4.set_title('D. Most Frequent Drugs in Top Combinations',
                     fontsize=14, fontweight='bold', pad=15)
        ax4.legend(frameon=True, fancybox=True, shadow=True)
        ax4.grid(axis='y', alpha=0.3)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        plt.suptitle('Cell Line Synergy Profile Comparison',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(self.figures_dir / 'Figure6_Cell_Line_Comparison.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        self._log("[OK] Figure 6: Cell Line Comparison")

    def plot_distribution_analysis(self):
        """
        Figure 7: Statistical distribution analysis.

        Detailed analysis of synergy score distributions.
        """
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

        for idx, (cell_line, df) in enumerate(self.screening_results.items()):
            # Histogram with fit
            ax = fig.add_subplot(gs[idx // 3, idx % 3])

            synergies = df['predicted_synergy'].values

            # Plot histogram
            n, bins, patches = ax.hist(synergies, bins=40, density=True,
                                       color=COLORS['cell_lines'][idx],
                                       alpha=0.7, edgecolor='black', linewidth=0.8)

            # Fit and plot normal distribution
            from scipy import stats
            mu, sigma = stats.norm.fit(synergies)
            x_fit = np.linspace(synergies.min(), synergies.max(), 100)
            ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sigma),
                   'r-', linewidth=2.5, label=f'Normal Fit\nμ={mu:.3f}, σ={sigma:.3f}')

            # Add percentile lines
            percentiles = [25, 50, 75, 90]
            colors_p = ['blue', 'green', 'orange', 'red']
            for p, c in zip(percentiles, colors_p):
                val = np.percentile(synergies, p)
                ax.axvline(val, color=c, linestyle='--', linewidth=2, alpha=0.7,
                          label=f'P{p}: {val:.3f}')

            ax.set_xlabel('Synergy Score', fontsize=11, fontweight='bold')
            ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
            ax.set_title(f'{chr(65+idx)}. Distribution: {cell_line}',
                        fontsize=13, fontweight='bold', pad=12)
            ax.legend(fontsize=8, frameon=True, fancybox=True)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Combined Q-Q plot
        ax_qq = fig.add_subplot(gs[1, :])

        for idx, (cell_line, df) in enumerate(self.screening_results.items()):
            synergies = df['predicted_synergy'].values
            stats.probplot(synergies, dist="norm", plot=ax_qq)
            ax_qq.get_lines()[-2].set_color(COLORS['cell_lines'][idx])
            ax_qq.get_lines()[-2].set_marker('o')
            ax_qq.get_lines()[-2].set_markersize(4)
            ax_qq.get_lines()[-2].set_alpha(0.6)
            ax_qq.get_lines()[-2].set_label(cell_line)
            ax_qq.get_lines()[-1].set_visible(False)  # Hide fit line

        ax_qq.set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
        ax_qq.set_ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
        ax_qq.set_title('D. Q-Q Plot (Normality Test)',
                       fontsize=14, fontweight='bold', pad=15)
        ax_qq.legend(frameon=True, fancybox=True, shadow=True)
        ax_qq.grid(True, alpha=0.3)
        ax_qq.spines['top'].set_visible(False)
        ax_qq.spines['right'].set_visible(False)

        plt.suptitle('Synergy Score Distribution Analysis',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(self.figures_dir / 'Figure7_Distribution_Analysis.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        self._log("[OK] Figure 7: Distribution Analysis")

    def plot_drug_network(self):
        """
        Figure 8: Drug interaction network visualization.

        Network graph showing synergistic relationships.
        """
        try:
            import networkx as nx
        except ImportError:
            self._log("[WARNING] networkx not installed, skipping network plot")
            self._log("          Install with: pip install networkx")
            return

        fig = plt.figure(figsize=(20, 12))

        for idx, (cell_line, df) in enumerate(self.screening_results.items()):
            ax = plt.subplot(1, len(self.screening_results), idx + 1)

            # Create network from top combinations
            G = nx.Graph()
            top_100 = df.nlargest(100, 'predicted_synergy')

            # Add edges with synergy as weight
            for _, row in top_100.iterrows():
                G.add_edge(row['drug1'], row['drug2'], 
                          weight=row['predicted_synergy'])

            # Calculate node sizes based on degree centrality
            degree_cent = nx.degree_centrality(G)
            node_sizes = [3000 * degree_cent[node] for node in G.nodes()]

            # Calculate edge widths based on synergy
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights)
            edge_widths = [5 * (w / max_weight) for w in weights]

            # Use spring layout
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

            # Draw network
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                  node_color=COLORS['cell_lines'][idx],
                                  alpha=0.7, edgecolors='black',
                                  linewidths=2, ax=ax)

            nx.draw_networkx_edges(G, pos, width=edge_widths, 
                                  alpha=0.3, edge_color='gray', ax=ax)

            nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold',
                                   font_family='Arial', ax=ax)

            ax.set_title(f'{chr(65+idx)}. Drug Synergy Network: {cell_line}',
                        fontsize=14, fontweight='bold', pad=15)
            ax.axis('off')

            # Add legend
            legend_elements = [
                mpatches.Patch(facecolor=COLORS['cell_lines'][idx], 
                             edgecolor='black', label='Drug Node'),
                plt.Line2D([0], [0], color='gray', linewidth=2, 
                          label='Synergy Edge')
            ]
            ax.legend(handles=legend_elements, loc='upper left', 
                     frameon=True, fancybox=True, shadow=True)

        plt.suptitle('Drug Synergy Interaction Networks (Top 100 Combinations)',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(self.figures_dir / 'Figure8_Drug_Network.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        self._log("[OK] Figure 8: Drug Network")

    def plot_statistical_summary(self):
        """
        Figure 9: Comprehensive statistical summary.

        Panel showing key statistics and findings.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

        # A: Summary statistics table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')

        summary_data = []
        for cell_line, df in self.screening_results.items():
            synergies = df['predicted_synergy']
            summary_data.append([
                cell_line,
                f"{synergies.mean():.4f}",
                f"{synergies.median():.4f}",
                f"{synergies.std():.4f}",
                f"{synergies.min():.4f}",
                f"{synergies.max():.4f}",
                f"{synergies.quantile(0.25):.4f}",
                f"{synergies.quantile(0.75):.4f}",
                f"{len(df)}"
            ])

        columns = ['Cell Line', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75', 'N']
        table = ax1.table(cellText=summary_data, colLabels=columns,
                         cellLoc='center', loc='center',
                         colColours=[COLORS['primary']] * len(columns),
                         colWidths=[0.12] * len(columns))

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor(COLORS['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(summary_data) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(1)

        ax1.set_title('A. Summary Statistics Across Cell Lines',
                     fontsize=14, fontweight='bold', pad=20)

        # B: Correlation matrix (if multiple cell lines)
        if len(self.screening_results) > 1:
            ax2 = fig.add_subplot(gs[1, :2])

            # Create correlation matrix of synergy scores
            cell_lines = list(self.screening_results.keys())
            corr_matrix = np.zeros((len(cell_lines), len(cell_lines)))

            for i, cl1 in enumerate(cell_lines):
                for j, cl2 in enumerate(cell_lines):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        # Match common drug pairs
                        df1 = self.screening_results[cl1]
                        df2 = self.screening_results[cl2]

                        # Create comparison key
                        df1['pair'] = df1['drug1'] + '_' + df1['drug2']
                        df2['pair'] = df2['drug1'] + '_' + df2['drug2']

                        merged = pd.merge(df1[['pair', 'predicted_synergy']], 
                                        df2[['pair', 'predicted_synergy']], 
                                        on='pair', suffixes=('_1', '_2'))

                        if len(merged) > 10:
                            corr = merged['predicted_synergy_1'].corr(merged['predicted_synergy_2'])
                            corr_matrix[i, j] = corr

            im2 = ax2.imshow(corr_matrix, cmap='coolwarm', aspect='auto', 
                           vmin=-1, vmax=1)
            ax2.set_xticks(range(len(cell_lines)))
            ax2.set_yticks(range(len(cell_lines)))
            ax2.set_xticklabels(cell_lines, fontsize=12, fontweight='bold')
            ax2.set_yticklabels(cell_lines, fontsize=12, fontweight='bold')
            ax2.set_title('B. Synergy Profile Correlation',
                         fontsize=14, fontweight='bold', pad=15)

            # Add correlation values
            for i in range(len(cell_lines)):
                for j in range(len(cell_lines)):
                    text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                    ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                            ha='center', va='center', color=text_color,
                            fontsize=11, fontweight='bold')

            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Correlation Coefficient', fontsize=11, fontweight='bold')

        # C: Top drug frequency
        ax3 = fig.add_subplot(gs[1, 2])

        all_drugs = {}
        for cell_line, df in self.screening_results.items():
            top_50 = df.nlargest(50, 'predicted_synergy')
            for drug in pd.concat([top_50['drug1'], top_50['drug2']]):
                all_drugs[drug] = all_drugs.get(drug, 0) + 1

        top_drugs = sorted(all_drugs.items(), key=lambda x: x[1], reverse=True)[:15]

        drugs, counts = zip(*top_drugs)
        colors_grad = plt.cm.viridis(np.linspace(0.2, 0.9, len(drugs)))

        ax3.barh(range(len(drugs)), counts, color=colors_grad,
                edgecolor='black', linewidth=1, alpha=0.8)
        ax3.set_yticks(range(len(drugs)))
        ax3.set_yticklabels(drugs, fontsize=9)
        ax3.set_xlabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('C. Most Frequent Drugs\n(Top 50 per Cell Line)',
                     fontsize=13, fontweight='bold', pad=12)
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)

        # D-F: Individual cell line highlights
        for idx, (cell_line, df) in enumerate(self.screening_results.items()):
            ax = fig.add_subplot(gs[2, idx])

            # Get top 10
            top_10 = df.nlargest(10, 'predicted_synergy')

            # Create word cloud-style visualization
            y_pos = np.arange(len(top_10))
            synergies = top_10['predicted_synergy'].values

            bars = ax.barh(y_pos, synergies,
                          color=COLORS['cell_lines'][idx],
                          alpha=0.7, edgecolor='black', linewidth=1)

            labels = [f"{row['drug1'][:8]}+{row['drug2'][:8]}" 
                     for _, row in top_10.iterrows()]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel('Synergy', fontsize=10, fontweight='bold')
            ax.set_title(f'{chr(68+idx)}. Top 10: {cell_line}',
                        fontsize=12, fontweight='bold', pad=10)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)

        plt.suptitle('Comprehensive Statistical Summary',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(self.figures_dir / 'Figure9_Statistical_Summary.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        self._log("[OK] Figure 9: Statistical Summary")


# Integration function to add to your main pipeline
def create_publication_figures(results_dir, figures_dir, training_history=None, logger=None):
    """
    Create all publication-quality figures.

    Usage:
        from visualization_publication import create_publication_figures

        create_publication_figures(
            results_dir='cancer_synergy_pipeline/results',
            figures_dir='cancer_synergy_pipeline/figures',
            training_history=history,  # Optional
            logger=logger  # Optional
        )

    Args:
        results_dir: Path to results directory with CSV files
        figures_dir: Path to save figures
        training_history: Dictionary with training metrics (optional)
        logger: Logger instance (optional)
    """
    visualizer = PublicationVisualizer(results_dir, figures_dir, logger)
    visualizer.create_all_figures(training_history)

    return visualizer


if __name__ == "__main__":
    # Example standalone usage
    import sys
    from pathlib import Path

    # Default paths
    results_dir = Path("cancer_synergy_pipeline/results")
    figures_dir = Path("cancer_synergy_pipeline/figures")

    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        figures_dir = Path(sys.argv[2])

    print("\n" + "="*80)
    print("PUBLICATION-QUALITY FIGURE GENERATOR")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Figures directory: {figures_dir}")
    print("="*80 + "\n")

    create_publication_figures(results_dir, figures_dir)

    print("\n" + "="*80)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*80 + "\n")
