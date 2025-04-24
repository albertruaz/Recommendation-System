"""
Hyperparameter Tuning Visualization Script
Focusing on Huber Loss Analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Directory setup
TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # Current test directory
OUTPUT_DIR = os.path.join(os.path.dirname(TEST_DIR), "output")  # Parent dir/output
VISUALIZATION_DIR = os.path.join(TEST_DIR, f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# Create visualization directory
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def load_results(file_path):
    """Load tuning results from CSV file"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        results_df = pd.read_csv(file_path)
        print(f"Loaded {len(results_df)} results from CSV file.")
        
        # Check data columns
        print("\nAvailable columns:")
        for col in results_df.columns:
            print(f"- {col}: {results_df[col].dtype}")
        
        return results_df
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None

def plot_huber_loss_heatmap(results_df, output_dir):
    """Create heatmaps showing Huber Loss for different hyperparameter combinations"""
    if results_df is None or results_df.empty:
        print("No data available for visualization.")
        return
    
    # Check if the required columns exist
    if 'test_huber_loss' not in results_df.columns:
        print("Error: test_huber_loss column not found in data.")
        return
    
    # Group by rank values (if available)
    if 'rank' in results_df.columns:
        ranks = sorted(results_df['rank'].unique())
    else:
        ranks = [None]
    
    for rank in ranks:
        # Filter data for current rank
        if rank is not None:
            df_rank = results_df[results_df['rank'] == rank].copy()
            title_prefix = f"Rank={rank}: "
        else:
            df_rank = results_df.copy()
            title_prefix = ""
        
        if len(df_rank) <= 1:
            continue
            
        # Check for max_iter and reg_param columns
        if 'max_iter' not in df_rank.columns or 'reg_param' not in df_rank.columns:
            print("Error: Required columns (max_iter, reg_param) not found.")
            continue
        
        try:
            # Create pivot table for heatmap
            pivot = df_rank.pivot_table(
                index='max_iter', 
                columns='reg_param', 
                values='test_huber_loss',
                aggfunc='mean'
            )
            
            # Skip if too many missing values
            if pivot.isna().sum().sum() > pivot.size * 0.5:
                print(f"Too many missing values for rank={rank}. Skipping...")
                continue
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            
            # Use viridis_r colormap (reversed viridis) for better visibility with lower values being darker
            sns.heatmap(pivot, annot=True, 
                      cmap='viridis_r',  # Dark blue for lower values (better)
                      fmt='.4f', linewidths=.5)
            
            plt.title(f"{title_prefix}Test Huber Loss (Lower is Better)", fontsize=16)
            plt.xlabel("Regularization Parameter", fontsize=14)
            plt.ylabel("Max Iterations", fontsize=14)
            
            # Save the plot
            output_path = os.path.join(output_dir, f"huber_loss_heatmap_rank{rank}.png" if rank is not None 
                                     else "huber_loss_heatmap.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Huber Loss heatmap saved: {output_path}")
            plt.close()
            
            # Additional visualization: 3D surface plot for better insights
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get x, y, z data
            x = pivot.columns.astype(float)
            y = pivot.index.astype(float)
            X, Y = np.meshgrid(x, y)
            Z = pivot.values
            
            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis_r', 
                                  linewidth=0, antialiased=True, alpha=0.8)
            
            # Add color bar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            # Set labels
            ax.set_xlabel("Regularization Parameter", fontsize=14)
            ax.set_ylabel("Max Iterations", fontsize=14)
            ax.set_zlabel("Huber Loss", fontsize=14)
            ax.set_title(f"{title_prefix}Test Huber Loss 3D Surface", fontsize=16)
            
            # Save the plot
            output_path = os.path.join(output_dir, f"huber_loss_3d_rank{rank}.png" if rank is not None 
                                     else "huber_loss_3d.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Huber Loss 3D surface plot saved: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"Error creating heatmap for rank={rank}: {str(e)}")

def plot_parameter_impact(results_df, output_dir):
    """Analyze the impact of each hyperparameter on Huber Loss"""
    if results_df is None or results_df.empty or 'test_huber_loss' not in results_df.columns:
        return
    
    for param in ['max_iter', 'reg_param', 'rank']:
        if param not in results_df.columns:
            continue
            
        param_values = sorted(results_df[param].unique())
        
        if len(param_values) <= 1:
            continue
        
        # Calculate mean and std of Huber Loss for each parameter value
        means = []
        errors = []
        
        for val in param_values:
            subset = results_df[results_df[param] == val]
            mean = subset['test_huber_loss'].mean()
            std = subset['test_huber_loss'].std() if len(subset) > 1 else 0
            means.append(mean)
            errors.append(std)
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(param_values)), means, yerr=errors, capsize=5, 
                      color='mediumseagreen', edgecolor='darkgreen', alpha=0.7)
        
        # Add value annotations
        for i, (mean, std) in enumerate(zip(means, errors)):
            if not np.isnan(mean):
                plt.text(i, mean + std + (max([m for m in means if not np.isnan(m)]) * 0.02), 
                        f'{mean:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Format the plot
        plt.xticks(range(len(param_values)), [str(x) for x in param_values])
        plt.xlabel(param.replace('_', ' ').title(), fontsize=14)
        plt.ylabel('Test Huber Loss', fontsize=14)
        
        param_name = param.replace('_', ' ').title()
        plt.title(f'Impact of {param_name} on Huber Loss (Lower is Better)', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Save the plot
        output_path = os.path.join(output_dir, f"{param}_vs_huber_loss.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Parameter impact plot saved: {output_path}")
        plt.close()
        
        # Add a line plot for trend visualization
        plt.figure(figsize=(12, 6))
        plt.errorbar(param_values, means, yerr=errors, marker='o', color='darkblue', 
                   ecolor='lightblue', elinewidth=2, capsize=5, linewidth=2)
        
        plt.xlabel(param.replace('_', ' ').title(), fontsize=14)
        plt.ylabel('Test Huber Loss', fontsize=14)
        plt.title(f'Trend of {param_name} Impact on Huber Loss', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Save the trend plot
        output_path = os.path.join(output_dir, f"{param}_trend_huber_loss.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Parameter trend plot saved: {output_path}")
        plt.close()

def find_best_parameters(results_df):
    """Find optimal hyperparameter combinations for Huber Loss"""
    if results_df is None or results_df.empty:
        return
    
    print("\n======= OPTIMAL HYPERPARAMETERS =======")
    
    if 'test_huber_loss' in results_df.columns:
        # Find the combination with lowest Huber Loss
        best_idx = results_df['test_huber_loss'].fillna(float('inf')).idxmin()
        best_value = results_df.loc[best_idx, 'test_huber_loss']
        
        print(f"\nBest Test Huber Loss: {best_value:.6f}")
        
        # Print hyperparameters for the best model
        for col in ['max_iter', 'reg_param', 'rank']:
            if col in results_df.columns:
                print(f"{col.replace('_', ' ').title()}: {results_df.loc[best_idx, col]}")
        
        # Create a summary table with top N best configurations
        top_n = 5
        top_configs = results_df.sort_values('test_huber_loss').head(top_n)
        
        print(f"\nTop {top_n} Configurations:")
        summary_columns = ['max_iter', 'reg_param', 'rank', 'test_huber_loss']
        summary_columns = [col for col in summary_columns if col in top_configs.columns]
        
        # Create and save the table as a CSV
        summary_table = top_configs[summary_columns].copy()
        
        for col in summary_table.columns:
            if 'loss' in col or 'rmse' in col:
                summary_table[col] = summary_table[col].map('{:.6f}'.format)
        
        summary_path = os.path.join(VISUALIZATION_DIR, "best_configurations.csv")
        summary_table.to_csv(summary_path, index=False)
        print(f"Best configurations saved to: {summary_path}")
        
        # Also print the table
        print("\n" + summary_table.to_string(index=False))

def create_sorted_hyperparameter_table(results_df, output_dir):
    """
    Create a table showing all hyperparameter combinations sorted by Huber Loss
    Visualize hyperparameters as heights instead of colors and show both Huber Loss and RMSE
    """
    if results_df is None or results_df.empty:
        print("No data available for visualization.")
        return
    
    # Check if the required columns exist
    if 'test_huber_loss' not in results_df.columns:
        print("Error: test_huber_loss column not found in data.")
        return
    
    # Check for RMSE column, use only if available
    has_rmse = 'test_rmse' in results_df.columns
    
    # Check for hyperparameter columns
    required_params = ['max_iter', 'reg_param', 'rank']
    missing_params = [param for param in required_params if param not in results_df.columns]
    if missing_params:
        print(f"Error: Required hyperparameter columns {missing_params} not found.")
        return
    
    try:
        # Sort the DataFrame by test_huber_loss value (ascending)
        sorted_df = results_df.sort_values('test_huber_loss').reset_index(drop=True)
        
        # Select relevant columns
        display_cols = ['max_iter', 'reg_param', 'rank', 'test_huber_loss']
        if has_rmse:
            display_cols.append('test_rmse')
        sorted_df = sorted_df[display_cols].copy()
        
        # Create a figure with subplots - more compact layout
        fig = plt.figure(figsize=(20, 12))
        
        # Create a GridSpec layout with no extra space
        rows = 5 if has_rmse else 4
        gs = plt.GridSpec(rows, 1, height_ratios=[2, 1, 2, 2, 2] if has_rmse else [2, 1, 2, 2])
        
        # Plot each hyperparameter as a bar chart
        param_axs = []
        for i, param in enumerate(['max_iter', 'reg_param', 'rank']):
            ax = fig.add_subplot(gs[i])
            param_axs.append(ax)
            
            # Plot bars with height representing the parameter values
            bars = ax.bar(range(len(sorted_df)), sorted_df[param], alpha=0.7)
            
            # Add value annotations (only if not too many bars)
            if len(sorted_df) <= 36:  # Only add text if not too crowded
                for idx, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height) if param != "reg_param" else height:.3f}', 
                           ha='center', va='bottom', fontsize=8)
            
            # Set axis labels
            ax.set_ylabel(param.replace('_', ' ').title(), fontsize=12)
            ax.set_xticks([])
            ax.set_title(f'{param.replace("_", " ").title()} Values', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Reduce whitespace by setting ylim
            if param == 'reg_param':
                buffer = max(sorted_df[param]) * 0.1
                ax.set_ylim(0, max(sorted_df[param]) + buffer)
        
        # Plot the Huber Loss values
        ax_loss = fig.add_subplot(gs[3])
        bars = ax_loss.bar(range(len(sorted_df)), sorted_df['test_huber_loss'], 
                   color='tab:blue', alpha=0.7)
                   
        # Add value annotations (only if not too many bars)
        if len(sorted_df) <= 36:
            for idx, bar in enumerate(bars):
                height = bar.get_height()
                ax_loss.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax_loss.set_ylabel('Huber Loss', fontsize=12)
        ax_loss.set_title('Huber Loss (Ascending Order)', fontsize=12)
        ax_loss.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add RMSE plot if available
        if has_rmse:
            ax_rmse = fig.add_subplot(gs[4])
            rmse_sorted = sorted_df.sort_values('test_rmse').reset_index(drop=True)
            bars = ax_rmse.bar(range(len(rmse_sorted)), rmse_sorted['test_rmse'], 
                       color='tab:red', alpha=0.7)
            
            # Add value annotations (only if not too many bars)
            if len(sorted_df) <= 36:
                for idx, bar in enumerate(bars):
                    height = bar.get_height()
                    ax_rmse.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
                
            ax_rmse.set_ylabel('RMSE', fontsize=12)
            ax_rmse.set_xlabel('Hyperparameter Combination (Sorted by RMSE)', fontsize=12)
            ax_rmse.set_title('RMSE (Ascending Order)', fontsize=12)
            ax_rmse.grid(axis='y', linestyle='--', alpha=0.7)
            ax_rmse.set_xticks(range(len(rmse_sorted)))
            ax_rmse.set_xticklabels([str(i+1) for i in range(len(rmse_sorted))],
                                  rotation=90 if len(rmse_sorted) > 20 else 0)
            
            # Twin axes for RMSE are removed to simplify the plot
        
        # Add labels for Huber Loss plot
        ax_loss.set_xlabel('Hyperparameter Combination (Sorted by Huber Loss)' if not has_rmse else '', fontsize=12)
        ax_loss.set_xticks(range(len(sorted_df)))
        ax_loss.set_xticklabels([str(i+1) for i in range(len(sorted_df))],
                               rotation=90 if len(sorted_df) > 20 else 0)
        
        # Adjust layout to minimize whitespace
        plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.1, left=0.1, right=0.95)
        
        # Save the figure
        output_path = os.path.join(output_dir, "hyperparameter_combinations_3d.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Hyperparameter combinations visualization saved: {output_path}")
        plt.close()
        
        # Save detailed table as CSV
        csv_path = os.path.join(output_dir, "hyperparameter_combinations_sorted.csv")
        sorted_df.to_csv(csv_path, index=False)
        print(f"Hyperparameter combinations data saved to: {csv_path}")
        
        # Also create a separate visualization for RMSE if available
        if has_rmse:
            # Create a separate figure for RMSE-sorted combinations
            fig = plt.figure(figsize=(20, 12))
            gs = plt.GridSpec(4, 1, height_ratios=[2, 1, 2, 2])
            
            # Plot each hyperparameter as a bar chart
            for i, param in enumerate(['max_iter', 'reg_param', 'rank']):
                ax = fig.add_subplot(gs[i])
                
                # Plot bars with height representing the parameter values
                bars = ax.bar(range(len(rmse_sorted)), rmse_sorted[param], alpha=0.7, color='tab:red')
                
                # Add value annotations (only if not too many bars)
                if len(sorted_df) <= 36:
                    for idx, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height) if param != "reg_param" else height:.3f}', 
                               ha='center', va='bottom', fontsize=8)
                
                # Set axis labels
                ax.set_ylabel(param.replace('_', ' ').title(), fontsize=12)
                ax.set_xticks([])
                ax.set_title(f'{param.replace("_", " ").title()} Values (RMSE Sort)', fontsize=12)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Reduce whitespace by setting ylim
                if param == 'reg_param':
                    buffer = max(rmse_sorted[param]) * 0.1
                    ax.set_ylim(0, max(rmse_sorted[param]) + buffer)
            
            # Plot the RMSE values
            ax_rmse = fig.add_subplot(gs[3])
            bars = ax_rmse.bar(range(len(rmse_sorted)), rmse_sorted['test_rmse'], 
                       color='tab:red', alpha=0.7)
                       
            # Add value annotations (only if not too many bars)
            if len(sorted_df) <= 36:
                for idx, bar in enumerate(bars):
                    height = bar.get_height()
                    ax_rmse.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
            
            ax_rmse.set_ylabel('RMSE', fontsize=12)
            ax_rmse.set_xlabel('Hyperparameter Combination (Sorted by RMSE)', fontsize=12)
            ax_rmse.set_title('RMSE (Ascending Order)', fontsize=12)
            ax_rmse.grid(axis='y', linestyle='--', alpha=0.7)
            ax_rmse.set_xticks(range(len(rmse_sorted)))
            ax_rmse.set_xticklabels([str(i+1) for i in range(len(rmse_sorted))],
                                  rotation=90 if len(rmse_sorted) > 20 else 0)
            
            # Adjust layout to minimize whitespace
            plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.1, left=0.1, right=0.95)
            
            # Save the figure
            output_path = os.path.join(output_dir, "hyperparameter_combinations_rmse.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"RMSE-sorted hyperparameter combinations visualization saved: {output_path}")
            plt.close()
        
    except Exception as e:
        print(f"Error creating hyperparameter visualization: {str(e)}")

def main():
    """Main execution function"""
    print("\n=== Hyperparameter Tuning Visualization (Huber Loss & RMSE Focus) ===")
    
    # Target file settings
    target_file = 'wandb_export_2025-04-24T14_00_10.158+09_00.csv'
    
    # Check file location
    if os.path.exists(os.path.join(OUTPUT_DIR, target_file)):
        file_path = os.path.join(OUTPUT_DIR, target_file)
    elif os.path.exists(target_file):
        file_path = target_file
    else:
        print(f"Target file not found: {target_file}")
        # Try to find any CSV file in output dir
        csv_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
        if csv_files:
            file_path = os.path.join(OUTPUT_DIR, csv_files[0])
            print(f"Using alternative file: {csv_files[0]}")
        else:
            file_path = input("Enter the CSV file path: ")
    
    # Load results
    results_df = load_results(file_path)
    
    if results_df is None:
        print("Failed to load results. Exiting.")
        return
    
    # Add trial_id if missing
    if 'trial_id' not in results_df.columns:
        results_df['trial_id'] = range(len(results_df))
    
    # Convert columns to numeric
    numeric_cols = ['max_iter', 'reg_param', 'rank', 
                  'train_rmse', 'test_rmse', 'train_huber_loss', 'test_huber_loss',
                  'train_time', 'total_time']
    
    # Only process columns that exist
    numeric_cols = [col for col in numeric_cols if col in results_df.columns]
    
    # Convert to numeric
    for col in numeric_cols:
        try:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        except:
            print(f"Could not convert column '{col}' to numeric type.")
    
    print("\n=== Creating Height-based Hyperparameter Visualization ===")
    
    # Only run the sorted hyperparameter visualization
    create_sorted_hyperparameter_table(results_df, VISUALIZATION_DIR)
    
    print(f"\nAll visualizations saved to: {VISUALIZATION_DIR}")
    
    # Save processed data
    processed_path = os.path.join(VISUALIZATION_DIR, "processed_data.csv")
    results_df.to_csv(processed_path, index=False)
    print(f"Processed data saved to: {processed_path}")

if __name__ == "__main__":
    main() 