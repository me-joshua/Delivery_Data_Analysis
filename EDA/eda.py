import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, normaltest, skew, kurtosis
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CourierDataAnalyzer:
    def __init__(self, data_path):
        self.data = self.load_and_preprocess_data(data_path)
        self.create_output_directory()
        
    def create_output_directory(self):
        self.output_dir = "courier_analysis_plots"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_and_preprocess_data(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(how='all')
        
        # Clean status data - remove empty/null values
        df = df[df['latest_status'].notna()]
        df = df[df['latest_status'].str.strip() != '']
        df['latest_status'] = df['latest_status'].str.strip().str.upper()
        
        date_columns = [
            'Order_date', 'Manifest Date', 'pickup_date', 
            '1st scan date at orgin courier dc', 'Last scan date destination dc',
            'Last Undelivered reason status Date', '1st_attempt_date', 
            '2nd_attempt_date', '3rd_attempt_date', 'last_ofd_date',
            'delivery_date', 'rto_initiated_date', 'rto_intransit_date',
            'rto_delivered_date', 'last_scan_date'
        ]
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        df['delivery_time_hours'] = self.calculate_delivery_time(df)
        df['is_delivered'] = df['latest_status'] == 'DELIVERED'
        df['has_undelivered_reason'] = df['last_undelivered_reason'].notna()
        df['delivery_attempts'] = df['no_of_attempt'].fillna(0)
        
        return df
    
    def calculate_delivery_time(self, df):
        delivery_time = []
        for idx, row in df.iterrows():
            if pd.notna(row['Order_date']) and pd.notna(row['delivery_date']):
                time_diff = (row['delivery_date'] - row['Order_date']).total_seconds() / 3600
                delivery_time.append(time_diff)
            else:
                delivery_time.append(np.nan)
        return delivery_time
    
    def generate_data_quality_report(self):
        print("="*80)
        print("COURIER DATA QUALITY & OVERVIEW REPORT")
        print("="*80)
        
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Total Records: {len(self.data):,}")
        print(f"Total Columns: {len(self.data.columns)}")
        print(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n" + "="*50)
        print("MISSING DATA ANALYSIS")
        print("="*50)
        
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        print("\n" + "="*50)
        print("DATA SPARSITY ASSESSMENT")
        print("="*50)
        
        total_cells = self.data.shape[0] * self.data.shape[1]
        missing_cells = self.data.isnull().sum().sum()
        sparsity = (missing_cells / total_cells) * 100
        
        print(f"Total Cells: {total_cells:,}")
        print(f"Missing Cells: {missing_cells:,}")
        print(f"Data Sparsity: {sparsity:.2f}%")
        
        if sparsity < 10:
            print("Data Quality: EXCELLENT - Very low sparsity")
        elif sparsity < 25:
            print("Data Quality: GOOD - Moderate sparsity")
        elif sparsity < 50:
            print("Data Quality: FAIR - High sparsity, needs attention")
        else:
            print("Data Quality: POOR - Very high sparsity, critical issues")
        
        return missing_df
    
    def analyze_delivery_performance(self):
        print("\n" + "="*50)
        print("DELIVERY PERFORMANCE ANALYSIS")
        print("="*50)
        
        delivered_orders = self.data[self.data['latest_status'] == 'DELIVERED']
        delivery_success_rate = (len(delivered_orders) / len(self.data)) * 100
        
        print(f"Overall Delivery Success Rate: {delivery_success_rate:.2f}%")
        
        # Clean status distribution - remove any empty or invalid entries
        status_dist = self.data['latest_status'].value_counts()
        status_dist = status_dist[status_dist > 0]  # Remove zero counts
        
        print("\nOrder Status Distribution:")
        print(status_dist)
        
        avg_delivery_time = self.data['delivery_time_hours'].mean()
        median_delivery_time = self.data['delivery_time_hours'].median()
        
        print(f"\nAverage Delivery Time: {avg_delivery_time:.1f} hours ({avg_delivery_time/24:.1f} days)")
        print(f"Median Delivery Time: {median_delivery_time:.1f} hours ({median_delivery_time/24:.1f} days)")
        
        self.plot_delivery_performance(status_dist, delivery_success_rate)
    
    def plot_delivery_performance(self, status_dist, success_rate):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        
        # Clean pie chart - only show meaningful segments
        valid_status = status_dist[status_dist > 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(valid_status)))
        
        wedges, texts, autotexts = ax1.pie(
            valid_status.values, 
            labels=valid_status.index, 
            autopct=lambda pct: f'{pct:.1f}%' if pct > 0.1 else '',  # Hide tiny percentages
            startangle=90,
            colors=colors,
            pctdistance=0.85
        )
        
        # Improve text visibility
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax1.set_title('Order Status Distribution', fontsize=14, fontweight='bold')
        
        colors_bar = ['red' if success_rate < 80 else 'orange' if success_rate < 90 else 'green']
        ax2.bar(['Success Rate'], [success_rate], color=colors_bar[0], alpha=0.7)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Percentage')
        ax2.set_title(f'Delivery Success Rate: {success_rate:.1f}%', fontsize=14, fontweight='bold')
        ax2.axhline(y=90, color='green', linestyle='--', label='Target (90%)')
        ax2.legend()
        
        delivery_times = self.data['delivery_time_hours'].dropna()
        if len(delivery_times) > 0:
            ax3.hist(delivery_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_xlabel('Delivery Time (Hours)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Delivery Time Distribution', fontsize=14, fontweight='bold')
            ax3.axvline(delivery_times.mean(), color='red', linestyle='--', label=f'Mean: {delivery_times.mean():.1f}h')
            ax3.legend()
        
        courier_performance = self.data.groupby('courier_name').agg({
            'latest_status': lambda x: (x == 'DELIVERED').sum() / len(x) * 100,
            'delivery_time_hours': 'mean'
        }).round(2)
        
        courier_performance.columns = ['Success_Rate', 'Avg_Delivery_Time']
        courier_performance = courier_performance.sort_values('Success_Rate', ascending=True)
        
        ax4.barh(courier_performance.index, courier_performance['Success_Rate'], 
                color='lightcoral', alpha=0.7)
        ax4.set_xlabel('Success Rate (%)')
        ax4.set_title('Courier Performance Comparison', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/delivery_performance_overview.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
    
    def analyze_geographical_patterns(self):
        print("\n" + "="*50)
        print("GEOGRAPHICAL ANALYSIS")
        print("="*50)
        
        zone_analysis = self.data.groupby('zone').agg({
            'latest_status': lambda x: (x == 'DELIVERED').sum() / len(x) * 100,
            'delivery_time_hours': ['mean', 'median'],
            'actual_weight': 'mean',
            'no_of_attempt': 'mean'
        }).round(2)
        
        zone_analysis.columns = ['Success_Rate', 'Avg_Delivery_Time', 'Median_Delivery_Time', 
                                'Avg_Weight', 'Avg_Attempts']
        
        print("Zone-wise Performance:")
        print(zone_analysis)
        
        self.plot_geographical_analysis(zone_analysis)
    
    def plot_geographical_analysis(self, zone_analysis):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        
        ax1.bar(zone_analysis.index, zone_analysis['Success_Rate'], 
                color='lightgreen', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Zone-wise Delivery Success Rate', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(zone_analysis.index, zone_analysis['Avg_Delivery_Time'], 
                color='lightblue', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Average Delivery Time (Hours)')
        ax2.set_title('Zone-wise Average Delivery Time', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        ax3.bar(zone_analysis.index, zone_analysis['Avg_Attempts'], 
                color='orange', alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Average Delivery Attempts')
        ax3.set_title('Zone-wise Average Delivery Attempts', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        ax4.scatter(zone_analysis['Avg_Weight'], zone_analysis['Success_Rate'], 
                   s=100, alpha=0.7, c='purple')
        for i, zone in enumerate(zone_analysis.index):
            ax4.annotate(zone, (zone_analysis['Avg_Weight'].iloc[i], 
                               zone_analysis['Success_Rate'].iloc[i]))
        ax4.set_xlabel('Average Weight (kg)')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('Weight vs Success Rate by Zone', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/geographical_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
    
    def analyze_statistical_distributions(self):
        print("\n" + "="*50)
        print("STATISTICAL DISTRIBUTION ANALYSIS")
        print("="*50)
        
        numerical_cols = ['actual_weight', 'delivery_time_hours', 'no_of_attempt']
        
        stats_summary = {}
        
        for col in numerical_cols:
            if col in self.data.columns:
                data_col = self.data[col].dropna()
                
                if len(data_col) > 0:
                    stats_summary[col] = {
                        'count': len(data_col),
                        'mean': data_col.mean(),
                        'median': data_col.median(),
                        'std': data_col.std(),
                        'min': data_col.min(),
                        'max': data_col.max(),
                        'skewness': skew(data_col),
                        'kurtosis': kurtosis(data_col),
                        'normality_test': normaltest(data_col)[1]
                    }
        
        stats_df = pd.DataFrame(stats_summary).T
        print("Statistical Summary:")
        print(stats_df.round(4))
        
        self.plot_statistical_distributions(numerical_cols, stats_summary)
        
        return stats_df
    
    def plot_statistical_distributions(self, numerical_cols, stats_summary):
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
        fig.patch.set_facecolor('white')
        
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(numerical_cols):
            if col in self.data.columns:
                data_col = self.data[col].dropna()
                
                axes[0, i].hist(data_col, bins=50, density=True, alpha=0.7, 
                               color='skyblue', edgecolor='black')
                
                mu, sigma = stats.norm.fit(data_col)
                x = np.linspace(data_col.min(), data_col.max(), 100)
                axes[0, i].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                               label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
                
                axes[0, i].set_title(f'{col} Distribution\nSkewness: {stats_summary[col]["skewness"]:.3f}',
                                    fontsize=12, fontweight='bold')
                axes[0, i].set_ylabel('Density')
                axes[0, i].legend()
                
                stats.probplot(data_col, dist="norm", plot=axes[1, i])
                axes[1, i].set_title(f'{col} Q-Q Plot\nNormality p-value: {stats_summary[col]["normality_test"]:.4f}',
                                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/statistical_distributions.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
    
    def analyze_delivery_attempts(self):
        print("\n" + "="*50)
        print("DELIVERY ATTEMPT ANALYSIS")
        print("="*50)
        
        attempt_dist = self.data['no_of_attempt'].value_counts().sort_index()
        print("Delivery Attempt Distribution:")
        print(attempt_dist)
        
        undelivered_reasons = self.data['last_undelivered_reason'].value_counts()
        print("\nTop Undelivered Reasons:")
        print(undelivered_reasons.head(10))
        
        self.plot_attempt_analysis(attempt_dist, undelivered_reasons)
    
    def plot_attempt_analysis(self, attempt_dist, undelivered_reasons):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('white')
        
        top_reasons = undelivered_reasons.head(10)
        ax1.barh(range(len(top_reasons)), top_reasons.values, color='orange', alpha=0.7)
        ax1.set_yticks(range(len(top_reasons)))
        ax1.set_yticklabels(top_reasons.index, fontsize=10)
        ax1.set_xlabel('Frequency')
        ax1.set_title('Top 10 Undelivered Reasons', fontsize=14, fontweight='bold')
        
        payment_success = self.data.groupby('pay_type').agg({
            'latest_status': lambda x: (x == 'DELIVERED').sum() / len(x) * 100
        }).round(2)
        
        ax2.bar(payment_success.index, payment_success['latest_status'], 
               color=['lightblue', 'lightgreen'], alpha=0.7)
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate by Payment Type', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/attempt_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
    
    def analyze_temporal_patterns(self):
        print("\n" + "="*50)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*50)
        
        if 'order_day' not in self.data.columns:
            self.data['order_day'] = self.data['Order_date'].dt.day_name()
        
        valid_data = self.data.dropna(subset=['order_day'])
        
        daily_patterns = valid_data.groupby('order_day').agg({
            'latest_status': lambda x: (x == 'DELIVERED').sum() / len(x) * 100,
            'delivery_time_hours': 'mean'
        }).round(2)
        
        print("Day-wise Performance:")
        print(daily_patterns)
        
        self.plot_temporal_analysis(daily_patterns)

    def plot_temporal_analysis(self, daily_patterns):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('white')
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_ordered = daily_patterns.reindex(day_order)
        
        ax1.bar(daily_ordered.index, daily_ordered['latest_status'], 
                color='lightgreen', alpha=0.7)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Day-wise Delivery Success Rate', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(daily_ordered.index, daily_ordered['delivery_time_hours'], 
                color='lightblue', alpha=0.7)
        ax2.set_ylabel('Average Delivery Time (Hours)')
        ax2.set_title('Day-wise Average Delivery Time', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/temporal_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()

    def generate_business_insights(self):
        print("\n" + "="*80)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*80)
        
        insights = []
        
        total_orders = len(self.data)
        delivered_orders = len(self.data[self.data['latest_status'] == 'DELIVERED'])
        success_rate = (delivered_orders / total_orders) * 100
        
        if success_rate < 85:
            insights.append(f"CRITICAL: Overall delivery success rate is {success_rate:.1f}%, below industry standard of 85%+")
        elif success_rate < 90:
            insights.append(f"WARNING: Delivery success rate is {success_rate:.1f}%, room for improvement")
        else:
            insights.append(f"GOOD: Delivery success rate is {success_rate:.1f}%, meeting industry standards")
        
        avg_delivery_time = self.data['delivery_time_hours'].mean()
        if avg_delivery_time > 72:
            insights.append(f"CRITICAL: Average delivery time is {avg_delivery_time/24:.1f} days, too slow for customer satisfaction")
        elif avg_delivery_time > 48:
            insights.append(f"WARNING: Average delivery time is {avg_delivery_time/24:.1f} days, consider optimization")
        else:
            insights.append(f"GOOD: Average delivery time is {avg_delivery_time/24:.1f} days, competitive")
        
        zone_performance = self.data.groupby('zone')['latest_status'].apply(
            lambda x: (x == 'DELIVERED').sum() / len(x) * 100
        ).sort_values()
        
        worst_zone = zone_performance.index[0]
        worst_performance = zone_performance.iloc[0]
        
        insights.append(f"ATTENTION: '{worst_zone}' zone has lowest success rate at {worst_performance:.1f}%")
        
        courier_performance = self.data.groupby('courier_name')['latest_status'].apply(
            lambda x: (x == 'DELIVERED').sum() / len(x) * 100
        ).sort_values()
        
        if len(courier_performance) > 1:
            worst_courier = courier_performance.index[0]
            best_courier = courier_performance.index[-1]
            
            insights.append(f"COURIER INSIGHT: '{best_courier}' outperforms '{worst_courier}' by {courier_performance.iloc[-1] - courier_performance.iloc[0]:.1f}%")
        
        high_attempt_orders = len(self.data[self.data['no_of_attempt'] > 3])
        if high_attempt_orders > 0:
            percentage = (high_attempt_orders / total_orders) * 100
            insights.append(f"EFFICIENCY: {percentage:.1f}% of orders require 4+ delivery attempts, indicating address/availability issues")
        
        top_reason = self.data['last_undelivered_reason'].value_counts().index[0] if not self.data['last_undelivered_reason'].isnull().all() else None
        if top_reason:
            insights.append(f"TOP ISSUE: '{top_reason}' is the most common undelivered reason - needs targeted solution")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        print("\n" + "="*50)
        print("STRATEGIC RECOMMENDATIONS")
        print("="*50)
        
        recommendations = [
            "PERFORMANCE: Implement zone-specific improvement plans for underperforming areas",
            "TARGETING: Focus on reducing multiple delivery attempts through better customer communication",
            "TECHNOLOGY: Implement real-time tracking and SMS notifications to reduce 'consignee not available' issues",
            "PARTNERSHIPS: Evaluate courier partnerships and consider redistributing volume based on performance",
            "MONITORING: Set up automated alerts for orders exceeding 48-hour delivery time",
            "OPTIMIZATION: Implement predictive analytics to identify high-risk deliveries early",
            "INNOVATION: Consider pickup points or lockers in areas with frequent delivery failures"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    def run_complete_analysis(self):
        print("Starting Comprehensive Courier Data Analysis...")
        print("="*80)
        
        missing_data_report = self.generate_data_quality_report()
        self.analyze_delivery_performance()
        self.analyze_geographical_patterns()
        stats_report = self.analyze_statistical_distributions()
        self.analyze_delivery_attempts()
        self.analyze_temporal_patterns()
        self.generate_business_insights()
        
        print(f"\nAnalysis complete! All plots saved in '{self.output_dir}' directory")
        
        return {
            'missing_data_report': missing_data_report,
            'statistical_report': stats_report
        }

if __name__ == "__main__":
    analyzer = CourierDataAnalyzer('delivery_data_updated.csv')
    reports = analyzer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("ADDITIONAL ANALYSIS AVAILABLE")
    print("="*80)
    print("You can now access:")
    print("- analyzer.data: Complete processed dataset")
    print("- analyzer.plot_*(): Individual plotting functions")
    print("- reports: Generated analysis reports")