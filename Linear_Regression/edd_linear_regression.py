import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EDDPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the delivery data"""
        try:

            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} records from {file_path}")

            print("\nDataset Info:")
            print(df.info())
            print("\nFirst few rows:")
            print(df.head())

            return self.preprocess_data(df)

        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def preprocess_data(self, df):
        """Preprocess the data for modeling"""

        data = df.copy()

        date_columns = ['Order_date', 'Manifest Date(RTS)', 'Pickup Date', 
                       'Last Scan Date', '1st attempt Date', 'Last OFD_Date', 'Delivery Date']

        for col in date_columns:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], format='%d-%m-%Y', errors='coerce')

        data['Delivery_Days'] = (data['Delivery Date'] - data['Order_date']).dt.days

        data = data.dropna(subset=['Delivery_Days'])

        data = data.drop(columns=['Delivery Date', 'Delivery Time', 'Last OFD_Date', 'Last_OFD_Time'], errors='ignore')

        data['Weight_Category'] = pd.cut(data['Actual Weight'], 
                                       bins=[0, 0.5, 1, 3, 5, float('inf')], 
                                       labels=[1, 2, 3, 4, 5])
        data['Weight_Category'] = data['Weight_Category'].astype(float)

        data['Order_Hour'] = pd.to_datetime(data['Order_time'], format='%H:%M:%S', errors='coerce').dt.hour
        data['Order_Day_of_Week'] = data['Order_date'].dt.dayofweek
        data['Order_Month'] = data['Order_date'].dt.month

        data['Pincode_Diff'] = abs(data['From Pincode'] - data['To  Pincode'])

        categorical_cols = ['Pay Type', 'Status', 'Zone', 'Courier Name']
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[f'{col}_Encoded'] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le

        data['Weight_Log'] = np.log1p(data['Actual Weight'])
        data['Pincode_Distance_Bucket'] = pd.cut(data['Pincode_Diff'], 
                                                bins=[0, 50000, 100000, 200000, 500000, float('inf')], 
                                                labels=[1, 2, 3, 4, 5]).astype(float)

        feature_columns = [
            'Actual Weight', 'Weight_Log', 'Weight_Category', 'Order_Hour', 'Order_Day_of_Week', 
            'Order_Month', 'Pincode_Diff', 'Pincode_Distance_Bucket', 'Pay Type_Encoded', 
            'Zone_Encoded', 'Courier Name_Encoded'
        ]

        feature_data = data[feature_columns + ['Delivery_Days']].dropna()

        print(f"\nFeature Engineering Complete:")
        print(f"Final dataset shape: {feature_data.shape}")
        print(f"Features used: {feature_columns}")

        return feature_data, feature_columns

    def train_model(self, data, feature_columns, test_size=0.2, random_state=42):
        """Train the linear regression model"""
        X = data[feature_columns]
        y = data['Delivery_Days']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.feature_names = feature_columns

        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state)
        }

        best_model = None
        best_score = -float('inf')
        model_results = {}

        for name, model in models.items():

            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            model_results[name] = {
                'model': model,
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }

            if r2 > best_score:
                best_score = r2
                best_model = model

        self.model = best_model

        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)

        for name, results in model_results.items():
            print(f"\n{name}:")
            print(f"  R² Score: {results['r2']:.4f}")
            print(f"  RMSE: {results['rmse']:.4f}")
            print(f"  MAE: {results['mae']:.4f}")
            print(f"  Cross-validation R²: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")

        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nFeature Importance (Top 10):")
            print(feature_importance.head(10))

        return X_test, y_test, model_results

    def predict_edd(self, features_dict):
        """Predict EDD for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if isinstance(features_dict, dict):
            features_df = pd.DataFrame([features_dict])
        else:
            features_df = features_dict

        missing_features = set(self.feature_names) - set(features_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        features_ordered = features_df[self.feature_names]

        features_scaled = self.scaler.transform(features_ordered)

        prediction = self.model.predict(features_scaled)

        return prediction

    def plot_results(self, y_true, y_pred, model_name="Best Model"):
        """Plot comprehensive model results"""

        import matplotlib
        matplotlib.use('Agg')  

        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Delivery Days')
        axes[0, 0].set_ylabel('Predicted Delivery Days')
        axes[0, 0].set_title(f'{model_name}: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Delivery Days')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)

        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        print("Model performance plots saved as 'model_performance.png'")

        print(f"\nModel Performance Summary:")
        print(f"Mean Residual: {residuals.mean():.4f}")
        print(f"Std Residual: {residuals.std():.4f}")
        print(f"Min Residual: {residuals.min():.4f}")
        print(f"Max Residual: {residuals.max():.4f}")

    def plot_feature_analysis(self, data, feature_columns):
        """Plot feature analysis and relationships"""
        import matplotlib
        matplotlib.use('Agg')

        if hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)

            plt.barh(range(len(feature_importance)), feature_importance['importance'])
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance Analysis')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("Feature importance plot saved as 'feature_importance.png'")

        plt.figure(figsize=(12, 10))
        correlation_matrix = data[feature_columns + ['Delivery_Days']].corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("Correlation matrix saved as 'correlation_matrix.png'")

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(data['Delivery_Days'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        plt.xlabel('Delivery Days')
        plt.ylabel('Frequency')
        plt.title('Distribution of Delivery Days')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.boxplot(data['Delivery_Days'])
        plt.ylabel('Delivery Days')
        plt.title('Delivery Days Box Plot')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('delivery_days_distribution.png', dpi=300, bbox_inches='tight')
        print("Delivery days distribution saved as 'delivery_days_distribution.png'")

    def plot_business_insights(self, data):
        """Plot business insights and patterns"""
        import matplotlib
        matplotlib.use('Agg')

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        if 'Pay Type' in data.columns:
            pay_type_stats = data.groupby('Pay Type')['Delivery_Days'].agg(['mean', 'std', 'count'])
            axes[0, 0].bar(pay_type_stats.index, pay_type_stats['mean'], 
                          yerr=pay_type_stats['std'], capsize=5, alpha=0.7)
            axes[0, 0].set_title('Average Delivery Days by Pay Type')
            axes[0, 0].set_ylabel('Average Delivery Days')
            for i, v in enumerate(pay_type_stats['mean']):
                axes[0, 0].text(i, v + 0.1, f'{v:.1f}', ha='center')

        if 'Zone' in data.columns:
            zone_stats = data.groupby('Zone')['Delivery_Days'].agg(['mean', 'std', 'count'])
            axes[0, 1].bar(range(len(zone_stats)), zone_stats['mean'], 
                          yerr=zone_stats['std'], capsize=5, alpha=0.7)
            axes[0, 1].set_title('Average Delivery Days by Zone')
            axes[0, 1].set_ylabel('Average Delivery Days')
            axes[0, 1].set_xticks(range(len(zone_stats)))
            axes[0, 1].set_xticklabels(zone_stats.index, rotation=45)
            for i, v in enumerate(zone_stats['mean']):
                axes[0, 1].text(i, v + 0.1, f'{v:.1f}', ha='center')

        if 'Courier Name' in data.columns:
            courier_stats = data.groupby('Courier Name')['Delivery_Days'].agg(['mean', 'std', 'count'])
            axes[0, 2].bar(courier_stats.index, courier_stats['mean'], 
                          yerr=courier_stats['std'], capsize=5, alpha=0.7)
            axes[0, 2].set_title('Average Delivery Days by Courier')
            axes[0, 2].set_ylabel('Average Delivery Days')
            for i, v in enumerate(courier_stats['mean']):
                axes[0, 2].text(i, v + 0.1, f'{v:.1f}', ha='center')

        if 'Weight_Category' in data.columns:
            weight_stats = data.groupby('Weight_Category')['Delivery_Days'].agg(['mean', 'std', 'count'])
            axes[1, 0].bar(weight_stats.index, weight_stats['mean'], 
                          yerr=weight_stats['std'], capsize=5, alpha=0.7)
            axes[1, 0].set_title('Average Delivery Days by Weight Category')
            axes[1, 0].set_ylabel('Average Delivery Days')
            axes[1, 0].set_xlabel('Weight Category')
            for i, v in enumerate(weight_stats['mean']):
                axes[1, 0].text(i, v + 0.1, f'{v:.1f}', ha='center')

        if 'Order_Hour' in data.columns:
            hour_stats = data.groupby('Order_Hour')['Delivery_Days'].mean()
            axes[1, 1].plot(hour_stats.index, hour_stats.values, marker='o')
            axes[1, 1].set_title('Average Delivery Days by Order Hour')
            axes[1, 1].set_ylabel('Average Delivery Days')
            axes[1, 1].set_xlabel('Order Hour')
            axes[1, 1].grid(True, alpha=0.3)

        if 'Order_Day_of_Week' in data.columns:
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_stats = data.groupby('Order_Day_of_Week')['Delivery_Days'].mean()
            axes[1, 2].bar(range(len(dow_stats)), dow_stats.values, alpha=0.7)
            axes[1, 2].set_title('Average Delivery Days by Day of Week')
            axes[1, 2].set_ylabel('Average Delivery Days')
            axes[1, 2].set_xlabel('Day of Week')
            axes[1, 2].set_xticks(range(len(dow_stats)))
            axes[1, 2].set_xticklabels([day_names[i] for i in dow_stats.index])
            for i, v in enumerate(dow_stats.values):
                axes[1, 2].text(i, v + 0.1, f'{v:.1f}', ha='center')

        plt.tight_layout()
        plt.savefig('business_insights.png', dpi=300, bbox_inches='tight')
        print("Business insights plots saved as 'business_insights.png'")

    def plot_model_comparison(self, model_results):
        """Plot model comparison results"""
        import matplotlib
        matplotlib.use('Agg')

        models = list(model_results.keys())
        r2_scores = [model_results[model]['r2'] for model in models]
        rmse_scores = [model_results[model]['rmse'] for model in models]
        mae_scores = [model_results[model]['mae'] for model in models]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        bars1 = axes[0].bar(models, r2_scores, alpha=0.7, color='lightblue')
        axes[0].set_title('R² Score Comparison')
        axes[0].set_ylabel('R² Score')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        for bar, score in zip(bars1, r2_scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')

        bars2 = axes[1].bar(models, rmse_scores, alpha=0.7, color='lightcoral')
        axes[1].set_title('RMSE Comparison')
        axes[1].set_ylabel('RMSE (Days)')
        axes[1].tick_params(axis='x', rotation=45)
        for bar, score in zip(bars2, rmse_scores):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')

        bars3 = axes[2].bar(models, mae_scores, alpha=0.7, color='lightgreen')
        axes[2].set_title('MAE Comparison')
        axes[2].set_ylabel('MAE (Days)')
        axes[2].tick_params(axis='x', rotation=45)
        for bar, score in zip(bars3, mae_scores):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("Model comparison plots saved as 'model_comparison.png'")

    def plot_prediction_intervals(self, X_test, y_test, y_pred):
        """Plot prediction intervals and confidence bands"""
        import matplotlib
        matplotlib.use('Agg')

        residuals = y_test - y_pred
        std_residual = np.std(residuals)

        confidence_95 = 1.96 * std_residual
        confidence_80 = 1.28 * std_residual

        sorted_indices = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_indices]
        y_test_sorted = y_test.iloc[sorted_indices]

        plt.figure(figsize=(12, 8))

        plt.scatter(y_pred, y_test, alpha=0.6, label='Actual vs Predicted')

        plt.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], 'r--', lw=2, label='Perfect Prediction')

        plt.fill_between(y_pred_sorted, y_pred_sorted - confidence_95, y_pred_sorted + confidence_95, 
                        alpha=0.2, color='red', label='95% Confidence Interval')
        plt.fill_between(y_pred_sorted, y_pred_sorted - confidence_80, y_pred_sorted + confidence_80, 
                        alpha=0.3, color='orange', label='80% Confidence Interval')

        plt.xlabel('Predicted Delivery Days')
        plt.ylabel('Actual Delivery Days')
        plt.title('Prediction Intervals and Confidence Bands')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('prediction_intervals.png', dpi=300, bbox_inches='tight')
        print("Prediction intervals plot saved as 'prediction_intervals.png'")

def main():
    """Main function to run the EDD prediction model"""

    predictor = EDDPredictor()

    print("Loading and preprocessing data...")
    file_path = "delivery_data_original 1.csv"  

    np.random.seed(42)
    n_samples = 2000

    base_delivery_days = np.random.normal(5, 2, n_samples)  
    base_delivery_days = np.clip(base_delivery_days, 1, 15)  

    sample_data = pd.DataFrame({
        'Pay Type': np.random.choice(['COD', 'PPD'], n_samples),
        'Actual Weight': np.random.exponential(1.5, n_samples),
        'Status': ['DELIVERED'] * n_samples,
        'Zone': np.random.choice(['Metro To Metro', 'Metro To Non-Metro', 'Non-Metro To Metro'], n_samples),
        'From Pincode': np.random.randint(100000, 999999, n_samples),
        'To  Pincode': np.random.randint(100000, 999999, n_samples),
        'Courier Name': np.random.choice(['Blue Dart', 'FedEx', 'DHL'], n_samples),
        'Order_date': pd.date_range('2025-01-01', periods=n_samples, freq='H'),
        'Order_time': [f"{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}" for _ in range(n_samples)]
    })

    order_dates = pd.date_range('2025-01-01', periods=n_samples, freq='H')
    delivery_dates = []

    for i in range(n_samples):
        days_to_add = base_delivery_days[i]

        if sample_data.iloc[i]['Pay Type'] == 'COD':
            days_to_add += 1  
        if sample_data.iloc[i]['Zone'] == 'Non-Metro To Metro':
            days_to_add += 1.5  
        if sample_data.iloc[i]['Actual Weight'] > 3:
            days_to_add += 0.5  

        delivery_dates.append(order_dates[i] + pd.Timedelta(days=int(days_to_add)))

    sample_data['Delivery Date'] = delivery_dates

    processed_data, feature_columns = predictor.preprocess_data(sample_data)

    print("\nTraining models...")
    X_test, y_test, model_results = predictor.train_model(processed_data, feature_columns)

    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
    best_predictions = model_results[best_model_name]['predictions']

    print("\nGenerating comprehensive analysis plots...")
    print("="*50)

    predictor.plot_results(y_test, best_predictions, best_model_name)

    predictor.plot_feature_analysis(processed_data, feature_columns)

    predictor.plot_business_insights(processed_data)

    predictor.plot_model_comparison(model_results)

    predictor.plot_prediction_intervals(X_test, y_test, best_predictions)

    print("\nAll plots have been saved as PNG files in the current directory!")
    print("Generated files:")
    print("- model_performance.png: Model accuracy and residual analysis")
    print("- feature_importance.png: Feature importance ranking")
    print("- correlation_matrix.png: Feature correlation heatmap")
    print("- delivery_days_distribution.png: Target variable distribution")
    print("- business_insights.png: Business factor analysis")
    print("- model_comparison.png: Algorithm performance comparison")
    print("- prediction_intervals.png: Confidence intervals")

    print(f"\n{best_model_name} selected as the best model!")
    print(f"Best R² Score: {model_results[best_model_name]['r2']:.4f}")
    print(f"Best RMSE: {model_results[best_model_name]['rmse']:.4f}")
    print(f"Best MAE: {model_results[best_model_name]['mae']:.4f}")

    print(f"\nExample Prediction:")
    print("="*40)

    example_features = {
        'Actual Weight': 1.5,
        'Weight_Log': np.log1p(1.5),
        'Weight_Category': 2.0,
        'Order_Hour': 14,
        'Order_Day_of_Week': 1,
        'Order_Month': 5,
        'Pincode_Diff': 150000,
        'Pincode_Distance_Bucket': 3.0,
        'Pay Type_Encoded': 1,  
        'Zone_Encoded': 0,      
        'Courier Name_Encoded': 0  
    }

    try:
        predicted_days = predictor.predict_edd(example_features)
        print(f"Predicted delivery days: {predicted_days[0]:.2f}")
        print(f"Estimated delivery date: {pd.Timestamp.now() + pd.Timedelta(days=predicted_days[0])}")
    except Exception as e:
        print(f"Error in prediction: {e}")
        print("Available features in model:", predictor.feature_names)
        print("Provided features:", list(example_features.keys()))

if __name__ == "__main__":
    main()