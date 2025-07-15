import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, classification_report, confusion_matrix,
                           precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EDDPredictor:
    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.delivery_categories = {}

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

        data['Delivery_Category'] = pd.cut(data['Delivery_Days'], 
                                         bins=[0, 3, 7, 14, float('inf')], 
                                         labels=['Very Fast', 'Fast', 'Normal', 'Slow'])

        data['On_Time'] = (data['Delivery_Days'] <= 7).astype(int)

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

        regression_data = data[feature_columns + ['Delivery_Days']].dropna()
        classification_data = data[feature_columns + ['On_Time', 'Delivery_Category']].dropna()

        print(f"\nFeature Engineering Complete:")
        print(f"Regression dataset shape: {regression_data.shape}")
        print(f"Classification dataset shape: {classification_data.shape}")
        print(f"Features used: {feature_columns}")

        return regression_data, classification_data, feature_columns

    def train_regression_models(self, data, feature_columns, test_size=0.2, random_state=42):
        """Train regression models for predicting delivery days"""
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
        regression_results = {}

        for name, model in models.items():

            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            regression_results[name] = {
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

        self.regression_model = best_model
        return X_test, y_test, regression_results

    def train_classification_models(self, data, feature_columns, test_size=0.2, random_state=42):
        """Train classification models for predicting delivery categories"""
        X = data[feature_columns]
        y = data['On_Time']  

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Ridge Logistic': LogisticRegression(penalty='l2', C=1.0, random_state=random_state, max_iter=1000),
            'Lasso Logistic': LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=random_state),
            'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=random_state)
        }

        best_model = None
        best_score = -float('inf')
        classification_results = {}

        for name, model in models.items():

            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

            classification_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }

            if accuracy > best_score:
                best_score = accuracy
                best_model = model

        self.classification_model = best_model
        return X_test, y_test, classification_results

    def predict_delivery_days(self, features_dict):
        """Predict delivery days using regression model"""
        if self.regression_model is None:
            raise ValueError("Regression model not trained yet!")

        if isinstance(features_dict, dict):
            features_df = pd.DataFrame([features_dict])
        else:
            features_df = features_dict

        missing_features = set(self.feature_names) - set(features_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        features_ordered = features_df[self.feature_names]
        features_scaled = self.scaler.transform(features_ordered)
        prediction = self.regression_model.predict(features_scaled)
        return prediction

    def predict_on_time_delivery(self, features_dict):
        """Predict on-time delivery probability using logistic regression"""
        if self.classification_model is None:
            raise ValueError("Classification model not trained yet!")

        if isinstance(features_dict, dict):
            features_df = pd.DataFrame([features_dict])
        else:
            features_df = features_dict

        missing_features = set(self.feature_names) - set(features_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        features_ordered = features_df[self.feature_names]
        features_scaled = self.scaler.transform(features_ordered)

        prediction = self.classification_model.predict(features_scaled)
        probability = self.classification_model.predict_proba(features_scaled)[:, 1]

        return prediction, probability

    def plot_regression_results(self, y_true, y_pred, model_name="Best Regression Model"):
        """Plot regression model results"""
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
        plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
        print("Regression results saved as 'regression_results.png'")

    def plot_classification_results(self, y_true, y_pred, y_pred_proba, classification_results, model_name="Best Classification Model"):
        """Plot classification model results"""
        import matplotlib
        matplotlib.use('Agg')

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f'{model_name}: Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Delayed', color='red')
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='On-time', color='green')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Probability Distribution by Class')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        models = list(classification_results.keys())
        accuracies = [classification_results[model]['accuracy'] for model in models]

        bars = axes[1, 1].bar(models, accuracies, alpha=0.7, color='skyblue')
        axes[1, 1].set_title('Model Accuracy Comparison')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].tick_params(axis='x', rotation=45)

        for bar, acc in zip(bars, accuracies):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
        print("Classification results saved as 'classification_results.png'")

    def plot_logistic_regression_analysis(self, classification_results):
        """Plot logistic regression specific analysis"""
        import matplotlib
        matplotlib.use('Agg')

        logistic_model = classification_results['Logistic Regression']['model']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        coefficients = logistic_model.coef_[0]
        feature_coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=True)

        axes[0, 0].barh(range(len(feature_coef_df)), feature_coef_df['coefficient'])
        axes[0, 0].set_yticks(range(len(feature_coef_df)))
        axes[0, 0].set_yticklabels(feature_coef_df['feature'])
        axes[0, 0].set_xlabel('Coefficient Value')
        axes[0, 0].set_title('Logistic Regression Coefficients')
        axes[0, 0].grid(True, alpha=0.3)

        odds_ratios = np.exp(coefficients)
        odds_df = pd.DataFrame({
            'feature': self.feature_names,
            'odds_ratio': odds_ratios
        }).sort_values('odds_ratio', ascending=True)

        axes[0, 1].barh(range(len(odds_df)), odds_df['odds_ratio'])
        axes[0, 1].set_yticks(range(len(odds_df)))
        axes[0, 1].set_yticklabels(odds_df['feature'])
        axes[0, 1].set_xlabel('Odds Ratio')
        axes[0, 1].set_title('Feature Odds Ratios')
        axes[0, 1].axvline(x=1, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].grid(True, alpha=0.3)

        models = list(classification_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

        model_metrics = np.array([[classification_results[model][metric] for metric in metrics] 
                                 for model in models])

        im = axes[1, 0].imshow(model_metrics, cmap='RdYlGn', aspect='auto')
        axes[1, 0].set_xticks(range(len(metrics)))
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].set_yticks(range(len(models)))
        axes[1, 0].set_yticklabels(models)
        axes[1, 0].set_title('Model Performance Heatmap')

        for i in range(len(models)):
            for j in range(len(metrics)):
                text = axes[1, 0].text(j, i, f'{model_metrics[i, j]:.3f}',
                                     ha="center", va="center", color="black")

        plt.colorbar(im, ax=axes[1, 0])

        cv_means = [classification_results[model]['cv_mean'] for model in models]
        cv_stds = [classification_results[model]['cv_std'] for model in models]

        bars = axes[1, 1].bar(models, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='lightcoral')
        axes[1, 1].set_title('Cross-Validation Accuracy')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].tick_params(axis='x', rotation=45)

        for bar, mean, std in zip(bars, cv_means, cv_stds):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                           f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('logistic_regression_analysis.png', dpi=300, bbox_inches='tight')
        print("Logistic regression analysis saved as 'logistic_regression_analysis.png'")

    def print_model_comparison(self, regression_results, classification_results):
        """Print detailed model comparison"""
        print("\n" + "="*80)
        print("REGRESSION MODEL PERFORMANCE COMPARISON")
        print("="*80)

        for name, results in regression_results.items():
            print(f"\n{name}:")
            print(f"  R² Score: {results['r2']:.4f}")
            print(f"  RMSE: {results['rmse']:.4f}")
            print(f"  MAE: {results['mae']:.4f}")
            print(f"  Cross-validation R²: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")

        print("\n" + "="*80)
        print("CLASSIFICATION MODEL PERFORMANCE COMPARISON")
        print("="*80)

        for name, results in classification_results.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  AUC Score: {results['auc']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1 Score: {results['f1']:.4f}")
            print(f"  Cross-validation Accuracy: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")

def main():
    """Main function to run the EDD prediction models"""
    predictor = EDDPredictor()

    print("Generating sample data...")

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

    regression_data, classification_data, feature_columns = predictor.preprocess_data(sample_data)

    print("\nTraining regression models...")
    X_test_reg, y_test_reg, regression_results = predictor.train_regression_models(regression_data, feature_columns)

    print("\nTraining classification models...")
    X_test_class, y_test_class, classification_results = predictor.train_classification_models(classification_data, feature_columns)

    predictor.print_model_comparison(regression_results, classification_results)

    best_regression_name = max(regression_results.keys(), key=lambda x: regression_results[x]['r2'])
    best_classification_name = max(classification_results.keys(), key=lambda x: classification_results[x]['accuracy'])

    print("\nGenerating analysis plots...")
    predictor.plot_regression_results(y_test_reg, regression_results[best_regression_name]['predictions'], best_regression_name)
    predictor.plot_classification_results(y_test_class, 
                                        classification_results[best_classification_name]['predictions'],
                                        classification_results[best_classification_name]['probabilities'],
                                        classification_results,
                                        best_classification_name)
    predictor.plot_logistic_regression_analysis(classification_results)

    print(f"\n{best_regression_name} selected as best regression model!")
    print(f"Best R² Score: {regression_results[best_regression_name]['r2']:.4f}")

    print(f"\n{best_classification_name} selected as best classification model!")
    print(f"Best Accuracy: {classification_results[best_classification_name]['accuracy']:.4f}")

    print(f"\nExample Predictions:")
    print("="*50)

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

        predicted_days = predictor.predict_delivery_days(example_features)
        print(f"Predicted delivery days: {predicted_days[0]:.2f}")

        on_time_pred, on_time_prob = predictor.predict_on_time_delivery(example_features)
        print(f"On-time delivery prediction: {'Yes' if on_time_pred[0] else 'No'}")
        print(f"On-time probability: {on_time_prob[0]:.3f}")

        print(f"Estimated delivery date: {pd.Timestamp.now() + pd.Timedelta(days=predicted_days[0])}")

    except Exception as e:
        print(f"Error in prediction: {e}")

    print("\nAll plots have been saved as PNG files:")
    print("- regression_results.png: Regression model analysis")
    print("- classification_results.png: Classification model analysis") 
    print("- logistic_regression_analysis.png: Detailed logistic regression analysis")

if __name__ == "__main__":
    main()