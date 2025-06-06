import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from mango import Tuner
from scipy.stats import uniform

class Enrico:
    """
    A class for optimizing machine learning models using Bayesian optimization.
    
    Attributes:
        X (array-like): Feature matrix
        y (array-like): Target vector
        
    Methods:
        optimize_extratrees_regressor: Optimizes ExtraTreesRegressor using Bayesian optimization
        optimize_xgb_classifier: Optimizes XGBClassifier using Bayesian optimization
    """
    
    def __init__(self, X, y):
        """
        Initialize ModelOptimizer with feature matrix and target vector.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector
        """
        self.X = X
        self.y = y
    
    def optimize_extratrees_regressor(self, test_size=0.2, n_iterations=80):
        """
        Optimize ExtraTreesRegressor using Bayesian optimization.
        
        Args:
            test_size (float): Proportion of dataset to include in test split
            n_iterations (int): Number of optimization iterations
            
        Returns:
            tuple: Contains train/test splits, best score, best parameters, and fitted model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        # Define parameter space
        param_space = {
            'max_depth': range(3, 10),
            'min_samples_split': range(int(0.01*self.X.shape[0]), int(0.1*self.X.shape[0])),
            'min_samples_leaf': range(int(0.01*self.X.shape[0]), int(0.1*self.X.shape[0])),
            'max_features': ["sqrt", "log2", "auto"]
        }
        
        # Define objective function
        def objective(space):
            return [cross_val_score(
                ExtraTreesRegressor(**params), 
                X_train, y_train, 
                scoring='neg_mean_squared_error', 
                cv=5
            ).mean() for params in space]
        
        # Optimize
        tuner = Tuner(param_space, objective, dict(num_iteration=n_iterations, initial_random=10))
        results = tuner.maximize()
        
        # Fit final model
        model = ExtraTreesRegressor(**results['best_params'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f'Best RMSE on train-set: {results["best_objective"]}')
        print(f'RMSE on test-set: {test_rmse}')
        print(f'Best Parameters: {results["best_params"]}')
        
        return X_train, X_test, y_train, y_test, results["best_objective"], results["best_params"], model
    
    def optimize_xgb_classifier(self, test_size=0.2, n_iterations=10):
        """
        Optimize XGBClassifier using Bayesian optimization.
        
        Args:
            test_size (float): Proportion of dataset to include in test split
            n_iterations (int): Number of optimization iterations
            
        Returns:
            tuple: Contains train/test splits, best score, best parameters, optimization results, and fitted model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        # Define parameter space
        param_space = {
            "learning_rate": uniform(0, 1),
            "gamma": uniform(0, 5)
        }
        
        # Define objective function
        def objective(space):
            return [cross_val_score(
                XGBClassifier(**params), 
                X_train, y_train, 
                scoring='accuracy', 
                cv=5
            ).mean() for params in space]
        
        # Optimize
        tuner = Tuner(param_space, objective, dict(num_iteration=n_iterations, initial_random=10))
        results = tuner.maximize()
        
        # Fit final model
        model = XGBClassifier(**results['best_params'])
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        print(f'Accuracy on test set: {accuracy:.4f}')
        
        return X_train, X_test, y_train, y_test, results["best_objective"], results["best_params"], results, model
