import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List

class Maestro:
    """
    A utility class for data preprocessing and feature engineering.
    
    Static Methods:
        outliers: IQR range outliers
        timestamp_feature_extractor: Extract timestamp features from a column
    """
    @staticmethod
    def outliers(data: pd.DataFrame, column_name: str, output: str = None) -> pd.DataFrame:
        """Detect and handle outliers in specified column.
        
        Args:
            data: Input DataFrame
            column_name: Column to check for outliers
            output: Handling method ('create_feature', 'replace_with_na', 'drop_outliers')
        
        Returns:
            Processed DataFrame
        """
        q25, q75 = data[column_name].quantile([0.25, 0.75])
        iqr = q75 - q25
        lower, upper = q25 - 1.5*iqr, q75 + 1.5*iqr
        
        outliers = data[(data[column_name] < lower) | (data[column_name] > upper)]
        clean_data = data[(data[column_name] > lower) & (data[column_name] < upper)]
        
        # Print statistics
        stats = {
            '25th quantile': q25, '75th quantile': q75, 'IQR': iqr,
            'Lower Bound': lower, 'Upper Bound': upper,
            '# of outliers': len(outliers),
            '% of outliers': len(outliers)/len(data)
        }
        print('\n'.join(f'{k}: {v}' for k, v in stats.items()))
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plot_params = dict(notch=True, showcaps=False,
                         flierprops={"marker": "o"},
                         boxprops={"facecolor": (.4, .6, .8, .5)},
                         medianprops={"color": "coral"}, fliersize=5)
        
        sns.boxplot(data[column_name], ax=ax1, **plot_params).set(title='Outliers boxplot')
        sns.boxplot(clean_data[column_name], ax=ax2, **plot_params).set(title='Cleaned series')
        fig.show()
        
        # Handle outliers based on output parameter
        if output == 'create_feature':
            data[f'{column_name}_outliers'] = data[column_name].apply(
                lambda x: 'outlier' if (x < lower) | (x > upper) else np.nan)
        elif output == 'replace_with_na':
            data[column_name] = data[column_name].mask((data[column_name] < lower) | (data[column_name] > upper))
        elif output == 'drop_outliers':
            data = clean_data
            
        return data

    @staticmethod
    def timestamp_feature_extractor(self, data: pd.DataFrame, column_name: str, 
                                  opts: List[str] = None) -> pd.DataFrame:
        """Extract various datetime features from a timestamp column.
        
        Args:
            data: Input DataFrame
            column_name: Name of datetime column
            opts: List of features to extract. Available options:
                ['datetime', 'year', 'month', 'quarter', 'day', 'weekday', 'weekend',
                 'hour', 'minute', 'seconds', 'week', 'sin_month', 'cos_month',
                 'sin_week', 'cos_week', 'sin_weekday', 'cos_weekday',
                 'sin_hour', 'cos_hour']
        
        Returns:
            DataFrame with additional datetime features
        """
        if not opts:
            print("You've not provided any options, try with some options.")
            return data
            
        data[column_name] = pd.to_datetime(data[column_name]).dt.tz_localize(None)
        
        feature_extractors = {
            'date': lambda x: pd.to_datetime(x.dt.strftime('%Y-%m-%d')),
            'datetime': lambda x: pd.to_datetime(x.dt.strftime('%Y-%m-%d %H:00:00')),
            'year': lambda x: x.dt.year,
            'quarter': lambda x: x.dt.quarter,
            'month': lambda x: x.dt.month,
            'day': lambda x: x.dt.day,
            'hour': lambda x: x.dt.strftime('%H').astype(np.int64),
            'minute': lambda x: x.dt.minute,
            'seconds': lambda x: x.dt.second,
            'weekday': lambda x: x.dt.dayofweek,
            'weekend': lambda x: x.dt.dayofweek.apply(lambda d: 1 if d in [5, 6] else 0),
            'week': lambda x: x.dt.isocalendar().week,
            'sin_month': lambda x: np.sin(2*np.pi*x.dt.month/12),
            'cos_month': lambda x: np.cos(2*np.pi*x.dt.month/12),
            'sin_week': lambda x: np.sin(2*np.pi*x.dt.isocalendar().week/52),
            'cos_week': lambda x: np.cos(2*np.pi*x.dt.isocalendar().week/52),
            'sin_weekday': lambda x: np.sin(2*np.pi*x.dt.dayofweek/7),
            'cos_weekday': lambda x: np.cos(2*np.pi*x.dt.dayofweek/7),
            'sin_hour': lambda x: np.sin(2*np.pi*x.dt.hour/24),
            'cos_hour': lambda x: np.cos(2*np.pi*x.dt.hour/24)
        }
        
        for opt in opts:
            if opt.lower() in feature_extractors:
                data[f'{column_name}_{opt}'] = feature_extractors[opt.lower()](data[column_name])
                
        return data