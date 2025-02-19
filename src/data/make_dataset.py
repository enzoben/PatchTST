import pandas as pd
import torch
from torch.utils.data import Dataset

def get_train_test_datasets(data_path: str, test_size: float = 0.2, window_size: int = 100, validation_size: float = None, 
                           sliding_size: int = 1, forecasting_horizon: int = 10, kind: str = 'sliding_window'):
    """
    Used to create a train and a test dataset from a given file. The test set is created using the 
    last (test_size)*100% of the entire dataset while the train set is the first (1-test_size)*100%.

    Args:
        data_path (str): path to the data file.
        test_size (float, optional): pourcentage of the dataset to be use for testing. Defaults to 0.2.
        window_size (int, optional): size of the window, also called look back window. Defaults to 100.
        sliding_size (int, optional): size of the stride for the sliding window . Defaults to 1.
        forecasting_horizon (int, optional): forecasting horizon. Defaults to 10.
        kind (str, optional): the kind of splitting strategie. Defaults to 'sliding_window'.

    Returns:
        train_set (nn.utils.data.Dataset): the train dataset.
        test_set (nn.utils.data.Dataset): the test dataset.
    """

    data = pd.read_csv(data_path, index_col='date')

    if kind == 'sliding_window':
        if validation_size:
            train_set = sliding_widow_dataset(data.iloc[:-int(len(data)*(test_size+validation_size))], window_size=window_size,
                                            sliding_size=sliding_size, forecasting_horizon=forecasting_horizon)
            val_set = sliding_widow_dataset(data.iloc[-int(len(data)*(test_size+validation_size)): -int(len(data)*test_size)], window_size=window_size,
                                            sliding_size=sliding_size, forecasting_horizon=forecasting_horizon)
            test_set = sliding_widow_dataset(data.iloc[-int(len(data)*test_size):], window_size=window_size,
                                            sliding_size=sliding_size, forecasting_horizon=forecasting_horizon)
        else :
            train_set = sliding_widow_dataset(data.iloc[:-int(len(data)*test_size)], window_size=window_size, 
                                            sliding_size=sliding_size, forecasting_horizon=forecasting_horizon)
            test_set = sliding_widow_dataset(data.iloc[-int(len(data)*test_size):], window_size=window_size, 
                                            sliding_size=sliding_size, forecasting_horizon=forecasting_horizon)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    
    if validation_size:
        return train_set, val_set, test_set
    
    return train_set, test_set

class sliding_widow_dataset(Dataset):

    def __init__(self, data: pd.DataFrame, window_size: int = 100, 
                 sliding_size: int = 1, forecasting_horizon: int = 10):
        """
        the sliding window dataset class is used to create a dataset with the 
        sliding window technique, preventing data leakage.

        Args:
            data (pd.DataFrame): the data to be used for the dataset.
            window_size (int, optional): size of the window, also called look back window. Defaults to 100.
            sliding_size (int, optional): size of the stride for the sliding window . Defaults to 1.
            forecasting_horizon (int, optional): forecasting horizon. Defaults to 10.
        """
        super(sliding_widow_dataset, self).__init__()

        self.window_size = window_size
        self.sliding_size = sliding_size
        self.forecasting_horizon = forecasting_horizon

        self.data = data
        self.data_size = len(self.data)

        self.nb_of_series = self.data.shape[1]
        self.nb_of_sequences = (self.data_size - self.window_size - self.forecasting_horizon) // self.sliding_size + 1

    def __len__(self):
        return (self.data_size - self.window_size - self.forecasting_horizon) // self.sliding_size + 1
    
    def __getitem__(self, idx):
        idx = idx * self.sliding_size

        x = self.data.iloc[idx:idx+self.window_size].values
        x = torch.tensor(x, dtype = torch.float32).T

        x_time = self.data.index[idx:idx+self.window_size].values

        y = self.data.iloc[idx+self.window_size:idx+self.window_size+self.forecasting_horizon].values
        y = torch.tensor(y, dtype = torch.float32).T

        y_time = self.data.index[idx+self.window_size:idx+self.window_size+self.forecasting_horizon].values

        return x, y, list(x_time), list(y_time)
