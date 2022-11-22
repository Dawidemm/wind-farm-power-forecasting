import pandas as pd
import numpy as np
import tensorflow as tf

class PrepareDataset():

    def __init__(self, dataset: pd.DataFrame) -> None:
        
        self.dataset = dataset
        self.lenght = len(self.dataset)

    def early_preparation(self):
        
        self.unix = self.dataset['Time - Unix Format']
        self.dataset = self.dataset.drop(['Time - Unix Format'], axis=1)

        self.wind_speed = self.dataset.pop('Wind Speed [m/s]')
        wind_degree = self.dataset.pop('Wind Degree [°]') * np.pi/180

        self.dataset['Wind in axis X'] = self.wind_speed * np.cos(wind_degree)
        self.dataset['Wind in axis Y'] = self.wind_speed * np.sin(wind_degree)

        date_time = pd.to_datetime(self.dataset.pop('Time - y/m/d/h Format'), format='%Y.%m.%d %H')
        timestamp = date_time.map(pd.Timestamp.timestamp)
        day = 24*60*60
        year = (365.2425)*day

        self.dataset['Day sin'] = np.sin(timestamp * (2 * np.pi / day))
        self.dataset['Day cos'] = np.cos(timestamp * (2 * np.pi / day))
        self.dataset['Year sin'] = np.sin(timestamp * (2 * np.pi / year))
        self.dataset['Year cos'] = np.cos(timestamp * (2 * np.pi / year))

        self.date_time = date_time

        return self

    def set_rows_to_zeros(self):

        for i in range(len(self.dataset.index)):

            if self.dataset.at[i, 'Total Average Power [W]'] == 0:

                self.dataset.at[i, 'Temperature [K]'] = 0
                self.dataset.at[i, 'Dew Point [K]'] = 0
                self.dataset.at[i, 'Pressure [hPa]'] = 0
                self.dataset.at[i, 'Humidity [%]'] = 0
                self.dataset.at[i, 'Cloudiness [%]'] = 0
                self.dataset.at[i, 'Wind in axis X'] = 0
                self.dataset.at[i, 'Wind in axis Y'] = 0
                self.dataset.at[i, 'Day sin'] = 0
                self.dataset.at[i, 'Day cos'] = 0
                self.dataset.at[i, 'Year sin'] = 0
                self.dataset.at[i, 'Year cos'] = 0

        return self

    def split_power(self):

        self.power = np.array(self.dataset.pop('Total Average Power [W]'))//(10**6)

        return self

    def standard_scaled_dataset(self):

        self.dataset_mean = self.dataset.mean()
        self.dataset_std = self.dataset.std()
        return self

    def train(self) -> tuple:

        train_dataset = self.dataset[0:int(self.lenght * 0.7)]
        train_dataset = (train_dataset - self.dataset_mean) / self.dataset_std
        self.train_dataset = np.array(train_dataset).reshape(train_dataset.shape[0], train_dataset.shape[1], 1)

        return self.train_dataset, self.power[0:int(self.lenght * 0.7)]

    def val(self) -> tuple:

        val_dataset = self.dataset[int(self.lenght * 0.8):int(self.lenght * 0.9)]
        val_dataset = (val_dataset - self.dataset_mean) / self.dataset_std
        self.val_dataset = np.array(val_dataset).reshape(val_dataset.shape[0], val_dataset.shape[1], 1)

        return self.val_dataset, self.power[int(self.lenght * 0.8):int(self.lenght * 0.9)]

    def test(self) -> tuple:
        
        test_dataset = self.dataset[int(self.lenght * 0.9):]
        test_dataset = (test_dataset - self.dataset_mean) / self.dataset_std
        self.test_dataset = np.array(test_dataset).reshape(test_dataset.shape[0], test_dataset.shape[1], 1)

        return self.test_dataset, self.power[int(self.lenght * 0.9):]