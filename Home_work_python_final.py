'''Задание 44

В ячейке ниже представлен код генерирующий DataFrame, которая состоит всего из 1 столбца. 
Ваша задача перевести его в one hot вид. Сможете ли вы это сделать без get_dummies?'''

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import random

lst = ['robot'] * 10
lst += ['human'] * 10
random.shuffle(lst)
data = pd.DataFrame({'whoAmI': lst})

# C использованием get_dummies
one_hot = pd.get_dummies(data['whoAmI'], sparse=False)

data = pd.concat([data, one_hot], axis=1)
print(data.head())


# Без использования get_dummies
enc = OneHotEncoder()
enc.fit(data[['whoAmI']])

one_hot = enc.transform(data[['whoAmI']])
cols = enc.get_feature_names_out(['whoAmI'])

one_hot_df = pd.DataFrame(one_hot.toarray(), columns=cols)
print(one_hot_df.head())