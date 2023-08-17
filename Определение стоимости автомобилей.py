#!/usr/bin/env python
# coding: utf-8

# # Определение стоимости автомобилей

# Сервис по продаже автомобилей с пробегом разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Вам нужно построить модель для определения стоимости. 
# 
# Заказчику важны:
# 
# - качество предсказания;
# - скорость предсказания;
# - время обучения.
# 
# Необходимо проверить как минимум LightGBM и одну модель не бустинг. Параметр качества- RSME, должен быть не выше 2500.

# ## Подготовка данных

# In[1]:


pip install lightgbm


# In[2]:


get_ipython().system('pip install scikit-learn==1.1.3')


# In[3]:


get_ipython().system('pip3 install catboost')


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
import time


# In[5]:


data = pd.read_csv('autos.csv.crdownload')


# In[6]:


data.head(5)


# In[7]:


data.info()


# для предсказания цены автомобиля не играют роли такие признаки как: DateCrawled,PostalCode,LastSeen,RegistrationMonth,RegistrationMonth

# In[8]:


data.drop(['DateCrawled', 'PostalCode', 'LastSeen','RegistrationMonth','DateCreated'], axis = 1, inplace= True)


# In[9]:


data.info()


# In[10]:


data.columns = data.columns.str.lower()


# In[11]:


data


# In[12]:


data.rename(columns= {'vehicletype':'vehicle_type', 'registrationyear':'registration_year','fueltype':'fuel_type', 'numberofpictures':'number_of_pictures'}, inplace=True)


# In[13]:


data.hist(figsize=(15,20))
plt.show


# По первому взгляду на графики можно сразу отметить, что точно есть какая-то ошибка или выброс в признаках RegistrationYear и Power, а еще огромное количество автомобилей с нулевой ценой. Еще вызывает подозрение признак NumberOfPictures(там что, одни нули?). ПОсмотрим колонку number_of_pictures, чтобы сразу удалить ее сразу, если там действительно нули

# In[14]:


data.number_of_pictures.describe()


# Действительно, одни нуди.удаляем этот признак

# In[15]:


data.drop('number_of_pictures', axis=1,inplace=True)


# In[16]:


data.info()


# In[17]:


data.duplicated().sum()


# In[18]:


data = data.drop_duplicates()


# In[19]:


data.reset_index(drop=True, inplace=True)


# ## Подготовка выборок

# разобьем выборку на обучающу, тестовую и валидационную.

# In[20]:


all_train, test = train_test_split(data, test_size=0.2, random_state=2023)


# In[21]:


train, valid = train_test_split(all_train, test_size=0.2, random_state=2023)


# In[22]:


valid_features = valid.drop('price', axis=1)
test_features = test.drop('price', axis=1)


# In[23]:


valid_target = valid['price']
test_target = test['price']


# Обработаем порпуски и выбросы в обучающей выборке. посмотрим внимательнее на данные, где год регистрации выше текущего (2023)

# In[24]:


train.info()


# In[25]:


train.hist(figsize=(13,8))
plt.show()


# In[26]:


train.loc[train['registration_year']>2023]


# Видим, что эти данные неинформативны. Логичнее всего будет их удалить. Подозреваю, что была какая-то ошибка в выгрузке данных или ранее. Теперь посмотрим данные с очень ранними годами регистрации. Первый автомобиль был изобретен в 1886 году. Автомобили старше 30 лет уже считаются раритетными. Т.о. делаем вывод, что со 100% вероятностью все года регистрации ранее 1886 года - ошибки, а все, что между 1886 и 1989 это либо ошибка либо старые машины и раритет (разница в цене может быть колоссальная). Предполагаю, что данные за 2019 год.

# Также данные выглядят скорее результатом ошибки. Подлежат удалению. Теперь посмотрим что там в промежутке 1886 - 1993

# In[27]:


train.loc[train['registration_year']<1886]


# In[28]:


short_data = train.query('registration_year>1886 & registration_year<1989')


# In[29]:


short_data


# In[30]:


short_data['registration_year'].hist()
plt.show


# In[31]:


short_data.sort_values(by='registration_year', ascending =False)


# Выглядит так, что все, что меньше 1950 года- что-то ошибочное. По крайней мере 1910 год. Оставим срез с 1910 по 1950 год. 

# In[32]:


train = train.query('registration_year>1910 & registration_year<=2016')


# Теперь расмотрим все данные, у которых слишком низкая цена. Поставим границу 50 евро. 

# In[33]:


train.query('price<50')


# Видим около 6,6 тысяч записей с ценой меньше 100. Много цен 0 и 1 евро. Откуда эти значения- узнать бы у коллег, но т.к. такой возможности нет, то считаю, что эти данные нужно удалить. Вряд ли хоть кому-то из пользвоателей захочется увидеть предсказание от модели "ваш автомобиль стоит 0 евро". 

# In[34]:


train = train.loc[train['price']>49]


# также по гистограммам было видно, что есть выбросы в признаке power

# на данный момент самый мощный автомобиль имеет 2000 лошадиные силы. Посмотрим на данные, у которых мощности двигателя указаны выше 2000

# In[35]:


train.loc[train['power']>2100]


# In[36]:


train.info()


# Таких автомобилей всего 70 на 191 тысячe (меньше 0,04%). Целесообразнее удалить эти данные из датасета, они все равно мало на что повлияют

# In[37]:


train = train.loc[train['power']<2100]


# In[38]:


train.power.hist()
plt.title('Распределение по мощностям')
plt.xlabel('Мощность')
plt.ylabel('Количество')
plt.show


# In[39]:


train.query('power>500')


# всего 190 значений. т.к. мощность обычных автомобилей - не выше 250, все, что выше 500 уже являеься спорткарами, однако глядя на данные мы видим, что модели и бренды автомобилей никак не относятся к спорткарам, это обычные автомобили. исправлять на что-то среднее не представляется целесообразным, удаляю

# посмотрим теперь авто с очень низкими мощностями.

# In[40]:


train = train.query('power<=500')


# In[41]:


train.query('power<=10')


# In[42]:


power_data = train.query('power<=10')


# In[43]:


power_data.hist(figsize=(20,15))
plt.show


# Похоже, автомобили с нулевыми мощностями- автомобили на разборку, те, которые не на ходу. Тогда они должны стоить дешевле обычных авто. 

# Сравним стоимости автомобилей с очень низкими мощностями и нормальными мощностями на нормализованной гистограмме

# In[44]:


train[train['power']<10]['price'].hist(bins=35, alpha=0.5, color='green', label= 'Низкая мощность', density = True)
train[train['power']>10]['price'].hist(bins=35, alpha=0.5, color='blue',label= 'Высокая мощность', density = True)
plt.show
plt.legend()
plt.title('Сревнение цен разномощностных авто')
plt.xlabel('Цена')
plt.ylabel('Частота встречаемости')


# Из графика видим, что автомобили с нулевыми мощностями и низкой ценой почти в 2 раза больше автомобилей с нормальной мощностью. Делаем вывод, что скорее всего предположение, что это авто на разборку верно.

# In[45]:


train.info()


# выведем данные, у которых есть пропуски сразу по 3 признакам- vehicle_type, gearbox, model

# In[46]:


train.query('vehicle_type.isnull()&gearbox.isnull()&model.isnull()')


# в этих данных также есть пропуски в fuel_type и repaired. Эти строки также подлежат удалению, т.к. достаточно бессмысленно ждать достоверных предсказаний только по бренду машины, году регистрации и киллометражу. Слишком мало информации.

# обработаем пропуски в признаке vehicle_type

# In[47]:


train.dropna(axis=0, how='all', subset=['vehicle_type','gearbox','model'], inplace=True)


# In[48]:


train[train.vehicle_type.isnull()]


# In[49]:


train.vehicle_type.unique()


# Заменим пропуски на заглушки "other" в столбце vehicle_type

# In[50]:


train.loc[train['vehicle_type'].isnull(), 'vehicle_type'] = 'other'


# In[51]:


train[train.gearbox.isnull()]


# заменим пропуски в gearbox на manual

# In[52]:


train.loc[train['gearbox'].isnull(), 'gearbox'] = 'manual'


# In[53]:


train.info()


# In[54]:


train[train.model.isnull()]


# In[55]:


train[train.model == 'other']


# Думаю, целесообразно поставить заглушку 'other' в колонке model, 'petrol' в колонке 'fuel_type'

# In[56]:


train.loc[train['model'].isnull(), 'model'] = 'other'


# In[57]:


train.loc[train['fuel_type'].isnull(), 'fuel_type'] = 'petrol'


# In[58]:


train.info()


# Теперь обработаем пропуски в признаке repaired. ПОсмотрим,какие уникальные значения есть в этой колонке

# In[59]:


train.repaired.unique()


# Думаю, здесь лучше поставить заглушку "unknown". Т.к. факт того, была ли машина в ремонте или нет влияет на ее итоговую стоимость.

# In[60]:


train.loc[train['repaired'].isnull(),'repaired'] = 'unknown'


# проверим, что нет больше пропусков и далее проверим данные на дубликаты

# In[61]:


train.info()


# In[62]:


train_features = train.drop('price',axis=1)
train_target = train['price']


# Теперь переведем категориальные данные в численные. Посмотрим, сколько значений в каждом категориальном признаке

# In[63]:


len(data.vehicle_type.unique())


# In[64]:


len(data.gearbox.unique())


# In[65]:


len(data.model.unique())


# In[66]:


len(data.fuel_type.unique())


# In[67]:


len(data.brand.unique())


# In[68]:


len(data.repaired.unique())


# Делать огромное количество признаков не хочется, поэтому для признаков vehicle_type,gearbox,fuel_type и repaired применим метод OHE,а для model и brand - ordinary

# In[69]:


ohe_names = ['vehicle_type','gearbox','fuel_type','repaired']


# In[70]:


encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
encoder.fit(train_features[ohe_names])
ohe_train = encoder.transform(train_features[ohe_names])
ohe_data_train = pd.DataFrame(ohe_train, columns=encoder.get_feature_names(ohe_names), index=train_features.index)

ohe_valid = encoder.transform(valid_features[ohe_names])
ohe_data_valid = pd.DataFrame(ohe_valid, columns=encoder.get_feature_names(ohe_names), index=valid_features.index)

ohe_test = encoder.transform(test_features[ohe_names])
ohe_data_test = pd.DataFrame(ohe_test, columns=encoder.get_feature_names(ohe_names), index=test_features.index)


# In[71]:


num_train = train_features.drop(ohe_names, axis=1)
num_valid = valid_features.drop(ohe_names, axis=1)
num_test = test_features.drop(ohe_names, axis=1)


# In[72]:


features_train = pd.concat([num_train,ohe_data_train], axis=1)
features_valid = pd.concat([num_valid, ohe_data_valid], axis=1)
features_test = pd.concat([num_test, ohe_data_test], axis=1)


# Теперь признаки model и brand

# In[73]:


encoder = OrdinalEncoder(handle_unknown= 'use_encoded_value',unknown_value=300)
encoder.fit(features_train[['model']])
features_train['model'] = encoder.transform(features_train[['model']])
features_valid['model'] = encoder.transform(features_valid[['model']])
features_test['model'] = encoder.transform(features_test[['model']])


# In[74]:


encoder = OrdinalEncoder(handle_unknown= 'use_encoded_value',unknown_value=300)
encoder.fit(features_train[['brand']])
features_train['brand'] = encoder.transform(features_train[['brand']])
features_valid['brand'] = encoder.transform(features_valid[['brand']])
features_test['brand'] = encoder.transform(features_test[['brand']])


# Теперь промасштабируем признаки

# In[75]:


numeric = ['registration_year', 'power', 'kilometer']
scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
features_test[numeric] = scaler.transform(features_test[numeric])


# ## Обучение моделей

# Напишем функцию вычисления RMSE

# In[76]:


def RMSE(target,predictions):
    MSE = mean_squared_error(target, predictions)
    return MSE ** 0.5
RMSE_score = make_scorer(RMSE, greater_is_better=False)


# In[77]:


model_list=[]


# In[78]:


def best_model(model, params, features_train, target_train, features_valid, target_valid):
    start_time = time.time()
    model_name = GridSearchCV(model, param_grid = params, cv=5,
                                            scoring = RMSE_score, n_jobs=-1)
    model_name.fit(features_train,target_train)
    first_time = time.time() - start_time
    start_time = time.time()
    predict = model_name.predict(features_valid)
    second_time = time.time() - start_time
    model_list.append(model_name)
    n = len(model_list)
    return first_time, second_time, RMSE(target_valid, predict),n


# Модель линейной регресии с регуляризацией

# In[79]:


ridge = Ridge()
ridge_param = {'alpha': np.logspace(-4, 4, 20)}


# In[80]:


ridge_model = best_model(ridge, ridge_param, features_train, train_target, features_valid, valid_target)


# In[81]:


ridge_model


# Время обучения- 20 секунд, время предсказания- 0.0083, метрика- 11806.73599709747. 

# Модель LGBM

# In[82]:


lgbm_params = {
    'n_estimators': [10,50,100],
    'max_depth': [15,30],
    'num_leaves': [10,20,31],
    'learning_rate': [0.1]
}
lgbm_reg_model = LGBMRegressor(random_state=2023)


# In[83]:


lgbm_model = best_model(lgbm_reg_model, lgbm_params, features_train, train_target, features_valid, valid_target )


# In[84]:


lgbm_model


# In[85]:


cat_param = { 'learning_rate': [0.1],
    'iterations': [10,50,100],
    'max_depth': [10,20]}
cat_model = CatBoostRegressor(random_state=2023)


# In[86]:


cat_model = best_model(cat_model, cat_param, features_train, train_target, features_valid, valid_target )


# In[87]:


cat_model


# In[88]:


constant_model = DummyRegressor(strategy='mean')


# In[89]:


constant_model.fit(features_train,train_target)
constant_pred = constant_model.predict(features_valid)
RMSE(valid_target,constant_pred)


# ## Анализ моделей

# In[90]:


model_table = pd.DataFrame([ridge_model, lgbm_model, cat_model], columns=['Время обучения', 'Время предсказания', 'RMSE','номер модели'], index=['Ridge','LGBM','CatBoost'])


# In[91]:


model_table


# ## Тестирование лучшей модели

# Были проверены 3 модели: линейная регрессия, Модель LGBM, catboost. По результатам метрики RMSE лучше всего сея показали модели LGBM и catboost. В обоих случаях метрика ниже 2500. Но Так как в условиях было подобрать модель с наименьшим временем обучения и предсказания, делаем выбор в пользу catboost. Проверка случайно моделью показала, что предсказания адекватны. ПРоверим итоговую модель на тестовой выборке

# In[92]:


end_predict = model_list[2].predict(features_test)


# In[93]:


RMSE (test_target, end_predict)


# Модель выбора- catboost
