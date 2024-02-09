# Лабораторная 9.1
# 1. Загрузить файл, разделить его на train и test. Для test взять 10% случайно выбранных строк таблицы.
# 2. Обучить модели: Decision Tree, XGBoost, Logistic Regression из библиотек sklearn и xgboost. Обучить модели предсказывать столбец label по остальным столбцам таблицы.
# 3. Наладить замер Accuracy - доли верно угаданных ответов.
# 4. Точности всех моделей не должны быть ниже 85%
# 5. С помощью Decision Tree выбрать 2 самых важных признака и проверить точность модели, обученной только на них.
import pandas as pd
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
# 1. Загрузить файл, разделить его на train и test. Для test взять 10% случайно выбранных строк таблицы.
df = pd.read_csv('titanic_prepared.csv')

selected_features = df.drop('label', axis=1)
target_variable = df['label']
# print(df)
# print(selected_features)
features_train_data, features_test_data, target_train_data, target_test_data = train_test_split(selected_features,
                                                                                                target_variable,
                                                                                                test_size=0.1,
                                                                                                random_state=43)
# 2. Обучить модели: Decision Tree, XGBoost, Logistic Regression из библиотек sklearn и xgboost. Обучить модели предсказывать столбец label по остальным столбцам таблицы.
# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=43)
dt_model.fit(features_train_data, target_train_data)
dt_predictions = dt_model.predict(features_test_data)

# XGBoost
from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=43)
xgb_model.fit(features_train_data, target_train_data)
xgb_predictions = xgb_model.predict(features_test_data)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=43)
lr_model.fit(features_train_data, target_train_data)
lr_predictions = lr_model.predict(features_test_data)

# 3. Наладить замер Accuracy - доли верно угаданных ответов.
from sklearn.metrics import accuracy_score

xgb_accuracy = accuracy_score(target_test_data, xgb_predictions)
dt_accuracy = accuracy_score(target_test_data, dt_predictions)
lr_accuracy = accuracy_score(target_test_data, lr_predictions)

# 4. Точности всех моделей не должны быть ниже 85%
# Вывод точности моделей
print("Точность Decision Tree:", dt_accuracy)
print("Точность XGBoost:", xgb_accuracy)
print("Точность Logistic Regression:", lr_accuracy)

# 5. С помощью Decision Tree выбрать 2 самых важных признака и проверить точность модели, обученной только на них.
# Выбор 2 самых важных признаков с помощью Decision Tree
from sklearn.feature_selection import SelectKBest, f_classif

dt_feature_selector = SelectKBest(score_func=f_classif, k=2)
dt_feature_selector.fit(features_train_data, target_train_data)
features_train_selected = dt_feature_selector.transform(features_train_data)
features_test_selected = dt_feature_selector.transform(features_test_data)

# Обучение модели Decision Tree на выбранных признаках
dt_model_selected = DecisionTreeClassifier(random_state=43)
dt_model_selected.fit(features_train_selected, target_train_data)

# Оценка точности модели Decision Tree на выбранных признаках
dt_selected_accuracy = accuracy_score(target_test_data, dt_model_selected.predict(features_test_selected))
print("Дерево решений по 2 признакам:", dt_selected_accuracy)
