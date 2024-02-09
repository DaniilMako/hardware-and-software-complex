# Семинар 10 от 31.10.23
# 1. Разделите данные Титаника (train.csv) на тренировочную, валидационную и тестовую часть. С помощью валидационной части подберите гиперпараметры для моделей Random Forest, XGBoost, Logistic Regression и KNN. Получите точность этих моделей на тестовой части.
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

import warnings  # чтобы откл в терминале ошибки
warnings.filterwarnings("ignore")

# pd.set_option("display.max_columns", None)  # для вывода всех столбцов

df = pd.read_csv('train.csv')
# print(df.head(5))
selected_features = df[
    ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin",
     "Embarked"]]

# Создание dummy-переменных для столбцов Name, Sex, Ticket, Cabin, Embarked
dummy_variables = pd.get_dummies(selected_features["Name"], prefix="name")
selected_features = selected_features.drop("Name", axis=1)
selected_features = pd.concat([selected_features, dummy_variables], axis=1)

dummy_variables = pd.get_dummies(selected_features["Sex"], prefix="sex")
selected_features = selected_features.drop("Sex", axis=1)
selected_features = pd.concat([selected_features, dummy_variables], axis=1)

dummy_variables = pd.get_dummies(selected_features["Ticket"], prefix="ticket")
selected_features = selected_features.drop("Ticket", axis=1)
selected_features = pd.concat([selected_features, dummy_variables], axis=1)

dummy_variables = pd.get_dummies(selected_features["Cabin"], prefix="cabin")
selected_features = selected_features.drop("Cabin", axis=1)
selected_features = pd.concat([selected_features, dummy_variables], axis=1)

dummy_variables = pd.get_dummies(selected_features["Embarked"], prefix="embarked")
selected_features = selected_features.drop("Embarked", axis=1)
selected_features = pd.concat([selected_features, dummy_variables], axis=1)


selected_features = selected_features.astype(float)
selected_features = selected_features.dropna()
# print(selected_features.head(5), "\n")

features = selected_features.drop("Survived", axis=1)
target_var = selected_features["Survived"]
# print(features.head(5))
# print(selected_features.head(5))
# print()
# Разделите данные Титаника (train.csv) на тренировочную, валидационную и тестовую часть
features_train_data, features_test_data, target_train_data, target_test_data = train_test_split(features, target_var,
                                                                                                test_size=0.2,
                                                                                                random_state=43)
# С помощью валидационной части подберите гиперпараметры для моделей...
# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=43)  # создание модели
rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}  # определение набора параметров для настройки модели: число деревьев, макс глубина деревьев
rf_grid = GridSearchCV(rf_model, rf_params, cv=5)  # передача модели, параметром и числа разбиений для кросс-валидации
rf_grid.fit(features_train_data, target_train_data)  # обучение модели
rf_best_model = rf_grid.best_estimator_  # сохранение лучшей модели

# XGBoost
from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=43)
xgb_params = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5)
xgb_grid.fit(features_train_data, target_train_data)
xgb_best_model = xgb_grid.best_estimator_

# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=43)
lr_params = {'C': [0.1, 1, 10]}  # коэффициент регуляризации
lr_grid = GridSearchCV(lr_model, lr_params, cv=5)
lr_grid.fit(features_train_data, target_train_data)
lr_best_model = lr_grid.best_estimator_

# KNN
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_params = {'n_neighbors': [3, 5, 7]}  # число соседей
knn_grid = GridSearchCV(knn_model, knn_params, cv=5)
knn_grid.fit(features_train_data, target_train_data)
knn_best_model = knn_grid.best_estimator_

# Получите точность этих моделей на тестовой части
from sklearn.metrics import accuracy_score
# в функцию accuracy_score передаем  фактические значения целевой переменной (target_test_data) и предсказанные значения, которые возвращает лучшая модель (..._best_model) для тестовых данных (features_test_data)
rf_accuracy = accuracy_score(target_test_data, rf_best_model.predict(features_test_data))
print("Точность Random Forest:", rf_accuracy)
xgb_accuracy = accuracy_score(target_test_data, xgb_best_model.predict(features_test_data))
print("Точность XGBoost:", xgb_accuracy)
lr_accuracy = accuracy_score(target_test_data, lr_best_model.predict(features_test_data))
print("Точность Logistic Regression:", lr_accuracy)
knn_accuracy = accuracy_score(target_test_data, knn_best_model.predict(features_test_data))
print("Точность KNN:", knn_accuracy)

# 2. С помощью RandomForest выберите 2, 4, 8 самых важных признаков и проверьте точность моделей только на этих признаках.
from sklearn.feature_selection import SelectKBest, f_classif

def accuracy_of_model(x):
    rf_feature_selector = SelectKBest(score_func=f_classif, k=x)
    rf_feature_selector.fit(features_train_data, target_train_data)  # подготовка данных обучения и целевых к выбору признаков
    features_train_selected = rf_feature_selector.transform(features_train_data)  # выбор отобранных признаков
    features_test_selected = rf_feature_selector.transform(features_test_data)

    # Обучение модели Random Forest на выбранных признаках
    rf_model_selected = RandomForestClassifier(random_state=43)
    rf_model_selected.fit(features_train_selected, target_train_data)

    # Оценка точности модели Random Forest на выбранных признаках
    rf_selected_accuracy = accuracy_score(target_test_data, rf_model_selected.predict(features_test_selected))
    print(f"Случайный лес по {x} признакам:", rf_selected_accuracy)


model_accuracy(2)
model_accuracy(4)
model_accuracy(8)
