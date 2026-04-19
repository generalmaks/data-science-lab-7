import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. Завантаження даних
# =========================
FILE_NAME = "Data_Set_6.xlsx"

df = pd.read_excel(FILE_NAME)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# =========================
# 2. Первинний огляд
# =========================
print("Перші 5 рядків:")
print(df.head(), "\n")

print("Розмір таблиці:")
print(df.shape, "\n")

print("Типи даних до очищення:")
print(df.dtypes, "\n")


# =========================
# 3. Очищення даних
# =========================
# У файлі є текстові пропуски типу n.a., not avilable, not available
# а також числа у форматі 3,469.00
missing_values = ["n.a.", "not avilable", "not available", "n.a", "N/A", "NA"]

df = df.replace(missing_values, np.nan)

# Прибираємо зайві пробіли у назвах колонок
df.columns = df.columns.str.strip()

# Якщо в регіонах є зайві пробіли — теж приберемо
df["SALES_BY_REGION"] = df["SALES_BY_REGION"].astype(str).str.strip()

# Список місяців
months = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
]

# Переводимо колонки місяців у числа
for col in months:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# SALES_ID теж у число
df["SALES_ID"] = pd.to_numeric(df["SALES_ID"], errors="coerce")

# Значення -1 вважаємо аномалією/пропуском
for col in months:
    df.loc[df[col] == -1, col] = np.nan

print("Кількість пропусків по колонках після очищення:")
print(df.isna().sum(), "\n")

print("Типи даних після очищення:")
print(df.dtypes, "\n")


# =========================
# 4. Заповнення пропусків
# =========================
# Заповнюємо пропуски середнім по відповідному місяцю
for col in months:
    df[col] = df[col].fillna(df[col].mean())

print("Пропуски після заповнення:")
print(df[months].isna().sum(), "\n")


# =========================
# 5. Базовий статистичний аналіз
# =========================
print("Описова статистика:")
print(df[months].describe(), "\n")

# Загальні показники по кожному запису
df["YEARLY_TOTAL"] = df[months].sum(axis=1)
df["YEARLY_MEAN"] = df[months].mean(axis=1)
df["YEARLY_STD"] = df[months].std(axis=1)

print("Перші рядки з новими показниками:")
print(df[["SALES_ID", "SALES_BY_REGION", "YEARLY_TOTAL", "YEARLY_MEAN", "YEARLY_STD"]].head(), "\n")


# =========================
# 6. Середні продажі по місяцях
# =========================
mean_sales_by_month = df[months].mean()

print("Середні продажі по місяцях:")
print(mean_sales_by_month, "\n")


# =========================
# 7. Середні продажі по регіонах
# =========================
region_mean_sales = df.groupby("SALES_BY_REGION")[months].mean()

print("Середні продажі по регіонах:")
print(region_mean_sales, "\n")


# =========================
# 8. Візуалізація
# =========================

# 8.1. Гістограми продажів по місяцях
df[months].hist(figsize=(14, 8), bins=10)
plt.suptitle("Гістограми продажів по місяцях")
plt.tight_layout()
plt.show()

# 8.2. Кількість записів по регіонах
df["SALES_BY_REGION"].value_counts().plot(kind="bar", figsize=(8, 5))
plt.title("Кількість записів по регіонах")
plt.xlabel("Регіон")
plt.ylabel("Кількість записів")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 8.3. Середні продажі по місяцях
mean_sales_by_month.plot(marker="o", figsize=(10, 5))
plt.title("Середні продажі по місяцях")
plt.xlabel("Місяць")
plt.ylabel("Середні продажі")
plt.grid()
plt.tight_layout()
plt.show()

# 8.4. Динаміка середніх продажів за регіонами
region_mean_sales.T.plot(marker="o", figsize=(12, 6))
plt.title("Динаміка середніх продажів за регіонами")
plt.xlabel("Місяць")
plt.ylabel("Середні продажі")
plt.grid()
plt.tight_layout()
plt.show()

# 8.5. Середні річні продажі за регіонами
region_total = df.groupby("SALES_BY_REGION")["YEARLY_TOTAL"].mean().sort_values(ascending=False)
region_total.plot(kind="bar", figsize=(8, 5))
plt.title("Середні річні продажі за регіонами")
plt.xlabel("Регіон")
plt.ylabel("Середній річний обсяг продажів")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# =========================
# 9. Побудова моделі
# =========================
# Будемо прогнозувати річний обсяг продажів
# за значеннями з січня по листопад
feature_months = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER"
]

X = df[feature_months]
y = df["YEARLY_TOTAL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Оцінка моделі прогнозування YEARLY_TOTAL:")
print(f"MAE  = {mae:.2f}")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R2   = {r2:.4f}\n")


# =========================
# 10. Порівняння реальних і прогнозованих значень
# =========================
results = pd.DataFrame({
    "Actual_YEARLY_TOTAL": y_test.values,
    "Predicted_YEARLY_TOTAL": y_pred
})

print("Порівняння реальних та прогнозованих значень:")
print(results.head(15), "\n")

results = results.reset_index(drop=True)

plt.figure(figsize=(10, 5))
plt.plot(results.index, results["Actual_YEARLY_TOTAL"], marker="o", label="Реальні значення")
plt.plot(results.index, results["Predicted_YEARLY_TOTAL"], marker="o", label="Прогноз моделі")
plt.title("Порівняння реальних і прогнозованих річних продажів")
plt.xlabel("Номер спостереження у тестовій вибірці")
plt.ylabel("Річний обсяг продажів")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# =========================
# 11. Коефіцієнти моделі
# =========================
coef_df = pd.DataFrame({
    "Month": feature_months,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("Коефіцієнти лінійної регресії:")
print(coef_df, "\n")

coef_df.plot(x="Month", y="Coefficient", kind="bar", figsize=(10, 5))
plt.title("Вплив місяців на прогноз річних продажів")
plt.xlabel("Місяць")
plt.ylabel("Коефіцієнт")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# =========================
# 12. Простий прогноз для середнього профілю
# =========================
average_profile = pd.DataFrame([df[feature_months].mean()])

forecast_yearly_total = model.predict(average_profile)[0]

print(f"Прогноз YEARLY_TOTAL для середнього профілю: {forecast_yearly_total:.2f}\n")


# =========================
# 13. Збереження результатів у файл
# =========================
with pd.ExcelWriter("lab7_analysis_results.xlsx") as writer:
    df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
    mean_sales_by_month.to_frame("Mean_Sales").to_excel(writer, sheet_name="Mean_By_Month")
    region_mean_sales.to_excel(writer, sheet_name="Mean_By_Region")
    results.to_excel(writer, sheet_name="Model_Results", index=False)
    coef_df.to_excel(writer, sheet_name="Model_Coefficients", index=False)

print("Результати збережено у файл: lab7_analysis_results.xlsx")