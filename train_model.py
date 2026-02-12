import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# โหลดข้อมูล
data = pd.read_csv("data.csv")

# กำหนด X และ y
X = data[['num_aircon',
          'aircon_hours',
          'num_people',
          'last_3_month_avg',
          'num_appliances']]

y = data['electricity_bill']

# แบ่ง train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# สร้างโมเดล
model = LinearRegression()
model.fit(X_train, y_train)

# ทำนาย
y_pred = model.predict(X_test)

# ประเมินผล
print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# บันทึกโมเดล
joblib.dump(model, "electricity_model.pkl")

print("Model saved successfully!")
