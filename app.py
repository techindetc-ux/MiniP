import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load("electricity_model.pkl")

st.title("ระบบทำนายค่าไฟฟ้ารายเดือน")

st.write("กรอกข้อมูลเพื่อทำนายค่าไฟเดือนถัดไป")

num_aircon = st.number_input("จำนวนเครื่องปรับอากาศ", min_value=0)
aircon_hours = st.number_input("ชั่วโมงเปิดแอร์ต่อวัน", min_value=0.0)
num_people = st.number_input("จำนวนคนในบ้าน", min_value=0)
last_3_month_avg = st.number_input("ค่าไฟเฉลี่ย 3 เดือนล่าสุด (บาท)", min_value=0.0)
num_appliances = st.number_input("จำนวนเครื่องใช้ไฟฟ้าหลัก", min_value=0)

if st.button("ทำนายค่าไฟ"):
    input_data = np.array([[num_aircon,
                            aircon_hours,
                            num_people,
                            last_3_month_avg,
                            num_appliances]])

    prediction = model.predict(input_data)

    st.success(f"ค่าไฟเดือนถัดไปโดยประมาณ: {prediction[0]:,.2f} บาท")
