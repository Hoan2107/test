import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Đọc dữ liệu từ file CSV
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Tạo và huấn luyện mô hình dự đoán
def train_model(data):
    X = data[['Số tầng', 'Số phòng ngủ', 'Diện tích']]  # Chọn các tính năng đầu vào
    y = data['Giá (triệu đồng/m2)']  # Chọn biến mục tiêu
    model = LinearRegression()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Đánh giá độ chính xác
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    st.write("Đánh giá độ chính xác của mô hình:")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R-squared (R²): {r2:.2f}")

    return model

# Load dữ liệu
data = load_data("file_new/du_lieu_lam_sach.csv")

# Huấn luyện mô hình và đánh giá độ chính xác
model = train_model(data)

# Giao diện Streamlit
st.title("Dự Đoán Giá Nhà")

st.write("Đây là một số dòng dữ liệu trong file CSV:")
st.write(data.head())

# Thêm các trường nhập liệu
st.sidebar.header("Nhập thông tin để dự đoán giá nhà")
num_floors = st.sidebar.number_input("Số tầng", min_value=1, max_value=11)
num_bedrooms = st.sidebar.number_input("Số phòng ngủ", min_value=1, max_value=11)
area = st.sidebar.number_input("Diện tích (m2)", min_value=10, max_value=200)

# Thêm trường nhập liệu chọn quận
selected_district = st.sidebar.selectbox("Chọn Quận", data['Quận'].unique())

# Dự đoán khi người dùng nhấn nút
if st.sidebar.button("Dự Đoán"):
    input_data = [[num_floors, num_bedrooms, area]]
    prediction = model.predict(input_data)
    st.sidebar.write(f"Giá Nhà Dự Đoán: {prediction[0]:.2f} triệu đồng/m2")

    # Lọc dữ liệu dựa trên quận và diện tích đã chọn
    filtered_data = data[(data['Quận'] == selected_district) & 
                         (data['Diện tích'] >= area - 5) & (data['Diện tích'] <= area + 5) & 
                         (data['Số tầng'] >= num_floors - 2) & (data['Số tầng'] <= num_floors + 2) & 
                         (data['Số phòng ngủ'] >= num_bedrooms - 2) & (data['Số phòng ngủ'] <= num_bedrooms + 2)]

    # Độ sai lệch cho giá (có thể điều chỉnh)
    price_tolerance = 2.0

    # Lọc ngôi nhà phù hợp trong quận đã chọn
    similar_houses = filtered_data[
        (filtered_data['Giá (triệu đồng/m2)'] >= prediction[0] - price_tolerance) &
        (filtered_data['Giá (triệu đồng/m2)'] <= prediction[0] + price_tolerance)
    ]

    st.sidebar.write(f"Các ngôi nhà phù hợp trong quận {selected_district} với diện tích gần {area} m2:")
    st.sidebar.write(similar_houses[['Địa chỉ', 'Giá (triệu đồng/m2)', 'Diện tích', 'Số phòng ngủ', 'Số tầng']])
