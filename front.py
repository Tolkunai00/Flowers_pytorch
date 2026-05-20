import streamlit as st
import requests

st.title("Flowers Project")
st.header("Загрузите изображение")

api_url = "http://127.0.0.1:8003/predict"
file = st.file_uploader("Выберите", type=["png", "jpg", "jpeg"])

if file is not None:
    st.image(file, caption="Загруженное изображение", width=200)

    if st.button("Predict"):
        try:
            files = {
                "image": ("image.jpg", file.getvalue(), "image/jpeg")
            }

            response = requests.post(api_url, files=files, timeout=10)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Результат: {result['predict']}")
            else:
                st.error(f"Ошибка API: {response.status_code}")

        except requests.exceptions.RequestException:
            st.error("Не удалось подключиться к API")