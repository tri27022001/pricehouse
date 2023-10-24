from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Đường dẫn đến file pkl cho các mô hình
model_paths = {
    'model1': 'decision_tree_model.pkl'
    # 'model2': 'model2.pkl',
    # 'model3': 'model3.pkl',
    # 'model4': 'model4.pkl',
    # 'model5': 'model5.pkl',
    # 'model6': 'model6.pkl'
}

# Định nghĩa trang chủ
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Lấy mô hình được chọn từ giao diện
        selected_model = request.form['model']

        # Tải mô hình từ file pkl
        model = joblib.load(model_paths[selected_model])

        # Lấy dữ liệu đầu vào từ giao diện
        input_features = []
        for i in range(1, 17):
            feature_name = f'feature{i}'
            input_value = request.form.get(feature_name)

            if not input_value:
                # Nếu input trống, hiển thị thông báo và không thực hiện dự đoán
                return render_template('index.html', predicted_price=None, message="Enter input please!")

            input_features.append(float(input_value))

        # Dự đoán giá nhà
        predicted_price = model.predict([input_features])

        return render_template('index.html', predicted_price=predicted_price[0])
    return render_template('index.html', predicted_price=None)

if __name__ == '__main__':
    app.run(debug=True)
