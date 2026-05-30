# FraudGuard

Notebook Machine Learning cho bài toán phát hiện gian lận thẻ tín dụng trên dataset Kaggle `mlg-ulb/creditcardfraud`.

## Mục Tiêu

Mục tiêu của project là xây dựng và phân tích pipeline phát hiện giao dịch gian lận:

- Khám phá dữ liệu và mức độ mất cân bằng nhãn.
- Tiền xử lý dữ liệu đúng cách, tránh data leakage.
- Huấn luyện và so sánh nhiều mô hình Machine Learning.
- Đánh giá bằng các metric phù hợp với bài toán fraud detection.
- Xuất biểu đồ `.pdf` để dùng trực tiếp trong báo cáo.

## Dataset

Dataset sử dụng:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

File cần có:

```text
data/creditcard.csv
```

Notebook cũng kiểm tra thêm:

- `creditcard.csv` ở thư mục gốc project.
- Đường dẫn Kaggle cũ nếu chạy trên Kaggle Notebook.

Dataset có 31 cột:

- `Time`: thời gian tính từ giao dịch đầu tiên.
- `Amount`: số tiền giao dịch.
- `V1` đến `V28`: đặc trưng đã được PCA.
- `Class`: nhãn, `0` là normal và `1` là fraud.

Điểm quan trọng: fraud chỉ chiếm khoảng `0.17%`, nên đây là bài toán mất cân bằng dữ liệu rất nặng.

## Cách Chạy Local

Tạo môi trường Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài thư viện:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tải dataset bằng Kaggle CLI:

```powershell
pip install kaggle
mkdir data
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

Hoặc tải thủ công từ Kaggle rồi đặt file tại:

```text
data/creditcard.csv
```

Mở notebook:

```powershell
jupyter notebook fraudguard.ipynb
```

Trong Jupyter, chọn kernel:

```text
FraudGuard Python 3.11
```

## Cấu Trúc Quan Trọng

```text
fraudguard.ipynb     Notebook chính
requirements.txt     Danh sách thư viện cần cài
figures/             Nơi lưu biểu đồ PDF xuất từ notebook
data/                Nơi đặt dataset, không commit lên git
```

## Luồng Phân Tích Trong Notebook

1. Import thư viện và cấu hình lưu biểu đồ.
2. Load dataset từ local hoặc Kaggle path.
3. EDA: kiểm tra shape, missing values, phân phối nhãn, Amount, Time.
4. Tạo biểu đồ phục vụ báo cáo và lưu PDF.
5. Chia train/test bằng `stratify=y`.
6. Chuẩn hóa `Amount` và `Time` bằng `StandardScaler`, fit trên train và transform trên test.
7. Huấn luyện baseline `LogisticRegression`.
8. Huấn luyện `RandomForestClassifier`.
9. Tune threshold cho Random Forest.
10. Thử `SMOTE`.
11. Huấn luyện `XGBoost` với `scale_pos_weight`.
12. So sánh mô hình bằng Precision, Recall, F1, ROC-AUC, PR-AUC.
13. Chạy Stratified Cross-Validation.
14. Tune threshold cho XGBoost.
15. Xuất kết luận gợi ý cho báo cáo.

## Metric Cần Ưu Tiên

Không nên dùng Accuracy làm metric chính vì dữ liệu quá mất cân bằng. Một mô hình dự đoán tất cả là normal vẫn có accuracy rất cao nhưng không phát hiện fraud.

Metric nên nhấn mạnh:

- `Precision`: trong các giao dịch bị báo fraud, bao nhiêu giao dịch thật sự là fraud.
- `Recall`: trong tất cả fraud thật, mô hình bắt được bao nhiêu.
- `F1-score`: cân bằng giữa Precision và Recall.
- `PR-AUC`: metric rất quan trọng cho dữ liệu mất cân bằng.
- `ROC-AUC`: vẫn báo cáo được, nhưng không nên là metric duy nhất.

## Mô Hình Đang So Sánh

- `LogisticRegression(class_weight='balanced')`: baseline tuyến tính.
- `RandomForestClassifier(class_weight='balanced')`: mô hình cây ensemble.
- `RandomForestClassifier + SMOTE`: thử oversampling fraud class.
- `XGBoost(scale_pos_weight=...)`: mô hình boosting, thường cho kết quả tốt nhất trong notebook.

## Baseline Kết Quả Đã Lưu Trước Đó

| Model | PR-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Random Forest balanced | 0.8425 | 0.7961 | 0.8367 | 0.8159 |
| Random Forest + threshold tuning | - | 0.7905 | 0.8469 | 0.8177 |
| Random Forest + SMOTE | - | 0.4118 | 0.8571 | 0.5563 |
| XGBoost balanced | 0.8680 | 0.8058 | 0.8469 | 0.8259 |

Nhận xét quan trọng: SMOTE tăng Recall nhẹ nhưng làm Precision giảm mạnh, nghĩa là số báo động giả tăng nhiều. XGBoost đang là hướng tốt hơn để tối ưu tiếp.

## Biểu Đồ PDF

Khi chạy notebook, mỗi cell tạo biểu đồ sẽ lưu biểu đồ ngay trong cell đó vào thư mục `figures/`.

Các biểu đồ chính:

- `01_class_imbalance.pdf`: mất cân bằng nhãn, có cả thang log.
- `02_amount_time_by_class.pdf`: Amount và Time theo từng class.
- `03_feature_correlation.pdf`: feature tương quan mạnh nhất với fraud.
- `04_top_feature_distributions.pdf`: phân phối các PCA feature nổi bật.
- `05_model_comparison.pdf`: so sánh Precision, Recall, F1, PR-AUC.
- `06_roc_pr_comparison.pdf`: so sánh ROC curve và Precision-Recall curve.
- `07_xgboost_threshold_tuning.pdf`: tuning threshold cho XGBoost.
- `08_train_test_class_distribution.pdf`: kiểm tra phân phối nhãn sau train/test split.
- `09_cross_validation_results.pdf`: kết quả Stratified Cross-Validation.
- `10_xgboost_feature_importance.pdf`: feature importance của XGBoost.
- `11_xgboost_tuned_confusion_matrix.pdf`: confusion matrix của XGBoost sau tuning threshold.
- `12_original_class_distribution.pdf`: biểu đồ phân phối nhãn phiên bản ban đầu.
- `13_original_amount_time_distribution.pdf`: biểu đồ Amount/Time phiên bản ban đầu.
- `14_random_forest_confusion_matrix.pdf`: confusion matrix của Random Forest.
- `15_random_forest_roc_pr_curves.pdf`: ROC và PR curve của Random Forest.
- `16_random_forest_feature_importance.pdf`: feature importance của Random Forest.
