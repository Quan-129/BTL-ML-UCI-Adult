# Đặc trưng sau bước 4

- `baseline/`, `pca_90/`, `pca_95/`: mỗi thư mục có X_train.npy, X_validation.npy, X_test.npy (float32), feature_names.json và manifest.json.
- `y_train.npy`, `y_validation.npy`, `y_test.npy`: int8, <=50K=0, >50K=1.
- `source_row_train.npy`, `source_row_validation.npy`, `source_row_test.npy`: int64, khớp metadata ở data/splits/.
- `manifest.json`: toàn bộ cấu hình, phiên bản thư viện, tên đặc trưng, nguồn train và SHA-256.

Tải bằng numpy.load(..., allow_pickle=False). Tất cả X không NaN/Inf. Các đối tượng imputer/encoder/scaler/PCA chỉ fit trên train.

Các X này được fit từ toàn bộ train cho quy trình holdout; KHÔNG dùng trực tiếp để tính CV. Ở bước 5 phải ghép build_preprocessor(config) chưa fit với mô hình rồi fit lại trên raw train trong từng fold theo nhóm.
