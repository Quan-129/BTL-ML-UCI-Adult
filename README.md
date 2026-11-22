# Bài tập lớn Học máy — UCI Adult

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Quan-129/BTL-ML-UCI-Adult/blob/main/notebooks/Adult_BTL.ipynb)

**Môn học:** CO3117, học kỳ 261, năm học 2026–2027.  
**Giảng viên:** Trương Vĩnh Lân.  
**Nhóm:** A01.

| Thành viên | MSSV | Email |
|---|---|---|
| Hồng Chấn Phước | 2352963 | phuoc.hongchan@hcmut.edu.vn |
| Trần Bá Minh Quân | 2353015 | quan.tranbaminh12@hcmut.edu.vn |
| Đinh Ngụy Nguyệt Hà | 2352286 | ha.dinhnguy2005@hcmut.edu.vn |

## Mục tiêu

Dự đoán thu nhập `<=50K` hoặc `>50K` từ 14 thuộc tính UCI Adult. So sánh Logistic Regression, LinearSVC, Random Forest và MLP; khảo sát tác động của cách điền thiếu, encoding, scaling và PCA.

## Thực hiện

Mở `notebooks/Adult_BTL.ipynb` bằng Google Colab và chọn **Runtime → Run all**. Notebook tự chuẩn bị thư viện, tải dữ liệu và tạo các đầu ra. Đối với môi trường Python riêng, cài các thư viện trong `requirements.txt`.

Dữ liệu: https://archive.ics.uci.edu/dataset/2/adult  
ZIP: https://archive.ics.uci.edu/static/public/2/adult.zip

## Cấu trúc

- `notebooks/`: notebook thực nghiệm.
- `modules/`: mã nguồn xử lý dữ liệu, huấn luyện, đánh giá và xuất báo cáo.
- `configs/`: cấu hình pipeline và mô hình.
- `data/`: dữ liệu gốc và các tập đã chia.
- `features/`: ba bộ đặc trưng baseline, PCA90 và PCA95; các tệp `.npy` đọc bằng `allow_pickle=False`.
- `artifacts/`: pipeline tiền xử lý và mô hình đã lưu.
- `results/`: số liệu, biểu đồ và dự đoán của từng giai đoạn.
- `reports/`: [PDF](reports/BTL_ML_Bao_Cao.pdf), [ZIP LaTeX](reports/BTL_ML_LaTeX_Final.zip) và nguồn tách theo từng mục.

Các kết quả của báo cáo tương ứng với môi trường trong `environment_tested.json` và `requirements-tested.txt`. Seed được cố định bằng 42. Tiền xử lý được fit riêng trong từng fold CV; các ma trận fit trên toàn bộ train chỉ dùng để xuất đặc trưng.

PDF do notebook xuất được dàn bằng ReportLab. Nguồn LaTeX sử dụng XeLaTeX trên Overleaf; cách ngắt dòng và số trang có thể khác.

## Phân công công việc

| Thành viên | Tỷ lệ phân công | Mức hoàn thành |
|---|---:|---:|
| Hồng Chấn Phước | 33% | 100% |
| Trần Bá Minh Quân | 34% | 100% |
| Đinh Nguy Nguyệt Hà | 33% | 100% |

## Liên kết

- [Repository GitHub](https://github.com/Quan-129/BTL-ML-UCI-Adult)
- [Chạy notebook trên Google Colab](https://colab.research.google.com/github/Quan-129/BTL-ML-UCI-Adult/blob/main/notebooks/Adult_BTL.ipynb)
- [Báo cáo PDF](reports/BTL_ML_Bao_Cao.pdf)
- [Mã nguồn LaTeX](reports/latex/main.tex)
