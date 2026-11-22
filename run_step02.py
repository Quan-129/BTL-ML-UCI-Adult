from modules.data_loader import download_adult, load_adult
from modules.eda import analyze_train, update_latex

if __name__ == "__main__":
    download_adult("data")
    train, _ = load_adult("data")
    analyze_train(train)
    update_latex()
    print("EDA và các mục LaTeX đã cập nhật. Dùng notebook để tái dựng khung và tạo ZIP.")
