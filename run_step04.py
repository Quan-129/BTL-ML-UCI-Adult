from modules.splitting import load_saved_splits
from modules.preprocessing import export_features, plot_preprocessing, update_latex

if __name__ == "__main__":
    frames, metadata = load_saved_splits()
    export_features(frames, metadata)
    plot_preprocessing()
    update_latex()
    print("Đã xuất features và cập nhật nguồn LaTeX. Notebook tạo thêm ZIP báo cáo.")
