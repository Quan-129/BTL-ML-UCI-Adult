from modules.data_loader import download_adult, load_adult, export_step01

if __name__ == "__main__":
    info = download_adult("data")
    train, test = load_adult("data")
    export_step01(train, test, info)
    print("Train:", train.shape, "Test:", test.shape)
