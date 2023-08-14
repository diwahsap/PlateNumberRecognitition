from ultralytics import YOLO
import glob
import pandas as pd
import os
import time
from combine import *


def main():
    user_input = input(
        "Masukkan lokasi folder berisi gambar yang ingin diprediksi (format .png): ")
    filename_output = input('Masukkan nama output file : ')

    img_fns = glob.glob(user_input + '/*.png')

    model_path = 'Model/best.pt'
    model = YOLO(model_path)

    results = model.predict(img_fns)
    df_combined = combine_results(results, img_fns)

    df_combined = df_combined.sort_values(by='Name of File', key=lambda x: x.map(sorting_filename))
    df_combined = df_combined.reset_index().iloc[:, 1:3]

    df_combined.to_csv(f'{filename_output}.csv')


if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
