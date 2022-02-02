import cv2
import numpy as np

import glob
import matplotlib.pyplot as plt
import warnings
import math

import tensorflow as tf
from tensorflow import keras


def load_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    """downloadフォルダに保存されたMNISTの画像とラベルデータを取得する
    Args:
        name(str): ImageとLabelのファイル名にあるプレフィックス train or test

    Returns:
        tuple[ndarray, ndarray]: (画像データ(3次元配列), ラベルデータ(1次元配列))
    """
    train_images = glob.glob(f"./mvtec/screw/test/thread_top/*.png")
    return np.array(
        list(
            map(lambda l: load_image(l), train_images)
        )
    )


def load_image(name: str):
    print(name)
    image = cv2.imread(name)
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/256
    return image


def plot_images(images: np.ndarray, labels: np.ndarray) -> None:
    """画像イメージ複数を並べてプロットする
    Args:
        images (np.ndarray): 画像イメージ配列
        labels (np.ndarray): ラベル配列
    """
    # if len(images) != len(labels):
    #     warnings.warn(f"データ数が一致しません\n画像数：{len(images)} ラベル数：{len(labels)}")
    #     return

    # 100 => 10x10のように正方形に均す処理
    image_count = len(images)
    col = math.ceil(math.sqrt(image_count))
    row = col - 1 if col * (col - 1) > image_count else col

    plt.figure(figsize=(row, col))
    for i in range(image_count):
        plt.subplot(row, col, i+1)
        plt.xticks([])
        plt.yticks([])

        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(labels[i])
    plt.tight_layout()
    plt.show()


def create_model():
    """Tensorflow Modelを作成する

    Returns:
        作成したモデル
    """
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D((2, 2), padding="same"),
        keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2), padding="same"),
        keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
        keras.layers.UpSampling2D(size=(2, 2)),
        keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        keras.layers.UpSampling2D(size=(2, 2)),
        keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same"),
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy")
    print(model.summary())
    return model


def main():
    data = load_data("")
    max = len(data)
    m = math.ceil(max*0.9)
    print(data[0])
    lb = list(map(lambda l: "DATA", data))
    #model = create_model()

    # model.fit(data[0:m], data[0:m], epochs=2500, batch_size=256,
    #          shuffle=True, validation_data=(data[m:max], data[m:max]))

    # model.save("tensorflow_model")
    model = keras.models.load_model("tensorflow_model")

    f = 0
    t = 0+12
    x_test_sampled_pred = model.predict(data, verbose=0)

    print((data[f:t] - x_test_sampled_pred[f:t])**2)
    lb[(t-f)*2:] = list(map(lambda l: f"{l:.4f}", ((data[f:t] - x_test_sampled_pred[f:t])**2).mean(axis=(1, 2, 3))))
    print(lb)
    plot_images(np.concatenate([
        data[f:t],
        x_test_sampled_pred[f:t],
        abs(data[f:t] - x_test_sampled_pred[f:t])
    ]), lb)
    pass


if __name__ == "__main__":
    main()
