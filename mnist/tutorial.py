import math
import struct
import warnings

import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, metrics


def load_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    """downloadフォルダに保存されたMNISTの画像とラベルデータを取得する
    Args:
        name(str): ImageとLabelのファイル名にあるプレフィックス train or test
    Returns:
        tuple[ndarray, ndarray]: (画像データ(3次元配列), ラベルデータ(1次元配列))
    """
    f_images = open(f"./download/{name}-images", "rb")
    f_labels = open(f"./download/{name}-labels", "rb")
    (_, img_count) = struct.unpack(">II", f_images.read(8))
    (_, lbl_count) = struct.unpack(">II", f_labels.read(8))
    (row, col) = struct.unpack(">II", f_images.read(8))

    # データをndarray配列に処理し、reshapeをすることで
    # データ数x28x28の画像ファイル配列と、ラベル数のラベル配列を作る
    img_body = np.frombuffer(f_images.read(), "B")
    lbl_body = np.frombuffer(f_labels.read(), "B")
    data = img_body.reshape(img_count, row, col)
    data = data / 256
    label = lbl_body.reshape(lbl_count)
    f_images.close()
    f_labels.close()
    return (data, label)


def plot_images(images: np.ndarray, labels: np.ndarray) -> None:
    """画像イメージ複数を並べてプロットする
    Args:
        images (np.ndarray): 画像イメージ配列
        labels (np.ndarray): ラベル配列
    """
    if len(images) != len(labels):
        warnings.warn(f"データ数が一致しません\n画像数：{len(images)} ラベル数：{len(labels)}")
        return

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
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(labels[i])
    plt.show()


def main():
    DATA_MAX = 3500
    m = 500
    (train_images, train_labels) = load_data("train")
    train_images = train_images[0:DATA_MAX]
    train_labels = train_labels[0:DATA_MAX]

    # count = 16
    # plot_images(train_images[0:count], train_labels[0:count])

    train_images_flatten = np.array(list(map(lambda l: l.flatten(), train_images)))
    predict_images_flatten = train_images_flatten[m:DATA_MAX]
    train_images_flatten = train_images_flatten[0:m]
    predict_labels = train_labels[m:DATA_MAX]
    train_labels = train_labels[0:m]

    #svm.SVC(kernel="rbf", C=1, gamma=0.5, random_state=0)
    print("SVM train start")
    clf = svm.SVC()
    clf.fit(train_images_flatten, train_labels)

    print("SVM predict start")
    predict = clf.predict(predict_images_flatten)

    print("SVM result")
    score = metrics.accuracy_score(predict_labels, predict)
    report = metrics.classification_report(predict_labels, predict)
    print("Accuracy: ", score)
    print(report)


if __name__ == "__main__":
    main()
