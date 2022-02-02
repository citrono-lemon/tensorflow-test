import math
import struct
import warnings
import cv2
import pickle

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
    plt.tight_layout()
    plt.show()


def create_model(images: np.ndarray, labels: np.ndarray, save_name: str) -> svm.SVC | None:
    """予測モデルを作成する(SVM)
    モデルは名前を付けて保存される

    Returns:
        SVMモデルを返す
    """
    if len(images) != len(labels):
        warnings.warn(
            f"The number of data does not match\nimage: {len(images)} / label: {len(labels)}")
        return
    # svm.SVC(kernel="rbf", C=1, gamma=0.5, random_state=0)
    print("SVM training has begun")
    clf = svm.SVC()
    clf.fit(images, labels)

    print(f"This model has been created under the name \"{save_name}\"!")
    with open(save_name, 'wb') as f:
        pickle.dump(clf, f, protocol=2)

    return clf


def get_data_by_file(file_name: str) -> np.ndarray:
    """画像データを読み込み、28x28の白黒画像に変換したのち、
    大きさが0~1のndarrayにして返す

    Args:
        file_name (str): 画像データ
    Returns:
        np.ndarray: 画像の二次元配列
    """
    data = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    data = cv2.resize(data, (28, 28))
    data = cv2.bitwise_not(data)
    data = data/256
    return data


def main2():
    data1 = get_data_by_file("sample1.png")
    data2 = get_data_by_file("sample2.png")
    test = np.array([data1.flatten(), data2.flatten()])

    with open("svm_model.pickle", "rb") as f:
        clf = pickle.load(f)
        res = clf.predict(test)

    plot_images(np.array([data1, data2]), res)


def main():
    # count = 12
    # plot_images(train_images[0:count], train_labels[0:count])
    DATA_MAX = 2000
    m = 1500
    (train_images, train_labels) = load_data("train")
    train_images = train_images[0:DATA_MAX]
    train_labels = train_labels[0:DATA_MAX]

    flatten_images = np.array(list(map(lambda l: l.flatten(), train_images)))

    predict_images_flatten = flatten_images[m:DATA_MAX]
    train_images_flatten = flatten_images[0:m]
    predict_labels = train_labels[m:DATA_MAX]
    train_labels = train_labels[0:m]

    clf = create_model(train_images_flatten, train_labels, "svm_model.pickle")

    print("SVM predictions has started")
    predict = clf.predict(predict_images_flatten)

    print("-- Result")
    score = metrics.accuracy_score(predict_labels, predict)
    report = metrics.classification_report(predict_labels, predict)
    print("Accuracy: ", score)
    print(report)


if __name__ == "__main__":
    main2()
