import numpy as np
import tensorflow as tf
from tensorflow import keras

from tutorial import get_data_by_file, load_data, plot_images


def create_model():
    """Tensorflow Modelを作成する

    Returns:
        作成したモデル
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def create_model_cnn():
    """Tensorflow CNN Modelを作成する

    Returns:
        作成したモデル
    """
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    print("Loading data...")
    (train_images, train_labels) = load_data("train")
    (test_images, test_labels) = load_data("test")

    # モデルを作成する
    print("TF Training has begun")
    #model = create_model_cnn()
    #model.fit(train_images, train_labels, epochs=5)

    # model.save("tensorflow_model")
    model = keras.models.load_model("tensorflow_model")

    # モデルの評価をする
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nTest loss', test_loss)

    # 予測結果
    predictions = model.predict(test_images)
    result_labels = []

    d = 64
    # ラベルを作る(上位2つ)
    for i in range(d):
        pred = predictions[i]
        # 確率の高い順にインデックスを格納する
        pred_arg_index = np.flip(np.argsort(pred))
        result_labels.append(
            f"{pred_arg_index[0]}[{pred[pred_arg_index[0]]*100:.2f}%]\n"
            f"{pred_arg_index[1]}[{pred[pred_arg_index[1]]*100:.2f}%]"
        )

    plot_images(test_images[0:d], result_labels)


if __name__ == "__main__":
    main()
