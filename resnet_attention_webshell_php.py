import os
from model import attention_resnet
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tensorflow.keras.callbacks import Callback
from utils import load_data
from tensorflow.keras.optimizers import Adam

height = 128
language = "php"
benign_ = "white"
malicious_ = "black"
train_black_dir = './codeImage/' + language + "/height" + str(height) + '/train/' + malicious_ + '/*.png'
train_white_dir = './codeImage/' + language + "/height" + str(height) + '/train/' + benign_ + '/*.png'
test_black_dir = './codeImage/' + language + "/height" + str(height) + '/test/' + malicious_ + '/*.png'
test_white_dir = './codeImage/' + language + "/height" + str(height) + '/test/' + benign_ + '/*.png'

width = 128
epochs = 20
model = attention_resnet(input_shape=(height, width, 1))
starting_epoch = 0
# model_path = './models/'+"height"+str(height)+'/model_epoch_08.h5'
# model.load_weights(model_path)

(train_images, train_labels), (test_images, test_labels) = load_data(train_black_dir, train_white_dir,
                                                                     test_black_dir, test_white_dir)

x_train = train_images
x_test = test_images
y_train = train_labels
y_test = test_labels

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print(x_train.shape[0], 'train set')
print(x_test.shape[0], 'test set')

optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


class TestEvaluationCallback(Callback):
    def __init__(self, x_test, y_test):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_test)
        y_pred_class = (y_pred > 0.5).astype(int).flatten()
        y_test_class = self.y_test.flatten()

        acc = accuracy_score(y_test_class, y_pred_class)
        recall = recall_score(y_test_class, y_pred_class)
        precision = precision_score(y_test_class, y_pred_class)
        f1 = f1_score(y_test_class, y_pred_class)
        print(f"..............................................")
        print(f"Epoch {epoch + 1}")
        print(f"Test Accuracy: {acc}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"F1 Score: {f1}")


class SaveModelCallback(Callback):
    def __init__(self, save_dir, starting_epoch=0):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.starting_epoch = starting_epoch

    def on_epoch_end(self, epoch, logs=None):
        model_path = os.path.join(self.save_dir, f'model_epoch_{self.starting_epoch + epoch + 1:02d}.h5')
        self.model.save_weights(model_path)
        print(f"Model saved to: {model_path}")
        print(f"..............................................")


test_eval_callback = TestEvaluationCallback(x_test, y_test)
save_model_callback = SaveModelCallback('./models/' + language + "/height" + str(height), starting_epoch=starting_epoch)
history = model.fit(x_train, y_train, epochs=epochs, batch_size=16, callbacks=[test_eval_callback, save_model_callback])
