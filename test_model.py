from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from utils import load_data
from model import attention_resnet

height = 32
width = 128
language = "php"
model_path = './models/height'+str(height)+'/model_epoch_20.h5'
model = attention_resnet(input_shape=(height, width, 1))
model.load_weights(model_path)

benign_ = "white"
malicious_ = "black"
train_black_dir = './codeImage/' + language + "/height" + str(height) + '/train/' + malicious_ + '/*.png'
train_white_dir = './codeImage/' + language + "/height" + str(height) + '/train/' + benign_ + '/*.png'
test_black_dir = './codeImage/' + language + "/height" + str(height) + '/test/' + malicious_ + '/*.png'
test_white_dir = './codeImage/' + language + "/height" + str(height) + '/test/' + benign_ + '/*.png'
_, (test_images, test_labels) = load_data(train_black_dir, train_white_dir, test_black_dir, test_white_dir)

x_test = test_images
y_test = test_labels


y_pred = model.predict(x_test)
y_pred_class = (y_pred > 0.5).astype(int).flatten()
y_test_class = y_test.flatten()
acc = accuracy_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)
print(f"..............................................")
print(f"Test Accuracy: {acc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print(f"..............................................")
