import tensorflow as tf
from minist import MNISTLoader

model = tf.saved_model.load('./savedmodel')

loader = MNISTLoader()
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  # 初始化测试集评估工具

y_pred = model(loader.x_test)  # 获取测试集运算结果
sparse_categorical_accuracy.update_state(y_true=loader.y_test, y_pred=y_pred)  # 评估测试集准确率
test_acc = sparse_categorical_accuracy.result()  # 获取准确率数值
print(test_acc.numpy())
