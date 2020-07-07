import tensorflow as tf
from mnist import MyCNN, MNISTLoader

save_path = './save'

filters_size_ = [2, 4, 8]  # CNN模型每层卷积核个数
model = MyCNN(filters_size=filters_size_)  # 初始化模型

# learning_rate = 1e-3  # 学习率
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # 初始化优化器
#
# current_epoch = tf.Variable(initial_value=0, dtype=tf.int32)  # 当前训练回合
#
# max_acc = tf.Variable(initial_value=0.0, dtype=tf.float32)  # 最高测试集准确率标志位

# checkpoint = tf.train.Checkpoint(model=model,
#                                  optimizer=optimizer,
#                                  current_epoch=current_epoch,
#                                  max_acc=max_acc)  # 模型及优化器参数保存节点

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(save_path))

loader = MNISTLoader()
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  # 初始化测试集评估工具

y_pred = model.predict(loader.x_test)  # 获取测试集运算结果
sparse_categorical_accuracy.update_state(y_true=loader.y_test, y_pred=y_pred)  # 评估测试集准确率
test_acc = sparse_categorical_accuracy.result()  # 获取准确率数值
print(test_acc.numpy())

tf.saved_model.save(model, './savedmodel')
