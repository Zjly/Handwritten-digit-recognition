from keras.datasets import mnist
import keras  # 导入Keras
from keras.models import Sequential  # 导入序贯模型
from keras.layers import Dense, Dropout  # 导入全连接层
from keras.optimizers import SGD  # 导入优化函数


def train_model():
	"""
	训练模型
	从keras中读取数据集并进行模型训练 而后将训练好的模型保存在文件当中
	:return:
	"""
	# 读取mnist数据集
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# 图片转化为一维
	x_train = x_train.reshape(60000, 784)
	x_test = x_test.reshape(10000, 784)

	# 图片数据归一化处理
	x_train = x_train / 255.0
	x_test = x_test / 255.0

	# 标签转化为独热编码
	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)

	# 构建模型
	model = Sequential()
	model.add(Dense(512, activation="relu", input_shape=(784,)))
	model.add(Dropout(0.2))
	# model.add(Dense(256, activation="relu"))
	model.add(Dense(10, activation='softmax'))
	model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

	# 训练模型
	model.fit(x=x_train, y=y_train, batch_size=64, epochs=3, verbose=1, callbacks=None, validation_split=0.2,
			  validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
			  steps_per_epoch=None, validation_steps=None)

	# 评估模型
	score = model.evaluate(x_test, y_test, batch_size=64)
	print(score)

	# 保存模型
	model.save('./digit_model.h5')


if __name__ == '__main__':
	train_model()
