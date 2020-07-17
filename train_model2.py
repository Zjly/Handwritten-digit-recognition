import numpy as np
from keras.datasets import mnist
import keras  # 导入Keras
from keras.models import Sequential  # 导入序贯模型
from keras.layers import Dense, Dropout, Activation  # 导入全连接层
from keras.optimizers import SGD  # 导入优化函数


def get_feature(image):
	"""
	计算图片的特征
	:param image: 待计算的图像
	:return: 图像的49个特征值
	"""
	i_height = image.shape[0]
	i_width = image.shape[1]

	feature_list = [0] * 49
	for i in range(i_height):
		for j in range(i_width):
			if image[i, j] != 0:
				index = int(i / 4) * 7 + int(j / 4)
				feature_list[index] += 1

	return feature_list


def train_model():
	"""
	训练模型
	从keras中读取数据集并进行模型训练 而后将训练好的模型保存在文件当中
	:return:
	"""
	# 读取mnist数据集
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	num = 0
	x_train_list = []
	for image in x_train:
		x_train_list.append(get_feature(image))
		num += 1
		if num % 1000 == 0:
			print("当前进度: " + str(num) + "/60000")

	num = 0
	x_test_list = []
	for image in x_test:
		x_test_list.append(get_feature(image))
		num += 1
		if num % 1000 == 0:
			print("当前进度: " + str(num) + "/10000")

	# 转化为array
	x_train_array = np.array(x_train_list)
	x_test_array = np.array(x_test_list)

	# 标签转化为独热编码
	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)

	# 构建模型
	model = Sequential()
	model.add(Dense(512, activation="relu", input_shape=(49,)))
	model.add(Dense(256, activation="relu"))
	model.add(Dense(10, activation='softmax'))

	# 模型编译
	model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

	# 训练模型
	model.fit(x=x_train_array, y=y_train, batch_size=64, epochs=100, verbose=1, callbacks=None, validation_split=0.2,
			  validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
			  steps_per_epoch=None, validation_steps=None)

	# 评估模型
	score = model.evaluate(x_test_array, y_test, batch_size=64)
	print(score)

	# 保存模型
	model.save('./digit_model2.h5')


if __name__ == '__main__':
	train_model()
