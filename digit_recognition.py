import numpy as np
from tensorflow.python.keras.models import load_model

from image_processing import processing
from train_model2 import get_feature

# 加载模型
model1 = load_model('./digit_model1.h5')
model2 = load_model('./digit_model2.h5')


def recognition1(image):
	"""
	使用模型1识别
	:param image: 待识别图片
	:return: 识别结果
	"""
	image = image / 255.0
	image = image.reshape(784, )
	image = np.expand_dims(image, axis=0)  # p转换成了2维,模型第一层定义了输入格式为2维: input_shape=(784,)

	# 测试图片
	pre = model1.predict(image)
	return np.argmax(pre, axis=1)[0]


def recognition2(image):
	"""
	使用模型2识别
	:param image: 待识别图片
	:return: 识别结果
	"""
	# 转化为列表的array格式
	image_list = []
	image = get_feature(image)
	image_list.append(image)
	image_array = np.array(image_list)

	# 测试图片
	pre = model2.predict(image_array)
	return np.argmax(pre, axis=1)[0]


if __name__ == '__main__':
	p, o_image = processing("./image/1.png")
	recognition1(o_image)
	recognition2(o_image)
