import numpy as np
from tensorflow.python.keras.models import load_model

from image_processing import processing

model = load_model('./digit_model.h5')


def recognition(image):
	image = image / 255.0
	image = image.reshape(784, )
	image = np.expand_dims(image, axis=0)  # p转换成了2维,模型第一层定义了输入格式为2维: input_shape=(784,)

	# 测试图片
	pre = model.predict(image)
	# print(np.argmax(pre, axis=1))
	return np.argmax(pre, axis=1)[0]


if __name__ == '__main__':
	p, o_image = processing("./image/7.png")  # p是1维
	recognition(o_image)
