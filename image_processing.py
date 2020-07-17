import numpy
import cv2


def pretreatment(image):
	"""
	图像预处理
	将所传入图像进行灰度化、二值化，并进行噪点处理
	:param image: 原始图像
	:return: 处理后的二值化图像
	"""
	# 图像灰度化
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# 图像二值化
	ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	image = noise_processing(image)

	return image


def noise_processing(image):
	# 噪点处理
	i_height = image.shape[0]
	i_width = image.shape[1]
	threshold = 7

	# 判定每一个像素
	for i in range(1, i_height - 1):
		for j in range(1, i_width - 1):
			i_value = image[i, j]

			# 去除黑点周围噪点
			if i_value == 0:
				i_count = 0
				for m in range(i - 1, i + 2):
					for n in range(j - 1, j + 2):
						if image[m, n] == 255:
							i_count = i_count + 1
				if i_count >= threshold:
					image[i, j] = 255
			else:
				i_count = 0
				for m in range(i - 1, i + 2):
					for n in range(j - 1, j + 2):
						if image[m, n] == 0:
							i_count = i_count + 1
				if i_count >= threshold:
					image[i, j] = 0

	return image


def resize(image):
	"""
	重新设置图像的大小
	将图像进行放缩 并将周围无用空白进行去除
	:param image: 原始图像
	:return: 用于展示的图片 image_show(128 * 128)
			用于识别的图片 image_use(28 * 28)
	"""
	image = 255 - image

	i_height = image.shape[0]
	i_width = image.shape[1]

	# 查找图像的顶部与底部
	up = 9999
	down = 0
	left = 9999
	right = 0
	for i in range(i_height):
		for j in range(i_width):
			if image[i, j] == 255:
				if j < left:
					left = j
				if j > right:
					right = j
				if i < up:
					up = i
				if i > down:
					down = i

	# 计算图像的合适大小
	size = int((down - up) / 0.7) + 1
	space_height = int(size * 0.3 / 2)
	space_width = int((size - (right - left)) / 2)

	# 创建新的图像 去除周围无用空白
	new_image = numpy.zeros([size, size])
	new_image.fill(0)
	for i in range(space_height, size - space_height):
		for j in range(space_width, size - space_width):
			if image[up + i - space_height, left + j - space_width] == 255:
				new_image[i, j] = 255

	# 统一缩放图像到128, 128
	image_show = cv2.resize(new_image, (128, 128), interpolation=cv2.INTER_LANCZOS4)
	image_show = binarization(image_show)

	# 统一缩放图像到28, 28
	image_use = cv2.resize(new_image, (28, 28), interpolation=cv2.INTER_LANCZOS4)
	image_use = binarization(image_use)

	return image_show, image_use


def binarization(image):
	"""
	图像二值化
	:param image: 原始图像
	:return:  二值化后的图像
	"""
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i, j] > 180:
				image[i, j] = 255
			else:
				image[i, j] = 0

	return image


def processing(path):
	"""
	图像处理
	:param path: 图像路径
	:return: 用于展示的图片 image_show(128 * 128)
			用于识别的图片 image_use(28 * 28)
	"""
	image = cv2.imread(path)
	image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LANCZOS4)
	image = pretreatment(image)
	image_show, image_use = resize(image)
	return image_show, image_use


if __name__ == '__main__':
	a, b = processing("./image/4.png")
