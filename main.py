import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from digit_recognition import recognition1, recognition2
from image_processing import processing

global o_image, p_image, image_use, file_path
global label_get_img, label_processing_img, label_result1, label_result2

# 基础界面
root = tk.Tk()


def run():
	"""
	运行手写数字识别
	:return:
	"""
	global label_get_img, label_processing_img, label_result1, label_result2
	root.title('手写数字识别')

	# 选择图片按钮
	get_image_button = tk.Button(root, text="选择图片", command=get_image)
	get_image_button.pack()

	# 选择图片显示
	label_get_img = tk.Label(root, image="")
	label_get_img.pack()

	# 处理图片按钮
	processing_image_button = tk.Button(root, text="处理图片", command=processing_image)
	processing_image_button.pack()

	# 处理图片显示
	label_processing_img = tk.Label(root, image="")
	label_processing_img.pack()

	# 模型1识别图片按钮
	recognition_image_button1 = tk.Button(root, text="模型1识别图片", command=recognition_image1)
	recognition_image_button1.pack()

	# 模型1识别图片显示
	label_result1 = tk.Label(root, text="待识别")
	label_result1.pack()

	# 模型2识别图片按钮
	recognition_image_button2 = tk.Button(root, text="模型2识别图片", command=recognition_image2)
	recognition_image_button2.pack()

	# 模型2识别图片显示
	label_result2 = tk.Label(root, text="待识别")
	label_result2.pack()

	root.mainloop()


def get_image():
	"""
	得到图片按钮事件
	:return:
	"""
	global o_image, file_path, label_get_img

	# 从文件中读取图片
	file_path = filedialog.askopenfilename(title=u'选择图片', filetypes=[('所有文件', '*'), ('JPG', '*.jpg'), ('PNG', '*.png')])
	o_image = cv2.imread(file_path)
	o_image = cv2.resize(o_image, (128, 128), interpolation=cv2.INTER_LANCZOS4)
	o_image = ImageTk.PhotoImage(Image.fromarray(o_image))

	# 图片显示
	label_get_img.configure(image=o_image)


def processing_image():
	"""
	处理图片按钮事件
	:return:
	"""
	global p_image, image_use, label_processing_img

	# 图片处理 调用image_processing.py
	image_show, image_use = processing(file_path)
	p_image = ImageTk.PhotoImage(Image.fromarray(image_show))

	# 处理完毕按钮显示
	label_processing_img.configure(image=p_image)


def recognition_image1():
	"""
	模型1识别图片按钮事件
	:return:
	"""
	global label_result1

	# 调用模型1进行图片识别
	result = "模型1图片识别结果为: " + str(recognition1(image_use))
	label_result1.configure(text=result)


def recognition_image2():
	"""
	模型2识别图片按钮事件
	:return:
	"""
	global label_result2

	# 调用模型2进行图片识别
	result = "模型2图片识别结果为: " + str(recognition2(image_use))
	label_result2.configure(text=result)


if __name__ == '__main__':
	run()
