import tkinter as tk
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk

from digit_recognition import recognition
from image_processing import processing

root = tk.Tk()
global o_image, p_image, image_use, file_path
global label_get_img, label_processing_img, label_result


def run():
	global label_get_img, label_processing_img, label_result
	root.title('手写数字识别')

	get_image_button = tk.Button(root, text="选择图片", command=get_image)
	get_image_button.pack()

	label_get_img = tk.Label(root, image="")
	label_get_img.pack()

	processing_image_button = tk.Button(root, text="处理图片", command=processing_image)
	processing_image_button.pack()

	label_processing_img = tk.Label(root, image="")
	label_processing_img.pack()

	recognition_image_button = tk.Button(root, text="识别图片", command=recognition_image)
	recognition_image_button.pack()

	label_result = tk.Label(root, text="")
	label_result.pack()

	root.mainloop()


def get_image():
	global o_image, file_path, label_get_img
	file_path = filedialog.askopenfilename(title=u'选择图片', filetypes=[('所有文件', '*'), ('JPG', '*.jpg'), ('PNG', '*.png')])
	o_image = cv2.imread(file_path)
	o_image = cv2.resize(o_image, (128, 128), interpolation=cv2.INTER_LANCZOS4)
	o_image = ImageTk.PhotoImage(Image.fromarray(o_image))

	label_get_img.configure(image=o_image)


def processing_image():
	global p_image, image_use, label_processing_img
	image_show, image_use = processing(file_path)
	p_image = ImageTk.PhotoImage(Image.fromarray(image_show))

	label_processing_img.configure(image=p_image)


def recognition_image():
	global label_result
	result = "图片识别结果为: " + str(recognition(image_use))
	label_result.configure(text=result)


if __name__ == '__main__':
	run()
