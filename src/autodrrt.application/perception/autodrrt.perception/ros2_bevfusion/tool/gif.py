from PIL import Image
import imageio
import os

# 设置图片文件夹和输出 GIF 文件的路径
input_folder = 'input_images'
output_gif_path = './output.gif'

# 获取图片文件夹中的所有图片文件

sorted_file = sorted(os.listdir('./dump/'))

image_files = [os.path.join('./dump/', filename) for filename in sorted_file]
print(image_files)

# # 读取每个图片并添加到一个图片列表中
images = []
i = 0
for image_file in image_files:
    image_file = image_file + '/cuda-bevfusion.jpg'
    img = Image.open(image_file)
    images.append(img)
    # i = i + 1
    # if i > 30:
    #     break

# # 保存为 GIF 动图
imageio.mimsave(output_gif_path, images, duration=1.8)  # 设置每帧的显示时间（秒）

print(f'动图已保存为 {output_gif_path}')
