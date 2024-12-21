import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 图片文件夹路径
before_trained_indomain = './before_trained_output/indomain/'
trained_indomain = './trained_output/indomain/'
before_trained_outdomain = './before_trained_output/outdomain/'
trained_outdomain = './trained_output/outdomain/'

# 获取图片文件列表
before_indomain_images = sorted(os.listdir(before_trained_indomain))
trained_indomain_images = sorted(os.listdir(trained_indomain))
before_outdomain_images = sorted(os.listdir(before_trained_outdomain))
trained_outdomain_images = sorted(os.listdir(trained_outdomain))

def wrap_title(text, width=20):
    """自动将标题文本换行，宽度超过指定值时换行"""
    import textwrap
    return '\n'.join(textwrap.wrap(text, width))

# 创建并保存indomain对比图
def create_indomain_comparison():
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    
    for i in range(8):
        # 读取图片
        before_img = Image.open(os.path.join(before_trained_indomain, before_indomain_images[i]))
        trained_img = Image.open(os.path.join(trained_indomain, trained_indomain_images[i]))
        
        # 处理为numpy数组，以确保大小一致
        before_img = np.array(before_img)
        trained_img = np.array(trained_img)
        
        # 将训练前的图片放在上排
        axes[0, i].imshow(before_img)
        axes[0, i].axis('off')
        axes[0, i].set_title(wrap_title(f"Before: {before_indomain_images[i].split('.')[0]}"))
        
        # 将训练后的图片放在下排
        axes[1, i].imshow(trained_img)
        axes[1, i].axis('off')
        axes[1, i].set_title(wrap_title(f"After:"))
    
    plt.tight_layout()
    plt.savefig('indomain_comparison.png')
    plt.show()

# 创建并保存outdomain的图片展示
def create_outdomain_comparison():
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    
    for i in range(8):
        # 读取图片
        before_img = Image.open(os.path.join(before_trained_outdomain, before_outdomain_images[i]))
        trained_img = Image.open(os.path.join(trained_outdomain, trained_outdomain_images[i]))
        
        # 处理为numpy数组，以确保大小一致
        before_img = np.array(before_img)
        trained_img = np.array(trained_img)
        
        # 将训练前的图片放在上排
        axes[0, i].imshow(before_img)
        axes[0, i].axis('off')
        axes[0, i].set_title(wrap_title(f"Before: {before_outdomain_images[i].split('.')[0]}"))
        
        # 将训练后的图片放在下排
        axes[1, i].imshow(trained_img)
        axes[1, i].axis('off')
        axes[1, i].set_title(wrap_title(f"After:"))
    
    plt.tight_layout()
    plt.savefig('outdomain_comparison.png')
    plt.show()


if __name__ == '__main__':
# 调用创建图片函数
    create_indomain_comparison()
    create_outdomain_comparison()