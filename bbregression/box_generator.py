import matplotlib.pyplot as plt
import numpy as np
import random
import os

batch_size = 128
img_h = 100
img_w = img_h
total_iter_num = 100000
num_gpus = 1

ln_rate = 0.001

exp_name = 'regressor_study_more_features'
controller = 'CPU'


def generate_img(img_h,img_w):
    while 1:
        img = np.zeros([img_h, img_w])

        curr_w = random.randint(10, 50)
        curr_h = curr_w

        x_offset = random.randint(0, img_w - curr_w)
        y_offset = random.randint(0, img_h - curr_h)

        img[y_offset:y_offset + curr_h, x_offset:x_offset + curr_w] = 1
        label = [x_offset, y_offset, curr_w, curr_h]

        return img, label





if __name__ == '__main__':
    img_h = 100
    img_w = img_h
    label_file = open("/home/mitom/dlcv/data/boundingbox/box_label.txt", 'w')
    plt.ion()
    
    for i in range(1000):
        img, label = generate_img(img_h,img_w)
        label = [i/100 for i in label]
        print(label)
        plt.title('x:{}, y:{}, w:{}, h:{}'.format(label[0], label[1], label[2], label[3]))
#         plt.imshow(img, cmap='gray')
#         plt.imshow(img)
#         plt.pause(2)
        imagefile = '/home/mitom/dlcv/data/boundingbox/'+str(i)+'.jpg'
        plt.imsave(imagefile, img)
        plt.pause(0.01)
        label_file.writelines(imagefile + ' ' + str(label[0]) + ' ' + str(label[1]) + ' ' + str(label[2]) + ' ' + str(label[3]) + '\n')
        
    label_file.close()
        
        
        
        
        
        
