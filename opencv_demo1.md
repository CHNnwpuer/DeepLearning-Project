#### 用python-opencv改的C++demo
原链接： https://blog.csdn.net/hello_yxc/article/details/69362643
```python
import cv2
import argparse


def whiteFace(img, alpha, beta):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            for k in range(0, 3):
                if(alpha * img[i,j,k] + beta < 0):
                    img[i,j,k] = 0
                if (alpha * img[i,j,k] + beta > 255):
                    img[i,j,k] = 255
                else:
                    img[i,j,k] = alpha * img[i,j,k] + beta




def main(args):
    img_ori = cv2.imread(args.input)
    img_result = cv2.imread(args.input)
    img_final = cv2.imread(args.input)
    bilateralFilterval = 30
    cv2.imshow('1111', img_ori)
    whiteFace(img_ori, 1.1, 68)
    cv2.imshow('2222', img_ori)
    cv2.GaussianBlur(img_ori,(9,9),0,img_ori,0)
    cv2.bilateralFilter(img_ori,bilateralFilterval,bilateralFilterval * 2,bilateralFilterval / 2,img_result)
    cv2.GaussianBlur(img_result,(0,0),9,img_final)
    cv2.addWeighted(img_result,1.5,img_final,-0.5,0,img_final)
    cv2.imshow('4444',img_final)
    cv2.imwrite(args.output,img_final)
    cv2.waitKey(0)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output.jpeg', help='path to save the processed images')
    parser.add_argument('--input', default='input.jpeg', help='path to the input images')
    args = parser.parse_args()
    main(args)
```
