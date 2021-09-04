import numpy as np
import cv2


img_open = cv2.imread('./img.png',cv2.IMREAD_GRAYSCALE) # 적용 시킬 이미지 

ret, dst = cv2.threshold(img_open,100,255 ,cv2.THRESH_OTSU)


h, w = img_open.shape
img_open32 = np.float32(img_open)
img_open32 = np.clip(img_open32,0,255)


count =0
A= np.zeros((w*h, 3))
Z= np.zeros((w*h, 1))

for i in range(0, h):
    for j in range(0, w):
        A[count, 0] = j
        A[count, 1] = i
        A[count, 2] = 1
        Z[count, 0] = img_open32[i,j]
        count += 1
count = 0
intensity = 0

for i in range(0, h):
    for j in range(0, w):
        intensity  += img_open32[i,j]

mean = intensity / (h*w)

TransposA = A.T 
invA = np.linalg.inv(np.dot(TransposA, A))
X = np.dot(invA,TransposA)
X = np.dot(X,Z)

aV = X[0]
bV = X[1]
cV = X[2]

for i in range(0, h):
    for j in range(0, w):
        Z1 = aV*j + bV*i + cV
        img_open32[i,j] = abs(Z1 - img_open32[i,j] + mean)


gray_img = np.uint8(img_open32)
gray_img = np.clip(gray_img,0,255)

ret, gray_img = cv2.threshold(gray_img,100,255 ,cv2.THRESH_OTSU)
cv2.imshow('hi2', dst) #적용 안 한 이미지
cv2.imshow('hi', gray_img) # 적용한 이미지
cv2.waitKey()
cv2.destroyAllWindows() 