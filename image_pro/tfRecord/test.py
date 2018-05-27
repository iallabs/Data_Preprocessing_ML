import cv2

path = "C:/Users/Lenovo\Desktop/testML\Work2/train/train/cat.42.jpg"
img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
img= cv2.resize(img, dsize=(224, 224))
print(img)
cv2.imshow("image", img)
cv2.waitKey()