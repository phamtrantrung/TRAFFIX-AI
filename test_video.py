import cv2, os

path = r"D:\DAI_HOC_GTVT\Thuctap\Video test\Video 1.mp4"
print("Exists:", os.path.exists(path))

cap = cv2.VideoCapture(path)
print("Opened:", cap.isOpened())
