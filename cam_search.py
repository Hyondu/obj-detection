import cv2

index = 0
arr = []
i = 10
while i > 0:
    cap = cv2.VideoCapture(index)
    if cap.read()[0]:
        arr.append(index)
        cap.release()
    index += 1
    i -= 1
print("사용 가능한 카메라 인덱스:", arr)