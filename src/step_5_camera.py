# import cv2
# import numpy as np
# import onnxruntime as ort

# # This function crops a rectangular frame to a square shape by extracting the center region, ensuring consistent dimensions for image processing tasks.
# def center_crop(frame):
#     h, w, _ = frame.shape
#     start = abs(h - w) // 2
#     if h > w:
#         return frame[start: start + w]
#     return frame[:, start: start + h]

# # This function captures video frames from a webcam, preprocesses them, uses an ONNX-exported model to classify hand gestures in real-time, and displays the predicted letter on the screen.
# def main():
#     # constants
#     index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
#     mean = 0.485 * 255.
#     std = 0.229 * 255.

#     # create runnable session with exported model
#     ort_session = ort.InferenceSession("signlanguage.onnx")

#     cap = cv2.VideoCapture(0)
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         # preprocess data
#         frame = center_crop(frame)
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         x = cv2.resize(frame, (28, 28))
#         x = (x - mean) / std

#         x = x.reshape(1, 1, 28, 28).astype(np.float32)
#         y = ort_session.run(None, {'input': x})[0]

#         index = np.argmax(y, axis=1)
#         letter = index_to_letter[int(index)]

#         cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
#         cv2.imshow("Sign Language Translator", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()






import cv2
camera_index=1
cam = cv2.VideoCapture(1)
if not cam.read()[0]:
    cam = cv2.VideoCapture(0)
    camera_index=0
while True:
    ret, frame = cam.read()
    # cv2.imshow(f"image from camera {camera_index}", frame)
    if not ret:
        break
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
cam.release()
cv2.destroyAllWindows()