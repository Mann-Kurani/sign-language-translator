import cv2
import av
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, ClientSettings
import onnxruntime as ort
import numpy as np

# Define a custom video transformer to display the webcam stream
class VideoTransformer(VideoTransformerBase):
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Preprocess data
        frame = center_crop(frm)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x = cv2.resize(frame, (28, 28))
        x = (x - mean) / std

        x = x.reshape(1, 1, 28, 28).astype(np.float32)
        y = ort_session.run(None, {'input': x})[0]

        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]

        # Draw the predicted letter on the frame
        cv2.putText(frm, letter, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        return frame[start: start + w]
    return frame[:, start: start + h]

index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
mean = 0.485 * 255.
std = 0.229 * 255.

# Load the ONNX model
ort_session = ort.InferenceSession("signlanguage.onnx")

def main():
    st.title("Sign Language Translator")

    # Configure WebRTC settings
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    # Display the webcam stream and predicted letter
    if webrtc_ctx.video_receiver:
        predicted_letter = st.empty()
        while True:
            frame = webrtc_ctx.video_receiver.frame
            if frame is not None:
                letter = index_to_letter[np.argmax(ort_session.run(None, {'input': preprocess_frame(frame.to_ndarray(format="bgr24"))})[0], axis=1)]
                predicted_letter.write(f"Predicted Letter: {letter}")

def preprocess_frame(frame):
    frame = center_crop(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(frame, (28, 28))
    x = (x - mean) / std
    x = x.reshape(1, 1, 28, 28).astype(np.float32)
    return x

if __name__ == "__main__":
    main()