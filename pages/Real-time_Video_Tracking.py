import streamlit as st
import altair as alt
import cv2
import cv2
import socket
import pickle
import struct
import json

# Create a socket client


from ultralytics import YOLO

model = YOLO('yolov8s-world.pt')
model = model.to("cuda")

with open("parameters.json", "r") as f:
    data = json.load(f)


with open("config.json", "r") as f:
    config = json.load(f)

classNames = []


if data["people"]:
    classNames.append("person")
if data["bottle"]:
    classNames.append("bottle")


model.set_classes(classNames)

def main():

    alt.themes.enable("dark")

    rtsp_url = f'rtsp://{config["user"]}:{config["pass"]}@{config["ip"]}:{config["port"]}/stream1'

    camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    with st.sidebar:
        st.title('Event Tracking Dashboard')
        st.text("This view enables online tracking of the venue.")

    st.markdown('#### Event scene')
    FRAME_WINDOW = st.image([])

    while True:
        _, frame = camera.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.track(img, persist=True, show=False, device="cuda")
        annotated_frame = results[0].plot()
        FRAME_WINDOW.image(annotated_frame)

if __name__ == "__main__":
    main()
