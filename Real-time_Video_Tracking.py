import streamlit as st
import altair as alt
import cv2


def main():

    alt.themes.enable("dark")

    rtsp_url = "rtsp://smartbi:aigeeks@192.168.28.243:554/stream1"

    # Sidebar

    camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    with st.sidebar:
        st.title('Event Tracking Dashboard')
        st.text("This view enables online tracking of the venue.")

    st.markdown('#### Event scene')

    option = st.selectbox(
        'What would you like to track?',
        ('People', 'Tables', 'Food'))

    # PLACEHOLDER the option should be passed to YOLO model
    st.write(option)

    FRAME_WINDOW = st.image([])
    _, frame = camera.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # run = st.checkbox('Stream')
    # # if run:
    FRAME_WINDOW.image(img)
    stream = st.button('Stream')
    if stream:
        while True:
            _, frame = camera.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # run = st.checkbox('Stream')
            # # if run:
            FRAME_WINDOW.image(img)


if __name__ == "__main__":
    main()
