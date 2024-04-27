import streamlit as st
import altair as alt
import cv2
import numpy as np
import sys
import os

# sys.path.append(os.path.abspath('.'))
# from .. import utils
# import utils
# from ...an import utils

import streamlit as st
import pandas as pd
import altair as alt
from sklearn.ensemble import IsolationForest
import json

from ultralytics import YOLO



with open("config.json", "r") as f:
    config = json.load(f)

with open("parameters.json", "r") as f:
    data = json.load(f)


def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100 - input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            # range=['#29b5e8', '#155F7A']),  # 31333F
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)

    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700,
                          fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            range=chart_color),  # 31333F
                        legend=None),
    ).properties(width=130, height=130)
    return plot_bg + plot + text


def darkness_estimation(pic):
    if 0 <= pic <= 33:
        return "The space might be too dark", round(pic), 'red'
    if 33 < pic <= 66:
        return "The space is perfectly lighted", round(pic), 'green'
    if pic > 66:
        return "The space might be too bright", round(pic), 'red'


def crowdedness_estimation(n_people, n_max=data["max_capacity"]):
    crowd = round(n_people / n_max * 100)
    if crowd < 30:
        return f"The space is filled by {crowd}%", crowd, 'red'
    else:
        return f"The space is filled by {crowd}%", crowd, 'red'



def isolation_estimation(X):
    # X should be centroids of bounding boxes (n_samples, n_features)
    clf = IsolationForest(random_state=0).fit(X)
    preds = clf.predict(X)
    return sum(preds[preds == -1])

def main():

    model = YOLO('yolov8s-world.pt')
    model = model.to("cuda")

    classNames = set(["person"])
    model.set_classes(list(classNames))

    alt.themes.enable("dark")

    description, brightness, color_1 = "No data on the brightness", 0, 'green'
    suggestion, people, color_2 = "No data on people", 0,  'blue'
    rtsp_url = f'rtsp://{config["user"]}:{config["pass"]}@{config["ip"]}:{config["port"]}/stream1'
    n_people_before = 0

    # Sidebar
    camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    placeholder = st.empty()

    with st.sidebar:
        st.title('Event Tracking Dashboard')
    
    # FRAME_WINDOW = st.image([])

    while True:

        _, frame = camera.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.track(img, persist=True, show=False, device="cuda")
        img = results[0].plot()


        # FRAME_WINDOW.image(img)
        with placeholder.container():
            col = st.columns((4, 4), gap='medium')
            # _, frame = camera.read()
            # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            avg = np.mean(img[:, :, 2])
            per = (avg / 255) * 100
            # PLACEHOLDER FOR N_PEOPLE CALCULATION. WE ASSUME THAT THERE ARE 100 people in the room max
            
            X = []
            total_people = len(results[0])
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    X.append([x1+x2 // 2, y1+y2 // 2])
            suggestion, people, color_2 = crowdedness_estimation(total_people)
            description, brightness, color_1 = darkness_estimation(per)

            # Dashboard Main Panel
            with col[0]:

                st.markdown('#### Scene parameters')

                donut_chart_greater = make_donut(brightness, description, color_1)
                donut_chart_less = make_donut(people, suggestion, color_2)

                migrations_col = st.columns((1.5, 2, 0.5))
                with migrations_col[1]:
                    st.write('Brightness')
                    st.altair_chart(donut_chart_greater)
                    st.write('Crowdedness')
                    st.altair_chart(donut_chart_less)

                st.markdown('#### Attendies statistics')

                # PLACEHOLDER, here we should get bounding boxes centroids
                if len(X) > 0:
                    n_people = int(isolation_estimation(X))
                else:
                    n_people = 0
                st.metric(label="Number of isolated people", value=n_people,
                          delta=n_people - n_people_before, delta_color="inverse")
                n_people_before = n_people
                st.metric(label="Attendance", value=total_people)

            with col[1]:
                # FRAME_WINDOW = st.image([])
                # FRAME_WINDOW.image(img)
                st.markdown('#### Overtime Heatmap Here')
                # Heatmap mean over certain time period
                st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTRdq-m-6_c0As92g82el-tsfO31XGKFO6h0Cn6b28oUA&s')


if __name__ == "__main__":
    main()
