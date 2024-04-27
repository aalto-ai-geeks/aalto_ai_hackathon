import streamlit as st
import altair as alt
import cv2
import numpy as np
import sys

# sys.path.append('/Users/aksenovaanna/PycharmProjects/smartbi/space_analytics')
from .. import utils


def main():

    alt.themes.enable("dark")

    description, brightness, color_1 = "No data on the brightness", 0, 'green'
    suggestion, people, color_2 = "No data on people", 0,  'blue'
    rtsp_url = "rtsp://smartbi:aigeeks@192.168.28.243:554/stream1"
    n_people_before = 0

    # Sidebar
    camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    placeholder = st.empty()

    with st.sidebar:
        st.title('Event Tracking Dashboard')

    while True:
        with placeholder.container():
            col = st.columns((4, 4), gap='medium')
            _, frame = camera.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            avg = np.mean(img[:, :, 2])
            per = (avg / 255) * 100
            # PLACEHOLDER FOR N_PEOPLE CALCULATION. WE ASSUME THAT THERE ARE 100 people in the room max
            n_people = 60
            suggestion, people, color_2 = utils.crowdedness_estimation(n_people)
            description, brightness, color_1 = utils.darkness_estimation(per)

            # Dashboard Main Panel
            with col[0]:

                st.markdown('#### Scene parameters')

                donut_chart_greater = utils.make_donut(brightness, description, color_1)
                donut_chart_less = utils.make_donut(people, suggestion, color_2)

                migrations_col = st.columns((1.5, 2, 0.5))
                with migrations_col[1]:
                    st.write('Brightness')
                    st.altair_chart(donut_chart_greater)
                    st.write('Crowdedness')
                    st.altair_chart(donut_chart_less)

                st.markdown('#### Attendies statistics')

                # PLACEHOLDER, here we should get bounding boxes centroids
                X = [[-1.1], [0.3], [0.5], [100]]
                n_people = utils.isolation_estimation(X)
                st.metric(label="Number of isolated people", value=n_people,
                          delta=n_people - n_people_before, delta_color="inverse")
                n_people_before = n_people

            with col[1]:
                st.markdown('#### Overtime Heatmap Here')
                # Heatmap mean over certain time period
                st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTRdq-m-6_c0As92g82el-tsfO31XGKFO6h0Cn6b28oUA&s')


if __name__ == "__main__":
    main()




