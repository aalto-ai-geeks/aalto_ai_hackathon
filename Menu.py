import streamlit as st
import json


st.set_page_config(
    page_title="Event Tracking Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")
st.markdown("# Select venue parameters")

col = st.columns((3,3), gap="medium")
with col[0]:
    st.markdown('### Tracking details')
    people = st.toggle("People")
    bottle = st.toggle("Bottles")

with col[1]:
    st.markdown("### Venue details")
    size = st.number_input("Size (m^2)", value=100)
    max_capacity = st.number_input("Maximum Attendance", value=50)


start = st.button("Start")

if start:
    data = {
        "people": people,
        "bottle": bottle,
        "size": size,
        "max_capacity": max_capacity
    }
    with open("parameters.json", "w") as f:
        json.dump(data, f)
    st.switch_page("./pages/Video_tracking.py")
