import streamlit as st
import pandas as pd
import altair as alt
from sklearn.ensemble import IsolationForest


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


def crowdedness_estimation(n_people, n_max=100):
    crowd = round(n_people / n_max * 100)
    return f"The space is filled by {crowd}%", crowd, 'blue'


def isolation_estimation(X):
    # X should be centroids of bounding boxes (n_samples, n_features)
    clf = IsolationForest(random_state=0).fit(X)
    preds = clf.predict(X)
    return sum(preds[preds == -1])
