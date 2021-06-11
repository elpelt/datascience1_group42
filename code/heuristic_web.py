import streamlit as st
from dbscan_heuristic import DBSCANHeuristic
import pandas as pd
import altair as alt

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="DBSCAN Heuristic", page_icon=":goat:")
st.title('DBSCAN Heuristic for determining minPts and eps')

st.header("Settings")

with st.form(key='settings'):
    col1, col2 = st.beta_columns(2)
    dataset = col1.selectbox('Choose a beautiful dataset',['iris', 'wine', 'diabetes', 'housevotes'])

    cluster_dist_desc = {'euclidean': 'd(x,y) = \sqrt{\sum_{i=1}^{n}(|x_i-y_i|)^2}',
                        'manhattan': 'd(x,y) = \sum\limits_{i=1}^{n}|x_i - y_i|',
                        'chebyshev': 'd(x,y) = \max(|x_i - y_i|)',
                        'cosine': 'd(x,y) = \\frac{\\arccos(\\frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2 \sum_{i=1}^{n} y_i^2}})}{\pi}'}
    
    cluster_dist = col1.selectbox('Choose an awesome distance measure',list(cluster_dist_desc.keys()))
    
    k = col2.slider("Choose a nice value for k", min_value=1, max_value=20, step=1, value=4)

    col2.latex(cluster_dist_desc[cluster_dist])
    
    submit_button = st.form_submit_button(label='Calculate kdist Graph')

heu = DBSCANHeuristic()
heu.set_metric(cluster_dist)
heu.set_dataset(dataset)

kdist = heu.kdist(k)
kdist.sort(reverse=True)

df = pd.DataFrame(
    [[i+1, kdist[i]] for i in range(len(kdist))],
    columns=["points", "dist"])

# this code is mostly derived from the altair examples gallery, especially this example:
# https://altair-viz.github.io/gallery/multiline_tooltip.html

nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['points'], empty='none')

yaxis = alt.Y("dist", axis=alt.Axis(title=f"{k}-dist"))
line = alt.Chart(df).mark_line(point=True).encode(x="points", y=yaxis).properties(
                    title=f"DBSCAN Heuristic k={k}, {cluster_dist} distance")

selectors = alt.Chart(df).mark_point().encode(x='points', opacity=alt.value(0)).add_selection(nearest)

points = line.mark_point(color="red").encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))

text = line.mark_text(align='left', dx=5, dy=-5, color="red").encode(text=alt.condition(nearest, "label:N", alt.value(' '))).transform_calculate(label=f'"distance: " + format(datum.dist, ".2f")')

textp = line.mark_text(align='left', dx=5, dy=-15, color="red").encode(text=alt.condition(nearest, "label:N", alt.value(' '))).transform_calculate(label=f'format( (1 - (datum.points-1) / {len(kdist)}) * 100, ".2f") + "% core points"')

rules = alt.Chart(df).mark_rule(color='gray').encode(x="points").transform_filter(nearest)

st.altair_chart(line+selectors+points+rules+text+textp, use_container_width=True)

#st.pyplot(heu.plot_kdist(kdist))