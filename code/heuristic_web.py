import streamlit as st
from dbscan_heuristic import DBSCANHeuristic
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
st.pyplot(heu.plot_kdist(kdist))