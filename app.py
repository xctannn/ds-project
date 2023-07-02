import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('data_clean.csv')
df_train = pd.read_csv('X_train.csv')
features = df_train.columns.to_list()

def get_features_mean():
    means_dict = {}
    for i in range(len(features)):
        means_dict.update({features[i]: df_train[features[i]].mean()})
    return means_dict

def get_features_std_dev():
    std_dict = {}
    for i in range(len(features)):
        std_dict.update({features[i]: df_train[features[i]].std()})
    return std_dict

def standardize_data(col_name, value):
    means_dict = get_features_mean()
    std_dict = get_features_std_dev()
    return (value - means_dict[col_name])/std_dict[col_name]

yes_no_op = {
    'Yes': 1,
    'No': 0
}

studytime_op = {
    '<2 hours': 1,
    '2 to 5 hours': 2,
    '5 to 10 hours': 3,
    '>10 hours': 4,
}

good_bad_op = {
    'Very bad': 1,
    'Bad': 2,
    'Normal': 3,
    'Good': 4,
    'Very Good': 5,
}

goout_op = {
    'Not Often': 1,
    'Less Often': 2,
    'Often': 3,
    'Slightly Often': 4,
    'Very Often': 5,
}

failures_op = {
    "Never Failed": 0,
    "Failed once": 1,
    "Failed 2-3 times": 2,
    "Failures > 4": 3,
}

def get_user_inputs():
    age = st.number_input("Age", min_value=15, max_value=22, value=15)
    activities = st.selectbox("Active in Extra-Curricular Activities?", list(yes_no_op.keys()))
    absences = st.slider("Number of school absences", 0,30,5)
    failures = st.select_slider("Number of Past Class Failures",list(failures_op.keys()))
    col1, col2 = st.columns(2)
    with col1:
        studytime = st.select_slider("Weekly Study Time",list(studytime_op.keys()),"2 to 5 hours")
        health = st.select_slider("Current Health Status", list(good_bad_op.keys()), "Good")  
        g1 = st.number_input("First Period Grade (%)", min_value= 0, max_value = 100, value = 50)
    with col2:
        goout = st.select_slider("Going Out with Friends", list(goout_op.keys()), 'Often')
        famrel = st.select_slider("Quality of Family Relatonsips",list(good_bad_op.keys()),"Normal")
        g2 = st.number_input("Second Period Grade (%)", min_value= 0, max_value = 100, value = 50)
    
    df_pred = pd.DataFrame(data=np.zeros(shape=(1, len(features))), columns=features, dtype='float')
    df_pred['age'] = standardize_data('age', age)
    df_pred['activities'] = standardize_data('activities', yes_no_op[activities])
    df_pred['absences'] = standardize_data('absences', absences)
    df_pred['health'] = standardize_data('health', good_bad_op[health])
    df_pred['failures'] = standardize_data('failures', failures_op[failures])
    df_pred['famrel'] = standardize_data('famrel', good_bad_op[famrel])
    df_pred['goout'] = standardize_data('goout', goout_op[goout])
    df_pred['studytime'] = standardize_data('studytime', studytime_op[studytime])
    df_pred['G1'] = standardize_data('G1', g1/5)
    df_pred['G2'] = standardize_data('G2', g2/5)

    return df_pred

def plot_G1_G3():
    rc = {'figure.figsize':(5,5),
        'axes.facecolor':'#0e1117',
        'axes.edgecolor': '#0e1117',
        'axes.labelcolor': 'white',
        'figure.facecolor': '#0e1117',
        'patch.edgecolor': '#0e1117',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'grey',
        'font.size' : 8,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ax = sns.regplot(x=df['G1']*5, y=df['G3']*5, x_jitter=.1, color = '#f21111',scatter_kws={"color": "#f21111"},line_kws={"color": "#c2dbfc"})
    ax.set(title='First Period Grade vs Final Grade', xlabel = 'First Period Grade (%)', ylabel = 'Final Grade (%)')
    st.pyplot(fig)

def plot_G2_G3():
    fig, ax = plt.subplots()
    ax = sns.regplot(x=df['G2']*5, y=df['G3']*5)
    ax.set(title='Second Period Grade vs Final Grade', xlabel = 'Second Period Grade (%)', ylabel = 'Final Grade (%)')
    st.pyplot(fig)

def main():
    # st.set_page_config(layout="wide")
    st.title("Predicting Student Performance in Secondary Education")
    col1, col2 = st.columns(2,gap='large')
    with col1: 
        plot_G1_G3()
    with col2:
        plot_G2_G3()
    st.markdown('First period grade and second period grade have a significance impact on the Final Grade.')
    st.divider()

    st.subheader("Student's information:")
    df_pred = get_user_inputs()

    model = joblib.load('linear_reg_model.joblib')
    value_pred = model.predict(df_pred)[0]*5 
    if value_pred < 0:
        value_pred = 0
    pred = f"{value_pred:.2f}%"
    st.divider()
    st.metric(label="The final grade score is estimated to be", value=pred)

main()