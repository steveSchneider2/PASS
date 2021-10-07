"""
Created on Thu Sep 16 18:16:19 2021

@author: steve
An example of showing geographic data.
open terminal, cd to github\streamlit
streamlit run  streamlitTraffic.py
Shows traffic accidents in a map in the web-app.
"""

import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import pyodbc, os
#from pandas_profiling import ProfileReport #needs pandas < 1.3 Pandas v1.3 renamed the ABCIndexClass to ABCIndex. The visions dependency of the pandas-profiling package hasn't caught up yet
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")

st.title('Car Crash App  \nStreamlit: {:.6s}  Conda environment: {:12s}'.
         format(st.__version__,os.environ['CONDA_DEFAULT_ENV']))

# LOADING DATA
DATE_TIME = "date/time"

#%% Set up to get data!

def getdata(records):
    start = datetime.now()
    conn = pyodbc.connect('DRIVER={SQL Server};'
                          'SERVER=(local);'
                          'DATABASE=Traffic;'
                          'Trusted_Connection=yes;')
    SqlParameters = f'@cnt={records},@yr=2014,@veh#=1,@day1=3,@day2=28'
#    SqlParameters = '@cnt=35000,@yr=2014,@veh#=2,@day1=11,@day2=19'
#    SQL = 'usp_GetEqualNbrMajorMinorCrashesA '
    SQL = 'usp_GetEqualNbrMajorMinorCrashesPASS '
#    SQL = 'select * from Crash110'
    SQLCommand = SQL
    SQLCommand = SQL + SqlParameters
    # SQLCommand = ("usp_GetEqualNbrMajorMinorCrashes  @cnt=100,@yr=2014,@veh#=1")
    # SQLCommand = 'select * from vw_CrashSubsetEngineeredR'  #' where vehmph > 0 '
    dt_string = start.strftime("%m/%d/%Y %H:%M:%S")
    dt_string = start.strftime("%m/%d/%Y %H:%M")
    dt_string
#    url = 'https://github.com/steveSchneider2/data/tree/main/FloridaTraffic/traffic116k_88_76pc.csv?raw=true'
    #url = 'https://github.com/topics/floridatraffic.csv'
#    pr = pd.read_csv(url)
#    pr = pd.read_csv('D:/ML/Python/data/traffic/traffic236k_95_16.csv', header=0)
#    pr = pd.read_csv('data/traffic72k_94_83.csv', header=0)
    pr = pd.read_csv('C:/Users/steve/Documents/GitHub/Misc/trafficPASS.csv')
    #pr = pd.read_sql(SQLCommand, conn)
    end = datetime.now()
    processTime = end - start
    print('Seconds to download SQL records is: ', processTime.total_seconds())
    #pr.to_csv(r'trafficPASS.csv', index=False)
    return pr, SQLCommand, SQL, SqlParameters, dt_string
#%% Not using this
@st.cache(persist=True)
def load_data(nrows):
    data, SQLcmd, sql, sqlParameters, dt_string= getdata(nrows)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis="columns", inplace=True)
    # data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
    return data
data = load_data(118000)
# CREATING FUNCTION FOR MAPS
#%% 
#['crash', 'vehmph', 'drvage', 'vehage', 'dayhour', 'speeddif', 'vehmakegrp', 'wday', 'vehtype', 'weather', 'light', 'roadtyp', 'UrbanLoc', 'vhmvnt', 'vhmtn', 'license', 'drDistract', 'drSitu', 'Vision', 'drvsex', 'vehicle_number', 'monthday', 'Crash_year', 'rdSurfac', 'vMphGrp', 'drvagegrp', 'hitrun', 'latitude', 'longitude']
data1 = data[['crash', 'vehmph', 'drvage', 'vehage', 'dayhour', 'speeddif', 
        'vehmakegrp', 'wday', 'vehtype', 
        #'weather', 'light', 'roadtyp', 
        #'UrbanLoc', 'vhmvnt', 'vhmtn', 'license', 'drDistract', 'drSitu', 
        #'Vision', 'drvsex', 'vehicle_number', 'monthday', 'Crash_year', 
        #'rdSurfac', 'vMphGrp', 'drvagegrp', 'hitrun', 
        'latitude', 'longitude']]
#data = data1[['dayhour','latitude','longitude']]
data1.latitude =data.latitude.astype(float).round(3)
data1.longitude =data.longitude.astype(float).round(3)
def map(data1, latitude, longitude, zoom):
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": latitude,
            "longitude": longitude,
            "zoom": zoom,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data1,
                get_position=['longitude', 'latitude'],
                radius=100,
                elevation_scale=4,
#                get_color='[200 , 30, 0, 160]',
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ]
    ))

# LAYING OUT THE TOP SECTION OF THE APP
if st.__version__ == '0.81.1':
    row1_1, row1_2 = st.beta_columns((2,3))
else: 
    row1_1, row1_2 = st.columns((2,3))
weekdays = ('Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday')
with row1_1:
    st.title("Jacksonville area accidents")
    hour_selected = st.slider("Select hour of crash", 0, 23, 12)
#    weekday = st.radio("days:", weekdays, 1)
    drvage = st.slider('max driver age:',0, 100, 18)

with row1_2:
    st.write(
    """
    ##
    Examining how accidents vary over time in the Jacksonville area.
    By sliding the slider on the left you can view different slices of time and explore different transportation trends.
    """)
# with row2_1:
#     st.title('Week day:')
    

# FILTERING DATA BY HOUR SELECTED
#data = data[data.dayhour == hour_selected & data.drvage.lt(drvage)]
data1 = data1[data.dayhour.ge(hour_selected)
#         & data.wday == weekday
         & data.drvage.ge(drvage)]

# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
if st.__version__ == '0.81.1':
    row2_1, row2_2 = st.beta_columns((2,1))
else: 
    row2_1, row2_2 = st.columns((2,1))

# SETTING THE ZOOM LOCATIONS FOR THE AIRPORTS
jacksonville= [30.33147, -81.65622]
miami       = [25.76168, -80.19178]
#newark = [40.7090, -74.1805]
zoom_level = 12
midpoint = (np.average(data.latitude), np.average(data.longitude))

with row2_1:
    st.write("**Jacksonville from %i:00 and %i:00**" % 
             (hour_selected, (hour_selected + 1) % 24))
    map(data1, jacksonville[0], jacksonville[1], 11)

with row2_2:
    st.write("**Miami**")
    map(data1, miami[0],miami[1], zoom_level)

# FILTERING DATA FOR THE HISTOGRAM
filtered = data1[
    (data1.dayhour >= hour_selected) & (data1.dayhour < (hour_selected + 1))
    ]

#hist = np.histogram(filtered.dayhour, bins=60, range=(0, 60))[0]

#chart_data = pd.DataFrame({"minute": range(60), "pickups": hist})

# LAYING OUT THE HISTOGRAM SECTION
'''
st.write("")

st.write("**Breakdown of rides per minute between %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))

st.altair_chart(alt.Chart(chart_data)
    .mark_area(
        interpolate='step-after',
    ).encode(
        x=alt.X("minute:Q", scale=alt.Scale(nice=False)),
        y=alt.Y("pickups:Q"),
        tooltip=['minute', 'pickups']
    ).configure_mark(
        opacity=0.5,
        color='red'
    ), use_container_width=True)
'''