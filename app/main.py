import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import pickle
import sklearn.tree as tree
from six import StringIO 
from IPython.display import Image
model_random = pickle.load(open("model/forest.pkl", "rb"))

from utils import head, body

hasil = head()

if st.button("Submit"):
    name, MDVP_FO,MDVP_FHI, MDVP_FloHz,MDVP_JitterPercent,MDVP_JitterAbs, MDVP_RAP, MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_ShimmerDb,Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ,Shimmer_DDA,NHR, HNR, RPDE, DFA, spread1,spread2, D2, PPE = hasil


    input_model = [[MDVP_FO,MDVP_FHI, MDVP_FloHz,MDVP_JitterPercent,MDVP_JitterAbs, MDVP_RAP, MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_ShimmerDb,Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ,Shimmer_DDA,NHR, HNR, RPDE, DFA, spread1,spread2, D2, PPE]]
    hasil_prediksi = model_random.predict(input_model)[0]
    
    status = {0:"Healthy", 1:"Parkinsson"}

    body("Result : "+ " " + name + " current status is " + status[hasil_prediksi])