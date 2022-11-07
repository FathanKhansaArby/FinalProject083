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
def head():
    st.title("Parkinsson Prediction")
    st.subheader('Based on Phyton Random Forest Classifier, up to 92 percent accuracy')
    st.caption('Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection, Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)')
    st.caption('https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection')
    name = st.text_input('yourname')
    MDVP_FO = st.number_input("MDVP_FO(Hz) - Average vocal fundamental frequency:", min_value=0.0, max_value=1000.0,step=1e-5,format="%.5f")
    MDVP_FHI = st.number_input("MDVP_FHI(Hz) - Maximum vocal fundamental frequency:", min_value=0.0, max_value=1000.0,step=1e-5,format="%.5f")
    MDVP_FloHz = st.number_input("MDVP_FLo(Hz) - Minimum vocal fundamental frequency:", min_value=0.0, max_value=1000.0,step=1e-5,format="%.5f")
    MDVP_JitterPercent = st.number_input("MDVP_JitterPercent(%) - Several measures of variation in fundamental frequency:",min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    MDVP_JitterAbs = st.number_input("MDVP_Jitter(Abs) - Several measures of variation in fundamental frequency:",min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    MDVP_RAP = st.number_input("MDVP_RAP - Several measures of variation in fundamental frequency:",min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    MDVP_PPQ = st.number_input("MDVP_PPQ - Several measures of variation in fundamental frequency:",min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    Jitter_DDP = st.number_input("Jitter_DDP - Several measures of variation in fundamental frequency:",min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    MDVP_Shimmer = st.number_input("MDVP_Shimmer - Several measures of variation in amplitude:", min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    MDVP_ShimmerDb = st.number_input("MDVP_ShimmerDb - Several measures of variation in amplitude:",min_value=0.0, max_value=10.0,step=1e-5,format="%.5f")
    Shimmer_APQ3 = st.number_input("Shimmer_APQ3 - Several measures of variation in amplitude:", min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    Shimmer_APQ5 = st.number_input("Shimmer_APQ5 - Several measures of variation in amplitude:", min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    MDVP_APQ = st.number_input("MDVP_APQ - Several measures of variation in amplitude:", min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    Shimmer_DDA = st.number_input("Shimmer_DDA - Several measures of variation in amplitude:",min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    NHR = st.number_input("NHR - measures of ratio of noise to tonal components in the voic:",min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    HNR = st.number_input("HNR - measures of ratio of noise to tonal components in the voic:",min_value=0.0, max_value=100.0,step=1e-5,format="%.5f")
    RPDE = st.number_input("RPDE - nonlinear dynamical complexity measures:",min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    DFA= st.number_input("DFA - Signal fractal scaling exponent:", min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")
    spread1 = st.number_input("spread1 - nonlinear measures of fundamental frequency variation:",min_value=-10.0, max_value=10.0,step=1e-5,format="%.5f")
    spread2 = st.number_input("spread2 - nonlinear measures of fundamental frequency variation:", min_value=0.0, max_value=10.0,step=1e-5,format="%.5f")
    D2 = st.number_input("D2 - nonlinear dynamical complexity measure:", min_value=0.0, max_value=10.0,step=1e-5,format="%.5f")
    PPE = st.number_input("PPE - nonlinear measures of fundamental frequency variation:",min_value=0.0, max_value=1.0,step=1e-5,format="%.5f")

    return [ name,MDVP_FO,MDVP_FHI, MDVP_FloHz,MDVP_JitterPercent,MDVP_JitterAbs, MDVP_RAP, MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_ShimmerDb,Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ,Shimmer_DDA,NHR, HNR, RPDE, DFA, spread1,spread2, D2, PPE]
    

def body(result):
    st.text("Parkinsson Prediction")
    st.text(result)


