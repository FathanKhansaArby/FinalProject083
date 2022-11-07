import streamlit as st
import pickle
model_random = pickle.load(open("model/forest.pkl", "rb"))

from utils import head, body

hasil = head()

if st.button("Submit"):
    name, MDVP_FO,MDVP_FHI, MDVP_FloHz,MDVP_JitterPercent,MDVP_JitterAbs, MDVP_RAP, MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_ShimmerDb,Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ,Shimmer_DDA,NHR, HNR, RPDE, DFA, spread1,spread2, D2, PPE = hasil


    input_model = [[MDVP_FO,MDVP_FHI, MDVP_FloHz,MDVP_JitterPercent,MDVP_JitterAbs, MDVP_RAP, MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_ShimmerDb,Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ,Shimmer_DDA,NHR, HNR, RPDE, DFA, spread1,spread2, D2, PPE]]
    hasil_prediksi = model_random.predict(input_model)[0]
    
    status = {0:"Healthy", 1:"Parkinsson"}

    body("Result : "+ " " + name + " current status is " + status[hasil_prediksi])
