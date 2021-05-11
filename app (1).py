import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/drive/My Drive/my_model.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('/content/drive/My Drive/ML_Lab/Classification Dataset3.csv')
# Extracting independent variable:
X = dataset.iloc[:,0:9].values
# Extracting dependent variable:
y = dataset.iloc[:, 9].values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,[1,2,3,4,5,6,7]]) 
#Replacing missing data with the calculated mean value  
X[:,[1,2,3,4,5,6,7]]= imputer.transform(X[:,[1,2,3,4,5,6,7]]) 
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y[:] = labelencoder_y.fit_transform(y[:])
y=y.astype('int')
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

def predict_note_authentication(meanfreq,sd,median,
                                          IQR,skew,kurt,
                                          mode,centroid,dfrange):
  output= model.predict(sc_x.transform([[meanfreq,sd,median,
                                          IQR,skew,kurt,
                                          mode,centroid,dfrange]]))
  print("Voice Recognition =", output)
  print("Voice Recognition", output)
  if output==[0]:
   print("Voice is of Female 0")
  elif output==[1]:
   print("Voice is of Male 1")
  return output

def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Voice Recognition using NaiveBayes Classifier")

    meanfreq = st.number_input('Insert meanfreq')
    sd = st.number_input('Insert sd')
    median = st.number_input('Insert Median')
    IQR = st.number_input('Inert IQR')
    skew = st.number_input('Insert skew')
    kurt = st.number_input('Insert kurt')
    mode = st.number_input('Insert mode')
    centroid = st.number_input('Insert centriod')
    dfrange = st.number_input('Insert dfrange')
    
    resul=""
    if st.button("Prediction"):
      result=predict_note_authentication(meanfreq,sd,median,
                                          IQR,skew,kurt,
                                          mode,centroid,dfrange)
      st.success('Model has predicted Voice Recognition {}'.format(result))  
    if st.button("About"):
      st.header("Developed by Palak Jain")
      st.subheader("Student, Department of Computer Engineering")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Experiment : NaiveBayes Classification</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()