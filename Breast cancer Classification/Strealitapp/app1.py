import streamlit as st 
import pandas as pd 
import plotly.graph_objects as go
import plotly.express as px
from bokeh.plotting import figure
import pickle
import cv2
import streamlit.components.v1 as components
st. set_page_config(layout="centered", page_icon=":hospital:")
st.set_option('deprecation.showPyplotGlobalUse', False)
import numpy as np
import pandas as pd
import altair as alt
import sweetviz as sv 
import codecs
from sklearn.preprocessing import StandardScaler
#import cancer_detection
import cv2
from keras.models import load_model
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv
from pandas_profiling import ProfileReport
import streamlit.components.v1 as components

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier

from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import hydralit_components as hc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Activation, Input, Dense, Dropout,Add, BatchNormalization,Conv1D,Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,average_precision_score,f1_score,precision_score,recall_score,roc_auc_score

from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
   
#menu_data = [
 #       {'icon': "far fa-copy", 'label':"Left End"},
  #      {'id':'Copy','icon':"🐙",'label':"Copy"},
   #     {'icon': "far fa-chart-bar", 'label':"Chart"},#no tooltip message
    #    {'icon': "far fa-address-book", 'label':"Book"},
     #   {'id':' Crazy return value 💀','icon': "💀", 'label':"Calendar"},
      #  {'icon': "far fa-clone", 'label':"Component"},
       # {'icon': "fas fa-tachometer-alt", 'label':"Dashboard",'ttip':"I'm the Dashboard tooltip!"}, #can add a tooltip message
        #{'icon': "far fa-copy", 'label':"Right End"},
#]
# we can override any part of the primary colors of the menu
#over_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
#over_theme = {'txc_inactive': '#FFFFFF'}
#menu_id = hc.nav_bar(menu_definition=menu_data,home_name='Home',override_theme=over_theme)

    
#get the id of the menu item clicked
#st.info(f"{menu_id=}")


st.sidebar.image('fight_like_girl.jpg', channels="BGR")


    
st.sidebar.title("Navigation")
page=st.sidebar.radio("Click on the radio button to know more",['Overview','EDA','Visualization','Classification','Hypertuning','Result','Predict Unseen Data','Image Prediction'])

st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 5px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)
#page=st.sidebar.beta_expander(st.sidebar.title("Hypertuning"), expanded=False)
#page=st.sidebar.expander ("Hypertuning", expanded=False)
#st.sidebar.beta_expander("Hypertuning", expanded=False):

pickle_in = open('model.pkl', 'rb') 
classifier = pickle.load(pickle_in)
scaler = pickle.load(open('scaler.pkl', 'rb'))
    
df=pd.read_csv("data.csv")
df=df.drop(['id',"Unnamed: 32"],axis=1)
X=df.drop(['diagnosis'],axis=1)
y=df['diagnosis']
le=LabelEncoder()
y=le.fit_transform(y)



fs_corr = ['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean','symmetry_mean',
                     'fractal_dimension_mean', 'texture_se', 'area_se','smoothness_se', 'concavity_se',
                     'symmetry_se', 'fractal_dimension_se','smoothness_worst', 'concavity_worst', 
                     'symmetry_worst', 'fractal_dimension_worst']
    
    # 2. Univariate feature selection SelectKBest, chi2
fs_chi2 = ['texture_mean', 'area_mean', 'concavity_mean', 'symmetry_mean', 'area_se', 
                     'concavity_se', 'smoothness_worst', 'concavity_worst', 'symmetry_worst', 
                     'fractal_dimension_worst']
    
    # 3. Recursive feature elimination (RFE) with random forest
fs_rfe = ['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'area_se', 
              'smoothness_se', 'concavity_se', 'smoothness_worst', 'concavity_worst', 'symmetry_worst']
    
    # 4. Recursive feature elimination with cross validation(RFECV) with random forest
fs_rfecv = ['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean','fractal_dimension_mean'
                , 'area_se', 'concavity_se', 'concavity_worst', 'symmetry_worst']
    
    # 5. Tree based feature selection with random forest classification
fs_rf = ['texture_mean', 'area_mean', 'concavity_mean', 'area_se', 'concavity_se', 
             'fractal_dimension_se', 'smoothness_worst','concavity_worst', 'symmetry_worst', 
             'fractal_dimension_worst']
    
    # 6. ExtraTree based feature selection 
fs_extraTree = ['texture_mean', 'area_mean', 'concavity_mean', 'fractal_dimension_mean', 'area_se', 
                    'concavity_se','smoothness_worst', 'concavity_worst', 
                    'symmetry_worst','fractal_dimension_worst']
    

    # 8. Vote based feature selection
fs_voted = ['texture_mean',  'area_mean',  'smoothness_mean',  'concavity_mean',  
                     'fractal_dimension_mean',  'area_se',  'concavity_se',  'smoothness_worst',  
                     'concavity_worst',  'symmetry_worst',  'fractal_dimension_worst']




if page=="Overview":
    st.title("Overview")
    st.write(df)
    st.write("Describe")
    st.write(df.describe())
    st.write("Shape",df.shape)

elif page=="Visualization":

        #st.write(df)
        chart_select = st.selectbox(label = "Select Chart Type", options = ['ScatterPlot', 'Heatmap','3D Linechart', 'Histogram', 'BoxPlot','Bar'])


        if chart_select == "ScatterPlot":

            st.subheader("Settings")
            x_values = st.selectbox('X axis', options = df.columns)
            y_values = st.selectbox('Y axis', options = df.columns)
            plot = px.scatter(data_frame = df, x=x_values, y=y_values)
            st.plotly_chart(plot)

        if chart_select == "3D Linechart":

            st.subheader("Settings")
            x_values = st.selectbox('X axis', options = df.columns)
            y_values = st.selectbox('Y axis', options = df.columns)
            z_values = st.selectbox('Z axis', options = df.columns)
            plot = px.line_3d(data_frame = df, x=x_values, y=y_values, z = z_values)
            st.plotly_chart(plot)


     
        if chart_select == "Histogram":

            st.subheader("Settings")
            x_values = st.selectbox('X axis', options = df.columns)
            #y_values = st.selectbox('Y axis', options = df.columns)
            plot = px.histogram(data_frame = df, x=x_values, hover_data = df.columns)
            st.plotly_chart(plot)

        if chart_select == "BoxPlot":

            st.subheader("Settings")
            x_values = st.selectbox('X axis', options = df.columns)
            y_values = st.selectbox('Y axis', options = df.columns)
            plot = px.box(data_frame = df, x=x_values, y=y_values)
            st.plotly_chart(plot)


        if chart_select == "Heatmap":

            st.subheader("Settings")
            x_values = st.selectbox('X axis', options = df.columns)
            y_values = st.selectbox('Y axis', options = df.columns)
            plot = px.density_heatmap(data_frame = df, x=x_values, y=y_values)
            st.plotly_chart(plot)

        if chart_select == "Bar":

            st.subheader("Settings")
            x_values = st.selectbox('X axis', options = df.columns)
            y_values = st.selectbox('Y axis', options = df.columns)
            plot = px.histogram(data_frame = df, x=x_values, y=y_values, orientation = "v")
            st.plotly_chart(plot)
            
 
#    st.title("Visualization")
 #   sel_col=st.selectbox("select columns",df.columns)
  #  st.line_chart(df[sel_col])

   # x_axis=st.selectbox("Select columns x axis",df.columns)
    #y_axis=st.selectbox("Select columns y axis",df.columns)
    #st.write(alt.Chart(df).mark_point().encode(
    #    x = alt.X(x_axis, type="quantitative", title=x_axis),
     #   y = alt.Y(y_axis, type="quantitative", title=y_axis),
    #))
elif page=='Classification':
    classifier_name=st.selectbox(
        'select classifier',
        ('KNN','Random Forest','Logistic Regression','GradientBoostingClassifier','SVM','AdaBoostClassifier','CatBoostClassifier','ExtraTreesClassifier','XGBClassifier','DecisionTreeClassifier',"ANN",'CNN')
    )
    feature_name=st.selectbox('select Feature',
        ('fs_corr','fs_chi2','fs_rfe','fs_rfecv','fs_rf','fs_extraTree','fs_voted')
    )    

    if(feature_name=='fs_corr'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_corr
    elif(feature_name=='fs_chi2'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_chi2
    elif(feature_name=='fs_rfe'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_rfe
    elif(feature_name=='fs_rfecv'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_rfecv
    elif(feature_name=='fs_rf'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_rf
    elif(feature_name=='fs_extraTree'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_extraTree
    elif(feature_name=='fs_voted'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_voted

    
    def cal_result(X_test,y_test,clf,y_pred):
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Classifier : {classifier_name}')
        st.write(f'Accuracy :', acc)
        binary1=classification_report(y_pred,y_test)
        class_names = [1, 0]
        plot_confusion_matrix(clf,X_test,y_test,display_labels =class_names)
        st.set_option('deprecation.showPyplotGlobalUse',False)
        st.pyplot(figsize=(7,5))
        plot_precision_recall_curve(clf,X_test,y_test)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(figsize=(7,5))

    def plot_deep(history):
        st.title('Loss Vs Epochs')
        st.line_chart(history.history['loss'])
        st.title('validation loss Vs Epochs')
        st.line_chart(history.history['val_loss'])
        st.title('Binary Accuracy Vs Epochs')
        st.line_chart(history.history['binary_accuracy'])
        st.title('validation Binary Accuracy Vs Epochs')
        st.line_chart(history.history['val_binary_accuracy'])        
    
    def apply_algo(classifier_name,feature_name,X,y):
        if(classifier_name=='KNN' or classifier_name=='Random Forest' or classifier_name=='CatBoostClassifier'or classifier_name=='AdaBoostClassifier'or classifier_name=='DecisionTreeClassifier'or classifier_name=='Logistic Regression' or classifier_name=='GradientBoostingClassifier' or classifier_name=='SVM' or classifier_name=='GradientBoostingClassifier' or classifier_name=='ExtraTreesClassifier' or classifier_name=='XGBClassifier'):
            if(classifier_name=='KNN'):
                
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=KNeighborsClassifier()
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)

            elif(classifier_name=='Random Forest'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=RandomForestClassifier()
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)

            elif(classifier_name=='CatBoostClassifier'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=CatBoostClassifier()
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)

            elif(classifier_name=='AdaBoostClassifier'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=AdaBoostClassifier()
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)

            elif(classifier_name=='DecisionTreeClassifier'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=DecisionTreeClassifier()
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)
            
            elif(classifier_name=='Logistic Regression'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=LogisticRegression()
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)      

            elif(classifier_name=='GradientBoostingClassifier'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=GradientBoostingClassifier()
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)    

            elif(classifier_name=='SVM'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=clf = SVC(kernel='rbf',probability=True)
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)   

            elif(classifier_name=='ExtraTreesClassifier'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=clf = ExtraTreesClassifier()
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)   

            elif(classifier_name=='XGBClassifier'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=clf = XGBClassifier()
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)       

        else:
            st.write("Deep learning")
            le=LabelEncoder()
            y=le.fit_transform(y)
            if classifier_name=="ANN":
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
                input_tensor=Input(shape=(len(feature_name),))
                D1=Dense(512, input_shape=(30,))(input_tensor)
                A1=Activation('relu')(D1)
                A1=Dropout(0.5)(A1)
                A1=BatchNormalization()(A1)
                D2=Dense(512)(A1)
                A2=Activation('relu')(D2)
                A2=Dropout(0.5)(A2)
                A2=BatchNormalization()(A2)
                D3=Dense(512)(A2)
                A3=Activation('relu')(D3)
                A3=Dropout(0.5)(A3)
                A3=BatchNormalization()(A3)
                D4=Dense(512)(A3)
                D4=Add()([D4,A1])
                A4=Activation('relu')(D4)
                A4=Dropout(0.5)(A4)
                A4=BatchNormalization()(A4)
                D5=Dense(256)(A4)
                A5=Activation('relu')(D5)
                A5=Dropout(0.2)(A5)
                A5=BatchNormalization()(A5)
                D6=Dense(256)(A5)
                A6=Activation('relu')(D6)
                A6=Dropout(0.2)(A6)
                A6=BatchNormalization()(A6)
                output_tensor=Dense(1, activation='sigmoid')(A6)
                functional_model=Model(inputs= input_tensor,outputs=output_tensor)
                functional_model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['binary_accuracy'])
                history=functional_model.fit(X_train,y_train,
                        validation_data=(X_test,y_test),
                        epochs=50,batch_size=128)
                y_pred=functional_model.predict(X_test)
                y_pred=(y_pred>0.5)
                acc = accuracy_score(y_test, y_pred)
                st.write(f'Classifier : {classifier_name}')
                st.write(f'Accuracy :', acc)
                plot_deep(history)

                    

            elif classifier_name=='CNN':
                X=X[feature_name]
                X = X.values.reshape(X.shape[0], X.shape[1], 1)
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)
                model = Sequential()
                model.add(Conv1D(32, 2, activation="relu", input_shape=(len(feature_name), 1)))
                model.add(Flatten())
                model.add(Dense(64, activation="relu"))
                model.add(Dense(1,activation="sigmoid"))
                model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
                history=model.fit(X_train, y_train, batch_size=12,epochs=20,validation_data=(X_test,y_test))   
                y_pred=model.predict(X_test) 
                y_pred=(y_pred>0.5)
                acc = accuracy_score(y_test, y_pred)
                st.write(f'Classifier : {classifier_name}')
                st.write(f'Accuracy :', acc)
                plot_deep(history)        

    apply_algo(classifier_name,feature_name,X,y)
elif page=='Result':
     st.title("Result Page")
     st.image('final_image.PNG', channels="BGR")
     st.image('feature.png',channels='BGR')


elif page=='EDA':
     st.subheader("Automated EDA with Sweetviz")
     report = sv.analyze(df)
     report.show_html()
     report_file = codecs.open("SWEETVIZ_REPORT.html",'r')
     page = report_file.read()
     components.html(page, width=2000, height=750, scrolling=True)
     #components.html(page,width=width,height=height,scrolling=True)

elif page=='Image Prediction':
     model = load_model("CNN_image_best_model_17102021.h5")
     st.title("Prediction")
     st.write("Upload Image")

     img = st.file_uploader("Choose an image...")
     if img is not None:
        img = Image.open(img)

        img.save("temp.png")
        img = "temp.png"

        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img_size = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
        img_size = cv2.normalize(
        img_size, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

        img = img_size.reshape(-1, 50, 50, 3)
        ans = model.predict(img)

        st.write("")
        st.write("Classifying...")
        for x in ans:
            if x > 0.5:
                st.write("Malignant")
            else:
                st.write("Benign")
    
elif page=='Hypertuning':
    classifier_name=st.selectbox(
        'select classifier',
        ('KNN','Random Forest','Logistic Regression','GradientBoostingClassifier','SVM','ExtraTreesClassifier','XGBClassifier',"Decision Trees",'AdaBoostClassifier','CatBoostClassifier')
    )
    feature_name=st.selectbox('select Feature',
        ('fs_corr','fs_chi2','fs_rfe','fs_rfecv','fs_rf','fs_extraTree','fs_voted')
    )    

    if(feature_name=='fs_corr'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_corr
    elif(feature_name=='fs_chi2'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_chi2
    elif(feature_name=='fs_rfe'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_rfe
    elif(feature_name=='fs_rfecv'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_rfecv
    elif(feature_name=='fs_rf'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_rf
    elif(feature_name=='fs_extraTree'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_extraTree
    elif(feature_name=='fs_voted'):
        st.write(f'Feature Select:{feature_name}')
        feature_name=fs_voted

    def cal_result(X_test,y_test,clf,y_pred):
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Classifier : {classifier_name}')
        st.write(f'Accuracy :', acc)
        binary1=classification_report(y_pred,y_test)
        class_names = [1, 0]
        plot_confusion_matrix(clf,X_test,y_test,display_labels =class_names)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        plot_precision_recall_curve(clf,X_test,y_test)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    def plot_deep(history):
        st.title('Loss Vs Epochs')
        st.line_chart(history.history['loss'])
        st.title('validation loss Vs Epochs')
        st.line_chart(history.history['val_loss'])
        st.title('Binary Accuracy Vs Epochs')
        st.line_chart(history.history['binary_accuracy'])
        st.title('validation Binary Accuracy Vs Epochs')
        st.line_chart(history.history['val_binary_accuracy'])        

    def add_parameter_ui(clf):
        params={}
        st.sidebar.write("Select values: ")

        if clf == "Logistic Regression":
            R = st.sidebar.slider("Regularization",1,100,step=1)
            MI = st.sidebar.slider("max_iter",10,400,step=50)
            params["R"] = R
            params["MI"] = MI

        elif clf == "KNN":
            K = st.sidebar.slider("n_neighbors",1,20)
            params["K"] = K
            
        elif clf == "ExtraTreesClassifier":
            K = st.sidebar.slider("n_neighbors",1,1200)
            params["K"] = K
            
        elif clf == "AdaBoostClassifier":
            N = st.sidebar.slider("n_estimators",50,1200,step=50,value=100)
            params["N"] = N

           
        elif clf == "CatBoostClassifier":
            N = st.sidebar.slider("n_estimators",50,1200,step=50,value=100)
            params["N"] = N
            #LR = st.sidebar.slider("Learning_Rate",0.01,1.0)
             
        elif clf == "SVM":
            C = st.sidebar.slider("Regularization",0.01,10.0,step=0.01)
            kernel = st.sidebar.selectbox("Kernel",("linear", "poly", "rbf", "sigmoid", "precomputed"))
            params["C"] = C
            params["kernel"] = kernel
    
        elif clf == "Decision Trees":
            M = st.sidebar.slider("max_depth", 2, 20)
            C = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
            SS = st.sidebar.slider("min_samples_split",1,10)
            params["M"] = M
            params["C"] = C
            params["SS"] = SS

        elif clf == "Random Forest":

            N = st.sidebar.slider("n_estimators",50,1200,step=50,value=100)
            M = st.sidebar.slider("max_depth",2,100)
            C = st.sidebar.selectbox("Criterion",("gini","entropy"))
            params["N"] = N
            params["M"] = M
            params["C"] = C

        elif clf == "Gradient Boosting":
            
            N = st.sidebar.slider("n_estimators",50,1200,step=50,value=100)
            LR = st.sidebar.slider("Learning_Rate",0.01,1)
            L = st.sidebar.selectbox("Loss", ('deviance', 'exponential'))
            M = st.sidebar.slider("max_depth",2,100)
            params["N"] = N
            params["LR"] = LR
            params["L"] = L
            params["M"] = M

        elif clf == "XGBoost":
            N = st.sidebar.slider("n_estimators",50,1200,step=50,value=100)
            LR = st.sidebar.slider("Learning_Rate",0.01,1.0,value=0.1)
            O = st.sidebar.selectbox("Objective", ('binary:logistic','reg:logistic','reg:squarederror',"reg:gamma"))
            M = st.sidebar.slider("max_depth",1,20,value=6)
            G = st.sidebar.slider("Gamma",0,10,value=5)
            L = st.sidebar.slider("reg_lambda",1.0,5.0,step=0.1)
            A = st.sidebar.slider("reg_alpha",0.0,5.0,step=0.1)
            CS = st.sidebar.slider("colsample_bytree",0.5,1.0,step=0.1)
            params["N"] = N
            params["LR"] = LR
            params["O"] = O
            params["M"] = M
            params["G"] = G
            params["L"] = L
            params["A"] = A
            params["CS"] = CS

        #RS=st.sidebar.slider("Random State",0,100)
        #params["RS"] = RS
        return params

    params = add_parameter_ui(classifier_name)

    def apply_algo1(classifier_name,feature_name,X,params,y):
        #if(classifier_name=='KNN' or classifier_name=='Random Forest' or classifier_name=='Logistic Regression' or classifier_name=='GradientBoostingClassifier' or classifier_name=='SVM' or classifier_name=='GradientBoostingClassifier' or classifier_name=='ExtraTreesClassifier' or classifier_name=='XGBClassifier'):
            if(classifier_name=='KNN'):
                
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=43)
                clf=KNeighborsClassifier(n_neighbors=params["K"])
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)
            
            elif(classifier_name=='Random Forest'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=43)
                clf=RandomForestClassifier(n_estimators=params["N"],max_depth=params["M"],criterion=params["C"])
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)
            
            elif(classifier_name=='Logistic Regression'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=43)
                clf=LogisticRegression(C=params["R"],max_iter=params["MI"])
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)      

            elif(classifier_name=='GradientBoostingClassifier'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=GradientBoostingClassifier(n_estimators=params["N"],Learning_Rate=params["LR"],loss=params["L"],max_depth=params["M"])
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)
                
            elif(classifier_name=='SVM'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=clf = SVC(kernel=params["kernel"],C=params["C"])
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)      

            elif(classifier_name=='Decision Trees'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=clf = DecisionTreeClassifier(max_depth=params["M"],criterion=params["C"],min_impurity_split=params["SS"])
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)   

            elif(classifier_name=='ExtraTreesClassifier'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=clf = ExtraTreesClassifier(n_estimators=params["N"])
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)   

            elif(classifier_name=='XGBClassifier'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=clf = XGBClassifier(booster="gbtree",n_estimators=params["N"],max_depth=params["M"],Learning_Rate=params["LR"],
                            objective=params["O"],gamma=params["G"],reg_alpha=params["A"],reg_lambda=params["L"],colsample_bytree=params["CS"])
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)

            elif(classifier_name=='AdaBoostClassifier'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                clf=clf = AdaBoostClassifier(n_estimators=params["N"])
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)

            elif(classifier_name=='CatBoostClassifier'):
                X=X[feature_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)
                #clf=clf = CatBoostClassifier(n_estimators=params["N"],Learning_Rate=params["LR"])
                clf=clf = CatBoostClassifier(n_estimators=params["N"])
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                cal_result(X_test,y_test,clf,y_pred)   


    apply_algo1(classifier_name,feature_name,X,params,y)
    
elif page=='Predict Unseen Data':
    st.title("Prediction")

#    texture_mean=st.slider('texture_mean', X.texture_mean.min(), X.texture_mean.max(), X.texture_mean.mean())
 #   area_mean=st.slider('area_mean', X.area_mean.min(), X.area_mean.max(), X.area_mean.mean())
  #  concavity_mean=st.slider('concavity_mean', X.concavity_mean.min(), X.concavity_mean.max(), X.concavity_mean.mean())
   # fractal_dimension_mean=st.slider('fractal_dimension_mean', X.fractal_dimension_mean.min(), X.fractal_dimension_mean.max(), X.fractal_dimension_mean.mean())

    #area_se=st.slider('area_se', X.area_se.min(), X.area_se.max(), X.area_se.mean())
    #concavity_se=st.slider('concavity_se', X.concavity_se.min(), X.concavity_se.max(), X.concavity_se.mean())
    #smoothness_worst=st.slider('smoothness_worst', X.smoothness_worst.min(), X.smoothness_worst.max(), X.smoothness_worst.mean())
   # concavity_worst=st.slider('concavity_worst', X.concavity_worst.min(), X.concavity_worst.max(), X.concavity_worst.mean())
   # symmetry_worst=st.slider('symmetry_worst', X.symmetry_worst.min(), X.symmetry_worst.max(), X.symmetry_worst.mean())
   # fractal_dimension_worst=st.slider('fractal_dimension_worst', X.fractal_dimension_worst.min(), X.fractal_dimension_worst.max(), X.fractal_dimension_worst.mean())


    col1, col2 ,col3 = st.columns(3)

    with col1:
  
        texture_mean=st.number_input("texture_mean")
        area_mean=st.number_input("area_mean")
        concavity_mean=st.number_input("concavity_mean")
        fractal_dimension_mean=st.number_input("fractal_dimension_mean")
	

    with col2:  

        area_se=st.number_input("area_se")
        concavity_se=st.number_input("concavity_se")
        smoothness_worst=st.number_input("smoothness_worst")
        
    with col3: 
        concavity_worst=st.number_input("concavity_worst")
        symmetry_worst=st.number_input("symmetry_worst")
        fractal_dimension_worst=st.number_input("fractal_dimension_worst")

    def prediction(texture_mean,area_mean,concavity_mean,fractal_dimension_mean,area_se, concavity_se,smoothness_worst,concavity_worst,symmetry_worst,fractal_dimension_worst):
        #final_features = scaler.transform(texture_mean,area_mean,concavity_mean,fractal_dimension_mean,area_se, concavity_se,smoothness_worst,concavity_worst,symmetry_worst,fractal_dimension_worst)   
        prediction = classifier.predict([[texture_mean,area_mean,concavity_mean,fractal_dimension_mean,area_se,concavity_se,smoothness_worst,concavity_worst,symmetry_worst,fractal_dimension_worst]])
        #prediction = classifier.predict(final_features)
        if prediction <5:
            pred = 'Benign(noncancerous)'
        else:
            pred = 'Malignant(cancerous)'
        return pred


    if st.button("Predict"): 
        result = prediction(texture_mean,area_mean,concavity_mean,fractal_dimension_mean,area_se,concavity_se,smoothness_worst,concavity_worst,symmetry_worst,fractal_dimension_worst) 
        st.success('The Result is  {}'.format(result))
