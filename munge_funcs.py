import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

from mord import LogisticAT

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import numpy as np


def df_clean (df,cat_cols,target_col):
    
    fm_cols = ['Benefits Management Importance', 'Benefits Management Satisfaction',
       'Bill Collection Services Importance',
       'Bill Collection Services Satisfaction', 'Bill Payment Importance',
       'Bill Payment Support Satisfaction', 'Budget Execution Importance',
       'Budget Execution Support Satisfaction',
       'Budget Formulation Importance', 'Budget Formulation Satisfaction','Financial Management Information Satisfaction',
       'Financial Management Information and Analysis Importance','Financial Management Strategic Partner Satisfaction',
       'Financial Management System',
       'Financial Risk Management Services Importance',
       'Financial Risk Management Services Satisfaction']
    
    acq_cols = ['Contract Administration Services Importance',
       'Contract Administration Services Satisfaction','Contracting Function Partner Satisfaction',
       'Contracting Function Systems Satisfaction','Pre-Award Services Importance', 'Pre-Award Services Satisfaction',
       'Purchase Card Management Services Importance',
       'Purchase Card Management Services Satisfaction']

    it_cols =  ['IT Comms Importance', 'IT Comms Satisfaction',
       'IT Equipment Importance', 'IT Equipment Satisfaction', 'IT Function Strategic Partner',
       'IT Function System Satisfaction', 'IT Support Importance',
       'IT Support Satisfaction','DM&E Importance',
       'DM&E Satisfaction','O&M Importance',
       'O&M Satisfaction']

    hr_cols = ['Recruiting and Hiring Services Importance',
       'Recruiting and Hiring Services Satisfaction',
       'Reporting Responsibilities Awareness','Human Capital System','Human Capital Strategic Partner',
       'Time and Attendance Management Importance',
       'Time and Attendance Management Satisfaction','Worklife Support Services Importance',
       'Worklife Support Services Satisfaction','Workforce and Succession Planning Importance',
       'Workforce and Succession Planning Satisfaction','Training and Development Services Importance',
       'Training and Development Services Satisfaction','Retirement Planning and Processing Importance',
       'Retirement Planning and Processing Satisfaction','Employee Relations Services Importance',
       'Employee Relations Services Satisfaction','Labor Relations Services Importance',
       'Labor Relations Services Satisfaction']
    
    service_area_cols = ['Financial Management Satisfaction','Human Capital Satisfaction','IT Function Satisfaction',
                         'Contracting Function Quality of Support Satisfaction']
    
    drop_target_cols = [x for x in service_area_cols if x != target_col]
    one_hot = pd.get_dummies(df[cat_cols],drop_first=True)
    df_new = df.join(one_hot)
    df_new.drop(columns=cat_cols,inplace=True)
    df_new.drop(columns=drop_target_cols,inplace=True)
    
    # .replace() throws an error if there are no values to replace
    if True in df_new['YRSAGENCY2'].isin(['50+']):
        df_new.replace({'YRSAGENCY2':{'50+':50}},inplace=True)
    else:
        pass
    
    if target_col == 'Financial Management Satisfaction':
        df_new.drop(columns=acq_cols+it_cols+hr_cols,inplace=True)
        
    elif target_col == 'Human Capital Satisfaction':
        df_new.drop(columns=fm_cols+acq_cols+it_cols,inplace=True)
        
    elif target_col == 'IT Function Satisfaction':
        df_new.drop(columns=fm_cols+acq_cols+hr_cols,inplace=True)
        
    elif target_col == 'Contracting Function Quality of Support Satisfaction':
        df_new.drop(columns=fm_cols+it_cols+hr_cols,inplace=True)
        
    else:
        pass
    
    df_new['YRSAGENCY2']=df_new['YRSAGENCY2'].astype(float).astype(int)
    df_new['intercept']=1
    #df_new= df_new[df_new[target_col].isna()==False]
    df_new.dropna(axis=0,how='any',inplace=True)
    df_new=df_new.astype(int)
    
    return df_new     


def import_data():
    # the csv extract contains a mixture of wide and long formatted data.  This is inefficient for a number of reasons
    # so I need to create a wide format with the pivot_table method 

    df = pd.read_csv('css_2019.csv')
    table = pd.pivot_table(df,values='NUMBERRESPONSE',index=['surveyfeedbacksurveyfeedbacknumb',
                                                              'TELEWORK','AGENCY', 'GRADELEVEL', 'GRADE',
                                                              'YRSAGENCY2','OCCUPATION','SUP_STATUS',
                                                              'WORK_LOCATION','EDUCATION_LEVEL','AGE','LEAVING','FED_YRS'],
                          columns='QUESTION')
    df_wide = pd.DataFrame(table).reset_index()
    
    # drop columns of survey responses we will not be using
                    
    df_wide.drop(columns =[ 
       'Federal Government Vet Process Satisfaction',
       'Insider Threats and Workplace Violence Confidentiality',
        'Performance and Recognition Management Importance',
       'Performance and Recognition Management Satisfaction',
       'Security of Data Systems Trust', 'Supervisory Status',
        'OCCUPATION','surveyfeedbacksurveyfeedbacknumb','GRADE','WORK_LOCATION','FED_YRS'],inplace=True )
    
    return df_wide