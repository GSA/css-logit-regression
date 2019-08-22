import munge_funcs
import pandas as pd

def main():
    
    cat_cols = ['GRADELEVEL','EDUCATION_LEVEL','LEAVING','AGENCY']
    data = munge_funcs.import_data()
    data_fm = munge_funcs.df_clean(data,cat_cols,'Financial Management Satisfaction')
    data_hr = munge_funcs.df_clean(data,cat_cols,'Human Capital Satisfaction')
    data_it = munge_funcs.df_clean(data,cat_cols,'IT Function Satisfaction')
    data_acq = munge_funcs.df_clean(data,cat_cols,'Contracting Function Quality of Support Satisfaction')
    
    #return data_fm,data_hr,data_it,data_acq
    data_fm.to_csv('fm_data.csv')
    data_hr.to_csv('hr_data.csv')
    data_it.to_csv('it_data.csv')
    data_acq.to_csv('acq_data.csv')
    
if __name__ == '__main__':
    main()