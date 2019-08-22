import unittest
import munge_funcs
import pandas as pd

class TestClean(unittest.TestCase):
    
    
    def setUp(self):
        self.service_area_cols = ['Financial Management Satisfaction','Human Capital Satisfaction','IT Function Satisfaction',
                         'Contracting Function Quality of Support Satisfaction']
        
        self.cat_cols = ['GRADELEVEL','EDUCATION_LEVEL','LEAVING','AGENCY']
        
        self.test_df = pd.DataFrame({'TELEWORK':[30.0,20.0],
                                'AGENCY':['Department of Veterans Affairs','Department of Justice'],
                                'GRADELEVEL':['GS-12','GS-13'],
                                'YRSAGENCY2':['50+',22.0],
                                'SUP_STATUS':[1,1],
                                'EDUCATION_LEVEL':['Master\'s Degree','High School Diploma, GED, or Equivalent'],
                                'AGE':[25.0,65.0],
                                'LEAVING':['No','Yes - to take another job'],
                                'Contracting Function Quality of Support Satisfaction':[5.0,1.0],
                                'Financial Management Satisfaction':[2.0,4.0],
                                'Human Capital Satisfaction':[4.0,7.0],
                                'IT Function Satisfaction':[2.0,5.0]})
    
    #test whether a newly created dataframe has only one service area as a target column.  
    def test_target_names_fm(self):
        
        target = 'Financial Management Satisfaction'        
        
        result = munge_funcs.df_clean(self.test_df,self.cat_cols,target)
        expected_cols = ['TELEWORK', 'YRSAGENCY2', 'SUP_STATUS', 'AGE',
        target, 'GRADELEVEL_GS-12', 'GRADELEVEL_GS-13',
        'EDUCATION_LEVEL_High School Diploma, GED, or Equivalent',
        'EDUCATION_LEVEL_Master\'s Degree', 'LEAVING_No',
        'LEAVING_Yes - to take another job', 'AGENCY_Department of Justice',
        'AGENCY_Department of Veterans Affairs', 'intercept']
  
        self.assertEqual(list(result.columns).sort(),list(expected_cols).sort())
        
    def test_target_names_hr(self):
        
        target = 'Human Capital Satisfaction'
        
        result = munge_funcs.df_clean(self.test_df,self.cat_cols,target)
        expected_cols = ['TELEWORK', 'YRSAGENCY2', 'SUP_STATUS', 'AGE',
        target, 'GRADELEVEL_GS-12', 'GRADELEVEL_GS-13',
       'EDUCATION_LEVEL_High School Diploma, GED, or Equivalent',
       'EDUCATION_LEVEL_Master\'s Degree', 'LEAVING_No',
       'LEAVING_Yes - to take another job', 'AGENCY_Department of Justice',
       'AGENCY_Department of Veterans Affairs', 'intercept']
  
        self.assertEqual(list(result.columns).sort(),list(expected_cols).sort())
        
    def test_target_names_it(self):
        
        target = 'IT Function Satisfaction'
        
        result = munge_funcs.df_clean(self.test_df,self.cat_cols,target)
        expected_cols = ['TELEWORK', 'YRSAGENCY2', 'SUP_STATUS', 'AGE',
        target, 'GRADELEVEL_GS-12', 'GRADELEVEL_GS-13',
        'EDUCATION_LEVEL_High School Diploma, GED, or Equivalent',
        'EDUCATION_LEVEL_Master\'s Degree', 'LEAVING_No',
        'LEAVING_Yes - to take another job', 'AGENCY_Department of Justice',
        'AGENCY_Department of Veterans Affairs', 'intercept']
  
        self.assertEqual(list(result.columns).sort(),list(expected_cols).sort())
        
    def test_target_names_acq(self):
        
        target = 'Contracting Function Quality of Support Satisfaction'
        
        result = munge_funcs.df_clean(self.test_df,self.cat_cols,target)
        expected_cols = ['TELEWORK', 'YRSAGENCY2', 'SUP_STATUS', 'AGE',
        target, 'GRADELEVEL_GS-12', 'GRADELEVEL_GS-13',
        'EDUCATION_LEVEL_High School Diploma, GED, or Equivalent',
        'EDUCATION_LEVEL_Master\'s Degree', 'LEAVING_No',
        'LEAVING_Yes - to take another job', 'AGENCY_Department of Justice',
        'AGENCY_Department of Veterans Affairs', 'intercept']
  
        self.assertEqual(list(result.columns).sort(),list(expected_cols).sort())
    
    def tearDown(self):
        #self.service_area_cols.dispose()
        #self.cat_cols.dispose()
        #self.test_df.dispose()
        
        self.service_area_cols = None
        self.cat_cols = None
        self.test_df = None
    
if __name__ == '__main__':
    unittest.main()