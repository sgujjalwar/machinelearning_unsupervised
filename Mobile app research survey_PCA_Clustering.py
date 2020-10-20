# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:15:46 2019

@author: Sushrutha Gujjalwar
"""

#######################
#CODE FOR DATA ANALYSIS
######################

# Importing new libraries
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis
from sklearn.cluster import KMeans # k-means clustering


# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

customers_df = pd.read_excel('finalExam_Mobile_App_Survey_Data_final_exam-2.xlsx')

customers_df.shape

#column names
customers_df.columns = ['caseID',
                        '1_age',
                        '2_p_iPhone',
                        '2_p_iPodtouch',
                        '2_p_Android',
                        '2_p_Blackberry',
                        '2_p_Nokia',
                        '2_p_Windows',
                        '2_d_HP/palm',
                        '2_d_tab',
                        '2_p_other',
                        '2_pd_none',
                        '4_ap_music',
                        '4_ap_tvcheck',
                        '4_ap_entertainment',
                        '4_ap_tvshow',
                        '4_ap_gaming',
                        '4_ap_social',
                        '4_ap_generalnew',
                        '4_ap_shopping',
                        '4_ap_newspub',
                        '4_ap_other',
                        '4_ap_none',
                        '11_ap_number',
                        '12_ap_free',
                        '13_visit_FB',
                        '13_visit_Twit',
                        '13_visit_myspace',
                        '13_visit_pandrad',
                        '13_visit_Vevo',
                        '13_visit_Youtube',
                        '13_visit_AOLrad',
                        '13_visit_lastfm',
                        '13_visit_Yahoo_m',
                        '13_visit_IMDB',
                        '13_visit_LinkedIn',
                        '13_visit_Netflix',
                        '24_tech_update',
                        '24_tech_advice',
                        '24_tech_purchase',
                        '24_tech_every_life',
                        '24_tech_control_life',
                        '24_tech_save_time',
                        '24_tech_music_imp',
                        '24_int_tv shows',
                        '24_int_more info',
                        '24_int_sneak_social',
                        '24_int_touch_family',
                        '24_int_easy_family',
                        '25_me_opinion_leader',
                        '25_me_standout',
                        '25_me_office_advisor',
                        '25_me_decision_lead',
                        '25_me_try_new',
                        '25_me_guide',
                        '25_me_incontrol',
                        '25_me_risktaker',
                        '25_me_creative',
                        '25_me_optimistic',
                        '25_me_active',
                        '25_me_stretched',
                        '26_me_luxurybrand',
                        '26_me_bargain',
                        '26_me_shop_fun',
                        '26_me_package_deal',
                        '26_me_shop_online',
                        '26_me_designerbrand',
                        '26_me_shop_noapps',
                        '26_me_shop_appcool',
                        '26_me_newapp_showoff',
                        '26_child_impact_app',
                        '26_me_extra$_apps',
                        '26_me_nospend_yearn',
                        '26_me_influence',
                        '26_me_style_reflect',
                        '26_me_morepurchase',
                        '26_me_tech_alwaysphone',
                        '48_education',
                        '49_maritalstatus',
                        '50_nochild',
                        '50_cage<6',
                        '50_cage6-12',
                        '50_cage13-17',
                        '50_cage>18',
                        '54_race',
                        '55_ethnicity',
                        '56_incomelevel',
                        '57_gender']
                  
                  
                  
                  
            

########################
# Step 2: Scale to get equal variance
########################

customer_features_reduced = customers_df.iloc[ : , 2:-11 ] #Remove demographic information

scaler = StandardScaler()

scaler.fit(customer_features_reduced)

X_scaled_reduced = scaler.transform(customer_features_reduced)


########################
# Step 3: Run PCA without limiting the number of components
########################
customer_pca_reduced = PCA(n_components = None,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)


X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)

########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(customer_pca_reduced.n_components_)

plt.plot(features,
         customer_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')

plt.title('Mobile App Research Survey')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()


########################
# Step 5: Run PCA again based on the desired number of components
########################

customer_pca_reduced = PCA(n_components = 5,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)


########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_loadings_df = pd.DataFrame(pd.np.transpose(customer_pca_reduced.components_))

factor_loadings_df = factor_loadings_df.set_index(customers_df.columns[2:-11])

factor_loadings_df.to_excel('SG_FinalExam.xlsx')

########################
# Step 7: Analyze factor strengths per customer
########################

X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)

X_pca_df = pd.DataFrame(X_pca_reduced)


########################
# Step 8: Renaming my principal components and reattach demographic information
########################

X_pca_df.columns = ['13_visit_FB',
                    '24_tech_purchase',
                    '2_p_iPhone',
                    '25_me_try_new',
                    '25_me_decision_lead']

final_pca_df = pd.concat([customers_df.loc[ : , ['1_age',
                                                 '48_education',
                                                 '49_maritalstatus',
                                                 '50_nochild',
                                                 '50_cage<6',
                                                 '50_cage6-12',
                                                 '50_cage13-17',
                                                 '50_cage>18',
                                                 '54_race',
                                                 '55_ethnicity',
                                                 '56_incomelevel',
                                                 '57_gender']],
                                                  X_pca_df], axis = 1)


###############################################################################
# Combining PCA and Clustering!!!
###############################################################################

########################
# Step 1: Taking your transformed dataframe
########################

print(X_pca_df.head(n = 5))

print(pd.np.var(X_pca_df))


########################
# Step 2: Scaling to get equal variance
########################

scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns



########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k_pca = KMeans(n_clusters = 5,
                         random_state = 508)


customers_k_pca.fit(X_pca_clust_df)


customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})

customers_k = KMeans(n_clusters = 5,
                      random_state = 508)

customers_k.fit(X_scaled_reduced)

customers_kmeans_clusters = pd.DataFrame({'cluster': customers_k.labels_})


print(customers_kmeans_pca.iloc[: , 0].value_counts())

########################
# Step 4: Analyze cluster centers
########################

centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['13_visit_FB',
                            '24_tech_purchase',
                            '2_p_iPhone',
                            '25_me_try_new',
                            '25_me_decision_lead']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('customers_pca_kmeans_centriods.xlsx')



########################
# Step 5: Analyze cluster memberships
########################
clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)

X_scaled_reduced_df = pd.DataFrame(X_scaled_reduced)


X_scaled_reduced_df.columns = customer_features_reduced.columns

########################
# Step 6: Reattach demographic information
########################


clusters_df = pd.concat([customers_kmeans_clusters,
                         X_scaled_reduced_df],
                         axis = 1)


final_pca_clust_df = pd.concat([customers_df.loc[ : , ['1_age',
                                                       '48_education',
                                                       '49_maritalstatus',
                                                       '50_nochild',
                                                       '50_cage<6',
                                                       '50_cage6-12',
                                                       '50_cage13-17',
                                                       '50_cage>18',
                                                       '54_race',
                                                       '55_ethnicity',
                                                       '56_incomelevel',
                                                       '57_gender']],
                                                        clst_pca_df], 
                                                        axis = 1)


print(final_pca_clust_df.head(n = 5))

final_clusters_df = pd.concat([customers_df.loc[ : , ['1_age',
                                                      '48_education',
                                                      '49_maritalstatus',
                                                      '50_nochild',
                                                      '50_cage<6',
                                                      '50_cage6-12',
                                                      '50_cage13-17',
                                                      '50_cage>18',
                                                      '54_race',
                                                      '55_ethnicity',
                                                      '56_incomelevel',
                                                      '57_gender']] ,
                                                       clusters_df],
                                                       axis = 1)
print(final_clusters_df.head(n = 5))


###########
#MODEL CODE
###########


# Renaming age- 

age = {1 : 'under 18',
       2 : '18-24',
       3 : '25-29',
       4 : '30-34',
       5 : '35-39',
       6 : '40-44',
       7 : '45-49',
       8 : '50-54',
       9 : '55-59',
       10 : '60-64',
       12 : '65 or over'}

final_clusters_df['1_age'].replace(age, inplace = True)

# Renaming education
education = {1 : 'Some high school',
             2 : 'High school graduate',
             3 : 'Some college',
             4 : 'College graduate',
             5 : 'Some post-graduate studies',
             6 : 'Post graduate degree'}

final_clusters_df['48_education'].replace(education, inplace = True)

# Renaming marital_status
marital_status = {1 : 'Married',
                  2 : 'Single',
                  3 : 'Single with a partner',
                  4 : 'Separated/Widowed/Divorced'}

final_clusters_df['49_maritalstatus'].replace(marital_status, inplace = True)

# Renaming race
race = {1 : 'White or Caucasian',
                2 : 'Black or African American',
                3 : 'Asian',
                4 : 'Native Hawaiian or Other Pacific Islander',
                5 : 'American Indian or Alaska Native',
                6 : 'Other race'}

final_clusters_df['54_race'].replace(race, inplace = True)

# Renaming ethnicity
ethnicity = {1 : 'Yes',
             2 : 'No'}

final_clusters_df['55_ethnicity'].replace(ethnicity, inplace = True)

# Renaming annual_income
annual_income = {1 : 'Under $10,000',
                 2 : '$10,000-$14,999',
                 3 : '$15,000-$19,999',
                 4 : '$20,000-$29,999',
                 5 : '$30,000-$39,999',
                 6 : '$40,000-$49,999',
                 7 : '$50,000-$59,999',
                 8 : '$60,000-$69,999',
                 9 : '$70,000-$79,999',
                 10 : '$80,000-$89,999',
                 11 : '$90,000-$99,999',
                 12 : '$100,000-$124,999',
                 13 : '$125,000-$149,999',
                 14 : '$150,000 and over'}

final_clusters_df['56_incomelevel'].replace(annual_income, inplace = True)


# Renaming gender
gender = {1 : 'Male',
          2 : 'Female'}

final_clusters_df['57_gender'].replace(gender, inplace = True)

# Adding a productivity step
data_df = final_clusters_df

####################
#Vizualizations
####################

# Age

# 25_me_try_new
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '1_age',
            y = '25_me_try_new',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

# 13_visit_FB

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '1_age',
            y = '13_visit_FB',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()


# 24_tech_purchase.
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '1_age',
            y = '24_tech_purchase',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

# 2_p_iPhone
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '1_age',
            y = '2_p_iPhone',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

# 25_me_decision_lead
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '1_age',
            y = '25_me_decision_lead',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()


#income level

#25_me_try_new

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '56_incomelevel',
            y = '25_me_try_new',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()


#13_visit_FB

fig, ax = plt.subplots(figsize = (20, 7))
sns.boxplot(x = '56_incomelevel',
            y = '13_visit_FB',
            hue = 'cluster',
            data = data_df)

plt.ylim(-1, 6)
plt.tight_layout()
plt.show()


#24_tech_purchase
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '56_incomelevel',
            y = '24_tech_purchase',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

# 2_p_iPhone
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '56_incomelevel',
            y = '2_p_iPhone',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

# 25_me_decision_lead
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '56_incomelevel',
            y = '25_me_decision_lead',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

#education

#25_me_try_new

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '48_education',
            y = '25_me_try_new',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

#13_visit_FB
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '48_education',
            y = '13_visit_FB',
            hue = 'cluster',
            data = data_df)

plt.ylim(-1, 6)
plt.tight_layout()
plt.show()


#24_tech_purchase
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '48_education',
            y = '24_tech_purchase',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

#2_p_iPhone
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '48_education',
            y = '2_p_iPhone',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

# 25_me_decision_lead
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '48_education',
            y = '25_me_decision_lead',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()




#Race

#25_me_try_new

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '54_race',
            y = '25_me_try_new',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

#13_visit_FB
fig, ax = plt.subplots(figsize = (20, 8))
sns.boxplot(x = '54_race',
            y = '13_visit_FB',
            hue = 'cluster',
            data = data_df)

plt.ylim(-1, 6)
plt.tight_layout()
plt.show()


#24_tech_purchase
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '54_race',
            y = '24_tech_purchase',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

# 2_p_iPhone
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '54_race',
            y = '2_p_iPhone',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

# 25_me_decision_lead

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '54_race',
            y = '25_me_decision_lead',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()


#ethnicity

#25_me_try_new
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '55_ethnicity',
            y = '25_me_try_new',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

#13_visit_FB
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '55_ethnicity',
            y = '13_visit_FB',
            hue = 'cluster',
            data = data_df)

plt.ylim(-1, 6)
plt.tight_layout()
plt.show()


#24_tech_purchase
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '55_ethnicity',
            y = '24_tech_purchase',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

# 2_p_iPhone
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '55_ethnicity',
            y = '2_p_iPhone',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

# 25_me_decision_lead

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = '55_ethnicity',
            y = '25_me_decision_lead',####write your group col.name
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

# Sending data to Excel
data_df.to_excel('analysis_data.xlsx')

