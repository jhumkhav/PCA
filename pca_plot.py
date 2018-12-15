import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("pca.csv", names = ["title", "fv", "fv", "fv", "fv", "fv", "fv", "fv", "fv", "fv", "fv", "cereal", "cereal", "cereal", "cereal", "cereal", "cereal", "cereal", "meat", "meat", "meat", "meat", "meat", "meat", "dairy", "dairy", "dairy", "dairy", "dairy", "dairy", "dairy", "dairy", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other"], delimiter = "\t")

features_fv = ["fv", "fv", "fv", "fv", "fv", "fv", "fv", "fv", "fv", "fv"]
features_cereal = ["cereal", "cereal", "cereal", "cereal", "cereal", "cereal", "cereal"]
features_meat = ["meat", "meat", "meat", "meat", "meat", "meat"]
features_dairy = ["dairy", "dairy", "dairy", "dairy", "dairy", "dairy", "dairy", "dairy"]
features_other = ["other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other", "other"]

#Separate features
x_fv = df.loc[:, features_fv].values
x_cereal = df.loc[:, features_cereal].values
x_meat = df.loc[:, features_meat].values
x_dairy = df.loc[:, features_dairy].values
x_other = df.loc[:, features_other].values

#Separate target
y = df.loc[:, ['title']].values

x_fv = StandardScaler().fit_transform(x_fv)
x_cereal = StandardScaler().fit_transform(x_cereal)
x_meat = StandardScaler().fit_transform(x_meat)
x_dairy = StandardScaler().fit_transform(x_dairy)
x_other = StandardScaler().fit_transform(x_other)

#Standardize features
x_fv  = StandardScaler().fit_transform(x_fv)
x_cereal = StandardScaler().fit_transform(x_cereal)
x_meat = StandardScaler().fit_transform(x_meat)
x_dairy = StandardScaler().fit_transform(x_dairy)
x_other = StandardScaler().fit_transform(x_other)

#Run PCA
pca = PCA(n_components = 2)
principalComponents_fv = pca.fit_transform(x_fv)
principalComponents_cereal = pca.fit_transform(x_cereal)
principalComponents_meat = pca.fit_transform(x_meat)
principalComponents_dairy = pca.fit_transform(x_dairy)
principalComponents_other = pca.fit_transform(x_other)

#Set new dataframe
principalDf_fv = pd.DataFrame(data = principalComponents_fv, columns = ['pca1', 'pca2'])
principalDf_cereal = pd.DataFrame(data = principalComponents_cereal, columns = ['pca1', 'pca2'])
principalDf_meat = pd.DataFrame(data = principalComponents_meat, columns = ['pca1', 'pca2'])
principalDf_dairy = pd.DataFrame(data = principalComponents_dairy, columns = ['pca1', 'pca2'])
principalDf_other = pd.DataFrame(data = principalComponents_other, columns = ['pca1', 'pca2'])

#Create final dataset
finalDf_fv = pd.concat([principalDf_fv, df[['title']]], axis = 1)
finalDf_cereal = pd.concat([principalDf_cereal, df[['title']]], axis = 1)
finalDf_meat = pd.concat([principalDf_meat, df[['title']]], axis = 1)
finalDf_dairy = pd.concat([principalDf_dairy, df[['title']]], axis = 1)
finalDf_other = pd.concat([principalDf_other, df[['title']]], axis = 1)

#Make Plot
df = pd.read_csv("pcadata.csv", names=['pca1','pca2','Group'])
graph = sns.scatterplot(x="pca1", y="pca2", data=df, hue="Group")
plt.show()
