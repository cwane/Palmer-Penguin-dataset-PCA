
#Importing all necessary Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

#Loading penguin dataset
penguins = sns.load_dataset('penguins')

#print("Shape of DataFrame:", df.shape)


#shape of dataset
df = pd.DataFrame(data=penguins)
print("Shape of DataFrame:", df.shape)

df.head()

df.head(344)

df.info()


print(df)


df['target'] = penguins['species']

#mapping species as target 
df['target'] = pd.Series(penguins['species'], dtype="category").cat.codes
df.head(344)


# Filter the DataFrame to include only numeric columns
numeric_columns = penguins.select_dtypes(include=[np.number])

# Drop rows with missing values
numeric_columns = numeric_columns.dropna()
print(numeric_columns)

# Extract the feature values
X = numeric_columns.values
X = X[:,0:-1]
print(X)

print(np.mean(X))
print(np.std(X))

print(X)

print(np.mean(X))
print(np.std(X))                                                 

# Calculate the mean and standard deviation
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)

print("Mean of X:", mean_X)
print("Standard deviation of X:", std_X)


mean_X = np.mean(X)
std_X = np.std(X)

print("Mean of X:", mean_X)
print("Standard deviation of X:", std_X)

#standardize the features in dataset
X_std = StandardScaler().fit_transform(X)

print(np.mean(X_std))
print(np.std(X_std))

#shape of dataset after dropping null values
print(X_std.shape)


#Finding covariance matrix
cov_mat = np.cov(np.transpose(X_std))
print(cov_mat)
cov_mat.shape


#calculation of eigen values and eigen vectors

eigen_value,eigen_vector = np.linalg.eig(cov_mat)
print(eigen_value)
print(eigen_vector)
eigen_value.shape

eigen_vector.shape

#calculation of proportion of each eigen values
prop1 = eigen_value[0] / np.sum(eigen_value)
print(prop1*100)

prop2 = eigen_value[1] / np.sum(eigen_value)
print(prop2*100)

prop3 = eigen_value[2] / np.sum(eigen_value)
print(prop3*100)

prop4 = eigen_value[3] / np.sum(eigen_value)
print(prop4*100)

#Scree plot of proportions
eigenvalues = [eigen_value[0], eigen_value[1], eigen_value[2], eigen_value[3]]

# Calculate the total sum of eigenvalues (total variance)
total_variance = np.sum(eigenvalues)

# Calculate the proportion of variance explained by each principal component
explained_variances = [(eig_val / total_variance) * 100 for eig_val in eigenvalues]

# Plot the scree plot
plt.figure(figsize=(6, 6))
bars = plt.bar(range(1, len(explained_variances) + 1), explained_variances, color='b', alpha=0.7)
plt.xticks(range(1, len(explained_variances) + 1))
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.title('Scree Plot - Variance Explained by Principal Components')

# Add percentage labels on top of each bar
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{explained_variances[i]:.1f}%', 
             ha='center', va='bottom')
    roll_numbers = ['Roll[35,42]']
roll_number_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8)
plt.legend(handles=[roll_number_legend], labels=roll_numbers, loc='upper right')

plt.show()

#selection of principle components

best_eigen_vector = np.transpose( eigen_vector[:,0:1])
selected_eigenvectors = np.transpose(eigen_vector[:, :2])
mat_trans = np.transpose(X_std)

new_data = np.dot(selected_eigenvectors,mat_trans)
new_data.shape



#shape of selected eigen vectors
selected_eigenvectors.shape

#Plot for PCA of palmer penguin dataset by taking two principle components
# Define the species-color mapping
species_color_dict = {
    'Adelie': 'blue',
    'Chinstrap': 'orange',
    'Gentoo': 'green'
}

new_dataframe = pd.DataFrame(data=new_data.T, columns=['PC1', 'PC2'])
# Map the species to colors in the dataframe
new_dataframe['color'] = penguins['species'].map(species_color_dict)

# Scatter plot with colors based on species
scatter = plt.scatter(new_dataframe['PC1'], new_dataframe['PC2'], c=new_dataframe['color'])
plt.title('Penguins Dataset Scatter Plot')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Create a legend for the species-color mapping
species_legend = plt.legend(handles=legend_labels, labels=species_color_dict.keys(), loc='lower left')

# Add roll number legend
roll_numbers = ['Roll[35,42]']
roll_number_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8)
plt.legend(handles=[roll_number_legend], labels=roll_numbers, loc='upper right')

# Add the species legend back to the plot
plt.gca().add_artist(species_legend)

plt.show()


#Plot for PCA of palmer penguin dataset using Scikit by taking two principle components
# Fit PCA on the data
pca = PCA(n_components=2)
new_data = pca.fit_transform(X)

# Define the species-color mapping
species_color_dict = {
    'Adelie': 'blue',
    'Chinstrap': 'orange',
    'Gentoo': 'green'
}

# Create a DataFrame with new_data and species labels
new_dataframe = pd.DataFrame(data=new_data, columns=['PC1', 'PC2'])
new_dataframe['species'] = penguins['species']  # Assuming penguins DataFrame has a 'species' column

# Filter the rows of the DataFrame based on the relevant species used for PCA
relevant_species = ['Adelie', 'Chinstrap', 'Gentoo']
new_dataframe = new_dataframe[new_dataframe['species'].isin(relevant_species)]

# Map the species to colors in the dataframe
new_dataframe['color'] = new_dataframe['species'].map(species_color_dict)

# Scatter plot with colors based on species
scatter = plt.scatter(new_dataframe['PC1'], new_dataframe['PC2'], c=new_dataframe['color'])
plt.title('PCA on Penguins Dataset using Scikit')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Create a list of legend handles for species-color mapping
legend_labels = []
for species, color in species_color_dict.items():
    legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))

# Create a legend for the species-color mapping
species_legend = plt.legend(handles=legend_labels, labels=species_color_dict.keys(), loc='upper left')

# Add roll number legend
roll_numbers = ['Roll[35,42]']
roll_number_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8)
plt.legend(handles=[roll_number_legend], labels=roll_numbers, loc='upper right')

# Add the species legend back to the plot
plt.gca().add_artist(species_legend)

plt.show()


#PCA plot taking three principle components
best_eigen_vector = np.transpose( eigen_vector[:,0:1])
selected_eigenvectors = np.transpose(eigen_vector[:, :3])
mat_trans = np.transpose(X_std)

new_data = np.dot(selected_eigenvectors,mat_trans)
new_data.shape




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the species-color mapping
species_color_dict = {
    'Adelie': 'blue',
    'Chinstrap': 'orange',
    'Gentoo': 'green'
}

new_dataframe = pd.DataFrame(data=new_data.T, columns=['PC1', 'PC2', 'PC3'])
# Map the species to colors in the dataframe
new_dataframe['color'] = penguins['species'].map(species_color_dict)

# Create a 3D plot
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

# Scatter plot with colors based on species
ax.scatter(new_dataframe['PC1'], new_dataframe['PC2'], new_dataframe['PC3'], c=new_dataframe['color'])

ax.set_title('Penguins dataset Scatter Plot (3D)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3', rotation=90)
ax.zaxis.labelpad = -4.0

# Create a legend for the species-color mapping
legend_labels = []
for species, color in species_color_dict.items():
    legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
species_legend = ax.legend(legend_labels, species_color_dict.keys(), loc='upper left')

# Add a separate legend for roll numbers
roll_numbers = ['Rollno[35, 42]']  # Replace with the actual roll numbers
roll_number_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8)
roll_number_legend = ax.legend([roll_number_legend], roll_numbers, loc='upper right')

# Add both legends to the plot
ax.add_artist(species_legend)
ax.add_artist(roll_number_legend)

plt.show()



##PCA plot using Scikit taking three principle components
# Fit PCA on the data
pca = PCA(n_components=3)
new_data = pca.fit_transform(X)

# Define the species-color mapping
species_color_dict = {
    'Adelie': 'blue',
    'Chinstrap': 'orange',
    'Gentoo': 'green'
}

# Create a DataFrame with new_data and species labels
new_dataframe = pd.DataFrame(data=new_data, columns=['PC1', 'PC2', 'PC3'])
new_dataframe['species'] = penguins['species']  # Assuming penguins DataFrame has a 'species' column

# Filter the rows of the DataFrame based on the relevant species used for PCA
relevant_species = ['Adelie', 'Chinstrap', 'Gentoo']
new_dataframe = new_dataframe[new_dataframe['species'].isin(relevant_species)]

# Map the species to colors in the dataframe
new_dataframe['color'] = new_dataframe['species'].map(species_color_dict)

# Scatter plot with colors based on species in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(new_dataframe['PC1'], new_dataframe['PC2'], new_dataframe['PC3'], c=new_dataframe['color'])
ax.set_title('Penguins dataset Scatter Plot using Scikit (3D)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3', rotation=90)
ax.zaxis.labelpad = -4.0


# Create a legend for the species-color mapping
legend_labels = []
for species, color in species_color_dict.items():
    legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
species_legend = ax.legend(legend_labels, species_color_dict.keys(), loc='upper left')

# Add a separate legend for roll numbers
roll_numbers = ['Rollno[35, 42]']  # Replace with the actual roll numbers
roll_number_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8)
roll_number_legend = ax.legend([roll_number_legend], roll_numbers, loc='upper right')

# Add both legends to the plot
ax.add_artist(species_legend)
ax.add_artist(roll_number_legend)

plt.show()







