import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Streamlit Example")

st.write("""
# Explore different Classifiers
which one of the best?  
""")

# dataset_name = st.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine") )
# st.write(dataset_name)


dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine") )
# st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))
# st.write(classifier_name)

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target

    return X, y

X, y = get_dataset(dataset_name) 
st.write("Shape of the selected dataset {} is {}".format(dataset_name, X.shape)) 
st.write("Number of Classes ", len(np.unique(y)))  


def add_parameter_ui(clf_name):
    params = dict()

    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C

        kernel = st.sidebar.selectbox("Select Kernel", ("rbf", "linear", "poly"))
        if kernel == "poly":
            degree = st.sidebar.slider("degree", 1, 3)
            params["kernel"] = kernel
            params["degree"] = degree
        else:
            params["kernel"] = kernel
            params["degree"] = 0
    else :
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)
# st.write("Selected Clasifier is {} and the hyper-parameters are {}". format(classifier_name, params)) 

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "SVM":
        if params['degree'] == 0:
            clf = SVC(C = params["C"], kernel=params["kernel"], degree=params["degree"])    

        clf = SVC(C = params["C"], kernel=params["kernel"])

    else :
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=99)

    return clf

clf = get_classifier(classifier_name, params)    

# Train the Model
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=99)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)

acc = accuracy_score(y_valid, y_pred)
st.write(f"For Classifier {classifier_name}")
st.write(f"The Accuracy is :  {acc}")


# Plot the data.
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar()

# plt.show() # with streamlit we have to use st.pyplot() and not plt.show()
st.pyplot(fig)