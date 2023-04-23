# Import libraries
import pandas
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.preprocessing import StandardScaler
from sklearn2pmml import sklearn2pmml

# Define columns name
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']

# Import dataset
iris_df = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names = col_names)

# Define features and target
iris_X = iris_df[iris_df.columns.difference(["Species"])]
iris_y = iris_df["Species"]

# Define the pipeline
pipeline = PMMLPipeline([
	("scaler", StandardScaler()),
	("pca", PCA(n_components = 3)),
	("classifier", LogisticRegression(multi_class = "ovr"))
])

# Fit the model
pipeline.fit(iris_X, iris_y)
pipeline.verify(iris_X.sample(n = 15))

# Save the pipeline into a pmml file
sklearn2pmml(pipeline, "LogisticRegressionIris.pmml", with_repr = True)
