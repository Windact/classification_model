import pathlib
import classification_model

from classification_model import config
import pandas as pd

from classification_model import __version__ as _version

from classification_model.processing import utils
import sklearn


# print("this is it : ",pathlib.Path(classification_model.__file__))

# print("this is it : ",pathlib.Path(classification_model.__file__).resolve())

# print("this is it : ",pathlib.Path(classification_model.__file__).resolve().parent)


# import os

# os.chdir(r"C:\Users\geoff\Desktop\udemy\badpipes\pumps\lol")

# for file in os.listdir(os.getcwd()):
#     if file != "f.txt":
#         os.remove(file)

# for file in classification_model.config.TRAINED_MODEL_DIR.iterdir():
#     print("the print")
#     print(file)
#     print(type(file))
#     print(help(file.unlink))
#     print(file.name)


# data = pd.read_csv(config.DATASET_DIR/"train.csv")
# # Check if the features expected are in the input_data
# features_missing = [feature for feature in config.VARIABLES_TO_KEEP if feature not in data.columns]
# print(features_missing)
# # if len(features_missing)> 0:
# #     return False

# # Check if the features are in their expected type
# input_data_num_var = [var for var in config.NUMERICAL_VARIABLES if data[var].dtype != "O"]
# print(input_data_num_var)
# # if len(input_data_num_var) != len(config.NUMERICAL_VARIABLES):
# #     return False
# print(isinstance(data["population"],(int,float)))
# cat_var = [var for var in config.VARIABLES_TO_KEEP if var not in config.NUMERICAL_VARIABLES]
# input_data_cat_var = [var for var in cat_var if isinstance(data[var],(object,bool))]

# print([file.name for file in classification_model.config.TRAINED_MODEL_DIR.iterdir()])
# print(f"{config.MODEL_PIPELINE_NAME}{_version}.pkl")

from classification_model.processing import preprocessors as pp




# for file in classification_model.config.TRAINED_MODEL_DIR.iterdir():
#     if file.name  not in ["blabla.pkl"] + ["__init__.py"]:
#         print(file.name)


# print(config.TRAINED_MODEL_DIR/"__init__.py" )


# p = utils.load_pipeline(f"{config.MODEL_PIPELINE_NAME}{_version}.pkl")
# print(type(p))

# print(isinstance(p,sklearn.pipeline.Pipeline))


# open('fileff.txt', 'w').close()

dataset_file_path = config.DATASET_DIR/config.TESTING_DATA_FILE
test_data = pd.read_csv(dataset_file_path)

single_row = test_data.iloc[0,:]

print(single_row)
print("********")

print(dir(single_row))
    