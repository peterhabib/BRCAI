#%%
from tensorflow.keras.models import load_model
import pandas as pd

from Scripts import common
import sys
from Scripts import common_scaler
from Scripts import common_categorical
from Scripts import common_pre_post_processing
import matplotlib.pyplot as plt
import pandas as pd




Data = sys.argv[1]
Savefile = sys.argv[2]
# Load the previous state of the model
model = load_model(common.model_file_name)

WriteTo = open(Savefile, 'w')
WriteTo.write('Clump_Thickness	Uniformity_of_Cell_Size	Uniformity_of_Cell_Shape	Marginal_Adhesion	Single_Epithelial_Cell_Size	Bare_Nuclei	Bland_Chromatin	Normal_Nucleoli	Mitoses	ID\tPrediction\n')

# Load the previous state of the enconders and scalers
label_encoder, onehot_encoder = common_categorical.load_categorical_feature_encoder()
x_scaler, y_scaler = common_scaler.load_scaler()


Data = pd.read_csv(Data, na_values='?', index_col=False)
Data = Data.dropna()
# print(Data)

#Clump_Thickness	Uniformity_of_Cell_Size	Uniformity_of_Cell_Shape	Marginal_Adhesion	Single_Epithelial_Cell_Size
# Bare_Nuclei	Bland_Chromatin	Normal_Nucleoli	Mitoses	ID

Row_list = []

for index, rows in Data.iterrows():
    # Create list for the current row
    my_list = [rows.Clump_Thickness, rows.Uniformity_of_Cell_Size,
               rows.Uniformity_of_Cell_Shape, rows.Marginal_Adhesion,
               rows.Single_Epithelial_Cell_Size, rows.Bare_Nuclei,
               rows.Bland_Chromatin, rows.Normal_Nucleoli,
               rows.Mitoses, rows.ID]

    # append the list to the final list
    Row_list.append(my_list)


values = Row_list



# Transform inputs to the format that the model expects
model_inputs, _ = common_pre_post_processing.transform_inputs(values, label_encoder, onehot_encoder, x_scaler)

# Use the model to predict the price for a house
y_predicted = model.predict(model_inputs)

# Transform the results into a user friendly representation
y_predicted_unscaled = common_pre_post_processing.transform_outputs(y_predicted, y_scaler)

print('Results when:')
print('Scale Input Features = ', common.scale_features_input)
print('Scale Output Features = ', common.scale_features_output)
print('Use Categorical Feature Eencoder  = ', common.use_categorical_feature_encoder)

for i in range(0, len(values)):
    result = int(y_predicted_unscaled[i])
    if result == 1: result = 'Benign'
    elif result == 2: result = 'Malignant'
    else: result = 'Unknown'
    Output = "\t".join(str(x) for x in values[i])+'\t'+ result+'\n'
    print(Output)

    WriteTo.write(Output)

print('Predicted Results Saved To: %s'%Savefile)



