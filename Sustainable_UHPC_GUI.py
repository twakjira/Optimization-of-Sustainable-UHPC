#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import PySimpleGUI as sg
from PIL import Image
from PIL import ImageOps

df = pd.read_excel('optimized_UHPC_all_objectives.xlsx')

dff1 = df[['C', 'SF', 'FA', 'GBFS', 'LP', 'W', 'SP', 'QP', 'MSA',
       'A', 'fc (MPa)', 'Climate Change', 'Fossil Depletion',
       'Freshwater Ecotoxicity', 'Freshwater Eutrophication', 'Human Toxicity',
       'Ionising Radiation', 'Marine Ecotoxicity', 'Marine Eutrophication',
       'Metal Depletion', 'Natural Land Transformation', 'Ozone Depletion',
       'Particulate Matter Formation', 'Photochemical Oxidant Formation',
       'Terrestrial Acidification', 'Terrestrial Ecotoxicity',
       'Urban Land Occupation', 'Water Depletion']]

# only objective functions
dff2 = dff1.copy(deep=True)


def get_option_1_values(fc_value):
    df.columns = ['C', 'SF', 'FA', 'GBFS', 'LP', 'W', 'SP', 'QP', 'MSA', 'A', 'fc (MPa)'] + list(df.columns[11:])
    closest_row = df.iloc[(df['fc (MPa)']-fc_value).abs().argsort()[0]]
    default_values = closest_row.round(3)
    return default_values


def get_option_2_values(weights):
    # ... (the rest of the option_2 function)
    # Normalize the decision matrix
    normalized_df = dff2 / np.sqrt((dff2 ** 2).sum())

    # Weight the normalized decision matrix
    weighted_df = normalized_df * weights

    # Determine the positive and negative ideal solutions
    positive_ideal_solution = np.zeros(len(dff2.columns))
    negative_ideal_solution = np.zeros(len(dff2.columns))

    # Since the first objective is a benefit objective and others are cost objectives
    positive_ideal_solution[0] = weighted_df.iloc[:, 0].max()
    negative_ideal_solution[0] = weighted_df.iloc[:, 0].min()

    for i in range(1, len(dff2.columns)):
        positive_ideal_solution[i] = weighted_df.iloc[:, i].min()
        negative_ideal_solution[i] = weighted_df.iloc[:, i].max()

    # Calculate the separation measures for each alternative
    positive_separation = np.sqrt(((weighted_df - positive_ideal_solution) ** 2).sum(axis=1))
    negative_separation = np.sqrt(((weighted_df - negative_ideal_solution) ** 2).sum(axis=1))

    # Calculate the relative closeness to the ideal solution
    closeness = negative_separation / (positive_separation + negative_separation)

    # Check if the sum of the weights input by the user is 1
    if np.abs(np.sum(weights) - 1) > 0.001:
        raise ValueError("The sum of the weights should be equal to 1.")

    # Get the best option (row with highest rank)
    best_option = dff1.iloc[closeness.argmax()]

    return pd.DataFrame(best_option).transpose().values
output_layout = [    [sg.Text(f"Output:")],
]

output_columns = ['C', 'SF', 'FA', 'GBFS', 'LP', 'W', 'SP', 'QP', 'MSA', 'A', 'fc (MPa)', 'Climate Change', 'Fossil Depletion', 'Freshwater Ecotoxicity', 'Freshwater Eutrophication', 'Human Toxicity', 'Ionising Radiation', 'Marine Ecotoxicity', 'Marine Eutrophication', 'Metal Depletion', 'Natural Land Transformation', 'Ozone Depletion', 'Particulate Matter Formation', 'Photochemical Oxidant Formation', 'Terrestrial Acidification', 'Terrestrial Ecotoxicity', 'Urban Land Occupation', 'Water Depletion']

column_1 = output_columns[:10]
column_2 = output_columns[10:20]

output_layout_column_1 = [[sg.Text(column, size=(10, 1)), sg.Input(default_text="", key=f"-{column}-", size=(10, 1), disabled=True, border_width=1, justification='center')] for column in column_1]
output_layout_column_2 = [[sg.Text(column, size=(20, 1)), sg.Input(default_text="", key=f"-{column}-", size=(10, 1), disabled=True, border_width=1, justification='center')] for column in column_2]

column_3 = output_columns[20:]  # select columns from "Photochemical Oxidant Formation" onwards
output_layout_column_3 = [[sg.Text(column, size=(25, 1)), sg.Input(default_text="", key=f"-{column}-", size=(10, 1), disabled=True, border_width=1, justification='center')] for column in column_3]

# output_layout.append([sg.Column(output_layout_column_1), sg.Column(output_layout_column_2), sg.Column(output_layout_column_3)])

# Add titles for the three images

# Open the images
img1 = Image.open('ALAMS.png')

# Get the minimum width and height among the images
widths = [img1.width]
heights = [img1.height]
min_width = min(widths)
min_height = min(heights)

# Resize the images to the minimum size
img1 = ImageOps.fit(img1, (min_width, min_height))

# Define the scale factor
scale_factor = 0.65

# Resize the images
img1 = img1.resize((int(min_width * scale_factor), int(min_height * scale_factor)))

# Save the resized images
img1.save('ALAMS1.png')


# Open the images
img1 = Image.open('fig11.png')

# Get the minimum width and height among the images
widths = [img1.width]
heights = [img1.height]
min_width = min(widths)
min_height = min(heights)

# Resize the images to the minimum size
img1 = ImageOps.fit(img1, (min_width, min_height))

# Define the scale factor
scale_factor = 0.7

# Resize the images
img1 = img1.resize((int(min_width * scale_factor), int(min_height * scale_factor)))

# Save the resized images
img1.save('fig.png')


output_layout.append([sg.Column(output_layout_column_1), sg.Column(output_layout_column_2), 
                      sg.Column(output_layout_column_3), 
                      ])

# output_layout.append([sg.Column(output_layout_column_1), sg.Column(output_layout_column_2)])
# Add the image element at the bottom
output_layout.append([sg.Image(filename='fig.png')])



layout = [    [sg.Text("Choose an option:", font=("Helvetica", 12))],
    [sg.Text("Use Option 1 to determine the optimized mix based on desired fc value",             
             text_color='black', font=("Helvetica", 12))],
    [sg.Text("Use Option 2 to assign weight to each objective function and determine optimimum mix using TOPSIS", 
             text_color='black', font=("Helvetica", 12))],     
    [sg.Button("Option 1", button_color=('white', 'gray')), sg.Button("Option 2", button_color=('white', 'gray'))],


    [sg.Text("Enter desired fc value (MPa) for Option 1:", visible=False, font=("Helvetica", 14)), sg.Input(key="-FC_VALUE-", size=(10, 1), visible=False)],
    *output_layout
]



window = sg.Window("Optimization of Sustainable UHPC", layout)

output_values = None

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break

    if event == "Option 1":
        window['-FC_VALUE-'].update(visible=False)
        fc_value_input = sg.popup_get_text("Please enter the desired fc value (MPa)")
        try:
            fc_value = float(fc_value_input)
            output_values = get_option_1_values(fc_value)
            for column in output_columns:
                window[f"-{column}-"].update(output_values[column])
        except ValueError:
            sg.popup("Invalid fc value input. Please enter a number.")



    elif event == "Option 2":
        window['-FC_VALUE-'].update(visible=False)
        weights_input = sg.popup_get_text("Please enter weights for each objective function, separated by commas.")
        try:
            weights = [float(x.strip()) for x in weights_input.split(',')]
            if len(weights) != len(dff2.columns):
                raise ValueError()
            # Check if the sum of the weights input by the user is 1
            if np.abs(np.sum(weights) - 1) > 0.001:
                raise ValueError("The sum of the weights should be equal to 1.")
            weights = np.array(weights) / np.sum(weights)
            output_values = get_option_2_values(weights)
            for column in output_columns:
                 window[f"-{column}-"].update(output_values[0][output_columns.index(column)])

        except ValueError as e:
            sg.popup(str(e))

window.close()

