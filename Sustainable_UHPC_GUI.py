import pandas as pd
import numpy as np
import PySimpleGUI as sg
from PIL import Image
from PIL import ImageOps
import pickle

df = pd.read_excel('optimized_UHPC_all_objectives.xlsx')

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

dff1 = df[['C (m3/m3)', 'SF (m3/m3)', 'FA (m3/m3)', 'GBFS (m3/m3)', 'LP (m3/m3)',
       'W (m3/m3)', 'SP (m3/m3)', 'QP (m3/m3)', 'MSA (m3/m3)', 'A (m3/m3)', 'fc (Mpa)',
       'Cost (USD)', 'Climate Change', 'Fossil Depletion',
       'Freshwater Ecotoxicity', 'Freshwater Eutrophication', 'Human Toxicity',
       'Ionising Radiation', 'Marine Ecotoxicity', 'Marine Eutrophication',
       'Metal Depletion', 'Natural Land Transformation', 'Ozone Depletion',
       'Particulate Matter Formation', 'Photochemical Oxidant Formation',
       'Terrestrial Acidification', 'Terrestrial Ecotoxicity',
       'Urban Land Occupation', 'Water Depletion']]


# only objective functions
dff2 = dff1[['fc (Mpa)',
       'Cost (USD)', 'Climate Change', 'Fossil Depletion',
       'Freshwater Ecotoxicity', 'Freshwater Eutrophication', 'Human Toxicity',
       'Ionising Radiation', 'Marine Ecotoxicity', 'Marine Eutrophication',
       'Metal Depletion', 'Natural Land Transformation', 'Ozone Depletion',
       'Particulate Matter Formation', 'Photochemical Oxidant Formation',
       'Terrestrial Acidification', 'Terrestrial Ecotoxicity',
       'Urban Land Occupation', 'Water Depletion']]


min_max_values = {
    'C': (0.06164037855, 0.6136507937),
    'SF': (0, 0.2055688492),
    'FA': (0, 0.2497716895),
    'GBFS': (0, 0.2465517241),
    'LP': (0, 0.1902560232),
    'W': (0.04892065, 0.387),
    'SP': (0.002571746051, 0.3047381132),
    'QP': (0, 0.234811988),
    'MSA': (0, 16000),
    'A': (0, 0.6936381777),
#     'fc (MPa)': (100.048, 241.462),
}

def normalize_input_data(input_data):
    min_values = np.array([0.06164037855, 0, 0, 0, 0, 0.04892065, 0.002571746051, 0, 0, 0])
    max_values = np.array([0.6136507937, 0.2055688492, 0.2497716895, 0.2465517241, 0.1902560232, 0.387, 0.3047381132, 0.234811988, 16000, 0.6936381777])
    normalized_input_data = (input_data - min_values) / (max_values - min_values)
    return normalized_input_data

def predict_fc_value(input_data):
    normalized_input_data = normalize_input_data(input_data)
    X = np.array(normalized_input_data).reshape(1, -1)
    prediction = model.predict(X)[0]
    denormalized_prediction = 100.048 + (241.462 - 100.048) * prediction  # Denormalize the predicted value
    return denormalized_prediction, input_data

def get_option_1_values(fc_value):
    df.columns = ['C (m3/m3)', 'SF (m3/m3)', 'FA (m3/m3)', 'GBFS (m3/m3)', 'LP (m3/m3)',
       'W (m3/m3)', 'SP (m3/m3)', 'QP (m3/m3)', 'MSA (m3/m3)', 'A (m3/m3)', 'fc (MPa)'] + list(df.columns[11:])
    closest_row = df.iloc[(df['fc (MPa)']-fc_value).abs().argsort()[0]]
    default_values = closest_row.round(3)
    return default_values


def get_option_2_values(weights):
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

output_columns = ['C (m3/m3)', 'SF (m3/m3)', 'FA (m3/m3)', 'GBFS (m3/m3)', 'LP (m3/m3)',
       'W (m3/m3)', 'SP (m3/m3)', 'QP (m3/m3)', 'MSA (m3/m3)', 'A (m3/m3)', 'fc (MPa)', 'Cost (USD)', 'Climate Change', 'Fossil Depletion', 'Freshwater Ecotoxicity', 'Freshwater Eutrophication', 'Human Toxicity', 'Ionising Radiation', 'Marine Ecotoxicity', 'Marine Eutrophication', 'Metal Depletion', 'Natural Land Transformation', 'Ozone Depletion', 'Particulate Matter Formation', 'Photochemical Oxidant Formation', 'Terrestrial Acidification', 'Terrestrial Ecotoxicity', 'Urban Land Occupation', 'Water Depletion']

# output_columns = ['C', 'SF', 'FA', 'GBFS', 'LP', 'W', 'SP', 'QP', 'MSA', 'A', 'fc (MPa)']

column_1 = output_columns[:10]
column_2 = output_columns[10:20]

output_layout_column_1 = [[sg.Text(column, size=(12, 1)), sg.Input(default_text="", key=f"-{column}-", size=(10, 1), disabled=True, border_width=1, justification='center')] for column in column_1]
output_layout_column_2 = [[sg.Text(column, size=(20, 1)), sg.Input(default_text="", key=f"-{column}-", size=(10, 1), disabled=True, border_width=1, justification='center')] for column in column_2]

column_3 = output_columns[20:]  # select columns from "Photochemical Oxidant Formation" onwards
output_layout_column_3 = [[sg.Text(column, size=(25, 1)), sg.Input(default_text="", key=f"-{column}-", size=(10, 1), disabled=True, border_width=1, justification='center')] for column in column_3]

# output_layout.append([sg.Column(output_layout_column_1), sg.Column(output_layout_column_2), sg.Column(output_layout_column_3)])

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
#     [sg.Button("Option 1", button_color=('white', 'gray')), sg.Button("Option 2", button_color=('white', 'gray'))],
    [sg.Button("Option 1", button_color=('white', 'gray')), sg.Button("Option 2", button_color=('white', 'gray')), 
     sg.Button("Predict 28-day Compressive Strength", button_color=('white', 'gray')),
     sg.Button("Reset", button_color=('white', 'gray'))],  # Add the Reset button

    [sg.Text("Enter desired fc value (MPa) for Option 1:", visible=False, font=("Helvetica", 14)), sg.Input(key="-FC_VALUE-", size=(10, 1), visible=False)],
    *output_layout
]






# Open the images
img1 = Image.open('image1.png')
img2 = Image.open('image2.png')
img3 = Image.open('image3.png')

# Get the minimum width and height among the images
widths = [img1.width, img2.width, img3.width]
heights = [img1.height, img2.height, img3.height]
min_width = min(widths)
min_height = min(heights)

# Resize the images to the minimum size
img1 = ImageOps.fit(img1, (min_width, min_height))
img2 = ImageOps.fit(img2, (min_width, min_height))
img3 = ImageOps.fit(img3, (min_width, min_height))

# Define the scale factor
scale_factor = 0.4

# Resize the images
img1 = img1.resize((int(min_width * scale_factor), int(min_height * scale_factor)))
img2 = img2.resize((int(min_width * scale_factor), int(min_height * scale_factor)))
img3 = img3.resize((int(min_width * scale_factor), int(min_height * scale_factor)))

# Save the resized images
img1.save('image11.png')
img2.save('image22.png')
img3.save('image33.png')

# To add figures in two columns
fig1 = sg.Image(filename='image11.png', key='-fig1-', size=(min_width * scale_factor, min_height * scale_factor))
fig2 = sg.Image(filename='image22.png', key='-fig2-', size=(min_width * scale_factor, min_height * scale_factor))
fig3 = sg.Image(filename='image33.png', key='-fig3-', size=(min_width * scale_factor, min_height * scale_factor))



# fig1_desc = sg.Text('Image 1')
# fig2_desc = sg.Text('Image 2')
layout += [[sg.Column([[sg.Text('Authors: Tadesse G. Wakjira, Adeeb A. Kutty, M. Shahria Alam')],
                [sg.Text('Contact: tgwakjira@gmail.com,'+ '\n'
                         '             www.tadessewakjira.com/Contact')],
            ],
            element_justification='left'
        ),
        sg.Column(
            [   [fig1,
                fig2,
                fig3,],
            ],
            element_justification='center'
        ),
    ]
]




column_1 = [
    ("                                         URL:", "https://github.com/twakjira/Optimization-of-Sustainable-UHPC"),
]

output_layout_column_1 = [
    [
        sg.Text(column, size=(24, 1)),
        sg.Input(
            default_text=value,
            key=f"-{column}-",
            size=(60, 2),
            disabled=True,
            border_width=1,
            justification="center",
            background_color="white",
            text_color="black",
            disabled_readonly_background_color="white",
            disabled_readonly_text_color="black",
        ) if value != "Authors:" and value != "Website:" else sg.Text(value, background_color="white", text_color="black"),
    ]
    for column, value in column_1
]

layout += output_layout_column_1



window = sg.Window("FAI-OSUSCONCRET", layout)

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

    elif event == "Reset":
        for column in output_columns:
            window[f"-{column}-"].update("")            
            
    elif event == "Predict 28-day Compressive Strength":
        window['-FC_VALUE-'].update(visible=False)
        input_data_text = sg.popup_get_text("Please enter the volumes of 'C', 'SF', 'FA', 'GBFS', 'LP', 'W', 'SP', 'QP', 'MSA', 'A', separated by commas.")
        try:
            input_data = [float(x.strip()) for x in input_data_text.split(',')]
            if len(input_data) != 10:
                raise ValueError("Please provide exactly 10 values.")
            fc_predicted, input_data = predict_fc_value(input_data)
            sg.popup(f"The predicted fc (MPa) value is: {fc_predicted:.3f}")

            # Update the window with the input data values
            for i, column in enumerate(column_1[:-1]):  # Exclude the last column (fc (MPa))
                window[f"-{column}-"].update(input_data[i])

            # Update the 'A' value in the window
            window['-A (m3/m3)-'].update(input_data[-1])

            # Update the fc (MPa) value in the window
            window['-fc (MPa)-'].update(fc_predicted)


        except ValueError as e:
            sg.popup(str(e))
        except Exception as e:
            sg.popup(f"Error: {str(e)}")


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
                 window[f"-{column}-"].update(round(output_values[0][output_columns.index(column)], 6))

        except ValueError as e:
            sg.popup(str(e))           

window.close()
