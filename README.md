# README

# FAI-OSUSCONCRET
## Introduction

This code is used to optimize the mix design of environmentally sustainable and economical ultra-high-performance concrete (UHPC) based on either a desired compressive strength (Option 1) or weighted objective functions (Option 2). A total of 19 objective functions that comprehensively evalute the compressive strength, cost, and environmental impacts of the mix are optimized.

The code reads data from an Excel file ('optimized_UHPC_all_objectives.xlsx') containing the input variables and objective functions. The user can choose between Option 1 and Option 2 by clicking on the corresponding button.

### Option 1

If the user selects Option 1 by clicking Option 1 button, then the user will be prompted to enter a desired compressive strength (MPa) for the UHPC mix. The software will then use this value to determine the optimized mix considering equal weight for each objective function. The output values are displayed in the corresponding text boxes in the screen.

### Option 2

If the user selects Option 2 by clicking Option 2 button, the user will be prompted to enter weights for each objective function, separated by commas (total of 19 objective functions). The software will then use these weights to determine the optimal mix using the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) method. The output values are displayed in the corresponding text boxes.

### Predict 28-day Compressive Strength of UHPC
Use "Predict 28-day Compressive Strength" button to predict the 28-day compressive strength of a given UHPC mix. For this purpose, enter the mixture content in m3/m3 of UHPC mixture in the prompted window.

### Reset button
Use "Reset" button to clear the field.

## Requirements

To use this application, the following dependencies should be installed:

    Python 3.6 or higher
    PySimpleGUI
    NumPy
    Pandas
    Pillow (PIL)
    openpyxl
    pickle
    tensorflow
    xgboost
    scikit-learn
    
These dependencies can be installed using pip:
E.g., pip install pysimplegui in order to install pysimplegui

## How to Use

    Clone or download the repository.
    Navigate to the directory containing the Python script and the dataset.
    Run the Python script: Sustainable_UHPC_GUI.py
    
    The GUI will open and the user can select either Option 1 or Option 2. The user will be prompted to enter either the desired compressive strength (Option 1) or    weights for each objective function (Option 2). The output values will be displayed in the corresponding text boxes, and the two images will be displayed at the bottom of the GUI.
  
## Contact

For any inquiries or feedback, please contact the authors at tgwakjira@gmail.com and visit www.tadessewakjira.com/Contact.

## Acknowledgements

The application was developed by Abushanab A., Wakjira T., Alnahhal W., and Alam M.S. from the University of British Columbia Okanagan and Qatar University.
