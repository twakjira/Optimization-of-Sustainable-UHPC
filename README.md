# Optimization-of-Sustainable-UHPC
README

This code is used to optimize the mix design of sustainable ultra-high-performance concrete (UHPC) based on either a desired compressive strength (Option 1) or weighted objective functions (Option 2).

The code reads data from an Excel file ('optimized_UHPC_all_objectives.xlsx') containing the input variables and objective functions. The user can choose between Option 1 and Option 2 by clicking on the corresponding button.
Option 1

If the user selects Option 1, they will be prompted to enter a desired compressive strength (MPa) for the UHPC mix. The code will then use this value to determine the optimized mix based on the closest row in the input data that matches the desired compressive strength. The output values are displayed in the corresponding text boxes.
Option 2

If the user selects Option 2, they will be prompted to enter weights for each objective function, separated by commas. The code will then use these weights to determine the optimal mix using the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) method. The output values are displayed in the corresponding text boxes.
Requirements

The code requires the following packages to be installed:

    pandas
    numpy
    PySimpleGUI
    Pillow (PIL)

How to use

To use the code, , simply execute the script in a Python environment that has the required packages installed. The GUI will open and the user can select either Option 1 or Option 2. The user will be prompted to enter either the desired compressive strength (Option 1) or weights for each objective function (Option 2). The output values will be displayed in the corresponding text boxes, and the two images will be displayed at the bottom of the GUI.
