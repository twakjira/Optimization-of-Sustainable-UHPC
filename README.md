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

Outputs

The code also displays two images at the bottom of the GUI:

    'fig.png': a plot showing the trade-off between the compressive strength and the environmental impact categories for the optimized UHPC mix.
    'ALAMS1.png': a logo for the ALkali Activated Materials Society.

How to use

To use the code, run the script and select either Option 1 or Option 2. If you select Option 1, enter a desired compressive strength (MPa) when prompted. If you select Option 2, enter weights for each objective function separated by commas when prompted. The output values will be displayed in the corresponding text boxes.
Note

The code assumes that the input Excel file 'optimized_UHPC_all_objectives.xlsx' is located in the same directory as the script. If the file is located elsewhere, the path to the file should be updated in the script.
