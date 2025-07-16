This is a README file which explains how to set up this application. 

The first step is to have the Python version of 3.11.6. This was the version which the application was developed. Python 3.13 is not going to work because the depedencies needed are not up to date, and the minimum version is 3.10. The full list of the requirements are saved as requirements.txt. Use pip install -r requirements.txt. The development was done using Ubuntu, but it should work cross platform if the python version is 3.11 

The final user application is in the Graphical User Interface folder, with the assets used for designing it. The streamlit library is used, and is a dependency. For output, the following command is used in the command line: streamlit run gui2.py. if it does not work, use streamlit run (gui2.py path). 

Here is an example:
streamlit run "/home/danny/Desktop/001254746_FYP_Code/Graphical User Interface/gui2.py"


The processed dataset was the output from the data_preprocessing, and the raw data was the initial dataset. The processed one is then used for the classification algorithm and in the final user interface. 

in the model folder, it can be found the saved models which were used, and in the src file, the python files can be found. 
The first one being the preprocessing, Random Forest model, LSTM model and the classification testing, where it can be seen how explainable AI libraries are implemented
