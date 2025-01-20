import pandas as pd
import os
import gettext

# Path to the Modelica file
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir,"data\PipeDataULg151202.mo")

# Load the .mo file using gettext
translation = gettext.GNUTranslations(open(file_path, 'rb'))

# Retrieve the translation for a string (msgid)
translated_string = translation.gettext("Hello")

# Print the translated string
print("Translated string:", translated_string)