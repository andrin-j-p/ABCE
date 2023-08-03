import unittest
import sys
import os

# Get the current directory (where the test file is located)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (containing the 'functions' folder) to the module search path
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

# Import the function from the 'functions' folder
from Sugarsim.read_data import read_dataframe

"""
Name: TestReadDataFram
Purpose: Unittesting for functions in read_data
Input: None
Output: 
"""
class TestReadData(unittest.TestCase):

    def test_columns_exist(self):
        # Test if the DataFrame contains certain columns
        file_path = 'CleanGeography_PUBLIC.dta'
        df = read_dataframe(file_path)
        print(df.head())

        # List of expected columns
        expected_columns = ['subcounty']

        # Check if all expected columns exist in the DataFrame
        for col in expected_columns:
            self.assertIn(col, df.columns)

if __name__ == '__main__':
    unittest.main()