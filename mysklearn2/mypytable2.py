"""Module providing a way to pretty print, read files"""
import copy
import csv
from tabulate import tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        if len(self.data) == 0:
            print("The data table is empty.")
            return 0,0
        else:
            number_of_rows = len(self.data)
            number_of_columns = len(self.data[0])
            return number_of_rows, number_of_columns

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """

        #gets column names from self.column names
        column_names = self.column_names

        #first case: check if the col_identifier is a string and handle appropriately
        if isinstance(col_identifier, str):
            if col_identifier in column_names:
                column_index = column_names.index(col_identifier)  # Use index() here
            else:
                raise ValueError("This is an Invalid Column Name")
        #second case: check if the col_identifier is an int for the column index
        elif isinstance(col_identifier, int):
            if col_identifier < 0 or col_identifier >= len(column_names):
                raise ValueError("This is an Invalid Column Index")
            column_index = col_identifier #if valid, store column_index directly from col_identifier

        #third case: the col_identifier isn't a string or int
        else:
            raise ValueError("The Column Identifer must be a column name or index")

        #now need to extract list of values from that column
        column_values = []
        for row in self.data:  #will iterate over entire data
            value = row[column_index]

            if include_missing_values: #checks to see if include_missing_values parameter is true
                column_values.append(value)
            elif include_missing_values is False:
                self.remove_rows_with_missing_values()
                column_values.append(value)
        return column_values

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data: #goes through each row in data
            index = 0 #starts index counter at 0
            for value in row: #goes through each value in the row
                try:
                    row[index] = float(value)
                except ValueError:
                    #if value can't be converted, pass means original value is unchanged
                    pass
                index = index + 1

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        cleaned_data = [] # have an empty cleaned data set
        index = 0

        for row in self.data:  # loop through rows of table
            if index not in row_indexes_to_drop:
                cleaned_data.append(row)
            index = index +1
        self.data = cleaned_data

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        #first initialize an empty list
        table = []

        #next open the file
        input_file = open(filename, "r", encoding = "utf-8")
        #use csv reader object
        csv_data = csv.reader(input_file)

        #assign first row of file to be header
        self.column_names = next(csv_data)

        for row in csv_data:
            table.append(row)

        #now need to assign table to self.data
        self.data = table
        self.convert_to_numeric() #calls this instance method

        #close the file
        input_file.close()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """

        #first open file in write mode
        output_file = open(filename, "w", encoding="utf-8")
         #reads file line by line using csv writer object
        csv_data = csv.writer(output_file)

        #next need to write headers first then data
        csv_data.writerow(self.column_names)
        csv_data.writerows(self.data)

        #close the file
        output_file.close()
