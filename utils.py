import numpy as np
from mysklearn2.mypytable2 import MyPyTable

def get_column(table, header, col_name):
    """Extracts a column from the table data as a list.

        Args:
            col_name(str or int): string for a column name or int
                for a column index
            table(list of obj): 2D data structure 
            header(list of str): M column names

        Returns:
            list of obj: 1D list of values in the column
        """
    col_index = header.index(col_name)
    col = []
    for row in table:
        col.append(row[col_index])

    return col

def compute_equal_width_cutoffs(values, num_bins):
    """Computes the cutoff values for discretization.

        Args:
            values(list of int): 1D list of data in a column
            num_bins(int): int for M bins 

        Returns:
            list of float: 1D list of calculated bin values
        """
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins
    cutoffs = [min(values) + i * bin_width for i in range(num_bins)]
    cutoffs.append(max(values))
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]

    return cutoffs

def get_frequencies(table, header, col_name):
    """Get the frequencies of values in column

        Args:
            col_name(str or int): string for a column name or int
                for a column index
            table(list of obj): 2D data structure 
            header(list of str): M column names

        Returns:
            list of obj: 1D list of values in the column
            list of obj: 1D list of frequency of values
        """
    col = get_column(table, header, col_name)

    frequency_dict = {}

    for val in col:
        if val in frequency_dict:
            frequency_dict[val] += 1
        else:
            frequency_dict[val] = 1

    unique_col_values = sorted(frequency_dict.keys())
    counts = [frequency_dict[val] for val in unique_col_values]

    return unique_col_values, counts


def compute_bin_frequencies(values, cutoffs):
    """Find frequencies of values at defined cutoffs

        Args:
            values(list of obj): 1D list of int or str
            cutoffs(list of obj): 1D list of int or string

        Returns:
            list of obj: 1D list of frequencies of values at cutoffs
        """
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for value in values:
        if value == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1

    return freqs

def compute_slope_intercept(x, y):
    """Computes the slope and y-intercept

        Args:
            x(list of int): 1D list of int 
            y(list of int): 1D list of int

        Returns:
            m(int): slope
            b(int): y-intercept
        """
    meanx = sum(x) / len(x)
    meany = sum(y) / len(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = sum([(x[i] - meanx) ** 2 for i in range(len(x))])
    m = num / den
    b = meany - m * meanx

    return m, b

def calculate_r_value(x, y):
    """Calculate r coefficient

        Args:
            x(list of int): 1D list of int
            y(list of int): 1D list of int

        Returns:
            r(int): value computed between [-1, 0]
        """
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x)
    denominator_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = (denominator_x * denominator_y) ** 0.5

    if denominator == 0:
        return 0

    r = numerator / denominator
    return r

def class_discretizer(prediction):
    """
    Discretizes the list of predicted MPG values into DOE mpg ratings.

    Args:
        predictions (list of float): A list of predicted mpg values.

    Returns:
        ratings (list of int): A list of corresponding DOE mpg ratings.
    """
    ranges = [0, 14, 15, 17, 20, 24, 27, 31, 37, 45, 46]
    ratings = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    y_predicted = []

    for value in prediction:
        rating = None

        for i in range(len(ranges) - 1):
            if ranges[i] <= value < ranges[i + 1]:
                rating = ratings[i]
                break

        if rating is None:
            if value >= 45:
                rating = 10
            elif value <= 13:
                rating = 1

        y_predicted.append(rating)

    return y_predicted

def compute_euclidean_distance(x, y):
    """Computes the Euclidean distance between x and y.
    altered for categorical vals

    Args:
        x (list of int or float): first value
        y (list of int or float): second value

    Returns:
        dists: The Euclidean distance between vectors x and y.     
    """
    distances = []
    for index, value in enumerate(x):
        if isinstance(value, (int, float)):
            distances.append((value - y[index]) ** 2)
        elif value == y[index]:
            distances.append(0)
        else:
            distances.append(1)

    return np.sqrt(sum(distances))

def min_max_scale(column):
    """Scales a list of numeric values using min-max scaling.
    
    Args: 
        column (1D list): numeric values in a list
    
    Returns: 
        values: the scaled values of the values in column

    """
    min_val = min(column)
    max_val = max(column)

    if max_val == min_val:
        return [0 for _ in column]

    return [(x - min_val) / (max_val - min_val) for x in column]

def clean_and_save_data(input_file, output_file):
    """Load data, remove rows with 'no info' in the smoking history, and save the cleaned data."""

    # create instance of MyPyTable class and load the data
    table = MyPyTable()
    table.load_from_file(input_file)

    # check the shape of the data before cleaning
    before_drop_shape = table.get_shape()
    print(f"Before cleaning - Rows: {before_drop_shape[0]}, Columns: {before_drop_shape[1]}")

    # extract the smoking history column and find rows to drop
    smoking_column = table.get_column("smoking_history")

    rows_to_drop = []
    for index, value in enumerate(smoking_column):
        if value == "No Info":
            rows_to_drop.append(index)

    # drop the rows with "No Info" in the smoking history column
    table.drop_rows(rows_to_drop)

    # check the shape of the data after cleaning
    after_drop_shape = table.get_shape()
    print(f"After cleaning - Rows: {after_drop_shape[0]}, Columns: {after_drop_shape[1]}")

    # save the cleaned data to a new file
    table.save_to_file(output_file)
