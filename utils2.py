from mysklearn2.mypytable2 import MyPyTable

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

