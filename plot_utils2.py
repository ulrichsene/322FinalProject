import matplotlib.pyplot as plt
from mysklearn2.mypytable2 import MyPyTable
import numpy as np

def plot_diabetes_status():

    # get the diabetes column
    # create instance of MyPyTable class and load the data
    table = MyPyTable()
    table.load_from_file("input_data2/diabetes_prediction_dataset2.csv")

    diabetes_column = table.get_column("diabetes")

    # count the occurences of 0 and 1 in the diabetes column
    diabetes_counts = [diabetes_column.count(0) , diabetes_column.count(1)]

    labels = ["No Diabetes", "Diabetes"]

    plt.figure()
    plt.pie(diabetes_counts, labels = labels)
    plt.title("Diabetes Distribution")
    plt.show()


def plot_gender_distribution():
    """Plot a pie chart for the gender distribution (Male vs Female)."""
     # create instance of MyPyTable class and load the data
    table = MyPyTable()
    table.load_from_file("input_data2/diabetes_prediction_dataset2.csv")

    gender_column = table.get_column("gender")

    male_count = gender_column.count("Male")
    female_count = gender_column.count("Female")

    labels = ["Male", "Female"]

    plt.figure()
    plt.pie([male_count, female_count], labels = labels)
    plt.title("Gender Distribution")
    plt.show()


def plot_diabetes_by_gender():
    """Plot a bar chart for diabetes distribution by gender (show counts for each gender and diabetes status)."""

     # create instance of MyPyTable class and load the data
    table = MyPyTable()
    table.load_from_file("input_data2/diabetes_prediction_dataset2.csv")
    
    # extract 'diabetes' and 'gender' columns
    diabetes_column = table.get_column("diabetes")
    gender_column = table.get_column("gender")

    # count the number of diabetic and non-diabetic individuals by gender
    male_diabetes = 0
    female_diabetes = 0
    male_no_diabetes = 0
    female_no_diabetes = 0

    for i in range(len(gender_column)):
        if gender_column[i] == "Male":
            if diabetes_column[i] == 1:
                male_diabetes += 1
            else:
                male_no_diabetes += 1
        elif gender_column[i] == "Female":
            if diabetes_column[i] == 1:
                female_diabetes += 1
            else:
                female_no_diabetes += 1

    # data for the bar chart
    labels = ['Male', 'Female']
    diabetes_counts = [male_diabetes, female_diabetes]
    no_diabetes_counts = [male_no_diabetes, female_no_diabetes]

    n = len(labels)

    # use numpy to create positions of bars
    x = np.arange(n)
    bar_width = 0.35

    fig, ax = plt.subplots()

    # plot bars for Diabetes and No Diabetes
    ax.bar(x - bar_width/2, diabetes_counts, bar_width, label='Diabetes', color='green')
    ax.bar(x + bar_width/2, no_diabetes_counts, bar_width, label='No Diabetes', color='red')

    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    ax.set_title('Diabetes Distribution by Gender')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    plt.show()



