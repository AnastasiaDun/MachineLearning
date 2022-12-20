import pandas as pd
import matplotlib.pyplot as plt


def read_csv(filename):
    data = pd.read_csv(filename)
    return data


def plot(dataset):
    genders = dataset.Sex
    filter_male = genders == 'male'
    filter_female = genders == 'female'

    maleDataset = dataset.loc[filter_male]
    femaleDataset = dataset.loc[filter_female]


    m = maleDataset[['Survived', 'Pclass']][maleDataset['Survived'] == 1].groupby('Pclass').count()
    print(m)
    m.plot.bar()
    plt.show()


    f = femaleDataset[['Survived', 'Pclass']][femaleDataset['Survived'] == 1].groupby('Pclass').count()
    f.plot.bar()
    plt.show()



    df = maleDataset['Pclass'].value_counts()
    print(df)
    df.plot(kind="pie", label="")
    plt.show()


    Age_sort_dataset = dataset['Age'].value_counts()
    print(Age_sort_dataset)
    Age_sort_dataset.plot.bar()
    plt.show()


if __name__ == '__main__':
    file = "train_titanic.csv"
    dataset = read_csv(file)
    plot(dataset)
