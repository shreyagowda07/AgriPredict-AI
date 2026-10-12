import pandas as pd


def load_data(file_path):
    """Load the crop yield dataset."""
    data = pd.read_csv(file_path)
    return data


def clean_data(data):
    """Clean the dataset."""
    
    # Remove duplicate rows
    data = data.drop_duplicates()

    # Remove rows with missing values
    data = data.dropna()

    return data


if __name__ == "__main__":

    file_path = "data/raw/crop_yield.csv"

    # Load the dataset
    data = load_data(file_path)

    print("Original Dataset:")
    print(data)

    # Clean the dataset
    cleaned_data = clean_data(data)

    print("\nCleaned Dataset:")
    print(cleaned_data)

    print("\nDataset Information:")
    print(cleaned_data.info())