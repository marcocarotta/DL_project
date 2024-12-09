from dataset import CharDataset


def main():
    with open ('data/dataset.txt', 'r') as f:
        data = f.read()
    # Split the data into train and validation sets
    # since i have a transformer i will do the first 80% of the data as train and the last 20% as validation
    # with an overlap of more or less 2*overlap characters
    train_size = 0.8
    overlap = 250
    train_data = data[:int(len(data)*train_size) + overlap]
    val_data = data[int(len(data)*train_size) - overlap:]

    # check the frequency of each character in the dataset
    block_size = 128
    train_dataset = CharDataset(train_data, block_size).preprocess()
    val_dataset = CharDataset(val_data, block_size).preprocess()

    train_dataset.dataset_analysis()
    val_dataset.dataset_analysis()

    # COMMENT
    # from the dataset analysis we can see that the dataset is balanced and that the characters are well distributed
    # if we apply the preprocess, apart form very few char wrt to the dataset size. (that appear only in train and non in val so there is no problem)


if __name__ == '__main__':
    main()

    
    