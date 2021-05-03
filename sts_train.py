import datasets

if __name__ == "__main__":
    dataset = datasets.load_dataset('glue', 'stsb')
    for x in dataset['train']:
        print(x)
