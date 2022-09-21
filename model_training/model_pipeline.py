
from model_training.train_model import fit_save_model, read_split_data
from model_training.validate_model import evaluate_per_slice


def main():
   train, test = read_split_data(data_path="../data/census.csv")
   fit_save_model(model_filename="model_output", train=train)
   evaluate_per_slice("model_output", test)

if __name__ == '__main__':
    main()