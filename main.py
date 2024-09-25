import argparse

from time import time
from src.aux import load_config
from src.data_loader import DataLoader
from src.model import NeuralNet
from src.plotter import Plotter


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, 
                        default='config/default.yaml',
                        help="Path to configuration file")
    args = parser.parse_args()
    return args


def train_and_test(config):

    # Initializing DataLoader instance
    data = DataLoader(config)

    # Initializing model instance
    model = NeuralNet(config)

    # Model training
    model.train(data)

    # Initializing Plotter instance
    plots = Plotter(data, model)

    # Create learning curves plot
    plots.learning_curves()

    # Create performance plot
    plots.performance()

    return model.log


if __name__ == '__main__':

    args = parse_arguments()
    time_start = time()

    # Load configuration file
    config = load_config(args.config)

    # Training and testing with specified config
    logs = train_and_test(config)
    # Read and print final L2-error
    L2_error = logs['L2_error'][0]
    print(f'Final L2-Error: {L2_error:1.1e}')

    time_end = time()
    print(f"Finished in {time_end-time_start:.2f} seconds!")
