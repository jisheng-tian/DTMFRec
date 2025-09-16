import sys
import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed, set_color
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, get_flops, get_environment

from DTMFRec import DTMFRec

if __name__ == '__main__':
    # 1. Configuration initialization
    config = Config(
        model=DTMFRec,  # Specify the model class
        config_file_list=['config.yaml']  # Path to config file
    )

    # 2. Set random seed
    init_seed(config['seed'], config['reproducibility'])

    # 3. Initialize logger
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(f"Loaded config: {config}")

    # 4. Load dataset
    dataset = create_dataset(config)
    logger.info(f"Dataset loaded: {dataset}")

    # 5. Split dataset (train / valid / test)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 6. Load and initialize model
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = DTMFRec(config, train_data.dataset).to(config['device'])
    logger.info(f"Model initialized:\n{model}")

    # 7. Compute FLOPs (optional, requires torchvision)
    from recbole.data.transform import construct_transform

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # 8. Initialize trainer
    trainer = Trainer(config, model)

    # 9. Train the model
    logger.info(set_color("Start training...", "green"))
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # 10. Evaluate the model
    logger.info(set_color("Start evaluation...", "green"))
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )

    # 11. Output environment and results
    environment_tb = get_environment(config)
    logger.info(
        "Training environment:\n" + environment_tb.draw()
    )
    logger.info(set_color("Best validation result", "yellow") + f": {best_valid_result}")
    logger.info(set_color("Test result", "yellow") + f": {test_result}")
