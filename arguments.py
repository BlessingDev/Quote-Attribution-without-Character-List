from argparse import Namespace, ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    
    # data configuration
    parser.add_argument(
        "--dataset_path", 
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_books",
        type=str,
        default="0 1 2 4 5 6 7 9 10 11 13 14 15 16",
        nargs='?'
    )
    parser.add_argument(
        "--val_books",
        type=str,
        default="3 8 12 17",
        nargs='?'
    )
    parser.add_argument(
        "--test_books",
        type=str,
        default="18 19 20 21",
        nargs='?'
    )
    
    # file configuration
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_state_file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--reload_from_files",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--expand_filepaths_to_save_dir",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--log_json_file",
        default="train_at_{time}.json"
    )
    
    # model configuration
    parser.add_argument(
        "--feature_dimension",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--decoder_hidden",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--feature_freedom",
        type=int,
        default=1
    )
    
    # train configuration
    parser.add_argument(
        "--seed",
        type=int,
        default=10025
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-8,
        type=float
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=int
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float
    )
    parser.add_argument(
        "--saved_anchor_num",
        default=2,
        type=int
    )
    parser.add_argument(
        "--detach_mems_step",
        default=2,
        type=int
    )
    parser.add_argument(
        "--loss_sig",
        default=0.5,
        type=float
    )
    parser.add_argument(
        "--loss_epsilon",
        default=10,
        type=int
    )
    parser.add_argument(
        "--early_stopping_criteria",
        default=3,
        type=int
    )
    parser.add_argument(
        "--max_epochs",
        default=10,
        type=int
    )
    parser.add_argument(
        "--book_train_iter",
        default=2,
        type=int
    )
    
    # etc
    parser.add_argument(
        "--catch_keyboard_interrupt",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False
    )
    
    return parser