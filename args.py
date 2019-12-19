# Command-line arguments for setup, train, test
import argparse

def get_setup_args():
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""

    parser = argparse.ArgumentParser('Download and preprocess dataset')

    parser.add_argument('--download_url',
                        type=str,
                        default='https://storage.googleapis.com/kagglesdsdata/competitions/3364/31151/fer2013.tar.gz?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1576831866&Signature=ltaqYmggNs6vnr67Zc71XoqqarHsXyx1RtdvKkut0bM5k0C13nTIAhlASHfvsQIsF7BEXWFVXp7V2PmPtqY3%2FESzp2EXwPHrXhk9ERBb4Q6V1Sn8viBq%2FFwmvX%2FS7X%2F8Rcd185I0iCxunubOCCUNAIBukwQYbdi6gyF8KzvZ4fEXHJxtD2YkWTx%2B%2BuE6P1EBBgyU1FYsmmx9Jz9JqoyOTkRvp6m3fwi65oNkI%2FlH3i7YPrDJPdvMHvYlJK194HQfHt1fuBdJBcjeInQjEh3xBdmfDJmldIF9xOQ59KyuD4uk0yaQdgXf08c1hSkPH7sSk%2F%2BUHVVWLCa04p9MCXmuDA%3D%3D&response-content-disposition=attachment%3B+filename%3Dfer2013.tar.gz')

    parser.add_argument('--dataset_path',
                        type=str,
                        default='./data/fer2013/fer2013.csv')

    parser.add_argument('--save',
                        type=str,
                        default=None)

    parser.add_argument('--train_faces',
                        type=str,
                        default='./data/train_faces.npy')

    parser.add_argument('--dev_faces',
                        type=str,
                        default='./data/dev_faces.npy')

    parser.add_argument('--test_faces',
                        type=str,
                        default='./data/test_faces.npy')

    parser.add_argument('--train_labels',
                        type=str,
                        default='./data/train_labels.npy')

    parser.add_argument('--dev_labels',
                        type=str,
                        default='./data/dev_labels.npy')

    parser.add_argument('--test_labels',
                        type=str,
                        default='./data/test_labels.npy')

    args = parser.parse_args()

    return args

def get_train_args():
    parser = argparse.ArgumentParser('Train_model')

    parser.add_argument('--dataset_path',
                        type=str,
                        default='./data/fer2013/fer2013.csv')

    parser.add_argument('--load_from_path',
                        type=str,
                        default=None,
                        help='Load data from path or not')

    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs')

    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Number of batch size')

    parser.add_argument('--lr',
                        type=float,
                        default=1e-2,
                        help='Learning rate')

    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='Momentum for SGD')

    parser.add_argument('--model',
                        type=str,
                        default='VGG19',
                        help='model for training')

    parser.add_argument('--weight',
                        type=str,
                        default=None,
                        help='imagenet weight')

    args = parser.parse_args()

    return args