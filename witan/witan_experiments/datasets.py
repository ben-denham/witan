import os
import re
import numpy as np
import pandas as pd
import requests
import json
from copy import deepcopy
from typing import cast, Set, Optional, Mapping, Callable

# Use a different seed for each stage of an experiment to prevent
# overlaps and unintended correlation. Different orders of magnitude
# so that they can be incremented several times if needed.
DATASET_RANDOM_SEED = 1_000_000_000


class Dataset():
    """Class for managing datasets."""

    def __init__(self,
                 df: pd.DataFrame, *,
                 test_mask: pd.Series,
                 text_features: Optional[Set[str]] = None,
                 description: str = '') -> None:
        self.name = 'UNNAMED'
        self.description = description

        pd.testing.assert_index_equal(df.index, test_mask.index, exact=True)
        self.df = df.reset_index()
        self.test_mask = test_mask.reset_index(drop=True)

        self.classes = np.array(sorted(self.df['class'].unique()))
        self.text_features = (set() if text_features is None
                              else set(text_features))

    def set_name(self, name: str) -> None:
        self.name = name

    @property
    def all_features(self) -> Set[str]:
        return self.text_features

    @property
    def X(self) -> pd.DataFrame:
        return self.df[self.all_features]

    @property
    def y(self) -> pd.Series:
        return self.df['class']

    @property
    def indexes(self) -> np.ndarray:
        return self.df.index.to_numpy()

    def subset(self, subset_df: pd.DataFrame) -> 'Dataset':
        """Return a dataset with the given subset of this dataset's df."""
        # Don't use the constructor, otherwise we'd reset df.index.
        subdataset = deepcopy(self)
        subdataset.df = subset_df
        return subdataset

    @property
    def train(self):
        """Return dataset of the training dataset."""
        if not hasattr(self, '_train'):
            self._train = self.subset(self.df[~self.test_mask])
            self._train.test_mask = False
        return self._train

    @property
    def test(self):
        """Return dataset of the test dataset."""
        if not hasattr(self, '_test'):
            self._test = self.subset(self.df[self.test_mask])
            self._test.test_mask = True
        return self._test


# Downloading / Caching

def uci_url(path: str) -> str:
    return 'https://archive.ics.uci.edu/ml/machine-learning-databases/{}'.format(path)


def cache_path(filename: str) -> str:
    return os.path.join('../data', filename)


def cache_file_locally(local_path: str, remote_url: str) -> None:
    """If local_path does not exist, Downloads the remote_url and saves it
    to local_url."""
    if os.path.isfile(local_path):
        return
    r = requests.get(remote_url, verify=True)
    with open(local_path, 'wb') as f:
        f.write(r.content)


def cache_and_unzip(local_compressed_path: str, local_target_dir_path: str,
                    remote_url: str, targz: bool = False) -> None:
    """If local_compressed_path does not exist, downloads the remote_url,
    and ensures it is uncompressed inside local_target_dir_path."""
    cache_file_locally(local_compressed_path, remote_url)
    # Decompress
    if not os.path.isdir(local_target_dir_path):
        os.mkdir(local_target_dir_path)
        unzip_command = 'tar -xvzf' if targz else 'unzip'
        command = (f'cd {local_target_dir_path} && '
                   f'{unzip_command} ../{os.path.basename(local_compressed_path)}')
        os.system(command)


# DATASETS

# Each dataset function:
# * Makes use of caching to prevent redownloading each dataset.
# * Returns a Dataset containing a DataFrame and metadata.
# * Stores the target feature for classification in a 'class' column.

def iws_dataset_loader(*, csv_name: str, class_labels: Mapping[int, str], **kwargs):
    """Generic dataset loader for datasets used by the IWS codebase:
    https://github.com/benbo/interactive-weak-supervision"""
    local_targz = cache_path('iws_datasets.tar.gz')
    local_dir = cache_path('iws_datasets')
    cache_and_unzip(local_targz, local_dir, 'https://ndownloader.figshare.com/files/25732838?private_link=860788136944ad107def', targz=True)
    local_file = os.path.join(local_dir, f'{csv_name}.csv')
    df = pd.read_csv(local_file, sep=',', index_col=False)
    df = df.rename(columns={'label': 'class'})
    df['class'] = df['class'].replace(class_labels)

    return Dataset(
        df=df,
        text_features={'text'},
        test_mask=(df['fold'] == 1),
        **kwargs,
    )


def imdb_dataset() -> Dataset:
    return iws_dataset_loader(
        csv_name='IMDB',
        class_labels={1: 'positive', -1: 'negative'},
        description='IMDB Reviews',
    )


def bias_pa_dataset() -> Dataset:
    return iws_dataset_loader(
        csv_name='painter_architect',
        class_labels={1: 'architect', -1: 'painter'},
        description='Bias Bios: Painter vs Architect',
    )


def bias_pt_dataset() -> Dataset:
    return iws_dataset_loader(
        csv_name='professor_teacher',
        class_labels={1: 'teacher', -1: 'professor'},
        description='Bias Bios: Professor vs Teacher',
    )


def bias_jp_dataset() -> Dataset:
    return iws_dataset_loader(
        csv_name='journalist_photographer',
        class_labels={1: 'photographer', -1: 'journalist'},
        description='Bias Bios: Journalist vs Photographer',
    )


def bias_pp_dataset() -> Dataset:
    return iws_dataset_loader(
        csv_name='professor_physician',
        class_labels={1: 'physician', -1: 'professor'},
        description='Bias Bios: Professor vs Physician',
    )


def amazon_dataset() -> Dataset:
    return iws_dataset_loader(
        csv_name='Amazon',
        class_labels={1: 'positive', -1: 'negative'},
        description='Amazon Reviews',
    )


def make_test_mask(df: pd.DataFrame, *, test_frac: float) -> pd.Series:
    """Return a random mask covering the given test_frac of the df."""
    test_mask = pd.Series(np.full((df.shape[0],), fill_value=False))
    test_count = round(df.shape[0] * test_frac)
    test_mask.iloc[:test_count] = True
    test_mask = cast(pd.Series, test_mask.sample(frac=1.0, random_state=1))
    test_mask.index = df.index
    return test_mask


def xclass_dataset_loader(name: str, url: str,
                          sample_frac: float = 1.0,
                          test_frac: float = 0.5,
                          manual_download_required: bool = False,
                          **kwargs) -> Dataset:
    """Generic dataset loader for datasets used by the xclass codebase:
    https://github.com/ZihanWangKi/XClass"""
    local_zip = cache_path(f'{name}.zip')
    local_dir = cache_path(name)

    if manual_download_required and not os.path.isfile(local_zip):
        raise ValueError(
            f'Please download dataset zip from {url} and place at: {local_zip}')

    cache_and_unzip(local_zip, local_dir, url)

    with open(os.path.join(local_dir, name, f'classes.txt'), 'r') as classes_file:
        classes = np.array([target_class.strip() for target_class in classes_file.readlines()])
    with open(os.path.join(local_dir, name, f'dataset.txt'), 'r') as text_file:
        X_list = list(text_file.readlines())
    with open(os.path.join(local_dir, name, f'labels.txt'), 'r') as labels_file:
        y_list = [classes[int(label)] for label in labels_file.readlines()]

    df = pd.DataFrame({
        'text': X_list,
        'class': y_list,
    })
    df = (cast(pd.DataFrame, df.sample(frac=sample_frac, random_state=0))
          .reset_index(drop=True))

    return Dataset(
        df=df,
        text_features={'text'},
        test_mask=make_test_mask(df, test_frac=test_frac),
        **kwargs,
    )


def twentynews_dataset() -> Dataset:
    return xclass_dataset_loader(
        name='20News',
        url='https://drive.google.com/uc?export=download&id=1y5x3MGVqUW0h5yLEDKdJdd5hZTztSQUo',
        description='20Newsgroups Topics'
    )


def agnews_dataset() -> Dataset:
    return xclass_dataset_loader(
        name='AGNews',
        url='https://drive.google.com/uc?export=download&id=1G4yyaT80l0U5blOkJs4J2m2_DO4_Kx-a',
        description='AGNews Topics',
    )


def binary_agnews_dataset() -> Dataset:
    dataset = agnews_dataset()
    df = dataset.df
    binary_mask = df['class'].isin(['Business', 'Technology'])
    return Dataset(
        df=df[binary_mask],
        text_features=dataset.text_features,
        test_mask=cast(pd.Series, dataset.test_mask[binary_mask]),
        description='AGNews Topics: Business vs Technology',
    )


def dbpedia_dataset() -> Dataset:
    return xclass_dataset_loader(
        name='DBpedia',
        url='https://drive.google.com/uc?export=download&confirm=&id=1pRvMzHaxFhBzes01C6hvzKw1fv-luqjl',
        # NOTE: This file needs to be download manually due to
        # failed Google virus scanning on larger files.
        manual_download_required=True,
        sample_frac=0.2,
        description='DBPedia Categories',
    )


def binary_dbpedia_dataset() -> Dataset:
    dataset = xclass_dataset_loader(
        name='DBpedia',
        url='https://drive.google.com/uc?export=download&confirm=&id=1pRvMzHaxFhBzes01C6hvzKw1fv-luqjl',
        # NOTE: This file needs to be download manually due to
        # failed Google virus scanning on larger files.
        manual_download_required=True,
    )
    df = dataset.df
    binary_mask = df['class'].isin(['Politics', 'Company'])
    return Dataset(
        df=df[binary_mask],
        text_features=dataset.text_features,
        test_mask=cast(pd.Series, dataset.test_mask[binary_mask]),
        description='DBPedia Categories: Politics vs Company',
    )


def nyttopics_dataset() -> Dataset:
    return xclass_dataset_loader(
        name='NYT-Topics',
        url='https://drive.google.com/uc?export=download&id=1fLu1lzmIRW29283cTivloqO4CLn0Hhpy',
        # NOTE: This file needs to be download manually due to
        # failed Google virus scanning on larger files.
        manual_download_required=True,
        description='NYT Topics',
    )


def yelp_dataset() -> Dataset:
    return xclass_dataset_loader(
        name='Yelp',
        url='https://drive.google.com/uc?export=download&id=1_jvk39CrZxPPrIVFyN1GbSXaJReTTqn-',
        description='Yelp Reviews',
    )


def plots_dataset(test_frac: float = 0.5) -> Dataset:
    local_path = cache_path('plots.txt')
    url = 'https://raw.githubusercontent.com/HazyResearch/reef/master/data/imdb/budgetandactors.txt'
    cache_file_locally(local_path, url)

    with open(local_path) as f:
        movies = [json.loads(line) for line in f]

    def get_class(movie):
        genre = movie['Genre']
        if 'Action' in genre:
            return 'action'
        elif 'Romance' in genre:
            return 'romance'
        else:
            return None

    df = pd.DataFrame({
        'text': [movie['Plot'] for movie in movies],
        'class': [get_class(movie) for movie in movies],
    })
    df = df[~df['class'].isna()]

    return Dataset(
        df=df,
        text_features={'text'},
        test_mask=make_test_mask(df, test_frac=test_frac),
        description='IMDB Plots: Action vs Romance',
    )


def airline_tweets_dataset(test_frac: float = 0.5) -> Dataset:
    local_file = cache_path('Tweets.csv')
    try:
        df = pd.read_csv(local_file)
    except FileNotFoundError as ex:
        raise ValueError(
            'Please download Tweets.csv from '
            'https://www.kaggle.com/crowdflower/twitter-airline-sentiment '
            f'and place at: {local_file}') from ex

    df = df[['airline_sentiment', 'text']]
    df = df.rename(columns={'airline_sentiment': 'class'})
    df = df[df['class'].isin(['positive', 'negative'])]

    return Dataset(
        df=df,
        text_features={'text'},
        test_mask=make_test_mask(df, test_frac=test_frac),
        description='Airline Tweet Sentiment',
    )


def damage_dataset(test_frac: float = 0.5):
    local_zip = cache_path('damage.zip')
    local_dir = cache_path('damage')
    cache_and_unzip(local_zip, local_dir, uci_url('00456/multimodal-deep-learning-disaster-response-mouzannar.zip'))
    data_dir = os.path.join(local_dir, 'multimodal')
    class_dfs = []
    for class_name in os.listdir(data_dir):
        class_texts = []
        class_texts_dir = os.path.join(data_dir, class_name, 'text')
        for instance_file in os.listdir(class_texts_dir):
            with open(os.path.join(class_texts_dir, instance_file)) as f:
                class_texts.append(f.read())
        class_dfs.append(pd.DataFrame({
            'text': class_texts,
            'category': class_name,
        }))
    df = pd.concat(class_dfs)

    df['class'] = 'damage'
    df.loc[(df['category'] == 'non_damage'), 'class'] = 'non_damage'

    return Dataset(
        df=df,
        text_features={'text'},
        test_mask=make_test_mask(df, test_frac=test_frac),
        description='Disaster tweets',
    )


def spam_dataset(test_frac: float = 0.5):
    local_zip = cache_path('spam.zip')
    local_dir = cache_path('spam')
    cache_and_unzip(local_zip, local_dir, uci_url('00228/smsspamcollection.zip'))
    local_file = os.path.join(local_dir, 'SMSSpamCollection')
    df = pd.read_csv(local_file, sep='\t', names=['class', 'text'])
    return Dataset(
        df=df,
        text_features={'text'},
        test_mask=make_test_mask(df, test_frac=test_frac),
        description='SMS Spam',
    )


def fakenews_dataset(test_frac: float = 0.5) -> Dataset:
    local_dir = cache_path('fakenews')
    try:
        true_df = pd.read_csv(os.path.join(local_dir, 'True.csv'))
        fake_df = pd.read_csv(os.path.join(local_dir, 'Fake.csv'))
    except FileNotFoundError as ex:
        raise ValueError(
            'Please download True.csv and Fake.csv from '
            'https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset '
            f'and place in: {local_dir}') from ex

    true_df['class'] = 'true'
    fake_df['class'] = 'fake'
    df = pd.concat([true_df, fake_df])
    # Remove initial Reuters formatting boilerplate
    df['text'] = df['text'].str.replace(r'.*\(Reuters\) -', '', regex=True)
    df['all_text'] = ((df['title'] + ' ' + df['text'])
                      .str.replace('reuters', '', flags=re.IGNORECASE))

    return Dataset(
        df=df,
        text_features={'all_text'},
        test_mask=make_test_mask(df, test_frac=test_frac),
        description='Fake News',
    )


# Dataset collection

def named_datasets(datasets: Mapping[str, Callable[[], Dataset]]) -> Mapping[str, Callable[[], Dataset]]:
    """Given a dict of dataset names to dataset building functions, return
    an analogous dict where the dataset building function will set the
    dataset's name after construction."""

    def named_dataset_decorator(
            name: str,
            dataset_func: Callable[[], Dataset]) -> Callable[[], Dataset]:

        def named_dataset_func() -> Dataset:
            dataset = dataset_func()
            dataset.set_name(name)
            return dataset

        return named_dataset_func

    return {
        name: named_dataset_decorator(name, dataset_func)
        for name, dataset_func in datasets.items()
    }


DATASETS = named_datasets({
    'imdb': imdb_dataset,
    'bias_pa': bias_pa_dataset,
    'bias_pt': bias_pt_dataset,
    'bias_jp': bias_jp_dataset,
    'bias_pp': bias_pp_dataset,
    'amazon': amazon_dataset,
    'twentynews': twentynews_dataset,
    'agnews': agnews_dataset,
    'binary_agnews': binary_agnews_dataset,
    'dbpedia': dbpedia_dataset,
    'binary_dbpedia': binary_dbpedia_dataset,
    'nyttopics': nyttopics_dataset,
    'yelp': yelp_dataset,
    'plots': plots_dataset,
    'airline_tweets': airline_tweets_dataset,
    'damage': damage_dataset,
    'spam': spam_dataset,
    'fakenews': fakenews_dataset,
})


DATASET_LABELS = {
    'imdb': 'IMD',
    'bias_pa': 'BPA',
    'bias_pt': 'BPT',
    'bias_jp': 'BJP',
    'bias_pp': 'BPP',
    'amazon': 'AZN',
    'yelp': 'YLP',
    'plots': 'PLT',
    'twentynews': 'TWN',
    'binary_dbpedia': 'BDB',
    'binary_agnews': 'BAG',
    'dbpedia': 'DBP',
    'agnews': 'AGN',
    'nyttopics': 'NYT',
    'airline_tweets': 'ATW',
    'damage': 'DMG',
    'spam': 'SPM',
    'fakenews': 'FNS',
}
