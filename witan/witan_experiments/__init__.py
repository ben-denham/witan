from datetime import datetime
import os
from joblib import dump, load
from functools import partial, wraps
import multiprocessing as mp
import numpy as np
import pandas as pd
from typing import Optional, Any, Sequence, List, Dict, Callable

from .config import (Config, PrepConfig, RulesConfig, LabelsConfig,
                     ExperimentConfig, condense_rules_configs)
from .datasets import DATASETS, Dataset
from .models import build_feature_transformer, binarize_features
from .utils import log, new_log_file
from .utils.executors import get_executor, as_completed

# Use a different seed for each stage of an experiment to prevent
# overlaps and unintended correlation. Different orders of magnitude
# so that they can be incremented several times if needed.
SEED_RULES_RANDOM_SEED = 1_000
RULE_GENERATION_RANDOM_SEED = 1_000_000
LABELLING_RANDOM_SEED = 1_000_000_000


# CACHING

CACHE_DIR = 'results'
BREAK_CACHE = False


class ExperimentError(Exception):
    pass


def cache_filepath(key: str) -> str:
    filename = f'dump.joblib'
    return os.path.join(CACHE_DIR, key, filename)


def is_cached(key: str) -> bool:
    if BREAK_CACHE:
        return False
    return os.path.isfile(cache_filepath(key))


def save_to_cache(key: str, obj: Any) -> None:
    filepath = cache_filepath(key)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    dump({key: obj, 'timestamp': datetime.now()}, filepath)


def load_from_cache(key: str) -> Any:
    return load(cache_filepath(key))[key]


def config_cache(execute_func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for functions that accept a config as first argument,
    which handles caching results for that config."""

    @wraps(execute_func)
    def run_func(config: Config, *args, **kwargs):
        if not is_cached(config.cache_key()):
            try:
                result = execute_func(config, *args, **kwargs)
            except Exception as ex:
                raise ExperimentError(f'Exception raised for config: {config}') from ex
            save_to_cache(config.cache_key(), result)
        # Load from cache even after execution in case the save/load
        # introduces subtle changes.
        return config, load_from_cache(config.cache_key())

    return run_func


# RUNNING EXPERIMENTS

def run_in_parallel(funcs: List[Callable[[], Any]], *,
                    continue_on_failure: bool,
                    workers: int) -> List[Any]:
    """Run the given funcs in parallel processes, and return the results."""
    results = []
    failure_count = 0
    with get_executor(max_workers=workers, mp_context=mp.get_context('spawn')) as executor:
        future_to_func = {executor.submit(func): func for func in funcs}
        futures = list(future_to_func.keys())
        for i, future in enumerate(as_completed(executor, futures), start=1):
            try:
                result = future.result()
            except Exception as ex:
                failure_count += 1
                if continue_on_failure:
                    config_desc = ''
                    try:
                        # If the func is a partial, get the first
                        # argument (typically the config object) as a
                        # description.
                        args = future_to_func[future].args  # type: ignore
                        config_desc = f' ({args[0]})'
                    except (AttributeError, IndexError):
                        pass
                    log(f'- Failed task {i}/{len(futures)}{config_desc}: {type(ex).__name__}({ex})')
                else:
                    for future in futures:
                        future.cancel()
                    raise ex
            else:
                results.append(result)
                log(f'- Completed task {i}/{len(futures)}')
    if failure_count > 0:
        log(f'- WARNING! there were {failure_count} failures!')
    return results


@config_cache
def run_data_preparation(prep_config: PrepConfig, *,
                         dataset: Dataset) -> Dict[str, Any]:
    """Run data preparation for the given prep_config and dataset."""
    feature_pipeline = build_feature_transformer(dataset)
    train_features = feature_pipeline.fit_transform(dataset.train.X)
    test_features = feature_pipeline.transform(dataset.test.X)
    return {
        'prep_config': prep_config,
        'train_features': train_features,
        'test_features': test_features,
    }


@config_cache
def run_rules_generation(rules_config: RulesConfig, *,
                         dataset: Dataset,
                         train_features: pd.DataFrame) -> Dict[str, Any]:
    """Run rule generation for the given rules_config."""
    train_binary_features = binarize_features(train_features)
    log('Generating seed rules')
    seed_rules = rules_config.rule_seeder.get_seed_rules(
        classes=dataset.classes,
        train_binary_features=train_binary_features,
        train_y=dataset.train.y,
        rngseed=rules_config.rngseed * SEED_RULES_RANDOM_SEED,
    )
    log('Generating rules')
    ruleset = rules_config.ruleset_generator.generate_ruleset(
        classes=dataset.classes,
        train_binary_features=train_binary_features,
        train_y=dataset.train.y,
        seed_rules=seed_rules,
        rngseed=rules_config.rngseed * RULE_GENERATION_RANDOM_SEED,
        interaction_count=rules_config.interaction_count,
    )
    return {
        'rules_config': rules_config,
        'seed_rules': seed_rules,
        'ruleset': ruleset,
    }


@config_cache
def run_labels_generation(labels_config: LabelsConfig, *,
                          dataset: Dataset,
                          train_features: pd.DataFrame,
                          rule_df: pd.DataFrame) -> Dict[str, Any]:
    """Run probabilistic labels generation for the given labels_config."""
    covered_train_mask = ((~rule_df.isna()).sum(axis=1) > 0)
    if covered_train_mask.sum() == 0:
        covered_train_prob_labels = np.zeros((0, len(dataset.classes)))
    else:
        covered_rule_df = rule_df[covered_train_mask]
        covered_train_prob_labels = labels_config.labeller.get_probabilistic_labels(
            classes=dataset.classes,
            rngseed=labels_config.rules_config.rngseed * LABELLING_RANDOM_SEED,
            rule_df=covered_rule_df,
        )
    return {
        'labels_config': labels_config,
        'covered_train_mask': covered_train_mask,
        'covered_train_prob_labels': covered_train_prob_labels,
    }


@config_cache
def run_classification(experiment_config: ExperimentConfig, *,
                       dataset: Dataset,
                       train_features: pd.DataFrame,
                       test_features: pd.DataFrame,
                       rule_df: pd.DataFrame,
                       covered_train_mask: pd.DataFrame,
                       covered_train_prob_labels: np.array) -> Dict[str, Any]:
    """Run classification the given experiment_config."""
    if covered_train_mask.sum() == 0:
        test_probs = np.zeros((0, len(dataset.classes)))
    else:
        covered_train_features = train_features[covered_train_mask]
        test_probs = experiment_config.classifier.predict_probs(
            classes=dataset.classes,
            covered_train_features=covered_train_features,
            test_features=test_features,
            covered_train_prob_labels=covered_train_prob_labels,
        )

    return {
        'experiment_config': experiment_config,
        'test_probs': test_probs,
    }


def execute_experiments(experiment_configs: Sequence[ExperimentConfig], *,
                        continue_on_failure: bool,
                        default_workers: int,
                        rule_workers: Optional[int]) -> Dict[ExperimentConfig, Dict[str, Any]]:
    """Execute experiments for the given experiment_configs"""
    if rule_workers is None:
        rule_workers = default_workers
    parallelise = partial(run_in_parallel,
                          continue_on_failure=continue_on_failure,
                          workers=default_workers)

    dataset_names = set([config.dataset_name for config in experiment_configs])
    log(f'Loading datasets ({len(dataset_names)} tasks)')
    datasets = {}
    for dataset_name in dataset_names:
        try:
            datasets[dataset_name] = DATASETS[dataset_name]()
        except Exception as ex:
            if continue_on_failure:
                log(f'Failure loading dataset: {type(ex).__name__}({ex})')
            else:
                raise ex

    prep_configs = set([
        config.prep_config for config in experiment_configs
        # Skip prep_configs where the dataset failed to load.
        if config.prep_config.dataset_name in datasets
    ])
    log(f'Preparing data ({len(prep_configs)} tasks)')
    prep_results = dict(
        parallelise([
            partial(
                run_data_preparation,
                prep_config,
                dataset=datasets[prep_config.dataset_name],
            )
            for prep_config in prep_configs
        ]),
    )

    rules_configs = set([
        config.rules_config for config in experiment_configs
        # Skip rules_configs where the prep_result failed.
        if config.rules_config.prep_config in prep_results
    ])
    rules_configs_map = condense_rules_configs(rules_configs)
    condensed_rules_configs = rules_configs_map.keys()
    log(f'Generating rules ({len(condensed_rules_configs)} tasks)')
    condensed_rules_results = dict(
        parallelise([
            partial(
                run_rules_generation,
                rules_config,
                dataset=datasets[rules_config.prep_config.dataset_name],
                train_features=prep_results[rules_config.prep_config]['train_features'],
            )
            for rules_config in condensed_rules_configs
        ], workers=rule_workers),
    )
    # Expand condensed_rules_results into rules_results for different
    # interaction counts.
    rules_results = {}
    for condensed_rules_config, rules_config_group in rules_configs_map.items():
        for rules_config in rules_config_group:
            try:
                condensed_rules_result = condensed_rules_results[condensed_rules_config]
                ruleset = condensed_rules_result['ruleset']
                rules_results[rules_config] = {
                    'rules_config': rules_config,
                    'seed_rules': condensed_rules_result['seed_rules'],
                    'rule_df': rules_config.ruleset_generator.get_rule_df(
                        ruleset=ruleset,
                        interaction_count=rules_config.interaction_count,
                    ),
                    'elapsed_wall_seconds': rules_config.ruleset_generator.get_elapsed_wall_seconds(
                        ruleset=ruleset,
                        interaction_count=rules_config.interaction_count,
                    ),
                    'rule_extras': rules_config.ruleset_generator.get_extras(
                        ruleset=ruleset,
                        interaction_count=rules_config.interaction_count,
                    ),
                }
            except Exception as ex:
                if continue_on_failure:
                    log(f'Failure processing rules result: {type(ex).__name__}({ex})')
                else:
                    raise ex

    labels_configs = set([
        config.labels_config for config in experiment_configs
        # Skip labels_configs where the rules_result failed.
        if config.labels_config.rules_config in rules_results
    ])
    log(f'Generating labels ({len(labels_configs)} tasks)')
    labels_results = dict(
        parallelise([
            partial(
                run_labels_generation,
                labels_config,
                dataset=datasets[labels_config.rules_config.prep_config.dataset_name],
                train_features=prep_results[labels_config.rules_config.prep_config]['train_features'],
                rule_df=rules_results[labels_config.rules_config]['rule_df'],
            )
            for labels_config in labels_configs
        ]),
    )

    classification_experiment_configs = [
        config for config in experiment_configs
        # Skip experiment_configs where the labels_result failed.
        if config.labels_config in labels_results
    ]
    log(f'Performing classification ({len(classification_experiment_configs)} tasks)')
    classification_results = dict(
        parallelise([
            partial(
                run_classification,
                experiment_config,
                dataset=datasets[experiment_config.dataset_name],
                train_features=prep_results[experiment_config.prep_config]['train_features'],
                test_features=prep_results[experiment_config.prep_config]['test_features'],
                rule_df=rules_results[experiment_config.rules_config]['rule_df'],
                covered_train_mask=labels_results[experiment_config.labels_config]['covered_train_mask'],
                covered_train_prob_labels=labels_results[experiment_config.labels_config]['covered_train_prob_labels'],
            )
            for experiment_config in classification_experiment_configs
        ])
    )

    log('Experiments complete!')
    failed_configs = set(experiment_configs).difference(set(classification_results.keys()))
    if len(failed_configs) > 0:
        failed_desc = ',\n'.join([
            f'- {config}' for config in failed_configs
        ])
        if continue_on_failure:
            log(
                (f'FAILURE! The following {len(failed_configs)} experiments '
                 f'failed, and are excluded from the results:\n{failed_desc}\n')
            )
        else:
            raise RuntimeError(
                (f'{len(failed_configs)} experiment_configs missing '
                 f'from classification_results:\n{failed_desc}')
            )

    experiment_results = {}
    for experiment_config in classification_results.keys():
        dataset = datasets[experiment_config.dataset_name]
        prep_result = prep_results[experiment_config.prep_config]
        assert prep_result['prep_config'] == experiment_config.prep_config
        rules_result = rules_results[experiment_config.rules_config]
        assert rules_result['rules_config'] == experiment_config.rules_config
        labels_result = labels_results[experiment_config.labels_config]
        assert labels_result['labels_config'] == experiment_config.labels_config
        classification_result = classification_results[experiment_config]
        assert classification_result['experiment_config'] == experiment_config

        experiment_results[experiment_config] = {
            'experiment_config': experiment_config,
            'dataset_description': dataset.description,
            'classes': dataset.classes,
            'train_y': dataset.train.y,
            'test_y': dataset.test.y,
            'seed_rules': rules_result['seed_rules'],
            'rule_df': rules_result['rule_df'],
            'rule_generation_wall_seconds': rules_result['elapsed_wall_seconds'],
            'rule_extras': rules_result['rule_extras'],
            'covered_train_mask': labels_result['covered_train_mask'],
            'covered_train_prob_labels': labels_result['covered_train_prob_labels'],
            'test_probs': classification_result['test_probs'],
        }
    return experiment_results


def run_experiments(experiment_configs: Sequence[ExperimentConfig], *,
                    continue_on_failure: bool = False,
                    default_workers: int = 12,
                    rule_workers: Optional[int] = None) -> Dict[ExperimentConfig, Dict[str, Any]]:
    """Execute or load cached experiments for the given experiment_configs."""
    new_log_file()

    experiment_results = {}
    result_cache_keys = {
        config: os.path.join(config.cache_key(), 'result')
        for config in experiment_configs
    }

    uncached_experiment_configs = []
    for config in experiment_configs:
        if is_cached(result_cache_keys[config]):
            log(f'- Loading result from cache for: {config}')
            try:
                experiment_results[config] = load_from_cache(result_cache_keys[config])
            except Exception:
                log(f'-- Failed to load from cache: {config}')

        if config not in experiment_results:
            uncached_experiment_configs.append(config)
            experiment_results[config] = None

    if len(uncached_experiment_configs) > 0:
        executed_results = execute_experiments(
            uncached_experiment_configs,
            continue_on_failure=continue_on_failure,
            default_workers=default_workers,
            rule_workers=rule_workers,
        )
        log('Caching results...')
        for config, result in executed_results.items():
            save_to_cache(result_cache_keys[config], result)
            experiment_results[config] = load_from_cache(result_cache_keys[config])
        log('Cached results!')

    return experiment_results
