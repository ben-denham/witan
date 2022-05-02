# coding: utf-8

from collections import defaultdict
from math import ceil
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score)
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, Markdown
from scipy.stats import friedmanchisquare
import Orange
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, cast

from ..config import ExperimentConfig
from ..datasets import DATASETS, DATASET_LABELS, Dataset
from ..rule_generators import RulesetGenerator, TrueRG
from ..rule_seeders import RuleSeeder, BlankRS, AccRS
from ..utils import log, prefix_keys, inverse_dict
from ..utils.executors import get_executor, as_completed
from ..models import build_feature_transformer, binarize_features

METRIC_LABELS: Dict[str, str] = {
    'test_macro_f1': 'F1 Score',
    'test_macro_auc': 'AUC',
}
SEEDING_LABELS: Dict[RuleSeeder, str] = {
    BlankRS(): 'Unseeded',
    AccRS(): 'Seeded',
}

STYLE_CONFS: List[Dict[str, Any]] = [
    {
        'labels': ['Full supervision'],
        'color': '#000000',
        'symbol': 'line-ew',
    },
    {
        'labels': ['Wɪᴛᴀɴ', 'Baseline'],
        'color': '#1D2294',
        'symbol': 'square',
    },
    {
        'labels': ['Wɪᴛᴀɴ-Core', 'Core'],
        'color': '#15d2f2',
        'symbol': 'diamond',
    },
    {
        'labels': ['Without ANDs'],
        'color': '#ad5df6',
        'symbol': 'triangle-up',
    },
    {
        'labels': ['Without ORs'],
        'color': '#64ac24',
        'symbol': 'triangle-down',
    },
    {
        'labels': ['Without GE'],
        'color': '#8c564b',
        'symbol': 'circle',
    },
    {
        'labels': ['With feedback'],
        'color': '#ff8016',
        'symbol': 'x',
    },
    {
        'labels': ['IWS-AS', 'IWS-AS-Distinct'],
        'color': '#04a67b',
        'symbol': 'triangle-up',
    },
    {
        'labels': ['IWS-AS-Multi'],
        'color': '#92DECA',
        'symbol': 'triangle-left',
    },
    {
        'labels': ['IWS-LSE-AC', 'IWS-LSE-AC-Distinct'],
        'color': '#7ebb61',
        'symbol': 'triangle-down',
    },
    {
        'labels': ['IWS-LSE-AC-Multi'],
        'color': '#BAD6AE',
        'symbol': 'triangle-right',
    },
    {
        'labels': ['Snuba'],
        'color': '#f87fbd',
        'symbol': 'cross',
    },
    {
        'labels': ['HDC'],
        'color': '#70453c',
        'symbol': 'bowtie',
    },
    {
        'labels': ['CBI'],
        'color': '#eb907e',
        'symbol': 'hourglass',
    },
    {
        'labels': ['Semi-supervised'],
        'color': '#e6ac24',
        'symbol': 'x',
    },
    {
        'labels': ['Active learning'],
        'color': 'red',
        'symbol': 'circle',
    },
]
SERIES_STYLES: pd.DataFrame = pd.DataFrame.from_dict({
    label: conf
    for conf in STYLE_CONFS
    for label in conf['labels']
}, orient='index')
SERIES_COLORS: Dict[str, str] = cast(Dict[str, str], SERIES_STYLES['color'].to_dict())
SERIES_SYMBOLS: Dict[str, str] = cast(Dict[str, str], SERIES_STYLES['symbol'].to_dict())


# RESULT SUMMARISATION

def get_class_prior(classes: np.ndarray, labels: np.ndarray) -> Optional[np.ndarray]:
    """Compute the class prior/distribution, or None if there are no class labels."""
    if len(labels) == 0:
        return None
    return np.array([
        np.sum(labels == target_class)
        for target_class in classes
    ]) / len(labels)


def get_preds(classes: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Convert classifier probs to predictions."""
    return classes[probs.argmax(axis=1)]


def get_rule_stats(classes: np.ndarray,
                   train_y: pd.Series,
                   rule_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistics for the set of rules in the given rule_df."""
    assert len(train_y.shape) == 1
    assert train_y.shape[0] == rule_df.shape[0]

    if rule_df.shape[1] == 0:
        coverage = 0.0
        overlap = 0.0
        class_coverage = [0.0 for target_class in classes]
    else:
        coverage_counts = (~rule_df.isna()).sum(axis=1)
        coverage = (coverage_counts > 0).sum() / rule_df.shape[0]
        overlap = (coverage_counts > 1).sum() / rule_df.shape[0]
        class_coverage = [
            (rule_df == target_class).max(axis=1).sum() / rule_df.shape[0]
            for target_class in classes
        ]

    return {
        'count': rule_df.shape[1],
        'coverage': coverage,
        'overlap': overlap,
        'class_coverage': class_coverage,
    }


def get_pred_stats(classes: np.ndarray,
                   y: pd.Series,
                   preds: np.ndarray) -> Dict[str, Any]:
    """Compute statistics for the given classification predictions."""
    assert len(y.shape) == 1

    if preds.shape[0] == 0:
        accuracy = None
        macro_f1 = None
    else:
        assert preds.shape == y.shape
        accuracy = accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average='macro', zero_division=0)

    return {
        'pred_prior': get_class_prior(classes, preds),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
    }


def get_prob_stats(classes: np.ndarray,
                   y: pd.Series,
                   probs: np.ndarray) -> Dict[str, Any]:
    """Compute statistics for the given classification probabilities."""
    assert len(y.shape) == 1

    if probs.shape[0] == 0:
        prob_prior = None
        macro_auc = None
    else:
        assert probs.shape == (y.shape[0], len(classes))
        prob_prior = np.mean(probs, axis=0)
        if y.nunique() == 1:
            # Only a single class in y (likely because it is covered_y
            # where rules only covered a single class).
            macro_auc = None
        elif len(classes) == 2:
            # Handle binary input specially, as multiclass auc
            # calculation fails for binary input
            macro_auc = roc_auc_score(y_true=(y == classes[0]), y_score=probs[:, 0])
        else:
            macro_auc = roc_auc_score(y_true=y, y_score=probs, labels=classes,
                                      average='macro', multi_class='ovo')

    return {
        'prob_prior': prob_prior,
        'macro_auc': macro_auc,
    }


def summarise_experiment(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return statistics for a given experiment's results."""
    config = result['experiment_config']
    classes = result['classes']
    train_y = result['train_y']
    test_y = result['test_y']

    rule_df = result['rule_df']
    covered_y = train_y[result['covered_train_mask']]
    covered_label_probs = result['covered_train_prob_labels']
    covered_label_preds = get_preds(classes, covered_label_probs)
    test_probs = result['test_probs']
    test_preds = get_preds(classes, test_probs)

    return {
        'config': config,
        **config.to_dict(),
        'train_prior': get_class_prior(classes, test_y),
        'test_prior': get_class_prior(classes, test_y),
        # Rule stats
        **prefix_keys(get_rule_stats(classes, train_y, rule_df), 'rule_'),
        # Rule generation time
        'rule_gen_wall_secs': result['rule_generation_wall_seconds'],
        # Label classifications
        **prefix_keys(get_pred_stats(classes, covered_y, covered_label_preds), 'label_'),
        **prefix_keys(get_prob_stats(classes, covered_y, covered_label_probs), 'label_'),
        # Test-set classifications
        **prefix_keys(get_pred_stats(classes, test_y, test_preds), 'test_'),
        **prefix_keys(get_prob_stats(classes, test_y, test_probs), 'test_'),
    }


def summarise_experiments(results: Dict[ExperimentConfig, Dict[str, Any]],
                          workers: int = 1) -> pd.DataFrame:
    """Return statistics for the given experiment results, computed in parallel."""
    summaries = {}
    with get_executor(max_workers=workers) as executor:
        future_to_key = {}
        for key, result in results.items():
            if result is None:
                log(f'- Skipping missing result for: {key}')
                continue
            future = executor.submit(summarise_experiment, result)
            future_to_key[future] = key
        futures_as_completed = as_completed(executor, list(future_to_key.keys()))
        for i, future in enumerate(futures_as_completed, start=1):
            key = future_to_key[future]
            log(f'- Summarised experiment {i}/{len(future_to_key)}: {key}')
            summaries[key] = future.result()
    return pd.DataFrame([summaries[key] for key in results.keys()])


# PLOTTING

def metric_line_grid(df: pd.DataFrame,
                     facet_row: Optional[str] = None,
                     facet_col: Optional[str] = None,
                     ruleset_generators: Optional[Dict[str, RulesetGenerator]] = None, *,
                     metric: str,
                     legend_y: float = 1.03,
                     facet_row_spacing: float = 0.05,
                     facet_col_spacing: float = 0.05,
                     # Prevents legend labels being cut off
                     legend_label_suffix: str = '  ',
                     **kwargs: Dict[str, Any]) -> go.Figure:
    """Plot a grid of metric line charts for the given results summary
    df. Accepts columns to use as the metric, facet_row, and
    facet_col. Accepts ruleset_generators to provide user-friendly
    names. Displays the mean value when multiple rngseeds are present."""
    # Compute plot sizes.
    cell_height = 350
    if facet_row is None:
        col_count = 2
        row_count = ceil(df[facet_col].nunique() / col_count)
    else:
        col_count = cast(pd.Series, df[facet_col]).nunique()
        row_count = cast(pd.Series, df[facet_row]).nunique()
    total_height = cell_height * row_count

    df = df.copy()
    df = df.fillna(0)

    # Compute mean values over rngseeds
    point_groups = df.groupby(['dataset_name', 'rule_seeder', 'ruleset_generator',
                               'interaction_count', 'labeller', 'classifier'])
    for _, point_group_df in point_groups:
        assert point_group_df.shape[0] == point_group_df['rngseed'].nunique()
    df = point_groups.mean().reset_index().drop(columns=['rngseed'])

    # Duplicate TrueRG results on all plots.
    truerg_rows = df[(df['rule_seeder'] == BlankRS()) & (df['ruleset_generator'] == TrueRG())]
    for rule_seeder in df['rule_seeder'].unique():
        df = pd.concat([df, truerg_rows.assign(rule_seeder=rule_seeder)])
    # Prepare user-friendly names
    df['dataset_name'] = df['dataset_name'].map(DATASET_LABELS)
    if ruleset_generators is not None:
        df['ruleset_generator'] = df['ruleset_generator'].map(inverse_dict(ruleset_generators))
    df['rule_seeder'] = df['rule_seeder'].map(SEEDING_LABELS)
    df['ruleset_generator'] = df['ruleset_generator'] + legend_label_suffix

    row_facets = [] if facet_row is None else df[facet_row].unique()
    col_facets = [] if facet_col is None else df[facet_col].unique()
    fig = px.line(
        df.fillna(0),
        x='interaction_count',
        y=metric,
        color='ruleset_generator',
        symbol='ruleset_generator',
        facet_row=facet_row,
        facet_col=facet_col,
        facet_col_wrap=col_count,
        facet_row_spacing=(facet_row_spacing * (cell_height / total_height)),
        facet_col_spacing=(facet_col_spacing * (cell_height / total_height)),
        color_discrete_map={(label + legend_label_suffix): color for label, color
                            in SERIES_COLORS.items()},
        symbol_map={(label + legend_label_suffix): symbol for label, symbol
                    in SERIES_SYMBOLS.items()},
        labels={
            'ruleset_generator': '',
            'interaction_count': 'Interaction Count',
            **METRIC_LABELS,
        },
        **kwargs,
    )
    fig.update_traces(
        marker=dict(
            size=13,
            opacity=0.5,
        ),
    )
    fig.update_xaxes(
        gridcolor='#AAAAAA',
        zerolinecolor='#888888',
        range=[
            (df['interaction_count'].min() - 5),
            (df['interaction_count'].max() + 5),
        ],
        tickvals=df['interaction_count'].unique(),
    )
    fig.update_yaxes(
        gridcolor='#AAAAAA',
        zerolinecolor='#888888',
    )
    fig.update_layout(
        width=950,
        height=total_height,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            xanchor='center',
            x=0.5,
            y=legend_y,
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(b=10, t=25, l=10, r=10, pad=0),
        font=dict(
            size=20,
            color='black',
        ),
    )

    def annotation_formatter(label):
        field, value = label.text.split("=")
        return label.update(
            text=(f'Dataset: {value}' if (field == 'dataset_name') else value),
            font=dict(size=24),
        )
    fig.for_each_annotation(annotation_formatter)

    fig.update_yaxes(matches=None)
    if facet_row is None:
        fig.for_each_yaxis(lambda axis: axis.update(showticklabels=True))
    else:
        # Use the same range for each row
        y_axis_mins: Dict[go.layout.YAxis, float] = {}
        y_axis_maxs: Dict[go.layout.YAxis, float] = {}

        def update_y_extrema(trace: go.layout.YAxis):
            y_axis_mins[trace.yaxis] = min(
                y_axis_mins.get(trace.yaxis, 1),
                np.min(trace.y),
            )
            y_axis_maxs[trace.yaxis] = max(
                y_axis_maxs.get(trace.yaxis, 0),
                np.max(trace.y),
            )
        fig.for_each_trace(update_y_extrema)

        for row, _ in enumerate(row_facets, start=1):
            row_subplot_y_anchors = [
                fig.get_subplot(row=row, col=col).xaxis.anchor
                for col, _ in enumerate(col_facets, start=1)
            ]
            row_min_y = np.min([y_axis_mins[axis] for axis in row_subplot_y_anchors])
            row_max_y = np.max([y_axis_maxs[axis] for axis in row_subplot_y_anchors])
            row_range = [max(row_min_y - 0.02, 0), (row_max_y + 0.02)]
            fig.update_yaxes(range=row_range, row=row)

    return fig


def build_metric_df(df: pd.DataFrame, *,
                    method: str,
                    metric: str,
                    labelled_methods: Optional[Dict[str, Any]] = None,
                    rngseed_agg: str = 'mean') -> pd.DataFrame:
    """Return a DataFrame presenting the given metric column on each
    dataset for each given method column and interaction count
    combination. Can provide a dict of labelled method values to
    rename them."""
    dataset_names = df['dataset_name'].unique()

    group_rows = []
    for (method_value, interaction_count), group_df in df.groupby([method, 'interaction_count']):
        group_row = {
            'method': method_value,
            'interaction_count': interaction_count,
        }
        for dataset_name in dataset_names:
            cell_df = group_df[group_df['dataset_name'] == dataset_name]
            # We should only have one row per rngseed
            if cell_df.shape[0] != cell_df['rngseed'].nunique():
                display(cell_df)
            assert cell_df.shape[0] == cell_df['rngseed'].nunique()
            # Aggregate values for different rngseeds
            if rngseed_agg == 'mean':
                cell_metric = cell_df[metric].mean()
            elif rngseed_agg == 'std':
                cell_metric = cell_df[metric].std()
            elif rngseed_agg == 'count':
                cell_metric = cell_df[metric].count()
            else:
                raise ValueError(f'Unrecognised rngseed_agg: {rngseed_agg}')
            group_row[dataset_name] = cell_metric
        group_rows.append(group_row)
    metric_df = pd.DataFrame(group_rows)

    if labelled_methods is not None:
        # Use ruleset generator labels
        metric_df['method'] = metric_df['method'].map(inverse_dict(labelled_methods))
        # Sort by provided ordering of methods
        method_order = {label: i for i, label in enumerate(labelled_methods.keys())}

        def sort_key(col):
            if col.name == 'method':
                return col.map(method_order)
            return col
        metric_df = metric_df.sort_values(['method', 'interaction_count'], key=sort_key)

    return metric_df.set_index(['method', 'interaction_count']).fillna(0)


def median_stds_df(metric_std_df: pd.DataFrame, *,
                   datasets: List[str],
                   ics: List[str]) -> pd.DataFrame:
    """Return a dataframe of median standard deviation for each method in
    a given metric_df of standard deviations."""
    methods = metric_std_df.index.get_level_values('method').unique()
    method_stds = defaultdict(list)
    for dataset in datasets:
        for ic in ics:
            for method in methods:
                method_stds[method].append(metric_std_df.loc[(method, ic), dataset])
    return pd.DataFrame({
        'median_std': pd.Series({
            method: np.median(stds)
            for method, stds in method_stds.items()
        }),
        'experiment_count': pd.Series({
            method: len(stds)
            for method, stds in method_stds.items()
        }),
    })


def display_metric_table(metric_df: pd.DataFrame,
                         baseline_label=None,
                         small_margin: float = 0.01,
                         big_margin: float = 0.05,
                         larger_is_better: bool = True,
                         rank_excluded_methods: Optional[List[str]] = None,
                         formatter: str = '{:.3f}') -> Styler:
    """Formats a metric table, with bold highlighting of the best ranked
    method for each dataset and interaction count (with optional
    exclusion of methods via rank_excluded_methods). Also supports
    optional conditional formatting to a baseline method, with two colour
    changes triggered at the small_margin and big_margin respectively.
    """
    rank_excluded_methods = [] if rank_excluded_methods is None else rank_excluded_methods

    table_df = metric_df.copy()
    # Use dataset labels
    table_df = table_df.rename(columns=DATASET_LABELS)  # type: ignore

    # Rename index labels.
    table_df.index = (table_df.index
                      .rename('IC', level='interaction_count')
                      .rename('', level='method'))
    # Set base table styles.
    table_style = (table_df.style
                   .format(formatter=formatter)
                   .set_table_styles([
                       {
                           'selector': ('tbody tr:nth-child(even) td,'
                                        'tbody tr:nth-child(even) th,'
                                        'tbody tr th:first-child'),
                           'props': [('border-bottom', '1px black solid')]
                       },
                       {
                           'selector': 'caption',
                           'props': [('text-align', 'center'),
                                     ('font-weight', 'bold'),
                                     ('color', 'black')],
                       },
                   ]))

    # Optionally highlight differences to baseline ruleset generator.
    if baseline_label is not None:
        diff_df = pd.DataFrame(0, index=table_df.index, columns=table_df.columns)
        for (rg_label, interaction_count), row in table_df.iterrows():  # type: ignore
            # Set cell values relative to baseline.
            for dataset_name, cell_metric in row.iteritems():
                diff_df.loc[(rg_label, interaction_count), dataset_name] = (
                    table_df.loc[(rg_label, interaction_count), dataset_name] -
                    table_df.loc[(baseline_label, interaction_count), dataset_name]
                )

        def diff_format(value):
            if value <= -big_margin:
                if larger_is_better:
                    return 'background-color: #d28977;'
                else:
                    return 'background-color: #79a5b2;'
            elif value <= -small_margin:
                if larger_is_better:
                    return 'background-color: #e6cac3;'
                else:
                    return 'background-color: #c4d6db;'
            elif value >= big_margin:
                if larger_is_better:
                    return 'background-color: #79a5b2;'
                else:
                    return 'background-color: #d28977;'
            elif value >= small_margin:
                if larger_is_better:
                    return 'background-color: #c4d6db;'
                else:
                    return 'background-color: #e6cac3;'
            return None

        format_df = diff_df.applymap(diff_format)
        table_style = table_style.apply(lambda _: format_df, axis=None)

    else:
        def format_best(col_series):
            bold_mask = pd.Series(False, index=col_series.index)
            rank_included_mask = ~col_series.index.get_level_values(0).isin(rank_excluded_methods)
            ic_groups = col_series[rank_included_mask].groupby(level='IC')
            if larger_is_better:
                best_by_ic = ic_groups.max()
            else:
                best_by_ic = ic_groups.min()
            for ic, best_value in best_by_ic.to_dict().items():
                bold_mask = bold_mask | (
                    rank_included_mask &
                    (col_series.index.get_level_values('IC') == ic) &
                    (col_series == best_value)
                )
            return np.where(bold_mask.values, 'font-weight: bold', '')

        # Highlight best values.
        table_style = (table_style
                       .apply(format_best, axis=0))

    return table_style


def display_friedman_test(metric_df: pd.DataFrame,
                          svg_file_prefix: Optional[str] = None) -> None:
    """Display Friedman test p-values and critical distance diagrams for
    each interaction count, also saves the diagrams as SVGs."""
    interaction_counts = metric_df.index.get_level_values('interaction_count').unique()
    for interaction_count in sorted(interaction_counts):
        ic_mask = metric_df.index.get_level_values('interaction_count') == interaction_count
        ic_df = metric_df[ic_mask]
        _, friedman_p = friedmanchisquare(*[
            ic_df[dataset_name].to_list() for dataset_name in metric_df.columns
        ])
        display(Markdown(f'### Interactions: {interaction_count}'))
        print(f'Friedman test p-value: {friedman_p}')
        ranks = ic_df.rank(ascending=False)
        avg_ranks_series = ranks.mean(axis=1)
        avg_ranks = avg_ranks_series.tolist()
        names = avg_ranks_series.index.get_level_values('method').tolist()
        dataset_count = ranks.shape[1]
        cd = Orange.evaluation.compute_CD(avg_ranks, dataset_count, alpha='0.05')
        print('Critical value:', cd)
        Orange.evaluation.graph_ranks(avg_ranks, names, cd=cd,
                                      width=6, textspace=1.5, reverse=False)
        if svg_file_prefix is not None:
            plt.savefig(f'{svg_file_prefix}nemenyi-{interaction_count}.svg',
                        bbox_inches='tight', pad_inches=0)
        plt.show()


# DATASET SUMMARIES

def get_dataset_to_binary_features_df(dataset_names: List[str]) -> Dict[Dataset, pd.DataFrame]:
    """Load and return a dictionary of datasets to training set binary features."""
    result = {}
    for dataset_name in dataset_names:
        dataset = DATASETS[dataset_name]()
        feature_pipeline = build_feature_transformer(dataset)
        train_features = feature_pipeline.fit_transform(dataset.train.X)
        train_binary_features = binarize_features(train_features)
        result[dataset] = train_binary_features
    return result


def dataset_summary(dataset: Dataset,
                    binary_feature_df: pd.DataFrame,
                    min_feature_cov: float = 0.02) -> Dict[str, Any]:
    """Return stats summarising a given dataset and training binary features."""
    feature_count = binary_feature_df.shape[1]
    min_cov_feature_count = (binary_feature_df.mean(axis=0) >= min_feature_cov).sum()
    class_prevalences = dataset.y.value_counts(normalize=True)
    return {
        'Dataset': DATASET_LABELS[dataset.name],
        'Description': dataset.description,
        'Train N': f'{dataset.train.X.shape[0]:,}',
        'Test N': f'{dataset.test.X.shape[0]:,}',
        'Classes': len(dataset.classes),
        'Min/Max Class Prevalence': f'{class_prevalences.min():.0%} / {class_prevalences.max():.0%}',
        f'Features (with {min_feature_cov:.0%} coverage)': f'{feature_count:,} ({min_cov_feature_count:,})',
    }


def datasets_table(dataset_to_binary_feature_df: Dict[Dataset, pd.DataFrame]) -> pd.DataFrame:
    """Return a DataFrame summarising the given datasets"""
    return pd.DataFrame([
        dataset_summary(dataset=dataset, binary_feature_df=binary_feature_df)
        for dataset, binary_feature_df in dataset_to_binary_feature_df.items()
    ])
