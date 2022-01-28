# coding: utf-8

import html
from ipywidgets import Widget, VBox, HTML
from ipytree import Tree, Node
import pandas as pd
import plotly.express as px
from string import ascii_uppercase
from textwrap import indent
from typing import Any, List, Dict

from witan.taxonomy import TaxonomyNode

from ..datasets import DATASETS
from ..models import build_feature_transformer, binarize_features


def escape_html(text: str) -> str:
    """Prepare text for safe printing as HTML."""
    return html.escape(text, quote=False)


class BrowserArgs:
    """Class to store arguments for a taxonomy browser."""

    def __init__(self, dataset_name: str):
        dataset = DATASETS[dataset_name]()
        feature_pipeline = build_feature_transformer(dataset)
        train_features = feature_pipeline.fit_transform(dataset.train.X)
        self.X = binarize_features(train_features)
        self.y = dataset.train.y


class TaxonomyTreeNode(Node):
    """Node for ipytree to represent a TaxonomyNode."""

    def __init__(self, tax_node: TaxonomyNode, *,
                 class_colours: Dict[str, str],
                 y: pd.Series,
                 X_df: pd.DataFrame):
        super().__init__()
        self.opened = True
        self.show_icon = False
        self.tax_node = tax_node

        node_mask = tax_node.get_mask(X_df)
        coverage = node_mask.mean()
        accuracy = (y[node_mask] == tax_node.target_class).mean()

        if tax_node.target_class:
            colour = class_colours[tax_node.target_class]
            target_class = f'<span style="font-weight: bold;">{tax_node.target_class}:</span> '
            stats = f' (coverage: {coverage:.0%}, accuracy: {accuracy:.0%})'
        else:
            colour = '#777777'
            target_class = ''
            stats = ''

        name = escape_html(tax_node.short_name
                           .replace('text__', '')
                           .replace('/', ' / '))
        self.name = (f'<span style="color: {colour};">{target_class}{name}</span>{stats}')

        for child in tax_node.children:
            self.add_node(TaxonomyTreeNode(child,
                                           class_colours=class_colours,
                                           y=y,
                                           X_df=X_df))


def witan_rule_browser(result: Dict[str, Any], browser_args: BrowserArgs) -> Widget:
    """Render an ipytree for a Witan-generated taxonomy."""
    taxonomy = result['rule_extras']['taxonomy']
    class_colours = {
        target_class: colour for target_class, colour in
        zip(sorted(browser_args.y.unique()),
            px.colors.qualitative.Set1)
    }
    tree = Tree([
        TaxonomyTreeNode(node,
                         class_colours=class_colours,
                         y=browser_args.y,
                         X_df=browser_args.X)
        for node in taxonomy.children
    ])
    return VBox([
        HTML('''
<style type="text/css">
    .jstree-icon {
        display: none !important;
    }
    .jstree-default .jstree-wholerow-clicked {
        background: #a8d5fa;
    }
</style>
        '''),
        tree,
    ])


def witan_rule_latex(result: Dict[str, Any], browser_args: BrowserArgs):
    """Return a LaTeX representation of a Witan-generated taxonomy."""
    taxonomy = result['rule_extras']['taxonomy']
    X_df = browser_args.X
    y = browser_args.y
    class_letters = {
        target_class: letter for target_class, letter in
        zip(sorted(browser_args.y.unique()), ascii_uppercase)
    }

    def node_latex(node: TaxonomyNode):
        name = escape_html(node.short_name
                           .replace('text__', '')
                           .replace('/', r'{\lfOr}'))

        if node.target_class:
            letter = class_letters[node.target_class]
            node_mask = node.get_mask(X_df)
            coverage = node_mask.mean()
            accuracy = (y[node_mask] == node.target_class).mean()

            stats = f'(coverage: {coverage:.0%}, accuracy: {accuracy:.0%})'
            latex = fr'\lfClass{letter}{{\textbf{{{node.target_class}:}} {name}}}\lfStats{{{stats}}}'
        else:
            latex = fr'\lfClassNull{{{name}}}'

        if node.children:
            latex += '\n' + indent(node_list_latex(node.children), (' ' * 2))

        return latex

    def node_list_latex(nodes: List[TaxonomyNode]):
        return '\n'.join([
            r'\begin{lfList}',
            *[fr'\item {node_latex(node)}' for node in nodes],
            r'\end{lfList}',
        ])

    return (node_list_latex(taxonomy.children)
            .replace('%', r'\%'))
