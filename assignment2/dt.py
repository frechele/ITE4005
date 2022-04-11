import pandas as pd
import numpy as np

import sys
from typing import Dict, List, Tuple


def calc_info(children: List[pd.DataFrame]):
    col_frac = np.zeros(len(children))
    infos = np.zeros(len(children))

    for i, df in enumerate(children):
        frac = df['label'].value_counts(normalize=True).to_numpy()
        infos[i] = -np.sum(frac * np.log2(frac))
        col_frac[i] = len(df)

    col_frac /= np.sum(col_frac)
    return np.sum(np.multiply(col_frac, infos))

def calc_gini(children: List[pd.DataFrame]):
    col_frac = np.zeros(len(children))
    ginis = np.zeros(len(children))

    for i, df in enumerate(children):
        frac = df['label'].value_counts(normalize=True).to_numpy()
        ginis[i] = 1 - np.sum(frac * frac)
        col_frac[i] = len(df)

    col_frac /= np.sum(col_frac)
    return np.sum(np.multiply(col_frac, ginis))

def calc_gain_ratio(parent: pd.DataFrame, children: List[pd.DataFrame]):
    if len(children) == 1:
        return -np.inf

    gain = calc_info([parent]) - calc_info(children)
    split_info = np.array([len(child) for child in children])
    split_info = split_info / np.sum(split_info)
    split_info = -np.sum(split_info * np.log2(split_info))

    return gain / split_info

def calc_gini_index(parent: pd.DataFrame, children: List[pd.DataFrame]):
    if len(children) == 1:
        return -np.inf

    return calc_gini([parent]) - calc_gini(children)


class DTNode:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.is_leaf = (len(df['label'].unique()) == 1)
        self.label = df['label'].unique()[0]

        self.children: Dict[str, DTNode] = None
        self.split_pivot = None

    def _split(self, column: str) -> Dict[str, pd.DataFrame]:
        children = dict()

        types = self.df[column].unique()
        for t in types:
            children[t] = self.df[self.df[column] == t]

        return children

    def fit(self, metric):
        columns = self.df.columns[:-1]
        scores = np.zeros(len(columns))

        for i, column in enumerate(columns):
            children = self._split(column)
            scores[i] = metric(self.df, list(children.values()))

        column = columns[np.argmax(scores)]
        self.split_pivot = column
        self.children = {t: DTNode(child) for t, child in self._split(column).items()}

        if np.isneginf(np.max(scores)):
            self.is_leaf = True
            return

        for child in self.children.values():
            if not child.is_leaf:
                child.fit(metric)


class DecisionTree:
    def fit(self, df: pd.DataFrame, metric):
        self.tree = DTNode(df)
        self.tree.fit(metric)

    def predict(self, df: pd.DataFrame) -> str:
        node = self.tree
        while not node.is_leaf:
            key = df[node.split_pivot]
            if key not in node.children:
                break

            node = node.children[key]

        return node.label


def parse_dataset(filename: str, training: bool) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(filename, sep='\t')
    orig_label = df.columns[-1]
    if training:
        df.rename(columns = { orig_label : 'label' }, inplace = True)
        
    return df, orig_label


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python {} <training file> <test file> <output file>'.format(sys.argv[0]))
        sys.exit(-1)
    
    training_file, test_file, output_file = sys.argv[1:]

    df_train, orig_label = parse_dataset(training_file, True)
    df_test, _ = parse_dataset(test_file, False)

    model = DecisionTree()
    model.fit(df_train, calc_gain_ratio)

    for i in range(len(df_test)):
        df_test.loc[i, orig_label] = model.predict(df_test.iloc[i])

    df_test.to_csv(output_file, sep='\t', index=False)