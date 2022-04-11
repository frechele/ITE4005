import pandas as pd
import numpy as np
import random

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

def calc_gain(parent: pd.DataFrame, children: List[pd.DataFrame]):
    if len(children) == 1:
        return -np.inf

    return calc_info([parent]) - calc_info(children)
    

def calc_gain_ratio(parent: pd.DataFrame, children: List[pd.DataFrame]):
    if len(children) == 1:
        return -np.inf

    gain = calc_gain(parent, children)
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

        label_counts = df['label'].value_counts(normalize=True)
        max_label = np.argmax(label_counts.to_numpy())

        self.label = label_counts.index[max_label]

        self.children: Dict[str, DTNode] = None
        self.split_pivot = None

    def _split(self, column: str) -> Dict[str, pd.DataFrame]:
        children = dict()

        types = self.df[column].unique()
        for t in types:
            children[t] = self.df[self.df[column] == t]

        return children

    def predict(self, df: pd.DataFrame) -> str:
        if self.is_leaf:
            return self.label

        key = df[self.split_pivot]
        if key not in self.children:
            return self.label

        return self.children[key].predict(df)

    def fit(self):
        columns = self.df.columns[:-1]
        scores = np.zeros(len(columns))

        if len(self.df) < 3:
            self.is_leaf = True
            return

        for i, column in enumerate(columns):
            children = self._split(column)
            scores[i] = calc_gain_ratio(self.df, list(children.values()))

        max_score = np.max(scores)
        if np.isneginf(max_score) or max_score < 1e-1:
            self.is_leaf = True
            return

        column = columns[np.argmax(scores)]
        self.split_pivot = column
        self.children = {t: DTNode(child) for t, child in self._split(column).items()}
        
        for child in self.children.values():
            if not child.is_leaf:
                child.fit()


class DTNode2:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.is_leaf = (len(df['label'].unique()) == 1)

        label_counts = df['label'].value_counts(normalize=True)
        max_label = np.argmax(label_counts.to_numpy())

        self.label = label_counts.index[max_label]

        self.children: List[DTNode2] = None
        self.split_pivot = None
        self.split_value = None

    def _split(self, column: str, value: str) -> List[pd.DataFrame]:
        children = list()
        children.append(self.df[self.df[column] == value])
        children.append(self.df[self.df[column] != value])

        return children

    def predict(self, df: pd.DataFrame) -> str:
        if self.is_leaf:
            return self.label

        key = 0 if df[self.split_pivot] == self.split_value else 1
        return self.children[key].predict(df)

    def fit(self):
        features = []
        for column in self.df.columns[:-1]:
            values = self.df[column].unique()
            for value in values:
                features.append([column, value])

        scores = np.zeros(len(features))

        if len(self.df) < 3:
            self.is_leaf = True
            return

        for i, feat in enumerate(features):
            column, value = feat
            children = self._split(column, value)


            if len(children[0]) == 0 or len(children[1]) == 0:
                scores[i] = -np.inf
            else:
                scores[i] = calc_gain_ratio(self.df, children)

        max_score = np.max(scores)
        if np.isneginf(max_score):
            self.is_leaf = True
            return

        column, value = features[np.argmax(scores)]
        self.split_pivot = column
        self.split_value = value
        self.children = [DTNode2(df) for df in self._split(column, value)]
        
        for child in self.children:
            if not child.is_leaf:
                child.fit()


class BDTNode:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.is_leaf = (len(df['label'].unique()) == 1)

        label_counts = df['label'].value_counts(normalize=True)
        max_label = np.argmax(label_counts.to_numpy())

        self.label = label_counts.index[max_label]

        self.children: List[BDTNode] = None
        self.split_pivot = None
        self.split_value = None

    def _split(self, column: str, value: str) -> List[pd.DataFrame]:
        children = list()
        children.append(self.df[self.df[column] == value])
        children.append(self.df[self.df[column] != value])

        return children

    def predict(self, df: pd.DataFrame) -> str:
        if self.is_leaf:
            return self.label

        key = 0 if df[self.split_pivot] == self.split_value else 1
        return self.children[key].predict(df)

    def fit(self):
        features = []
        for column in self.df.columns[:-1]:
            values = self.df[column].unique()
            for value in values:
                features.append([column, value])

        scores = np.zeros(len(features))

        if len(self.df) < 3:
            self.is_leaf = True
            return

        for i, feat in enumerate(features):
            column, value = feat
            children = self._split(column, value)


            if len(children[0]) == 0 or len(children[1]) == 0:
                scores[i] = -np.inf
            else:
                scores[i] = calc_gini_index(self.df, children)

        max_score = np.max(scores)
        if np.isneginf(max_score):
            self.is_leaf = True
            return

        column, value = features[np.argmax(scores)]
        self.split_pivot = column
        self.split_value = value
        self.children = [BDTNode(df) for df in self._split(column, value)]
        
        for child in self.children:
            if not child.is_leaf:
                child.fit()


class DecisionTree:
    def __init__(self, backend):
        self.backend = backend

    def fit(self, df: pd.DataFrame):
        self.tree = self.backend(df)
        self.tree.fit()

    def predict(self, df: pd.DataFrame) -> str:
        return self.tree.predict(df)


class RandomForest:
    def __init__(self, n_estimators: int):
        self.n_estimators: int = n_estimators
        self.estimators: List[DecisionTree] = []

    def fit(self, df: pd.DataFrame):
        indices = list(range(len(df)))

        dfs = []
        stride = len(indices)//self.n_estimators
        for i in range(self.n_estimators):
            random.shuffle(indices)
            dfs.append(df.iloc[indices[:-stride]])

        for i in range(self.n_estimators//2):
            tree = DecisionTree(DTNode2)
            tree.fit(dfs[i])
            self.estimators.append(tree)

        for i in range(self.n_estimators//2):
            tree = DecisionTree(BDTNode)
            tree.fit(dfs[i])
            self.estimators.append(tree)

    def predict(self, df: pd.DataFrame) -> str:
        votes = dict()

        for i, estimator in enumerate(self.estimators):
            result = estimator.predict(df)
            votes[result] = votes.get(result, 0) + 1

        max_result = None
        max_value = -999
        for res, val in votes.items():
            if val > max_value:
                max_value = val
                max_result = res
        return max_result


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

    if len(df_train) < 100:
        model = DecisionTree()
        model.fit(df_train, calc_gain_ratio)
    else:
        model = RandomForest(n_estimators=100)
        model.fit(df_train)

    for i in range(len(df_test)):
        df_test.loc[i, orig_label] = model.predict(df_test.iloc[i])

    df_test.to_csv(output_file, sep='\t', index=False)
