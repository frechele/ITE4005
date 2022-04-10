import pandas as pd
import sys
from typing import List


class DTNode:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.is_leaf = (len(df['label'].unique()) == 1)

        self.children: List[DTNode] = []

    def _split(self, column: str, feat: str):
        assert not self.is_leaf

        self.children.append(self.df[self.df[column] == feat])
        self.children.append(self.df[self.df[column] != feat])

    


class DecisionTree:
    def fit(self, df: pd.DataFrame):
        features = []
        for column in df.columns:
            for feat in df[column].unique():
                features.append(f'{column}\t{feat}')

        self.tree = DTNode(df)


def parse_dataset(filename: str, training: bool) -> pd.DataFrame:
    df = pd.read_csv(filename, sep='\t')
    if training:
        df.rename(columns = { df.columns[-1] : 'label' }, inplace = True)
        
    return df


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python {} <training file> <test file> <output file>'.format(sys.argv[0]))
        sys.exit(-1)
    
    training_file, test_file, output_file = sys.argv[1:]

    df_train = parse_dataset(training_file, True)
    df_test = parse_dataset(test_file, False)

    model = DecisionTree()
    model.fit(df_train)
