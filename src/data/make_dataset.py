import logging
import multiprocessing as mp
import spacy
import pandas as pd
from settings import RANDOM_STATE

logger = logging.getLogger(__name__)


class HamSpamDataset:
    """A Dataset class for Ham Spam message classification."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self._load()
        self._extract_features()
        self._train_data = None
        self._test_data = None

    def _load(self):
        """Load the dataset from the attribute path."""
        self._data = pd.read_csv(
            self.data_path, delimiter="\t", header=0, names=["label", "text"]
        )

    def get_train_test_split(
        self, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns a train test split of the dataset.

        Args:
            test_size (float, optional): Defaults to 0.2.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
        """
        suffled_data = self._processed_data.sample(
            frac=1, random_state=RANDOM_STATE
        )
        split_index = int(len(suffled_data) * (1 - test_size))

        df_train = suffled_data.iloc[:split_index]
        df_test = suffled_data.iloc[split_index:]

        self._train_data = df_train
        self._test_data = df_test

        return df_train, df_test

    def _extract_features(self):
        nlp = spacy.load("en_core_web_sm")
        texts = self._data["text"]
        labels = self._data["label"]

        features = []

        for doc in nlp.pipe(
            texts,
            n_process=int(0.8 * mp.cpu_count()),
            disable=["tok2vec"],
        ):
            token_count = len(doc)
            char_count = len(doc.text)
            sentences_count = len(list(doc.sents))
            exclam_count = doc.text.count("!")
            question_count = doc.text.count("?")
            uppercase_count = sum(1 for c in doc.text if c.isupper())

            url_count = 0
            pos_counts = {}
            ent_types = {}

            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

                if token.like_url:
                    url_count += 1

            for ent in doc.ents:
                ent_types[ent.label_] = ent_types.get(ent.label_, 0) + 1

            ent_count = len(ent_types)
            avg_tok_len = char_count / token_count if token_count > 0 else 0
            avg_sent_len = (
                token_count / sentences_count if sentences_count > 0 else 0
            )
            upper_ratio = uppercase_count / char_count if char_count > 0 else 0
            money_count = ent_types.get("MONEY", 0)

            feature = {
                "token_count": token_count,
                "char_count": char_count,
                "sentences_count": sentences_count,
                "exclam_count": exclam_count,
                "question_count": question_count,
                "uppercase_count": uppercase_count,
                "url_count": url_count,
                "ent_count": ent_count,
                "avg_tok_len": avg_tok_len,
                "avg_sent_len": avg_sent_len,
                "upper_ratio": upper_ratio,
                "money_count": money_count,
            }

            for pos, count in pos_counts.items():
                feature[f"{pos}_count"] = count

            features.append(feature)
        df = pd.DataFrame(features)
        df["label"] = labels
        df["label"] = df["label"].apply(lambda x: 1 if x == "spam" else 0)

        df.fillna(0, inplace=True)
        self._processed_data = df
