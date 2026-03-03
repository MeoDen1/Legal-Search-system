import re
import ast
import pandas as pd
from loguru import logger

class DataframeProcessor:
    def __init__(self):
        # Remove any record having the pattern
        self.exclude_patterns = [r".*Giải thích từ ngữ.*"]

        # Delete any text in `remove_texts` from the record
        self.remove_texts = [" Phân tích"]

    def _filter(self, df: pd.DataFrame):
        def s1_check(x):
            text = x["input"]
            x["valid"] = True

            for pattern in self.exclude_patterns:
                if re.search(pattern, text):
                    x["valid"] = False
                    break

            if x["valid"]:
                x["input"] = x["input"].strip()
                for remove_text in self.remove_texts:
                    x["input"].replace(remove_text, "")

            return x

        df = df.apply(s1_check, axis=1)
        df = df[df["valid"] == True].drop("valid", axis=1)

        return df

    def _format(self, df: pd.DataFrame):
        depths = []

        def convert_labels(val):
            if isinstance(val, str):
                try:
                    lst = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    lst = [val] # Fallback
            else:
                lst = val if isinstance(val, list) else [val]
            
            depths.append(len(lst))
            return lst

        # Apply conversion and track depths
        df["labels"] = df["labels"].apply(convert_labels)
        
        if not depths:
            return df
            
        min_depth = min(depths)
        
        def split_labels(row):
            lst = row["labels"]
            for i in range(min_depth):
                row[f"label{i}"] = lst[i]
            return row

        df = df.apply(split_labels, axis=1)
        return df.drop(columns=["labels"])

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Number of records before processing: {}".format(len(df)))

        df = self._filter(df)
        df = self._format(df)

        logger.info("Number of records after processing: {}".format(len(df)))
        return df
