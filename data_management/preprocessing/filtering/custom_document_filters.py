from hojichar import document_filters, Document
from fugashi import Tagger

from os import PathLike
import os
from typing import Any, Union
import re

tagger = Tagger('-Owakati')


class DiscardAdultContentJa(document_filters.NgWordsFilterJa):
    """
    TokenFilter の実装例です.
    日本語の成人向けコンテンツを閾値に応じて排除します.
    """

    def __init__(
        self,
        dict_path: Union[str, PathLike] = document_filters.BASE_PATH / "dict/adult_keywords_ja.txt",
        threshold: float = 0.01,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)
        self.threshold = threshold

    def apply(self, doc: Document) -> Document:
        adult_keywords_pattern = self.keyword_pat
        matches = re.findall(adult_keywords_pattern, doc.text)
        adult_content_count = len(matches)
        total_words_count = len(tagger.parse(doc.text).split())

        if total_words_count > 0 and adult_content_count / total_words_count > self.threshold:
            doc.is_rejected = True

        return doc

class DiscardAdultContentEn(document_filters.NgWordsFilterEn):
    """
    英語の成人向けコンテンツを閾値に応じて排除.
    thresholdの割合を超えていればreject
    """

    def __init__(
        self,
        dict_path: Union[str, PathLike] = document_filters.BASE_PATH / "dict/adult_keywords_en.txt",
        threshold: float = 0.01,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)
        self.threshold = threshold

    def apply(self, doc: Document) -> Document:
        adult_keywords_pattern = self.keyword_pat
        matches = re.findall(adult_keywords_pattern, doc.text)
        adult_content_count = len(matches)
        total_words_count = len(doc.text.split())   # divide by space

        if total_words_count > 0 and adult_content_count / total_words_count > self.threshold:
            doc.is_rejected = True

        return doc
    
class DiscardBadWordEn(document_filters.NgWordsFilterEn):
    """
    英語のbad wordを閾値に応じて排除.
    thresholdの割合を超えていればreject
    """

    def __init__(
        self,
        dict_path: Union[str, PathLike] = os.path.dirname(__file__) + "/bad_keywords_en.txt",
        threshold: float = 0.01,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)
        self.threshold = threshold

    def apply(self, doc: Document) -> Document:
        bad_keywords_pattern = self.keyword_pat
        matches = re.findall(bad_keywords_pattern, doc.text)
        bad_content_count = len(matches)
        total_words_count = len(doc.text.split())   # divide by space

        if total_words_count > 0 and bad_content_count / total_words_count > self.threshold:
            doc.is_rejected = True

        return doc