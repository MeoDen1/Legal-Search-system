import uuid
import re
from bs4 import Tag
from abc import ABC, abstractmethod
from typing import List, Dict

def value_node (
    key: str,
    value: str,
    subitems: List[str]
):
    return {
        "uid": str(uuid.uuid4()),
        "key": key,
        "value": value,
        "subitems": subitems
    }

class ExtractorInterface(ABC):
    @abstractmethod
    def __init__(self):
        self.layer = None

    @abstractmethod
    def __call__(self, element: Tag) -> Dict | str:
        pass

    def process_text(self, text: str):
        text = text.replace(u'\xa0', u' ')
        # ADHOC
        text = text.replace("Đang theo dõi", "")
        text = text.replace(" Phân tích", "")

        text = text.strip("“\n ")
        while text.find("  ") != -1:
            text = text.replace("  ", " ")

        return text

    def get_text_only(self, element: Tag):
        return self.process_text(element.get_text(separator=" ", strip=True))

class Extractor:
    @staticmethod
    def get_extractor(class_: str):
        extractors = {
            "docitem-2": L1Extractor,
            "docitem-5": L2Extractor,
            "docitem-11": L3Extractor,
            "docitem-12": L4Extractor
        }

        extractor_class = extractors.get(class_)
        if extractor_class:
            return extractor_class()
        
        return None

class L1Extractor(ExtractorInterface):
    def __init__(self):
        self.layer = 1

    def __call__(self, element: Tag) -> Dict:
        text: str = self.process_text(element.get_text(separator="\n", strip=True))
        
        _break = text.find("\n")
        key = self.process_text(text[:_break])
        value = self.process_text(text[_break+1:])

        return value_node(
            key=key,
            value=value,
            subitems=[]
        )

class L2Extractor(ExtractorInterface):
    def __init__(self):
        self.layer = 2
        self.pattern = r"^.* ([0-9]+)\s*\.([^\n]*)"

    def __call__(self, element: Tag) -> Dict | str:
        text: str = self.process_text(element.get_text(separator=" ", strip=True))
        text = text.replace("\n", "")
        groups = re.findall(self.pattern, text)

        if len(groups) == 0 or len(groups[0]) < 2:
            return f"L2: Can not extract str: {text}"
        
        key = groups[0][0].strip()
        value = groups[0][1].strip()

        return value_node(
            key=f"Điều {key}",
            value=value,
            subitems=[]
        )

class L3Extractor(ExtractorInterface):
    def __init__(self):
        self.layer = 3
        self.pattern= r"([0-9]+)\s*\.([^\n]*)"

    def __call__(self, element: Tag) -> Dict | str:
        text: str = self.process_text(element.get_text(separator=" ", strip=True))
        text = text.replace("\n", "")
        groups = re.findall(self.pattern, text)

        if len(groups) == 0 or len(groups[0]) < 2:
            return f"L3: Can not extract str: {text}"
        
        key = groups[0][0].strip()
        value = groups[0][1].strip()

        return value_node(
            key=f"Khoản {key}",
            value=value,
            subitems=[]
        )

class L4Extractor(ExtractorInterface):
    def __init__(self):
        self.layer = 4
        self.pattern= r"([a-zđ]+)\)([^\n]*)"

    def __call__(self, element: Tag) -> Dict | str:
        text: str = self.process_text(element.get_text(separator=" ", strip=True))
        text = text.replace("\n", "")
        groups = re.findall(self.pattern, text)

        if len(groups) == 0 or len(groups[0]) < 2:
            return f"L4: Can not extract str: {text}"
        
        key = groups[0][0].strip()
        value = groups[0][1].strip()

        return value_node(
            key=f"Điểm {key}",
            value=value,
            subitems=[]
        )
