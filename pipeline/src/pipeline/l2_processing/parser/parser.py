import json
from bs4 import BeautifulSoup
from typing import Dict, List, Any
from .extractor import *
from loguru import logger

class ParserV1:
    @staticmethod
    def parse(
        page_source: str,
        tree_depth: int = 4
    ) -> Dict[str, Any]:
        """
        Process the page source and hierarchical structure of the document.

        Parameters:
            name (LiteralString): Name of the document.
            metadata (Dict[LiteralString, LiteralString]): Metadata of the document.
            page_source (str): Page source to process.

        Returns:
            Dict: The root node of the document.
        """

        def _value_node (
            uid: str,
            key: str,
            value: str,
            subitems: List[str]
        ):
            return {
                "uid": uid,
                "key": key,
                "value": value,
                "subitems": subitems
            }

        soup = BeautifulSoup(page_source, features="lxml")
        stack : List[Dict] = []

        # Leave the root node empty for later initialization
        node = _value_node("", "", "", [])
        stack.append(node)

        for element in soup.find_all("div", class_=re.compile(r"^docitem")):
            if element.attrs.get("class") is None:
                continue

            class_ = element.attrs["class"][0]
            extractor = Extractor.get_extractor(class_)

            if extractor is None:
                continue

            if extractor.layer > tree_depth:
                # Instead of creating a node, append text to the current parent
                content = extractor.get_text_only(element)
                if stack[-1]["value"]:
                    stack[-1]["value"] += f" {content}"
                else:
                    stack[-1]["value"] = content
                continue # Skip the stack logic below

            obj = extractor(element)

            if isinstance(obj, str):
                # Ensure not content is missed
                content = extractor.get_text_only(element)
                if stack[-1]["value"]:
                    stack[-1]["value"] += f" {content}"
                else:
                    stack[-1]["value"] = content
                continue

            if extractor.layer - len(stack) > 1:
                logger.error(f"Invalid layer: {stack[-1]} - {obj}")
                continue

            while extractor.layer < len(stack):
                stack.pop()

            stack[-1]["subitems"].append(obj)
            stack.append(obj)
            
        return stack[0]
