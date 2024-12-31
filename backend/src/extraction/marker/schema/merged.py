from collections import Counter
from typing import Optional

from marker.schema.bbox import BboxElement
from pydantic import BaseModel


class MergedLine(BboxElement):
    text: str
    fonts: list[str]

    def most_common_font(self):
        counter = Counter(self.fonts)
        return counter.most_common(1)[0][0]


class MergedBlock(BboxElement):
    lines: list[MergedLine]
    pnum: int
    block_type: Optional[str]


class FullyMergedBlock(BaseModel):
    text: str
    block_type: str
    pnum: Optional[list[int]] = None
