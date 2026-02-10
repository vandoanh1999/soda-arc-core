from __future__ import annotations
from typing import Callable, List, Optional, Any
from soda.grid import Grid
from soda.primitives import Primitive


class Anchor:
    __slots__ = ("_fn", "name", "_frozen", "_metadata", "_composition_chain")

    def __init__(self, primitive: Primitive):
        self._fn = primitive
        self.name = primitive.name
        self._frozen = False
        self._metadata = primitive.metadata
        self._composition_chain: List[str] = [primitive.name]

    def freeze(self) -> Anchor:
        self._frozen = True
        return self

    def unfreeze(self) -> Anchor:
        self._frozen = False
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def metadata(self) -> dict:
        return self._metadata.copy()

    @property
    def composition_chain(self) -> List[str]:
        return self._composition_chain.copy()

    def apply(self, grid: Grid) -> Grid:
        return self._fn(grid)

    def compose(self, other: Anchor) -> Anchor:
        def _composed(grid: Grid) -> Grid:
            return self.apply(other.apply(grid))

        composed = Anchor(Primitive(
            name=f"{self.name}_◦_{other.name}",
            fn=_composed,
            metadata={"left": self.metadata, "right": other.metadata}
        ))
        
        composed._composition_chain = other._composition_chain + self._composition_chain
        
        if self._frozen and other._frozen:
            composed.freeze()
        
        return composed

    def __mul__(self, other: Anchor) -> Anchor:
        return self.compose(other)

    def __repr__(self) -> str:
        state = "frozen" if self._frozen else "mutable"
        return f"Anchor({self.name}, {state})"


class AnchorChain:
    __slots__ = ("_anchors", "_name")

    def __init__(self, anchors: List[Anchor], name: Optional[str] = None):
        if not anchors:
            raise ValueError("AnchorChain cannot be empty")
        self._anchors = anchors.copy()
        self._name = name or f"chain_{len(anchors)}"

    @property
    def anchors(self) -> List[Anchor]:
        return self._anchors.copy()

    @property
    def name(self) -> str:
        return self._name

    def apply(self, grid: Grid) -> Grid:
        current = grid
        for anchor in self._anchors:
            current = anchor.apply(current)
        return current

    def freeze_all(self) -> AnchorChain:
        for anchor in self._anchors:
            anchor.freeze()
        return self

    def optimize(self) -> Anchor:
        if len(self._anchors) == 1:
            return self._anchors[0]
        
        result = self._anchors[0]
        for anchor in self._anchors[1:]:
            result = result.compose(anchor)
        
        return result

    def __repr__(self) -> str:
        return f"AnchorChain({self._name}, length={len(self._anchors)})"
