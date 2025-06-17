class KnowledgeGraph:
    """Simple in-memory knowledge graph for relationships between symbols."""

    def __init__(self) -> None:
        # adjacency map: node -> relation -> set(nodes)
        self._graph: dict[str, dict[str, set[str]]] = {}

    def add(self, src: str, relation: str, dst: str) -> None:
        self._graph.setdefault(src, {}).setdefault(relation, set()).add(dst)

    def neighbors(self, src: str, relation: str) -> list[str]:
        return list(self._graph.get(src, {}).get(relation, []))

    def to_dict(self) -> dict[str, dict[str, list[str]]]:
        return {
            n: {r: list(t) for r, t in rels.items()} for n, rels in self._graph.items()
        }
