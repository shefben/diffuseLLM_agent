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

    def query(
        self, src: str, relation: str | None = None, depth: int = 1
    ) -> dict[str, list[str]]:
        """Breadth-first search for neighbors up to ``depth`` hops.

        If ``relation`` is provided only that edge type is followed, otherwise
        all relations from each node are expanded. The returned mapping uses the
        form ``"node:relation"`` as keys when multiple relations are traversed.
        """
        results: dict[str, list[str]] = {}
        frontier = {src}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                rel_map = self._graph.get(node, {})
                if relation:
                    neighbours = rel_map.get(relation, set())
                    if neighbours:
                        results.setdefault(node, []).extend(sorted(neighbours))
                        next_frontier.update(neighbours)
                else:
                    for rel, neighbours in rel_map.items():
                        key = f"{node}:{rel}"
                        results.setdefault(key, []).extend(sorted(neighbours))
                        next_frontier.update(neighbours)
            frontier = next_frontier
            if not frontier:
                break
        return results
