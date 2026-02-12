```python
# soda/core.py
from __future__ import annotations
import numpy as np
import hashlib
import heapq
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Callable, Optional, Set, Any
from scipy.stats import entropy
from scipy.ndimage import label, find_objects
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter
import pickle

class ObjectExtractor:
    """Color-invariant object extraction with stable structural hashing."""
    
    @staticmethod
    def extract(grid: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Extract objects with structural hash ID."""
        objects = []
        obj_map = np.zeros(grid.shape, dtype=np.int32)
        obj_id = 1
        
        unique_vals = np.unique(grid)
        for val in unique_vals:
            if val == 0:
                continue
            mask = (grid == val).astype(np.int32)
            labeled, n_components = label(mask)
            
            for comp_id in range(1, n_components + 1):
                comp_mask = (labeled == comp_id)
                slices = find_objects(comp_mask)[0]
                if slices is None:
                    continue
                
                local_mask = comp_mask[slices]
                coords = np.argwhere(local_mask)
                
                # Normalize to canonical form
                canonical = ObjectExtractor._canonicalize(coords)
                struct_hash = hashlib.sha256(canonical.tobytes()).hexdigest()[:16]
                
                bbox = (slices[0].start, slices[1].start, 
                       slices[0].stop, slices[1].stop)
                
                objects.append({
                    'id': struct_hash,
                    'color': int(val),
                    'coords': coords,
                    'bbox': bbox,
                    'canonical': canonical,
                    'slice': slices,
                    'mask': local_mask
                })
                
                obj_map[comp_mask] = obj_id
                obj_id += 1
        
        return objects, obj_map
    
    @staticmethod
    def _canonicalize(coords: np.ndarray) -> np.ndarray:
        """Orientation-invariant canonicalization."""
        coords = coords - coords.min(axis=0)
        
        candidates = []
        for k in range(4):
            rotated = ObjectExtractor._rotate_coords(coords, k)
            candidates.append(rotated)
            candidates.append(ObjectExtractor._flip_coords(rotated))
        
        canonical = min(candidates, key=lambda x: (x[:, 0].sum(), x[:, 1].sum(), tuple(x.flatten())))
        return canonical
    
    @staticmethod
    def _rotate_coords(coords: np.ndarray, k: int) -> np.ndarray:
        for _ in range(k):
            coords = np.column_stack([coords[:, 1], -coords[:, 0]])
        coords = coords - coords.min(axis=0)
        return coords
    
    @staticmethod
    def _flip_coords(coords: np.ndarray) -> np.ndarray:
        coords = np.column_stack([-coords[:, 0], coords[:, 1]])
        coords = coords - coords.min(axis=0)
        return coords

class ObjectGraph:
    """Graph representation with spatial relations."""
    
    def __init__(self, objects: List[Dict], grid_shape: Tuple[int, int]):
        self.objects = objects
        self.grid_shape = grid_shape
        self.edges = self._compute_edges()
    
    def _compute_edges(self) -> List[Tuple[str, str, str]]:
        """Compute spatial relations between objects."""
        edges = []
        n = len(self.objects)
        
        for i in range(n):
            for j in range(i + 1, n):
                obj_i, obj_j = self.objects[i], self.objects[j]
                relations = self._compute_relations(obj_i, obj_j)
                for rel in relations:
                    edges.append((obj_i['id'], obj_j['id'], rel))
        
        return edges
    
    def _compute_relations(self, obj_i: Dict, obj_j: Dict) -> List[str]:
        """Encode spatial relations."""
        relations = []
        
        bi = obj_i['bbox']
        bj = obj_j['bbox']
        
        # Inside
        if bi[0] >= bj[0] and bi[1] >= bj[1] and bi[2] <= bj[2] and bi[3] <= bj[3]:
            relations.append('inside')
        elif bj[0] >= bi[0] and bj[1] >= bi[1] and bj[2] <= bi[2] and bj[3] <= bi[3]:
            relations.append('contains')
        
        # Adjacent
        if abs(bi[2] - bj[0]) <= 1 or abs(bj[2] - bi[0]) <= 1:
            relations.append('adjacent_v')
        if abs(bi[3] - bj[1]) <= 1 or abs(bj[3] - bi[1]) <= 1:
            relations.append('adjacent_h')
        
        # Aligned
        if bi[0] == bj[0] or bi[2] == bj[2]:
            relations.append('aligned_v')
        if bi[1] == bj[1] or bi[3] == bj[3]:
            relations.append('aligned_h')
        
        # Symmetric
        center_i = ((bi[0] + bi[2]) / 2, (bi[1] + bi[3]) / 2)
        center_j = ((bj[0] + bj[2]) / 2, (bj[1] + bj[3]) / 2)
        if abs(center_i[0] - center_j[0]) < 0.5:
            relations.append('symmetric_v')
        if abs(center_i[1] - center_j[1]) < 0.5:
            relations.append('symmetric_h')
        
        return relations if relations else ['disjoint']
    
    def reconstruct(self) -> np.ndarray:
        """Lossless reconstruction from object graph."""
        grid = np.zeros(self.grid_shape, dtype=np.int32)
        
        for obj in self.objects:
            slices = obj['slice']
            mask = obj['mask']
            grid[slices][mask] = obj['color']
        
        return grid
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to structural embedding input."""
        n_obj = len(self.objects)
        
        # Node features: [canonical_shape_hash, bbox_normalized]
        node_features = []
        for obj in self.objects:
            shape_feat = hash(obj['id']) % 1000 / 1000.0
            bbox_feat = np.array([
                obj['bbox'][0] / self.grid_shape[0],
                obj['bbox'][1] / self.grid_shape[1],
                (obj['bbox'][2] - obj['bbox'][0]) / self.grid_shape[0],
                (obj['bbox'][3] - obj['bbox'][1]) / self.grid_shape[1]
            ])
            node_features.append(np.concatenate([[shape_feat], bbox_feat]))
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Edge features
        edge_index = []
        edge_attr = []
        relation_vocab = ['inside', 'contains', 'adjacent_v', 'adjacent_h', 
                         'aligned_v', 'aligned_h', 'symmetric_v', 'symmetric_h', 'disjoint']
        
        for i, obj_i in enumerate(self.objects):
            for j, obj_j in enumerate(self.objects):
                if i == j:
                    continue
                relations = self._compute_relations(obj_i, obj_j)
                edge_index.append([i, j])
                edge_vec = [1.0 if r in relations else 0.0 for r in relation_vocab]
                edge_attr.append(edge_vec)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).T if edge_index else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.zeros((0, len(relation_vocab)), dtype=torch.float32)
        
        return node_features, edge_index, edge_attr

class GraphEncoder(nn.Module):
    """Structure-aware encoder for object graphs."""
    
    def __init__(self, node_dim=5, edge_dim=9, hidden_dim=64, embed_dim=32):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        self.conv1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.pool = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, node_features, edge_index, edge_attr):
        x = F.relu(self.node_encoder(node_features))
        
        if edge_index.shape[1] > 0:
            edge_feat = F.relu(self.edge_encoder(edge_attr))
            
            for conv in [self.conv1, self.conv2]:
                messages = []
                for k in range(edge_index.shape[1]):
                    src, dst = edge_index[0, k], edge_index[1, k]
                    msg = torch.cat([x[src], edge_feat[k]])
                    messages.append((dst.item(), msg))
                
                new_x = x.clone()
                for dst, msg in messages:
                    new_x[dst] = new_x[dst] + F.relu(conv(msg))
                x = new_x
        
        graph_embed = torch.mean(x, dim=0)
        return self.pool(graph_embed)

class MacroLearner:
    """Primitive auto-composition via frequent pattern mining."""
    
    def __init__(self, min_support=2):
        self.min_support = min_support
        self.macros: Dict[str, List[str]] = {}
        self.macro_stats: Dict[str, int] = defaultdict(int)
    
    def mine_sequences(self, solutions: List[List[str]]):
        """Mine frequent primitive sequences."""
        seq_counter = Counter()
        
        for sol in solutions:
            for length in range(2, min(6, len(sol) + 1)):
                for i in range(len(sol) - length + 1):
                    subseq = tuple(sol[i:i+length])
                    seq_counter[subseq] += 1
        
        for seq, count in seq_counter.items():
            if count >= self.min_support:
                macro_name = f"macro_{'_'.join(seq[:2])}"
                self.macros[macro_name] = list(seq)
                self.macro_stats[macro_name] = count
    
    def expand_macro(self, macro_name: str) -> List[str]:
        """Expand macro to primitive sequence."""
        return self.macros.get(macro_name, [macro_name])
    
    def get_primitives(self, base_primitives: Dict) -> Dict:
        """Combine base primitives with learned macros."""
        combined = base_primitives.copy()
        
        for macro_name, seq in self.macros.items():
            def make_macro(sequence):
                def macro_fn(d):
                    result = d
                    for prim in sequence:
                        if prim in base_primitives:
                            result = base_primitives[prim](result)
                    return result
                return macro_fn
            
            combined[macro_name] = make_macro(seq)
        
        return combined

class MemoryBank:
    """Cross-task memory with retrieval and continual learning."""
    
    def __init__(self, embed_dim=32):
        self.encoder = GraphEncoder(embed_dim=embed_dim)
        self.memory: List[Dict[str, Any]] = []
        self.primitive_stats = defaultdict(lambda: {'success': 0, 'total': 0})
        self.task_embeddings: List[torch.Tensor] = []
        
    def store(self, obj_graph: ObjectGraph, solution: List[str], success: bool):
        """Store task instance."""
        node_feat, edge_idx, edge_attr = obj_graph.to_tensor()
        
        with torch.no_grad():
            embed = self.encoder(node_feat, edge_idx, edge_attr)
        
        self.memory.append({
            'graph': obj_graph,
            'solution': solution,
            'embedding': embed,
            'success': success
        })
        
        self.task_embeddings.append(embed)
        
        for prim in solution:
            self.primitive_stats[prim]['total'] += 1
            if success:
                self.primitive_stats[prim]['success'] += 1
    
    def retrieve(self, query_graph: ObjectGraph, top_k=5) -> List[Dict]:
        """Similarity-based retrieval."""
        if not self.memory:
            return []
        
        node_feat, edge_idx, edge_attr = query_graph.to_tensor()
        
        with torch.no_grad():
            query_embed = self.encoder(node_feat, edge_idx, edge_attr)
        
        similarities = []
        for entry in self.memory:
            sim = F.cosine_similarity(query_embed.unsqueeze(0), 
                                     entry['embedding'].unsqueeze(0)).item()
            similarities.append((sim, entry))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [entry for _, entry in similarities[:top_k]]
    
    def get_primitive_weights(self) -> Dict[str, float]:
        """Adaptive primitive weighting."""
        weights = {}
        for prim, stats in self.primitive_stats.items():
            if stats['total'] > 0:
                weights[prim] = stats['success'] / stats['total']
            else:
                weights[prim] = 0.5
        return weights
    
    def train_contrastive(self, positive_pairs: List[Tuple[ObjectGraph, ObjectGraph]], 
                         negative_pairs: List[Tuple[ObjectGraph, ObjectGraph]], epochs=10):
        """Contrastive training for encoder."""
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for g1, g2 in positive_pairs:
                nf1, ei1, ea1 = g1.to_tensor()
                nf2, ei2, ea2 = g2.to_tensor()
                
                e1 = self.encoder(nf1, ei1, ea1)
                e2 = self.encoder(nf2, ei2, ea2)
                
                pos_loss = 1 - F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
                
                optimizer.zero_grad()
                pos_loss.backward()
                optimizer.step()
                
                total_loss += pos_loss.item()
            
            for g1, g2 in negative_pairs:
                nf1, ei1, ea1 = g1.to_tensor()
                nf2, ei2, ea2 = g2.to_tensor()
                
                e1 = self.encoder(nf1, ei1, ea1)
                e2 = self.encoder(nf2, ei2, ea2)
                
                neg_loss = F.relu(F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)) - 0.5)
                
                optimizer.zero_grad()
                neg_loss.backward()
                optimizer.step()
                
                total_loss += neg_loss.item()

class LearnedHeuristic:
    """Embedding-based heuristic estimation."""
    
    def __init__(self, encoder: GraphEncoder):
        self.encoder = encoder
    
    def estimate(self, current_graph: ObjectGraph, target_graph: ObjectGraph) -> float:
        """Structure-aware distance estimation."""
        with torch.no_grad():
            nf1, ei1, ea1 = current_graph.to_tensor()
            nf2, ei2, ea2 = target_graph.to_tensor()
            
            e1 = self.encoder(nf1, ei1, ea1)
            e2 = self.encoder(nf2, ei2, ea2)
            
            dist = torch.norm(e1 - e2, p=2).item()
        
        return dist * 10.0

@dataclass(order=True)
class AGISearchNode:
    f: float
    g: float = field(compare=False)
    graph: ObjectGraph = field(compare=False)
    path: Tuple[str, ...] = field(compare=False)

class AGISolver:
    """AGI-approaching solver with all required components."""
    
    def __init__(self, primitives: Dict[str, Callable[[np.ndarray], np.ndarray]]):
        self.base_primitives = primitives
        self.memory_bank = MemoryBank()
        self.macro_learner = MacroLearner()
        self.heuristic = LearnedHeuristic(self.memory_bank.encoder)
        
    def solve(self, start_data: np.ndarray, target_data: np.ndarray, max_depth: int = 10) -> Optional[List[str]]:
        # Extract objects
        start_objects, _ = ObjectExtractor.extract(start_data)
        target_objects, _ = ObjectExtractor.extract(target_data)
        
        start_graph = ObjectGraph(start_objects, start_data.shape)
        target_graph = ObjectGraph(target_objects, target_data.shape)
        
        # Verify lossless reconstruction
        if not np.array_equal(start_graph.reconstruct(), start_data):
            return None
        if not np.array_equal(target_graph.reconstruct(), target_data):
            return None
        
        # Retrieve similar tasks
        similar = self.memory_bank.retrieve(start_graph, top_k=3)
        
        # Get adaptive primitives
        primitives = self.macro_learner.get_primitives(self.base_primitives)
        weights = self.memory_bank.get_primitive_weights()
        
        # A* search with learned heuristic
        start_h = self.heuristic.estimate(start_graph, target_graph)
        frontier = [AGISearchNode(start_h, 0.0, start_graph, ())]
        visited: Dict[str, float] = {}
        
        start_hash = self._graph_hash(start_graph)
        visited[start_hash] = 0.0
        
        while frontier:
            node = heapq.heappop(frontier)
            
            # Goal test
            if self._graphs_equal(node.graph, target_graph):
                solution = list(node.path)
                self.memory_bank.store(start_graph, solution, True)
                self.macro_learner.mine_sequences([solution])
                return solution
            
            if len(node.path) >= max_depth:
                continue
            
            current_grid = node.graph.reconstruct()
            
            for p_name, p_fn in primitives.items():
                try:
                    next_data = p_fn(current_grid)
                    
                    if np.array_equal(next_data, current_grid):
                        continue
                    
                    next_objects, _ = ObjectExtractor.extract(next_data)
                    next_graph = ObjectGraph(next_objects, next_data.shape)
                    
                    if not np.array_equal(next_graph.reconstruct(), next_data):
                        continue
                    
                    next_hash = self._graph_hash(next_graph)
                    
                    weight = weights.get(p_name, 0.5)
                    move_cost = (1.0 / (weight + 0.1)) + (0.1 * len(p_name))
                    new_g = node.g + move_cost
                    
                    if next_hash not in visited or new_g < visited[next_hash]:
                        visited[next_hash] = new_g
                        h = self.heuristic.estimate(next_graph, target_graph)
                        heapq.heappush(frontier, AGISearchNode(new_g + h, new_g, next_graph, node.path + (p_name,)))
                
                except:
                    continue
        
        self.memory_bank.store(start_graph, [], False)
        return None
    
    def _graph_hash(self, graph: ObjectGraph) -> str:
        """Stable hash for object graph."""
        obj_hashes = sorted([obj['id'] for obj in graph.objects])
        edge_hashes = sorted([f"{e[0]}_{e[1]}_{e[2]}" for e in graph.edges])
        combined = ''.join(obj_hashes + edge_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _graphs_equal(self, g1: ObjectGraph, g2: ObjectGraph) -> bool:
        """Check structural equality."""
        return self._graph_hash(g1) == self._graph_hash(g2)

def run_agi_engine(input_arr: np.ndarray, output_arr: np.ndarray):
    """AGI-approaching ARC solver."""
    prims = {
        "rot90": lambda d: np.rot90(d, k=1),
        "rot180": lambda d: np.rot90(d, k=2),
        "rot270": lambda d: np.rot90(d, k=3),
        "flip_h": lambda d: np.fliplr(d),
        "flip_v": lambda d: np.flipud(d),
        "transpose": lambda d: d.T,
        "shift_u": lambda d: np.roll(d, -1, axis=0),
        "shift_d": lambda d: np.roll(d, 1, axis=0),
        "shift_l": lambda d: np.roll(d, -1, axis=1),
        "shift_r": lambda d: np.roll(d, 1, axis=1),
    }