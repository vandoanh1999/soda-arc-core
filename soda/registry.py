# soda/registry.py
from __future__ import annotations
import inspect
import functools
import threading
from typing import Dict, List, Callable, Optional, Any, Type, Union, get_type_hints
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from enum import Enum
import warnings
import json
import pickle
from pathlib import Path

class ComponentType(Enum):
    PRIMITIVE = "primitive"
    ENCODER = "encoder"
    POLICY = "policy"
    VALUE = "value"
    MEMORY = "memory"
    ABSTRACTOR = "abstractor"
    OPTIMIZER = "optimizer"
    INDUCTOR = "inductor"
    METRIC = "metric"
    TRANSFORM = "transform"
    EXTRACTOR = "extractor"
    SOLVER = "solver"

class PrimitiveCategory(Enum):
    SPATIAL = "spatial"
    COLOR = "color"
    TOPOLOGY = "topology"
    COMPOSITE = "composite"
    LEARNED = "learned"

@dataclass
class ComponentMetadata:
    name: str
    component_type: ComponentType
    version: str
    author: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    category: Optional[PrimitiveCategory] = None
    input_spec: Optional[Dict] = None
    output_spec: Optional[Dict] = None
    complexity: float = 1.0
    priority: int = 0
    enabled: bool = True

@dataclass
class RegistrationRecord:
    metadata: ComponentMetadata
    component: Any
    timestamp: float
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    last_used: Optional[float] = None

class Registry:
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        with self._lock:
            if not self._initialized:
                self._components: Dict[ComponentType, Dict[str, RegistrationRecord]] = defaultdict(dict)
                self._aliases: Dict[str, tuple[ComponentType, str]] = {}
                self._hooks: Dict[str, List[Callable]] = defaultdict(list)
                self._stats: Dict[str, Dict[str, float]] = defaultdict(dict)
                self._config: Dict[str, Any] = {}
                self._initialized = True
    
    def register(
        self,
        name: str,
        component: Any,
        component_type: ComponentType,
        version: str = "1.0.0",
        author: str = "Claude",
        description: str = "",
        category: Optional[PrimitiveCategory] = None,
        dependencies: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        complexity: float = 1.0,
        priority: int = 0,
        override: bool = False
    ) -> None:
        with self._lock:
            if not override and name in self._components[component_type]:
                raise ValueError(f"Component '{name}' already registered in {component_type.value}")
            
            metadata = ComponentMetadata(
                name=name,
                component_type=component_type,
                version=version,
                author=author,
                description=description,
                dependencies=dependencies or [],
                tags=tags or [],
                category=category,
                complexity=complexity,
                priority=priority
            )
            
            if callable(component):
                try:
                    sig = inspect.signature(component)
                    metadata.input_spec = {
                        param.name: str(param.annotation) 
                        for param in sig.parameters.values()
                    }
                    metadata.output_spec = {"return": str(sig.return_annotation)}
                except Exception:
                    pass
            
            record = RegistrationRecord(
                metadata=metadata,
                component=component,
                timestamp=self._get_timestamp()
            )
            
            self._components[component_type][name] = record
            self._trigger_hook("on_register", component_type, name, record)
    
    def get(
        self,
        name: str,
        component_type: Optional[ComponentType] = None,
        default: Any = None
    ) -> Optional[Any]:
        with self._lock:
            if name in self._aliases:
                alias_type, alias_name = self._aliases[name]
                if component_type is None or component_type == alias_type:
                    component_type = alias_type
                    name = alias_name
            
            if component_type:
                record = self._components[component_type].get(name)
                if record and record.metadata.enabled:
                    self._update_usage(component_type, name)
                    return record.component
                return default
            
            for ctype in ComponentType:
                record = self._components[ctype].get(name)
                if record and record.metadata.enabled:
                    self._update_usage(ctype, name)
                    return record.component
            
            return default
    
    def get_all(
        self,
        component_type: ComponentType,
        category: Optional[PrimitiveCategory] = None,
        tags: Optional[List[str]] = None,
        enabled_only: bool = True
    ) -> Dict[str, Any]:
        with self._lock:
            results = {}
            
            for name, record in self._components[component_type].items():
                if enabled_only and not record.metadata.enabled:
                    continue
                
                if category and record.metadata.category != category:
                    continue
                
                if tags:
                    if not all(tag in record.metadata.tags for tag in tags):
                        continue
                
                results[name] = record.component
            
            return results
    
    def unregister(self, name: str, component_type: ComponentType) -> bool:
        with self._lock:
            if name in self._components[component_type]:
                record = self._components[component_type].pop(name)
                self._trigger_hook("on_unregister", component_type, name, record)
                return True
            return False
    
    def alias(self, alias: str, target_name: str, component_type: ComponentType) -> None:
        with self._lock:
            if target_name not in self._components[component_type]:
                raise ValueError(f"Target component '{target_name}' not found")
            self._aliases[alias] = (component_type, target_name)
    
    def enable(self, name: str, component_type: ComponentType) -> None:
        with self._lock:
            if name in self._components[component_type]:
                self._components[component_type][name].metadata.enabled = True
    
    def disable(self, name: str, component_type: ComponentType) -> None:
        with self._lock:
            if name in self._components[component_type]:
                self._components[component_type][name].metadata.enabled = False
    
    def get_metadata(self, name: str, component_type: ComponentType) -> Optional[ComponentMetadata]:
        with self._lock:
            record = self._components[component_type].get(name)
            return record.metadata if record else None
    
    def get_stats(self, name: str, component_type: ComponentType) -> Dict[str, float]:
        with self._lock:
            record = self._components[component_type].get(name)
            if not record:
                return {}
            
            return {
                "usage_count": record.usage_count,
                "success_count": record.success_count,
                "failure_count": record.failure_count,
                "success_rate": record.success_count / max(record.usage_count, 1),
                "avg_execution_time": record.avg_execution_time,
                "last_used": record.last_used or 0.0
            }
    
    def update_stats(
        self,
        name: str,
        component_type: ComponentType,
        success: bool,
        execution_time: Optional[float] = None
    ) -> None:
        with self._lock:
            if name not in self._components[component_type]:
                return
            
            record = self._components[component_type][name]
            
            if success:
                record.success_count += 1
            else:
                record.failure_count += 1
            
            if execution_time is not None:
                alpha = 0.1
                if record.avg_execution_time == 0.0:
                    record.avg_execution_time = execution_time
                else:
                    record.avg_execution_time = (
                        alpha * execution_time + 
                        (1 - alpha) * record.avg_execution_time
                    )
            
            record.last_used = self._get_timestamp()
    
    def list_components(
        self,
        component_type: Optional[ComponentType] = None
    ) -> List[str]:
        with self._lock:
            if component_type:
                return list(self._components[component_type].keys())
            
            all_names = []
            for ctype in ComponentType:
                all_names.extend([
                    f"{ctype.value}::{name}" 
                    for name in self._components[ctype].keys()
                ])
            return all_names
    
    def search(
        self,
        query: str,
        component_type: Optional[ComponentType] = None,
        search_fields: Optional[List[str]] = None
    ) -> List[tuple[ComponentType, str, ComponentMetadata]]:
        search_fields = search_fields or ["name", "description", "tags"]
        results = []
        
        with self._lock:
            types_to_search = [component_type] if component_type else list(ComponentType)
            
            for ctype in types_to_search:
                for name, record in self._components[ctype].items():
                    meta = record.metadata
                    
                    if "name" in search_fields and query.lower() in name.lower():
                        results.append((ctype, name, meta))
                        continue
                    
                    if "description" in search_fields and query.lower() in meta.description.lower():
                        results.append((ctype, name, meta))
                        continue
                    
                    if "tags" in search_fields and any(query.lower() in tag.lower() for tag in meta.tags):
                        results.append((ctype, name, meta))
                        continue
        
        return results
    
    def get_sorted_by_priority(
        self,
        component_type: ComponentType,
        descending: bool = True
    ) -> List[tuple[str, Any]]:
        with self._lock:
            items = [
                (name, record.component, record.metadata.priority)
                for name, record in self._components[component_type].items()
                if record.metadata.enabled
            ]
            
            items.sort(key=lambda x: x[2], reverse=descending)
            return [(name, comp) for name, comp, _ in items]
    
    def get_sorted_by_performance(
        self,
        component_type: ComponentType,
        metric: str = "success_rate"
    ) -> List[tuple[str, Any, float]]:
        with self._lock:
            items = []
            
            for name, record in self._components[component_type].items():
                if not record.metadata.enabled:
                    continue
                
                stats = self.get_stats(name, component_type)
                metric_value = stats.get(metric, 0.0)
                items.append((name, record.component, metric_value))
            
            items.sort(key=lambda x: x[2], reverse=True)
            return items
    
    def add_hook(self, event: str, callback: Callable) -> None:
        with self._lock:
            self._hooks[event].append(callback)
    
    def remove_hook(self, event: str, callback: Callable) -> None:
        with self._lock:
            if event in self._hooks and callback in self._hooks[event]:
                self._hooks[event].remove(callback)
    
    def _trigger_hook(self, event: str, *args, **kwargs) -> None:
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                warnings.warn(f"Hook {callback} failed: {e}")
    
    def _update_usage(self, component_type: ComponentType, name: str) -> None:
        if name in self._components[component_type]:
            self._components[component_type][name].usage_count += 1
            self._components[component_type][name].last_used = self._get_timestamp()
    
    def _get_timestamp(self) -> float:
        import time
        return time.time()
    
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            state = {
                "components": {},
                "aliases": self._aliases,
                "stats": self._stats,
                "config": self._config
            }
            
            for ctype in ComponentType:
                state["components"][ctype.value] = {}
                for name, record in self._components[ctype].items():
                    state["components"][ctype.value][name] = {
                        "metadata": {
                            "name": record.metadata.name,
                            "version": record.metadata.version,
                            "author": record.metadata.author,
                            "description": record.metadata.description,
                            "tags": record.metadata.tags,
                            "complexity": record.metadata.complexity,
                            "priority": record.metadata.priority,
                            "enabled": record.metadata.enabled
                        },
                        "stats": {
                            "usage_count": record.usage_count,
                            "success_count": record.success_count,
                            "failure_count": record.failure_count,
                            "avg_execution_time": record.avg_execution_time
                        }
                    }
            
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
    
    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {path}")
        
        with self._lock:
            with open(path, "r") as f:
                state = json.load(f)
            
            self._aliases = state.get("aliases", {})
            self._stats = state.get("stats", {})
            self._config = state.get("config", {})
    
    def clear(self, component_type: Optional[ComponentType] = None) -> None:
        with self._lock:
            if component_type:
                self._components[component_type].clear()
            else:
                self._components.clear()
                self._aliases.clear()
    
    def get_config(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        with self._lock:
            self._config[key] = value
    
    def validate_dependencies(self, name: str, component_type: ComponentType) -> bool:
        with self._lock:
            record = self._components[component_type].get(name)
            if not record:
                return False
            
            for dep in record.metadata.dependencies:
                if "::" in dep:
                    dep_type_str, dep_name = dep.split("::", 1)
                    dep_type = ComponentType(dep_type_str)
                    if dep_name not in self._components[dep_type]:
                        return False
                else:
                    found = False
                    for ctype in ComponentType:
                        if dep in self._components[ctype]:
                            found = True
                            break
                    if not found:
                        return False
            
            return True
    
    def export_metrics(self) -> Dict[str, Any]:
        with self._lock:
            metrics = {
                "total_components": sum(len(comps) for comps in self._components.values()),
                "by_type": {},
                "top_performers": {},
                "statistics": {}
            }
            
            for ctype in ComponentType:
                count = len(self._components[ctype])
                enabled = sum(1 for r in self._components[ctype].values() if r.metadata.enabled)
                
                metrics["by_type"][ctype.value] = {
                    "total": count,
                    "enabled": enabled,
                    "disabled": count - enabled
                }
                
                if count > 0:
                    top = self.get_sorted_by_performance(ctype, "success_rate")[:5]
                    metrics["top_performers"][ctype.value] = [
                        {"name": name, "success_rate": rate} 
                        for name, _, rate in top
                    ]
            
            return metrics
    
    def __repr__(self) -> str:
        total = sum(len(comps) for comps in self._components.values())
        return f"<Registry: {total} components across {len(ComponentType)} types>"

def register(
    component_type: ComponentType,
    name: Optional[str] = None,
    version: str = "1.0.0",
    author: str = "Claude",
    description: str = "",
    category: Optional[PrimitiveCategory] = None,
    tags: Optional[List[str]] = None,
    complexity: float = 1.0,
    priority: int = 0
):
    def decorator(func_or_class):
        nonlocal name
        if name is None:
            name = func_or_class.__name__
        
        registry = Registry()
        registry.register(
            name=name,
            component=func_or_class,
            component_type=component_type,
            version=version,
            author=author,
            description=description,
            category=category,
            tags=tags,
            complexity=complexity,
            priority=priority
        )
        
        @functools.wraps(func_or_class)
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            try:
                result = func_or_class(*args, **kwargs)
                elapsed = time.time() - start
                registry.update_stats(name, component_type, True, elapsed)
                return result
            except Exception as e:
                elapsed = time.time() - start
                registry.update_stats(name, component_type, False, elapsed)
                raise
        
        if inspect.isclass(func_or_class):
            return func_or_class
        
        return wrapper
    
    return decorator

_global_registry = Registry()

def get_registry() -> Registry:
    return _global_registry

def reset_registry() -> None:
    global _global_registry
    Registry._instance = None
    _global_registry = Registry()