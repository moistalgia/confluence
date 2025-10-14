"""
One Trade Bot - Core Package
🎯 Core components for "One Good Trade Per Day" system

Components:
- DataProvider: Market data fetching with strict validation
- WorkflowEngine: Daily 5-filter orchestration system
- PositionManager: 1% risk position management (coming soon)
"""

from .data_provider import DataProvider, TechnicalIndicators

__all__ = [
    'DataProvider',
    'TechnicalIndicators'
]