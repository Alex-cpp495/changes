"""
Web API路由模块
"""

from . import detection, feedback, monitoring, reports, config, auth

__all__ = [
    'detection',
    'feedback', 
    'monitoring',
    'reports',
    'config',
    'auth'
] 