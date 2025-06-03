"""
ToF Module for MLX7502x sensors
===============================

This module provides Python bindings for Melexis MLX7502x Time-of-Flight sensors.
Supports MLX75026 and MLX75027.
"""

from .mlx7502x import MLX7502x, ToFConfig

__all__ = ['MLX7502x', 'ToFConfig']