"""
API module

Contains all REST API endpoints organized by functionality
"""

# Don't import submodules here - let main.py handle errors gracefully
# This allows auth endpoints to load even if chat has import errors

__all__ = ["chat", "auth"]