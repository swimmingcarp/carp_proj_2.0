"""
Top-level package marker for stock_trading_advisor.

This allows subpackages such as ``stock_trading_advisor.src`` to be imported
when third-party libraries (like numba) try to resolve modules via their full
package path during cache loading.
"""

__all__ = []
