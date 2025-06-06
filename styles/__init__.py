import os
from contextlib import contextmanager

STYLES_DIR = os.path.abspath(os.path.dirname(__file__))
DARK_THEME = os.path.join(STYLES_DIR, 'dark-theme.mplstyle')
LIGHT_THEME = os.path.join(STYLES_DIR, 'light-theme.mplstyle')

AVAILABLE_THEMES = {
    'dark': DARK_THEME,
    'light': LIGHT_THEME
}

@contextmanager
def theme(name='dark'):
    """Context manager for temporary style settings"""
    import matplotlib.pyplot as plt
    
    if name not in AVAILABLE_THEMES:
        raise ValueError(f"Theme must be one of {list(AVAILABLE_THEMES.keys())}")
    
    with plt.style.context(AVAILABLE_THEMES[name]):
        yield
