import sys

if sys.version_info.major == 3:
    try:
        import png3 as png
    except ImportError:
        from . import png3 as png
elif sys.version_info.major == 2:
    import png2 as png
    # use backported functools
    import functools32 as functools
