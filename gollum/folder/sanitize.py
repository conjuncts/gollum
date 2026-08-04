def sanitize_dirname(dirname: str):
    """
    Makes dirname a valid filename.
    Only allow alphanumeric characters, underscores, and hyphens.
    """
    return "".join(c for c in dirname if c.isalnum() or c in "_-")