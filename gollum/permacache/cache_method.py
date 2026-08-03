from gollum.types import GollumRequest


class CacheMethod:
    """
    Produces a cache key for a given request.
    """

    def generate_cache_key(
        self,
        request: GollumRequest
    ):
        pass