import hashlib
import json

from gollum.types import GollumRequest


class CacheMethod:
    """
    Produces a cache key for a given request.
    """

    def generate_cache_key(
        self,
        request: GollumRequest
    ):
        """simple implementation: hash the json-dumped request, plus a salt"""
        json_str = json.dumps(request.chat_completion)
        salt_str = request.metadata.get("gollum.salt", "")
        return hashlib.sha256((json_str + salt_str).encode()).hexdigest()
