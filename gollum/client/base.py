
from gollum.worklist.base import Worklist


class GollumClient:
    """
    A client for interacting with the Gollum system.
    """

    def __init__(self, worklist: Worklist):
        """
        :param worklist: 
        """
        self.worklist = worklist
