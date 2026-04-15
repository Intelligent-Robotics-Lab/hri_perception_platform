from abc import ABC, abstractmethod
from typing import Optional


class LatestStore(ABC):
    @abstractmethod
    def get_latest(self):
        pass


class IngestAdapter(ABC):
    @abstractmethod
    def ingest(self, *args, **kwargs):
        pass