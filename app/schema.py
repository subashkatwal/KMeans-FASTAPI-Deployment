from pydantic import BaseModel
from typing import List 
class ClusterInput(BaseModel):
    data: List[float]