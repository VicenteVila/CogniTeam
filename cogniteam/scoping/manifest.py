from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ClassificationInfo(BaseModel):
    domain_key: str = ""
    domain_name: str = ""
    archetype_key: str = ""
    archetype_name: str = ""
    priority: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    domain_rules: List[str] = []
    stack: Dict[str, Any] = {}
    is_primary: bool = True


class TaskManifest(BaseModel):
    manifest_version: str = "1.0"
    status: str = "READY_TO_EXECUTE"
    classification: ClassificationInfo = Field(default_factory=ClassificationInfo)
    secondary_classifications: List[ClassificationInfo] = Field(default_factory=list)
    parameters: Dict[str, Any] = {}
    constraints: List[str] = []
    clarified_task: str = ""
    original_task: str = ""

    def all_classifications(self) -> List[ClassificationInfo]:
        result = [self.classification]
        result.extend(self.secondary_classifications)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
