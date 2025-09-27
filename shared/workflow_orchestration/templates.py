"""
Workflow Template System for Q2 Platform.

Provides reusable workflow templates with parameterization and inheritance.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from .engine import WorkflowDefinition, WorkflowStep, StepType

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Parameter data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    ANY = "any"


@dataclass
class ParameterDefinition:
    """Template parameter definition."""
    name: str
    type: ParameterType
    description: str = ""
    default_value: Any = None
    required: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)
    
    def validate(self, value: Any) -> bool:
        """Validate a parameter value."""
        if value is None and self.required:
            return False
        
        if value is None and not self.required:
            return True
        
        # Type validation
        if self.type == ParameterType.STRING and not isinstance(value, str):
            return False
        elif self.type == ParameterType.INTEGER and not isinstance(value, int):
            return False
        elif self.type == ParameterType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif self.type == ParameterType.BOOLEAN and not isinstance(value, bool):
            return False
        elif self.type == ParameterType.LIST and not isinstance(value, list):
            return False
        elif self.type == ParameterType.DICT and not isinstance(value, dict):
            return False
        
        # Apply validation rules
        for rule, rule_value in self.validation_rules.items():
            if rule == "min_length" and len(str(value)) < rule_value:
                return False
            elif rule == "max_length" and len(str(value)) > rule_value:
                return False
            elif rule == "min_value" and value < rule_value:
                return False
            elif rule == "max_value" and value > rule_value:
                return False
            elif rule == "regex" and self.type == ParameterType.STRING:
                import re
                if not re.match(rule_value, value):
                    return False
        
        return True


@dataclass
class WorkflowTemplate:
    """Reusable workflow template."""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    category: str = "general"
    tags: Set[str] = field(default_factory=set)
    parameters: List[ParameterDefinition] = field(default_factory=list)
    steps_template: List[Dict[str, Any]] = field(default_factory=list)
    parent_template_id: Optional[str] = None  # For template inheritance
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    
    def validate_parameters(self, values: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate parameter values against definitions."""
        errors = {}
        
        # Check required parameters
        param_names = {p.name for p in self.parameters}
        for param in self.parameters:
            if param.required and param.name not in values:
                if param.default_value is None:
                    errors.setdefault(param.name, []).append("Required parameter missing")
        
        # Validate provided parameters
        for name, value in values.items():
            param = next((p for p in self.parameters if p.name == name), None)
            if param and not param.validate(value):
                errors.setdefault(name, []).append("Parameter validation failed")
        
        return errors
    
    def instantiate(self, parameters: Dict[str, Any]) -> WorkflowDefinition:
        """Create a workflow instance from the template."""
        # Validate parameters
        validation_errors = self.validate_parameters(parameters)
        if validation_errors:
            raise ValueError(f"Parameter validation failed: {validation_errors}")
        
        # Apply defaults for missing parameters
        final_params = {}
        for param in self.parameters:
            if param.name in parameters:
                final_params[param.name] = parameters[param.name]
            elif param.default_value is not None:
                final_params[param.name] = param.default_value
        
        # Generate workflow steps from template
        steps = []
        for step_template in self.steps_template:
            step = self._instantiate_step(step_template, final_params)
            steps.append(step)
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            id=str(uuid4()),
            name=f"{self.name}_instance",
            description=f"Instance of template {self.name}",
            steps=steps,
            global_parameters=dict(self.global_parameters),
            tags=self.tags.copy(),
            version=self.version
        )
        
        return workflow
    
    def _instantiate_step(self, step_template: Dict[str, Any], 
                         parameters: Dict[str, Any]) -> WorkflowStep:
        """Create a workflow step from template."""
        # Substitute parameters in step template
        substituted = self._substitute_parameters(step_template, parameters)
        
        # Create step object
        step = WorkflowStep(
            id=substituted.get("id", str(uuid4())),
            name=substituted.get("name", "Unnamed Step"),
            type=StepType(substituted.get("type", "action")),
            parameters=substituted.get("parameters", {}),
            dependencies=substituted.get("dependencies", []),
            retry_config=substituted.get("retry_config", {}),
            timeout_seconds=substituted.get("timeout_seconds"),
            on_success=substituted.get("on_success"),
            on_failure=substituted.get("on_failure"),
            metadata=substituted.get("metadata", {})
        )
        
        return step
    
    def _substitute_parameters(self, template_dict: Dict[str, Any], 
                             parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute parameters in template dictionary."""
        result = {}
        
        for key, value in template_dict.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Parameter substitution
                param_name = value[2:-1]
                if param_name in parameters:
                    result[key] = parameters[param_name]
                else:
                    result[key] = value  # Keep original if parameter not found
            elif isinstance(value, dict):
                result[key] = self._substitute_parameters(value, parameters)
            elif isinstance(value, list):
                result[key] = [
                    self._substitute_parameters(item, parameters) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result


class TemplateRegistry:
    """Registry for workflow templates."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self._templates: Dict[str, WorkflowTemplate] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._base_path = base_path or Path("/tmp/workflow_templates")
        self._base_path.mkdir(parents=True, exist_ok=True)
    
    def register_template(self, template: WorkflowTemplate) -> None:
        """Register a workflow template."""
        # Validate template inheritance
        if template.parent_template_id:
            if template.parent_template_id not in self._templates:
                raise ValueError(f"Parent template {template.parent_template_id} not found")
            
            # Merge with parent template
            template = self._merge_with_parent(template)
        
        self._templates[template.id] = template
        
        # Update category index
        if template.category not in self._categories:
            self._categories[template.category] = set()
        self._categories[template.category].add(template.id)
        
        logger.info(f"Registered workflow template: {template.name} ({template.id})")
    
    def get_template(self, template_id: str) -> WorkflowTemplate:
        """Get a template by ID."""
        if template_id not in self._templates:
            raise ValueError(f"Template {template_id} not found")
        
        return self._templates[template_id]
    
    def list_templates(self, category: Optional[str] = None, 
                      tags: Optional[Set[str]] = None) -> List[WorkflowTemplate]:
        """List templates, optionally filtered."""
        templates = list(self._templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if tags:
            templates = [t for t in templates if t.tags.intersection(tags)]
        
        return templates
    
    def list_categories(self) -> List[str]:
        """List all template categories."""
        return list(self._categories.keys())
    
    def search_templates(self, query: str) -> List[WorkflowTemplate]:
        """Search templates by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for template in self._templates.values():
            if (query_lower in template.name.lower() or 
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return results
    
    def save_template(self, template: WorkflowTemplate) -> Path:
        """Save template to disk."""
        template_path = self._base_path / f"{template.id}.json"
        
        # Convert to serializable format
        template_data = {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "version": template.version,
            "category": template.category,
            "tags": list(template.tags),
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.value,
                    "description": p.description,
                    "default_value": p.default_value,
                    "required": p.required,
                    "validation_rules": p.validation_rules,
                    "examples": p.examples
                } for p in template.parameters
            ],
            "steps_template": template.steps_template,
            "parent_template_id": template.parent_template_id,
            "global_parameters": template.global_parameters,
            "metadata": template.metadata,
            "created_at": template.created_at.isoformat(),
            "created_by": template.created_by
        }
        
        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        logger.info(f"Saved template to {template_path}")
        return template_path
    
    def load_template(self, template_path: Path) -> WorkflowTemplate:
        """Load template from disk."""
        with open(template_path, 'r') as f:
            template_data = json.load(f)
        
        # Convert from serializable format
        parameters = []
        for param_data in template_data.get("parameters", []):
            param = ParameterDefinition(
                name=param_data["name"],
                type=ParameterType(param_data["type"]),
                description=param_data.get("description", ""),
                default_value=param_data.get("default_value"),
                required=param_data.get("required", True),
                validation_rules=param_data.get("validation_rules", {}),
                examples=param_data.get("examples", [])
            )
            parameters.append(param)
        
        template = WorkflowTemplate(
            id=template_data["id"],
            name=template_data["name"],
            description=template_data["description"],
            version=template_data.get("version", "1.0.0"),
            category=template_data.get("category", "general"),
            tags=set(template_data.get("tags", [])),
            parameters=parameters,
            steps_template=template_data.get("steps_template", []),
            parent_template_id=template_data.get("parent_template_id"),
            global_parameters=template_data.get("global_parameters", {}),
            metadata=template_data.get("metadata", {}),
            created_at=datetime.fromisoformat(template_data.get("created_at", datetime.now(timezone.utc).isoformat())),
            created_by=template_data.get("created_by")
        )
        
        return template
    
    def load_templates_from_directory(self, directory: Path) -> int:
        """Load all templates from a directory."""
        count = 0
        for template_file in directory.glob("*.json"):
            try:
                template = self.load_template(template_file)
                self.register_template(template)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load template from {template_file}: {e}")
        
        logger.info(f"Loaded {count} templates from {directory}")
        return count
    
    def _merge_with_parent(self, template: WorkflowTemplate) -> WorkflowTemplate:
        """Merge template with its parent template."""
        parent = self._templates[template.parent_template_id]
        
        # Merge parameters (child overrides parent)
        merged_params = {p.name: p for p in parent.parameters}
        for param in template.parameters:
            merged_params[param.name] = param
        
        # Merge steps (child appends to parent)
        merged_steps = parent.steps_template.copy()
        merged_steps.extend(template.steps_template)
        
        # Merge global parameters
        merged_globals = dict(parent.global_parameters)
        merged_globals.update(template.global_parameters)
        
        # Merge tags
        merged_tags = parent.tags.union(template.tags)
        
        # Create merged template
        merged_template = WorkflowTemplate(
            id=template.id,
            name=template.name,
            description=template.description,
            version=template.version,
            category=template.category,
            tags=merged_tags,
            parameters=list(merged_params.values()),
            steps_template=merged_steps,
            parent_template_id=template.parent_template_id,
            global_parameters=merged_globals,
            metadata=template.metadata,
            created_at=template.created_at,
            created_by=template.created_by
        )
        
        return merged_template


class TemplateManager:
    """High-level template management interface."""
    
    def __init__(self, registry: Optional[TemplateRegistry] = None):
        self._registry = registry or TemplateRegistry()
    
    @property
    def registry(self) -> TemplateRegistry:
        """Get the underlying registry."""
        return self._registry
    
    def create_template_from_workflow(self, workflow: WorkflowDefinition,
                                    parameterize: Dict[str, ParameterDefinition]) -> WorkflowTemplate:
        """Create a template from an existing workflow definition."""
        # Convert workflow steps to template format
        steps_template = []
        for step in workflow.steps:
            step_template = {
                "id": step.id,
                "name": step.name,
                "type": step.type.value,
                "parameters": step.parameters,
                "dependencies": step.dependencies,
                "retry_config": step.retry_config,
                "timeout_seconds": step.timeout_seconds,
                "on_success": step.on_success,
                "on_failure": step.on_failure,
                "metadata": step.metadata
            }
            steps_template.append(step_template)
        
        # Create template
        template = WorkflowTemplate(
            id=str(uuid4()),
            name=f"{workflow.name}_template",
            description=f"Template created from workflow {workflow.name}",
            version=workflow.version,
            parameters=list(parameterize.values()),
            steps_template=steps_template,
            global_parameters=workflow.global_parameters,
            tags=workflow.tags
        )
        
        return template
    
    def create_common_templates(self) -> None:
        """Create common workflow templates."""
        
        # Data Processing Pipeline Template
        data_pipeline_template = WorkflowTemplate(
            id="data_pipeline_v1",
            name="Data Processing Pipeline",
            description="Generic data processing pipeline with validation and transformation",
            category="data_processing",
            tags={"data", "pipeline", "etl"},
            parameters=[
                ParameterDefinition(
                    name="input_source",
                    type=ParameterType.STRING,
                    description="Input data source",
                    required=True,
                    examples=["s3://bucket/data", "database://table"]
                ),
                ParameterDefinition(
                    name="output_destination", 
                    type=ParameterType.STRING,
                    description="Output destination",
                    required=True
                ),
                ParameterDefinition(
                    name="batch_size",
                    type=ParameterType.INTEGER,
                    description="Processing batch size",
                    default_value=1000,
                    required=False,
                    validation_rules={"min_value": 1, "max_value": 10000}
                )
            ],
            steps_template=[
                {
                    "id": "validate_input",
                    "name": "Validate Input Data",
                    "type": "action",
                    "parameters": {
                        "action_name": "validate_data",
                        "source": "${input_source}"
                    }
                },
                {
                    "id": "transform_data",
                    "name": "Transform Data",
                    "type": "action",
                    "parameters": {
                        "action_name": "transform_data",
                        "source": "${input_source}",
                        "batch_size": "${batch_size}"
                    },
                    "dependencies": ["validate_input"]
                },
                {
                    "id": "save_output",
                    "name": "Save Output",
                    "type": "action", 
                    "parameters": {
                        "action_name": "save_data",
                        "destination": "${output_destination}"
                    },
                    "dependencies": ["transform_data"]
                }
            ]
        )
        
        # ML Model Training Template
        ml_training_template = WorkflowTemplate(
            id="ml_training_v1",
            name="ML Model Training Pipeline",
            description="Machine learning model training with evaluation and deployment",
            category="machine_learning",
            tags={"ml", "training", "deployment"},
            parameters=[
                ParameterDefinition(
                    name="model_type",
                    type=ParameterType.STRING,
                    description="Type of ML model to train",
                    required=True,
                    examples=["linear_regression", "random_forest", "neural_network"]
                ),
                ParameterDefinition(
                    name="training_data",
                    type=ParameterType.STRING,
                    description="Training dataset path",
                    required=True
                ),
                ParameterDefinition(
                    name="validation_split",
                    type=ParameterType.FLOAT,
                    description="Validation split ratio",
                    default_value=0.2,
                    required=False,
                    validation_rules={"min_value": 0.1, "max_value": 0.5}
                )
            ],
            steps_template=[
                {
                    "id": "load_data",
                    "name": "Load Training Data",
                    "type": "action",
                    "parameters": {
                        "action_name": "load_training_data",
                        "data_path": "${training_data}",
                        "validation_split": "${validation_split}"
                    }
                },
                {
                    "id": "train_model",
                    "name": "Train Model",
                    "type": "action",
                    "parameters": {
                        "action_name": "train_ml_model",
                        "model_type": "${model_type}"
                    },
                    "dependencies": ["load_data"]
                },
                {
                    "id": "evaluate_model",
                    "name": "Evaluate Model",
                    "type": "action",
                    "parameters": {
                        "action_name": "evaluate_model"
                    },
                    "dependencies": ["train_model"]
                },
                {
                    "id": "deploy_model",
                    "name": "Deploy Model",
                    "type": "condition",
                    "condition": "step_results['evaluate_model']['output']['accuracy'] > 0.8",
                    "dependencies": ["evaluate_model"]
                }
            ]
        )
        
        # Register templates
        self._registry.register_template(data_pipeline_template)
        self._registry.register_template(ml_training_template)
        
        logger.info("Created common workflow templates")


# Global instances
template_registry = TemplateRegistry()
template_manager = TemplateManager(template_registry)