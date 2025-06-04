"""Data models and schemas for the MCP Agent Framework.

This package contains Pydantic data models, state management schemas, and
validation utilities used throughout the framework.

Authors:
    Fernando Delgado Chaves, Piyush Kulkarni, Gautam Chug,
    Nitanshu Mayur Idnani, Reeju Bhattacharjee

Example:
    Basic model usage:

    >>> from mcp_agent.models import SearchQuery, SearchResult, ToolInfo
    >>> query = SearchQuery(text="RNA-seq tools", max_results=10)
    >>> tool = ToolInfo(name="BLAST", description="Sequence alignment tool")
    >>> result = SearchResult(query="RNA-seq", tools=[tool])
"""

from typing import Any, Dict, List, Optional, Type, Union
import json
from datetime import datetime

# Import core model classes
try:
    from mcp_agent.models.schemas import (
        # Core data models
        SearchQuery,
        SearchResult,
        ToolInfo,
        AgentTask,
        AgentResponse,
        ResearchData,
        
        # Report models
        ReportTemplate,
        ReportMetadata,
        ReportSection,
        
        # User interaction models
        UserQuery,
        UserPreferences,
        
        # Tool and resource models
        BioinformaticsToolInfo,
        DatabaseInfo,
        PublicationInfo,
        
        # Validation models
        ValidationResult,
        ErrorInfo,
    )
    
    from mcp_agent.models.state import (
        # Workflow state models
        WorkflowState,
        AgentState,
        TaskState,
        
        # Graph state models  
        GraphState,
        NodeState,
        EdgeState,
        
        # Execution state models
        ExecutionContext,
        ExecutionResult,
        ExecutionMetrics,
    )
    
    # Make all key model classes available
    __all__ = [
        # Core data models
        "SearchQuery",
        "SearchResult", 
        "ToolInfo",
        "AgentTask",
        "AgentResponse",
        "ResearchData",
        
        # Report models
        "ReportTemplate",
        "ReportMetadata", 
        "ReportSection",
        
        # User models
        "UserQuery",
        "UserPreferences",
        
        # Specialized models
        "BioinformaticsToolInfo",
        "DatabaseInfo",
        "PublicationInfo",
        
        # Validation models
        "ValidationResult",
        "ErrorInfo",
        
        # State models
        "WorkflowState",
        "AgentState",
        "TaskState",
        "GraphState",
        "NodeState", 
        "EdgeState",
        "ExecutionContext",
        "ExecutionResult",
        "ExecutionMetrics",
        
        # Model utilities
        "ModelRegistry",
        "create_model",
        "validate_model",
        "serialize_model",
        "deserialize_model",
        "get_model_schema",
        
        # Model management
        "list_available_models",
        "get_model_info",
        "validate_models_compatibility",
    ]
    
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(
        f"Model components not available: {e}. "
        "This is normal during initial setup.",
        ImportWarning,
        stacklevel=2
    )
    
    # Provide minimal exports for development
    __all__ = [
        "create_model",
        "validate_model", 
        "list_available_models",
        "get_model_info",
    ]
    
    # Mock classes for development
    class SearchQuery:
        """Mock SearchQuery for development."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class SearchResult:
        """Mock SearchResult for development."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ToolInfo:
        """Mock ToolInfo for development."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class AgentTask:
        """Mock AgentTask for development."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class AgentResponse:
        """Mock AgentResponse for development."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class WorkflowState:
        """Mock WorkflowState for development."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class ModelRegistry:
    """Registry for managing available data models and their schemas.
    
    This class provides a centralized way to discover, validate, and manage
    different types of data models in the framework.
    
    Example:
        >>> registry = ModelRegistry()
        >>> model = registry.create_model("SearchQuery", text="RNA-seq")
        >>> schema = registry.get_schema("ToolInfo")
    """
    
    def __init__(self) -> None:
        """Initialize the model registry."""
        self._models: Dict[str, Type[Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_models()
    
    def _register_builtin_models(self) -> None:
        """Register the built-in model types."""
        try:
            self._models = {
                # Core models
                "SearchQuery": SearchQuery,
                "SearchResult": SearchResult,
                "ToolInfo": ToolInfo,
                "AgentTask": AgentTask,
                "AgentResponse": AgentResponse,
                "ResearchData": ResearchData,
                
                # Report models
                "ReportTemplate": ReportTemplate,
                "ReportMetadata": ReportMetadata,
                "ReportSection": ReportSection,
                
                # User models
                "UserQuery": UserQuery,
                "UserPreferences": UserPreferences,
                
                # Specialized models
                "BioinformaticsToolInfo": BioinformaticsToolInfo,
                "DatabaseInfo": DatabaseInfo,
                "PublicationInfo": PublicationInfo,
                
                # Validation models
                "ValidationResult": ValidationResult,
                "ErrorInfo": ErrorInfo,
                
                # State models
                "WorkflowState": WorkflowState,
                "AgentState": AgentState,
                "TaskState": TaskState,
                "GraphState": GraphState,
                "NodeState": NodeState,
                "EdgeState": EdgeState,
                "ExecutionContext": ExecutionContext,
                "ExecutionResult": ExecutionResult,
                "ExecutionMetrics": ExecutionMetrics,
            }
            
            # Generate schemas for models that have them
            for name, model_class in self._models.items():
                if hasattr(model_class, 'model_json_schema'):
                    try:
                        self._schemas[name] = model_class.model_json_schema()
                    except Exception:
                        # Some models might not be fully initialized yet
                        pass
                        
        except NameError:
            # During development when classes aren't available
            pass
    
    def register_model(
        self,
        name: str,
        model_class: Type[Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new model type.
        
        Args:
            name: Unique name for the model type.
            model_class: Model class to register.
            schema: Optional JSON schema for the model.
            
        Example:
            >>> registry.register_model(
            ...     "CustomModel",
            ...     CustomModel,
            ...     {"type": "object", "properties": {...}}
            ... )
        """
        self._models[name] = model_class
        
        if schema:
            self._schemas[name] = schema
        elif hasattr(model_class, 'model_json_schema'):
            try:
                self._schemas[name] = model_class.model_json_schema()
            except Exception:
                pass
    
    def create_model(self, model_type: str, *args, **kwargs) -> Any:
        """Create a model instance of the specified type.
        
        Args:
            model_type: Type of model to create.
            *args: Positional arguments for model constructor.
            **kwargs: Keyword arguments for model constructor.
            
        Returns:
            Any: Created model instance.
            
        Raises:
            ValueError: If model type is not registered.
            
        Example:
            >>> query = registry.create_model(
            ...     "SearchQuery",
            ...     text="RNA-seq tools",
            ...     max_results=10
            ... )
        """
        if model_type not in self._models:
            available = list(self._models.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
        
        model_class = self._models[model_type]
        return model_class(*args, **kwargs)
    
    def get_schema(self, model_type: str) -> Dict[str, Any]:
        """Get JSON schema for a model type.
        
        Args:
            model_type: Model type to get schema for.
            
        Returns:
            Dict[str, Any]: JSON schema.
            
        Raises:
            ValueError: If model type is not registered.
        """
        if model_type not in self._schemas:
            if model_type in self._models:
                # Try to generate schema
                model_class = self._models[model_type]
                if hasattr(model_class, 'model_json_schema'):
                    try:
                        schema = model_class.model_json_schema()
                        self._schemas[model_type] = schema
                        return schema
                    except Exception:
                        pass
            
            available = list(self._schemas.keys())
            raise ValueError(f"No schema available for model type: {model_type}. Available: {available}")
        
        return self._schemas[model_type].copy()
    
    def list_models(self) -> List[str]:
        """List all registered model types.
        
        Returns:
            List[str]: List of model type names.
        """
        return list(self._models.keys())
    
    def validate_model_type(self, model_type: str) -> bool:
        """Validate that a model type is registered.
        
        Args:
            model_type: Model type to validate.
            
        Returns:
            bool: True if model type is registered.
        """
        return model_type in self._models
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get comprehensive information about a model type.
        
        Args:
            model_type: Model type to get info for.
            
        Returns:
            Dict[str, Any]: Model information.
        """
        if model_type not in self._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self._models[model_type]
        
        info = {
            "name": model_type,
            "class": model_class.__name__,
            "module": getattr(model_class, '__module__', 'unknown'),
            "has_schema": model_type in self._schemas,
            "is_pydantic": hasattr(model_class, 'model_fields'),
        }
        
        # Add Pydantic-specific info
        if info["is_pydantic"]:
            try:
                info["fields"] = list(model_class.model_fields.keys())
                info["required_fields"] = [
                    name for name, field in model_class.model_fields.items()
                    if field.is_required()
                ]
            except Exception:
                pass
        
        # Add docstring if available
        if hasattr(model_class, '__doc__') and model_class.__doc__:
            info["description"] = model_class.__doc__.strip().split('\n')[0]
        
        return info


# Global registry instance
_registry = ModelRegistry()


def create_model(model_type: str, *args, **kwargs) -> Any:
    """Create a model instance using the global registry.
    
    Args:
        model_type: Type of model to create.
        *args: Positional arguments for model constructor.
        **kwargs: Keyword arguments for model constructor.
        
    Returns:
        Any: Created model instance.
        
    Example:
        >>> from mcp_agent.models import create_model
        >>> query = create_model("SearchQuery", text="RNA-seq", max_results=10)
        >>> tool = create_model("ToolInfo", name="BLAST", description="Alignment tool")
    """
    return _registry.create_model(model_type, *args, **kwargs)


def validate_model(model_instance: Any) -> Dict[str, Any]:
    """Validate a model instance.
    
    Args:
        model_instance: Model instance to validate.
        
    Returns:
        Dict[str, Any]: Validation results.
        
    Example:
        >>> validation = validate_model(search_query)
        >>> if validation["valid"]:
        ...     print("Model is valid")
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "model_type": type(model_instance).__name__,
    }
    
    try:
        # Check if it's a Pydantic model
        if hasattr(model_instance, 'model_validate'):
            # For Pydantic v2
            try:
                model_instance.model_validate(model_instance.model_dump())
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Pydantic validation failed: {e}")
        
        elif hasattr(model_instance, 'dict'):
            # For Pydantic v1
            try:
                model_instance.dict()
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Model serialization failed: {e}")
        
        # Additional custom validations can be added here
        
    except Exception as e:
        validation_result["valid"] = False
        validation_result["errors"].append(f"General validation error: {e}")
    
    return validation_result


def serialize_model(model_instance: Any, format: str = "json") -> Union[str, Dict[str, Any]]:
    """Serialize a model instance to the specified format.
    
    Args:
        model_instance: Model instance to serialize.
        format: Output format ("json", "dict").
        
    Returns:
        Union[str, Dict[str, Any]]: Serialized model.
        
    Example:
        >>> json_str = serialize_model(search_query, "json")
        >>> data_dict = serialize_model(search_query, "dict")
    """
    try:
        # Handle Pydantic models
        if hasattr(model_instance, 'model_dump'):
            # Pydantic v2
            data = model_instance.model_dump()
        elif hasattr(model_instance, 'dict'):
            # Pydantic v1
            data = model_instance.dict()
        else:
            # Generic object
            data = vars(model_instance)
        
        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == "dict":
            return data
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    except Exception as e:
        raise RuntimeError(f"Serialization failed: {e}") from e


def deserialize_model(data: Union[str, Dict[str, Any]], model_type: str) -> Any:
    """Deserialize data to a model instance.
    
    Args:
        data: Data to deserialize (JSON string or dictionary).
        model_type: Target model type.
        
    Returns:
        Any: Deserialized model instance.
        
    Example:
        >>> query = deserialize_model(json_data, "SearchQuery")
        >>> tool = deserialize_model(dict_data, "ToolInfo")
    """
    try:
        # Parse JSON if needed
        if isinstance(data, str):
            data = json.loads(data)
        
        # Create model instance
        return _registry.create_model(model_type, **data)
        
    except Exception as e:
        raise RuntimeError(f"Deserialization failed: {e}") from e


def get_model_schema(model_type: str) -> Dict[str, Any]:
    """Get JSON schema for a model type.
    
    Args:
        model_type: Model type to get schema for.
        
    Returns:
        Dict[str, Any]: JSON schema.
        
    Example:
        >>> schema = get_model_schema("SearchQuery")
        >>> print(schema["properties"])
    """
    return _registry.get_schema(model_type)


def list_available_models() -> List[str]:
    """Get list of available model types.
    
    Returns:
        List[str]: List of model type names.
        
    Example:
        >>> models = list_available_models()
        >>> print("Available models:", models)
    """
    return _registry.list_models()


def get_model_info(model_type: str) -> Dict[str, Any]:
    """Get comprehensive information about a model type.
    
    Args:
        model_type: Model type to get info for.
        
    Returns:
        Dict[str, Any]: Model information.
        
    Example:
        >>> info = get_model_info("SearchQuery")
        >>> print(f"Required fields: {info['required_fields']}")
    """
    return _registry.get_model_info(model_type)


def validate_models_compatibility(*model_instances) -> Dict[str, Any]:
    """Validate compatibility between multiple model instances.
    
    Args:
        *model_instances: Model instances to check compatibility.
        
    Returns:
        Dict[str, Any]: Compatibility validation results.
        
    Example:
        >>> compatibility = validate_models_compatibility(query, result, tool)
        >>> if compatibility["compatible"]:
        ...     print("Models are compatible")
    """
    validation_result = {
        "compatible": True,
        "issues": [],
        "warnings": [],
        "models_checked": len(model_instances),
        "model_types": [type(m).__name__ for m in model_instances],
    }
    
    if len(model_instances) < 2:
        validation_result["warnings"].append("Less than 2 models provided for compatibility check")
        return validation_result
    
    try:
        # Basic compatibility checks
        for i, model1 in enumerate(model_instances):
            for j, model2 in enumerate(model_instances[i+1:], i+1):
                # Check if models can be used together
                # This is a basic implementation - could be enhanced with specific rules
                
                # Example: SearchQuery and SearchResult should be compatible
                if (isinstance(model1, SearchQuery) and isinstance(model2, SearchResult)) or \
                   (isinstance(model2, SearchQuery) and isinstance(model1, SearchResult)):
                    # Check if query text matches
                    if hasattr(model1, 'text') and hasattr(model2, 'query'):
                        if model1.text != model2.query:
                            validation_result["warnings"].append(
                                f"Query text mismatch between {type(model1).__name__} and {type(model2).__name__}"
                            )
                
                # Add more compatibility rules as needed
        
    except Exception as e:
        validation_result["compatible"] = False
        validation_result["issues"].append(f"Compatibility check failed: {e}")
    
    return validation_result


# Package-level constants
MODEL_TYPES = [
    "SearchQuery", "SearchResult", "ToolInfo", "AgentTask", "AgentResponse",
    "WorkflowState", "ReportTemplate", "UserQuery", "BioinformaticsToolInfo"
]

# Model categories
MODEL_CATEGORIES = {
    "core": ["SearchQuery", "SearchResult", "ToolInfo", "AgentTask", "AgentResponse"],
    "state": ["WorkflowState", "AgentState", "TaskState", "GraphState"],
    "reports": ["ReportTemplate", "ReportMetadata", "ReportSection"],
    "user": ["UserQuery", "UserPreferences"],
    "bioinformatics": ["BioinformaticsToolInfo", "DatabaseInfo", "PublicationInfo"],
    "validation": ["ValidationResult", "ErrorInfo"],
}

# Version compatibility
MODELS_VERSION = "0.1.0"
PYDANTIC_VERSION_REQUIRED = ">=2.0.0"

# Feature flags for optional model components
MODEL_FEATURES = {
    "validation": True,
    "serialization": True,
    "schema_generation": True,
    "compatibility_checking": True,
    "type_coercion": True,
}


def list_model_features() -> Dict[str, bool]:
    """List all available model features and their status.
    
    Returns:
        Dict[str, bool]: Dictionary of feature names and availability.
        
    Example:
        >>> features = list_model_features()
        >>> if features['validation']:
        ...     print("Model validation is available")
    """
    return MODEL_FEATURES.copy()


def get_models_by_category(category: str) -> List[str]:
    """Get model types by category.
    
    Args:
        category: Category name ('core', 'state', 'reports', etc.).
        
    Returns:
        List[str]: List of model types in the category.
        
    Example:
        >>> core_models = get_models_by_category("core")
        >>> print("Core models:", core_models)
    """
    return MODEL_CATEGORIES.get(category, [])


def validate_model_data(data: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Validate raw data against a model schema without creating an instance.
    
    Args:
        data: Data to validate.
        model_type: Model type to validate against.
        
    Returns:
        Dict[str, Any]: Validation results.
        
    Example:
        >>> validation = validate_model_data(
        ...     {"text": "RNA-seq", "max_results": 10},
        ...     "SearchQuery"
        ... )
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "model_type": model_type,
    }
    
    try:
        # Get model schema
        schema = get_model_schema(model_type)
        
        # Basic schema validation
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
        
        # Check field types
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type:
                    # Basic type checking (could be enhanced)
                    if expected_type == "string" and not isinstance(value, str):
                        validation_result["warnings"].append(f"Field {field} should be string")
                    elif expected_type == "integer" and not isinstance(value, int):
                        validation_result["warnings"].append(f"Field {field} should be integer")
                    elif expected_type == "array" and not isinstance(value, list):
                        validation_result["warnings"].append(f"Field {field} should be array")
        
    except Exception as e:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Schema validation error: {e}")
    
    return validation_result


# Backwards compatibility aliases
Query = SearchQuery  # Alias for backwards compatibility
Result = SearchResult  # Alias for backwards compatibility
Tool = ToolInfo  # Alias for backwards compatibility
Task = AgentTask  # Alias for backwards compatibility
Response = AgentResponse  # Alias for backwards compatibility