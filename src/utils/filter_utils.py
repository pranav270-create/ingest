from typing import Any, TypeVar, Tuple, List

from src.schemas.schemas import ExtractedFeatureType, EmbeddedFeatureType

T = TypeVar('T')


def filter_basemodels(basemodels: List[T], filter_params: dict[str, Any]) -> Tuple[List[T], List[T]]:
    """
    Filter basemodels based on provided field conditions.
    Returns tuple of (filtered_models, unfiltered_models)

    Example usage in yaml:
        filter_params:
            consolidated_feature_type: "image"
            chunk_index: 1
            embedded_feature_type: ["raw", "synthetic"]
    """
    # Map of field names to their enum classes
    enum_fields = {
        'consolidated_feature_type': ExtractedFeatureType,
        'embedded_feature_type': EmbeddedFeatureType,
    }

    # Preprocess filter_params to convert string values to enums where needed
    processed_params = {}
    for key, value in filter_params.items():
        if key in enum_fields:
            enum_class = enum_fields[key]
            if isinstance(value, list):
                # Handle list of values
                processed_values = []
                for v in value:
                    if isinstance(v, str):
                        try:
                            matching_member = next(
                                member for member in enum_class
                                if member.value == v.lower()
                            )
                            processed_values.append(matching_member)
                        except StopIteration:
                            valid_values = [member.value for member in enum_class]
                            raise ValueError(f"Invalid value '{v}' for {key}. Must be one of {valid_values}")
                    else:
                        processed_values.append(v)
                processed_params[key] = processed_values
            elif isinstance(value, str):
                # Handle single string value
                try:
                    matching_member = next(
                        member for member in enum_class 
                        if member.value == value.lower()
                    )
                    processed_params[key] = matching_member
                except StopIteration:
                    valid_values = [member.value for member in enum_class]
                    raise ValueError(f"Invalid value '{value}' for {key}. Must be one of {valid_values}")
            else:
                processed_params[key] = value
        else:
            processed_params[key] = value

    filtered = []
    unfiltered = []

    for model in basemodels:
        matches_all_conditions = True
        for field, expected_value in processed_params.items():
            actual_value = getattr(model, field, None)

            # Handle list of allowed values
            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    matches_all_conditions = False
                    break
            # Handle single value match
            elif actual_value != expected_value:
                matches_all_conditions = False
                break

        if matches_all_conditions:
            filtered.append(model)
        else:
            unfiltered.append(model)

    return filtered, unfiltered
