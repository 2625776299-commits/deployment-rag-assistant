"""Extract explicit depends_on links; selecting answer evidence is a later step."""

import yaml


def extract_dependencies(text: str, *, config_id: str) -> list[dict]:
    """Retain every declared depends_on edge within one configuration block.

    This does not infer runtime dependencies or recursively select evidence.
    A missing target is retained and marked, rather than silently discarded.
    declared_condition is None when the YAML does not state one explicitly.
    """
    if not isinstance(config_id, str) or not config_id.strip():
        raise ValueError("config_id must identify one configuration block")
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError("Expected one valid YAML document") from exc
    if not isinstance(document, dict):
        raise ValueError("Expected a Compose mapping")
    services = document.get("services", {})
    if not isinstance(services, dict):
        raise ValueError("services must be a mapping")
    if any(not isinstance(name, str) or not name.strip() for name in services):
        raise ValueError("Service names must be non-empty strings")

    relations = []
    for source_name, configuration in services.items():
        if not isinstance(configuration, dict):
            raise ValueError(f"Service {source_name!r} must be a mapping")
        dependencies = configuration.get("depends_on", [])
        if isinstance(dependencies, list):
            entries = [(target, None) for target in dependencies]
        elif isinstance(dependencies, dict):
            entries = list(dependencies.items())
        else:
            raise ValueError(f"depends_on for {source_name!r} must be a list or mapping")

        for target_name, options in entries:
            if not isinstance(target_name, str) or not target_name.strip():
                raise ValueError("Dependency targets must be non-empty strings")
            if isinstance(dependencies, dict) and not isinstance(options, dict):
                raise ValueError("Long-form dependency options must be mappings")
            condition = options.get("condition") if options is not None else None
            if condition is not None and not isinstance(condition, str):
                raise ValueError("A declared dependency condition must be a string")
            relations.append({
                "config_id": config_id,
                "relation_type": "depends_on",
                "source_service": source_name,
                "target_service": target_name,
                "evidence_path": ["services", source_name, "depends_on"],
                "target_path": ["services", target_name],
                "declared_condition": condition,
                "target_defined": target_name in services,
            })

    return relations
