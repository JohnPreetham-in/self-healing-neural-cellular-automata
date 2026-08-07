"""Validated, immutable experiment configuration loading."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from src.utils.io import load_yaml

Config: TypeAlias = Mapping[str, Any]

_REQUIRED_SECTIONS: tuple[str, ...] = (
    "run",
    "paths",
    "device",
    "seed",
    "model",
    "training",
    "optimizer",
    "evaluation",
    "visualization",
    "logging",
    "pool",
    "damage",
)

_REQUIRED_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "run": ("name", "mode"),
        "paths": ("outputs", "checkpoints", "logs"),
        "device": ("preference",),
        "seed": ("value", "deterministic"),
        "model": (
            "grid_size",
            "channels",
            "visible_channels",
            "hidden_channels",
            "perception",
            "update_rate",
            "alive_threshold",
        ),
        "training": ("epochs", "batch_size", "rollout_steps", "checkpoint_interval"),
        "optimizer": ("name", "learning_rate", "weight_decay"),
        "evaluation": ("rollout_steps", "num_trials", "stochastic_updates"),
        "visualization": ("enabled", "save_animations", "dpi"),
        "logging": ("level", "file_enabled"),
        "pool": ("enabled", "size", "reseed_fraction"),
        "damage": ("enabled", "shape", "fraction"),
    }
)


def load_config(path: str | Path) -> Config:
    """Load, validate, and recursively freeze an experiment configuration.

    Args:
        path: Path to a standalone YAML configuration file.

    Returns:
        A read-only nested configuration mapping.

    Raises:
        FileNotFoundError: If the configuration file is missing.
        ValueError: If required sections or fields are missing or invalid.
    """
    raw_config = load_yaml(path)
    validate_config(raw_config, path)
    return _freeze_config(raw_config)


def validate_config(config: Mapping[str, Any], source: str = "configuration") -> None:
    """Validate the shared experiment configuration schema.

    Args:
        config: Parsed configuration mapping to validate.
        source: Human-readable source used in error messages.

    Raises:
        ValueError: If the configuration does not satisfy the project schema.
    """
    _validate_required_sections(config, source)
    for section_name, field_names in _REQUIRED_FIELDS.items():
        section = config[section_name]
        if not isinstance(section, Mapping):
            raise ValueError(f"'{section_name}' in {source} must be a mapping.")
        _validate_required_fields(section_name, section, field_names, source)

    perception = config["model"]["perception"]
    if not isinstance(perception, Mapping):
        raise ValueError(f"'model.perception' in {source} must be a mapping.")
    _validate_required_fields("model.perception", perception, ("filters",), source)

    _validate_types(config, source)
    _validate_ranges(config, source)


def _validate_required_sections(config: Mapping[str, Any], source: str) -> None:
    missing_sections = [name for name in _REQUIRED_SECTIONS if name not in config]
    if missing_sections:
        joined_sections = ", ".join(missing_sections)
        raise ValueError(f"{source} is missing required section(s): {joined_sections}.")


def _validate_required_fields(
    section_name: str,
    section: Mapping[str, Any],
    field_names: tuple[str, ...],
    source: str,
) -> None:
    missing_fields = [name for name in field_names if name not in section]
    if missing_fields:
        joined_fields = ", ".join(missing_fields)
        raise ValueError(
            f"Section '{section_name}' in {source} is missing required field(s): "
            f"{joined_fields}."
        )


def _validate_types(config: Mapping[str, Any], source: str) -> None:
    _require_string(config, "run.name", source)
    _require_string(config, "run.mode", source)
    _require_strings(config, "paths", ("outputs", "checkpoints", "logs"), source)
    _require_string(config, "device.preference", source)
    _require_integer(config, "seed.value", source)
    _require_boolean(config, "seed.deterministic", source)
    _require_integer(config, "model.grid_size", source)
    _require_integer(config, "model.channels", source)
    _require_sequence_of_integers(config, "model.visible_channels", source)
    _require_sequence_of_integers(config, "model.hidden_channels", source)
    _require_mapping(config, "model.perception", source)
    _require_sequence_of_strings(config, "model.perception.filters", source)
    _require_number(config, "model.update_rate", source)
    _require_number(config, "model.alive_threshold", source)
    _require_integers(config, "training", ("epochs", "batch_size", "rollout_steps", "checkpoint_interval"), source)
    _require_string(config, "optimizer.name", source)
    _require_numbers(config, "optimizer", ("learning_rate", "weight_decay"), source)
    _require_integers(config, "evaluation", ("rollout_steps", "num_trials"), source)
    _require_boolean(config, "evaluation.stochastic_updates", source)
    _require_booleans(config, "visualization", ("enabled", "save_animations"), source)
    _require_integer(config, "visualization.dpi", source)
    _require_string(config, "logging.level", source)
    _require_boolean(config, "logging.file_enabled", source)
    _require_boolean(config, "pool.enabled", source)
    _require_integer(config, "pool.size", source)
    _require_number(config, "pool.reseed_fraction", source)
    _require_boolean(config, "damage.enabled", source)
    _require_string(config, "damage.shape", source)
    _require_number(config, "damage.fraction", source)


def _validate_ranges(config: Mapping[str, Any], source: str) -> None:
    for dotted_key in (
        "model.grid_size",
        "model.channels",
        "training.epochs",
        "training.batch_size",
        "training.rollout_steps",
        "training.checkpoint_interval",
        "evaluation.rollout_steps",
        "evaluation.num_trials",
        "visualization.dpi",
        "pool.size",
    ):
        if _get_value(config, dotted_key) <= 0:
            raise ValueError(f"'{dotted_key}' in {source} must be greater than zero.")

    for dotted_key in (
        "model.update_rate",
        "pool.reseed_fraction",
        "damage.fraction",
    ):
        value = _get_value(config, dotted_key)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"'{dotted_key}' in {source} must be between 0 and 1.")

    if _get_value(config, "model.alive_threshold") < 0.0:
        raise ValueError(f"'model.alive_threshold' in {source} must be non-negative.")
    if _get_value(config, "optimizer.learning_rate") <= 0.0:
        raise ValueError(f"'optimizer.learning_rate' in {source} must be greater than zero.")
    if _get_value(config, "optimizer.weight_decay") < 0.0:
        raise ValueError(f"'optimizer.weight_decay' in {source} must be non-negative.")

    _validate_model_channels(config, source)


def _get_value(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        value = value[key]
    return value


def _require_mapping(config: Mapping[str, Any], dotted_key: str, source: str) -> None:
    if not isinstance(_get_value(config, dotted_key), Mapping):
        raise ValueError(f"'{dotted_key}' in {source} must be a mapping.")


def _require_string(config: Mapping[str, Any], dotted_key: str, source: str) -> None:
    value = _get_value(config, dotted_key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{dotted_key}' in {source} must be a non-empty string.")


def _require_strings(
    config: Mapping[str, Any],
    section_name: str,
    field_names: tuple[str, ...],
    source: str,
) -> None:
    for field_name in field_names:
        _require_string(config, f"{section_name}.{field_name}", source)


def _require_boolean(config: Mapping[str, Any], dotted_key: str, source: str) -> None:
    if not isinstance(_get_value(config, dotted_key), bool):
        raise ValueError(f"'{dotted_key}' in {source} must be a boolean.")


def _require_booleans(
    config: Mapping[str, Any],
    section_name: str,
    field_names: tuple[str, ...],
    source: str,
) -> None:
    for field_name in field_names:
        _require_boolean(config, f"{section_name}.{field_name}", source)


def _require_integer(config: Mapping[str, Any], dotted_key: str, source: str) -> None:
    value = _get_value(config, dotted_key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"'{dotted_key}' in {source} must be an integer.")


def _require_integers(
    config: Mapping[str, Any],
    section_name: str,
    field_names: tuple[str, ...],
    source: str,
) -> None:
    for field_name in field_names:
        _require_integer(config, f"{section_name}.{field_name}", source)


def _require_number(config: Mapping[str, Any], dotted_key: str, source: str) -> None:
    value = _get_value(config, dotted_key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"'{dotted_key}' in {source} must be a number.")


def _require_numbers(
    config: Mapping[str, Any],
    section_name: str,
    field_names: tuple[str, ...],
    source: str,
) -> None:
    for field_name in field_names:
        _require_number(config, f"{section_name}.{field_name}", source)


def _require_sequence_of_integers(
    config: Mapping[str, Any],
    dotted_key: str,
    source: str,
) -> None:
    value = _get_value(config, dotted_key)
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"'{dotted_key}' in {source} must be a non-empty sequence.")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"'{dotted_key}' in {source} must contain only integers.")


def _require_sequence_of_strings(
    config: Mapping[str, Any],
    dotted_key: str,
    source: str,
) -> None:
    value = _get_value(config, dotted_key)
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"'{dotted_key}' in {source} must be a non-empty sequence.")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"'{dotted_key}' in {source} must contain only non-empty strings.")


def _validate_model_channels(config: Mapping[str, Any], source: str) -> None:
    channel_count = _get_value(config, "model.channels")
    visible_channels = _get_value(config, "model.visible_channels")
    hidden_channels = _get_value(config, "model.hidden_channels")
    all_channels = [*visible_channels, *hidden_channels]

    if len(all_channels) != channel_count or len(set(all_channels)) != channel_count:
        raise ValueError(
            f"Visible and hidden channels in {source} must be unique and cover "
            "exactly 'model.channels' entries."
        )
    if any(channel < 0 or channel >= channel_count for channel in all_channels):
        raise ValueError(
            f"Visible and hidden channel indices in {source} must be within the "
            "model channel range."
        )


def _freeze_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen_mapping = {key: _freeze_config(item) for key, item in value.items()}
        return MappingProxyType(frozen_mapping)
    if isinstance(value, list):
        return tuple(_freeze_config(item) for item in value)
    return value
