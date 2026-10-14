"""
Dataclass construction of camera context resolvers.
"""

from __future__ import annotations

# %% Imports
from dataclasses import is_dataclass
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    SemDTRayTracerCameraConfig,
)
from robokudo.io.semdt_camera_context import SemDTCameraContextResolver
from robokudo.utils.module_loader import ModuleLoader

# %% Resolver construction


def test_camera_context_resolver_is_dataclass() -> None:
    """
    Resolver configuration is represented by dataclass fields.
    """
    assert is_dataclass(SemDTCameraContextResolver)


def test_camera_context_resolver_preserves_injected_loader() -> None:
    """
    Construction retains the caller's configuration and loader.
    """
    configuration = SemDTRayTracerCameraConfig()
    loader = ModuleLoader()
    resolver = SemDTCameraContextResolver(configuration, module_loader=loader)
    assert resolver.camera_config is configuration
    assert resolver.module_loader is loader


def test_camera_context_resolvers_have_separate_default_loaders() -> None:
    """
    Default loaders belong to individual resolver instances.
    """
    configuration = SemDTRayTracerCameraConfig()
    first_resolver = SemDTCameraContextResolver(configuration)
    second_resolver = SemDTCameraContextResolver(configuration)
    assert isinstance(first_resolver.module_loader, ModuleLoader)
    assert isinstance(second_resolver.module_loader, ModuleLoader)
    assert first_resolver.module_loader is not second_resolver.module_loader
