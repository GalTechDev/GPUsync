"""`gpusync.gfx` — Framework graphique 3D / 2D / UI temps réel 100 % GPUsync."""

from .app import App
from .atmosphere import Atmosphere, AtmosphereConfig
from .camera import Camera2D, Camera3D, Viewport
from .effects import ChromaticAberration, ColorGrade, Pixelate, PostEffect, Scanlines
from .fluids import (
    FloatingBody,
    GpuFloaters,
    Fluid2DConfig,
    Fluid3DConfig,
    FluidSPHConfig,
    GpuFluid2D,
    GpuFluid3D,
    GpuFluidSPH3D,
    GpuWaterSurface,
    WaterSurfaceConfig,
)
from .gltf import load_gltf_mesh, load_gltf_nodes, write_glb_from_mesh
from .input import Input, PickResult
from .local import LocalClouds
from .material import (
    PATTERN_BANDS,
    PATTERN_CHECKER,
    PATTERN_GRID,
    PATTERN_MARBLE,
    PATTERN_MIRROR_Y,
    PATTERN_MIRROR_Z,
    PATTERN_SOLID,
    DirectionalLight,
    Environment,
    Material,
    PointLight,
    SpotLight,
)
from .mesh import Mesh
from .nest import NestedAtmosphere
from .node import InstancedMesh3D, Node, Object3D
from .objects2d import Circle2D, Line2D, Object2D, Particles2D, Rect2D, Sprite2D
from .physics import (
    BoxCollider,
    GpuSpherePhysics,
    PhysicsWorld,
    PlaneCollider,
    RayHit,
    RigidBody,
    SphereCollider,
)
from .renderer2d import Renderer2D
from .renderer3d import Renderer3D
from .scene import Scene
from .sky import CloudRenderer, SkyRenderer
from .texture import PlanarReflection, RenderTarget, Texture, TexturePool
from .ui import Button, FPSOverlay, Label, Panel, ProgressBar, UILayer, Widget
from .weather import Weather, WeatherConfig, fictional_geography

__all__ = [
    "LocalClouds",
    "Weather",
    "WeatherConfig",
    "fictional_geography",
    "App",
    "Scene",
    "Node",
    "Object3D",
    "InstancedMesh3D",
    "Object2D",
    "Rect2D",
    "Circle2D",
    "Line2D",
    "Sprite2D",
    "Particles2D",
    "Mesh",
    "Material",
    "Texture",
    "RenderTarget",
    "PlanarReflection",
    "TexturePool",
    "DirectionalLight",
    "PointLight",
    "SpotLight",
    "Environment",
    "PATTERN_SOLID",
    "PATTERN_CHECKER",
    "PATTERN_GRID",
    "PATTERN_BANDS",
    "PATTERN_MARBLE",
    "PATTERN_MIRROR_Z",
    "PATTERN_MIRROR_Y",
    "Camera3D",
    "Camera2D",
    "Viewport",
    "PostEffect",
    "ChromaticAberration",
    "Scanlines",
    "ColorGrade",
    "Pixelate",
    "Atmosphere",
    "NestedAtmosphere",
    "AtmosphereConfig",
    "CloudRenderer",
    "SkyRenderer",
    "FloatingBody",
    "GpuFloaters",
    "WaterSurfaceConfig",
    "GpuWaterSurface",
    "Fluid3DConfig",
    "GpuFluid3D",
    "FluidSPHConfig",
    "GpuFluidSPH3D",
    "Fluid2DConfig",
    "GpuFluid2D",
    "PhysicsWorld",
    "RigidBody",
    "SphereCollider",
    "BoxCollider",
    "PlaneCollider",
    "RayHit",
    "GpuSpherePhysics",
    "load_gltf_mesh",
    "load_gltf_nodes",
    "write_glb_from_mesh",
    "Renderer3D",
    "Renderer2D",
    "Input",
    "PickResult",
    "UILayer",
    "Widget",
    "Panel",
    "Label",
    "Button",
    "ProgressBar",
    "FPSOverlay",
]
