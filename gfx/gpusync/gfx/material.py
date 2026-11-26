"""Cook-Torrance PBR materials, unified light sources and environment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .math3d import normalize
from .texture import Texture

# Procedural patterns evaluated directly in the GPU shading kernel
PATTERN_SOLID = 0
PATTERN_CHECKER = 1
PATTERN_GRID = 2
PATTERN_BANDS = 3
PATTERN_MARBLE = 4
PATTERN_MIRROR_Z = 5
PATTERN_MIRROR_Y = 6

# Light types in the unified buffer (16 floats per light)
LIGHT_OFF = 0
LIGHT_POINT = 1
LIGHT_SPOT = 2


@dataclass
class Material:
    """Cook-Torrance PBR material (albedo, metallic, roughness, emissive, pattern or 2D / RenderTarget textures)."""

    albedo: tuple[float, float, float] = (0.8, 0.8, 0.8)
    metallic: float = 0.0
    roughness: float = 0.4
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pattern: int = PATTERN_SOLID
    pattern_scale: float = 2.0
    albedo2: tuple[float, float, float] = (0.15, 0.17, 0.22)
    albedo_map: Texture | None = None
    emissive_map: Texture | None = None
    roughness_map: Texture | None = None
    emissive_strength: float = 1.0

    @classmethod
    def solid(
        cls,
        color: tuple[float, float, float],
        *,
        metallic: float = 0.0,
        roughness: float = 0.4,
        emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Material:
        return cls(albedo=color, metallic=metallic, roughness=roughness, emissive=emissive)

    @classmethod
    def metal(
        cls,
        color: tuple[float, float, float] = (0.9, 0.92, 0.96),
        *,
        roughness: float = 0.14,
        metallic: float = 0.95,
    ) -> Material:
        return cls(albedo=color, metallic=metallic, roughness=roughness)

    @classmethod
    def checker(
        cls,
        c1: tuple[float, float, float] = (0.24, 0.27, 0.34),
        c2: tuple[float, float, float] = (0.10, 0.12, 0.16),
        *,
        scale: float = 1.5,
        metallic: float = 0.18,
        roughness: float = 0.32,
    ) -> Material:
        return cls(
            albedo=c1,
            albedo2=c2,
            metallic=metallic,
            roughness=roughness,
            pattern=PATTERN_CHECKER,
            pattern_scale=scale,
        )

    @classmethod
    def grid(
        cls,
        c1: tuple[float, float, float] = (0.30, 0.33, 0.40),
        c2: tuple[float, float, float] = (0.16, 0.18, 0.24),
        *,
        scale: float = 2.0,
        metallic: float = 0.35,
        roughness: float = 0.45,
    ) -> Material:
        return cls(
            albedo=c1,
            albedo2=c2,
            metallic=metallic,
            roughness=roughness,
            pattern=PATTERN_GRID,
            pattern_scale=scale,
        )

    @classmethod
    def marble(
        cls,
        c1: tuple[float, float, float] = (0.18, 0.20, 0.26),
        c2: tuple[float, float, float] = (0.45, 0.50, 0.60),
        *,
        scale: float = 2.2,
        metallic: float = 0.25,
        roughness: float = 0.25,
    ) -> Material:
        return cls(
            albedo=c1,
            albedo2=c2,
            metallic=metallic,
            roughness=roughness,
            pattern=PATTERN_MARBLE,
            pattern_scale=scale,
        )

    @classmethod
    def glow(
        cls,
        color: tuple[float, float, float],
        intensity: float = 1.8,
        *,
        roughness: float = 0.2,
    ) -> Material:
        return cls(
            albedo=color,
            metallic=0.1,
            roughness=roughness,
            emissive=(color[0] * intensity, color[1] * intensity, color[2] * intensity),
        )

    @classmethod
    def screen(
        cls,
        texture: Texture,
        *,
        brightness: float = 1.05,
        roughness: float = 0.55,
    ) -> Material:
        """Anti-reflective glowing screen material (3D monitor, security camera, interactive 2D / UI kiosk)."""
        return cls(
            albedo=(0.02, 0.02, 0.03),
            metallic=0.0,
            roughness=roughness,
            emissive=(brightness, brightness, brightness),
            emissive_map=texture,
            emissive_strength=brightness,
        )

    @classmethod
    def mirror(
        cls,
        reflection: Any,
        *,
        tint: tuple[float, float, float] = (0.94, 0.97, 1.0),
        reflectivity: float = 0.94,
    ) -> Material:
        """Optical planar mirror material (exact screen projection + silvered glass).

        Directly accepts a `PlanarReflection` (automatically detects a vertical `plane_z` mirror
        or a horizontal `plane_y` one) or a `RenderTarget`.
        """
        from .texture import PlanarReflection

        if isinstance(reflection, PlanarReflection):
            tex = reflection.target
            pat = PATTERN_MIRROR_Y if (reflection.plane_y is not None and reflection.plane_z is None) else PATTERN_MIRROR_Z
        else:
            tex = reflection
            pat = PATTERN_MIRROR_Z
        return cls(
            albedo=(0.015, 0.02, 0.028),
            metallic=0.08,
            roughness=0.06,
            emissive=(tint[0] * reflectivity, tint[1] * reflectivity, tint[2] * reflectivity),
            pattern=pat,
            emissive_map=tex,
            emissive_strength=reflectivity,
        )

    def pack(
        self,
        *,
        albedo_tid: int = -1,
        emissive_tid: int = -1,
        roughness_tid: int = -1,
    ) -> np.ndarray:
        """Packs the material into 16 `float32` for `shade_pbr` (texture indices in slots 13..15)."""
        return np.array(
            [
                self.albedo[0],
                self.albedo[1],
                self.albedo[2],
                self.metallic,
                self.roughness,
                float(self.pattern),
                self.pattern_scale,
                self.emissive[0],
                self.emissive[1],
                self.emissive[2],
                self.albedo2[0],
                self.albedo2[1],
                self.albedo2[2],
                float(albedo_tid),
                float(emissive_tid),
                float(roughness_tid),
            ],
            dtype=np.float32,
        )

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "albedo": [float(c) for c in self.albedo],
            "metallic": float(self.metallic),
            "roughness": float(self.roughness),
            "emissive": [float(c) for c in self.emissive],
            "pattern": int(self.pattern),
            "pattern_scale": float(self.pattern_scale),
            "albedo2": [float(c) for c in self.albedo2],
            "emissive_strength": float(self.emissive_strength),
        }
        for key, tex in (
            ("albedo_map", self.albedo_map),
            ("emissive_map", self.emissive_map),
            ("roughness_map", self.roughness_map),
        ):
            if tex is not None:
                td = tex.to_dict(base_dir=base_dir)
                if td is not None:
                    d[key] = td
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> Material:
        return cls(
            albedo=tuple(float(x) for x in data.get("albedo", (0.8, 0.8, 0.8))),
            metallic=float(data.get("metallic", 0.0)),
            roughness=float(data.get("roughness", 0.4)),
            emissive=tuple(float(x) for x in data.get("emissive", (0.0, 0.0, 0.0))),
            pattern=int(data.get("pattern", PATTERN_SOLID)),
            pattern_scale=float(data.get("pattern_scale", 2.0)),
            albedo2=tuple(float(x) for x in data.get("albedo2", (0.15, 0.17, 0.22))),
            albedo_map=Texture.from_dict(data["albedo_map"], base_dir=base_dir) if "albedo_map" in data else None,
            emissive_map=Texture.from_dict(data["emissive_map"], base_dir=base_dir) if "emissive_map" in data else None,
            roughness_map=Texture.from_dict(data["roughness_map"], base_dir=base_dir) if "roughness_map" in data else None,
            emissive_strength=float(data.get("emissive_strength", 1.0)),
        )


@dataclass
class DirectionalLight:
    """Directional light (sun) with a directional shadow map."""

    direction: tuple[float, float, float] = (0.55, 0.80, 0.42)
    color: tuple[float, float, float] = (1.0, 0.94, 0.84)
    intensity: float = 3.2
    shadow_extent: float = 10.5
    shadow_distance: float = 18.0
    shadow_target: tuple[float, float, float] = (0.0, 1.0, 0.0)

    @property
    def dir_normalized(self) -> np.ndarray:
        return normalize(self.direction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": [float(x) for x in self.direction],
            "color": [float(x) for x in self.color],
            "intensity": float(self.intensity),
            "shadow_extent": float(self.shadow_extent),
            "shadow_distance": float(self.shadow_distance),
            "shadow_target": [float(x) for x in self.shadow_target],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DirectionalLight:
        return cls(
            direction=tuple(float(x) for x in data.get("direction", (0.55, 0.80, 0.42))),
            color=tuple(float(x) for x in data.get("color", (1.0, 0.94, 0.84))),
            intensity=float(data.get("intensity", 3.2)),
            shadow_extent=float(data.get("shadow_extent", 10.5)),
            shadow_distance=float(data.get("shadow_distance", 18.0)),
            shadow_target=tuple(float(x) for x in data.get("shadow_target", (0.0, 1.0, 0.0))),
        )


@dataclass
class PointLight:
    """Omnidirectional point light."""

    pos: tuple[float, float, float] = (0.0, 3.0, 0.0)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 12.0
    range: float = 0.0  # 0 = unlimited range (quadratic attenuation)
    enabled: bool = True

    def pack(self) -> np.ndarray:
        out = np.zeros(16, dtype=np.float32)
        if not self.enabled:
            return out
        out[0:3] = self.pos
        out[3] = self.intensity
        out[4:7] = self.color
        out[7] = float(LIGHT_POINT)
        out[13] = float(self.range)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "point",
            "pos": [float(x) for x in self.pos],
            "color": [float(x) for x in self.color],
            "intensity": float(self.intensity),
            "range": float(self.range),
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointLight:
        return cls(
            pos=tuple(float(x) for x in data.get("pos", (0.0, 3.0, 0.0))),
            color=tuple(float(x) for x in data.get("color", (1.0, 1.0, 1.0))),
            intensity=float(data.get("intensity", 12.0)),
            range=float(data.get("range", 0.0)),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class SpotLight:
    """Cone spotlight (direction + inner/outer angles in degrees)."""

    pos: tuple[float, float, float] = (0.0, 5.0, 0.0)
    direction: tuple[float, float, float] = (0.0, -1.0, 0.0)
    color: tuple[float, float, float] = (1.0, 0.95, 0.85)
    intensity: float = 24.0
    inner_deg: float = 18.0
    outer_deg: float = 32.0
    range: float = 0.0
    enabled: bool = True

    def look_at(self, target: tuple[float, float, float]) -> None:
        d = normalize(np.asarray(target, dtype=np.float32) - np.asarray(self.pos, dtype=np.float32))
        self.direction = (float(d[0]), float(d[1]), float(d[2]))

    def pack(self) -> np.ndarray:
        out = np.zeros(16, dtype=np.float32)
        if not self.enabled:
            return out
        d = normalize(self.direction)
        ci = math.cos(math.radians(min(self.inner_deg, self.outer_deg)))
        co = math.cos(math.radians(max(self.inner_deg, self.outer_deg)))
        out[0:3] = self.pos
        out[3] = self.intensity
        out[4:7] = self.color
        out[7] = float(LIGHT_SPOT)
        out[8:11] = d
        out[11] = ci
        out[12] = co
        out[13] = float(self.range)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "spot",
            "pos": [float(x) for x in self.pos],
            "direction": [float(x) for x in self.direction],
            "color": [float(x) for x in self.color],
            "intensity": float(self.intensity),
            "inner_deg": float(self.inner_deg),
            "outer_deg": float(self.outer_deg),
            "range": float(self.range),
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpotLight:
        return cls(
            pos=tuple(float(x) for x in data.get("pos", (0.0, 5.0, 0.0))),
            direction=tuple(float(x) for x in data.get("direction", (0.0, -1.0, 0.0))),
            color=tuple(float(x) for x in data.get("color", (1.0, 0.95, 0.85))),
            intensity=float(data.get("intensity", 24.0)),
            inner_deg=float(data.get("inner_deg", 18.0)),
            outer_deg=float(data.get("outer_deg", 32.0)),
            range=float(data.get("range", 0.0)),
            enabled=bool(data.get("enabled", True)),
        )


Light = PointLight | SpotLight


def light_from_dict(data: dict[str, Any]) -> Light:
    t = data.get("type", "point")
    if t == "spot":
        return SpotLight.from_dict(data)
    return PointLight.from_dict(data)


@dataclass
class Environment:
    """Global environment: sun, exponential fog and equirectangular / HDRI skybox."""

    sun: DirectionalLight = field(default_factory=DirectionalLight)
    fog_density: float = 0.0012
    fog_color: tuple[float, float, float] = (0.09, 0.08, 0.16)
    skybox: Texture | None = None
    skybox_intensity: float = 1.0

    @staticmethod
    def studio_skybox(width: int = 512, height: int = 256) -> Texture:
        """Generates a studio / neon equirectangular skybox (light strips, ideal for IBL reflections)."""
        yy, xx = np.indices((height, width), dtype=np.float32)
        u = (xx + 0.5) / width
        v = (yy + 0.5) / height
        phi = u * 2.0 * math.pi
        theta = v * math.pi
        ry = np.cos(theta)
        # Deep-sky gradient + amber horizon + 2 studio softboxes for crisp chrome reflections
        r = np.clip(0.06 + 0.35 * np.power(1.0 - np.abs(ry), 3.0), 0.0, 1.0)
        g = np.clip(0.08 + 0.28 * np.power(1.0 - np.abs(ry), 3.0) + 0.06 * np.maximum(ry, 0.0), 0.0, 1.0)
        b = np.clip(0.18 + 0.38 * np.power(1.0 - np.abs(ry), 2.5) + 0.18 * np.maximum(ry, 0.0), 0.0, 1.0)
        # Soft (Gaussian) light panels: warm + cyan for natural chrome reflections
        soft1 = np.exp(-((phi - 1.1) / 0.36) ** 2 - ((ry - 0.52) / 0.22) ** 2).astype(np.float32)
        soft2 = np.exp(-((phi - 4.3) / 0.42) ** 2 - ((ry - 0.44) / 0.22) ** 2).astype(np.float32)
        r = np.clip(r + soft1 * 0.92 + soft2 * 0.22, 0.0, 1.0)
        g = np.clip(g + soft1 * 0.76 + soft2 * 0.80, 0.0, 1.0)
        b = np.clip(b + soft1 * 0.50 + soft2 * 0.96, 0.0, 1.0)
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, 0] = (r * 255.0).astype(np.uint8)
        rgba[:, :, 1] = (g * 255.0).astype(np.uint8)
        rgba[:, :, 2] = (b * 255.0).astype(np.uint8)
        rgba[:, :, 3] = 255
        tex = Texture.from_array(rgba, bilinear=True)
        tex.recipe = {"kind": "studio_skybox", "width": int(width), "height": int(height)}
        return tex

    @staticmethod
    def daylight_skybox(width: int = 512, height: int = 256) -> Texture:
        """Generates a sunny full-daylight equirectangular skybox (light blue sky, bright horizon, sun, soft clouds)."""
        yy, xx = np.indices((height, width), dtype=np.float32)
        u = (xx + 0.5) / width
        v = (yy + 0.5) / height
        phi = u * 2.0 * math.pi
        theta = v * math.pi
        ry = np.cos(theta)
        rx = np.sin(theta) * np.cos(phi)
        rz = np.sin(theta) * np.sin(phi)

        # Realistic daytime sky gradient: azure at the zenith, bright blue-white at the horizon, light ground below
        up_t = np.clip(ry, 0.0, 1.0)
        hz = np.power(1.0 - np.abs(ry), 4.0)
        r = 0.24 + 0.54 * (1.0 - np.power(up_t, 0.55)) + 0.18 * hz
        g = 0.48 + 0.36 * (1.0 - np.power(up_t, 0.60)) + 0.14 * hz
        b = 0.86 + 0.12 * (1.0 - np.power(up_t, 0.65)) + 0.04 * hz

        # Light neutral ground below the horizon (for realistic reflections from below)
        below = (ry < 0.0).astype(np.float32)
        gt = np.clip(-ry * 2.5, 0.0, 1.0)
        r = r * (1.0 - below) + (0.68 - 0.30 * gt) * below
        g = g * (1.0 - below) + (0.72 - 0.30 * gt) * below
        b = b * (1.0 - below) + (0.78 - 0.28 * gt) * below

        # Soft clouds high in the sky
        cloud = (
            0.5
            + 0.28 * np.sin(phi * 3.0 + ry * 5.0) * np.cos(phi * 2.0 - ry * 4.0)
            + 0.22 * np.sin(phi * 7.0 - ry * 8.0)
        )
        cloud_m = np.clip((cloud - 0.56) * 3.2, 0.0, 1.0) * np.clip((ry - 0.08) * 3.0, 0.0, 1.0) * np.clip(1.0 - ry * 0.6, 0.0, 1.0)
        r = r * (1.0 - cloud_m * 0.55) + 0.98 * (cloud_m * 0.55)
        g = g * (1.0 - cloud_m * 0.55) + 0.99 * (cloud_m * 0.55)
        b = b * (1.0 - cloud_m * 0.55) + 1.00 * (cloud_m * 0.55)

        # Disque solaire lumineux + halo chaud
        sdx, sdy, sdz = 0.48, 0.78, 0.40
        sn = math.sqrt(sdx * sdx + sdy * sdy + sdz * sdz)
        sdot = np.clip((rx * sdx + ry * sdy + rz * sdz) / sn, 0.0, 1.0)
        halo = np.power(sdot, 24.0) * 0.45 + np.power(sdot, 260.0) * 1.8
        r = np.clip(r + halo * 1.0, 0.0, 1.0)
        g = np.clip(g + halo * 0.95, 0.0, 1.0)
        b = np.clip(b + halo * 0.85, 0.0, 1.0)

        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, 0] = (r * 255.0).astype(np.uint8)
        rgba[:, :, 1] = (g * 255.0).astype(np.uint8)
        rgba[:, :, 2] = (b * 255.0).astype(np.uint8)
        rgba[:, :, 3] = 255
        tex = Texture.from_array(rgba, bilinear=True)
        tex.recipe = {"kind": "daylight_skybox", "width": int(width), "height": int(height)}
        return tex

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "sun": self.sun.to_dict(),
            "fog_density": float(self.fog_density),
            "fog_color": [float(x) for x in self.fog_color],
            "skybox_intensity": float(self.skybox_intensity),
        }
        if self.skybox is not None:
            sd = self.skybox.to_dict(base_dir=base_dir)
            if sd is not None:
                d["skybox"] = sd
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> Environment:
        return cls(
            sun=DirectionalLight.from_dict(data.get("sun", {})),
            fog_density=float(data.get("fog_density", 0.0012)),
            fog_color=tuple(float(x) for x in data.get("fog_color", (0.09, 0.08, 0.16))),
            skybox=Texture.from_dict(data["skybox"], base_dir=base_dir) if "skybox" in data else None,
            skybox_intensity=float(data.get("skybox_intensity", 1.0)),
        )
