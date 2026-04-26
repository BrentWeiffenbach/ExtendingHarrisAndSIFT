"""Interactive parameter playground for Harris/SIFT detectors.

Opens napari with a control panel where you can:
  - Select dimension (2D / 3D-Voxel / 3D-PointCloud)
  - Select algorithm (SIFT, Harris, etc.)
  - Select data source
  - Tune all algorithm parameters via live widgets
  - Toggle intermediate pipeline outputs as napari layers

Usage:
  python playground.py
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import magicgui.widgets as mw
import napari
import numpy as np
from napari.qt.threading import thread_worker
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.common.io import (
    ModelNetLoader,
    SyntheticVoxelLoader,
    load_image,
    load_pointcloud,
)
from src.common.visualization import rasterize_extrema_blobs_3d
from src.pointcloud.params import SIFTRadiiPCParams, SIFTVoxelPCParams
from src.pointcloud.sift_pc import SIFTRadiiPC, SIFTVoxelPC
from src.voxel.harris3d import Harris3DVoxel
from src.voxel.params import Harris3DParams, SIFT2DParams, SIFT3DParams
from src.voxel.sift2d import SIFT2D
from src.voxel.sift3d import SIFT3DVoxel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYNTH_SHAPES = ["cone", "cube", "cuboid", "cylinder", "pyramid", "sphere", "torus"]

_DIM_CHOICES = ["2D", "3D-Voxel", "3D-PointCloud"]

_ALGO_CHOICES: dict[str, list[str]] = {
    "2D": ["SIFT"],
    "3D-Voxel": ["Harris", "SIFT"],
    "3D-PointCloud": ["SIFT-Radii", "SIFT-Voxel"],
}

_PARAMS_CLASS: dict[str, type] = {
    "2D/SIFT": SIFT2DParams,
    "3D-Voxel/Harris": Harris3DParams,
    "3D-Voxel/SIFT": SIFT3DParams,
    "3D-PointCloud/SIFT-Radii": SIFTRadiiPCParams,
    "3D-PointCloud/SIFT-Voxel": SIFTVoxelPCParams,
}

_PG_TAG = "pg"


def _algo_key(dimension: str, algorithm: str) -> str:
    return f"{dimension}/{algorithm}"


# ---------------------------------------------------------------------------
# Parameter widget helpers
# ---------------------------------------------------------------------------


def _make_param_widgets(
    params_instance: Any, prefix: str = ""
) -> tuple[list, dict[str, Any]]:
    """Build magicgui widgets from a dataclass instance.

    Returns (widget_list, widget_dict).  widget_dict maps name -> widget for
    reading values back. Tuple fields produce N sub-widgets named {field}_{i}.
    Nested dataclasses are expanded inline with {parent}.{child} keys.
    """
    widgets: list = []
    widget_dict: dict[str, Any] = {}

    for f in dataclasses.fields(params_instance):
        val = getattr(params_instance, f.name)
        key = f"{prefix}.{f.name}" if prefix else f.name
        label = f.name.replace("_", " ")

        if isinstance(val, bool):
            w = mw.CheckBox(name=key, value=val, label=label)
            widgets.append(w)
            widget_dict[key] = w

        elif isinstance(val, int):
            w = mw.SpinBox(name=key, value=val, label=label, min=0, max=100_000)
            widgets.append(w)
            widget_dict[key] = w

        elif isinstance(val, float):
            step = max(abs(val) * 0.1, 1e-9)
            w = mw.FloatSpinBox(
                name=key,
                value=val,
                label=label,
                step=step,
                min=-1e9,
                max=1e9,
            )
            widgets.append(w)
            widget_dict[key] = w

        elif isinstance(val, str):
            w = mw.LineEdit(name=key, value=val, label=label)
            widgets.append(w)
            widget_dict[key] = w

        elif isinstance(val, (tuple, list)):
            for i, sub_val in enumerate(val):
                sub_key = f"{key}_{i}"
                sub_lbl = f"{f.name}[{i}]"
                if isinstance(sub_val, float):
                    step = max(abs(sub_val) * 0.1, 1e-9)
                    w = mw.FloatSpinBox(
                        name=sub_key,
                        value=sub_val,
                        label=sub_lbl,
                        step=step,
                        min=-1e9,
                        max=1e9,
                    )
                else:
                    w = mw.SpinBox(
                        name=sub_key,
                        value=int(sub_val),
                        label=sub_lbl,
                        min=0,
                        max=100_000,
                    )
                widgets.append(w)
                widget_dict[sub_key] = w

        elif dataclasses.is_dataclass(val):
            widgets.append(mw.Label(value=f"── {f.name} ──"))
            sub_widgets, sub_dict = _make_param_widgets(val, prefix=key)
            widgets.extend(sub_widgets)
            widget_dict.update(sub_dict)

    return widgets, widget_dict


def _read_params(
    widget_dict: dict[str, Any], params_class: type, prefix: str = ""
) -> Any:
    """Reconstruct a params dataclass from widget_dict values."""
    kwargs: dict[str, Any] = {}

    for f in dataclasses.fields(params_class):
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default = f.default_factory()  # type: ignore[misc]
        else:
            default = None

        key = f"{prefix}.{f.name}" if prefix else f.name

        if isinstance(default, bool):
            kwargs[f.name] = (
                bool(widget_dict[key].value) if key in widget_dict else default
            )
        elif isinstance(default, int):
            kwargs[f.name] = (
                int(widget_dict[key].value) if key in widget_dict else default
            )
        elif isinstance(default, float):
            kwargs[f.name] = (
                float(widget_dict[key].value) if key in widget_dict else default
            )
        elif isinstance(default, str):
            kwargs[f.name] = (
                str(widget_dict[key].value) if key in widget_dict else default
            )
        elif isinstance(default, (tuple, list)):
            parts = []
            for i, sub_val in enumerate(default):
                sub_key = f"{key}_{i}"
                raw = widget_dict[sub_key].value if sub_key in widget_dict else sub_val
                parts.append(type(sub_val)(raw))
            kwargs[f.name] = type(default)(parts)
        elif dataclasses.is_dataclass(default):
            kwargs[f.name] = _read_params(widget_dict, type(default), prefix=key)

    return params_class(**kwargs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _data_sources(dimension: str) -> list[str]:
    if dimension == "2D":
        paths = sorted(Path("data/2d").glob("*.jpg"))
        return [p.stem for p in paths] or ["(no images)"]
    elif dimension == "3D-Voxel":
        return _SYNTH_SHAPES + [f"modelnet_{i}" for i in range(10)]
    else:
        return _SYNTH_SHAPES


def _load_data(dimension: str, source_name: str) -> np.ndarray:
    if dimension == "2D":
        matches = list(Path("data/2d").glob(f"{source_name}.jpg"))
        if not matches:
            raise FileNotFoundError(f"No image found: {source_name}.jpg")
        return load_image(str(matches[0]))

    elif dimension == "3D-Voxel":
        if source_name.startswith("modelnet_"):
            idx = int(source_name.split("_")[1])
            loader = ModelNetLoader(
                "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
            )
            return loader.load_by_index(idx).astype(np.float32)
        return SyntheticVoxelLoader().load_by_name(source_name).astype(np.float32)

    else:  # 3D-PointCloud
        pcd = load_pointcloud(f"data/Pointcloud/synthetic/{source_name}.ply")
        pts = np.asarray(pcd.points, dtype=np.float32)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        rng = np.where(hi - lo > 0, hi - lo, 1.0)
        return (pts - lo) / rng


# ---------------------------------------------------------------------------
# Detector runner
# ---------------------------------------------------------------------------


def _run_detector(
    dimension: str, algorithm: str, params: Any, data: np.ndarray
) -> dict:
    """Run the selected detector; always uses run() for intermediary access."""
    key = _algo_key(dimension, algorithm)

    if key == "2D/SIFT":
        result = SIFT2D(params).run(data)
        kp = (
            result.extrema_global[:, :2].astype(np.float32)
            if result.extrema_global.shape[0] > 0
            else np.empty((0, 2), np.float32)
        )
        return {
            "keypoints": kp,
            "raw": data,
            "intermediaries": {
                "gaussian_pyramid": result.gaussian_pyramid,
                "dog_pyramid": result.dog_pyramid,
            },
        }

    elif key == "3D-Voxel/Harris":
        detector = Harris3DVoxel(params)
        kp = detector.detect(data).astype(np.float32)
        return {
            "keypoints": kp,
            "raw": data,
            "intermediaries": {
                "response": detector.last_response.astype(np.float32),
            },
        }

    elif key == "3D-Voxel/SIFT":
        result = SIFT3DVoxel(params).run(data)
        extrema = (
            result.extrema_global.astype(np.float32)
            if result.extrema_global.shape[0] > 0
            else np.empty((0, 7), np.float32)
        )
        kp = (
            extrema[:, [2, 1, 0]]
            if extrema.shape[0] > 0
            else np.empty((0, 3), np.float32)
        )
        return {
            "keypoints": kp,
            "raw": data,
            "extrema_global": extrema,
            "intermediaries": {
                "gaussian_pyramid": result.gaussian_pyramid,
                "dog_pyramid": result.dog_pyramid,
            },
        }

    elif key == "3D-PointCloud/SIFT-Radii":
        result = SIFTRadiiPC(params).run(data)
        kp5 = (
            result.keypoints.astype(np.float32)
            if result.keypoints.shape[0] > 0
            else np.empty((0, 5), np.float32)
        )
        return {
            "keypoints": kp5[:, :3],
            "keypoints_full": kp5,
            "raw": data,
            "intermediaries": {
                "density_pyramid": result.density_pyramid,
                "dog_pyramid": result.dog_pyramid,
                "radii_pyramid": result.radii_pyramid,
                "points_per_octave": result.points_per_octave,
                "signal_name": "KDE",
            },
        }

    elif key == "3D-PointCloud/SIFT-Voxel":
        result = SIFTVoxelPC(params).run(data)
        kp = result["keypoints"].astype(np.float32)
        sift3d = result["sift3d_result"]
        return {
            "keypoints": kp,
            "raw": data,
            "intermediaries": {
                "voxel_volume": result["volume"].astype(np.float32),
                "gaussian_pyramid": sift3d.gaussian_pyramid,
                "dog_pyramid": sift3d.dog_pyramid,
            },
        }

    raise ValueError(f"Unknown combination: {key}")


# ---------------------------------------------------------------------------
# Layer management
# ---------------------------------------------------------------------------


def _clear_pg_layers(viewer: napari.Viewer) -> None:
    to_remove = [layer for layer in viewer.layers if layer.metadata.get(_PG_TAG)]
    for layer in to_remove:
        viewer.layers.remove(layer)


def _pg_meta() -> dict:
    return {_PG_TAG: True}


def _update_layers(
    viewer: napari.Viewer,
    result: dict,
    dimension: str,
    algorithm: str,
    show_inter: bool,
) -> None:
    _clear_pg_layers(viewer)
    raw = result["raw"]
    kp = result["keypoints"]
    inter = result["intermediaries"]
    pg = _pg_meta()

    if dimension == "2D":
        viewer.dims.ndisplay = 2
        viewer.add_image(raw, name="image", colormap="gray", metadata=pg)
        if kp.shape[0] > 0:
            viewer.add_points(
                kp[:, :2],
                name="keypoints",
                size=6,
                face_color="red",
                symbol="disc",
                metadata=pg,
            )
        if show_inter:
            _add_inter_2d(viewer, inter, pg)

    elif dimension == "3D-Voxel":
        viewer.dims.ndisplay = 3
        viewer.add_image(
            raw.astype(np.float32),
            name="volume",
            colormap="gray",
            rendering="mip",
            depiction="volume",
            metadata=pg,
        )
        extrema_global = result.get("extrema_global")
        if (
            algorithm == "SIFT"
            and extrema_global is not None
            and extrema_global.shape[0] > 0
        ):
            blob_labels, centers = rasterize_extrema_blobs_3d(
                raw.shape, extrema_global, radius_factor=1.0, max_blobs=2000
            )
            viewer.add_labels(
                blob_labels, name="keypoint_blobs", opacity=0.55, metadata=pg
            )
            if centers.shape[0] > 0:
                viewer.add_points(
                    centers, name="keypoints", size=4, face_color="yellow", metadata=pg
                )
        elif kp.shape[0] > 0:
            kp_zyx = kp[:, [2, 1, 0]].astype(np.float32)
            viewer.add_points(
                kp_zyx, name="keypoints", size=4, face_color="red", metadata=pg
            )
        if show_inter:
            _add_inter_3d_voxel(viewer, inter, algorithm, pg)

    else:  # 3D-PointCloud
        viewer.dims.ndisplay = 3
        pc_zyx = raw[:, [2, 1, 0]].astype(np.float32)
        viewer.add_points(
            pc_zyx,
            name="point_cloud",
            size=0.008,
            face_color="white",
            opacity=0.2,
            metadata=pg,
        )
        if kp.shape[0] > 0:
            kp_zyx = kp[:, [2, 1, 0]].astype(np.float32)
            viewer.add_points(
                kp_zyx,
                name="keypoints",
                size=0.05,
                face_color="red",
                opacity=0.95,
                metadata=pg,
            )
        if show_inter:
            _add_inter_3d_pc(viewer, inter, pg)


def _add_inter_2d(viewer: napari.Viewer, inter: dict, pg: dict) -> None:
    for o, octave in enumerate(inter.get("gaussian_pyramid", [])):
        for s, img in enumerate(octave):
            viewer.add_image(
                img.astype(np.float32),
                name=f"gauss_o{o}_s{s}",
                colormap="gray",
                visible=False,
                metadata=pg,
            )
    for o, octave in enumerate(inter.get("dog_pyramid", [])):
        for d, img in enumerate(octave):
            viewer.add_image(
                img.astype(np.float32),
                name=f"dog_o{o}_d{d}",
                colormap="bwr",
                visible=False,
                metadata=pg,
            )


def _add_inter_3d_voxel(
    viewer: napari.Viewer, inter: dict, algorithm: str, pg: dict
) -> None:
    if algorithm == "Harris":
        resp = inter.get("response")
        if resp is not None:
            viewer.add_image(
                resp.astype(np.float32),
                name="harris_response",
                colormap="viridis",
                opacity=0.6,
                visible=False,
                metadata=pg,
            )
        return

    for o, octave in enumerate(inter.get("gaussian_pyramid", [])):
        for s, vol in enumerate(octave):
            viewer.add_image(
                vol.astype(np.float32),
                name=f"gauss_o{o}_s{s}",
                colormap="gray",
                rendering="mip",
                depiction="volume",
                visible=False,
                metadata=pg,
            )
    for o, octave in enumerate(inter.get("dog_pyramid", [])):
        for d, vol in enumerate(octave):
            viewer.add_image(
                vol.astype(np.float32),
                name=f"dog_o{o}_d{d}",
                colormap="bwr",
                rendering="mip",
                depiction="volume",
                visible=False,
                metadata=pg,
            )


def _add_inter_3d_pc(viewer: napari.Viewer, inter: dict, pg: dict) -> None:
    voxel_vol = inter.get("voxel_volume")
    if voxel_vol is not None:
        viewer.add_image(
            voxel_vol.astype(np.float32),
            name="voxel_occupancy",
            colormap="gray",
            rendering="mip",
            depiction="volume",
            visible=False,
            metadata=pg,
        )
        for o, octave in enumerate(inter.get("gaussian_pyramid", [])):
            for s, vol in enumerate(octave):
                viewer.add_image(
                    vol.astype(np.float32),
                    name=f"gauss_o{o}_s{s}",
                    colormap="gray",
                    rendering="mip",
                    depiction="volume",
                    visible=False,
                    metadata=pg,
                )
        for o, octave in enumerate(inter.get("dog_pyramid", [])):
            for d, vol in enumerate(octave):
                viewer.add_image(
                    vol.astype(np.float32),
                    name=f"dog_o{o}_d{d}",
                    colormap="bwr",
                    rendering="mip",
                    depiction="volume",
                    visible=False,
                    metadata=pg,
                )
        return

    signal_name = inter.get("signal_name", "signal")
    pts_per_oct = inter.get("points_per_octave", [])
    density_pyr = inter.get("density_pyramid", [])
    dog_pyr = inter.get("dog_pyramid", [])
    radii_pyr = inter.get("radii_pyramid", [])
    short_sig = signal_name.replace(" ", "_")

    first_density = True
    for o, (oct_signals, pts) in enumerate(zip(density_pyr, pts_per_oct)):
        pts_zyx = pts[:, [2, 1, 0]].astype(np.float32)
        oct_radii = radii_pyr[o] if o < len(radii_pyr) else [None] * len(oct_signals)
        for s, sig in enumerate(oct_signals):
            sig = np.asarray(sig, dtype=np.float32)
            d_min, d_max = float(sig.min()), float(sig.max())
            if d_max <= d_min:
                d_max = d_min + 1.0
            r = oct_radii[s]
            lbl = f"{short_sig} o={o} s={s}" + (f" r={r:.3f}" if r is not None else "")
            viewer.add_points(
                pts_zyx,
                name=lbl,
                features={"signal": sig},
                face_color="signal",
                face_colormap="plasma",
                face_contrast_limits=(d_min, d_max),
                size=0.015,
                opacity=0.85,
                visible=first_density,
                metadata=pg,
            )
            first_density = False

    for o, (oct_dogs, pts) in enumerate(zip(dog_pyr, pts_per_oct)):
        pts_zyx = pts[:, [2, 1, 0]].astype(np.float32)
        for d, dog in enumerate(oct_dogs):
            dog = np.asarray(dog, dtype=np.float32)
            vmax = float(np.percentile(np.abs(dog), 99)) or 1.0
            viewer.add_points(
                pts_zyx,
                name=f"DoG o={o} d={d}",
                features={"dog": dog},
                face_color="dog",
                face_colormap="bwr",
                face_contrast_limits=(-vmax, vmax),
                size=0.015,
                opacity=0.85,
                visible=False,
                metadata=pg,
            )


# ---------------------------------------------------------------------------
# PlaygroundPanel — dock widget
# ---------------------------------------------------------------------------


class PlaygroundPanel(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = viewer
        self._widget_dict: dict[str, Any] = {}
        self._params_mgui_container: mw.Container | None = None

        self._build_ui()
        # Initialize combos without triggering double-rebuild
        self._dim_combo.blockSignals(True)
        self._dim_combo.setCurrentIndex(0)
        self._dim_combo.blockSignals(False)
        self._on_dim_changed()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # ── Selection ──────────────────────────────────────────
        sel = QGroupBox("Selection")
        sel_lay = QVBoxLayout(sel)
        sel_lay.setSpacing(3)

        sel_lay.addWidget(QLabel("Dimension:"))
        self._dim_combo = QComboBox()
        self._dim_combo.addItems(_DIM_CHOICES)
        sel_lay.addWidget(self._dim_combo)

        sel_lay.addWidget(QLabel("Algorithm:"))
        self._algo_combo = QComboBox()
        sel_lay.addWidget(self._algo_combo)

        sel_lay.addWidget(QLabel("Data source:"))
        self._data_combo = QComboBox()
        sel_lay.addWidget(self._data_combo)

        root.addWidget(sel)

        # ── Options ─────────────────────────────────────────────
        opt = QGroupBox("Options")
        opt_lay = QVBoxLayout(opt)
        opt_lay.setSpacing(3)
        self._show_inter_cb = QCheckBox("Show intermediary layers")
        self._auto_run_cb = QCheckBox("Auto-run on parameter change")
        opt_lay.addWidget(self._show_inter_cb)
        opt_lay.addWidget(self._auto_run_cb)
        root.addWidget(opt)

        # ── Run / status ─────────────────────────────────────────
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setMinimumHeight(30)
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        run_row.addWidget(self._run_btn, 2)
        run_row.addWidget(self._status_lbl, 1)
        root.addLayout(run_row)

        # ── Parameters scroll area ───────────────────────────────
        params_box = QGroupBox("Parameters")
        params_lay = QVBoxLayout(params_box)
        params_lay.setContentsMargins(0, 4, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        params_lay.addWidget(self._scroll)
        root.addWidget(params_box, stretch=1)

        # ── Signals ─────────────────────────────────────────────
        self._dim_combo.currentIndexChanged.connect(self._on_dim_changed)
        self._algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        self._run_btn.clicked.connect(self._on_run_clicked)

    # ── Combo change handlers ──────────────────────────────────

    def _on_dim_changed(self) -> None:
        dim = self._dim_combo.currentText()

        self._algo_combo.blockSignals(True)
        self._algo_combo.clear()
        self._algo_combo.addItems(_ALGO_CHOICES.get(dim, []))
        self._algo_combo.blockSignals(False)

        self._data_combo.clear()
        self._data_combo.addItems(_data_sources(dim))

        self._rebuild_params_panel()

    def _on_algo_changed(self) -> None:
        self._rebuild_params_panel()

    def _rebuild_params_panel(self) -> None:
        dim = self._dim_combo.currentText()
        algo = self._algo_combo.currentText()
        params_class = _PARAMS_CLASS.get(_algo_key(dim, algo))
        if params_class is None:
            return

        widget_list, widget_dict = _make_param_widgets(params_class())
        self._widget_dict = widget_dict

        container = mw.Container(widgets=widget_list, labels=True)
        self._params_mgui_container = container

        # Wrap in a plain QWidget (scroll area needs a widget, not a Container)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(container.native)
        lay.addStretch()
        self._scroll.setWidget(inner)

        # Wire each param widget to the auto-run gate
        for w in widget_dict.values():
            try:
                w.changed.connect(self._maybe_auto_run)
            except Exception:
                pass

    # ── Run ───────────────────────────────────────────────────

    def _maybe_auto_run(self, *_) -> None:
        if self._auto_run_cb.isChecked() and self._run_btn.isEnabled():
            self._on_run_clicked()

    def _on_run_clicked(self, *_) -> None:
        dim = self._dim_combo.currentText()
        algo = self._algo_combo.currentText()
        source = self._data_combo.currentText()
        params_class = _PARAMS_CLASS.get(_algo_key(dim, algo))
        if params_class is None:
            return

        params = _read_params(self._widget_dict, params_class)
        show_inter = self._show_inter_cb.isChecked()
        viewer = self._viewer

        self._run_btn.setEnabled(False)
        self._status_lbl.setText("Loading…")

        try:
            data = _load_data(dim, source)
        except Exception as exc:
            self._status_lbl.setText(f"Load error: {exc}")
            self._run_btn.setEnabled(True)
            return

        self._status_lbl.setText("Running…")

        @thread_worker
        def _worker():
            return _run_detector(dim, algo, params, data)

        def _on_done(result: dict) -> None:
            _update_layers(viewer, result, dim, algo, show_inter)
            n = result["keypoints"].shape[0]
            self._status_lbl.setText(f"Done: {n} keypoints")
            self._run_btn.setEnabled(True)

        def _on_error(exc_info: tuple) -> None:
            self._status_lbl.setText(f"Error: {exc_info[1]}")
            self._run_btn.setEnabled(True)

        worker = _worker()
        worker.returned.connect(_on_done)
        worker.errored.connect(_on_error)
        worker.start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    viewer = napari.Viewer(title="Detector Playground", ndisplay=3)
    panel = PlaygroundPanel(viewer)
    panel.setMinimumWidth(300)
    viewer.window.add_dock_widget(panel, area="right", name="Playground")
    napari.run()


if __name__ == "__main__":
    main()
