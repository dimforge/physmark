"""Instanced, depth-buffered rendering of large saved trajectories (never timed)."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import findfont


def render_large(record, out, scene, frames, bodies, bounds, geometry, rotation, color, version, no_video):
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

    def rgb(value):
        return tuple(int(value[i:i+2], 16) / 255 for i in (1, 3, 5))

    def mesh(faces):
        faces = np.asarray(faces, dtype=np.float32)
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(faces.reshape(-1, 3), deep=True))
        cells = vtk.vtkCellArray()
        cells.SetData(
            numpy_to_vtkIdTypeArray(np.arange(0, faces.shape[0] * faces.shape[1] + 1, faces.shape[1], dtype=np.int64), deep=True),
            numpy_to_vtkIdTypeArray(np.arange(faces.shape[0] * faces.shape[1], dtype=np.int64), deep=True),
        )
        data = vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetPolys(cells)
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(data)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(clean.GetOutputPort())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SetFeatureAngle(35)
        normals.Update()
        result = vtk.vtkPolyData()
        result.DeepCopy(normals.GetOutput())
        return result

    def actor(mapper, color):
        obj = vtk.vtkActor()
        obj.SetMapper(mapper)
        prop = obj.GetProperty()
        prop.SetColor(rgb(color))
        prop.SetAmbient(0.30)
        prop.SetDiffuse(0.70)
        prop.SetSpecular(0.05)
        prop.SetOpacity(1.0)
        renderer.AddActor(obj)
        return obj

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(rgb("#101923"))
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(720, 540)
    window.SetMultiSamples(4)
    window.AddRenderer(renderer)
    groups = {}
    for i, body in enumerate(bodies):
        groups.setdefault((body["shape"], tuple(body["size"]), json.dumps(body.get("parts"), sort_keys=True)), []).append(i)
    instances = []
    for indices in groups.values():
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(len(indices))
        data = vtk.vtkPolyData()
        data.SetPoints(points)
        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetInputData(data)
        mapper.SetSourceData(mesh(geometry(bodies[indices[0]])))
        mapper.SetOrientationModeToQuaternion()
        mapper.SetOrientationArray("orientation")
        mapper.OrientOn()
        mapper.ScalingOff()
        mapper.ScalarVisibilityOff()
        actor(mapper, color)
        instances.append((np.array(indices), points, data))
    lo, hi = bounds
    center = (lo + hi) / 2
    span = float(max(hi - lo))
    if scene["scenario"] != "free_fall":
        floor = [[[lo[0]-span, lo[1]-span, 0], [hi[0]+span, lo[1]-span, 0],
                  [hi[0]+span, hi[1]+span, 0], [lo[0]-span, hi[1]+span, 0]]]
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh(floor))
        actor(mapper, "#263746")
    for body in scene["bodies"]:
        if not body["dynamic"] and body["position"][2] > 0:
            faces = [v @ rotation(body["rotation"]).T + body["position"] for v in geometry(body)]
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh(faces))
            boundary_actor = actor(mapper, "#64778a")
            if scene.get("mpm"):
                boundary_actor.GetProperty().SetRepresentationToWireframe()
    camera = renderer.GetActiveCamera()
    elevation, azimuth = np.radians([23, -62])
    direction = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    camera.SetPosition(*(center + 3 * span * direction))
    camera.SetFocalPoint(*center)
    camera.SetViewUp(0, 0, 1)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(span * 0.70)
    camera.SetClippingRange(span * 0.01, span * 10)
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    font = ImageFont.truetype(findfont("DejaVu Sans"), 15)
    media = out / record["directory"]
    has_video = not no_video and bool(shutil.which("ffmpeg"))

    def draw(frame):
        poses = np.asarray(frame["poses"], dtype=np.float64)
        if poses.shape != (len(bodies), 7) or not np.isfinite(poses).all():
            raise ValueError("invalid large-scene poses")
        for indices, points, data in instances:
            selected = poses[indices]
            points.SetData(numpy_to_vtk(np.ascontiguousarray(selected[:, :3]), deep=True))
            # PhysMark XYZW -> VTK quaternion WXYZ. No Euler conversion.
            orientation = numpy_to_vtk(np.ascontiguousarray(selected[:, [6, 3, 4, 5]]), deep=True)
            orientation.SetName("orientation")
            data.GetPointData().RemoveArray("orientation")
            data.GetPointData().AddArray(orientation)
            data.Modified()
        window.Render()
        capture.Modified()
        capture.Update()
        pixels = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(540, 720, 3)
        image = Image.fromarray(pixels[::-1].copy())
        painter = ImageDraw.Draw(image)
        painter.rectangle((0, 0, 720, 57), fill="#101923")
        painter.multiline_text((24, 12),
            f"{record['engine'].upper()} / {record['backend']}  ·  {scene['scenario'].replace('_', ' ')}  ·  n={len(bodies):,}\n"
            f"World 0  ·  t={frame['step'] * scene['dt']:.3f} s  ·  untimed replay",
            font=font, fill="white", spacing=3)
        return image

    try:
        draw(frames[-1]).save(media / "simulation.png")
        if has_video:
            duration = frames[-1]["step"] * scene["dt"]
            times = np.array([f["step"] * scene["dt"] for f in frames])
            indices = np.searchsorted(times, np.linspace(0, duration, max(2, int(duration*30)+1))).clip(max=len(frames)-1)
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
                   "-s", "720x540", "-pix_fmt", "rgb24", "-r", "30", "-i", "-", "-an",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(media / "simulation.mp4")]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                previous, pixels = None, None
                for index in indices:
                    if index != previous:
                        pixels = draw(frames[index]).tobytes()
                        previous = index
                    process.stdin.write(pixels)
            finally:
                process.stdin.close()
                status = process.wait()
            if status:
                raise RuntimeError(f"ffmpeg exited {status}")
    finally:
        window.Finalize()
    (media / "render-version.json").write_text(json.dumps({
        "png": version, "mp4": version if has_video else None,
        "renderer": "vtk_instanced", "vtk_version": vtk.vtkVersion.GetVTKVersion(),
        ("rendered_particles" if scene.get("mpm") else "rendered_bodies"): len(bodies),
        "sampling": "all particles in world 0" if scene.get("mpm") else "all bodies in world 0"
    }))
