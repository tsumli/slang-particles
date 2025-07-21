import slangpy as spy
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class CameraParams:
    world_view_projection: npt.NDArray[np.float32]

    def __init__(self):
        self.world_view_projection = np.eye(4, dtype=np.float32)

    def pack(self) -> npt.NDArray[np.float32]:
        return self.world_view_projection.flatten()


def look_at_lh(
    eye: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    up: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    z = target - eye
    z /= np.linalg.norm(z)

    x = np.cross(up, z)
    x /= np.linalg.norm(x)

    y = np.cross(z, x)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = x
    view[1, :3] = y
    view[2, :3] = z
    view[0, 3] = -np.dot(x, eye)
    view[1, 3] = -np.dot(y, eye)
    view[2, 3] = -np.dot(z, eye)

    return view


def perspective_lh(
    fov_y: float, aspect: float, near: float, far: float
) -> npt.NDArray[np.float32]:
    f = 1.0 / np.tan(np.radians(fov_y) * 0.5)

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = far / (far - near)
    proj[2, 3] = 1.0
    proj[3, 2] = -near * far / (far - near)

    return proj


class Camera:
    def __init__(self, device: spy.Device, width: int, height: int):
        self.device = device

        self.camera_params = CameraParams()
        self.world_view_projection_buffer = self.device.create_buffer(
            element_count=1,
            struct_size=64,  # 4x4 matrix = 16 floats * 4 bytes = 64 bytes
            usage=spy.BufferUsage.shader_resource,
            label="camera",
        )

        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.position = np.array([2000.0, 2000.0, 2000.0], dtype=np.float32)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.fov = 60.0
        self.recompute()

    def recompute(self):
        self.aspect_ratio = float(self.width) / float(self.height)

        # Calculate camera basis vectors
        fwd = self.target - self.position
        fwd = fwd / np.linalg.norm(fwd)

        right = np.cross(fwd, self.up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, fwd)
        up = up / np.linalg.norm(up)

        # Store basis vectors for controller
        self.fwd = fwd
        self.right = right
        self.up = up

        # Create world-view-projection matrix
        view_matrix = look_at_lh(self.position, self.target, self.up.astype(np.float32))
        distance_to_origin = float(np.linalg.norm(self.position))
        far_plane = max(1000.0, distance_to_origin * 10.0)
        proj_matrix = perspective_lh(self.fov, self.aspect_ratio, 0.1, far_plane)
        self.world_view_projection = proj_matrix @ view_matrix
        self.camera_params.world_view_projection = (
            self.world_view_projection.T
        )  # column major

        # Copy the flattened matrix data to the buffer
        matrix_data = self.camera_params.pack()
        self.world_view_projection_buffer.copy_from_numpy(matrix_data)

    def get_world_view_projection(self) -> np.ndarray:
        """Get the current world-view-projection matrix"""
        return self.camera_params.world_view_projection


class CameraController:
    MOVE_KEYS = {
        spy.KeyCode.a: np.array([-10.0, 0.0, 0.0], dtype=np.float32),
        spy.KeyCode.d: np.array([10.0, 0.0, 0.0], dtype=np.float32),
        spy.KeyCode.e: np.array([0.0, -10.0, 0.0], dtype=np.float32),
        spy.KeyCode.q: np.array([0.0, 10.0, 0.0], dtype=np.float32),
        spy.KeyCode.w: np.array([0.0, 0.0, 10.0], dtype=np.float32),
        spy.KeyCode.s: np.array([0.0, 0.0, -10.0], dtype=np.float32),
    }
    MOVE_SHIFT_FACTOR = 50.0

    def __init__(self, camera: Camera):
        self.camera = camera
        self.left_mouse_down = False
        self.right_mouse_down = False
        self.mouse_pos = np.zeros(2, dtype=np.float32)
        self.key_state = {k: False for k in CameraController.MOVE_KEYS.keys()}
        self.shift_down = False

        self.move_delta: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
        self.rotate_delta: npt.NDArray[np.float32] = np.zeros(2, dtype=np.float32)

        self.move_speed = 1.0
        self.rotate_speed = 0.001

    def update(self, dt: float):
        changed = False
        position = self.camera.position.copy()
        fwd = self.camera.fwd.copy()
        up = self.camera.up.copy()
        right = self.camera.right.copy()

        # Move
        if np.linalg.norm(self.move_delta) > 0:
            offset = right * self.move_delta[0]
            offset += up * self.move_delta[1]
            offset += fwd * self.move_delta[2]
            factor = CameraController.MOVE_SHIFT_FACTOR if self.shift_down else 1.0
            offset *= self.move_speed * factor * dt
            position += offset
            changed = True

        # Rotate
        if np.linalg.norm(self.rotate_delta) > 0:
            yaw = np.arctan2(fwd[2], fwd[0])
            pitch = np.arcsin(fwd[1])
            yaw -= self.rotate_speed * self.rotate_delta[0]
            pitch -= self.rotate_speed * self.rotate_delta[1]
            pitch = np.clip(pitch, -np.pi * 0.49, np.pi * 0.49)
            fwd = np.array(
                [
                    np.cos(yaw) * np.cos(pitch),
                    np.sin(pitch),
                    np.sin(yaw) * np.cos(pitch),
                ],
                dtype=np.float32,
            )
            self.rotate_delta = np.zeros(2, dtype=np.float32)
            changed = True

        if changed:
            self.camera.position = position
            self.camera.target = position + fwd
            self.camera.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.camera.recompute()

        return changed

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if event.is_key_press() or event.is_key_release():
            down = event.is_key_press()
            if event.key in CameraController.MOVE_KEYS:
                self.key_state[event.key] = down
            elif event.key == spy.KeyCode.left_shift:
                self.shift_down = down
        self.move_delta = np.zeros(3, dtype=np.float32)
        for key, state in self.key_state.items():
            if state:
                self.move_delta += CameraController.MOVE_KEYS[key]

    def on_mouse_event(self, event: spy.MouseEvent):
        self.rotate_delta = np.zeros(2, dtype=np.float32)
        if event.is_button_down() and event.button == spy.MouseButton.left:
            self.left_mouse_down = True
        if event.is_button_up() and event.button == spy.MouseButton.left:
            self.left_mouse_down = False
        if event.is_button_down() and event.button == spy.MouseButton.right:
            self.right_mouse_down = True
        if event.is_button_up() and event.button == spy.MouseButton.right:
            self.right_mouse_down = False

        if event.is_move():
            mouse_delta = np.array(event.pos) - self.mouse_pos
            if self.left_mouse_down:
                self.rotate_delta = mouse_delta
            self.mouse_pos = event.pos
