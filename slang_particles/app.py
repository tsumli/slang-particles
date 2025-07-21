import slangpy as spy
import time
import enum
import numpy as np

from slang_particles.camera import CameraController, Camera
from slang_particles.timer import Timer
from slang_particles.particles import Particle
from slang_particles.shader import SHADER_DIR
from slang_particles.particle_shader import ParticleShader
from slang_particles.renderer import ParticleRenderer, PrepareFrame


class RenderType(enum.Enum):
    GREEDY = enum.auto()
    TILED = enum.auto()


def create_particles(
    num_particles=100000, radius=15.0, center=np.zeros(3)
) -> list[Particle]:
    particles = []

    for i in range(num_particles):
        x = np.random.uniform(-1.0, 1.0) * radius
        y = np.random.uniform(-1.0, 1.0) * radius
        z = np.random.uniform(-0.1, 0.1) * radius

        position = np.array([x, y, z], dtype=np.float32) + center

        distance_from_center = np.linalg.norm(position)
        orbital_speed = 20.0 * (1.0 - distance_from_center / 100.0) + 5.0

        if distance_from_center > 0.1:
            tangent = np.array([-position[1], position[0], 0.0]) / distance_from_center
            orbital_velocity = tangent * orbital_speed
        else:
            orbital_velocity = np.array([0.0, 0.0, 0.0])

        random_velocity = np.random.normal(0, 10.0, 3)

        velocity = orbital_velocity + random_velocity
        velocity = velocity.astype(np.float32)

        mass = np.random.uniform(0.5, 2.0)

        particles.append(Particle(mass=mass, position=position, velocity=velocity))

    return particles


class App:
    def __init__(self, render_type: RenderType, num_particles: int):
        super().__init__()
        self.window = spy.Window(
            width=1920, height=1080, title="Particle viewer", resizable=True
        )
        self.device = spy.Device(
            enable_debug_layers=False,
            compiler_options={
                "include_paths": [SHADER_DIR],
            },
        )
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(
            width=self.window.width, height=self.window.height, vsync=False
        )

        self.render_type = render_type

        self.output_texture: spy.Texture = None  # type: ignore (will be set immediately)

        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        self.camera = Camera(self.device, self.window.width, self.window.height)
        self.camera_controller = CameraController(self.camera)

        self.particles = create_particles(num_particles=num_particles)

        if self.render_type == RenderType.GREEDY:
            self.compute = ParticleShader(
                self.device,
                SHADER_DIR / "greedy.slang",
                self.particles,
                (len(self.particles), 1, 1),
            )
        elif self.render_type == RenderType.TILED:
            self.compute = ParticleShader(
                self.device,
                SHADER_DIR / "tiled.slang",
                self.particles,
                (len(self.particles), 1, 1),
            )
        else:
            raise ValueError(f"Unknown render type: {self.render_type}")

        self.particle_renderer = ParticleRenderer(self.device)
        self.prepare_frame = PrepareFrame(self.device, PrepareFrame.Strategy.ATTENUATE)

        self.timer = Timer()

        self.ui = spy.ui.Context(self.device)
        self.setup_ui()

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Settings", size=spy.float2(300, 100))
        self.fps_text = spy.ui.Text(window, f"FPS: null")
        self.camera_position_text = spy.ui.Text(window, f"Camera position: null")

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()
        self.camera_controller.on_keyboard_event(event)

    def on_mouse_event(self, event: spy.MouseEvent):
        if self.ui.handle_mouse_event(event):
            return
        self.camera_controller.on_mouse_event(event)

    def on_resize(self, width: int, height: int):
        self.device.wait()
        self.surface.configure(width=width, height=height, vsync=False)

    def main_loop(self):
        frame = 0
        while not self.window.should_close():
            self.window.process_events()
            self.ui.process_events()

            self.timer.tick()
            self.fps_text.text = f"FPS: {self.timer.get_fps():.2f}"
            if self.timer.get_frame_count() % 100 == 0:
                print(
                    f"frame {self.timer.get_frame_count()}: FPS: {self.timer.get_fps():.2f}"
                )
            self.camera_position_text.text = f"Camera position: {self.camera.position[0]:.2f}, {self.camera.position[1]:.2f}, {self.camera.position[2]:.2f}"

            dt = self.timer.avg_dt()
            self.camera_controller.update(dt)

            if not self.surface.config:
                continue

            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            if (
                self.output_texture == None
                or self.output_texture.width != surface_texture.width
                or self.output_texture.height != surface_texture.height
            ):
                self.output_texture = self.device.create_texture(
                    format=spy.Format.rgba32_float,
                    width=surface_texture.width,
                    height=surface_texture.height,
                    usage=spy.TextureUsage.shader_resource
                    | spy.TextureUsage.unordered_access,
                    label="output_texture",
                )

            command_encoder = self.device.create_command_encoder()

            self.prepare_frame.execute(command_encoder, self.output_texture)

            self.compute.execute(command_encoder, dt)

            self.particle_renderer.execute(
                command_encoder,
                self.output_texture,
                self.compute.particle_buffer,
                self.camera.world_view_projection_buffer,
                self.camera.position,
            )

            command_encoder.blit(surface_texture, self.output_texture)

            self.ui.new_frame(surface_texture.width, surface_texture.height)
            self.ui.render(surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()

            frame += 1

        self.device.wait()
