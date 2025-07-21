#include "types.slang"

RWTexture2D<float4> output;
StructuredBuffer<Particle> particles;

ParameterBlock<Camera> camera;

bool IsWithinBounds(float2 target, float2 bounds) {
  return target.x >= 0.0 && target.x < bounds.x && target.y >= 0.0 && target.y < bounds.y;
}

float2 project_position(float3 world_pos) {
  let world_view_projection = camera.world_view_projection[0];
  let clip_pos = mul(float4(world_pos, 1.0), world_view_projection);

  if (clip_pos.w >= 0.0) {
    return float2(-1.0, -1.0);
  }

  let ndc_pos = clip_pos.xyz / clip_pos.w;
  return float2(ndc_pos.x * 0.5 + 0.5, -ndc_pos.y * 0.5 + 0.5);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID) {
  let index = tid.x;
  let num_particles = particles.getCount();
  if (index >= num_particles) {
    return;
  }
  let particle = particles[index];

  let position = particle.position;
  let mass = particle.mass;

  let mass_normalized = saturate(mass / 10.0);
  let color = lerp(float3(1.0, 1.0, 1.0), float3(1.0, 1.0, 1.0), mass_normalized);

  uint2 resolution;
  output.GetDimensions(resolution.x, resolution.y);

  // Project 3D position to screen space
  let screen_pos = project_position(position);

  if (!IsWithinBounds(screen_pos, float2(1.0, 1.0))) {
    return;
  }

  let pixel_pos = int2(screen_pos * float2(resolution));
  if (!IsWithinBounds(pixel_pos, resolution)) {
    return;
  }

  int2 pixel_center = int2(screen_pos * float2(resolution));

  if (!IsWithinBounds(pixel_center, resolution)) {
    return;
  }

  output[uint2(pixel_center)] = float4(color, 1.0);
}
