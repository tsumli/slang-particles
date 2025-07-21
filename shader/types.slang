#ifndef SHADER_TYPES_SLANG
#define SHADER_TYPES_SLANG

struct Particle {
  float mass;
  float3 position;
  float3 velocity;
};

struct Camera {
  StructuredBuffer<float4x4> world_view_projection;
  StructuredBuffer<float3> position;
};

#endif  // SHADER_TYPES_SLANG