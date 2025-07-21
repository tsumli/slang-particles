#include "particle_utils.slang"
#include "types.slang"

static const uint kTileSize = 256;

struct FrameInfo {
  StructuredBuffer<float> elapsed;
  StructuredBuffer<float> dt;
};

RWStructuredBuffer<Particle> particles;
ParameterBlock<FrameInfo> frame_info;

groupshared Particle tile_particles[kTileSize];

[shader("compute")]
[numthreads(kTileSize, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, uint3 gtid: SV_GroupThreadID) {
  let index = tid.x;
  let group_thread_id = gtid.x;
  let num_particles = particles.getCount();
  if (index >= num_particles) {
    return;
  }

  let dt = frame_info.dt[0];
  var particle = particles[index];
  var forces = float3(0.0);

  [unroll]
  for (uint tile_start = 0; tile_start < num_particles; tile_start += kTileSize) {
    uint load_index = tile_start + group_thread_id;
    if (load_index < num_particles) {
      tile_particles[group_thread_id] = particles[load_index];
    }

    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint tile_offset = 0;
         tile_offset < kTileSize && (tile_start + tile_offset) < num_particles; ++tile_offset) {
      if (tile_start + tile_offset != index) {
        ParticleInteraction(particle, tile_particles[tile_offset], forces);
      }
    }

    GroupMemoryBarrierWithGroupSync();
  }

  UpdateParticle(particle, forces, dt);
  particles[index] = particle;
}
