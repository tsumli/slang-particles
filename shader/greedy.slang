#include "particle_utils.slang"
#include "types.slang"

struct FrameInfo {
  StructuredBuffer<float> elapsed;
  StructuredBuffer<float> dt;
};

RWStructuredBuffer<Particle> particles;
ParameterBlock<FrameInfo> frame_info;

[shader("compute")]
[numthreads(16, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID) {
  let index = tid.x;
  let num_particles = particles.getCount();

  if (index >= num_particles) {
    return;
  }

  let dt = frame_info.dt[0];

  var particle = particles[index];

  var forces = float3(0.0);
  for (int i = 0; i < num_particles; ++i) {
    if (i == index) {
      continue;
    }
    let other_particle = particles[i];
    ParticleInteraction(particle, other_particle, forces);
  }

  UpdateParticle(particle, forces, dt);
  particles[index] = particle;
}
