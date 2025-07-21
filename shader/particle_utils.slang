#ifndef PARTICLE_UTILS_SLANG
#define PARTICLE_UTILS_SLANG

#include "types.slang"

static const float kGravityStrength = 1.0;

void ParticleInteraction(Particle particle, Particle other_particle, inout float3 forces) {
  let r = other_particle.position - particle.position;
  let dist_squared = dot(r, r) + 1.0;
  let dist = sqrt(dist_squared);

  if (dist > 0.1) {
    let inv_dist_cube = 1.0 / (dist_squared * dist);
    let s = kGravityStrength * other_particle.mass * inv_dist_cube;
    forces += r * s;
  }
}

void UpdateParticle(inout Particle particle, float3 forces, float dt) {
  let acceleration = forces / particle.mass;
  particle.velocity += acceleration * dt;
  particle.position += particle.velocity * dt;
}

#endif 