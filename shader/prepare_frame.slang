RWTexture2D<float4> output;

bool IsWithinBounds(uint2 target, uint2 bounds) {
  return target.x < bounds.x && target.y < bounds.y;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void clear_main(uint3 tid: SV_DispatchThreadID) {
  uint2 pixel = tid.xy;
  uint2 resolution;
  output.GetDimensions(resolution.x, resolution.y);

  if (!IsWithinBounds(pixel, resolution)) {
    return;
  }

  output[pixel] = float4(0.0, 0.0, 0.0, 1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void attenuate_main(uint3 tid: SV_DispatchThreadID) {
  uint2 pixel = tid.xy;
  uint2 resolution;
  output.GetDimensions(resolution.x, resolution.y);

  if (!IsWithinBounds(pixel, resolution)) {
    return;
  }

  let color = output[pixel];
  output[pixel] = float4(color.rgb * 0.90, color.a);
}
