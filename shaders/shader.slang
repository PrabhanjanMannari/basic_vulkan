struct VSInput {
    float3 inPosition;
    float3 inColor;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
ConstantBuffer<UniformBuffer> ubo;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
    float pointSize : SV_PointSize;
};

struct FragmentInput {
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertMain(VSInput input) {
    VertexOutput output;
    output.position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0))));
    output.color = input.inColor;
    output.pointSize = 1.0f;
    return output;
}

[shader("fragment")]
float4 fragMain(FragmentInput inVert) : SV_Target {
    float3 color = inVert.color;
    return float4(color, 1.0);
}
