#version 460

#define MAX_BALL_COUNT 63

layout (location = 0) out vec4 fragColor;

in vec2 uv;

uniform float aspect_ratio;
uniform uint  num_balls;
uniform vec3  ball_pos_rad[MAX_BALL_COUNT];
uniform vec3  ball_color[MAX_BALL_COUNT];

float dist_sqrd(vec2 a, vec2 b)
{
	vec2 delta = a - b;
	return dot(delta, delta);
}

float falloff(vec2 a, vec2 b, float r)
{
	return pow(r, 2) / dist_sqrd(a, b);
}

void main()
{
	fragColor = vec4(uv, 1.0, 1.0);
	vec2 uv_corr = vec2(uv.x * aspect_ratio, uv.y);

	float val = 0.0;

	vec3 color = vec3(0.0, 0.0, 0.0);
	float saturation = 0.0;

	for (int i = 0; i < num_balls; i++) {
		vec2 curr_pos   = ball_pos_rad[i].xy;
		float curr_rad  = ball_pos_rad[i].z * 1.0;
		vec3 curr_color = ball_color[i];

		float field_str = falloff(uv_corr, curr_pos, curr_rad);
		float field_clamped = clamp(field_str, 0.0, 1.0);

		color += field_clamped * curr_color;
		saturation += field_clamped;
	}
	saturation = clamp(saturation, 0.0, 1.0);
	color = clamp(color, 0.0, 1.0);

	float inv_sat = 1.0 - saturation;
	vec3 whiteness = vec3(inv_sat, inv_sat, inv_sat);
	vec3 final = color + whiteness;

	fragColor = vec4(final, 1.0);
}
