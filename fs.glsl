#version 460

#define MAX_BALL_COUNT 63
#define PI 3.14159
#define WARP_FACTOR 70.0

layout (location = 0) out vec4 fragColor;

in vec2 uv;

uniform float aspect_ratio;
uniform uint  num_balls;
uniform vec3  ball_pos_rad[MAX_BALL_COUNT];
uniform vec3  ball_color[MAX_BALL_COUNT];

// x: number of points in star
// y: rotation angle of star
// z: plumpness factor of star, 0.0 - 1.0
// w: warp the star
uniform vec4  ball_params[MAX_BALL_COUNT];

const float tail_critical_value_0 = 0.10;
const float tail_critical_value_1 = 0.60;

float vec_angle(vec2 delta)
{
	float xsign = sign(delta.x);
	float xabs  = abs (delta.x);

	xabs    = max(xabs, 1e-6);
	delta.x = xabs * xsign;
	return atan(delta.y, delta.x);
}

float star_func(float ang, float num_points, float plumpness)
{
	float inv_plump = 1.0 - plumpness;
	float ang_corr  = ang + PI;

	return (1.0 - pow(cos(ang * num_points * 0.5), 2)) * inv_plump + plumpness;
}

float falloff(float dist_sqrd, float r)
{
	return pow(r, 2) / dist_sqrd;
}

// Copied from https://github.com/hughsk/glsl-hsv2rgb
vec3 hsv2rgb(vec3 c)
{
	vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float kill_tail(float f)
{
	return f * smoothstep(tail_critical_value_0,
	                      tail_critical_value_1,
	                      f);
}

void main()
{
	vec2 uv_corr = vec2(uv.x * aspect_ratio, uv.y);

	float val = 0.0;

	vec3 color = vec3(0.0, 0.0, 0.0);
	float saturation = 0.0;

	for (int i = 0; i < num_balls; i++) {
		vec2 curr_pos    = ball_pos_rad[i].xy;
		float curr_r     = ball_pos_rad[i].z * 1.0;
		vec3 curr_color  = hsv2rgb(ball_color[i]);
		float curr_n_pts = ball_params[i].x;
		float curr_ang   = ball_params[i].y;
		float plumpness  = ball_params[i].z;
		float curr_warp  = ball_params[i].w;

		vec2  delta_pos  = uv_corr - curr_pos;
		float dist_sqrd  = dot(delta_pos, delta_pos);

		float warp_ang = PI * curr_warp * dist_sqrd * WARP_FACTOR;
		float scr_ang = vec_angle(delta_pos);
		float ang = scr_ang + curr_ang + warp_ang;
		float star_param = star_func(ang, curr_n_pts, plumpness);

		float field_str = falloff(dist_sqrd, curr_r * star_param);
		float field_clamped = min(1.0, kill_tail(field_str));

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
