#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#define LOG_SZ 1024

#define STEP_PER_US_1HZ 1e-8f

// At which field strength a pixel should be drawn completely white
#define INITIAL_TAIL_CRITICAL_CALUE 0.10f

#define BALL_COUNT 32
#define SATURATION_COEFF 12.0f
#define VALUE_COEFF 14.0f

#define HUE_VELOCITY_FACTOR 0.002f

#define AVG_BALL_RADIUS 0.035f
#define BALL_RADIUS_DEVIATION 0.007f
#define MIN_BALL_RADIUS 0.01f
#define MAX_BALL_RADIUS 0.5f

// Too small and balls will escape, too large and they will oscillate
#define FORCE_STRENGTH 0.3f
#define BIAS_STRENGTH (0.005f * (FORCE_STRENGTH))

// The larger boundary strictness is set, the farther from boundaries the
// balls will be forced to retreat back towards center. The tendency should
// be related to sqrt(BIAS_BOUNDARY_STRICTNESS) so large values could be ok
#define BIAS_BOUNDARY_STRICTNESS 64.0f
#define TARGET_MAX_VELOCITY 0.05f

#define INITIAL_FRICTION 0.15f

#define ROT_SPEED_FACTOR   0.10f
#define WRP_SPEED_FACTOR   0.10f
#define PLP_SPEED_FACTOR   0.03f

#define SHARPNESS_STEP 0.05f
#define FRICTION_STEP 1.3f // Note: friction grows geometrically

struct vec2 {
	float x, y;
	vec2() : x(0.0f), y(0.0f) {}
	vec2(float x_, float y_) : x(x_), y(y_) {}
};

struct vec3 {
	float x, y, z;
	vec3() : x(0.0f), y(0.0f), z(0.0f) {}
	vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct vec4 {
	float x, y, z, w;
	vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
	vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};

struct rwp_vs {
	float rot_v, wrp_v, plp_v;
	rwp_vs() : rot_v(0.0f), wrp_v(0.0f), plp_v(0.0f) {}
	rwp_vs(float rot_v_, float wrp_v_, float plp_v_)
		: rot_v(rot_v_)
		, wrp_v(wrp_v_)
		, plp_v(plp_v_)
	{}
};

struct user_params {
	float tail_critical_value;
	float friction;
	bool do_draw;
	bool limit_time;

	user_params()
		: tail_critical_value(INITIAL_TAIL_CRITICAL_CALUE)
		, friction(INITIAL_FRICTION)
		, do_draw(true)
		, limit_time(true)
	{}

	user_params(float tcv_, float friction_, bool do_draw_, bool limit_time_)
		: tail_critical_value(tcv_)
		, friction(friction_)
		, do_draw(do_draw_)
		, limit_time(limit_time_)
	{}
};

// Hacky.. the flag indicates whether aspect ratio change has been handled
static float aspect_ratio;
std::atomic_flag aspect_ratio_clean = ATOMIC_FLAG_INIT;

struct {
	GLuint num_balls_loc;
	GLuint aspect_ratio_loc;
	GLuint tail_critical_value_loc;
	GLuint ball_pos_rad_loc;
	GLuint ball_color_loc;
	GLuint ball_params_loc;
} uniform_locs;

typedef std::function<void(struct user_params *)> key_callback;

typedef std::pair<const char *, GLuint *> uniform_name_loc_mapping;
typedef std::tuple<int, GLuint, key_callback> key_to_count_mapping;

// God I wish there was std::make_array that would infer its size from
// initializer list size
const std::array<uniform_name_loc_mapping, 6> un2l = {
	std::make_pair("num_balls", &(uniform_locs.num_balls_loc)),
	std::make_pair("aspect_ratio", &(uniform_locs.aspect_ratio_loc)),
	std::make_pair("tail_critical_value", &(uniform_locs.tail_critical_value_loc)),
	std::make_pair("ball_pos_rad", &(uniform_locs.ball_pos_rad_loc)),
	std::make_pair("ball_color", &(uniform_locs.ball_color_loc)),
	std::make_pair("ball_params", &(uniform_locs.ball_params_loc)),
};

static void sharpen_balls_callback    (struct user_params *);
static void unsharpen_balls_callback  (struct user_params *);
static void toggle_draw_callback      (struct user_params *);
static void toggle_limit_time_callback(struct user_params *);
static void more_friction_callback    (struct user_params *);
static void less_friction_callback    (struct user_params *);

std::array<key_to_count_mapping, 6> interesting_keys = {
	std::make_tuple(GLFW_KEY_UP,   0, sharpen_balls_callback),
	std::make_tuple(GLFW_KEY_DOWN, 0, unsharpen_balls_callback),
	std::make_tuple(GLFW_KEY_D,    0, toggle_draw_callback),
	std::make_tuple(GLFW_KEY_L,    0, toggle_limit_time_callback),
	std::make_tuple(GLFW_KEY_F,    0, more_friction_callback),
	std::make_tuple(GLFW_KEY_V,    0, less_friction_callback),
};

std::mutex key_mtx;

static void sharpen_balls_callback(struct user_params *params)
{
	float *tcv = &(params->tail_critical_value);
	*tcv = std::min(*tcv + SHARPNESS_STEP, 1.0f);
}

static void unsharpen_balls_callback(struct user_params *params)
{
	float *tcv = &(params->tail_critical_value);
	*tcv = std::max(*tcv - SHARPNESS_STEP, 0.0f);
}

static void toggle_draw_callback(struct user_params *params)
{
	params->do_draw = !(params->do_draw);
}

static void toggle_limit_time_callback(struct user_params *params)
{
	params->limit_time = !(params->limit_time);
}

static void more_friction_callback(struct user_params *params)
{
	params->friction *= FRICTION_STEP;
}

static void less_friction_callback(struct user_params *params)
{
	params->friction *= (1.0f / FRICTION_STEP);
}

static void get_uniform_locs(GLuint prg)
{
	for (auto it = un2l.begin(); it != un2l.end(); ++it) {
		const char *uniform_name = it->first;
		GLuint *loc = it->second;

		*loc = glGetUniformLocation(prg, uniform_name);
	}
}

static int read_file(const char *fn, char **dst, GLint *sz)
{
	int rv = 0;
	FILE *f = fopen(fn, "r");
	if (f == NULL) {
		rv = 1;
		goto out;
	}
	fseek(f, 0, SEEK_END);
	*sz = ftell(f);
	fseek(f, 0, SEEK_SET);

	*dst = (char *)malloc(*sz);
	if (*dst == NULL) {
		rv = 1;
		goto out_close_f;
	}
	if (fread(*dst, 1, *sz, f) != *sz) {
		rv = 1;
		goto out_dealloc;
	}
	goto out_close_f;
out_dealloc:
	free(*dst);
out_close_f:
	fclose(f);
out:
	return rv;
}

static void process_input(GLFWwindow *window, struct user_params *params)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	std::lock_guard<std::mutex> lck(key_mtx);
	for (auto it = interesting_keys.begin(); it != interesting_keys.end(); ++it) {
		GLuint &count = std::get<1>(*it);
		auto callback = std::get<2>(*it);

		for (; count > 0; count--)
			callback(params);
	}
}

static void resize_callback(GLFWwindow *window, int w, int h)
{
	glViewport(0, 0, w, h);
	aspect_ratio = (float)w / (float)h;
	aspect_ratio_clean.clear();
}

static void draw(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

static void update_num_balls(GLuint prg, GLuint num_balls)
{
	glProgramUniform1ui(prg, uniform_locs.num_balls_loc, num_balls);
}

static void update_ball_pos_rad(GLuint prg, GLuint num_balls, const struct vec3 *ball_pos_rad)
{
	glProgramUniform3fv(prg, uniform_locs.ball_pos_rad_loc, num_balls, (const GLfloat *)ball_pos_rad);
}

static void update_ball_color(GLuint prg, GLuint num_balls, const struct vec3 *ball_color)
{
	glProgramUniform3fv(prg, uniform_locs.ball_color_loc, num_balls, (const GLfloat *)ball_color);
}

static void update_ball_params(GLuint prg, GLuint num_balls, const struct vec4 *ball_params)
{
	glProgramUniform4fv(prg, uniform_locs.ball_params_loc, num_balls, (const GLfloat *)ball_params);
}

static void update_aspect_ratio_maybe(GLuint prg)
{
	if (aspect_ratio_clean.test_and_set())
		glProgramUniform1f(prg, uniform_locs.aspect_ratio_loc, aspect_ratio);
}

static void update_tail_cv(GLuint prg, const struct user_params *params)
{
	glProgramUniform1f(prg, uniform_locs.tail_critical_value_loc, params->tail_critical_value);
}

static void gen_vao(GLuint *vao)
{
	glGenVertexArrays(1, vao);
	glBindVertexArray(*vao);
}

static GLuint shader_from_src(const char *fn, GLenum type)
{
	GLuint shader = 0;
	char *src;
	GLint sz;
	GLint success;
	GLchar log[LOG_SZ];

	if (read_file(fn, &src, &sz) != 0)
		goto out;

	shader = glCreateShader(type);
	glShaderSource(shader, 1, &src, &sz);
	glCompileShader(shader);
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(shader, LOG_SZ - 1, NULL, log);
		fprintf(stderr, "Failed to compile shader %s:\n%s\n", fn, log);
		goto out_delete_shader;
	}
	goto out_free;

out_delete_shader:
	glDeleteShader(shader);
	shader = 0;
out_free:
	free(src);
out:
	return shader;
}

static GLuint create_shader_program(const char *vs_fn, const char *fs_fn)
{
	GLuint prg = 0, vs, fs;
	GLint success;
	GLchar log[LOG_SZ];

	vs = shader_from_src(vs_fn, GL_VERTEX_SHADER);
	if (vs == 0)
		goto out;

	fs = shader_from_src(fs_fn, GL_FRAGMENT_SHADER);
	if (fs == 0)
		goto out_delete_vs;

	prg = glCreateProgram();
	if (prg == 0)
		goto out_delete_fs_vs;

	glAttachShader(prg, vs);
	glAttachShader(prg, fs);
	glLinkProgram(prg);

	glGetProgramiv(prg, GL_LINK_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(prg, LOG_SZ - 1, NULL, log);
		fprintf(stderr, "Failed to link shaders %s and %s:\n%s\n", vs_fn, fs_fn, log);
		goto out_delete_prg;
	}
	goto out_delete_fs_vs;

out_delete_prg:
	glDeleteProgram(prg);
	prg = 0;
out_delete_fs_vs:
	glDeleteShader(fs);
out_delete_vs:
	glDeleteShader(vs);
out:
	return prg;
}

static float rnd_f_minmax(std::minstd_rand &gen, float lo, float hi)
{
	std::normal_distribution<float> distr(0.0f, 1.0f);
	return distr(gen);
}

static void random_saturated_color(struct vec3 *color, std::minstd_rand &gen)
{
	std::uniform_real_distribution<float> hue_dist(0.0f, 1.0f);
	std::exponential_distribution<float>  sat_dist(SATURATION_COEFF);
	std::exponential_distribution<float>  val_dist(VALUE_COEFF);

	color->x = hue_dist(gen);
	color->y = 1.0f - std::min(sat_dist(gen), 1.0f);
	color->z = 1.0f - std::min(val_dist(gen), 1.0f);
}

static float clamp(float f, float lo, float hi)
{
	return std::max(std::min(f, hi), lo);
}

static void random_ball_pos_rad(struct vec3 *ball_pos_rad, std::minstd_rand &gen)
{
	std::normal_distribution<float> coords(0.5f, 0.22f);
	std::normal_distribution<float> radius(AVG_BALL_RADIUS, BALL_RADIUS_DEVIATION);

	ball_pos_rad->x = clamp(coords(gen), 0.0f, 1.0f) * aspect_ratio;
	ball_pos_rad->y = clamp(coords(gen), 0.0f, 1.0f);
	ball_pos_rad->z = clamp(radius(gen), MIN_BALL_RADIUS, MAX_BALL_RADIUS);
}

static void random_ball_params(struct vec4 *ball_params, std::minstd_rand &gen)
{
	std::gamma_distribution<float> corners_dist(5.0f, 0.28f);

	ball_params->x = std::round(4.0f + corners_dist(gen));
}

static float random_sign(std::minstd_rand &gen)
{
	return gen() & (1 << 16) ? 1.0f : -1.0f;
}

static void random_ball_hue_velocity(float *hue_velocity, std::minstd_rand &gen)
{
	std::gamma_distribution<float> distr(7.0f, 2.0f); // TODO
	*hue_velocity = distr(gen) * random_sign(gen) * HUE_VELOCITY_FACTOR;
}

// Rotation / Warp / Plumpness velocities
static void random_ball_rwp_velocity(struct rwp_vs *rwp, std::minstd_rand &gen)
{
	std::gamma_distribution<float> dist(12.0f, 0.4f); // TODO

	rwp->rot_v = dist(gen) * random_sign(gen) * ROT_SPEED_FACTOR;
	rwp->wrp_v = dist(gen) * random_sign(gen) * WRP_SPEED_FACTOR;
	rwp->plp_v = dist(gen) * random_sign(gen) * PLP_SPEED_FACTOR;
}

static struct vec2 biased_random_force(const struct vec3 &curr_pos, std::minstd_rand &gen)
{
	const float xlo = 0.0f, xhi = aspect_ratio,
	            ylo = 0.0f, yhi = 1.0f;

	// Delta to low/high boundaries should be an approximate measure
	// how willing the ball is to move towards that boundary
	float xld = curr_pos.x - xlo;
	float xhd = xhi - curr_pos.x;
	float yld = curr_pos.y - ylo;
	float yhd = yhi - curr_pos.y;

	float xbias = xhd - xld;
	float ybias = yhd - yld;

	xbias *= abs(xbias) * BIAS_BOUNDARY_STRICTNESS;
	ybias *= abs(ybias) * BIAS_BOUNDARY_STRICTNESS;

	std::uniform_real_distribution<float> distr(-FORCE_STRENGTH, FORCE_STRENGTH);
	struct vec2 force = {
		.x = distr(gen) + xbias * BIAS_STRENGTH,
		.y = distr(gen) + ybias * BIAS_STRENGTH,
	};
	return force;
}

static void move_balls(std::vector<struct vec3> &ball_pos_rad,
                       std::vector<struct vec2> &ball_velocity,
                       std::minstd_rand &gen,
		       float step,
                       float friction)
{
	for (int i = 0; i < ball_pos_rad.size(); i++) {
		struct vec3 &pos_rad  = ball_pos_rad.at(i);
		struct vec2 &velocity = ball_velocity.at(i);

		// Limit velocities
		float vx_sqrd = velocity.x * velocity.x;
		float vy_sqrd = velocity.y * velocity.y;
		float v_sqrd = velocity.x * velocity.x + velocity.y * velocity.y;
		struct vec2 friction_force = {
			.x = -velocity.x * v_sqrd * friction,
			.y = -velocity.y * v_sqrd * friction,
		};

		struct vec2 rnd_force = biased_random_force(pos_rad, gen);
		struct vec2 force = {
			.x = rnd_force.x + friction_force.x,
			.y = rnd_force.y + friction_force.y,
		};
		struct vec2 delta_pos = {
			.x = velocity.x * step + 0.5f * force.x * step * step,
			.y = velocity.y * step + 0.5f * force.y * step * step,
		};
		pos_rad.x += delta_pos.x;
		pos_rad.y += delta_pos.y;

		velocity.x += force.x * step;
		velocity.y += force.y * step;
	}
}

static void move_ball_hues(std::vector<struct vec3> &ball_color,
                           std::vector<float> &ball_hue_velocity,
                           float step)
{
	for (int i = 0; i < ball_color.size(); i++) {
		const float &hue_v = ball_hue_velocity.at(i);
		struct vec3 &color = ball_color.at(i);
		float &hue = color.x;

		hue += hue_v * step;
		while (hue > 1.0f)
			hue -= 1.0f;
		while (hue < 0.0f)
			hue += 1.0f;
	}
}

static float cos_0to1(float f)
{
	return 0.5f * (cos(f) + 1.0f);
}

static float cos_minmax(float f, float min, float max)
{
	return cos_0to1(f) * (max - min) + min;
}

static void rotate_warp_balls(std::vector<struct vec4> &ball_params,
                        const std::vector<struct rwp_vs> &ball_rwp_velocity,
                              float time)
{
	for (int i = 0; i < ball_params.size(); i++) {
		struct vec4 &curr_params = ball_params.at(i);
		const struct rwp_vs &curr_rwp = ball_rwp_velocity.at(i);

		curr_params.y =            curr_rwp.rot_v * time;
		curr_params.z = cos_minmax(curr_rwp.plp_v * time,  0.7f,  1.0f);
		curr_params.w = cos_minmax(curr_rwp.wrp_v * time, -0.2f,  0.2f);
	}
}

static void key_callback_f(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	std::lock_guard<std::mutex> lck(key_mtx);

	for (auto it = interesting_keys.begin(); it != interesting_keys.end(); ++it) {
		int keycode = std::get<0>(*it);
		GLuint &count = std::get<1>(*it);

		if (key == keycode && action == GLFW_PRESS)
			count++;
	}
}

int main(void)
{
	int rv = 0;
	GLenum err;
	GLuint prg, vao;
	float time;
	struct user_params params;

	GLFWmonitor *monitor;
	const GLFWvidmode *mode;

	float us = 0.0f;
	float step_per_us, target_frametime_us;
	auto last_frame = std::chrono::steady_clock::now();
	GLuint rndseed = last_frame.time_since_epoch().count();
	std::minstd_rand rndgen(rndseed);

	std::uniform_real_distribution<float> startingtime_distr(1e3f, 2e3f);
	time = startingtime_distr(rndgen);

	GLuint num_balls = BALL_COUNT;

	std::vector<struct vec3> ball_pos_rad(num_balls);
	std::vector<struct vec3> ball_color(num_balls);
	std::vector<struct vec2> ball_velocity(num_balls);
	std::vector<struct vec4> ball_params(num_balls);
	std::vector<float> ball_hue_velocity(num_balls);
	std::vector<struct rwp_vs> ball_rwp_velocity(num_balls);

	glfwInit();
	monitor = glfwGetPrimaryMonitor();
	mode    = glfwGetVideoMode(monitor);

	step_per_us = STEP_PER_US_1HZ * (float)mode->refreshRate;
	target_frametime_us = 1e6f / (float)mode->refreshRate;

	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow *window = glfwCreateWindow(mode->width, mode->height, "mä nään värejä", monitor, NULL);

	if (window == NULL) {
		rv = 1;
		goto out;
	}
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, resize_callback);
	glfwSetKeyCallback(window, key_callback_f);

	err = glewInit();
	if (err != GLEW_OK) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		goto out_terminate;
	}

	prg = create_shader_program("vs.glsl", "fs.glsl");
	if (prg == 0) {
		fprintf(stderr, "Failed to create shader program\n");
		goto out_terminate;
	}
	gen_vao(&vao);
	get_uniform_locs(prg);

	resize_callback(window, mode->width, mode->height);

	for (GLuint i = 0; i < num_balls; i++) {
		random_ball_pos_rad(ball_pos_rad.data() + i, rndgen);
		random_saturated_color(ball_color.data() + i, rndgen);
		random_ball_params(ball_params.data() + i, rndgen);
		random_ball_hue_velocity(ball_hue_velocity.data() + i, rndgen);
		random_ball_rwp_velocity(ball_rwp_velocity.data() + i, rndgen);
	}
	glUseProgram(prg);
	update_num_balls(prg, num_balls);
	update_ball_color(prg, num_balls, ball_color.data());

	while (!glfwWindowShouldClose(window)) {
		float step;
		if (params.limit_time)
			step = step_per_us * us;
		else
			step = step_per_us * target_frametime_us;

		time += step;
		move_balls(ball_pos_rad, ball_velocity, rndgen, step, params.friction);
		move_ball_hues(ball_color, ball_hue_velocity, step);
		rotate_warp_balls(ball_params, ball_rwp_velocity, time);
		update_aspect_ratio_maybe(prg);

		update_ball_pos_rad(prg, num_balls, ball_pos_rad.data());
		update_ball_color(prg, num_balls, ball_color.data());
		update_ball_params(prg, num_balls, ball_params.data());
		update_tail_cv(prg, &params);

		glfwPollEvents();
		process_input(window, &params);
		if (params.do_draw) {
			glBindVertexArray(vao);
			draw();
		}
		if (params.limit_time || params.do_draw)
			glfwSwapBuffers(window);

		auto this_frame = std::chrono::steady_clock::now();
		us = std::chrono::duration_cast<std::chrono::microseconds>(this_frame - last_frame).count();
		last_frame = this_frame;
	}
out_terminate:
	glfwTerminate();
out:
	return rv;
}
