#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <random>
#include <utility>
#include <vector>

#define LOG_SZ 1024

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
#define BIAS_BOUNDARY_STRICTNESS 48.0f
#define TARGET_MAX_VELOCITY 0.05f


#define ROT_SPEED_FACTOR   0.10f
#define WRP_SPEED_FACTOR   1.0f
#define PLP_SPEED_FACTOR   0.03f

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

// Hacky.. the flag indicates whether aspect ratio change has been handled
static float aspect_ratio;
std::atomic_flag aspect_ratio_clean = ATOMIC_FLAG_INIT;

struct {
	GLuint num_balls_loc;
	GLuint aspect_ratio_loc;
	GLuint ball_pos_rad_loc;
	GLuint ball_color_loc;
	GLuint ball_params_loc;
} uniform_locs;

typedef std::pair<const char *, GLuint *> uniform_name_loc_mapping;

// God I wish there was std::make_array that would infer its size from
// initializer list size
const std::array<uniform_name_loc_mapping, 5> un2l = {
	std::make_pair("num_balls", &(uniform_locs.num_balls_loc)),
	std::make_pair("aspect_ratio", &(uniform_locs.aspect_ratio_loc)),
	std::make_pair("ball_pos_rad", &(uniform_locs.ball_pos_rad_loc)),
	std::make_pair("ball_color", &(uniform_locs.ball_color_loc)),
	std::make_pair("ball_params", &(uniform_locs.ball_params_loc)),
};

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

static void process_input(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
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

static float rnd_f_minmax(std::mt19937 &gen, float lo, float hi)
{
	std::normal_distribution<float> distr(0.0f, 1.0f);
	return distr(gen);
}

static void random_saturated_color(struct vec3 *color, std::mt19937 &gen)
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

static void random_ball_pos_rad(struct vec3 *ball_pos_rad, std::mt19937 &gen)
{
	std::normal_distribution<float> coords(0.5f, 0.28f);
	std::normal_distribution<float> radius(AVG_BALL_RADIUS, BALL_RADIUS_DEVIATION);

	ball_pos_rad->x = clamp(coords(gen), 0.0f, 1.0f) * aspect_ratio;
	ball_pos_rad->y = clamp(coords(gen), 0.0f, 1.0f);
	ball_pos_rad->z = clamp(radius(gen), MIN_BALL_RADIUS, MAX_BALL_RADIUS);
}

static void random_ball_params(struct vec4 *ball_params, std::mt19937 &gen)
{
	std::gamma_distribution<float> corners_dist(2.9f, 0.35f);

	ball_params->x = std::round(4.0f + corners_dist(gen));
}

static float random_sign(std::mt19937 &gen)
{
	return gen() & 1 ? 1.0f : -1.0f;
}

static void random_ball_hue_velocity(float *hue_velocity, std::mt19937 &gen)
{
	std::gamma_distribution<float> distr(7.0f, 2.0f); // TODO
	*hue_velocity = distr(gen) * random_sign(gen) * HUE_VELOCITY_FACTOR;
}

// Rotation / Warp / Plumpness velocities
static void random_ball_rwp_velocity(struct rwp_vs *rwp, std::mt19937 &gen)
{
	std::gamma_distribution<float> dist(12.0f, 0.4f); // TODO

	rwp->rot_v = dist(gen) * random_sign(gen) * ROT_SPEED_FACTOR;
	rwp->wrp_v = dist(gen) * random_sign(gen) * WRP_SPEED_FACTOR;
	rwp->plp_v = dist(gen) * random_sign(gen) * PLP_SPEED_FACTOR;
}

static struct vec2 biased_random_force(const struct vec3 &curr_pos, std::mt19937 &gen)
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
                       std::mt19937 &gen,
		       float step)
{
	for (int i = 0; i < ball_pos_rad.size(); i++) {
		struct vec3 &pos_rad  = ball_pos_rad.at(i);
		struct vec2 &velocity = ball_velocity.at(i);

		// Limit velocities
		float v_sqrd = velocity.x * velocity.x + velocity.y * velocity.y;
		float inv_inertia = std::min(1.0f, (TARGET_MAX_VELOCITY * TARGET_MAX_VELOCITY) / v_sqrd);

		struct vec2 force = biased_random_force(pos_rad, gen);
		struct vec2 delta_pos = {
			.x = velocity.x * step + 0.5f * force.x * step * step,
			.y = velocity.y * step + 0.5f * force.y * step * step,
		};
		pos_rad.x += delta_pos.x;
		pos_rad.y += delta_pos.y;

		velocity.x += force.x * step * inv_inertia;
		velocity.y += force.y * step * inv_inertia;
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

int main(void)
{
	int rv = 0;
	GLenum err;
	GLuint prg, vao;
	float time;

	GLuint rndseed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 rndgen(rndseed);

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
	GLFWmonitor *monitor = glfwGetPrimaryMonitor();
	const GLFWvidmode *mode = glfwGetVideoMode(monitor);

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
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, resize_callback);

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
		time += 0.01f; // TODO TODO
		move_balls(ball_pos_rad, ball_velocity, rndgen, 0.01f); // TODO add correct time step
		move_ball_hues(ball_color, ball_hue_velocity, 0.01f);
		rotate_warp_balls(ball_params, ball_rwp_velocity, time);
		update_aspect_ratio_maybe(prg);

		update_ball_pos_rad(prg, num_balls, ball_pos_rad.data());
		update_ball_color(prg, num_balls, ball_color.data());
		update_ball_params(prg, num_balls, ball_params.data());

		process_input(window);
		glBindVertexArray(vao);
		draw();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
out_terminate:
	glfwTerminate();
out:
	return rv;
}
