#if 0
if [ "$#" -eq  "0" ]
then
    echo "No argument supplied"
    exit
fi
dir="$PWD"
mkdir -p build && cd build
if [ "$1" = "native" ]
then
  g++ -g $dir/main.cpp --std=c++14 -lSDL2 -lOpenGL && ./a.out
else
  if [ "$1" = "web" ]
  then
    emcc -std=c++14 -O3 $dir/main.cpp -s ALLOW_MEMORY_GROWTH=1 -s USE_WEBGL2=1 -s FULL_ES3=1 -s USE_SDL=2 -s USE_SDL_IMAGE=2 -s USE_SDL_TTF=2 -s WASM=1 -o index.html && \
    python3 -m http.server
  else
    echo "Unknown command $1; valid: [\"web\", \"native\"]"
    exit
  fi
fi
exit
#endif
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#if __EMSCRIPTEN__
#include <GLES3/gl3.h>
#include <SDL.h>
#include <emscripten/emscripten.h>
#else
#include <GLES3/gl32.h>
#include <SDL2/SDL.h>
void MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                     GLsizei length, const GLchar *message,
                     const void *userParam) {
  fprintf(stderr,
          "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
          (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type, severity,
          message);
}
#endif
double my_clock() {
  std::chrono::time_point<std::chrono::system_clock> now =
      std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return 1.0e-3 *
         (double)std::chrono::duration_cast<std::chrono::milliseconds>(duration)
             .count();
}
template <typename F> struct __Defer__ {
  F f;
  __Defer__(F f) : f(f) {}
  ~__Defer__() { f(); }
};

template <typename F> __Defer__<F> defer_func(F f) { return __Defer__<F>(f); }

#define DEFER_1(x, y) x##y
#define DEFER_2(x, y) DEFER_1(x, y)
#define DEFER_3(x) DEFER_2(x, __COUNTER__)
#define defer(code) auto DEFER_3(_defer_) = defer_func([&]() { code; })
#define ito(N) for (uint32_t i = 0; i < N; ++i)
#define jto(N) for (uint32_t j = 0; j < N; ++j)
#define kto(N) for (uint32_t k = 0; k < N; ++k)
#define ASSERT_ALWAYS(x)                                                       \
  {                                                                            \
    if (!(x)) {                                                                \
      fprintf(stderr, "[FAIL] at %s:%i %s\n", __FILE__, __LINE__, #x);         \
      (void)(*(int *)(NULL) = 0);                                              \
    }                                                                          \
  }
#define ASSERT_DEBUG(x) ASSERT_ALWAYS(x)

#define Real double
#define LowPFP float

#if Real == double
Real Real_cos(Real a) { return cos((double)a); }
Real Real_sin(Real a) { return sin((double)a); }
Real Real_sqrt(Real a) { return sqrt((double)a); }
Real Real_tan(Real a) { return tan((double)a); }
#endif

struct real3 {
  Real x, y, z;
};

real3 real3_cross(real3 a, real3 b) {
  return real3{.x = (a.y * b.z - a.z * b.y),
               .y = (a.z * b.x - a.x * b.z),
               .z = (a.x * b.y - a.y * b.x)};
}

real3 real3_print(real3 a) { fprintf(stdout, "{%f, %f, %f}", a.x, a.y, a.z); }

real3 real3_sub(real3 a, real3 b) {
  return real3{.x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z};
}

real3 real3_mat3(Real *mat3, real3 a) {
  // clang-format off
  return real3{
  .x = a.x * mat3[0] + a.y * mat3[1] + a.z * mat3[2],
  .y = a.x * mat3[3] + a.y * mat3[4] + a.z * mat3[5],
  .z = a.x * mat3[6] + a.y * mat3[7] + a.z * mat3[8],
  };
  // clang-format on
}

real3 real3_mat4(Real *mat4, real3 a) {
  // clang-format off
  return real3{
  .x = a.x * mat4[0] + a.y * mat4[1] + a.z * mat4[2] + mat4[3],
  .y = a.x * mat4[4] + a.y * mat4[5] + a.z * mat4[6] + mat4[7],
  .z = a.x * mat4[8] + a.y * mat4[9] + a.z * mat4[10] + mat4[11],
  };
  // clang-format on
}

real3 real3_add(real3 a, real3 b) {
  return real3{.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z};
}

real3 real3_mul(real3 a, Real b) {
  return real3{.x = a.x * b, .y = a.y * b, .z = a.z * b};
}

Real real3_dot(real3 a, real3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

real3 real3_norm(real3 a) {
  Real length = Real_sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
  return real3{.x = a.x / length, .y = a.y / length, .z = a.z / length};
}

// CONSTANTS
static size_t BALL_COUNT = 1 << 10;
static Real BALL_SPEED = 10.0; // Avg bullet travels at 700 meters/second
static Real ring_angle = 0.0;
static Real last_time = my_clock();
static Real RING_RADIUS = 6 * 1.0e6;
static Real RING_ANGULAR_VELOCITY = Real_sqrt(9.80665 / RING_RADIUS); // Rad/Sec
static Real RING_TANGENTAL_VELOCITY = Real_sqrt(9.80665 * RING_RADIUS); // m/Sec
static Real CAMERA_Z_OFFSET = -RING_RADIUS + 1.0e-1;
static Real BALLS_PER_SECOND = 30.0;
static Real BALLS_PERIOD = 1.0 / BALLS_PER_SECOND;
static Real BALLS_PER_SECOND_TIMER = 0.0;
//////////////////////////

struct Camera {
  // input:
  real3 pos;
  Real phi, theta;
  Real fov_angle;
  Real nearz;
  // calculated:
  real3 look;
  real3 up;
  real3 left;
  float proj[16];
} g_camera;

void Camera_update(Camera *cam, real3 up, float viewport_width,
                   float viewport_height) {
  cam->look =
      real3{Real_cos(cam->theta) * Real_cos(cam->phi),
            Real_cos(cam->theta) * Real_sin(cam->phi), Real_sin(cam->theta)};
  cam->left = real3_norm(real3_cross(up, cam->look));
  cam->up = real3_cross(cam->left, cam->look);
  // real3_print(cam->look);
  // real3_print(cam->left);
  // real3_print(cam->up);
  // fprintf(stdout, "____\n");
  float fov = (float)(1.0 / Real_tan(cam->fov_angle / 2.0));
  float aspect = ((float)viewport_width / viewport_height);
  float e = (float)2.4e-7f;
  // clang-format off
  float proj[16] = {
    fov,       0.0f,          0.0f,       0.0f,
    0.0f,      fov * aspect,  0.0f,       0.0f,
    0.0f,      0.0f,          e - 1.0f,   (float)((e - 2.0f) * cam->nearz),
    0.0f,      0.0f,          -1.0f,      0.0f
  };
  // clang-format on
  memcpy(&cam->proj[0], &proj[0], sizeof(proj));
}

// [0 ..                     .. N-1]
// [stack bytes...][memory bytes...]
struct Temporary_Storage {
  uint8_t *ptr;
  size_t cursor;
  size_t capacity;
  size_t stack_capacity;
  size_t stack_cursor;
};

Temporary_Storage Temporary_Storage_new(size_t capacity) {
  ASSERT_DEBUG(capacity > 0);
  Temporary_Storage out;
  size_t STACK_CAPACITY = 0x100 * sizeof(size_t);
  out.ptr = (uint8_t *)malloc(STACK_CAPACITY + capacity);
  out.capacity = capacity;
  out.cursor = 0;
  out.stack_capacity = STACK_CAPACITY;
  out.stack_cursor = 0;
  return out;
}

void Temporary_Storage_delete(Temporary_Storage *ts) {
  ASSERT_DEBUG(ts != nullptr);
  free(ts->ptr);
  memset(ts, 0, sizeof(Temporary_Storage));
}

void *Temporary_Storage_allocate(Temporary_Storage *ts, size_t size) {
  ASSERT_DEBUG(ts != nullptr);
  ASSERT_DEBUG(size != 0);
  void *ptr = (void *)(ts->ptr + ts->stack_capacity + ts->cursor);
  ts->cursor += size;
  ASSERT_DEBUG(ts->cursor < ts->capacity);
  return ptr;
}

void Temporary_Storage_enter_scope(Temporary_Storage *ts) {
  ASSERT_DEBUG(ts != nullptr);
  // Save the cursor to the stack
  size_t *top = (size_t *)(ts->ptr + ts->stack_cursor);
  *top = ts->cursor;
  // Increment stack cursor
  ts->stack_cursor += sizeof(size_t);
  ASSERT_DEBUG(ts->stack_cursor < ts->stack_capacity);
}

void Temporary_Storage_exit_scope(Temporary_Storage *ts) {
  ASSERT_DEBUG(ts != nullptr);
  // Decrement stack cursor
  ASSERT_DEBUG(ts->stack_cursor >= sizeof(size_t));
  ts->stack_cursor -= sizeof(size_t);
  // Restore the cursor from the stack
  size_t *top = (size_t *)(ts->ptr + ts->stack_cursor);
  ts->cursor = *top;
}

void Temporary_Storage_reset(Temporary_Storage *ts) {
  ASSERT_DEBUG(ts != nullptr);
  ts->cursor = 0;
  ts->stack_cursor = 0;
}

struct Ring_Geom {
  real3 *vertices;
  uint32_t *indices;
  uint32_t vertex_count;
  uint32_t index_count;
};

Ring_Geom Ring_Geom_new(Temporary_Storage *ts, uint32_t section_count,
                        Real radius, Real width) {
  ASSERT_ALWAYS(section_count >= 4);
  Ring_Geom out;
  memset(&out, 0, sizeof(out));
  out.vertex_count = section_count * 2;
  out.vertices =
      (real3 *)Temporary_Storage_allocate(ts, out.vertex_count * sizeof(real3));
  out.index_count = section_count * 6;
  out.indices = (uint32_t *)Temporary_Storage_allocate(
      ts, out.index_count * sizeof(uint32_t));

  const Real section_angle_step = (2.0 * M_PI) / section_count;

  ito(section_count) {
    /*
        Sections
        Vertex ids:
      N-2 0   2   4   6
    .._____________..
      |\  |\  |\  |
      | \ | \ | \ |
    ..|__\L__\|__\|..
      N-1 1   3   5   7

      Row ids:
      N-1 0   1   2 ...

        N = section_count * 2
        vertex buffer layout : [v0, v1, v2, v3, ... , v_{N-1}]
        index buffer layout  : [0, 3, 1, 0, 2, 3, ... , N-3, 2, 0]
    */
    uint32_t vertex_offset = i * 2;
    uint32_t V_0_id = (vertex_offset + 0) % out.vertex_count; // V_0
    uint32_t V_1_id = (vertex_offset + 1) % out.vertex_count; // V_1
    uint32_t V_2_id = (vertex_offset + 2) % out.vertex_count; // V_2
    uint32_t V_3_id = (vertex_offset + 3) % out.vertex_count; // V_3
    uint32_t index_offset = i * 6;
    out.indices[index_offset + 0] = V_0_id;
    out.indices[index_offset + 1] = V_3_id;
    out.indices[index_offset + 2] = V_1_id;
    out.indices[index_offset + 3] = V_0_id;
    out.indices[index_offset + 4] = V_2_id;
    out.indices[index_offset + 5] = V_3_id;
    Real section_angle = section_angle_step * (Real)i;
    Real section_2d_x_0 = Real_cos(section_angle);
    Real section_2d_y_0 = Real_sin(section_angle);
    out.vertices[V_0_id] = real3{.x = section_2d_x_0 * radius,
                                 .y = -width / 2,
                                 .z = section_2d_y_0 * radius};
    out.vertices[V_1_id] = real3{.x = section_2d_x_0 * radius,
                                 .y = width / 2,
                                 .z = section_2d_y_0 * radius};
  }

  return out;
}

Ring_Geom Ring_Geom_delete(Ring_Geom *rg) {
  // free(rg->vertices);
  // free(rg->indices);
  memset(rg, 0, sizeof(Ring_Geom));
}

void Ring_Geom_convert_to_lowfp(Ring_Geom *rg, float *out, real3 pos,
                                real3 look, real3 left, real3 up) {
  ito(rg->vertex_count) {
    real3 src_pos = rg->vertices[i];
    real3 dp = real3_sub(pos, src_pos);
    Real dp_x = real3_dot(dp, left);
    Real dp_y = real3_dot(dp, up);
    Real dp_z = real3_dot(dp, look);
    real3 dst_pos = real3{.x = dp_x, .y = dp_y, .z = dp_z};
    out[i * 3 + 0] = (float)dst_pos.x;
    out[i * 3 + 1] = (float)dst_pos.y;
    out[i * 3 + 2] = (float)dst_pos.z;
  }
}

static int quit_loop = 0;

SDL_Window *window = NULL;
SDL_GLContext glc;
int SCREEN_WIDTH, SCREEN_HEIGHT;

void compile_shader(GLuint shader) {
  glCompileShader(shader);
  GLint isCompiled = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
  if (isCompiled == GL_FALSE) {
    GLint maxLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
    GLchar *errorLog = (GLchar *)malloc(maxLength);
    defer(free(errorLog));
    glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

    glDeleteShader(shader);
    fprintf(stderr, "[ERROR]: %s\n", &errorLog[0]);
    exit(1);
  }
}

void link_program(GLuint program) {
  glLinkProgram(program);
  GLint isLinked = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &isLinked);
  if (isLinked == GL_FALSE) {
    GLint maxLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
    GLchar *infoLog = (GLchar *)malloc(maxLength);
    defer(free(infoLog));
    glGetProgramInfoLog(program, maxLength, &maxLength, &infoLog[0]);
    glDeleteProgram(program);
    fprintf(stderr, "[ERROR]: %s\n", &infoLog[0]);
    exit(1);
  }
}

GLuint create_program(GLchar const *vsrc, GLchar const *fsrc) {
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vsrc, NULL);
  compile_shader(vertexShader);
  defer(glDeleteShader(vertexShader););
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fsrc, NULL);
  compile_shader(fragmentShader);
  defer(glDeleteShader(fragmentShader););

  GLuint program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  link_program(program);
  glDetachShader(program, vertexShader);
  glDetachShader(program, fragmentShader);

  return program;
}

void render() {
  const GLchar *vsrc =
      R"(#version 300 es
  layout (location=0) in vec3 position;
  uniform mat4 projection;
  void main() {
      /*mat2 rot = mat2(
        cos(angle), sin(angle),
        -sin(angle), cos(angle)
      );
      color = vec3(1.0, 0.0, 0.0);*/
      gl_Position =  vec4(position, 1.0)  * projection;
      
  })";
  const GLchar *fsrc =
      R"(#version 300 es
  precision highp float;
  layout(location = 0) out vec4 SV_TARGET0;
  uniform vec3 ucolor;
  void main() {
    /*float color = 1.0 / ((
        //pow(abs(dFdx(gl_FragCoord.z)) + abs(dFdy(gl_FragCoord.z)), 1.0)
        pow(abs(gl_FragCoord.z) * 0.5 + 0.5, 3.5)
    ) * 1.0e7 + 1.0);*/
    SV_TARGET0 = vec4(ucolor, 1.0);
  })";
  static Temporary_Storage ts = Temporary_Storage_new(64 * (1 << 20));
  static real3 *ball_positions = [&] {
    real3 *out =
        (real3 *)Temporary_Storage_allocate(&ts, BALL_COUNT * sizeof(real3));
    ito(BALL_COUNT) out[i] = real3{
        .x = 1.0e10,
        .y = 1.0e10,
        .z = 1.0e10,
    };
    return out;
  }();
  static real3 *ball_velocities = [&] {
    real3 *out =
        (real3 *)Temporary_Storage_allocate(&ts, BALL_COUNT * sizeof(real3));
    ito(BALL_COUNT) out[i] = real3{
        .x = 0.0,
        .y = 0.0,
        .z = 0.0,
    };
    return out;
  }();
  static size_t cur_ball_id = 0;
  static GLuint program = create_program(vsrc, fsrc);
  struct Ring_Buffer_GL {
    GLuint vao;
    GLuint vbo;
    GLuint ibo;
  };
  static Ring_Buffer_GL ring_vao = [&] {
    Ring_Buffer_GL out;
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    GLuint ibo;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    out.vbo = vbo;
    out.ibo = ibo;
    out.vao = vao;
    return out;
  }();
  struct Balls_GL {
    GLuint vao;
    GLuint vbo;
    GLuint ibo;
  };
  static Balls_GL balls_vao = [&] {
    Balls_GL out;
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    GLuint ibo;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    out.vbo = vbo;
    out.ibo = ibo;
    out.vao = vao;
    return out;
  }();
  static int init = [] {
    g_camera.fov_angle = M_PI / 2.0;
    g_camera.nearz = 1.0;
    g_camera.pos = real3{.x = 0.0, .y = 0.0, .z = 0.0};
    g_camera.phi = 0.0;
    g_camera.theta = 0.0f;
    return 0;
  }();

  // Update delta time
  Real cur_time = my_clock();
  Real dt = cur_time - last_time;
  last_time = cur_time;

  ring_angle += RING_ANGULAR_VELOCITY * dt;

  Camera_update(&g_camera, real3{.x = 0.0, .y = 0.0, .z = 1.0}, SCREEN_WIDTH,
                SCREEN_HEIGHT);
  // Scoped temporary allocator
  Temporary_Storage_enter_scope(&ts);
  defer(Temporary_Storage_exit_scope(&ts));

  Real world_to_camera[16];
  {
    // clang-format off
    Real ring_rot[9] = {
        Real_cos(ring_angle),   0.0,    -Real_sin(ring_angle),
        0.0,                    1.0,    0.0,
        Real_sin(ring_angle),   0.0,    Real_cos(ring_angle)
    };
    real3 camera_left = real3_mat3(ring_rot, g_camera.left);
    real3 camera_up = real3_mat3(ring_rot, g_camera.up);
    real3 camera_look = real3_mat3(ring_rot, g_camera.look);
    real3 camera_offset = real3_mat3(ring_rot, real3_add(g_camera.pos, real3{.x = 0.0, .y = 0.0, .z = CAMERA_Z_OFFSET}));
    real3 tangent = real3_mat3(ring_rot, real3{.x = 1.0, .y = 0.0, .z = 0.0});

    Real world_to_camera_tmp[16] = {
      -camera_left.x,  -camera_left.y,  -camera_left.z,  real3_dot(camera_offset, camera_left),
      -camera_up.x,    -camera_up.y,    -camera_up.z,    real3_dot(camera_offset, camera_up),
      -camera_look.x,  -camera_look.y,  -camera_look.z,  real3_dot(camera_offset, camera_look),
      0.0,            0.0,            0.0,            1.0
    };
    memcpy(world_to_camera, world_to_camera_tmp, sizeof(world_to_camera));
    // clang-format on
    // Update balls
    {
      ito(BALL_COUNT) {
        ball_positions[i] =
            real3_add(ball_positions[i], real3_mul(ball_velocities[i], dt));
      }
      BALLS_PER_SECOND_TIMER += dt;
      if (BALLS_PER_SECOND_TIMER > BALLS_PERIOD) {
        ball_positions[cur_ball_id] = real3_sub(camera_offset, real3{.x=0.0, .y=0.0, .z=-1.0});
        ball_velocities[cur_ball_id] = // real3{.x = 0.0, .y = 0.0, .z = 0.0};
            real3_add(real3_mul(tangent, RING_TANGENTAL_VELOCITY),
                      real3_mul(camera_look, BALL_SPEED));
        cur_ball_id = (cur_ball_id + 1) % BALL_COUNT;
        BALLS_PER_SECOND_TIMER = 0.0;
      }
    }
  }
  // Draw
  glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
  glClearColor(0.0f, 0.3f, 0.4f, 1.0f);
  glClearDepthf(100000.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_DEPTH_WRITEMASK);
  glDepthFunc(GL_LEQUAL);
  // Balls
  {
    size_t ball_billboards_buffer_size = BALL_COUNT * 4 * sizeof(float) * 3;
    size_t ball_indices_buffer_size = BALL_COUNT * 6 * sizeof(uint32_t);
    float *ball_billboards =
        (float *)Temporary_Storage_allocate(&ts, ball_billboards_buffer_size);
    uint32_t *ball_indices =
        (uint32_t *)Temporary_Storage_allocate(&ts, ball_indices_buffer_size);
    ito(BALL_COUNT) {
      size_t position_offset = i * 4 * 3;
      size_t vertex_offset = i * 4;
      size_t index_offset = i * 6;
      real3 billboard_center = real3_mat4(world_to_camera, ball_positions[i]);
      float billboard_center_x = (float)billboard_center.x;
      float billboard_center_y = (float)billboard_center.y;
      float billboard_center_z = (float)billboard_center.z;
      /*
        0   1
        ____
        |\  |
        L__\|
        2   3
      */
      float billboard_size = 1.0e-1 + billboard_center.z * 4.0e-3;
      // clang-format off
      ball_billboards[position_offset + 0]  = billboard_center_x - 1.0f * billboard_size; // V_0.x
      ball_billboards[position_offset + 1]  = billboard_center_y + 1.0f * billboard_size; // V_0.y
      ball_billboards[position_offset + 2]  = billboard_center_z + 0.0f * billboard_size; // V_0.z
      ball_billboards[position_offset + 3]  = billboard_center_x + 1.0f * billboard_size; // V_1.x
      ball_billboards[position_offset + 4]  = billboard_center_y + 1.0f * billboard_size; // V_1.y
      ball_billboards[position_offset + 5]  = billboard_center_z + 0.0f * billboard_size; // V_1.z
      ball_billboards[position_offset + 6]  = billboard_center_x - 1.0f * billboard_size; // V_2.x
      ball_billboards[position_offset + 7]  = billboard_center_y - 1.0f * billboard_size; // V_2.y
      ball_billboards[position_offset + 8]  = billboard_center_z + 0.0f * billboard_size; // V_2.z
      ball_billboards[position_offset + 9]  = billboard_center_x + 1.0f * billboard_size; // V_3.x
      ball_billboards[position_offset + 10] = billboard_center_y - 1.0f * billboard_size; // V_3.y
      ball_billboards[position_offset + 11] = billboard_center_z + 0.0f * billboard_size; // V_3.z

      ball_indices[index_offset + 0] = vertex_offset + 0;
      ball_indices[index_offset + 1] = vertex_offset + 3;
      ball_indices[index_offset + 2] = vertex_offset + 1;
      ball_indices[index_offset + 3] = vertex_offset + 0;
      ball_indices[index_offset + 4] = vertex_offset + 2;
      ball_indices[index_offset + 5] = vertex_offset + 3;
      // clang-format on
    }
    glBindVertexArray(balls_vao.vao);
    glBindBuffer(GL_ARRAY_BUFFER, balls_vao.vbo);
    glBufferData(GL_ARRAY_BUFFER, ball_billboards_buffer_size, ball_billboards,
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, balls_vao.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, ball_indices_buffer_size,
                 ball_indices, GL_DYNAMIC_DRAW);
    ASSERT_ALWAYS(program != 0);
    glUseProgram(program);
    glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, GL_FALSE,
                       g_camera.proj);
    glUniform3f(glGetUniformLocation(program, "ucolor"), 0.73f, 0.53f, 0.3f);
    GLint posAttrib = glGetAttribLocation(program, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawElements(GL_TRIANGLES, BALL_COUNT * 6, GL_UNSIGNED_INT, NULL);
  }
  // Ring
  {
    // Upload ring vertex data to GPU
    Ring_Geom ring_geom =
        Ring_Geom_new(&ts, 1 << 10, RING_RADIUS, RING_RADIUS * 1.0e-2);
    {

      size_t lowp_buffer_size = ring_geom.vertex_count * 3 * sizeof(float);
      float *lowp_vertices =
          (float *)Temporary_Storage_allocate(&ts, lowp_buffer_size);
      Ring_Geom_convert_to_lowfp(
          &ring_geom, lowp_vertices,
          real3_add(g_camera.pos,
                    real3{.x = 0.0, .y = 0.0, .z = CAMERA_Z_OFFSET}),
          g_camera.look, g_camera.left, g_camera.up);
      glBindVertexArray(ring_vao.vao);
      glBindBuffer(GL_ARRAY_BUFFER, ring_vao.vbo);
      glBufferData(GL_ARRAY_BUFFER, lowp_buffer_size, lowp_vertices,
                   GL_DYNAMIC_DRAW);

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ring_vao.ibo);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                   sizeof(ring_geom.indices[0]) * ring_geom.index_count,
                   ring_geom.indices, GL_DYNAMIC_DRAW);
      ASSERT_ALWAYS(program != 0);
      glUseProgram(program);
      glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1,
                         GL_FALSE, g_camera.proj);
      glUniform3f(glGetUniformLocation(program, "ucolor"), 0.23f, 0.43f, 0.3f);
      GLint posAttrib = glGetAttribLocation(program, "position");
      glEnableVertexAttribArray(posAttrib);
      glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
      glDrawElements(GL_TRIANGLES, ring_geom.index_count, GL_UNSIGNED_INT,
                     NULL);
    }
  }

  // glDeleteProgram(program);
  // glDeleteVertexArrays(1, &vao);
  SDL_GL_SwapWindow(window);
}

#if __EMSCRIPTEN__
void main_tick() {
#else
int main_tick() {
#endif

  SDL_Event event;
  SDL_GetWindowSize(window, &SCREEN_WIDTH, &SCREEN_HEIGHT);
  Real camera_speed = 1.0;
  while (SDL_PollEvent(&event)) {
    switch (event.type) {
    case SDL_QUIT: {
      quit_loop = 1;
      break;
    }
    case SDL_KEYDOWN: {
      switch (event.key.keysym.sym) {

      case SDLK_w: {
        g_camera.pos =
            real3_add(g_camera.pos, real3_mul(g_camera.look, camera_speed));
        break;
      }
      case SDLK_ESCAPE: {
        quit_loop = 1;
        break;
      }
      case SDLK_s: {
        g_camera.pos =
            real3_add(g_camera.pos, real3_mul(g_camera.look, -camera_speed));
        break;
      }
      case SDLK_a: {
        g_camera.pos =
            real3_add(g_camera.pos, real3_mul(g_camera.left, camera_speed));
        break;
      }
      case SDLK_d: {
        g_camera.pos =
            real3_add(g_camera.pos, real3_mul(g_camera.left, -camera_speed));
        break;
      }
      }
      break;
    }
    case SDL_MOUSEMOTION: {

      SDL_MouseMotionEvent *m = (SDL_MouseMotionEvent *)&event;
      static int old_mp_x = m->x;
      static int old_mp_y = m->y;
      int dx = m->x - old_mp_x;
      int dy = m->y - old_mp_y;
      g_camera.phi -= (Real)dx * 1.0e1 / SCREEN_HEIGHT;
      g_camera.theta -= (Real)dy * 1.0e1 / SCREEN_HEIGHT;
      g_camera.theta =
          std::max(std::min(g_camera.theta, M_PI_2 - 1.0e-6), -M_PI_2 + 1.0e-6);
      old_mp_x = m->x;
      old_mp_y = m->y;

    } break;
    case SDL_MOUSEWHEEL: {

      BALL_SPEED += (Real)event.wheel.y * 1.0e1;

    } break;
    }
  }

  render();
  SDL_UpdateWindowSurface(window);

#if !__EMSCRIPTEN__
  return 0;
#endif
}

void main_loop() {

#if __EMSCRIPTEN__
  emscripten_set_main_loop(main_tick, 0, true);
#else
  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(MessageCallback, 0);
  while (0 == quit_loop) {
    main_tick();
  }
#endif
}

int main() {
  SDL_Init(SDL_INIT_VIDEO);

  window = SDL_CreateWindow(
      "Star Ring 3D", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 1024,
      1024, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
#if __EMSCRIPTEN__
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#else
  // 3.2 is minimal requirement for renderdoc
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#endif
  SDL_GL_SetSwapInterval(1);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 23);
  glc = SDL_GL_CreateContext(window);
  ASSERT_ALWAYS(glc);

  main_loop();

  SDL_GL_DeleteContext(glc);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}