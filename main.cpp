#if 0
emcc -std=c++14 -O3 ../main.cpp -s ALLOW_MEMORY_GROWTH=1 -s USE_WEBGL2=1 -s FULL_ES3=1 -s USE_SDL=2 -s USE_SDL_IMAGE=2 -s USE_SDL_TTF=2 -s WASM=1 -o index.html
python3 -m http.server
exit
#endif

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

static int quit_loop = 0;

#include <vector>

#define ASSERT_ALWAYS(x)                                                       \
  {                                                                            \
    if (!(x)) {                                                                \
      fprintf(stderr, "[FAIL] at %s:%i %s\n", __FILE__, __LINE__, #x);         \
      (void)(*(int *)(NULL) = 0);                                              \
    }                                                                          \
  }
#define ASSERT_DEBUG(x) ASSERT_ALWAYS(x)

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
    std::vector<GLchar> errorLog(maxLength);
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
    std::vector<GLchar> infoLog(maxLength);
    glGetProgramInfoLog(program, maxLength, &maxLength, &infoLog[0]);
    glDeleteProgram(program);
    fprintf(stderr, "[ERROR]: %s\n", &infoLog[0]);
    exit(1);
  }
}

GLuint create_program(GLchar const*vsrc, GLchar const*fsrc) {
  // Create and compile vertex shader
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vsrc, NULL);
  compile_shader(vertexShader);
  defer(glDeleteShader(vertexShader););
  // Create and compile fragment shader
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fsrc, NULL);
  compile_shader(fragmentShader);
  defer(glDeleteShader(fragmentShader););

  // Link vertex and fragment shader into shader program and use it
  GLuint program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  link_program(program);
  glDetachShader(program, vertexShader);
  glDetachShader(program, fragmentShader);
  glUseProgram(program);

  return program;
}

void redraw() {
  const GLchar *vsrc =
      R"(#version 300 es
  layout (location=0) in vec3 position;
  out vec3 color;
  void main() {
      color = position;
      gl_Position = vec4(position, 1.0);
      
  })";
  const GLchar *fsrc =
      R"(#version 300 es
  precision highp float;
  in vec3 color;
  layout(location = 0) out vec4 SV_TARGET0;

  void main() {
    SV_TARGET0 = vec4(color, 1.0);
  })";
  static GLuint program = create_program(vsrc, fsrc);
  static GLuint vao = [&] {
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    GLfloat vertices[] = {0.0f, 0.5f, 0.0f,  -0.5f, -0.5f,
                          0.0f, 0.5f, -0.5f, 0.0f};
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    GLint posAttrib = glGetAttribLocation(program, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindVertexArray(0);
    return vao;
  }();
  ASSERT_ALWAYS(program != 0);
  ASSERT_ALWAYS(vao != 0);
  glUseProgram(program);
  glBindVertexArray(vao);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  // glDeleteProgram(program);
  // glDeleteVertexArrays(1, &vao);
}

void render() {
  glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
  glClearColor(0.0f, 0.4f, 1.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  redraw();
  SDL_GL_SwapWindow(window);
}

#if __EMSCRIPTEN__
void main_tick() {
#else
int main_tick() {
#endif

  SDL_Event event;
  SDL_GetWindowSize(window, &SCREEN_WIDTH, &SCREEN_HEIGHT);
  while (SDL_PollEvent(&event)) {
    switch (event.type) {
    case SDL_QUIT: {
      quit_loop = 1;
      break;
    }
    case SDL_KEYDOWN: {
      switch (event.key.keysym.sym) {
      case SDLK_UP: {

        break;
      }
      case SDLK_DOWN: {

        break;
      }
      case SDLK_LEFT: {

        break;
      }
      case SDLK_RIGHT: {

        break;
      }
      }
      break;
    }
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
      "WEBASM", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 1920, 1080,
      SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
#if __EMSCRIPTEN__
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#else
  // 3.2 is minimal requirement for renderdoc
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#endif
  SDL_GL_SetSwapInterval(1);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  glc = SDL_GL_CreateContext(window);
  ASSERT_ALWAYS(glc);

  main_loop();

  SDL_GL_DeleteContext(glc);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
