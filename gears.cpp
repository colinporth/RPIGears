// gears.cpp
//{{{  includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <termio.h>
#include <sys/time.h>

#include "bcm_host.h"

#include "GLES2/gl2.h"
#include "EGL/egl.h"
//#include "EGL/eglext.h"

#include "RPi_Logo256.h"
//}}}
//{{{
typedef struct {
  GLfloat pos[3];
  GLfloat norm[3];
  GLfloat texCoords[2];
  } vertex_t;
//}}}
//{{{
typedef struct {
  vertex_t *vertices;
  GLshort *indices;
  GLfloat color[4];

  int nvertices, nindices;

  GLuint vboId; // ID for vertex buffer object
  GLuint iboId; // ID for index buffer object

  GLuint tricount; // number of triangles to draw
  GLvoid *vertex_p; // offset or pointer to first vertex
  GLvoid *normal_p; // offset or pointer to first normal
  GLvoid *index_p;  // offset or pointer to first index
  GLvoid *texCoords_p;  // offset or pointer to first texcoord
  } gear_t;
//}}}
//{{{
typedef struct {
  uint32_t screen_width;
  uint32_t screen_height;

 // OpenGL|ES objects
  EGLDisplay display;
  EGLSurface surface;
  EGLContext context;
  GLenum drawMode;

 // current distance from camera
  GLfloat viewDist;
  GLfloat distance_inc;

 // number of seconds to run the demo
  uint timeToRun;
  GLuint texId;

  gear_t *gear1, *gear2, *gear3;

  // The location of the shader uniforms
  GLuint ModelViewProjectionMatrix_location;
  GLuint ModelViewMatrix_location;
  GLuint NormalMatrix_location;
  GLuint LightSourcePosition_location;
  GLuint MaterialColor_location;
  GLuint DiffuseMap_location;

 // The projection matrix
  GLfloat ProjectionMatrix[16];

  GLfloat angle;
  GLfloat angleFrame;
  GLfloat angleVel;

  // Average Frames Per Second
  float avgfps;
  int useVBO;
  int useVSync;
  } CUBE_STATE_T;
//}}}

const GLfloat LightSourcePosition[4] = { 5.0, 5.0, 10.0, 1.0};
//{{{
// vertex shader for gles2
const char vertex_shader[] =
  "attribute vec3 position;\n"
  "attribute vec3 normal;\n"
  "attribute vec2 uv;\n"
  "uniform mat4 ModelViewMatrix;\n"
  "uniform mat4 ModelViewProjectionMatrix;\n"
  "uniform mat4 NormalMatrix;\n"
  "uniform vec4 LightSourcePosition;\n"
  "varying lowp vec3 L;\n"
  "varying lowp vec3 N;\n"
  "varying lowp vec3 H;\n"
  "varying lowp vec2 oUV;\n"
  "void main(void)\n"
  "    vec4 pos = vec4(position, 1.0);\n"
  "    N = vec3(NormalMatrix * vec4(normal, 0.0));\n"
  "    L = vec3(LightSourcePosition - (ModelViewMatrix * pos));\n"
  "    lowp vec3 V = vec3(ModelViewMatrix * pos);\n"
  "    H = L - V;\n"
  "    oUV = uv;\n"
  "    gl_Position = ModelViewProjectionMatrix * pos;\n"
  "}";
//}}}
//{{{
// fragment shader for gles2
const char fragment_shader[] =
  "uniform vec4 MaterialColor;\n"
  "uniform sampler2D DiffuseMap;\n"
  "varying lowp vec3 L;\n"
  "varying lowp vec3 N;\n"
  "varying lowp vec3 H;\n"
  "varying lowp vec2 oUV;\n"
  "void main(void)\n"
  "    lowp vec3 l = normalize(L);\n"
  "    lowp vec3 n = normalize(N);\n"
  "    lowp vec3 h = normalize(H);\n"
  "    lowp float diffuse = max(dot(l, n), 0.0);\n"
  "    vec4 diffCol = texture2D(DiffuseMap, oUV);\n"
  "    gl_FragColor = vec4(MaterialColor.xyz * diffuse, 1.0) * diffCol;\n"
  "    gl_FragColor += pow(max(0.0, dot(n, h)), 7.0) * diffCol.r;\n"
  "}";
//}}}

CUBE_STATE_T _state;
CUBE_STATE_T* state = &_state;
EGL_DISPMANX_WINDOW_T nativewindow;
GLfloat view_rotx = 25.0, view_roty = 30.0, view_rotz = 0.0;

//{{{
uint getMilliseconds() {

  struct timespec spec;
  clock_gettime(CLOCK_REALTIME, &spec);
  return (spec.tv_sec * 1000L + spec.tv_nsec / 1000000L);
  }
//}}}
//{{{
int _kbhit() {

  static const int STDIN = 0;
  static int initialized = 0;

  if (!initialized) {
    // Use termios to turn off line buffering
    struct termios term;
    tcgetattr (STDIN, &term);
    term.c_lflag &= ~ICANON;
    tcsetattr (STDIN, TCSANOW, &term);
    setbuf (stdin, NULL);
    initialized = 1;
    }

  int bytesWaiting;
  ioctl(STDIN, FIONREAD, &bytesWaiting);

  //if (bytesWaiting > 0) printf("key count: %d", bytesWaiting);
  return bytesWaiting;
  }
//}}}

//{{{
void m4x4_copy (GLfloat* md, const GLfloat* ms)
{
   memcpy(md, ms, sizeof(GLfloat)*16);
}
//}}}
//{{{
void m4x4_multiply( GLfloat* m, const GLfloat* n)
{
   GLfloat tmp[16];
   const GLfloat* row,* column;
   div_t d;
   int i, j;

   for (i = 0; i < 16; i++) {
      tmp[i] = 0;
      d = div(i, 4);
      row = n + d.quot * 4;
      column = m + d.rem;
      for (j = 0; j < 4; j++)
         tmp[i] += row[j] * column[j * 4];
   }
   m4x4_copy(m, tmp);
}
//}}}
//{{{
void m4x4_rotate (GLfloat* m, GLfloat angle, GLfloat x, GLfloat y, GLfloat z)
{
   float s, c;

   angle = 2.0f * M_PI * angle / 360.0f;
   s = sinf(angle);
   c = cosf(angle);

   GLfloat r[16] = {
      x * x * (1 - c) + c,     y * x * (1 - c) + z * s, x * z * (1 - c) - y * s, 0,
      x * y * (1 - c) - z * s, y * y * (1 - c) + c,     y * z * (1 - c) + x * s, 0,
      x * z * (1 - c) + y * s, y * z * (1 - c) - x * s, z * z * (1 - c) + c,     0,
      0, 0, 0, 1
   };

   m4x4_multiply(m, r);
}

//}}}
//{{{
void m4x4_translate (GLfloat* m, GLfloat x, GLfloat y, GLfloat z)
{
   GLfloat t[16] = { 1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  x, y, z, 1 };

   m4x4_multiply(m, t);
}
//}}}
//{{{
void m4x4_identity (GLfloat* m)
{
   const GLfloat t[16] = {
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
   };

   m4x4_copy(m, t);
}
//}}}
//{{{
void m4x4_transpose (GLfloat* m)
{
   const GLfloat t[16] = {
      m[0], m[4], m[8],  m[12],
      m[1], m[5], m[9],  m[13],
      m[2], m[6], m[10], m[14],
      m[3], m[7], m[11], m[15]};

   m4x4_copy(m, t);
}
//}}}
//{{{
void m4x4_invert (GLfloat* m) {

  GLfloat t[16];
  m4x4_identity(t);

  // Extract and invert the translation part 't'. The inverse of a
  // translation matrix can be calculated by negating the translation
  // coordinates.
  t[12] = -m[12]; t[13] = -m[13]; t[14] = -m[14];

  // Invert the rotation part 'r'. The inverse of a rotation matrix is
  // equal to its transpose.
  m[12] = m[13] = m[14] = 0;
  m4x4_transpose(m);

  // inv(m) = inv(r) * inv(t)
  m4x4_multiply(m, t);
  }
//}}}
//{{{
void m4x4_perspective (GLfloat* m, GLfloat fovy, GLfloat aspect, GLfloat zNear, GLfloat zFar) {

  GLfloat tmp[16];
  m4x4_identity(tmp);

  float sine, cosine, cotangent, deltaZ;
  GLfloat radians = fovy / 2.0 * M_PI / 180.0;

  deltaZ = zFar - zNear;
  sine = sinf(radians);
  cosine = cosf(radians);

  if ((deltaZ == 0) || (sine == 0) || (aspect == 0))
     return;

  cotangent = cosine / sine;

  tmp[0] = cotangent / aspect;
  tmp[5] = cotangent;
  tmp[10] = -(zFar + zNear) / deltaZ;
  tmp[11] = -1;
  tmp[14] = -2 * zNear * zFar / deltaZ;
  tmp[15] = 0;

  m4x4_copy(m, tmp);
  }
//}}}

//{{{
gear_t* gear (const GLfloat inner_radius, const GLfloat outer_radius,
              const GLfloat width, const GLint teeth, const GLfloat tooth_depth, const GLfloat color[]) {

  GLint i, j;
  GLfloat r0, r1, r2;
  GLfloat ta, da;
  GLfloat u1, v1, u2, v2, len;
  GLfloat cos_ta, cos_ta_1da, cos_ta_2da, cos_ta_3da, cos_ta_4da;
  GLfloat sin_ta, sin_ta_1da, sin_ta_2da, sin_ta_3da, sin_ta_4da;
  GLshort ix0, ix1, ix2, ix3, ix4;
  vertex_t *vt, *nm, *tx;
  GLshort *ix;

  gear_t* gear = (gear_t*)calloc(1, sizeof(gear_t));
  gear->nvertices = teeth * 38;
  gear->nindices = teeth * 64 * 3;
  gear->vertices = (vertex_t*)calloc(gear->nvertices, sizeof(vertex_t));
  gear->indices = (GLshort*)calloc(gear->nindices, sizeof(GLshort));
  memcpy(&gear->color[0], &color[0], sizeof(GLfloat) * 4);

  r0 = inner_radius;
  r1 = outer_radius - tooth_depth / 2.0;
  r2 = outer_radius + tooth_depth / 2.0;
  da = 2.0 * M_PI / teeth / 4.0;

  vt = gear->vertices;
  nm = gear->vertices;
  tx = gear->vertices;
  ix = gear->indices;

  #define VERTEX(x,y,z) ((vt->pos[0] = x),(vt->pos[1] = y),(vt->pos[2] = z), \
    (tx->texCoords[0] = x / r2 * 0.8 + 0.5),(tx->texCoords[1] = y / r2 * 0.8 + 0.5), (tx++), \
    (vt++ - gear->vertices))
  #define NORMAL(x,y,z) ((nm->norm[0] = x),(nm->norm[1] = y),(nm->norm[2] = z), \
                         (nm++))
  #define INDEX(a,b,c) ((*ix++ = a),(*ix++ = b),(*ix++ = c))

  for (i = 0; i < teeth; i++) {
    ta = i * 2.0 * M_PI / teeth;

    cos_ta = cos(ta);
    cos_ta_1da = cos(ta + da);
    cos_ta_2da = cos(ta + 2 * da);
    cos_ta_3da = cos(ta + 3 * da);
    cos_ta_4da = cos(ta + 4 * da);
    sin_ta = sin(ta);
    sin_ta_1da = sin(ta + da);
    sin_ta_2da = sin(ta + 2 * da);
    sin_ta_3da = sin(ta + 3 * da);
    sin_ta_4da = sin(ta + 4 * da);

    u1 = r2 * cos_ta_1da - r1 * cos_ta;
    v1 = r2 * sin_ta_1da - r1 * sin_ta;
    len = sqrt(u1 * u1 + v1 * v1);
    u1 /= len;
    v1 /= len;
    u2 = r1 * cos_ta_3da - r2 * cos_ta_2da;
    v2 = r1 * sin_ta_3da - r2 * sin_ta_2da;

    /* front face */
    ix0 = VERTEX(r1 * cos_ta,          r1 * sin_ta,          width * 0.5);
    ix1 = VERTEX(r0 * cos_ta,          r0 * sin_ta,          width * 0.5);
    ix2 = VERTEX(r1 * cos_ta_3da,      r1 * sin_ta_3da,      width * 0.5);
    ix3 = VERTEX(r0 * cos_ta_4da,      r0 * sin_ta_4da,      width * 0.5);
    ix4 = VERTEX(r1 * cos_ta_4da,      r1 * sin_ta_4da,      width * 0.5);
    for (j = 0; j < 5; j++) {
      NORMAL(0.0,                  0.0,                  1.0);
    }
    INDEX(ix0, ix2, ix1);
    INDEX(ix1, ix2, ix3);
    INDEX(ix2, ix4, ix3);

    /* front sides of teeth */
    ix0 = VERTEX(r1 * cos_ta,          r1 * sin_ta,          width * 0.5);
    ix1 = VERTEX(r2 * cos_ta_1da,      r2 * sin_ta_1da,      width * 0.5);
    ix2 = VERTEX(r1 * cos_ta_3da,      r1 * sin_ta_3da,      width * 0.5);
    ix3 = VERTEX(r2 * cos_ta_2da,      r2 * sin_ta_2da,      width * 0.5);
    for (j = 0; j < 4; j++) {
      NORMAL(0.0,                  0.0,                  1.0);
    }
    INDEX(ix0, ix1, ix2);
    INDEX(ix1, ix3, ix2);
    /* back face */
    ix0 = VERTEX(r1 * cos_ta,          r1 * sin_ta,          -width * 0.5);
    ix1 = VERTEX(r1 * cos_ta_3da,      r1 * sin_ta_3da,      -width * 0.5);
    ix2 = VERTEX(r0 * cos_ta,          r0 * sin_ta,          -width * 0.5);
    ix3 = VERTEX(r1 * cos_ta_4da,      r1 * sin_ta_4da,      -width * 0.5);
    ix4 = VERTEX(r0 * cos_ta_4da,      r0 * sin_ta_4da,      -width * 0.5);
    for (j = 0; j < 5; j++) {
      NORMAL(0.0,                  0.0,                  -1.0);
    }
    INDEX(ix0, ix2, ix1);
    INDEX(ix1, ix2, ix3);
    INDEX(ix2, ix4, ix3);

 /* back sides of teeth */
    ix0 = VERTEX(r1 * cos_ta_3da,      r1 * sin_ta_3da,      -width * 0.5);
    ix1 = VERTEX(r2 * cos_ta_2da,      r2 * sin_ta_2da,      -width * 0.5);
    ix2 = VERTEX(r1 * cos_ta,          r1 * sin_ta,          -width * 0.5);
    ix3 = VERTEX(r2 * cos_ta_1da,      r2 * sin_ta_1da,      -width * 0.5);

    for (j = 0; j < 4; j++) {
      NORMAL(0.0,                  0.0,                  -1.0);
    }
    INDEX(ix0, ix1, ix2);
    INDEX(ix1, ix3, ix2);

    /* draw outward faces of teeth */
    ix0 = VERTEX(r1 * cos_ta,          r1 * sin_ta,          width * 0.5);
    ix1 = VERTEX(r1 * cos_ta,          r1 * sin_ta,          -width * 0.5);
    ix2 = VERTEX(r2 * cos_ta_1da,      r2 * sin_ta_1da,      width * 0.5);
    ix3 = VERTEX(r2 * cos_ta_1da,      r2 * sin_ta_1da,      -width * 0.5);

    for (j = 0; j < 4; j++) {
      NORMAL(v1,                   -u1,                  0.0);
    }
    INDEX(ix0, ix1, ix2);
    INDEX(ix1, ix3, ix2);
    ix0 = VERTEX(r2 * cos_ta_1da,      r2 * sin_ta_1da,      width * 0.5);
    ix1 = VERTEX(r2 * cos_ta_1da,      r2 * sin_ta_1da,      -width * 0.5);
    ix2 = VERTEX(r2 * cos_ta_2da,      r2 * sin_ta_2da,      width * 0.5);
    ix3 = VERTEX(r2 * cos_ta_2da,      r2 * sin_ta_2da,      -width * 0.5);
    for (j = 0; j < 4; j++) {
      NORMAL(cos_ta,               sin_ta,               0.0);
    }
    INDEX(ix0, ix1, ix2);
    INDEX(ix1, ix3, ix2);
    ix0 = VERTEX(r2 * cos_ta_2da,      r2 * sin_ta_2da,      width * 0.5);
    ix1 = VERTEX(r2 * cos_ta_2da,      r2 * sin_ta_2da,      -width * 0.5);
    ix2 = VERTEX(r1 * cos_ta_3da,      r1 * sin_ta_3da,      width * 0.5);
    ix3 = VERTEX(r1 * cos_ta_3da,      r1 * sin_ta_3da,      -width * 0.5);
    for (j = 0; j < 4; j++) {
      NORMAL(v2,                   -u2,                  0.0);
    }
    INDEX(ix0, ix1, ix2);
    INDEX(ix1, ix3, ix2);
    ix0 = VERTEX(r1 * cos_ta_3da,      r1 * sin_ta_3da,      width * 0.5);
    ix1 = VERTEX(r1 * cos_ta_3da,      r1 * sin_ta_3da,      -width * 0.5);
    ix2 = VERTEX(r1 * cos_ta_4da,      r1 * sin_ta_4da,      width * 0.5);
    ix3 = VERTEX(r1 * cos_ta_4da,      r1 * sin_ta_4da,      -width * 0.5);
    for (j = 0; j < 4; j++) {
      NORMAL(cos_ta,               sin_ta,               0.0);
    }
    INDEX(ix0, ix1, ix2);
    INDEX(ix1, ix3, ix2);
 /* draw inside radius cylinder */
    ix0 = VERTEX(r0 * cos_ta,          r0 * sin_ta,          -width * 0.5);
    ix1 = VERTEX(r0 * cos_ta,          r0 * sin_ta,          width * 0.5);
    ix2 = VERTEX(r0 * cos_ta_4da,      r0 * sin_ta_4da,      -width * 0.5);
    ix3 = VERTEX(r0 * cos_ta_4da,      r0 * sin_ta_4da,      width * 0.5);
    NORMAL(-cos_ta,              -sin_ta,              0.0);
    NORMAL(-cos_ta,              -sin_ta,              0.0);
    NORMAL(-cos_ta_4da,          -sin_ta_4da,          0.0);
    NORMAL(-cos_ta_4da,          -sin_ta_4da,          0.0);
    INDEX(ix0, ix1, ix2);
    INDEX(ix1, ix3, ix2);
  }

  // setup pointers/offsets for draw operations
  if (state->useVBO) {
    // for VBO use offsets into the buffer object
    gear->vertex_p = 0;
    gear->normal_p = (GLvoid *)sizeof(gear->vertices[0].pos);
    gear->texCoords_p = (GLvoid *)(sizeof(gear->vertices[0].pos) + sizeof(gear->vertices[0].norm));
    gear->index_p = 0;
    }
  else {
    // for Vertex Array use pointers to where the buffer starts
    gear->vertex_p = gear->vertices[0].pos;
    gear->normal_p = gear->vertices[0].norm;
    gear->texCoords_p = gear->vertices[0].texCoords;
    gear->index_p = gear->indices;
    }

  gear->tricount = gear->nindices / 3;
  return gear;
  }
//}}}
//{{{
void init_textures() {

  // load a texture buffer but use them on six OGL|ES texture surfaces
  glGenTextures(1, &state->texId);

  // setup texture
  glBindTexture (GL_TEXTURE_2D, state->texId);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB, rpi_image.width, rpi_image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, rpi_image.pixel_data);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }
//}}}

//{{{
void init_model_projGLES2() {
  m4x4_perspective(state->ProjectionMatrix, 45.0, (float)state->screen_width / (float)state->screen_height, 1.0, 50.0);
  glViewport(0, 0, (GLsizei)state->screen_width, (GLsizei)state->screen_height);
  }
//}}}
//{{{
void init_scene_GLES2() {

  GLuint v, f, program;
  const char *p;
  char msg[512];

  glEnable (GL_CULL_FACE);
  glEnable (GL_DEPTH_TEST);

  //Compile the vertex shader
  p = vertex_shader;
  v = glCreateShader (GL_VERTEX_SHADER);
  glShaderSource (v, 1, &p, NULL);
  glCompileShader (v);
  glGetShaderInfoLog (v, sizeof msg, NULL, msg);
  printf ("vertex shader info: %s\n", msg);

  // Compile the fragment shader
  p = fragment_shader;
  f = glCreateShader (GL_FRAGMENT_SHADER);
  glShaderSource (f, 1, &p, NULL);
  glCompileShader (f);
  glGetShaderInfoLog (f, sizeof msg, NULL, msg);
  printf ("fragment shader info: %s\n", msg);

  // Create and link the shader program
  program = glCreateProgram();
  glAttachShader (program, v);
  glAttachShader (program, f);
  glBindAttribLocation (program, 0, "position");
  glBindAttribLocation (program, 1, "normal");
  glBindAttribLocation (program, 2, "uv");

  glLinkProgram (program);
  glGetProgramInfoLog (program, sizeof msg, NULL, msg);
  printf ("info: %s\n", msg);

  // Enable the shaders
  glUseProgram (program);

  // Get the locations of the uniforms so we can access them
  state->ModelViewProjectionMatrix_location = glGetUniformLocation (program, "ModelViewProjectionMatrix");
  state->ModelViewMatrix_location = glGetUniformLocation (program, "ModelViewMatrix");
  state->NormalMatrix_location = glGetUniformLocation (program, "NormalMatrix");
  state->LightSourcePosition_location = glGetUniformLocation (program, "LightSourcePosition");
  state->MaterialColor_location = glGetUniformLocation (program, "MaterialColor");
  state->DiffuseMap_location = glGetUniformLocation (program, "DiffuseMap");
  }
//}}}
//{{{
void draw_gearGLES2 (gear_t* gear, GLfloat* transform, GLfloat x, GLfloat y, GLfloat angle) {

  GLfloat model_view[16];
  GLfloat normal_matrix[16];
  GLfloat model_view_projection[16];

  /* Translate and rotate the gear */
  m4x4_copy (model_view, transform);
  m4x4_translate (model_view, x, y, 0);
  m4x4_rotate (model_view, angle, 0, 0, 1);

  /* Create and set the ModelViewProjectionMatrix */
  m4x4_copy (model_view_projection, state->ProjectionMatrix);
  m4x4_multiply (model_view_projection, model_view);

  glUniformMatrix4fv (state->ModelViewProjectionMatrix_location, 1, GL_FALSE, model_view_projection);
  glUniformMatrix4fv (state->ModelViewMatrix_location, 1, GL_FALSE, model_view);

  /* Set the LightSourcePosition uniform in relation to the object */
  glUniform4fv (state->LightSourcePosition_location, 1, LightSourcePosition);
  glUniform1i (state->DiffuseMap_location, 0);

  // Create and set the NormalMatrix. It's the inverse transpose of the ModelView matrix.
  m4x4_copy (normal_matrix, model_view);
  m4x4_invert (normal_matrix);
  m4x4_transpose (normal_matrix);
  glUniformMatrix4fv (state->NormalMatrix_location, 1, GL_FALSE, normal_matrix);

  /* Set the gear color */
  glUniform4fv (state->MaterialColor_location, 1, gear->color);
  if (state->useVBO) {
    glBindBuffer (GL_ARRAY_BUFFER, gear->vboId);
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, gear->iboId);
    }

  /* Set up the position of the attributes in the vertex buffer object */
  // setup where vertex data is
  glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_t), gear->vertex_p);
  // setup where normal data is
  glVertexAttribPointer (1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_t), gear->normal_p);
  // setup where uv data is
  glVertexAttribPointer (2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex_t), gear->texCoords_p);

  /* Enable the attributes */
  glEnableVertexAttribArray (0);
  glEnableVertexAttribArray (1);
  glEnableVertexAttribArray (2);

  // Bind texture surface to current vertices
  glBindTexture (GL_TEXTURE_2D, state->texId);

  glDrawElements (state->drawMode, gear->tricount, GL_UNSIGNED_SHORT, gear->index_p);

  /* Disable the attributes */
  glDisableVertexAttribArray (2);
  glDisableVertexAttribArray (1);
  glDisableVertexAttribArray (0);
  }
//}}}
//{{{
void draw_sceneGLES2() {

  GLfloat transform[16];
  m4x4_identity(transform);

  /* Translate and rotate the view */
  m4x4_translate (transform, 0.9, 0.0, -state->viewDist);
  m4x4_rotate (transform, view_rotx, 1, 0, 0);
  m4x4_rotate (transform, view_roty, 0, 1, 0);
  m4x4_rotate (transform, view_rotz, 0, 0, 1);

  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  draw_gearGLES2 (state->gear1, transform, -3.0, -2.0, state->angle);
  draw_gearGLES2 (state->gear2, transform, 3.1, -2.0, -2 * state->angle - 9.0);
  draw_gearGLES2 (state->gear3, transform, -3.1, 4.2, -2 * state->angle - 25.0);

  eglSwapBuffers (state->display, state->surface);
  }
//}}}

//{{{
void make_gear_vbo (gear_t* gear) {

  // setup the vertex buffer that will hold the vertices and normals
  glGenBuffers (1, &gear->vboId);
  glBindBuffer (GL_ARRAY_BUFFER, gear->vboId);
  glBufferData (GL_ARRAY_BUFFER, sizeof(vertex_t) * gear->nvertices, gear->vertices, GL_STATIC_DRAW);

  // setup the index buffer that will hold the indices
  glGenBuffers (1, &gear->iboId);
  glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, gear->iboId);
  glBufferData (GL_ELEMENT_ARRAY_BUFFER, sizeof(GLshort) * gear->nindices, gear->indices, GL_STATIC_DRAW);
  }
//}}}
//{{{
void build_gears() {

  const GLfloat red[4] = {0.9, 0.3, 0.3, 1.0};
  const GLfloat green[4] = {0.3, 0.9, 0.3, 1.0};
  const GLfloat blue[4] = {0.3, 0.3, 0.9, 1.0};

  /* make the meshes for the gears */
  state->gear1 = gear (1.0, 4.0, 2.5, 20, 0.7, red);
  state->gear2 = gear (0.5, 2.0, 3.0, 10, 0.7, green);
  state->gear3 = gear (1.3, 2.0, 1.5, 10, 0.7, blue);

  // if VBO enabled then set them up for each gear
  if (state->useVBO) {
    make_gear_vbo (state->gear1);
    make_gear_vbo (state->gear2);
    make_gear_vbo (state->gear3);
    }
}
//}}}
//{{{
void free_gear (gear_t* gear) {

  if (gear) {
    if (gear->vboId)
      glDeleteBuffers(1, &gear->vboId);
    if (gear->iboId)
      glDeleteBuffers(1, &gear->iboId);

    free (gear->vertices);
    free (gear->indices);
    free (gear);
    }
  }
//}}}

//{{{
void update_angleFrame() {
  state->angleFrame = state->angleVel / state->avgfps;
  }
//}}}
//{{{
void update_gear_rotation() {

  /* advance gear rotation for next frame */
  state->angle += state->angleFrame;
  if (state->angle > 360.0)
    state->angle -= 360.0;
  }
//}}}
//{{{
void run_gears() {

  const uint ttr = state->timeToRun;
  const uint st = getMilliseconds();

  uint ct = st;
  uint prevct = ct, seconds = st;

  float dt;
  float fps;
  int frames = 0;
  int active = 30;

  // keep doing the loop while no key hit and ttr  is either 0 or time since start is less than time to run (ttr)
  while (active && ((ttr == 0) || ((ct - st) < ttr))) {
    ct = getMilliseconds();
    frames++;

    dt = (float)(ct - seconds)/1000.0f;
    // adjust angleFrame each half second
    if ((ct - prevct) > 500) {
      if (dt > 0.0f) {
        state->avgfps = state->avgfps * 0.4f + (float)frames / dt * 0.6f;
        update_angleFrame();
        }
      prevct = ct;
      }

    update_gear_rotation();
    draw_sceneGLES2();

    if (dt >= 1.0f) {
      fps = (float)frames  / dt;
      printf("%d frames in %3.1f seconds = %3.1f FPS\n", frames, dt, fps);
      seconds = ct;
      frames = 0;
      }

    // once in a while check if the user hit the keyboard stop the program if a key was hit
    if (active-- == 1) {
      if (_kbhit())
        active = 0;
      else
        active = 30;
      }
    }
  }
//}}}

//{{{
void init_egl() {

  DISPMANX_ELEMENT_HANDLE_T dispman_element;
  DISPMANX_DISPLAY_HANDLE_T dispman_display;
  DISPMANX_UPDATE_HANDLE_T dispman_update;

  // get an EGL display connection
  state->display = eglGetDisplay (EGL_DEFAULT_DISPLAY);
  assert (state->display != EGL_NO_DISPLAY);

  // initialize the EGL display connection
  EGLBoolean result = eglInitialize (state->display, NULL, NULL);
  assert (EGL_FALSE != result);

  // get an appropriate EGL frame buffer configuration
  //{{{
  const EGLint attribute_list[] = {
     EGL_RED_SIZE, 8,
     EGL_GREEN_SIZE, 8,
     EGL_BLUE_SIZE, 8,
     EGL_ALPHA_SIZE, 8,
     EGL_DEPTH_SIZE, 16,
     EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
     EGL_NONE
  };
  //}}}
  EGLConfig config;
  EGLint num_config;
  result = eglChooseConfig (state->display, attribute_list, &config, 1, &num_config);
  assert (EGL_FALSE != result);
  printf ("eglChooseConfig %d\n", num_config);

  // bind the gles api to this thread - this is default so not required
  result = eglBindAPI (EGL_OPENGL_ES_API);
  assert (EGL_FALSE != result);

  // create an EGL rendering context
  //{{{
  EGLint context_attributes[] = {
     EGL_CONTEXT_CLIENT_VERSION, 2,
     EGL_NONE
  };
  //}}}
  state->context = eglCreateContext (state->display, config, EGL_NO_CONTEXT, context_attributes);
  assert (state->context!=EGL_NO_CONTEXT);

  // create an EGL window surface
  int32_t success = graphics_get_display_size (0, &state->screen_width, &state->screen_height);
  assert( success >= 0 );

  VC_RECT_T dst_rect;
  dst_rect.x = 0;
  dst_rect.y = 0;
  dst_rect.width = state->screen_width;
  dst_rect.height = state->screen_height;

  VC_RECT_T src_rect;
  src_rect.x = 0;
  src_rect.y = 0;
  src_rect.width = state->screen_width << 16;
  src_rect.height = state->screen_height << 16;

  dispman_display = vc_dispmanx_display_open (0);
  dispman_update = vc_dispmanx_update_start (0);

  //VC_DISPMANX_ALPHA_T kAlpha = { DISPMANX_FLAGS_ALPHA_FROM_SOURCE, 200, 0 };
  VC_DISPMANX_ALPHA_T kAlpha = { DISPMANX_FLAGS_ALPHA_FIXED_ALL_PIXELS, 210, 0 };
  dispman_element = vc_dispmanx_element_add (dispman_update, dispman_display, 0, &dst_rect, 0, &src_rect,
                                             DISPMANX_PROTECTION_NONE, &kAlpha, NULL, DISPMANX_NO_ROTATE);
  nativewindow.element = dispman_element;
  nativewindow.width = state->screen_width;
  nativewindow.height = state->screen_height;
  vc_dispmanx_update_submit_sync (dispman_update);
  state->surface = eglCreateWindowSurface (state->display, config, &nativewindow, NULL);

  assert (state->surface != EGL_NO_SURFACE);
  assert (eglMakeCurrent (state->display, state->surface, state->surface, state->context) != EGL_FALSE);
  assert (eglSwapInterval (state->display, state->useVSync) != EGL_FALSE);

  // Set background color and clear buffers
  glClearColor (0.25f, 0.45f, 0.55f, 1.0f);

  // Enable back face culling.
  glEnable (GL_CULL_FACE);
  glFrontFace (GL_CCW);

  printf ("GL_RENDERER   = %s\n", (char*)glGetString (GL_RENDERER));
  printf ("GL_VERSION    = %s\n", (char*)glGetString (GL_VERSION));
  printf ("GL_VENDOR     = %s\n", (char*)glGetString (GL_VENDOR));
  printf ("GL_EXTENSIONS = %s\n", (char*)glGetString (GL_EXTENSIONS));
  }
//}}}
//{{{
void exit_func()
// Function to be passed to atexit().
{
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

   // clear screen
   glClear( GL_COLOR_BUFFER_BIT );
   eglSwapBuffers(state->display, state->surface);

   // Release OpenGL resources
   eglMakeCurrent( state->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT );
   eglDestroySurface( state->display, state->surface );
   eglDestroyContext( state->display, state->context );
   eglTerminate( state->display );

   // release memory used for gear and associated vertex arrays
   free_gear(state->gear1);
   free_gear(state->gear2);
   free_gear(state->gear3);

   printf("\nRPIGears finished\n");

} // exit_func()
//}}}
//{{{
void setup_user_options (int argc, char* argv[]) {

  // setup some default states
  state->viewDist = 18.0;
  state->avgfps = 300;
  state->angleVel = 70;
  state->useVBO = 0;
  state->drawMode = GL_TRIANGLES;

  for (int i = 1; i < argc; i++ ) {
    if (!strcmp(argv[i], "-vsync")) { state->useVSync = 1; state->avgfps = 60; }
    else if (!strcmp(argv[i], "-vbo")) state->useVBO = 1;
    else if (!strcmp(argv[i], "-line")) state->drawMode = GL_LINES;
    else if (!strcmp(argv[i], "-nospin")) state->angleVel = 0.0f;
    }
  update_angleFrame();
}
//}}}
//{{{
int main (int argc, char* argv[]) {

  bcm_host_init();

  memset( state, 0, sizeof( *state ) );
  setup_user_options(argc, argv);

  init_egl();
  init_textures();
  build_gears();

  init_scene_GLES2();
  init_model_projGLES2();

  run_gears();

  exit_func();

  return 0;
  }
//}}}
