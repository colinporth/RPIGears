/*{{{*/
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*}}}*/
//#include "glinfo_common.h"
/*{{{  .h*/

/*
 * Copyright (C) 1999-2014  Brian Paul   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * BRIAN PAUL BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


/**
 * Common code shared by glxinfo and wglinfo.
 */

#ifndef GLINFO_COMMON_H
#define GLINFO_COMMON_H


#ifdef _WIN32
/* GL/glext.h is not commonly available on Windows. */
#include <GL/glew.h>
#else
#include <GL/gl.h>
#include <GL/glext.h>
#endif

typedef void (APIENTRY * GETPROGRAMIVARBPROC) (GLenum target, GLenum pname, GLint *params);
typedef const GLubyte *(APIENTRY * GETSTRINGIPROC) (GLenum name, GLuint index);
typedef void (APIENTRY * GETCONVOLUTIONPARAMETERIVPROC) (GLenum target, GLenum pname, GLint *params);


/**
 * Ext functions needed in common code but must be provided by
 * glxinfo or wglinfo.
 */
struct ext_functions
{
   GETPROGRAMIVARBPROC GetProgramivARB;
   GETSTRINGIPROC GetStringi;
   GETCONVOLUTIONPARAMETERIVPROC GetConvolutionParameteriv;
};


#define ELEMENTS(array) (sizeof(array) / sizeof(array[0]))


struct bit_info
{
   int bit;
   const char *name;
};


typedef enum
{
   Normal,
   Wide,
   Verbose,
   Brief
} InfoMode;


struct options
{
   InfoMode mode;
   GLboolean findBest;
   GLboolean limits;
   GLboolean singleLine;
   /* GLX only */
   char *displayName;
   GLboolean allowDirect;
};


/** list of known OpenGL versions */
static const struct { int major, minor; } gl_versions[] = {
   {4, 5},
   {4, 4},
   {4, 3},
   {4, 2},
   {4, 1},
   {4, 0},

   {3, 3},
   {3, 2},
   {3, 1},
   {3, 0},

   {2, 1},
   {2, 0},

   {1, 5},
   {1, 4},
   {1, 3},
   {1, 2},
   {1, 1},
   {1, 0},

   {0, 0} /* end of list */
};


void
print_extension_list(const char *ext, GLboolean singleLine);

char *
build_core_profile_extension_list(const struct ext_functions *extfuncs);

GLboolean
extension_supported(const char *ext, const char *extensionsList);

void
print_limits(const char *extensions, const char *oglstring, int version,
             const struct ext_functions *extfuncs);

const char *
bitmask_to_string(const struct bit_info bits[], int numBits, int mask);

const char *
profile_mask_string(int mask);

const char *
context_flags_string(int mask);


void
parse_args(int argc, char *argv[], struct options *options);

void
print_gpu_memory_info(const char *glExtensions);

#endif /* GLINFO_COMMON_H */
/*}}}*/

/*{{{*/
/*
 * Return the GL enum name for a numeric value.
 * We really only care about the compressed texture formats for now.
 */
static const char * enum_name(GLenum val) {
   /*{{{*/
   static const struct {
      const char *name;
      GLenum val;
   } enums [] = {
      { "GL_COMPRESSED_ALPHA", 0x84E9 },
      { "GL_COMPRESSED_LUMINANCE", 0x84EA },
      { "GL_COMPRESSED_LUMINANCE_ALPHA", 0x84EB },
      { "GL_COMPRESSED_INTENSITY", 0x84EC },
      { "GL_COMPRESSED_RGB", 0x84ED },
      { "GL_COMPRESSED_RGBA", 0x84EE },
      { "GL_COMPRESSED_TEXTURE_FORMATS", 0x86A3 },
      { "GL_COMPRESSED_RGB", 0x84ED },
      { "GL_COMPRESSED_RGBA", 0x84EE },
      { "GL_COMPRESSED_TEXTURE_FORMATS", 0x86A3 },
      { "GL_COMPRESSED_ALPHA", 0x84E9 },
      { "GL_COMPRESSED_LUMINANCE", 0x84EA },
      { "GL_COMPRESSED_LUMINANCE_ALPHA", 0x84EB },
      { "GL_COMPRESSED_INTENSITY", 0x84EC },
      { "GL_COMPRESSED_SRGB", 0x8C48 },
      { "GL_COMPRESSED_SRGB_ALPHA", 0x8C49 },
      { "GL_COMPRESSED_SLUMINANCE", 0x8C4A },
      { "GL_COMPRESSED_SLUMINANCE_ALPHA", 0x8C4B },
      { "GL_COMPRESSED_RED", 0x8225 },
      { "GL_COMPRESSED_RG", 0x8226 },
      { "GL_COMPRESSED_RED_RGTC1", 0x8DBB },
      { "GL_COMPRESSED_SIGNED_RED_RGTC1", 0x8DBC },
      { "GL_COMPRESSED_RG_RGTC2", 0x8DBD },
      { "GL_COMPRESSED_SIGNED_RG_RGTC2", 0x8DBE },
      { "GL_COMPRESSED_RGB8_ETC2", 0x9274 },
      { "GL_COMPRESSED_SRGB8_ETC2", 0x9275 },
      { "GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2", 0x9276 },
      { "GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2", 0x9277 },
      { "GL_COMPRESSED_RGBA8_ETC2_EAC", 0x9278 },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC", 0x9279 },
      { "GL_COMPRESSED_R11_EAC", 0x9270 },
      { "GL_COMPRESSED_SIGNED_R11_EAC", 0x9271 },
      { "GL_COMPRESSED_RG11_EAC", 0x9272 },
      { "GL_COMPRESSED_SIGNED_RG11_EAC", 0x9273 },
      { "GL_COMPRESSED_ALPHA_ARB", 0x84E9 },
      { "GL_COMPRESSED_LUMINANCE_ARB", 0x84EA },
      { "GL_COMPRESSED_LUMINANCE_ALPHA_ARB", 0x84EB },
      { "GL_COMPRESSED_INTENSITY_ARB", 0x84EC },
      { "GL_COMPRESSED_RGB_ARB", 0x84ED },
      { "GL_COMPRESSED_RGBA_ARB", 0x84EE },
      { "GL_COMPRESSED_TEXTURE_FORMATS_ARB", 0x86A3 },
      { "GL_COMPRESSED_RGBA_BPTC_UNORM_ARB", 0x8E8C },
      { "GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_ARB", 0x8E8D },
      { "GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_ARB", 0x8E8E },
      { "GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_ARB", 0x8E8F },
      { "GL_COMPRESSED_RGBA_ASTC_4x4_KHR", 0x93B0 },
      { "GL_COMPRESSED_RGBA_ASTC_5x4_KHR", 0x93B1 },
      { "GL_COMPRESSED_RGBA_ASTC_5x5_KHR", 0x93B2 },
      { "GL_COMPRESSED_RGBA_ASTC_6x5_KHR", 0x93B3 },
      { "GL_COMPRESSED_RGBA_ASTC_6x6_KHR", 0x93B4 },
      { "GL_COMPRESSED_RGBA_ASTC_8x5_KHR", 0x93B5 },
      { "GL_COMPRESSED_RGBA_ASTC_8x6_KHR", 0x93B6 },
      { "GL_COMPRESSED_RGBA_ASTC_8x8_KHR", 0x93B7 },
      { "GL_COMPRESSED_RGBA_ASTC_10x5_KHR", 0x93B8 },
      { "GL_COMPRESSED_RGBA_ASTC_10x6_KHR", 0x93B9 },
      { "GL_COMPRESSED_RGBA_ASTC_10x8_KHR", 0x93BA },
      { "GL_COMPRESSED_RGBA_ASTC_10x10_KHR", 0x93BB },
      { "GL_COMPRESSED_RGBA_ASTC_12x10_KHR", 0x93BC },
      { "GL_COMPRESSED_RGBA_ASTC_12x12_KHR", 0x93BD },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR", 0x93D0 },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR", 0x93D1 },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR", 0x93D2 },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR", 0x93D3 },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR", 0x93D4 },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR", 0x93D5 },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR", 0x93D6 },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR", 0x93D7 },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR", 0x93D8 },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR", 0x93D9 },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR", 0x93DA },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR", 0x93DB },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR", 0x93DC },
      { "GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR", 0x93DD },
      { "GL_COMPRESSED_RGB_FXT1_3DFX", 0x86B0 },
      { "GL_COMPRESSED_RGBA_FXT1_3DFX", 0x86B1 },
      { "GL_COMPRESSED_LUMINANCE_LATC1_EXT", 0x8C70 },
      { "GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT", 0x8C71 },
      { "GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT", 0x8C72 },
      { "GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT", 0x8C73 },
      { "GL_COMPRESSED_RED_RGTC1_EXT", 0x8DBB },
      { "GL_COMPRESSED_SIGNED_RED_RGTC1_EXT", 0x8DBC },
      { "GL_COMPRESSED_RED_GREEN_RGTC2_EXT", 0x8DBD },
      { "GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT", 0x8DBE },
      { "GL_COMPRESSED_RGB_S3TC_DXT1_EXT", 0x83F0 },
      { "GL_COMPRESSED_RGBA_S3TC_DXT1_EXT", 0x83F1 },
      { "GL_COMPRESSED_RGBA_S3TC_DXT3_EXT", 0x83F2 },
      { "GL_COMPRESSED_RGBA_S3TC_DXT5_EXT", 0x83F3 },
      { "GL_COMPRESSED_SRGB_EXT", 0x8C48 },
      { "GL_COMPRESSED_SRGB_ALPHA_EXT", 0x8C49 },
      { "GL_COMPRESSED_SLUMINANCE_EXT", 0x8C4A },
      { "GL_COMPRESSED_SLUMINANCE_ALPHA_EXT", 0x8C4B },
      { "GL_COMPRESSED_SRGB_S3TC_DXT1_EXT", 0x8C4C },
      { "GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT", 0x8C4D },
      { "GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT", 0x8C4E },
      { "GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT", 0x8C4F },
      { "GL_PALETTE4_RGB8_OES", 0x8B90 },
      { "GL_PALETTE4_RGBA8_OES", 0x8B91 },
      { "GL_PALETTE4_R5_G6_B5_OES", 0x8B92 },
      { "GL_PALETTE4_RGBA4_OES", 0x8B93 },
      { "GL_PALETTE4_RGB5_A1_OES", 0x8B94 },
      { "GL_PALETTE8_RGB8_OES", 0x8B95 },
      { "GL_PALETTE8_RGBA8_OES", 0x8B96 },
      { "GL_PALETTE8_R5_G6_B5_OES", 0x8B97 },
      { "GL_PALETTE8_RGBA4_OES", 0x8B98 },
      { "GL_PALETTE8_RGB5_A1_OES", 0x8B99 }
   };
   /*}}}*/
   const int n = sizeof(enums) / sizeof(enums[0]);
   static char buffer[100];
   int i;
   for (i = 0; i < n; i++) {
      if (enums[i].val == val) {
         return enums[i].name;
      }
   }
   /* enum val not found, just print hexadecimal value into static buffer */
   snprintf(buffer, sizeof(buffer), "0x%x", val);
   return buffer;
}
/*}}}*/
/*{{{*/
/*
 * qsort callback for string comparison.
 */
static int compare_string_ptr(const void *p1, const void *p2)
{
   return strcmp(* (char * const *) p1, * (char * const *) p2);
}
/*}}}*/
/*{{{*/
/*
 * Print a list of extensions, with word-wrapping.
 */
void print_extension_list(const char *ext, GLboolean singleLine)
{
   char **extensions;
   int num_extensions;
   const char *indentString = "    ";
   const int indent = 4;
   const int max = 79;
   int width, i, j, k;

   if (!ext || !ext[0])
      return;

   /* count the number of extensions, ignoring successive spaces */
   num_extensions = 0;
   j = 1;
   do {
      if ((ext[j] == ' ' || ext[j] == 0) && ext[j - 1] != ' ') {
         ++num_extensions;
      }
   } while(ext[j++]);

   /* copy individual extensions to an array */
   extensions = malloc(num_extensions * sizeof *extensions);
   if (!extensions) {
      fprintf(stderr, "Error: malloc() failed\n");
      exit(1);
   }
   i = j = k = 0;
   while (1) {
      if (ext[j] == ' ' || ext[j] == 0) {
         /* found end of an extension name */
         const int len = j - i;

         if (len) {
            assert(k < num_extensions);

            extensions[k] = malloc(len + 1);
            if (!extensions[k]) {
               fprintf(stderr, "Error: malloc() failed\n");
               exit(1);
            }

            memcpy(extensions[k], ext + i, len);
            extensions[k][len] = 0;

            ++k;
         };

         i += len + 1;

         if (ext[j] == 0) {
            break;
         }
      }
      j++;
   }
   assert(k == num_extensions);

   /* sort extensions alphabetically */
   qsort(extensions, num_extensions, sizeof extensions[0], compare_string_ptr);

   /* print the extensions */
   width = indent;
   printf("%s", indentString);
   for (k = 0; k < num_extensions; ++k) {
      const int len = strlen(extensions[k]);
      if ((!singleLine) && (width + len > max)) {
         /* start a new line */
         printf("\n");
         width = indent;
         printf("%s", indentString);
      }
      /* print the extension name */
      printf("%s", extensions[k]);

      /* either we're all done, or we'll continue with next extension */
      width += len + 1;

      if (singleLine) {
         printf("\n");
         width = indent;
         printf("%s", indentString);
      }
      else if (k < (num_extensions -1)) {
         printf(", ");
         width += 2;
      }
   }
   printf("\n");

   for (k = 0; k < num_extensions; ++k) {
      free(extensions[k]);
   }
   free(extensions);
}
/*}}}*/

/*{{{*/
/**
 * Get list of OpenGL extensions using core profile's glGetStringi().
 */
char * build_core_profile_extension_list(const struct ext_functions *extfuncs)
{
   GLint i, n, totalLen;
   char *buffer;

   glGetIntegerv(GL_NUM_EXTENSIONS, &n);

   /* compute totalLen */
   totalLen = 0;
   for (i = 0; i < n; i++) {
      const char *ext = (const char *) extfuncs->GetStringi(GL_EXTENSIONS, i);
      if (ext)
          totalLen += strlen(ext) + 1; /* plus a space */
   }

   if (!totalLen)
     return NULL;

   buffer = malloc(totalLen + 1);
   if (buffer) {
      int pos = 0;
      for (i = 0; i < n; i++) {
         const char *ext = (const char *) extfuncs->GetStringi(GL_EXTENSIONS, i);
         strcpy(buffer + pos, ext);
         pos += strlen(ext);
         buffer[pos++] = ' ';
      }
      buffer[pos] = '\0';
   }
   return buffer;
}
/*}}}*/
/*{{{*/
/** Is extension 'ext' supported? */
GLboolean extension_supported(const char *ext, const char *extensionsList)
{
   while (1) {
      const char *p = strstr(extensionsList, ext);
      if (p) {
         /* check that next char is a space or end of string */
         int extLen = strlen(ext);
         if (p[extLen] == 0 || p[extLen] == ' ') {
            return 1;
         }
         else {
            /* We found a superset string, keep looking */
            extensionsList += extLen;
         }
      }
      else {
         break;
      }
   }
   return 0;
}
/*}}}*/
/*{{{*/
/**
 * Is verNum >= verString?
 * \param verString  such as "2.1", "3.0", etc.
 * \param verNum  such as 20, 21, 30, 31, 32, etc.
 */
static GLboolean version_supported(const char *verString, int verNum)
{
   int v;

   if (!verString ||
       !isdigit(verString[0]) ||
       verString[1] != '.' ||
       !isdigit(verString[2])) {
      return GL_FALSE;
   }

   v = (verString[0] - '0') * 10 + (verString[2] - '0');

   return verNum >= v;
}
/*}}}*/

/*{{{*/
struct token_name
{
   GLenum token;
   const char *name;
};
/*}}}*/
/*{{{*/
static void print_shader_limit_list(const struct token_name *lim)
{
   GLint max[1];
   unsigned i;

   for (i = 0; lim[i].token; i++) {
      glGetIntegerv(lim[i].token, max);
      if (glGetError() == GL_NO_ERROR) {
         printf("        %s = %d\n", lim[i].name, max[0]);
      }
   }
}
/*}}}*/
/*{{{*/
/*
 * Print interesting limits for vertex/fragment shaders.
 */
static void print_shader_limits(GLenum target)
{
   static const struct token_name vertex_limits[] = {
      { GL_MAX_VERTEX_UNIFORM_COMPONENTS_ARB, "GL_MAX_VERTEX_UNIFORM_COMPONENTS_ARB" },
      { GL_MAX_VARYING_FLOATS_ARB, "GL_MAX_VARYING_FLOATS_ARB" },
      { GL_MAX_VERTEX_ATTRIBS_ARB, "GL_MAX_VERTEX_ATTRIBS_ARB" },
      { GL_MAX_TEXTURE_IMAGE_UNITS_ARB, "GL_MAX_TEXTURE_IMAGE_UNITS_ARB" },
      { GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS_ARB, "GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS_ARB" },
      { GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS_ARB, "GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS_ARB" },
      { GL_MAX_TEXTURE_COORDS_ARB, "GL_MAX_TEXTURE_COORDS_ARB" },
      { GL_MAX_VERTEX_OUTPUT_COMPONENTS  , "GL_MAX_VERTEX_OUTPUT_COMPONENTS  " },
      { (GLenum) 0, NULL }
   };
   static const struct token_name fragment_limits[] = {
      { GL_MAX_FRAGMENT_UNIFORM_COMPONENTS_ARB, "GL_MAX_FRAGMENT_UNIFORM_COMPONENTS_ARB" },
      { GL_MAX_TEXTURE_COORDS_ARB, "GL_MAX_TEXTURE_COORDS_ARB" },
      { GL_MAX_TEXTURE_IMAGE_UNITS_ARB, "GL_MAX_TEXTURE_IMAGE_UNITS_ARB" },
      { GL_MAX_FRAGMENT_INPUT_COMPONENTS , "GL_MAX_FRAGMENT_INPUT_COMPONENTS " },
      { (GLenum) 0, NULL }
   };
   static const struct token_name geometry_limits[] = {
      { GL_MAX_GEOMETRY_UNIFORM_COMPONENTS, "GL_MAX_GEOMETRY_UNIFORM_COMPONENTS" },
      { GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS, "GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS" },
      { GL_MAX_GEOMETRY_OUTPUT_VERTICES  , "GL_MAX_GEOMETRY_OUTPUT_VERTICES  " },
      { GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS, "GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS" },
      { GL_MAX_GEOMETRY_INPUT_COMPONENTS , "GL_MAX_GEOMETRY_INPUT_COMPONENTS " },
      { GL_MAX_GEOMETRY_OUTPUT_COMPONENTS, "GL_MAX_GEOMETRY_OUTPUT_COMPONENTS" },
      { (GLenum) 0, NULL }
   };

   switch (target) {
   case GL_VERTEX_SHADER:
      printf("    GL_VERTEX_SHADER_ARB:\n");
      print_shader_limit_list(vertex_limits);
      break;

   case GL_FRAGMENT_SHADER:
      printf("    GL_FRAGMENT_SHADER_ARB:\n");
      print_shader_limit_list(fragment_limits);
      break;

   case GL_GEOMETRY_SHADER:
      printf("    GL_GEOMETRY_SHADER:\n");
      print_shader_limit_list(geometry_limits);
      break;
   }
}
/*}}}*/
/*{{{*/
/**
 * Print interesting limits for vertex/fragment programs.
 */
static void print_program_limits(GLenum target, const struct ext_functions *extfuncs)
{
#if defined(GL_ARB_vertex_program) || defined(GL_ARB_fragment_program)
   struct token_name {
      GLenum token;
      const char *name;
   };
   static const struct token_name common_limits[] = {
      { GL_MAX_PROGRAM_INSTRUCTIONS_ARB, "GL_MAX_PROGRAM_INSTRUCTIONS_ARB" },
      { GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB, "GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB" },
      { GL_MAX_PROGRAM_TEMPORARIES_ARB, "GL_MAX_PROGRAM_TEMPORARIES_ARB" },
      { GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB, "GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB" },
      { GL_MAX_PROGRAM_PARAMETERS_ARB, "GL_MAX_PROGRAM_PARAMETERS_ARB" },
      { GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB, "GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB" },
      { GL_MAX_PROGRAM_ATTRIBS_ARB, "GL_MAX_PROGRAM_ATTRIBS_ARB" },
      { GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB, "GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB" },
      { GL_MAX_PROGRAM_ADDRESS_REGISTERS_ARB, "GL_MAX_PROGRAM_ADDRESS_REGISTERS_ARB" },
      { GL_MAX_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB, "GL_MAX_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB" },
      { GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB, "GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB" },
      { GL_MAX_PROGRAM_ENV_PARAMETERS_ARB, "GL_MAX_PROGRAM_ENV_PARAMETERS_ARB" },
      { (GLenum) 0, NULL }
   };
   static const struct token_name fragment_limits[] = {
      { GL_MAX_PROGRAM_ALU_INSTRUCTIONS_ARB, "GL_MAX_PROGRAM_ALU_INSTRUCTIONS_ARB" },
      { GL_MAX_PROGRAM_TEX_INSTRUCTIONS_ARB, "GL_MAX_PROGRAM_TEX_INSTRUCTIONS_ARB" },
      { GL_MAX_PROGRAM_TEX_INDIRECTIONS_ARB, "GL_MAX_PROGRAM_TEX_INDIRECTIONS_ARB" },
      { GL_MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB, "GL_MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB" },
      { GL_MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB, "GL_MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB" },
      { GL_MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB, "GL_MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB" },
      { (GLenum) 0, NULL }
   };

   GLint max[1];
   int i;

   if (target == GL_VERTEX_PROGRAM_ARB) {
      printf("    GL_VERTEX_PROGRAM_ARB:\n");
   }
   else if (target == GL_FRAGMENT_PROGRAM_ARB) {
      printf("    GL_FRAGMENT_PROGRAM_ARB:\n");
   }
   else {
      return; /* something's wrong */
   }

   for (i = 0; common_limits[i].token; i++) {
      extfuncs->GetProgramivARB(target, common_limits[i].token, max);
      if (glGetError() == GL_NO_ERROR) {
         printf("        %s = %d\n", common_limits[i].name, max[0]);
      }
   }
   if (target == GL_FRAGMENT_PROGRAM_ARB) {
      for (i = 0; fragment_limits[i].token; i++) {
         extfuncs->GetProgramivARB(target, fragment_limits[i].token, max);
         if (glGetError() == GL_NO_ERROR) {
            printf("        %s = %d\n", fragment_limits[i].name, max[0]);
         }
      }
   }
#endif /* GL_ARB_vertex_program / GL_ARB_fragment_program */
}
/*}}}*/

/*{{{*/
/**
 * Print interesting OpenGL implementation limits.
 * \param version  20, 21, 30, 31, 32, etc.
 */
void print_limits(const char *extensions, const char *oglstring, int version, const struct ext_functions *extfuncs)
{
   struct token_name {
      GLuint count;
      GLenum token;
      const char *name;
      const char *extension; /* NULL or GL extension name or version string */
   };
   static const struct token_name limits[] = {
      { 1, GL_MAX_ATTRIB_STACK_DEPTH, "GL_MAX_ATTRIB_STACK_DEPTH", NULL },
      { 1, GL_MAX_CLIENT_ATTRIB_STACK_DEPTH, "GL_MAX_CLIENT_ATTRIB_STACK_DEPTH", NULL },
      { 1, GL_MAX_CLIP_PLANES, "GL_MAX_CLIP_PLANES", NULL },
      { 1, GL_MAX_COLOR_MATRIX_STACK_DEPTH, "GL_MAX_COLOR_MATRIX_STACK_DEPTH", "GL_ARB_imaging" },
      { 1, GL_MAX_ELEMENTS_VERTICES, "GL_MAX_ELEMENTS_VERTICES", NULL },
      { 1, GL_MAX_ELEMENTS_INDICES, "GL_MAX_ELEMENTS_INDICES", NULL },
      { 1, GL_MAX_EVAL_ORDER, "GL_MAX_EVAL_ORDER", NULL },
      { 1, GL_MAX_LIGHTS, "GL_MAX_LIGHTS", NULL },
      { 1, GL_MAX_LIST_NESTING, "GL_MAX_LIST_NESTING", NULL },
      { 1, GL_MAX_MODELVIEW_STACK_DEPTH, "GL_MAX_MODELVIEW_STACK_DEPTH", NULL },
      { 1, GL_MAX_NAME_STACK_DEPTH, "GL_MAX_NAME_STACK_DEPTH", NULL },
      { 1, GL_MAX_PIXEL_MAP_TABLE, "GL_MAX_PIXEL_MAP_TABLE", NULL },
      { 1, GL_MAX_PROJECTION_STACK_DEPTH, "GL_MAX_PROJECTION_STACK_DEPTH", NULL },
      { 1, GL_MAX_TEXTURE_STACK_DEPTH, "GL_MAX_TEXTURE_STACK_DEPTH", NULL },
      { 1, GL_MAX_TEXTURE_SIZE, "GL_MAX_TEXTURE_SIZE", NULL },
      { 1, GL_MAX_3D_TEXTURE_SIZE, "GL_MAX_3D_TEXTURE_SIZE", NULL },
#if defined(GL_EXT_texture_array)
      { 1, GL_MAX_ARRAY_TEXTURE_LAYERS_EXT, "GL_MAX_ARRAY_TEXTURE_LAYERS", "GL_EXT_texture_array" },
#endif
      { 2, GL_MAX_VIEWPORT_DIMS, "GL_MAX_VIEWPORT_DIMS", NULL },
      { 2, GL_ALIASED_LINE_WIDTH_RANGE, "GL_ALIASED_LINE_WIDTH_RANGE", NULL },
      { 2, GL_SMOOTH_LINE_WIDTH_RANGE, "GL_SMOOTH_LINE_WIDTH_RANGE", NULL },
      { 2, GL_ALIASED_POINT_SIZE_RANGE, "GL_ALIASED_POINT_SIZE_RANGE", NULL },
      { 2, GL_SMOOTH_POINT_SIZE_RANGE, "GL_SMOOTH_POINT_SIZE_RANGE", NULL },
#if defined(GL_ARB_texture_cube_map)
      { 1, GL_MAX_CUBE_MAP_TEXTURE_SIZE_ARB, "GL_MAX_CUBE_MAP_TEXTURE_SIZE_ARB", "GL_ARB_texture_cube_map" },
#endif
#if defined(GL_NV_texture_rectangle)
      { 1, GL_MAX_RECTANGLE_TEXTURE_SIZE_NV, "GL_MAX_RECTANGLE_TEXTURE_SIZE_NV", "GL_NV_texture_rectangle" },
#endif
#if defined(GL_ARB_multitexture)
      { 1, GL_MAX_TEXTURE_UNITS_ARB, "GL_MAX_TEXTURE_UNITS_ARB", "GL_ARB_multitexture" },
#endif
#if defined(GL_EXT_texture_lod_bias)
      { 1, GL_MAX_TEXTURE_LOD_BIAS_EXT, "GL_MAX_TEXTURE_LOD_BIAS_EXT", "GL_EXT_texture_lod_bias" },
#endif
#if defined(GL_EXT_texture_filter_anisotropic)
      { 1, GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, "GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT", "GL_EXT_texture_filter_anisotropic" },
#endif
#if defined(GL_ARB_draw_buffers)
      { 1, GL_MAX_DRAW_BUFFERS_ARB, "GL_MAX_DRAW_BUFFERS_ARB", "GL_ARB_draw_buffers" },
#endif
#if defined(GL_ARB_blend_func_extended)
      { 1, GL_MAX_DUAL_SOURCE_DRAW_BUFFERS, "GL_MAX_DUAL_SOURCE_DRAW_BUFFERS", "GL_ARB_blend_func_extended" },
#endif
#if defined (GL_ARB_framebuffer_object)
      { 1, GL_MAX_RENDERBUFFER_SIZE, "GL_MAX_RENDERBUFFER_SIZE", "GL_ARB_framebuffer_object" },
      { 1, GL_MAX_COLOR_ATTACHMENTS, "GL_MAX_COLOR_ATTACHMENTS", "GL_ARB_framebuffer_object" },
      { 1, GL_MAX_SAMPLES, "GL_MAX_SAMPLES", "GL_ARB_framebuffer_object" },
#endif
#if defined (GL_EXT_transform_feedback)
     { 1, GL_MAX_TRANSFORM_FEEDBACK_BUFFERS, "GL_MAX_TRANSFORM_FEEDBACK_BUFFERS", "GL_EXT_transform_feedback" },
     { 1, GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS_EXT, "GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS", "GL_EXT_transform_feedback" },
     { 1, GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS_EXT, "GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS", "GL_EXT_transform_feedback", },
     { 1, GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS_EXT, "GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS", "GL_EXT_transform_feedback" },
#endif
#if defined (GL_ARB_texture_buffer_object)
      { 1, GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT, "GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT", "GL_ARB_texture_buffer_object" },
      { 1, GL_MAX_TEXTURE_BUFFER_SIZE, "GL_MAX_TEXTURE_BUFFER_SIZE", "GL_ARB_texture_buffer_object" },
#endif
#if defined (GL_ARB_texture_multisample)
      { 1, GL_MAX_COLOR_TEXTURE_SAMPLES, "GL_MAX_COLOR_TEXTURE_SAMPLES", "GL_ARB_texture_multisample" },
      { 1, GL_MAX_DEPTH_TEXTURE_SAMPLES, "GL_MAX_DEPTH_TEXTURE_SAMPLES", "GL_ARB_texture_multisample" },
      { 1, GL_MAX_INTEGER_SAMPLES, "GL_MAX_INTEGER_SAMPLES", "GL_ARB_texture_multisample" },
#endif
#if defined (GL_ARB_uniform_buffer_object)
      { 1, GL_MAX_VERTEX_UNIFORM_BLOCKS, "GL_MAX_VERTEX_UNIFORM_BLOCKS", "GL_ARB_uniform_buffer_object" },
      { 1, GL_MAX_FRAGMENT_UNIFORM_BLOCKS, "GL_MAX_FRAGMENT_UNIFORM_BLOCKS", "GL_ARB_uniform_buffer_object" },
      { 1, GL_MAX_GEOMETRY_UNIFORM_BLOCKS, "GL_MAX_GEOMETRY_UNIFORM_BLOCKS" , "GL_ARB_uniform_buffer_object" },
      { 1, GL_MAX_COMBINED_UNIFORM_BLOCKS, "GL_MAX_COMBINED_UNIFORM_BLOCKS", "GL_ARB_uniform_buffer_object" },
      { 1, GL_MAX_UNIFORM_BUFFER_BINDINGS, "GL_MAX_UNIFORM_BUFFER_BINDINGS", "GL_ARB_uniform_buffer_object" },
      { 1, GL_MAX_UNIFORM_BLOCK_SIZE, "GL_MAX_UNIFORM_BLOCK_SIZE", "GL_ARB_uniform_buffer_object" },
      { 1, GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS, "GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS", "GL_ARB_uniform_buffer_object" },
      { 1, GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS, "GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS", "GL_ARB_uniform_buffer_object" },
      { 1, GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS, "GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS", "GL_ARB_uniform_buffer_object" },
      { 1, GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, "GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT", "GL_ARB_uniform_buffer_object" },
#endif
#if defined (GL_ARB_vertex_attrib_binding)
      { 1, GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET, "GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET", "GL_ARB_vertex_attrib_binding" },
      { 1, GL_MAX_VERTEX_ATTRIB_BINDINGS, "GL_MAX_VERTEX_ATTRIB_BINDINGS", "GL_ARB_vertex_attrib_binding" },
#endif
#if defined(GL_VERSION_4_4)
      { 1, GL_MAX_VERTEX_ATTRIB_STRIDE, "GL_MAX_VERTEX_ATTRIB_STRIDE", "4.4" },
#endif
      { 0, (GLenum) 0, NULL, NULL }
   };
   GLint i, max[2];

   printf("%s limits:\n", oglstring);
   for (i = 0; limits[i].count; i++) {
      if (!limits[i].extension ||
          version_supported(limits[i].extension, version) ||
          extension_supported(limits[i].extension, extensions)) {
         glGetIntegerv(limits[i].token, max);
         if (glGetError() == GL_NO_ERROR) {
            if (limits[i].count == 1)
               printf("    %s = %d\n", limits[i].name, max[0]);
            else /* XXX fix if we ever query something with more than 2 values */
               printf("    %s = %d, %d\n", limits[i].name, max[0], max[1]);
         }
      }
   }

   /* these don't fit into the above mechanism, unfortunately */
   if (extension_supported("GL_ARB_imaging", extensions)) {
      extfuncs->GetConvolutionParameteriv(GL_CONVOLUTION_2D,
                                          GL_MAX_CONVOLUTION_WIDTH, max);
      extfuncs->GetConvolutionParameteriv(GL_CONVOLUTION_2D,
                                          GL_MAX_CONVOLUTION_HEIGHT, max+1);
      printf("    GL_MAX_CONVOLUTION_WIDTH/HEIGHT = %d, %d\n", max[0], max[1]);
   }

   if (extension_supported("GL_ARB_texture_compression", extensions)) {
      GLint i, n;
      GLint *formats;
      glGetIntegerv(GL_NUM_COMPRESSED_TEXTURE_FORMATS, &n);
      printf("    GL_NUM_COMPRESSED_TEXTURE_FORMATS = %d\n", n);
      formats = (GLint *) malloc(n * sizeof(GLint));
      glGetIntegerv(GL_COMPRESSED_TEXTURE_FORMATS, formats);
      for (i = 0; i < n; i++) {
         printf("        %s\n", enum_name(formats[i]));
      }
      free(formats);
   }
#if defined(GL_ARB_vertex_program)
   if (extension_supported("GL_ARB_vertex_program", extensions)) {
      print_program_limits(GL_VERTEX_PROGRAM_ARB, extfuncs);
   }
#endif
#if defined(GL_ARB_fragment_program)
   if (extension_supported("GL_ARB_fragment_program", extensions)) {
      print_program_limits(GL_FRAGMENT_PROGRAM_ARB, extfuncs);
   }
#endif
   if (extension_supported("GL_ARB_vertex_shader", extensions)) {
      print_shader_limits(GL_VERTEX_SHADER_ARB);
   }
   if (extension_supported("GL_ARB_fragment_shader", extensions)) {
      print_shader_limits(GL_FRAGMENT_SHADER_ARB);
   }
   if (version >= 32) {
      print_shader_limits(GL_GEOMETRY_SHADER);
   }
}

/*}}}*/
/*{{{*/
/**
 * Return string representation for bits in a bitmask.
 */
const char * bitmask_to_string(const struct bit_info bits[], int numBits, int mask)
{
   static char buffer[256], *p;
   int i;

   strcpy(buffer, "(none)");
   p = buffer;
   for (i = 0; i < numBits; i++) {
      if (mask & bits[i].bit) {
         if (p > buffer)
            *p++ = ',';
         strcpy(p, bits[i].name);
         p += strlen(bits[i].name);
      }
   }

   return buffer;
}
/*}}}*/
/*{{{*/
/**
 * Return string representation for the bitmask returned by
 * GL_CONTEXT_PROFILE_MASK (OpenGL 3.2 or later).
 */
const char * profile_mask_string(int mask)
{
   const static struct bit_info bits[] = {
#ifdef GL_CONTEXT_CORE_PROFILE_BIT
      { GL_CONTEXT_CORE_PROFILE_BIT, "core profile"},
#endif
#ifdef GL_CONTEXT_COMPATIBILITY_PROFILE_BIT
      { GL_CONTEXT_COMPATIBILITY_PROFILE_BIT, "compatibility profile" }
#endif
   };

   return bitmask_to_string(bits, ELEMENTS(bits), mask);
}
/*}}}*/
/*{{{*/
/**
 * Return string representation for the bitmask returned by
 * GL_CONTEXT_FLAGS (OpenGL 3.0 or later).
 */
const char * context_flags_string(int mask)
{
   const static struct bit_info bits[] = {
#ifdef GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT
      { GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT, "forward-compatible" },
#endif
#ifdef GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT_ARB
      { GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT_ARB, "robust-access" },
#endif
   };

   return bitmask_to_string(bits, ELEMENTS(bits), mask);
}
/*}}}*/

/*{{{*/
static void usage(void)
{
#ifdef _WIN32
   printf("Usage: wglinfo [-v] [-t] [-h] [-b] [-l] [-s]\n");
#else
   printf("Usage: glxinfo [-v] [-t] [-h] [-b] [-l] [-s] [-i] [-display <dname>]\n");
   printf("\t-display <dname>: Print GLX visuals on specified server.\n");
   printf("\t-i: Force an indirect rendering context.\n");
#endif
   printf("\t-B: brief output, print only the basics.\n");
   printf("\t-v: Print visuals info in verbose form.\n");
   printf("\t-t: Print verbose table.\n");
   printf("\t-h: This information.\n");
   printf("\t-b: Find the 'best' visual and print its number.\n");
   printf("\t-l: Print interesting OpenGL limits.\n");
   printf("\t-s: Print a single extension per line.\n");
}
/*}}}*/
/*{{{*/
void parse_args(int argc, char *argv[], struct options *options)
{
   int i;

   options->mode = Normal;
   options->findBest = GL_FALSE;
   options->limits = GL_FALSE;
   options->singleLine = GL_FALSE;
   options->displayName = NULL;
   options->allowDirect = GL_TRUE;

   for (i = 1; i < argc; i++) {
#ifndef _WIN32
      if (strcmp(argv[i], "-display") == 0 && i + 1 < argc) {
         options->displayName = argv[i + 1];
         i++;
      }
      else if (strcmp(argv[i], "-i") == 0) {
         options->allowDirect = GL_FALSE;
      }
      else
#endif
      if (strcmp(argv[i], "-t") == 0) {
         options->mode = Wide;
      }
      else if (strcmp(argv[i], "-v") == 0) {
         options->mode = Verbose;
      }
      else if (strcmp(argv[i], "-B") == 0) {
         options->mode = Brief;
      }
      else if (strcmp(argv[i], "-b") == 0) {
         options->findBest = GL_TRUE;
      }
      else if (strcmp(argv[i], "-l") == 0) {
         options->limits = GL_TRUE;
      }
      else if (strcmp(argv[i], "-h") == 0) {
         usage();
         exit(0);
      }
      else if(strcmp(argv[i], "-s") == 0) {
         options->singleLine = GL_TRUE;
      }
      else {
         printf("Unknown option `%s'\n", argv[i]);
         usage();
         exit(0);
      }
   }
}
/*}}}*/

/*{{{*/
static void query_ATI_meminfo(void)
{
#ifdef GL_ATI_meminfo
    int i[4];

    printf("Memory info (GL_ATI_meminfo):\n");

    glGetIntegerv(GL_VBO_FREE_MEMORY_ATI, i);
    printf("    VBO free memory - total: %u MB, largest block: %u MB\n",
           i[0] / 1024, i[1] / 1024);
    printf("    VBO free aux. memory - total: %u MB, largest block: %u MB\n",
           i[2] / 1024, i[3] / 1024);

    glGetIntegerv(GL_TEXTURE_FREE_MEMORY_ATI, i);
    printf("    Texture free memory - total: %u MB, largest block: %u MB\n",
           i[0] / 1024, i[1] / 1024);
    printf("    Texture free aux. memory - total: %u MB, largest block: %u MB\n",
           i[2] / 1024, i[3] / 1024);

    glGetIntegerv(GL_RENDERBUFFER_FREE_MEMORY_ATI, i);
    printf("    Renderbuffer free memory - total: %u MB, largest block: %u MB\n",
           i[0] / 1024, i[1] / 1024);
    printf("    Renderbuffer free aux. memory - total: %u MB, largest block: %u MB\n",
           i[2] / 1024, i[3] / 1024);
#endif
}
/*}}}*/
/*{{{*/
static void query_NVX_gpu_memory_info(void)
{
#ifdef GL_NVX_gpu_memory_info
    int i;

    printf("Memory info (GL_NVX_gpu_memory_info):\n");

    glGetIntegerv(GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX, &i);
    printf("    Dedicated video memory: %u MB\n", i / 1024);

    glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &i);
    printf("    Total available memory: %u MB\n", i / 1024);

    glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &i);
    printf("    Currently available dedicated video memory: %u MB\n", i / 1024);
#endif
}
/*}}}*/
/*{{{*/
void print_gpu_memory_info(const char *glExtensions)
{
   if (strstr(glExtensions, "GL_ATI_meminfo"))
      query_ATI_meminfo();
   if (strstr(glExtensions, "GL_NVX_gpu_memory_info"))
      query_NVX_gpu_memory_info();
}
/*}}}*/


/*{{{*/
/*
 * This program is a work-alike of the IRIX glxinfo program.
 * Command line options:
 *  -t                     print wide table
 *  -v                     print verbose information
 *  -display DisplayName   specify the X display to interogate
 *  -B                     brief, print only the basics
 *  -b                     only print ID of "best" visual on screen 0
 *  -i                     use indirect rendering connection only
 *  -l                     print interesting OpenGL limits (added 5 Sep 2002)
 *
 * Brian Paul  26 January 2000
 */
/*}}}*/

/*{{{  includes*/
#define GLX_GLXEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES

#include <assert.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "glinfo_common.h"


#ifndef GLX_NONE_EXT
#define GLX_NONE_EXT  0x8000
#endif

#ifndef GLX_TRANSPARENT_RGB
#define GLX_TRANSPARENT_RGB 0x8008
#endif

#ifndef GLX_RGBA_BIT
#define GLX_RGBA_BIT      0x00000001
#endif

#ifndef GLX_COLOR_INDEX_BIT
#define GLX_COLOR_INDEX_BIT   0x00000002
#endif
/*}}}*/


/*{{{*/
struct visual_attribs
{
   /* X visual attribs */
   int id;    /* May be visual ID or FBConfig ID */
   int vis_id;    /* Visual ID.  Only set for FBConfigs */
   int klass;
   int depth;
   int redMask, greenMask, blueMask;
   int colormapSize;
   int bitsPerRGB;

   /* GL visual attribs */
   int supportsGL;
   int drawableType;
   int transparentType;
   int transparentRedValue;
   int transparentGreenValue;
   int transparentBlueValue;
   int transparentAlphaValue;
   int transparentIndexValue;
   int bufferSize;
   int level;
   int render_type;
   int doubleBuffer;
   int stereo;
   int auxBuffers;
   int redSize, greenSize, blueSize, alphaSize;
   int depthSize;
   int stencilSize;
   int accumRedSize, accumGreenSize, accumBlueSize, accumAlphaSize;
   int numSamples, numMultisample;
   int visualCaveat;
   int floatComponents;
   int packedfloatComponents;
   int srgb;
};
/*}}}*/


/**
 * Version of the context that was created
 *
 * 20, 21, 30, 31, 32, etc.
 */
static int version;

/**
 * GL Error checking/warning.
 */
static void CheckError(int line)
{
   int n;
   n = glGetError();
   if (n)
      printf("Warning: GL error 0x%x at line %d\n", n, line);
}


static void print_display_info(Display *dpy)
{
   printf("name of display: %s\n", DisplayString(dpy));
}


/**
 * Choose a simple FB Config.
 */
static GLXFBConfig * choose_fb_config(Display *dpy, int scrnum)
{
   int fbAttribSingle[] = {
      GLX_RENDER_TYPE,   GLX_RGBA_BIT,
      GLX_RED_SIZE,      1,
      GLX_GREEN_SIZE,    1,
      GLX_BLUE_SIZE,     1,
      GLX_DOUBLEBUFFER,  False,
      None };
   int fbAttribDouble[] = {
      GLX_RENDER_TYPE,   GLX_RGBA_BIT,
      GLX_RED_SIZE,      1,
      GLX_GREEN_SIZE,    1,
      GLX_BLUE_SIZE,     1,
      GLX_DOUBLEBUFFER,  True,
      None };
   GLXFBConfig *configs;
   int nConfigs;

   configs = glXChooseFBConfig(dpy, scrnum, fbAttribSingle, &nConfigs);
   if (!configs)
      configs = glXChooseFBConfig(dpy, scrnum, fbAttribDouble, &nConfigs);

   return configs;
}


static Bool CreateContextErrorFlag;

static int create_context_error_handler(Display *dpy, XErrorEvent *error)
{
   (void) dpy;
   (void) error->error_code;
   CreateContextErrorFlag = True;
   return 0;
}


/**
 * Try to create a GLX context of the given version with flags/options.
 * Note: A version number is required in order to get a core profile
 * (at least w/ NVIDIA).
 */
static GLXContext create_context_flags(Display *dpy, GLXFBConfig fbconfig, int major, int minor,
                     int contextFlags, int profileMask, Bool direct)
{
#ifdef GLX_ARB_create_context
   static PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB_func = 0;
   static Bool firstCall = True;
   int (*old_handler)(Display *, XErrorEvent *);
   GLXContext context;
   int attribs[20];
   int n = 0;

   if (firstCall) {
      /* See if we have GLX_ARB_create_context_profile and get pointer to
       * glXCreateContextAttribsARB() function.
       */
      const char *glxExt = glXQueryExtensionsString(dpy, 0);
      if (extension_supported("GLX_ARB_create_context_profile", glxExt)) {
         glXCreateContextAttribsARB_func = (PFNGLXCREATECONTEXTATTRIBSARBPROC)
            glXGetProcAddress((const GLubyte *) "glXCreateContextAttribsARB");
      }
      firstCall = False;
   }

   if (!glXCreateContextAttribsARB_func)
      return 0;

   /* setup attribute array */
   if (major) {
      attribs[n++] = GLX_CONTEXT_MAJOR_VERSION_ARB;
      attribs[n++] = major;
      attribs[n++] = GLX_CONTEXT_MINOR_VERSION_ARB;
      attribs[n++] = minor;
   }
   if (contextFlags) {
      attribs[n++] = GLX_CONTEXT_FLAGS_ARB;
      attribs[n++] = contextFlags;
   }
#ifdef GLX_ARB_create_context_profile
   if (profileMask) {
      attribs[n++] = GLX_CONTEXT_PROFILE_MASK_ARB;
      attribs[n++] = profileMask;
   }
#endif
   attribs[n++] = 0;

   /* install X error handler */
   old_handler = XSetErrorHandler(create_context_error_handler);
   CreateContextErrorFlag = False;

   /* try creating context */
   context = glXCreateContextAttribsARB_func(dpy,
                                             fbconfig,
                                             0, /* share_context */
                                             direct,
                                             attribs);

   /* restore error handler */
   XSetErrorHandler(old_handler);

   if (CreateContextErrorFlag)
      context = 0;

   if (context && direct) {
      if (!glXIsDirect(dpy, context)) {
         glXDestroyContext(dpy, context);
         return 0;
      }
   }

   return context;
#else
   return 0;
#endif
}


/**
 * Try to create a GLX context of the newest version.
 */
static GLXContext create_context_with_config(Display *dpy, GLXFBConfig config,
                           Bool coreProfile, Bool es2Profile, Bool direct)
{
   GLXContext ctx = 0;

   if (coreProfile) {
      /* Try to create a core profile, starting with the newest version of
       * GL that we're aware of.  If we don't specify the version
       */
      int i;
      for (i = 0; gl_versions[i].major > 0; i++) {
          /* don't bother below GL 3.0 */
          if (gl_versions[i].major == 3 &&
              gl_versions[i].minor == 0)
             return 0;
         ctx = create_context_flags(dpy, config,
                                    gl_versions[i].major,
                                    gl_versions[i].minor,
                                    0x0,
                                    GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
                                    direct);
         if (ctx)
            return ctx;
      }
      /* couldn't get core profile context */
      return 0;
   }

   if (es2Profile) {
#ifdef GLX_CONTEXT_ES2_PROFILE_BIT_EXT
      if (extension_supported("GLX_EXT_create_context_es2_profile",
                              glXQueryExtensionsString(dpy, 0))) {
         ctx = create_context_flags(dpy, config, 2, 0, 0x0,
                                    GLX_CONTEXT_ES2_PROFILE_BIT_EXT,
                                    direct);
         return ctx;
      }
#endif
      return 0;
   }

   /* GLX should return a context of the latest GL version that supports
    * the full profile.
    */
   ctx = glXCreateNewContext(dpy, config, GLX_RGBA_TYPE, NULL, direct);

   /* make sure the context is direct, if direct was requested */
   if (ctx && direct) {
      if (!glXIsDirect(dpy, ctx)) {
         glXDestroyContext(dpy, ctx);
         return 0;
      }
   }

   return ctx;
}


static XVisualInfo * choose_xvisinfo(Display *dpy, int scrnum)
{
   int attribSingle[] = {
      GLX_RGBA,
      GLX_RED_SIZE, 1,
      GLX_GREEN_SIZE, 1,
      GLX_BLUE_SIZE, 1,
      None };
   int attribDouble[] = {
      GLX_RGBA,
      GLX_RED_SIZE, 1,
      GLX_GREEN_SIZE, 1,
      GLX_BLUE_SIZE, 1,
      GLX_DOUBLEBUFFER,
      None };
   XVisualInfo *visinfo;

   visinfo = glXChooseVisual(dpy, scrnum, attribSingle);
   if (!visinfo)
      visinfo = glXChooseVisual(dpy, scrnum, attribDouble);

   return visinfo;
}


static void query_renderer(void)
{
#ifdef GLX_MESA_query_renderer
    PFNGLXQUERYCURRENTRENDERERINTEGERMESAPROC queryInteger;
    PFNGLXQUERYCURRENTRENDERERSTRINGMESAPROC queryString;
    unsigned int v[3];

    queryInteger = (PFNGLXQUERYCURRENTRENDERERINTEGERMESAPROC)
        glXGetProcAddressARB((const GLubyte *)
                             "glXQueryCurrentRendererIntegerMESA");
    queryString = (PFNGLXQUERYCURRENTRENDERERSTRINGMESAPROC)
        glXGetProcAddressARB((const GLubyte *)
                             "glXQueryCurrentRendererStringMESA");

    printf("Extended renderer info (GLX_MESA_query_renderer):\n");
    queryInteger(GLX_RENDERER_VENDOR_ID_MESA, v);
    printf("    Vendor: %s (0x%x)\n",
           queryString(GLX_RENDERER_VENDOR_ID_MESA), *v);
    queryInteger(GLX_RENDERER_DEVICE_ID_MESA, v);
    printf("    Device: %s (0x%x)\n",
           queryString(GLX_RENDERER_DEVICE_ID_MESA), *v);
    queryInteger(GLX_RENDERER_VERSION_MESA, v);
    printf("    Version: %d.%d.%d\n", v[0], v[1], v[2]);
    queryInteger(GLX_RENDERER_ACCELERATED_MESA, v);
    printf("    Accelerated: %s\n", *v ? "yes" : "no");
    queryInteger(GLX_RENDERER_VIDEO_MEMORY_MESA, v);
    printf("    Video memory: %dMB\n", *v);
    queryInteger(GLX_RENDERER_UNIFIED_MEMORY_ARCHITECTURE_MESA, v);
    printf("    Unified memory: %s\n", *v ? "yes" : "no");
    queryInteger(GLX_RENDERER_PREFERRED_PROFILE_MESA, v);
    printf("    Preferred profile: %s (0x%x)\n",
           *v == GLX_CONTEXT_CORE_PROFILE_BIT_ARB ? "core" :
           *v == GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB ? "compat" :
           "unknown", *v);
    queryInteger(GLX_RENDERER_OPENGL_CORE_PROFILE_VERSION_MESA, v);
    printf("    Max core profile version: %d.%d\n", v[0], v[1]);
    queryInteger(GLX_RENDERER_OPENGL_COMPATIBILITY_PROFILE_VERSION_MESA, v);
    printf("    Max compat profile version: %d.%d\n", v[0], v[1]);
    queryInteger(GLX_RENDERER_OPENGL_ES_PROFILE_VERSION_MESA, v);
    printf("    Max GLES1 profile version: %d.%d\n", v[0], v[1]);
    queryInteger(GLX_RENDERER_OPENGL_ES2_PROFILE_VERSION_MESA, v);
    printf("    Max GLES[23] profile version: %d.%d\n", v[0], v[1]);
#endif
}


static Bool print_screen_info(Display *dpy, int scrnum,
                  const struct options *opts,
                  Bool coreProfile, Bool es2Profile, Bool limits,
                  Bool coreWorked)
{
   Window win;
   XSetWindowAttributes attr;
   unsigned long mask;
   Window root;
   GLXContext ctx = NULL;
   XVisualInfo *visinfo;
   int width = 100, height = 100;
   GLXFBConfig *fbconfigs;
   const char *oglstring = coreProfile ? "OpenGL core profile" :
                           es2Profile ? "OpenGL ES profile" : "OpenGL";

   root = RootWindow(dpy, scrnum);

   /*
    * Choose FBConfig or XVisualInfo and create a context.
    */
   fbconfigs = choose_fb_config(dpy, scrnum);
   if (fbconfigs) {
      ctx = create_context_with_config(dpy, fbconfigs[0],
                                       coreProfile, es2Profile,
                                       opts->allowDirect);
      if (!ctx && opts->allowDirect && !coreProfile) {
         /* try indirect */
         ctx = create_context_with_config(dpy, fbconfigs[0],
                                          coreProfile, es2Profile, False);
      }

      visinfo = glXGetVisualFromFBConfig(dpy, fbconfigs[0]);
      XFree(fbconfigs);
   }
   else if (!coreProfile && !es2Profile) {
      visinfo = choose_xvisinfo(dpy, scrnum);
      if (visinfo)
         ctx = glXCreateContext(dpy, visinfo, NULL, opts->allowDirect);
   } else
      visinfo = NULL;

   if (!visinfo && !coreProfile && !es2Profile) {
      fprintf(stderr, "Error: couldn't find RGB GLX visual or fbconfig\n");
      return False;
   }

   if (!ctx) {
      if (!coreProfile && !es2Profile)
         fprintf(stderr, "Error: glXCreateContext failed\n");
      XFree(visinfo);
      return False;
   }

   /*
    * Create a window so that we can just bind the context.
    */
   attr.background_pixel = 0;
   attr.border_pixel = 0;
   attr.colormap = XCreateColormap(dpy, root, visinfo->visual, AllocNone);
   attr.event_mask = StructureNotifyMask | ExposureMask;
   mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask;
   win = XCreateWindow(dpy, root, 0, 0, width, height,
                       0, visinfo->depth, InputOutput,
                       visinfo->visual, mask, &attr);

   if (glXMakeCurrent(dpy, win, ctx)) {
      const char *serverVendor = glXQueryServerString(dpy, scrnum, GLX_VENDOR);
      const char *serverVersion = glXQueryServerString(dpy, scrnum, GLX_VERSION);
      const char *serverExtensions = glXQueryServerString(dpy, scrnum, GLX_EXTENSIONS);
      const char *clientVendor = glXGetClientString(dpy, GLX_VENDOR);
      const char *clientVersion = glXGetClientString(dpy, GLX_VERSION);
      const char *clientExtensions = glXGetClientString(dpy, GLX_EXTENSIONS);
      const char *glxExtensions = glXQueryExtensionsString(dpy, scrnum);
      const char *glVendor = (const char *) glGetString(GL_VENDOR);
      const char *glRenderer = (const char *) glGetString(GL_RENDERER);
      const char *glVersion = (const char *) glGetString(GL_VERSION);
      char *glExtensions = NULL;
      int glxVersionMajor = 0;
      int glxVersionMinor = 0;
      char *displayName = NULL;
      char *colon = NULL, *period = NULL;
      struct ext_functions extfuncs;

      CheckError(__LINE__);

      /* Get some ext functions */
      extfuncs.GetProgramivARB = (GETPROGRAMIVARBPROC)
         glXGetProcAddressARB((GLubyte *) "glGetProgramivARB");
      extfuncs.GetStringi = (GETSTRINGIPROC)
         glXGetProcAddressARB((GLubyte *) "glGetStringi");
      extfuncs.GetConvolutionParameteriv = (GETCONVOLUTIONPARAMETERIVPROC)
         glXGetProcAddressARB((GLubyte *) "glGetConvolutionParameteriv");

      if (!glXQueryVersion(dpy, & glxVersionMajor, & glxVersionMinor)) {
         fprintf(stderr, "Error: glXQueryVersion failed\n");
         exit(1);
      }

      /* Get list of GL extensions */
      if (coreProfile && extfuncs.GetStringi)
         glExtensions = build_core_profile_extension_list(&extfuncs);
      if (!glExtensions) {
         coreProfile = False;
         glExtensions = (char *) glGetString(GL_EXTENSIONS);
      }

      CheckError(__LINE__);

      if (!coreWorked) {
         /* Strip the screen number from the display name, if present. */
         if (!(displayName = (char *) malloc(strlen(DisplayString(dpy)) + 1))) {
            fprintf(stderr, "Error: malloc() failed\n");
            exit(1);
         }
         strcpy(displayName, DisplayString(dpy));
         colon = strrchr(displayName, ':');
         if (colon) {
            period = strchr(colon, '.');
            if (period)
               *period = '\0';
         }

         printf("display: %s  screen: %d\n", displayName, scrnum);
         free(displayName);
         printf("direct rendering: ");
         if (glXIsDirect(dpy, ctx)) {
            printf("Yes\n");
         }
         else {
            if (!opts->allowDirect) {
               printf("No (-i specified)\n");
            }
            else if (getenv("LIBGL_ALWAYS_INDIRECT")) {
               printf("No (LIBGL_ALWAYS_INDIRECT set)\n");
            }
            else {
               printf("No (If you want to find out why, try setting "
                      "LIBGL_DEBUG=verbose)\n");
            }
         }
         if (opts->mode != Brief) {
            printf("server glx vendor string: %s\n", serverVendor);
            printf("server glx version string: %s\n", serverVersion);
            printf("server glx extensions:\n");
            print_extension_list(serverExtensions, opts->singleLine);
            printf("client glx vendor string: %s\n", clientVendor);
            printf("client glx version string: %s\n", clientVersion);
            printf("client glx extensions:\n");
            print_extension_list(clientExtensions, opts->singleLine);
            printf("GLX version: %u.%u\n", glxVersionMajor, glxVersionMinor);
            printf("GLX extensions:\n");
            print_extension_list(glxExtensions, opts->singleLine);
         }
         if (strstr(glxExtensions, "GLX_MESA_query_renderer"))
            query_renderer();
         print_gpu_memory_info(glExtensions);
         printf("OpenGL vendor string: %s\n", glVendor);
         printf("OpenGL renderer string: %s\n", glRenderer);
      } else
         printf("\n");

      printf("%s version string: %s\n", oglstring, glVersion);

      version = (glVersion[0] - '0') * 10 + (glVersion[2] - '0');

      CheckError(__LINE__);

#ifdef GL_VERSION_2_0
      if (version >= 20) {
         char *v = (char *) glGetString(GL_SHADING_LANGUAGE_VERSION);
         printf("%s shading language version string: %s\n", oglstring, v);
      }
#endif
      CheckError(__LINE__);
#ifdef GL_VERSION_3_0
      if (version >= 30 && !es2Profile) {
         GLint flags;
         glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
         printf("%s context flags: %s\n", oglstring, context_flags_string(flags));
      }
#endif
      CheckError(__LINE__);
#ifdef GL_VERSION_3_2
      if (version >= 32 && !es2Profile) {
         GLint mask;
         glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);
         printf("%s profile mask: %s\n", oglstring, profile_mask_string(mask));
      }
#endif

      CheckError(__LINE__);

      if (opts->mode != Brief) {
         printf("%s extensions:\n", oglstring);
         print_extension_list(glExtensions, opts->singleLine);
      }

      if (limits) {
         print_limits(glExtensions, oglstring, version, &extfuncs);
      }

      if (coreProfile)
         free(glExtensions);
   }
   else {
      fprintf(stderr, "Error: glXMakeCurrent failed\n");
   }

   glXDestroyContext(dpy, ctx);
   XFree(visinfo);
   XDestroyWindow(dpy, win);
   XSync(dpy, 1);
   return True;
}


static const char * visual_class_name(int cls)
{
   switch (cls) {
      case StaticColor:
         return "StaticColor";
      case PseudoColor:
         return "PseudoColor";
      case StaticGray:
         return "StaticGray";
      case GrayScale:
         return "GrayScale";
      case TrueColor:
         return "TrueColor";
      case DirectColor:
         return "DirectColor";
      default:
         return "";
   }
}

static const char * visual_drawable_type(int type)
{
   const static struct bit_info bits[] = {
      { GLX_WINDOW_BIT, "window" },
      { GLX_PIXMAP_BIT, "pixmap" },
      { GLX_PBUFFER_BIT, "pbuffer" }
   };

   return bitmask_to_string(bits, ELEMENTS(bits), type);
}

static const char * visual_class_abbrev(int cls)
{
   switch (cls) {
      case StaticColor:
         return "sc";
      case PseudoColor:
         return "pc";
      case StaticGray:
         return "sg";
      case GrayScale:
         return "gs";
      case TrueColor:
         return "tc";
      case DirectColor:
         return "dc";
      default:
         return "";
   }
}

static const char * visual_render_type_name(int type)
{
   switch (type) {
      case GLX_RGBA_UNSIGNED_FLOAT_BIT_EXT:
         return "ufloat";
      case GLX_RGBA_FLOAT_BIT_ARB:
         return "float";
      case GLX_RGBA_BIT:
         return "rgba";
      case GLX_COLOR_INDEX_BIT:
         return "ci";
      case GLX_RGBA_BIT | GLX_COLOR_INDEX_BIT:
         return "rgba|ci";
      default:
         return "";
   }
}

static const char * caveat_string(int caveat)
{
   switch (caveat) {
#ifdef GLX_EXT_visual_rating
      case GLX_SLOW_VISUAL_EXT:
         return "Slow";
      case GLX_NON_CONFORMANT_VISUAL_EXT:
         return "Ncon";
      case GLX_NONE_EXT:
         /* fall-through */
#endif
      case 0:
         /* fall-through */
      default:
         return "None";
   }
}


static Bool get_visual_attribs(Display *dpy, XVisualInfo *vInfo,
                   struct visual_attribs *attribs)
{
   const char *ext = glXQueryExtensionsString(dpy, vInfo->screen);
   int rgba;

   memset(attribs, 0, sizeof(struct visual_attribs));

   attribs->id = vInfo->visualid;
#if defined(__cplusplus) || defined(c_plusplus)
   attribs->klass = vInfo->c_class;
#else
   attribs->klass = vInfo->class;
#endif
   attribs->depth = vInfo->depth;
   attribs->redMask = vInfo->red_mask;
   attribs->greenMask = vInfo->green_mask;
   attribs->blueMask = vInfo->blue_mask;
   attribs->colormapSize = vInfo->colormap_size;
   attribs->bitsPerRGB = vInfo->bits_per_rgb;

   if (glXGetConfig(dpy, vInfo, GLX_USE_GL, &attribs->supportsGL) != 0 ||
       !attribs->supportsGL)
      return False;
   glXGetConfig(dpy, vInfo, GLX_BUFFER_SIZE, &attribs->bufferSize);
   glXGetConfig(dpy, vInfo, GLX_LEVEL, &attribs->level);
   glXGetConfig(dpy, vInfo, GLX_RGBA, &rgba);
   if (rgba)
      attribs->render_type = GLX_RGBA_BIT;
   else
      attribs->render_type = GLX_COLOR_INDEX_BIT;

   glXGetConfig(dpy, vInfo, GLX_DRAWABLE_TYPE, &attribs->drawableType);
   glXGetConfig(dpy, vInfo, GLX_DOUBLEBUFFER, &attribs->doubleBuffer);
   glXGetConfig(dpy, vInfo, GLX_STEREO, &attribs->stereo);
   glXGetConfig(dpy, vInfo, GLX_AUX_BUFFERS, &attribs->auxBuffers);
   glXGetConfig(dpy, vInfo, GLX_RED_SIZE, &attribs->redSize);
   glXGetConfig(dpy, vInfo, GLX_GREEN_SIZE, &attribs->greenSize);
   glXGetConfig(dpy, vInfo, GLX_BLUE_SIZE, &attribs->blueSize);
   glXGetConfig(dpy, vInfo, GLX_ALPHA_SIZE, &attribs->alphaSize);
   glXGetConfig(dpy, vInfo, GLX_DEPTH_SIZE, &attribs->depthSize);
   glXGetConfig(dpy, vInfo, GLX_STENCIL_SIZE, &attribs->stencilSize);
   glXGetConfig(dpy, vInfo, GLX_ACCUM_RED_SIZE, &attribs->accumRedSize);
   glXGetConfig(dpy, vInfo, GLX_ACCUM_GREEN_SIZE, &attribs->accumGreenSize);
   glXGetConfig(dpy, vInfo, GLX_ACCUM_BLUE_SIZE, &attribs->accumBlueSize);
   glXGetConfig(dpy, vInfo, GLX_ACCUM_ALPHA_SIZE, &attribs->accumAlphaSize);

   /* get transparent pixel stuff */
   glXGetConfig(dpy, vInfo,GLX_TRANSPARENT_TYPE, &attribs->transparentType);
   if (attribs->transparentType == GLX_TRANSPARENT_RGB) {
     glXGetConfig(dpy, vInfo, GLX_TRANSPARENT_RED_VALUE, &attribs->transparentRedValue);
     glXGetConfig(dpy, vInfo, GLX_TRANSPARENT_GREEN_VALUE, &attribs->transparentGreenValue);
     glXGetConfig(dpy, vInfo, GLX_TRANSPARENT_BLUE_VALUE, &attribs->transparentBlueValue);
     glXGetConfig(dpy, vInfo, GLX_TRANSPARENT_ALPHA_VALUE, &attribs->transparentAlphaValue);
   }
   else if (attribs->transparentType == GLX_TRANSPARENT_INDEX) {
     glXGetConfig(dpy, vInfo, GLX_TRANSPARENT_INDEX_VALUE, &attribs->transparentIndexValue);
   }

   /* multisample attribs */
#ifdef GLX_ARB_multisample
   if (ext && strstr(ext, "GLX_ARB_multisample")) {
      glXGetConfig(dpy, vInfo, GLX_SAMPLE_BUFFERS_ARB, &attribs->numMultisample);
      glXGetConfig(dpy, vInfo, GLX_SAMPLES_ARB, &attribs->numSamples);
   }
#endif
   else {
      attribs->numSamples = 0;
      attribs->numMultisample = 0;
   }

#if defined(GLX_EXT_visual_rating)
   if (ext && strstr(ext, "GLX_EXT_visual_rating")) {
      glXGetConfig(dpy, vInfo, GLX_VISUAL_CAVEAT_EXT, &attribs->visualCaveat);
   }
   else {
      attribs->visualCaveat = GLX_NONE_EXT;
   }
#else
   attribs->visualCaveat = 0;
#endif

#if defined(GLX_EXT_framebuffer_sRGB)
   if (ext && strstr(ext, "GLX_EXT_framebuffer_sRGB")) {
      glXGetConfig(dpy, vInfo, GLX_FRAMEBUFFER_SRGB_CAPABLE_EXT, &attribs->srgb);
   }
#endif

   return True;
}

#ifdef GLX_VERSION_1_3

static int glx_token_to_visual_class(int visual_type)
{
   switch (visual_type) {
   case GLX_TRUE_COLOR:
      return TrueColor;
   case GLX_DIRECT_COLOR:
      return DirectColor;
   case GLX_PSEUDO_COLOR:
      return PseudoColor;
   case GLX_STATIC_COLOR:
      return StaticColor;
   case GLX_GRAY_SCALE:
      return GrayScale;
   case GLX_STATIC_GRAY:
      return StaticGray;
   case GLX_NONE:
   default:
      return None;
   }
}

static Bool get_fbconfig_attribs(Display *dpy, GLXFBConfig fbconfig,
                     struct visual_attribs *attribs)
{
   const char *ext = glXQueryExtensionsString(dpy, 0);
   int visual_type;
   XVisualInfo *vInfo;

   memset(attribs, 0, sizeof(struct visual_attribs));

   glXGetFBConfigAttrib(dpy, fbconfig, GLX_FBCONFIG_ID, &attribs->id);

   vInfo = glXGetVisualFromFBConfig(dpy, fbconfig);

   if (vInfo != NULL) {
      attribs->vis_id = vInfo->visualid;
      attribs->depth = vInfo->depth;
      attribs->redMask = vInfo->red_mask;
      attribs->greenMask = vInfo->green_mask;
      attribs->blueMask = vInfo->blue_mask;
      attribs->colormapSize = vInfo->colormap_size;
      attribs->bitsPerRGB = vInfo->bits_per_rgb;
   }

   glXGetFBConfigAttrib(dpy, fbconfig, GLX_X_VISUAL_TYPE, &visual_type);
   attribs->klass = glx_token_to_visual_class(visual_type);

   glXGetFBConfigAttrib(dpy, fbconfig, GLX_DRAWABLE_TYPE, &attribs->drawableType);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_BUFFER_SIZE, &attribs->bufferSize);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_LEVEL, &attribs->level);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_RENDER_TYPE, &attribs->render_type);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_DOUBLEBUFFER, &attribs->doubleBuffer);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_STEREO, &attribs->stereo);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_AUX_BUFFERS, &attribs->auxBuffers);

   glXGetFBConfigAttrib(dpy, fbconfig, GLX_RED_SIZE, &attribs->redSize);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_GREEN_SIZE, &attribs->greenSize);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_BLUE_SIZE, &attribs->blueSize);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_ALPHA_SIZE, &attribs->alphaSize);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_DEPTH_SIZE, &attribs->depthSize);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_STENCIL_SIZE, &attribs->stencilSize);

   glXGetFBConfigAttrib(dpy, fbconfig, GLX_ACCUM_RED_SIZE, &attribs->accumRedSize);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_ACCUM_GREEN_SIZE, &attribs->accumGreenSize);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_ACCUM_BLUE_SIZE, &attribs->accumBlueSize);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_ACCUM_ALPHA_SIZE, &attribs->accumAlphaSize);

   /* get transparent pixel stuff */
   glXGetFBConfigAttrib(dpy, fbconfig,GLX_TRANSPARENT_TYPE, &attribs->transparentType);
   if (attribs->transparentType == GLX_TRANSPARENT_RGB) {
     glXGetFBConfigAttrib(dpy, fbconfig, GLX_TRANSPARENT_RED_VALUE, &attribs->transparentRedValue);
     glXGetFBConfigAttrib(dpy, fbconfig, GLX_TRANSPARENT_GREEN_VALUE, &attribs->transparentGreenValue);
     glXGetFBConfigAttrib(dpy, fbconfig, GLX_TRANSPARENT_BLUE_VALUE, &attribs->transparentBlueValue);
     glXGetFBConfigAttrib(dpy, fbconfig, GLX_TRANSPARENT_ALPHA_VALUE, &attribs->transparentAlphaValue);
   }
   else if (attribs->transparentType == GLX_TRANSPARENT_INDEX) {
     glXGetFBConfigAttrib(dpy, fbconfig, GLX_TRANSPARENT_INDEX_VALUE, &attribs->transparentIndexValue);
   }

   glXGetFBConfigAttrib(dpy, fbconfig, GLX_SAMPLE_BUFFERS, &attribs->numMultisample);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_SAMPLES, &attribs->numSamples);
   glXGetFBConfigAttrib(dpy, fbconfig, GLX_CONFIG_CAVEAT, &attribs->visualCaveat);

#if defined(GLX_NV_float_buffer)
   if (ext && strstr(ext, "GLX_NV_float_buffer")) {
      glXGetFBConfigAttrib(dpy, fbconfig, GLX_FLOAT_COMPONENTS_NV, &attribs->floatComponents);
   }
#endif
#if defined(GLX_ARB_fbconfig_float)
   if (ext && strstr(ext, "GLX_ARB_fbconfig_float")) {
      if (attribs->render_type & GLX_RGBA_FLOAT_BIT_ARB) {
         attribs->floatComponents = True;
      }
   }
#endif
#if defined(GLX_EXT_fbconfig_packed_float)
   if (ext && strstr(ext, "GLX_EXT_fbconfig_packed_float")) {
      if (attribs->render_type & GLX_RGBA_UNSIGNED_FLOAT_BIT_EXT) {
         attribs->packedfloatComponents = True;
      }
   }
#endif

#if defined(GLX_EXT_framebuffer_sRGB)
   if (ext && strstr(ext, "GLX_EXT_framebuffer_sRGB")) {
      glXGetFBConfigAttrib(dpy, fbconfig, GLX_FRAMEBUFFER_SRGB_CAPABLE_EXT, &attribs->srgb);
   }
#endif
   return True;
}

#endif



static void print_visual_attribs_verbose(const struct visual_attribs *attribs,
                             int fbconfigs)
{
   if (fbconfigs) {
      printf("FBConfig ID: %x  Visual ID=%x  depth=%d  class=%s, type=%s\n",
             attribs->id, attribs->vis_id, attribs->depth,
             visual_class_name(attribs->klass),
             visual_drawable_type(attribs->drawableType));
   }
   else {
      printf("Visual ID: %x  depth=%d  class=%s, type=%s\n",
             attribs->id, attribs->depth, visual_class_name(attribs->klass),
             visual_drawable_type(attribs->drawableType));
   }
   printf("    bufferSize=%d level=%d renderType=%s doubleBuffer=%d stereo=%d\n",
          attribs->bufferSize, attribs->level,
          visual_render_type_name(attribs->render_type),
          attribs->doubleBuffer, attribs->stereo);
   printf("    rgba: redSize=%d greenSize=%d blueSize=%d alphaSize=%d float=%c sRGB=%c\n",
          attribs->redSize, attribs->greenSize,
          attribs->blueSize, attribs->alphaSize,
          attribs->packedfloatComponents ? 'P' : attribs->floatComponents ? 'Y' : 'N',
          attribs->srgb ? 'Y' : 'N');
   printf("    auxBuffers=%d depthSize=%d stencilSize=%d\n",
          attribs->auxBuffers, attribs->depthSize, attribs->stencilSize);
   printf("    accum: redSize=%d greenSize=%d blueSize=%d alphaSize=%d\n",
          attribs->accumRedSize, attribs->accumGreenSize,
          attribs->accumBlueSize, attribs->accumAlphaSize);
   printf("    multiSample=%d  multiSampleBuffers=%d\n",
          attribs->numSamples, attribs->numMultisample);
#ifdef GLX_EXT_visual_rating
   if (attribs->visualCaveat == GLX_NONE_EXT || attribs->visualCaveat == 0)
      printf("    visualCaveat=None\n");
   else if (attribs->visualCaveat == GLX_SLOW_VISUAL_EXT)
      printf("    visualCaveat=Slow\n");
   else if (attribs->visualCaveat == GLX_NON_CONFORMANT_VISUAL_EXT)
      printf("    visualCaveat=Nonconformant\n");
#endif
   if (attribs->transparentType == GLX_NONE) {
     printf("    Opaque.\n");
   }
   else if (attribs->transparentType == GLX_TRANSPARENT_RGB) {
     printf("    Transparent RGB: Red=%d Green=%d Blue=%d Alpha=%d\n",attribs->transparentRedValue,attribs->transparentGreenValue,attribs->transparentBlueValue,attribs->transparentAlphaValue);
   }
   else if (attribs->transparentType == GLX_TRANSPARENT_INDEX) {
     printf("    Transparent index=%d\n",attribs->transparentIndexValue);
   }
}


static void print_visual_attribs_short_header(void)
{
 printf("    visual  x   bf lv rg d st  colorbuffer  sr ax dp st accumbuffer  ms  cav\n");
 printf("  id dep cl sp  sz l  ci b ro  r  g  b  a F gb bf th cl  r  g  b  a ns b eat\n");
 printf("----------------------------------------------------------------------------\n");
}


static void print_visual_attribs_short(const struct visual_attribs *attribs)
{
   const char *caveat = caveat_string(attribs->visualCaveat);

   printf("0x%03x %2d %2s %2d %3d %2d %c%c %c %c  %2d %2d %2d %2d %c  %c %2d %2d %2d",
          attribs->id,
          attribs->depth,
          visual_class_abbrev(attribs->klass),
          attribs->transparentType != GLX_NONE,
          attribs->bufferSize,
          attribs->level,
          (attribs->render_type & GLX_RGBA_BIT) ? 'r' : ' ',
          (attribs->render_type & GLX_COLOR_INDEX_BIT) ? 'c' : ' ',
          attribs->doubleBuffer ? 'y' : '.',
          attribs->stereo ? 'y' : '.',
          attribs->redSize, attribs->greenSize,
          attribs->blueSize, attribs->alphaSize,
          attribs->packedfloatComponents ? 'u' : attribs->floatComponents ? 'f' : '.',
          attribs->srgb ? 's' : '.',
          attribs->auxBuffers,
          attribs->depthSize,
          attribs->stencilSize
          );

   printf(" %2d %2d %2d %2d %2d %1d %s\n",
          attribs->accumRedSize, attribs->accumGreenSize,
          attribs->accumBlueSize, attribs->accumAlphaSize,
          attribs->numSamples, attribs->numMultisample,
          caveat
          );
}


static void print_visual_attribs_long_header(void)
{
 printf("Vis   Vis   Visual Trans  buff lev render DB ste  r   g   b   a      s  aux dep ste  accum buffer   MS   MS         \n");
 printf(" ID  Depth   Type  parent size el   type     reo sz  sz  sz  sz flt rgb buf th  ncl  r   g   b   a  num bufs caveats\n");
 printf("--------------------------------------------------------------------------------------------------------------------\n");
}


static void print_visual_attribs_long(const struct visual_attribs *attribs)
{
   const char *caveat = caveat_string(attribs->visualCaveat);

   printf("0x%3x %2d %-11s %2d    %3d %2d  %4s %3d %3d %3d %3d %3d %3d",
          attribs->id,
          attribs->depth,
          visual_class_name(attribs->klass),
          attribs->transparentType != GLX_NONE,
          attribs->bufferSize,
          attribs->level,
          visual_render_type_name(attribs->render_type),
          attribs->doubleBuffer,
          attribs->stereo,
          attribs->redSize, attribs->greenSize,
          attribs->blueSize, attribs->alphaSize
          );

   printf("  %c   %c %3d %4d %2d %3d %3d %3d %3d  %2d  %2d %6s\n",
          attribs->floatComponents ? 'f' : '.',
          attribs->srgb ? 's' : '.',
          attribs->auxBuffers,
          attribs->depthSize,
          attribs->stencilSize,
          attribs->accumRedSize, attribs->accumGreenSize,
          attribs->accumBlueSize, attribs->accumAlphaSize,
          attribs->numSamples, attribs->numMultisample,
          caveat
          );
}


static void print_visual_info(Display *dpy, int scrnum, InfoMode mode)
{
   XVisualInfo theTemplate;
   XVisualInfo *visuals;
   int numVisuals, numGlxVisuals;
   long mask;
   int i;
   struct visual_attribs attribs;

   /* get list of all visuals on this screen */
   theTemplate.screen = scrnum;
   mask = VisualScreenMask;
   visuals = XGetVisualInfo(dpy, mask, &theTemplate, &numVisuals);

   numGlxVisuals = 0;
   for (i = 0; i < numVisuals; i++) {
      if (get_visual_attribs(dpy, &visuals[i], &attribs))
         numGlxVisuals++;
   }

   if (numGlxVisuals == 0)
      return;

   printf("%d GLX Visuals\n", numGlxVisuals);

   if (mode == Normal)
      print_visual_attribs_short_header();
   else if (mode == Wide)
      print_visual_attribs_long_header();

   for (i = 0; i < numVisuals; i++) {
      if (!get_visual_attribs(dpy, &visuals[i], &attribs))
         continue;

      if (mode == Verbose)
         print_visual_attribs_verbose(&attribs, False);
      else if (mode == Normal)
         print_visual_attribs_short(&attribs);
      else if (mode == Wide) 
         print_visual_attribs_long(&attribs);
   }
   printf("\n");

   XFree(visuals);
}

#ifdef GLX_VERSION_1_3

static void print_fbconfig_info(Display *dpy, int scrnum, InfoMode mode)
{
   int numFBConfigs = 0;
   struct visual_attribs attribs;
   GLXFBConfig *fbconfigs;
   int i;

   /* get list of all fbconfigs on this screen */
   fbconfigs = glXGetFBConfigs(dpy, scrnum, &numFBConfigs);

   if (numFBConfigs == 0) {
      XFree(fbconfigs);
      return;
   }

   printf("%d GLXFBConfigs:\n", numFBConfigs);
   if (mode == Normal)
      print_visual_attribs_short_header();
   else if (mode == Wide)
      print_visual_attribs_long_header();

   for (i = 0; i < numFBConfigs; i++) {
      get_fbconfig_attribs(dpy, fbconfigs[i], &attribs);

      if (mode == Verbose) 
         print_visual_attribs_verbose(&attribs, True);
      else if (mode == Normal)
         print_visual_attribs_short(&attribs);
      else if (mode == Wide)
         print_visual_attribs_long(&attribs);
   }
   printf("\n");

   XFree(fbconfigs);
}

#endif

/*
 * Stand-alone Mesa doesn't really implement the GLX protocol so it
 * doesn't really know the GLX attributes associated with an X visual.
 * The first time a visual is presented to Mesa's pseudo-GLX it
 * attaches ancilliary buffers to it (like depth and stencil).
 * But that usually only works if glXChooseVisual is used.
 * This function calls glXChooseVisual() to sort of "prime the pump"
 * for Mesa's GLX so that the visuals that get reported actually
 * reflect what applications will see.
 * This has no effect when using true GLX.
 */
static void mesa_hack(Display *dpy, int scrnum)
{
   static int attribs[] = {
      GLX_RGBA,
      GLX_RED_SIZE, 1,
      GLX_GREEN_SIZE, 1,
      GLX_BLUE_SIZE, 1,
      GLX_DEPTH_SIZE, 1,
      GLX_STENCIL_SIZE, 1,
      GLX_ACCUM_RED_SIZE, 1,
      GLX_ACCUM_GREEN_SIZE, 1,
      GLX_ACCUM_BLUE_SIZE, 1,
      GLX_ACCUM_ALPHA_SIZE, 1,
      GLX_DOUBLEBUFFER,
      None
   };
   XVisualInfo *visinfo;

   visinfo = glXChooseVisual(dpy, scrnum, attribs);
   if (visinfo)
      XFree(visinfo);
}


/*
 * Examine all visuals to find the so-called best one.
 * We prefer deepest RGBA buffer with depth, stencil and accum
 * that has no caveats.
 */
static int find_best_visual(Display *dpy, int scrnum)
{
   XVisualInfo theTemplate;
   XVisualInfo *visuals;
   int numVisuals;
   long mask;
   int i;
   struct visual_attribs bestVis;

   /* get list of all visuals on this screen */
   theTemplate.screen = scrnum;
   mask = VisualScreenMask;
   visuals = XGetVisualInfo(dpy, mask, &theTemplate, &numVisuals);

   /* init bestVis with first visual info */
   get_visual_attribs(dpy, &visuals[0], &bestVis);

   /* try to find a "better" visual */
   for (i = 1; i < numVisuals; i++) {
      struct visual_attribs vis;

      get_visual_attribs(dpy, &visuals[i], &vis);

      /* always skip visuals with caveats */
      if (vis.visualCaveat != GLX_NONE_EXT)
         continue;

      /* see if this vis is better than bestVis */
      if ((!bestVis.supportsGL && vis.supportsGL) ||
          (bestVis.visualCaveat != GLX_NONE_EXT) ||
          (!(bestVis.render_type & GLX_RGBA_BIT) && (vis.render_type & GLX_RGBA_BIT)) ||
          (!bestVis.doubleBuffer && vis.doubleBuffer) ||
          (bestVis.redSize < vis.redSize) ||
          (bestVis.greenSize < vis.greenSize) ||
          (bestVis.blueSize < vis.blueSize) ||
          (bestVis.alphaSize < vis.alphaSize) ||
          (bestVis.depthSize < vis.depthSize) ||
          (bestVis.stencilSize < vis.stencilSize) ||
          (bestVis.accumRedSize < vis.accumRedSize)) {
         /* found a better visual */
         bestVis = vis;
      }
   }

   XFree(visuals);

   return bestVis.id;
}


int
main(int argc, char *argv[]) {
   Display *dpy;
   int numScreens, scrnum;
   struct options opts;
   Bool coreWorked;

   parse_args(argc, argv, &opts);

   dpy = XOpenDisplay(opts.displayName);
   if (!dpy) {
      fprintf(stderr, "Error: unable to open display %s\n",
              XDisplayName(opts.displayName));
      return -1;
   }

   if (opts.findBest) {
      int b;
      mesa_hack(dpy, 0);
      b = find_best_visual(dpy, 0);
      printf("%d\n", b);
   }
   else {
      numScreens = ScreenCount(dpy);
      print_display_info(dpy);
      for (scrnum = 0; scrnum < numScreens; scrnum++) {
         mesa_hack(dpy, scrnum);
         coreWorked = print_screen_info(dpy, scrnum, &opts,
                                        True, False, opts.limits, False);
         print_screen_info(dpy, scrnum, &opts, False, False,
                           opts.limits, coreWorked);
         print_screen_info(dpy, scrnum, &opts, False, True, False, True);

         printf("\n");

         if (opts.mode != Brief) {
            print_visual_info(dpy, scrnum, opts.mode);
#ifdef GLX_VERSION_1_3
            print_fbconfig_info(dpy, scrnum, opts.mode);
#endif
         }

         if (scrnum + 1 < numScreens)
            printf("\n\n");
      }
   }

   XCloseDisplay(dpy);

   return 0;
}
generated by cgit v0.10.2 at 2017-03-11 11:10:40 (GMT)
