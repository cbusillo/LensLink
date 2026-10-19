/* Just enough of libobs for matte.c to build in the standalone tests:
 * the frame fields and formats it touches, and a silent blog(). */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#define MAX_AV_PLANES 8
#define LOG_INFO 300

enum video_format {
	VIDEO_FORMAT_NONE,
	VIDEO_FORMAT_I420,
	VIDEO_FORMAT_NV12,
	VIDEO_FORMAT_I40A,
};

struct obs_source_frame {
	uint8_t *data[MAX_AV_PLANES];
	uint32_t linesize[MAX_AV_PLANES];
	uint32_t width;
	uint32_t height;
	uint64_t timestamp;
	enum video_format format;
};

static inline void blog(int level, const char *format, ...)
{
	(void)level;
	(void)format;
}
