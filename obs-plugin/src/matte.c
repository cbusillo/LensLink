#include "matte.h"

#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define MATTE_SLOTS 6
#define MATTE_MAX_SIDE 4096
/* A frame whose own matte was lost borrows the newest older one if it is
 * at most this old (two frames at 30 fps). Older than that, the subject
 * may have moved enough that opaque is the better miss. */
#define MATTE_MAX_AGE_NS 70000000ULL
#define MATTE_HEADER 6

struct lenslink_matte {
	uint32_t w, h; /* every slot shares the current matte size */
	uint8_t *slot_data[MATTE_SLOTS];
	uint64_t slot_pts[MATTE_SLOTS];
	bool slot_valid[MATTE_SLOTS];
	size_t next;

	z_stream zs;
	bool zs_ready;

	/* Output planes and the upsampling tables, rebuilt on size change. */
	uint32_t out_w, out_h;
	uint8_t *alpha;
	uint8_t *u, *v;
	uint8_t *expanded; /* every matte row widened to out_w */
	uint32_t table_w, table_h; /* matte size expanded/col0 were built for */
	uint32_t *col0;
	uint8_t *colw;

	uint64_t applied, missed, rejected;
	bool logged_first;
};

struct lenslink_matte *matte_create(void)
{
	struct lenslink_matte *m = calloc(1, sizeof(*m));
	if (!m)
		return NULL;
	/* Raw deflate (no zlib header): what Apple's COMPRESSION_ZLIB
	 * writes. */
	m->zs_ready = inflateInit2(&m->zs, -15) == Z_OK;
	return m;
}

static void free_slots(struct lenslink_matte *m)
{
	for (size_t i = 0; i < MATTE_SLOTS; i++) {
		free(m->slot_data[i]);
		m->slot_data[i] = NULL;
		m->slot_valid[i] = false;
	}
	m->w = m->h = 0;
}

static void free_output(struct lenslink_matte *m)
{
	free(m->alpha);
	free(m->u);
	free(m->v);
	free(m->expanded);
	free(m->col0);
	free(m->colw);
	m->alpha = m->u = m->v = m->expanded = m->colw = NULL;
	m->col0 = NULL;
	m->out_w = m->out_h = 0;
	m->table_w = m->table_h = 0;
}

void matte_destroy(struct lenslink_matte *m)
{
	if (!m)
		return;
	if (m->applied || m->missed || m->rejected)
		blog(LOG_INFO,
		     "[lenslink] matte: %llu frames cut out, %llu without a "
		     "matte, %llu bad packets",
		     (unsigned long long)m->applied,
		     (unsigned long long)m->missed,
		     (unsigned long long)m->rejected);
	if (m->zs_ready)
		inflateEnd(&m->zs);
	free_slots(m);
	free_output(m);
	free(m);
}

void matte_push(struct lenslink_matte *m, uint64_t pts,
		const uint8_t *payload, size_t len)
{
	if (!m || len < MATTE_HEADER)
		return;
	uint32_t w = ((uint32_t)payload[0] << 8) | payload[1];
	uint32_t h = ((uint32_t)payload[2] << 8) | payload[3];
	uint8_t encoding = payload[4];
	if (!w || !h || w > MATTE_MAX_SIDE || h > MATTE_MAX_SIDE) {
		m->rejected++;
		return;
	}
	if (w != m->w || h != m->h) {
		free_slots(m);
		for (size_t i = 0; i < MATTE_SLOTS; i++) {
			m->slot_data[i] = malloc((size_t)w * h);
			if (!m->slot_data[i]) {
				free_slots(m);
				return;
			}
		}
		m->w = w;
		m->h = h;
	}

	size_t slot = m->next;
	m->next = (m->next + 1) % MATTE_SLOTS;
	m->slot_valid[slot] = false;
	const uint8_t *body = payload + MATTE_HEADER;
	size_t body_len = len - MATTE_HEADER;
	size_t size = (size_t)w * h;

	if (encoding == 0) {
		if (body_len != size) {
			m->rejected++;
			return;
		}
		memcpy(m->slot_data[slot], body, size);
	} else if (encoding == 1 && m->zs_ready) {
		inflateReset(&m->zs);
		m->zs.next_in = (Bytef *)body;
		m->zs.avail_in = (uInt)body_len;
		m->zs.next_out = m->slot_data[slot];
		m->zs.avail_out = (uInt)size;
		if (inflate(&m->zs, Z_FINISH) != Z_STREAM_END ||
		    m->zs.total_out != size) {
			m->rejected++;
			return;
		}
	} else {
		m->rejected++;
		return;
	}
	m->slot_pts[slot] = pts;
	m->slot_valid[slot] = true;
}

/* The slot for `pts`: exact match first, else the newest older one within
 * MATTE_MAX_AGE_NS. -1 when none fits. */
static int find_slot(const struct lenslink_matte *m, uint64_t pts)
{
	int best = -1;
	for (int i = 0; i < MATTE_SLOTS; i++) {
		if (!m->slot_valid[i] || m->slot_pts[i] > pts)
			continue;
		if (m->slot_pts[i] == pts)
			return i;
		if (pts - m->slot_pts[i] <= MATTE_MAX_AGE_NS &&
		    (best < 0 || m->slot_pts[i] > m->slot_pts[best]))
			best = i;
	}
	return best;
}

static bool ensure_output(struct lenslink_matte *m, uint32_t w, uint32_t h)
{
	if (m->out_w == w && m->out_h == h && m->alpha)
		return true;
	free_output(m);
	size_t cw = (w + 1) / 2, ch = (h + 1) / 2;
	m->alpha = malloc((size_t)w * h);
	m->u = malloc(cw * ch);
	m->v = malloc(cw * ch);
	m->col0 = malloc(w * sizeof(*m->col0));
	m->colw = malloc(w);
	if (!m->alpha || !m->u || !m->v || !m->col0 || !m->colw) {
		free_output(m);
		return false;
	}
	m->out_w = w;
	m->out_h = h;
	return true;
}

/* Pixel-centre-aligned linear position of output index i on a source of
 * n samples: integer index and 8-bit weight toward the next sample. */
static void src_pos(uint32_t i, uint32_t out_n, uint32_t n, uint32_t *i0,
		    uint8_t *w)
{
	int64_t fixed = (((int64_t)i * 2 + 1) * n * 256) / ((int64_t)out_n * 2) -
			128;
	if (fixed < 0)
		fixed = 0;
	int64_t max = ((int64_t)n - 1) * 256;
	if (fixed > max)
		fixed = max;
	*i0 = (uint32_t)(fixed >> 8);
	*w = (uint8_t)(fixed & 0xFF);
}

/* Bilinear upsample of slot `slot` to the output size: each matte row is
 * widened once, then output rows blend two widened rows. Both loops are
 * plain byte arithmetic the compiler vectorizes. */
static bool upsample(struct lenslink_matte *m, int slot)
{
	uint32_t ow = m->out_w, oh = m->out_h, mw = m->w, mh = m->h;
	size_t expanded_size = (size_t)ow * mh;
	if (!m->expanded || m->table_w != mw || m->table_h != mh) {
		free(m->expanded);
		m->expanded = malloc(expanded_size);
		if (!m->expanded)
			return false;
		for (uint32_t x = 0; x < ow; x++)
			src_pos(x, ow, mw, &m->col0[x], &m->colw[x]);
		m->table_w = mw;
		m->table_h = mh;
	}
	const uint8_t *src = m->slot_data[slot];
	for (uint32_t y = 0; y < mh; y++) {
		const uint8_t *row = src + (size_t)y * mw;
		uint8_t *out = m->expanded + (size_t)y * ow;
		for (uint32_t x = 0; x < ow; x++) {
			uint32_t x0 = m->col0[x];
			uint32_t x1 = x0 + 1 < mw ? x0 + 1 : x0;
			uint32_t wx = m->colw[x];
			out[x] = (uint8_t)((row[x0] * (256 - wx) +
					    row[x1] * wx + 128) >>
					   8);
		}
	}
	for (uint32_t y = 0; y < oh; y++) {
		uint32_t y0;
		uint8_t wy8;
		src_pos(y, oh, mh, &y0, &wy8);
		uint32_t y1 = y0 + 1 < mh ? y0 + 1 : y0;
		uint32_t wy = wy8;
		const uint8_t *r0 = m->expanded + (size_t)y0 * ow;
		const uint8_t *r1 = m->expanded + (size_t)y1 * ow;
		uint8_t *out = m->alpha + (size_t)y * ow;
		for (uint32_t x = 0; x < ow; x++)
			out[x] = (uint8_t)((r0[x] * (256 - wy) + r1[x] * wy +
					    128) >>
					   8);
	}
	return true;
}

bool matte_apply(struct lenslink_matte *m, struct obs_source_frame *frame)
{
	if (!m || (frame->format != VIDEO_FORMAT_NV12 &&
		   frame->format != VIDEO_FORMAT_I420))
		return false;
	int slot = find_slot(m, frame->timestamp);
	if (slot < 0) {
		m->missed++;
		return false;
	}
	if (!ensure_output(m, frame->width, frame->height) ||
	    !upsample(m, slot)) {
		m->missed++;
		return false;
	}

	if (frame->format == VIDEO_FORMAT_NV12) {
		uint32_t cw = (frame->width + 1) / 2;
		uint32_t ch = (frame->height + 1) / 2;
		for (uint32_t y = 0; y < ch; y++) {
			const uint8_t *uv =
				frame->data[1] + (size_t)y * frame->linesize[1];
			uint8_t *u = m->u + (size_t)y * cw;
			uint8_t *v = m->v + (size_t)y * cw;
			for (uint32_t x = 0; x < cw; x++) {
				u[x] = uv[2 * x];
				v[x] = uv[2 * x + 1];
			}
		}
		frame->data[1] = m->u;
		frame->data[2] = m->v;
		frame->linesize[1] = cw;
		frame->linesize[2] = cw;
	}
	frame->data[3] = m->alpha;
	frame->linesize[3] = frame->width;
	frame->format = VIDEO_FORMAT_I40A;

	m->applied++;
	if (!m->logged_first) {
		m->logged_first = true;
		blog(LOG_INFO,
		     "[lenslink] matte: cutting out %ux%u frames with a "
		     "%ux%u matte",
		     frame->width, frame->height, m->w, m->h);
	}
	return true;
}
