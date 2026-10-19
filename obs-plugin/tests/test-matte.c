#include "test.h"
#include "matte.h"

#include <stdlib.h>
#include <zlib.h>

/* A MATTE payload: 6-byte header, then the alpha, raw or raw deflate. */
static size_t make_payload(uint8_t *out, size_t cap, uint16_t w, uint16_t h,
			   const uint8_t *alpha, bool deflated)
{
	size_t size = (size_t)w * h;
	out[0] = (uint8_t)(w >> 8);
	out[1] = (uint8_t)w;
	out[2] = (uint8_t)(h >> 8);
	out[3] = (uint8_t)h;
	out[4] = deflated ? 1 : 0;
	out[5] = 0;
	if (!deflated) {
		memcpy(out + 6, alpha, size);
		return 6 + size;
	}
	z_stream zs = {0};
	deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8,
		     Z_DEFAULT_STRATEGY);
	zs.next_in = (Bytef *)alpha;
	zs.avail_in = (uInt)size;
	zs.next_out = out + 6;
	zs.avail_out = (uInt)(cap - 6);
	deflate(&zs, Z_FINISH);
	size_t n = zs.total_out;
	deflateEnd(&zs);
	return 6 + n;
}

/* 8x4 NV12 frame: Y = 100, U = 10..13, V = 20..23 per chroma column. */
static uint8_t y_plane[32], uv_plane[16];
static struct obs_source_frame nv12_frame(uint64_t pts)
{
	memset(y_plane, 100, sizeof(y_plane));
	for (int r = 0; r < 2; r++)
		for (int c = 0; c < 4; c++) {
			uv_plane[r * 8 + 2 * c] = (uint8_t)(10 + c);
			uv_plane[r * 8 + 2 * c + 1] = (uint8_t)(20 + c);
		}
	struct obs_source_frame f = {.data = {y_plane, uv_plane},
				     .linesize = {8, 8},
				     .width = 8,
				     .height = 4,
				     .timestamp = pts,
				     .format = VIDEO_FORMAT_NV12};
	return f;
}

int main(void)
{
	uint8_t payload[256];
	struct lenslink_matte *m = matte_create();
	CHECK(m != NULL);

	/* 2x1 matte, left kept and right dropped, deflated, for pts 1000. */
	const uint8_t half[2] = {255, 0};
	size_t len = make_payload(payload, sizeof(payload), 2, 1, half, true);
	matte_push(m, 1000, payload, len);

	struct obs_source_frame f = nv12_frame(1000);
	CHECK(matte_apply(m, &f));
	CHECK(f.format == VIDEO_FORMAT_I40A);
	/* NV12 chroma split into planar U and V, Y untouched. */
	CHECK(f.data[0] == y_plane);
	CHECK(f.linesize[1] == 4 && f.linesize[2] == 4);
	CHECK(f.data[1][3] == 13 && f.data[2][3] == 23);
	CHECK(f.data[1][4] == 10 && f.data[2][7] == 23);
	/* Upsampled 4x: outer columns keep the matte's values, the middle
	 * ramps between them, and every row matches. */
	const uint8_t *a = f.data[3];
	CHECK(f.linesize[3] == 8);
	CHECK(a[0] == 255 && a[1] == 255 && a[6] == 0 && a[7] == 0);
	CHECK(a[3] > a[4] && a[3] < 255 && a[4] > 0);
	CHECK(memcmp(a, a + 24, 8) == 0);

	/* A frame whose matte was lost borrows the newest older one within
	 * two frames, and goes opaque beyond that. */
	const uint8_t full[2] = {255, 255};
	len = make_payload(payload, sizeof(payload), 2, 1, full, false);
	matte_push(m, 2000, payload, len);
	f = nv12_frame(2000 + 33000000);
	CHECK(matte_apply(m, &f));
	CHECK(f.data[3][7] == 255);
	f = nv12_frame(2000 + 100000000);
	CHECK(!matte_apply(m, &f));
	CHECK(f.format == VIDEO_FORMAT_NV12);

	/* A matte from the future never applies to an earlier frame. */
	f = nv12_frame(999);
	CHECK(!matte_apply(m, &f));

	/* Malformed payloads are dropped: short, oversize, truncated raw. */
	matte_push(m, 3000, payload, 3);
	payload[0] = 0xFF;
	matte_push(m, 3000, payload, len);
	len = make_payload(payload, sizeof(payload), 2, 1, half, false);
	matte_push(m, 3000, payload, len - 1);
	f = nv12_frame(3000);
	CHECK(matte_apply(m, &f));
	CHECK(f.data[3][7] == 255); /* still the pts-2000 matte */

	matte_destroy(m);
	TEST_MAIN_END();
}
