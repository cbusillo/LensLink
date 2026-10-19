#pragma once

#include <obs-module.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * Alpha matte (this fork): the phone's green-screen keep-mask for each
 * frame arrives as a MATTE packet (docs/PROTOCOL.md §13) ahead of the
 * VIDEO frame with the same pts. The standard decode path pairs them and
 * hands OBS an I40A frame (I420 + alpha), so the source arrives already
 * cut out. Dial-loop thread only: packets and decoded frames both arrive
 * there, so there is no locking.
 */
struct lenslink_matte;

struct lenslink_matte *matte_create(void);
void matte_destroy(struct lenslink_matte *m);

/* Stores one MATTE payload for `pts`; malformed payloads are dropped. */
void matte_push(struct lenslink_matte *m, uint64_t pts,
		const uint8_t *payload, size_t len);

/*
 * Attaches the matte stored for frame->timestamp (or a slightly older
 * one when a matte was lost) by rewriting `frame` as I40A over planes
 * the matte owns until the next call. NV12 and I420 frames only. Returns
 * false and leaves `frame` untouched when no matte fits: the frame then
 * shows opaque, never blank.
 */
bool matte_apply(struct lenslink_matte *m, struct obs_source_frame *frame);
