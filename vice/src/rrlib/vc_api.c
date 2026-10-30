/*
 * vc_api.c - VICE as a library, one frame at a time (RetroRecomp).
 *
 * Linked with the objects of the headless hybrid build (CPPFLAGS=-DRR_HOOK)
 * into vice-core.dll, minus main.o's executable entry. The host program
 * (RetroRecomp's engine, in Rust) loads the DLL and drives it:
 *
 *   vc_init(argc, argv)  start VICE with these arguments on a thread of its
 *                        own and return when the first frame has ended;
 *   vc_run_frame()       let the emulation run to the end of the next frame;
 *   vc_frame(...)        the last frame as palette indices, as the binary
 *                        monitor's DISPLAY_GET gives it;
 *   vc_joystick(...)     a joystick port's bits;
 *   vc_key(...)          a key of the keyboard matrix;
 *   vc_request_*(...)    reset, save or load a snapshot, warp: queued, and
 *                        done on the emulation thread through a CPU trap at
 *                        the next instruction boundary, as VICE's own UI
 *                        does them. Safe to call from any thread, while the
 *                        machine runs or while it is parked;
 *   vc_setenv(...)       set a variable in the C runtime's environment,
 *                        which getenv reads (a host's SetEnvironmentVariable
 *                        after startup does not reach it). VICE's hook code
 *                        reads RR_BLOCKS_DLL and RR_INPUT_* this way.
 *
 * The emulation thread is parked inside vsync_do_vsync (the rr_vsync_gate
 * hook) whenever the host holds it, so the host's calls between frames see
 * a machine that is not running. Only plain data and C functions cross the
 * boundary, so the MSVC-built host and this MinGW-built library keep their
 * own C runtimes.
 *
 * Plan and status: RetroRecomp's docs/engine-host.md.
 *
 * This file is part of VICE and, like it, GPL-2.0-or-later.
 */

#include "vice.h"

#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "interrupt.h"
#include "joyport/joystick.h"
#include "keyboard.h"
#include "lib.h"
#include "machine-video.h"
#include "machine.h"
#include "main.h"
#include "screenshot.h"
#include "vsync.h"

#define VC_EXPORT __declspec(dllexport)

extern void (*rr_vsync_gate)(void);

static sem_t vc_frame_done;
static sem_t vc_go;
static volatile long vc_frames = 0;
static volatile int vc_failed = 0;
static int vc_argc = 0;
static char **vc_argv = NULL;

/* Requests from the host or the port's menu, done on the emulation thread. */
enum vc_request_kind { VC_NONE, VC_RESET, VC_POWER_CYCLE, VC_SAVE, VC_LOAD, VC_WARP_ON, VC_WARP_OFF };

static pthread_mutex_t vc_lock = PTHREAD_MUTEX_INITIALIZER;
static enum vc_request_kind vc_pending = VC_NONE;
static char vc_pending_path[1024];
static volatile int vc_last_result = 0;

static void vc_snapshot_trap(uint16_t addr, void *data)
{
    enum vc_request_kind kind = (enum vc_request_kind)(intptr_t)data;
    char path[sizeof vc_pending_path];

    pthread_mutex_lock(&vc_lock);
    memcpy(path, vc_pending_path, sizeof path);
    pthread_mutex_unlock(&vc_lock);
    if (kind == VC_SAVE) {
        vc_last_result = machine_write_snapshot(path, 1, 0, 0);
    } else {
        vc_last_result = machine_read_snapshot(path, 0);
    }
}

/* Take a queued request: snapshots become a CPU trap (run at the next
   instruction boundary, registers in place), the rest happen here. */
static void vc_service(void)
{
    enum vc_request_kind kind;

    pthread_mutex_lock(&vc_lock);
    kind = vc_pending;
    vc_pending = VC_NONE;
    pthread_mutex_unlock(&vc_lock);
    switch (kind) {
        case VC_RESET:
            machine_trigger_reset(MACHINE_RESET_MODE_RESET_CPU);
            break;
        case VC_POWER_CYCLE:
            machine_trigger_reset(MACHINE_RESET_MODE_POWER_CYCLE);
            break;
        case VC_SAVE:
        case VC_LOAD:
            interrupt_maincpu_trigger_trap(vc_snapshot_trap, (void *)(intptr_t)kind);
            break;
        case VC_WARP_ON:
            vsync_set_warp_mode(1);
            break;
        case VC_WARP_OFF:
            vsync_set_warp_mode(0);
            break;
        case VC_NONE:
            break;
    }
}

static int vc_request(enum vc_request_kind kind, const char *path)
{
    pthread_mutex_lock(&vc_lock);
    if (vc_pending != VC_NONE) {
        pthread_mutex_unlock(&vc_lock);
        return -1; /* one at a time */
    }
    vc_pending = kind;
    if (path != NULL) {
        strncpy(vc_pending_path, path, sizeof vc_pending_path - 1);
        vc_pending_path[sizeof vc_pending_path - 1] = 0;
    }
    pthread_mutex_unlock(&vc_lock);
    return 0;
}

/* On the emulation thread, at the end of every frame. */
static void vc_gate(void)
{
    vc_service();
    vc_frames++;
    sem_post(&vc_frame_done);
    sem_wait(&vc_go);
}

static void *vc_emulation_thread(void *unused)
{
    /* Runs the machine until the process ends; returns only on a failed
       start, which the host must hear about. */
    main_program(vc_argc, vc_argv);
    vc_failed = 1;
    sem_post(&vc_frame_done);
    return NULL;
}

VC_EXPORT int vc_init(int argc, const char **argv)
{
    pthread_t thread;
    pthread_attr_t attr;
    int i;

    if (vc_argv != NULL) {
        return -1; /* once per process */
    }
    vc_argc = argc;
    vc_argv = malloc(sizeof(char *) * (size_t)(argc + 1));
    for (i = 0; i < argc; i++) {
        vc_argv[i] = strdup(argv[i]);
    }
    vc_argv[argc] = NULL;

    sem_init(&vc_frame_done, 0, 0);
    sem_init(&vc_go, 0, 0);
    rr_vsync_gate = vc_gate;

    pthread_attr_init(&attr);
    /* As VICE's own thread: wic64 and friends use large stack buffers. */
    pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);
    if (pthread_create(&thread, &attr, vc_emulation_thread, NULL) != 0) {
        return -2;
    }
    sem_wait(&vc_frame_done);
    return vc_failed ? -3 : 0;
}

/* Run to the end of the next frame. Returns the number of frames ended so
   far, or -1 if the machine has stopped. */
VC_EXPORT long vc_run_frame(void)
{
    if (vc_failed) {
        return -1;
    }
    sem_post(&vc_go);
    sem_wait(&vc_frame_done);
    return vc_failed ? -1 : vc_frames;
}

/* The last frame, one palette index per pixel, in `buf` (capacity `cap`
   bytes), with the same geometry DISPLAY_GET reports: `info` receives
   debug_width, debug_height, debug_offset_x, debug_offset_y, inner_width
   and inner_height. Returns the number of bytes written, or -1. */
VC_EXPORT long vc_frame(uint8_t *buf, long cap, uint16_t *info)
{
    screenshot_t s;
    struct video_canvas_s *canvas = machine_video_canvas_get(0);
    unsigned int line, i, excess_w, excess_h, off_x, off_y;
    long need;

    if (canvas == NULL || machine_screenshot(&s, canvas) < 0) {
        return -1;
    }
    s.width = s.max_width & ~3;
    s.height = s.last_displayed_line - s.first_displayed_line + 1;
    s.y_offset = s.first_displayed_line;
    need = (long)s.debug_width * (long)s.debug_height;
    info[0] = (uint16_t)s.debug_width;
    info[1] = (uint16_t)s.debug_height;
    info[2] = (uint16_t)s.debug_offset_x;
    info[3] = (uint16_t)s.debug_offset_y;
    info[4] = (uint16_t)s.inner_width;
    info[5] = (uint16_t)s.inner_height;
    if (buf == NULL || cap < need) {
        return need;
    }

    /* As monitor_binary_screenshot_line_data. */
    excess_w = (s.width - s.inner_width) / 2;
    excess_h = (s.height - s.inner_height) / 2;
    off_x = s.debug_offset_x - excess_w;
    off_y = s.debug_offset_y - excess_h;
    if (off_x + s.width > s.debug_width) {
        off_x = 0;
    }
    if (off_y + s.height > s.debug_height) {
        off_y = 0;
    }
    memset(buf, 0, (size_t)need);
    for (line = 0; line < s.debug_height; line++) {
        uint8_t *data = buf + (long)line * s.debug_width;
        uint8_t *base;
        if (line < off_y || line > off_y + s.height) {
            continue;
        }
        base = s.draw_buffer + ((line - off_y) + s.y_offset) * s.size_height * s.draw_buffer_line_size;
        for (i = 0; i < s.width && i + off_x < s.debug_width; i++) {
            data[i + off_x] = base[i * s.size_width + s.x_offset];
        }
    }
    return need;
}

/* A joystick port's bits, 1 = pressed: up, down, left, right, fire. */
VC_EXPORT void vc_joystick(unsigned int port, uint16_t bits)
{
    joystick_set_value_absolute(port, bits);
}

/* A key of the matrix, as VICE's keyarr[row] bit col. */
VC_EXPORT void vc_key(int row, int col, int down)
{
    keyboard_set_keyarr(row, col, down);
}

/* Set an environment variable as the C runtime sees it. */
VC_EXPORT int vc_setenv(const char *name, const char *value)
{
    return _putenv_s(name, value);
}

/* Reset: 0 the CPU (the reset button), 1 a power cycle. */
VC_EXPORT int vc_request_reset(int power_cycle)
{
    return vc_request(power_cycle ? VC_POWER_CYCLE : VC_RESET, NULL);
}

/* Save a snapshot (with ROMs, without disks) at the next boundary. */
VC_EXPORT int vc_request_save(const char *path)
{
    return vc_request(VC_SAVE, path);
}

/* Load a snapshot at the next boundary. */
VC_EXPORT int vc_request_load(const char *path)
{
    return vc_request(VC_LOAD, path);
}

/* Warp: run as fast as the host can (VICE's own pacing off). */
VC_EXPORT int vc_request_warp(int on)
{
    return vc_request(on ? VC_WARP_ON : VC_WARP_OFF, NULL);
}

/* The result of the last snapshot save or load: 0 on success. */
VC_EXPORT int vc_last_snapshot_result(void)
{
    return vc_last_result;
}
