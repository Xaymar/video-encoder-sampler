#include "priority.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>

bool set_low_priority() {
	// Background mode lowers the CPU priority to idle and the I/O priority to very low (and the memory priority) for the whole process.
	if (SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_BEGIN)) return true;
	return SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS) != 0; // no I/O part, but the CPU part
}
#else
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

bool set_low_priority() {
	const bool cpu = setpriority(PRIO_PROCESS, 0, 19) == 0;
	constexpr int IOPRIO_WHO_PROCESS = 1, IOPRIO_CLASS_IDLE = 3, IOPRIO_CLASS_SHIFT = 13;
	const bool io = syscall(SYS_ioprio_set, IOPRIO_WHO_PROCESS, 0, IOPRIO_CLASS_IDLE << IOPRIO_CLASS_SHIFT) == 0;
	return cpu && io;
}
#endif
