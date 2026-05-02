#include "priority.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>

bool set_low_priority() {
	// Not PROCESS_MODE_BACKGROUND_BEGIN: besides the CPU and disk priorities it lowers the memory priority, and with that ves made no progress at all
	// on a busy machine. The two priorities asked for are set on their own. The disk one has no documented call for a whole process, so this is
	// NtSetInformationProcess(ProcessIoPriority), which Windows has had since Vista; if it is missing only the CPU part applies.
	const bool cpu = SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS) != 0;
	using SetInfo = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG);
	const SetInfo set_info = (SetInfo)GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtSetInformationProcess");
	ULONG very_low = 0;
	const bool io = set_info && set_info(GetCurrentProcess(), 33 /* ProcessIoPriority */, &very_low, sizeof very_low) == 0;
	return cpu && io;
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
