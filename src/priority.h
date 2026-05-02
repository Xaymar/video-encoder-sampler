#pragma once

// Makes ves yield to everything else on the machine: the lowest CPU priority and the lowest disk priority (Windows: background processing mode;
// Linux: nice 19 and the idle I/O class). A sweep runs for hours and should not slow down what the person at the computer is doing. The GPU is not
// affected. Returns false if the system refused.
bool set_low_priority();
