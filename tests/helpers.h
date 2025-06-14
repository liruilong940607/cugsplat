#pragma once

// Helper macro: Returns 1 if the condition is false, 0 otherwise
#define CHECK(condition, message)                                                      \
    ((condition)                                                                       \
         ? 0                                                                           \
         : (printf("[FAIL] condition: %s, message: %s\n", #condition, message), 1))
