

// Helper macro for assertions with messages
#define ASSERT_MSG(condition, message)                                                 \
    do {                                                                               \
        if (!(condition)) {                                                            \
            printf("[FAIL] condition: %s, message: %s\n", #condition, message);        \
            assert(condition);                                                         \
        }                                                                              \
    } while (0)
