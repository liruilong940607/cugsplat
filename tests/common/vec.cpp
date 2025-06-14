#include <cassert>
#include <stdio.h>

#include "../helpers.h"
#include "tinyrend/common/vec.h"

using namespace tinyrend;

int test_vec() {
    int fails = 0;

    // Initialize from values and pointer
    {
        float data[3] = {1.2f, 2.0f, 3.0f};
        fvec3 v1 = fvec3::from_ptr(data);
        fvec3 v2 = fvec3(1.2f, 2.0f, 3.0f);
        fails += CHECK(v1 == v2, "");
    }

    // Sum
    {
        fvec3 v1 = fvec3(1.2f, 2.0f, 3.0f);
        CHECK(v1.sum() == 6.2f, "");
    }

    // Vector-Vector operations
    {
        fvec3 v1 = fvec3(1.2f, 2.0f, 3.0f);
        fvec3 v2 = fvec3(4.0f, 5.0f, 6.0f);
        CHECK((v1 + v2).is_close(fvec3(5.2f, 7.0f, 9.0f)), "");
        CHECK((v1 - v2).is_close(fvec3(-2.8f, -3.0f, -3.0f)), "");
        CHECK((v1 * v2).is_close(fvec3(4.8f, 10.0f, 18.0f)), "");
        CHECK((v1 / v2).is_close(fvec3(0.3f, 0.4f, 0.5f)), "");
    }

    // Vector-Scalar operations
    {
        fvec3 v1 = fvec3(1.2f, 2.0f, 3.0f);
        CHECK((v1 + 1.0f).is_close(fvec3(2.2f, 3.0f, 4.0f)), "");
        CHECK((v1 - 1.0f).is_close(fvec3(0.2f, 1.0f, 2.0f)), "");
        CHECK((v1 * 2.0f).is_close(fvec3(2.4f, 4.0f, 6.0f)), "");
        CHECK((v1 / 2.0f).is_close(fvec3(0.6f, 1.0f, 1.5f)), "");
    }

    // Scalar-Vector operations
    {
        fvec3 v1 = fvec3(1.2f, 2.0f, 3.0f);
        CHECK((1.0f + v1).is_close(fvec3(2.2f, 3.0f, 4.0f)), "");
        CHECK((1.0f - v1).is_close(-fvec3(0.2f, 1.0f, 2.0f)), "");
        CHECK((1.0f * v1).is_close(fvec3(1.2f, 2.0f, 3.0f)), "");
    }

    // Compound assignment operators
    {
        fvec3 v1 = fvec3(1.2f, 2.0f, 3.0f);
        v1 += fvec3(4.0f, 5.0f, 6.0f);
        CHECK(v1.is_close(fvec3(5.2f, 7.0f, 9.0f)), "");
        v1 -= fvec3(4.0f, 5.0f, 6.0f);
        CHECK(v1.is_close(fvec3(1.2f, 2.0f, 3.0f)), "");
        v1 *= fvec3(4.0f, 5.0f, 6.0f);
        CHECK(v1.is_close(fvec3(4.8f, 10.0f, 18.0f)), "");
        v1 /= fvec3(4.0f, 5.0f, 6.0f);
        CHECK(v1.is_close(fvec3(1.2f, 2.0f, 3.0f)), "");
        v1 += 1.0f;
        CHECK(v1.is_close(fvec3(2.2f, 3.0f, 4.0f)), "");
        v1 -= 1.0f;
        CHECK(v1.is_close(fvec3(1.2f, 2.0f, 3.0f)), "");
        v1 *= 2.0f;
        CHECK(v1.is_close(fvec3(2.4f, 4.0f, 6.0f)), "");
        v1 /= 2.0f;
    }

    // Comparison operators
    {
        fvec3 v1 = fvec3(1.2f, 2.0f, 3.0f);
        fvec3 v2 = fvec3(1.2f, 2.0f, 3.0f);
        CHECK(v1 == v2, "");
        CHECK(v1 != fvec3(4.0f, 5.0f, 6.0f), "");
    }

    // To string
    {
        fvec3 v1 = fvec3(1.2f, 2.0f, 3.0f);
        CHECK(v1.to_string() == "vec3(1.2, 2, 3)", "");
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_vec();

    if (fails > 0) {
        printf("[common/vec.cpp] %d tests failed!\n", fails);
    } else {
        printf("[common/vec.cpp] All tests passed!\n");
    }

    return fails;
}