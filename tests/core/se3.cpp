#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

#include "../helpers.h"
#include "core/se3.h"

using namespace cugsplat::se3;

int test_interpolate() {
    int fails = 0;

    // Test case 1: Quaternion interpolation
    {
        auto const rot1 = glm::fquat(1.0f, 0.0f, 0.0f, 0.0f); // identity
        auto const transl1 = glm::fvec3(0.0f, 0.0f, 0.0f);
        auto const rot2 = glm::fquat(0.0f, 1.0f, 0.0f, 0.0f); // 180° around x
        auto const transl2 = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const ratio = 0.5f;

        auto const [rot, transl] = interpolate(ratio, rot1, transl1, rot2, transl2);
        auto const expected_rot =
            glm::fquat(0.707106781f, 0.707106781f, 0.0f, 0.0f); // 90° around x
        auto const expected_transl = glm::fvec3(0.5f, 0.5f, 0.5f);

        if (!is_close(rot, expected_rot) || !is_close(transl, expected_transl)) {
            printf("\n=== Testing interpolate (quaternion) ===\n");
            printf("\n[FAIL] Test 1: Quaternion interpolation\n");
            printf("  Rot: %s\n", glm::to_string(rot).c_str());
            printf("  Expected rot: %s\n", glm::to_string(expected_rot).c_str());
            printf("  Transl: %s\n", glm::to_string(transl).c_str());
            printf("  Expected transl: %s\n", glm::to_string(expected_transl).c_str());
            fails += 1;
        }
    }

    // Test case 2: Matrix interpolation
    {
        // Identity matrix (column-major)
        auto const rot1 = glm::fmat3(
            1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            1.0f
        ); // third column
        auto const transl1 = glm::fvec3(0.0f, 0.0f, 0.0f);
        // 180° rotation around y (column-major)
        auto const rot2 = glm::fmat3(
            -1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            -1.0f
        ); // third column
        auto const transl2 = glm::fvec3(2.0f, 2.0f, 2.0f);
        auto const ratio = 0.5f;

        auto const [rot, transl] = interpolate(ratio, rot1, transl1, rot2, transl2);
        // 90° rotation around y (column-major)
        auto const expected_rot = glm::fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f
        ); // third column
        auto const expected_transl = glm::fvec3(1.0f, 1.0f, 1.0f);

        if (!is_close(rot, expected_rot) || !is_close(transl, expected_transl)) {
            printf("\n[FAIL] Test 2: Matrix interpolation\n");
            printf("  Rot: %s\n", glm::to_string(rot).c_str());
            printf("  Expected rot: %s\n", glm::to_string(expected_rot).c_str());
            printf("  Transl: %s\n", glm::to_string(transl).c_str());
            printf("  Expected transl: %s\n", glm::to_string(expected_transl).c_str());
            fails += 1;
        }
    }

    return fails;
}

int test_transform_point() {
    int fails = 0;

    // Test case 1: Quaternion transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = glm::fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const transl = glm::fvec3(1.0f, 2.0f, 3.0f);
        auto const point = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const transformed = transform_point(rot, transl, point);
        // For quaternion rotation: q * p * q^-1 + t
        // 90° around y: (x,y,z) -> (z,y,-x)
        // (1,1,1) -> (1,1,-1) + (1,2,3) = (2,3,2)
        auto const expected = glm::fvec3(2.0f, 3.0f, 2.0f);

        if (!is_close(transformed, expected)) {
            printf("\n=== Testing transform_point (quaternion) ===\n");
            printf("\n[FAIL] Test 1: Quaternion transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    // Test case 2: Matrix transform
    {
        // 90° rotation around y (column-major)
        auto const rot = glm::fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f
        ); // third column
        auto const transl = glm::fvec3(1.0f, 2.0f, 3.0f);
        auto const point = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const transformed = transform_point(rot, transl, point);
        // For column-major matrix multiplication: R * p + t
        // [0 0 1] [1]   [1]   [2]
        // [0 1 0] [1] + [2] = [3]
        // [-1 0 0][1]   [3]   [2]
        auto const expected = glm::fvec3(2.0f, 3.0f, 2.0f);

        if (!is_close(transformed, expected)) {
            printf("\n[FAIL] Test 2: Matrix transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    return fails;
}

int test_invtransform_point() {
    int fails = 0;

    // Test case 1: Quaternion inverse transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = glm::fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const transl = glm::fvec3(1.0f, 2.0f, 3.0f);
        auto const point = glm::fvec3(2.0f, 3.0f, 2.0f);
        auto const transformed = invtransform_point(rot, transl, point);
        // For inverse quaternion rotation: q^-1 * (p - t) * q
        // 90° around y inverse: (x,y,z) -> (-z,y,x)
        // (1,1,-1) -> (1,1,1)
        auto const expected = glm::fvec3(1.0f, 1.0f, 1.0f);

        if (!is_close(transformed, expected)) {
            printf("\n=== Testing invtransform_point (quaternion) ===\n");
            printf("\n[FAIL] Test 1: Quaternion inverse transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    // Test case 2: Matrix inverse transform
    {
        // 90° rotation around y (column-major)
        auto const rot = glm::fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f
        ); // third column
        auto const transl = glm::fvec3(1.0f, 2.0f, 3.0f);
        auto const point = glm::fvec3(4.0f, 3.0f, 0.0f);
        auto const transformed = invtransform_point(rot, transl, point);
        // For inverse transform: R^T * (p - t)
        // [0 0 -1] [3]   [3]
        // [0 1 0]  [1] = [1]
        // [1 0 0]  [-3]  [3]
        auto const expected = glm::fvec3(3.0f, 1.0f, 3.0f);

        if (!is_close(transformed, expected)) {
            printf("\n[FAIL] Test 2: Matrix inverse transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    return fails;
}

int test_transform_dir() {
    int fails = 0;

    // Test case 1: Quaternion transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = glm::fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const dir = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const transformed = transform_dir(rot, dir);
        // For quaternion rotation: q * d * q^-1
        // 90° around y: (x,y,z) -> (z,y,-x)
        auto const expected = glm::fvec3(1.0f, 1.0f, -1.0f);

        if (!is_close(transformed, expected)) {
            printf("\n=== Testing transform_dir (quaternion) ===\n");
            printf("\n[FAIL] Test 1: Quaternion transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    // Test case 2: Matrix transform
    {
        // 90° rotation around y (column-major)
        auto const rot = glm::fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f
        ); // third column
        auto const dir = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const transformed = transform_dir(rot, dir);
        auto const expected = glm::fvec3(1.0f, 1.0f, -1.0f);

        if (!is_close(transformed, expected)) {
            printf("\n[FAIL] Test 2: Matrix transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    return fails;
}

int test_invtransform_dir() {
    int fails = 0;

    // Test case 1: Quaternion inverse transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = glm::fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const dir = glm::fvec3(1.0f, 1.0f, -1.0f);
        auto const transformed = invtransform_dir(rot, dir);
        // For inverse quaternion rotation: q^-1 * d * q
        // 90° around y inverse: (x,y,z) -> (-z,y,x)
        auto const expected = glm::fvec3(1.0f, 1.0f, 1.0f);

        if (!is_close(transformed, expected)) {
            printf("\n=== Testing invtransform_dir (quaternion) ===\n");
            printf("\n[FAIL] Test 1: Quaternion inverse transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    // Test case 2: Matrix inverse transform
    {
        // 90° rotation around y (column-major)
        auto const rot = glm::fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f
        ); // third column
        auto const dir = glm::fvec3(1.0f, 1.0f, -1.0f);
        auto const transformed = invtransform_dir(rot, dir);
        auto const expected = glm::fvec3(1.0f, 1.0f, 1.0f);

        if (!is_close(transformed, expected)) {
            printf("\n[FAIL] Test 2: Matrix inverse transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    return fails;
}

int test_transform_ray() {
    int fails = 0;

    // Test case 1: Quaternion transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = glm::fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const transl = glm::fvec3(1.0f, 2.0f, 3.0f);
        auto const ray_o = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const ray_d = glm::fvec3(0.0f, 1.0f, 0.0f);
        auto const [transformed_o, transformed_d] =
            transform_ray(rot, transl, ray_o, ray_d);
        // For quaternion rotation: q * p * q^-1 + t and q * d * q^-1
        // 90° around y: (x,y,z) -> (z,y,-x)
        auto const expected_o = glm::fvec3(2.0f, 3.0f, 2.0f);
        auto const expected_d = glm::fvec3(0.0f, 1.0f, 0.0f);

        if (!is_close(transformed_o, expected_o) ||
            !is_close(transformed_d, expected_d)) {
            printf("\n=== Testing transform_ray (quaternion) ===\n");
            printf("\n[FAIL] Test 1: Quaternion transform\n");
            printf("  Transformed origin: %s\n", glm::to_string(transformed_o).c_str());
            printf("  Expected origin: %s\n", glm::to_string(expected_o).c_str());
            printf(
                "  Transformed direction: %s\n", glm::to_string(transformed_d).c_str()
            );
            printf("  Expected direction: %s\n", glm::to_string(expected_d).c_str());
            fails += 1;
        }
    }

    // Test case 2: Matrix transform
    {
        // 90° rotation around y (column-major)
        auto const rot = glm::fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f
        ); // third column
        auto const transl = glm::fvec3(1.0f, 2.0f, 3.0f);
        auto const ray_o = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const ray_d = glm::fvec3(0.0f, 1.0f, 0.0f);
        auto const [transformed_o, transformed_d] =
            transform_ray(rot, transl, ray_o, ray_d);
        auto const expected_o = glm::fvec3(2.0f, 3.0f, 2.0f);
        auto const expected_d = glm::fvec3(0.0f, 1.0f, 0.0f);

        if (!is_close(transformed_o, expected_o) ||
            !is_close(transformed_d, expected_d)) {
            printf("\n[FAIL] Test 2: Matrix transform\n");
            printf("  Transformed origin: %s\n", glm::to_string(transformed_o).c_str());
            printf("  Expected origin: %s\n", glm::to_string(expected_o).c_str());
            printf(
                "  Transformed direction: %s\n", glm::to_string(transformed_d).c_str()
            );
            printf("  Expected direction: %s\n", glm::to_string(expected_d).c_str());
            fails += 1;
        }
    }

    return fails;
}

int test_invtransform_ray() {
    int fails = 0;

    // Test case 1: Quaternion inverse transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = glm::fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const transl = glm::fvec3(1.0f, 2.0f, 3.0f);
        auto const ray_o = glm::fvec3(2.0f, 3.0f, 2.0f);
        auto const ray_d = glm::fvec3(0.0f, 1.0f, 0.0f);
        auto const [transformed_o, transformed_d] =
            invtransform_ray(rot, transl, ray_o, ray_d);
        // For inverse quaternion rotation: q^-1 * (p - t) * q and q^-1 * d * q
        // 90° around y inverse: (x,y,z) -> (-z,y,x)
        auto const expected_o = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const expected_d = glm::fvec3(0.0f, 1.0f, 0.0f);

        if (!is_close(transformed_o, expected_o) ||
            !is_close(transformed_d, expected_d)) {
            printf("\n=== Testing invtransform_ray (quaternion) ===\n");
            printf("\n[FAIL] Test 1: Quaternion inverse transform\n");
            printf("  Transformed origin: %s\n", glm::to_string(transformed_o).c_str());
            printf("  Expected origin: %s\n", glm::to_string(expected_o).c_str());
            printf(
                "  Transformed direction: %s\n", glm::to_string(transformed_d).c_str()
            );
            printf("  Expected direction: %s\n", glm::to_string(expected_d).c_str());
            fails += 1;
        }
    }

    // Test case 2: Matrix inverse transform
    {
        // 90° rotation around y (column-major)
        auto const rot = glm::fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f
        ); // third column
        auto const transl = glm::fvec3(1.0f, 2.0f, 3.0f);
        auto const ray_o = glm::fvec3(4.0f, 3.0f, 0.0f);
        auto const ray_d = glm::fvec3(0.0f, 1.0f, 0.0f);
        auto const [transformed_o, transformed_d] =
            invtransform_ray(rot, transl, ray_o, ray_d);
        auto const expected_o = glm::fvec3(3.0f, 1.0f, 3.0f);
        auto const expected_d = glm::fvec3(0.0f, 1.0f, 0.0f);

        if (!is_close(transformed_o, expected_o) ||
            !is_close(transformed_d, expected_d)) {
            printf("\n[FAIL] Test 2: Matrix inverse transform\n");
            printf("  Transformed origin: %s\n", glm::to_string(transformed_o).c_str());
            printf("  Expected origin: %s\n", glm::to_string(expected_o).c_str());
            printf(
                "  Transformed direction: %s\n", glm::to_string(transformed_d).c_str()
            );
            printf("  Expected direction: %s\n", glm::to_string(expected_d).c_str());
            fails += 1;
        }
    }

    return fails;
}

int test_transform_covar() {
    int fails = 0;

    // Test case 1: Quaternion transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = glm::fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const covar = glm::fmat3(
            1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            3.0f
        ); // third column
        auto const transformed = transform_covar(rot, covar);
        // For quaternion rotation: R * covar * R^T
        // 90° around y: (x,y,z) -> (z,y,-x)
        auto const expected = glm::fmat3(
            3.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            1.0f
        ); // third column

        if (!is_close(transformed, expected)) {
            printf("\n=== Testing transform_covar (quaternion) ===\n");
            printf("\n[FAIL] Test 1: Quaternion transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    // Test case 2: Matrix transform
    {
        // 90° rotation around y (column-major)
        auto const rot = glm::fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f
        ); // third column
        auto const covar = glm::fmat3(
            1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            3.0f
        ); // third column
        auto const transformed = transform_covar(rot, covar);
        auto const expected = glm::fmat3(
            3.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            1.0f
        ); // third column

        if (!is_close(transformed, expected)) {
            printf("\n[FAIL] Test 2: Matrix transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    return fails;
}

int test_invtransform_covar() {
    int fails = 0;

    // Test case 1: Quaternion inverse transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = glm::fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const covar = glm::fmat3(
            3.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            1.0f
        ); // third column
        auto const transformed = invtransform_covar(rot, covar);
        // For inverse quaternion rotation: R^T * covar * R
        // 90° around y inverse: (x,y,z) -> (-z,y,x)
        auto const expected = glm::fmat3(
            1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            3.0f
        ); // third column

        if (!is_close(transformed, expected)) {
            printf("\n=== Testing invtransform_covar (quaternion) ===\n");
            printf("\n[FAIL] Test 1: Quaternion inverse transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    // Test case 2: Matrix inverse transform
    {
        // 90° rotation around y (column-major)
        auto const rot = glm::fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f
        ); // third column
        auto const covar = glm::fmat3(
            3.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            1.0f
        ); // third column
        auto const transformed = invtransform_covar(rot, covar);
        auto const expected = glm::fmat3(
            1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            3.0f
        ); // third column

        if (!is_close(transformed, expected)) {
            printf("\n[FAIL] Test 2: Matrix inverse transform\n");
            printf("  Transformed: %s\n", glm::to_string(transformed).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_interpolate();
    fails += test_transform_point();
    fails += test_invtransform_point();
    fails += test_transform_dir();
    fails += test_invtransform_dir();
    fails += test_transform_ray();
    fails += test_invtransform_ray();
    fails += test_transform_covar();
    fails += test_invtransform_covar();

    if (fails > 0) {
        printf("[core/se3.cpp] %d tests failed!\n", fails);
    } else {
        printf("[core/se3.cpp] All tests passed!\n");
    }

    return fails;
}