#pragma once
#include <string>
#include <vector>
#include <functional>
#include <iostream>

struct TestCase {
    std::string name;
    std::function<void()> func;
};

class TestRegistry {
public:
    static std::vector<TestCase>& get_tests() {
        static std::vector<TestCase> tests;
        return tests;
    }
    static void add(const std::string& name, std::function<void()> func) {
        get_tests().push_back({name, func});
    }
};

// Macro to make adding tests effortless
// NOTE
// The first `void name();` to declare a function void with name();
// The second `void name()` without `;`  is where our function definition {..} continues

// REGISTER_TEST(TEST_NAME) { ... }
// -> void TEST_NAME; static int ...; void TEST_NAME() { ... }

// # convert the function name in to a string
// ## is used to concatenate tokens
// #define ArgArg(x,y) x##y -> ArgArg(lady, bug) "ladybug"

#define REGISTER_TEST(name) \
    void name(); \
    static int dummy_##name = []() { \
        TestRegistry::add(#name, name); \
        return 0; \
    }(); \
    void name()
