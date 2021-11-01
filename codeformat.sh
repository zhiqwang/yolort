#!/usr/bin/env bash

# we run clang-format and astyle twice to get stable format output

find deployment/ tools/ tests/ -type f -name '*.c' -o -name '*.cpp' -o -name '*.cc' -o -name '*.h' | xargs -i clang-format -i {}
astyle -n -r "tests/*.h,*.cpp,*.cc" "tools/*.h,*.cpp,*.cc"
astyle -n -r "deployment/*.h,*.cpp,*.cc"

find deployment/ tools/ tests/ -type f -name '*.c' -o -name '*.cpp' -o -name '*.cc' -o -name '*.h' | xargs -i clang-format -i {}
astyle -n -r "tests/*.h,*.cpp,*.cc" "tools/*.h,*.cpp,*.cc"
astyle -n -r "deployment/*.h,*.cpp,*.cc"
