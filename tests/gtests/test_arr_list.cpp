#include <gtest/gtest.h>
#include "fvar.hpp"

extern "C"
{
  void test_ad_exit(const int exit_code);
}

class test_arr_list: public ::testing::Test {};

TEST_F(test_arr_list, default_constructor)
{
  arr_list arrlist;
}
