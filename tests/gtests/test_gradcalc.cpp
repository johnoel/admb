#include <gtest/gtest.h>
#include <fvar.hpp>
#include <admodel.h>

extern "C"
{
  void test_ad_exit(const int exit_code);
}

class test_gradcalc: public ::testing::Test {};

TEST_F(test_gradcalc, nvar_zero)
{
  ad_exit=&test_ad_exit;

  dvector g;

  ASSERT_EQ(0, gradient_structure::get_NVAR());
  ASSERT_EQ(0, gradient_structure::get_instances());

  ASSERT_ANY_THROW({
    gradcalc(1, g);
  });
}
TEST_F(test_gradcalc, nvar_one_g_empty)
{
  ad_exit=&test_ad_exit;

  dvector g;

  ASSERT_EQ(0, gradient_structure::get_instances());
  gradient_structure gs;
  ASSERT_EQ(1, gradient_structure::get_instances());

  independent_variables x(1, 1);
  x(1) = 5;

  ASSERT_EQ(0, gradient_structure::get_NVAR());
  dvar_vector variables(x);
  ASSERT_EQ(1, gradient_structure::get_NVAR());

  ASSERT_ANY_THROW({
    gradcalc(1, g);
  });
}
TEST_F(test_gradcalc, nvar_one_g_allocated)
{
  ad_exit=&test_ad_exit;

  dvector g(1, 1);
  g(1) = 12345;

  //Increases gradient_structure::instances.
  gradient_structure gs;

  independent_variables x(1, 1);
  x(1) = 5;

  ASSERT_EQ(0, gradient_structure::get_NVAR());
  dvar_vector variables(x);
  ASSERT_EQ(1, gradient_structure::get_NVAR());

  ASSERT_EQ(gradient_structure::totalbytes(), 0);
  gradcalc(1, g);
  ASSERT_EQ(gradient_structure::totalbytes(), 0);

  //No operations done...gradients are just zero.
  ASSERT_DOUBLE_EQ(0, g(1));
}
TEST_F(test_gradcalc, objective_function_value)
{
  ad_exit=&test_ad_exit;

  dvector g(1, 1);
  g(1) = 12345;

  independent_variables x(1, 1);
  x(1) = 5;

  //Increases gradient_structure::instances.
  gradient_structure gs;

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  dvar_vector variables(x);

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  dvariable result = variables(1);

  ASSERT_EQ(1, gradient_structure::GRAD_STACK1->ptr_index());

  double f = value(result);
  ASSERT_DOUBLE_EQ(5, f);

  ASSERT_EQ(1, gradient_structure::GRAD_STACK1->ptr_index());

  ASSERT_EQ(gradient_structure::totalbytes(), 0);
  gradcalc(1, g);
  ASSERT_EQ(gradient_structure::totalbytes(), 0);

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  //No operations done...gradients are just zero.
  ASSERT_DOUBLE_EQ(1, g(1));
}
TEST_F(test_gradcalc, square)
{
  ad_exit=&test_ad_exit;

  dvector g(1, 1);
  g(1) = 12345;

  independent_variables x(1, 1);
  x(1) = 5;

  //Increases gradient_structure::instances.
  gradient_structure gs;

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  dvar_vector variables(x);

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  dvariable result = square(variables(1));

  ASSERT_EQ(2, gradient_structure::GRAD_STACK1->ptr_index());

  double f = value(result);
  ASSERT_DOUBLE_EQ(25, f);

  ASSERT_EQ(2, gradient_structure::GRAD_STACK1->ptr_index());

  ASSERT_EQ(gradient_structure::totalbytes(), 0);
  gradcalc(1, g);
  ASSERT_EQ(gradient_structure::totalbytes(), 0);

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  //No operations done...gradients are just zero.
  ASSERT_DOUBLE_EQ(10, g(1));
}
