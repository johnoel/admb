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
TEST_F(test_gradcalc, square_minus_square)
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

  dvariable result = square(variables(1)) - square(variables(1));

  ASSERT_EQ(4, gradient_structure::GRAD_STACK1->ptr_index());

  double f = value(result);
  ASSERT_DOUBLE_EQ(0, f);

  ASSERT_EQ(4, gradient_structure::GRAD_STACK1->ptr_index());

  ASSERT_EQ(gradient_structure::totalbytes(), 0);
  gradcalc(1, g);
  ASSERT_EQ(gradient_structure::totalbytes(), 0);

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  //No operations done...gradients are just zero.
  ASSERT_DOUBLE_EQ(0, g(1));
}
TEST_F(test_gradcalc, square_plus_square)
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

  dvariable result = square(variables(1)) + square(variables(1));

  ASSERT_EQ(4, gradient_structure::GRAD_STACK1->ptr_index());

  double f = value(result);
  ASSERT_DOUBLE_EQ(50, f);

  ASSERT_EQ(4, gradient_structure::GRAD_STACK1->ptr_index());

  ASSERT_EQ(gradient_structure::totalbytes(), 0);
  gradcalc(1, g);
  ASSERT_EQ(gradient_structure::totalbytes(), 0);

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  //No operations done...gradients are just zero.
  ASSERT_DOUBLE_EQ(20, g(1));
}
TEST_F(test_gradcalc, cube_minus_square)
{
  ad_exit=&test_ad_exit;

  dvector g(1, 1);
  g(1) = 12345;

  independent_variables x(1, 1);
  x(1) = 5;

  //Increases gradient_structure::instances.
  gradient_structure gs;

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());
  ASSERT_EQ(0, gradient_structure::ARR_LIST1->get_number_arr_links());
  ASSERT_EQ(0, gradient_structure::ARR_LIST1->get_last_offset());

  dvar_vector variables(x);
  ASSERT_EQ(1, gradient_structure::ARR_LIST1->get_number_arr_links());
  ASSERT_EQ(8, gradient_structure::ARR_LIST1->get_last_offset());
  ASSERT_EQ(8, gradient_structure::ARR_LIST1->get_max_last_offset());

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  dvariable result = cube(variables(1)) - square(variables(1));

  ASSERT_EQ(4, gradient_structure::GRAD_STACK1->ptr_index());

  double f = value(result);
  ASSERT_DOUBLE_EQ(100, f);

  ASSERT_EQ(4, gradient_structure::GRAD_STACK1->ptr_index());

  ASSERT_EQ(gradient_structure::totalbytes(), 0);
  gradcalc(1, g);
  ASSERT_EQ(gradient_structure::totalbytes(), 0);

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  //No operations done...gradients are just zero.
  ASSERT_DOUBLE_EQ(65, g(1));
  ASSERT_EQ(1, gradient_structure::ARR_LIST1->get_number_arr_links());
  ASSERT_EQ(8, gradient_structure::ARR_LIST1->get_last_offset());
  ASSERT_EQ(8, gradient_structure::ARR_LIST1->get_max_last_offset());
}
TEST_F(test_gradcalc, cube_square_then_minus)
{
  ad_exit=&test_ad_exit;

  independent_variables x(1, 1);
  x(1) = 5;

  //Increases gradient_structure::instances.
  gradient_structure gs;

  dvar_vector variables(x);

  dvariable result = cube(variables(1));
  double f = value(result);
  ASSERT_DOUBLE_EQ(125, f);
  dvector g(1, 1);
  g.initialize();
  gradcalc(1, g);

  //Wierd the value of variables is the gradient after gradcalc.
  ASSERT_DOUBLE_EQ(75, value(variables(1)));
  ASSERT_DOUBLE_EQ(75, g(1));

  //Must reconstruct variables with x.
  dvar_vector variables2(x);
  dvariable result2 = square(variables2(1));
  double f2 = value(result2);
  ASSERT_DOUBLE_EQ(25, f2);
  dvector g2(1, 1);
  g2.initialize();
  gradcalc(1, g2);
  ASSERT_DOUBLE_EQ(10, g2(1));
  ASSERT_DOUBLE_EQ(value(variables2(1)), g2(1));

  //Able to take two independant variables
  //and sum them up.  Idea for multiple parrallel
  //gradient stacks.  f_total = cube(x) - square(x)
  ASSERT_DOUBLE_EQ(100, f - f2);
  ASSERT_DOUBLE_EQ(65, g(1) - g2(1));
}
/**
Compute the gradient from the data stored in the global \ref gradient_structure.

\param nvar Number of variables in the gradient.
\param _g Vector from 1 to nvar. On return contains the gradient.
\param f objective function
\returns likelihood value
*/
double gradcalc(int nvar, const dvector& _g, dvariable& f)
{
  double v = value(f);
  gradcalc(nvar, _g);
  return v;
}
TEST_F(test_gradcalc, why_did_profile_likelihood_break)
{
  ad_exit=&test_ad_exit;

  independent_variables x(1, 1);
  x(1) = 5;

  //Increases gradient_structure::instances.
  gradient_structure gs;

  dvar_vector variables(x);

  dvariable result = cube(variables(1));
  dvector g(1, 1);
  g.initialize();

  double f = gradcalc(1, g, result);
  ASSERT_DOUBLE_EQ(125, f);

  //Wierd the value of variables is the gradient after gradcalc.
  ASSERT_DOUBLE_EQ(75, value(variables(1)));
  ASSERT_DOUBLE_EQ(75, g(1));

  //Must reconstruct variables with x.
  dvar_vector variables2(x);
  dvariable result2 = square(variables2(1));
  dvector g2(1, 1);
  g2.initialize();
  double f2 = gradcalc(1, g2, result2);
  ASSERT_DOUBLE_EQ(25, f2);
  ASSERT_DOUBLE_EQ(10, g2(1));
  ASSERT_DOUBLE_EQ(value(variables2(1)), g2(1));

  //Able to take two independant variables
  //and sum them up.  Idea for multiple parrallel
  //gradient stacks.  f_total = cube(x) - square(x)
  ASSERT_DOUBLE_EQ(100, f - f2);
  ASSERT_DOUBLE_EQ(65, g(1) - g2(1));
}
TEST_F(test_gradcalc, split_cube_minus_square)
{
  ad_exit=&test_ad_exit;

  dvector g(1, 1);
  g(1) = 12345;

  independent_variables x(1, 1);
  x(1) = 5;

  //Increases gradient_structure::instances.
  gradient_structure gs;

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());
  ASSERT_EQ(0, gradient_structure::ARR_LIST1->get_number_arr_links());
  ASSERT_EQ(0, gradient_structure::ARR_LIST1->get_last_offset());

  dvar_vector variables(x);
  ASSERT_EQ(1, gradient_structure::ARR_LIST1->get_number_arr_links());
  ASSERT_EQ(8, gradient_structure::ARR_LIST1->get_last_offset());
  ASSERT_EQ(8, gradient_structure::ARR_LIST1->get_max_last_offset());

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  dvariable a = cube(variables(1));
  ASSERT_EQ(2, gradient_structure::GRAD_STACK1->ptr_index());
  dvariable b = square(variables(1));
  ASSERT_EQ(4, gradient_structure::GRAD_STACK1->ptr_index());
  dvariable result = a - b;
  ASSERT_EQ(6, gradient_structure::GRAD_STACK1->ptr_index());

  double f = value(result);
  ASSERT_DOUBLE_EQ(100, f);

  ASSERT_EQ(6, gradient_structure::GRAD_STACK1->ptr_index());

  ASSERT_EQ(gradient_structure::totalbytes(), 0);
  gradcalc(1, g);
  ASSERT_EQ(gradient_structure::totalbytes(), 0);

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  //No operations done...gradients are just zero.
  ASSERT_DOUBLE_EQ(65, g(1));
  ASSERT_EQ(1, gradient_structure::ARR_LIST1->get_number_arr_links());
  ASSERT_EQ(8, gradient_structure::ARR_LIST1->get_last_offset());
  ASSERT_EQ(8, gradient_structure::ARR_LIST1->get_max_last_offset());
}
