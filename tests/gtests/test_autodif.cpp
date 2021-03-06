#include <gtest/gtest.h>
#include <fvar.hpp>

class test_autodif: public ::testing::Test {};

TEST_F(test_autodif, gradcalc)
{
  const int nvar = 1;
  dvector g(1, nvar);

  ASSERT_DEATH(gradcalc(nvar, g), "nvar != gradient_structure::NVAR in gradcalc");
}
/*
  gradient_structure gs(1500);

  independent_variables variables(1, nvar);
  dvar_vector x(variables);

  dvariable variable = sum(x);

  double f = value(variable);
  dvector g(1, nvar);
*/

TEST_F(test_autodif, dvar_vector)
{
  const int nvar = 1;
  independent_variables variables(1, nvar);
  ASSERT_DEATH
  (
    {
      dvar_vector x(variables);
    },
    "Error -- you are trying to create a dvar_vector object"
  );
}
TEST_F(test_autodif, grad_stack1)
{
  ASSERT_TRUE(gradient_structure::GRAD_STACK1 == NULL);
  {
    gradient_structure gs(1500);
    ASSERT_TRUE(gradient_structure::GRAD_STACK1 != NULL);

    const int nvar = 1;
    independent_variables variables(1, nvar);
    dvar_vector x(variables);
    ASSERT_TRUE(gradient_structure::GRAD_STACK1 != NULL);
  }
  ASSERT_TRUE(gradient_structure::GRAD_STACK1 == NULL);
}
/*
TEST_F(test_autodif, dvar_vector3)
{
  ASSERT_TRUE(gradient_structure::GRAD_STACK1 == NULL);
  gradient_structure gs(1500);
  const int nvar = 1;
  independent_variables variables(1, nvar);
  variables[1] = 10;
  dvar_vector x(variables);
  
  ASSERT_DOUBLE_EQ(value(x[1]), 10);

  dvariable v = sum(x);
  double f = value(v);
  ASSERT_DOUBLE_EQ(f, 10);

  dvector g(1, nvar);
  gradcalc(nvar, g);
}
*/
