#include <gtest/gtest.h>
#include <fvar.hpp>
#include <admodel.h>
#include <thread>

extern "C"
{
  void test_ad_exit(const int exit_code);
}

class test_simple: public ::testing::Test
{
protected:
  void SetUp()
  {
    ad_comm::argc = 0;
    ad_comm::argv = nullptr;
    x.allocate(1, 10);
    x(1) = -1;
    x(2) = 0;
    x(3) = 1;
    x(4) = 2;
    x(5) = 3;
    x(6) = 4;
    x(7) = 5;
    x(8) = 6;
    x(9) = 7;
    x(10) = 8;
  }
  void TearDown()
  {
    ad_comm::argc = 0;
    ad_comm::argv = nullptr;
  }

  dvector x;
};

TEST_F(test_simple, ax_b_simple)
{
  ad_exit=&test_ad_exit;

  dvector g(1, 2);
  g(1) = 12345;
  g(2) = 12345;

  independent_variables independants(1, 2);
  independants(1) = 5;
  independants(2) = 10;

  //Increases gradient_structure::instances.
  gradient_structure gs;

  dvar_vector variables(independants);
  
  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());
  //ax + b
  dvariable result = variables(1) * x(1) + variables(2);
  ASSERT_EQ(3, gradient_structure::GRAD_STACK1->ptr_index());

  double f = value(result);
  ASSERT_EQ(3, gradient_structure::GRAD_STACK1->ptr_index());

  gradcalc(2, g);
  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());

  ASSERT_DOUBLE_EQ(x(1), g(1));
  ASSERT_DOUBLE_EQ(1.0, g(2));
  ASSERT_DOUBLE_EQ(independants(1) * x(1) + independants(2), f);
  //Gradient
  ASSERT_DOUBLE_EQ(g(1), variables.elem(1).v->x);
  ASSERT_DOUBLE_EQ(g(2), variables.elem(2).v->x);
}
TEST_F(test_simple, ax_b_simple_loop)
{
  ad_exit=&test_ad_exit;

  //Increases gradient_structure::instances.
  independent_variables independants(1, 2);
  independants(1) = 5;
  independants(2) = 10;

  gradient_structure gs;
  dvar_vector variables(independants);
  
  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());
  dvariable result = 0;
  ASSERT_EQ(1, gradient_structure::GRAD_STACK1->ptr_index());

  double expected = 0;
  int index = 1; 
  for (int i = 1; i <= 10; ++i)
  {
    ASSERT_EQ(index, gradient_structure::GRAD_STACK1->ptr_index());

    //ax + b
    result += variables(1) * x(i) + variables(2);
    expected += independants(1) * x(i) + independants(2);

    index += 3;
    ASSERT_EQ(index, gradient_structure::GRAD_STACK1->ptr_index());
  }

  double f = value(result);
  ASSERT_EQ(index, 31);
  ASSERT_EQ(index, gradient_structure::GRAD_STACK1->ptr_index());

  dvector g(1, 2);
  g.initialize();
  gradcalc(2, g);

  ASSERT_DOUBLE_EQ(sum(x), g(1));
  ASSERT_DOUBLE_EQ(10.0, g(2));
  ASSERT_DOUBLE_EQ(expected, f);
  //Gradient
  ASSERT_DOUBLE_EQ(g(1), variables.elem(1).v->x);
  ASSERT_DOUBLE_EQ(g(2), variables.elem(2).v->x);
}
/*
TEST_F(test_simple, log_manual_simple)
{
  ad_exit=&test_ad_exit;

  dvector g(1, 1);
  g(1) = 12345;

  independent_variables x(1, 1);
  x(1) = 5;

  //Increases gradient_structure::instances.
  gradient_structure gs;

  dvar_vector variables(x);
  ASSERT_DOUBLE_EQ(5.0, x(1));
  ASSERT_DOUBLE_EQ(5.0, variables.elem(1).v->x);

  ASSERT_EQ(0, gradient_structure::GRAD_STACK1->ptr_index());
  dvariable result = log(variables(1));

  double f = value(result);
  ASSERT_DOUBLE_EQ(std::log(x(1)), f);

  //Points at next available element.
  ASSERT_EQ(2, gradient_structure::GRAD_STACK1->ptr_index());
  ASSERT_EQ(nullptr, gradient_structure::GRAD_STACK1->get_element(2));

  ///Begin Needed!!!
  ASSERT_DOUBLE_EQ(value(gradient_structure::RETURN_PTR[0]), f);
  gradient_structure::GRAD_LIST->initialize();
  ASSERT_DOUBLE_EQ(value(gradient_structure::RETURN_PTR[0]), 0.0);

  double_and_int* ptr = (double_and_int*)gradient_structure::get_ARRAY_MEMBLOCK_BASE();
  ASSERT_DOUBLE_EQ(ptr->x, 5.0);
  ptr->x = 0.0;
  ///End Needed!!!

  //assigment operator
  grad_stack_entry* e1 = gradient_structure::GRAD_STACK1->get_element(1);
  *e1->dep_addr = 1.0;
  ASSERT_EQ(nullptr, e1->func);
  ASSERT_EQ(&default_evaluation1, e1->func2);
  ASSERT_DOUBLE_EQ(1.0, *e1->dep_addr);
  ASSERT_EQ(&value(result), e1->dep_addr);
  ASSERT_DOUBLE_EQ(0.0, *e1->ind_addr1);
  //Return of pow
  ASSERT_EQ(&value(gradient_structure::RETURN_PTR[0]), e1->ind_addr1);
  std::thread t2([e1]()
  {
    (*(e1->func2))(e1);
  });
  t2.join();
  ASSERT_DOUBLE_EQ(0.0, *e1->dep_addr);
  ASSERT_DOUBLE_EQ(1.0, *e1->ind_addr1);
  ASSERT_DOUBLE_EQ(0.0, e1->mult1);
  ASSERT_DOUBLE_EQ(0.0, e1->mult2);
  ASSERT_EQ(nullptr, e1->ind_addr2);

  //pow function fvar_fn.cpp
  grad_stack_entry* e0 = gradient_structure::GRAD_STACK1->get_element(0);
  ASSERT_EQ(nullptr, e0->func);
  //ASSERT_EQ(&default_evaluation, e0->func2);
  ASSERT_DOUBLE_EQ(1.0, *e0->dep_addr);
  //Return of pow
  ASSERT_EQ(&value(gradient_structure::RETURN_PTR[0]), e0->dep_addr);
  ASSERT_DOUBLE_EQ(0.2, e0->mult1);
  ASSERT_DOUBLE_EQ(0.0, *e0->ind_addr1);
  ASSERT_EQ(&value(variables(1)), e0->ind_addr1);
  ASSERT_DOUBLE_EQ(0.0, e0->mult2);
  ASSERT_EQ(nullptr, e0->ind_addr2);
  std::thread t1([e0]()
  {
    (*(e0->func2))(e0);
  });
  t1.join();
  ASSERT_DOUBLE_EQ(0.0, *e0->dep_addr);
  ASSERT_DOUBLE_EQ(0.2, e0->mult1);
  ASSERT_DOUBLE_EQ(0.2, *e0->ind_addr1);
  ASSERT_DOUBLE_EQ(0.0, e0->mult2);
  ASSERT_EQ(nullptr, e0->ind_addr2);

  ASSERT_DOUBLE_EQ(5.0, x(1));
  //Gradient
  ASSERT_DOUBLE_EQ(1.0 / x(1), variables.elem(1).v->x);
}
*/
