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
TEST_F(test_simple, sum_ax_b_gradcalc)
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
TEST_F(test_simple, sum_ax_b_manual)
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

  gradient_structure::GRAD_LIST->initialize();
  ASSERT_DOUBLE_EQ(value(gradient_structure::RETURN_PTR[0]), 0.0);

  double_and_int* tmp =
    (double_and_int*)gradient_structure::get_ARRAY_MEMBLOCK_BASE();

  unsigned long int imax = gradient_structure::ARR_LIST1->get_max_last_offset() / sizeof(double_and_int);
  for (unsigned int i = 0; i < imax; ++i)
  {
    tmp->x = 0.0;
    ++tmp;
  }

  //gradcalc(2, g);
 
  int icount = 0;
  int element_index = 30;
  while (element_index >= 0)
  {
    grad_stack_entry* element =
      gradient_structure::GRAD_STACK1->get_element(element_index);
    *element->dep_addr = 1.0;

    cout << "Begin *dep_addr: " << *element->dep_addr << endl;
    cout << "Begin *ind_addr1: ";
    if (element->ind_addr1)
    {
      cout << *element->ind_addr1;
    }
    cout << endl;
    cout << "Begin mult1: " << element->mult1<< endl;
    cout << "Begin mult2: " << element->mult2 << endl;
    if (element->func != nullptr)
    {
      (*(element->func))();
      ASSERT_TRUE(false);
    }
    else if (element->func2 != nullptr)
    {
      (*(element->func2))(element);
    }
    cout << "End *dep_addr: " << *element->dep_addr << endl;
    cout << "End *ind_addr1: ";
    if (element->ind_addr1)
    {
      cout << *element->ind_addr1;
    }
    cout << endl;
    cout << "End mult1: " << element->mult1<< endl;
    cout << "End mult2: " << element->mult2 << endl;

    --element_index;

    cout << "icount: " << icount++ << endl;
  }

  ASSERT_DOUBLE_EQ(sum(x), *(gradient_structure::get_INDVAR_LIST()->get_address(0)));
  ASSERT_DOUBLE_EQ(10.0, *(gradient_structure::get_INDVAR_LIST()->get_address(1)));
  ASSERT_DOUBLE_EQ(expected, f);
}
TEST_F(test_simple, sum_ax_b_thread)
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

  gradient_structure::GRAD_LIST->initialize();
  ASSERT_DOUBLE_EQ(value(gradient_structure::RETURN_PTR[0]), 0.0);

  double_and_int* tmp =
    (double_and_int*)gradient_structure::get_ARRAY_MEMBLOCK_BASE();

  unsigned long int imax = gradient_structure::ARR_LIST1->get_max_last_offset() / sizeof(double_and_int);
  for (unsigned int i = 0; i < imax; ++i)
  {
    tmp->x = 0.0;
    ++tmp;
  }

  //gradcalc(2, g);
 
  int icount = 0;
  int element_index = 30;
  while (element_index >= 0)
  {
    grad_stack_entry* element =
      gradient_structure::GRAD_STACK1->get_element(element_index);
    *element->dep_addr = 1.0;

    cout << "Begin *dep_addr: " << *element->dep_addr << endl;
    cout << "Begin *ind_addr1: ";
    if (element->ind_addr1)
    {
      cout << *element->ind_addr1;
    }
    cout << endl;
    cout << "Begin mult1: " << element->mult1<< endl;
    cout << "Begin mult2: " << element->mult2 << endl;
    if (element->func != nullptr)
    {
      (*(element->func))();
      ASSERT_TRUE(false);
    }
    else if (element->func2 != nullptr)
    {
      std::thread t([element]()
      {
        (*(element->func2))(element);
      });
      t.join();
    }
    cout << "End *dep_addr: " << *element->dep_addr << endl;
    cout << "End *ind_addr1: ";
    if (element->ind_addr1)
    {
      cout << *element->ind_addr1;
    }
    cout << endl;
    cout << "End mult1: " << element->mult1<< endl;
    cout << "End mult2: " << element->mult2 << endl;

    --element_index;

    cout << "icount: " << icount++ << endl;
  }

  ASSERT_DOUBLE_EQ(sum(x), *(gradient_structure::get_INDVAR_LIST()->get_address(0)));
  ASSERT_DOUBLE_EQ(10.0, *(gradient_structure::get_INDVAR_LIST()->get_address(1)));
  ASSERT_DOUBLE_EQ(expected, f);
}
TEST_F(test_simple, sum_ax_b_Checkreturns)
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

  gradient_structure::GRAD_LIST->initialize();
  ASSERT_DOUBLE_EQ(value(gradient_structure::RETURN_PTR[0]), 0.0);

  double_and_int* tmp =
    (double_and_int*)gradient_structure::get_ARRAY_MEMBLOCK_BASE();

  unsigned long int imax = gradient_structure::ARR_LIST1->get_max_last_offset() / sizeof(double_and_int);
  for (unsigned int i = 0; i < imax; ++i)
  {
    tmp->x = 0.0;
    ++tmp;
  }

  //gradcalc(2, g);

  int xi = 10;
  double grad0 = 0.0;
  double grad1 = 0.0;
  for (int i = 30; i > 0; i -= 3)
  {
    grad_stack_entry* a = gradient_structure::GRAD_STACK1->get_element(i);
    *a->dep_addr = 1.0;
    std::thread t1([a]()
    {
      (*(a->func2))(a);
    });
    t1.join();
    ASSERT_DOUBLE_EQ(grad0, *(gradient_structure::get_INDVAR_LIST()->get_address(0)));
    ASSERT_DOUBLE_EQ(grad1, *(gradient_structure::get_INDVAR_LIST()->get_address(1)));
    grad_stack_entry* b = gradient_structure::GRAD_STACK1->get_element(i - 1);
    *b->dep_addr = 1.0;
    std::thread t2([b]()
    {
      (*(b->func2))(b);
    });
    t2.join();
    grad1 += 1.0;
    ASSERT_DOUBLE_EQ(grad0, *(gradient_structure::get_INDVAR_LIST()->get_address(0)));
    ASSERT_DOUBLE_EQ(grad1, *(gradient_structure::get_INDVAR_LIST()->get_address(1)));
    grad_stack_entry* c = gradient_structure::GRAD_STACK1->get_element(i - 2);
    *c->dep_addr = 1.0;
    std::thread t3([c]()
    {
      (*(c->func2))(c);
    });
    t3.join();
    grad0 += x(xi);
    --xi;
    ASSERT_DOUBLE_EQ(grad0, *(gradient_structure::get_INDVAR_LIST()->get_address(0)));
    ASSERT_DOUBLE_EQ(grad1, *(gradient_structure::get_INDVAR_LIST()->get_address(1)));
  }

  ASSERT_DOUBLE_EQ(expected, f);
}
