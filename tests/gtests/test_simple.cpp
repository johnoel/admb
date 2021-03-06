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

  unsigned int arrayindex = 20;
  int xi = 10;
  double grad0 = 0.0;
  double grad1 = 0.0;
  for (int i = 30; i > 0; i -= 3)
  {
cout << __FILE__ << ':' << __LINE__ << endl;
    grad_stack_entry* a = gradient_structure::GRAD_STACK1->get_element(i);
    ASSERT_TRUE(a->dep_addr == &result.v->x);
    *a->dep_addr = 1.0;
    ASSERT_DOUBLE_EQ(0.0, *a->ind_addr1);
    std::thread t1([a]()
    {
      (*(a->func2))(a);
    });
    t1.join();
    ASSERT_DOUBLE_EQ(1.0, *a->ind_addr1);
    ASSERT_DOUBLE_EQ(1.0, *a->dep_addr);
    ASSERT_TRUE(a->ind_addr1 == gradient_structure::get_RETURN_ARRAYS(0, arrayindex));
    --arrayindex;
    ASSERT_TRUE(a->ind_addr2 == NULL);
    ASSERT_DOUBLE_EQ(grad0, *(gradient_structure::get_INDVAR_LIST()->get_address(0)));
    ASSERT_DOUBLE_EQ(grad1, *(gradient_structure::get_INDVAR_LIST()->get_address(1)));

    grad_stack_entry* b = gradient_structure::GRAD_STACK1->get_element(i - 1);
    *b->dep_addr = 1.0;
    ASSERT_DOUBLE_EQ(0.0, *b->ind_addr1);
    ASSERT_DOUBLE_EQ(grad1, *b->ind_addr2);
    std::thread t2([b]()
    {
      (*(b->func2))(b);
    });
    t2.join();
    grad1 += 1.0;
    ASSERT_DOUBLE_EQ(0.0, *b->dep_addr);
    ASSERT_DOUBLE_EQ(1.0, *b->ind_addr1);
    ASSERT_DOUBLE_EQ(grad1, *b->ind_addr2);
    ASSERT_TRUE(b->ind_addr1 == gradient_structure::get_RETURN_ARRAYS(0, arrayindex));
    --arrayindex;
    ASSERT_TRUE(b->ind_addr2 == gradient_structure::get_INDVAR_LIST()->get_address(1));
    ASSERT_DOUBLE_EQ(grad0, *(gradient_structure::get_INDVAR_LIST()->get_address(0)));
    ASSERT_DOUBLE_EQ(grad1, *(gradient_structure::get_INDVAR_LIST()->get_address(1)));

    grad_stack_entry* c = gradient_structure::GRAD_STACK1->get_element(i - 2);
    ASSERT_TRUE(c->dep_addr == gradient_structure::get_RETURN_ARRAYS(0, arrayindex + 1));
    *c->dep_addr = 1.0;
    ASSERT_DOUBLE_EQ(grad0, *c->ind_addr1);
    std::thread t3([c]()
    {
      (*(c->func2))(c);
    });
    t3.join();
    ASSERT_DOUBLE_EQ(0.0, *c->dep_addr);
    grad0 += x(xi);
    --xi;
    ASSERT_DOUBLE_EQ(grad0, *c->ind_addr1);
    ASSERT_TRUE(c->ind_addr1 == gradient_structure::get_INDVAR_LIST()->get_address(0));
    ASSERT_TRUE(c->ind_addr2 == NULL);
    ASSERT_DOUBLE_EQ(grad0, *(gradient_structure::get_INDVAR_LIST()->get_address(0)));
    ASSERT_DOUBLE_EQ(grad1, *(gradient_structure::get_INDVAR_LIST()->get_address(1)));
  }

  ASSERT_DOUBLE_EQ(expected, f);
}
TEST_F(test_simple, sum_ax_b_threading)
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

  unsigned int arrayindex = 20;
  std::vector<std::thread> threads;
  for (int i = 30; i > 0; i -= 3)
  {
    threads.push_back(std::thread([&result](int i, int arrayindex, double xi)
    {
      cout << __FILE__ << ':' << __LINE__ << ' ' << arrayindex << endl;
      grad_stack_entry* a = gradient_structure::GRAD_STACK1->get_element(i);
      //DataRace with next line
      ASSERT_TRUE(a->dep_addr == &result.v->x);
      *a->dep_addr = 1.0;
      ASSERT_DOUBLE_EQ(1.0, *a->dep_addr);
      ASSERT_DOUBLE_EQ(0.0, a->ind_value1);
      (*(a->func2))(a);
      ASSERT_DOUBLE_EQ(0.0, a->ind_value1);
      ASSERT_DOUBLE_EQ(1.0, *a->dep_addr);
      ASSERT_TRUE(a->ind_addr1 == gradient_structure::get_RETURN_ARRAYS(0, arrayindex));
      ASSERT_TRUE(a->ind_addr2 == NULL);

      grad_stack_entry* b = gradient_structure::GRAD_STACK1->get_element(i - 1);
      ASSERT_TRUE(b->dep_addr == gradient_structure::get_RETURN_ARRAYS(0, arrayindex));
      *b->dep_addr = 1.0;
      ASSERT_DOUBLE_EQ(1.0, *b->dep_addr);
      ASSERT_DOUBLE_EQ(0.0, b->ind_value1);
      ASSERT_DOUBLE_EQ(0.0, b->ind_value2);
      (*(b->func2))(b);
      ASSERT_DOUBLE_EQ(0.0, *b->dep_addr);
      ASSERT_DOUBLE_EQ(1.0, b->ind_value1);
      ASSERT_DOUBLE_EQ(1.0, b->ind_value2);
      ASSERT_TRUE(b->ind_addr1 == gradient_structure::get_RETURN_ARRAYS(0, arrayindex - 1));
      ASSERT_TRUE(b->ind_addr2 == gradient_structure::get_INDVAR_LIST()->get_address(1));

      grad_stack_entry* c = gradient_structure::GRAD_STACK1->get_element(i - 2);
      ASSERT_TRUE(c->dep_addr == gradient_structure::get_RETURN_ARRAYS(0, arrayindex - 1));
      *c->dep_addr = 1.0;
      ASSERT_DOUBLE_EQ(0.0, c->ind_value1);
      (*(c->func2))(c);
      ASSERT_DOUBLE_EQ(0.0, *c->dep_addr);
      ASSERT_DOUBLE_EQ(xi, c->ind_value1);
      ASSERT_TRUE(c->ind_addr1 == gradient_structure::get_INDVAR_LIST()->get_address(0));
      ASSERT_TRUE(c->ind_addr2 == NULL);
    }, i, arrayindex, x(i / 3)));
    arrayindex -= 2;
  }
  std::for_each(threads.begin(), threads.end(), [](std::thread &t)
  {
    t.join();
  });
  double grad0 = 0.0;
  double* grad_ptr0 = gradient_structure::get_INDVAR_LIST()->get_address(0);
  double grad1 = 0.0;
  double* grad_ptr1 = gradient_structure::get_INDVAR_LIST()->get_address(1);
  for (int i = 30; i > 0; --i)
  {
    grad_stack_entry* gs = gradient_structure::GRAD_STACK1->get_element(i);
    if (gs->ind_addr1 == grad_ptr0)
    {
      grad0 += gs->ind_value1;
    }
    if (gs->ind_addr2 == grad_ptr0)
    {
      grad0 += gs->ind_value2;
    }
    if (gs->ind_addr1 == grad_ptr1)
    {
      grad1 += gs->ind_value1;
    }
    if (gs->ind_addr2 == grad_ptr1)
    {
      grad1 += gs->ind_value2;
    }
  }
  ASSERT_DOUBLE_EQ(grad1, x.size());
  ASSERT_DOUBLE_EQ(grad0, sum(x));
  ASSERT_DOUBLE_EQ(expected, f);
}
void df_plus_eq_pvpv2(grad_stack_entry* grad_ptr)
{
  *grad_ptr->ind_addr1 += *grad_ptr->dep_addr;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
TEST_F(test_simple, sum_ax_b_threading_timer)
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
  for (int i = 30; i > 0; i -= 3)
  {
    grad_stack_entry* a = gradient_structure::GRAD_STACK1->get_element(i);
    a->func2 = &df_plus_eq_pvpv2;
  }
  auto start = std::chrono::system_clock::now();
  unsigned int arrayindex = 20;
  std::vector<std::thread> threads;
  for (int i = 30; i > 0; i -= 3)
  {
    threads.push_back(std::thread([&result](int i, int arrayindex, double xi)
    {
      grad_stack_entry* a = gradient_structure::GRAD_STACK1->get_element(i);
      //DataRace with next line
      *a->dep_addr = 1.0;
      (*(a->func2))(a);

      grad_stack_entry* b = gradient_structure::GRAD_STACK1->get_element(i - 1);
      *b->dep_addr = 1.0;
      (*(b->func2))(b);

      grad_stack_entry* c = gradient_structure::GRAD_STACK1->get_element(i - 2);
      *c->dep_addr = 1.0;
      (*(c->func2))(c);
    }, i, arrayindex, x(i / 3)));
    arrayindex -= 2;
  }
  double grad0 = 0.0;
  double grad1 = 0.0;
  double* grad_ptr0 = gradient_structure::get_INDVAR_LIST()->get_address(0);
  double* grad_ptr1 = gradient_structure::get_INDVAR_LIST()->get_address(1);
  int index2 = 0;
  std::for_each(threads.begin(), threads.end(),
    [&index2, &grad0, &grad_ptr0, &grad1, &grad_ptr1](std::thread &t)
    {
      t.join();
      for (int i = 0; i < 3; ++i)
      {
        grad_stack_entry* gs =
          gradient_structure::GRAD_STACK1->get_element(index2);
        if (gs->ind_addr1 == grad_ptr0)
        {
          grad0 += gs->ind_value1;
        }
        if (gs->ind_addr2 == grad_ptr0)
        {
          grad0 += gs->ind_value2;
        }
        if (gs->ind_addr1 == grad_ptr1)
        {
          grad1 += gs->ind_value1;
        }
        if (gs->ind_addr2 == grad_ptr1)
        {
          grad1 += gs->ind_value2;
        }
        ++index2;
      }
    }
  );
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  ASSERT_DOUBLE_EQ(grad1, x.size());
  ASSERT_DOUBLE_EQ(grad0, sum(x));

  auto start2 = std::chrono::system_clock::now();
  gradcalc(2, g);
  auto end2 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
  cout << "Time: "
       << elapsed_seconds.count() << ' ' << elapsed_seconds2.count() << endl;
  ASSERT_TRUE(elapsed_seconds.count() < elapsed_seconds2.count());
}
void devgradcalc(int nvar, dvector& g)
{
  gradient_structure::GRAD_LIST->initialize();

  double_and_int* tmp =
    (double_and_int*)gradient_structure::get_ARRAY_MEMBLOCK_BASE();

  unsigned long int imax =
    gradient_structure::ARR_LIST1->get_max_last_offset() /
    sizeof(double_and_int);
  for (unsigned int i = 0; i < imax; ++i)
  {
    tmp->x = 0.0;
    ++tmp;
  }
  unsigned int arrayindex = 20;
  std::vector<std::thread> threads;
  for (int i = 30; i > 0; i -= 3)
  {
    threads.push_back(std::thread([](int i, int arrayindex)
    {
      grad_stack_entry* a = gradient_structure::GRAD_STACK1->get_element(i);
      //DataRace with next line
      *a->dep_addr = 1.0;
      (*(a->func2))(a);

      grad_stack_entry* b = gradient_structure::GRAD_STACK1->get_element(i - 1);
      *b->dep_addr = 1.0;
      (*(b->func2))(b);

      grad_stack_entry* c = gradient_structure::GRAD_STACK1->get_element(i - 2);
      *c->dep_addr = 1.0;
      (*(c->func2))(c);
    }, i, arrayindex));
    arrayindex -= 2;
  }
  double grad0 = 0.0;
  double grad1 = 0.0;
  double* grad_ptr0 = gradient_structure::get_INDVAR_LIST()->get_address(0);
  double* grad_ptr1 = gradient_structure::get_INDVAR_LIST()->get_address(1);
  int index2 = 0;
  std::for_each(threads.begin(), threads.end(),
    [&index2, &grad0, &grad_ptr0, &grad1, &grad_ptr1](std::thread &t)
    {
      t.join();
      for (int i = 0; i < 3; ++i)
      {
        grad_stack_entry* gs =
          gradient_structure::GRAD_STACK1->get_element(index2);
        if (gs->ind_addr1 == grad_ptr0)
        {
          grad0 += gs->ind_value1;
        }
        if (gs->ind_addr2 == grad_ptr0)
        {
          grad0 += gs->ind_value2;
        }
        if (gs->ind_addr1 == grad_ptr1)
        {
          grad1 += gs->ind_value1;
        }
        if (gs->ind_addr2 == grad_ptr1)
        {
          grad1 += gs->ind_value2;
        }
        ++index2;
      }
    }
  );
  *grad_ptr0 = grad0;
  *grad_ptr1 = grad1;
}
TEST_F(test_simple, sum_ax_b_thread_devgradcalc)
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

  for (int i = 30; i > 0; i -= 3)
  {
    grad_stack_entry* a = gradient_structure::GRAD_STACK1->get_element(i);
    a->func2 = &df_plus_eq_pvpv2;
  }

  dvector g(1, 2);
  g.initialize();

  auto start = std::chrono::system_clock::now();
  devgradcalc(2, g);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  ASSERT_DOUBLE_EQ(sum(x), *(gradient_structure::get_INDVAR_LIST()->get_address(0)));
  ASSERT_DOUBLE_EQ(x.size(), *(gradient_structure::get_INDVAR_LIST()->get_address(1)));

  double* grad_ptr0 = gradient_structure::get_INDVAR_LIST()->get_address(0);
  double* grad_ptr1 = gradient_structure::get_INDVAR_LIST()->get_address(1);
  *grad_ptr0 = 0.0;
  *grad_ptr1 = 0.0;
  auto start2 = std::chrono::system_clock::now();
  gradcalc(2, g);
  auto end2 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
  ASSERT_DOUBLE_EQ(sum(x), *(gradient_structure::get_INDVAR_LIST()->get_address(0)));
  ASSERT_DOUBLE_EQ(x.size(), *(gradient_structure::get_INDVAR_LIST()->get_address(1)));

  cout << "Time: "
       << elapsed_seconds.count() << ' ' << elapsed_seconds2.count() << endl;
  ASSERT_TRUE(elapsed_seconds.count() < elapsed_seconds2.count());
}
