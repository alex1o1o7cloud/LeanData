import Mathlib

namespace NUMINAMATH_CALUDE_g_neg_one_value_l3578_357892

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of y = f(x) + x^2 being an odd function
def is_odd_composite (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem g_neg_one_value (f : ℝ → ℝ) 
  (h1 : is_odd_composite f) 
  (h2 : f 1 = 1) : 
  g f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_one_value_l3578_357892


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l3578_357817

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : average_children = 3)
  (h3 : childless_families = 3)
  : (total_families * average_children) / (total_families - childless_families) = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l3578_357817


namespace NUMINAMATH_CALUDE_cos_double_angle_special_point_l3578_357804

/-- Given that the terminal side of angle α passes through point (1,2), prove that cos 2α = -3/5 -/
theorem cos_double_angle_special_point (α : ℝ) :
  (∃ r : ℝ, r > 0 ∧ r * Real.cos α = 1 ∧ r * Real.sin α = 2) →
  Real.cos (2 * α) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_point_l3578_357804


namespace NUMINAMATH_CALUDE_min_value_theorem_l3578_357802

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3578_357802


namespace NUMINAMATH_CALUDE_population_scientific_notation_l3578_357875

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem population_scientific_notation :
  let population : ℝ := 1412.60 * 1000000
  let scientific_form := toScientificNotation population
  scientific_form.coefficient = 1.4126 ∧ scientific_form.exponent = 5 := by
  sorry

end NUMINAMATH_CALUDE_population_scientific_notation_l3578_357875


namespace NUMINAMATH_CALUDE_f_derivative_neg_one_l3578_357834

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Theorem statement
theorem f_derivative_neg_one (a b c : ℝ) : 
  f' a b 1 = 2 → f' a b (-1) = -2 := by
sorry

end NUMINAMATH_CALUDE_f_derivative_neg_one_l3578_357834


namespace NUMINAMATH_CALUDE_childs_movie_ticket_cost_l3578_357864

/-- Proves that the cost of a child's movie ticket is $3 given the specified conditions. -/
theorem childs_movie_ticket_cost (total_money : ℚ) (adult_ticket_cost : ℚ) (num_children : ℕ) 
  (h1 : total_money = 35)
  (h2 : adult_ticket_cost = 8)
  (h3 : num_children = 9) :
  ∃ (child_ticket_cost : ℚ), 
    child_ticket_cost = 3 ∧ 
    adult_ticket_cost + num_children * child_ticket_cost ≤ total_money :=
by sorry

end NUMINAMATH_CALUDE_childs_movie_ticket_cost_l3578_357864


namespace NUMINAMATH_CALUDE_group_average_difference_l3578_357889

/-- Represents the first element of the n-th group -/
def a (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n % 2 = 1 then a (n - 1) + (n - 1)
  else a (n - 1) + (n - 2)

/-- Sum of elements in the n-th group -/
def S (n : ℕ) : ℕ :=
  n * (2 * a n + (n - 1) * 2) / 2

/-- Average of elements in the n-th group -/
def avg (n : ℕ) : ℚ :=
  (S n : ℚ) / n

theorem group_average_difference (n : ℕ) :
  avg (2 * n + 1) - avg (2 * n) = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_group_average_difference_l3578_357889


namespace NUMINAMATH_CALUDE_salmon_count_l3578_357842

theorem salmon_count (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261) 
  (h2 : female_salmon = 259378) : 
  male_salmon + female_salmon = 971639 := by
  sorry

end NUMINAMATH_CALUDE_salmon_count_l3578_357842


namespace NUMINAMATH_CALUDE_girls_in_first_grade_l3578_357863

/-- Represents the first grade class configuration -/
structure FirstGrade where
  classrooms : ℕ
  boys : ℕ
  students_per_classroom : ℕ

/-- Calculates the number of girls in the first grade -/
def girls_count (fg : FirstGrade) : ℕ :=
  fg.classrooms * fg.students_per_classroom - fg.boys

/-- Theorem stating the number of girls in the first grade -/
theorem girls_in_first_grade (fg : FirstGrade) 
  (h1 : fg.classrooms = 4)
  (h2 : fg.boys = 56)
  (h3 : fg.students_per_classroom = 25)
  (h4 : ∀ c, c ≤ fg.classrooms → fg.boys / fg.classrooms = (girls_count fg) / fg.classrooms) :
  girls_count fg = 44 := by
  sorry

#eval girls_count ⟨4, 56, 25⟩

end NUMINAMATH_CALUDE_girls_in_first_grade_l3578_357863


namespace NUMINAMATH_CALUDE_range_of_a_l3578_357883

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 ≥ a

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  prop_p a ∧ prop_q a ↔ a ∈ Set.Iic (-2) ∪ {1} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3578_357883


namespace NUMINAMATH_CALUDE_janet_pay_calculation_l3578_357866

/-- Represents Janet's work parameters and calculates her pay per post. -/
def janet_pay_per_post (check_time : ℕ) (hourly_rate : ℚ) : ℚ :=
  let seconds_per_hour : ℕ := 3600
  let posts_per_hour : ℕ := seconds_per_hour / check_time
  hourly_rate / posts_per_hour

/-- Proves that Janet's pay per post is $0.25 given the specified conditions. -/
theorem janet_pay_calculation :
  janet_pay_per_post 10 90 = 1/4 := by
  sorry

#eval janet_pay_per_post 10 90

end NUMINAMATH_CALUDE_janet_pay_calculation_l3578_357866


namespace NUMINAMATH_CALUDE_smallest_prime_q_l3578_357813

theorem smallest_prime_q (p : ℕ) : 
  Prime p → Prime (13 * p + 2) → (13 * p + 2) ≥ 41 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_q_l3578_357813


namespace NUMINAMATH_CALUDE_aluminum_atomic_weight_l3578_357871

/-- The atomic weight of chlorine in atomic mass units (amu) -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of the compound in atomic mass units (amu) -/
def compound_weight : ℝ := 132

/-- The number of chlorine atoms in the compound -/
def chlorine_count : ℕ := 3

/-- The atomic weight of aluminum in atomic mass units (amu) -/
def aluminum_weight : ℝ := compound_weight - chlorine_count * chlorine_weight

theorem aluminum_atomic_weight :
  aluminum_weight = 25.65 := by sorry

end NUMINAMATH_CALUDE_aluminum_atomic_weight_l3578_357871


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3578_357803

theorem tan_alpha_value (α : ℝ) :
  (3 * Real.sin (Real.pi + α) + Real.cos (-α)) / (4 * Real.sin (-α) - Real.cos (9 * Real.pi + α)) = 2 →
  Real.tan α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3578_357803


namespace NUMINAMATH_CALUDE_import_tax_calculation_l3578_357848

theorem import_tax_calculation (total_value : ℝ) (tax_rate : ℝ) (threshold : ℝ) (tax_amount : ℝ) : 
  total_value = 2580 →
  tax_rate = 0.07 →
  threshold = 1000 →
  tax_amount = (total_value - threshold) * tax_rate →
  tax_amount = 110.60 := by
sorry

end NUMINAMATH_CALUDE_import_tax_calculation_l3578_357848


namespace NUMINAMATH_CALUDE_water_consumption_l3578_357886

theorem water_consumption (initial_water : ℚ) : 
  initial_water > 0 →
  let remaining_day1 := initial_water / 2
  let remaining_day2 := remaining_day1 * 2 / 3
  let remaining_day3 := remaining_day2 / 2
  remaining_day3 = 250 →
  initial_water = 1500 := by
sorry

end NUMINAMATH_CALUDE_water_consumption_l3578_357886


namespace NUMINAMATH_CALUDE_initial_girls_count_l3578_357818

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 18) = b) →
  (4 * (b - 36) = g - 18) →
  g = 31 :=
by sorry

end NUMINAMATH_CALUDE_initial_girls_count_l3578_357818


namespace NUMINAMATH_CALUDE_remainder_3012_div_97_l3578_357887

theorem remainder_3012_div_97 : 3012 % 97 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3012_div_97_l3578_357887


namespace NUMINAMATH_CALUDE_no_real_solutions_for_abs_equation_l3578_357881

theorem no_real_solutions_for_abs_equation :
  ∀ x : ℝ, |2*x - 6| ≠ x^2 - x + 2 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_abs_equation_l3578_357881


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3578_357806

-- Define the propositions p and q
def p (x : ℝ) : Prop := x = 2
def q (x : ℝ) : Prop := 0 < x ∧ x < 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ 
  (∃ x, q x ∧ ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3578_357806


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3578_357876

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3578_357876


namespace NUMINAMATH_CALUDE_no_number_with_specific_digit_sums_l3578_357831

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: No natural number exists with sum of digits 1000 and sum of square's digits 1000000 -/
theorem no_number_with_specific_digit_sums :
  ¬ ∃ n : ℕ, sumOfDigits n = 1000 ∧ sumOfDigits (n^2) = 1000000 := by sorry

end NUMINAMATH_CALUDE_no_number_with_specific_digit_sums_l3578_357831


namespace NUMINAMATH_CALUDE_mutually_expressible_implies_symmetric_zero_l3578_357839

/-- A function f is symmetric if f(x, y) = f(y, x) for all x and y. -/
def IsSymmetric (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = f y x

/-- Two variables x and y are mutually expressible if there exists a symmetric function f
    such that f(x, y) = 0 implies both y = g(x) and x = g(y) for some function g. -/
def MutuallyExpressible (x y : ℝ) : Prop :=
  ∃ (f : ℝ → ℝ → ℝ) (g : ℝ → ℝ), IsSymmetric f ∧ f x y = 0 ∧ y = g x ∧ x = g y

/-- Theorem: If two variables are mutually expressible, then there exists a symmetric function
    that equals zero for those variables. -/
theorem mutually_expressible_implies_symmetric_zero (x y : ℝ) :
  MutuallyExpressible x y → ∃ (f : ℝ → ℝ → ℝ), IsSymmetric f ∧ f x y = 0 := by
  sorry

end NUMINAMATH_CALUDE_mutually_expressible_implies_symmetric_zero_l3578_357839


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3578_357809

/-- Given an income, dividend rate, and investment amount, calculate the price of a stock. -/
theorem stock_price_calculation (income investment : ℚ) (dividend_rate : ℚ) : 
  income = 650 →
  dividend_rate = 1/10 →
  investment = 6240 →
  (investment / (income / dividend_rate)) * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l3578_357809


namespace NUMINAMATH_CALUDE_car_speed_problem_l3578_357824

/-- Proves that the speed at which a car takes 15 seconds less to travel 1 kilometer
    compared to traveling at 48 km/h is 60 km/h. -/
theorem car_speed_problem (v : ℝ) : v > 0 →
  (1 / (48 / 3600) = 1 / (v / 3600) + 15) → v = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3578_357824


namespace NUMINAMATH_CALUDE_binary_101_to_decimal_l3578_357878

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 101 in base 2. -/
def binary_101 : List Bool := [true, false, true]

theorem binary_101_to_decimal :
  binary_to_decimal binary_101 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_to_decimal_l3578_357878


namespace NUMINAMATH_CALUDE_factorization_equality_l3578_357885

theorem factorization_equality (x y : ℝ) : 2 * x^2 - 4 * x * y = 2 * x * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3578_357885


namespace NUMINAMATH_CALUDE_certain_number_problem_l3578_357811

theorem certain_number_problem : ∃ x : ℝ, (0.60 * 50 = 0.40 * x + 18) ∧ (x = 30) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3578_357811


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_1001_l3578_357897

-- Define a three-digit number
def three_digit_number (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

-- Define the six-digit number formed by repeating the three-digit number
def six_digit_number (a b c : Nat) : Nat :=
  1000 * (three_digit_number a b c) + (three_digit_number a b c)

-- Theorem statement
theorem six_digit_divisible_by_1001 (a b c : Nat) :
  (a < 10) → (b < 10) → (c < 10) →
  (six_digit_number a b c) % 1001 = 0 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_1001_l3578_357897


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3578_357899

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 67 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute 6 3 = 67 := by sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3578_357899


namespace NUMINAMATH_CALUDE_unique_integer_for_complex_sixth_power_l3578_357851

theorem unique_integer_for_complex_sixth_power : 
  ∃! (n : ℤ), ∃ (m : ℤ), (n + Complex.I) ^ 6 = m := by sorry

end NUMINAMATH_CALUDE_unique_integer_for_complex_sixth_power_l3578_357851


namespace NUMINAMATH_CALUDE_rectangle_area_l3578_357828

theorem rectangle_area (perimeter : ℝ) (length width : ℝ) (h1 : perimeter = 280) 
  (h2 : length / width = 5 / 2) (h3 : perimeter = 2 * (length + width)) 
  (h4 : width * Real.sqrt 2 = length / 2) : length * width = 4000 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3578_357828


namespace NUMINAMATH_CALUDE_symmetric_function_sum_zero_l3578_357830

theorem symmetric_function_sum_zero 
  (v : ℝ → ℝ) 
  (h : ∀ x ∈ Set.Icc (-1.75) 1.75, v (-x) = -v x) : 
  v (-1.75) + v (-0.5) + v 0.5 + v 1.75 = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_sum_zero_l3578_357830


namespace NUMINAMATH_CALUDE_sum_with_twenty_equals_thirty_l3578_357891

theorem sum_with_twenty_equals_thirty (x : ℝ) : 20 + x = 30 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_twenty_equals_thirty_l3578_357891


namespace NUMINAMATH_CALUDE_smallest_solutions_l3578_357822

def is_solution (k : ℕ) : Prop :=
  Real.cos (k^2 + 8^2 : ℝ) ^ 2 = 1

theorem smallest_solutions :
  (∀ k : ℕ, k > 0 ∧ k < 48 → ¬ is_solution k) ∧
  is_solution 48 ∧
  (∀ k : ℕ, k > 48 ∧ k < 53 → ¬ is_solution k) ∧
  is_solution 53 :=
sorry

end NUMINAMATH_CALUDE_smallest_solutions_l3578_357822


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_sum_l3578_357845

theorem seventh_root_of_unity_sum (z : ℂ) :
  z ^ 7 = 1 ∧ z ≠ 1 →
  z + z^2 + z^4 = (-1 + Complex.I * Real.sqrt 11) / 2 ∨
  z + z^2 + z^4 = (-1 - Complex.I * Real.sqrt 11) / 2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_sum_l3578_357845


namespace NUMINAMATH_CALUDE_fold_and_cut_square_l3578_357823

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a straight line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Represents a folding operation -/
def Fold := Point → Point

/-- Checks if a line intersects all four 1x1 squares in a 2x2 square -/
def intersectsAllSquares (l : Line) : Prop :=
  ∃ (p1 p2 p3 p4 : Point),
    p1.x + p1.y = 1 ∧ 
    p2.x + p2.y = 3 ∧ 
    p3.x - p3.y = 1 ∧ 
    p4.x - p4.y = -1 ∧
    l.a * p1.x + l.b * p1.y + l.c = 0 ∧
    l.a * p2.x + l.b * p2.y + l.c = 0 ∧
    l.a * p3.x + l.b * p3.y + l.c = 0 ∧
    l.a * p4.x + l.b * p4.y + l.c = 0

/-- The main theorem stating that it's possible to fold and cut a 2x2 square into four 1x1 squares -/
theorem fold_and_cut_square : 
  ∃ (f1 f2 : Fold) (l : Line),
    intersectsAllSquares l :=
sorry

end NUMINAMATH_CALUDE_fold_and_cut_square_l3578_357823


namespace NUMINAMATH_CALUDE_function_composition_equality_l3578_357882

theorem function_composition_equality (f g : ℝ → ℝ) (b : ℝ) : 
  (∀ x, f x = x / 6 + 2) → 
  (∀ x, g x = 5 - 2 * x) → 
  f (g b) = 4 → 
  b = -7 / 2 := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3578_357882


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3578_357832

theorem regular_polygon_sides (n : ℕ) (angle_OAB : ℝ) : 
  n > 0 → 
  angle_OAB = 72 → 
  (360 : ℝ) / angle_OAB = n → 
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3578_357832


namespace NUMINAMATH_CALUDE_partition_product_property_l3578_357814

theorem partition_product_property (S : Finset ℕ) (h : S = Finset.range (3^5 - 2) ∪ {3^5}) :
  ∀ (A B : Finset ℕ), A ∪ B = S → A ∩ B = ∅ →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a * b = c) :=
by sorry

end NUMINAMATH_CALUDE_partition_product_property_l3578_357814


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l3578_357890

theorem fraction_equals_zero (x : ℝ) : 
  (x - 5) / (4 * x^2 - 1) = 0 ↔ x = 5 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l3578_357890


namespace NUMINAMATH_CALUDE_positive_real_solution_floor_product_l3578_357801

theorem positive_real_solution_floor_product (x : ℝ) : 
  x > 0 → x * ⌊x⌋ = 72 → x = 9 := by sorry

end NUMINAMATH_CALUDE_positive_real_solution_floor_product_l3578_357801


namespace NUMINAMATH_CALUDE_one_basket_of_peaches_l3578_357873

def basket_count (red_peaches green_peaches total_peaches : ℕ) : ℕ :=
  if red_peaches + green_peaches = total_peaches then 1 else 0

theorem one_basket_of_peaches (red_peaches green_peaches total_peaches : ℕ) 
  (h1 : red_peaches = 7)
  (h2 : green_peaches = 3)
  (h3 : total_peaches = 10) :
  basket_count red_peaches green_peaches total_peaches = 1 := by
sorry

end NUMINAMATH_CALUDE_one_basket_of_peaches_l3578_357873


namespace NUMINAMATH_CALUDE_board_intersection_area_l3578_357829

/-- The area of intersection of two rectangular boards crossing at a 45-degree angle -/
theorem board_intersection_area (width1 width2 : ℝ) (angle : ℝ) :
  width1 = 5 →
  width2 = 7 →
  angle = 45 →
  (width1 * width2 : ℝ) = 35 :=
by sorry

end NUMINAMATH_CALUDE_board_intersection_area_l3578_357829


namespace NUMINAMATH_CALUDE_games_to_sell_l3578_357853

def playstation_cost : ℝ := 500
def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def game_price : ℝ := 7.5

theorem games_to_sell : 
  ⌈(playstation_cost - (birthday_money + christmas_money)) / game_price⌉ = 20 := by sorry

end NUMINAMATH_CALUDE_games_to_sell_l3578_357853


namespace NUMINAMATH_CALUDE_complex_number_properties_l3578_357846

theorem complex_number_properties (z : ℂ) (h : z = 1 + I) : 
  (Complex.abs z = Real.sqrt 2) ∧ 
  (z ≠ 1 - I) ∧
  (z.im ≠ 1) ∧
  (0 < z.re ∧ 0 < z.im) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3578_357846


namespace NUMINAMATH_CALUDE_lindas_furniture_fraction_l3578_357893

/-- Given Linda's original savings and the cost of a TV, prove the fraction spent on furniture. -/
theorem lindas_furniture_fraction (original_savings : ℚ) (tv_cost : ℚ) 
  (h1 : original_savings = 600)
  (h2 : tv_cost = 300) :
  (original_savings - tv_cost) / original_savings = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_lindas_furniture_fraction_l3578_357893


namespace NUMINAMATH_CALUDE_closest_to_division_l3578_357847

def options : List ℝ := [0.2, 2, 20, 200, 2000]

theorem closest_to_division (x y : ℝ) (h1 : y ≠ 0) :
  ∃ z ∈ options, ∀ w ∈ options, |x / y - z| ≤ |x / y - w| :=
sorry

end NUMINAMATH_CALUDE_closest_to_division_l3578_357847


namespace NUMINAMATH_CALUDE_kate_red_balloons_l3578_357840

/-- Given Kate's balloon scenario, prove she initially had 2 red balloons -/
theorem kate_red_balloons (R : ℕ) : 
  (R + 2) / (R + 8 : ℚ) = 2/5 → R = 2 := by
  sorry

end NUMINAMATH_CALUDE_kate_red_balloons_l3578_357840


namespace NUMINAMATH_CALUDE_panda_equation_l3578_357870

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Condition that all digits are distinct -/
def all_distinct (a b c d e : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

/-- Convert a two-digit number to a natural number -/
def to_nat (tens units : Digit) : ℕ :=
  10 * tens.val + units.val

/-- Convert a three-digit number to a natural number -/
def to_nat_3 (hundreds tens units : Digit) : ℕ :=
  100 * hundreds.val + 10 * tens.val + units.val

theorem panda_equation (tuan yuan da xiong mao : Digit)
  (h_distinct : all_distinct tuan yuan da xiong mao)
  (h_eq : to_nat tuan tuan * to_nat yuan yuan = to_nat_3 da xiong mao) :
  da.val + xiong.val + mao.val = 23 := by
  sorry

end NUMINAMATH_CALUDE_panda_equation_l3578_357870


namespace NUMINAMATH_CALUDE_polygon_sides_l3578_357880

theorem polygon_sides (n : ℕ) (angle_sum : ℝ) (excluded_angle : ℝ) :
  angle_sum = 2970 ∧
  angle_sum = (n - 2) * 180 - 2 * excluded_angle ∧
  excluded_angle > 0 ∧
  excluded_angle < 180 →
  n = 19 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l3578_357880


namespace NUMINAMATH_CALUDE_heidi_painting_rate_l3578_357867

theorem heidi_painting_rate (total_time minutes : ℕ) (fraction : ℚ) : 
  (total_time = 30) → (minutes = 10) → (fraction = 1 / 3) →
  (fraction = (minutes : ℚ) / total_time) :=
by sorry

end NUMINAMATH_CALUDE_heidi_painting_rate_l3578_357867


namespace NUMINAMATH_CALUDE_largest_eight_digit_even_digits_proof_l3578_357895

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  10000000 ≤ n ∧ n ≤ 99999999

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k : Nat, n / (10^k) % 10 = d

def largest_eight_digit_with_even_digits : Nat := 99986420

theorem largest_eight_digit_even_digits_proof :
  is_eight_digit largest_eight_digit_with_even_digits ∧
  contains_all_even_digits largest_eight_digit_with_even_digits ∧
  ∀ m : Nat, is_eight_digit m ∧ contains_all_even_digits m →
    m ≤ largest_eight_digit_with_even_digits :=
by sorry

end NUMINAMATH_CALUDE_largest_eight_digit_even_digits_proof_l3578_357895


namespace NUMINAMATH_CALUDE_problem_solution_l3578_357861

theorem problem_solution (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 1) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3578_357861


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3578_357833

/-- The complex number i -/
noncomputable def i : ℂ := Complex.I

/-- Proof that (1+i)/(1-i) = i -/
theorem complex_fraction_simplification : (1 + i) / (1 - i) = i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3578_357833


namespace NUMINAMATH_CALUDE_ratio_of_amounts_l3578_357827

theorem ratio_of_amounts (total_amount : ℕ) (r_amount : ℕ) 
  (h1 : total_amount = 8000)
  (h2 : r_amount = 3200) :
  r_amount / (total_amount - r_amount) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_amounts_l3578_357827


namespace NUMINAMATH_CALUDE_infinite_sums_equalities_l3578_357858

/-- Given a right triangle ΔAB₀C₀ with right angle at B₀ and angle α at A,
    with perpendiculars drawn as described in the problem -/
structure RightTriangleWithPerpendiculars (α : ℝ) :=
  (A B₀ C₀ : ℝ × ℝ)
  (is_right_angle : (B₀.1 - A.1) * (C₀.1 - A.1) + (B₀.2 - A.2) * (C₀.2 - A.2) = 0)
  (angle_α : Real.cos α = (B₀.1 - A.1) / Real.sqrt ((B₀.1 - A.1)^2 + (B₀.2 - A.2)^2))
  (B : ℕ → ℝ × ℝ)
  (C : ℕ → ℝ × ℝ)

/-- The theorem stating the equalities for the infinite sums -/
theorem infinite_sums_equalities {α : ℝ} (t : RightTriangleWithPerpendiculars α) :
  (∑' i, Real.sqrt ((t.B i).1 - (t.C i).1)^2 + ((t.B i).2 - (t.C i).2)^2) = 
    Real.sqrt ((t.A.1 - t.C₀.1)^2 + (t.A.2 - t.C₀.2)^2) / Real.sin α ∧
  (∑' i, Real.sqrt ((t.A.1 - (t.B i).1)^2 + (t.A.2 - (t.B i).2)^2)) = 
    Real.sqrt ((t.A.1 - t.B₀.1)^2 + (t.A.2 - t.B₀.2)^2) / Real.sin α^2 ∧
  (∑' i, Real.sqrt ((t.A.1 - (t.C i).1)^2 + (t.A.2 - (t.C i).2)^2)) = 
    Real.sqrt ((t.A.1 - t.C₀.1)^2 + (t.A.2 - t.C₀.2)^2) / Real.sin α^2 ∧
  (∑' i, Real.sqrt (((t.C (i+1)).1 - (t.B i).1)^2 + ((t.C (i+1)).2 - (t.B i).2)^2)) = 
    Real.sqrt ((t.A.1 - t.B₀.1)^2 + (t.A.2 - t.B₀.2)^2) / Real.sin α :=
sorry

end NUMINAMATH_CALUDE_infinite_sums_equalities_l3578_357858


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_square_l3578_357838

theorem cubic_sum_over_product_square (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (sum_zero : x + y + z = 0) (sum_prod_nonzero : x*y + x*z + y*z ≠ 0) :
  (x^3 + y^3 + z^3) / (x*y*z * (x*y + x*z + y*z)^2) = 3 / (x^2 + x*y + y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_square_l3578_357838


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l3578_357888

theorem smallest_number_of_eggs (total_eggs : ℕ) (containers : ℕ) : 
  total_eggs > 130 →
  total_eggs = 15 * containers - 3 →
  (∀ n : ℕ, n > 130 ∧ ∃ m : ℕ, n = 15 * m - 3 → n ≥ total_eggs) →
  total_eggs = 132 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l3578_357888


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3578_357856

-- Define the quadratic equation and its roots
def quadratic_eq (p q x : ℝ) := x^2 + p*x + q = 0

theorem quadratic_roots_properties (p q : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic_eq p q x₁) (h₂ : quadratic_eq p q x₂) (h₃ : x₁ ≠ x₂) :
  (1/x₁ + 1/x₂ = -p/q) ∧
  (1/x₁^2 + 1/x₂^2 = (p^2 - 2*q)/q^2) ∧
  (x₁^3 + x₂^3 = -p^3 + 3*p*q) ∧
  (1/(x₁ + p)^2 + 1/(x₂ + p)^2 = (p^2 - 2*q)/q^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3578_357856


namespace NUMINAMATH_CALUDE_function_value_alternation_l3578_357808

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) where a, b, α, β are non-zero real numbers,
    if f(2013) = -1, then f(2014) = 1 -/
theorem function_value_alternation (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 2013 = -1 → f 2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_alternation_l3578_357808


namespace NUMINAMATH_CALUDE_unique_solution_for_g_l3578_357820

/-- Given functions f and g where g(x) = 4f⁻¹(x) and f(x) = 30 / (x + 4),
    prove that the unique value of x satisfying g(x) = 20 is 10/3 -/
theorem unique_solution_for_g (f g : ℝ → ℝ) 
    (h1 : ∀ x, g x = 4 * (f⁻¹ x)) 
    (h2 : ∀ x, f x = 30 / (x + 4)) : 
    ∃! x, g x = 20 ∧ x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_g_l3578_357820


namespace NUMINAMATH_CALUDE_system_solution_unique_l3578_357874

theorem system_solution_unique : 
  ∃! (x y : ℝ), (x + 3 * y = 1) ∧ (3 * x + y = -5) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3578_357874


namespace NUMINAMATH_CALUDE_sodium_hydroxide_combined_l3578_357884

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the reaction between acetic acid and sodium hydroxide -/
structure Reaction where
  acetic_acid : Moles
  sodium_hydroxide : Moles
  sodium_acetate : Moles

/-- The reaction occurs in a 1:1 molar ratio -/
axiom reaction_ratio (r : Reaction) : r.acetic_acid = r.sodium_hydroxide

/-- The number of moles of sodium acetate formed equals the number of moles of acetic acid used -/
axiom sodium_acetate_formation (r : Reaction) : r.sodium_acetate = r.acetic_acid

theorem sodium_hydroxide_combined (r : Reaction) :
  r.sodium_hydroxide = r.sodium_acetate :=
by sorry

end NUMINAMATH_CALUDE_sodium_hydroxide_combined_l3578_357884


namespace NUMINAMATH_CALUDE_mixture_alcohol_percentage_l3578_357850

/-- Represents the properties of an alcohol solution -/
structure Solution where
  volume : ℝ
  alcoholPercentage : ℝ

/-- Calculates the volume of alcohol in a solution -/
def alcoholVolume (s : Solution) : ℝ :=
  s.volume * s.alcoholPercentage

/-- Theorem: Adding 50 mL of 30% alcohol solution to 200 mL of 10% alcohol solution results in 14% alcohol solution -/
theorem mixture_alcohol_percentage 
  (x : Solution) 
  (y : Solution) 
  (h1 : x.volume = 200)
  (h2 : x.alcoholPercentage = 0.1)
  (h3 : y.volume = 50)
  (h4 : y.alcoholPercentage = 0.3) :
  let finalSolution : Solution := {
    volume := x.volume + y.volume,
    alcoholPercentage := (alcoholVolume x + alcoholVolume y) / (x.volume + y.volume)
  }
  finalSolution.alcoholPercentage = 0.14 := by
  sorry

#check mixture_alcohol_percentage

end NUMINAMATH_CALUDE_mixture_alcohol_percentage_l3578_357850


namespace NUMINAMATH_CALUDE_wharf_length_l3578_357852

/-- The length of the wharf in meters -/
def L_wharf : ℝ := 64

/-- The average speed in meters per second -/
def V_avg : ℝ := 2

/-- The travel time in seconds -/
def T_travel : ℝ := 16

/-- Theorem: The length of the wharf is 64 meters -/
theorem wharf_length : L_wharf = 2 * V_avg * T_travel := by
  sorry

end NUMINAMATH_CALUDE_wharf_length_l3578_357852


namespace NUMINAMATH_CALUDE_intersection_value_l3578_357869

theorem intersection_value (a : ℝ) : 
  let A := {x : ℝ | x^2 - 4 ≤ 0}
  let B := {x : ℝ | 2*x + a ≤ 0}
  (A ∩ B = {x : ℝ | -2 ≤ x ∧ x ≤ 1}) → a = -4 := by
sorry

end NUMINAMATH_CALUDE_intersection_value_l3578_357869


namespace NUMINAMATH_CALUDE_negation_of_x_squared_plus_two_gt_zero_is_false_l3578_357855

theorem negation_of_x_squared_plus_two_gt_zero_is_false :
  ¬(∃ x : ℝ, x^2 + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_x_squared_plus_two_gt_zero_is_false_l3578_357855


namespace NUMINAMATH_CALUDE_cistern_emptying_l3578_357810

/-- Represents the rate at which a pipe can empty a cistern -/
structure EmptyingRate where
  fraction : ℚ
  time : ℕ

/-- Represents the operation of pipes emptying a cistern -/
def empty_cistern (pipe1 : EmptyingRate) (pipe2 : EmptyingRate) (time1 : ℕ) (time2 : ℕ) : ℚ :=
  sorry

theorem cistern_emptying :
  let pipe1 : EmptyingRate := ⟨3/4, 12⟩
  let pipe2 : EmptyingRate := ⟨1/2, 15⟩
  empty_cistern pipe1 pipe2 4 10 = 7/12 :=
by sorry

end NUMINAMATH_CALUDE_cistern_emptying_l3578_357810


namespace NUMINAMATH_CALUDE_split_meal_cost_l3578_357860

def meal_contribution (total_price : ℚ) (coupon_value : ℚ) (num_people : ℕ) : ℚ :=
  (total_price - coupon_value) / num_people

theorem split_meal_cost :
  meal_contribution 67 4 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_split_meal_cost_l3578_357860


namespace NUMINAMATH_CALUDE_iron_cotton_mass_equality_l3578_357844

-- Define the conversion factor from kilograms to grams
def kgToGrams : ℝ → ℝ := (· * 1000)

-- Define the masses in their given units
def ironMassKg : ℝ := 5
def cottonMassG : ℝ := 5000

-- Theorem stating that the masses are equal
theorem iron_cotton_mass_equality :
  kgToGrams ironMassKg = cottonMassG := by sorry

end NUMINAMATH_CALUDE_iron_cotton_mass_equality_l3578_357844


namespace NUMINAMATH_CALUDE_pink_roses_count_is_300_l3578_357825

/-- Calculates the number of pink roses in Mrs. Dawson's garden -/
def pink_roses_count : ℕ :=
  let total_rows : ℕ := 30
  let roses_per_row : ℕ := 50
  let red_roses : ℕ := (2 * roses_per_row) / 5
  let blue_roses : ℕ := 1
  let remaining_after_blue : ℕ := roses_per_row - red_roses - blue_roses
  let white_roses : ℕ := remaining_after_blue / 4
  let yellow_roses : ℕ := 2
  let remaining_after_yellow : ℕ := remaining_after_blue - white_roses - yellow_roses
  let purple_roses : ℕ := (3 * remaining_after_yellow) / 8
  let orange_roses : ℕ := 3
  let pink_roses_per_row : ℕ := remaining_after_yellow - purple_roses - orange_roses
  total_rows * pink_roses_per_row

theorem pink_roses_count_is_300 : pink_roses_count = 300 := by
  sorry

end NUMINAMATH_CALUDE_pink_roses_count_is_300_l3578_357825


namespace NUMINAMATH_CALUDE_seven_balls_two_boxes_at_least_two_in_first_l3578_357859

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes -/
def total_distributions (n : ℕ) : ℕ := 2^n

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes
    where the first box contains exactly k balls -/
def distributions_with_k_in_first_box (n k : ℕ) : ℕ := n.choose k

theorem seven_balls_two_boxes_at_least_two_in_first : 
  total_distributions 7 - (distributions_with_k_in_first_box 7 0 + distributions_with_k_in_first_box 7 1) = 120 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_two_boxes_at_least_two_in_first_l3578_357859


namespace NUMINAMATH_CALUDE_expression_value_l3578_357854

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (3 * x^3 + 4 * y^3) / 9 = 37 + 4/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3578_357854


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3578_357816

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let L' := 1.3 * L
  let B' := 0.75 * B
  let A := L * B
  let A' := L' * B'
  A' / A = 0.975 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3578_357816


namespace NUMINAMATH_CALUDE_like_terms_imply_x_power_y_equals_nine_l3578_357826

theorem like_terms_imply_x_power_y_equals_nine (a b x y : ℝ) 
  (h : ∃ (k : ℝ), 3 * a^(x+7) * b^4 = k * (-a^4 * b^(2*y))) : 
  x^y = 9 := by sorry

end NUMINAMATH_CALUDE_like_terms_imply_x_power_y_equals_nine_l3578_357826


namespace NUMINAMATH_CALUDE_greg_earnings_l3578_357868

/-- Calculates the earnings for dog walking based on the given parameters -/
def dog_walking_earnings (base_charge : ℕ) (per_minute_charge : ℕ) 
  (dogs : List (ℕ × ℕ)) : ℕ :=
  dogs.foldl (fun acc (num_dogs, minutes) => 
    acc + num_dogs * base_charge + num_dogs * minutes * per_minute_charge) 0

/-- Theorem stating Greg's earnings from his dog walking business -/
theorem greg_earnings : 
  dog_walking_earnings 20 1 [(1, 10), (2, 7), (3, 9)] = 171 := by
  sorry

#eval dog_walking_earnings 20 1 [(1, 10), (2, 7), (3, 9)]

end NUMINAMATH_CALUDE_greg_earnings_l3578_357868


namespace NUMINAMATH_CALUDE_probability_of_no_growth_pie_l3578_357896

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def pies_given : ℕ := 3

def probability_no_growth_pie : ℚ := 7/10

theorem probability_of_no_growth_pie :
  (1 - (Nat.choose (total_pies - growth_pies) (pies_given - growth_pies) : ℚ) / 
   (Nat.choose total_pies pies_given : ℚ)) = probability_no_growth_pie :=
sorry

end NUMINAMATH_CALUDE_probability_of_no_growth_pie_l3578_357896


namespace NUMINAMATH_CALUDE_fill_time_is_100_l3578_357898

/-- Represents the water filling system with three pipes and a tank -/
structure WaterSystem where
  tankCapacity : ℕ
  pipeARate : ℕ
  pipeBRate : ℕ
  pipeCRate : ℕ
  pipeATime : ℕ
  pipeBTime : ℕ
  pipeCTime : ℕ

/-- Calculates the time required to fill the tank -/
def fillTime (sys : WaterSystem) : ℕ :=
  let cycleAmount := sys.pipeARate * sys.pipeATime + sys.pipeBRate * sys.pipeBTime - sys.pipeCRate * sys.pipeCTime
  let cycles := (sys.tankCapacity + cycleAmount - 1) / cycleAmount
  cycles * (sys.pipeATime + sys.pipeBTime + sys.pipeCTime)

/-- Theorem stating that the fill time for the given system is 100 minutes -/
theorem fill_time_is_100 (sys : WaterSystem) 
  (h1 : sys.tankCapacity = 5000)
  (h2 : sys.pipeARate = 200)
  (h3 : sys.pipeBRate = 50)
  (h4 : sys.pipeCRate = 25)
  (h5 : sys.pipeATime = 1)
  (h6 : sys.pipeBTime = 2)
  (h7 : sys.pipeCTime = 2) :
  fillTime sys = 100 := by
  sorry

#eval fillTime { tankCapacity := 5000, pipeARate := 200, pipeBRate := 50, pipeCRate := 25, 
                 pipeATime := 1, pipeBTime := 2, pipeCTime := 2 }

end NUMINAMATH_CALUDE_fill_time_is_100_l3578_357898


namespace NUMINAMATH_CALUDE_king_then_ace_probability_l3578_357836

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of Kings in a standard deck -/
def numKings : ℕ := 4

/-- The number of Aces in a standard deck -/
def numAces : ℕ := 4

/-- The probability of drawing a King first and an Ace second from a standard deck -/
def probKingThenAce : ℚ := (numKings : ℚ) / standardDeckSize * numAces / (standardDeckSize - 1)

theorem king_then_ace_probability :
  probKingThenAce = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_king_then_ace_probability_l3578_357836


namespace NUMINAMATH_CALUDE_exist_same_color_perfect_square_diff_l3578_357877

/-- A coloring of integers using three colors. -/
def Coloring := ℤ → Fin 3

/-- Predicate to check if a number is a perfect square. -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

/-- Main theorem: For any coloring of integers using three colors,
    there exist two different integers of the same color
    whose difference is a perfect square. -/
theorem exist_same_color_perfect_square_diff (c : Coloring) :
  ∃ a b : ℤ, a ≠ b ∧ c a = c b ∧ is_perfect_square (a - b) := by
  sorry


end NUMINAMATH_CALUDE_exist_same_color_perfect_square_diff_l3578_357877


namespace NUMINAMATH_CALUDE_weight_loss_program_result_l3578_357819

/-- Calculates the final weight after a weight loss program -/
def finalWeight (initialWeight : ℕ) (weeklyLoss1 weeklyLoss2 : ℕ) (weeks1 weeks2 : ℕ) : ℕ :=
  initialWeight - (weeklyLoss1 * weeks1 + weeklyLoss2 * weeks2)

/-- Proves that the weight loss program results in the correct final weight -/
theorem weight_loss_program_result :
  finalWeight 250 3 2 4 8 = 222 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_program_result_l3578_357819


namespace NUMINAMATH_CALUDE_rectangle_ratio_l3578_357821

/-- Proves that a rectangle with area 100 m² and length 20 m has a length-to-width ratio of 4:1 -/
theorem rectangle_ratio (area : ℝ) (length : ℝ) (width : ℝ) 
  (h_area : area = 100) 
  (h_length : length = 20) 
  (h_rect : area = length * width) : 
  length / width = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l3578_357821


namespace NUMINAMATH_CALUDE_line_y_intercept_l3578_357815

/-- A straight line in the xy-plane with slope 2 and passing through (259, 520) has y-intercept 2 -/
theorem line_y_intercept : 
  ∀ (f : ℝ → ℝ), 
  (∀ x y, f y = 2 * x + f 0) →  -- slope is 2
  f 520 = 2 * 259 + f 0 →      -- point (259, 520) lies on the line
  f 0 = 2 := by               -- y-intercept is 2
sorry

end NUMINAMATH_CALUDE_line_y_intercept_l3578_357815


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l3578_357894

theorem greatest_integer_problem : 
  ∃ (m : ℕ), m < 150 ∧ 
  (∃ (k : ℕ), m = 9 * k - 2) ∧ 
  (∃ (j : ℕ), m = 11 * j - 4) ∧
  (∀ (n : ℕ), n < 150 → 
    (∃ (k' : ℕ), n = 9 * k' - 2) → 
    (∃ (j' : ℕ), n = 11 * j' - 4) → 
    n ≤ m) ∧
  m = 142 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l3578_357894


namespace NUMINAMATH_CALUDE_computer_price_increase_l3578_357879

/-- Given a computer with original price x dollars, where 2x = 540,
    prove that after a 30% increase, the new price is 351 dollars. -/
theorem computer_price_increase (x : ℝ) (h1 : 2 * x = 540) :
  x * 1.3 = 351 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l3578_357879


namespace NUMINAMATH_CALUDE_age_difference_proof_l3578_357857

theorem age_difference_proof (patrick_age michael_age monica_age : ℕ) :
  patrick_age * 5 = michael_age * 3 →
  michael_age * 4 = monica_age * 3 →
  patrick_age + michael_age + monica_age = 88 →
  monica_age - patrick_age = 22 := by
sorry


end NUMINAMATH_CALUDE_age_difference_proof_l3578_357857


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3578_357812

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_property (f : ℝ → ℝ) (h_quad : is_quadratic f)
  (h_cond : ∀ a b : ℝ, a ≠ b → f a = f b → f (a^2 - 6*b - 1) = f (b^2 + 8)) :
  f 2 = f 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3578_357812


namespace NUMINAMATH_CALUDE_candy_seller_problem_l3578_357862

/-- The number of candies the seller had initially, given the number of clowns,
    children, candies per person, and candies left after selling. -/
def initial_candies (clowns children candies_per_person candies_left : ℕ) : ℕ :=
  (clowns + children) * candies_per_person + candies_left

/-- Theorem stating that given the specific conditions in the problem,
    the initial number of candies is 700. -/
theorem candy_seller_problem :
  initial_candies 4 30 20 20 = 700 := by
  sorry

end NUMINAMATH_CALUDE_candy_seller_problem_l3578_357862


namespace NUMINAMATH_CALUDE_choir_size_l3578_357807

/-- Represents a choir split into three groups -/
structure Choir :=
  (group1 : ℕ)
  (group2 : ℕ)
  (group3 : ℕ)

/-- The total number of members in the choir -/
def Choir.total (c : Choir) : ℕ :=
  c.group1 + c.group2 + c.group3

/-- Theorem stating that a choir with the given group sizes has 70 members -/
theorem choir_size (c : Choir) 
  (h1 : c.group1 = 25) 
  (h2 : c.group2 = 30) 
  (h3 : c.group3 = 15) : 
  c.total = 70 := by
  sorry

/-- The specific choir instance from the problem -/
def problem_choir : Choir :=
  { group1 := 25
    group2 := 30
    group3 := 15 }

#eval Choir.total problem_choir

end NUMINAMATH_CALUDE_choir_size_l3578_357807


namespace NUMINAMATH_CALUDE_crossword_solvable_l3578_357805

-- Define the structure of a crossword puzzle
structure Crossword :=
  (grid : List (List Char))
  (vertical_clues : List String)
  (horizontal_clues : List String)

-- Define the words for the crossword
def words : List String := ["счет", "евро", "доллар", "вклад", "золото", "ломбард", "обмен", "система"]

-- Define the clues for the crossword
def vertical_clues : List String := [
  "What a bank opens for a person who wants to become its client",
  "This currency is used in Italy and other places",
  "One of the most well-known international currencies, accepted for payment in many countries",
  "The way to store and gradually increase family money in the bank",
  "A precious metal, whose reserves are accounted for by the Bank of Russia",
  "An organization from which you can borrow money and pay a small interest"
]

def horizontal_clues : List String := [
  "To pay abroad, you need to carry out ... of currency",
  "In Russia, there is a multi-level banking ...: the Central Bank of the Russian Federation, banks with a universal license, and with a basic one",
  "The place where you can take jewelry and get a loan for it"
]

-- Define the function to check if the crossword is valid
def is_valid_crossword (c : Crossword) : Prop :=
  c.vertical_clues.length = 6 ∧
  c.horizontal_clues.length = 3 ∧
  c.grid.all (λ row => row.length = 6) ∧
  c.grid.length = 7

-- Define the theorem to prove
theorem crossword_solvable :
  ∃ (c : Crossword), is_valid_crossword c ∧
    (∀ w ∈ words, w.length ≤ 7) ∧
    (∀ clue ∈ c.vertical_clues ++ c.horizontal_clues, ∃ w ∈ words, clue.length > 0 ∧ w.length > 0) :=
sorry

end NUMINAMATH_CALUDE_crossword_solvable_l3578_357805


namespace NUMINAMATH_CALUDE_expression_evaluation_l3578_357837

theorem expression_evaluation :
  let a : ℚ := -1/2
  (a + 3)^2 + (a + 3)*(a - 3) - 2*a*(3 - a) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3578_357837


namespace NUMINAMATH_CALUDE_intersection_line_circle_l3578_357872

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define a line with slope 1
def line_with_slope_1 (x y b : ℝ) : Prop := y = x + b

-- Define a point on both the circle and the line
def point_on_circle_and_line (x y b : ℝ) : Prop :=
  circle_C x y ∧ line_with_slope_1 x y b

-- Define that a circle passes through the origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂)^2 + (y₁ + y₂)^2 = x₁^2 + y₁^2 + x₂^2 + y₂^2

theorem intersection_line_circle :
  ∃ b : ℝ, b = 1 ∨ b = -4 ∧
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    point_on_circle_and_line x₁ y₁ b ∧
    point_on_circle_and_line x₂ y₂ b ∧
    x₁ ≠ x₂ ∧
    circle_through_origin x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l3578_357872


namespace NUMINAMATH_CALUDE_one_fourth_of_hundred_equals_ten_percent_of_250_l3578_357865

theorem one_fourth_of_hundred_equals_ten_percent_of_250 : 
  (1 / 4 : ℚ) * 100 = (10 / 100 : ℚ) * 250 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_hundred_equals_ten_percent_of_250_l3578_357865


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l3578_357841

/-- Convert a list of binary digits to a natural number -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Convert a natural number to a list of binary digits -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Multiply two binary numbers represented as lists of booleans -/
def binary_multiply (a b : List Bool) : List Bool :=
  nat_to_binary ((binary_to_nat a) * (binary_to_nat b))

theorem binary_multiplication_theorem :
  let a := [true, false, true, true, false, true, true]  -- 1101101₂
  let b := [true, true, true]  -- 111₂
  let product := binary_multiply a b
  binary_to_nat product = 1267 ∧ 
  product = [true, true, false, false, true, true, true, true, false, false, true] :=
by sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l3578_357841


namespace NUMINAMATH_CALUDE_circular_pool_volume_l3578_357835

/-- The volume of a circular cylinder with diameter 80 feet and height 10 feet is approximately 50265.6 cubic feet. -/
theorem circular_pool_volume :
  let diameter : ℝ := 80
  let height : ℝ := 10
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * height
  ∃ ε > 0, |volume - 50265.6| < ε :=
by sorry

end NUMINAMATH_CALUDE_circular_pool_volume_l3578_357835


namespace NUMINAMATH_CALUDE_system_solution_l3578_357849

theorem system_solution : 
  ∃ (x y : ℝ), (6 / (x^2 + y^2) + x^2 * y^2 = 10) ∧ 
               (x^4 + y^4 + 7 * x^2 * y^2 = 81) ∧
               ((x = Real.sqrt 3 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨
                (x = -Real.sqrt 3 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3578_357849


namespace NUMINAMATH_CALUDE_correct_average_l3578_357843

theorem correct_average (n : ℕ) (initial_avg : ℚ) (error1 : ℚ) (error2 : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 16 →
  error2 = 18 →
  (n : ℚ) * initial_avg - error1 + error2 = n * 40.4 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l3578_357843


namespace NUMINAMATH_CALUDE_product_of_fractions_l3578_357800

theorem product_of_fractions : (1 / 3 : ℚ) * (4 / 7 : ℚ) * (5 / 8 : ℚ) = 5 / 42 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3578_357800
