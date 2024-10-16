import Mathlib

namespace NUMINAMATH_CALUDE_intersection_M_N_l1167_116776

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1167_116776


namespace NUMINAMATH_CALUDE_f_negative_expression_l1167_116712

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the function f for x > 0
def f_positive (x : ℝ) : ℝ :=
  x^3 + x + 1

-- Theorem statement
theorem f_negative_expression 
  (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x < 0, f x = -x^3 - x + 1 := by
sorry

end NUMINAMATH_CALUDE_f_negative_expression_l1167_116712


namespace NUMINAMATH_CALUDE_expression_values_l1167_116785

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = -5 ∨ expr = -1 ∨ expr = 1 ∨ expr = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l1167_116785


namespace NUMINAMATH_CALUDE_max_abs_z_l1167_116703

theorem max_abs_z (z : ℂ) (h : Complex.abs (z + 5 - 12*I) = 3) : 
  ∃ (max_abs : ℝ), max_abs = 16 ∧ Complex.abs z ≤ max_abs ∧ 
  ∀ (w : ℂ), Complex.abs (w + 5 - 12*I) = 3 → Complex.abs w ≤ max_abs :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_l1167_116703


namespace NUMINAMATH_CALUDE_reinforcement_is_1900_l1167_116757

/-- Calculates the reinforcement size given initial garrison size, provision duration, and remaining provision duration after reinforcement --/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions_left := initial_garrison * (initial_duration - days_before_reinforcement)
  (provisions_left / remaining_duration) - initial_garrison

/-- The reinforcement size for the given problem --/
def problem_reinforcement : ℕ := calculate_reinforcement 2000 54 15 20

/-- Theorem stating that the reinforcement size for the given problem is 1900 --/
theorem reinforcement_is_1900 : problem_reinforcement = 1900 := by
  sorry

#eval problem_reinforcement

end NUMINAMATH_CALUDE_reinforcement_is_1900_l1167_116757


namespace NUMINAMATH_CALUDE_solve_equation_l1167_116705

theorem solve_equation (y : ℝ) : 
  5 * y^(1/4) - 3 * (y / y^(3/4)) = 9 + y^(1/4) ↔ y = 6561 :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l1167_116705


namespace NUMINAMATH_CALUDE_anne_cleaning_time_l1167_116793

variable (B A C : ℝ)

-- Define the conditions
def condition1 : Prop := B + A + C = 1/4
def condition2 : Prop := B + 2*A + 3*C = 1/3
def condition3 : Prop := B + C = 1/6

-- Theorem statement
theorem anne_cleaning_time 
  (h1 : condition1 B A C) 
  (h2 : condition2 B A C) 
  (h3 : condition3 B C) : 
  1/A = 12 := by
sorry

end NUMINAMATH_CALUDE_anne_cleaning_time_l1167_116793


namespace NUMINAMATH_CALUDE_greenfield_high_school_teachers_l1167_116733

/-- The number of students at Greenfield High School -/
def num_students : ℕ := 900

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- The number of students in each class -/
def students_per_class : ℕ := 25

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- The number of teachers at Greenfield High School -/
def num_teachers : ℕ := 44

theorem greenfield_high_school_teachers :
  (num_students * classes_per_student) / students_per_class / classes_per_teacher = num_teachers := by
  sorry

end NUMINAMATH_CALUDE_greenfield_high_school_teachers_l1167_116733


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1167_116767

/-- Given a quadratic function f(x) = ax^2 - 2x + c where the solution set of f(x) > 0 is {x | x ≠ 1/a},
    this theorem states that the minimum value of f(2) is 0 and when f(2) is minimum,
    the maximum value of m that satisfies f(x) + 4 ≥ m(x-2) for all x > 2 is 2√2. -/
theorem quadratic_function_properties (a c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 - 2*x + c)
  (h2 : ∀ x, f x > 0 ↔ x ≠ 1/a) :
  (∃ (f_min : ℝ → ℝ), (∀ x, f_min x = (1/2) * x^2 - 2*x + 2) ∧ 
   f_min 2 = 0 ∧ 
   (∀ m : ℝ, (∀ x > 2, f_min x + 4 ≥ m * (x - 2)) ↔ m ≤ 2 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1167_116767


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l1167_116760

theorem discounted_price_calculation (original_price discount_percentage : ℝ) 
  (h1 : original_price = 975)
  (h2 : discount_percentage = 20) : 
  original_price * (1 - discount_percentage / 100) = 780 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l1167_116760


namespace NUMINAMATH_CALUDE_janice_purchase_problem_l1167_116795

theorem janice_purchase_problem :
  ∃ (a d b c : ℕ), 
    a + d + b + c = 50 ∧ 
    30 * a + 150 * d + 200 * b + 300 * c = 6000 :=
by sorry

end NUMINAMATH_CALUDE_janice_purchase_problem_l1167_116795


namespace NUMINAMATH_CALUDE_first_triple_winner_lcm_of_prizes_first_triple_winner_is_900_l1167_116770

theorem first_triple_winner (n : ℕ) : 
  (n % 25 = 0 ∧ n % 36 = 0 ∧ n % 45 = 0) → n ≥ 900 :=
by sorry

theorem lcm_of_prizes : Nat.lcm (Nat.lcm 25 36) 45 = 900 :=
by sorry

theorem first_triple_winner_is_900 : 
  900 % 25 = 0 ∧ 900 % 36 = 0 ∧ 900 % 45 = 0 :=
by sorry

end NUMINAMATH_CALUDE_first_triple_winner_lcm_of_prizes_first_triple_winner_is_900_l1167_116770


namespace NUMINAMATH_CALUDE_find_k_value_l1167_116778

theorem find_k_value : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 11)*x - 8 = -(x - 2)*(x - 4) → k = -17 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l1167_116778


namespace NUMINAMATH_CALUDE_rd_sum_formula_count_rd_sum_3883_is_18_count_rd_sum_equal_is_143_l1167_116740

/-- Represents a four-digit positive integer ABCD where A and D are non-zero digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h1 : a > 0 ∧ a < 10
  h2 : b < 10
  h3 : c < 10
  h4 : d > 0 ∧ d < 10

/-- Calculates the reverse of a four-digit number -/
def reverse (n : FourDigitNumber) : Nat :=
  1000 * n.d + 100 * n.c + 10 * n.b + n.a

/-- Calculates the RD sum of a four-digit number -/
def rdSum (n : FourDigitNumber) : Nat :=
  (1000 * n.a + 100 * n.b + 10 * n.c + n.d) + reverse n

/-- Theorem: The RD sum of ABCD is equal to 1001(A + D) + 110(B + C) -/
theorem rd_sum_formula (n : FourDigitNumber) :
  rdSum n = 1001 * (n.a + n.d) + 110 * (n.b + n.c) := by
  sorry

/-- The number of four-digit integers whose RD sum is 3883 -/
def count_rd_sum_3883 : Nat := 18

/-- Theorem: The number of four-digit integers whose RD sum is 3883 is 18 -/
theorem count_rd_sum_3883_is_18 :
  count_rd_sum_3883 = 18 := by
  sorry

/-- The number of four-digit integers that are equal to the RD sum of a four-digit integer -/
def count_rd_sum_equal : Nat := 143

/-- Theorem: The number of four-digit integers that are equal to the RD sum of a four-digit integer is 143 -/
theorem count_rd_sum_equal_is_143 :
  count_rd_sum_equal = 143 := by
  sorry

end NUMINAMATH_CALUDE_rd_sum_formula_count_rd_sum_3883_is_18_count_rd_sum_equal_is_143_l1167_116740


namespace NUMINAMATH_CALUDE_circle_equation_correct_l1167_116781

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- The equation of a circle in polar coordinates -/
def circleEquation (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.r = 2 * Real.cos (p.θ - c.center.θ)

theorem circle_equation_correct (c : PolarCircle) :
  c.center.r = 1 ∧ c.center.θ = π/4 ∧ c.radius = 1 →
  ∀ p : PolarPoint, circleEquation c p ↔ (p.r * Real.cos p.θ - c.center.r)^2 + (p.r * Real.sin p.θ - c.center.r)^2 = c.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l1167_116781


namespace NUMINAMATH_CALUDE_fraction_comparison_l1167_116752

theorem fraction_comparison : (1 / (Real.sqrt 5 - 2)) < (1 / (Real.sqrt 6 - Real.sqrt 5)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1167_116752


namespace NUMINAMATH_CALUDE_sqrt_neg_two_squared_l1167_116761

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_two_squared_l1167_116761


namespace NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l1167_116700

/-- An arithmetic sequence with first term 3 and ninth term 27 has its thirtieth term equal to 90 -/
theorem arithmetic_sequence_30th_term : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 3 →                            -- first term is 3
  a 9 = 27 →                           -- ninth term is 27
  a 30 = 90 :=                         -- thirtieth term is 90
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l1167_116700


namespace NUMINAMATH_CALUDE_third_term_5_4_l1167_116729

/-- Decomposition function that returns the n-th term in the decomposition of m^k -/
def decomposition (m : ℕ) (k : ℕ) (n : ℕ) : ℕ :=
  2 * m * k - 1 + 2 * (n - 1)

/-- Theorem stating that the third term in the decomposition of 5^4 is 125 -/
theorem third_term_5_4 : decomposition 5 4 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_third_term_5_4_l1167_116729


namespace NUMINAMATH_CALUDE_ball_placement_methods_l1167_116724

/-- The number of different balls -/
def n : ℕ := 4

/-- The number of different boxes -/
def m : ℕ := 4

/-- The number of ways to place n different balls into m different boxes without any empty boxes -/
def noEmptyBoxes (n m : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to place n different balls into m different boxes allowing empty boxes -/
def allowEmptyBoxes (n m : ℕ) : ℕ := m ^ n

/-- The number of ways to place n different balls into m different boxes with exactly one box left empty -/
def oneEmptyBox (n m : ℕ) : ℕ := 
  Nat.choose m 1 * Nat.choose n (m - 1) * Nat.factorial (m - 1)

theorem ball_placement_methods :
  noEmptyBoxes n m = 24 ∧
  allowEmptyBoxes n m = 256 ∧
  oneEmptyBox n m = 144 := by sorry

end NUMINAMATH_CALUDE_ball_placement_methods_l1167_116724


namespace NUMINAMATH_CALUDE_circle_radius_comparison_l1167_116745

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of three circles being pairwise disjoint and collinear
def pairwiseDisjointCollinear (c1 c2 c3 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let (x3, y3) := c3.center
  (x2 - x1) * (y3 - y1) = (y2 - y1) * (x3 - x1) ∧ 
  (c1.radius + c2.radius < Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)) ∧
  (c2.radius + c3.radius < Real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)) ∧
  (c3.radius + c1.radius < Real.sqrt ((x3 - x1)^2 + (y3 - y1)^2))

-- Define the property of a circle touching three other circles externally
def touchesExternally (c : Circle) (c1 c2 c3 : Circle) : Prop :=
  let (x, y) := c.center
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let (x3, y3) := c3.center
  (Real.sqrt ((x - x1)^2 + (y - y1)^2) = c.radius + c1.radius) ∧
  (Real.sqrt ((x - x2)^2 + (y - y2)^2) = c.radius + c2.radius) ∧
  (Real.sqrt ((x - x3)^2 + (y - y3)^2) = c.radius + c3.radius)

-- The main theorem
theorem circle_radius_comparison 
  (c1 c2 c3 c : Circle) 
  (h1 : pairwiseDisjointCollinear c1 c2 c3)
  (h2 : touchesExternally c c1 c2 c3) :
  c.radius > c2.radius :=
sorry

end NUMINAMATH_CALUDE_circle_radius_comparison_l1167_116745


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l1167_116774

theorem smallest_n_divisibility (x y z : ℕ+) 
  (h1 : x ∣ y^3) 
  (h2 : y ∣ z^3) 
  (h3 : z ∣ x^3) : 
  (∀ n : ℕ, n ≥ 13 → (x*y*z : ℕ) ∣ (x + y + z : ℕ)^n) ∧ 
  (∀ m : ℕ, m < 13 → ∃ a b c : ℕ+, 
    a ∣ b^3 ∧ b ∣ c^3 ∧ c ∣ a^3 ∧ ¬((a*b*c : ℕ) ∣ (a + b + c : ℕ)^m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l1167_116774


namespace NUMINAMATH_CALUDE_multiples_properties_l1167_116726

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 12 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ (∃ p : ℤ, a - b = 4 * p) :=
by sorry

end NUMINAMATH_CALUDE_multiples_properties_l1167_116726


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l1167_116730

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l1167_116730


namespace NUMINAMATH_CALUDE_x_eq_one_iff_quadratic_eq_zero_l1167_116734

theorem x_eq_one_iff_quadratic_eq_zero : ∀ x : ℝ, x = 1 ↔ x^2 - 2*x + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_x_eq_one_iff_quadratic_eq_zero_l1167_116734


namespace NUMINAMATH_CALUDE_isosceles_triangle_min_ratio_l1167_116711

/-- The minimum value of (2a + b) / a for an isosceles triangle with two equal sides of length a and base of length b is 3 -/
theorem isosceles_triangle_min_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a ≥ b) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x ≥ y → (2 * x + y) / x ≥ (2 * a + b) / a ∧ (2 * a + b) / a = 3 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_min_ratio_l1167_116711


namespace NUMINAMATH_CALUDE_real_gdp_change_omega_l1167_116780

/-- Represents the production and price data for Omega in a given year -/
structure YearData where
  vegetable_production : ℕ
  fruit_production : ℕ
  vegetable_price : ℕ
  fruit_price : ℕ

/-- Calculates the nominal GDP for a given year -/
def nominalGDP (data : YearData) : ℕ :=
  data.vegetable_production * data.vegetable_price + data.fruit_production * data.fruit_price

/-- Calculates the real GDP for a given year using base year prices -/
def realGDP (data : YearData) (base : YearData) : ℕ :=
  data.vegetable_production * base.vegetable_price + data.fruit_production * base.fruit_price

/-- Calculates the percentage change in GDP -/
def percentageChange (old : ℕ) (new : ℕ) : ℚ :=
  100 * (new - old : ℚ) / old

/-- The main theorem stating the percentage change in real GDP -/
theorem real_gdp_change_omega :
  let data2014 : YearData := {
    vegetable_production := 1200,
    fruit_production := 750,
    vegetable_price := 90000,
    fruit_price := 75000
  }
  let data2015 : YearData := {
    vegetable_production := 900,
    fruit_production := 900,
    vegetable_price := 100000,
    fruit_price := 70000
  }
  let nominal2014 := nominalGDP data2014
  let real2015 := realGDP data2015 data2014
  let change := percentageChange nominal2014 real2015
  ∃ ε > 0, |change + 9.59| < ε :=
by sorry

end NUMINAMATH_CALUDE_real_gdp_change_omega_l1167_116780


namespace NUMINAMATH_CALUDE_product_of_y_values_l1167_116716

theorem product_of_y_values (y₁ y₂ : ℝ) : 
  (|5 * y₁| + 7 = 47) → 
  (|5 * y₂| + 7 = 47) → 
  y₁ ≠ y₂ → 
  y₁ * y₂ = -64 := by
sorry

end NUMINAMATH_CALUDE_product_of_y_values_l1167_116716


namespace NUMINAMATH_CALUDE_range_of_m_l1167_116797

/-- A decreasing function on the open interval (-2, 2) -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 < x ∧ x < y ∧ y < 2 → f x > f y

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (m - 1) > f (2 * m - 1)) :
  0 < m ∧ m < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1167_116797


namespace NUMINAMATH_CALUDE_spurs_basketball_count_l1167_116766

theorem spurs_basketball_count :
  let num_players : ℕ := 22
  let balls_per_player : ℕ := 11
  num_players * balls_per_player = 242 :=
by sorry

end NUMINAMATH_CALUDE_spurs_basketball_count_l1167_116766


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1167_116735

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ p = 2^n - 1 ∧ Prime p

theorem largest_mersenne_prime_under_500 :
  ∃ p : ℕ, p = 127 ∧ 
    is_mersenne_prime p ∧ 
    p < 500 ∧ 
    ∀ q : ℕ, is_mersenne_prime q → q < 500 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1167_116735


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1167_116794

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x^(1/4) = 18 / (9 - x^(1/4))) ↔ (x = 81 ∨ x = 1296) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1167_116794


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l1167_116763

/-- Given a right triangle with sides 6, 8, and 10, prove that a similar triangle
    with shortest side 15 has a perimeter of 60. -/
theorem similar_triangle_perimeter : 
  ∀ (a b c : ℝ) (x y z : ℝ),
  a = 6 ∧ b = 8 ∧ c = 10 →  -- Original triangle sides
  x = 15 →                  -- Shortest side of similar triangle
  x / a = y / b ∧ x / a = z / c →  -- Similarity condition
  x + y + z = 60 :=  -- Perimeter of similar triangle
by sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l1167_116763


namespace NUMINAMATH_CALUDE_expression_simplification_l1167_116720

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 2 + 1) 
  (hb : b = Real.sqrt 2 - 1) : 
  (a^2 - b^2) / a / (a + (2*a*b + b^2) / a) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1167_116720


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1167_116771

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n, 
    prove that if S_n / T_n = (2n - 5) / (4n + 3) for all n, then a_6 / b_6 = 17 / 47 -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) 
  (S T : ℕ → ℚ) 
  (h_arithmetic_a : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_arithmetic_b : ∀ n, b (n + 1) - b n = b (n + 2) - b (n + 1))
  (h_sum_a : ∀ n, S n = (n / 2) * (a 1 + a n))
  (h_sum_b : ∀ n, T n = (n / 2) * (b 1 + b n))
  (h_ratio : ∀ n, S n / T n = (2 * n - 5) / (4 * n + 3)) :
  a 6 / b 6 = 17 / 47 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1167_116771


namespace NUMINAMATH_CALUDE_delta_value_l1167_116791

-- Define the simultaneous equations
def simultaneous_equations (x y z : ℝ) : Prop :=
  x - y - z = -1 ∧ y - x - z = -2 ∧ z - x - y = -4

-- Define β
def β : ℕ := 5

-- Define γ
def γ : ℕ := 2

-- Define the polynomial equation
def polynomial_equation (a b δ : ℝ) : Prop :=
  ∃ t : ℝ, t^4 + a*t^2 + b*t + δ = 0 ∧
           1^4 + a*1^2 + b*1 + δ = 0 ∧
           γ^4 + a*γ^2 + b*γ + δ = 0 ∧
           (γ^2)^4 + a*(γ^2)^2 + b*(γ^2) + δ = 0

-- Theorem statement
theorem delta_value :
  ∃ x y z a b : ℝ,
    simultaneous_equations x y z →
    polynomial_equation a b (-56) :=
sorry

end NUMINAMATH_CALUDE_delta_value_l1167_116791


namespace NUMINAMATH_CALUDE_competition_probability_l1167_116727

/-- The probability of correctly answering a single question -/
def p_correct : ℝ := 0.8

/-- The probability of incorrectly answering a single question -/
def p_incorrect : ℝ := 1 - p_correct

/-- The number of preset questions in the competition -/
def num_questions : ℕ := 5

/-- The probability of answering exactly 4 questions before advancing -/
def prob_four_questions : ℝ := p_correct * p_incorrect * p_correct * p_correct

theorem competition_probability :
  prob_four_questions = 0.128 :=
sorry

end NUMINAMATH_CALUDE_competition_probability_l1167_116727


namespace NUMINAMATH_CALUDE_johnny_video_game_cost_l1167_116721

/-- The amount Johnny spent on the video game -/
def video_game_cost (september_savings october_savings november_savings amount_left : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - amount_left

/-- Theorem: Johnny spent $58 on the video game -/
theorem johnny_video_game_cost :
  video_game_cost 30 49 46 67 = 58 := by
  sorry

end NUMINAMATH_CALUDE_johnny_video_game_cost_l1167_116721


namespace NUMINAMATH_CALUDE_value_of_x_l1167_116749

theorem value_of_x : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1167_116749


namespace NUMINAMATH_CALUDE_not_sum_of_two_squares_l1167_116782

theorem not_sum_of_two_squares (n m : ℤ) (h : n = 4 * m + 3) : 
  ¬ ∃ (a b : ℤ), n = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_of_two_squares_l1167_116782


namespace NUMINAMATH_CALUDE_gas_price_difference_l1167_116755

/-- The difference between actual and expected gas prices -/
theorem gas_price_difference (actual_gallons : ℕ) (actual_price : ℕ) (expected_gallons : ℕ) :
  actual_gallons = 10 →
  actual_price = 150 →
  expected_gallons = 12 →
  actual_price - (actual_gallons * actual_price / expected_gallons) = 25 := by
sorry

end NUMINAMATH_CALUDE_gas_price_difference_l1167_116755


namespace NUMINAMATH_CALUDE_alex_shirts_l1167_116789

theorem alex_shirts (alex joe ben : ℕ) 
  (h1 : joe = alex + 3) 
  (h2 : ben = joe + 8) 
  (h3 : ben = 15) : 
  alex = 4 := by
sorry

end NUMINAMATH_CALUDE_alex_shirts_l1167_116789


namespace NUMINAMATH_CALUDE_hospital_staff_remaining_l1167_116702

/-- Given a hospital with an initial count of doctors and nurses,
    calculate the remaining staff after some quit. -/
def remaining_staff (initial_doctors : ℕ) (initial_nurses : ℕ)
                    (doctors_quit : ℕ) (nurses_quit : ℕ) : ℕ :=
  (initial_doctors - doctors_quit) + (initial_nurses - nurses_quit)

/-- Theorem stating that with 11 doctors and 18 nurses initially,
    if 5 doctors and 2 nurses quit, 22 staff members remain. -/
theorem hospital_staff_remaining :
  remaining_staff 11 18 5 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_remaining_l1167_116702


namespace NUMINAMATH_CALUDE_or_propagation_l1167_116722

theorem or_propagation (p q r : Prop) (h1 : p ∨ q) (h2 : ¬p ∨ r) : q ∨ r := by
  sorry

end NUMINAMATH_CALUDE_or_propagation_l1167_116722


namespace NUMINAMATH_CALUDE_box_office_scientific_notation_equality_l1167_116742

-- Define the box office revenue
def box_office_revenue : ℝ := 1824000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.824 * (10 ^ 9)

-- Theorem stating that the box office revenue is equal to its scientific notation representation
theorem box_office_scientific_notation_equality : 
  box_office_revenue = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_box_office_scientific_notation_equality_l1167_116742


namespace NUMINAMATH_CALUDE_max_x_value_l1167_116765

theorem max_x_value (x : ℤ) 
  (h : Real.log (2 * x + 1) / Real.log (1/4) < Real.log (x - 1) / Real.log (1/2)) : 
  x ≤ 3 ∧ ∃ y : ℤ, y ≤ 3 ∧ Real.log (2 * y + 1) / Real.log (1/4) < Real.log (y - 1) / Real.log (1/2) := by
  sorry

#check max_x_value

end NUMINAMATH_CALUDE_max_x_value_l1167_116765


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l1167_116754

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area := (1 / 2) * a * b
  ∀ c : ℝ, (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → area ≤ (1 / 2) * a * c ∧ area ≤ (1 / 2) * b * c :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l1167_116754


namespace NUMINAMATH_CALUDE_telescope_visual_range_increase_l1167_116783

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 50)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_increase_l1167_116783


namespace NUMINAMATH_CALUDE_x_equals_y_l1167_116738

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x = 1 + 1 / y) (h2 : y = 1 + 1 / x) : y = x := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_l1167_116738


namespace NUMINAMATH_CALUDE_expression_simplification_l1167_116701

theorem expression_simplification :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 10) / 4) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1167_116701


namespace NUMINAMATH_CALUDE_helicopter_performance_l1167_116772

/-- Heights of helicopter A's performances in km -/
def heights_A : List ℝ := [3.6, -2.4, 2.8, -1.5, 0.9]

/-- Heights of helicopter B's performances in km -/
def heights_B : List ℝ := [3.8, -2, 4.1, -2.3]

/-- The highest altitude reached by helicopter A -/
def highest_altitude_A : ℝ := 3.6

/-- The final altitude of helicopter A after 5 performances -/
def final_altitude_A : ℝ := 3.4

/-- The required height change for helicopter B's 5th performance -/
def height_change_B : ℝ := -0.2

theorem helicopter_performance :
  (heights_A.maximum? = some highest_altitude_A) ∧
  (heights_A.sum = final_altitude_A) ∧
  (heights_B.sum + height_change_B = final_altitude_A) := by
  sorry

end NUMINAMATH_CALUDE_helicopter_performance_l1167_116772


namespace NUMINAMATH_CALUDE_fraction_gt_one_not_equivalent_to_a_gt_b_l1167_116704

theorem fraction_gt_one_not_equivalent_to_a_gt_b :
  ¬(∀ (a b : ℝ), a / b > 1 ↔ a > b) :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_gt_one_not_equivalent_to_a_gt_b_l1167_116704


namespace NUMINAMATH_CALUDE_paths_avoiding_diagonal_l1167_116719

/-- The number of paths on an 8x8 grid from corner to corner, avoiding a diagonal line --/
def num_paths : ℕ := sorry

/-- Binomial coefficient function --/
def binom (n k : ℕ) : ℕ := sorry

theorem paths_avoiding_diagonal :
  num_paths = binom 7 1 * binom 7 1 + (binom 7 3) ^ 2 := by sorry

end NUMINAMATH_CALUDE_paths_avoiding_diagonal_l1167_116719


namespace NUMINAMATH_CALUDE_rectangle_from_equal_bisecting_diagonals_parallelogram_from_bisecting_diagonals_square_from_rhombus_equal_diagonals_l1167_116777

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def diagonals_bisect_each_other (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorems to prove
theorem rectangle_from_equal_bisecting_diagonals (q : Quadrilateral) :
  has_equal_diagonals q → diagonals_bisect_each_other q → is_rectangle q := by sorry

theorem parallelogram_from_bisecting_diagonals (q : Quadrilateral) :
  diagonals_bisect_each_other q → is_parallelogram q := by sorry

theorem square_from_rhombus_equal_diagonals (q : Quadrilateral) :
  is_rhombus q → has_equal_diagonals q → is_square q := by sorry

end NUMINAMATH_CALUDE_rectangle_from_equal_bisecting_diagonals_parallelogram_from_bisecting_diagonals_square_from_rhombus_equal_diagonals_l1167_116777


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l1167_116756

/-- Proves the factorization of x²y - 4xy + 4y -/
theorem factorization_1 (x y : ℝ) : x^2*y - 4*x*y + 4*y = y*(x-2)^2 := by
  sorry

/-- Proves the factorization of x² - 4y² -/
theorem factorization_2 (x y : ℝ) : x^2 - 4*y^2 = (x+2*y)*(x-2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l1167_116756


namespace NUMINAMATH_CALUDE_apples_picked_l1167_116714

theorem apples_picked (benny_apples dan_apples : ℕ) 
  (h1 : benny_apples = 2) 
  (h2 : dan_apples = 9) : 
  benny_apples + dan_apples = 11 := by
sorry

end NUMINAMATH_CALUDE_apples_picked_l1167_116714


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1167_116773

/-- An arithmetic sequence with first term 11, last term 101, and common difference 5 has 19 terms. -/
theorem arithmetic_sequence_terms : ∀ (a : ℕ → ℕ),
  (a 0 = 11) →  -- First term is 11
  (∀ n, a (n + 1) - a n = 5) →  -- Common difference is 5
  (∃ k, a k = 101) →  -- Last term is 101
  (∃ k, k = 19 ∧ a (k - 1) = 101) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1167_116773


namespace NUMINAMATH_CALUDE_exam_time_allocation_l1167_116715

theorem exam_time_allocation :
  ∀ (total_time : ℕ) (total_questions : ℕ) (type_a_questions : ℕ) (type_b_questions : ℕ),
    total_time = 3 * 60 →
    total_questions = 200 →
    type_a_questions = 50 →
    type_b_questions = total_questions - type_a_questions →
    2 * (total_time / total_questions) * type_b_questions = 
      (total_time / total_questions) * type_a_questions →
    (total_time / total_questions) * type_a_questions = 72 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_allocation_l1167_116715


namespace NUMINAMATH_CALUDE_intersection_difference_l1167_116750

theorem intersection_difference (A B : Set ℕ) (m n : ℕ) :
  A = {1, 2, m} →
  B = {2, 3, 4, n} →
  A ∩ B = {1, 2, 3} →
  m - n = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_difference_l1167_116750


namespace NUMINAMATH_CALUDE_jessica_watermelons_l1167_116796

/-- The number of watermelons Jessica initially grew -/
def initial_watermelons : ℕ := 35

/-- The number of watermelons eaten by rabbits -/
def eaten_watermelons : ℕ := 27

/-- The number of carrots Jessica grew (not used in the proof, but included for completeness) -/
def carrots : ℕ := 30

theorem jessica_watermelons : initial_watermelons - eaten_watermelons = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessica_watermelons_l1167_116796


namespace NUMINAMATH_CALUDE_compute_expression_l1167_116739

theorem compute_expression : (3 + 7)^2 + Real.sqrt (3^2 + 7^2) = 100 + Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1167_116739


namespace NUMINAMATH_CALUDE_ferry_time_difference_l1167_116788

/-- A ferry route between an island and the mainland -/
structure FerryRoute where
  speed : ℝ  -- Speed in km/h
  time : ℝ   -- Time in hours
  distance : ℝ -- Distance in km

/-- The problem setup for two ferry routes -/
def ferry_problem (p q : FerryRoute) : Prop :=
  p.speed = 8 ∧
  p.time = 3 ∧
  p.distance = p.speed * p.time ∧
  q.distance = 3 * p.distance ∧
  q.speed = p.speed + 1

theorem ferry_time_difference (p q : FerryRoute) 
  (h : ferry_problem p q) : q.time - p.time = 5 := by
  sorry

end NUMINAMATH_CALUDE_ferry_time_difference_l1167_116788


namespace NUMINAMATH_CALUDE_yoongi_has_smallest_number_l1167_116728

def jungkook_number : ℕ := 6 * 3
def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5

theorem yoongi_has_smallest_number : 
  yoongi_number ≤ jungkook_number ∧ yoongi_number ≤ yuna_number :=
sorry

end NUMINAMATH_CALUDE_yoongi_has_smallest_number_l1167_116728


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_4410_l1167_116741

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem count_perfect_square_factors_4410 :
  prime_factorization 4410 = [(2, 1), (3, 2), (5, 1), (7, 2)] →
  count_perfect_square_factors 4410 = 4 := by sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_4410_l1167_116741


namespace NUMINAMATH_CALUDE_compare_two_point_five_and_sqrt_six_l1167_116762

theorem compare_two_point_five_and_sqrt_six :
  2.5 > Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_compare_two_point_five_and_sqrt_six_l1167_116762


namespace NUMINAMATH_CALUDE_average_of_xyz_l1167_116706

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) :
  (x + y + z) / 3 = 4 := by sorry

end NUMINAMATH_CALUDE_average_of_xyz_l1167_116706


namespace NUMINAMATH_CALUDE_most_cost_effective_plan_verify_bus_capacities_l1167_116731

/-- Represents the capacity and cost of buses -/
structure BusInfo where
  small_capacity : ℕ
  large_capacity : ℕ
  small_cost : ℕ
  large_cost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  small_buses : ℕ
  large_buses : ℕ

/-- Checks if a rental plan is valid for the given number of students -/
def is_valid_plan (info : BusInfo) (students : ℕ) (plan : RentalPlan) : Prop :=
  plan.small_buses * info.small_capacity + plan.large_buses * info.large_capacity = students

/-- Calculates the cost of a rental plan -/
def plan_cost (info : BusInfo) (plan : RentalPlan) : ℕ :=
  plan.small_buses * info.small_cost + plan.large_buses * info.large_cost

/-- Theorem stating the most cost-effective plan -/
theorem most_cost_effective_plan (info : BusInfo) (students : ℕ) : 
  info.small_capacity = 20 →
  info.large_capacity = 45 →
  info.small_cost = 200 →
  info.large_cost = 400 →
  students = 400 →
  ∃ (optimal_plan : RentalPlan),
    is_valid_plan info students optimal_plan ∧
    optimal_plan.small_buses = 2 ∧
    optimal_plan.large_buses = 8 ∧
    plan_cost info optimal_plan = 3600 ∧
    ∀ (plan : RentalPlan), 
      is_valid_plan info students plan → 
      plan_cost info optimal_plan ≤ plan_cost info plan :=
by
  sorry

/-- Verifies the given bus capacities -/
theorem verify_bus_capacities (info : BusInfo) :
  info.small_capacity = 20 →
  info.large_capacity = 45 →
  3 * info.small_capacity + info.large_capacity = 105 ∧
  info.small_capacity + 2 * info.large_capacity = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_most_cost_effective_plan_verify_bus_capacities_l1167_116731


namespace NUMINAMATH_CALUDE_reflection_line_equation_l1167_116753

-- Define the triangle vertices and their images
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (8, 7)
def C : ℝ × ℝ := (6, -4)
def A' : ℝ × ℝ := (-5, 2)
def B' : ℝ × ℝ := (-10, 7)
def C' : ℝ × ℝ := (-8, -4)

-- Define the reflection line
def L (x : ℝ) : Prop := x = -1

-- Theorem statement
theorem reflection_line_equation :
  (∀ p p', (p = A ∧ p' = A') ∨ (p = B ∧ p' = B') ∨ (p = C ∧ p' = C') →
    p.2 = p'.2 ∧ L ((p.1 + p'.1) / 2)) →
  L (-1) :=
sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l1167_116753


namespace NUMINAMATH_CALUDE_central_cell_value_l1167_116784

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the 3x3 table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

/-- The theorem stating that the central cell value is 0.00081 -/
theorem central_cell_value (t : Table) (h : satisfies_conditions t) : t.e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l1167_116784


namespace NUMINAMATH_CALUDE_fraction_power_product_l1167_116707

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l1167_116707


namespace NUMINAMATH_CALUDE_difference_of_squares_75_45_l1167_116709

theorem difference_of_squares_75_45 : 75^2 - 45^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_75_45_l1167_116709


namespace NUMINAMATH_CALUDE_market_equilibrium_change_l1167_116717

-- Define the demand and supply functions
def demand (p : ℝ) : ℝ := 150 - p
def supply (p : ℝ) : ℝ := 3 * p - 10

-- Define the new demand function after increase
def new_demand (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

-- Define the equilibrium condition
def is_equilibrium (p : ℝ) : Prop := demand p = supply p

-- Define the new equilibrium condition
def is_new_equilibrium (α : ℝ) (p : ℝ) : Prop := new_demand α p = supply p

-- State the theorem
theorem market_equilibrium_change (α : ℝ) :
  (∃ p₀ : ℝ, is_equilibrium p₀) →
  (∃ p₁ : ℝ, is_new_equilibrium α p₁ ∧ p₁ = 1.25 * p₀) →
  α = 1.4 := by sorry

end NUMINAMATH_CALUDE_market_equilibrium_change_l1167_116717


namespace NUMINAMATH_CALUDE_problem_solution_l1167_116744

def f (x : ℝ) := |x + 1| + |x - 2|
def g (a x : ℝ) := |x + 1| - |x - a| + a

theorem problem_solution :
  (∀ x : ℝ, f x ≤ 5 ↔ x ∈ Set.Icc (-2) 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1167_116744


namespace NUMINAMATH_CALUDE_max_earnings_l1167_116759

def max_work_hours : ℕ := 80
def regular_hours : ℕ := 20
def regular_wage : ℚ := 8
def regular_tips : ℚ := 2
def overtime_wage_multiplier : ℚ := 1.25
def overtime_tips : ℚ := 3
def bonus_per_5_hours : ℚ := 20

def overtime_hours : ℕ := max_work_hours - regular_hours

def regular_earnings : ℚ := regular_hours * (regular_wage + regular_tips)
def overtime_wage : ℚ := regular_wage * overtime_wage_multiplier
def overtime_earnings : ℚ := overtime_hours * (overtime_wage + overtime_tips)
def bonus : ℚ := (overtime_hours / 5) * bonus_per_5_hours

def total_earnings : ℚ := regular_earnings + overtime_earnings + bonus

theorem max_earnings :
  total_earnings = 1220 := by sorry

end NUMINAMATH_CALUDE_max_earnings_l1167_116759


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l1167_116758

/-- Calculates the cost of tax-free items given total purchase, sales tax, and tax rate -/
def cost_of_tax_free_items (total_purchase : ℚ) (sales_tax : ℚ) (tax_rate : ℚ) : ℚ :=
  total_purchase - sales_tax / tax_rate

/-- Theorem stating that given the problem conditions, the cost of tax-free items is 20 -/
theorem tax_free_items_cost : 
  let total_purchase : ℚ := 25
  let sales_tax : ℚ := 30 / 100  -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 6 / 100    -- 6%
  cost_of_tax_free_items total_purchase sales_tax tax_rate = 20 := by
  sorry


end NUMINAMATH_CALUDE_tax_free_items_cost_l1167_116758


namespace NUMINAMATH_CALUDE_cucumber_salad_problem_l1167_116723

theorem cucumber_salad_problem (total : ℕ) (cucumber : ℕ) (tomato : ℕ) : 
  total = 280 →
  tomato = 3 * cucumber →
  total = cucumber + tomato →
  cucumber = 70 := by
sorry

end NUMINAMATH_CALUDE_cucumber_salad_problem_l1167_116723


namespace NUMINAMATH_CALUDE_starters_with_triplet_l1167_116787

def total_players : ℕ := 12
def triplets : ℕ := 3
def starters : ℕ := 5

theorem starters_with_triplet (total_players : ℕ) (triplets : ℕ) (starters : ℕ) :
  total_players = 12 →
  triplets = 3 →
  starters = 5 →
  (Nat.choose total_players starters) - (Nat.choose (total_players - triplets) starters) = 666 :=
by sorry

end NUMINAMATH_CALUDE_starters_with_triplet_l1167_116787


namespace NUMINAMATH_CALUDE_two_numbers_problem_l1167_116748

theorem two_numbers_problem :
  ∃ (x y : ℝ), y = 33 ∧ x + y = 51 ∧ y = 2 * x - 3 :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l1167_116748


namespace NUMINAMATH_CALUDE_palindrome_with_seven_percentage_l1167_116798

-- Define a palindrome in the range 100 to 999
def IsPalindrome (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

-- Define a number containing at least one 7
def ContainsSeven (n : Nat) : Prop :=
  (n / 100 = 7) ∨ ((n / 10) % 10 = 7) ∨ (n % 10 = 7)

-- Count of palindromes with at least one 7
def PalindromeWithSeven : Nat :=
  19

-- Total count of palindromes between 100 and 999
def TotalPalindromes : Nat :=
  90

-- Theorem statement
theorem palindrome_with_seven_percentage :
  (PalindromeWithSeven : ℚ) / TotalPalindromes = 19 / 90 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_with_seven_percentage_l1167_116798


namespace NUMINAMATH_CALUDE_expected_value_is_thirteen_eighths_l1167_116732

/-- Represents the outcome of a die roll -/
inductive DieOutcome
| Prime (n : Nat)
| NonPrimeSquare (n : Nat)
| Other (n : Nat)

/-- The set of possible outcomes for an 8-sided die -/
def dieOutcomes : Finset DieOutcome := sorry

/-- The probability of each outcome, assuming a fair die -/
def prob (outcome : DieOutcome) : ℚ := sorry

/-- The winnings for each outcome -/
def winnings (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Prime n => n
  | DieOutcome.NonPrimeSquare _ => 2
  | DieOutcome.Other _ => -4

/-- The expected value of winnings for one die toss -/
def expectedValue : ℚ := sorry

/-- Theorem stating that the expected value of winnings is 13/8 -/
theorem expected_value_is_thirteen_eighths :
  expectedValue = 13 / 8 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_thirteen_eighths_l1167_116732


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1167_116769

/-- Given an angle whose complement is 7° more than five times the angle,
    prove that the angle measures 13.833°. -/
theorem angle_measure_proof (x : ℝ) : 
  x + (5 * x + 7) = 90 → x = 13.833 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1167_116769


namespace NUMINAMATH_CALUDE_bridgets_skittles_proof_l1167_116743

/-- The number of Skittles Henry has -/
def henrys_skittles : ℕ := 4

/-- The total number of Skittles after Henry gives his to Bridget -/
def total_skittles : ℕ := 8

/-- Bridget's initial number of Skittles -/
def bridgets_initial_skittles : ℕ := total_skittles - henrys_skittles

theorem bridgets_skittles_proof :
  bridgets_initial_skittles = 4 :=
by sorry

end NUMINAMATH_CALUDE_bridgets_skittles_proof_l1167_116743


namespace NUMINAMATH_CALUDE_root_equation_problem_l1167_116736

theorem root_equation_problem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) →
  r = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l1167_116736


namespace NUMINAMATH_CALUDE_root_existence_l1167_116764

/-- The logarithm function with base 10 -/
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The function f(x) = x + lg(x - 2) -/
noncomputable def f (x : ℝ) : ℝ := x + lg (x - 2)

theorem root_existence :
  ∃ c ∈ Set.Ioo 2.001 2.01, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_root_existence_l1167_116764


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l1167_116790

theorem cos_two_theta_value (θ : ℝ) 
  (h1 : 3 * Real.sin (2 * θ) = 4 * Real.tan θ) 
  (h2 : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.cos (2 * θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l1167_116790


namespace NUMINAMATH_CALUDE_books_added_to_shelf_l1167_116746

theorem books_added_to_shelf (initial_action_figures initial_books added_books : ℕ) :
  initial_action_figures = 7 →
  initial_books = 2 →
  initial_action_figures = (initial_books + added_books) + 1 →
  added_books = 4 :=
by sorry

end NUMINAMATH_CALUDE_books_added_to_shelf_l1167_116746


namespace NUMINAMATH_CALUDE_equation_solution_l1167_116737

theorem equation_solution :
  ∃ x : ℝ, (x + 2 ≠ 0 ∧ x - 2 ≠ 0) ∧ (2 * x / (x + 2) + x / (x - 2) = 3) ∧ x = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1167_116737


namespace NUMINAMATH_CALUDE_annual_interest_calculation_l1167_116708

def total_amount : ℝ := 3000
def first_part : ℝ := 299.99999999999994
def second_part : ℝ := total_amount - first_part
def interest_rate1 : ℝ := 0.03
def interest_rate2 : ℝ := 0.05

theorem annual_interest_calculation :
  let interest1 := first_part * interest_rate1
  let interest2 := second_part * interest_rate2
  interest1 + interest2 = 144 := by sorry

end NUMINAMATH_CALUDE_annual_interest_calculation_l1167_116708


namespace NUMINAMATH_CALUDE_beijing_spirit_max_l1167_116710

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- The equation: Patriotism × Innovation × Inclusiveness + Integrity = Beijing Spirit -/
def equation (patriotism innovation inclusiveness integrity : Digit) (beijingSpirit : Nat) :=
  (patriotism.val * innovation.val * inclusiveness.val + integrity.val = beijingSpirit)

/-- All digits are different -/
def all_different (patriotism nation creation new inclusiveness tolerance integrity virtue : Digit) :=
  patriotism ≠ nation ∧ patriotism ≠ creation ∧ patriotism ≠ new ∧ patriotism ≠ inclusiveness ∧
  patriotism ≠ tolerance ∧ patriotism ≠ integrity ∧ patriotism ≠ virtue ∧
  nation ≠ creation ∧ nation ≠ new ∧ nation ≠ inclusiveness ∧ nation ≠ tolerance ∧
  nation ≠ integrity ∧ nation ≠ virtue ∧ creation ≠ new ∧ creation ≠ inclusiveness ∧
  creation ≠ tolerance ∧ creation ≠ integrity ∧ creation ≠ virtue ∧ new ≠ inclusiveness ∧
  new ≠ tolerance ∧ new ≠ integrity ∧ new ≠ virtue ∧ inclusiveness ≠ tolerance ∧
  inclusiveness ≠ integrity ∧ inclusiveness ≠ virtue ∧ tolerance ≠ integrity ∧ tolerance ≠ virtue ∧
  integrity ≠ virtue

theorem beijing_spirit_max (patriotism nation creation new inclusiveness tolerance integrity virtue : Digit) :
  all_different patriotism nation creation new inclusiveness tolerance integrity virtue →
  equation patriotism creation inclusiveness integrity 9898 →
  integrity.val = 98 := by
  sorry

end NUMINAMATH_CALUDE_beijing_spirit_max_l1167_116710


namespace NUMINAMATH_CALUDE_book_exchange_count_l1167_116718

/-- Represents a book exchange in a book club --/
structure BookExchange where
  friends : Finset (Fin 6)
  give : Fin 6 → Fin 6
  receive : Fin 6 → Fin 6

/-- Conditions for a valid book exchange --/
def ValidExchange (e : BookExchange) : Prop :=
  (∀ i, i ∈ e.friends) ∧
  (∀ i, e.give i ≠ i) ∧
  (∀ i, e.receive i ≠ i) ∧
  (∀ i, e.give i ≠ e.receive i) ∧
  (∀ i j, e.give i = e.give j → i = j) ∧
  (∀ i j, e.receive i = e.receive j → i = j)

/-- The number of valid book exchanges --/
def NumberOfExchanges : ℕ := sorry

/-- Theorem stating that the number of valid book exchanges is 160 --/
theorem book_exchange_count : NumberOfExchanges = 160 := by sorry

end NUMINAMATH_CALUDE_book_exchange_count_l1167_116718


namespace NUMINAMATH_CALUDE_parabola_equation_l1167_116799

/-- Represents a parabola -/
structure Parabola where
  -- The equation of the parabola in the form y² = 2px or x² = 2py
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

/-- Checks if the parabola has a given focus -/
def Parabola.hasFocus (p : Parabola) (fx fy : ℝ) : Prop :=
  ∃ (a : ℝ), (p.equation = fun x y ↦ (y - fy)^2 = 4*a*(x - fx)) ∨
             (p.equation = fun x y ↦ (x - fx)^2 = 4*a*(y - fy))

/-- Checks if the parabola has a given directrix -/
def Parabola.hasDirectrix (p : Parabola) (d : ℝ) : Prop :=
  ∃ (a : ℝ), (p.equation = fun x y ↦ y^2 = 4*a*(x + a)) ∧ d = -a ∨
             (p.equation = fun x y ↦ x^2 = 4*a*(y + a)) ∧ d = -a

theorem parabola_equation (p : Parabola) :
  p.hasFocus (-2) 0 →
  p.hasDirectrix (-1) →
  p.contains 1 2 →
  (p.equation = fun x y ↦ y^2 = 4*x) ∨
  (p.equation = fun x y ↦ x^2 = 1/2*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1167_116799


namespace NUMINAMATH_CALUDE_visitors_three_months_l1167_116725

/-- The total number of visitors for three months given specific conditions -/
theorem visitors_three_months (october : ℕ) (november_increase_percent : ℕ) (december_increase : ℕ)
  (h1 : october = 100)
  (h2 : november_increase_percent = 15)
  (h3 : december_increase = 15) :
  october + 
  (october + october * november_increase_percent / 100) + 
  (october + october * november_increase_percent / 100 + december_increase) = 345 := by
  sorry

end NUMINAMATH_CALUDE_visitors_three_months_l1167_116725


namespace NUMINAMATH_CALUDE_z_purely_imaginary_iff_m_eq_neg_one_l1167_116747

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- A complex number z defined as i(5-i) + m -/
def z (m : ℝ) : ℂ := i * (5 - i) + m

/-- Theorem stating that z is purely imaginary if and only if m = -1 -/
theorem z_purely_imaginary_iff_m_eq_neg_one (m : ℝ) :
  z m = Complex.I * (z m).im ↔ m = -1 := by sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_iff_m_eq_neg_one_l1167_116747


namespace NUMINAMATH_CALUDE_golf_ball_distribution_returns_to_initial_state_l1167_116751

/-- Represents the distribution of golf balls in boxes and the next starting box -/
structure State :=
  (balls : Fin 10 → ℕ)
  (next_box : Fin 10)

/-- The set of all possible states -/
def S : Set State := {s | ∀ i, s.balls i > 0}

/-- Represents one move in the game -/
def move (s : State) : State :=
  sorry

theorem golf_ball_distribution_returns_to_initial_state :
  ∀ (initial : State),
  initial ∈ S →
  ∃ (n : ℕ+),
  (move^[n] initial) = initial :=
sorry

end NUMINAMATH_CALUDE_golf_ball_distribution_returns_to_initial_state_l1167_116751


namespace NUMINAMATH_CALUDE_plates_added_before_fall_l1167_116786

theorem plates_added_before_fall (initial_plates : Nat) (second_addition : Nat) (total_plates : Nat)
  (h1 : initial_plates = 27)
  (h2 : second_addition = 37)
  (h3 : total_plates = 83) :
  total_plates - (initial_plates + second_addition) = 19 := by
  sorry

end NUMINAMATH_CALUDE_plates_added_before_fall_l1167_116786


namespace NUMINAMATH_CALUDE_square_of_prime_divisibility_l1167_116779

theorem square_of_prime_divisibility (n p : ℕ) : 
  n > 1 → 
  Nat.Prime p → 
  (n ∣ p - 1) → 
  (p ∣ n^3 - 1) → 
  ∃ k : ℕ, 4*p - 3 = k^2 :=
sorry

end NUMINAMATH_CALUDE_square_of_prime_divisibility_l1167_116779


namespace NUMINAMATH_CALUDE_rhombus_q_value_l1167_116713

/-- A rhombus ABCD on a Cartesian plane -/
structure Rhombus where
  P : ℝ
  u : ℝ
  v : ℝ
  A : ℝ × ℝ := (0, 0)
  B : ℝ × ℝ := (P, 1)
  C : ℝ × ℝ := (u, v)
  D : ℝ × ℝ := (1, P)

/-- The sum of u and v coordinates of point C -/
def Q (r : Rhombus) : ℝ := r.u + r.v

/-- Theorem: For a rhombus ABCD with given coordinates, Q equals 2P + 2 -/
theorem rhombus_q_value (r : Rhombus) : Q r = 2 * r.P + 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_q_value_l1167_116713


namespace NUMINAMATH_CALUDE_bennys_seashells_l1167_116768

theorem bennys_seashells (initial_seashells given_away_seashells : ℚ) 
  (h1 : initial_seashells = 66.5)
  (h2 : given_away_seashells = 52.5) :
  initial_seashells - given_away_seashells = 14 :=
by sorry

end NUMINAMATH_CALUDE_bennys_seashells_l1167_116768


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1167_116792

theorem rectangle_max_area (p : ℝ) (h1 : p = 40) : 
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * (l + w) = p ∧ w = 2 * l ∧ l * w = 800 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1167_116792


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1167_116775

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1167_116775
