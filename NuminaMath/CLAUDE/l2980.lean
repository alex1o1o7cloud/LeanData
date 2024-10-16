import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_partial_sum_l2980_298057

/-- Given a geometric sequence {a_n} with S_2 = 7 and S_6 = 91, prove that S_4 = 35 -/
theorem geometric_sequence_partial_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence condition
  (a 1 + a 1 * r = 7) →         -- S_2 = 7
  (a 1 + a 1 * r + a 1 * r^2 + a 1 * r^3 + a 1 * r^4 + a 1 * r^5 = 91) →  -- S_6 = 91
  (a 1 + a 1 * r + a 1 * r^2 + a 1 * r^3 = 35) :=  -- S_4 = 35
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_partial_sum_l2980_298057


namespace NUMINAMATH_CALUDE_calculator_result_is_very_large_l2980_298071

/-- The calculator function that replaces x with x^2 - 2 -/
def calc_function (x : ℝ) : ℝ := x^2 - 2

/-- Applies the calculator function n times to the initial value -/
def apply_n_times (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => calc_function (apply_n_times n x)

/-- A number is considered "very large" if it's greater than 10^100 -/
def is_very_large (x : ℝ) : Prop := x > 10^100

/-- Theorem stating that after 50 applications of the calculator function starting from 3, the result is very large -/
theorem calculator_result_is_very_large : 
  is_very_large (apply_n_times 50 3) := by sorry

end NUMINAMATH_CALUDE_calculator_result_is_very_large_l2980_298071


namespace NUMINAMATH_CALUDE_acute_triangle_exists_l2980_298007

/-- Given 5 real numbers representing lengths of line segments,
    if any three of these numbers can form a triangle,
    then there exists a combination of three numbers that forms a triangle with all acute angles. -/
theorem acute_triangle_exists (a b c d e : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) 
  (h_triangle : ∀ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) →
                               (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) →
                               (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) →
                               x ≠ y ∧ y ≠ z ∧ x ≠ z →
                               x + y > z ∧ y + z > x ∧ x + z > y) :
  ∃ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
                 (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
                 (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
                 x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                 x^2 + y^2 > z^2 ∧ y^2 + z^2 > x^2 ∧ x^2 + z^2 > y^2 :=
by sorry


end NUMINAMATH_CALUDE_acute_triangle_exists_l2980_298007


namespace NUMINAMATH_CALUDE_alloy_mixture_l2980_298050

/-- The amount of alloy A in kg -/
def alloy_A : ℝ := 130

/-- The ratio of lead to tin in alloy A -/
def ratio_A : ℚ := 2/3

/-- The ratio of tin to copper in alloy B -/
def ratio_B : ℚ := 3/4

/-- The amount of tin in the new alloy in kg -/
def tin_new : ℝ := 146.57

/-- The amount of alloy B mixed with alloy A in kg -/
def alloy_B : ℝ := 160.33

theorem alloy_mixture :
  alloy_B * (ratio_B / (1 + ratio_B)) + alloy_A * (ratio_A / (1 + ratio_A)) = tin_new := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_l2980_298050


namespace NUMINAMATH_CALUDE_purchase_cost_l2980_298010

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 4

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 3/2

/-- The discount applied when purchasing at least 10 sandwiches -/
def bulk_discount : ℚ := 5

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 10

/-- The number of sodas purchased -/
def num_sodas : ℕ := 6

/-- The total cost of the purchase -/
def total_cost : ℚ := 
  (num_sandwiches * sandwich_cost - bulk_discount) + (num_sodas * soda_cost)

theorem purchase_cost : total_cost = 44 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l2980_298010


namespace NUMINAMATH_CALUDE_smallest_m_for_inequality_l2980_298015

theorem smallest_m_for_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) :
  ∀ m : ℝ, (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 → 
    m * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1) → 
  m ≥ 27 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_inequality_l2980_298015


namespace NUMINAMATH_CALUDE_haley_small_gardens_l2980_298072

/-- The number of small gardens Haley had -/
def num_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

/-- Theorem stating that Haley had 7 small gardens -/
theorem haley_small_gardens : 
  num_small_gardens 56 35 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_haley_small_gardens_l2980_298072


namespace NUMINAMATH_CALUDE_valid_q_values_are_zero_two_neg_two_l2980_298068

/-- Given a set of 10 distinct real numbers, this function determines the values of q
    such that every number in the second line is also in the third line. -/
def valid_q_values (napkin : Finset ℝ) : Set ℝ :=
  { q : ℝ | ∀ (a b c d : ℝ), a ∈ napkin → b ∈ napkin → c ∈ napkin → d ∈ napkin →
    ∃ (w x y z : ℝ), w ∈ napkin ∧ x ∈ napkin ∧ y ∈ napkin ∧ z ∈ napkin →
    q * (a - b) * (c - d) = (w - x)^2 + (y - z)^2 - (x - y)^2 - (z - w)^2 }

/-- Theorem stating that for any set of 10 distinct real numbers, 
    the only valid q values are 0, 2, and -2. -/
theorem valid_q_values_are_zero_two_neg_two (napkin : Finset ℝ) 
  (h : napkin.card = 10) :
  valid_q_values napkin = {0, 2, -2} := by
  sorry


end NUMINAMATH_CALUDE_valid_q_values_are_zero_two_neg_two_l2980_298068


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_80_sevens_80_threes_l2980_298022

/-- A number consisting of n repeated digits d -/
def repeated_digit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_product_80_sevens_80_threes : 
  sum_of_digits (repeated_digit 80 7 * repeated_digit 80 3) = 240 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_80_sevens_80_threes_l2980_298022


namespace NUMINAMATH_CALUDE_solution_set_for_m_eq_2_range_of_m_l2980_298090

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m|

-- Part I
theorem solution_set_for_m_eq_2 :
  {x : ℝ | f 2 x > 7 - |x - 1|} = {x : ℝ | x < -4 ∨ x > 5} := by sorry

-- Part II
theorem range_of_m :
  {m : ℝ | ∃ x : ℝ, f m x > 7 + |x - 1|} = {m : ℝ | m < -6 ∨ m > 8} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_m_eq_2_range_of_m_l2980_298090


namespace NUMINAMATH_CALUDE_coeff_bound_theorem_l2980_298012

/-- Represents a real polynomial -/
def RealPolynomial := ℝ → ℝ

/-- The degree of a polynomial -/
def degree (p : RealPolynomial) : ℕ := sorry

/-- The largest absolute value of the coefficients of a polynomial -/
def max_coeff (p : RealPolynomial) : ℝ := sorry

/-- Multiplication of polynomials -/
def poly_mul (p q : RealPolynomial) : RealPolynomial := sorry

/-- Addition of a constant to x -/
def add_const (a : ℝ) : RealPolynomial := sorry

theorem coeff_bound_theorem (p q : RealPolynomial) (a : ℝ) (n : ℕ) (h k : ℝ) :
  p = poly_mul (add_const a) q →
  degree p = n →
  max_coeff p = h →
  max_coeff q = k →
  k ≤ h * n := by sorry

end NUMINAMATH_CALUDE_coeff_bound_theorem_l2980_298012


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2980_298019

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧ (∃ a, a^2 > 1 ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2980_298019


namespace NUMINAMATH_CALUDE_snow_at_brecknock_l2980_298025

/-- The amount of snow at Mrs. Hilt's house in inches -/
def mrs_hilt_snow : ℕ := 29

/-- The difference in snow between Mrs. Hilt's house and Brecknock Elementary School in inches -/
def snow_difference : ℕ := 12

/-- The amount of snow at Brecknock Elementary School in inches -/
def brecknock_snow : ℕ := mrs_hilt_snow - snow_difference

theorem snow_at_brecknock : brecknock_snow = 17 := by
  sorry

end NUMINAMATH_CALUDE_snow_at_brecknock_l2980_298025


namespace NUMINAMATH_CALUDE_problem_statement_l2980_298043

theorem problem_statement :
  ∀ d : ℕ, (5 ^ 5) * (9 ^ 3) = 3 * d ∧ d = 15 ^ 5 → d = 759375 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2980_298043


namespace NUMINAMATH_CALUDE_max_underwear_is_four_l2980_298095

/-- Represents the washing machine and clothing weights --/
structure WashingMachine where
  limit : Nat
  sock_weight : Nat
  underwear_weight : Nat
  shirt_weight : Nat
  shorts_weight : Nat
  pants_weight : Nat

/-- Represents the clothes Tony is washing --/
structure ClothesInWash where
  pants : Nat
  shirts : Nat
  shorts : Nat
  socks : Nat

/-- Calculates the maximum number of additional pairs of underwear that can be added --/
def max_additional_underwear (wm : WashingMachine) (clothes : ClothesInWash) : Nat :=
  let current_weight := 
    clothes.pants * wm.pants_weight +
    clothes.shirts * wm.shirt_weight +
    clothes.shorts * wm.shorts_weight +
    clothes.socks * wm.sock_weight
  let remaining_weight := wm.limit - current_weight
  remaining_weight / wm.underwear_weight

/-- Theorem stating that the maximum number of additional pairs of underwear is 4 --/
theorem max_underwear_is_four :
  let wm : WashingMachine := {
    limit := 50,
    sock_weight := 2,
    underwear_weight := 4,
    shirt_weight := 5,
    shorts_weight := 8,
    pants_weight := 10
  }
  let clothes : ClothesInWash := {
    pants := 1,
    shirts := 2,
    shorts := 1,
    socks := 3
  }
  max_additional_underwear wm clothes = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_underwear_is_four_l2980_298095


namespace NUMINAMATH_CALUDE_f_properties_l2980_298018

def f (x y : ℝ) : ℝ × ℝ := (x - y, x + y)

theorem f_properties :
  (f 3 5 = (-2, 8)) ∧ (f 4 1 = (3, 5)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2980_298018


namespace NUMINAMATH_CALUDE_race_finish_time_difference_l2980_298058

/-- Race problem statement -/
theorem race_finish_time_difference 
  (race_distance : ℝ) 
  (malcolm_speed : ℝ) 
  (joshua_speed : ℝ) 
  (h1 : race_distance = 12) 
  (h2 : malcolm_speed = 7) 
  (h3 : joshua_speed = 9) : 
  joshua_speed * race_distance - malcolm_speed * race_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_time_difference_l2980_298058


namespace NUMINAMATH_CALUDE_probability_of_e_in_theorem_l2980_298061

theorem probability_of_e_in_theorem :
  let total_letters : ℕ := 7
  let number_of_e : ℕ := 2
  let probability : ℚ := number_of_e / total_letters
  probability = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_of_e_in_theorem_l2980_298061


namespace NUMINAMATH_CALUDE_green_shirts_count_l2980_298037

/-- Proves that the number of green shirts is 17 given the total number of shirts and the number of blue shirts. -/
theorem green_shirts_count (total_shirts : ℕ) (blue_shirts : ℕ) (h1 : total_shirts = 23) (h2 : blue_shirts = 6) :
  total_shirts - blue_shirts = 17 := by
  sorry

#check green_shirts_count

end NUMINAMATH_CALUDE_green_shirts_count_l2980_298037


namespace NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l2980_298086

/-- Calculates the cost of windows given the quantity and discount offer -/
def windowCost (quantity : ℕ) (regularPrice : ℕ) (discountRate : ℕ) : ℕ :=
  let discountedQuantity := quantity - (quantity / (discountRate + 2)) * 2
  discountedQuantity * regularPrice

/-- Proves that joint purchase does not lead to savings compared to separate purchases -/
theorem no_savings_on_joint_purchase (regularPrice : ℕ) (discountRate : ℕ) :
  windowCost 22 regularPrice discountRate =
  windowCost 10 regularPrice discountRate + windowCost 12 regularPrice discountRate :=
by sorry

end NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l2980_298086


namespace NUMINAMATH_CALUDE_expression_simplification_l2980_298099

theorem expression_simplification (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ 1/2) :
  1 - (1 / (1 - a / (1 - a))) = -a / (1 - 2*a) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2980_298099


namespace NUMINAMATH_CALUDE_closest_cube_approximation_l2980_298097

def x : Real := 0.48017

theorem closest_cube_approximation :
  ∀ y ∈ ({0.011, 1.10, 11.0, 110} : Set Real),
  |x^3 - 0.110| < |x^3 - y| := by sorry

end NUMINAMATH_CALUDE_closest_cube_approximation_l2980_298097


namespace NUMINAMATH_CALUDE_f_of_2_eq_0_l2980_298044

/-- The function f(x) = x^3 - 3x^2 + 2x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- Theorem: f(2) = 0 -/
theorem f_of_2_eq_0 : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_eq_0_l2980_298044


namespace NUMINAMATH_CALUDE_stratified_sample_problem_l2980_298092

theorem stratified_sample_problem (m : ℕ) : 
  let total_male : ℕ := 56
  let sample_size : ℕ := 28
  let male_in_sample : ℕ := (sample_size + 4) / 2
  let female_in_sample : ℕ := (sample_size - 4) / 2
  (male_in_sample : ℚ) / female_in_sample = (total_male : ℚ) / m →
  m = 42 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_problem_l2980_298092


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2980_298016

theorem consecutive_integers_square_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x^2 + (x + 1)^2 = 1625 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2980_298016


namespace NUMINAMATH_CALUDE_quadratic_equation_from_condition_l2980_298088

theorem quadratic_equation_from_condition (a b : ℝ) :
  a^2 - 4*a*b + 5*b^2 - 2*b + 1 = 0 →
  ∃ (x : ℝ → ℝ), (x a = 0 ∧ x b = 0) ∧ (∀ y, x y = y^2 - 3*y + 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_condition_l2980_298088


namespace NUMINAMATH_CALUDE_expected_points_is_seventeen_thirds_l2980_298021

/-- Represents the outcomes of the biased die -/
inductive Outcome
| Odd
| EvenNotSix
| Six

/-- The probability of each outcome -/
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Odd => 1/2
  | Outcome.EvenNotSix => 1/3
  | Outcome.Six => 1/6

/-- The points gained for each outcome -/
def points (o : Outcome) : ℚ :=
  match o with
  | Outcome.Odd => 9/2  -- Average of 1, 3, and 5
  | Outcome.EvenNotSix => 3  -- Average of 2 and 4
  | Outcome.Six => -5

/-- The expected value of points gained -/
def expected_value : ℚ :=
  (probability Outcome.Odd * points Outcome.Odd) +
  (probability Outcome.EvenNotSix * points Outcome.EvenNotSix) +
  (probability Outcome.Six * points Outcome.Six)

theorem expected_points_is_seventeen_thirds :
  expected_value = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_points_is_seventeen_thirds_l2980_298021


namespace NUMINAMATH_CALUDE_triangle_inequality_l2980_298080

/-- Given a triangle with side lengths a, b, c and area S, 
    prove that a^2 + b^2 + c^2 ≥ 4√3 S -/
theorem triangle_inequality (a b c S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2980_298080


namespace NUMINAMATH_CALUDE_calculation_proof_l2980_298054

theorem calculation_proof :
  (2/3 - 1/4 - 1/6) * 24 = 6 ∧
  (-2)^3 + (-9 + (-3)^2 * (1/3)) = -14 :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l2980_298054


namespace NUMINAMATH_CALUDE_odd_function_solution_set_l2980_298049

/-- A function is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The solution set of an inequality -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x*f x - Real.exp (abs x) > 0}

theorem odd_function_solution_set
  (f : ℝ → ℝ)
  (hodd : OddFunction f)
  (hf1 : f 1 = Real.exp 1)
  (hineq : ∀ x ≥ 0, (x - 1) * f x < x * (deriv f x)) :
  SolutionSet f = Set.Ioi 1 ∪ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_odd_function_solution_set_l2980_298049


namespace NUMINAMATH_CALUDE_bird_cage_problem_l2980_298077

theorem bird_cage_problem (B : ℕ) (F : ℚ) : 
  B = 60 → 
  (1/3 : ℚ) * (2/3 : ℚ) * B * (1 - F) = 8 → 
  F = 4/5 := by
sorry

end NUMINAMATH_CALUDE_bird_cage_problem_l2980_298077


namespace NUMINAMATH_CALUDE_ab_value_l2980_298059

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2980_298059


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l2980_298040

/-- Represents the different types of sampling methods. -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a school with different student groups. -/
structure School where
  elementaryStudents : ℕ
  juniorHighStudents : ℕ
  highSchoolStudents : ℕ

/-- Determines if there are significant differences between student groups. -/
def hasDifferences (s : School) : Prop :=
  sorry -- Definition of significant differences

/-- Determines the most appropriate sampling method given a school and sample size. -/
def mostAppropriateSamplingMethod (s : School) (sampleSize : ℕ) : SamplingMethod :=
  sorry -- Definition of most appropriate sampling method

/-- Theorem stating that stratified sampling is the most appropriate method for the given conditions. -/
theorem stratified_sampling_most_appropriate (s : School) (sampleSize : ℕ) :
  s.elementaryStudents = 125 →
  s.juniorHighStudents = 280 →
  s.highSchoolStudents = 95 →
  sampleSize = 100 →
  hasDifferences s →
  mostAppropriateSamplingMethod s sampleSize = SamplingMethod.Stratified :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l2980_298040


namespace NUMINAMATH_CALUDE_x_intercept_of_specific_line_l2980_298032

/-- The x-intercept of a line is a point on the x-axis where the line intersects it. -/
def x_intercept (a b c : ℚ) : ℚ × ℚ :=
  (c / a, 0)

/-- The line equation is of the form ax + by = c, where a, b, and c are rational numbers. -/
structure LineEquation where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Theorem: The x-intercept of the line 4x - 3y = 24 is the point (6, 0). -/
theorem x_intercept_of_specific_line :
  let line : LineEquation := { a := 4, b := -3, c := 24 }
  x_intercept line.a line.b line.c = (6, 0) := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_specific_line_l2980_298032


namespace NUMINAMATH_CALUDE_three_digit_equation_l2980_298093

/-- 
Given a three-digit number A7B where 7 is the tens digit, 
prove that A = 6 if A7B + 23 = 695
-/
theorem three_digit_equation (A B : ℕ) : 
  (A * 100 + 70 + B) + 23 = 695 → 
  0 ≤ A ∧ A ≤ 9 → 
  0 ≤ B ∧ B ≤ 9 → 
  A = 6 := by
sorry

end NUMINAMATH_CALUDE_three_digit_equation_l2980_298093


namespace NUMINAMATH_CALUDE_box_volume_formula_l2980_298064

/-- The volume of an open box formed from a rectangular sheet --/
def boxVolume (x y : ℝ) : ℝ := (16 - 2*x) * (12 - 2*y) * y

/-- Theorem stating the volume of the box --/
theorem box_volume_formula (x y : ℝ) :
  boxVolume x y = 192*y - 32*y^2 - 24*x*y + 4*x*y^2 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_formula_l2980_298064


namespace NUMINAMATH_CALUDE_johnny_attend_probability_l2980_298003

-- Define the probabilities
def p_rain : ℝ := 0.3
def p_sunny : ℝ := 0.5
def p_cloudy : ℝ := 1 - p_rain - p_sunny

def p_attend_given_rain : ℝ := 0.5
def p_attend_given_sunny : ℝ := 0.9
def p_attend_given_cloudy : ℝ := 0.7

-- Define the theorem
theorem johnny_attend_probability :
  p_attend_given_rain * p_rain + p_attend_given_sunny * p_sunny + p_attend_given_cloudy * p_cloudy = 0.74 := by
  sorry


end NUMINAMATH_CALUDE_johnny_attend_probability_l2980_298003


namespace NUMINAMATH_CALUDE_triangulation_has_differently_colored_triangle_l2980_298084

structure Triangle where
  vertices : Finset (Fin 3)
  edges : Finset (Fin 3 × Fin 3)

structure Triangulation where
  triangles : Finset Triangle
  colors : Fin 3 → Fin 3

def has_differently_colored_vertices (t : Triangle) (coloring : Fin 3 → Fin 3) : Prop :=
  ∃ (v1 v2 v3 : Fin 3), v1 ∈ t.vertices ∧ v2 ∈ t.vertices ∧ v3 ∈ t.vertices ∧
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    coloring v1 ≠ coloring v2 ∧ coloring v2 ≠ coloring v3 ∧ coloring v1 ≠ coloring v3

theorem triangulation_has_differently_colored_triangle (t : Triangulation) :
  ∃ (small_triangle : Triangle), small_triangle ∈ t.triangles ∧
    has_differently_colored_vertices small_triangle t.colors :=
  sorry

end NUMINAMATH_CALUDE_triangulation_has_differently_colored_triangle_l2980_298084


namespace NUMINAMATH_CALUDE_percentage_decrease_l2980_298085

theorem percentage_decrease (t : ℝ) (x : ℝ) : 
  t = 80 → 
  t * (1 + 0.125) - (t * (1 - x / 100)) = 30 → 
  x = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_l2980_298085


namespace NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l2980_298098

-- Part 1: Equation
theorem equation_solution :
  ∃! x : ℝ, (1 / (x - 3) = 3 + x / (3 - x)) ∧ x = 5 := by sorry

-- Part 2: System of Inequalities
theorem inequalities_solution :
  ∀ x : ℝ, ((x - 1) / 2 < (x + 1) / 3 ∧ x - 3 * (x - 2) ≤ 4) ↔ (1 ≤ x ∧ x < 5) := by sorry

end NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l2980_298098


namespace NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l2980_298030

theorem sum_of_square_roots_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab_c : a + b + c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l2980_298030


namespace NUMINAMATH_CALUDE_intersection_parallel_to_l_l2980_298052

structure GeometricSpace where
  Line : Type
  Plane : Type
  perpendicular : Line → Line → Prop
  perpendicular_line_plane : Line → Plane → Prop
  in_plane : Line → Plane → Prop
  skew : Line → Line → Prop
  intersect : Plane → Plane → Prop
  parallel : Line → Line → Prop
  intersection_line : Plane → Plane → Line

variable (S : GeometricSpace)

theorem intersection_parallel_to_l 
  (m n l : S.Line) (α β : S.Plane) :
  S.skew m n →
  S.perpendicular_line_plane m α →
  S.perpendicular_line_plane n β →
  S.perpendicular l m →
  S.perpendicular l n →
  ¬S.in_plane l α →
  ¬S.in_plane l β →
  S.intersect α β ∧ S.parallel l (S.intersection_line α β) :=
sorry

end NUMINAMATH_CALUDE_intersection_parallel_to_l_l2980_298052


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2980_298002

/-- Represents a tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  pq : ℝ
  pr : ℝ
  ps : ℝ
  qr : ℝ
  qs : ℝ
  rs : ℝ

/-- Calculates the volume of a tetrahedron -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 18 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    pq := 6,
    pr := 4,
    ps := 5,
    qr := 5,
    qs := 7,
    rs := 8
  }
  tetrahedronVolume t = 18 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2980_298002


namespace NUMINAMATH_CALUDE_problem_solution_l2980_298094

theorem problem_solution (a b : ℝ) (h1 : a - 2*b = 0) (h2 : b ≠ 0) :
  (b / (a - b) + 1) * (a^2 - b^2) / a^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2980_298094


namespace NUMINAMATH_CALUDE_jesse_blocks_count_l2980_298014

theorem jesse_blocks_count (building_blocks farm_blocks fence_blocks remaining_blocks : ℕ) 
  (h1 : building_blocks = 80)
  (h2 : farm_blocks = 123)
  (h3 : fence_blocks = 57)
  (h4 : remaining_blocks = 84) :
  building_blocks + farm_blocks + fence_blocks + remaining_blocks = 344 :=
by sorry

end NUMINAMATH_CALUDE_jesse_blocks_count_l2980_298014


namespace NUMINAMATH_CALUDE_three_correct_statements_l2980_298024

theorem three_correct_statements : 
  (0 ∉ (∅ : Set ℕ)) ∧ 
  ((∅ : Set ℕ) ⊆ {1,2}) ∧ 
  ({(x,y) : ℝ × ℝ | 2*x + y = 10 ∧ 3*x - y = 5} ≠ {3,4}) ∧ 
  (∀ A B : Set α, A ⊆ B → A ∩ B = A) :=
by sorry

end NUMINAMATH_CALUDE_three_correct_statements_l2980_298024


namespace NUMINAMATH_CALUDE_sequence_modulo_eight_property_l2980_298066

theorem sequence_modulo_eight_property (s : ℕ → ℕ) 
  (h : ∀ n : ℕ, s (n + 2) = s (n + 1) + s n) : 
  ∃ r : ℤ, ∀ n : ℕ, ¬ (8 ∣ (s n - r)) :=
sorry

end NUMINAMATH_CALUDE_sequence_modulo_eight_property_l2980_298066


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l2980_298070

/-- An ellipse with given properties -/
structure Ellipse where
  /-- First focus of the ellipse -/
  F₁ : ℝ × ℝ
  /-- Second focus of the ellipse -/
  F₂ : ℝ × ℝ
  /-- Length of the chord AB passing through F₂ and perpendicular to x-axis -/
  AB_length : ℝ
  /-- The first focus is at (-1, 0) -/
  F₁_constraint : F₁ = (-1, 0)
  /-- The second focus is at (1, 0) -/
  F₂_constraint : F₂ = (1, 0)
  /-- The length of AB is 3 -/
  AB_length_constraint : AB_length = 3

/-- The equation of the ellipse -/
def ellipse_equation (C : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Theorem stating that the given ellipse satisfies the equation x²/4 + y²/3 = 1 -/
theorem ellipse_equation_proof (C : Ellipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ellipse_equation C p.1 p.2} ↔ 
  (x, y) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = (C.F₂.1, t) ∧ 
  abs (2 * t) = C.AB_length ∧ 
  (p.1 - C.F₁.1)^2 + p.2^2 + (p.1 - C.F₂.1)^2 + p.2^2 = 
  ((p.1 - C.F₁.1)^2 + p.2^2)^(1/2) + ((p.1 - C.F₂.1)^2 + p.2^2)^(1/2)} := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_proof_l2980_298070


namespace NUMINAMATH_CALUDE_equation_factorization_l2980_298006

theorem equation_factorization :
  ∀ x : ℝ, (5*x - 1)^2 = 3*(5*x - 1) ↔ (5*x - 1)*(5*x - 4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_factorization_l2980_298006


namespace NUMINAMATH_CALUDE_thirty_sixth_bead_is_white_l2980_298075

/-- Represents the color of a bead -/
inductive BeadColor
| Black
| White

/-- Defines the sequence of bead colors -/
def beadSequence : ℕ → BeadColor
| 0 => BeadColor.White
| n + 1 => match (n + 1) % 5 with
  | 1 => BeadColor.White
  | 2 => BeadColor.Black
  | 3 => BeadColor.White
  | 4 => BeadColor.Black
  | _ => BeadColor.White

/-- Theorem: The 36th bead in the sequence is white -/
theorem thirty_sixth_bead_is_white : beadSequence 35 = BeadColor.White := by
  sorry

end NUMINAMATH_CALUDE_thirty_sixth_bead_is_white_l2980_298075


namespace NUMINAMATH_CALUDE_problem_solution_l2980_298033

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : Real.sqrt x / Real.sqrt y - Real.sqrt y / Real.sqrt x = 7/12)
  (h2 : x - y = 7) :
  x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2980_298033


namespace NUMINAMATH_CALUDE_no_real_roots_l2980_298091

/-- Given a function f and constants a and b, prove that f(ax + b) has no real roots -/
theorem no_real_roots (f : ℝ → ℝ) (a b : ℝ) : 
  (∀ x, f x = x^2 + 2*x + a) →
  (∀ x, f (b*x) = 9*x - 6*x + 2) →
  (∀ x, f (a*x + b) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_l2980_298091


namespace NUMINAMATH_CALUDE_simplify_expression_l2980_298069

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : a^3 + b^3 = a + b) (h2 : a^2 + b^2 = 3*a + b) :
  a/b + b/a + 1/(a*b) = (9*a + 3*b + 3)/(3*a + b - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2980_298069


namespace NUMINAMATH_CALUDE_basketball_team_selection_l2980_298078

/-- The number of ways to choose k elements from n elements -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the team -/
def total_players : ℕ := 18

/-- The number of quadruplets (who must be included in the starting lineup) -/
def quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def starters : ℕ := 7

theorem basketball_team_selection :
  binomial (total_players - quadruplets) (starters - quadruplets) = 364 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l2980_298078


namespace NUMINAMATH_CALUDE_wendy_furniture_time_l2980_298042

/-- The time Wendy spent putting together all the furniture -/
def total_time (num_chairs num_tables time_per_piece : ℕ) : ℕ :=
  (num_chairs + num_tables) * time_per_piece

/-- Proof that Wendy spent 48 minutes putting together all the furniture -/
theorem wendy_furniture_time :
  total_time 4 4 6 = 48 := by
  sorry

end NUMINAMATH_CALUDE_wendy_furniture_time_l2980_298042


namespace NUMINAMATH_CALUDE_lcm_problem_l2980_298008

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by sorry

end NUMINAMATH_CALUDE_lcm_problem_l2980_298008


namespace NUMINAMATH_CALUDE_binary_multiplication_division_equality_l2980_298096

def binary_to_nat (s : String) : Nat :=
  s.foldl (fun acc c => 2 * acc + c.toNat - '0'.toNat) 0

theorem binary_multiplication_division_equality : 
  (binary_to_nat "1100101" * binary_to_nat "101101" * binary_to_nat "110") / 
  binary_to_nat "100" = binary_to_nat "1111101011011011000" := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_equality_l2980_298096


namespace NUMINAMATH_CALUDE_section_area_of_specific_pyramid_l2980_298060

/-- Regular quadrilateral pyramid with square base -/
structure RegularQuadPyramid where
  base_side : ℝ
  height : ℝ

/-- Plane intersecting the pyramid -/
structure IntersectingPlane where
  angle_with_base : ℝ

/-- The area of the section created by the intersecting plane -/
noncomputable def section_area (p : RegularQuadPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

theorem section_area_of_specific_pyramid :
  let p : RegularQuadPyramid := ⟨8, 9⟩
  let plane : IntersectingPlane := ⟨Real.arctan (3/4)⟩
  section_area p plane = 45 := by sorry

end NUMINAMATH_CALUDE_section_area_of_specific_pyramid_l2980_298060


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2980_298082

/-- The ellipse equation -/
def ellipse (k x y : ℝ) : ℝ := k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1

/-- Theorem stating the range of k for the given ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, (∀ x y : ℝ, ellipse k x y = 0 → ellipse k 0 0 < 0) →
  (0 < |k| ∧ |k| < 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2980_298082


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l2980_298065

theorem magnitude_of_complex_fraction (z : ℂ) : z = (2 - I) / (1 + 2*I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l2980_298065


namespace NUMINAMATH_CALUDE_hamburger_sales_average_l2980_298004

theorem hamburger_sales_average (total_hamburgers : ℕ) (days_in_week : ℕ) 
  (h1 : total_hamburgers = 63)
  (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
sorry

end NUMINAMATH_CALUDE_hamburger_sales_average_l2980_298004


namespace NUMINAMATH_CALUDE_supervisors_per_bus_l2980_298020

theorem supervisors_per_bus (total_buses : ℕ) (total_supervisors : ℕ) 
  (h1 : total_buses = 7) 
  (h2 : total_supervisors = 21) : 
  total_supervisors / total_buses = 3 := by
  sorry

end NUMINAMATH_CALUDE_supervisors_per_bus_l2980_298020


namespace NUMINAMATH_CALUDE_factor_expression_l2980_298083

theorem factor_expression (x : ℝ) : 3*x*(x+1) + 7*(x+1) = (3*x+7)*(x+1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2980_298083


namespace NUMINAMATH_CALUDE_erasers_remaining_l2980_298039

/-- The number of erasers left in a box after some are removed -/
def erasers_left (initial : ℕ) (removed : ℕ) : ℕ := initial - removed

/-- Theorem: Given 69 initial erasers and 54 removed, 15 erasers are left -/
theorem erasers_remaining : erasers_left 69 54 = 15 := by
  sorry

end NUMINAMATH_CALUDE_erasers_remaining_l2980_298039


namespace NUMINAMATH_CALUDE_three_digit_special_property_l2980_298081

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ 0 ∧
    b < 10 ∧
    c < 10 ∧
    3 * a * (10 * b + c) = n

theorem three_digit_special_property :
  {n : ℕ | is_valid_number n} = {150, 240, 735} :=
sorry

end NUMINAMATH_CALUDE_three_digit_special_property_l2980_298081


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l2980_298011

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) : 
  cube_side = 5 → cylinder_radius = 1.5 → 
  cube_side^3 - π * cylinder_radius^2 * cube_side = 125 - 11.25 * π := by
  sorry

#check remaining_cube_volume

end NUMINAMATH_CALUDE_remaining_cube_volume_l2980_298011


namespace NUMINAMATH_CALUDE_chris_candy_distribution_l2980_298034

/-- Given that Chris gave each friend 12 pieces of candy and gave away a total of 420 pieces of candy,
    prove that the number of friends Chris gave candy to is 35. -/
theorem chris_candy_distribution (candy_per_friend : ℕ) (total_candy : ℕ) (num_friends : ℕ) :
  candy_per_friend = 12 →
  total_candy = 420 →
  num_friends * candy_per_friend = total_candy →
  num_friends = 35 := by
sorry

end NUMINAMATH_CALUDE_chris_candy_distribution_l2980_298034


namespace NUMINAMATH_CALUDE_sandwich_ratio_l2980_298074

/-- The number of sandwiches Samson ate at lunch on Monday -/
def lunch_sandwiches : ℕ := 3

/-- The number of sandwiches Samson ate at dinner on Monday -/
def dinner_sandwiches : ℕ := 6

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_sandwiches : ℕ := 1

/-- The difference in total sandwiches eaten between Monday and Tuesday -/
def sandwich_difference : ℕ := 8

theorem sandwich_ratio :
  (dinner_sandwiches : ℚ) / lunch_sandwiches = 2 ∧
  lunch_sandwiches + dinner_sandwiches = tuesday_sandwiches + sandwich_difference :=
sorry

end NUMINAMATH_CALUDE_sandwich_ratio_l2980_298074


namespace NUMINAMATH_CALUDE_reading_time_difference_l2980_298000

-- Define the reading speeds and book length
def xanthia_speed : ℝ := 150  -- pages per hour
def molly_speed : ℝ := 75     -- pages per hour
def book_length : ℝ := 300    -- pages

-- Define the time difference in minutes
def time_difference : ℝ := 120 -- minutes

-- Theorem statement
theorem reading_time_difference :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = time_difference := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l2980_298000


namespace NUMINAMATH_CALUDE_train_crossing_time_l2980_298029

/-- Proves that a train with given specifications takes 20 seconds to cross a platform -/
theorem train_crossing_time (train_length : ℝ) (platform_length : ℝ) (passing_time : ℝ) :
  train_length = 180 →
  platform_length = 270 →
  passing_time = 8 →
  let train_speed := train_length / passing_time
  let total_distance := train_length + platform_length
  let crossing_time := total_distance / train_speed
  crossing_time = 20 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2980_298029


namespace NUMINAMATH_CALUDE_abs_T_equals_1024_l2980_298001

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^19 - (1 - i)^19

-- Theorem statement
theorem abs_T_equals_1024 : Complex.abs T = 1024 := by
  sorry

end NUMINAMATH_CALUDE_abs_T_equals_1024_l2980_298001


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2980_298087

/-- Given a hyperbola E and a parabola C with specific properties, 
    prove that the eccentricity of E is in the range (1, 3√2/4] -/
theorem hyperbola_eccentricity_range 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (E : Set (ℝ × ℝ)) 
  (C : Set (ℝ × ℝ)) 
  (A : ℝ × ℝ) 
  (F : ℝ × ℝ) 
  (P : ℝ × ℝ) 
  (hE : E = {(x, y) | x^2/a^2 - y^2/b^2 = 1})
  (hC : C = {(x, y) | y^2 = 8*a*x})
  (hA : A = (a, 0))
  (hF : F = (2*a, 0))
  (hP : P ∈ {(x, y) | y = (b/a)*x})  -- P is on the asymptote of E
  (hPerp : (P.1 - A.1) * (P.1 - F.1) + (P.2 - A.2) * (P.2 - F.2) = 0)  -- AP ⊥ FP
  : 1 < Real.sqrt (a^2 + b^2) / a ∧ Real.sqrt (a^2 + b^2) / a ≤ 3 * Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2980_298087


namespace NUMINAMATH_CALUDE_smallest_period_five_cycles_l2980_298067

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def completes_n_cycles (f : ℝ → ℝ) (n : ℕ) (a b : ℝ) : Prop :=
  ∃ T > 0, is_periodic f T ∧ n * T = b - a

theorem smallest_period_five_cycles (f : ℝ → ℝ) 
  (h : completes_n_cycles f 5 0 (2 * Real.pi)) :
  ∃ T > 0, is_periodic f T ∧ 
    (∀ T' > 0, is_periodic f T' → T ≤ T') ∧
    T = 2 * Real.pi / 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_period_five_cycles_l2980_298067


namespace NUMINAMATH_CALUDE_leadership_selection_ways_l2980_298073

/-- The number of ways to choose a president, vice-president, and committee from a group. -/
def choose_leadership (n : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) 3)

/-- The problem statement as a theorem. -/
theorem leadership_selection_ways :
  choose_leadership 10 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_leadership_selection_ways_l2980_298073


namespace NUMINAMATH_CALUDE_complex_magnitude_three_fifths_minus_four_sevenths_i_l2980_298089

theorem complex_magnitude_three_fifths_minus_four_sevenths_i :
  Complex.abs (3/5 - (4/7)*Complex.I) = 29/35 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_three_fifths_minus_four_sevenths_i_l2980_298089


namespace NUMINAMATH_CALUDE_negative_three_halves_less_than_negative_one_l2980_298076

theorem negative_three_halves_less_than_negative_one :
  -((3 : ℚ) / 2) < -1 := by sorry

end NUMINAMATH_CALUDE_negative_three_halves_less_than_negative_one_l2980_298076


namespace NUMINAMATH_CALUDE_negation_existence_inequality_l2980_298056

theorem negation_existence_inequality :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_inequality_l2980_298056


namespace NUMINAMATH_CALUDE_jellybean_count_l2980_298047

theorem jellybean_count (steve matt matilda katy : ℕ) : 
  steve = 84 →
  matt = 10 * steve →
  matilda = matt / 2 →
  katy = 3 * matilda →
  katy = matt / 2 →
  katy = 1260 := by
sorry

end NUMINAMATH_CALUDE_jellybean_count_l2980_298047


namespace NUMINAMATH_CALUDE_division_subtraction_l2980_298036

theorem division_subtraction : ((-150) / (-50)) - 15 = -12 := by
  sorry

end NUMINAMATH_CALUDE_division_subtraction_l2980_298036


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l2980_298079

theorem neither_sufficient_nor_necessary (p q : Prop) :
  ¬(((¬p ∧ ¬q) → (p ∨ q)) ∧ ((p ∨ q) → (¬p ∧ ¬q))) :=
by sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l2980_298079


namespace NUMINAMATH_CALUDE_shape_D_symmetric_l2980_298053

-- Define the shape type
inductive Shape
| A
| B
| C
| D
| E

-- Define the property of being symmetric with respect to a horizontal line
def isSymmetric (s1 s2 : Shape) : Prop := sorry

-- Define the given shape
def givenShape : Shape := sorry

-- Theorem statement
theorem shape_D_symmetric : 
  isSymmetric givenShape Shape.D := by sorry

end NUMINAMATH_CALUDE_shape_D_symmetric_l2980_298053


namespace NUMINAMATH_CALUDE_brick_height_l2980_298048

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Proves that the height of a brick with given dimensions and surface area is 3 cm -/
theorem brick_height (l w sa : ℝ) (hl : l = 10) (hw : w = 4) (hsa : sa = 164) :
  ∃ h : ℝ, h = 3 ∧ surface_area l w h = sa :=
by sorry

end NUMINAMATH_CALUDE_brick_height_l2980_298048


namespace NUMINAMATH_CALUDE_expand_product_l2980_298041

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2980_298041


namespace NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l2980_298026

theorem smallest_solution_for_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 10 → x ≤ y) ∧
  ⌊x^2⌋ - x * ⌊x⌋ = 10 ∧
  x = 131 / 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l2980_298026


namespace NUMINAMATH_CALUDE_selection_ways_l2980_298009

def total_students : ℕ := 10
def selected_students : ℕ := 4
def specific_students : ℕ := 2

theorem selection_ways : 
  (Nat.choose total_students selected_students - 
   Nat.choose (total_students - specific_students) selected_students) = 140 :=
by sorry

end NUMINAMATH_CALUDE_selection_ways_l2980_298009


namespace NUMINAMATH_CALUDE_average_book_price_l2980_298063

/-- The average price of books Sandy bought, given the number of books and total cost from two shops. -/
theorem average_book_price 
  (shop1_books : ℕ) 
  (shop1_cost : ℕ) 
  (shop2_books : ℕ) 
  (shop2_cost : ℕ) 
  (h1 : shop1_books = 65) 
  (h2 : shop1_cost = 1480) 
  (h3 : shop2_books = 55) 
  (h4 : shop2_cost = 920) : 
  (shop1_cost + shop2_cost) / (shop1_books + shop2_books) = 20 := by
sorry

end NUMINAMATH_CALUDE_average_book_price_l2980_298063


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2980_298051

theorem complex_division_simplification : 
  (2 + Complex.I) / (1 - 2 * Complex.I) = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2980_298051


namespace NUMINAMATH_CALUDE_space_shuttle_speed_l2980_298027

-- Define the speed in kilometers per hour
def speed_kmh : ℝ := 14400

-- Define the conversion factor from hours to seconds
def seconds_per_hour : ℝ := 3600

-- The theorem to prove
theorem space_shuttle_speed : speed_kmh / seconds_per_hour = 4 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_l2980_298027


namespace NUMINAMATH_CALUDE_xyz_product_is_27_l2980_298023

theorem xyz_product_is_27 
  (x y z : ℂ) 
  (h1 : x * y + 3 * y = -9)
  (h2 : y * z + 3 * z = -9)
  (h3 : z * x + 3 * x = -9) :
  x * y * z = 27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_is_27_l2980_298023


namespace NUMINAMATH_CALUDE_range_of_expression_l2980_298045

def line_equation (x : ℝ) : ℝ := -2 * x + 8

theorem range_of_expression (x₁ y₁ : ℝ) :
  y₁ = line_equation x₁ →
  x₁ ∈ Set.Icc 2 5 →
  (y₁ + 1) / (x₁ + 1) ∈ Set.Icc (-1/6) (5/3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l2980_298045


namespace NUMINAMATH_CALUDE_inequality_representation_l2980_298028

/-- 
Theorem: The inequality 3x - 2 > 0 correctly represents the statement 
"x is three times the difference between 2".
-/
theorem inequality_representation (x : ℝ) : 
  (3 * x - 2 > 0) ↔ (∃ y : ℝ, x = 3 * y ∧ y > 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_representation_l2980_298028


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_smallest_satisfying_number_value_l2980_298035

def is_perfect_power (n : ℕ) (k : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ k

def satisfies_conditions (N : ℕ) : Prop :=
  is_perfect_power (N / 2) 2 ∧
  is_perfect_power (N / 3) 3 ∧
  is_perfect_power (N / 5) 5

theorem smallest_satisfying_number :
  ∃ N : ℕ, satisfies_conditions N ∧
    ∀ M : ℕ, satisfies_conditions M → N ≤ M :=
by
  -- The proof goes here
  sorry

theorem smallest_satisfying_number_value :
  ∃ N : ℕ, N = 2^15 * 3^10 * 5^6 ∧ satisfies_conditions N ∧
    ∀ M : ℕ, satisfies_conditions M → N ≤ M :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_smallest_satisfying_number_value_l2980_298035


namespace NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_l2980_298062

def A (m : ℝ) : Set ℝ := {0, m^2}
def B : Set ℝ := {1, 2}

theorem m_equals_one_sufficient_not_necessary :
  (∃ m : ℝ, A m ∩ B = {1} ∧ m ≠ 1) ∧
  (∀ m : ℝ, m = 1 → A m ∩ B = {1}) :=
sorry

end NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_l2980_298062


namespace NUMINAMATH_CALUDE_n_has_nine_digits_l2980_298046

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect fourth power -/
axiom n_sq_fourth_power : ∃ k : ℕ, n^2 = k^4

/-- n^4 is a perfect cube -/
axiom n_fourth_cube : ∃ k : ℕ, n^4 = k^3

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m < n → ¬(30 ∣ m ∧ (∃ k : ℕ, m^2 = k^4) ∧ (∃ k : ℕ, m^4 = k^3))

/-- The number of digits in n -/
def num_digits (x : ℕ) : ℕ := sorry

/-- Theorem stating that n has 9 digits -/
theorem n_has_nine_digits : num_digits n = 9 := by sorry

end NUMINAMATH_CALUDE_n_has_nine_digits_l2980_298046


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l2980_298055

theorem reciprocal_sum_of_quadratic_roots (γ δ : ℝ) : 
  (∃ r s : ℝ, 7 * r^2 + 5 * r + 3 = 0 ∧ 
              7 * s^2 + 5 * s + 3 = 0 ∧ 
              γ = 1 / r ∧ 
              δ = 1 / s) → 
  γ + δ = -5/3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l2980_298055


namespace NUMINAMATH_CALUDE_determine_origin_l2980_298013

/-- Given two points A and B in a 2D coordinate system, we can uniquely determine the origin. -/
theorem determine_origin (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (3, 1)) :
  ∃! O : ℝ × ℝ, O = (0, 0) ∧ 
  (O.1 - A.1) ^ 2 + (O.2 - A.2) ^ 2 = (O.1 - B.1) ^ 2 + (O.2 - B.2) ^ 2 ∧
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = (O.1 - A.1) ^ 2 + (O.2 - A.2) ^ 2 + (O.1 - B.1) ^ 2 + (O.2 - B.2) ^ 2 :=
by sorry


end NUMINAMATH_CALUDE_determine_origin_l2980_298013


namespace NUMINAMATH_CALUDE_min_values_theorem_l2980_298038

/-- Given positive real numbers a and b satisfying 4a + b = ab, 
    prove the following statements about their minimum values. -/
theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4*a + b = a*b) :
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4*a₀ + b₀ = a₀*b₀ ∧ ∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a₀*b₀ ≤ a'*b') ∧
  (∃ (a₁ b₁ : ℝ), a₁ > 0 ∧ b₁ > 0 ∧ 4*a₁ + b₁ = a₁*b₁ ∧ ∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a₁ + b₁ ≤ a' + b') ∧
  (∃ (a₂ b₂ : ℝ), a₂ > 0 ∧ b₂ > 0 ∧ 4*a₂ + b₂ = a₂*b₂ ∧ ∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → 1/a₂^2 + 4/b₂^2 ≤ 1/a'^2 + 4/b'^2) ∧
  (∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a'*b' ≥ 16) ∧
  (∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a' + b' ≥ 9) ∧
  (∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → 1/a'^2 + 4/b'^2 ≥ 1/5) :=
by sorry

end NUMINAMATH_CALUDE_min_values_theorem_l2980_298038


namespace NUMINAMATH_CALUDE_marble_probabilities_and_total_l2980_298017

/-- The probability of drawing a white marble -/
def prob_white : ℚ := 1/4

/-- The probability of drawing a green marble -/
def prob_green : ℚ := 2/7

/-- The probability of drawing either a red or blue marble -/
def prob_red_or_blue : ℚ := 13/28

/-- The total number of marbles in the box -/
def total_marbles : ℕ := 28

/-- Theorem stating that the given probabilities sum to 1 and the total number of marbles is 28 -/
theorem marble_probabilities_and_total : 
  prob_white + prob_green + prob_red_or_blue = 1 ∧ total_marbles = 28 := by sorry

end NUMINAMATH_CALUDE_marble_probabilities_and_total_l2980_298017


namespace NUMINAMATH_CALUDE_expression_value_l2980_298031

theorem expression_value (x : ℝ) (h : x^2 - 5*x - 2006 = 0) :
  ((x-2)^3 - (x-1)^2 + 1) / (x-2) = 2010 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2980_298031


namespace NUMINAMATH_CALUDE_both_samples_stratified_l2980_298005

/-- Represents a sample of students -/
structure Sample :=
  (numbers : List Nat)

/-- Represents the school population -/
structure School :=
  (total_students : Nat)
  (first_grade : Nat)
  (second_grade : Nat)
  (third_grade : Nat)

/-- Checks if a sample is valid for stratified sampling -/
def is_valid_stratified_sample (school : School) (sample : Sample) : Prop :=
  sample.numbers.length = 10 ∧
  sample.numbers.all (λ n => n > 0 ∧ n ≤ school.total_students) ∧
  sample.numbers.Nodup

/-- The given school configuration -/
def junior_high : School :=
  { total_students := 300
  , first_grade := 120
  , second_grade := 90
  , third_grade := 90 }

/-- Sample ① -/
def sample1 : Sample :=
  { numbers := [7, 37, 67, 97, 127, 157, 187, 217, 247, 277] }

/-- Sample ③ -/
def sample3 : Sample :=
  { numbers := [11, 41, 71, 101, 131, 161, 191, 221, 251, 281] }

theorem both_samples_stratified :
  is_valid_stratified_sample junior_high sample1 ∧
  is_valid_stratified_sample junior_high sample3 := by
  sorry

end NUMINAMATH_CALUDE_both_samples_stratified_l2980_298005
