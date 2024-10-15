import Mathlib

namespace NUMINAMATH_CALUDE_max_andy_consumption_l3292_329270

def total_cookies : ℕ := 36

def cookie_distribution (andy alexa ann : ℕ) : Prop :=
  ∃ k : ℕ+, alexa = k * andy ∧ ann = 2 * andy ∧ andy + alexa + ann = total_cookies

def max_andy_cookies : ℕ := 9

theorem max_andy_consumption :
  ∀ andy alexa ann : ℕ,
    cookie_distribution andy alexa ann →
    andy ≤ max_andy_cookies :=
by sorry

end NUMINAMATH_CALUDE_max_andy_consumption_l3292_329270


namespace NUMINAMATH_CALUDE_largest_integer_x_l3292_329264

theorem largest_integer_x : ∃ x : ℤ, 
  (∀ y : ℤ, (7 - 3 * y > 20 ∧ y ≥ -10) → y ≤ x) ∧ 
  (7 - 3 * x > 20 ∧ x ≥ -10) ∧ 
  x = -5 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_x_l3292_329264


namespace NUMINAMATH_CALUDE_f_at_negative_five_l3292_329288

/-- Given a function f(x) = x^2 + 2x - 3, prove that f(-5) = 12 -/
theorem f_at_negative_five (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x - 3) : f (-5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_five_l3292_329288


namespace NUMINAMATH_CALUDE_regression_line_not_always_through_point_l3292_329254

/-- A sample data point in a regression analysis -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Linear regression equation -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Check if a point lies on a line defined by a linear regression equation -/
def pointOnLine (p : DataPoint) (reg : LinearRegression) : Prop :=
  p.y = reg.b * p.x + reg.a

/-- Theorem stating that it's not necessarily true that a linear regression line passes through at least one sample point -/
theorem regression_line_not_always_through_point :
  ∃ (n : ℕ) (data : Fin n → DataPoint) (reg : LinearRegression),
    ∀ i : Fin n, ¬(pointOnLine (data i) reg) :=
sorry

end NUMINAMATH_CALUDE_regression_line_not_always_through_point_l3292_329254


namespace NUMINAMATH_CALUDE_triangle_value_l3292_329289

theorem triangle_value (Δ q : ℤ) 
  (h1 : 3 * Δ * q = 63) 
  (h2 : 7 * (Δ + q) = 161) : 
  Δ = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l3292_329289


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3292_329247

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3292_329247


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3292_329291

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 7*x - 5 = 0 ↔ (x + 7/2)^2 = 69/4 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3292_329291


namespace NUMINAMATH_CALUDE_tenth_letter_shift_l3292_329285

def shift_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem tenth_letter_shift :
  ∀ (letter : Char),
  (shift_sum 10) % 26 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tenth_letter_shift_l3292_329285


namespace NUMINAMATH_CALUDE_b_eventually_constant_iff_square_l3292_329218

/-- The greatest integer m such that m^2 ≤ n -/
def m (n : ℕ) : ℕ := Nat.sqrt n

/-- d(n) = n - m^2, where m is the greatest integer such that m^2 ≤ n -/
def d (n : ℕ) : ℕ := n - (m n)^2

/-- The sequence b_i defined by b_{k+1} = b_k + d(b_k) -/
def b : ℕ → ℕ → ℕ
  | b_0, 0 => b_0
  | b_0, k + 1 => b b_0 k + d (b b_0 k)

/-- A sequence is eventually constant if there exists an N such that
    for all i ≥ N, the i-th term equals the N-th term -/
def EventuallyConstant (s : ℕ → ℕ) : Prop :=
  ∃ N, ∀ i, N ≤ i → s i = s N

/-- Main theorem: b_i is eventually constant iff b_0 is a perfect square -/
theorem b_eventually_constant_iff_square (b_0 : ℕ) :
  EventuallyConstant (b b_0) ↔ ∃ k, b_0 = k^2 := by sorry

end NUMINAMATH_CALUDE_b_eventually_constant_iff_square_l3292_329218


namespace NUMINAMATH_CALUDE_focus_coordinates_l3292_329215

/-- A parabola is defined by the equation x^2 = 4y -/
structure Parabola where
  equation : ∀ x y : ℝ, x^2 = 4*y

/-- The focus of a parabola is a point (h, k) on its axis of symmetry -/
structure Focus (p : Parabola) where
  h : ℝ
  k : ℝ

/-- Theorem: The focus of the parabola x^2 = 4y has coordinates (0, 1) -/
theorem focus_coordinates (p : Parabola) : 
  ∃ f : Focus p, f.h = 0 ∧ f.k = 1 := by
  sorry

end NUMINAMATH_CALUDE_focus_coordinates_l3292_329215


namespace NUMINAMATH_CALUDE_gcd_binomial_integrality_l3292_329252

theorem gcd_binomial_integrality (m n : ℕ) (h1 : 1 ≤ m) (h2 : m ≤ n) :
  ∃ (a b : ℤ), (Nat.gcd m n : ℚ) / n * Nat.choose n m = a * Nat.choose (n-1) (m-1) + b * Nat.choose n m := by
  sorry

end NUMINAMATH_CALUDE_gcd_binomial_integrality_l3292_329252


namespace NUMINAMATH_CALUDE_equation_identity_l3292_329230

theorem equation_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_equation_identity_l3292_329230


namespace NUMINAMATH_CALUDE_lg_calculation_l3292_329284

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem lg_calculation : (lg 2)^2 + lg 20 * lg 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_calculation_l3292_329284


namespace NUMINAMATH_CALUDE_frozen_yoghurt_cost_l3292_329260

theorem frozen_yoghurt_cost (ice_cream_quantity : ℕ) (frozen_yoghurt_quantity : ℕ) 
  (ice_cream_cost : ℕ) (ice_cream_total : ℕ) (price_difference : ℕ) :
  ice_cream_quantity = 10 →
  frozen_yoghurt_quantity = 4 →
  ice_cream_cost = 4 →
  ice_cream_total = ice_cream_quantity * ice_cream_cost →
  ice_cream_total = price_difference + (frozen_yoghurt_quantity * 1) →
  1 = (ice_cream_total - price_difference) / frozen_yoghurt_quantity :=
by
  sorry

end NUMINAMATH_CALUDE_frozen_yoghurt_cost_l3292_329260


namespace NUMINAMATH_CALUDE_gan_is_axisymmetric_l3292_329208

/-- A figure is axisymmetric if it can be folded along a line so that the parts on both sides of the line coincide. -/
def is_axisymmetric (figure : Type*) : Prop :=
  ∃ (line : Set figure), ∀ (point : figure), 
    ∃ (reflected_point : figure), 
      point ≠ reflected_point ∧ 
      (point ∈ line ∨ reflected_point ∈ line) ∧
      (∀ (p : figure), p ∉ line → (p = point ↔ p = reflected_point))

/-- The Chinese character "干" -/
def gan : Type* := sorry

/-- Theorem: The Chinese character "干" is an axisymmetric figure -/
theorem gan_is_axisymmetric : is_axisymmetric gan := by sorry

end NUMINAMATH_CALUDE_gan_is_axisymmetric_l3292_329208


namespace NUMINAMATH_CALUDE_class_average_problem_l3292_329222

theorem class_average_problem (total_students : Nat) (high_score_students : Nat) 
  (zero_score_students : Nat) (high_score : Nat) (class_average : Rat) :
  total_students = 25 →
  high_score_students = 3 →
  zero_score_students = 3 →
  high_score = 95 →
  class_average = 45.6 →
  let remaining_students := total_students - high_score_students - zero_score_students
  let total_score := (total_students : Rat) * class_average
  let high_score_total := (high_score_students : Rat) * high_score
  let remaining_average := (total_score - high_score_total) / remaining_students
  remaining_average = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l3292_329222


namespace NUMINAMATH_CALUDE_monotonic_condition_l3292_329205

/-- A function f is monotonic on an interval [a, b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The main theorem stating the condition for the function to be monotonic. -/
theorem monotonic_condition (a : ℝ) :
  (IsMonotonic (fun x => -x^2 + 4*a*x) 2 4) ↔ (a ≤ 1 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_condition_l3292_329205


namespace NUMINAMATH_CALUDE_sum_of_coordinates_is_60_l3292_329279

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions for the points
def satisfies_conditions (p : Point) : Prop :=
  (abs (p.y - 10) = 4) ∧
  ((p.x - 5)^2 + (p.y - 10)^2 = 12^2)

-- Theorem statement
theorem sum_of_coordinates_is_60 :
  ∀ (p1 p2 p3 p4 : Point),
    satisfies_conditions p1 →
    satisfies_conditions p2 →
    satisfies_conditions p3 →
    satisfies_conditions p4 →
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 →
    p1.x + p1.y + p2.x + p2.y + p3.x + p3.y + p4.x + p4.y = 60 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_is_60_l3292_329279


namespace NUMINAMATH_CALUDE_maria_ate_two_cookies_l3292_329272

/-- Given Maria's cookie distribution, prove she ate 2 cookies. -/
theorem maria_ate_two_cookies
  (initial_cookies : ℕ)
  (friend_cookies : ℕ)
  (final_cookies : ℕ)
  (h1 : initial_cookies = 19)
  (h2 : friend_cookies = 5)
  (h3 : final_cookies = 5)
  (h4 : ∃ (family_cookies : ℕ), 
    2 * family_cookies = initial_cookies - friend_cookies) :
  initial_cookies - friend_cookies - 
    ((initial_cookies - friend_cookies) / 2) - final_cookies = 2 :=
by sorry


end NUMINAMATH_CALUDE_maria_ate_two_cookies_l3292_329272


namespace NUMINAMATH_CALUDE_correct_num_tables_l3292_329209

/-- The number of tables in the lunchroom -/
def num_tables : ℕ := 6

/-- The initial number of students per table -/
def initial_students_per_table : ℚ := 6

/-- The desired number of students per table -/
def desired_students_per_table : ℚ := 17 / 3

/-- Theorem stating that the number of tables is correct -/
theorem correct_num_tables :
  (initial_students_per_table * num_tables : ℚ) =
  (desired_students_per_table * num_tables : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_correct_num_tables_l3292_329209


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3292_329283

theorem decimal_to_fraction : (2.24 : ℚ) = 56 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3292_329283


namespace NUMINAMATH_CALUDE_number_times_three_equals_33_l3292_329261

theorem number_times_three_equals_33 : ∃ x : ℝ, 3 * x = 33 ∧ x = 11 := by sorry

end NUMINAMATH_CALUDE_number_times_three_equals_33_l3292_329261


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3292_329240

theorem absolute_value_simplification (a b : ℚ) (ha : a < 0) (hb : b > 0) :
  |a - b| + b = -a + 2*b := by sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3292_329240


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3292_329246

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x) / (3 * x - 1) > 1 ↔ 1 / 3 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3292_329246


namespace NUMINAMATH_CALUDE_log_equation_solution_l3292_329292

theorem log_equation_solution (x : ℝ) (hx : x > 0) :
  Real.log x / Real.log 3 + Real.log 3 / Real.log x - 2 * (Real.log x / Real.log 3) * (Real.log 3 / Real.log x) = 1/2 ↔ 
  x = Real.sqrt 3 ∨ x = 9 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3292_329292


namespace NUMINAMATH_CALUDE_max_sum_squares_given_sum_cubes_l3292_329231

theorem max_sum_squares_given_sum_cubes 
  (a b c d : ℝ) 
  (h : a^3 + b^3 + c^3 + d^3 = 8) : 
  ∃ (m : ℝ), m = 4 ∧ ∀ (x y z w : ℝ), x^3 + y^3 + z^3 + w^3 = 8 → x^2 + y^2 + z^2 + w^2 ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_sum_squares_given_sum_cubes_l3292_329231


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l3292_329212

/-- The number of zeros between the decimal point and the first non-zero digit when 7/8000 is written as a decimal -/
def zeros_before_first_nonzero : ℕ :=
  3

/-- The fraction we're considering -/
def fraction : ℚ :=
  7 / 8000

theorem zeros_before_first_nonzero_of_fraction :
  zeros_before_first_nonzero = 3 ∧ fraction = 7 / 8000 := by
  sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l3292_329212


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_5_l3292_329286

theorem x_plus_2y_equals_5 (x y : ℝ) 
  (h1 : (x + y) / 3 = 1) 
  (h2 : 2 * x + y = 4) : 
  x + 2 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_5_l3292_329286


namespace NUMINAMATH_CALUDE_multiples_properties_l3292_329299

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k) 
  (hb : ∃ m : ℤ, b = 6 * m) : 
  (∃ n : ℤ, b = 3 * n) ∧ 
  (∃ p : ℤ, a - b = 3 * p) := by
sorry

end NUMINAMATH_CALUDE_multiples_properties_l3292_329299


namespace NUMINAMATH_CALUDE_games_given_to_neil_l3292_329259

theorem games_given_to_neil (henry_initial : ℕ) (neil_initial : ℕ) (games_given : ℕ) : 
  henry_initial = 33 →
  neil_initial = 2 →
  henry_initial - games_given = 4 * (neil_initial + games_given) →
  games_given = 5 := by
sorry

end NUMINAMATH_CALUDE_games_given_to_neil_l3292_329259


namespace NUMINAMATH_CALUDE_salary_expenditure_percentage_l3292_329206

theorem salary_expenditure_percentage (initial_salary : ℝ) 
  (house_rent_percentage : ℝ) (education_percentage : ℝ) 
  (final_amount : ℝ) : 
  initial_salary = 2125 →
  house_rent_percentage = 20 →
  education_percentage = 10 →
  final_amount = 1377 →
  let remaining_after_rent := initial_salary * (1 - house_rent_percentage / 100)
  let remaining_after_education := remaining_after_rent * (1 - education_percentage / 100)
  let clothes_percentage := (remaining_after_education - final_amount) / remaining_after_education * 100
  clothes_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_salary_expenditure_percentage_l3292_329206


namespace NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l3292_329256

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio -/
def originalRatio : RecipeRatio :=
  { flour := 11, water := 8, sugar := 1 }

/-- The new recipe ratio -/
def newRatio : RecipeRatio :=
  { flour := 22, water := 8, sugar := 1 }

/-- The amount of water in the new recipe -/
def newWaterAmount : ℚ := 4

/-- Theorem stating that the amount of sugar in the new recipe is 0.5 cups -/
theorem sugar_amount_in_new_recipe :
  (newWaterAmount * newRatio.sugar) / newRatio.water = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l3292_329256


namespace NUMINAMATH_CALUDE_even_number_less_than_square_l3292_329242

theorem even_number_less_than_square (m : ℕ) (h1 : m > 1) (h2 : Even m) : m < m^2 := by
  sorry

end NUMINAMATH_CALUDE_even_number_less_than_square_l3292_329242


namespace NUMINAMATH_CALUDE_function_zeros_sum_l3292_329290

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.exp (1 - x) - a * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x + Real.exp (1 - x) - a

theorem function_zeros_sum (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : g a x₁ = 0) 
  (h₂ : g a x₂ = 0) 
  (h₃ : f a x₁ + f a x₂ = -4) : 
  a = 4 := by sorry

end NUMINAMATH_CALUDE_function_zeros_sum_l3292_329290


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3292_329210

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 10 ∧ ∀ m : ℤ, |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3292_329210


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3292_329273

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 → 
  E = 2 * F + 15 → 
  D + E + F = 180 → 
  F = 30 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3292_329273


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3292_329219

theorem rectangle_dimension_change (w l : ℝ) (h_w_pos : w > 0) (h_l_pos : l > 0) :
  let new_w := 1.4 * w
  let new_l := l / 1.4
  let area := w * l
  let new_area := new_w * new_l
  new_area = area ∧ (1 - new_l / l) * 100 = 100 * (1 - 1 / 1.4) := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3292_329219


namespace NUMINAMATH_CALUDE_mia_weight_l3292_329236

/-- 
Given two people, Anna and Mia, with the following conditions:
1. The sum of their weights is 220 pounds.
2. The difference between Mia's weight and Anna's weight is twice Anna's weight.
This theorem proves that Mia's weight is 165 pounds.
-/
theorem mia_weight (anna_weight mia_weight : ℝ) 
  (sum_condition : anna_weight + mia_weight = 220)
  (difference_condition : mia_weight - anna_weight = 2 * anna_weight) :
  mia_weight = 165 := by
sorry

end NUMINAMATH_CALUDE_mia_weight_l3292_329236


namespace NUMINAMATH_CALUDE_base_8_to_10_fraction_l3292_329234

theorem base_8_to_10_fraction (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- c and d are base-10 digits
  (5 * 8^2 + 6 * 8 + 3 = 3 * 100 + c * 10 + d) →  -- 563_8 = 3cd_10
  (c * d) / 12 = 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_base_8_to_10_fraction_l3292_329234


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3292_329202

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k ^ n

/-- Proof that distributing 5 distinguishable balls into 4 distinguishable boxes results in 1024 ways -/
theorem distribute_five_balls_four_boxes : distribute 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3292_329202


namespace NUMINAMATH_CALUDE_sum_of_series_l3292_329251

theorem sum_of_series : 
  (∑' n : ℕ, (4 * n + 1 : ℝ) / (3 : ℝ) ^ n) = 7 / 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_series_l3292_329251


namespace NUMINAMATH_CALUDE_triangle_area_l3292_329203

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : Real.cos C = -3/5) :
  let S := (1/2) * a * b * Real.sin C
  S = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3292_329203


namespace NUMINAMATH_CALUDE_tin_in_new_alloy_tin_amount_is_correct_l3292_329296

/-- The amount of tin in a new alloy formed by mixing two alloys -/
theorem tin_in_new_alloy (alloy_a_mass : ℝ) (alloy_b_mass : ℝ) 
  (lead_tin_ratio_a : ℝ × ℝ) (tin_copper_ratio_b : ℝ × ℝ) : ℝ :=
  let tin_in_a := (lead_tin_ratio_a.2 / (lead_tin_ratio_a.1 + lead_tin_ratio_a.2)) * alloy_a_mass
  let tin_in_b := (tin_copper_ratio_b.1 / (tin_copper_ratio_b.1 + tin_copper_ratio_b.2)) * alloy_b_mass
  tin_in_a + tin_in_b

/-- The amount of tin in the new alloy is 139.5 kg -/
theorem tin_amount_is_correct : 
  tin_in_new_alloy 120 180 (2, 3) (3, 5) = 139.5 := by
  sorry

end NUMINAMATH_CALUDE_tin_in_new_alloy_tin_amount_is_correct_l3292_329296


namespace NUMINAMATH_CALUDE_point_distance_to_y_axis_l3292_329287

theorem point_distance_to_y_axis (a : ℝ) : 
  (a + 3 > 0) →  -- Point is in the first quadrant (x-coordinate is positive)
  (a > 0) →      -- Point is in the first quadrant (y-coordinate is positive)
  (a + 3 = 5) →  -- Distance to y-axis is 5
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_point_distance_to_y_axis_l3292_329287


namespace NUMINAMATH_CALUDE_equal_area_triangles_l3292_329243

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem equal_area_triangles :
  triangleArea 25 25 30 = triangleArea 25 25 40 := by sorry

end NUMINAMATH_CALUDE_equal_area_triangles_l3292_329243


namespace NUMINAMATH_CALUDE_third_term_range_l3292_329235

/-- A sequence of positive real numbers satisfying certain conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (a 1 = 1) ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1)^2 + a n^2 < (5/2) * a (n + 1) * a n) ∧
  (a 2 = 3/2) ∧
  (a 4 = 4)

/-- The third term of the sequence is within the range (2, 3) -/
theorem third_term_range (a : ℕ → ℝ) (h : SpecialSequence a) :
  ∃ x, a 3 = x ∧ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_third_term_range_l3292_329235


namespace NUMINAMATH_CALUDE_bob_bake_time_proof_l3292_329280

/-- The time it takes Alice to bake a pie, in minutes -/
def alice_bake_time : ℝ := 5

/-- The time it takes Bob to bake a pie, in minutes -/
def bob_bake_time : ℝ := 6

/-- The total time available for baking, in minutes -/
def total_time : ℝ := 60

/-- The number of additional pies Alice can bake compared to Bob in the total time -/
def additional_pies : ℕ := 2

theorem bob_bake_time_proof :
  alice_bake_time = 5 ∧
  (total_time / alice_bake_time - total_time / bob_bake_time = additional_pies) →
  bob_bake_time = 6 := by
sorry

end NUMINAMATH_CALUDE_bob_bake_time_proof_l3292_329280


namespace NUMINAMATH_CALUDE_inequality_proof_l3292_329200

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3292_329200


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3292_329255

-- Problem 1
theorem problem_1 (a b : ℝ) (h1 : (a + b)^2 = 6) (h2 : (a - b)^2 = 2) : 
  a^2 + b^2 = 4 ∧ a * b = 1 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x + 1/x = 3) : 
  x^2 + 1/x^2 = 7 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3292_329255


namespace NUMINAMATH_CALUDE_subtraction_decimal_l3292_329239

theorem subtraction_decimal : 3.56 - 1.29 = 2.27 := by sorry

end NUMINAMATH_CALUDE_subtraction_decimal_l3292_329239


namespace NUMINAMATH_CALUDE_inequality_solution_l3292_329233

theorem inequality_solution (x : ℝ) : -3 * x^2 + 5 * x + 4 < 0 ∧ x > 0 → x ∈ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3292_329233


namespace NUMINAMATH_CALUDE_total_fans_l3292_329224

/-- The number of students who like basketball -/
def basketball_fans : ℕ := 7

/-- The number of students who like cricket -/
def cricket_fans : ℕ := 5

/-- The number of students who like both basketball and cricket -/
def both_fans : ℕ := 3

/-- Theorem: The number of students who like basketball or cricket or both is 9 -/
theorem total_fans : basketball_fans + cricket_fans - both_fans = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_fans_l3292_329224


namespace NUMINAMATH_CALUDE_school_workbooks_calculation_l3292_329238

/-- The number of workbooks a school should buy given the number of classes,
    workbooks per class, and spare workbooks. -/
def total_workbooks (num_classes : ℕ) (workbooks_per_class : ℕ) (spare_workbooks : ℕ) : ℕ :=
  num_classes * workbooks_per_class + spare_workbooks

/-- Theorem stating that the total number of workbooks the school should buy
    is equal to 25 * 144 + 80, given the specific conditions of the problem. -/
theorem school_workbooks_calculation :
  total_workbooks 25 144 80 = 25 * 144 + 80 := by
  sorry

end NUMINAMATH_CALUDE_school_workbooks_calculation_l3292_329238


namespace NUMINAMATH_CALUDE_negative_integral_of_negative_function_l3292_329227

theorem negative_integral_of_negative_function 
  {f : ℝ → ℝ} {a b : ℝ} 
  (hf : Continuous f) 
  (hneg : ∀ x, f x < 0) 
  (hab : a < b) : 
  ∫ x in a..b, f x < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_integral_of_negative_function_l3292_329227


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l3292_329216

theorem consecutive_even_integers_sum (n : ℤ) : 
  (n + (n + 4) = 156) → (n + (n + 2) + (n + 4) = 234) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l3292_329216


namespace NUMINAMATH_CALUDE_max_value_of_a_plus_2b_l3292_329281

/-- Two circles in a 2D plane -/
structure TwoCircles where
  a : ℝ
  b : ℝ
  circle1 : (x y : ℝ) → x^2 + y^2 + 2*a*x + a^2 - 4 = 0
  circle2 : (x y : ℝ) → x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The property that two circles have exactly three common tangents -/
def have_three_common_tangents (c : TwoCircles) : Prop := sorry

/-- The theorem stating the maximum value of a+2b -/
theorem max_value_of_a_plus_2b (c : TwoCircles) 
  (h : have_three_common_tangents c) : 
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
  ∀ (a b : ℝ), c.a = a → c.b = b → a + 2*b ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_a_plus_2b_l3292_329281


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3292_329278

def is_valid_number (n : Nat) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧
  ∃ (k : Nat), 
    (n / (n % 100) = k^2) ∧
    (k^2 = (n / 100 + 1)^2)

theorem smallest_valid_number : 
  is_valid_number 1805 ∧ 
  ∀ (m : Nat), is_valid_number m → m ≥ 1805 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3292_329278


namespace NUMINAMATH_CALUDE_g_value_l3292_329268

-- Define the polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_def : ∀ x, f x = x^4 - x^2 - 3
axiom sum_eq : ∀ x, f x + g x = 3 * x^2 - 1

-- State the theorem
theorem g_value : ∀ x, g x = -x^4 + 4 * x^2 + 2 := by sorry

end NUMINAMATH_CALUDE_g_value_l3292_329268


namespace NUMINAMATH_CALUDE_perpendicular_vectors_tan_2x_l3292_329211

theorem perpendicular_vectors_tan_2x (x : ℝ) : 
  let a : ℝ × ℝ := (Real.cos x, Real.sin x)
  let b : ℝ × ℝ := (Real.sqrt 3, -1)
  (a.1 * b.1 + a.2 * b.2 = 0) → Real.tan (2 * x) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_tan_2x_l3292_329211


namespace NUMINAMATH_CALUDE_five_line_intersections_l3292_329229

/-- Represents a configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_intersections : ℕ
  no_three_point_intersection : Bool

/-- The maximum number of intersections for n lines -/
def max_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the impossibility of 11 intersections and possibility of 9 intersections -/
theorem five_line_intersections (config : LineConfiguration) :
  config.num_lines = 5 ∧ config.no_three_point_intersection = true →
  (config.num_intersections ≠ 11 ∧ 
   ∃ (config' : LineConfiguration), 
     config'.num_lines = 5 ∧ 
     config'.no_three_point_intersection = true ∧ 
     config'.num_intersections = 9) := by
  sorry

end NUMINAMATH_CALUDE_five_line_intersections_l3292_329229


namespace NUMINAMATH_CALUDE_greatest_whole_number_inequality_l3292_329262

theorem greatest_whole_number_inequality :
  ∀ x : ℤ, (7 * x - 8 < 4 - 2 * x) → x ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_whole_number_inequality_l3292_329262


namespace NUMINAMATH_CALUDE_average_of_geometric_sequence_l3292_329275

theorem average_of_geometric_sequence (z : ℝ) :
  let sequence := [0, 2*z, 4*z, 8*z, 16*z]
  (sequence.sum / sequence.length : ℝ) = 6*z :=
by sorry

end NUMINAMATH_CALUDE_average_of_geometric_sequence_l3292_329275


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l3292_329253

/-- A regular octagon with a square inscribed such that one side of the square
    coincides with one side of the octagon. -/
structure OctagonWithSquare where
  /-- The measure of an interior angle of the regular octagon -/
  octagon_interior_angle : ℝ
  /-- The measure of an interior angle of the square -/
  square_interior_angle : ℝ
  /-- A is a vertex of the octagon -/
  A : Point
  /-- B is the next vertex of the octagon after A -/
  B : Point
  /-- C is a vertex of the inscribed square on the line extended from side AB -/
  C : Point
  /-- The measure of angle ABC -/
  angle_ABC : ℝ
  /-- The octagon is regular -/
  octagon_regular : octagon_interior_angle = 135
  /-- The square has right angles -/
  square_right_angle : square_interior_angle = 90

/-- The measure of angle ABC in the described configuration is 67.5 degrees -/
theorem angle_ABC_measure (config : OctagonWithSquare) : config.angle_ABC = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l3292_329253


namespace NUMINAMATH_CALUDE_f_range_and_triangle_property_l3292_329201

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, 1 - Real.sqrt 2 * Real.sin x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1 + Real.sqrt 2 * Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem f_range_and_triangle_property :
  (∀ y ∈ Set.Icc (-1 : ℝ) 2, ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = y) ∧
  (∀ {a b c A B C : ℝ},
    b / a = Real.sqrt 3 →
    (Real.sin B * Real.cos A) / Real.sin A = 2 - Real.cos B →
    f B = 1) :=
sorry

end NUMINAMATH_CALUDE_f_range_and_triangle_property_l3292_329201


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l3292_329225

theorem point_on_terminal_side (t : ℝ) (θ : ℝ) : 
  ((-2 : ℝ) = Real.cos θ * Real.sqrt (4 + t^2)) →
  (t = Real.sin θ * Real.sqrt (4 + t^2)) →
  (Real.sin θ + Real.cos θ = Real.sqrt 5 / 5) →
  t = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l3292_329225


namespace NUMINAMATH_CALUDE_sin_alpha_plus_7pi_over_6_l3292_329266

theorem sin_alpha_plus_7pi_over_6 (α : ℝ) 
  (h : Real.cos (α - π / 6) + Real.sin α = (4 / 5) * Real.sqrt 3) : 
  Real.sin (α + 7 * π / 6) = -(4 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_7pi_over_6_l3292_329266


namespace NUMINAMATH_CALUDE_reciprocal_problem_l3292_329265

theorem reciprocal_problem (x : ℚ) : 8 * x = 6 → 60 * (1 / x) = 80 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l3292_329265


namespace NUMINAMATH_CALUDE_equation_D_is_linear_l3292_329228

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x - 3 = 5 -/
def equation_D (x : ℝ) : ℝ := 2 * x - 3

theorem equation_D_is_linear : is_linear_equation equation_D := by
  sorry

end NUMINAMATH_CALUDE_equation_D_is_linear_l3292_329228


namespace NUMINAMATH_CALUDE_subtract_square_thirty_l3292_329214

theorem subtract_square_thirty : 30 - 5^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_square_thirty_l3292_329214


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3292_329297

theorem sum_of_two_numbers (x y : ℝ) : x * y = 437 ∧ |x - y| = 4 → x + y = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3292_329297


namespace NUMINAMATH_CALUDE_common_tangent_line_l3292_329244

/-- Two circles O₁ and O₂ in the Cartesian coordinate system -/
structure TwoCircles where
  m : ℝ
  r₁ : ℝ
  r₂ : ℝ
  h₁ : m > 0
  h₂ : r₁ > 0
  h₃ : r₂ > 0
  h₄ : r₁ * r₂ = 2
  h₅ : (3 : ℝ) = r₁ / m
  h₆ : (1 : ℝ) = r₁
  h₇ : (2 : ℝ) ^ 2 + (2 : ℝ) ^ 2 = (2 - r₁ / m) ^ 2 + (2 - r₁) ^ 2 + r₁ ^ 2
  h₈ : (2 : ℝ) ^ 2 + (2 : ℝ) ^ 2 = (2 - r₂ / m) ^ 2 + (2 - r₂) ^ 2 + r₂ ^ 2

/-- The equation of another common tangent line is y = (4/3)x -/
theorem common_tangent_line (c : TwoCircles) :
  ∃ (k : ℝ), k = 4 / 3 ∧ ∀ (x y : ℝ), y = k * x → 
  (∃ (t : ℝ), (x - 3) ^ 2 + (y - 1) ^ 2 = t ^ 2 ∧ (x - c.r₂ / c.m) ^ 2 + (y - c.r₂) ^ 2 = t ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_common_tangent_line_l3292_329244


namespace NUMINAMATH_CALUDE_orphanage_donation_l3292_329282

theorem orphanage_donation (total donation1 donation3 : ℚ) 
  (h1 : total = 650)
  (h2 : donation1 = 175)
  (h3 : donation3 = 250) :
  total - donation1 - donation3 = 225 := by
  sorry

end NUMINAMATH_CALUDE_orphanage_donation_l3292_329282


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l3292_329248

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem stating the total wet surface area of a specific cistern -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 8 6 1.25 = 83 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l3292_329248


namespace NUMINAMATH_CALUDE_table_capacity_l3292_329267

theorem table_capacity (invited : ℕ) (no_shows : ℕ) (tables : ℕ) : 
  invited = 18 → no_shows = 12 → tables = 2 → 
  (invited - no_shows) / tables = 3 := by sorry

end NUMINAMATH_CALUDE_table_capacity_l3292_329267


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_expression_l3292_329245

theorem largest_prime_divisor_of_expression : 
  ∃ p : ℕ, 
    Prime p ∧ 
    p ∣ (Nat.factorial 12 + Nat.factorial 13 + 17) ∧
    ∀ q : ℕ, Prime q → q ∣ (Nat.factorial 12 + Nat.factorial 13 + 17) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_expression_l3292_329245


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3292_329295

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    a(n+1) = r * a(n) for all n -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  IsGeometric a →
  (3 * (a 3)^2 - 11 * (a 3) + 9 = 0) →
  (3 * (a 7)^2 - 11 * (a 7) + 9 = 0) →
  a 5 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3292_329295


namespace NUMINAMATH_CALUDE_equation_solution_l3292_329204

theorem equation_solution :
  let f (x : ℝ) := x * ((6 - x) / (x + 1)) * ((6 - x) / (x + 1) + x)
  ∀ x : ℝ, f x = 8 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3292_329204


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3292_329221

theorem unique_solution_for_equation (m p q : ℕ) : 
  m > 0 ∧ 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  2^m * p^2 + 1 = q^5 → 
  m = 1 ∧ p = 11 ∧ q = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3292_329221


namespace NUMINAMATH_CALUDE_jasons_punch_problem_l3292_329220

/-- Represents the recipe for Jason's punch -/
structure PunchRecipe where
  water : ℝ
  lemon_juice : ℝ
  sugar : ℝ

/-- Represents the actual amounts used in Jason's punch -/
structure PunchIngredients where
  water : ℝ
  lemon_juice : ℝ
  sugar : ℝ

/-- The recipe ratios are correct -/
def recipe_ratios_correct (recipe : PunchRecipe) : Prop :=
  recipe.water = 5 * recipe.lemon_juice ∧ 
  recipe.lemon_juice = 3 * recipe.sugar

/-- The actual ingredients follow the recipe ratios -/
def ingredients_follow_recipe (recipe : PunchRecipe) (ingredients : PunchIngredients) : Prop :=
  ingredients.water / ingredients.lemon_juice = recipe.water / recipe.lemon_juice ∧
  ingredients.lemon_juice / ingredients.sugar = recipe.lemon_juice / recipe.sugar

/-- Jason's punch problem -/
theorem jasons_punch_problem (recipe : PunchRecipe) (ingredients : PunchIngredients) :
  recipe_ratios_correct recipe →
  ingredients_follow_recipe recipe ingredients →
  ingredients.lemon_juice = 5 →
  ingredients.water = 25 := by
  sorry

end NUMINAMATH_CALUDE_jasons_punch_problem_l3292_329220


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l3292_329263

/-- Given two positive integers with LCM 750 and product 18750, prove their HCF is 25 -/
theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750)
  (h_product : a * b = 18750) : 
  Nat.gcd a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l3292_329263


namespace NUMINAMATH_CALUDE_function_identity_l3292_329274

theorem function_identity (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (f m + f n) = m + n) : 
  ∀ x : ℕ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_identity_l3292_329274


namespace NUMINAMATH_CALUDE_simultaneous_divisibility_by_17_l3292_329271

theorem simultaneous_divisibility_by_17 : ∃ (x y : ℤ), 
  (17 ∣ (2*x + 3*y)) ∧ (17 ∣ (9*x + 5*y)) := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_divisibility_by_17_l3292_329271


namespace NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l3292_329276

theorem coefficient_m5n5_in_expansion : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l3292_329276


namespace NUMINAMATH_CALUDE_proposition_count_l3292_329269

theorem proposition_count : ∃! n : ℕ, n = 2 ∧ 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x * y ≥ 0) ∧ 
  (∀ x y : ℝ, x * y ≥ 0 → x ≥ 0 ∧ y ≥ 0 ∨ x ≤ 0 ∧ y ≤ 0) ∧
  (∃ x y : ℝ, ¬(x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)) ∧
  (∀ x y : ℝ, x * y < 0 → x < 0 ∨ y < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_proposition_count_l3292_329269


namespace NUMINAMATH_CALUDE_equation_solution_l3292_329277

theorem equation_solution (x : ℝ) :
  (|Real.cos x| - Real.cos (3 * x)) / (Real.cos x * Real.sin (2 * x)) = 2 / Real.sqrt 3 ↔
  (∃ k : ℤ, x = π / 6 + 2 * k * π ∨ x = 5 * π / 6 + 2 * k * π ∨ x = 4 * π / 3 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3292_329277


namespace NUMINAMATH_CALUDE_curve_intersects_median_l3292_329241

theorem curve_intersects_median (a b c : ℝ) (h : a + c - 2*b ≠ 0) :
  ∃! p : ℂ, 
    (∀ t : ℝ, p ≠ Complex.I * a * (Real.cos t)^4 + 2 * (1/2 + Complex.I * b) * (Real.cos t)^2 * (Real.sin t)^2 + (1 + Complex.I * c) * (Real.sin t)^4) ∧
    (p.re = 1/2) ∧
    (p.im = (a + c + 2*b) / 4) ∧
    (∃ k : ℝ, p.im - (a + b)/2 = (c - a) * (p.re - 1/4) + k * ((3/4 - 1/4) * Complex.I - ((b + c)/2 - (a + b)/2))) := by
  sorry

end NUMINAMATH_CALUDE_curve_intersects_median_l3292_329241


namespace NUMINAMATH_CALUDE_sqrt_three_times_sqrt_twelve_l3292_329258

theorem sqrt_three_times_sqrt_twelve : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_times_sqrt_twelve_l3292_329258


namespace NUMINAMATH_CALUDE_school_c_sample_size_l3292_329293

/-- Represents the number of teachers in each school -/
structure SchoolPopulation where
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ

/-- Represents the sampling parameters -/
structure SamplingParams where
  totalSample : ℕ
  population : SchoolPopulation

/-- Calculates the stratified sample size for a given school -/
def stratifiedSampleSize (params : SamplingParams) (schoolSize : ℕ) : ℕ :=
  (schoolSize * params.totalSample) / (params.population.schoolA + params.population.schoolB + params.population.schoolC)

/-- Theorem stating that the stratified sample size for School C is 10 -/
theorem school_c_sample_size :
  let params : SamplingParams := {
    totalSample := 60,
    population := {
      schoolA := 180,
      schoolB := 270,
      schoolC := 90
    }
  }
  stratifiedSampleSize params params.population.schoolC = 10 := by
  sorry


end NUMINAMATH_CALUDE_school_c_sample_size_l3292_329293


namespace NUMINAMATH_CALUDE_clock_hand_overlaps_l3292_329232

/-- Represents the number of revolutions a clock hand makes in a day -/
structure ClockHand where
  revolutions : ℕ

/-- Calculates the number of overlaps between two clock hands in a day -/
def overlaps (hand1 hand2 : ClockHand) : ℕ :=
  hand2.revolutions - hand1.revolutions

theorem clock_hand_overlaps :
  let hour_hand : ClockHand := ⟨2⟩
  let minute_hand : ClockHand := ⟨24⟩
  let second_hand : ClockHand := ⟨1440⟩
  (overlaps hour_hand minute_hand = 22) ∧
  (overlaps minute_hand second_hand = 1416) :=
by sorry

end NUMINAMATH_CALUDE_clock_hand_overlaps_l3292_329232


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3292_329294

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 70 = (X - 7) * q + 63 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3292_329294


namespace NUMINAMATH_CALUDE_bug_crawl_distance_l3292_329237

-- Define the bug's movement
def bugPath : List ℤ := [-3, -7, 0, 8]

-- Function to calculate distance between two points
def distance (a b : ℤ) : ℕ := (a - b).natAbs

-- Function to calculate total distance traveled
def totalDistance (path : List ℤ) : ℕ :=
  List.sum (List.zipWith distance path path.tail)

-- Theorem statement
theorem bug_crawl_distance :
  totalDistance bugPath = 19 := by sorry

end NUMINAMATH_CALUDE_bug_crawl_distance_l3292_329237


namespace NUMINAMATH_CALUDE_disjunction_implies_conjunction_false_l3292_329217

theorem disjunction_implies_conjunction_false :
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_implies_conjunction_false_l3292_329217


namespace NUMINAMATH_CALUDE_f_derivative_sum_l3292_329257

noncomputable def f (x : ℝ) : ℝ := Real.log 9 * (Real.log x / Real.log 3)

theorem f_derivative_sum : 
  (deriv (λ _ : ℝ => f 2)) 0 + (deriv f) 2 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_sum_l3292_329257


namespace NUMINAMATH_CALUDE_festival_lineup_solution_valid_l3292_329207

/-- Represents the minimum number of Gennadys required for the festival lineup -/
def min_gennadys (alexanders borises vasilies : ℕ) : ℕ :=
  max 0 (borises - 1 - alexanders - vasilies)

/-- Theorem stating the minimum number of Gennadys required for the given problem -/
theorem festival_lineup (alexanders borises vasilies : ℕ) 
  (h_alex : alexanders = 45)
  (h_boris : borises = 122)
  (h_vasily : vasilies = 27) :
  min_gennadys alexanders borises vasilies = 49 := by
  sorry

/-- Verifies that the solution satisfies the problem constraints -/
theorem solution_valid (alexanders borises vasilies gennadys : ℕ)
  (h_alex : alexanders = 45)
  (h_boris : borises = 122)
  (h_vasily : vasilies = 27)
  (h_gennady : gennadys = min_gennadys alexanders borises vasilies) :
  alexanders + borises + vasilies + gennadys ≥ borises + (borises - 1) := by
  sorry

end NUMINAMATH_CALUDE_festival_lineup_solution_valid_l3292_329207


namespace NUMINAMATH_CALUDE_basketball_time_calc_l3292_329250

/-- Calculates the time spent playing basketball given total play time and football play time. -/
def basketball_time (total_time : Real) (football_time : Nat) : Real :=
  total_time * 60 - football_time

/-- Proves that given a total play time of 1.5 hours and 60 minutes of football,
    the time spent playing basketball is 30 minutes. -/
theorem basketball_time_calc :
  basketball_time 1.5 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_basketball_time_calc_l3292_329250


namespace NUMINAMATH_CALUDE_parallelogram_bisecting_line_slope_l3292_329223

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Determines if a line through the origin cuts a parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (m : ℝ) : Prop := sorry

/-- The specific parallelogram from the problem -/
def problem_parallelogram : Parallelogram :=
  { v1 := { x := 2, y := 5 }
  , v2 := { x := 2, y := 23 }
  , v3 := { x := 7, y := 38 }
  , v4 := { x := 7, y := 20 }
  }

theorem parallelogram_bisecting_line_slope :
  cuts_into_congruent_polygons problem_parallelogram (43/9) := by sorry

end NUMINAMATH_CALUDE_parallelogram_bisecting_line_slope_l3292_329223


namespace NUMINAMATH_CALUDE_system_a_solution_system_b_solutions_l3292_329298

-- Part (a)
theorem system_a_solution (x y : ℝ) : 
  x^2 - 3*x*y - 4*y^2 = 0 ∧ x^3 + y^3 = 65 → (x = 4 ∧ y = 1) :=
sorry

-- Part (b)
theorem system_b_solutions (x y : ℝ) :
  x^2 + 2*y^2 = 17 ∧ 2*x*y - x^2 = 3 →
  ((x = 3 ∧ y = 2) ∨ 
   (x = -3 ∧ y = -2) ∨ 
   (x = Real.sqrt 3 / 3 ∧ y = 5 * Real.sqrt 3 / 3) ∨ 
   (x = -Real.sqrt 3 / 3 ∧ y = -5 * Real.sqrt 3 / 3)) :=
sorry

end NUMINAMATH_CALUDE_system_a_solution_system_b_solutions_l3292_329298


namespace NUMINAMATH_CALUDE_tiling_combination_l3292_329226

def interior_angle (n : ℕ) : ℚ := (n - 2 : ℚ) * 180 / n

def can_tile (a b c : ℕ) : Prop :=
  ∃ (m n p : ℕ), m * interior_angle a + n * interior_angle b + p * interior_angle c = 360 ∧
  m + n + p = 4 ∧ m > 0 ∧ n > 0 ∧ p > 0

theorem tiling_combination :
  can_tile 3 4 6 ∧
  ¬can_tile 3 4 5 ∧
  ¬can_tile 3 4 7 ∧
  ¬can_tile 3 4 8 :=
sorry

end NUMINAMATH_CALUDE_tiling_combination_l3292_329226


namespace NUMINAMATH_CALUDE_acid_mixture_percentage_l3292_329249

theorem acid_mixture_percentage :
  ∀ (a w : ℝ),
  a > 0 ∧ w > 0 →
  a / (a + w + 2) = 0.3 →
  (a + 2) / (a + w + 4) = 0.4 →
  a / (a + w) = 0.36 := by
sorry

end NUMINAMATH_CALUDE_acid_mixture_percentage_l3292_329249


namespace NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l3292_329213

theorem complex_arithmetic_evaluation : (7 - 3*I) - 3*(2 - 5*I) = 1 + 12*I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l3292_329213
