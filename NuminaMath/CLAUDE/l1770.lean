import Mathlib

namespace NUMINAMATH_CALUDE_ivy_cupcakes_l1770_177009

def morning_cupcakes : ℕ := 20
def afternoon_difference : ℕ := 15

def total_cupcakes : ℕ := morning_cupcakes + (morning_cupcakes + afternoon_difference)

theorem ivy_cupcakes : total_cupcakes = 55 := by
  sorry

end NUMINAMATH_CALUDE_ivy_cupcakes_l1770_177009


namespace NUMINAMATH_CALUDE_factorization_proof_l1770_177002

theorem factorization_proof (x y : ℝ) : 75 * x^10 * y^3 - 150 * x^20 * y^6 = 75 * x^10 * y^3 * (1 - 2 * x^10 * y^3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1770_177002


namespace NUMINAMATH_CALUDE_odd_integer_sequence_sum_l1770_177067

theorem odd_integer_sequence_sum (n : ℕ) : n > 0 → (
  let sum := n / 2 * (5 + (6 * n - 1))
  sum = 597 ↔ n = 13
) := by sorry

end NUMINAMATH_CALUDE_odd_integer_sequence_sum_l1770_177067


namespace NUMINAMATH_CALUDE_sum_x_y_l1770_177031

/-- The smallest positive integer x such that 480x is a perfect square -/
def x : ℕ := 30

/-- The smallest positive integer y such that 480y is a perfect cube -/
def y : ℕ := 450

/-- 480 * x is a perfect square -/
axiom x_square : ∃ n : ℕ, 480 * x = n^2

/-- 480 * y is a perfect cube -/
axiom y_cube : ∃ n : ℕ, 480 * y = n^3

/-- x is the smallest positive integer such that 480x is a perfect square -/
axiom x_smallest : ∀ z : ℕ, z > 0 → z < x → ¬∃ n : ℕ, 480 * z = n^2

/-- y is the smallest positive integer such that 480y is a perfect cube -/
axiom y_smallest : ∀ z : ℕ, z > 0 → z < y → ¬∃ n : ℕ, 480 * z = n^3

theorem sum_x_y : x + y = 480 := by sorry

end NUMINAMATH_CALUDE_sum_x_y_l1770_177031


namespace NUMINAMATH_CALUDE_train_passing_time_l1770_177080

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 20 → 
  train_speed_kmph = 36 → 
  (train_length / (train_speed_kmph * (1000 / 3600))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1770_177080


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1770_177063

theorem simplify_sqrt_expression (y : ℝ) (hy : y ≠ 0) :
  Real.sqrt (4 + ((y^6 - 4) / (3 * y^3))^2) = (Real.sqrt (y^12 + 28 * y^6 + 16)) / (3 * y^3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1770_177063


namespace NUMINAMATH_CALUDE_greatest_possible_area_l1770_177029

/-- A convex equilateral pentagon with side length 2 and two right angles -/
structure ConvexEquilateralPentagon where
  side_length : ℝ
  has_two_right_angles : Prop
  is_convex : Prop
  is_equilateral : Prop
  side_length_eq_two : side_length = 2

/-- The area of a ConvexEquilateralPentagon -/
def area (p : ConvexEquilateralPentagon) : ℝ := sorry

theorem greatest_possible_area (p : ConvexEquilateralPentagon) :
  area p ≤ 4 + Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_greatest_possible_area_l1770_177029


namespace NUMINAMATH_CALUDE_absolute_value_calculation_l1770_177003

theorem absolute_value_calculation : |-2| - Real.sqrt 4 + 3^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_calculation_l1770_177003


namespace NUMINAMATH_CALUDE_sarah_flour_total_l1770_177030

/-- The total amount of flour Sarah has -/
def total_flour (rye whole_wheat chickpea pastry : ℕ) : ℕ :=
  rye + whole_wheat + chickpea + pastry

/-- Theorem: Sarah has 20 pounds of flour in total -/
theorem sarah_flour_total :
  total_flour 5 10 3 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sarah_flour_total_l1770_177030


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1770_177012

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x + |2*x - 3| > 0} = Set.Ioi 2 ∪ Set.Iic (2/3) :=
sorry

-- Part II
theorem range_of_a_part_ii :
  {a : ℝ | ∀ x, f a x < |x - 3|} = Set.Ioo 1 5 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1770_177012


namespace NUMINAMATH_CALUDE_range_of_q_l1770_177004

def q (x : ℝ) : ℝ := x^4 + 4*x^2 + 4

theorem range_of_q :
  {y : ℝ | ∃ x ≥ 0, q x = y} = {y : ℝ | y ≥ 4} := by sorry

end NUMINAMATH_CALUDE_range_of_q_l1770_177004


namespace NUMINAMATH_CALUDE_f_neg_l1770_177015

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_pos : ∀ x > 0, f x = -x * (1 + x)

-- Theorem to prove
theorem f_neg : ∀ x < 0, f x = -x * (1 - x) := by sorry

end NUMINAMATH_CALUDE_f_neg_l1770_177015


namespace NUMINAMATH_CALUDE_mult_B_not_binomial_square_or_diff_squares_other_mults_are_diff_squares_l1770_177033

-- Define the multiplications
def mult_A (x y : ℝ) := (3*x + 7*y) * (3*x - 7*y)
def mult_B (m n : ℝ) := (5*m - n) * (n - 5*m)
def mult_C (x : ℝ) := (-0.2*x - 0.3) * (-0.2*x + 0.3)
def mult_D (m n : ℝ) := (-3*n - m*n) * (3*n - m*n)

-- Define the square of binomial form
def square_of_binomial (a b : ℝ) := (a + b)^2

-- Define the difference of squares form
def diff_of_squares (a b : ℝ) := a^2 - b^2

theorem mult_B_not_binomial_square_or_diff_squares :
  ∀ m n : ℝ, ¬∃ a b : ℝ, mult_B m n = square_of_binomial a b ∨ mult_B m n = diff_of_squares a b :=
sorry

theorem other_mults_are_diff_squares :
  (∀ x y : ℝ, ∃ a b : ℝ, mult_A x y = diff_of_squares a b) ∧
  (∀ x : ℝ, ∃ a b : ℝ, mult_C x = diff_of_squares a b) ∧
  (∀ m n : ℝ, ∃ a b : ℝ, mult_D m n = diff_of_squares a b) :=
sorry

end NUMINAMATH_CALUDE_mult_B_not_binomial_square_or_diff_squares_other_mults_are_diff_squares_l1770_177033


namespace NUMINAMATH_CALUDE_grace_age_l1770_177072

-- Define the ages as natural numbers
def Harriet : ℕ := 18
def Ian : ℕ := Harriet + 5
def Jack : ℕ := Ian - 7
def Grace : ℕ := 2 * Jack

-- Theorem statement
theorem grace_age : Grace = 32 := by
  sorry

end NUMINAMATH_CALUDE_grace_age_l1770_177072


namespace NUMINAMATH_CALUDE_rosemary_pots_correct_l1770_177066

/-- The number of pots of rosemary Annie planted -/
def rosemary_pots : ℕ := 
  let basil_pots : ℕ := 3
  let thyme_pots : ℕ := 6
  let basil_leaves_per_pot : ℕ := 4
  let rosemary_leaves_per_pot : ℕ := 18
  let thyme_leaves_per_pot : ℕ := 30
  let total_leaves : ℕ := 354
  9

theorem rosemary_pots_correct : 
  let basil_pots : ℕ := 3
  let thyme_pots : ℕ := 6
  let basil_leaves_per_pot : ℕ := 4
  let rosemary_leaves_per_pot : ℕ := 18
  let thyme_leaves_per_pot : ℕ := 30
  let total_leaves : ℕ := 354
  rosemary_pots * rosemary_leaves_per_pot + 
  basil_pots * basil_leaves_per_pot + 
  thyme_pots * thyme_leaves_per_pot = total_leaves :=
by sorry

end NUMINAMATH_CALUDE_rosemary_pots_correct_l1770_177066


namespace NUMINAMATH_CALUDE_parts_per_day_calculation_l1770_177020

/-- The number of parts initially planned per day -/
def initial_parts_per_day : ℕ := 142

/-- The number of days with initial production rate -/
def initial_days : ℕ := 3

/-- The increase in parts per day after the initial days -/
def increase_in_parts : ℕ := 5

/-- The total number of parts produced -/
def total_parts : ℕ := 675

/-- The number of extra parts produced compared to the plan -/
def extra_parts : ℕ := 100

/-- The number of days after the initial period -/
def additional_days : ℕ := 1

theorem parts_per_day_calculation :
  initial_parts_per_day * initial_days + 
  (initial_parts_per_day + increase_in_parts) * additional_days = 
  total_parts - extra_parts :=
by sorry

#check parts_per_day_calculation

end NUMINAMATH_CALUDE_parts_per_day_calculation_l1770_177020


namespace NUMINAMATH_CALUDE_matrix_cube_computation_l1770_177007

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_cube_computation :
  A ^ 3 = !![(-4), 2; (-2), 1] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_computation_l1770_177007


namespace NUMINAMATH_CALUDE_expression_evaluation_l1770_177013

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1770_177013


namespace NUMINAMATH_CALUDE_exactly_one_hit_probability_l1770_177019

/-- The probability that both A and B hit the target -/
def prob_both_hit : ℝ := 0.6

/-- The probability that A hits the target -/
def prob_A_hit : ℝ := prob_both_hit

/-- The probability that B hits the target -/
def prob_B_hit : ℝ := prob_both_hit

/-- The probability that exactly one of A and B hits the target -/
def prob_exactly_one_hit : ℝ := prob_A_hit * (1 - prob_B_hit) + (1 - prob_A_hit) * prob_B_hit

theorem exactly_one_hit_probability :
  prob_exactly_one_hit = 0.48 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_hit_probability_l1770_177019


namespace NUMINAMATH_CALUDE_multiple_of_eleven_with_specific_digits_l1770_177032

theorem multiple_of_eleven_with_specific_digits : ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧
  (85 * 10^5 + A * 10^4 + 3 * 10^3 + 6 * 10^2 + B * 10 + 4) % 11 = 0 ∧
  (9 * 10^6 + 1 * 10^5 + 7 * 10^4 + B * 10^3 + A * 10^2 + 5 * 10 + 0) % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_eleven_with_specific_digits_l1770_177032


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1770_177099

theorem sufficient_not_necessary (a : ℝ) :
  (a > 9 → (1 / a) < (1 / 9)) ∧
  ∃ b : ℝ, (1 / b) < (1 / 9) ∧ b ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1770_177099


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l1770_177011

/-- A parabola passing through (2, 3) and (0, 7) has coefficients a, b, c such that a + b + c = 4 -/
theorem parabola_coefficient_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (x - 2)^2 + 3) → -- Vertex form condition
  (a * 0^2 + b * 0 + c = 7) →                      -- Passes through (0, 7)
  (a + b + c = 4) := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l1770_177011


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1770_177091

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 7b₃ is -16/7 -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) :
  b₁ = 2 →
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) →
  (∀ x y : ℝ, x = b₂ ∧ y = b₃ → 3 * x + 7 * y ≥ -16/7) ∧
  (∃ x y : ℝ, x = b₂ ∧ y = b₃ ∧ 3 * x + 7 * y = -16/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1770_177091


namespace NUMINAMATH_CALUDE_unique_solution_l1770_177024

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 81

-- State the theorem
theorem unique_solution :
  ∃! x, f x = 1/4 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1770_177024


namespace NUMINAMATH_CALUDE_travel_fraction_proof_l1770_177028

def initial_amount : ℚ := 750
def clothes_fraction : ℚ := 1/3
def food_fraction : ℚ := 1/5
def final_amount : ℚ := 300

theorem travel_fraction_proof :
  let remaining_after_clothes := initial_amount * (1 - clothes_fraction)
  let remaining_after_food := remaining_after_clothes * (1 - food_fraction)
  let spent_on_travel := remaining_after_food - final_amount
  spent_on_travel / remaining_after_food = 1/4 := by sorry

end NUMINAMATH_CALUDE_travel_fraction_proof_l1770_177028


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1770_177055

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 9) :
  2/y + 1/x ≥ 1 ∧ (2/y + 1/x = 1 ↔ x = 3 ∧ y = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1770_177055


namespace NUMINAMATH_CALUDE_min_diff_composite_sum_105_l1770_177064

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def sum_to_105 (a b : ℕ) : Prop := a + b = 105

theorem min_diff_composite_sum_105 :
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ sum_to_105 a b ∧
  ∀ (c d : ℕ), is_composite c → is_composite d → sum_to_105 c d →
  (c : ℤ) - (d : ℤ) ≥ 3 ∨ (d : ℤ) - (c : ℤ) ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_diff_composite_sum_105_l1770_177064


namespace NUMINAMATH_CALUDE_triangle_tangent_inequality_l1770_177065

/-- Given a triangle ABC with sides a, b, c, and points A₁, A₂, B₁, B₂, C₁, C₂ defined by lines
    parallel to the opposite sides and tangent to the incircle, prove the inequality. -/
theorem triangle_tangent_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (AA₁ AA₂ BB₁ BB₂ CC₁ CC₂ : ℝ)
  (hAA₁ : AA₁ = b * (b + c - a) / (a + b + c))
  (hAA₂ : AA₂ = c * (b + c - a) / (a + b + c))
  (hBB₁ : BB₁ = c * (c + a - b) / (a + b + c))
  (hBB₂ : BB₂ = a * (c + a - b) / (a + b + c))
  (hCC₁ : CC₁ = a * (a + b - c) / (a + b + c))
  (hCC₂ : CC₂ = b * (a + b - c) / (a + b + c)) :
  AA₁ * AA₂ + BB₁ * BB₂ + CC₁ * CC₂ ≥ (1 / 9) * (a^2 + b^2 + c^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_tangent_inequality_l1770_177065


namespace NUMINAMATH_CALUDE_square_side_length_l1770_177018

theorem square_side_length (s : ℝ) (h : s > 0) : s ^ 2 = 2 * (4 * s) → s = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1770_177018


namespace NUMINAMATH_CALUDE_percentage_ratio_l1770_177034

theorem percentage_ratio (x : ℝ) (a b : ℝ) (ha : a = 0.08 * x) (hb : b = 0.16 * x) :
  a / b = 0.5 := by sorry

end NUMINAMATH_CALUDE_percentage_ratio_l1770_177034


namespace NUMINAMATH_CALUDE_rosie_pies_theorem_l1770_177084

def apples_per_pie (total_apples : ℕ) (pies : ℕ) : ℕ := total_apples / pies

def pies_from_apples (available_apples : ℕ) (apples_per_pie : ℕ) : ℕ := available_apples / apples_per_pie

def leftover_apples (available_apples : ℕ) (pies : ℕ) (apples_per_pie : ℕ) : ℕ := available_apples - pies * apples_per_pie

theorem rosie_pies_theorem (available_apples : ℕ) (base_apples : ℕ) (base_pies : ℕ) :
  available_apples = 55 →
  base_apples = 15 →
  base_pies = 3 →
  let apples_per_pie := apples_per_pie base_apples base_pies
  let pies := pies_from_apples available_apples apples_per_pie
  let leftovers := leftover_apples available_apples pies apples_per_pie
  pies = 11 ∧ leftovers = 0 := by sorry

end NUMINAMATH_CALUDE_rosie_pies_theorem_l1770_177084


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l1770_177025

/-- Simple interest calculation for a loan -/
theorem loan_principal_calculation 
  (rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ) :
  rate = 0.12 →
  time = 3 →
  interest = 5400 →
  principal * rate * time = interest →
  principal = 15000 := by
  sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l1770_177025


namespace NUMINAMATH_CALUDE_football_club_arrangements_l1770_177048

theorem football_club_arrangements (n : ℕ) (k : ℕ) 
  (h1 : n = 9) 
  (h2 : k = 2) : 
  (Nat.factorial n) * (Nat.choose n k) = 13063680 :=
by sorry

end NUMINAMATH_CALUDE_football_club_arrangements_l1770_177048


namespace NUMINAMATH_CALUDE_neg_p_and_q_implies_not_p_and_q_l1770_177083

theorem neg_p_and_q_implies_not_p_and_q (p q : Prop) :
  (¬p ∧ q) → (¬p ∧ q) :=
by
  sorry

end NUMINAMATH_CALUDE_neg_p_and_q_implies_not_p_and_q_l1770_177083


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_two_l1770_177073

theorem binomial_coefficient_seven_two : Nat.choose 7 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_two_l1770_177073


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1770_177014

/-- A line with equation x - 2y = r is tangent to a parabola with equation y = x^2 - r
    if and only if r = -1/8 -/
theorem line_tangent_to_parabola (r : ℝ) :
  (∃ x y, x - 2*y = r ∧ y = x^2 - r ∧
    ∀ x' y', x' - 2*y' = r ∧ y' = x'^2 - r → (x', y') = (x, y)) ↔
  r = -1/8 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1770_177014


namespace NUMINAMATH_CALUDE_log_product_equals_four_implies_y_equals_81_l1770_177089

theorem log_product_equals_four_implies_y_equals_81 (m y : ℝ) 
  (h : m > 0) (k : y > 0) (eq : Real.log y / Real.log m * Real.log m / Real.log 3 = 4) : 
  y = 81 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_four_implies_y_equals_81_l1770_177089


namespace NUMINAMATH_CALUDE_property_P_theorems_l1770_177022

/-- Property (P): A number n ≥ 2 has property (P) if in its prime factorization,
    at least one of the factors has an exponent of 3 -/
def has_property_P (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∃ p : ℕ, Prime p ∧ (∃ k : ℕ, n = p^(3*k+3) * (n / p^(3*k+3)))

/-- The smallest N such that any N consecutive natural numbers contain
    at least one number with property (P) -/
def smallest_N : ℕ := 16

/-- The smallest 15 consecutive numbers without property (P) such that
    their sum multiplied by 5 has property (P) -/
def smallest_15_consecutive : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

theorem property_P_theorems :
  (∀ k : ℕ, ∃ n ∈ List.range smallest_N, has_property_P (k + n)) ∧
  (∀ n ∈ smallest_15_consecutive, ¬ has_property_P n) ∧
  has_property_P (5 * smallest_15_consecutive.sum) := by
  sorry

end NUMINAMATH_CALUDE_property_P_theorems_l1770_177022


namespace NUMINAMATH_CALUDE_function_values_l1770_177045

noncomputable def f (A : ℝ) (x : ℝ) : ℝ := A * Real.sin x - Real.sqrt 3 * Real.cos x

theorem function_values (A : ℝ) :
  f A (π / 3) = 0 →
  A = 1 ∧ f A (π / 12) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_function_values_l1770_177045


namespace NUMINAMATH_CALUDE_larger_number_proof_l1770_177036

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 23) 
  (h2 : Nat.lcm a b = 23 * 13 * 15) (h3 : a > b) : a = 345 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1770_177036


namespace NUMINAMATH_CALUDE_parabola_intersection_points_l1770_177010

/-- Given a quadratic equation with specific roots, prove the intersection points of a related parabola with the x-axis -/
theorem parabola_intersection_points 
  (a m : ℝ) 
  (h1 : a * (-1 + m)^2 = 3) 
  (h2 : a * (3 + m)^2 = 3) :
  let f (x : ℝ) := a * (x + m - 2)^2 - 3
  ∃ (x1 x2 : ℝ), x1 = 5 ∧ x2 = 1 ∧ f x1 = 0 ∧ f x2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_points_l1770_177010


namespace NUMINAMATH_CALUDE_remainder_of_4n_mod_4_l1770_177077

theorem remainder_of_4n_mod_4 (n : ℤ) (h : n % 4 = 3) : (4 * n) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_4n_mod_4_l1770_177077


namespace NUMINAMATH_CALUDE_union_of_sets_l1770_177075

theorem union_of_sets : 
  let A : Set ℤ := {0, 1}
  let B : Set ℤ := {0, -1}
  A ∪ B = {-1, 0, 1} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1770_177075


namespace NUMINAMATH_CALUDE_base_eight_addition_l1770_177039

/-- Given a base-8 addition where 5XY₈ + 32₈ = 62X₈, prove that X + Y = 12 in base 10 --/
theorem base_eight_addition (X Y : ℕ) : 
  (5 * 8^2 + X * 8 + Y) + 32 = 6 * 8^2 + 2 * 8 + X → X + Y = 12 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_addition_l1770_177039


namespace NUMINAMATH_CALUDE_quadratic_properties_l1770_177098

/-- A quadratic function y = x² + mx + n -/
def quadratic (m n : ℝ) (x : ℝ) : ℝ := x^2 + m*x + n

theorem quadratic_properties (m n : ℝ) :
  (∀ y₁ y₂ : ℝ, quadratic m n 1 = y₁ ∧ quadratic m n 3 = y₂ ∧ y₁ = y₂ → m = -4) ∧
  (m = -4 ∧ ∃! x, quadratic m n x = 0 → n = 4) ∧
  (∀ a b₁ b₂ : ℝ, quadratic m n a = b₁ ∧ quadratic m n 3 = b₂ ∧ b₁ > b₂ → a < 1 ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1770_177098


namespace NUMINAMATH_CALUDE_point_on_bisector_l1770_177051

/-- If A(a, b) and B(b, a) represent the same point, then this point lies on the line y = x. -/
theorem point_on_bisector (a b : ℝ) : (a, b) = (b, a) → a = b :=
by sorry

end NUMINAMATH_CALUDE_point_on_bisector_l1770_177051


namespace NUMINAMATH_CALUDE_beef_weight_after_processing_l1770_177050

theorem beef_weight_after_processing (initial_weight : ℝ) (loss_percentage : ℝ) 
  (processed_weight : ℝ) (h1 : initial_weight = 840) (h2 : loss_percentage = 35) :
  processed_weight = initial_weight * (1 - loss_percentage / 100) → 
  processed_weight = 546 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_after_processing_l1770_177050


namespace NUMINAMATH_CALUDE_min_value_when_a_is_quarter_range_of_a_for_full_range_l1770_177017

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - 4*a) * a^x + a else Real.log x

-- Theorem 1: Minimum value of f(x) when a = 1/4 is 0
theorem min_value_when_a_is_quarter :
  ∀ x : ℝ, f (1/4) x ≥ 0 ∧ ∃ x₀ : ℝ, f (1/4) x₀ = 0 :=
sorry

-- Theorem 2: Range of f(x) is R iff 1/2 < a ≤ 3/4
theorem range_of_a_for_full_range :
  ∀ a : ℝ, (a > 0 ∧ a ≠ 1) →
    (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (1/2 < a ∧ a ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_quarter_range_of_a_for_full_range_l1770_177017


namespace NUMINAMATH_CALUDE_intersection_M_N_l1770_177016

def M : Set ℕ := {1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2*a - 1}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1770_177016


namespace NUMINAMATH_CALUDE_spherical_coords_negated_y_theorem_l1770_177068

/-- Given a point with rectangular coordinates (x, y, z) and spherical coordinates (ρ, θ, φ),
    this function returns the spherical coordinates of the point (x, -y, z) -/
def spherical_coords_negated_y (x y z ρ θ φ : Real) : Real × Real × Real :=
  sorry

/-- Theorem stating that if a point has rectangular coordinates (x, y, z) and 
    spherical coordinates (3, 5π/6, 5π/12), then the point with rectangular 
    coordinates (x, -y, z) has spherical coordinates (3, π/6, 5π/12) -/
theorem spherical_coords_negated_y_theorem (x y z : Real) :
  let (ρ, θ, φ) := (3, 5*π/6, 5*π/12)
  (x = ρ * Real.sin φ * Real.cos θ) →
  (y = ρ * Real.sin φ * Real.sin θ) →
  (z = ρ * Real.cos φ) →
  spherical_coords_negated_y x y z ρ θ φ = (3, π/6, 5*π/12) :=
by
  sorry

end NUMINAMATH_CALUDE_spherical_coords_negated_y_theorem_l1770_177068


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l1770_177052

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 25
def C₂ (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 100

-- Define the tangent line segment
def is_tangent (R S : ℝ × ℝ) : Prop :=
  C₁ R.1 R.2 ∧ C₂ S.1 S.2 ∧
  ∀ T : ℝ × ℝ, (T ≠ R ∧ T ≠ S) →
    (C₁ T.1 T.2 → (T.1 - R.1)^2 + (T.2 - R.2)^2 < (S.1 - R.1)^2 + (S.2 - R.2)^2) ∧
    (C₂ T.1 T.2 → (T.1 - S.1)^2 + (T.2 - S.2)^2 < (R.1 - S.1)^2 + (R.2 - S.2)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∃ R S : ℝ × ℝ, is_tangent R S ∧
    ∀ R' S' : ℝ × ℝ, is_tangent R' S' →
      Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) ≤ Real.sqrt ((S'.1 - R'.1)^2 + (S'.2 - R'.2)^2) ∧
      Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) = 5 * Real.sqrt 15 + 10 :=
sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l1770_177052


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1770_177054

/-- Given positive real numbers a, b, and c such that a + b + c = 1,
    the maximum value of √(a+1) + √(b+1) + √(c+1) is achieved
    when applying the Cauchy-Schwarz inequality. -/
theorem max_value_sqrt_sum (a b c : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (hsum : a + b + c = 1) : 
    ∃ (max : ℝ), ∀ (x y z : ℝ), 
    x > 0 → y > 0 → z > 0 → x + y + z = 1 →
    Real.sqrt (x + 1) + Real.sqrt (y + 1) + Real.sqrt (z + 1) ≤ max ∧
    Real.sqrt (a + 1) + Real.sqrt (b + 1) + Real.sqrt (c + 1) = max :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1770_177054


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1770_177057

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, (a * x^2 + b * x + 2 > 0) ↔ (-1/2 < x ∧ x < 1/3)) →
  a - b = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1770_177057


namespace NUMINAMATH_CALUDE_train_speed_l1770_177078

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : Real) (time : Real) (h1 : length = 200) (h2 : time = 12) :
  (length / 1000) / (time / 3600) = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1770_177078


namespace NUMINAMATH_CALUDE_cherry_trees_planted_l1770_177006

/-- The number of trees planted by each group in a tree-planting event --/
structure TreePlanting where
  apple : ℕ
  orange : ℕ
  cherry : ℕ

/-- The conditions of the tree-planting event --/
def tree_planting_conditions (t : TreePlanting) : Prop :=
  t.apple = 2 * t.orange ∧
  t.orange = t.apple - 15 ∧
  t.cherry = t.apple + t.orange - 10 ∧
  t.apple = 47 ∧
  t.orange = 27

/-- Theorem stating that under the given conditions, 64 cherry trees were planted --/
theorem cherry_trees_planted (t : TreePlanting) 
  (h : tree_planting_conditions t) : t.cherry = 64 := by
  sorry


end NUMINAMATH_CALUDE_cherry_trees_planted_l1770_177006


namespace NUMINAMATH_CALUDE_conference_room_capacity_l1770_177071

theorem conference_room_capacity 
  (num_rooms : ℕ) 
  (current_occupancy : ℕ) 
  (occupancy_ratio : ℚ) :
  num_rooms = 6 →
  current_occupancy = 320 →
  occupancy_ratio = 2/3 →
  (current_occupancy : ℚ) / occupancy_ratio / num_rooms = 80 := by
  sorry

end NUMINAMATH_CALUDE_conference_room_capacity_l1770_177071


namespace NUMINAMATH_CALUDE_loan_sum_proof_l1770_177008

theorem loan_sum_proof (x y : ℝ) : 
  x * (3 / 100) * 5 = y * (5 / 100) * 3 →
  y = 1332.5 →
  x + y = 2665 := by
sorry

end NUMINAMATH_CALUDE_loan_sum_proof_l1770_177008


namespace NUMINAMATH_CALUDE_march_greatest_drop_l1770_177094

/-- Represents the months of the year --/
inductive Month
| January
| February
| March
| April
| May
| June

/-- The price change for each month --/
def price_change : Month → ℝ
| Month.January => -0.5
| Month.February => 1.5
| Month.March => -3.0
| Month.April => 2.0
| Month.May => -1.0
| Month.June => -2.5

/-- The fixed transaction fee --/
def transaction_fee : ℝ := 1.0

/-- The adjusted price change after applying the transaction fee --/
def adjusted_price_change (m : Month) : ℝ :=
  price_change m - transaction_fee

/-- Theorem stating that March has the greatest monthly drop --/
theorem march_greatest_drop :
  ∀ m : Month, m ≠ Month.March →
  adjusted_price_change Month.March ≤ adjusted_price_change m :=
by sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l1770_177094


namespace NUMINAMATH_CALUDE_perimeter_sum_equals_original_l1770_177060

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithIncircle where
  perimeter : ℝ
  incircle : Set ℝ × ℝ

/-- Represents a triangle cut off from the original triangle -/
structure CutOffTriangle where
  perimeter : ℝ
  touchesIncircle : Bool

/-- The theorem stating that the perimeter of the original triangle
    is equal to the sum of the perimeters of the cut-off triangles -/
theorem perimeter_sum_equals_original
  (original : TriangleWithIncircle)
  (cutoff1 cutoff2 cutoff3 : CutOffTriangle)
  (h1 : cutoff1.touchesIncircle = true)
  (h2 : cutoff2.touchesIncircle = true)
  (h3 : cutoff3.touchesIncircle = true) :
  original.perimeter = cutoff1.perimeter + cutoff2.perimeter + cutoff3.perimeter :=
sorry

end NUMINAMATH_CALUDE_perimeter_sum_equals_original_l1770_177060


namespace NUMINAMATH_CALUDE_arithmetic_mean_fractions_l1770_177049

theorem arithmetic_mean_fractions : 
  let a := 8 / 11
  let b := 9 / 11
  let c := 5 / 6
  c = (a + b) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_fractions_l1770_177049


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1770_177059

theorem geometric_sequence_third_term
  (a : ℕ → ℕ)  -- The sequence
  (h1 : a 1 = 5)  -- First term is 5
  (h2 : a 4 = 320)  -- Fourth term is 320
  (h_geom : ∀ n : ℕ, n > 0 → a (n + 1) = a n * (a 2 / a 1))  -- Geometric sequence property
  : a 3 = 80 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1770_177059


namespace NUMINAMATH_CALUDE_import_value_calculation_l1770_177097

/-- Given the export value and its relationship to the import value, 
    calculate the import value. -/
theorem import_value_calculation (export_value : ℝ) (import_value : ℝ) : 
  export_value = 8.07 ∧ 
  export_value = 1.5 * import_value + 1.11 → 
  sorry

end NUMINAMATH_CALUDE_import_value_calculation_l1770_177097


namespace NUMINAMATH_CALUDE_missing_number_proof_l1770_177041

theorem missing_number_proof (x : ℝ) (y : ℝ) : 
  (12 + x + y + 78 + 104) / 5 = 62 ∧ 
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 → 
  y = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1770_177041


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l1770_177085

/-- The point symmetric to A(3, 4) with respect to the x-axis -/
def symmetric_point : ℝ × ℝ := (3, -4)

/-- The original point A -/
def point_A : ℝ × ℝ := (3, 4)

/-- Theorem stating that symmetric_point is indeed symmetric to point_A with respect to the x-axis -/
theorem symmetric_point_correct :
  symmetric_point.1 = point_A.1 ∧
  symmetric_point.2 = -point_A.2 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l1770_177085


namespace NUMINAMATH_CALUDE_sector_area_l1770_177027

/-- The area of a sector with radius 2 and central angle π/4 is π/2 -/
theorem sector_area (r : ℝ) (α : ℝ) (S : ℝ) : 
  r = 2 → α = π / 4 → S = (1 / 2) * r^2 * α → S = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1770_177027


namespace NUMINAMATH_CALUDE_inverse_proportional_m_range_l1770_177040

/-- Given an inverse proportional function y = (1 - 2m) / x with two points
    A(x₁, y₁) and B(x₂, y₂) on its graph, where x₁ < 0 < x₂ and y₁ < y₂,
    prove that the range of m is m < 1/2. -/
theorem inverse_proportional_m_range (m x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = (1 - 2*m) / x₁)
  (h2 : y₂ = (1 - 2*m) / x₂)
  (h3 : x₁ < 0)
  (h4 : 0 < x₂)
  (h5 : y₁ < y₂) :
  m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_inverse_proportional_m_range_l1770_177040


namespace NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coord_l1770_177042

theorem degenerate_ellipse_max_y_coord :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ (x^2 / 36) + ((y + 5)^2 / 16)
  ∀ (x y : ℝ), f (x, y) = 0 → y ≤ -5 :=
by sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coord_l1770_177042


namespace NUMINAMATH_CALUDE_vertical_tangent_iff_negative_a_l1770_177092

/-- A function f(x) = ax^2 + ln(x) has a vertical tangent line if and only if a < 0 -/
theorem vertical_tangent_iff_negative_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ ¬ ∃ y : ℝ, HasDerivAt (fun x => a * x^2 + Real.log x) y x) ↔ a < 0 :=
sorry

end NUMINAMATH_CALUDE_vertical_tangent_iff_negative_a_l1770_177092


namespace NUMINAMATH_CALUDE_abc_inequality_l1770_177096

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c = a^(1/7) + b^(1/7) + c^(1/7)) :
  a^a * b^b * c^c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1770_177096


namespace NUMINAMATH_CALUDE_pilot_weeks_flown_l1770_177087

def miles_tuesday : ℕ := 1134
def miles_thursday : ℕ := 1475
def total_miles : ℕ := 7827

theorem pilot_weeks_flown : 
  (total_miles : ℚ) / (miles_tuesday + miles_thursday : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_pilot_weeks_flown_l1770_177087


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l1770_177069

/-- The functional equation that f must satisfy for all x and y -/
def functional_equation (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

/-- The theorem stating the only solutions to the functional equation -/
theorem functional_equation_solutions :
  ∀ α : ℝ, ∀ f : ℝ → ℝ,
    functional_equation f α →
    ((α = 1 ∧ ∀ x, f x = -x) ∨ (α = -1 ∧ ∀ x, f x = x)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l1770_177069


namespace NUMINAMATH_CALUDE_sheila_work_hours_l1770_177001

/-- Represents Sheila's work schedule and earnings -/
structure WorkSchedule where
  hourly_rate : ℕ
  weekly_earnings : ℕ
  tue_thu_hours : ℕ
  mon_wed_fri_hours : ℕ

/-- Theorem stating that given Sheila's work conditions, she works 24 hours on Mon, Wed, Fri -/
theorem sheila_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.hourly_rate = 7)
  (h2 : schedule.weekly_earnings = 252)
  (h3 : schedule.tue_thu_hours = 6 * 2)
  (h4 : schedule.weekly_earnings = 
        schedule.hourly_rate * (schedule.tue_thu_hours + schedule.mon_wed_fri_hours)) :
  schedule.mon_wed_fri_hours = 24 := by
  sorry


end NUMINAMATH_CALUDE_sheila_work_hours_l1770_177001


namespace NUMINAMATH_CALUDE_square_roots_problem_l1770_177043

theorem square_roots_problem (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2*a + 1)^2 = x ∧ (a + 5)^2 = x) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1770_177043


namespace NUMINAMATH_CALUDE_cube_greater_than_one_iff_l1770_177070

theorem cube_greater_than_one_iff (x : ℝ) : x > 1 ↔ x^3 > 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_greater_than_one_iff_l1770_177070


namespace NUMINAMATH_CALUDE_three_fourths_of_48_plus_5_l1770_177086

theorem three_fourths_of_48_plus_5 : (3 / 4 : ℚ) * 48 + 5 = 41 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_of_48_plus_5_l1770_177086


namespace NUMINAMATH_CALUDE_geometric_progressions_terms_l1770_177037

theorem geometric_progressions_terms (a₁ b₁ q₁ q₂ : ℚ) (sum : ℚ) :
  a₁ = 20 →
  q₁ = 3/4 →
  b₁ = 4 →
  q₂ = 2/3 →
  sum = 158.75 →
  (∃ n : ℕ, sum = (a₁ * b₁) * (1 - (q₁ * q₂)^n) / (1 - q₁ * q₂)) →
  (∃ n : ℕ, n = 7 ∧
    sum = (a₁ * b₁) * (1 - (q₁ * q₂)^n) / (1 - q₁ * q₂)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progressions_terms_l1770_177037


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1770_177056

theorem simplify_and_evaluate (m : ℝ) (h : m = 2) : 
  (2 * m - 6) / (m^2 - 9) / ((2 * m + 2) / (m + 3)) - m / (m + 1) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1770_177056


namespace NUMINAMATH_CALUDE_students_favoring_both_proposals_l1770_177026

theorem students_favoring_both_proposals 
  (total : ℕ) 
  (favor_A : ℕ) 
  (favor_B : ℕ) 
  (against_both : ℕ) 
  (h1 : total = 232)
  (h2 : favor_A = 172)
  (h3 : favor_B = 143)
  (h4 : against_both = 37) :
  favor_A + favor_B - (total - against_both) = 120 := by
  sorry

end NUMINAMATH_CALUDE_students_favoring_both_proposals_l1770_177026


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1770_177079

theorem trigonometric_inequality : 
  let a := Real.sin (2 * Real.pi / 5)
  let b := Real.cos (5 * Real.pi / 6)
  let c := Real.tan (7 * Real.pi / 5)
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1770_177079


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1770_177081

theorem coefficient_x_cubed_in_expansion : 
  let n : ℕ := 5
  let k : ℕ := 3
  let a : ℤ := 1
  let b : ℤ := -2
  (n.choose k) * (b ^ k) * (a ^ (n - k)) = -80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1770_177081


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_difference_product_relation_l1770_177044

theorem reciprocal_sum_of_difference_product_relation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x - y = 3 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_difference_product_relation_l1770_177044


namespace NUMINAMATH_CALUDE_election_percentage_l1770_177038

/-- Given an election with 700 total votes where the winning candidate has a majority of 476 votes,
    prove that the winning candidate received 84% of the votes. -/
theorem election_percentage (total_votes : ℕ) (winning_majority : ℕ) (winning_percentage : ℚ) :
  total_votes = 700 →
  winning_majority = 476 →
  winning_percentage = 84 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = winning_majority :=
by sorry

end NUMINAMATH_CALUDE_election_percentage_l1770_177038


namespace NUMINAMATH_CALUDE_vectors_form_basis_l1770_177023

theorem vectors_form_basis (a b : ℝ × ℝ) : 
  a = (2, 6) → b = (-1, 3) → LinearIndependent ℝ ![a, b] := by sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l1770_177023


namespace NUMINAMATH_CALUDE_distance_BA_is_54_l1770_177095

/-- Represents a circular path with three points -/
structure CircularPath where
  -- Distance from A to B
  dAB : ℝ
  -- Distance from B to C
  dBC : ℝ
  -- Distance from C to A
  dCA : ℝ
  -- Ensure all distances are positive
  all_positive : 0 < dAB ∧ 0 < dBC ∧ 0 < dCA

/-- The distance from B to A in the opposite direction on the circular path -/
def distance_BA (path : CircularPath) : ℝ :=
  path.dBC + path.dCA

/-- Theorem stating the distance from B to A in the opposite direction -/
theorem distance_BA_is_54 (path : CircularPath) 
  (h1 : path.dAB = 30) 
  (h2 : path.dBC = 28) 
  (h3 : path.dCA = 26) : 
  distance_BA path = 54 := by
  sorry

end NUMINAMATH_CALUDE_distance_BA_is_54_l1770_177095


namespace NUMINAMATH_CALUDE_range_of_a_l1770_177046

-- Define the open interval (1, 2)
def open_interval := {x : ℝ | 1 < x ∧ x < 2}

-- Define the inequality condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x ∈ open_interval, (x - 1)^2 < Real.log x / Real.log a

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, inequality_holds a ↔ a ∈ {a : ℝ | 1 < a ∧ a ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1770_177046


namespace NUMINAMATH_CALUDE_sector_central_angle_l1770_177076

/-- Proves that a circular sector with arc length 4 and area 2 has a central angle of 4 radians -/
theorem sector_central_angle (l : ℝ) (A : ℝ) (θ : ℝ) (r : ℝ) :
  l = 4 →
  A = 2 →
  l = r * θ →
  A = 1/2 * r^2 * θ →
  θ = 4 := by
sorry


end NUMINAMATH_CALUDE_sector_central_angle_l1770_177076


namespace NUMINAMATH_CALUDE_sequence_sum_l1770_177005

theorem sequence_sum (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n > 1, a n - a (n-1) = n) :
  ∀ n : ℕ, n ≥ 1 → a n = n * (n + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1770_177005


namespace NUMINAMATH_CALUDE_sine_product_less_than_quarter_l1770_177093

-- Define a structure for a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Theorem statement
theorem sine_product_less_than_quarter (t : Triangle) :
  Real.sin (t.A / 2) * Real.sin (t.B / 2) * Real.sin (t.C / 2) < 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sine_product_less_than_quarter_l1770_177093


namespace NUMINAMATH_CALUDE_variance_of_five_numbers_l1770_177000

theorem variance_of_five_numbers (m : ℝ) 
  (h : (1 + 2 + 3 + 4 + m) / 5 = 3) : 
  ((1 - 3)^2 + (2 - 3)^2 + (3 - 3)^2 + (4 - 3)^2 + (m - 3)^2) / 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_five_numbers_l1770_177000


namespace NUMINAMATH_CALUDE_sum_of_erased_numbers_l1770_177058

/-- Represents a sequence of odd numbers -/
def OddSequence (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * i + 1)

/-- The sum of the first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n ^ 2

/-- Theorem: Sum of erased numbers in the sequence -/
theorem sum_of_erased_numbers
  (n : ℕ) -- Length of the first part
  (h1 : sumOddNumbers (n + 2) = 4147) -- Sum of third part is 4147
  (h2 : n > 0) -- Ensure non-empty sequence
  : ∃ (a b : ℕ), a ∈ OddSequence (4 * n + 6) ∧ 
                 b ∈ OddSequence (4 * n + 6) ∧ 
                 a + b = 168 :=
sorry

end NUMINAMATH_CALUDE_sum_of_erased_numbers_l1770_177058


namespace NUMINAMATH_CALUDE_factorization_equality_l1770_177061

theorem factorization_equality (x y : ℝ) : 3*x^2 + 6*x*y + 3*y^2 = 3*(x+y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1770_177061


namespace NUMINAMATH_CALUDE_interest_difference_implies_principal_l1770_177090

/-- Proves that if the difference between compound interest and simple interest 
    on a sum at 10% per annum for 2 years is Rs. 61, then the sum (principal) is Rs. 6100. -/
theorem interest_difference_implies_principal (P : ℝ) : 
  P * (1 + 0.1)^2 - P - (P * 0.1 * 2) = 61 → P = 6100 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_implies_principal_l1770_177090


namespace NUMINAMATH_CALUDE_two_car_speeds_l1770_177035

/-- Represents the speed of two cars traveling in opposite directions -/
structure TwoCarSpeeds where
  slower : ℝ
  faster : ℝ
  speed_difference : faster = slower + 10
  total_distance : 5 * slower + 5 * faster = 500

/-- Theorem stating the speeds of the two cars -/
theorem two_car_speeds : ∃ (s : TwoCarSpeeds), s.slower = 45 ∧ s.faster = 55 := by
  sorry

end NUMINAMATH_CALUDE_two_car_speeds_l1770_177035


namespace NUMINAMATH_CALUDE_min_value_is_214_l1770_177053

-- Define the type for our permutations
def Permutation := Fin 9 → Fin 9

-- Define the function we want to minimize
def f (p : Permutation) : ℕ :=
  let x₁ := (p 0).val + 1
  let x₂ := (p 1).val + 1
  let x₃ := (p 2).val + 1
  let y₁ := (p 3).val + 1
  let y₂ := (p 4).val + 1
  let y₃ := (p 5).val + 1
  let z₁ := (p 6).val + 1
  let z₂ := (p 7).val + 1
  let z₃ := (p 8).val + 1
  x₁ * x₂ * x₃ + y₁ * y₂ * y₃ + z₁ * z₂ * z₃

theorem min_value_is_214 :
  (∃ (p : Permutation), f p = 214) ∧ (∀ (p : Permutation), f p ≥ 214) := by
  sorry

end NUMINAMATH_CALUDE_min_value_is_214_l1770_177053


namespace NUMINAMATH_CALUDE_cistern_length_l1770_177074

/-- Given a cistern with specified dimensions, prove its length is 12 meters. -/
theorem cistern_length (width : ℝ) (depth : ℝ) (total_area : ℝ) :
  width = 14 →
  depth = 1.25 →
  total_area = 233 →
  width * depth * 2 + width * (total_area / width / depth - width) + depth * (total_area / width / depth - width) * 2 = total_area →
  total_area / width / depth - width = 12 :=
by sorry

end NUMINAMATH_CALUDE_cistern_length_l1770_177074


namespace NUMINAMATH_CALUDE_tailor_cut_l1770_177082

theorem tailor_cut (skirt_cut pants_cut : ℝ) : 
  skirt_cut = 0.75 → 
  skirt_cut = pants_cut + 0.25 → 
  pants_cut = 0.50 := by
sorry

end NUMINAMATH_CALUDE_tailor_cut_l1770_177082


namespace NUMINAMATH_CALUDE_no_formula_matches_l1770_177021

def x : List ℕ := [1, 2, 3, 4, 5]
def y : List ℕ := [5, 15, 33, 61, 101]

def formula_a (x : ℕ) : ℕ := 2 * x^3 + 3 * x^2 - x + 1
def formula_b (x : ℕ) : ℕ := 3 * x^3 + x^2 + x + 1
def formula_c (x : ℕ) : ℕ := 2 * x^3 + x^2 + x + 1
def formula_d (x : ℕ) : ℕ := 2 * x^3 + x^2 + x - 1

theorem no_formula_matches : 
  (∃ i, List.get! x i ≠ 0 ∧ formula_a (List.get! x i) ≠ List.get! y i) ∧
  (∃ i, List.get! x i ≠ 0 ∧ formula_b (List.get! x i) ≠ List.get! y i) ∧
  (∃ i, List.get! x i ≠ 0 ∧ formula_c (List.get! x i) ≠ List.get! y i) ∧
  (∃ i, List.get! x i ≠ 0 ∧ formula_d (List.get! x i) ≠ List.get! y i) :=
by sorry

end NUMINAMATH_CALUDE_no_formula_matches_l1770_177021


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l1770_177047

theorem ellipse_parabola_intersection (n m : ℝ) :
  (∀ x y : ℝ, x^2/n + y^2/9 = 1 ∧ y = x^2 - m → 
    (3/n < m ∧ m < (4*m^2 + 9)/(4*m) ∧ m > 3/2)) ∧
  (m = 4 ∧ n = 4 → 
    ∀ x y : ℝ, x^2/n + y^2/9 = 1 ∧ y = x^2 - m → 
      4*x^2 + 4*y^2 - 5*y - 16 = 0) := by
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l1770_177047


namespace NUMINAMATH_CALUDE_custom_mul_property_l1770_177088

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := (a + b + 1)^2

/-- Theorem stating that (x-1) * (1-x) = 1 for all real x -/
theorem custom_mul_property (x : ℝ) : custom_mul (x - 1) (1 - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_property_l1770_177088


namespace NUMINAMATH_CALUDE_circle_symmetry_l1770_177062

-- Define the symmetry condition
def symmetric_circles (C : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C ↔ ((-y : ℝ) - 1)^2 + (-x)^2 = 1

-- State the theorem
theorem circle_symmetry :
  ∀ C : Set (ℝ × ℝ),
  symmetric_circles C →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + (y + 1)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1770_177062
