import Mathlib

namespace NUMINAMATH_GPT_area_of_square_l1790_179098

theorem area_of_square (d : ℝ) (hd : d = 14 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 196 := by
  sorry

end NUMINAMATH_GPT_area_of_square_l1790_179098


namespace NUMINAMATH_GPT_incorrect_comparison_l1790_179022

theorem incorrect_comparison :
  ¬ (- (2 / 3) < - (4 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_comparison_l1790_179022


namespace NUMINAMATH_GPT_perpendicular_tangent_inequality_l1790_179036

variable {A B C : Type} 

-- Definitions according to conditions in part a)
def isAcuteAngledTriangle (a b c : Type) : Prop :=
  -- A triangle being acute-angled in Euclidean geometry
  sorry

def triangleArea (a b c : Type) : ℝ :=
  -- Definition of the area of a triangle
  sorry

def perpendicularLengthToLine (point line : Type) : ℝ :=
  -- Length of the perpendicular from a point to a line
  sorry

def tangentOfAngleA (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle A in the triangle
  sorry

def tangentOfAngleB (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle B in the triangle
  sorry

def tangentOfAngleC (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle C in the triangle
  sorry

theorem perpendicular_tangent_inequality (a b c line : Type) 
  (ht : isAcuteAngledTriangle a b c)
  (u := perpendicularLengthToLine a line)
  (v := perpendicularLengthToLine b line)
  (w := perpendicularLengthToLine c line):
  u^2 * tangentOfAngleA a b c + v^2 * tangentOfAngleB a b c + w^2 * tangentOfAngleC a b c ≥ 
  2 * triangleArea a b c :=
sorry

end NUMINAMATH_GPT_perpendicular_tangent_inequality_l1790_179036


namespace NUMINAMATH_GPT_abs_sum_div_diff_sqrt_7_5_l1790_179024

theorem abs_sum_div_diff_sqrt_7_5 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 12 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 5) :=
by
  sorry

end NUMINAMATH_GPT_abs_sum_div_diff_sqrt_7_5_l1790_179024


namespace NUMINAMATH_GPT_quadratic_has_two_real_distinct_roots_and_find_m_l1790_179003

theorem quadratic_has_two_real_distinct_roots_and_find_m 
  (m : ℝ) :
  (x : ℝ) → 
  (h1 : x^2 - (2 * m - 2) * x + (m^2 - 2 * m) = 0) →
  (x1 x2 : ℝ) →
  (h2 : x1^2 + x2^2 = 10) →
  (x1 + x2 = 2 * m - 2) →
  (x1 * x2 = m^2 - 2 * m) →
  (x1 ≠ x2) ∧ (m = -1 ∨ m = 3) :=
by sorry

end NUMINAMATH_GPT_quadratic_has_two_real_distinct_roots_and_find_m_l1790_179003


namespace NUMINAMATH_GPT_time_to_fill_tank_with_leak_l1790_179049

theorem time_to_fill_tank_with_leak (A L : ℚ) (hA : A = 1/6) (hL : L = 1/24) :
  (1 / (A - L)) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_time_to_fill_tank_with_leak_l1790_179049


namespace NUMINAMATH_GPT_intersection_point_exists_l1790_179040

def line_param_eq (x y z : ℝ) (t : ℝ) := x = 5 + t ∧ y = 3 - t ∧ z = 2
def plane_eq (x y z : ℝ) := 3 * x + y - 5 * z - 12 = 0

theorem intersection_point_exists : 
  ∃ t : ℝ, ∃ x y z : ℝ, line_param_eq x y z t ∧ plane_eq x y z ∧ x = 7 ∧ y = 1 ∧ z = 2 :=
by {
  -- Skipping the proof
  sorry
}

end NUMINAMATH_GPT_intersection_point_exists_l1790_179040


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1790_179004

theorem negation_of_universal_proposition :
  ¬ (∀ (m : ℝ), ∃ (x : ℝ), x^2 + x + m = 0) ↔ ∃ (m : ℝ), ¬ ∃ (x : ℝ), x^2 + x + m = 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1790_179004


namespace NUMINAMATH_GPT_sandy_gain_percent_l1790_179069

def gain_percent (purchase_price repair_cost selling_price : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost
  let gain := selling_price - total_cost
  (gain * 100) / total_cost

theorem sandy_gain_percent :
  gain_percent 900 300 1260 = 5 :=
by
  sorry

end NUMINAMATH_GPT_sandy_gain_percent_l1790_179069


namespace NUMINAMATH_GPT_savings_percentage_is_correct_l1790_179019

-- Definitions for given conditions
def jacket_original_price : ℕ := 100
def shirt_original_price : ℕ := 50
def shoes_original_price : ℕ := 60

def jacket_discount : ℝ := 0.30
def shirt_discount : ℝ := 0.40
def shoes_discount : ℝ := 0.25

-- Definitions for savings
def jacket_savings : ℝ := jacket_original_price * jacket_discount
def shirt_savings : ℝ := shirt_original_price * shirt_discount
def shoes_savings : ℝ := shoes_original_price * shoes_discount

-- Definition for total savings and total original cost
def total_savings : ℝ := jacket_savings + shirt_savings + shoes_savings
def total_original_cost : ℕ := jacket_original_price + shirt_original_price + shoes_original_price

-- The theorems to be proven
theorem savings_percentage_is_correct : (total_savings / total_original_cost * 100) = 30.95 := by
  sorry

end NUMINAMATH_GPT_savings_percentage_is_correct_l1790_179019


namespace NUMINAMATH_GPT_sum_of_possible_values_l1790_179084

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 4) = -21) :
    ∃ N₁ N₂ : ℝ, (N₁ + N₂ = 4 ∧ N₁ * (N₁ - 4) = -21 ∧ N₂ * (N₂ - 4) = -21) :=
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l1790_179084


namespace NUMINAMATH_GPT_larger_triangle_perimeter_l1790_179077

-- Given conditions
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

def similar (t1 t2 : Triangle) (k : ℝ) : Prop :=
  t1.a / t2.a = k ∧ t1.b / t2.b = k ∧ t1.c / t2.c = k

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Define specific triangles based on the problem
def smaller_triangle : Triangle := {a := 12, b := 12, c := 15}
def larger_triangle_ratio : ℝ := 2
def larger_triangle : Triangle := {a := 12 * larger_triangle_ratio, b := 12 * larger_triangle_ratio, c := 15 * larger_triangle_ratio}

-- Main theorem statement
theorem larger_triangle_perimeter : perimeter larger_triangle = 78 :=
by 
  sorry

end NUMINAMATH_GPT_larger_triangle_perimeter_l1790_179077


namespace NUMINAMATH_GPT_black_eyes_ratio_l1790_179011

-- Define the number of people in the theater
def total_people : ℕ := 100

-- Define the number of people with blue eyes
def blue_eyes : ℕ := 19

-- Define the number of people with brown eyes
def brown_eyes : ℕ := 50

-- Define the number of people with green eyes
def green_eyes : ℕ := 6

-- Define the number of people with black eyes
def black_eyes : ℕ := total_people - (blue_eyes + brown_eyes + green_eyes)

-- Prove that the ratio of the number of people with black eyes to the total number of people is 1:4
theorem black_eyes_ratio :
  black_eyes * 4 = total_people := by
  sorry

end NUMINAMATH_GPT_black_eyes_ratio_l1790_179011


namespace NUMINAMATH_GPT_smallest_ratio_is_three_l1790_179093

theorem smallest_ratio_is_three (m n : ℕ) (a : ℕ) (h1 : 2^m + 1 = a * (2^n + 1)) (h2 : a > 1) : a = 3 :=
sorry

end NUMINAMATH_GPT_smallest_ratio_is_three_l1790_179093


namespace NUMINAMATH_GPT_express_A_using_roster_method_l1790_179052

def A := {x : ℕ | ∃ (n : ℕ), 8 / (2 - x) = n }

theorem express_A_using_roster_method :
  A = {0, 1} :=
sorry

end NUMINAMATH_GPT_express_A_using_roster_method_l1790_179052


namespace NUMINAMATH_GPT_find_single_digit_number_l1790_179006

theorem find_single_digit_number (n : ℕ) : 
  (5 < n ∧ n < 9 ∧ n > 7) ↔ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_single_digit_number_l1790_179006


namespace NUMINAMATH_GPT_positive_A_satisfies_eq_l1790_179045

theorem positive_A_satisfies_eq :
  ∃ (A : ℝ), A > 0 ∧ A^2 + 49 = 194 → A = Real.sqrt 145 :=
by
  sorry

end NUMINAMATH_GPT_positive_A_satisfies_eq_l1790_179045


namespace NUMINAMATH_GPT_profit_percentage_example_l1790_179082

noncomputable def profit_percentage (cp_total : ℝ) (cp_count : ℕ) (sp_total : ℝ) (sp_count : ℕ) : ℝ :=
  let cp_per_article := cp_total / cp_count
  let sp_per_article := sp_total / sp_count
  let profit_per_article := sp_per_article - cp_per_article
  (profit_per_article / cp_per_article) * 100

theorem profit_percentage_example : profit_percentage 25 15 33 12 = 65 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_example_l1790_179082


namespace NUMINAMATH_GPT_count_triangles_in_hexagonal_grid_l1790_179039

-- Define the number of smallest triangles in the figure.
def small_triangles : ℕ := 10

-- Define the number of medium triangles in the figure, composed of 4 small triangles each.
def medium_triangles : ℕ := 6

-- Define the number of large triangles in the figure, composed of 9 small triangles each.
def large_triangles : ℕ := 3

-- Define the number of extra-large triangle composed of 16 small triangles.
def extra_large_triangle : ℕ := 1

-- Define the total number of triangles in the figure.
def total_triangles : ℕ := small_triangles + medium_triangles + large_triangles + extra_large_triangle

-- The theorem we want to prove: the total number of triangles is 20.
theorem count_triangles_in_hexagonal_grid : total_triangles = 20 := by
  -- Placeholder for the proof.
  sorry

end NUMINAMATH_GPT_count_triangles_in_hexagonal_grid_l1790_179039


namespace NUMINAMATH_GPT_correct_operation_l1790_179009

variable (a b : ℝ)

theorem correct_operation : (-a^2 * b + 2 * a^2 * b = a^2 * b) :=
by sorry

end NUMINAMATH_GPT_correct_operation_l1790_179009


namespace NUMINAMATH_GPT_original_price_l1790_179096

theorem original_price (price_paid original_price : ℝ) 
  (h₁ : price_paid = 5) 
  (h₂ : price_paid = original_price / 10) : 
  original_price = 50 := by
  sorry

end NUMINAMATH_GPT_original_price_l1790_179096


namespace NUMINAMATH_GPT_value_of_a_m_minus_2n_l1790_179047

variable (a : ℝ) (m n : ℝ)

theorem value_of_a_m_minus_2n (h1 : a^m = 8) (h2 : a^n = 4) : a^(m - 2 * n) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_m_minus_2n_l1790_179047


namespace NUMINAMATH_GPT_cos_4theta_value_l1790_179081

theorem cos_4theta_value (theta : ℝ) 
  (h : ∑' n : ℕ, (Real.cos theta)^(2 * n) = 8) : 
  Real.cos (4 * theta) = 1 / 8 := 
sorry

end NUMINAMATH_GPT_cos_4theta_value_l1790_179081


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1790_179073

theorem simplify_and_evaluate_expression (a : ℚ) (h : a = -3/2) :
  (a + 2 - 5/(a - 2)) / ((2 * a^2 - 6 * a) / (a - 2)) = -1/2 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1790_179073


namespace NUMINAMATH_GPT_part1_part2_l1790_179005

theorem part1 (a : ℝ) (h : 48 * a^2 = 75) (ha : a > 0) : a = 5 / 4 :=
sorry

theorem part2 (θ : ℝ) 
  (h₁ : 10 * (Real.sin θ) ^ 2 = 5) 
  (h₀ : 0 < θ ∧ θ < Real.pi / 2) 
  : θ = Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1790_179005


namespace NUMINAMATH_GPT_regina_earnings_l1790_179008

def num_cows : ℕ := 20

def num_pigs (num_cows : ℕ) : ℕ := 4 * num_cows

def price_per_pig : ℕ := 400
def price_per_cow : ℕ := 800

def earnings (num_cows num_pigs price_per_cow price_per_pig : ℕ) : ℕ :=
  num_cows * price_per_cow + num_pigs * price_per_pig

theorem regina_earnings :
  earnings num_cows (num_pigs num_cows) price_per_cow price_per_pig = 48000 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_regina_earnings_l1790_179008


namespace NUMINAMATH_GPT_find_x_given_y_l1790_179020

variable (x y : ℝ)

theorem find_x_given_y :
  (0 < x) → (0 < y) → 
  (∃ k : ℝ, (3 * x^2 * y = k)) → 
  (y = 18 → x = 3) → 
  (y = 2400) → 
  x = 9 * Real.sqrt 6 / 85 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_x_given_y_l1790_179020


namespace NUMINAMATH_GPT_inequality_sqrt_sum_l1790_179054

theorem inequality_sqrt_sum (a b c : ℝ) : 
  (Real.sqrt (a^2 + b^2 - a * b) + Real.sqrt (b^2 + c^2 - b * c)) ≥ Real.sqrt (a^2 + c^2 + a * c) :=
sorry

end NUMINAMATH_GPT_inequality_sqrt_sum_l1790_179054


namespace NUMINAMATH_GPT_polynomial_divisible_by_3_l1790_179057

/--
Given q and p are integers where q is divisible by 3 and p+1 is divisible by 3,
prove that the polynomial Q(x) = x^3 - x + (p+1)x + q is divisible by 3 for any integer x.
-/
theorem polynomial_divisible_by_3 (q p : ℤ) (hq : 3 ∣ q) (hp1 : 3 ∣ (p + 1)) :
  ∀ x : ℤ, 3 ∣ (x^3 - x + (p+1) * x + q) :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_divisible_by_3_l1790_179057


namespace NUMINAMATH_GPT_emily_saves_more_using_promotion_a_l1790_179092

-- Definitions based on conditions
def price_per_pair : ℕ := 50
def promotion_a_cost : ℕ := price_per_pair + price_per_pair / 2
def promotion_b_cost : ℕ := price_per_pair + (price_per_pair - 20)

-- Statement to prove the savings
theorem emily_saves_more_using_promotion_a :
  promotion_b_cost - promotion_a_cost = 5 := by
  sorry

end NUMINAMATH_GPT_emily_saves_more_using_promotion_a_l1790_179092


namespace NUMINAMATH_GPT_sin_diff_l1790_179060

variable (θ : ℝ)
variable (hθ : 0 < θ ∧ θ < π / 2)
variable (h1 : Real.sin θ = 2 * Real.sqrt 5 / 5)

theorem sin_diff
  (hθ : 0 < θ ∧ θ < π / 2)
  (h1 : Real.sin θ = 2 * Real.sqrt 5 / 5) :
  Real.sin (θ - π / 4) = Real.sqrt 10 / 10 :=
sorry

end NUMINAMATH_GPT_sin_diff_l1790_179060


namespace NUMINAMATH_GPT_Miles_trombones_count_l1790_179085

theorem Miles_trombones_count :
  let fingers := 10
  let trumpets := fingers - 3
  let hands := 2
  let guitars := hands + 2
  let french_horns := guitars - 1
  let heads := 1
  let trombones := heads + 2
  trumpets + guitars + french_horns + trombones = 17 → trombones = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_Miles_trombones_count_l1790_179085


namespace NUMINAMATH_GPT_intersection_M_N_l1790_179086

def M := { y : ℝ | ∃ x : ℝ, y = 2^x }
def N := { y : ℝ | ∃ x : ℝ, y = 2 * Real.sin x }

theorem intersection_M_N : M ∩ N = { y : ℝ | 0 < y ∧ y ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1790_179086


namespace NUMINAMATH_GPT_inverse_proportion_quadrants_l1790_179007

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → k / x > 0) ∧ (x < 0 → k / x < 0))) ↔ k > 0 := by
  sorry

end NUMINAMATH_GPT_inverse_proportion_quadrants_l1790_179007


namespace NUMINAMATH_GPT_downstream_speed_is_28_l1790_179091

-- Define the speed of the man in still water
def speed_in_still_water : ℝ := 24

-- Define the speed of the man rowing upstream
def speed_upstream : ℝ := 20

-- Define the speed of the stream
def speed_stream : ℝ := speed_in_still_water - speed_upstream

-- Define the speed of the man rowing downstream
def speed_downstream : ℝ := speed_in_still_water + speed_stream

-- The main theorem stating that the speed of the man rowing downstream is 28 kmph
theorem downstream_speed_is_28 : speed_downstream = 28 := by
  sorry

end NUMINAMATH_GPT_downstream_speed_is_28_l1790_179091


namespace NUMINAMATH_GPT_find_m_l1790_179027

theorem find_m (m x : ℝ) (h : (m - 2) * x^2 + 3 * x + m^2 - 4 = 0) (hx : x = 0) : m = -2 :=
by sorry

end NUMINAMATH_GPT_find_m_l1790_179027


namespace NUMINAMATH_GPT_increased_cost_is_4_percent_l1790_179030

-- Initial declarations
variables (initial_cost : ℕ) (price_change_eggs price_change_apples percentage_increase : ℕ)

-- Cost definitions based on initial conditions
def initial_cost_eggs := 100
def initial_cost_apples := 100

-- Price adjustments
def new_cost_eggs := initial_cost_eggs - (initial_cost_eggs * 2 / 100)
def new_cost_apples := initial_cost_apples + (initial_cost_apples * 10 / 100)

-- New combined cost
def new_combined_cost := new_cost_eggs + new_cost_apples

-- Old combined cost
def old_combined_cost := initial_cost_eggs + initial_cost_apples

-- Increase in cost
def increase_in_cost := new_combined_cost - old_combined_cost

-- Percentage increase
def calculated_percentage_increase := (increase_in_cost * 100) / old_combined_cost

-- The proof statement
theorem increased_cost_is_4_percent :
  initial_cost = 100 →
  price_change_eggs = 2 →
  price_change_apples = 10 →
  percentage_increase = 4 →
  calculated_percentage_increase = percentage_increase :=
sorry

end NUMINAMATH_GPT_increased_cost_is_4_percent_l1790_179030


namespace NUMINAMATH_GPT_four_consecutive_even_impossible_l1790_179097

def is_four_consecutive_even_sum (S : ℕ) : Prop :=
  ∃ n : ℤ, S = 4 * n + 12

theorem four_consecutive_even_impossible :
  ¬ is_four_consecutive_even_sum 34 :=
by
  sorry

end NUMINAMATH_GPT_four_consecutive_even_impossible_l1790_179097


namespace NUMINAMATH_GPT_question_correctness_l1790_179043

theorem question_correctness (x : ℝ) :
  ¬(x^2 + x^4 = x^6) ∧
  ¬((x + 1) * (x - 1) = x^2 + 1) ∧
  ((x^3)^2 = x^6) ∧
  ¬(x^6 / x^3 = x^2) :=
by sorry

end NUMINAMATH_GPT_question_correctness_l1790_179043


namespace NUMINAMATH_GPT_sum_of_products_circle_l1790_179025

theorem sum_of_products_circle 
  (a b c d : ℤ) 
  (h : a + b + c + d = 0) : 
  -((a * (b + d)) + (b * (a + c)) + (c * (b + d)) + (d * (a + c))) = 2 * (a + c) ^ 2 :=
sorry

end NUMINAMATH_GPT_sum_of_products_circle_l1790_179025


namespace NUMINAMATH_GPT_similar_triangles_x_value_l1790_179050

-- Define the conditions of the problem
variables (x : ℝ) (h₁ : 10 / x = 8 / 5)

-- State the theorem/proof problem
theorem similar_triangles_x_value : x = 6.25 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_similar_triangles_x_value_l1790_179050


namespace NUMINAMATH_GPT_compare_abc_l1790_179014

noncomputable def a := Real.sqrt 0.3
noncomputable def b := Real.sqrt 0.4
noncomputable def c := Real.log 0.6 / Real.log 3

theorem compare_abc : c < a ∧ a < b :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_compare_abc_l1790_179014


namespace NUMINAMATH_GPT_no_such_ab_exists_l1790_179041

theorem no_such_ab_exists : ¬ ∃ (a b : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi → (a * x + b)^2 - Real.cos x * (a * x + b) < (1 / 4) * (Real.sin x)^2 :=
by
  sorry

end NUMINAMATH_GPT_no_such_ab_exists_l1790_179041


namespace NUMINAMATH_GPT_Tom_water_intake_daily_l1790_179051

theorem Tom_water_intake_daily (cans_per_day : ℕ) (oz_per_can : ℕ) (fluid_per_week : ℕ) (days_per_week : ℕ)
  (h1 : cans_per_day = 5) 
  (h2 : oz_per_can = 12) 
  (h3 : fluid_per_week = 868) 
  (h4 : days_per_week = 7) : 
  ((fluid_per_week - (cans_per_day * oz_per_can * days_per_week)) / days_per_week) = 64 := 
sorry

end NUMINAMATH_GPT_Tom_water_intake_daily_l1790_179051


namespace NUMINAMATH_GPT_value_at_points_zero_l1790_179012

def odd_function (v : ℝ → ℝ) := ∀ x : ℝ, v (-x) = -v x

theorem value_at_points_zero (v : ℝ → ℝ)
  (hv : odd_function v) :
  v (-2.1) + v (-1.2) + v (1.2) + v (2.1) = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_at_points_zero_l1790_179012


namespace NUMINAMATH_GPT_part_I_part_II_l1790_179031

open Real  -- Specify that we are working with real numbers

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 2) - abs (x + a)

-- The first theorem: Prove the result for a = 1
theorem part_I (x : ℝ) : f x 1 + x > 0 ↔ (x > -3 ∧ x < 1 ∨ x > 3) :=
by
  sorry

-- The second theorem: Prove the range of a such that f(x) ≤ 3 for all x
theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≤ 3) ↔ (-5 ≤ a ∧ a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1790_179031


namespace NUMINAMATH_GPT_sum_squares_of_solutions_eq_l1790_179023

noncomputable def sum_of_squares_of_solutions : ℚ := sorry

theorem sum_squares_of_solutions_eq :
  (∃ x : ℚ, abs (x^2 - x + (1 : ℚ) / 2010) = (1 : ℚ) / 2010) →
  sum_of_squares_of_solutions = (2008 : ℚ) / 1005 :=
sorry

end NUMINAMATH_GPT_sum_squares_of_solutions_eq_l1790_179023


namespace NUMINAMATH_GPT_sum_of_fractions_and_decimal_l1790_179076

theorem sum_of_fractions_and_decimal :
  (3 / 10) + (5 / 100) + (7 / 1000) + 0.001 = 0.358 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_and_decimal_l1790_179076


namespace NUMINAMATH_GPT_max_positive_integer_value_l1790_179056

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n: ℕ, ∃ q: ℝ, a (n + 1) = a n * q

theorem max_positive_integer_value
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : ∀ n, a n > 0)
  (h3 : a 2 * a 4 = 4)
  (h4 : a 1 + a 2 + a 3 = 14) : 
  ∃ n, n ≤ 4 ∧ a n * a (n+1) * a (n+2) > 1 / 9 :=
sorry

end NUMINAMATH_GPT_max_positive_integer_value_l1790_179056


namespace NUMINAMATH_GPT_average_of_numbers_l1790_179065

theorem average_of_numbers (x : ℝ) (h : (2 + x + 12) / 3 = 8) : x = 10 :=
by sorry

end NUMINAMATH_GPT_average_of_numbers_l1790_179065


namespace NUMINAMATH_GPT_geometric_sequence_eighth_term_l1790_179026

theorem geometric_sequence_eighth_term (a r : ℝ) (h₀ : a = 27) (h₁ : r = 1/3) :
  a * r^7 = 1/81 :=
by
  rw [h₀, h₁]
  sorry

end NUMINAMATH_GPT_geometric_sequence_eighth_term_l1790_179026


namespace NUMINAMATH_GPT_root_bounds_l1790_179089

noncomputable def sqrt (r : ℝ) (n : ℕ) := r^(1 / n)

theorem root_bounds (a b c d : ℝ) (n p x y : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hn : 0 < n) (hp : 0 < p) (hx : 0 < x) (hy : 0 < y) :
  sqrt d y < sqrt (a * b * c * d) (n + p + x + y) ∧
  sqrt (a * b * c * d) (n + p + x + y) < sqrt a n := 
sorry

end NUMINAMATH_GPT_root_bounds_l1790_179089


namespace NUMINAMATH_GPT_intersection_result_l1790_179013

open Set

namespace ProofProblem

def A : Set ℝ := {x | |x| ≤ 4}
def B : Set ℝ := {x | 4 ≤ x ∧ x < 5}

theorem intersection_result : A ∩ B = {4} :=
  sorry

end ProofProblem

end NUMINAMATH_GPT_intersection_result_l1790_179013


namespace NUMINAMATH_GPT_product_discount_rate_l1790_179035

theorem product_discount_rate (cost_price marked_price : ℝ) (desired_profit_rate : ℝ) :
  cost_price = 200 → marked_price = 300 → desired_profit_rate = 0.2 →
  (∃ discount_rate : ℝ, discount_rate = 0.8 ∧ marked_price * discount_rate = cost_price * (1 + desired_profit_rate)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_product_discount_rate_l1790_179035


namespace NUMINAMATH_GPT_min_squared_distance_l1790_179048

theorem min_squared_distance : 
  ∀ (x y : ℝ), (x - y = 1) → (∃ (a b : ℝ), 
  ((a - 2) ^ 2 + (b - 2) ^ 2 <= (x - 2) ^ 2 + (y - 2) ^ 2) ∧ ((a - 2) ^ 2 + (b - 2) ^ 2 = 1 / 2)) := 
by
  sorry

end NUMINAMATH_GPT_min_squared_distance_l1790_179048


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1790_179038

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 9) (h2 : b = 4) (h3 : b < a + a) : a + a + b = 22 := by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1790_179038


namespace NUMINAMATH_GPT_average_age_of_troupe_l1790_179002

theorem average_age_of_troupe
  (number_females : ℕ) (number_males : ℕ) 
  (average_age_females : ℕ) (average_age_males : ℕ)
  (total_people : ℕ) (total_age : ℕ)
  (h1 : number_females = 12) 
  (h2 : number_males = 18) 
  (h3 : average_age_females = 25) 
  (h4 : average_age_males = 30)
  (h5 : total_people = 30)
  (h6 : total_age = (25 * 12 + 30 * 18)) :
  total_age / total_people = 28 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_average_age_of_troupe_l1790_179002


namespace NUMINAMATH_GPT_Jack_sent_correct_number_of_BestBuy_cards_l1790_179061

def price_BestBuy_gift_card : ℕ := 500
def price_Walmart_gift_card : ℕ := 200
def initial_BestBuy_gift_cards : ℕ := 6
def initial_Walmart_gift_cards : ℕ := 9

def total_price_of_initial_gift_cards : ℕ :=
  (initial_BestBuy_gift_cards * price_BestBuy_gift_card) +
  (initial_Walmart_gift_cards * price_Walmart_gift_card)

def price_of_Walmart_sent : ℕ := 2 * price_Walmart_gift_card
def value_of_gift_cards_remaining : ℕ := 3900

def prove_sent_BestBuy_worth : Prop :=
  total_price_of_initial_gift_cards - value_of_gift_cards_remaining - price_of_Walmart_sent = 1 * price_BestBuy_gift_card

theorem Jack_sent_correct_number_of_BestBuy_cards :
  prove_sent_BestBuy_worth :=
by
  sorry

end NUMINAMATH_GPT_Jack_sent_correct_number_of_BestBuy_cards_l1790_179061


namespace NUMINAMATH_GPT_present_population_l1790_179055

theorem present_population (P : ℝ) (h : 1.04 * P = 1289.6) : P = 1240 :=
by
  sorry

end NUMINAMATH_GPT_present_population_l1790_179055


namespace NUMINAMATH_GPT_mayo_bottles_count_l1790_179017

theorem mayo_bottles_count
  (ketchup_ratio mayo_ratio : ℕ) 
  (ratio_multiplier ketchup_bottles : ℕ)
  (h_ratio_eq : 3 = ketchup_ratio)
  (h_mayo_ratio_eq : 2 = mayo_ratio)
  (h_ketchup_bottles_eq : 6 = ketchup_bottles)
  (h_ratio_condition : ketchup_bottles * mayo_ratio = ketchup_ratio * ratio_multiplier) :
  ratio_multiplier = 4 := 
by 
  sorry

end NUMINAMATH_GPT_mayo_bottles_count_l1790_179017


namespace NUMINAMATH_GPT_radius_of_C3_correct_l1790_179068

noncomputable def radius_of_C3
  (C1 C2 C3 : Type)
  (r1 r2 : ℝ)
  (A B T : Type)
  (TA : ℝ) : ℝ :=
if h1 : r1 = 2 ∧ r2 = 3
    ∧ (TA = 4) -- Conditions 1 and 2
   then 8
   else 0

-- Proof statement
theorem radius_of_C3_correct
  (C1 C2 C3 : Type)
  (r1 r2 : ℝ)
  (A B T : Type)
  (TA : ℝ)
  (h1 : r1 = 2)
  (h2 : r2 = 3)
  (h3 : TA = 4) :
  radius_of_C3 C1 C2 C3 r1 r2 A B T TA = 8 :=
by 
  sorry

end NUMINAMATH_GPT_radius_of_C3_correct_l1790_179068


namespace NUMINAMATH_GPT_units_digit_k_squared_plus_2_k_l1790_179015

noncomputable def k : ℕ := 2017^2 + 2^2017

theorem units_digit_k_squared_plus_2_k : (k^2 + 2^k) % 10 = 3 := 
  sorry

end NUMINAMATH_GPT_units_digit_k_squared_plus_2_k_l1790_179015


namespace NUMINAMATH_GPT_milan_billed_minutes_l1790_179063

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 2) 
  (h2 : cost_per_minute = 0.12) 
  (h3 : total_bill = 23.36) : 
  (total_bill - monthly_fee) / cost_per_minute = 178 := 
by 
  sorry

end NUMINAMATH_GPT_milan_billed_minutes_l1790_179063


namespace NUMINAMATH_GPT_frustum_volume_correct_l1790_179075

noncomputable def base_length := 20 -- cm
noncomputable def base_width := 10 -- cm
noncomputable def original_altitude := 12 -- cm
noncomputable def cut_height := 6 -- cm
noncomputable def base_area := base_length * base_width -- cm^2
noncomputable def original_volume := (1 / 3 : ℚ) * base_area * original_altitude -- cm^3
noncomputable def top_area := base_area / 4 -- cm^2
noncomputable def smaller_pyramid_volume := (1 / 3 : ℚ) * top_area * cut_height -- cm^3
noncomputable def frustum_volume := original_volume - smaller_pyramid_volume -- cm^3

theorem frustum_volume_correct :
  frustum_volume = 700 :=
by
  sorry

end NUMINAMATH_GPT_frustum_volume_correct_l1790_179075


namespace NUMINAMATH_GPT_pizza_problem_l1790_179001

noncomputable def pizza_slices (total_slices pepperoni_slices mushroom_slices : ℕ) : ℕ := 
  let slices_with_both := total_slices - (pepperoni_slices + mushroom_slices - total_slices)
  slices_with_both

theorem pizza_problem 
  (total_slices pepperoni_slices mushroom_slices : ℕ)
  (h_total: total_slices = 16)
  (h_pepperoni: pepperoni_slices = 8)
  (h_mushrooms: mushroom_slices = 12)
  (h_at_least_one: pepperoni_slices + mushroom_slices - total_slices ≥ 0)
  (h_no_three_toppings: total_slices = pepperoni_slices + mushroom_slices - 
   (total_slices - (pepperoni_slices + mushroom_slices - total_slices))) : 
  pizza_slices total_slices pepperoni_slices mushroom_slices = 4 :=
by 
  rw [h_total, h_pepperoni, h_mushrooms]
  sorry

end NUMINAMATH_GPT_pizza_problem_l1790_179001


namespace NUMINAMATH_GPT_A_n_squared_l1790_179087

-- Define C(n-2)
def C_n_2 (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define A_n_2
def A_n_2 (n : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - 2)

theorem A_n_squared (n : ℕ) (hC : C_n_2 n = 15) : A_n_2 n = 30 := by
  sorry

end NUMINAMATH_GPT_A_n_squared_l1790_179087


namespace NUMINAMATH_GPT_exists_line_intersecting_all_segments_l1790_179074

theorem exists_line_intersecting_all_segments 
  (segments : List (ℝ × ℝ)) 
  (h1 : ∀ (P Q R : (ℝ × ℝ)), P ∈ segments → Q ∈ segments → R ∈ segments → ∃ (L : ℝ × ℝ → Prop), L P ∧ L Q ∧ L R) :
  ∃ (L : ℝ × ℝ → Prop), ∀ (S : (ℝ × ℝ)), S ∈ segments → L S :=
by
  sorry

end NUMINAMATH_GPT_exists_line_intersecting_all_segments_l1790_179074


namespace NUMINAMATH_GPT_pyramid_volume_l1790_179094

theorem pyramid_volume (b : ℝ) (h₀ : b > 0) :
  let base_area := (b * b * (Real.sqrt 3)) / 4
  let height := b / 2
  let volume := (1 / 3) * base_area * height
  volume = (b^3 * (Real.sqrt 3)) / 24 :=
sorry

end NUMINAMATH_GPT_pyramid_volume_l1790_179094


namespace NUMINAMATH_GPT_range_of_a_l1790_179046

theorem range_of_a (a : ℚ) (h_pos : 0 < a) (h_int_count : ∀ n : ℕ, 2 * n + 1 = 2007 -> ∃ k : ℤ, -a < ↑k ∧ ↑k < a) : 1003 < a ∧ a ≤ 1004 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1790_179046


namespace NUMINAMATH_GPT_area_identity_tg_cos_l1790_179016

variable (a b c α β γ : Real)
variable (s t : Real) (area_of_triangle : Real)

-- Assume t is the area of the triangle and s is the semiperimeter
axiom area_of_triangle_eq_heron :
  t = Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Assume trigonometric identities for tangents and cosines of half-angles
axiom tg_half_angle_α : Real.tan (α / 2) = Real.sqrt ((s - b) * (s - c) / (s * (s - a)))
axiom tg_half_angle_β : Real.tan (β / 2) = Real.sqrt ((s - c) * (s - a) / (s * (s - b)))
axiom tg_half_angle_γ : Real.tan (γ / 2) = Real.sqrt ((s - a) * (s - b) / (s * (s - c)))

axiom cos_half_angle_α : Real.cos (α / 2) = Real.sqrt (s * (s - a) / (b * c))
axiom cos_half_angle_β : Real.cos (β / 2) = Real.sqrt (s * (s - b) / (c * a))
axiom cos_half_angle_γ : Real.cos (γ / 2) = Real.sqrt (s * (s - c) / (a * b))

theorem area_identity_tg_cos :
  t = s^2 * Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) ∧
  t = (a * b * c / s) * Real.cos (α / 2) * Real.cos (β / 2) * Real.cos (γ / 2) :=
by
  sorry

end NUMINAMATH_GPT_area_identity_tg_cos_l1790_179016


namespace NUMINAMATH_GPT_polygon_area_is_400_l1790_179021

-- Definition of the points and polygon
def Point := (ℝ × ℝ)
def Polygon := List Point

def points : List Point := [(0, 0), (20, 0), (20, 20), (0, 20), (10, 0), (20, 10), (10, 20), (0, 10)]

def polygon : Polygon := [(0,0), (10,0), (20,10), (20,20), (10,20), (0,10), (0,0)]

-- Function to calculate the area of the polygon
noncomputable def polygon_area (p : Polygon) : ℝ := 
  -- Assume we have the necessary function to calculate the area of a polygon given a list of vertices
  sorry

-- Theorem statement: The area of the given polygon is 400
theorem polygon_area_is_400 : polygon_area polygon = 400 := sorry

end NUMINAMATH_GPT_polygon_area_is_400_l1790_179021


namespace NUMINAMATH_GPT_water_pouring_problem_l1790_179029

theorem water_pouring_problem : ∃ n : ℕ, n = 3 ∧
  (1 / (2 * n - 1) = 1 / 5) :=
by
  sorry

end NUMINAMATH_GPT_water_pouring_problem_l1790_179029


namespace NUMINAMATH_GPT_jason_car_count_l1790_179095

theorem jason_car_count :
  ∀ (red green purple total : ℕ),
  (green = 4 * red) →
  (red = purple + 6) →
  (purple = 47) →
  (total = purple + red + green) →
  total = 312 :=
by
  intros red green purple total h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_jason_car_count_l1790_179095


namespace NUMINAMATH_GPT_larger_number_l1790_179072

theorem larger_number (HCF LCM a b : ℕ) (h_hcf : HCF = 28) (h_factors: 12 * 15 * HCF = LCM) (h_prod : a * b = HCF * LCM) :
  max a b = 180 :=
sorry

end NUMINAMATH_GPT_larger_number_l1790_179072


namespace NUMINAMATH_GPT_factor_expression_l1790_179080

theorem factor_expression (a b : ℕ) (h_factor : (x - a) * (x - b) = x^2 - 18 * x + 72) (h_nonneg : 0 ≤ a ∧ 0 ≤ b) (h_order : a > b) : 4 * b - a = 27 := by
  sorry

end NUMINAMATH_GPT_factor_expression_l1790_179080


namespace NUMINAMATH_GPT_g_neither_even_nor_odd_l1790_179037

noncomputable def g (x : ℝ) : ℝ := Real.log (2 * x)

theorem g_neither_even_nor_odd :
  (∀ x, g (-x) = g x → false) ∧ (∀ x, g (-x) = -g x → false) :=
by
  unfold g
  sorry

end NUMINAMATH_GPT_g_neither_even_nor_odd_l1790_179037


namespace NUMINAMATH_GPT_number_of_fowls_l1790_179078

theorem number_of_fowls (chickens : ℕ) (ducks : ℕ) (h1 : chickens = 28) (h2 : ducks = 18) : chickens + ducks = 46 :=
by
  sorry

end NUMINAMATH_GPT_number_of_fowls_l1790_179078


namespace NUMINAMATH_GPT_orange_count_in_bin_l1790_179042

-- Definitions of the conditions
def initial_oranges : Nat := 5
def oranges_thrown_away : Nat := 2
def new_oranges_added : Nat := 28

-- The statement of the proof problem
theorem orange_count_in_bin : initial_oranges - oranges_thrown_away + new_oranges_added = 31 :=
by
  sorry

end NUMINAMATH_GPT_orange_count_in_bin_l1790_179042


namespace NUMINAMATH_GPT_value_of_k_l1790_179062

theorem value_of_k (x y k : ℝ) (h1 : 3 * x + 2 * y = k + 1) (h2 : 2 * x + 3 * y = k) (h3 : x + y = 2) :
  k = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l1790_179062


namespace NUMINAMATH_GPT_least_number_to_addition_l1790_179058

-- Given conditions
def n : ℤ := 2496

-- The least number to be added to n to make it divisible by 5
def least_number_to_add (n : ℤ) : ℤ :=
  if (n % 5 = 0) then 0 else (5 - (n % 5))

-- Prove that adding 4 to 2496 makes it divisible by 5
theorem least_number_to_addition : (least_number_to_add n) = 4 :=
  by
    sorry

end NUMINAMATH_GPT_least_number_to_addition_l1790_179058


namespace NUMINAMATH_GPT_factor_of_quadratic_expression_l1790_179067

def is_factor (a b : ℤ) : Prop := ∃ k, b = k * a

theorem factor_of_quadratic_expression (m : ℤ) :
  is_factor (m - 8) (m^2 - 5 * m - 24) :=
sorry

end NUMINAMATH_GPT_factor_of_quadratic_expression_l1790_179067


namespace NUMINAMATH_GPT_greatest_good_number_smallest_bad_number_l1790_179099

def is_good (M : ℕ) : Prop :=
  ∃ (a b c d : ℕ), M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ (a * d = b * c)

def is_good_iff_exists_xy (M : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≤ y ∧ M ≤ x * y ∧ (x + 1) * (y + 1) ≤ M + 49

theorem greatest_good_number : ∃ (M : ℕ), is_good M ∧ ∀ (N : ℕ), is_good N → N ≤ M :=
  by
    use 576
    sorry

theorem smallest_bad_number : ∃ (M : ℕ), ¬is_good M ∧ ∀ (N : ℕ), ¬is_good N → M ≤ N :=
  by
    use 443
    sorry

end NUMINAMATH_GPT_greatest_good_number_smallest_bad_number_l1790_179099


namespace NUMINAMATH_GPT_wavelength_scientific_notation_l1790_179053

theorem wavelength_scientific_notation :
  (0.000000193 : Float) = 1.93 * (10 : Float) ^ (-7) :=
sorry

end NUMINAMATH_GPT_wavelength_scientific_notation_l1790_179053


namespace NUMINAMATH_GPT_fruit_seller_profit_percentage_l1790_179010

/-- Suppose a fruit seller sells mangoes at the rate of Rs. 12 per kg and incurs a loss of 15%. 
    The mangoes should have been sold at Rs. 14.823529411764707 per kg to make a specific profit percentage. 
    This statement proves that the profit percentage is 5%. 
-/
theorem fruit_seller_profit_percentage :
  ∃ P : ℝ, 
    (∀ (CP SP : ℝ), 
        SP = 14.823529411764707 ∧ CP = 12 / 0.85 → 
        SP = CP * (1 + P / 100)) → 
    P = 5 := 
sorry

end NUMINAMATH_GPT_fruit_seller_profit_percentage_l1790_179010


namespace NUMINAMATH_GPT_find_carl_age_l1790_179088

variables (Alice Bob Carl : ℝ)

-- Conditions
def average_age : Prop := (Alice + Bob + Carl) / 3 = 15
def carl_twice_alice : Prop := Carl - 5 = 2 * Alice
def bob_fraction_alice : Prop := Bob + 4 = (3 / 4) * (Alice + 4)

-- Conjecture
theorem find_carl_age : average_age Alice Bob Carl ∧ carl_twice_alice Alice Carl ∧ bob_fraction_alice Alice Bob → Carl = 34.818 :=
by
  sorry

end NUMINAMATH_GPT_find_carl_age_l1790_179088


namespace NUMINAMATH_GPT_simplify_expression_l1790_179018

theorem simplify_expression (x : ℝ) : 24 * (3 * x - 4) - 6 * x = 66 * x - 96 := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1790_179018


namespace NUMINAMATH_GPT_device_prices_within_budget_l1790_179070

-- Given conditions
def x : ℝ := 12 -- Price of each type A device in thousands of dollars
def y : ℝ := 10 -- Price of each type B device in thousands of dollars
def budget : ℝ := 110 -- The budget in thousands of dollars

-- Conditions as given equations and inequalities
def condition1 : Prop := 3 * x - 2 * y = 16
def condition2 : Prop := 3 * y - 2 * x = 6
def budget_condition (a : ℕ) : Prop := 12 * a + 10 * (10 - a) ≤ budget

-- Theorem to prove
theorem device_prices_within_budget :
  condition1 ∧ condition2 ∧
  (∀ a : ℕ, a ≤ 5 → budget_condition a) :=
by sorry

end NUMINAMATH_GPT_device_prices_within_budget_l1790_179070


namespace NUMINAMATH_GPT_problem_statement_l1790_179059

noncomputable def smallest_integer_exceeding := 
  let x : ℝ := (Real.sqrt 3 + Real.sqrt 2) ^ 8
  Int.ceil x

theorem problem_statement : smallest_integer_exceeding = 5360 :=
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_problem_statement_l1790_179059


namespace NUMINAMATH_GPT_find_x_l1790_179028

theorem find_x (x : ℝ) (h : (1 / Real.log x / Real.log 5 + 1 / Real.log x / Real.log 7 + 1 / Real.log x / Real.log 11) = 1) : x = 385 := 
sorry

end NUMINAMATH_GPT_find_x_l1790_179028


namespace NUMINAMATH_GPT_smallest_k_value_eq_sqrt475_div_12_l1790_179032

theorem smallest_k_value_eq_sqrt475_div_12 :
  ∀ (k : ℝ), (dist (⟨5 * Real.sqrt 3, k - 2⟩ : ℝ × ℝ) ⟨0, 0⟩ = 5 * k) →
  k = (1 + Real.sqrt 475) / 12 := 
by
  intro k
  sorry

end NUMINAMATH_GPT_smallest_k_value_eq_sqrt475_div_12_l1790_179032


namespace NUMINAMATH_GPT_range_of_a_l1790_179044

noncomputable
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}

def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) : (A a ∪ B = B) ↔ a < -4 ∨ a > 5 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1790_179044


namespace NUMINAMATH_GPT_initial_investment_l1790_179079

theorem initial_investment :
  ∃ x : ℝ, P = 705.03 ∧ r = 0.12 ∧ n = 5 ∧ P = x * (1 + r)^n ∧ x = 400 :=
by
  let P := 705.03
  let r := 0.12
  let n := 5
  use 400
  simp [P, r, n]
  sorry

end NUMINAMATH_GPT_initial_investment_l1790_179079


namespace NUMINAMATH_GPT_jayda_spending_l1790_179071

theorem jayda_spending
  (J A : ℝ)
  (h1 : A = J + (2/5) * J)
  (h2 : J + A = 960) :
  J = 400 :=
by
  sorry

end NUMINAMATH_GPT_jayda_spending_l1790_179071


namespace NUMINAMATH_GPT_cos_three_pi_over_two_l1790_179083

theorem cos_three_pi_over_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  -- Provided as correct by the solution steps role
  sorry

end NUMINAMATH_GPT_cos_three_pi_over_two_l1790_179083


namespace NUMINAMATH_GPT_power_sum_roots_l1790_179034

theorem power_sum_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + 3 * x₁ + 1 = 0) (h₂ : x₂^2 + 3 * x₂ + 1 = 0) : 
    x₁^7 + x₂^7 = -843 := 
by 
  sorry

end NUMINAMATH_GPT_power_sum_roots_l1790_179034


namespace NUMINAMATH_GPT_find_base_numerica_l1790_179000

theorem find_base_numerica (r : ℕ) (h_gadget_cost : 5*r^2 + 3*r = 530) (h_payment : r^3 + r^2 = 1100) (h_change : 4*r^2 + 6*r = 460) :
  r = 9 :=
sorry

end NUMINAMATH_GPT_find_base_numerica_l1790_179000


namespace NUMINAMATH_GPT_smallest_lcm_value_theorem_l1790_179064

-- Define k and l to be positive 4-digit integers where gcd(k, l) = 5
def is_positive_4_digit (n : ℕ) : Prop := 1000 <= n ∧ n < 10000

noncomputable def smallest_lcm_value : ℕ :=
  201000

theorem smallest_lcm_value_theorem (k l : ℕ) (hk : is_positive_4_digit k) (hl : is_positive_4_digit l) (h : Int.gcd k l = 5) :
  ∃ m, m = Int.lcm k l ∧ m = smallest_lcm_value :=
sorry

end NUMINAMATH_GPT_smallest_lcm_value_theorem_l1790_179064


namespace NUMINAMATH_GPT_total_matches_played_l1790_179066

-- Definitions
def victories_points := 3
def draws_points := 1
def defeats_points := 0
def points_after_5_games := 8
def games_played := 5
def target_points := 40
def remaining_wins_required := 9

-- Statement to prove
theorem total_matches_played :
  ∃ M : ℕ, points_after_5_games + victories_points * remaining_wins_required < target_points -> M = games_played + remaining_wins_required + 1 :=
sorry

end NUMINAMATH_GPT_total_matches_played_l1790_179066


namespace NUMINAMATH_GPT_minimum_police_officers_needed_l1790_179090

def grid := (5, 8)
def total_intersections : ℕ := 54
def max_distance_to_police := 2

theorem minimum_police_officers_needed (min_police_needed : ℕ) :
  (min_police_needed = 6) := sorry

end NUMINAMATH_GPT_minimum_police_officers_needed_l1790_179090


namespace NUMINAMATH_GPT_oliver_gave_janet_l1790_179033

def initial_candy : ℕ := 78
def remaining_candy : ℕ := 68

theorem oliver_gave_janet : initial_candy - remaining_candy = 10 :=
by
  sorry

end NUMINAMATH_GPT_oliver_gave_janet_l1790_179033
