import Mathlib

namespace NUMINAMATH_GPT_good_numbers_product_sum_digits_not_equal_l1022_102234

def is_good_number (n : ℕ) : Prop :=
  n.digits 10 ⊆ [0, 1]

theorem good_numbers_product_sum_digits_not_equal (A B : ℕ) (hA : is_good_number A) (hB : is_good_number B) (hAB : is_good_number (A * B)) :
  ¬ ( (A.digits 10).sum * (B.digits 10).sum = ((A * B).digits 10).sum ) :=
sorry

end NUMINAMATH_GPT_good_numbers_product_sum_digits_not_equal_l1022_102234


namespace NUMINAMATH_GPT_max_correct_answers_l1022_102212

variable (c w b : ℕ) -- Define c, w, b as natural numbers

theorem max_correct_answers (h1 : c + w + b = 30) (h2 : 4 * c - w = 70) : c ≤ 20 := by
  sorry

end NUMINAMATH_GPT_max_correct_answers_l1022_102212


namespace NUMINAMATH_GPT_complement_M_intersect_N_l1022_102221

def M : Set ℤ := {m | m ≤ -3 ∨ m ≥ 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}
def complement_M : Set ℤ := {m | -3 < m ∧ m < 2} 

theorem complement_M_intersect_N : (complement_M ∩ N) = {-1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_complement_M_intersect_N_l1022_102221


namespace NUMINAMATH_GPT_compute_sum_bk_ck_l1022_102262

theorem compute_sum_bk_ck 
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 3*x^2 - 2*x + 1 =
                (x^2 + b1*x + c1) * (x^2 + b2*x + c2) * (x^2 + b3*x + c3)) :
  b1 * c1 + b2 * c2 + b3 * c3 = -2 := 
sorry

end NUMINAMATH_GPT_compute_sum_bk_ck_l1022_102262


namespace NUMINAMATH_GPT_problem1_problem2_l1022_102239

-- Problem 1: Prove the expression equals 5
theorem problem1 : (1 : ℚ) * ((1/3 : ℚ) - (3/4) + (5/6)) / (1/12) = 5 := by
  sorry

-- Problem 2: Prove the expression equals 7
theorem problem2 : ((-1 : ℤ)^2023 + |(1 - 0.5 : ℚ)| * ((-4)^2)) = 7 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1022_102239


namespace NUMINAMATH_GPT_survived_trees_difference_l1022_102227

theorem survived_trees_difference {original_trees died_trees survived_trees: ℕ} 
  (h1 : original_trees = 13) 
  (h2 : died_trees = 6)
  (h3 : survived_trees = original_trees - died_trees) :
  survived_trees - died_trees = 1 :=
by
  sorry

end NUMINAMATH_GPT_survived_trees_difference_l1022_102227


namespace NUMINAMATH_GPT_function_increasing_in_range_l1022_102266

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - m) * x - m else Real.log x / Real.log m

theorem function_increasing_in_range (m : ℝ) :
  (3 / 2 ≤ m ∧ m < 3) ↔ (∀ x y : ℝ, x < y → f m x < f m y) := by
  sorry

end NUMINAMATH_GPT_function_increasing_in_range_l1022_102266


namespace NUMINAMATH_GPT_find_x_l1022_102255

theorem find_x (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : y = 4) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1022_102255


namespace NUMINAMATH_GPT_pirates_total_coins_l1022_102286

theorem pirates_total_coins (x : ℕ) (h : (x * (x + 1)) / 2 = 5 * x) : 6 * x = 54 := by
  -- The proof will go here, but it's currently omitted with 'sorry'
  sorry

end NUMINAMATH_GPT_pirates_total_coins_l1022_102286


namespace NUMINAMATH_GPT_max_number_soap_boxes_l1022_102204

-- Definition of dimensions and volumes
def carton_length : ℕ := 25
def carton_width : ℕ := 42
def carton_height : ℕ := 60
def soap_box_length : ℕ := 7
def soap_box_width : ℕ := 12
def soap_box_height : ℕ := 5

def volume (l w h : ℕ) : ℕ := l * w * h

-- Volumes of the carton and soap box
def carton_volume : ℕ := volume carton_length carton_width carton_height
def soap_box_volume : ℕ := volume soap_box_length soap_box_width soap_box_height

-- The maximum number of soap boxes that can be placed in the carton
def max_soap_boxes : ℕ := carton_volume / soap_box_volume

theorem max_number_soap_boxes :
  max_soap_boxes = 150 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_max_number_soap_boxes_l1022_102204


namespace NUMINAMATH_GPT_inscribed_circle_quadrilateral_l1022_102299

theorem inscribed_circle_quadrilateral
  (AB CD BC AD AC BD E : ℝ)
  (r1 r2 r3 r4 : ℝ)
  (h1 : BC = AD)
  (h2 : AB + CD = BC + AD)
  (h3 : ∃ E, ∃ AC BD, AC * BD = E∧ AC > 0 ∧ BD > 0)
  (h_r1 : r1 > 0)
  (h_r2 : r2 > 0)
  (h_r3 : r3 > 0)
  (h_r4 : r4 > 0):
  1 / r1 + 1 / r3 = 1 / r2 + 1 / r4 := 
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_quadrilateral_l1022_102299


namespace NUMINAMATH_GPT_temperature_difference_l1022_102215

theorem temperature_difference (T_south T_north : ℤ) (h1 : T_south = -7) (h2 : T_north = -15) :
  T_south - T_north = 8 :=
by
  sorry

end NUMINAMATH_GPT_temperature_difference_l1022_102215


namespace NUMINAMATH_GPT_pow_two_grows_faster_than_square_l1022_102208

theorem pow_two_grows_faster_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := sorry

end NUMINAMATH_GPT_pow_two_grows_faster_than_square_l1022_102208


namespace NUMINAMATH_GPT_find_radius_of_circle_l1022_102219

theorem find_radius_of_circle
  (a b R : ℝ)
  (h1 : R^2 = a * b) :
  R = Real.sqrt (a * b) :=
by
  sorry

end NUMINAMATH_GPT_find_radius_of_circle_l1022_102219


namespace NUMINAMATH_GPT_length_of_24_l1022_102278

def length_of_integer (k : ℕ) : ℕ :=
  k.factors.length

theorem length_of_24 : length_of_integer 24 = 4 :=
by
  sorry

end NUMINAMATH_GPT_length_of_24_l1022_102278


namespace NUMINAMATH_GPT_lunks_needed_for_12_apples_l1022_102235

/-- 
  Given:
  1. 7 lunks can be traded for 4 kunks.
  2. 3 kunks will buy 5 apples.

  Prove that the number of lunks needed to purchase one dozen (12) apples is equal to 14.
-/
theorem lunks_needed_for_12_apples (L K : ℕ)
  (h1 : 7 * L = 4 * K)
  (h2 : 3 * K = 5) :
  (8 * K = 14 * L) :=
by
  sorry

end NUMINAMATH_GPT_lunks_needed_for_12_apples_l1022_102235


namespace NUMINAMATH_GPT_number_of_children_on_bus_l1022_102233

theorem number_of_children_on_bus (initial_children : ℕ) (additional_children : ℕ) (total_children : ℕ) 
  (h1 : initial_children = 26) (h2 : additional_children = 38) : total_children = 64 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_on_bus_l1022_102233


namespace NUMINAMATH_GPT_distance_AC_in_terms_of_M_l1022_102264

-- Define the given constants and the relevant equations
variables (M x : ℝ) (AB BC AC : ℝ)
axiom distance_eq_add : AB = M + BC
axiom time_AB : (M + x) / 7 = x / 5
axiom time_BC : BC = x
axiom time_S : (M + x + x) = AC

theorem distance_AC_in_terms_of_M : AC = 6 * M :=
by
  sorry

end NUMINAMATH_GPT_distance_AC_in_terms_of_M_l1022_102264


namespace NUMINAMATH_GPT_find_m_l1022_102261

noncomputable def vector_sum (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

noncomputable def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_m (m : ℝ) :
  let a := (1, m)
  let b := (3, -2)
  are_parallel (vector_sum a b) b → m = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1022_102261


namespace NUMINAMATH_GPT_sum_of_probability_fractions_l1022_102257

def total_tree_count := 15
def non_birch_count := 9
def birch_count := 6
def total_arrangements := Nat.choose 15 6
def non_adjacent_birch_arrangements := Nat.choose 10 6
def birch_probability := non_adjacent_birch_arrangements / total_arrangements
def simplified_probability_numerator := 6
def simplified_probability_denominator := 143
def answer := simplified_probability_numerator + simplified_probability_denominator

theorem sum_of_probability_fractions :
  answer = 149 := by
  sorry

end NUMINAMATH_GPT_sum_of_probability_fractions_l1022_102257


namespace NUMINAMATH_GPT_product_b6_b8_is_16_l1022_102248

-- Given conditions
variable (a : ℕ → ℝ) -- Sequence a_n
variable (b : ℕ → ℝ) -- Sequence b_n

-- Condition 1: Arithmetic sequence a_n and non-zero
axiom a_is_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d
axiom a_non_zero : ∃ n, a n ≠ 0

-- Condition 2: Equation 2a_3 - a_7^2 + 2a_n = 0
axiom a_satisfies_eq : ∀ n : ℕ, 2 * a 3 - (a 7) ^ 2 + 2 * a n = 0

-- Condition 3: Geometric sequence b_n with b_7 = a_7
axiom b_is_geometric : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n
axiom b7_equals_a7 : b 7 = a 7

-- Prove statement
theorem product_b6_b8_is_16 : b 6 * b 8 = 16 := sorry

end NUMINAMATH_GPT_product_b6_b8_is_16_l1022_102248


namespace NUMINAMATH_GPT_min_value_112_l1022_102229

noncomputable def min_value_expr (a b c d : ℝ) : ℝ :=
  20 * (a^2 + b^2 + c^2 + d^2) - (a^3 * b + a^3 * c + a^3 * d + b^3 * a + b^3 * c + b^3 * d +
                                c^3 * a + c^3 * b + c^3 * d + d^3 * a + d^3 * b + d^3 * c)

theorem min_value_112 (a b c d : ℝ) (h : a + b + c + d = 8) : min_value_expr a b c d = 112 :=
  sorry

end NUMINAMATH_GPT_min_value_112_l1022_102229


namespace NUMINAMATH_GPT_temperature_at_80_degrees_l1022_102241

theorem temperature_at_80_degrees (t : ℝ) :
  (-t^2 + 10 * t + 60 = 80) ↔ (t = 5 + 3 * Real.sqrt 5 ∨ t = 5 - 3 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_GPT_temperature_at_80_degrees_l1022_102241


namespace NUMINAMATH_GPT_gcd_18_30_l1022_102279

theorem gcd_18_30 : Int.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_18_30_l1022_102279


namespace NUMINAMATH_GPT_probability_qualified_from_A_is_correct_l1022_102272

-- Given conditions:
def p_A : ℝ := 0.7
def pass_A : ℝ := 0.95

-- Define what we need to prove:
def qualified_from_A : ℝ := p_A * pass_A

-- Theorem statement
theorem probability_qualified_from_A_is_correct :
  qualified_from_A = 0.665 :=
by
  sorry

end NUMINAMATH_GPT_probability_qualified_from_A_is_correct_l1022_102272


namespace NUMINAMATH_GPT_total_points_scored_l1022_102228

theorem total_points_scored
    (Bailey_points Chandra_points Akiko_points Michiko_points : ℕ)
    (h1 : Bailey_points = 14)
    (h2 : Michiko_points = Bailey_points / 2)
    (h3 : Akiko_points = Michiko_points + 4)
    (h4 : Chandra_points = 2 * Akiko_points) :
  Bailey_points + Michiko_points + Akiko_points + Chandra_points = 54 := by
  sorry

end NUMINAMATH_GPT_total_points_scored_l1022_102228


namespace NUMINAMATH_GPT_all_values_achievable_all_values_achievable_1_all_values_achievable_2_all_values_achievable_3_all_values_achievable_4_l1022_102205

def coin_values : Set ℤ := {1, 5, 10, 25}

theorem all_values_achievable (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 30) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_1 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 40) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_2 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 50) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_3 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 60) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_4 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 70) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

end NUMINAMATH_GPT_all_values_achievable_all_values_achievable_1_all_values_achievable_2_all_values_achievable_3_all_values_achievable_4_l1022_102205


namespace NUMINAMATH_GPT_calculate_expression_l1022_102247

theorem calculate_expression :
  (5 / 19) * ((19 / 5) * (16 / 3) + (14 / 3) * (19 / 5)) = 10 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1022_102247


namespace NUMINAMATH_GPT_alcohol_percentage_in_new_mixture_l1022_102232

theorem alcohol_percentage_in_new_mixture :
  let afterShaveLotionVolume := 200
  let afterShaveLotionConcentration := 0.35
  let solutionVolume := 75
  let solutionConcentration := 0.15
  let waterVolume := 50
  let totalVolume := afterShaveLotionVolume + solutionVolume + waterVolume
  let alcoholVolume := (afterShaveLotionVolume * afterShaveLotionConcentration) + (solutionVolume * solutionConcentration)
  let alcoholPercentage := (alcoholVolume / totalVolume) * 100
  alcoholPercentage = 25 := 
  sorry

end NUMINAMATH_GPT_alcohol_percentage_in_new_mixture_l1022_102232


namespace NUMINAMATH_GPT_squares_area_relation_l1022_102222

/-- 
Given:
1. $\alpha$ such that $\angle 1 = \angle 2 = \angle 3 = \alpha$
2. The areas of the squares are given by:
   - $S_A = \cos^4 \alpha$
   - $S_D = \sin^4 \alpha$
   - $S_B = \cos^2 \alpha \sin^2 \alpha$
   - $S_C = \cos^2 \alpha \sin^2 \alpha$

Prove that:
$S_A \cdot S_D = S_B \cdot S_C$
--/

theorem squares_area_relation (α : ℝ) :
  (Real.cos α)^4 * (Real.sin α)^4 = (Real.cos α)^2 * (Real.sin α)^2 * (Real.cos α)^2 * (Real.sin α)^2 :=
by sorry

end NUMINAMATH_GPT_squares_area_relation_l1022_102222


namespace NUMINAMATH_GPT_ratio_red_to_yellow_l1022_102297

structure MugCollection where
  total_mugs : ℕ
  red_mugs : ℕ
  blue_mugs : ℕ
  yellow_mugs : ℕ
  other_mugs : ℕ
  colors : ℕ

def HannahCollection : MugCollection :=
  { total_mugs := 40,
    red_mugs := 6,
    blue_mugs := 6 * 3,
    yellow_mugs := 12,
    other_mugs := 4,
    colors := 4 }

theorem ratio_red_to_yellow
  (hc : MugCollection)
  (h_total : hc.total_mugs = 40)
  (h_blue : hc.blue_mugs = 3 * hc.red_mugs)
  (h_yellow : hc.yellow_mugs = 12)
  (h_other : hc.other_mugs = 4)
  (h_colors : hc.colors = 4) :
  hc.red_mugs / hc.yellow_mugs = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_red_to_yellow_l1022_102297


namespace NUMINAMATH_GPT_smallest_even_divisible_by_20_and_60_l1022_102283

theorem smallest_even_divisible_by_20_and_60 : ∃ x, (Even x) ∧ (x % 20 = 0) ∧ (x % 60 = 0) ∧ (∀ y, (Even y) ∧ (y % 20 = 0) ∧ (y % 60 = 0) → x ≤ y) → x = 60 :=
by
  sorry

end NUMINAMATH_GPT_smallest_even_divisible_by_20_and_60_l1022_102283


namespace NUMINAMATH_GPT_cost_per_gumball_l1022_102226

theorem cost_per_gumball (total_money : ℕ) (num_gumballs : ℕ) (cost_each : ℕ) 
  (h1 : total_money = 32) (h2 : num_gumballs = 4) : cost_each = 8 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_cost_per_gumball_l1022_102226


namespace NUMINAMATH_GPT_silly_bills_count_l1022_102288

theorem silly_bills_count (x : ℕ) (h1 : x + 2 * (x + 11) + 3 * (x - 18) = 100) : x = 22 :=
by { sorry }

end NUMINAMATH_GPT_silly_bills_count_l1022_102288


namespace NUMINAMATH_GPT_domain_of_sqrt_fraction_l1022_102273

theorem domain_of_sqrt_fraction {x : ℝ} (h1 : x - 3 ≥ 0) (h2 : 7 - x > 0) :
  3 ≤ x ∧ x < 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_of_sqrt_fraction_l1022_102273


namespace NUMINAMATH_GPT_fraction_result_l1022_102289

theorem fraction_result (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (2 * x + 3 * y) / (x - 2 * y) = 3) : 
  (x + 2 * y) / (2 * x - y) = 11 / 17 :=
sorry

end NUMINAMATH_GPT_fraction_result_l1022_102289


namespace NUMINAMATH_GPT_cricket_run_target_l1022_102256

/-- Assuming the run rate in the first 15 overs and the required run rate for the next 35 overs to
reach a target, prove that the target number of runs is 275. -/
theorem cricket_run_target
  (run_rate_first_15 : ℝ := 3.2)
  (overs_first_15 : ℝ := 15)
  (run_rate_remaining_35 : ℝ := 6.485714285714286)
  (overs_remaining_35 : ℝ := 35)
  (runs_first_15 := run_rate_first_15 * overs_first_15)
  (runs_remaining_35 := run_rate_remaining_35 * overs_remaining_35)
  (target_runs := runs_first_15 + runs_remaining_35) :
  target_runs = 275 := by
  sorry

end NUMINAMATH_GPT_cricket_run_target_l1022_102256


namespace NUMINAMATH_GPT_area_enclosed_by_graph_l1022_102236

theorem area_enclosed_by_graph : 
  ∃ A : ℝ, (∀ x y : ℝ, |x| + |3 * y| = 9 ↔ (x = 9 ∨ x = -9 ∨ y = 3 ∨ y = -3)) → A = 54 :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_graph_l1022_102236


namespace NUMINAMATH_GPT_distinct_arrays_for_48_chairs_with_conditions_l1022_102237

theorem distinct_arrays_for_48_chairs_with_conditions : 
  ∃ n : ℕ, n = 7 ∧ 
    ∀ (m r c : ℕ), 
      m = 48 ∧ 
      2 ≤ r ∧ 
      2 ≤ c ∧ 
      r * c = m ↔ 
      (∃ (k : ℕ), 
         ((k = (m / r) ∧ r * (m / r) = m) ∨ (k = (m / c) ∧ c * (m / c) = m)) ∧ 
         r * c = m) → 
    n = 7 :=
by
  sorry

end NUMINAMATH_GPT_distinct_arrays_for_48_chairs_with_conditions_l1022_102237


namespace NUMINAMATH_GPT_sqrt_factorial_product_squared_l1022_102251

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end NUMINAMATH_GPT_sqrt_factorial_product_squared_l1022_102251


namespace NUMINAMATH_GPT_merchant_spent_initially_500_rubles_l1022_102253

theorem merchant_spent_initially_500_rubles
  (x : ℕ)
  (h1 : x + 100 > x)
  (h2 : x + 220 > x + 100)
  (h3 : x * (x + 220) = (x + 100) * (x + 100))
  : x = 500 := sorry

end NUMINAMATH_GPT_merchant_spent_initially_500_rubles_l1022_102253


namespace NUMINAMATH_GPT_pablo_puzzle_l1022_102298

open Nat

theorem pablo_puzzle (pieces_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) 
    (pieces_per_five_puzzles : ℕ) (num_five_puzzles : ℕ) (total_pieces : ℕ) 
    (num_eight_puzzles : ℕ) :

    pieces_per_hour = 100 →
    hours_per_day = 7 →
    days = 7 →
    pieces_per_five_puzzles = 500 →
    num_five_puzzles = 5 →
    num_eight_puzzles = 8 →
    total_pieces = (pieces_per_hour * hours_per_day * days) →
    num_eight_puzzles * (total_pieces - num_five_puzzles * pieces_per_five_puzzles) / num_eight_puzzles = 300 :=
by
  intros
  sorry

end NUMINAMATH_GPT_pablo_puzzle_l1022_102298


namespace NUMINAMATH_GPT_average_speed_correct_l1022_102295

def biking_time : ℕ := 30 -- in minutes
def biking_speed : ℕ := 16 -- in mph
def walking_time : ℕ := 90 -- in minutes
def walking_speed : ℕ := 4 -- in mph

theorem average_speed_correct :
  (biking_time / 60 * biking_speed + walking_time / 60 * walking_speed) / ((biking_time + walking_time) / 60) = 7 := by
  sorry

end NUMINAMATH_GPT_average_speed_correct_l1022_102295


namespace NUMINAMATH_GPT_simplify_x_cubed_simplify_expr_l1022_102287

theorem simplify_x_cubed (x : ℝ) : x * (x + 3) * (x + 5) = x^3 + 8 * x^2 + 15 * x := by
  sorry

theorem simplify_expr (x y : ℝ) : (5 * x + 2 * y) * (5 * x - 2 * y) - 5 * x * (5 * x - 3 * y) = -4 * y^2 + 15 * x * y := by
  sorry

end NUMINAMATH_GPT_simplify_x_cubed_simplify_expr_l1022_102287


namespace NUMINAMATH_GPT_number_of_people_after_10_years_l1022_102275

def number_of_people_after_n_years (n : ℕ) : ℕ :=
  Nat.recOn n 30 (fun k a_k => 3 * a_k - 20)

theorem number_of_people_after_10_years :
  number_of_people_after_n_years 10 = 1180990 := by
  sorry

end NUMINAMATH_GPT_number_of_people_after_10_years_l1022_102275


namespace NUMINAMATH_GPT_prove_y_eq_x_l1022_102231

theorem prove_y_eq_x
  (x y : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : x = 2 + 1 / y)
  (h2 : y = 2 + 1 / x) : y = x :=
sorry

end NUMINAMATH_GPT_prove_y_eq_x_l1022_102231


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l1022_102202

theorem no_positive_integer_solutions (x n r : ℕ) (h1 : x > 1) (h2 : x > 0) (h3 : n > 0) (h4 : r > 0) :
  ¬(x^(2*n + 1) = 2^r + 1 ∨ x^(2*n + 1) = 2^r - 1) :=
sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l1022_102202


namespace NUMINAMATH_GPT_classroom_student_count_l1022_102210

-- Define the conditions and the question
theorem classroom_student_count (B G : ℕ) (h1 : B / G = 3 / 5) (h2 : G = B + 4) : B + G = 16 := by
  sorry

end NUMINAMATH_GPT_classroom_student_count_l1022_102210


namespace NUMINAMATH_GPT_line_equation_l1022_102294

theorem line_equation 
  (m b k : ℝ) 
  (h1 : ∀ k, abs ((k^2 + 4 * k + 4) - (m * k + b)) = 4)
  (h2 : m * 2 + b = 8) 
  (h3 : b ≠ 0) : 
  m = 8 ∧ b = -8 :=
by sorry

end NUMINAMATH_GPT_line_equation_l1022_102294


namespace NUMINAMATH_GPT_petya_friends_count_l1022_102230

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end NUMINAMATH_GPT_petya_friends_count_l1022_102230


namespace NUMINAMATH_GPT_find_min_length_seg_O1O2_l1022_102280

noncomputable def minimum_length_O1O2 
  (X Y Z W : ℝ × ℝ) 
  (dist_XY : ℝ) (dist_YZ : ℝ) (dist_YW : ℝ)
  (O1 O2 : ℝ × ℝ) 
  (circumcenter1 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (circumcenter2 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (h1 : dist X Y = dist_XY) 
  (h2 : dist Y Z = dist_YZ) 
  (h3 : dist Y W = dist_YW) 
  (hO1 : O1 = circumcenter1 W X Y)
  (hO2 : O2 = circumcenter2 W Y Z)
  : ℝ :=
  dist O1 O2

theorem find_min_length_seg_O1O2 
  (X Y Z W : ℝ × ℝ) 
  (dist_XY : ℝ := 1)
  (dist_YZ : ℝ := 3)
  (dist_YW : ℝ := 5)
  (O1 O2 : ℝ × ℝ) 
  (circumcenter1 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (circumcenter2 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (h1 : dist X Y = dist_XY) 
  (h2 : dist Y Z = dist_YZ) 
  (h3 : dist Y W = dist_YW) 
  (hO1 : O1 = circumcenter1 W X Y)
  (hO2 : O2 = circumcenter2 W Y Z)
  : minimum_length_O1O2 X Y Z W dist_XY dist_YZ dist_YW O1 O2 circumcenter1 circumcenter2 h1 h2 h3 hO1 hO2 = 2 :=
sorry

end NUMINAMATH_GPT_find_min_length_seg_O1O2_l1022_102280


namespace NUMINAMATH_GPT_linear_equation_check_l1022_102292

theorem linear_equation_check : 
  (∃ a b : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x + b = 1)) ∧ 
  ¬ (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, a * x + b * y = 3)) ∧ 
  ¬ (∀ x : ℝ, x^2 - 2 * x = 0) ∧ 
  ¬ (∀ x : ℝ, x - 1 / x = 0) := 
sorry

end NUMINAMATH_GPT_linear_equation_check_l1022_102292


namespace NUMINAMATH_GPT_max_area_of_sector_l1022_102216

theorem max_area_of_sector (r l : ℝ) (h₁ : 2 * r + l = 12) : 
  (1 / 2) * l * r ≤ 9 :=
by sorry

end NUMINAMATH_GPT_max_area_of_sector_l1022_102216


namespace NUMINAMATH_GPT_range_of_m_min_value_of_7a_4b_l1022_102263

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x - 1| + |x + 1| - m ≥ 0) → m ≤ 2 :=
sorry

theorem min_value_of_7a_4b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_eq : 2 / (3 * a + b) + 1 / (a + 2 * b) = 2) : 7 * a + 4 * b ≥ 9 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_min_value_of_7a_4b_l1022_102263


namespace NUMINAMATH_GPT_auction_theorem_l1022_102242

def auctionProblem : Prop :=
  let starting_value := 300
  let harry_bid_round1 := starting_value + 200
  let alice_bid_round1 := harry_bid_round1 * 2
  let bob_bid_round1 := harry_bid_round1 * 3
  let highest_bid_round1 := bob_bid_round1
  let carol_bid_round2 := highest_bid_round1 * 1.5
  let sum_previous_increases := (harry_bid_round1 - starting_value) + 
                                 (alice_bid_round1 - harry_bid_round1) + 
                                 (bob_bid_round1 - harry_bid_round1)
  let dave_bid_round2 := carol_bid_round2 + sum_previous_increases
  let highest_other_bid_round3 := dave_bid_round2
  let harry_final_bid_round3 := 6000
  let difference := harry_final_bid_round3 - highest_other_bid_round3
  difference = 2050

theorem auction_theorem : auctionProblem :=
by
  sorry

end NUMINAMATH_GPT_auction_theorem_l1022_102242


namespace NUMINAMATH_GPT_regular_polygon_is_octagon_l1022_102243

theorem regular_polygon_is_octagon (n : ℕ) (interior_angle exterior_angle : ℝ) :
  interior_angle = 3 * exterior_angle ∧ interior_angle + exterior_angle = 180 → n = 8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_regular_polygon_is_octagon_l1022_102243


namespace NUMINAMATH_GPT_prime_sum_diff_condition_unique_l1022_102259

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def can_be_written_as_sum_of_two_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime (p + q)

def can_be_written_as_difference_of_two_primes (p r : ℕ) : Prop :=
  is_prime p ∧ is_prime r ∧ is_prime (p - r)

-- Question rewritten as Lean statement
theorem prime_sum_diff_condition_unique (p q r : ℕ) :
  is_prime p →
  can_be_written_as_sum_of_two_primes (p - 2) p →
  can_be_written_as_difference_of_two_primes (p + 2) p →
  p = 5 :=
sorry

end NUMINAMATH_GPT_prime_sum_diff_condition_unique_l1022_102259


namespace NUMINAMATH_GPT_percentage_increase_B_over_C_l1022_102268

noncomputable def A_m : ℕ := 537600 / 12
noncomputable def C_m : ℕ := 16000
noncomputable def ratio : ℚ := 5 / 2

noncomputable def B_m (A_m : ℕ) : ℚ := (2 * A_m) / 5

theorem percentage_increase_B_over_C :
  B_m A_m = 17920 →
  C_m = 16000 →
  (B_m A_m - C_m) / C_m * 100 = 12 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_B_over_C_l1022_102268


namespace NUMINAMATH_GPT_Intersect_A_B_l1022_102293

-- Defining the sets A and B according to the problem's conditions
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x ∈ Set.univ | x^2 - 5*x + 4 < 0}

-- Prove that the intersection of A and B is {2}
theorem Intersect_A_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_GPT_Intersect_A_B_l1022_102293


namespace NUMINAMATH_GPT_fraction_not_integer_l1022_102225

def containsExactlyTwoOccurrences (d : List ℕ) : Prop :=
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7], d.count n = 2

theorem fraction_not_integer
  (k m : ℕ)
  (hk : 14 = (List.length (Nat.digits 10 k)))
  (hm : 14 = (List.length (Nat.digits 10 m)))
  (hkd : containsExactlyTwoOccurrences (Nat.digits 10 k))
  (hmd : containsExactlyTwoOccurrences (Nat.digits 10 m))
  (hkm : k ≠ m) :
  ¬ ∃ d : ℕ, k = m * d := 
sorry

end NUMINAMATH_GPT_fraction_not_integer_l1022_102225


namespace NUMINAMATH_GPT_determinant_roots_cubic_eq_l1022_102240

noncomputable def determinant_of_matrix (a b c : ℝ) : ℝ :=
  a * (b * c - 1) - (c - 1) + (1 - b)

theorem determinant_roots_cubic_eq {a b c p q r : ℝ}
  (h1 : a + b + c = p)
  (h2 : a * b + b * c + c * a = q)
  (h3 : a * b * c = r) :
  determinant_of_matrix a b c = r - p + 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_determinant_roots_cubic_eq_l1022_102240


namespace NUMINAMATH_GPT_doubled_volume_l1022_102224

theorem doubled_volume (V : ℕ) (h : V = 4) : 8 * V = 32 := by
  sorry

end NUMINAMATH_GPT_doubled_volume_l1022_102224


namespace NUMINAMATH_GPT_polynomial_relation_l1022_102218

theorem polynomial_relation (x y : ℕ) :
  (x = 1 ∧ y = 1) ∨ 
  (x = 2 ∧ y = 4) ∨ 
  (x = 3 ∧ y = 9) ∨ 
  (x = 4 ∧ y = 16) ∨ 
  (x = 5 ∧ y = 25) → 
  y = x^2 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_relation_l1022_102218


namespace NUMINAMATH_GPT_cookies_with_five_cups_of_flour_l1022_102290

-- Define the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def additional_flour : ℕ := 5

-- State the problem
theorem cookies_with_five_cups_of_flour :
  (initial_cookies / initial_flour) * additional_flour = 40 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_cookies_with_five_cups_of_flour_l1022_102290


namespace NUMINAMATH_GPT_find_alpha_l1022_102206

open Real

def alpha_is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2

theorem find_alpha (α : ℝ) (h1 : alpha_is_acute α) (h2 : sin (α - 10 * (pi / 180)) = sqrt 3 / 2) : α = 70 * (pi / 180) :=
sorry

end NUMINAMATH_GPT_find_alpha_l1022_102206


namespace NUMINAMATH_GPT_quadratic_completing_square_l1022_102270

theorem quadratic_completing_square (b p : ℝ) (hb : b < 0)
  (h_quad_eq : ∀ x : ℝ, x^2 + b * x + (1 / 6) = (x + p)^2 + (1 / 18)) :
  b = - (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_completing_square_l1022_102270


namespace NUMINAMATH_GPT_another_representation_l1022_102260

def positive_int_set : Set ℕ := {x | x > 0}

theorem another_representation :
  {x ∈ positive_int_set | x - 3 < 2} = {1, 2, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_another_representation_l1022_102260


namespace NUMINAMATH_GPT_monotonic_increasing_f_l1022_102269

theorem monotonic_increasing_f (f g : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) 
  (hg : ∀ x, g (-x) = g x) (hfg : ∀ x, f x + g x = 3^x) :
  ∀ a b : ℝ, a > b → f a > f b :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_f_l1022_102269


namespace NUMINAMATH_GPT_cos_double_angle_l1022_102200

theorem cos_double_angle (α : ℝ) (h : Real.sin ((Real.pi / 6) + α) = 1 / 3) :
  Real.cos ((2 * Real.pi / 3) - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1022_102200


namespace NUMINAMATH_GPT_solve_for_x_l1022_102223

theorem solve_for_x (x y : ℚ) (h1 : 2 * x - 3 * y = 15) (h2 : x + 2 * y = 8) : x = 54 / 7 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1022_102223


namespace NUMINAMATH_GPT_jane_picked_fraction_l1022_102274

-- Define the total number of tomatoes initially
def total_tomatoes : ℕ := 100

-- Define the number of tomatoes remaining at the end
def remaining_tomatoes : ℕ := 15

-- Define the number of tomatoes picked in the second week
def second_week_tomatoes : ℕ := 20

-- Define the number of tomatoes picked in the third week
def third_week_tomatoes : ℕ := 2 * second_week_tomatoes

theorem jane_picked_fraction :
  ∃ (f : ℚ), f = 1 / 4 ∧
    (f * total_tomatoes + second_week_tomatoes + third_week_tomatoes + remaining_tomatoes = total_tomatoes) :=
sorry

end NUMINAMATH_GPT_jane_picked_fraction_l1022_102274


namespace NUMINAMATH_GPT_number_of_friends_l1022_102213

-- Define the conditions
def initial_apples := 55
def apples_given_to_father := 10
def apples_per_person := 9

-- Define the formula to calculate the number of friends
def friends (initial_apples apples_given_to_father apples_per_person : ℕ) : ℕ :=
  (initial_apples - apples_given_to_father - apples_per_person) / apples_per_person

-- State the Lean theorem
theorem number_of_friends :
  friends initial_apples apples_given_to_father apples_per_person = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_l1022_102213


namespace NUMINAMATH_GPT_clock_four_different_digits_l1022_102203

noncomputable def total_valid_minutes : ℕ :=
  let minutes_from_00_00_to_19_59 := 20 * 60
  let valid_minutes_1 := 2 * 9 * 4 * 7
  let minutes_from_20_00_to_23_59 := 4 * 60
  let valid_minutes_2 := 1 * 3 * 4 * 7
  valid_minutes_1 + valid_minutes_2

theorem clock_four_different_digits : total_valid_minutes = 588 :=
by
  sorry

end NUMINAMATH_GPT_clock_four_different_digits_l1022_102203


namespace NUMINAMATH_GPT_simplify_eq_neg_one_l1022_102214

variable (a b c : ℝ)

noncomputable def simplify_expression := 
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)

theorem simplify_eq_neg_one 
  (a_ne_zero : a ≠ 0) 
  (b_ne_zero : b ≠ 0) 
  (c_ne_zero : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) 
  : simplify_expression a b c = -1 :=
by sorry

end NUMINAMATH_GPT_simplify_eq_neg_one_l1022_102214


namespace NUMINAMATH_GPT_actual_price_per_gallon_l1022_102265

variable (x : ℝ)
variable (expected_price : ℝ := x) -- price per gallon that the motorist expected to pay
variable (total_cash : ℝ := 12 * x) -- total cash to buy 12 gallons at expected price
variable (actual_price : ℝ := x + 0.30) -- actual price per gallon
variable (equation : 12 * x = 10 * (x + 0.30)) -- total cash equals the cost of 10 gallons at actual price

theorem actual_price_per_gallon (x : ℝ) (h : 12 * x = 10 * (x + 0.30)) : x + 0.30 = 1.80 := 
by 
  sorry

end NUMINAMATH_GPT_actual_price_per_gallon_l1022_102265


namespace NUMINAMATH_GPT_jenny_correct_number_l1022_102238

theorem jenny_correct_number (x : ℤ) (h : x - 26 = -14) : x + 26 = 38 :=
by
  sorry

end NUMINAMATH_GPT_jenny_correct_number_l1022_102238


namespace NUMINAMATH_GPT_find_second_number_l1022_102250

theorem find_second_number (a b c : ℚ) (h1 : a + b + c = 98) (h2 : a = (2 / 3) * b) (h3 : c = (8 / 5) * b) : b = 30 :=
by sorry

end NUMINAMATH_GPT_find_second_number_l1022_102250


namespace NUMINAMATH_GPT_maria_candy_remaining_l1022_102291

theorem maria_candy_remaining :
  let c := 520.75
  let e := c / 2
  let g := 234.56
  let r := e - g
  r = 25.815 := by
  sorry

end NUMINAMATH_GPT_maria_candy_remaining_l1022_102291


namespace NUMINAMATH_GPT_coordinates_of_point_P_in_third_quadrant_l1022_102207

noncomputable def distance_from_y_axis (P : ℝ × ℝ) : ℝ := abs P.1
noncomputable def distance_from_x_axis (P : ℝ × ℝ) : ℝ := abs P.2

theorem coordinates_of_point_P_in_third_quadrant : 
  ∃ P : ℝ × ℝ, P.1 < 0 ∧ P.2 < 0 ∧ distance_from_x_axis P = 2 ∧ distance_from_y_axis P = 5 ∧ P = (-5, -2) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_point_P_in_third_quadrant_l1022_102207


namespace NUMINAMATH_GPT_ratio_problem_l1022_102271

theorem ratio_problem (x n : ℕ) (h1 : 5 * x = n) (h2 : n = 65) : x = 13 :=
by
  sorry

end NUMINAMATH_GPT_ratio_problem_l1022_102271


namespace NUMINAMATH_GPT_domain_of_f_l1022_102254

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3)^2 + (x - 6))

theorem domain_of_f :
  ∀ x : ℝ, x ≠ (5 + Real.sqrt 13) / 2 ∧ x ≠ (5 - Real.sqrt 13) / 2 → ∃ y : ℝ, y = f x :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1022_102254


namespace NUMINAMATH_GPT_correct_divisor_l1022_102217

theorem correct_divisor (X : ℕ) (D : ℕ) (H1 : X = 24 * 87) (H2 : X / D = 58) : D = 36 :=
by
  sorry

end NUMINAMATH_GPT_correct_divisor_l1022_102217


namespace NUMINAMATH_GPT_actual_positions_correct_l1022_102276

-- Define the five athletes
inductive Athlete
| A | B | C | D | E
deriving DecidableEq, Repr

open Athlete

-- Define the two predictions as lists
def first_prediction : List Athlete := [A, B, C, D, E]
def second_prediction : List Athlete := [C, E, A, B, D]

-- Define the actual positions
def actual_positions : List Athlete := [C, B, A, D, E]

-- Prove that the first prediction correctly predicted exactly three athletes
def first_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD first_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

-- Prove that the second prediction correctly predicted exactly two athletes
def second_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD second_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

theorem actual_positions_correct :
  first_prediction_correct = 3 ∧ second_prediction_correct = 2 ∧
  actual_positions = [C, B, A, D, E] :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_actual_positions_correct_l1022_102276


namespace NUMINAMATH_GPT_number_of_valid_5_digit_numbers_l1022_102267

def is_multiple_of_16 (n : Nat) : Prop := 
  n % 16 = 0

theorem number_of_valid_5_digit_numbers : Nat := 
  sorry

example : number_of_valid_5_digit_numbers = 90 :=
  sorry

end NUMINAMATH_GPT_number_of_valid_5_digit_numbers_l1022_102267


namespace NUMINAMATH_GPT_last_digit_of_sum_of_powers_l1022_102252

theorem last_digit_of_sum_of_powers {a b c d : ℕ} 
  (h1 : a = 2311) (h2 : b = 5731) (h3 : c = 3467) (h4 : d = 6563) 
  : (a^b + c^d) % 10 = 4 := by
  sorry

end NUMINAMATH_GPT_last_digit_of_sum_of_powers_l1022_102252


namespace NUMINAMATH_GPT_total_apple_trees_l1022_102277

-- Definitions and conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3
def total_trees : ℕ := ava_trees + lily_trees

-- Statement to be proved
theorem total_apple_trees :
  total_trees = 15 := by
  sorry

end NUMINAMATH_GPT_total_apple_trees_l1022_102277


namespace NUMINAMATH_GPT_difference_of_numbers_l1022_102296

theorem difference_of_numbers 
  (a b : ℕ) 
  (h1 : a + b = 23976)
  (h2 : b % 8 = 0)
  (h3 : a = 7 * b / 8) : 
  b - a = 1598 :=
sorry

end NUMINAMATH_GPT_difference_of_numbers_l1022_102296


namespace NUMINAMATH_GPT_total_revenue_l1022_102209

-- Definitions based on the conditions
def ticket_price : ℕ := 25
def first_show_tickets : ℕ := 200
def second_show_tickets : ℕ := 3 * first_show_tickets

-- Statement to prove the problem
theorem total_revenue : (first_show_tickets * ticket_price + second_show_tickets * ticket_price) = 20000 :=
by
  sorry

end NUMINAMATH_GPT_total_revenue_l1022_102209


namespace NUMINAMATH_GPT_prime_for_all_k_l1022_102285

theorem prime_for_all_k (n : ℕ) (h_n : n ≥ 2) (h_prime : ∀ k : ℕ, k ≤ Nat.sqrt (n / 3) → Prime (k^2 + k + n)) :
  ∀ k : ℕ, k ≤ n - 2 → Prime (k^2 + k + n) :=
by
  intros
  sorry

end NUMINAMATH_GPT_prime_for_all_k_l1022_102285


namespace NUMINAMATH_GPT_ratio_eq_thirteen_fifths_l1022_102244

theorem ratio_eq_thirteen_fifths
  (a b c : ℝ)
  (h₁ : b / a = 4)
  (h₂ : c / b = 2) :
  (a + b + c) / (a + b) = 13 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_eq_thirteen_fifths_l1022_102244


namespace NUMINAMATH_GPT_thirteenth_term_geometric_sequence_l1022_102211

theorem thirteenth_term_geometric_sequence 
  (a : ℕ → ℕ) 
  (r : ℝ)
  (h₁ : a 7 = 7) 
  (h₂ : a 10 = 21)
  (h₃ : ∀ (n : ℕ), a (n + 1) = a n * r) : 
  a 13 = 63 := 
by
  -- proof needed
  sorry

end NUMINAMATH_GPT_thirteenth_term_geometric_sequence_l1022_102211


namespace NUMINAMATH_GPT_count_valid_propositions_is_zero_l1022_102245

theorem count_valid_propositions_is_zero :
  (∀ (a b : ℝ), (a > b → a^2 > b^2) = false) ∧
  (∀ (a b : ℝ), (a^2 > b^2 → a > b) = false) ∧
  (∀ (a b : ℝ), (a > b → b / a < 1) = false) ∧
  (∀ (a b : ℝ), (a > b → 1 / a < 1 / b) = false) :=
by
  sorry

end NUMINAMATH_GPT_count_valid_propositions_is_zero_l1022_102245


namespace NUMINAMATH_GPT_add_fractions_l1022_102220

-- Define the two fractions
def frac1 := 7 / 8
def frac2 := 9 / 12

-- The problem: addition of the two fractions and expressing in simplest form
theorem add_fractions : frac1 + frac2 = (13 : ℚ) / 8 := 
by 
  sorry

end NUMINAMATH_GPT_add_fractions_l1022_102220


namespace NUMINAMATH_GPT_percentage_not_caught_l1022_102281

theorem percentage_not_caught (x : ℝ) (h1 : 22 + x = 25.88235294117647) : x = 3.88235294117647 :=
sorry

end NUMINAMATH_GPT_percentage_not_caught_l1022_102281


namespace NUMINAMATH_GPT_wheres_waldo_books_published_l1022_102246

theorem wheres_waldo_books_published (total_minutes : ℕ) (minutes_per_puzzle : ℕ) (puzzles_per_book : ℕ)
  (h1 : total_minutes = 1350) (h2 : minutes_per_puzzle = 3) (h3 : puzzles_per_book = 30) :
  total_minutes / minutes_per_puzzle / puzzles_per_book = 15 :=
by
  sorry

end NUMINAMATH_GPT_wheres_waldo_books_published_l1022_102246


namespace NUMINAMATH_GPT_sum_of_zeros_of_even_function_is_zero_l1022_102201

open Function

theorem sum_of_zeros_of_even_function_is_zero (f : ℝ → ℝ) (hf: Even f) (hx: ∃ x1 x2 x3 x4 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) :
  x1 + x2 + x3 + x4 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_zeros_of_even_function_is_zero_l1022_102201


namespace NUMINAMATH_GPT_marcus_baseball_cards_l1022_102249

/-- 
Marcus initially has 210.0 baseball cards.
Carter gives Marcus 58.0 more baseball cards.
Prove that Marcus now has 268.0 baseball cards.
-/
theorem marcus_baseball_cards (initial_cards : ℝ) (additional_cards : ℝ) 
  (h_initial : initial_cards = 210.0) (h_additional : additional_cards = 58.0) : 
  initial_cards + additional_cards = 268.0 :=
  by
    sorry

end NUMINAMATH_GPT_marcus_baseball_cards_l1022_102249


namespace NUMINAMATH_GPT_sequence_n_value_l1022_102258

theorem sequence_n_value (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) (h3 : a n = 2008) : n = 670 :=
by
 sorry

end NUMINAMATH_GPT_sequence_n_value_l1022_102258


namespace NUMINAMATH_GPT_range_of_m_inequality_system_l1022_102282

theorem range_of_m_inequality_system (m : ℝ) :
  (∀ x : ℤ, (-5 < x ∧ x ≤ m + 1) ↔ (x = -4 ∨ x = -3 ∨ x = -2)) →
  -3 ≤ m ∧ m < -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_inequality_system_l1022_102282


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1022_102284

-- For Question 1

theorem system1_solution (x y : ℝ) :
  (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ↔ (x = 5 ∧ y = 5) := 
sorry

-- For Question 2

theorem system2_solution (x y : ℝ) :
  (3 * (x + y) - 4 * (x - y) = 16) ∧ ((x + y)/2 + (x - y)/6 = 1) ↔ (x = 1/3 ∧ y = 7/3) := 
sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1022_102284
