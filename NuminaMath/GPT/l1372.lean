import Mathlib

namespace NUMINAMATH_GPT_min_value_of_function_l1372_137213

theorem min_value_of_function (x : ℝ) (h : x > 5 / 4) : 
  ∃ ymin : ℝ, ymin = 7 ∧ ∀ y : ℝ, y = 4 * x + 1 / (4 * x - 5) → y ≥ ymin := 
sorry

end NUMINAMATH_GPT_min_value_of_function_l1372_137213


namespace NUMINAMATH_GPT_games_within_division_l1372_137249

variables (N M : ℕ)
  (h1 : N > 2 * M)
  (h2 : M > 4)
  (h3 : 3 * N + 4 * M = 76)

theorem games_within_division :
  3 * N = 48 :=
sorry

end NUMINAMATH_GPT_games_within_division_l1372_137249


namespace NUMINAMATH_GPT_average_typed_words_per_minute_l1372_137258

def rudy_wpm := 64
def joyce_wpm := 76
def gladys_wpm := 91
def lisa_wpm := 80
def mike_wpm := 89
def num_team_members := 5

theorem average_typed_words_per_minute : 
  (rudy_wpm + joyce_wpm + gladys_wpm + lisa_wpm + mike_wpm) / num_team_members = 80 := 
by
  sorry

end NUMINAMATH_GPT_average_typed_words_per_minute_l1372_137258


namespace NUMINAMATH_GPT_triangle_AC_range_l1372_137214

noncomputable def length_AB : ℝ := 12
noncomputable def length_CD : ℝ := 6

def is_valid_AC (AC : ℝ) : Prop :=
  AC > 6 ∧ AC < 24

theorem triangle_AC_range :
  ∃ m n : ℝ, 
    (6 < m ∧ m < 24) ∧ (6 < n ∧ n < 24) ∧
    m + n = 30 ∧
    ∀ AC : ℝ, is_valid_AC AC →
      6 < AC ∧ AC < 24 :=
by
  use 6
  use 24
  simp
  sorry

end NUMINAMATH_GPT_triangle_AC_range_l1372_137214


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1372_137265

theorem quadratic_inequality_solution_set (m t : ℝ)
  (h : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - m*x + t < 0) : 
  m - t = -1 := sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1372_137265


namespace NUMINAMATH_GPT_jamal_bought_4_half_dozens_l1372_137228

/-- Given that each crayon costs $2, the total cost is $48, and a half dozen is 6 crayons,
    prove that Jamal bought 4 half dozens of crayons. -/
theorem jamal_bought_4_half_dozens (cost_per_crayon : ℕ) (total_cost : ℕ) (half_dozen : ℕ) 
  (h1 : cost_per_crayon = 2) (h2 : total_cost = 48) (h3 : half_dozen = 6) : 
  (total_cost / cost_per_crayon) / half_dozen = 4 := 
by 
  sorry

end NUMINAMATH_GPT_jamal_bought_4_half_dozens_l1372_137228


namespace NUMINAMATH_GPT_caterpillar_prob_A_l1372_137202

-- Define the probabilities involved
def prob_move_to_A_from_1 (x y z : ℚ) : ℚ :=
  (1/3 : ℚ) * 1 + (1/3 : ℚ) * y + (1/3 : ℚ) * z

def prob_move_to_A_from_2 (x y u : ℚ) : ℚ :=
  (1/3 : ℚ) * 0 + (1/3 : ℚ) * x + (1/3 : ℚ) * u

def prob_move_to_A_from_0 (x y : ℚ) : ℚ :=
  (2/3 : ℚ) * x + (1/3 : ℚ) * y

def prob_move_to_A_from_3 (y u : ℚ) : ℚ :=
  (2/3 : ℚ) * y + (1/3 : ℚ) * u

theorem caterpillar_prob_A :
  exists (x y z u : ℚ), 
    x = prob_move_to_A_from_1 x y z ∧
    y = prob_move_to_A_from_2 x y y ∧
    z = prob_move_to_A_from_0 x y ∧
    u = prob_move_to_A_from_3 y y ∧
    u = y ∧
    x = 9/14 :=
sorry

end NUMINAMATH_GPT_caterpillar_prob_A_l1372_137202


namespace NUMINAMATH_GPT_find_number_divided_l1372_137218

theorem find_number_divided (x : ℝ) (h : x / 1.33 = 48) : x = 63.84 :=
by
  sorry

end NUMINAMATH_GPT_find_number_divided_l1372_137218


namespace NUMINAMATH_GPT_non_neg_int_solutions_l1372_137264

theorem non_neg_int_solutions :
  (∀ m n k : ℕ, 2 * m + 3 * n = k ^ 2 →
    (m = 0 ∧ n = 1 ∧ k = 2) ∨
    (m = 3 ∧ n = 0 ∧ k = 3) ∨
    (m = 4 ∧ n = 2 ∧ k = 5)) :=
by
  intro m n k h
  -- outline proof steps here
  sorry

end NUMINAMATH_GPT_non_neg_int_solutions_l1372_137264


namespace NUMINAMATH_GPT_sqrt_9_eq_pm3_l1372_137254

theorem sqrt_9_eq_pm3 : ∃ x : ℤ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by sorry

end NUMINAMATH_GPT_sqrt_9_eq_pm3_l1372_137254


namespace NUMINAMATH_GPT_jennie_rental_cost_is_306_l1372_137203

-- Definitions for the given conditions
def weekly_rate_mid_size : ℕ := 190
def daily_rate_mid_size_upto10 : ℕ := 25
def total_rental_days : ℕ := 13
def coupon_discount : ℝ := 0.10

-- Define the cost calculation
def rental_cost (days : ℕ) : ℕ :=
  let weeks := days / 7
  let extra_days := days % 7
  let cost_weeks := weeks * weekly_rate_mid_size
  let cost_extra := extra_days * daily_rate_mid_size_upto10
  cost_weeks + cost_extra

def discount (total : ℝ) (rate : ℝ) : ℝ := total * rate

def final_amount (initial_amount : ℝ) (discount_amount : ℝ) : ℝ := initial_amount - discount_amount

-- Main theorem to prove the final payment amount
theorem jennie_rental_cost_is_306 : 
  final_amount (rental_cost total_rental_days) (discount (rental_cost total_rental_days) coupon_discount) = 306 := 
by
  sorry

end NUMINAMATH_GPT_jennie_rental_cost_is_306_l1372_137203


namespace NUMINAMATH_GPT_term_of_sequence_l1372_137231

def S (n : ℕ) : ℚ := n^2 + 2/3

def a (n : ℕ) : ℚ :=
  if n = 1 then 5/3
  else 2 * n - 1

theorem term_of_sequence (n : ℕ) : a n = 
  if n = 1 then S n 
  else S n - S (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_term_of_sequence_l1372_137231


namespace NUMINAMATH_GPT_min_value_x_squared_plus_y_squared_plus_z_squared_l1372_137224

theorem min_value_x_squared_plus_y_squared_plus_z_squared (x y z : ℝ) (h : x + y + z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_x_squared_plus_y_squared_plus_z_squared_l1372_137224


namespace NUMINAMATH_GPT_min_area_rectangle_l1372_137252

theorem min_area_rectangle (l w : ℝ) 
  (hl : 3.5 ≤ l ∧ l ≤ 4.5) 
  (hw : 5.5 ≤ w ∧ w ≤ 6.5) 
  (constraint : l ≥ 2 * w) : 
  l * w = 60.5 := 
sorry

end NUMINAMATH_GPT_min_area_rectangle_l1372_137252


namespace NUMINAMATH_GPT_intersection_of_sets_l1372_137200

def set_A (x : ℝ) : Prop := |x - 1| < 3
def set_B (x : ℝ) : Prop := (x - 1) / (x - 5) < 0

theorem intersection_of_sets : ∀ x : ℝ, (set_A x ∧ set_B x) ↔ 1 < x ∧ x < 4 := 
by sorry

end NUMINAMATH_GPT_intersection_of_sets_l1372_137200


namespace NUMINAMATH_GPT_toothpicks_total_l1372_137291

-- Definitions based on the conditions
def grid_length : ℕ := 50
def grid_width : ℕ := 40

-- Mathematical statement to prove
theorem toothpicks_total : (grid_length + 1) * grid_width + (grid_width + 1) * grid_length = 4090 := by
  sorry

end NUMINAMATH_GPT_toothpicks_total_l1372_137291


namespace NUMINAMATH_GPT_veggie_patty_percentage_l1372_137223

-- Let's define the weights
def weight_total : ℕ := 150
def weight_additives : ℕ := 45

-- Let's express the proof statement as a theorem
theorem veggie_patty_percentage : (weight_total - weight_additives) * 100 / weight_total = 70 := by
  sorry

end NUMINAMATH_GPT_veggie_patty_percentage_l1372_137223


namespace NUMINAMATH_GPT_Mitzi_score_l1372_137294

-- Definitions based on the conditions
def Gretchen_score : ℕ := 120
def Beth_score : ℕ := 85
def average_score (total_score : ℕ) (num_bowlers : ℕ) : ℕ := total_score / num_bowlers

-- Theorem stating that Mitzi's bowling score is 113
theorem Mitzi_score (m : ℕ) (h : average_score (Gretchen_score + m + Beth_score) 3 = 106) :
  m = 113 :=
by sorry

end NUMINAMATH_GPT_Mitzi_score_l1372_137294


namespace NUMINAMATH_GPT_necessarily_positive_expressions_l1372_137211

theorem necessarily_positive_expressions
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 3) :
  (b + b^2 > 0) ∧ (b + 3 * b^2 > 0) :=
sorry

end NUMINAMATH_GPT_necessarily_positive_expressions_l1372_137211


namespace NUMINAMATH_GPT_train_length_l1372_137240

noncomputable def speed_kmph := 90
noncomputable def time_sec := 5
noncomputable def speed_mps := speed_kmph * 1000 / 3600

theorem train_length : (speed_mps * time_sec) = 125 := by
  -- We need to assert and prove this theorem
  sorry

end NUMINAMATH_GPT_train_length_l1372_137240


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l1372_137222

-- Definition of hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

-- The main theorem to prove
theorem asymptotes_of_hyperbola (x y : ℝ) :
  hyperbola_eq x y → (y = (1/2) * x ∨ y = -(1/2) * x) :=
by 
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l1372_137222


namespace NUMINAMATH_GPT_efficiency_ratio_l1372_137230

variable (A_eff B_eff : ℝ)

-- Condition 1: A and B together finish a piece of work in 36 days
def combined_efficiency := A_eff + B_eff = 1 / 36

-- Condition 2: B alone finishes the work in 108 days
def B_efficiency := B_eff = 1 / 108

-- Theorem: Prove that the ratio of A's efficiency to B's efficiency is 2:1
theorem efficiency_ratio (h1 : combined_efficiency A_eff B_eff) (h2 : B_efficiency B_eff) : (A_eff / B_eff) = 2 := by
  sorry

end NUMINAMATH_GPT_efficiency_ratio_l1372_137230


namespace NUMINAMATH_GPT_kerosene_cost_l1372_137272

/-- Given that:
    - A dozen eggs cost as much as a pound of rice.
    - A half-liter of kerosene costs as much as 8 eggs.
    - The cost of each pound of rice is $0.33.
    - One dollar has 100 cents.
Prove that a liter of kerosene costs 44 cents.
-/
theorem kerosene_cost :
  let egg_cost := 0.33 / 12  -- Cost per egg in dollars
  let kerosene_half_liter_cost := egg_cost * 8  -- Half-liter of kerosene cost in dollars
  let kerosene_liter_cost := kerosene_half_liter_cost * 2  -- Liter of kerosene cost in dollars
  let kerosene_liter_cost_cents := kerosene_liter_cost * 100  -- Liter of kerosene cost in cents
  kerosene_liter_cost_cents = 44 :=
by
  sorry

end NUMINAMATH_GPT_kerosene_cost_l1372_137272


namespace NUMINAMATH_GPT_symmetric_line_equation_l1372_137281

def line_1 (x y : ℝ) : Prop := 2 * x - y + 3 = 0
def line_2 (x y : ℝ) : Prop := x - y + 2 = 0
def symmetric_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

theorem symmetric_line_equation :
  ∀ x y : ℝ, line_1 x y → line_2 x y → symmetric_line x y := 
sorry

end NUMINAMATH_GPT_symmetric_line_equation_l1372_137281


namespace NUMINAMATH_GPT_pm_star_eq_6_l1372_137253

open Set

-- Definitions based on the conditions
def universal_set : Set ℕ := univ
def M : Set ℕ := {1, 2, 3, 4, 5}
def P : Set ℕ := {2, 3, 6}
def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- The theorem to prove
theorem pm_star_eq_6 : star P M = {6} :=
sorry

end NUMINAMATH_GPT_pm_star_eq_6_l1372_137253


namespace NUMINAMATH_GPT_general_term_a_general_term_b_sum_c_l1372_137226

-- Problem 1: General term formula for the sequence {a_n}
theorem general_term_a (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 2 - a n) :
  ∀ n, a n = (1 / 2) ^ (n - 1) := 
sorry

-- Problem 2: General term formula for the sequence {b_n}
theorem general_term_b (b : ℕ → ℝ) (a : ℕ → ℝ) (h_b1 : b 1 = 1)
  (h_b : ∀ n, b (n + 1) = b n + a n) (h_a : ∀ n, a n = (1 / 2) ^ (n - 1)) :
  ∀ n, b n = 3 - 2 * (1 / 2) ^ (n - 1) := 
sorry

-- Problem 3: Sum of the first n terms for the sequence {c_n}
theorem sum_c (c : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h_b : ∀ n, b n = 3 - 2 * (1 / 2) ^ (n - 1)) (h_c : ∀ n, c n = n * (3 - b n)) :
  ∀ n, T n = 8 - (8 + 4 * n) * (1 / 2) ^ n := 
sorry

end NUMINAMATH_GPT_general_term_a_general_term_b_sum_c_l1372_137226


namespace NUMINAMATH_GPT_correct_exponentiation_l1372_137204

theorem correct_exponentiation (a : ℝ) : (-2 * a^3) ^ 4 = 16 * a ^ 12 :=
by sorry

end NUMINAMATH_GPT_correct_exponentiation_l1372_137204


namespace NUMINAMATH_GPT_ships_meeting_count_l1372_137275

theorem ships_meeting_count :
  ∀ (n : ℕ) (east_sailing west_sailing : ℕ),
    n = 10 →
    east_sailing = 5 →
    west_sailing = 5 →
    east_sailing + west_sailing = n →
    (∀ (v : ℕ), v > 0) →
    25 = east_sailing * west_sailing :=
by
  intros n east_sailing west_sailing h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_ships_meeting_count_l1372_137275


namespace NUMINAMATH_GPT_gcd_of_sum_and_squares_l1372_137215

theorem gcd_of_sum_and_squares {a b : ℤ} (h : Int.gcd a b = 1) : 
  Int.gcd (a^2 + b^2) (a + b) = 1 ∨ Int.gcd (a^2 + b^2) (a + b) = 2 := 
by
  sorry

end NUMINAMATH_GPT_gcd_of_sum_and_squares_l1372_137215


namespace NUMINAMATH_GPT_smallest_sector_angle_3_l1372_137221

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_angles_is_360 (a : ℕ → ℕ) : Prop :=
  (Finset.range 15).sum a = 360

def smallest_possible_angle (a : ℕ → ℕ) (x : ℕ) : Prop :=
  ∀ i : ℕ, a i ≥ x

theorem smallest_sector_angle_3 :
  ∃ a : ℕ → ℕ,
    is_arithmetic_sequence a ∧
    sum_of_angles_is_360 a ∧
    smallest_possible_angle a 3 :=
sorry

end NUMINAMATH_GPT_smallest_sector_angle_3_l1372_137221


namespace NUMINAMATH_GPT_total_score_is_938_l1372_137287

-- Define the average score condition
def average_score (S : ℤ) : Prop := 85.25 ≤ (S : ℚ) / 11 ∧ (S : ℚ) / 11 < 85.35

-- Define the condition that each student's score is an integer
def total_score (S : ℤ) : Prop := average_score S ∧ ∃ n : ℕ, S = n

-- Lean 4 statement for the proof problem
theorem total_score_is_938 : ∃ S : ℤ, total_score S ∧ S = 938 :=
by
  sorry

end NUMINAMATH_GPT_total_score_is_938_l1372_137287


namespace NUMINAMATH_GPT_minimize_y_l1372_137282

noncomputable def y (x a b k : ℝ) : ℝ :=
  (x - a) ^ 2 + (x - b) ^ 2 + k * x

theorem minimize_y (a b k : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x a b k ≤ y x' a b k) ∧ x = (a + b - k / 2) / 2 :=
by
  have x := (a + b - k / 2) / 2
  use x
  sorry

end NUMINAMATH_GPT_minimize_y_l1372_137282


namespace NUMINAMATH_GPT_problem_statement_l1372_137298

-- Define C and D as specified in the problem conditions.
def C : ℕ := 4500
def D : ℕ := 3000

-- The final statement of the problem to prove C + D = 7500.
theorem problem_statement : C + D = 7500 := by
  -- This proof can be completed by checking arithmetic.
  sorry

end NUMINAMATH_GPT_problem_statement_l1372_137298


namespace NUMINAMATH_GPT_raisin_weight_l1372_137288

theorem raisin_weight (Wg : ℝ) (dry_grapes_fraction : ℝ) (dry_raisins_fraction : ℝ) :
  Wg = 101.99999999999999 → dry_grapes_fraction = 0.10 → dry_raisins_fraction = 0.85 → 
  Wg * dry_grapes_fraction / dry_raisins_fraction = 12 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_raisin_weight_l1372_137288


namespace NUMINAMATH_GPT_correct_systematic_sampling_method_l1372_137210

inductive SamplingMethod
| A
| B
| C
| D

def most_suitable_for_systematic_sampling (A B C D : SamplingMethod) : SamplingMethod :=
SamplingMethod.C

theorem correct_systematic_sampling_method : 
    most_suitable_for_systematic_sampling SamplingMethod.A SamplingMethod.B SamplingMethod.C SamplingMethod.D = SamplingMethod.C :=
by
  sorry

end NUMINAMATH_GPT_correct_systematic_sampling_method_l1372_137210


namespace NUMINAMATH_GPT_range_of_a_l1372_137261

open Set

variable {x a : ℝ}

def p (x a : ℝ) := x^2 + 2 * a * x - 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) := x^2 + 2 * x - 8 < 0

theorem range_of_a (h : ∀ x, p x a → q x): 0 < a ∧ a ≤ 4 / 3 := 
  sorry

end NUMINAMATH_GPT_range_of_a_l1372_137261


namespace NUMINAMATH_GPT_share_of_A_l1372_137235

-- Definitions corresponding to the conditions
variables (A B C : ℝ)
variable (total : ℝ := 578)
variable (share_ratio_B_C : ℝ := 1 / 4)
variable (share_ratio_A_B : ℝ := 2 / 3)

-- Conditions
def condition1 : B = share_ratio_B_C * C := by sorry
def condition2 : A = share_ratio_A_B * B := by sorry
def condition3 : A + B + C = total := by sorry

-- The equivalent math proof problem statement
theorem share_of_A :
  A = 68 :=
by sorry

end NUMINAMATH_GPT_share_of_A_l1372_137235


namespace NUMINAMATH_GPT_bounded_sequence_is_constant_two_l1372_137209

def is_bounded (l : ℕ → ℕ) := ∃ (M : ℕ), ∀ (n : ℕ), l n ≤ M

def satisfies_condition (a : ℕ → ℕ) : Prop :=
∀ n ≥ 3, a n = (a n.pred + a (n.pred.pred)) / (Nat.gcd (a n.pred) (a (n.pred.pred)))

theorem bounded_sequence_is_constant_two (a : ℕ → ℕ) 
  (h1 : is_bounded a) 
  (h2 : satisfies_condition a) : 
  ∀ n : ℕ, a n = 2 :=
sorry

end NUMINAMATH_GPT_bounded_sequence_is_constant_two_l1372_137209


namespace NUMINAMATH_GPT_remainder_of_n_div_1000_l1372_137201

noncomputable def setS : Set ℕ := {x | 1 ≤ x ∧ x ≤ 15}

def n : ℕ :=
  let T := {x | 4 ≤ x ∧ x ≤ 15}
  (3^12 - 2^12) / 2

theorem remainder_of_n_div_1000 : (n % 1000) = 672 := 
  by sorry

end NUMINAMATH_GPT_remainder_of_n_div_1000_l1372_137201


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1372_137263

theorem simplify_and_evaluate (x y : ℝ) (h1 : x = 1/25) (h2 : y = -25) :
  x * (x + 2 * y) - (x + 1) ^ 2 + 2 * x = -3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1372_137263


namespace NUMINAMATH_GPT_solve_opposite_numbers_product_l1372_137250

theorem solve_opposite_numbers_product :
  ∃ (x : ℤ), 3 * x - 2 * (-x) = 30 ∧ x * (-x) = -36 :=
by
  sorry

end NUMINAMATH_GPT_solve_opposite_numbers_product_l1372_137250


namespace NUMINAMATH_GPT_phil_final_quarters_l1372_137220

-- Define the conditions
def initial_quarters : ℕ := 50
def doubled_initial_quarters : ℕ := 2 * initial_quarters
def quarters_collected_each_month : ℕ := 3
def months_in_year : ℕ := 12
def quarters_collected_in_a_year : ℕ := quarters_collected_each_month * months_in_year
def quarters_collected_every_third_month : ℕ := 1
def quarters_collected_in_third_months : ℕ := months_in_year / 3 * quarters_collected_every_third_month
def total_before_losing : ℕ := doubled_initial_quarters + quarters_collected_in_a_year + quarters_collected_in_third_months
def lost_quarter_of_total : ℕ := total_before_losing / 4
def quarters_left : ℕ := total_before_losing - lost_quarter_of_total

-- Prove the final result
theorem phil_final_quarters : quarters_left = 105 := by
  sorry

end NUMINAMATH_GPT_phil_final_quarters_l1372_137220


namespace NUMINAMATH_GPT_certain_event_is_A_l1372_137241

def conditions (option_A option_B option_C option_D : Prop) : Prop :=
  option_A ∧ ¬option_B ∧ ¬option_C ∧ ¬option_D

theorem certain_event_is_A 
  (option_A option_B option_C option_D : Prop)
  (hconditions : conditions option_A option_B option_C option_D) : 
  ∀ e, (e = option_A) := 
by
  sorry

end NUMINAMATH_GPT_certain_event_is_A_l1372_137241


namespace NUMINAMATH_GPT_evaluate_cubic_diff_l1372_137284

theorem evaluate_cubic_diff (x y : ℝ) (h1 : x + y = 12) (h2 : 2 * x + y = 16) : x^3 - y^3 = -448 := 
by
    sorry

end NUMINAMATH_GPT_evaluate_cubic_diff_l1372_137284


namespace NUMINAMATH_GPT_jason_total_games_l1372_137260

theorem jason_total_games :
  let jan_games := 11
  let feb_games := 17
  let mar_games := 16
  let apr_games := 20
  let may_games := 14
  let jun_games := 14
  let jul_games := 14
  jan_games + feb_games + mar_games + apr_games + may_games + jun_games + jul_games = 106 :=
by
  sorry

end NUMINAMATH_GPT_jason_total_games_l1372_137260


namespace NUMINAMATH_GPT_base10_representation_of_n_l1372_137277

theorem base10_representation_of_n (a b c n : ℕ) (ha : a > 0)
  (h14 : n = 14^2 * a + 14 * b + c)
  (h15 : n = 15^2 * a + 15 * c + b)
  (h6 : n = 6^3 * a + 6^2 * c + 6 * a + c) : n = 925 :=
by sorry

end NUMINAMATH_GPT_base10_representation_of_n_l1372_137277


namespace NUMINAMATH_GPT_ellipse_with_given_foci_and_point_l1372_137243

noncomputable def areFociEqual (a b c₁ c₂ : ℝ) : Prop :=
  c₁ = Real.sqrt (a^2 - b^2) ∧ c₂ = Real.sqrt (a^2 - b^2)

noncomputable def isPointOnEllipse (x₀ y₀ a₂ b₂ : ℝ) : Prop :=
  (x₀^2 / a₂) + (y₀^2 / b₂) = 1

theorem ellipse_with_given_foci_and_point :
  ∃a b : ℝ, 
    areFociEqual 8 3 a b ∧
    a = Real.sqrt 5 ∧ b = Real.sqrt 5 ∧
    isPointOnEllipse 3 (-2) 15 10  :=
sorry

end NUMINAMATH_GPT_ellipse_with_given_foci_and_point_l1372_137243


namespace NUMINAMATH_GPT_find_missing_number_l1372_137270

theorem find_missing_number (x : ℝ) :
  (20 + 40 + 60) / 3 = (10 + x + 35) / 3 + 5 → x = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_number_l1372_137270


namespace NUMINAMATH_GPT_smallest_n_ineq_l1372_137273

theorem smallest_n_ineq : ∃ n : ℕ, 3 * Real.sqrt n - 2 * Real.sqrt (n - 1) < 0.03 ∧ 
  (∀ m : ℕ, (3 * Real.sqrt m - 2 * Real.sqrt (m - 1) < 0.03) → n ≤ m) ∧ n = 433715589 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_ineq_l1372_137273


namespace NUMINAMATH_GPT_fraction_ordering_l1372_137245

theorem fraction_ordering :
  (8 : ℚ) / 24 < (6 : ℚ) / 17 ∧ (6 : ℚ) / 17 < (10 : ℚ) / 27 :=
by
  sorry

end NUMINAMATH_GPT_fraction_ordering_l1372_137245


namespace NUMINAMATH_GPT_number_of_girls_l1372_137225

theorem number_of_girls (total_children boys girls : ℕ) 
    (total_children_eq : total_children = 60)
    (boys_eq : boys = 22)
    (compute_girls : girls = total_children - boys) : 
    girls = 38 :=
by
    rw [total_children_eq, boys_eq] at compute_girls
    simp at compute_girls
    exact compute_girls

end NUMINAMATH_GPT_number_of_girls_l1372_137225


namespace NUMINAMATH_GPT_evaluate_expression_l1372_137295

def ceil (x : ℚ) : ℤ := sorry -- Implement the ceiling function for rational numbers as needed

theorem evaluate_expression :
  (ceil ((23 : ℚ) / 9 - ceil ((35 : ℚ) / 23))) 
  / (ceil ((35 : ℚ) / 9 + ceil ((9 * 23 : ℚ) / 35))) = (1 / 10 : ℚ) :=
by
  intros
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1372_137295


namespace NUMINAMATH_GPT_line_intersects_ellipse_two_points_l1372_137217

theorem line_intersects_ellipse_two_points {m n : ℝ} (h1 : ¬∃ x y : ℝ, m*x + n*y = 4 ∧ x^2 + y^2 = 4)
  (h2 : m^2 + n^2 < 4) : 
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ (m * p1.1 + n * p1.2 = 4) ∧ (m * p2.1 + n * p2.2 = 4) ∧ 
  (p1.1^2 / 9 + p1.2^2 / 4 = 1) ∧ (p2.1^2 / 9 + p2.2^2 / 4 = 1) :=
sorry

end NUMINAMATH_GPT_line_intersects_ellipse_two_points_l1372_137217


namespace NUMINAMATH_GPT_beta_interval_solution_l1372_137257

/-- 
Prove that the values of β in the set {β | β = π/6 + 2*k*π, k ∈ ℤ} 
that satisfy the interval (-2*π, 2*π) are β = π/6 or β = -11*π/6.
-/
theorem beta_interval_solution :
  ∀ β : ℝ, (∃ k : ℤ, β = (π / 6) + 2 * k * π) → (-2 * π < β ∧ β < 2 * π) →
  (β = π / 6 ∨ β = -11 * π / 6) :=
by
  intros β h_exists h_interval
  sorry

end NUMINAMATH_GPT_beta_interval_solution_l1372_137257


namespace NUMINAMATH_GPT_solution_l1372_137280

/-- Definition of the number with 2023 ones. -/
def x_2023 : ℕ := (10^2023 - 1) / 9

/-- Definition of the polynomial equation. -/
def polynomial_eq (x : ℕ) : ℤ :=
  567 * x^3 + 171 * x^2 + 15 * x - (7 * x + 5 * 10^2023 + 3 * 10^(2*2023))

/-- The solution x_2023 satisfies the polynomial equation. -/
theorem solution : polynomial_eq x_2023 = 0 := sorry

end NUMINAMATH_GPT_solution_l1372_137280


namespace NUMINAMATH_GPT_hotel_charge_problem_l1372_137267

theorem hotel_charge_problem (R G P : ℝ) 
  (h1 : P = 0.5 * R) 
  (h2 : P = 0.9 * G) : 
  (R - G) / G * 100 = 80 :=
by
  sorry

end NUMINAMATH_GPT_hotel_charge_problem_l1372_137267


namespace NUMINAMATH_GPT_ab_divisible_by_6_l1372_137297

theorem ab_divisible_by_6
  (n : ℕ) (a b : ℕ)
  (h1 : 2^n = 10 * a + b)
  (h2 : n > 3)
  (h3 : b < 10) :
  (a * b) % 6 = 0 :=
sorry

end NUMINAMATH_GPT_ab_divisible_by_6_l1372_137297


namespace NUMINAMATH_GPT_area_DEFG_l1372_137271

-- Define points and the properties of the rectangle ABCD
variable (A B C D E G F : Type)
variables (area_ABCD : ℝ) (Eg_parallel_AB_CD Df_parallel_AD_BC : Prop)
variable (E_position_AD : ℝ) (G_position_CD : ℝ) (F_midpoint_BC : Prop)
variables (length_abcd width_abcd : ℝ)

-- Assumptions based on given conditions
axiom h1 : area_ABCD = 150
axiom h2 : E_position_AD = 1 / 3
axiom h3 : G_position_CD = 1 / 3
axiom h4 : Eg_parallel_AB_CD
axiom h5 : Df_parallel_AD_BC
axiom h6 : F_midpoint_BC

-- Theorem to prove the area of DEFG
theorem area_DEFG : length_abcd * width_abcd / 3 = 50 :=
    sorry

end NUMINAMATH_GPT_area_DEFG_l1372_137271


namespace NUMINAMATH_GPT_maximum_k_l1372_137296

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

theorem maximum_k (x : ℝ) (h₀ : x > 0) (k : ℤ) (a := 1) (h₁ : (x - k) * f_prime x a + x + 1 > 0) : k = 2 :=
sorry

end NUMINAMATH_GPT_maximum_k_l1372_137296


namespace NUMINAMATH_GPT_approximate_pi_value_l1372_137268

theorem approximate_pi_value (r h : ℝ) (L : ℝ) (V : ℝ) (π : ℝ) 
  (hL : L = 2 * π * r)
  (hV : V = 1 / 3 * π * r^2 * h) 
  (approxV : V = 2 / 75 * L^2 * h) :
  π = 25 / 8 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_approximate_pi_value_l1372_137268


namespace NUMINAMATH_GPT_range_of_m_l1372_137236

theorem range_of_m (m : ℝ) (x : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((m - 2023) * x₁ + m + 2023) > ((m - 2023) * x₂ + m + 2023)) → m < 2023 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1372_137236


namespace NUMINAMATH_GPT_smallest_value_for_x_9_l1372_137259

theorem smallest_value_for_x_9 :
  let x := 9
  ∃ i, i = (8 / (x + 2)) ∧ 
  (i < (8 / x) ∧ 
   i < (8 / (x - 2)) ∧ 
   i < (x / 8) ∧ 
   i < ((x + 2) / 8)) :=
by
  let x := 9
  use (8 / (x + 2))
  sorry

end NUMINAMATH_GPT_smallest_value_for_x_9_l1372_137259


namespace NUMINAMATH_GPT_ab_plus_cd_l1372_137276

variables (a b c d : ℝ)

axiom h1 : a + b + c = 1
axiom h2 : a + b + d = 6
axiom h3 : a + c + d = 15
axiom h4 : b + c + d = 10

theorem ab_plus_cd : a * b + c * d = 45.33333333333333 := 
by 
  sorry

end NUMINAMATH_GPT_ab_plus_cd_l1372_137276


namespace NUMINAMATH_GPT_construction_costs_l1372_137278

theorem construction_costs 
  (land_cost_per_sq_meter : ℕ := 50)
  (bricks_cost_per_1000 : ℕ := 100)
  (roof_tile_cost_per_tile : ℕ := 10)
  (land_area : ℕ := 2000)
  (number_of_bricks : ℕ := 10000)
  (number_of_roof_tiles : ℕ := 500) :
  50 * 2000 + (100 / 1000) * 10000 + 10 * 500 = 106000 :=
by sorry

end NUMINAMATH_GPT_construction_costs_l1372_137278


namespace NUMINAMATH_GPT_pizza_cost_is_correct_l1372_137237

noncomputable def total_pizza_cost : ℝ :=
  let triple_cheese_pizza_cost := (3 * 10) + (6 * 2 * 2.5)
  let meat_lovers_pizza_cost := (3 * 8) + (4 * 3 * 2.5)
  let veggie_delight_pizza_cost := (6 * 5) + (10 * 1 * 2.5)
  triple_cheese_pizza_cost + meat_lovers_pizza_cost + veggie_delight_pizza_cost

theorem pizza_cost_is_correct : total_pizza_cost = 169 := by
  sorry

end NUMINAMATH_GPT_pizza_cost_is_correct_l1372_137237


namespace NUMINAMATH_GPT_roundness_of_hundred_billion_l1372_137206

def roundness (n : ℕ) : ℕ :=
  let pf := n.factorization
  pf 2 + pf 5

theorem roundness_of_hundred_billion : roundness 100000000000 = 22 := by
  sorry

end NUMINAMATH_GPT_roundness_of_hundred_billion_l1372_137206


namespace NUMINAMATH_GPT_derivative_at_2_l1372_137266

theorem derivative_at_2 (f : ℝ → ℝ) (h : ∀ x, f x = x^2 * deriv f 2 + 5 * x) :
    deriv f 2 = -5/3 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_2_l1372_137266


namespace NUMINAMATH_GPT_sum_of_a_and_b_l1372_137299

theorem sum_of_a_and_b (a b : ℝ) (h : a^2 + b^2 + 2 * a - 4 * b + 5 = 0) :
  a + b = 1 :=
sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l1372_137299


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1372_137229

theorem solve_quadratic_eq (x : ℝ) :
  x^2 - 4 * x + 2 = 0 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1372_137229


namespace NUMINAMATH_GPT_simplify_expression_l1372_137293

variable (a b : ℝ)

theorem simplify_expression :
  3 * a * (3 * a^3 + 2 * a^2) - 2 * a^2 * (b^2 + 1) = 9 * a^4 + 6 * a^3 - 2 * a^2 * b^2 - 2 * a^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1372_137293


namespace NUMINAMATH_GPT_find_soma_cubes_for_shape_l1372_137227

def SomaCubes (n : ℕ) : Type := 
  if n = 1 
  then Fin 3 
  else if 2 ≤ n ∧ n ≤ 7 
       then Fin 4 
       else Fin 0

theorem find_soma_cubes_for_shape :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  SomaCubes a = Fin 3 ∧ SomaCubes b = Fin 4 ∧ SomaCubes c = Fin 4 ∧ 
  a + b + c = 11 ∧ ((a, b, c) = (1, 3, 5) ∨ (a, b, c) = (1, 3, 6)) := 
by
  sorry

end NUMINAMATH_GPT_find_soma_cubes_for_shape_l1372_137227


namespace NUMINAMATH_GPT_quadratic_sum_constants_l1372_137232

theorem quadratic_sum_constants (a b c : ℝ) 
  (h_eq : ∀ x, a * x^2 + b * x + c = 0 → x = -3 ∨ x = 5)
  (h_min : ∀ x, a * x^2 + b * x + c ≥ 36) 
  (h_at : a * 1^2 + b * 1 + c = 36) :
  a + b + c = 36 :=
sorry

end NUMINAMATH_GPT_quadratic_sum_constants_l1372_137232


namespace NUMINAMATH_GPT_minimum_additional_coins_l1372_137256

-- The conditions
def total_friends : ℕ := 15
def current_coins : ℕ := 100

-- The fact that the total coins required to give each friend a unique number of coins from 1 to 15 is 120
def total_required_coins : ℕ := (total_friends * (total_friends + 1)) / 2

-- The theorem stating the required number of additional coins
theorem minimum_additional_coins (total_friends : ℕ) (current_coins : ℕ) (total_required_coins : ℕ) : ℕ :=
  sorry

end NUMINAMATH_GPT_minimum_additional_coins_l1372_137256


namespace NUMINAMATH_GPT_five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one_l1372_137242

theorem five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one (n : ℕ) (hn : n > 0) : ¬ (4 ^ n - 1 ∣ 5 ^ n - 1) :=
sorry

end NUMINAMATH_GPT_five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one_l1372_137242


namespace NUMINAMATH_GPT_fraction_product_eq_l1372_137248

theorem fraction_product_eq :
  (1 / 3) * (3 / 5) * (5 / 7) * (7 / 9) = 1 / 9 := by
  sorry

end NUMINAMATH_GPT_fraction_product_eq_l1372_137248


namespace NUMINAMATH_GPT_find_expression_value_l1372_137207

theorem find_expression_value (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) :
  (x^3 + 3 * y^3) / 9 = 73 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_expression_value_l1372_137207


namespace NUMINAMATH_GPT_ratio_of_a_and_b_l1372_137292

theorem ratio_of_a_and_b (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : (a * Real.sin (Real.pi / 7) + b * Real.cos (Real.pi / 7)) / 
        (a * Real.cos (Real.pi / 7) - b * Real.sin (Real.pi / 7)) = 
        Real.tan (10 * Real.pi / 21)) :
  b / a = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_a_and_b_l1372_137292


namespace NUMINAMATH_GPT_chocolates_received_per_boy_l1372_137283

theorem chocolates_received_per_boy (total_chocolates : ℕ) (total_people : ℕ)
(boys : ℕ) (girls : ℕ) (chocolates_per_girl : ℕ)
(h_total_chocolates : total_chocolates = 3000)
(h_total_people : total_people = 120)
(h_boys : boys = 60)
(h_girls : girls = 60)
(h_chocolates_per_girl : chocolates_per_girl = 3) :
  (total_chocolates - (girls * chocolates_per_girl)) / boys = 47 :=
by
  sorry

end NUMINAMATH_GPT_chocolates_received_per_boy_l1372_137283


namespace NUMINAMATH_GPT_division_minutes_per_day_l1372_137269

-- Define the conditions
def total_hours : ℕ := 5
def minutes_multiplication_per_day : ℕ := 10
def days_total : ℕ := 10

-- Convert hours to minutes
def total_minutes : ℕ := total_hours * 60

-- Total minutes spent on multiplication
def total_minutes_multiplication : ℕ := minutes_multiplication_per_day * days_total

-- Total minutes spent on division
def total_minutes_division : ℕ := total_minutes - total_minutes_multiplication

-- Minutes spent on division per day
def minutes_division_per_day : ℕ := total_minutes_division / days_total

-- The theorem to prove
theorem division_minutes_per_day : minutes_division_per_day = 20 := by
  sorry

end NUMINAMATH_GPT_division_minutes_per_day_l1372_137269


namespace NUMINAMATH_GPT_geometric_sequence_division_condition_l1372_137262

variable {a : ℕ → ℝ}
variable {q : ℝ}

/-- a is a geometric sequence with common ratio q -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = a 1 * q ^ (n - 1)

/-- 3a₁, 1/2a₅, and 2a₃ forming an arithmetic sequence -/
def arithmetic_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  3 * a 1 + 2 * (a 1 * q ^ 2) = 2 * (1 / 2 * (a 1 * q ^ 4))

theorem geometric_sequence_division_condition
  (h1 : is_geometric_sequence a q)
  (h2 : arithmetic_sequence_condition a q) :
  (a 9 + a 10) / (a 7 + a 8) = 3 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_division_condition_l1372_137262


namespace NUMINAMATH_GPT_mean_rest_scores_l1372_137285

theorem mean_rest_scores (n : ℕ) (h : 15 < n) 
  (overall_mean : ℝ := 10)
  (mean_of_fifteen : ℝ := 12)
  (total_score : ℝ := n * overall_mean): 
  (180 + p * (n - 15) = total_score) →
  p = (10 * n - 180) / (n - 15) :=
sorry

end NUMINAMATH_GPT_mean_rest_scores_l1372_137285


namespace NUMINAMATH_GPT_associates_more_than_two_years_l1372_137208

-- Definitions based on the given conditions
def total_associates := 100
def second_year_associates_percent := 25
def not_first_year_associates_percent := 75

-- The theorem to prove
theorem associates_more_than_two_years :
  not_first_year_associates_percent - second_year_associates_percent = 50 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_associates_more_than_two_years_l1372_137208


namespace NUMINAMATH_GPT_geometric_sequence_a4_l1372_137238

theorem geometric_sequence_a4 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 4)
  (h3 : a 6 = 16) : 
  a 4 = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_l1372_137238


namespace NUMINAMATH_GPT_carmen_reaches_alex_in_17_5_minutes_l1372_137255

-- Define the conditions
variable (initial_distance : ℝ := 30) -- Initial distance in kilometers
variable (rate_of_closure : ℝ := 2) -- Rate at which the distance decreases in km per minute
variable (minutes_before_stop : ℝ := 10) -- Minutes before Alex stops

-- Define the speeds
variable (v_A : ℝ) -- Alex's speed in km per hour
variable (v_C : ℝ := 2 * v_A) -- Carmen's speed is twice Alex's speed
variable (total_closure_rate : ℝ := 120) -- Closure rate in km per hour (2 km per minute)

-- Main theorem to prove:
theorem carmen_reaches_alex_in_17_5_minutes : 
  ∃ (v_A v_C : ℝ), v_C = 2 * v_A ∧ v_C + v_A = total_closure_rate ∧ 
    (initial_distance - rate_of_closure * minutes_before_stop 
    - v_C * ((initial_distance - rate_of_closure * minutes_before_stop) / v_C) / 60 = 0) ∧ 
    (minutes_before_stop + ((initial_distance - rate_of_closure * minutes_before_stop) / v_C) * 60 = 17.5) :=
by
  sorry

end NUMINAMATH_GPT_carmen_reaches_alex_in_17_5_minutes_l1372_137255


namespace NUMINAMATH_GPT_floor_abs_sum_eq_501_l1372_137205

open Int

theorem floor_abs_sum_eq_501 (x : Fin 1004 → ℝ) (h : ∀ i, x i + (i : ℝ) + 1 = (Finset.univ.sum x) + 1005) : 
  Int.floor (abs (Finset.univ.sum x)) = 501 :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_floor_abs_sum_eq_501_l1372_137205


namespace NUMINAMATH_GPT_find_u_values_l1372_137234

namespace MathProof

variable (u v : ℝ)
variable (h1 : u ≠ 0) (h2 : v ≠ 0)
variable (h3 : u + 1/v = 8) (h4 : v + 1/u = 16/3)

theorem find_u_values : u = 4 + Real.sqrt 232 / 4 ∨ u = 4 - Real.sqrt 232 / 4 :=
by {
  sorry
}

end MathProof

end NUMINAMATH_GPT_find_u_values_l1372_137234


namespace NUMINAMATH_GPT_number_of_tables_l1372_137239

-- Defining the given parameters
def linen_cost : ℕ := 25
def place_setting_cost : ℕ := 10
def rose_cost : ℕ := 5
def lily_cost : ℕ := 4
def num_place_settings : ℕ := 4
def num_roses : ℕ := 10
def num_lilies : ℕ := 15
def total_decoration_cost : ℕ := 3500

-- Defining the cost per table
def cost_per_table : ℕ := linen_cost + (num_place_settings * place_setting_cost) + (num_roses * rose_cost) + (num_lilies * lily_cost)

-- Proof problem statement: Proving number of tables is 20
theorem number_of_tables : (total_decoration_cost / cost_per_table) = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_tables_l1372_137239


namespace NUMINAMATH_GPT_sin_A_eq_one_half_l1372_137251

theorem sin_A_eq_one_half (a b : ℝ) (sin_B : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sin_B = 2/3) : 
  ∃ (sin_A : ℝ), sin_A = 1/2 := 
by
  let sin_A := a * sin_B / b
  existsi sin_A
  sorry

end NUMINAMATH_GPT_sin_A_eq_one_half_l1372_137251


namespace NUMINAMATH_GPT_will_money_left_l1372_137219

theorem will_money_left (initial sweater tshirt shoes refund_percentage : ℕ) 
  (h_initial : initial = 74)
  (h_sweater : sweater = 9)
  (h_tshirt : tshirt = 11)
  (h_shoes : shoes = 30)
  (h_refund_percentage : refund_percentage = 90) : 
  initial - (sweater + tshirt + (100 - refund_percentage) * shoes / 100) = 51 := by
  sorry

end NUMINAMATH_GPT_will_money_left_l1372_137219


namespace NUMINAMATH_GPT_verify_squaring_method_l1372_137289

theorem verify_squaring_method (x : ℝ) :
  ((x + 1)^3 - (x - 1)^3 - 2) / 6 = x^2 :=
by
  sorry

end NUMINAMATH_GPT_verify_squaring_method_l1372_137289


namespace NUMINAMATH_GPT_benny_seashells_l1372_137246

-- Defining the conditions
def initial_seashells : ℕ := 66
def given_away_seashells : ℕ := 52

-- Statement of the proof problem
theorem benny_seashells : (initial_seashells - given_away_seashells) = 14 :=
by
  sorry

end NUMINAMATH_GPT_benny_seashells_l1372_137246


namespace NUMINAMATH_GPT_grandpa_max_movies_l1372_137216

-- Definition of the conditions
def movie_duration : ℕ := 90

def tuesday_total_minutes : ℕ := 4 * 60 + 30

def tuesday_movies_watched : ℕ := tuesday_total_minutes / movie_duration

def wednesday_movies_watched : ℕ := 2 * tuesday_movies_watched

def total_movies_watched : ℕ := tuesday_movies_watched + wednesday_movies_watched

theorem grandpa_max_movies : total_movies_watched = 9 := by
  sorry

end NUMINAMATH_GPT_grandpa_max_movies_l1372_137216


namespace NUMINAMATH_GPT_cost_of_each_steak_meal_l1372_137290

variable (x : ℝ)

theorem cost_of_each_steak_meal :
  (2 * x + 2 * 3.5 + 3 * 2 = 99 - 38) → x = 24 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_of_each_steak_meal_l1372_137290


namespace NUMINAMATH_GPT_max_g_on_interval_l1372_137274

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_on_interval : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_max_g_on_interval_l1372_137274


namespace NUMINAMATH_GPT_multiplication_simplification_l1372_137212

theorem multiplication_simplification :
  let y := 6742
  let z := 397778
  let approx_mult (a b : ℕ) := 60 * a - a
  z = approx_mult y 59 := sorry

end NUMINAMATH_GPT_multiplication_simplification_l1372_137212


namespace NUMINAMATH_GPT_sum_of_roots_l1372_137279

   theorem sum_of_roots : 
     let a := 2
     let b := 7
     let c := 3
     let roots := (-b / a : ℝ)
     roots = -3.5 :=
   by
     sorry
   
end NUMINAMATH_GPT_sum_of_roots_l1372_137279


namespace NUMINAMATH_GPT_c_investment_ratio_l1372_137233

-- Conditions as definitions
variables (x : ℕ) (m : ℕ) (total_profit a_share : ℕ)
variables (h_total_profit : total_profit = 19200)
variables (h_a_share : a_share = 6400)

-- Definition of total investment (investments weighted by time)
def total_investment (x m : ℕ) : ℕ :=
  (12 * x) + (6 * 2 * x) + (4 * m * x)

-- Definition of A's share in terms of total investment
def a_share_in_terms_of_total_investment (x : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (12 * x * total_profit) / total_investment

-- The theorem stating the ratio of C's investment to A's investment
theorem c_investment_ratio (x m total_profit a_share : ℕ) (h_total_profit : total_profit = 19200)
  (h_a_share : a_share = 6400) (h_a_share_eq : a_share_in_terms_of_total_investment x (total_investment x m) total_profit = a_share) :
  m = 3 :=
by sorry

end NUMINAMATH_GPT_c_investment_ratio_l1372_137233


namespace NUMINAMATH_GPT_set_operation_example_l1372_137247

def set_operation (A B : Set ℝ) := {x | (x ∈ A ∪ B) ∧ (x ∉ A ∩ B)}

def M := {x : ℝ | -2 < x ∧ x < 2}
def N := {x : ℝ | 1 < x ∧ x < 3}

theorem set_operation_example : set_operation M N = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (2 ≤ x ∧ x < 3)} :=
by {
  sorry
}

end NUMINAMATH_GPT_set_operation_example_l1372_137247


namespace NUMINAMATH_GPT_model_to_statue_ratio_l1372_137244

theorem model_to_statue_ratio 
  (statue_height : ℝ) 
  (model_height_feet : ℝ)
  (model_height_inches : ℝ)
  (conversion_factor : ℝ) :
  statue_height = 45 → model_height_feet = 3 → conversion_factor = 12 → model_height_inches = model_height_feet * conversion_factor →
  (45 / model_height_inches) = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_model_to_statue_ratio_l1372_137244


namespace NUMINAMATH_GPT_complement_of_angleA_is_54_l1372_137286

variable (A : ℝ)

-- Condition: \(\angle A = 36^\circ\)
def angleA := 36

-- Definition of complement
def complement (angle : ℝ) : ℝ := 90 - angle

-- Proof statement
theorem complement_of_angleA_is_54 (h : angleA = 36) : complement angleA = 54 :=
sorry

end NUMINAMATH_GPT_complement_of_angleA_is_54_l1372_137286
