import Mathlib

namespace NUMINAMATH_GPT_exactly_2_std_devs_less_than_mean_l50_5069

noncomputable def mean : ℝ := 14.5
noncomputable def std_dev : ℝ := 1.5
noncomputable def value : ℝ := mean - 2 * std_dev

theorem exactly_2_std_devs_less_than_mean : value = 11.5 := by
  sorry

end NUMINAMATH_GPT_exactly_2_std_devs_less_than_mean_l50_5069


namespace NUMINAMATH_GPT_total_bees_l50_5090

theorem total_bees 
    (B : ℕ) 
    (h1 : (1/5 : ℚ) * B + (1/3 : ℚ) * B + (2/5 : ℚ) * B + 1 = B) : 
    B = 15 := sorry

end NUMINAMATH_GPT_total_bees_l50_5090


namespace NUMINAMATH_GPT_find_lawn_width_l50_5025

/-- Given a rectangular lawn with a length of 80 m and roads each 10 m wide,
    one running parallel to the length and the other running parallel to the width,
    with a total travel cost of Rs. 3300 at Rs. 3 per sq m, prove that the width of the lawn is 30 m. -/
theorem find_lawn_width (w : ℕ) (h_area_road : 10 * w + 10 * 80 = 1100) : w = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_lawn_width_l50_5025


namespace NUMINAMATH_GPT_age_difference_l50_5014

noncomputable def years_older (A B : ℕ) : ℕ :=
A - B

theorem age_difference (A B : ℕ) (h1 : B = 39) (h2 : A + 10 = 2 * (B - 10)) :
  years_older A B = 9 :=
by
  rw [years_older]
  rw [h1] at h2
  sorry

end NUMINAMATH_GPT_age_difference_l50_5014


namespace NUMINAMATH_GPT_toys_per_day_l50_5017

theorem toys_per_day (total_toys_per_week : ℕ) (days_worked_per_week : ℕ)
  (production_rate_constant : Prop) (h1 : total_toys_per_week = 8000)
  (h2 : days_worked_per_week = 4)
  (h3 : production_rate_constant)
  : (total_toys_per_week / days_worked_per_week) = 2000 :=
by
  sorry

end NUMINAMATH_GPT_toys_per_day_l50_5017


namespace NUMINAMATH_GPT_exp_inequality_l50_5099

theorem exp_inequality (n : ℕ) (h : 0 < n) : 2 ≤ (1 + 1 / (n : ℝ)) ^ n ∧ (1 + 1 / (n : ℝ)) ^ n < 3 :=
sorry

end NUMINAMATH_GPT_exp_inequality_l50_5099


namespace NUMINAMATH_GPT_x_equals_neg_x_is_zero_abs_x_equals_2_is_pm_2_l50_5044

theorem x_equals_neg_x_is_zero (x : ℝ) (h : x = -x) : x = 0 := sorry

theorem abs_x_equals_2_is_pm_2 (x : ℝ) (h : |x| = 2) : x = 2 ∨ x = -2 := sorry

end NUMINAMATH_GPT_x_equals_neg_x_is_zero_abs_x_equals_2_is_pm_2_l50_5044


namespace NUMINAMATH_GPT_sequence_formula_l50_5080

theorem sequence_formula (a : ℕ → ℕ) (h₀ : a 1 = 2) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n ^ 2 - n * a n + 1) :
  ∀ n : ℕ, a n = n + 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l50_5080


namespace NUMINAMATH_GPT_no_real_solution_abs_eq_quadratic_l50_5036

theorem no_real_solution_abs_eq_quadratic (x : ℝ) : abs (2 * x - 6) ≠ x^2 - x + 2 := by
  sorry

end NUMINAMATH_GPT_no_real_solution_abs_eq_quadratic_l50_5036


namespace NUMINAMATH_GPT_product_of_two_numbers_l50_5071

theorem product_of_two_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 :=
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l50_5071


namespace NUMINAMATH_GPT_percentage_vanilla_orders_l50_5065

theorem percentage_vanilla_orders 
  (V C : ℕ) 
  (h1 : V = 2 * C) 
  (h2 : V + C = 220) 
  (h3 : C = 22) : 
  (V * 100) / 220 = 20 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_vanilla_orders_l50_5065


namespace NUMINAMATH_GPT_total_height_correct_l50_5051

def height_of_stairs : ℕ := 10

def num_flights : ℕ := 3

def height_of_all_stairs : ℕ := height_of_stairs * num_flights

def height_of_rope : ℕ := height_of_all_stairs / 2

def extra_height_of_ladder : ℕ := 10

def height_of_ladder : ℕ := height_of_rope + extra_height_of_ladder

def total_height_climbed : ℕ := height_of_all_stairs + height_of_rope + height_of_ladder

theorem total_height_correct : total_height_climbed = 70 := by
  sorry

end NUMINAMATH_GPT_total_height_correct_l50_5051


namespace NUMINAMATH_GPT_polynomial_simplification_l50_5082

theorem polynomial_simplification (p : ℝ) :
  (4 * p^4 + 2 * p^3 - 7 * p + 3) + (5 * p^3 - 8 * p^2 + 3 * p + 2) = 
  4 * p^4 + 7 * p^3 - 8 * p^2 - 4 * p + 5 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l50_5082


namespace NUMINAMATH_GPT_tent_ratio_l50_5022

-- Define the relevant variables
variables (N E S C T : ℕ)

-- State the conditions
def conditions : Prop :=
  N = 100 ∧
  E = 2 * N ∧
  S = 200 ∧
  T = 900 ∧
  N + E + S + C = T

-- State the theorem to prove the ratio
theorem tent_ratio (h : conditions N E S C T) : C = 4 * N :=
by sorry

end NUMINAMATH_GPT_tent_ratio_l50_5022


namespace NUMINAMATH_GPT_payment_difference_correct_l50_5059

noncomputable def prove_payment_difference (x : ℕ) (h₀ : x > 0) : Prop :=
  180 / x - 180 / (x + 2) = 3

theorem payment_difference_correct (x : ℕ) (h₀ : x > 0) : prove_payment_difference x h₀ :=
  by
    sorry

end NUMINAMATH_GPT_payment_difference_correct_l50_5059


namespace NUMINAMATH_GPT_math_equivalence_proof_problem_l50_5096

-- Define the initial radii in L0
def r1 := 50^2
def r2 := 53^2

-- Define the formula for constructing a new circle in subsequent layers
def next_radius (r1 r2 : ℕ) : ℕ :=
  (r1 * r2) / ((Nat.sqrt r1 + Nat.sqrt r2)^2)

-- Compute the sum of reciprocals of the square roots of the radii 
-- of all circles up to and including layer L6
def sum_of_reciprocals_of_square_roots_up_to_L6 : ℚ :=
  let initial_sum := (1 / (50 : ℚ)) + (1 / (53 : ℚ))
  (127 * initial_sum) / (50 * 53)

theorem math_equivalence_proof_problem : 
  sum_of_reciprocals_of_square_roots_up_to_L6 = 13021 / 2650 := 
sorry

end NUMINAMATH_GPT_math_equivalence_proof_problem_l50_5096


namespace NUMINAMATH_GPT_alice_catch_up_time_l50_5076

def alice_speed : ℝ := 45
def tom_speed : ℝ := 15
def initial_distance : ℝ := 4
def minutes_per_hour : ℝ := 60

theorem alice_catch_up_time :
  (initial_distance / (alice_speed - tom_speed)) * minutes_per_hour = 8 :=
by
  sorry

end NUMINAMATH_GPT_alice_catch_up_time_l50_5076


namespace NUMINAMATH_GPT_arithmetic_mean_median_l50_5028

theorem arithmetic_mean_median (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : a = 0) (h4 : (a + b + c) / 3 = 4 * b) : c / b = 11 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_median_l50_5028


namespace NUMINAMATH_GPT_minimum_value_of_f_l50_5016

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x - 15| + |x - a - 15|

theorem minimum_value_of_f {a : ℝ} (h0 : 0 < a) (h1 : a < 15) : ∃ Q, (∀ x, a ≤ x ∧ x ≤ 15 → f x a ≥ Q) ∧ Q = 15 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l50_5016


namespace NUMINAMATH_GPT_problem_1_problem_2_l50_5098

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y - 8 = 0
def line3 (x y : ℝ) : Prop := 3 * x + 2 * y + 5 = 0
def point_M : ℝ × ℝ := (1, 2)
def point_P : ℝ × ℝ := (3, 1)

def line_l1 (x y : ℝ) : Prop := x + 2 * y - 5 = 0
def line_l2 (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0

theorem problem_1 :
  (line1 point_M.1 point_M.2) ∧ (line2 point_M.1 point_M.2) → 
  (line_l1 point_P.1 point_P.2) ∧ (line_l1 point_M.1 point_M.2) :=
by 
  sorry

theorem problem_2 :
  (line1 point_M.1 point_M.2) ∧ (line2 point_M.1 point_M.2) →
  (∀ (x y : ℝ), line_l2 x y ↔ line3 x y) :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l50_5098


namespace NUMINAMATH_GPT_molly_age_l50_5026

variable (S M : ℕ)

theorem molly_age (h1 : S / M = 4 / 3) (h2 : S + 6 = 38) : M = 24 :=
by
  sorry

end NUMINAMATH_GPT_molly_age_l50_5026


namespace NUMINAMATH_GPT_tempo_insured_fraction_l50_5064

theorem tempo_insured_fraction (premium : ℝ) (rate : ℝ) (original_value : ℝ) (h1 : premium = 300) (h2 : rate = 0.03) (h3 : original_value = 14000) : 
  premium / rate / original_value = 5 / 7 :=
by 
  sorry

end NUMINAMATH_GPT_tempo_insured_fraction_l50_5064


namespace NUMINAMATH_GPT_combined_towel_weight_l50_5043

/-
Given:
1. Mary has 5 times as many towels as Frances.
2. Mary has 3 times as many towels as John.
3. The total weight of their towels is 145 pounds.
4. Mary has 60 towels.

To prove: 
The combined weight of Frances's and John's towels is 22.863 kilograms.
-/

theorem combined_towel_weight (total_weight_pounds : ℝ) (mary_towels frances_towels john_towels : ℕ) 
  (conversion_factor : ℝ) (combined_weight_kilograms : ℝ) :
  mary_towels = 60 →
  mary_towels = 5 * frances_towels →
  mary_towels = 3 * john_towels →
  total_weight_pounds = 145 →
  conversion_factor = 0.453592 →
  combined_weight_kilograms = 22.863 :=
by
  sorry

end NUMINAMATH_GPT_combined_towel_weight_l50_5043


namespace NUMINAMATH_GPT_smallest_n_l50_5056

theorem smallest_n (n : ℕ) (h1: n ≥ 100) (h2: n ≤ 999) 
  (h3: (n + 5) % 8 = 0) (h4: (n - 8) % 5 = 0) : 
  n = 123 :=
sorry

end NUMINAMATH_GPT_smallest_n_l50_5056


namespace NUMINAMATH_GPT_line_does_not_pass_first_quadrant_l50_5057

open Real

theorem line_does_not_pass_first_quadrant (a b : ℝ) (h₁ : a > 0) (h₂ : b < 0) : 
  ¬∃ x y : ℝ, (x > 0) ∧ (y > 0) ∧ (ax + y - b = 0) :=
sorry

end NUMINAMATH_GPT_line_does_not_pass_first_quadrant_l50_5057


namespace NUMINAMATH_GPT_cat_and_dog_positions_l50_5035

def cat_position_after_365_moves : Nat :=
  let cycle_length := 9
  365 % cycle_length

def dog_position_after_365_moves : Nat :=
  let cycle_length := 16
  365 % cycle_length

theorem cat_and_dog_positions :
  cat_position_after_365_moves = 5 ∧ dog_position_after_365_moves = 13 :=
by
  sorry

end NUMINAMATH_GPT_cat_and_dog_positions_l50_5035


namespace NUMINAMATH_GPT_product_of_roots_l50_5046

theorem product_of_roots (x1 x2 : ℝ) (h1 : x1 ^ 2 - 2 * x1 = 2) (h2 : x2 ^ 2 - 2 * x2 = 2) (hne : x1 ≠ x2) :
  x1 * x2 = -2 := 
sorry

end NUMINAMATH_GPT_product_of_roots_l50_5046


namespace NUMINAMATH_GPT_find_a_and_vertices_find_y_range_find_a_range_l50_5015

noncomputable def quadratic_function (x a : ℝ) : ℝ :=
  x^2 - 6 * a * x + 9

theorem find_a_and_vertices (a : ℝ) :
  quadratic_function 2 a = 7 →
  a = 1 / 2 ∧
  (3 * a, quadratic_function (3 * a) a) = (3 / 2, 27 / 4) :=
sorry

theorem find_y_range (x a : ℝ) :
  a = 1 / 2 →
  -1 ≤ x ∧ x < 3 →
  27 / 4 ≤ quadratic_function x a ∧ quadratic_function x a ≤ 13 :=
sorry

theorem find_a_range (a : ℝ) (x1 x2 : ℝ) :
  (3 * a - 2 ≤ x1 ∧ x1 ≤ 5 ∧ 3 * a - 2 ≤ x2 ∧ x2 ≤ 5) →
  (x1 ≥ 3 ∧ x2 ≥ 3 → quadratic_function x1 a - quadratic_function x2 a ≤ 9 * a^2 + 20) →
  1 / 6 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_find_a_and_vertices_find_y_range_find_a_range_l50_5015


namespace NUMINAMATH_GPT_factor_difference_of_squares_l50_5073

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l50_5073


namespace NUMINAMATH_GPT_sum_of_youngest_and_oldest_friend_l50_5023

-- Given definitions
def mean_age_5 := 12
def median_age_5 := 11
def one_friend_age := 10

-- The total sum of ages is given by mean * number of friends
def total_sum_ages : ℕ := 5 * mean_age_5

-- Third friend's age as defined by median
def third_friend_age := 11

-- Proving the sum of the youngest and oldest friend's ages
theorem sum_of_youngest_and_oldest_friend:
  (∃ youngest oldest : ℕ, youngest + oldest = 38) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_youngest_and_oldest_friend_l50_5023


namespace NUMINAMATH_GPT_sequence_length_l50_5068

theorem sequence_length :
  ∃ n : ℕ, ∀ (a_1 : ℤ) (d : ℤ) (a_n : ℤ), a_1 = -6 → d = 4 → a_n = 50 → a_n = a_1 + (n - 1) * d ∧ n = 15 :=
by
  sorry

end NUMINAMATH_GPT_sequence_length_l50_5068


namespace NUMINAMATH_GPT_number_of_pencils_selling_price_equals_loss_l50_5003

theorem number_of_pencils_selling_price_equals_loss :
  ∀ (S C L : ℝ) (N : ℕ),
  C = 1.3333333333333333 * S →
  L = C - S →
  (S / 60) * N = L →
  N = 20 :=
by
  intros S C L N hC hL hN
  sorry

end NUMINAMATH_GPT_number_of_pencils_selling_price_equals_loss_l50_5003


namespace NUMINAMATH_GPT_highest_price_per_shirt_l50_5019

theorem highest_price_per_shirt (x : ℝ) 
  (num_shirts : ℕ := 20)
  (total_money : ℝ := 180)
  (entrance_fee : ℝ := 5)
  (sales_tax : ℝ := 0.08)
  (whole_number: ∀ p : ℝ, ∃ n : ℕ, p = n) :
  (∀ (price_per_shirt : ℕ), price_per_shirt ≤ 8) :=
by
  sorry

end NUMINAMATH_GPT_highest_price_per_shirt_l50_5019


namespace NUMINAMATH_GPT_net_profit_positive_max_average_net_profit_l50_5089

def initial_investment : ℕ := 720000
def first_year_expense : ℕ := 120000
def annual_expense_increase : ℕ := 40000
def annual_sales : ℕ := 500000

def net_profit (n : ℕ) : ℕ := annual_sales - (first_year_expense + (n-1) * annual_expense_increase)
def average_net_profit (y n : ℕ) : ℕ := y / n

theorem net_profit_positive (n : ℕ) : net_profit n > 0 :=
sorry -- prove when net profit is positive

theorem max_average_net_profit (n : ℕ) : 
∀ m, average_net_profit (net_profit m) m ≤ average_net_profit (net_profit n) n :=
sorry -- prove when the average net profit is maximized

end NUMINAMATH_GPT_net_profit_positive_max_average_net_profit_l50_5089


namespace NUMINAMATH_GPT_systematic_sampling_third_group_number_l50_5010

theorem systematic_sampling_third_group_number :
  ∀ (total_members groups sample_number group_5_number group_gap : ℕ),
  total_members = 200 →
  groups = 40 →
  sample_number = total_members / groups →
  group_5_number = 22 →
  group_gap = 5 →
  (group_this_number : ℕ) = group_5_number - (5 - 3) * group_gap →
  group_this_number = 12 :=
by
  intros total_members groups sample_number group_5_number group_gap Htotal Hgroups Hsample Hgroup5 Hgap Hthis_group
  sorry

end NUMINAMATH_GPT_systematic_sampling_third_group_number_l50_5010


namespace NUMINAMATH_GPT_n_leq_84_l50_5060

theorem n_leq_84 (n : ℕ) (hn : 0 < n) (h: (1 / 2 + 1 / 3 + 1 / 7 + 1 / ↑n : ℚ).den ≤ 1): n ≤ 84 :=
sorry

end NUMINAMATH_GPT_n_leq_84_l50_5060


namespace NUMINAMATH_GPT_find_other_number_l50_5027

theorem find_other_number (lcm_ab : Nat) (gcd_ab : Nat) (a b : Nat) 
  (hlcm : Nat.lcm a b = lcm_ab) 
  (hgcd : Nat.gcd a b = gcd_ab) 
  (ha : a = 210) 
  (hlcm_ab : lcm_ab = 2310) 
  (hgcd_ab : gcd_ab = 55) 
  : b = 605 := 
by 
  sorry

end NUMINAMATH_GPT_find_other_number_l50_5027


namespace NUMINAMATH_GPT_stack_crates_height_l50_5095

theorem stack_crates_height :
  ∀ a b c : ℕ, (3 * a + 4 * b + 5 * c = 50) ∧ (a + b + c = 12) → false :=
by
  sorry

end NUMINAMATH_GPT_stack_crates_height_l50_5095


namespace NUMINAMATH_GPT_floor_e_eq_2_l50_5030

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_GPT_floor_e_eq_2_l50_5030


namespace NUMINAMATH_GPT_negation_of_all_squares_positive_l50_5042

theorem negation_of_all_squares_positive :
  ¬ (∀ x : ℝ, x * x > 0) ↔ ∃ x : ℝ, x * x ≤ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_all_squares_positive_l50_5042


namespace NUMINAMATH_GPT_find_coefficient_m_l50_5034

theorem find_coefficient_m :
  ∃ m : ℝ, (1 + 2 * x)^3 = 1 + 6 * x + m * x^2 + 8 * x^3 ∧ m = 12 := by
  sorry

end NUMINAMATH_GPT_find_coefficient_m_l50_5034


namespace NUMINAMATH_GPT_negation_of_implication_l50_5055

theorem negation_of_implication (x : ℝ) :
  ¬ (x ≠ 3 ∧ x ≠ 2 → x^2 - 5 * x + 6 ≠ 0) ↔ (x = 3 ∨ x = 2 → x^2 - 5 * x + 6 = 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_implication_l50_5055


namespace NUMINAMATH_GPT_least_number_remainder_l50_5072

open Nat

theorem least_number_remainder (n : ℕ) :
  (n ≡ 4 [MOD 5]) →
  (n ≡ 4 [MOD 6]) →
  (n ≡ 4 [MOD 9]) →
  (n ≡ 4 [MOD 12]) →
  n = 184 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_least_number_remainder_l50_5072


namespace NUMINAMATH_GPT_clay_weight_in_second_box_l50_5012

/-- Define the properties of the first and second boxes -/
structure Box where
  height : ℕ
  width : ℕ
  length : ℕ
  weight : ℕ

noncomputable def box1 : Box :=
  { height := 2, width := 3, length := 5, weight := 40 }

noncomputable def box2 : Box :=
  { height := 2 * 2, width := 3 * 3, length := 5, weight := 240 }

theorem clay_weight_in_second_box : 
  box2.weight = (box2.height * box2.width * box2.length) / 
                (box1.height * box1.width * box1.length) * box1.weight :=
by
  sorry

end NUMINAMATH_GPT_clay_weight_in_second_box_l50_5012


namespace NUMINAMATH_GPT_classroom_gpa_l50_5081

theorem classroom_gpa (n : ℕ) (x : ℝ)
  (h1 : n > 0)
  (h2 : (1/3 : ℝ) * n * 45 + (2/3 : ℝ) * n * x = n * 55) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_classroom_gpa_l50_5081


namespace NUMINAMATH_GPT_motorcyclist_initial_speed_l50_5048

theorem motorcyclist_initial_speed (x : ℝ) : 
  (120 = x * (120 / x)) ∧
  (120 = x + 6) → 
  (120 / x = 1 + 1/6 + (120 - x) / (x + 6)) →
  (x = 48) :=
by
  sorry

end NUMINAMATH_GPT_motorcyclist_initial_speed_l50_5048


namespace NUMINAMATH_GPT_screen_to_body_ratio_increases_l50_5013

theorem screen_to_body_ratio_increases
  (a b m : ℝ)
  (h1 : a > b)
  (h2 : 0 < m)
  (h3 : m < 1) :
  (b + m) / (a + m) > b / a :=
by
  sorry

end NUMINAMATH_GPT_screen_to_body_ratio_increases_l50_5013


namespace NUMINAMATH_GPT_sum_tens_ones_digit_l50_5007

theorem sum_tens_ones_digit (a : ℕ) (b : ℕ) (n : ℕ) (h : a - b = 3) :
  let d := (3^n)
  let ones_digit := d % 10
  let tens_digit := (d / 10) % 10
  ones_digit + tens_digit = 9 :=
by 
  let d := 3^17
  let ones_digit := d % 10
  let tens_digit := (d / 10) % 10
  sorry

end NUMINAMATH_GPT_sum_tens_ones_digit_l50_5007


namespace NUMINAMATH_GPT_min_value_of_quadratic_l50_5087

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 9

theorem min_value_of_quadratic : ∃ (x : ℝ), f x = 6 :=
by sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l50_5087


namespace NUMINAMATH_GPT_repeated_three_digit_divisible_101_l50_5045

theorem repeated_three_digit_divisible_101 (abc : ℕ) (h1 : 100 ≤ abc) (h2 : abc < 1000) :
  (1000000 * abc + 1000 * abc + abc) % 101 = 0 :=
by
  sorry

end NUMINAMATH_GPT_repeated_three_digit_divisible_101_l50_5045


namespace NUMINAMATH_GPT_quadratic_real_roots_l50_5024

theorem quadratic_real_roots (k : ℝ) (h1 : k ≠ 0) : (4 + 4 * k) ≥ 0 ↔ k ≥ -1 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l50_5024


namespace NUMINAMATH_GPT_find_ravish_marks_l50_5054

-- Define the data according to the conditions.
def max_marks : ℕ := 200
def passing_percentage : ℕ := 40
def failed_by : ℕ := 40

-- The main theorem we need to prove.
theorem find_ravish_marks (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) 
  (passing_marks := (max_marks * passing_percentage) / 100)
  (ravish_marks := passing_marks - failed_by) 
  : ravish_marks = 40 := by sorry

end NUMINAMATH_GPT_find_ravish_marks_l50_5054


namespace NUMINAMATH_GPT_problem_1_problem_2_l50_5037

def M : Set ℕ := {0, 1}

def A := { p : ℕ × ℕ | p.fst ∈ M ∧ p.snd ∈ M }

def B := { p : ℕ × ℕ | p.snd = 1 - p.fst }

theorem problem_1 : A = {(0,0), (0,1), (1,0), (1,1)} :=
by
  sorry

theorem problem_2 : 
  let AB := { p ∈ A | p ∈ B }
  AB = {(1,0), (0,1)} ∧
  {S : Set (ℕ × ℕ) | S ⊆ AB} = {∅, {(1,0)}, {(0,1)}, {(1,0), (0,1)}} :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l50_5037


namespace NUMINAMATH_GPT_arrange_polynomial_descending_l50_5078

variable (a b : ℤ)

def polynomial := -a + 3 * a^5 * b^3 + 5 * a^3 * b^5 - 9 + 4 * a^2 * b^2 

def rearranged_polynomial := 3 * a^5 * b^3 + 5 * a^3 * b^5 + 4 * a^2 * b^2 - a - 9

theorem arrange_polynomial_descending :
  polynomial a b = rearranged_polynomial a b :=
sorry

end NUMINAMATH_GPT_arrange_polynomial_descending_l50_5078


namespace NUMINAMATH_GPT_time_to_meet_l50_5040

variables {v_g : ℝ} -- speed of Petya and Vasya on the dirt road
variable {v_a : ℝ} -- speed of Petya on the paved road
variable {t : ℝ} -- time from start until Petya and Vasya meet
variable {x : ℝ} -- distance from the starting point to the bridge

-- Conditions
axiom Petya_speed : v_a = 3 * v_g
axiom Petya_bridge_time : x / v_a = 1
axiom Vasya_speed : v_a ≠ 0 ∧ v_g ≠ 0

-- Statement
theorem time_to_meet (h1 : v_a = 3 * v_g) (h2 : x / v_a = 1) (h3 : v_a ≠ 0 ∧ v_g ≠ 0) : t = 2 :=
by
  have h4 : x = 3 * v_g := sorry
  have h5 : (2 * x - 2 * v_g) / (2 * v_g) = 1 := sorry
  exact sorry

end NUMINAMATH_GPT_time_to_meet_l50_5040


namespace NUMINAMATH_GPT_barry_shirt_discount_l50_5009

theorem barry_shirt_discount 
  (original_price : ℤ) 
  (discount_percent : ℤ) 
  (discounted_price : ℤ) 
  (h1 : original_price = 80) 
  (h2 : discount_percent = 15)
  (h3 : discounted_price = original_price - (discount_percent * original_price / 100)) : 
  discounted_price = 68 :=
sorry

end NUMINAMATH_GPT_barry_shirt_discount_l50_5009


namespace NUMINAMATH_GPT_sum_of_areas_of_circles_l50_5006

theorem sum_of_areas_of_circles :
  (∑' n : ℕ, π * (9 / 16) ^ n) = π * (16 / 7) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_circles_l50_5006


namespace NUMINAMATH_GPT_inequality_x4_y4_l50_5083

theorem inequality_x4_y4 (x y : ℝ) : x^4 + y^4 + 8 ≥ 8 * x * y := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_x4_y4_l50_5083


namespace NUMINAMATH_GPT_range_of_a_l50_5039

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (1/2) 2 → x₂ ∈ Set.Icc (1/2) 2 → (a / x₁ + x₁ * Real.log x₁ ≥ x₂^3 - x₂^2 - 3)) →
  a ∈ Set.Ici 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l50_5039


namespace NUMINAMATH_GPT_project_completion_advance_l50_5008

variables (a : ℝ) -- efficiency of each worker (units of work per day)
variables (total_days : ℕ) (initial_workers added_workers : ℕ) (fraction_completed : ℝ)
variables (initial_days remaining_days total_initial_work total_remaining_work total_workers_efficiency : ℝ)

-- Conditions
def conditions : Prop :=
  total_days = 100 ∧
  initial_workers = 10 ∧
  initial_days = 30 ∧
  fraction_completed = 1 / 5 ∧
  added_workers = 10 ∧
  total_initial_work = initial_workers * initial_days * a * 5 ∧ 
  total_remaining_work = total_initial_work - (initial_workers * initial_days * a) ∧
  total_workers_efficiency = (initial_workers + added_workers) * a ∧
  remaining_days = total_remaining_work / total_workers_efficiency

-- Proof statement
theorem project_completion_advance (h : conditions a total_days initial_workers added_workers fraction_completed initial_days remaining_days total_initial_work total_remaining_work total_workers_efficiency) :
  total_days - (initial_days + remaining_days) = 10 :=
  sorry

end NUMINAMATH_GPT_project_completion_advance_l50_5008


namespace NUMINAMATH_GPT_A_inter_B_eq_A_A_union_B_l50_5041

-- Definitions for sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + 3 * a = (a + 3) * x}
def B : Set ℝ := {x | x^2 + 3 = 4 * x}

-- Proof problem for part (1)
theorem A_inter_B_eq_A (a : ℝ) : (A a ∩ B = A a) ↔ (a = 1 ∨ a = 3) :=
by
  sorry

-- Proof problem for part (2)
theorem A_union_B (a : ℝ) : A a ∪ B = if a = 1 then {1, 3} else if a = 3 then {1, 3} else {a, 1, 3} :=
by
  sorry

end NUMINAMATH_GPT_A_inter_B_eq_A_A_union_B_l50_5041


namespace NUMINAMATH_GPT_number_of_chickens_l50_5047

theorem number_of_chickens (c b : ℕ) (h1 : c + b = 9) (h2 : 2 * c + 4 * b = 26) : c = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_chickens_l50_5047


namespace NUMINAMATH_GPT_nth_equation_identity_l50_5005

theorem nth_equation_identity (n : ℕ) (h : n ≥ 1) : 
  (n / (n + 2 : ℚ)) * (1 - 1 / (n + 1 : ℚ)) = (n^2 / ((n + 1) * (n + 2) : ℚ)) := 
by 
  sorry

end NUMINAMATH_GPT_nth_equation_identity_l50_5005


namespace NUMINAMATH_GPT_find_question_mark_l50_5077

noncomputable def c1 : ℝ := (5568 / 87)^(1/3)
noncomputable def c2 : ℝ := (72 * 2)^(1/2)
noncomputable def sum_c1_c2 : ℝ := c1 + c2

theorem find_question_mark : sum_c1_c2 = 16 → 256 = 16^2 :=
by
  sorry

end NUMINAMATH_GPT_find_question_mark_l50_5077


namespace NUMINAMATH_GPT_range_of_m_l50_5086

noncomputable def f (x : ℝ) : ℝ :=
  if x >= -1 then x^2 + 3*x + 5 else (1/2)^x

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x > m^2 - m) ↔ -1 ≤ m ∧ m ≤ 2 := sorry

end NUMINAMATH_GPT_range_of_m_l50_5086


namespace NUMINAMATH_GPT_jack_pays_back_total_l50_5091

noncomputable def principal : ℝ := 1200
noncomputable def rate : ℝ := 0.10
noncomputable def interest : ℝ := principal * rate
noncomputable def total : ℝ := principal + interest

theorem jack_pays_back_total : total = 1320 := by
  sorry

end NUMINAMATH_GPT_jack_pays_back_total_l50_5091


namespace NUMINAMATH_GPT_find_abc_sol_l50_5084

theorem find_abc_sol (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (1 / ↑a + 1 / ↑b + 1 / ↑c = 1) →
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨
  (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_abc_sol_l50_5084


namespace NUMINAMATH_GPT_siblings_total_weekly_water_l50_5033

noncomputable def Theo_daily : ℕ := 8
noncomputable def Mason_daily : ℕ := 7
noncomputable def Roxy_daily : ℕ := 9

noncomputable def daily_to_weekly (daily : ℕ) : ℕ := daily * 7

theorem siblings_total_weekly_water :
  daily_to_weekly Theo_daily + daily_to_weekly Mason_daily + daily_to_weekly Roxy_daily = 168 := by
  sorry

end NUMINAMATH_GPT_siblings_total_weekly_water_l50_5033


namespace NUMINAMATH_GPT_triangle_perimeter_l50_5094

theorem triangle_perimeter (L R B : ℕ) (hL : L = 12) (hR : R = L + 2) (hB : B = 24) : L + R + B = 50 :=
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l50_5094


namespace NUMINAMATH_GPT_a_and_b_finish_work_in_72_days_l50_5079

noncomputable def work_rate_A_B {A B C : ℝ} 
  (h1 : B + C = 1 / 24) 
  (h2 : A + C = 1 / 36) 
  (h3 : A + B + C = 1 / 16.000000000000004) : ℝ :=
  A + B

theorem a_and_b_finish_work_in_72_days {A B C : ℝ} 
  (h1 : B + C = 1 / 24) 
  (h2 : A + C = 1 / 36) 
  (h3 : A + B + C = 1 / 16.000000000000004) : 
  work_rate_A_B h1 h2 h3 = 1 / 72 :=
sorry

end NUMINAMATH_GPT_a_and_b_finish_work_in_72_days_l50_5079


namespace NUMINAMATH_GPT_sequence_formula_l50_5063

theorem sequence_formula (a : ℕ → ℕ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, a (n + 1) - 2 * a n + 3 = 0) :
  ∀ n : ℕ, a n = 3 - 2^n :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l50_5063


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l50_5066

variable (x : ℝ)

theorem equation1_solution :
  ((2 * x - 5) / 6 - (3 * x + 1) / 2 = 1) → (x = -2) :=
by
  sorry

theorem equation2_solution :
  (3 * x - 7 * (x - 1) = 3 - 2 * (x + 3)) → (x = 5) :=
by
  sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l50_5066


namespace NUMINAMATH_GPT_mr_roper_lawn_cuts_l50_5020

theorem mr_roper_lawn_cuts (x : ℕ) (h_apr_sep : ℕ → ℕ) (h_total_cuts : 12 * 9 = 108) :
  (6 * x + 6 * 3 = 108) → x = 15 :=
by
  -- The proof is not needed as per the instructions, hence we use sorry.
  sorry

end NUMINAMATH_GPT_mr_roper_lawn_cuts_l50_5020


namespace NUMINAMATH_GPT_one_fourth_of_7point2_is_9div5_l50_5074

theorem one_fourth_of_7point2_is_9div5 : (7.2 / 4 : ℚ) = 9 / 5 := 
by sorry

end NUMINAMATH_GPT_one_fourth_of_7point2_is_9div5_l50_5074


namespace NUMINAMATH_GPT_scientific_notation_280000_l50_5085

theorem scientific_notation_280000 : 
  ∃ n: ℝ, n * 10^5 = 280000 ∧ n = 2.8 :=
by
-- our focus is on the statement outline, thus we use sorry to skip the proof part
  sorry

end NUMINAMATH_GPT_scientific_notation_280000_l50_5085


namespace NUMINAMATH_GPT_quadratic_equal_real_roots_l50_5092

theorem quadratic_equal_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 + x + m = 0 ∧ ∀ y : ℝ, y^2 + y + m = 0 → y = x) ↔ m = 1/4 :=
by sorry

end NUMINAMATH_GPT_quadratic_equal_real_roots_l50_5092


namespace NUMINAMATH_GPT_total_cost_proof_l50_5000

noncomputable def cost_of_4kg_mangos_3kg_rice_5kg_flour (M R F : ℝ) : ℝ :=
  4 * M + 3 * R + 5 * F

theorem total_cost_proof
  (M R F : ℝ)
  (h1 : 10 * M = 24 * R)
  (h2 : 6 * F = 2 * R)
  (h3 : F = 22) :
  cost_of_4kg_mangos_3kg_rice_5kg_flour M R F = 941.6 :=
  sorry

end NUMINAMATH_GPT_total_cost_proof_l50_5000


namespace NUMINAMATH_GPT_train_speed_l50_5032

theorem train_speed (v : ℕ) :
  (∀ (d : ℕ), d = 480 → ∀ (ship_speed : ℕ), ship_speed = 60 → 
  (∀ (ship_time : ℕ), ship_time = d / ship_speed →
  (∀ (train_time : ℕ), train_time = ship_time + 2 →
  v = d / train_time))) → v = 48 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l50_5032


namespace NUMINAMATH_GPT_compare_neg_two_powers_l50_5049

theorem compare_neg_two_powers : (-2)^3 = -2^3 := by sorry

end NUMINAMATH_GPT_compare_neg_two_powers_l50_5049


namespace NUMINAMATH_GPT_compare_m_n_l50_5021

theorem compare_m_n (b m n : ℝ) :
  m = -3 * (-2) + b ∧ n = -3 * (3) + b → m > n :=
by
  sorry

end NUMINAMATH_GPT_compare_m_n_l50_5021


namespace NUMINAMATH_GPT_overall_average_marks_is_57_l50_5067

-- Define the number of students and average mark per class
def students_class_A := 26
def avg_marks_class_A := 40

def students_class_B := 50
def avg_marks_class_B := 60

def students_class_C := 35
def avg_marks_class_C := 55

def students_class_D := 45
def avg_marks_class_D := 65

-- Define the total marks per class
def total_marks_class_A := students_class_A * avg_marks_class_A
def total_marks_class_B := students_class_B * avg_marks_class_B
def total_marks_class_C := students_class_C * avg_marks_class_C
def total_marks_class_D := students_class_D * avg_marks_class_D

-- Define the grand total of marks
def grand_total_marks := total_marks_class_A + total_marks_class_B + total_marks_class_C + total_marks_class_D

-- Define the total number of students
def total_students := students_class_A + students_class_B + students_class_C + students_class_D

-- Define the overall average marks
def overall_avg_marks := grand_total_marks / total_students

-- The target theorem we want to prove
theorem overall_average_marks_is_57 : overall_avg_marks = 57 := by
  sorry

end NUMINAMATH_GPT_overall_average_marks_is_57_l50_5067


namespace NUMINAMATH_GPT_trick_deck_cost_l50_5058

theorem trick_deck_cost :
  (∃ x : ℝ, 4 * x + 4 * x = 72) → ∃ x : ℝ, x = 9 := sorry

end NUMINAMATH_GPT_trick_deck_cost_l50_5058


namespace NUMINAMATH_GPT_expression_S_max_value_S_l50_5029

section
variable (x t : ℝ)
def f (x : ℝ) := -3 * x^2 + 6 * x

-- Define the integral expression for S(t)
noncomputable def S (t : ℝ) := ∫ x in t..(t + 1), f x

-- Assert the expression for S(t)
theorem expression_S (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 2) :
  S t = -3 * t^2 + 3 * t + 2 :=
by
  sorry

-- Assert the maximum value of S(t)
theorem max_value_S :
  ∀ t, (0 ≤ t ∧ t ≤ 2) → S t ≤ 5 / 4 :=
by
  sorry

end

end NUMINAMATH_GPT_expression_S_max_value_S_l50_5029


namespace NUMINAMATH_GPT_topless_cubical_box_l50_5038

def squares : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

def valid_placement (s : Char) : Bool :=
  match s with
  | 'A' => true
  | 'B' => true
  | 'C' => true
  | 'D' => false
  | 'E' => false
  | 'F' => true
  | 'G' => true
  | 'H' => false
  | _ => false

def valid_configurations : List Char := squares.filter valid_placement

theorem topless_cubical_box:
  valid_configurations.length = 5 := by
  sorry

end NUMINAMATH_GPT_topless_cubical_box_l50_5038


namespace NUMINAMATH_GPT_euclidean_division_mod_l50_5004

theorem euclidean_division_mod (h1 : 2022 % 19 = 8)
                               (h2 : 8^6 % 19 = 1)
                               (h3 : 2023 % 6 = 1)
                               (h4 : 2023^2024 % 6 = 1) 
: 2022^(2023^2024) % 19 = 8 := 
by
  sorry

end NUMINAMATH_GPT_euclidean_division_mod_l50_5004


namespace NUMINAMATH_GPT_D_72_eq_22_l50_5031

def D(n : ℕ) : ℕ :=
  if n = 72 then 22 else 0 -- the actual function logic should define D properly

theorem D_72_eq_22 : D 72 = 22 :=
  by sorry

end NUMINAMATH_GPT_D_72_eq_22_l50_5031


namespace NUMINAMATH_GPT_fourth_triangle_exists_l50_5050

theorem fourth_triangle_exists (a b c d : ℝ)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)
  (h4 : a + b > d) (h5 : a + d > b) (h6 : b + d > a)
  (h7 : a + c > d) (h8 : a + d > c) (h9 : c + d > a) :
  b + c > d ∧ b + d > c ∧ c + d > b :=
by
  -- I skip the proof with "sorry"
  sorry

end NUMINAMATH_GPT_fourth_triangle_exists_l50_5050


namespace NUMINAMATH_GPT_point_quadrant_l50_5097

theorem point_quadrant (x y : ℝ) (h : (x + y)^2 = x^2 + y^2 - 2) : 
  ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by
  sorry

end NUMINAMATH_GPT_point_quadrant_l50_5097


namespace NUMINAMATH_GPT_annual_average_growth_rate_estimated_output_value_2006_l50_5002

-- First problem: Prove the annual average growth rate from 2003 to 2005
theorem annual_average_growth_rate (x : ℝ) (h : 6.4 * (1 + x)^2 = 10) : 
  x = 1/4 :=
by
  sorry

-- Second problem: Prove the estimated output value for 2006 given the annual growth rate
theorem estimated_output_value_2006 (x : ℝ) (output_2005 : ℝ) (h_growth : x = 1/4) (h_2005 : output_2005 = 10) : 
  output_2005 * (1 + x) = 12.5 :=
by 
  sorry

end NUMINAMATH_GPT_annual_average_growth_rate_estimated_output_value_2006_l50_5002


namespace NUMINAMATH_GPT_min_value_frac_inv_l50_5018

noncomputable def min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) : ℝ :=
  1 / a + 1 / b

theorem min_value_frac_inv (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) :
  min_value a b h1 h2 h3 = 1 / 3 :=
sorry

end NUMINAMATH_GPT_min_value_frac_inv_l50_5018


namespace NUMINAMATH_GPT_measure_of_angle_C_l50_5093

variable (a b c : ℝ) (S : ℝ)

-- Conditions
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom area_equation : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)

-- The problem
theorem measure_of_angle_C (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) (h₄: S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  ∃ C : ℝ, C = Real.arctan (Real.sqrt 3) ∧ C = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_l50_5093


namespace NUMINAMATH_GPT_trigonometric_expression_value_l50_5011

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 5 := by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_value_l50_5011


namespace NUMINAMATH_GPT_number_of_balls_in_last_box_l50_5001

noncomputable def box_question (b : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2010 → b i + b (i + 1) = 14 + i) ∧
  (b 1 + b 2011 = 1023)

theorem number_of_balls_in_last_box (b : ℕ → ℕ) (h : box_question b) : b 2011 = 1014 :=
by
  sorry

end NUMINAMATH_GPT_number_of_balls_in_last_box_l50_5001


namespace NUMINAMATH_GPT_correct_divisor_l50_5052

theorem correct_divisor :
  ∀ (D : ℕ), (D = 12 * 63) → (D = x * 36) → (x = 21) := 
by 
  intros D h1 h2
  sorry

end NUMINAMATH_GPT_correct_divisor_l50_5052


namespace NUMINAMATH_GPT_collinear_dot_probability_computation_l50_5061

def collinear_dot_probability : ℚ := 12 / Nat.choose 25 5

theorem collinear_dot_probability_computation :
  collinear_dot_probability = 12 / 53130 :=
by
  -- This is where the proof steps would be if provided.
  sorry

end NUMINAMATH_GPT_collinear_dot_probability_computation_l50_5061


namespace NUMINAMATH_GPT_solution_for_a_if_fa_eq_a_l50_5053

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 2) / (x - 2)

theorem solution_for_a_if_fa_eq_a (a : ℝ) (h : f a = a) : a = -1 :=
sorry

end NUMINAMATH_GPT_solution_for_a_if_fa_eq_a_l50_5053


namespace NUMINAMATH_GPT_negation_of_proposition_l50_5075

theorem negation_of_proposition (a b : ℝ) (h : a > b → a^2 > b^2) : a ≤ b → a^2 ≤ b^2 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l50_5075


namespace NUMINAMATH_GPT_factorial_not_multiple_of_57_l50_5070

theorem factorial_not_multiple_of_57 (n : ℕ) (h : ¬ (57 ∣ n!)) : n < 19 := 
sorry

end NUMINAMATH_GPT_factorial_not_multiple_of_57_l50_5070


namespace NUMINAMATH_GPT_ellipse_eccentricity_range_of_ratio_l50_5062

-- The setup conditions
variables {a b c : ℝ}
variables (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) (h1 : a^2 - b^2 = c^2)
variables (M : ℝ) (m : ℝ)
variables (hM : M = a + c) (hm : m = a - c) (hMm : M * m = 3 / 4 * a^2)

-- Proof statement for the eccentricity of the ellipse
theorem ellipse_eccentricity : c / a = 1 / 2 := by
  sorry

-- The setup for the second part
variables {S1 S2 : ℝ}
variables (ellipse_eq : ∀ x y : ℝ, (x^2 / (4 * c^2) + y^2 / (3 * c^2) = 1) → x + y = 0)
variables (range_S : S1 / S2 > 9)

-- Proof statement for the range of the given ratio
theorem range_of_ratio : 0 < (2 * S1 * S2) / (S1^2 + S2^2) ∧ (2 * S1 * S2) / (S1^2 + S2^2) < 9 / 41 := by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_range_of_ratio_l50_5062


namespace NUMINAMATH_GPT_vector_parallel_solution_l50_5088

-- Define the vectors and the condition
def a (m : ℝ) := (2 * m + 1, 3)
def b (m : ℝ) := (2, m)

-- The proof problem statement
theorem vector_parallel_solution (m : ℝ) :
  (2 * m + 1) * m = 3 * 2 ↔ m = 3 / 2 ∨ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_vector_parallel_solution_l50_5088
