import Mathlib

namespace NUMINAMATH_GPT_trigonometric_identity_l350_35079

theorem trigonometric_identity 
  (α m : ℝ) 
  (h : Real.tan (α / 2) = m) :
  (1 - 2 * (Real.sin (α / 2))^2) / (1 + Real.sin α) = (1 - m) / (1 + m) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l350_35079


namespace NUMINAMATH_GPT_tomato_count_after_harvest_l350_35054

theorem tomato_count_after_harvest :
  let plant_A_initial := 150
  let plant_B_initial := 200
  let plant_C_initial := 250
  -- Day 1
  let plant_A_after_day1 := plant_A_initial - (plant_A_initial * 3 / 10)
  let plant_B_after_day1 := plant_B_initial - (plant_B_initial * 1 / 4)
  let plant_C_after_day1 := plant_C_initial - (plant_C_initial * 4 / 25)
  -- Day 7
  let plant_A_after_day7 := plant_A_after_day1 - ((plant_A_initial * 3 / 10) + 5)
  let plant_B_after_day7 := plant_B_after_day1 - (plant_B_after_day1 * 1 / 5)
  let plant_C_after_day7 := plant_C_after_day1 - ((plant_C_initial * 4 / 25) * 2)
  -- Day 14
  let plant_A_after_day14 := plant_A_after_day7 - ((plant_A_after_day1 - ((plant_A_initial * 3 / 10) + 5)) * 3)
  let plant_B_after_day14 := plant_B_after_day7 - ((plant_B_after_day1 * 1 / 5) + 15)
  let plant_C_after_day14 := plant_C_after_day7 - (plant_C_after_day7 * 1 / 5)
  (plant_A_after_day14 = 0) ∧ (plant_B_after_day14 = 75) ∧ (plant_C_after_day14 = 104) :=
by
  sorry

end NUMINAMATH_GPT_tomato_count_after_harvest_l350_35054


namespace NUMINAMATH_GPT_hcf_of_two_numbers_is_18_l350_35035

theorem hcf_of_two_numbers_is_18
  (product : ℕ)
  (lcm : ℕ)
  (hcf : ℕ) :
  product = 571536 ∧ lcm = 31096 → hcf = 18 := 
by sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_is_18_l350_35035


namespace NUMINAMATH_GPT_number_of_terms_in_expansion_l350_35091

theorem number_of_terms_in_expansion :
  (∃ (a1 a2 a3 a4 a5 b1 b2 b3 b4 c1 c2 c3 : ℕ), (a1 + a2 + a3 + a4 + a5) * (b1 + b2 + b3 + b4) * (c1 + c2 + c3) = 60) :=
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_expansion_l350_35091


namespace NUMINAMATH_GPT_solution_pairs_count_l350_35024

theorem solution_pairs_count : 
  ∃ (s : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ s → 5 * p.1 + 7 * p.2 = 708) ∧ s.card = 20 :=
sorry

end NUMINAMATH_GPT_solution_pairs_count_l350_35024


namespace NUMINAMATH_GPT_range_of_a_l350_35065

theorem range_of_a (a : ℝ) : 
  (∀ x1 x2 : ℝ, (x1 + x2 = -2 * a) ∧ (x1 * x2 = 1) ∧ (x1 < 0) ∧ (x2 < 0)) ↔ (a ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l350_35065


namespace NUMINAMATH_GPT_laundry_loads_l350_35092

theorem laundry_loads (usual_price : ℝ) (sale_price : ℝ) (cost_per_load : ℝ) (total_loads_2_bottles : ℝ) :
  usual_price = 25 ∧ sale_price = 20 ∧ cost_per_load = 0.25 ∧ total_loads_2_bottles = (2 * sale_price) / cost_per_load →
  (total_loads_2_bottles / 2) = 80 :=
by
  sorry

end NUMINAMATH_GPT_laundry_loads_l350_35092


namespace NUMINAMATH_GPT_largest_k_rooks_l350_35000

noncomputable def rooks_max_k (board_size : ℕ) : ℕ := 
  if board_size = 10 then 16 else 0

theorem largest_k_rooks {k : ℕ} (h : 0 ≤ k ∧ k ≤ 100) :
  k ≤ rooks_max_k 10 := 
sorry

end NUMINAMATH_GPT_largest_k_rooks_l350_35000


namespace NUMINAMATH_GPT_tangent_parallel_l350_35036

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel (P₀ : ℝ × ℝ) :
  (∃ x : ℝ, (P₀ = (x, f x) ∧ deriv f x = 4)) 
  ↔ (P₀ = (1, 0) ∨ P₀ = (-1, -4)) :=
by 
  sorry

end NUMINAMATH_GPT_tangent_parallel_l350_35036


namespace NUMINAMATH_GPT_three_digit_numbers_divisible_by_17_l350_35086

theorem three_digit_numbers_divisible_by_17 : ∃ (n : ℕ), n = 53 ∧ ∀ k, 100 <= 17 * k ∧ 17 * k <= 999 ↔ (6 <= k ∧ k <= 58) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_divisible_by_17_l350_35086


namespace NUMINAMATH_GPT_chili_problem_l350_35090

def cans_of_chili (x y z : ℕ) : Prop := x + 2 * y + z = 6

def percentage_more_tomatoes_than_beans (x y z : ℕ) : ℕ :=
  100 * (z - 2 * y) / (2 * y)

theorem chili_problem (x y z : ℕ) (h1 : cans_of_chili x y z) (h2 : x = 1) (h3 : y = 1) : 
  percentage_more_tomatoes_than_beans x y z = 50 :=
by
  sorry

end NUMINAMATH_GPT_chili_problem_l350_35090


namespace NUMINAMATH_GPT_total_time_to_pump_540_gallons_l350_35028

-- Definitions for the conditions
def initial_rate : ℝ := 360  -- gallons per hour
def increased_rate : ℝ := 480 -- gallons per hour
def target_volume : ℝ := 540  -- total gallons
def first_interval : ℝ := 0.5 -- first 30 minutes as fraction of hour

-- Proof problem statement
theorem total_time_to_pump_540_gallons : 
  (first_interval * initial_rate) + ((target_volume - (first_interval * initial_rate)) / increased_rate) * 60 = 75 := by
  sorry

end NUMINAMATH_GPT_total_time_to_pump_540_gallons_l350_35028


namespace NUMINAMATH_GPT_asymptote_of_hyperbola_l350_35051

theorem asymptote_of_hyperbola : 
  ∀ x y : ℝ, (y^2 / 4 - x^2 = 1) → (y = 2 * x) ∨ (y = -2 * x) := 
by
  sorry

end NUMINAMATH_GPT_asymptote_of_hyperbola_l350_35051


namespace NUMINAMATH_GPT_find_a_l350_35063

theorem find_a (a : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h1 : ∀ x, f (g x) = x)
  (h2 : f x = (Real.log (x + 1) / Real.log 2) + a)
  (h3 : g 4 = 1) :
  a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_l350_35063


namespace NUMINAMATH_GPT_ducks_cows_legs_l350_35066

theorem ducks_cows_legs (D C : ℕ) (L H X : ℤ)
  (hC : C = 13)
  (hL : L = 2 * D + 4 * C)
  (hH : H = D + C)
  (hCond : L = 3 * H + X) : X = 13 := by
  sorry

end NUMINAMATH_GPT_ducks_cows_legs_l350_35066


namespace NUMINAMATH_GPT_log_expression_zero_l350_35094

theorem log_expression_zero (log : Real → Real) (exp : Real → Real) (log_mul : ∀ a b, log (a * b) = log a + log b) :
  log 2 ^ 2 + log 2 * log 50 - log 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_log_expression_zero_l350_35094


namespace NUMINAMATH_GPT_local_min_c_value_l350_35077

-- Definition of the function f(x) with its local minimum condition
def f (x c : ℝ) := x * (x - c)^2

-- Theorem stating that for the given function f(x) to have a local minimum at x = 1, the value of c must be 1
theorem local_min_c_value (c : ℝ) (h : ∀ ε > 0, f 1 ε < f c ε) : c = 1 := sorry

end NUMINAMATH_GPT_local_min_c_value_l350_35077


namespace NUMINAMATH_GPT_find_fourth_number_l350_35031

theorem find_fourth_number : 
  ∀ (x y : ℝ),
  (28 + x + 42 + y + 104) / 5 = 90 ∧ (128 + 255 + 511 + 1023 + x) / 5 = 423 →
  y = 78 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_find_fourth_number_l350_35031


namespace NUMINAMATH_GPT_solve_system_of_equations_l350_35043

def solution_set : Set (ℝ × ℝ) := {(0, 0), (-1, 1), (-2 / (3^(1/3)), -2 * (3^(1/3)))}

theorem solve_system_of_equations (x y : ℝ) :
  (x * y^2 - 2 * y + 3 * x^2 = 0 ∧ y^2 + x^2 * y + 2 * x = 0) ↔ (x, y) ∈ solution_set := sorry

end NUMINAMATH_GPT_solve_system_of_equations_l350_35043


namespace NUMINAMATH_GPT_final_speed_is_zero_l350_35078

-- Define physical constants and conversion
def initial_speed_kmh : ℝ := 189
def initial_speed_ms : ℝ := initial_speed_kmh * 0.277778
def deceleration : ℝ := -0.5
def distance : ℝ := 4000

-- The goal is to prove the final speed is 0 m/s
theorem final_speed_is_zero (v_i : ℝ) (a : ℝ) (d : ℝ) (v_f : ℝ) 
  (hv_i : v_i = initial_speed_ms) 
  (ha : a = deceleration) 
  (hd : d = distance) 
  (h : v_f^2 = v_i^2 + 2 * a * d) : 
  v_f = 0 := 
by 
  sorry 

end NUMINAMATH_GPT_final_speed_is_zero_l350_35078


namespace NUMINAMATH_GPT_calculate_n_l350_35070

theorem calculate_n (n : ℕ) : 3^n = 3 * 9^5 * 81^3 -> n = 23 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_calculate_n_l350_35070


namespace NUMINAMATH_GPT_nina_total_spent_l350_35058

open Real

def toy_price : ℝ := 10
def toy_count : ℝ := 3
def toy_discount : ℝ := 0.15

def card_price : ℝ := 5
def card_count : ℝ := 2
def card_discount : ℝ := 0.10

def shirt_price : ℝ := 6
def shirt_count : ℝ := 5
def shirt_discount : ℝ := 0.20

def sales_tax_rate : ℝ := 0.07

noncomputable def discounted_price (price : ℝ) (count : ℝ) (discount : ℝ) : ℝ :=
  count * price * (1 - discount)

noncomputable def total_cost_before_tax : ℝ := 
  discounted_price toy_price toy_count toy_discount +
  discounted_price card_price card_count card_discount +
  discounted_price shirt_price shirt_count shirt_discount

noncomputable def total_cost_after_tax : ℝ :=
  total_cost_before_tax * (1 + sales_tax_rate)

theorem nina_total_spent : total_cost_after_tax = 62.60 :=
by
  sorry

end NUMINAMATH_GPT_nina_total_spent_l350_35058


namespace NUMINAMATH_GPT_simplify_polynomial_l350_35098

variable (x : ℝ)

theorem simplify_polynomial :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 6*x^3 =
  6*x^3 - x^2 + 23*x - 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l350_35098


namespace NUMINAMATH_GPT_unique_base_for_final_digit_one_l350_35026

theorem unique_base_for_final_digit_one :
  ∃! b : ℕ, 2 ≤ b ∧ b ≤ 15 ∧ 648 % b = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_base_for_final_digit_one_l350_35026


namespace NUMINAMATH_GPT_instantaneous_velocity_at_2_l350_35069

def s (t : ℝ) : ℝ := 3 * t^2 + t

theorem instantaneous_velocity_at_2 : (deriv s 2) = 13 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_2_l350_35069


namespace NUMINAMATH_GPT_TV_cost_exact_l350_35085

theorem TV_cost_exact (savings : ℝ) (fraction_furniture : ℝ) (fraction_tv : ℝ) (original_savings : ℝ) (tv_cost : ℝ) :
  savings = 880 →
  fraction_furniture = 3 / 4 →
  fraction_tv = 1 - fraction_furniture →
  tv_cost = fraction_tv * savings →
  tv_cost = 220 :=
by
  sorry

end NUMINAMATH_GPT_TV_cost_exact_l350_35085


namespace NUMINAMATH_GPT_scientific_notation_of_1300000_l350_35012

theorem scientific_notation_of_1300000 : 1300000 = 1.3 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_1300000_l350_35012


namespace NUMINAMATH_GPT_monkey_count_l350_35003

theorem monkey_count (piles_1 piles_2 hands_1 hands_2 bananas_1_per_hand bananas_2_per_hand total_bananas_per_monkey : ℕ) 
  (h1 : piles_1 = 6) 
  (h2 : piles_2 = 4) 
  (h3 : hands_1 = 9) 
  (h4 : hands_2 = 12) 
  (h5 : bananas_1_per_hand = 14) 
  (h6 : bananas_2_per_hand = 9) 
  (h7 : total_bananas_per_monkey = 99) : 
  (piles_1 * hands_1 * bananas_1_per_hand + piles_2 * hands_2 * bananas_2_per_hand) / total_bananas_per_monkey = 12 := 
by 
  sorry

end NUMINAMATH_GPT_monkey_count_l350_35003


namespace NUMINAMATH_GPT_problem_statement_l350_35015

open Complex

theorem problem_statement (x y : ℂ) (h : (x + y) / (x - y) - (3 * (x - y)) / (x + y) = 2) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 8320 / 4095 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l350_35015


namespace NUMINAMATH_GPT_triangle_angle_zero_degrees_l350_35052

theorem triangle_angle_zero_degrees {a b c : ℝ} (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  ∃ (C : ℝ), C = 0 ∧ c = 0 :=
sorry

end NUMINAMATH_GPT_triangle_angle_zero_degrees_l350_35052


namespace NUMINAMATH_GPT_find_vector_c_l350_35095

def angle_equal_coordinates (c : ℝ × ℝ) : Prop :=
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (1, -Real.sqrt 3)
  let cos_angle_ab (u v : ℝ × ℝ) : ℝ :=
    (u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2))
  cos_angle_ab c a = cos_angle_ab c b

theorem find_vector_c :
  angle_equal_coordinates (Real.sqrt 3, -1) :=
sorry

end NUMINAMATH_GPT_find_vector_c_l350_35095


namespace NUMINAMATH_GPT_proportion_of_fathers_with_full_time_jobs_l350_35060

theorem proportion_of_fathers_with_full_time_jobs
  (P : ℕ) -- Total number of parents surveyed
  (mothers_proportion : ℝ := 0.4) -- Proportion of mothers in the survey
  (mothers_ftj_proportion : ℝ := 0.9) -- Proportion of mothers with full-time jobs
  (parents_no_ftj_proportion : ℝ := 0.19) -- Proportion of parents without full-time jobs
  (hfathers : ℝ := 0.6) -- Proportion of fathers in the survey
  (hfathers_ftj_proportion : ℝ) -- Proportion of fathers with full-time jobs
  : hfathers_ftj_proportion = 0.75 := 
by 
  sorry

end NUMINAMATH_GPT_proportion_of_fathers_with_full_time_jobs_l350_35060


namespace NUMINAMATH_GPT_simplify_fraction_l350_35068

theorem simplify_fraction :
  (1 / (1 / (Real.sqrt 3 + 1) + 3 / (Real.sqrt 5 - 2))) = 2 / (Real.sqrt 3 + 6 * Real.sqrt 5 + 11) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l350_35068


namespace NUMINAMATH_GPT_combined_work_time_l350_35018

theorem combined_work_time (W : ℝ) (A B C : ℝ) (ha : A = W / 12) (hb : B = W / 18) (hc : C = W / 9) : 
  1 / (A + B + C) = 4 := 
by sorry

end NUMINAMATH_GPT_combined_work_time_l350_35018


namespace NUMINAMATH_GPT_shoe_length_increase_l350_35013

noncomputable def shoeSizeLength (l : ℕ → ℝ) (size : ℕ) : ℝ :=
  if size = 15 then 9.25
  else if size = 17 then 1.3 * l 8
  else l size

theorem shoe_length_increase :
  (forall l : ℕ → ℝ,
    (shoeSizeLength l 15 = 9.25) ∧
    (shoeSizeLength l 17 = 1.3 * (shoeSizeLength l 8)) ∧
    (forall n, shoeSizeLength l (n + 1) = shoeSizeLength l n + 0.25)
  ) :=
  sorry

end NUMINAMATH_GPT_shoe_length_increase_l350_35013


namespace NUMINAMATH_GPT_distinct_nonzero_reals_equation_l350_35025

theorem distinct_nonzero_reals_equation {a b c d : ℝ} 
  (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0) 
  (h₄ : a ≠ b) (h₅ : b ≠ c) (h₆ : c ≠ d) (h₇ : d ≠ a) (h₈ : a ≠ c) (h₉ : b ≠ d)
  (h₁₀ : a * c = b * d) 
  (h₁₁ : a / b + b / c + c / d + d / a = 4) :
  (a / c + c / a + b / d + d / b = 4) :=
by
  sorry

end NUMINAMATH_GPT_distinct_nonzero_reals_equation_l350_35025


namespace NUMINAMATH_GPT_sufficient_condition_range_a_l350_35059

theorem sufficient_condition_range_a (a : ℝ) :
  (∀ x, (2 * a ≤ x ∧ x ≤ a^2 + 1) → (x^2 - 3 * (a + 1) * x + 6 * a + 2 ≤ 0)) ↔
  (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) := by
  sorry

end NUMINAMATH_GPT_sufficient_condition_range_a_l350_35059


namespace NUMINAMATH_GPT_number_is_square_l350_35081

theorem number_is_square (x y : ℕ) : (∃ n : ℕ, (1100 * x + 11 * y = n^2)) ↔ (x = 7 ∧ y = 4) :=
by
  sorry

end NUMINAMATH_GPT_number_is_square_l350_35081


namespace NUMINAMATH_GPT_at_least_two_pass_written_test_expectation_number_of_admission_advantage_l350_35082

noncomputable def probability_of_passing_written_test_A : ℝ := 0.4
noncomputable def probability_of_passing_written_test_B : ℝ := 0.8
noncomputable def probability_of_passing_written_test_C : ℝ := 0.5

noncomputable def probability_of_passing_interview_A : ℝ := 0.8
noncomputable def probability_of_passing_interview_B : ℝ := 0.4
noncomputable def probability_of_passing_interview_C : ℝ := 0.64

theorem at_least_two_pass_written_test :
  (probability_of_passing_written_test_A * probability_of_passing_written_test_B * (1 - probability_of_passing_written_test_C) +
  probability_of_passing_written_test_A * (1 - probability_of_passing_written_test_B) * probability_of_passing_written_test_C +
  (1 - probability_of_passing_written_test_A) * probability_of_passing_written_test_B * probability_of_passing_written_test_C +
  probability_of_passing_written_test_A * probability_of_passing_written_test_B * probability_of_passing_written_test_C = 0.6) :=
sorry

theorem expectation_number_of_admission_advantage :
  (3 * (probability_of_passing_written_test_A * probability_of_passing_interview_A) +
  3 * (probability_of_passing_written_test_B * probability_of_passing_interview_B) +
  3 * (probability_of_passing_written_test_C * probability_of_passing_interview_C) = 0.96) :=
sorry

end NUMINAMATH_GPT_at_least_two_pass_written_test_expectation_number_of_admission_advantage_l350_35082


namespace NUMINAMATH_GPT_max_value_of_ab_l350_35056

theorem max_value_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 5 * a + 3 * b < 90) :
  ab * (90 - 5 * a - 3 * b) ≤ 1800 :=
sorry

end NUMINAMATH_GPT_max_value_of_ab_l350_35056


namespace NUMINAMATH_GPT_geometric_sequence_value_l350_35045

variable {a_n : ℕ → ℝ}

-- Condition: {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given: a_1 a_2 a_3 = -8
variable (a1 a2 a3 : ℝ) (h_seq : is_geometric_sequence a_n)
variable (h_cond : a1 * a2 * a3 = -8)

-- Prove: a2 = -2
theorem geometric_sequence_value : a2 = -2 :=
by
  -- Proof will be provided later
  sorry

end NUMINAMATH_GPT_geometric_sequence_value_l350_35045


namespace NUMINAMATH_GPT_isosceles_triangle_length_l350_35006

theorem isosceles_triangle_length (BC : ℕ) (area : ℕ) (h : ℕ)
  (isosceles : AB = AC)
  (BC_val : BC = 16)
  (area_val : area = 120)
  (height_val : h = (2 * area) / BC)
  (AB_square : ∀ BD AD : ℕ, BD = BC / 2 → AD = h → AB^2 = AD^2 + BD^2)
  : AB = 17 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_length_l350_35006


namespace NUMINAMATH_GPT_initial_deposit_l350_35016

theorem initial_deposit (A r : ℝ) (n t : ℕ) (hA : A = 169.40) 
  (hr : r = 0.20) (hn : n = 2) (ht : t = 1) :
  ∃ P : ℝ, P = 140 ∧ A = P * (1 + r / n)^(n * t) :=
by
  sorry

end NUMINAMATH_GPT_initial_deposit_l350_35016


namespace NUMINAMATH_GPT_value_of_a_is_minus_one_l350_35023

-- Define the imaginary unit i
def imaginary_unit_i : Complex := Complex.I

-- Define the complex number condition
def complex_number_condition (a : ℝ) : Prop :=
  let z := (a + imaginary_unit_i) / (1 + imaginary_unit_i)
  (Complex.re z) = 0 ∧ (Complex.im z) ≠ 0

-- Prove that the value of the real number a is -1 given the condition
theorem value_of_a_is_minus_one (a : ℝ) (h : complex_number_condition a) : a = -1 :=
sorry

end NUMINAMATH_GPT_value_of_a_is_minus_one_l350_35023


namespace NUMINAMATH_GPT_tourists_escape_l350_35099

theorem tourists_escape (T : ℕ) (hT : T = 10) (hats : Fin T → Bool) (could_see : ∀ (i : Fin T), Fin (i) → Bool) :
  ∃ strategy : (Fin T → Bool), (∀ (i : Fin T), (strategy i = hats i) ∨ (strategy i ≠ hats i)) →
  (∀ (i : Fin T), (∀ (j : Fin T), i < j → strategy i = hats i) → ∃ count : ℕ, count ≥ 9 ∧ ∀ (i : Fin T), count ≥ i → strategy i = hats i) := sorry

end NUMINAMATH_GPT_tourists_escape_l350_35099


namespace NUMINAMATH_GPT_petya_mistake_l350_35021

theorem petya_mistake (x : ℝ) (h : x - x / 10 = 19.71) : x = 21.9 := 
  sorry

end NUMINAMATH_GPT_petya_mistake_l350_35021


namespace NUMINAMATH_GPT_taxi_speed_is_60_l350_35008

theorem taxi_speed_is_60 (v_b v_t : ℝ) (h1 : v_b = v_t - 30) (h2 : 3 * v_t = 6 * v_b) : v_t = 60 := 
by 
  sorry

end NUMINAMATH_GPT_taxi_speed_is_60_l350_35008


namespace NUMINAMATH_GPT_kaleb_games_per_box_l350_35001

theorem kaleb_games_per_box (initial_games sold_games boxes remaining_games games_per_box : ℕ)
  (h1 : initial_games = 76)
  (h2 : sold_games = 46)
  (h3 : boxes = 6)
  (h4 : remaining_games = initial_games - sold_games)
  (h5 : games_per_box = remaining_games / boxes) :
  games_per_box = 5 :=
sorry

end NUMINAMATH_GPT_kaleb_games_per_box_l350_35001


namespace NUMINAMATH_GPT_cube_volume_from_surface_area_l350_35061

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_from_surface_area_l350_35061


namespace NUMINAMATH_GPT_club_members_l350_35002

theorem club_members (M W : ℕ) (h1 : M + W = 30) (h2 : M + 1/3 * (W : ℝ) = 18) : M = 12 :=
by
  -- proof step
  sorry

end NUMINAMATH_GPT_club_members_l350_35002


namespace NUMINAMATH_GPT_number_of_bookshelves_l350_35074

theorem number_of_bookshelves (books_in_each: ℕ) (total_books: ℕ) (h_books_in_each: books_in_each = 56) (h_total_books: total_books = 504) : total_books / books_in_each = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_bookshelves_l350_35074


namespace NUMINAMATH_GPT_total_candies_in_third_set_l350_35071

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_total_candies_in_third_set_l350_35071


namespace NUMINAMATH_GPT_probability_of_reaching_3_1_without_2_0_in_8_steps_l350_35062

theorem probability_of_reaching_3_1_without_2_0_in_8_steps :
  let n_total := 1680
  let invalid := 30
  let total := n_total - invalid
  let q := total / 4^8
  let gcd := Nat.gcd total 65536
  let m := total / gcd
  let n := 65536 / gcd
  (m + n = 11197) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_reaching_3_1_without_2_0_in_8_steps_l350_35062


namespace NUMINAMATH_GPT_profit_percentage_calc_l350_35073

noncomputable def sale_price_incl_tax : ℝ := 616
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def cost_price : ℝ := 531.03
noncomputable def expected_profit_percentage : ℝ := 5.45

theorem profit_percentage_calc :
  let sale_price_before_tax := sale_price_incl_tax / (1 + sales_tax_rate)
  let profit := sale_price_before_tax - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = expected_profit_percentage :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_calc_l350_35073


namespace NUMINAMATH_GPT_position_after_2010_transformations_l350_35050

-- Define the initial position of the square
def init_position := "ABCD"

-- Define the transformation function
def transform (position : String) (steps : Nat) : String :=
  match steps % 8 with
  | 0 => "ABCD"
  | 1 => "CABD"
  | 2 => "DACB"
  | 3 => "BCAD"
  | 4 => "ADCB"
  | 5 => "CBDA"
  | 6 => "BADC"
  | 7 => "CDAB"
  | _ => "ABCD"  -- Default case, should never happen

-- The theorem to prove the correct position after 2010 transformations
theorem position_after_2010_transformations : transform init_position 2010 = "CABD" := 
by
  sorry

end NUMINAMATH_GPT_position_after_2010_transformations_l350_35050


namespace NUMINAMATH_GPT_smallest_multiple_of_9_and_6_is_18_l350_35038

theorem smallest_multiple_of_9_and_6_is_18 :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_multiple_of_9_and_6_is_18_l350_35038


namespace NUMINAMATH_GPT_line_BC_l350_35093

noncomputable def Point := (ℝ × ℝ)
def A : Point := (-1, -4)
def l₁ := { p : Point | p.2 + 1 = 0 }
def l₂ := { p : Point | p.1 + p.2 + 1 = 0 }
def A' : Point := (-1, 2)
def A'' : Point := (3, 0)

theorem line_BC :
  ∃ (c₁ c₂ c₃ : ℝ), c₁ ≠ 0 ∨ c₂ ≠ 0 ∧
  ∀ (p : Point), (c₁ * p.1 + c₂ * p.2 + c₃ = 0) ↔ p ∈ { x | x = A ∨ x = A'' } :=
by sorry

end NUMINAMATH_GPT_line_BC_l350_35093


namespace NUMINAMATH_GPT_polynomial_solution_l350_35022

noncomputable def f (n : ℕ) (X Y : ℝ) : ℝ :=
  (X - 2 * Y) * (X + Y) ^ (n - 1)

theorem polynomial_solution (n : ℕ) (f : ℝ → ℝ → ℝ)
  (h1 : ∀ (t x y : ℝ), f (t * x) (t * y) = t^n * f x y)
  (h2 : ∀ (a b c : ℝ), f (a + b) c + f (b + c) a + f (c + a) b = 0)
  (h3 : f 1 0 = 1) :
  ∀ (X Y : ℝ), f X Y = (X - 2 * Y) * (X + Y) ^ (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l350_35022


namespace NUMINAMATH_GPT_average_weight_b_c_l350_35004

theorem average_weight_b_c (A B C : ℝ) (h1 : A + B + C = 126) (h2 : A + B = 80) (h3 : B = 40) : 
  (B + C) / 2 = 43 := 
by 
  -- Proof would go here, but is left as sorry as per instructions
  sorry

end NUMINAMATH_GPT_average_weight_b_c_l350_35004


namespace NUMINAMATH_GPT_squirrel_journey_time_l350_35087

theorem squirrel_journey_time : 
  (let distance := 2
  let speed_to_tree := 3
  let speed_return := 2
  let time_to_tree := distance / speed_to_tree
  let time_return := distance / speed_return
  let total_time := (time_to_tree + time_return) * 60
  total_time = 100) :=
by
  sorry

end NUMINAMATH_GPT_squirrel_journey_time_l350_35087


namespace NUMINAMATH_GPT_triangle_inequality_not_true_l350_35005

theorem triangle_inequality_not_true (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : ¬ (b + c > 2 * a) :=
by {
  -- assume (b + c > 2 * a)
  -- we need to reach a contradiction
  sorry
}

end NUMINAMATH_GPT_triangle_inequality_not_true_l350_35005


namespace NUMINAMATH_GPT_fermat_little_theorem_variant_l350_35046

theorem fermat_little_theorem_variant (p : ℕ) (m : ℤ) [hp : Fact (Nat.Prime p)] : 
  (m ^ p - m) % p = 0 :=
sorry

end NUMINAMATH_GPT_fermat_little_theorem_variant_l350_35046


namespace NUMINAMATH_GPT_intervals_of_monotonicity_range_of_a_l350_35055

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * log x

theorem intervals_of_monotonicity (h : ∀ x, 0 < x → x ≠ e → f (-2) x = -2 * x + x * log x) :
  ((∀ x, 0 < x ∧ x < exp 1 → deriv (f (-2)) x < 0) ∧ (∀ x, x > exp 1 → deriv (f (-2)) x > 0)) :=
sorry

theorem range_of_a (h : ∀ x, e ≤ x → deriv (f a) x ≥ 0) : a ≥ -2 :=
sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_range_of_a_l350_35055


namespace NUMINAMATH_GPT_circles_tangent_radii_product_eq_l350_35017

/-- Given two circles that pass through a fixed point \(M(x_1, y_1)\)
    and are tangent to both the x-axis and y-axis, with radii \(r_1\) and \(r_2\),
    prove that \(r_1 r_2 = x_1^2 + y_1^2\). -/
theorem circles_tangent_radii_product_eq (x1 y1 r1 r2 : ℝ)
  (h1 : (∃ (a : ℝ), ∃ (circle1 : ℝ → ℝ → ℝ), ∀ x y, circle1 x y = (x - a)^2 + (y - a)^2 - r1^2)
    ∧ (∃ (b : ℝ), ∃ (circle2 : ℝ → ℝ → ℝ), ∀ x y, circle2 x y = (x - b)^2 + (y - b)^2 - r2^2))
  (hm1 : (x1, y1) ∈ { p : ℝ × ℝ | (p.fst - r1)^2 + (p.snd - r1)^2 = r1^2 })
  (hm2 : (x1, y1) ∈ { p : ℝ × ℝ | (p.fst - r2)^2 + (p.snd - r2)^2 = r2^2 }) :
  r1 * r2 = x1^2 + y1^2 := sorry

end NUMINAMATH_GPT_circles_tangent_radii_product_eq_l350_35017


namespace NUMINAMATH_GPT_bulb_probability_gt4000_l350_35084

-- Definitions given in conditions
def P_X : ℝ := 0.60
def P_Y : ℝ := 0.40
def P_gt4000_X : ℝ := 0.59
def P_gt4000_Y : ℝ := 0.65

-- The proof statement
theorem bulb_probability_gt4000 : 
  (P_X * P_gt4000_X + P_Y * P_gt4000_Y) = 0.614 :=
  by
  sorry

end NUMINAMATH_GPT_bulb_probability_gt4000_l350_35084


namespace NUMINAMATH_GPT_no_odd_multiples_between_1500_and_3000_l350_35020

theorem no_odd_multiples_between_1500_and_3000 :
  ∀ n : ℤ, 1500 ≤ n → n ≤ 3000 → (18 ∣ n) → (24 ∣ n) → (36 ∣ n) → ¬(n % 2 = 1) :=
by
  -- The proof steps would go here, but we skip them according to the instructions.
  sorry

end NUMINAMATH_GPT_no_odd_multiples_between_1500_and_3000_l350_35020


namespace NUMINAMATH_GPT_factor_difference_of_squares_example_l350_35089

theorem factor_difference_of_squares_example :
    (m : ℝ) → (m ^ 2 - 4 = (m + 2) * (m - 2)) :=
by
    intro m
    sorry

end NUMINAMATH_GPT_factor_difference_of_squares_example_l350_35089


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l350_35014

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - 3 * x + 2 > 0)) ↔ (∃ x : ℝ, x^2 - 3 * x + 2 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l350_35014


namespace NUMINAMATH_GPT_savings_by_paying_cash_l350_35072

theorem savings_by_paying_cash
  (cash_price : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) (number_of_months : ℕ)
  (h1 : cash_price = 400) (h2 : down_payment = 120) (h3 : monthly_payment = 30) (h4 : number_of_months = 12) :
  cash_price + (monthly_payment * number_of_months - down_payment) - cash_price = 80 :=
by
  sorry

end NUMINAMATH_GPT_savings_by_paying_cash_l350_35072


namespace NUMINAMATH_GPT_calculate_altitude_l350_35080

-- Define the conditions
def Speed_up : ℕ := 18
def Speed_down : ℕ := 24
def Avg_speed : ℝ := 20.571428571428573

-- Define what we want to prove
theorem calculate_altitude : 
  2 * Speed_up * Speed_down / (Speed_up + Speed_down) = Avg_speed →
  (864 : ℝ) / 2 = 432 :=
by
  sorry

end NUMINAMATH_GPT_calculate_altitude_l350_35080


namespace NUMINAMATH_GPT_population_size_in_15th_year_l350_35030

theorem population_size_in_15th_year
  (a : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, y x = a * Real.logb 2 (x + 1))
  (h2 : y 1 = 100) :
  y 15 = 400 :=
by
  sorry

end NUMINAMATH_GPT_population_size_in_15th_year_l350_35030


namespace NUMINAMATH_GPT_fraction_of_cookies_l350_35009

-- Given conditions
variables 
  (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (H1 : Mike_cookies = 3 * Millie_cookies)
  (H2 : Millie_cookies = 4)
  (H3 : Frank_cookies = 3)

-- Proof statement
theorem fraction_of_cookies (Millie_cookies Mike_cookies Frank_cookies : ℕ)
  (H1 : Mike_cookies = 3 * Millie_cookies)
  (H2 : Millie_cookies = 4)
  (H3 : Frank_cookies = 3) : 
  (Frank_cookies / Mike_cookies : ℚ) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_cookies_l350_35009


namespace NUMINAMATH_GPT_find_angle_measure_l350_35067

theorem find_angle_measure (x : ℝ) (hx : 90 - x + 40 = (1 / 2) * (180 - x)) : x = 80 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_measure_l350_35067


namespace NUMINAMATH_GPT_james_out_of_pocket_cost_l350_35019

theorem james_out_of_pocket_cost (total_cost : ℝ) (coverage : ℝ) (out_of_pocket_cost : ℝ)
  (h1 : total_cost = 300) (h2 : coverage = 0.8) :
  out_of_pocket_cost = 60 :=
by
  sorry

end NUMINAMATH_GPT_james_out_of_pocket_cost_l350_35019


namespace NUMINAMATH_GPT_original_price_of_cycle_l350_35097

theorem original_price_of_cycle (SP : ℝ) (P : ℝ) (loss_percent : ℝ) 
  (h_loss : loss_percent = 18) 
  (h_SP : SP = 1148) 
  (h_eq : SP = (1 - loss_percent / 100) * P) : 
  P = 1400 := 
by 
  sorry

end NUMINAMATH_GPT_original_price_of_cycle_l350_35097


namespace NUMINAMATH_GPT_seq_positive_integers_seq_not_divisible_by_2109_l350_35048

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ a 2 = 6 ∧ ∀ n : ℕ, a (n + 2) = (a (n + 1) ^ 2 + 9) / a n

theorem seq_positive_integers (a : ℕ → ℤ) (h : seq a) : ∀ n : ℕ, 0 < a (n + 1) :=
sorry

theorem seq_not_divisible_by_2109 (a : ℕ → ℤ) (h : seq a) : ¬ ∃ m : ℕ, 2109 ∣ a (m + 1) :=
sorry

end NUMINAMATH_GPT_seq_positive_integers_seq_not_divisible_by_2109_l350_35048


namespace NUMINAMATH_GPT_determine_a_l350_35011

theorem determine_a (a : ℝ) : (∃ b : ℝ, (3 * (x : ℝ))^2 - 2 * 3 * b * x + b^2 = 9 * x^2 - 27 * x + a) → a = 20.25 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l350_35011


namespace NUMINAMATH_GPT_stratified_sampling_grade10_l350_35044

theorem stratified_sampling_grade10
  (total_students : ℕ)
  (grade10_students : ℕ)
  (grade11_students : ℕ)
  (grade12_students : ℕ)
  (sample_size : ℕ)
  (h1 : total_students = 700)
  (h2 : grade10_students = 300)
  (h3 : grade11_students = 200)
  (h4 : grade12_students = 200)
  (h5 : sample_size = 35)
  : (grade10_students * sample_size / total_students) = 15 := 
sorry

end NUMINAMATH_GPT_stratified_sampling_grade10_l350_35044


namespace NUMINAMATH_GPT_sum_of_a_b_vert_asymptotes_l350_35088

theorem sum_of_a_b_vert_asymptotes (a b : ℝ) 
  (h1 : ∀ x : ℝ, x = -1 → x^2 + a * x + b = 0) 
  (h2 : ∀ x : ℝ, x = 3 → x^2 + a * x + b = 0) : 
  a + b = -5 :=
sorry

end NUMINAMATH_GPT_sum_of_a_b_vert_asymptotes_l350_35088


namespace NUMINAMATH_GPT_one_meter_eq_jumps_l350_35007

theorem one_meter_eq_jumps 
  (x y a b p q s t : ℝ) 
  (h1 : x * hops = y * skips)
  (h2 : a * jumps = b * hops)
  (h3 : p * skips = q * leaps)
  (h4 : s * leaps = t * meters) :
  1 * meters = (sp * x * a / (tq * y * b)) * jumps :=
sorry

end NUMINAMATH_GPT_one_meter_eq_jumps_l350_35007


namespace NUMINAMATH_GPT_exists_n_for_binomial_congruence_l350_35076

theorem exists_n_for_binomial_congruence 
  (p : ℕ) (a k : ℕ) (prime_p : Nat.Prime p) 
  (positive_a : a > 0) (positive_k : k > 0)
  (h1 : p^a < k) (h2 : k < 2 * p^a) : 
  ∃ n : ℕ, n < p^(2 * a) ∧ (Nat.choose n k) % p^a = n % p^a ∧ n % p^a = k % p^a :=
by
  sorry

end NUMINAMATH_GPT_exists_n_for_binomial_congruence_l350_35076


namespace NUMINAMATH_GPT_restaurant_tip_difference_l350_35033

theorem restaurant_tip_difference
  (a b : ℝ)
  (h1 : 0.15 * a = 3)
  (h2 : 0.25 * b = 3)
  : a - b = 8 := 
sorry

end NUMINAMATH_GPT_restaurant_tip_difference_l350_35033


namespace NUMINAMATH_GPT_find_unique_n_k_l350_35057

theorem find_unique_n_k (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) :
    (n+1)^n = 2 * n^k + 3 * n + 1 ↔ (n = 3 ∧ k = 3) := by
  sorry

end NUMINAMATH_GPT_find_unique_n_k_l350_35057


namespace NUMINAMATH_GPT_alison_birth_weekday_l350_35041

-- Definitions for the problem conditions
def days_in_week : ℕ := 7

-- John's birth day
def john_birth_weekday : ℕ := 3  -- Assuming Monday=0, Tuesday=1, ..., Wednesday=3, ...

-- Number of days Alison was born later
def days_later : ℕ := 72

-- Proof that the resultant day is Friday
theorem alison_birth_weekday : (john_birth_weekday + days_later) % days_in_week = 5 :=
by
  sorry

end NUMINAMATH_GPT_alison_birth_weekday_l350_35041


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l350_35075

theorem arithmetic_sequence_problem (a₁ d S₁₀ : ℝ) (h1 : d < 0) (h2 : (a₁ + d) * (a₁ + 3 * d) = 12) 
  (h3 : (a₁ + d) + (a₁ + 3 * d) = 8) (h4 : S₁₀ = 10 * a₁ + 10 * (10 - 1) / 2 * d) : 
  true := sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l350_35075


namespace NUMINAMATH_GPT_f_eq_91_for_all_n_leq_100_l350_35039

noncomputable def f : ℤ → ℝ := sorry

theorem f_eq_91_for_all_n_leq_100 (n : ℤ) (h : n ≤ 100) : f n = 91 := sorry

end NUMINAMATH_GPT_f_eq_91_for_all_n_leq_100_l350_35039


namespace NUMINAMATH_GPT_paint_brush_ratio_l350_35042

theorem paint_brush_ratio 
  (s w : ℝ) 
  (h1 : s > 0) 
  (h2 : w > 0) 
  (h3 : (1 / 2) * w ^ 2 + (1 / 2) * (s - w) ^ 2 = (s ^ 2) / 3) 
  : s / w = 3 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_paint_brush_ratio_l350_35042


namespace NUMINAMATH_GPT_original_price_double_value_l350_35096

theorem original_price_double_value :
  ∃ (P : ℝ), P + 0.30 * P = 351 ∧ 2 * P = 540 :=
by
  sorry

end NUMINAMATH_GPT_original_price_double_value_l350_35096


namespace NUMINAMATH_GPT_find_X_l350_35027

variable (X : ℝ)  -- Threshold income level for the lower tax rate
variable (I : ℝ)  -- Income of the citizen
variable (T : ℝ)  -- Total tax amount

-- Conditions
def income : Prop := I = 50000
def tax_amount : Prop := T = 8000
def tax_formula : Prop := T = 0.15 * X + 0.20 * (I - X)

theorem find_X (h1 : income I) (h2 : tax_amount T) (h3 : tax_formula T I X) : X = 40000 :=
by
  sorry

end NUMINAMATH_GPT_find_X_l350_35027


namespace NUMINAMATH_GPT_evaluate_expression_l350_35047

theorem evaluate_expression : 3 + (-3)^2 = 12 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l350_35047


namespace NUMINAMATH_GPT_find_m_n_l350_35010

theorem find_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (hmn : m^n = n^(m - n)) : 
  (m = 9 ∧ n = 3) ∨ (m = 8 ∧ n = 2) :=
sorry

end NUMINAMATH_GPT_find_m_n_l350_35010


namespace NUMINAMATH_GPT_range_of_m_l350_35040

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ (Set.Ioc 0 2) then 2^x - 1 else sorry

def g (x m : ℝ) : ℝ :=
x^2 - 2*x + m

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-2:ℝ) 2, f (-x) = -f x) ∧
  (∀ x ∈ Set.Ioc (0:ℝ) 2, f x = 2^x - 1) ∧
  (∀ x1 ∈ Set.Icc (-2:ℝ) 2, ∃ x2 ∈ Set.Icc (-2:ℝ) 2, g x2 m = f x1) 
  → -5 ≤ m ∧ m ≤ -2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l350_35040


namespace NUMINAMATH_GPT_talias_fathers_age_l350_35064

-- Definitions based on the conditions
variable (T M F : ℕ)

-- The conditions
axiom h1 : T + 7 = 20
axiom h2 : M = 3 * T
axiom h3 : F + 3 = M

-- Goal: Prove that Talia's father (F) is currently 36 years old
theorem talias_fathers_age : F = 36 :=
by
  sorry

end NUMINAMATH_GPT_talias_fathers_age_l350_35064


namespace NUMINAMATH_GPT_find_list_price_l350_35037

theorem find_list_price (P : ℝ) (h1 : 0.873 * P = 61.11) : P = 61.11 / 0.873 :=
by
  sorry

end NUMINAMATH_GPT_find_list_price_l350_35037


namespace NUMINAMATH_GPT_find_a_value_l350_35083

theorem find_a_value (a : ℝ) (f : ℝ → ℝ)
  (h_def : ∀ x, f x = (Real.exp (x - a) - 1) * Real.log (x + 2 * a - 1))
  (h_ge_0 : ∀ x, x > 1 - 2 * a → f x ≥ 0) : a = 2 / 3 :=
by
  -- Omitted proof
  sorry

end NUMINAMATH_GPT_find_a_value_l350_35083


namespace NUMINAMATH_GPT_cos_330_is_sqrt3_over_2_l350_35053

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_cos_330_is_sqrt3_over_2_l350_35053


namespace NUMINAMATH_GPT_ratio_comparison_l350_35049

theorem ratio_comparison (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) (h_m_lt_n : m < n) :
  (m + 3) / (n + 3) > m / n :=
sorry

end NUMINAMATH_GPT_ratio_comparison_l350_35049


namespace NUMINAMATH_GPT_sphere_volume_l350_35034

theorem sphere_volume (h : 4 * π * r^2 = 256 * π) : (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end NUMINAMATH_GPT_sphere_volume_l350_35034


namespace NUMINAMATH_GPT_range_of_m_for_inequality_l350_35032

theorem range_of_m_for_inequality (x y m : ℝ) :
  (∀ x y : ℝ, 3*x^2 + y^2 ≥ m * x * (x + y)) ↔ (-6 ≤ m ∧ m ≤ 2) := sorry

end NUMINAMATH_GPT_range_of_m_for_inequality_l350_35032


namespace NUMINAMATH_GPT_employee_b_payment_l350_35029

theorem employee_b_payment (total_payment : ℝ) (A_ratio : ℝ) (payment_B : ℝ) : 
  total_payment = 550 ∧ A_ratio = 1.2 ∧ total_payment = payment_B + A_ratio * payment_B → payment_B = 250 := 
by
  sorry

end NUMINAMATH_GPT_employee_b_payment_l350_35029
