import Mathlib

namespace NUMINAMATH_GPT_one_inch_represents_feet_l100_10097

def height_statue : ℕ := 80 -- Height of the statue in feet

def height_model : ℕ := 5 -- Height of the model in inches

theorem one_inch_represents_feet : (height_statue / height_model) = 16 := 
by
  sorry

end NUMINAMATH_GPT_one_inch_represents_feet_l100_10097


namespace NUMINAMATH_GPT_cheese_pizzas_l100_10052

theorem cheese_pizzas (p b c total : ℕ) (h1 : p = 2) (h2 : b = 6) (h3 : total = 14) (ht : p + b + c = total) : c = 6 := 
by
  sorry

end NUMINAMATH_GPT_cheese_pizzas_l100_10052


namespace NUMINAMATH_GPT_gcd_45_75_105_l100_10037

theorem gcd_45_75_105 : Nat.gcd (45 : ℕ) (Nat.gcd 75 105) = 15 := 
by
  sorry

end NUMINAMATH_GPT_gcd_45_75_105_l100_10037


namespace NUMINAMATH_GPT_proof_problem_l100_10028

variable {a b c d : ℝ}
variable {x1 y1 x2 y2 x3 y3 x4 y4 : ℝ}

-- Assume the conditions
variable (habcd_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
variable (unity_circle : x1^2 + y1^2 = 1 ∧ x2^2 + y2^2 = 1 ∧ x3^2 + y3^2 = 1 ∧ x4^2 + y4^2 = 1)
variable (unit_sum : a * b + c * d = 1)

-- Statement to prove
theorem proof_problem :
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2
    ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := 
  sorry

end NUMINAMATH_GPT_proof_problem_l100_10028


namespace NUMINAMATH_GPT_comprehensive_score_l100_10057

theorem comprehensive_score :
  let w_c := 0.4
  let w_u := 0.6
  let s_c := 80
  let s_u := 90
  s_c * w_c + s_u * w_u = 86 :=
by
  sorry

end NUMINAMATH_GPT_comprehensive_score_l100_10057


namespace NUMINAMATH_GPT_rice_mixture_ratio_l100_10001

theorem rice_mixture_ratio
  (cost_variety1 : ℝ := 5) 
  (cost_variety2 : ℝ := 8.75) 
  (desired_cost_mixture : ℝ := 7.50) 
  (x y : ℝ) :
  5 * x + 8.75 * y = 7.50 * (x + y) → 
  y / x = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rice_mixture_ratio_l100_10001


namespace NUMINAMATH_GPT_nat_divides_power_difference_l100_10033

theorem nat_divides_power_difference (n : ℕ) : n ∣ 2 ^ (2 * n.factorial) - 2 ^ n.factorial := by
  sorry

end NUMINAMATH_GPT_nat_divides_power_difference_l100_10033


namespace NUMINAMATH_GPT_find_f_neg_one_l100_10043

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x >= 0 then 2^x + 2 * x + m else -(2^(-x) + 2 * (-x) + m)

theorem find_f_neg_one (m : ℝ) (h_m : f 0 m = 0) : f (-1) (-1) = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_one_l100_10043


namespace NUMINAMATH_GPT_ramu_profit_percent_l100_10017

theorem ramu_profit_percent
  (cost_of_car : ℕ)
  (cost_of_repairs : ℕ)
  (selling_price : ℕ)
  (total_cost : ℕ := cost_of_car + cost_of_repairs)
  (profit : ℕ := selling_price - total_cost)
  (profit_percent : ℚ := ((profit : ℚ) / total_cost) * 100)
  (h1 : cost_of_car = 42000)
  (h2 : cost_of_repairs = 15000)
  (h3 : selling_price = 64900) :
  profit_percent = 13.86 :=
by
  sorry

end NUMINAMATH_GPT_ramu_profit_percent_l100_10017


namespace NUMINAMATH_GPT_original_slices_proof_l100_10070

def original_slices (andy_consumption toast_slices leftover_slice: ℕ) : ℕ :=
  andy_consumption + toast_slices + leftover_slice

theorem original_slices_proof :
  original_slices (3 * 2) (10 * 2) 1 = 27 :=
by
  sorry

end NUMINAMATH_GPT_original_slices_proof_l100_10070


namespace NUMINAMATH_GPT_mr_thompson_third_score_is_78_l100_10090

theorem mr_thompson_third_score_is_78 :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧ 
                   (a = 58 ∧ b = 65 ∧ c = 70 ∧ d = 78) ∧ 
                   (a + b + c + d) % 4 = 3 ∧ 
                   (∀ i j k, (a + i + j + k) % 4 = 0) ∧ -- This checks that average is integer
                   c = 78 := sorry

end NUMINAMATH_GPT_mr_thompson_third_score_is_78_l100_10090


namespace NUMINAMATH_GPT_certain_number_value_l100_10013

theorem certain_number_value
  (t b c x : ℝ)
  (h1 : (t + b + c + x + 15) / 5 = 12)
  (h2 : (t + b + c + 29) / 4 = 15) :
  x = 14 :=
by 
  sorry

end NUMINAMATH_GPT_certain_number_value_l100_10013


namespace NUMINAMATH_GPT_sin_value_proof_l100_10071

theorem sin_value_proof (θ : ℝ) (h : Real.cos (5 * Real.pi / 12 - θ) = 1 / 3) :
  Real.sin (Real.pi / 12 + θ) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_sin_value_proof_l100_10071


namespace NUMINAMATH_GPT_one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero_l100_10099

theorem one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a < 1 / b) ↔ ((a * b) / (a^3 - b^3) > 0) := 
by
  sorry

end NUMINAMATH_GPT_one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero_l100_10099


namespace NUMINAMATH_GPT_problem_180_180_minus_12_l100_10095

namespace MathProof

theorem problem_180_180_minus_12 :
  180 * (180 - 12) - (180 * 180 - 12) = -2148 := 
by
  -- Placeholders for computation steps
  sorry

end MathProof

end NUMINAMATH_GPT_problem_180_180_minus_12_l100_10095


namespace NUMINAMATH_GPT_triangle_perimeter_l100_10047

def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

variable (a b c : ℝ)

theorem triangle_perimeter
  (h1 : 90 = (1/2) * 18 * b)
  (h2 : right_triangle 18 b c) :
  18 + b + c = 28 + 2 * Real.sqrt 106 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l100_10047


namespace NUMINAMATH_GPT_base_amount_calculation_l100_10076

theorem base_amount_calculation (tax_amount : ℝ) (tax_rate : ℝ) (base_amount : ℝ) 
  (h1 : tax_amount = 82) (h2 : tax_rate = 82) : base_amount = 100 :=
by
  -- Proof will be provided here.
  sorry

end NUMINAMATH_GPT_base_amount_calculation_l100_10076


namespace NUMINAMATH_GPT_geometric_sequence_178th_term_l100_10084

-- Conditions of the problem as definitions
def first_term : ℤ := 5
def second_term : ℤ := -20
def common_ratio : ℤ := second_term / first_term
def nth_term (a : ℤ) (r : ℤ) (n : ℕ) : ℤ := a * r^(n-1)

-- The translated problem statement in Lean 4
theorem geometric_sequence_178th_term :
  nth_term first_term common_ratio 178 = -5 * 4^177 :=
by
  repeat { sorry }

end NUMINAMATH_GPT_geometric_sequence_178th_term_l100_10084


namespace NUMINAMATH_GPT_fraction_equivalence_l100_10063

theorem fraction_equivalence (x y : ℝ) (h : x ≠ y) :
  (x - y)^2 / (x^2 - y^2) = (x - y) / (x + y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_equivalence_l100_10063


namespace NUMINAMATH_GPT_alex_mother_age_proof_l100_10086

-- Define the initial conditions
def alex_age_2004 : ℕ := 7
def mother_age_2004 : ℕ := 35
def initial_year : ℕ := 2004

-- Define the time variable and the relationship conditions
def years_after_2004 (x : ℕ) : Prop :=
  let alex_age := alex_age_2004 + x
  let mother_age := mother_age_2004 + x
  mother_age = 2 * alex_age

-- State the theorem to be proved
theorem alex_mother_age_proof : ∃ x : ℕ, years_after_2004 x ∧ initial_year + x = 2025 :=
by
  sorry

end NUMINAMATH_GPT_alex_mother_age_proof_l100_10086


namespace NUMINAMATH_GPT_units_digit_of_p_is_6_l100_10050

theorem units_digit_of_p_is_6 (p : ℤ) (h1 : p % 10 > 0) 
                             (h2 : ((p^3) % 10 - (p^2) % 10) = 0) 
                             (h3 : (p + 1) % 10 = 7) : 
                             p % 10 = 6 :=
by sorry

end NUMINAMATH_GPT_units_digit_of_p_is_6_l100_10050


namespace NUMINAMATH_GPT_total_books_left_l100_10060

def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def benny_lost_books : ℕ := 24

theorem total_books_left : sandy_books + tim_books - benny_lost_books = 19 :=
by
  sorry

end NUMINAMATH_GPT_total_books_left_l100_10060


namespace NUMINAMATH_GPT_min_value_of_A_div_B_l100_10056

noncomputable def A (g1 : Finset ℕ) : ℕ :=
  g1.prod id

noncomputable def B (g2 : Finset ℕ) : ℕ :=
  g2.prod id

theorem min_value_of_A_div_B : ∃ (g1 g2 : Finset ℕ), 
  g1 ∪ g2 = (Finset.range 31).erase 0 ∧ g1 ∩ g2 = ∅ ∧ A g1 % B g2 = 0 ∧ A g1 / B g2 = 1077205 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_A_div_B_l100_10056


namespace NUMINAMATH_GPT_ratio_of_men_to_women_l100_10074

theorem ratio_of_men_to_women 
  (M W : ℕ) 
  (h1 : W = M + 5) 
  (h2 : M + W = 15): M = 5 ∧ W = 10 ∧ (M + W) / Nat.gcd M W = 1 ∧ (W + M) / Nat.gcd M W = 2 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_men_to_women_l100_10074


namespace NUMINAMATH_GPT_solution_set_of_inequality_l100_10044

theorem solution_set_of_inequality : 
  {x : ℝ | x * (x + 3) ≥ 0} = {x : ℝ | x ≥ 0 ∨ x ≤ -3} := 
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l100_10044


namespace NUMINAMATH_GPT_Hulk_jump_l100_10042

theorem Hulk_jump :
  ∃ n : ℕ, 2^n > 500 ∧ ∀ m : ℕ, m < n → 2^m ≤ 500 :=
by
  sorry

end NUMINAMATH_GPT_Hulk_jump_l100_10042


namespace NUMINAMATH_GPT_adam_earning_per_lawn_l100_10019

theorem adam_earning_per_lawn 
  (total_lawns : ℕ) 
  (forgotten_lawns : ℕ) 
  (total_earnings : ℕ) 
  (h1 : total_lawns = 12) 
  (h2 : forgotten_lawns = 8) 
  (h3 : total_earnings = 36) : 
  total_earnings / (total_lawns - forgotten_lawns) = 9 :=
by
  sorry

end NUMINAMATH_GPT_adam_earning_per_lawn_l100_10019


namespace NUMINAMATH_GPT_cannot_make_120_cents_with_6_coins_l100_10049

def Coin := ℕ → ℕ -- represents a number of each type of coin

noncomputable def coin_value (c : Coin) : ℕ :=
  c 0 * 1 + c 1 * 5 + c 2 * 10 + c 3 * 25

def total_coins (c : Coin) : ℕ :=
  c 0 + c 1 + c 2 + c 3

theorem cannot_make_120_cents_with_6_coins (c : Coin) (h1 : total_coins c = 6) :
  coin_value c ≠ 120 :=
sorry

end NUMINAMATH_GPT_cannot_make_120_cents_with_6_coins_l100_10049


namespace NUMINAMATH_GPT_parallel_line_with_y_intercept_l100_10027

theorem parallel_line_with_y_intercept (x y : ℝ) (m : ℝ) : 
  ((x + y + 4 = 0) → (x + y + m = 0)) ∧ (m = 1)
 := by sorry

end NUMINAMATH_GPT_parallel_line_with_y_intercept_l100_10027


namespace NUMINAMATH_GPT_overall_profit_percentage_is_30_l100_10051

noncomputable def overall_profit_percentage (n_A n_B : ℕ) (price_A price_B profit_A profit_B : ℝ) : ℝ :=
  (n_A * profit_A + n_B * profit_B) / (n_A * price_A + n_B * price_B) * 100

theorem overall_profit_percentage_is_30 :
  overall_profit_percentage 5 10 850 950 225 300 = 30 :=
by
  sorry

end NUMINAMATH_GPT_overall_profit_percentage_is_30_l100_10051


namespace NUMINAMATH_GPT_green_notebook_cost_each_l100_10092

-- Definitions for conditions:
def num_notebooks := 4
def num_green_notebooks := 2
def num_black_notebooks := 1
def num_pink_notebooks := 1
def total_cost := 45
def black_notebook_cost := 15
def pink_notebook_cost := 10

-- Define the problem statement:
theorem green_notebook_cost_each : 
  (2 * g + black_notebook_cost + pink_notebook_cost = total_cost) → 
  g = 10 := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_green_notebook_cost_each_l100_10092


namespace NUMINAMATH_GPT_find_g_neg_2_l100_10055

-- Definitions
variable {R : Type*} [CommRing R] [Inhabited R]
variable (f g : R → R)

-- Conditions
axiom odd_y (x : R) : f (-x) + 2 * x^2 = -(f x + 2 * x^2)
axiom definition_g (x : R) : g x = f x + 1
axiom value_f_2 : f 2 = 2

-- Goal
theorem find_g_neg_2 : g (-2) = -17 :=
by
  sorry

end NUMINAMATH_GPT_find_g_neg_2_l100_10055


namespace NUMINAMATH_GPT_stable_state_exists_l100_10012

-- Definition of the problem
theorem stable_state_exists 
(N : ℕ) (N_ge_3 : N ≥ 3) (letters : Fin N → Fin 3) 
(perform_operation : ∀ (letters : Fin N → Fin 3), Fin N → Fin 3)
(stable : ∀ (letters : Fin N → Fin 3), Prop)
(initial_state : Fin N → Fin 3):
  ∃ (state : Fin N → Fin 3), (∀ i, perform_operation state i = state i) ∧ stable state :=
sorry

end NUMINAMATH_GPT_stable_state_exists_l100_10012


namespace NUMINAMATH_GPT_binom_n_plus_one_n_l100_10024

theorem binom_n_plus_one_n (n : ℕ) (h : 0 < n) : Nat.choose (n + 1) n = n + 1 := 
sorry

end NUMINAMATH_GPT_binom_n_plus_one_n_l100_10024


namespace NUMINAMATH_GPT_solve_x_l100_10014

theorem solve_x (x : ℝ) : (x - 1)^2 = 4 → (x = 3 ∨ x = -1) := 
by 
  sorry

end NUMINAMATH_GPT_solve_x_l100_10014


namespace NUMINAMATH_GPT_problem_eq_995_l100_10065

theorem problem_eq_995 :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) /
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400))
  = 995 := sorry

end NUMINAMATH_GPT_problem_eq_995_l100_10065


namespace NUMINAMATH_GPT_solve_for_F_l100_10054

theorem solve_for_F (C F : ℝ) (h1 : C = 5 / 9 * (F - 32)) (h2 : C = 40) : F = 104 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_F_l100_10054


namespace NUMINAMATH_GPT_trigonometric_product_l100_10067

theorem trigonometric_product :
  (1 - Real.sin (Real.pi / 12)) * 
  (1 - Real.sin (5 * Real.pi / 12)) * 
  (1 - Real.sin (7 * Real.pi / 12)) * 
  (1 - Real.sin (11 * Real.pi / 12)) = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_trigonometric_product_l100_10067


namespace NUMINAMATH_GPT_abs_gt_not_implies_gt_l100_10058

noncomputable def abs_gt_implies_gt (a b : ℝ) : Prop :=
  |a| > |b| → a > b

theorem abs_gt_not_implies_gt (a b : ℝ) :
  ¬ abs_gt_implies_gt a b :=
sorry

end NUMINAMATH_GPT_abs_gt_not_implies_gt_l100_10058


namespace NUMINAMATH_GPT_average_score_l100_10034

theorem average_score (a_males : ℕ) (a_females : ℕ) (n_males : ℕ) (n_females : ℕ)
  (h_males : a_males = 85) (h_females : a_females = 92) (h_n_males : n_males = 8) (h_n_females : n_females = 20) :
  (a_males * n_males + a_females * n_females) / (n_males + n_females) = 90 :=
by
  sorry

end NUMINAMATH_GPT_average_score_l100_10034


namespace NUMINAMATH_GPT_damaged_books_count_l100_10093

variables (o d : ℕ)

theorem damaged_books_count (h1 : o + d = 69) (h2 : o = 6 * d - 8) : d = 11 := 
by 
  sorry

end NUMINAMATH_GPT_damaged_books_count_l100_10093


namespace NUMINAMATH_GPT_crosswalk_red_light_wait_l100_10036

theorem crosswalk_red_light_wait :
  let red_light_duration := 40
  let wait_time_requirement := 15
  let favorable_duration := red_light_duration - wait_time_requirement
  (favorable_duration : ℝ) / red_light_duration = (5 : ℝ) / 8 :=
by
  sorry

end NUMINAMATH_GPT_crosswalk_red_light_wait_l100_10036


namespace NUMINAMATH_GPT_min_value_of_m_l100_10094

open Real

-- Definitions from the conditions
def condition1 (m : ℝ) : Prop :=
  m > 0

def condition2 (m : ℝ) : Prop :=
  ∀ (x : ℝ), 1 < x → 2 * exp (2 * m * x) - (log x) / m ≥ 0

-- The theorem statement for the minimum value of m
theorem min_value_of_m (m : ℝ) : condition1 m → condition2 m → m ≥ 1 / (2 * exp 1) := 
sorry

end NUMINAMATH_GPT_min_value_of_m_l100_10094


namespace NUMINAMATH_GPT_difference_is_24_l100_10069

namespace BuffaloesAndDucks

def numLegs (B D : ℕ) : ℕ := 4 * B + 2 * D

def numHeads (B D : ℕ) : ℕ := B + D

def diffLegsAndHeads (B D : ℕ) : ℕ := numLegs B D - 2 * numHeads B D

theorem difference_is_24 (D : ℕ) : diffLegsAndHeads 12 D = 24 := by
  sorry

end BuffaloesAndDucks

end NUMINAMATH_GPT_difference_is_24_l100_10069


namespace NUMINAMATH_GPT_compute_five_fold_application_l100_10040

def f (x : ℤ) : ℤ :=
  if x ≥ 0 then -2 * x^2 else x^2 + 4 * x + 12

theorem compute_five_fold_application :
  f (f (f (f (f 2)))) = -449183247763232 :=
  by
    sorry

end NUMINAMATH_GPT_compute_five_fold_application_l100_10040


namespace NUMINAMATH_GPT_tim_income_less_juan_l100_10091

variable {T M J : ℝ}

theorem tim_income_less_juan :
  (M = 1.60 * T) → (M = 0.6400000000000001 * J) → T = 0.4 * J :=
by
  sorry

end NUMINAMATH_GPT_tim_income_less_juan_l100_10091


namespace NUMINAMATH_GPT_gcd_polynomials_l100_10023

-- Define a as a multiple of 1836
def is_multiple_of (a b : ℤ) : Prop := ∃ k : ℤ, a = k * b

-- Problem statement: gcd of the polynomial expressions given the condition
theorem gcd_polynomials (a : ℤ) (h : is_multiple_of a 1836) : Int.gcd (2 * a^2 + 11 * a + 40) (a + 4) = 4 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomials_l100_10023


namespace NUMINAMATH_GPT_executed_is_9_l100_10082

-- Define the conditions based on given problem
variables (x K I : ℕ)

-- Condition 1: Number of killed
def number_killed (x : ℕ) : ℕ := 2 * x + 4

-- Condition 2: Number of injured
def number_injured (x : ℕ) : ℕ := (16 * x) / 3 + 8

-- Condition 3: Total of killed, injured, and executed is less than 98
def total_less_than_98 (x : ℕ) (k : ℕ) (i : ℕ) : Prop := k + i + x < 98

-- Condition 4: Relation between killed and executed
def killed_relation (x : ℕ) (k : ℕ) : Prop := k - 4 = 2 * x

-- The final theorem statement to prove
theorem executed_is_9 : ∃ x, number_killed x = 2 * x + 4 ∧
                       number_injured x = (16 * x) / 3 + 8 ∧
                       total_less_than_98 x (number_killed x) (number_injured x) ∧
                       killed_relation x (number_killed x) ∧
                       x = 9 :=
by
  sorry

end NUMINAMATH_GPT_executed_is_9_l100_10082


namespace NUMINAMATH_GPT_solve_system1_solve_system2_l100_10068

-- Definitions for the first system of equations
def system1_equation1 (x y : ℚ) := 3 * x - 6 * y = 4
def system1_equation2 (x y : ℚ) := x + 5 * y = 6

-- Definitions for the second system of equations
def system2_equation1 (x y : ℚ) := x / 4 + y / 3 = 3
def system2_equation2 (x y : ℚ) := 3 * (x - 4) - 2 * (y - 1) = -1

-- Lean statement for proving the solution to the first system
theorem solve_system1 :
  ∃ (x y : ℚ), system1_equation1 x y ∧ system1_equation2 x y ∧ x = 8 / 3 ∧ y = 2 / 3 :=
by
  sorry

-- Lean statement for proving the solution to the second system
theorem solve_system2 :
  ∃ (x y : ℚ), system2_equation1 x y ∧ system2_equation2 x y ∧ x = 6 ∧ y = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system1_solve_system2_l100_10068


namespace NUMINAMATH_GPT_megan_carrots_second_day_l100_10010

theorem megan_carrots_second_day : 
  ∀ (initial : ℕ) (thrown : ℕ) (total : ℕ) (second_day : ℕ),
  initial = 19 →
  thrown = 4 →
  total = 61 →
  second_day = (total - (initial - thrown)) →
  second_day = 46 :=
by
  intros initial thrown total second_day h_initial h_thrown h_total h_second_day
  rw [h_initial, h_thrown, h_total] at h_second_day
  sorry

end NUMINAMATH_GPT_megan_carrots_second_day_l100_10010


namespace NUMINAMATH_GPT_fixed_point_for_all_k_l100_10004

theorem fixed_point_for_all_k (k : ℝ) : (5, 225) ∈ { p : ℝ × ℝ | ∃ k : ℝ, p.snd = 9 * p.fst^2 + k * p.fst - 5 * k } :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_for_all_k_l100_10004


namespace NUMINAMATH_GPT_total_marbles_l100_10080

theorem total_marbles (y b g : ℝ) (h1 : y = 1.4 * b) (h2 : g = 1.75 * y) :
  b + y + g = 3.4643 * y :=
sorry

end NUMINAMATH_GPT_total_marbles_l100_10080


namespace NUMINAMATH_GPT_domain_eq_l100_10041

theorem domain_eq (f : ℝ → ℝ) : 
  (∀ x : ℝ, -1 ≤ 3 - 2 * x ∧ 3 - 2 * x ≤ 2) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 :=
by sorry

end NUMINAMATH_GPT_domain_eq_l100_10041


namespace NUMINAMATH_GPT_minimum_selling_price_l100_10046

theorem minimum_selling_price (total_cost : ℝ) (total_fruit : ℝ) (spoilage : ℝ) (min_price : ℝ) :
  total_cost = 760 ∧ total_fruit = 80 ∧ spoilage = 0.05 ∧ min_price = 10 → 
  ∀ price : ℝ, (price * total_fruit * (1 - spoilage) >= total_cost) → price >= min_price :=
by
  intros h price hp
  rcases h with ⟨hc, hf, hs, hm⟩
  sorry

end NUMINAMATH_GPT_minimum_selling_price_l100_10046


namespace NUMINAMATH_GPT_henrietta_has_three_bedrooms_l100_10098

theorem henrietta_has_three_bedrooms
  (living_room_walls_sqft : ℕ)
  (bedroom_walls_sqft : ℕ)
  (num_bedrooms : ℕ)
  (gallon_coverage_sqft : ℕ)
  (h1 : living_room_walls_sqft = 600)
  (h2 : bedroom_walls_sqft = 400)
  (h3 : gallon_coverage_sqft = 600)
  (h4 : num_bedrooms = 3) : 
  num_bedrooms = 3 :=
by
  exact h4

end NUMINAMATH_GPT_henrietta_has_three_bedrooms_l100_10098


namespace NUMINAMATH_GPT_cubic_representation_l100_10002

variable (a b : ℝ) (x : ℝ)
variable (v u w : ℝ)

axiom h1 : 6.266 * x^3 - 3 * a * x^2 + (3 * a^2 - b) * x - (a^3 - a * b) = 0
axiom h2 : b ≥ 0

theorem cubic_representation : v = a ∧ u = a ∧ w^2 = b → 
  6.266 * x^3 - 3 * v * x^2 + (3 * u^2 - w^2) * x - (u^3 - u * w^2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_cubic_representation_l100_10002


namespace NUMINAMATH_GPT_find_product_l100_10061

theorem find_product (a b c d : ℝ) 
  (h_avg : (a + b + c + d) / 4 = 7.1)
  (h_rel : 2.5 * a = b - 1.2 ∧ b - 1.2 = c + 4.8 ∧ c + 4.8 = 0.25 * d) :
  a * b * c * d = 49.6 := 
sorry

end NUMINAMATH_GPT_find_product_l100_10061


namespace NUMINAMATH_GPT_a_1_value_l100_10066

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ)

axiom a_n_def : ∀ n ≥ 2, a n + 2 * (S n) * (S (n - 1)) = 0
axiom S_5_value : S 5 = 1/11
axiom summation_def : ∀ k ≥ 1, S k = S (k - 1) + a k

theorem a_1_value : a 1 = 1/3 := by
  sorry

end NUMINAMATH_GPT_a_1_value_l100_10066


namespace NUMINAMATH_GPT_leftover_cents_l100_10038

noncomputable def total_cents (pennies nickels dimes quarters : Nat) : Nat :=
  (pennies * 1) + (nickels * 5) + (dimes * 10) + (quarters * 25)

noncomputable def total_cost (num_people : Nat) (cost_per_person : Nat) : Nat :=
  num_people * cost_per_person

theorem leftover_cents (h₁ : total_cents 123 85 35 26 = 1548)
                       (h₂ : total_cost 5 300 = 1500) :
  1548 - 1500 = 48 :=
sorry

end NUMINAMATH_GPT_leftover_cents_l100_10038


namespace NUMINAMATH_GPT_perpendicular_to_plane_l100_10073

theorem perpendicular_to_plane (Line : Type) (Plane : Type) (triangle : Plane) (circle : Plane)
  (perpendicular1 : Line → Plane → Prop)
  (perpendicular2 : Line → Plane → Prop) :
  (∀ l, ∃ t, perpendicular1 l t ∧ t = triangle) ∧ (∀ l, ∃ c, perpendicular2 l c ∧ c = circle) →
  (∀ l, ∃ p, (perpendicular1 l p ∨ perpendicular2 l p) ∧ (p = triangle ∨ p = circle)) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_to_plane_l100_10073


namespace NUMINAMATH_GPT_degree_product_l100_10083

-- Define the degrees of the polynomials p and q
def degree_p : ℕ := 3
def degree_q : ℕ := 4

-- Define the functions p(x) and q(x) as polynomials and their respective degrees
axiom degree_p_definition (p : Polynomial ℝ) : p.degree = degree_p
axiom degree_q_definition (q : Polynomial ℝ) : q.degree = degree_q

-- Define the degree of the product p(x^2) * q(x^4)
noncomputable def degree_p_x2_q_x4 (p q : Polynomial ℝ) : ℕ :=
  2 * degree_p + 4 * degree_q

-- Prove that the degree of p(x^2) * q(x^4) is 22
theorem degree_product (p q : Polynomial ℝ) (hp : p.degree = degree_p) (hq : q.degree = degree_q) :
  degree_p_x2_q_x4 p q = 22 :=
by
  sorry

end NUMINAMATH_GPT_degree_product_l100_10083


namespace NUMINAMATH_GPT_tan_half_angle_third_quadrant_l100_10005

theorem tan_half_angle_third_quadrant (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h : Real.sin α = -24/25) :
  Real.tan (α / 2) = -4/3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_half_angle_third_quadrant_l100_10005


namespace NUMINAMATH_GPT_composite_sum_l100_10030

theorem composite_sum (m n : ℕ) (h : 88 * m = 81 * n) : ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ (m + n) = p * q :=
by sorry

end NUMINAMATH_GPT_composite_sum_l100_10030


namespace NUMINAMATH_GPT_parallelogram_base_length_l100_10011

theorem parallelogram_base_length (A H : ℝ) (base : ℝ) 
    (hA : A = 72) (hH : H = 6) (h_area : A = base * H) : base = 12 := 
by 
  sorry

end NUMINAMATH_GPT_parallelogram_base_length_l100_10011


namespace NUMINAMATH_GPT_rectangular_prism_faces_edges_vertices_sum_l100_10016

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end NUMINAMATH_GPT_rectangular_prism_faces_edges_vertices_sum_l100_10016


namespace NUMINAMATH_GPT_roots_of_equation_l100_10059

theorem roots_of_equation (a x : ℝ) : x * (x + 5)^2 * (a - x) = 0 ↔ (x = 0 ∨ x = -5 ∨ x = a) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_equation_l100_10059


namespace NUMINAMATH_GPT_imaginary_part_of_z_l100_10062

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4 * I) * z = abs (4 + 3 * I)) : im z = 4 / 5 :=
sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l100_10062


namespace NUMINAMATH_GPT_weight_of_dried_grapes_l100_10029

def fresh_grapes_initial_weight : ℝ := 25
def fresh_grapes_water_percentage : ℝ := 0.90
def dried_grapes_water_percentage : ℝ := 0.20

theorem weight_of_dried_grapes :
  (fresh_grapes_initial_weight * (1 - fresh_grapes_water_percentage)) /
  (1 - dried_grapes_water_percentage) = 3.125 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_weight_of_dried_grapes_l100_10029


namespace NUMINAMATH_GPT_range_of_a_l100_10053

-- Definitions for propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, ¬(x^2 + (a-1)*x + 1 ≤ 0)

def q (a : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (a - 1)^x₁ < (a - 1)^x₂

-- The final theorem to prove
theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) → (-1 < a ∧ a ≤ 2) ∨ (a ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l100_10053


namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l100_10035

theorem isosceles_right_triangle_area (a b : ℝ) (h₁ : a = b) (h₂ : a + b = 20) : 
  (1 / 2) * a * b = 50 := 
by 
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l100_10035


namespace NUMINAMATH_GPT_probability_of_losing_weight_l100_10022

theorem probability_of_losing_weight (total_volunteers lost_weight : ℕ) (h_total : total_volunteers = 1000) (h_lost : lost_weight = 241) : 
    (lost_weight : ℚ) / total_volunteers = 0.24 := by
  sorry

end NUMINAMATH_GPT_probability_of_losing_weight_l100_10022


namespace NUMINAMATH_GPT_sum_infinite_partial_fraction_l100_10089

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_infinite_partial_fraction_l100_10089


namespace NUMINAMATH_GPT_permutations_five_three_eq_sixty_l100_10021

theorem permutations_five_three_eq_sixty : (Nat.factorial 5) / (Nat.factorial (5 - 3)) = 60 := 
by
  sorry

end NUMINAMATH_GPT_permutations_five_three_eq_sixty_l100_10021


namespace NUMINAMATH_GPT_range_of_a_l100_10039

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x + 1

def is_monotonic_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) a, 0 ≤ (deriv f) x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) a, 0 ≤ (deriv (f a)) x) → 1 ≤ a := 
sorry

end NUMINAMATH_GPT_range_of_a_l100_10039


namespace NUMINAMATH_GPT_impossible_to_use_up_all_parts_l100_10078

theorem impossible_to_use_up_all_parts (p q r : ℕ) :
  (∃ p q r : ℕ,
    2 * p + 2 * r + 2 = A ∧
    2 * p + q + 1 = B ∧
    q + r = C) → false :=
by {
  sorry
}

end NUMINAMATH_GPT_impossible_to_use_up_all_parts_l100_10078


namespace NUMINAMATH_GPT_second_integer_is_66_l100_10006

-- Define the conditions
def are_two_units_apart (a b c : ℤ) : Prop :=
  b = a + 2 ∧ c = a + 4

def sum_of_first_and_third_is_132 (a b c : ℤ) : Prop :=
  a + c = 132

-- State the theorem
theorem second_integer_is_66 (a b c : ℤ) 
  (H1 : are_two_units_apart a b c) 
  (H2 : sum_of_first_and_third_is_132 a b c) : b = 66 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_second_integer_is_66_l100_10006


namespace NUMINAMATH_GPT_jamie_school_distance_l100_10096

theorem jamie_school_distance
  (v : ℝ) -- usual speed in miles per hour
  (d : ℝ) -- distance to school in miles
  (h1 : (20 : ℝ) / 60 = 1 / 3) -- usual time to school in hours
  (h2 : (10 : ℝ) / 60 = 1 / 6) -- lighter traffic time in hours
  (h3 : d = v * (1 / 3)) -- distance equation for usual traffic
  (h4 : d = (v + 15) * (1 / 6)) -- distance equation for lighter traffic
  : d = 5 := by
  sorry

end NUMINAMATH_GPT_jamie_school_distance_l100_10096


namespace NUMINAMATH_GPT_special_op_eight_four_l100_10075

def special_op (a b : ℕ) : ℕ := 2 * a + a / b

theorem special_op_eight_four : special_op 8 4 = 18 := by
  sorry

end NUMINAMATH_GPT_special_op_eight_four_l100_10075


namespace NUMINAMATH_GPT_find_x_of_orthogonal_vectors_l100_10087

theorem find_x_of_orthogonal_vectors (x : ℝ) : 
  (⟨3, -4, 1⟩ : ℝ × ℝ × ℝ) • (⟨x, 2, -7⟩ : ℝ × ℝ × ℝ) = 0 → x = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_x_of_orthogonal_vectors_l100_10087


namespace NUMINAMATH_GPT_find_x2_y2_l100_10064

theorem find_x2_y2 (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = (10344 / 169) := by
  sorry

end NUMINAMATH_GPT_find_x2_y2_l100_10064


namespace NUMINAMATH_GPT_projection_of_orthogonal_vectors_l100_10018

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scale := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (scale * v.1, scale * v.2)

theorem projection_of_orthogonal_vectors
  (a b : ℝ × ℝ)
  (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj : proj (4, -2) a = (4 / 5, 8 / 5)) :
  proj (4, -2) b = (16 / 5, -18 / 5) :=
sorry

end NUMINAMATH_GPT_projection_of_orthogonal_vectors_l100_10018


namespace NUMINAMATH_GPT_initial_pens_eq_42_l100_10085

-- Definitions based on the conditions
def initial_books : ℕ := 143
def remaining_books : ℕ := 113
def remaining_pens : ℕ := 19
def sold_pens : ℕ := 23

-- Theorem to prove that the initial number of pens was 42
theorem initial_pens_eq_42 (b_init b_remain p_remain p_sold : ℕ) 
    (H_b_init : b_init = initial_books)
    (H_b_remain : b_remain = remaining_books)
    (H_p_remain : p_remain = remaining_pens)
    (H_p_sold : p_sold = sold_pens) : 
    (p_sold + p_remain = 42) := 
by {
    -- Provide proof later
    sorry
}

end NUMINAMATH_GPT_initial_pens_eq_42_l100_10085


namespace NUMINAMATH_GPT_rectangle_perimeter_equal_area_l100_10079

theorem rectangle_perimeter_equal_area (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 2 * a + 2 * b) : 2 * (a + b) = 18 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_equal_area_l100_10079


namespace NUMINAMATH_GPT_marge_final_plant_count_l100_10025

/-- Define the initial conditions of the garden -/
def initial_seeds : ℕ := 23
def marigold_seeds : ℕ := 10
def sunflower_seeds : ℕ := 8
def lavender_seeds : ℕ := 5
def seeds_without_growth : ℕ := 5

/-- Growth rates for each type of plant -/
def marigold_growth_rate : ℕ := 4
def sunflower_growth_rate : ℕ := 4
def lavender_growth_rate : ℕ := 3

/-- Impact of animals -/
def marigold_eaten_by_squirrels : ℕ := 2
def sunflower_eaten_by_rabbits : ℕ := 1

/-- Impact of pest control -/
def marigold_pest_control_reduction : ℕ := 0
def sunflower_pest_control_reduction : ℕ := 0
def lavender_pest_control_protected : ℕ := 2

/-- Impact of weeds -/
def weeds_strangled_plants : ℕ := 2

/-- Weeds left as plants -/
def weeds_kept_as_plants : ℕ := 1

/-- Marge's final number of plants -/
def survived_plants :=
  (marigold_growth_rate - marigold_eaten_by_squirrels - marigold_pest_control_reduction) +
  (sunflower_growth_rate - sunflower_eaten_by_rabbits - sunflower_pest_control_reduction) +
  (lavender_growth_rate - (lavender_growth_rate - lavender_pest_control_protected)) - weeds_strangled_plants

theorem marge_final_plant_count :
  survived_plants + weeds_kept_as_plants = 6 :=
by
  sorry

end NUMINAMATH_GPT_marge_final_plant_count_l100_10025


namespace NUMINAMATH_GPT_scientific_notation_l100_10003

variables (n : ℕ) (h : n = 505000)

theorem scientific_notation : n = 505000 → "5.05 * 10^5" = "scientific notation of 505000" :=
by
  intro h
  sorry

end NUMINAMATH_GPT_scientific_notation_l100_10003


namespace NUMINAMATH_GPT_units_digit_in_base_7_l100_10072

theorem units_digit_in_base_7 (n m : ℕ) (h1 : n = 312) (h2 : m = 57) : (n * m) % 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_in_base_7_l100_10072


namespace NUMINAMATH_GPT_tangent_line_through_origin_l100_10020

noncomputable def curve (x : ℝ) : ℝ := Real.exp (x - 1) + x

theorem tangent_line_through_origin :
  ∃ k : ℝ, k = 2 ∧ ∀ x y : ℝ, (y = k * x) ↔ (∃ m : ℝ, curve m = m + Real.exp (m - 1) ∧ (curve m) = (m + Real.exp (m - 1)) ∧ k = (Real.exp (m - 1) + 1) ∧ y = k * x ∧ y = 2*x) :=
by 
  sorry

end NUMINAMATH_GPT_tangent_line_through_origin_l100_10020


namespace NUMINAMATH_GPT_marble_probability_is_correct_l100_10088

def marbles_probability
  (total_marbles: ℕ) 
  (red_marbles: ℕ) 
  (blue_marbles: ℕ) 
  (green_marbles: ℕ)
  (choose_marbles: ℕ) 
  (required_red: ℕ) 
  (required_blue: ℕ) 
  (required_green: ℕ): ℚ := sorry

-- Define conditions
def total_marbles := 7
def red_marbles := 3
def blue_marbles := 2
def green_marbles := 2
def choose_marbles := 4
def required_red := 2
def required_blue := 1
def required_green := 1

-- Proof statement
theorem marble_probability_is_correct : 
  marbles_probability total_marbles red_marbles blue_marbles green_marbles choose_marbles required_red required_blue required_green = (12 / 35 : ℚ) :=
sorry

end NUMINAMATH_GPT_marble_probability_is_correct_l100_10088


namespace NUMINAMATH_GPT_point_K_outside_hexagon_and_length_KC_l100_10026

theorem point_K_outside_hexagon_and_length_KC :
    ∀ (A B C K : ℝ × ℝ),
    A = (0, 0) →
    B = (3, 0) →
    C = (3 / 2, (3 * Real.sqrt 3) / 2) →
    K = (15 / 2, - (3 * Real.sqrt 3) / 2) →
    (¬ (0 ≤ K.1 ∧ K.1 ≤ 3 ∧ 0 ≤ K.2 ∧ K.2 ≤ 3 * Real.sqrt 3)) ∧
    Real.sqrt ((K.1 - C.1) ^ 2 + (K.2 - C.2) ^ 2) = 3 * Real.sqrt 7 :=
by
  intros A B C K hA hB hC hK
  sorry

end NUMINAMATH_GPT_point_K_outside_hexagon_and_length_KC_l100_10026


namespace NUMINAMATH_GPT_general_term_formula_l100_10032

theorem general_term_formula (n : ℕ) : 
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (∀ n > 1, a n - a (n-1) = 2^(n-1)) → (a n = 2^n - 1) :=
  by 
  intros a h1 hdif
  sorry

end NUMINAMATH_GPT_general_term_formula_l100_10032


namespace NUMINAMATH_GPT_total_combined_area_l100_10081

-- Definition of the problem conditions
def base_parallelogram : ℝ := 20
def height_parallelogram : ℝ := 4
def base_triangle : ℝ := 20
def height_triangle : ℝ := 2

-- Given the conditions, we want to prove:
theorem total_combined_area :
  (base_parallelogram * height_parallelogram) + (0.5 * base_triangle * height_triangle) = 100 :=
by
  sorry  -- proof goes here

end NUMINAMATH_GPT_total_combined_area_l100_10081


namespace NUMINAMATH_GPT_tree_initial_leaves_l100_10009

theorem tree_initial_leaves (L : ℝ) (h1 : ∀ n : ℤ, 1 ≤ n ∧ n ≤ 4 → ∃ k : ℝ, L = k * (9/10)^n + k / 10^n)
                            (h2 : L * (9/10)^4 = 204) :
  L = 311 :=
by
  sorry

end NUMINAMATH_GPT_tree_initial_leaves_l100_10009


namespace NUMINAMATH_GPT_forum_posting_total_l100_10048

theorem forum_posting_total (num_members : ℕ) (num_answers_per_question : ℕ) (num_questions_per_hour : ℕ) (hours_per_day : ℕ) :
  num_members = 1000 ->
  num_answers_per_question = 5 ->
  num_questions_per_hour = 7 ->
  hours_per_day = 24 ->
  ((num_questions_per_hour * hours_per_day * num_members) + (num_answers_per_question * num_questions_per_hour * hours_per_day * num_members)) = 1008000 :=
by
  intros
  sorry

end NUMINAMATH_GPT_forum_posting_total_l100_10048


namespace NUMINAMATH_GPT_cos_B_value_l100_10000

theorem cos_B_value (A B C a b c : ℝ) (h₁ : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * a * Real.cos B) :
  Real.cos B = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_cos_B_value_l100_10000


namespace NUMINAMATH_GPT_wicket_keeper_older_than_captain_l100_10007

-- Define the team and various ages
def captain_age : ℕ := 28
def average_age_team : ℕ := 25
def number_of_players : ℕ := 11
def number_of_remaining_players : ℕ := number_of_players - 2
def average_age_remaining_players : ℕ := average_age_team - 1

theorem wicket_keeper_older_than_captain :
  ∃ (W : ℕ), W = captain_age + 3 ∧
  275 = number_of_players * average_age_team ∧
  216 = number_of_remaining_players * average_age_remaining_players ∧
  59 = 275 - 216 ∧
  W = 59 - captain_age :=
by
  sorry

end NUMINAMATH_GPT_wicket_keeper_older_than_captain_l100_10007


namespace NUMINAMATH_GPT_geometric_series_first_term_l100_10008

theorem geometric_series_first_term
  (a r : ℚ)
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 150) :
  a = 60 / 7 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_first_term_l100_10008


namespace NUMINAMATH_GPT_sample_size_calculation_l100_10077

-- Definitions based on the conditions
def num_classes : ℕ := 40
def num_representatives_per_class : ℕ := 3

-- Theorem statement we aim to prove
theorem sample_size_calculation : num_classes * num_representatives_per_class = 120 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_calculation_l100_10077


namespace NUMINAMATH_GPT_successive_product_4160_l100_10015

theorem successive_product_4160 (n : ℕ) (h : n * (n + 1) = 4160) : n = 64 :=
sorry

end NUMINAMATH_GPT_successive_product_4160_l100_10015


namespace NUMINAMATH_GPT_max_value_exp_l100_10045

theorem max_value_exp (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_constraint : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 8 + 7 * y * z + 5 * x * z ≤ 23.0219 :=
sorry

end NUMINAMATH_GPT_max_value_exp_l100_10045


namespace NUMINAMATH_GPT_f_3_equals_1000_l100_10031

-- Define the function property f(lg x) = x
axiom f : ℝ → ℝ
axiom lg : ℝ → ℝ -- log function
axiom f_property : ∀ x : ℝ, f (lg x) = x

-- Prove that f(3) = 10^3
theorem f_3_equals_1000 : f 3 = 10^3 :=
by 
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_f_3_equals_1000_l100_10031
