import Mathlib

namespace bananas_on_first_day_l214_214080

theorem bananas_on_first_day (total_bananas : ℕ) (days : ℕ) (increment : ℕ) (bananas_first_day : ℕ) :
  (total_bananas = 100) ∧ (days = 5) ∧ (increment = 6) ∧ ((bananas_first_day + (bananas_first_day + increment) + 
  (bananas_first_day + 2*increment) + (bananas_first_day + 3*increment) + (bananas_first_day + 4*increment)) = total_bananas) → 
  bananas_first_day = 8 :=
by
  sorry

end bananas_on_first_day_l214_214080


namespace barbara_typing_time_l214_214149

theorem barbara_typing_time :
  let original_speed := 212
  let reduction := 40
  let num_words := 3440
  let reduced_speed := original_speed - reduction
  let time := num_words / reduced_speed
  time = 20 :=
by
  sorry

end barbara_typing_time_l214_214149


namespace find_exponent_l214_214999

theorem find_exponent 
  (h1 : (1 : ℝ) / 9 = 3 ^ (-2 : ℝ))
  (h2 : (3 ^ (20 : ℝ) : ℝ) / 9 = 3 ^ x) : 
  x = 18 :=
by sorry

end find_exponent_l214_214999


namespace complete_the_square_d_l214_214553

theorem complete_the_square_d (x : ℝ) :
  ∃ c d, (x^2 + 10 * x + 9 = 0 → (x + c)^2 = d) ∧ d = 16 :=
sorry

end complete_the_square_d_l214_214553


namespace cost_of_graphing_calculator_l214_214284

/-
  Everton college paid $1625 for an order of 45 calculators.
  Each scientific calculator costs $10.
  The order included 20 scientific calculators and 25 graphing calculators.
  We need to prove that each graphing calculator costs $57.
-/

namespace EvertonCollege

theorem cost_of_graphing_calculator
  (total_cost : ℕ)
  (cost_scientific : ℕ)
  (num_scientific : ℕ)
  (num_graphing : ℕ)
  (cost_graphing : ℕ)
  (h_order : total_cost = 1625)
  (h_cost_scientific : cost_scientific = 10)
  (h_num_scientific : num_scientific = 20)
  (h_num_graphing : num_graphing = 25)
  (h_total_calc : num_scientific + num_graphing = 45)
  (h_pay : total_cost = num_scientific * cost_scientific + num_graphing * cost_graphing) :
  cost_graphing = 57 :=
by
  sorry

end EvertonCollege

end cost_of_graphing_calculator_l214_214284


namespace curlers_total_l214_214000

theorem curlers_total (P B G : ℕ) (h1 : 4 * P = P + B + G) (h2 : B = 2 * P) (h3 : G = 4) : 
  4 * P = 16 := 
by sorry

end curlers_total_l214_214000


namespace students_taking_history_but_not_statistics_l214_214541

theorem students_taking_history_but_not_statistics (H S U : ℕ) (total_students : ℕ) 
  (H_val : H = 36) (S_val : S = 30) (U_val : U = 59) (total_students_val : total_students = 90) :
  H - (H + S - U) = 29 := 
by
  sorry

end students_taking_history_but_not_statistics_l214_214541


namespace a_alone_completes_in_eight_days_l214_214598

variable (a b : Type)
variables (days_ab : ℝ) (days_a : ℝ) (days_ab_2 : ℝ)

noncomputable def days := ℝ

axiom work_together_four_days : days_ab = 4
axiom work_together_266666_days : days_ab_2 = 8 / 3

theorem a_alone_completes_in_eight_days (a b : Type) (days_ab : ℝ) (days_a : ℝ) (days_ab_2 : ℝ)
  (work_together_four_days : days_ab = 4)
  (work_together_266666_days : days_ab_2 = 8 / 3) :
  days_a = 8 :=
by
  sorry

end a_alone_completes_in_eight_days_l214_214598


namespace solve_equation_l214_214117

theorem solve_equation (x: ℝ) (h : (5 - x)^(1/3) = -5/2) : x = 165/8 :=
by
  -- proof skipped
  sorry

end solve_equation_l214_214117


namespace prove_scientific_notation_l214_214560

def scientific_notation_correct : Prop :=
  340000 = 3.4 * (10 ^ 5)

theorem prove_scientific_notation : scientific_notation_correct :=
  by
    sorry

end prove_scientific_notation_l214_214560


namespace remainder_of_10_pow_23_minus_7_mod_6_l214_214141

theorem remainder_of_10_pow_23_minus_7_mod_6 : ((10 ^ 23 - 7) % 6) = 3 := by
  sorry

end remainder_of_10_pow_23_minus_7_mod_6_l214_214141


namespace rate_downstream_l214_214905

-- Define the man's rate in still water
def rate_still_water : ℝ := 24.5

-- Define the rate of the current
def rate_current : ℝ := 7.5

-- Define the man's rate upstream (unused in the proof but given in the problem)
def rate_upstream : ℝ := 17.0

-- Prove that the man's rate when rowing downstream is as stated given the conditions
theorem rate_downstream : rate_still_water + rate_current = 32 := by
  simp [rate_still_water, rate_current]
  norm_num

end rate_downstream_l214_214905


namespace inequality_proof_l214_214664

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x + y + z ≥ 1/x + 1/y + 1/z) : 
  x/y + y/z + z/x ≥ 1/(x * y) + 1/(y * z) + 1/(z * x) :=
by
  sorry

end inequality_proof_l214_214664


namespace total_weight_is_correct_l214_214899

-- Define the weight of apples
def weight_of_apples : ℕ := 240

-- Define the multiplier for pears
def pears_multiplier : ℕ := 3

-- Define the weight of pears
def weight_of_pears := pears_multiplier * weight_of_apples

-- Define the total weight of apples and pears
def total_weight : ℕ := weight_of_apples + weight_of_pears

-- The theorem that states the total weight calculation
theorem total_weight_is_correct : total_weight = 960 := by
  sorry

end total_weight_is_correct_l214_214899


namespace ab_sum_l214_214681

theorem ab_sum (a b : ℕ) (h1: (a + b) % 9 = 8) (h2: (a - b) % 11 = 7) : a + b = 8 :=
sorry

end ab_sum_l214_214681


namespace arithmetic_sequence_sum_l214_214918

open Nat

theorem arithmetic_sequence_sum :
  let a := 50
  let d := 3
  let l := 98
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  3 * S = 3774 := 
by
  let a := 50
  let d := 3
  let l := 98
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  sorry

end arithmetic_sequence_sum_l214_214918


namespace part_a_part_b_part_c_l214_214595

def is_frameable (n : ℕ) : Prop :=
  n = 3 ∨ n = 4 ∨ n = 6

theorem part_a : is_frameable 3 ∧ is_frameable 4 ∧ is_frameable 6 :=
  sorry

theorem part_b (n : ℕ) (h : n ≥ 7) : ¬ is_frameable n :=
  sorry

theorem part_c : ¬ is_frameable 5 :=
  sorry

end part_a_part_b_part_c_l214_214595


namespace sum_of_squares_positive_l214_214355

theorem sum_of_squares_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2 > 0) ∧ (b^2 + c^2 > 0) ∧ (c^2 + a^2 > 0) :=
by
  sorry

end sum_of_squares_positive_l214_214355


namespace ratio_markus_age_son_age_l214_214970

variable (M S G : ℕ)

theorem ratio_markus_age_son_age (h1 : G = 20) (h2 : S = 2 * G) (h3 : M + S + G = 140) : M / S = 2 := by
  sorry

end ratio_markus_age_son_age_l214_214970


namespace a_5_is_9_l214_214616

-- Definition of the sequence sum S_n
def S : ℕ → ℕ
| n => n^2 - 1

-- Define the specific term in the sequence
def a (n : ℕ) :=
  if n = 1 then S 1 else S n - S (n - 1)

-- Theorem to prove
theorem a_5_is_9 : a 5 = 9 :=
sorry

end a_5_is_9_l214_214616


namespace functional_equation_zero_l214_214336

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + |y|) = f (|x|) + f (y)) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_equation_zero_l214_214336


namespace exists_close_points_l214_214622

theorem exists_close_points (r : ℝ) (h : r > 0) (points : Fin 5 → EuclideanSpace ℝ (Fin 3)) (hf : ∀ i, dist (points i) (0 : EuclideanSpace ℝ (Fin 3)) = r) :
  ∃ i j : Fin 5, i ≠ j ∧ dist (points i) (points j) ≤ r * Real.sqrt 2 :=
by 
  sorry

end exists_close_points_l214_214622


namespace repeating_decimal_division_l214_214542

theorem repeating_decimal_division:
  let x := (54 / 99 : ℚ)
  let y := (18 / 99 : ℚ)
  (x / y) * (1 / 2) = (3 / 2 : ℚ) := by
    sorry

end repeating_decimal_division_l214_214542


namespace find_number_l214_214011

theorem find_number :
  (∃ m : ℝ, 56 = (3 / 2) * m) ∧ (56 = 0.7 * 80) → m = 37 := by
  sorry

end find_number_l214_214011


namespace average_of_remaining_ten_numbers_l214_214013

theorem average_of_remaining_ten_numbers
  (avg_50 : ℝ)
  (n_50 : ℝ)
  (avg_40 : ℝ)
  (n_40 : ℝ)
  (sum_50 : n_50 * avg_50 = 3800)
  (sum_40 : n_40 * avg_40 = 3200)
  (n_10 : n_50 - n_40 = 10)
  : (3800 - 3200) / 10 = 60 :=
by
  sorry

end average_of_remaining_ten_numbers_l214_214013


namespace part1_solution_set_part2_minimum_value_l214_214748

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

theorem part1_solution_set (x : ℝ) :
  (f x ≥ -1) ↔ (2 / 3 ≤ x ∧ x ≤ 6) := sorry

variables {a b c : ℝ}
theorem part2_minimum_value (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = 6) :
  (1 / (2 * a + b) + 1 / (2 * a + c) ≥ 2 / 3) := 
sorry

end part1_solution_set_part2_minimum_value_l214_214748


namespace find_x_squared_perfect_square_l214_214339

theorem find_x_squared_perfect_square (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : n ≠ m)
  (h4 : n > m) (h5 : n % 2 ≠ m % 2) : 
  ∃ x : ℤ, x = 0 ∧ ∀ x, (x = 0) → ∃ k : ℕ, (x ^ (2 ^ n) - 1) / (x ^ (2 ^ m) - 1) = k^2 :=
sorry

end find_x_squared_perfect_square_l214_214339


namespace alexa_emily_profit_l214_214715

def lemonade_stand_profit : ℕ :=
  let total_expenses := 10 + 5 + 3
  let price_per_cup := 4
  let cups_sold := 21
  let total_revenue := price_per_cup * cups_sold
  total_revenue - total_expenses

theorem alexa_emily_profit : lemonade_stand_profit = 66 :=
  by
  sorry

end alexa_emily_profit_l214_214715


namespace ratio_amyl_alcohol_to_ethanol_l214_214238

noncomputable def mol_amyl_alcohol : ℕ := 3
noncomputable def mol_hcl : ℕ := 3
noncomputable def mol_ethanol : ℕ := 1
noncomputable def mol_h2so4 : ℕ := 1
noncomputable def mol_ch3_cl2_c5_h9 : ℕ := 3
noncomputable def mol_h2o : ℕ := 3
noncomputable def mol_ethyl_dimethylpropyl_sulfate : ℕ := 1

theorem ratio_amyl_alcohol_to_ethanol : 
  (mol_amyl_alcohol / mol_ethanol = 3) :=
by 
  have h1 : mol_amyl_alcohol = 3 := by rfl
  have h2 : mol_ethanol = 1 := by rfl
  sorry

end ratio_amyl_alcohol_to_ethanol_l214_214238


namespace other_root_l214_214516

theorem other_root (x : ℚ) (h : 48 * x^2 + 29 = 35 * x + 12) : x = 3 / 4 ∨ x = 1 / 3 := 
by {
  -- Proof can be filled in here
  sorry
}

end other_root_l214_214516


namespace prime_in_A_l214_214318

def A (n : ℕ) : Prop :=
  ∃ a b : ℤ, b ≠ 0 ∧ n = a^2 + 2 * b^2

theorem prime_in_A {p : ℕ} (h_prime : Nat.Prime p) (h_p2_in_A : A (p^2)) : A p :=
sorry

end prime_in_A_l214_214318


namespace eliot_account_balance_l214_214408

variable (A E : ℝ)

-- Condition 1: Al has more money than Eliot.
axiom h1 : A > E

-- Condition 2: The difference between their accounts is 1/12 of the sum of their accounts.
axiom h2 : A - E = (1 / 12) * (A + E)

-- Condition 3: If Al's account were increased by 10% and Eliot's by 20%, Al would have exactly $21 more than Eliot.
axiom h3 : 1.1 * A = 1.2 * E + 21

-- Conjecture: Eliot has $210 in his account.
theorem eliot_account_balance : E = 210 :=
by
  sorry

end eliot_account_balance_l214_214408


namespace max_A_plus_B_l214_214874

theorem max_A_plus_B:
  ∃ A B C D : ℕ,
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  A + B + C + D = 17 ∧ ∃ k : ℕ, C + D ≠ 0 ∧ A + B = k * (C + D) ∧
  A + B = 16 :=
by sorry

end max_A_plus_B_l214_214874


namespace fourth_derivative_at_0_l214_214800

noncomputable def f : ℝ → ℝ := sorry

axiom f_at_0 : f 0 = 1
axiom f_prime_at_0 : deriv f 0 = 2
axiom f_double_prime : ∀ t, deriv (deriv f) t = 4 * deriv f t - 3 * f t + 1

-- We want to prove that the fourth derivative of f at 0 equals 54
theorem fourth_derivative_at_0 : deriv (deriv (deriv (deriv f))) 0 = 54 :=
sorry

end fourth_derivative_at_0_l214_214800


namespace trig_identity_l214_214253

open Real

theorem trig_identity (α β : ℝ) (h : cos α * cos β - sin α * sin β = 0) : sin α * cos β + cos α * sin β = 1 ∨ sin α * cos β + cos α * sin β = -1 :=
by
  sorry

end trig_identity_l214_214253


namespace tan_of_angle_in_fourth_quadrant_l214_214020

-- Define the angle α in the fourth quadrant in terms of its cosine value
variable (α : Real)
variable (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) -- fourth quadrant condition
variable (h2 : Real.cos α = 4/5) -- given condition

-- Define the proof problem that tan α equals -3/4 given the conditions
theorem tan_of_angle_in_fourth_quadrant (α : Real) (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) (h2 : Real.cos α = 4/5) : 
  Real.tan α = -3/4 :=
sorry

end tan_of_angle_in_fourth_quadrant_l214_214020


namespace arrangements_with_AB_together_l214_214660

theorem arrangements_with_AB_together (n : ℕ) (A B: ℕ) (students: Finset ℕ) (h₁ : students.card = 6) (h₂ : A ∈ students) (h₃ : B ∈ students):
  ∃! (count : ℕ), count = 240 :=
by
  sorry

end arrangements_with_AB_together_l214_214660


namespace special_numbers_count_l214_214863

-- Define conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_zero (n : ℕ) : Prop := n % 10 = 0
def divisible_by_30 (n : ℕ) : Prop := n % 30 = 0

-- Define the count of numbers with the specified conditions
noncomputable def count_special_numbers : ℕ :=
  (9990 - 1020) / 30 + 1

-- The proof problem
theorem special_numbers_count : count_special_numbers = 300 := sorry

end special_numbers_count_l214_214863


namespace sufficient_prime_logarithms_l214_214677

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

-- Statement of the properties of logarithms
axiom log_mul (b x y : ℝ) : log_b b (x * y) = log_b b x + log_b b y
axiom log_div (b x y : ℝ) : log_b b (x / y) = log_b b x - log_b b y
axiom log_pow (b x : ℝ) (n : ℝ) : log_b b (x ^ n) = n * log_b b x

-- Main theorem
theorem sufficient_prime_logarithms (b : ℝ) (hb : 1 < b) :
  (∀ p : ℕ, is_prime p → ∃ Lp : ℝ, log_b b p = Lp) →
  ∀ n : ℕ, n > 0 → ∃ Ln : ℝ, log_b b n = Ln :=
by
  sorry

end sufficient_prime_logarithms_l214_214677


namespace ShepherdProblem_l214_214533

theorem ShepherdProblem (x y : ℕ) :
  (x + 9 = 2 * (y - 9) ∧ y + 9 = x - 9) ↔
  ((x + 9 = 2 * (y - 9)) ∧ (y + 9 = x - 9)) :=
by
  sorry

end ShepherdProblem_l214_214533


namespace frame_dimension_ratio_l214_214466

theorem frame_dimension_ratio (W H x : ℕ) (h1 : W = 20) (h2 : H = 30) (h3 : 2 * (W + 2 * x) * (H + 6 * x) - W * H = 2 * (W * H)) :
  (W + 2 * x) / (H + 6 * x) = 1/2 :=
by sorry

end frame_dimension_ratio_l214_214466


namespace value_of_expression_l214_214049

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end value_of_expression_l214_214049


namespace geom_seq_inequality_l214_214472

-- Define S_n as a.sum of the first n terms of a geometric sequence with ratio q and first term a_1
noncomputable def S (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
if h : q = 1 then (n + 1) * a_1 else a_1 * (1 - q ^ (n + 1)) / (1 - q)

-- Define a_n for geometric sequence
noncomputable def a_seq (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
a_1 * q ^ n

-- The main theorem to prove
theorem geom_seq_inequality (a_1 : ℝ) (q : ℝ) (n : ℕ) (hq_pos : 0 < q) :
  S a_1 q (n + 1) * a_seq a_1 q n > S a_1 q n * a_seq a_1 q (n + 1) :=
by {
  sorry -- Placeholder for actual proof
}

end geom_seq_inequality_l214_214472


namespace vertex_on_x_axis_l214_214351

theorem vertex_on_x_axis (m : ℝ) : 
  (∃ x : ℝ, x^2 - 8 * x + m = 0) ↔ m = 16 :=
by
  sorry

end vertex_on_x_axis_l214_214351


namespace sum_of_numbers_l214_214420

theorem sum_of_numbers (x : ℝ) (h1 : x^2 + (2 * x)^2 + (4 * x)^2 = 1701) : x + 2 * x + 4 * x = 63 :=
sorry

end sum_of_numbers_l214_214420


namespace min_value_x1_squared_plus_x2_squared_plus_x3_squared_l214_214508

theorem min_value_x1_squared_plus_x2_squared_plus_x3_squared
    (x1 x2 x3 : ℝ) 
    (h1 : 3 * x1 + 2 * x2 + x3 = 30) 
    (h2 : x1 > 0) 
    (h3 : x2 > 0) 
    (h4 : x3 > 0) : 
    x1^2 + x2^2 + x3^2 ≥ 125 := 
  by sorry

end min_value_x1_squared_plus_x2_squared_plus_x3_squared_l214_214508


namespace sum_of_coordinates_of_D_is_12_l214_214320

theorem sum_of_coordinates_of_D_is_12 :
  (exists (x y : ℝ), (5 = (11 + x) / 2) ∧ (9 = (5 + y) / 2) ∧ (x + y = 12)) :=
by
  sorry

end sum_of_coordinates_of_D_is_12_l214_214320


namespace minimum_value_l214_214647

open Real

variables {A B C M : Type}
variables (AB AC : ℝ) 
variables (S_MBC x y : ℝ)

-- Assume the given conditions
axiom dot_product_AB_AC : AB * AC = 2 * sqrt 3
axiom angle_BAC_30 : (30 : Real) = π / 6
axiom area_MBC : S_MBC = 1/2
axiom area_sum : x + y = 1/2

-- Define the minimum value problem
theorem minimum_value : 
  ∃ m, m = 18 ∧ (∀ x y, (1/x + 4/y) ≥ m) :=
sorry

end minimum_value_l214_214647


namespace remainder_of_13_pow_a_mod_37_l214_214228

theorem remainder_of_13_pow_a_mod_37 (a : ℕ) (h_pos : a > 0) (h_mult : ∃ k : ℕ, a = 3 * k) : (13^a) % 37 = 1 := 
sorry

end remainder_of_13_pow_a_mod_37_l214_214228


namespace base6_sum_l214_214388

-- Define each of the numbers in base 6
def base6_555 : ℕ := 5 * 6^2 + 5 * 6^1 + 5 * 6^0
def base6_55 : ℕ := 5 * 6^1 + 5 * 6^0
def base6_5 : ℕ := 5 * 6^0
def base6_1103 : ℕ := 1 * 6^3 + 1 * 6^2 + 0 * 6^1 + 3 * 6^0 

-- The problem statement is to prove the sum equals the expected result in base 6
theorem base6_sum : base6_555 + base6_55 + base6_5 = base6_1103 :=
by
  sorry

end base6_sum_l214_214388


namespace sum_due_l214_214857

theorem sum_due (BD TD S : ℝ) (hBD : BD = 18) (hTD : TD = 15) (hRel : BD = TD + (TD^2 / S)) : S = 75 :=
by
  sorry

end sum_due_l214_214857


namespace least_perimeter_of_triangle_l214_214183

theorem least_perimeter_of_triangle (a b : ℕ) (a_eq : a = 33) (b_eq : b = 42) (c : ℕ) (h1 : c + a > b) (h2 : c + b > a) (h3 : a + b > c) : a + b + c = 85 :=
sorry

end least_perimeter_of_triangle_l214_214183


namespace JessicaPathsAvoidRiskySite_l214_214229

-- Definitions for the conditions.
def West (x y : ℕ) : Prop := (x > 0)
def East (x y : ℕ) : Prop := (x < 4)
def North (x y : ℕ) : Prop := (y < 3)
def AtOrigin (x y : ℕ) : Prop := (x = 0 ∧ y = 0)
def AtAnna (x y : ℕ) : Prop := (x = 4 ∧ y = 3)
def RiskySite (x y : ℕ) : Prop := (x = 2 ∧ y = 1)

-- Function to calculate binomial coefficient, binom(n, k)
def binom : ℕ → ℕ → ℕ
  | n, 0 => 1
  | 0, k + 1 => 0
  | n + 1, k + 1 => binom n k + binom n (k + 1)

-- Number of total valid paths avoiding the risky site.
theorem JessicaPathsAvoidRiskySite :
  let totalPaths := binom 7 4
  let pathsThroughRisky := binom 3 2 * binom 4 2
  (totalPaths - pathsThroughRisky) = 17 :=
by
  sorry

end JessicaPathsAvoidRiskySite_l214_214229


namespace sum_of_three_exists_l214_214386

theorem sum_of_three_exists (n : ℤ) (X : Finset ℤ) 
  (hX_card : X.card = n + 2) 
  (hX_abs : ∀ x ∈ X, abs x ≤ n) : 
  ∃ a b c : ℤ, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ c = a + b := 
by 
  sorry

end sum_of_three_exists_l214_214386


namespace sum_and_times_l214_214961

theorem sum_and_times 
  (a : ℕ) (ha : a = 99) 
  (b : ℕ) (hb : b = 301) 
  (c : ℕ) (hc : c = 200) : 
  a + b = 2 * c :=
by 
  -- skipping proof 
  sorry

end sum_and_times_l214_214961


namespace at_least_one_not_less_than_two_l214_214849

open Real

theorem at_least_one_not_less_than_two (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  (a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) → false := 
sorry

end at_least_one_not_less_than_two_l214_214849


namespace swimmers_speed_in_still_water_l214_214637

theorem swimmers_speed_in_still_water
  (v : ℝ) -- swimmer's speed in still water
  (current_speed : ℝ) -- speed of the water current
  (time : ℝ) -- time taken to swim against the current
  (distance : ℝ) -- distance swum against the current
  (h_current_speed : current_speed = 2)
  (h_time : time = 3.5)
  (h_distance : distance = 7)
  (h_eqn : time = distance / (v - current_speed)) :
  v = 4 :=
by
  sorry

end swimmers_speed_in_still_water_l214_214637


namespace triangle_base_length_l214_214880

/-
Theorem: Given a triangle with height 5.8 meters and area 24.36 square meters,
the length of the base is 8.4 meters.
-/

theorem triangle_base_length (h : ℝ) (A : ℝ) (b : ℝ) :
  h = 5.8 ∧ A = 24.36 ∧ A = (b * h) / 2 → b = 8.4 :=
by
  sorry

end triangle_base_length_l214_214880


namespace seashells_total_now_l214_214555

def henry_collected : ℕ := 11
def paul_collected : ℕ := 24
def total_initial : ℕ := 59
def leo_initial (henry_collected paul_collected total_initial : ℕ) : ℕ := total_initial - (henry_collected + paul_collected)
def leo_gave (leo_initial : ℕ) : ℕ := leo_initial / 4
def total_now (total_initial leo_gave : ℕ) : ℕ := total_initial - leo_gave

theorem seashells_total_now :
  total_now total_initial (leo_gave (leo_initial henry_collected paul_collected total_initial)) = 53 :=
sorry

end seashells_total_now_l214_214555


namespace sqrt_ab_eq_18_l214_214305

noncomputable def a := Real.log 9 / Real.log 4
noncomputable def b := 108 * (Real.log 8 / Real.log 3)

theorem sqrt_ab_eq_18 : Real.sqrt (a * b) = 18 := by
  sorry

end sqrt_ab_eq_18_l214_214305


namespace find_m_l214_214971

-- Define vectors as tuples
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, -1)
def c (m : ℝ) : ℝ × ℝ := (4, m)

-- Define vector subtraction
def sub_vect (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Define dot product
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove the condition that (a - b) ⊥ c implies m = 4
theorem find_m (m : ℝ) (h : dot_prod (sub_vect a (b m)) (c m) = 0) : m = 4 :=
by
  sorry

end find_m_l214_214971


namespace even_function_behavior_l214_214219

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def condition (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 ≠ x2 → (x2 - x1) * (f x2 - f x1) > 0

theorem even_function_behavior (f : ℝ → ℝ) (h_even : is_even_function f) (h_condition : condition f) 
  (n : ℕ) (h_n : n > 0) : 
  f (n+1) < f (-n) ∧ f (-n) < f (n-1) :=
sorry

end even_function_behavior_l214_214219


namespace valid_combinations_l214_214612

def herbs : Nat := 4
def crystals : Nat := 6
def incompatible_pairs : Nat := 3

theorem valid_combinations : 
  (herbs * crystals) - incompatible_pairs = 21 := by
  sorry

end valid_combinations_l214_214612


namespace simplify_expression_l214_214517

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (x⁻¹ - x + 2) = (1 - (x - 1)^2) / x := 
sorry

end simplify_expression_l214_214517


namespace smallest_nonfactor_product_of_48_l214_214159

noncomputable def is_factor_of (a b : ℕ) : Prop :=
  b % a = 0

theorem smallest_nonfactor_product_of_48
  (m n : ℕ)
  (h1 : m ≠ n)
  (h2 : is_factor_of m 48)
  (h3 : is_factor_of n 48)
  (h4 : ¬is_factor_of (m * n) 48) :
  m * n = 18 :=
sorry

end smallest_nonfactor_product_of_48_l214_214159


namespace solution_set_inequality_l214_214217

noncomputable def solution_set := {x : ℝ | (x + 1) * (x - 2) ≤ 0 ∧ x ≠ -1}

theorem solution_set_inequality :
  solution_set = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by {
-- Insert proof here
sorry
}

end solution_set_inequality_l214_214217


namespace sector_area_half_triangle_area_l214_214094

theorem sector_area_half_triangle_area (θ : Real) (r : Real) (hθ1 : 0 < θ) (hθ2 : θ < π / 3) :
    2 * θ = Real.tan θ := by
  sorry

end sector_area_half_triangle_area_l214_214094


namespace unique_prime_with_conditions_l214_214867

theorem unique_prime_with_conditions (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (p + 2)) (hp4 : Nat.Prime (p + 4)) : p = 3 :=
by
  sorry

end unique_prime_with_conditions_l214_214867


namespace initial_percentage_of_water_is_12_l214_214922

noncomputable def initial_percentage_of_water (initial_volume : ℕ) (added_water : ℕ) (final_percentage : ℕ) : ℕ :=
  let final_volume := initial_volume + added_water
  let final_water_amount := (final_percentage * final_volume) / 100
  let initial_water_amount := final_water_amount - added_water
  (initial_water_amount * 100) / initial_volume

theorem initial_percentage_of_water_is_12 :
  initial_percentage_of_water 20 2 20 = 12 :=
by
  sorry

end initial_percentage_of_water_is_12_l214_214922


namespace comic_stack_ways_l214_214488

-- Define the factorial function for convenience
noncomputable def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Conditions: Define the number of each type of comic book
def batman_comics := 7
def superman_comics := 4
def wonder_woman_comics := 5
def flash_comics := 3

-- The total number of comic books
def total_comics := batman_comics + superman_comics + wonder_woman_comics + flash_comics

-- Proof problem: The number of ways to stack the comics
theorem comic_stack_ways :
  (factorial batman_comics) * (factorial superman_comics) * (factorial wonder_woman_comics) * (factorial flash_comics) * (factorial 4) = 1102489600 := sorry

end comic_stack_ways_l214_214488


namespace cost_per_box_l214_214544

theorem cost_per_box (trays : ℕ) (cookies_per_tray : ℕ) (cookies_per_box : ℕ) (total_cost : ℕ) (box_cost : ℝ) 
  (h1 : trays = 3) 
  (h2 : cookies_per_tray = 80) 
  (h3 : cookies_per_box = 60)
  (h4 : total_cost = 14) 
  (h5 : (trays * cookies_per_tray) = 240)
  (h6 : (240 / cookies_per_box : ℕ) = 4) 
  (h7 : (total_cost / 4 : ℝ) = box_cost) : 
  box_cost = 3.5 := 
by sorry

end cost_per_box_l214_214544


namespace nancy_water_intake_l214_214566

theorem nancy_water_intake (water_intake body_weight : ℝ) (h1 : water_intake = 54) (h2 : body_weight = 90) : 
  (water_intake / body_weight) * 100 = 60 :=
by
  -- using the conditions h1 and h2
  rw [h1, h2]
  -- skipping the proof
  sorry

end nancy_water_intake_l214_214566


namespace santa_chocolate_candies_l214_214030

theorem santa_chocolate_candies (C M : ℕ) (h₁ : C + M = 2023) (h₂ : C = 3 * M / 4) : C = 867 :=
sorry

end santa_chocolate_candies_l214_214030


namespace cryptarithm_solutions_unique_l214_214367

/- Definitions corresponding to the conditions -/
def is_valid_digit (d : Nat) : Prop := d < 10

def is_six_digit_number (n : Nat) : Prop := n >= 100000 ∧ n < 1000000

def matches_cryptarithm (abcdef bcdefa : Nat) : Prop := abcdef * 3 = bcdefa

/- Prove that the two identified solutions are valid and no other solutions exist -/
theorem cryptarithm_solutions_unique :
  ∀ (A B C D E F : Nat),
  is_valid_digit A → is_valid_digit B → is_valid_digit C →
  is_valid_digit D → is_valid_digit E → is_valid_digit F →
  let abcdef := 100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F
  let bcdefa := 100000 * B + 10000 * C + 1000 * D + 100 * E + 10 * F + A
  is_six_digit_number abcdef →
  is_six_digit_number bcdefa →
  matches_cryptarithm abcdef bcdefa →
  (abcdef = 142857 ∨ abcdef = 285714) :=
by
  intros A B C D E F A_valid B_valid C_valid D_valid E_valid F_valid abcdef bcdefa abcdef_six_digit bcdefa_six_digit cryptarithm_match
  sorry

end cryptarithm_solutions_unique_l214_214367


namespace least_number_subtracted_l214_214232

theorem least_number_subtracted (n : ℕ) (x : ℕ) (h_n : n = 4273981567) (h_x : x = 17) : 
  (n - x) % 25 = 0 := by
  sorry

end least_number_subtracted_l214_214232


namespace max_candies_takeable_l214_214944

theorem max_candies_takeable : 
  ∃ (max_take : ℕ), max_take = 159 ∧
  ∀ (boxes: Fin 5 → ℕ), 
    boxes 0 = 11 → 
    boxes 1 = 22 → 
    boxes 2 = 33 → 
    boxes 3 = 44 → 
    boxes 4 = 55 →
    (∀ (i : Fin 5), 
      ∀ (new_boxes : Fin 5 → ℕ),
      (new_boxes i = boxes i - 4) ∧ 
      (∀ (j : Fin 5), j ≠ i → new_boxes j = boxes j + 1) →
      boxes i = 0 → max_take = new_boxes i) :=
sorry

end max_candies_takeable_l214_214944


namespace number_of_incorrect_inequalities_l214_214414

theorem number_of_incorrect_inequalities (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  (ite (|a| > |b|) 0 1) + (ite (a < b) 0 1) + (ite (a + b < ab) 0 1) + (ite (a^3 > b^3) 0 1) = 3 :=
sorry

end number_of_incorrect_inequalities_l214_214414


namespace max_grapes_in_bag_l214_214156

theorem max_grapes_in_bag : ∃ (x : ℕ), x > 100 ∧ x % 3 = 1 ∧ x % 5 = 2 ∧ x % 7 = 4 ∧ x = 172 := by
  sorry

end max_grapes_in_bag_l214_214156


namespace emma_average_speed_l214_214297

-- Define the given conditions
def distance1 : ℕ := 420     -- Distance traveled in the first segment
def time1 : ℕ := 7          -- Time taken in the first segment
def distance2 : ℕ := 480    -- Distance traveled in the second segment
def time2 : ℕ := 8          -- Time taken in the second segment

-- Define the total distance and total time
def total_distance : ℕ := distance1 + distance2
def total_time : ℕ := time1 + time2

-- Define the expected average speed
def expected_average_speed : ℕ := 60

-- Prove that the average speed is 60 miles per hour
theorem emma_average_speed : (total_distance / total_time) = expected_average_speed := by
  sorry

end emma_average_speed_l214_214297


namespace twice_total_credits_l214_214666

-- Define the variables and conditions
variables (Aria Emily Spencer Hannah : ℕ)
variables (h1 : Aria = 2 * Emily) 
variables (h2 : Emily = 2 * Spencer)
variables (h3 : Emily = 20)
variables (h4 : Hannah = 3 * Spencer)

-- Proof statement
theorem twice_total_credits : 2 * (Aria + Emily + Spencer + Hannah) = 200 :=
by 
  -- Proof steps are omitted with sorry
  sorry

end twice_total_credits_l214_214666


namespace box_breadth_l214_214107

noncomputable def cm_to_m (cm : ℕ) : ℝ := cm / 100

theorem box_breadth :
  ∀ (length depth cm cubical_edge blocks : ℕ), 
    length = 160 →
    depth = 60 →
    cubical_edge = 20 →
    blocks = 120 →
    breadth = (blocks * (cubical_edge ^ 3)) / (length * depth) →
    breadth = 100 :=
by
  sorry

end box_breadth_l214_214107


namespace total_amount_correct_l214_214840

/-- Meghan has the following cash denominations: -/
def num_100_bills : ℕ := 2
def num_50_bills : ℕ := 5
def num_10_bills : ℕ := 10

/-- Value of each denomination: -/
def value_100_bill : ℕ := 100
def value_50_bill : ℕ := 50
def value_10_bill : ℕ := 10

/-- Meghan's total amount of money: -/
def total_amount : ℕ :=
  (num_100_bills * value_100_bill) +
  (num_50_bills * value_50_bill) +
  (num_10_bills * value_10_bill)

/-- The proof: -/
theorem total_amount_correct : total_amount = 550 :=
by
  -- sorry for now
  sorry

end total_amount_correct_l214_214840


namespace part1_l214_214801

theorem part1 (a : ℤ) (h : a = -2) : 
  ((a^2 + a) / (a^2 - 3 * a)) / ((a^2 - 1) / (a - 3)) - 1 / (a + 1) = 2 / 3 := by
  rw [h]
  sorry

end part1_l214_214801


namespace quadratic_solutions_l214_214362

theorem quadratic_solutions (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end quadratic_solutions_l214_214362


namespace rectangle_sides_l214_214975

theorem rectangle_sides :
  ∀ (x : ℝ), 
    (3 * x = 8) ∧ (8 / 3 * 3 = 8) →
    ((2 * (3 * x + x) = 3 * x^2) ∧ (2 * (3 * (8 / 3) + (8 / 3)) = 3 * (8 / 3)^2) →
    x = 8 / 3
      ∧ 3 * x = 8) := 
by
  sorry

end rectangle_sides_l214_214975


namespace single_colony_reaches_limit_in_24_days_l214_214895

/-- A bacteria colony doubles in size every day. -/
def double (n : ℕ) : ℕ := 2 ^ n

/-- Two bacteria colonies growing simultaneously will take 24 days to reach the habitat's limit. -/
axiom two_colonies_24_days : ∀ k : ℕ, double k + double k = double 24

/-- Prove that it takes 24 days for a single bacteria colony to reach the habitat's limit. -/
theorem single_colony_reaches_limit_in_24_days : ∃ x : ℕ, double x = double 24 :=
sorry

end single_colony_reaches_limit_in_24_days_l214_214895


namespace sum_of_digits_of_a_l214_214731

-- Define a as 10^10 - 47
def a : ℕ := (10 ^ 10) - 47

-- Function to compute the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- The theorem to prove that the sum of all the digits of a is 81
theorem sum_of_digits_of_a : sum_of_digits a = 81 := by
  sorry

end sum_of_digits_of_a_l214_214731


namespace find_simple_annual_rate_l214_214548

-- Conditions from part a).
-- 1. Principal initial amount (P) is $5,000.
-- 2. Annual interest rate for compounded interest (r) is 0.06.
-- 3. Number of times it compounds per year (n) is 2 (semi-annually).
-- 4. Time period (t) is 1 year.
-- 5. The interest earned after one year for simple interest is $6 less than compound interest.

noncomputable def principal : ℝ := 5000
noncomputable def annual_rate_compound : ℝ := 0.06
noncomputable def times_compounded : ℕ := 2
noncomputable def time_years : ℝ := 1
noncomputable def compound_interest : ℝ := principal * (1 + annual_rate_compound / times_compounded) ^ (times_compounded * time_years) - principal
noncomputable def simple_interest : ℝ := compound_interest - 6

-- Question from part a) translated to Lean statement using the condition that simple interest satisfaction
theorem find_simple_annual_rate : 
    ∃ r : ℝ, principal * r * time_years = simple_interest :=
by
  exists (0.0597)
  sorry

end find_simple_annual_rate_l214_214548


namespace abs_a_lt_abs_b_add_abs_c_l214_214726

theorem abs_a_lt_abs_b_add_abs_c (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end abs_a_lt_abs_b_add_abs_c_l214_214726


namespace triangle_ratio_condition_l214_214852

theorem triangle_ratio_condition (a b c : ℝ) (A B C : ℝ) (h1 : b * Real.cos C + c * Real.cos B = 2 * b)
  (h2 : a = b * Real.sin A / Real.sin B)
  (h3 : b = a * Real.sin B / Real.sin A)
  (h4 : c = a * Real.sin C / Real.sin A)
  (h5 : ∀ x, Real.sin (B + C) = Real.sin x): 
  b / a = 1 / 2 :=
by
  sorry

end triangle_ratio_condition_l214_214852


namespace hyperbola_problem_l214_214395

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def eccentricity (a c : ℝ) : Prop :=
  c / a = 2 * Real.sqrt 3 / 3

def focal_distance (c a : ℝ) : Prop :=
  2 * a^2 = 3 * c

def point_on_hyperbola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  hyperbola a b P.1 P.2

def point_satisfies_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 2

noncomputable def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

theorem hyperbola_problem (a b c : ℝ) (P F1 F2 : ℝ × ℝ) :
  (a > 0 ∧ b > 0) →
  eccentricity a c →
  focal_distance c a →
  point_on_hyperbola P a b →
  point_satisfies_condition P F1 F2 →
  distance F1 F2 = 2 * c →
  (distance P F1) * (distance P F2) = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hyperbola_problem_l214_214395


namespace triangular_prism_skew_pair_count_l214_214363

-- Definition of a triangular prism with 6 vertices and 15 lines through any two vertices
structure TriangularPrism :=
  (vertices : Fin 6)   -- 6 vertices
  (lines : Fin 15)     -- 15 lines through any two vertices

-- A function to check if two lines are skew lines 
-- (not intersecting and not parallel in three-dimensional space)
def is_skew (line1 line2 : Fin 15) : Prop := sorry

-- Function to count pairs of lines that are skew in a triangular prism
def count_skew_pairs (prism : TriangularPrism) : Nat := sorry

-- Theorem stating the number of skew pairs in a triangular prism is 36
theorem triangular_prism_skew_pair_count (prism : TriangularPrism) :
  count_skew_pairs prism = 36 := 
sorry

end triangular_prism_skew_pair_count_l214_214363


namespace cost_price_of_article_l214_214014

theorem cost_price_of_article (x : ℝ) (h : 66 - x = x - 22) : x = 44 :=
sorry

end cost_price_of_article_l214_214014


namespace solution_set_of_inequality_af_neg2x_pos_l214_214860

-- Given conditions:
-- f(x) = x^2 + ax + b has roots -1 and 2
-- We need to prove that the solution set for af(-2x) > 0 is -1 < x < 1/2
theorem solution_set_of_inequality_af_neg2x_pos (a b : ℝ) (x : ℝ) 
  (h1 : -1 + 2 = -a) 
  (h2 : -1 * 2 = b) : 
  (a * ((-2 * x)^2 + a * (-2 * x) + b) > 0) = (-1 < x ∧ x < 1/2) :=
by
  sorry

end solution_set_of_inequality_af_neg2x_pos_l214_214860


namespace parking_lot_motorcycles_l214_214471

theorem parking_lot_motorcycles
  (x y : ℕ)
  (h1 : x + y = 24)
  (h2 : 3 * x + 4 * y = 86) : x = 10 :=
by
  sorry

end parking_lot_motorcycles_l214_214471


namespace specified_time_eq_l214_214203

def distance : ℕ := 900
def ts (x : ℕ) : ℕ := x + 1
def tf (x : ℕ) : ℕ := x - 3

theorem specified_time_eq (x : ℕ) (h1 : x > 3) : 
  (distance / tf x) = 2 * (distance / ts x) :=
sorry

end specified_time_eq_l214_214203


namespace cars_meet_after_40_minutes_l214_214881

noncomputable def time_to_meet 
  (BC CD : ℝ) (speed : ℝ) 
  (constant_speed : ∀ t, t > 0 → speed = (BC + CD) / t) : ℝ :=
  (BC + CD) / speed * 40 / 60

-- Define the condition that must hold: cars meet at 40 minutes
theorem cars_meet_after_40_minutes
  (BC CD : ℝ) (speed : ℝ)
  (constant_speed : ∀ t, t > 0 → speed = (BC + CD) / t) :
  time_to_meet BC CD speed constant_speed = 40 := sorry

end cars_meet_after_40_minutes_l214_214881


namespace two_integer_solutions_iff_m_l214_214572

def op (p q : ℝ) : ℝ := p + q - p * q

theorem two_integer_solutions_iff_m (m : ℝ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ op 2 x1 > 0 ∧ op x1 3 ≤ m ∧ op 2 x2 > 0 ∧ op x2 3 ≤ m) ↔ 3 ≤ m ∧ m < 5 :=
by
  sorry

end two_integer_solutions_iff_m_l214_214572


namespace f_three_l214_214939

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_f_succ : ∀ x : ℝ, f (x + 1) = f (-x + 1)
axiom f_one : f 1 = 1 

-- Goal
theorem f_three : f 3 = -1 :=
by
  -- The proof will be provided here
  sorry

end f_three_l214_214939


namespace cost_of_bananas_l214_214819

-- Definitions of the conditions from the problem
namespace BananasCost

variables (A B : ℝ)

-- Condition equations
def condition1 : Prop := 2 * A + B = 7
def condition2 : Prop := A + B = 5

-- The theorem to prove the cost of a bunch of bananas
theorem cost_of_bananas (h1 : condition1 A B) (h2 : condition2 A B) : B = 3 := 
  sorry

end BananasCost

end cost_of_bananas_l214_214819


namespace incorrect_statement_D_l214_214269

theorem incorrect_statement_D 
  (population : Set ℕ)
  (time_spent_sample : ℕ → ℕ)
  (sample_size : ℕ)
  (individual : ℕ)
  (h1 : ∀ s, s ∈ population → s ≤ 24)
  (h2 : ∀ i, i < sample_size → population (time_spent_sample i))
  (h3 : sample_size = 300)
  (h4 : ∀ i, i < 300 → time_spent_sample i = individual):
  ¬ (∀ i, i < 300 → time_spent_sample i = individual) :=
sorry

end incorrect_statement_D_l214_214269


namespace proof_problem_l214_214290

noncomputable def polar_to_cartesian_O1 : Prop :=
  ∀ (ρ : ℝ) (θ : ℝ), ρ = 4 * Real.cos θ → (ρ^2 = 4 * ρ * Real.cos θ)

noncomputable def cartesian_O1 : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 = 4 * x → x^2 + y^2 - 4 * x = 0

noncomputable def polar_to_cartesian_O2 : Prop :=
  ∀ (ρ : ℝ) (θ : ℝ), ρ = -4 * Real.sin θ → (ρ^2 = -4 * ρ * Real.sin θ)

noncomputable def cartesian_O2 : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 = -4 * y → x^2 + y^2 + 4 * y = 0

noncomputable def intersections_O1_O2 : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x = 0) ∧ (x^2 + y^2 + 4 * y = 0) →
  (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -2)

noncomputable def line_through_intersections : Prop :=
  ∀ (x y : ℝ), ((x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -2)) → y = -x

theorem proof_problem : polar_to_cartesian_O1 ∧ cartesian_O1 ∧ polar_to_cartesian_O2 ∧ cartesian_O2 ∧ intersections_O1_O2 ∧ line_through_intersections :=
  sorry

end proof_problem_l214_214290


namespace product_of_real_solutions_l214_214275

theorem product_of_real_solutions :
  (∀ x : ℝ, (x + 1) / (3 * x + 3) = (3 * x + 2) / (8 * x + 2)) →
  x = -1 ∨ x = -4 →
  (-1) * (-4) = 4 := 
sorry

end product_of_real_solutions_l214_214275


namespace boarders_initial_count_l214_214650

noncomputable def initial_boarders (x : ℕ) : ℕ := 7 * x

theorem boarders_initial_count (x : ℕ) (h1 : 80 + initial_boarders x = (2 : ℝ) * 16) :
  initial_boarders x = 560 :=
by
  sorry

end boarders_initial_count_l214_214650


namespace first_night_percentage_is_20_l214_214687

-- Conditions
variable (total_pages : ℕ) (pages_left : ℕ)
variable (pages_second_night : ℕ)
variable (pages_third_night : ℕ)
variable (first_night_percentage : ℕ)

-- Definitions
def total_read_pages (total_pages pages_left : ℕ) : ℕ := total_pages - pages_left

def pages_first_night (total_pages first_night_percentage : ℕ) : ℕ :=
  (first_night_percentage * total_pages) / 100

def total_read_on_three_nights (total_pages pages_left pages_second_night pages_third_night first_night_percentage : ℕ) : Prop :=
  total_read_pages total_pages pages_left = pages_first_night total_pages first_night_percentage + pages_second_night + pages_third_night

-- Theorem
theorem first_night_percentage_is_20 :
  ∀ total_pages pages_left pages_second_night pages_third_night,
  total_pages = 500 →
  pages_left = 150 →
  pages_second_night = 100 →
  pages_third_night = 150 →
  total_read_on_three_nights total_pages pages_left pages_second_night pages_third_night 20 :=
by
  intros
  sorry

end first_night_percentage_is_20_l214_214687


namespace social_gathering_married_men_fraction_l214_214093

theorem social_gathering_married_men_fraction {W : ℝ} {MW : ℝ} {MM : ℝ} 
  (hW_pos : 0 < W)
  (hMW_def : MW = W * (3/7))
  (hMM_def : MM = W - MW)
  (h_total_people : 2 * MM + MW = 11) :
  (MM / 11) = 4/11 :=
by {
  sorry
}

end social_gathering_married_men_fraction_l214_214093


namespace car_price_l214_214091

/-- Prove that the price of the car Quincy bought is $20,000 given the conditions. -/
theorem car_price (years : ℕ) (monthly_payment : ℕ) (down_payment : ℕ) 
  (h1 : years = 5) 
  (h2 : monthly_payment = 250) 
  (h3 : down_payment = 5000) : 
  (down_payment + (monthly_payment * (12 * years))) = 20000 :=
by
  /- We provide the proof below with sorry because we are only writing the statement as requested. -/
  sorry

end car_price_l214_214091


namespace Nedy_crackers_total_l214_214708

theorem Nedy_crackers_total :
  let packs_from_Mon_to_Thu := 8
  let packs_on_Fri := 2 * packs_from_Mon_to_Thu
  (packs_from_Mon_to_Thu + packs_on_Fri) = 24 :=
by
  let packs_from_Mon_to_Thu := 8
  let packs_on_Fri := 2 * packs_from_Mon_to_Thu
  show packs_from_Mon_to_Thu + packs_on_Fri = 24
  sorry

end Nedy_crackers_total_l214_214708


namespace integer_quotient_is_perfect_square_l214_214702

theorem integer_quotient_is_perfect_square (a b : ℕ) (h : 0 < a ∧ 0 < b) (h_int : (a + b) ^ 2 % (4 * a * b + 1) = 0) :
  ∃ k : ℕ, (a + b) ^ 2 = k ^ 2 * (4 * a * b + 1) := sorry

end integer_quotient_is_perfect_square_l214_214702


namespace min_value_of_n_l214_214534

def is_prime (p : ℕ) : Prop := p ≥ 2 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def is_not_prime (n : ℕ) : Prop := ¬ is_prime n

def decomposable_into_primes_leq_10 (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≤ 10 ∧ q ≤ 10 ∧ n = p * q

theorem min_value_of_n : ∃ n : ℕ, is_not_prime n ∧ decomposable_into_primes_leq_10 n ∧ n = 6 :=
by
  -- The proof would go here.
  sorry

end min_value_of_n_l214_214534


namespace hyperbola_eccentricity_l214_214981

-- Definitions translated from conditions
noncomputable def parabola_focus : ℝ × ℝ := (0, -Real.sqrt 5)
noncomputable def a : ℝ := 2
noncomputable def c : ℝ := Real.sqrt 5

-- Eccentricity formula for the hyperbola
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

-- Statement to be proved
theorem hyperbola_eccentricity :
  eccentricity c a = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_eccentricity_l214_214981


namespace radius_of_circle_l214_214458

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = Real.pi * r ^ 2) : r = 6 := 
by 
  sorry

end radius_of_circle_l214_214458


namespace alice_meets_john_time_l214_214839

-- Definitions according to conditions
def john_speed : ℝ := 4
def bob_speed : ℝ := 6
def alice_speed : ℝ := 3
def initial_distance_alice_john : ℝ := 2

-- Prove the required meeting time
theorem alice_meets_john_time : 2 / (john_speed + alice_speed) * 60 = 17 := 
by
  sorry

end alice_meets_john_time_l214_214839


namespace daniel_earnings_l214_214894

theorem daniel_earnings :
  let monday_fabric := 20
  let monday_yarn := 15
  let tuesday_fabric := 2 * monday_fabric
  let tuesday_yarn := monday_yarn + 10
  let wednesday_fabric := (1 / 4) * tuesday_fabric
  let wednesday_yarn := (1 / 2) * tuesday_yarn
  let total_fabric := monday_fabric + tuesday_fabric + wednesday_fabric
  let total_yarn := monday_yarn + tuesday_yarn + wednesday_yarn
  let fabric_cost := 2
  let yarn_cost := 3
  let fabric_earnings_before_discount := total_fabric * fabric_cost
  let yarn_earnings_before_discount := total_yarn * yarn_cost
  let fabric_discount := if total_fabric > 30 then 0.10 * fabric_earnings_before_discount else 0
  let yarn_discount := if total_yarn > 20 then 0.05 * yarn_earnings_before_discount else 0
  let fabric_earnings_after_discount := fabric_earnings_before_discount - fabric_discount
  let yarn_earnings_after_discount := yarn_earnings_before_discount - yarn_discount
  let total_earnings := fabric_earnings_after_discount + yarn_earnings_after_discount
  total_earnings = 275.625 := by
  {
    sorry
  }

end daniel_earnings_l214_214894


namespace range_of_a_l214_214384

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - 3 < 0 → x > a) ∧ (∃ x : ℝ, x > a ∧ ¬(x^2 - 2 * x - 3 < 0)) → a ≤ -1 :=
by
  sorry

end range_of_a_l214_214384


namespace num_digits_divisible_l214_214710

theorem num_digits_divisible (h : Nat) :
  (∃ n : Fin 10, (10 * 24 + n) % n = 0) -> h = 7 :=
by sorry

end num_digits_divisible_l214_214710


namespace unique_solution_2023_plus_2_pow_n_eq_k_sq_l214_214593

theorem unique_solution_2023_plus_2_pow_n_eq_k_sq (n k : ℕ) (h : 2023 + 2^n = k^2) :
  (n = 1 ∧ k = 45) :=
by
  sorry

end unique_solution_2023_plus_2_pow_n_eq_k_sq_l214_214593


namespace stone_counting_l214_214604

theorem stone_counting (n : ℕ) (m : ℕ) : 
    10 > 0 ∧  (n ≡ 6 [MOD 20]) ∧ m = 126 → n = 6 := 
by
  sorry

end stone_counting_l214_214604


namespace solve_for_x_l214_214991

def operation (a b : ℝ) : ℝ := a^2 - 3*a + b

theorem solve_for_x (x : ℝ) : operation x 2 = 6 → (x = -1 ∨ x = 4) :=
by
  sorry

end solve_for_x_l214_214991


namespace min_operator_result_l214_214070

theorem min_operator_result : 
  min ((-3) + (-6)) (min ((-3) - (-6)) (min ((-3) * (-6)) ((-3) / (-6)))) = -9 := 
by 
  sorry

end min_operator_result_l214_214070


namespace area_within_square_outside_semicircles_l214_214110

theorem area_within_square_outside_semicircles (side_length : ℝ) (r : ℝ) (area_square : ℝ) (area_semicircles : ℝ) (area_shaded : ℝ) 
  (h1 : side_length = 4)
  (h2 : r = side_length / 2)
  (h3 : area_square = side_length * side_length)
  (h4 : area_semicircles = 4 * (1 / 2 * π * r^2))
  (h5 : area_shaded = area_square - area_semicircles)
  : area_shaded = 16 - 8 * π :=
sorry

end area_within_square_outside_semicircles_l214_214110


namespace smallest_lcm_for_80k_quadruples_l214_214348

-- Declare the gcd and lcm functions for quadruples
def gcd_quad (a b c d : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) (Nat.gcd c d)
def lcm_quad (a b c d : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) (Nat.lcm c d)

-- Main statement we need to prove
theorem smallest_lcm_for_80k_quadruples :
  ∃ m : ℕ, (∃ (a b c d : ℕ), gcd_quad a b c d = 100 ∧ lcm_quad a b c d = m) ∧
    (∀ m', m' < m → ¬ (∃ (a' b' c' d' : ℕ), gcd_quad a' b' c' d' = 100 ∧ lcm_quad a' b' c' d' = m')) ∧
    m = 2250000 :=
sorry

end smallest_lcm_for_80k_quadruples_l214_214348


namespace max_value_g_f_less_than_e_x_div_x_sq_l214_214859

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem max_value_g : ∃ x, x = 3 ∧ g x = 2 * Real.log 2 - 7 / 4 := by
  sorry

theorem f_less_than_e_x_div_x_sq (x : ℝ) (hx : x > 0) : f x < (Real.exp x - 1) / x ^ 2 := by
  sorry

end max_value_g_f_less_than_e_x_div_x_sq_l214_214859


namespace domain_log_base_2_l214_214372

theorem domain_log_base_2 (x : ℝ) : (1 - x > 0) ↔ (x < 1) := by
  sorry

end domain_log_base_2_l214_214372


namespace sequence_equiv_l214_214459

theorem sequence_equiv (n : ℕ) (hn : n > 0) : ∃ (p : ℕ), p > 0 ∧ (4 * p + 5 = (3^n)^2) :=
by
  sorry

end sequence_equiv_l214_214459


namespace multiple_choice_question_count_l214_214385

theorem multiple_choice_question_count (n : ℕ) : 
  (4 * 224 / (2^4 - 2) = 4^2) → n = 2 := 
by
  sorry

end multiple_choice_question_count_l214_214385


namespace extreme_point_of_f_l214_214729

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * x^2 - Real.log x

theorem extreme_point_of_f : 
  ∃ c : ℝ, c = Real.sqrt 3 / 3 ∧ (∀ x: ℝ, x > 0 → (f x > f c → x > c) ∧ (f x < f c → x < c)) := 
sorry

end extreme_point_of_f_l214_214729


namespace pool_people_count_l214_214929

theorem pool_people_count (P : ℕ) (total_money : ℝ) (cost_per_person : ℝ) (leftover_money : ℝ) 
  (h1 : total_money = 30) 
  (h2 : cost_per_person = 2.50) 
  (h3 : leftover_money = 5) 
  (h4 : total_money - leftover_money = cost_per_person * P) : 
  P = 10 :=
sorry

end pool_people_count_l214_214929


namespace flash_catches_ace_l214_214963

theorem flash_catches_ace (v : ℝ) (x : ℝ) (y : ℝ) (hx : x > 1) :
  let t := y / (v * (x - 1))
  let ace_distance := v * t
  let flash_distance := x * v * t
  flash_distance = (xy / (x - 1)) :=
by
  let t := y / (v * (x - 1))
  let ace_distance := v * t
  let flash_distance := x * v * t
  have h1 : x * v * t = xy / (x - 1) := sorry
  exact h1

end flash_catches_ace_l214_214963


namespace sides_of_triangle_inequality_l214_214587

theorem sides_of_triangle_inequality (a b c : ℝ) (h : a + b > c) : a + b > c := 
by 
  exact h

end sides_of_triangle_inequality_l214_214587


namespace avg_cost_of_6_toys_l214_214500

-- Define the given conditions
def dhoni_toys_count : ℕ := 5
def dhoni_toys_avg_cost : ℝ := 10
def sixth_toy_cost : ℝ := 16
def sales_tax_rate : ℝ := 0.10

-- Define the supposed answer
def supposed_avg_cost : ℝ := 11.27

-- Define the problem in Lean 4 statement
theorem avg_cost_of_6_toys :
  (dhoni_toys_count * dhoni_toys_avg_cost + sixth_toy_cost * (1 + sales_tax_rate)) / (dhoni_toys_count + 1) = supposed_avg_cost :=
by
  -- Proof goes here, replace with actual proof
  sorry

end avg_cost_of_6_toys_l214_214500


namespace number_of_days_in_first_part_l214_214034

variable {x : ℕ}

-- Conditions
def avg_exp_first_part (x : ℕ) : ℕ := 350 * x
def avg_exp_next_four_days : ℕ := 420 * 4
def total_days (x : ℕ) : ℕ := x + 4
def avg_exp_whole_week (x : ℕ) : ℕ := 390 * total_days x

-- Equation based on the conditions
theorem number_of_days_in_first_part :
  avg_exp_first_part x + avg_exp_next_four_days = avg_exp_whole_week x →
  x = 3 :=
by
  sorry

end number_of_days_in_first_part_l214_214034


namespace number_of_girls_calculation_l214_214169

theorem number_of_girls_calculation : 
  ∀ (number_of_boys number_of_girls total_children : ℕ), 
  number_of_boys = 27 → total_children = 62 → number_of_girls = total_children - number_of_boys → number_of_girls = 35 :=
by
  intros number_of_boys number_of_girls total_children 
  intros h_boys h_total h_calc
  rw [h_boys, h_total] at h_calc
  simp at h_calc
  exact h_calc

end number_of_girls_calculation_l214_214169


namespace total_time_to_complete_l214_214095

noncomputable def time_to_clean_keys (n : Nat) (t : Nat) : Nat := n * t

def assignment_time : Nat := 10
def time_per_key : Nat := 3
def remaining_keys : Nat := 14

theorem total_time_to_complete :
  time_to_clean_keys remaining_keys time_per_key + assignment_time = 52 := by
  sorry

end total_time_to_complete_l214_214095


namespace travel_distance_of_wheel_l214_214571

theorem travel_distance_of_wheel (r : ℝ) (revolutions : ℕ) (h_r : r = 2) (h_revolutions : revolutions = 2) : 
    ∃ d : ℝ, d = 8 * Real.pi :=
by
  sorry

end travel_distance_of_wheel_l214_214571


namespace ferris_wheel_capacity_l214_214294

-- Define the conditions
def number_of_seats : ℕ := 14
def people_per_seat : ℕ := 6

-- Theorem to prove the total capacity is 84
theorem ferris_wheel_capacity : number_of_seats * people_per_seat = 84 := sorry

end ferris_wheel_capacity_l214_214294


namespace find_m_l214_214631

-- Definition of the constraints and the values of x and y that satisfy them
def constraint1 (x y : ℝ) : Prop := x + y - 2 ≥ 0
def constraint2 (x y : ℝ) : Prop := x - y + 1 ≥ 0
def constraint3 (x : ℝ) : Prop := x ≤ 3

-- Given conditions
def satisfies_constraints (x y : ℝ) : Prop := 
  constraint1 x y ∧ constraint2 x y ∧ constraint3 x

-- The objective to prove
theorem find_m (x y m : ℝ) (h : satisfies_constraints x y) : 
  (∀ x y, satisfies_constraints x y → (- 3 = m * x + y)) → m = -2 / 3 :=
by
  sorry

end find_m_l214_214631


namespace twelve_million_plus_twelve_thousand_l214_214150

theorem twelve_million_plus_twelve_thousand :
  12000000 + 12000 = 12012000 :=
by
  sorry

end twelve_million_plus_twelve_thousand_l214_214150


namespace base_five_to_ten_3214_l214_214773

theorem base_five_to_ten_3214 : (3 * 5^3 + 2 * 5^2 + 1 * 5^1 + 4 * 5^0) = 434 := by
  sorry

end base_five_to_ten_3214_l214_214773


namespace rabbit_is_hit_l214_214136

noncomputable def P_A : ℝ := 0.6
noncomputable def P_B : ℝ := 0.5
noncomputable def P_C : ℝ := 0.4

noncomputable def P_none_hit : ℝ := (1 - P_A) * (1 - P_B) * (1 - P_C)
noncomputable def P_rabbit_hit : ℝ := 1 - P_none_hit

theorem rabbit_is_hit :
  P_rabbit_hit = 0.88 :=
by
  -- Proof is omitted
  sorry

end rabbit_is_hit_l214_214136


namespace tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_l214_214270

theorem tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2 (alpha : ℝ) 
  (h1 : Real.sin alpha = - (Real.sqrt 3) / 2) 
  (h2 : 3 * π / 2 < alpha ∧ alpha < 2 * π) : 
  Real.tan alpha = - Real.sqrt 3 := 
by 
  sorry

end tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_l214_214270


namespace find_moles_of_NaOH_l214_214308

-- Define the conditions
def reaction (NaOH HClO4 NaClO4 H2O : ℕ) : Prop :=
  NaOH = HClO4 ∧ NaClO4 = HClO4 ∧ H2O = 1

def moles_of_HClO4 := 3
def moles_of_NaClO4 := 3

-- Problem statement
theorem find_moles_of_NaOH : ∃ (NaOH : ℕ), NaOH = moles_of_HClO4 ∧ moles_of_NaClO4 = 3 ∧ NaOH = 3 :=
by sorry

end find_moles_of_NaOH_l214_214308


namespace quadratic_roots_l214_214539

theorem quadratic_roots (x : ℝ) : 
  (2 * x^2 - 4 * x - 5 = 0) ↔ 
  (x = (2 + Real.sqrt 14) / 2 ∨ x = (2 - Real.sqrt 14) / 2) :=
by
  sorry

end quadratic_roots_l214_214539


namespace find_total_price_l214_214982

-- Define the cost parameters
variables (sugar_price salt_price : ℝ)

-- Define the given conditions
def condition_1 : Prop := 2 * sugar_price + 5 * salt_price = 5.50
def condition_2 : Prop := sugar_price = 1.50

-- Theorem to be proven
theorem find_total_price (h1 : condition_1 sugar_price salt_price) (h2 : condition_2 sugar_price) : 
  3 * sugar_price + 1 * salt_price = 5.00 :=
by
  sorry

end find_total_price_l214_214982


namespace least_number_with_remainder_l214_214455

theorem least_number_with_remainder (N : ℕ) : (∃ k : ℕ, N = 12 * k + 4) → N = 256 :=
by
  intro h
  sorry

end least_number_with_remainder_l214_214455


namespace ratio_of_hypothetical_to_actual_children_l214_214891

theorem ratio_of_hypothetical_to_actual_children (C H : ℕ) 
  (h1 : H = 16 * 8)
  (h2 : ∀ N : ℕ, N = C / 8 → C * N = 512) 
  (h3 : C^2 = 512 * 8) : H / C = 2 := 
by 
  sorry

end ratio_of_hypothetical_to_actual_children_l214_214891


namespace remainder_of_h_x10_div_h_x_l214_214928

noncomputable def h (x : ℤ) : ℤ := x^5 - x^4 + x^3 - x^2 + x - 1

theorem remainder_of_h_x10_div_h_x (x : ℤ) : (h (x ^ 10)) % (h x) = -6 :=
by
  sorry

end remainder_of_h_x10_div_h_x_l214_214928


namespace hannah_highest_score_l214_214288

-- Definitions based on conditions
def total_questions : ℕ := 40
def wrong_questions : ℕ := 3
def correct_percent_student_1 : ℝ := 0.95

-- The Lean statement representing the proof problem
theorem hannah_highest_score :
  ∃ q : ℕ, (q > (total_questions - wrong_questions) ∧ q > (total_questions * correct_percent_student_1)) ∧ q = 39 :=
by
  sorry

end hannah_highest_score_l214_214288


namespace jack_lap_time_improvement_l214_214526

/-!
Jack practices running in a stadium. Initially, he completed 15 laps in 45 minutes.
After a month of training, he completed 18 laps in 42 minutes. By how many minutes 
has he improved his lap time?
-/

theorem jack_lap_time_improvement:
  ∀ (initial_laps current_laps : ℕ) 
    (initial_time current_time : ℝ), 
    initial_laps = 15 → 
    current_laps = 18 → 
    initial_time = 45 → 
    current_time = 42 → 
    (initial_time / initial_laps - current_time / current_laps = 2/3) :=
by 
  intros _ _ _ _ h_initial_laps h_current_laps h_initial_time h_current_time
  rw [h_initial_laps, h_current_laps, h_initial_time, h_current_time]
  sorry

end jack_lap_time_improvement_l214_214526


namespace log2_square_eq_37_l214_214551

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_square_eq_37
  {x y : ℝ}
  (hx : x ≠ 1)
  (hy : y ≠ 1)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_log : log2 x = Real.log 8 / Real.log y)
  (h_prod : x * y = 128) :
  (log2 (x / y))^2 = 37 := by
  sorry

end log2_square_eq_37_l214_214551


namespace fraction_equivalent_to_decimal_l214_214836

theorem fraction_equivalent_to_decimal : 
  ∃ (x : ℚ), x = 0.6 + 0.0037 * (1 / (1 - 0.01)) ∧ x = 631 / 990 :=
by
  sorry

end fraction_equivalent_to_decimal_l214_214836


namespace solution_eq_l214_214006

theorem solution_eq (a x : ℚ) :
  (2 * (x - 2 * (x - a / 4)) = 3 * x) ∧ ((x + a) / 9 - (1 - 3 * x) / 12 = 1) → 
  a = 65 / 11 ∧ x = 13 / 11 :=
by
  sorry

end solution_eq_l214_214006


namespace degrees_for_cherry_pie_l214_214780

theorem degrees_for_cherry_pie
  (n c a b : ℕ)
  (hc : c = 15)
  (ha : a = 10)
  (hb : b = 9)
  (hn : n = 48)
  (half_remaining_cherry : (n - (c + a + b)) / 2 = 7) :
  (7 / 48 : ℚ) * 360 = 52.5 := 
by sorry

end degrees_for_cherry_pie_l214_214780


namespace division_correct_multiplication_correct_l214_214240

theorem division_correct : 400 / 5 = 80 := by
  sorry

theorem multiplication_correct : 230 * 3 = 690 := by
  sorry

end division_correct_multiplication_correct_l214_214240


namespace red_marbles_in_A_l214_214968

-- Define the number of marbles in baskets A, B, and C
variables (R : ℕ)
def basketA := R + 2 -- Basket A: R red, 2 yellow
def basketB := 6 + 1 -- Basket B: 6 green, 1 yellow
def basketC := 3 + 9 -- Basket C: 3 white, 9 yellow

-- Define the greatest difference condition
def greatest_difference (A B C : ℕ) := max (max (A - B) (B - C)) (max (A - C) (C - B))

-- Define the hypothesis based on the conditions
axiom H1 : greatest_difference 3 9 0 = 6

-- The theorem we need to prove: The number of red marbles in Basket A is 8
theorem red_marbles_in_A : R = 8 := 
by {
  -- The proof would go here, but we'll use sorry to skip it
  sorry
}

end red_marbles_in_A_l214_214968


namespace incorrect_comparison_tan_138_tan_143_l214_214289

theorem incorrect_comparison_tan_138_tan_143 :
  ¬ (Real.tan (Real.pi * 138 / 180) > Real.tan (Real.pi * 143 / 180)) :=
by sorry

end incorrect_comparison_tan_138_tan_143_l214_214289


namespace problem1_problem2_l214_214131

-- First proof problem
theorem problem1 : - (2^2 : ℚ) + (2/3) * ((1 - 1/3) ^ 2) = -100/27 :=
by sorry

-- Second proof problem
theorem problem2 : (8 : ℚ) ^ (1 / 3) - |2 - (3 : ℚ) ^ (1 / 2)| - (3 : ℚ) ^ (1 / 2) = 0 :=
by sorry

end problem1_problem2_l214_214131


namespace fixed_point_l214_214829

variable (p : ℝ)

def f (x : ℝ) : ℝ := 9 * x^2 + p * x - 5 * p

theorem fixed_point : ∀ c d : ℝ, (∀ p : ℝ, f p c = d) → (c = 5 ∧ d = 225) :=
by
  intro c d h
  -- This is a placeholder for the proof
  sorry

end fixed_point_l214_214829


namespace sum_of_ages_3_years_hence_l214_214101

theorem sum_of_ages_3_years_hence (A B C D S : ℝ) (h1 : A = 2 * B) (h2 : C = A / 2) (h3 : D = A - C) (h_sum : A + B + C + D = S) : 
  (A + 3) + (B + 3) + (C + 3) + (D + 3) = S + 12 :=
by sorry

end sum_of_ages_3_years_hence_l214_214101


namespace total_units_is_34_l214_214302

-- Define the number of units on the first floor
def first_floor_units : Nat := 2

-- Define the number of units on the remaining floors (each floor) and number of such floors
def other_floors_units : Nat := 5
def number_of_other_floors : Nat := 3

-- Define the total number of floors per building
def total_floors : Nat := 4

-- Calculate the total units in one building
def units_in_one_building : Nat := first_floor_units + other_floors_units * number_of_other_floors

-- The number of buildings
def number_of_buildings : Nat := 2

-- Calculate the total number of units in both buildings
def total_units : Nat := units_in_one_building * number_of_buildings

-- Prove the total units is 34
theorem total_units_is_34 : total_units = 34 := by
  sorry

end total_units_is_34_l214_214302


namespace break_even_performances_l214_214130

def totalCost (x : ℕ) : ℕ := 81000 + 7000 * x
def totalRevenue (x : ℕ) : ℕ := 16000 * x

theorem break_even_performances : ∃ x : ℕ, totalCost x = totalRevenue x ∧ x = 9 := 
by
  sorry

end break_even_performances_l214_214130


namespace volume_tetrahedron_formula_l214_214838

-- Definitions of the problem elements
def distance (A B C D : Point) : ℝ := sorry
def angle (A B C D : Point) : ℝ := sorry
def length (A B : Point) : ℝ := sorry

-- The problem states you need to prove the volume of the tetrahedron
noncomputable def volume_tetrahedron (A B C D : Point) : ℝ := sorry

-- Conditions
variable (A B C D : Point)
variable (d : ℝ) (phi : ℝ) -- d = distance between lines AB and CD, phi = angle between lines AB and CD

-- Question reformulated as a proof statement
theorem volume_tetrahedron_formula (h1 : d = distance A B C D)
                                   (h2 : phi = angle A B C D) :
  volume_tetrahedron A B C D = (d * length A B * length C D * Real.sin phi) / 6 :=
sorry

end volume_tetrahedron_formula_l214_214838


namespace zoe_takes_correct_amount_of_money_l214_214475

def numberOfPeople : ℕ := 6
def costPerSoda : ℝ := 0.5
def costPerPizza : ℝ := 1.0

def totalCost : ℝ := (numberOfPeople * costPerSoda) + (numberOfPeople * costPerPizza)

theorem zoe_takes_correct_amount_of_money : totalCost = 9 := sorry

end zoe_takes_correct_amount_of_money_l214_214475


namespace right_triangle_legs_from_medians_l214_214713

theorem right_triangle_legs_from_medians
  (a b : ℝ) (x y : ℝ)
  (h1 : x^2 + 4 * y^2 = 4 * a^2)
  (h2 : 4 * x^2 + y^2 = 4 * b^2) :
  y^2 = (16 * a^2 - 4 * b^2) / 15 ∧ x^2 = (16 * b^2 - 4 * a^2) / 15 :=
by
  sorry

end right_triangle_legs_from_medians_l214_214713


namespace ab_value_l214_214967

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 172) : ab = 85 / 6 := 
by
  sorry

end ab_value_l214_214967


namespace road_signs_count_l214_214222

theorem road_signs_count (n1 n2 n3 n4 : ℕ) (h1 : n1 = 40) (h2 : n2 = n1 + n1 / 4) (h3 : n3 = 2 * n2) (h4 : n4 = n3 - 20) : 
  n1 + n2 + n3 + n4 = 270 := 
by
  sorry

end road_signs_count_l214_214222


namespace negative_fraction_comparison_l214_214012

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l214_214012


namespace find_m_l214_214452

noncomputable def vec_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem find_m (m : ℝ) (h : dot_product (vec_add (-1, 2) (m, 1)) (-1, 2) = 0) : m = 7 :=
  by 
  sorry

end find_m_l214_214452


namespace max_balls_drawn_l214_214706

structure DrawingConditions where
  yellow_items : ℕ
  round_items : ℕ
  edible_items : ℕ
  tomato_properties : Prop
  ball_properties : Prop
  banana_properties : Prop

axiom petya_conditions : DrawingConditions → Prop 

theorem max_balls_drawn (cond : DrawingConditions)
  (h1 : cond.yellow_items = 15)
  (h2 : cond.round_items = 18)
  (h3 : cond.edible_items = 13)
  (h4 : cond.tomato_properties)
  (h5 : cond.ball_properties)
  (h6 : cond.banana_properties) :
  ∀ balls : ℕ, balls ≤ 18 :=
by
  sorry

end max_balls_drawn_l214_214706


namespace find_speed_of_B_l214_214108

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l214_214108


namespace part1_part2_part3_l214_214037

-- Define the complex number z
def z (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m + 2, m^2 - 1⟩  -- Note: This forms a complex number with real and imaginary parts

-- (1) Proof for z = 0 if and only if m = 1
theorem part1 (m : ℝ) : z m = 0 ↔ m = 1 :=
by sorry

-- (2) Proof for z being a pure imaginary number if and only if m = 2
theorem part2 (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 :=
by sorry

-- (3) Proof for the point corresponding to z being in the second quadrant if and only if 1 < m < 2
theorem part3 (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ 1 < m ∧ m < 2 :=
by sorry

end part1_part2_part3_l214_214037


namespace expressions_inequivalence_l214_214750

theorem expressions_inequivalence (x : ℝ) (h : x > 0) :
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ 2 * (x + 1) ^ x) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ (x + 1) ^ (2 * x + 2)) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ 2 * (0.5 * x + x) ^ x) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ (2 * x + 2) ^ (2 * x + 2)) := by
  sorry

end expressions_inequivalence_l214_214750


namespace unique_x_value_l214_214174

theorem unique_x_value (x : ℝ) (h : x ≠ 0) (h_sqrt : Real.sqrt (5 * x / 7) = x) : x = 5 / 7 :=
by
  sorry

end unique_x_value_l214_214174


namespace annual_depletion_rate_l214_214662

theorem annual_depletion_rate
  (initial_value : ℝ) 
  (final_value : ℝ) 
  (time : ℝ) 
  (depletion_rate : ℝ)
  (h_initial_value : initial_value = 40000)
  (h_final_value : final_value = 36100)
  (h_time : time = 2)
  (decay_eq : final_value = initial_value * (1 - depletion_rate)^time) :
  depletion_rate = 0.05 :=
by 
  sorry

end annual_depletion_rate_l214_214662


namespace opposite_of_neg_three_l214_214657

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l214_214657


namespace ratio_Bill_Cary_l214_214060

noncomputable def Cary_height : ℝ := 72
noncomputable def Jan_height : ℝ := 42
noncomputable def Bill_height : ℝ := Jan_height - 6

theorem ratio_Bill_Cary : Bill_height / Cary_height = 1 / 2 :=
by
  sorry

end ratio_Bill_Cary_l214_214060


namespace cubic_boxes_properties_l214_214625

-- Define the lengths of the edges of the cubic boxes
def edge_length_1 : ℝ := 3
def edge_length_2 : ℝ := 5
def edge_length_3 : ℝ := 6

-- Define the volumes of the respective cubic boxes
def volume (edge_length : ℝ) : ℝ := edge_length ^ 3
def volume_1 := volume edge_length_1
def volume_2 := volume edge_length_2
def volume_3 := volume edge_length_3

-- Define the surface areas of the respective cubic boxes
def surface_area (edge_length : ℝ) : ℝ := 6 * (edge_length ^ 2)
def surface_area_1 := surface_area edge_length_1
def surface_area_2 := surface_area edge_length_2
def surface_area_3 := surface_area edge_length_3

-- Total volume and surface area calculations
def total_volume := volume_1 + volume_2 + volume_3
def total_surface_area := surface_area_1 + surface_area_2 + surface_area_3

-- Theorem statement to be proven
theorem cubic_boxes_properties :
  total_volume = 368 ∧ total_surface_area = 420 := by
  sorry

end cubic_boxes_properties_l214_214625


namespace permutations_count_5040_four_digit_numbers_count_4356_even_four_digit_numbers_count_1792_l214_214123

open Finset

def digits : Finset ℤ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def permutations_no_repetition : ℤ :=
  (digits.card.factorial) / ((digits.card - 4).factorial)

noncomputable def four_digit_numbers_no_repetition : ℤ :=
  9 * ((digits.card - 1).factorial / ((digits.card - 1 - 3).factorial))

noncomputable def even_four_digit_numbers_gt_3000_no_repetition : ℤ :=
  784 + 1008

theorem permutations_count_5040 : permutations_no_repetition = 5040 := by
  sorry

theorem four_digit_numbers_count_4356 : four_digit_numbers_no_repetition = 4356 := by
  sorry

theorem even_four_digit_numbers_count_1792 : even_four_digit_numbers_gt_3000_no_repetition = 1792 := by
  sorry

end permutations_count_5040_four_digit_numbers_count_4356_even_four_digit_numbers_count_1792_l214_214123


namespace simplify_polynomial_l214_214366

theorem simplify_polynomial (x : ℝ) : 
  (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1 = 32 * x ^ 5 := 
by 
  sorry

end simplify_polynomial_l214_214366


namespace least_three_digit_product_12_l214_214170

-- Problem statement: Find the least three-digit number whose digits multiply to 12
theorem least_three_digit_product_12 : ∃ (n : ℕ), n ≥ 100 ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a * b * c = 12 ∧ n = 126) :=
by
  sorry

end least_three_digit_product_12_l214_214170


namespace sequence_terms_distinct_l214_214345

theorem sequence_terms_distinct (n m : ℕ) (hnm : n ≠ m) : 
  (n / (n + 1) : ℚ) ≠ (m / (m + 1) : ℚ) :=
sorry

end sequence_terms_distinct_l214_214345


namespace collinear_points_inverse_sum_half_l214_214113

theorem collinear_points_inverse_sum_half (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
    (collinear : (a - 2) * (b - 2) - (-2) * a = 0) : 
    1 / a + 1 / b = 1 / 2 := 
by
  sorry

end collinear_points_inverse_sum_half_l214_214113


namespace lakshmi_share_annual_gain_l214_214909

theorem lakshmi_share_annual_gain (x : ℝ) (annual_gain : ℝ) (Raman_inv_months : ℝ) (Lakshmi_inv_months : ℝ) (Muthu_inv_months : ℝ) (Gowtham_inv_months : ℝ) (Pradeep_inv_months : ℝ)
  (total_inv_months : ℝ) (lakshmi_share : ℝ) :
  Raman_inv_months = x * 12 →
  Lakshmi_inv_months = 2 * x * 6 →
  Muthu_inv_months = 3 * x * 4 →
  Gowtham_inv_months = 4 * x * 9 →
  Pradeep_inv_months = 5 * x * 1 →
  total_inv_months = Raman_inv_months + Lakshmi_inv_months + Muthu_inv_months + Gowtham_inv_months + Pradeep_inv_months →
  annual_gain = 58000 →
  lakshmi_share = (Lakshmi_inv_months / total_inv_months) * annual_gain →
  lakshmi_share = 9350.65 :=
by
  sorry

end lakshmi_share_annual_gain_l214_214909


namespace dominoes_per_player_l214_214375

-- Define the conditions
def total_dominoes : ℕ := 28
def number_of_players : ℕ := 4

-- The theorem
theorem dominoes_per_player : total_dominoes / number_of_players = 7 :=
by sorry

end dominoes_per_player_l214_214375


namespace two_f_one_lt_f_four_l214_214077

theorem two_f_one_lt_f_four
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f (-x - 2))
  (h2 : ∀ x, x > 2 → x * (deriv f x) > 2 * (deriv f x) + f x) :
  2 * f 1 < f 4 :=
sorry

end two_f_one_lt_f_four_l214_214077


namespace at_least_one_greater_than_16000_l214_214711

open Nat

theorem at_least_one_greater_than_16000 (seq : Fin 20 → ℕ)
  (h_distinct : ∀ i j : Fin 20, i ≠ j → seq i ≠ seq j)
  (h_perfect_square : ∀ i : Fin 19, ∃ k : ℕ, (seq i) * (seq (i + 1)) = k^2)
  (h_first : seq 0 = 42) : ∃ i : Fin 20, seq i > 16000 :=
by
  sorry

end at_least_one_greater_than_16000_l214_214711


namespace repair_cost_l214_214796

theorem repair_cost
  (R : ℝ) -- R is the cost to repair the used shoes
  (new_shoes_cost : ℝ := 30) -- New shoes cost $30.00
  (new_shoes_lifetime : ℝ := 2) -- New shoes last for 2 years
  (percentage_increase : ℝ := 42.857142857142854) 
  (h1 : new_shoes_cost / new_shoes_lifetime = R + (percentage_increase / 100) * R) :
  R = 10.50 :=
by
  sorry

end repair_cost_l214_214796


namespace find_usual_time_l214_214154

variables (P D T : ℝ)
variable (h1 : P = D / T)
variable (h2 : 3 / 4 * P = D / (T + 20))

theorem find_usual_time (h1 : P = D / T) (h2 : 3 / 4 * P = D / (T + 20)) : T = 80 := 
  sorry

end find_usual_time_l214_214154


namespace trajectory_ellipse_l214_214904

/--
Given two fixed points A(-2,0) and B(2,0) in the Cartesian coordinate system, 
if a moving point P satisfies |PA| + |PB| = 6, 
then prove that the equation of the trajectory for point P is (x^2) / 9 + (y^2) / 5 = 1.
-/
theorem trajectory_ellipse (P : ℝ × ℝ)
  (A B : ℝ × ℝ)
  (hA : A = (-2, 0))
  (hB : B = (2, 0))
  (hPA_PB : dist P A + dist P B = 6) :
  (P.1 ^ 2) / 9 + (P.2 ^ 2) / 5 = 1 :=
sorry

end trajectory_ellipse_l214_214904


namespace length_PC_in_rectangle_l214_214821

theorem length_PC_in_rectangle (PA PB PD: ℝ) (P_inside: True) 
(h1: PA = 5) (h2: PB = 7) (h3: PD = 3) : PC = Real.sqrt 65 := 
sorry

end length_PC_in_rectangle_l214_214821


namespace arithmetic_sequence_number_of_terms_l214_214746

theorem arithmetic_sequence_number_of_terms 
  (a d : ℝ) (n : ℕ) 
  (h1 : a + (a + d) + (a + 2 * d) = 34) 
  (h2 : (a + (n-3) * d) + (a + (n-2) * d) + (a + (n-1) * d) = 146) 
  (h3 : (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 390) : 
  n = 13 :=
by 
  sorry

end arithmetic_sequence_number_of_terms_l214_214746


namespace delta_epsilon_time_l214_214237

variable (D E Z h t : ℕ)

theorem delta_epsilon_time :
  (t = D - 8) →
  (t = E - 3) →
  (t = Z / 3) →
  (h = 3 * t) → 
  h = 15 / 8 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end delta_epsilon_time_l214_214237


namespace bridgette_total_baths_l214_214244

def bridgette_baths (dogs baths_per_dog_per_month cats baths_per_cat_per_month birds baths_per_bird_per_month : ℕ) : ℕ :=
  (dogs * baths_per_dog_per_month * 12) + (cats * baths_per_cat_per_month * 12) + (birds * (12 / baths_per_bird_per_month))

theorem bridgette_total_baths :
  bridgette_baths 2 2 3 1 4 4 = 96 :=
by
  -- Proof omitted
  sorry

end bridgette_total_baths_l214_214244


namespace find_f_neg_one_l214_214781

theorem find_f_neg_one (f h : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x)
    (h2 : ∀ x, h x = f x - 9) (h3 : h 1 = 2) : f (-1) = -11 := 
by
  sorry

end find_f_neg_one_l214_214781


namespace total_legs_proof_l214_214776

def johnny_legs : Nat := 2
def son_legs : Nat := 2
def dog_legs : Nat := 4
def number_of_dogs : Nat := 2
def number_of_humans : Nat := 2

def total_legs : Nat :=
  (number_of_dogs * dog_legs) + (number_of_humans * johnny_legs)

theorem total_legs_proof : total_legs = 12 := by
  sorry

end total_legs_proof_l214_214776


namespace bus_speed_proof_l214_214483
noncomputable def speed_of_bus (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let relative_speed_mps := train_length / time_to_pass
  let bus_speed_mps := relative_speed_mps - train_speed_mps
  bus_speed_mps * 3.6

theorem bus_speed_proof : 
  speed_of_bus 220 90 5.279577633789296 = 60 :=
by
  sorry

end bus_speed_proof_l214_214483


namespace speed_of_canoe_downstream_l214_214295

-- Definition of the problem conditions
def speed_of_canoe_in_still_water (V_c : ℝ) (V_s : ℝ) (upstream_speed : ℝ) : Prop :=
  V_c - V_s = upstream_speed

def speed_of_stream (V_s : ℝ) : Prop :=
  V_s = 4

-- The statement we want to prove
theorem speed_of_canoe_downstream (V_c V_s : ℝ) (upstream_speed : ℝ) 
  (h1 : speed_of_canoe_in_still_water V_c V_s upstream_speed)
  (h2 : speed_of_stream V_s)
  (h3 : upstream_speed = 4) :
  V_c + V_s = 12 :=
by
  sorry

end speed_of_canoe_downstream_l214_214295


namespace nba_conference_division_impossible_l214_214876

theorem nba_conference_division_impossible :
  let teams := 30
  let games_per_team := 82
  let total_games := teams * games_per_team
  let unique_games := total_games / 2
  let inter_conference_games := unique_games / 2
  ¬∃ (A B : ℕ), A + B = teams ∧ A * B = inter_conference_games := 
by
  let teams := 30
  let games_per_team := 82
  let total_games := teams * games_per_team
  let unique_games := total_games / 2
  let inter_conference_games := unique_games / 2
  sorry

end nba_conference_division_impossible_l214_214876


namespace calculate_probability_l214_214226

theorem calculate_probability :
  let letters_in_bag : List Char := ['C', 'A', 'L', 'C', 'U', 'L', 'A', 'T', 'E']
  let target_letters : List Char := ['C', 'U', 'T']
  let total_outcomes := letters_in_bag.length
  let favorable_outcomes := (letters_in_bag.filter (λ c => c ∈ target_letters)).length
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 4 / 9 := sorry

end calculate_probability_l214_214226


namespace total_selection_methods_l214_214324

-- Define the students and days
inductive Student
| S1 | S2 | S3 | S4 | S5

inductive Day
| Wednesday | Thursday | Friday | Saturday | Sunday

-- The condition where S1 cannot be on Saturday and S2 cannot be on Sunday
def valid_arrangement (arrangement : Day → Student) : Prop :=
  arrangement Day.Saturday ≠ Student.S1 ∧
  arrangement Day.Sunday ≠ Student.S2

-- The main statement
theorem total_selection_methods : ∃ (arrangement_count : ℕ), 
  arrangement_count = 78 ∧
  ∀ (arrangement : Day → Student), valid_arrangement arrangement → 
  arrangement_count = 78 :=
sorry

end total_selection_methods_l214_214324


namespace find_C_coordinates_l214_214371

open Real

noncomputable def coordC (A B : ℝ × ℝ) : ℝ × ℝ :=
  let n := A.1
  let m := B.1
  let coord_n_y : ℝ := n
  let coord_m_y : ℝ := m
  let y_value (x : ℝ) : ℝ := sqrt 3 / x
  (sqrt 3 / 2, 2)

theorem find_C_coordinates :
  ∃ C : ℝ × ℝ, 
  (∃ A B : ℝ × ℝ, 
   A.2 = sqrt 3 / A.1 ∧
   B.2 = sqrt 3 / B.1 + 6 ∧
   A.2 + 6 = B.2 ∧
   B.2 > A.2 ∧ 
   (sqrt 3 / 2, 2) = coordC A B) ∧
   (sqrt 3 / 2, 2) = (C.1, C.2) :=
by
  sorry

end find_C_coordinates_l214_214371


namespace possible_values_for_xyz_l214_214047

theorem possible_values_for_xyz:
  (∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
   x + 2 * y = z →
   x^2 - 4 * y^2 + z^2 = 310 →
   (∃ (k : ℕ), k = x * y * z ∧ (k = 11935 ∨ k = 2015))) :=
by
  intros x y z hx hy hz h1 h2
  sorry

end possible_values_for_xyz_l214_214047


namespace price_increase_equivalence_l214_214276

theorem price_increase_equivalence (P : ℝ) : 
  let increase_35 := P * 1.35
  let increase_40 := increase_35 * 1.40
  let increase_20 := increase_40 * 1.20
  let final_increase := increase_20
  final_increase = P * 2.268 :=
by
  -- proof skipped
  sorry

end price_increase_equivalence_l214_214276


namespace boots_ratio_l214_214114

noncomputable def problem_statement : Prop :=
  let total_money : ℝ := 50
  let cost_toilet_paper : ℝ := 12
  let cost_groceries : ℝ := 2 * cost_toilet_paper
  let remaining_after_groceries : ℝ := total_money - cost_toilet_paper - cost_groceries
  let extra_money_per_person : ℝ := 35
  let total_extra_money : ℝ := 2 * extra_money_per_person
  let total_cost_boots : ℝ := remaining_after_groceries + total_extra_money
  let cost_per_pair_boots : ℝ := total_cost_boots / 2
  let ratio := cost_per_pair_boots / remaining_after_groceries
  ratio = 3

theorem boots_ratio (total_money : ℝ) (cost_toilet_paper : ℝ) (extra_money_per_person : ℝ) : 
  let cost_groceries := 2 * cost_toilet_paper
  let remaining_after_groceries := total_money - cost_toilet_paper - cost_groceries
  let total_extra_money := 2 * extra_money_per_person
  let total_cost_boots := remaining_after_groceries + total_extra_money
  let cost_per_pair_boots := total_cost_boots / 2
  let ratio := cost_per_pair_boots / remaining_after_groceries
  ratio = 3 :=
by
  sorry

end boots_ratio_l214_214114


namespace scientific_notation_of_300670_l214_214187

theorem scientific_notation_of_300670 : ∃ a : ℝ, ∃ n : ℤ, (1 ≤ |a| ∧ |a| < 10) ∧ 300670 = a * 10^n ∧ a = 3.0067 ∧ n = 5 :=
  by
    sorry

end scientific_notation_of_300670_l214_214187


namespace isosceles_triangle_l214_214040

theorem isosceles_triangle
  (α β γ x y z w : ℝ)
  (h_triangle : α + β + γ = 180)
  (h_quad : x + y + z + w = 360)
  (h_conditions : (x = α + β) ∧ (y = β + γ) ∧ (z = γ + α) ∨ (w = α + β) ∧ (x = β + γ) ∧ (y = γ + α) ∨ (z = α + β) ∧ (w = β + γ) ∧ (x = γ + α) ∨ (y = α + β) ∧ (z = β + γ) ∧ (w = γ + α))
  : α = β ∨ β = γ ∨ γ = α := 
sorry

end isosceles_triangle_l214_214040


namespace number_of_t_in_T_such_that_f_t_mod_8_eq_0_l214_214184

def f (x : ℤ) : ℤ := x^3 + 2 * x^2 + 3 * x + 4

def T := { n : ℤ | 0 ≤ n ∧ n ≤ 50 }

theorem number_of_t_in_T_such_that_f_t_mod_8_eq_0 : 
  (∃ t ∈ T, f t % 8 = 0) = false := sorry

end number_of_t_in_T_such_that_f_t_mod_8_eq_0_l214_214184


namespace total_apples_eaten_l214_214417

theorem total_apples_eaten : (1 / 2) * 16 + (1 / 3) * 15 + (1 / 4) * 20 = 18 := by
  sorry

end total_apples_eaten_l214_214417


namespace smallest_prime_dividing_sum_l214_214434

theorem smallest_prime_dividing_sum (h1 : Odd 7) (h2 : Odd 9) 
    (h3 : ∀ {a b : ℤ}, Odd a → Odd b → Even (a + b)) :
  ∃ p : ℕ, Prime p ∧ p ∣ (7 ^ 15 + 9 ^ 7) ∧ p = 2 := 
by
  sorry

end smallest_prime_dividing_sum_l214_214434


namespace cycling_distance_l214_214779

-- Define the conditions
def cycling_time : ℕ := 40  -- Total cycling time in minutes
def time_per_interval : ℕ := 10  -- Time per interval in minutes
def distance_per_interval : ℕ := 2  -- Distance per interval in miles

-- Proof statement
theorem cycling_distance : (cycling_time / time_per_interval) * distance_per_interval = 8 := by
  sorry

end cycling_distance_l214_214779


namespace total_selling_price_correct_l214_214605

-- Define the given conditions
def cost_price_per_metre : ℝ := 72
def loss_per_metre : ℝ := 12
def total_metres_of_cloth : ℝ := 200

-- Define the selling price per metre
def selling_price_per_metre : ℝ := cost_price_per_metre - loss_per_metre

-- Define the total selling price
def total_selling_price : ℝ := selling_price_per_metre * total_metres_of_cloth

-- The theorem we want to prove
theorem total_selling_price_correct : 
  total_selling_price = 12000 := 
by
  sorry

end total_selling_price_correct_l214_214605


namespace trajectory_eq_l214_214333

theorem trajectory_eq (M : Type) [MetricSpace M] : 
  (∀ (r x y : ℝ), (x + 2)^2 + y^2 = (r + 1)^2 ∧ |x - 1| = 1 → y^2 = -8 * x) :=
by sorry

end trajectory_eq_l214_214333


namespace inequality_geq_27_l214_214485

theorem inequality_geq_27 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_eq : a + b + c + 2 = a * b * c) : (a + 1) * (b + 1) * (c + 1) ≥ 27 := 
    sorry

end inequality_geq_27_l214_214485


namespace simplify_expression_l214_214398

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 := 
by 
  sorry

end simplify_expression_l214_214398


namespace Misha_earnings_needed_l214_214436

-- Define the conditions and the goal in Lean 4
def Misha_current_dollars : ℕ := 34
def Misha_target_dollars : ℕ := 47

theorem Misha_earnings_needed : Misha_target_dollars - Misha_current_dollars = 13 := by
  sorry

end Misha_earnings_needed_l214_214436


namespace find_y_l214_214004

theorem find_y (x y : ℝ) (h1 : 2 * (x - y) = 12) (h2 : x + y = 14) : y = 4 := 
by
  sorry

end find_y_l214_214004


namespace essentially_different_proportions_l214_214435

theorem essentially_different_proportions (x y z t : α) [DecidableEq α] 
  (h1 : x ≠ y) (h2 : x ≠ z) (h3 : x ≠ t) (h4 : y ≠ z) (h5 : y ≠ t) (h6 : z ≠ t) : 
  ∃ n : ℕ, n = 3 := by
  sorry

end essentially_different_proportions_l214_214435


namespace minimum_sugar_correct_l214_214721

noncomputable def minimum_sugar (f : ℕ) (s : ℕ) : ℕ := 
  if (f ≥ 8 + s / 2 ∧ f ≤ 3 * s) then s else sorry

theorem minimum_sugar_correct (f s : ℕ) : 
  (f ≥ 8 + s / 2 ∧ f ≤ 3 * s) → s ≥ 4 :=
by sorry

end minimum_sugar_correct_l214_214721


namespace y_intercept_of_line_l214_214636

theorem y_intercept_of_line (m : ℝ) (a : ℝ) (b : ℝ) (ha : a ≠ 0) (hb : b = 0) (h_slope : m = 3) (h_x_intercept : (a, b) = (4, 0)) :
  ∃ y : ℝ, (0, y) = (0, -12) :=
by 
  sorry

end y_intercept_of_line_l214_214636


namespace burger_cost_l214_214775

theorem burger_cost :
  ∃ b s f : ℕ, 4 * b + 2 * s + 3 * f = 480 ∧ 3 * b + s + 2 * f = 360 ∧ b = 80 :=
by
  sorry

end burger_cost_l214_214775


namespace ab_difference_l214_214645

theorem ab_difference (a b : ℝ) 
  (h1 : 10 = a * 3 + b)
  (h2 : 22 = a * 7 + b) : 
  a - b = 2 := 
  sorry

end ab_difference_l214_214645


namespace family_gathering_l214_214594

theorem family_gathering : 
  ∃ (total_people oranges bananas apples : ℕ), 
    total_people = 20 ∧ 
    oranges = total_people / 2 ∧ 
    bananas = (total_people - oranges) / 2 ∧ 
    apples = total_people - oranges - bananas ∧ 
    oranges < total_people ∧ 
    total_people - oranges = 10 :=
by
  sorry

end family_gathering_l214_214594


namespace sqrt_40_simplified_l214_214603

theorem sqrt_40_simplified : Real.sqrt 40 = 2 * Real.sqrt 10 := 
by
  sorry

end sqrt_40_simplified_l214_214603


namespace no_consecutive_integer_sum_to_36_l214_214889

theorem no_consecutive_integer_sum_to_36 :
  ∀ (a n : ℕ), n ≥ 2 → (n * a + n * (n - 1) / 2) = 36 → false :=
by
  sorry

end no_consecutive_integer_sum_to_36_l214_214889


namespace scientific_notation_of_258000000_l214_214570

theorem scientific_notation_of_258000000 :
  258000000 = 2.58 * 10^8 :=
sorry

end scientific_notation_of_258000000_l214_214570


namespace min_value_of_m_l214_214635

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x + 5

theorem min_value_of_m {m : ℝ} (h : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ m) : m = 5 := 
sorry

end min_value_of_m_l214_214635


namespace polynomial_properties_l214_214569

noncomputable def polynomial : Polynomial ℚ :=
  -3/8 * (Polynomial.X ^ 5) + 5/4 * (Polynomial.X ^ 3) - 15/8 * (Polynomial.X)

theorem polynomial_properties (f : Polynomial ℚ) :
  (Polynomial.degree f = 5) ∧
  (∃ q : Polynomial ℚ, f + 1 = Polynomial.X - 1 ^ 3 * q) ∧
  (∃ p : Polynomial ℚ, f - 1 = Polynomial.X + 1 ^ 3 * p) ↔
  f = polynomial :=
by sorry

end polynomial_properties_l214_214569


namespace current_average_age_of_seven_persons_l214_214768

theorem current_average_age_of_seven_persons (T : ℕ)
  (h1 : T + 12 = 6 * 43)
  (h2 : 69 = 69)
  : (T + 69) / 7 = 45 := by
  sorry

end current_average_age_of_seven_persons_l214_214768


namespace production_rate_equation_l214_214794

theorem production_rate_equation (x : ℝ) (h1 : ∀ t : ℝ, t = 600 / (x + 8)) (h2 : ∀ t : ℝ, t = 400 / x) : 
  600/(x + 8) = 400/x :=
by
  sorry

end production_rate_equation_l214_214794


namespace cos_105_sub_alpha_l214_214771

variable (α : ℝ)

-- Condition
def condition : Prop := Real.cos (75 * Real.pi / 180 + α) = 1 / 2

-- Statement
theorem cos_105_sub_alpha (h : condition α) : Real.cos (105 * Real.pi / 180 - α) = -1 / 2 :=
by
  sorry

end cos_105_sub_alpha_l214_214771


namespace geometric_sequence_17th_term_l214_214268

variable {α : Type*} [Field α]

def geometric_sequence (a r : α) (n : ℕ) : α :=
  a * r ^ (n - 1)

theorem geometric_sequence_17th_term :
  ∀ (a r : α),
    a * r ^ 4 = 9 →  -- Fifth term condition
    a * r ^ 12 = 1152 →  -- Thirteenth term condition
    a * r ^ 16 = 36864 :=  -- Seventeenth term conclusion
by
  intros a r h5 h13
  sorry

end geometric_sequence_17th_term_l214_214268


namespace det_A_eq_l214_214017

open Matrix

def A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, -3, 3],
    ![x, 5, -1],
    ![4, -2, 1]]

theorem det_A_eq (x : ℝ) : det (A x) = -3 * x - 45 :=
by sorry

end det_A_eq_l214_214017


namespace find_hundreds_digit_l214_214442

theorem find_hundreds_digit :
  ∃ n : ℕ, (n % 37 = 0) ∧ (n % 173 = 0) ∧ (10000 ≤ n) ∧ (n < 100000) ∧ ((n / 1000) % 10 = 3) ∧ (((n / 100) % 10) = 2) :=
sorry

end find_hundreds_digit_l214_214442


namespace find_smallest_d_l214_214537

noncomputable def smallest_possible_d (c d : ℕ) : ℕ :=
  if c - d = 8 ∧ Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16 then d else 0

-- Proving the smallest possible value of d given the conditions
theorem find_smallest_d :
  ∀ c d : ℕ, (0 < c) → (0 < d) → (c - d = 8) → 
  Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16 → d = 4 :=
by
  sorry

end find_smallest_d_l214_214537


namespace pump_B_rate_l214_214751

noncomputable def rate_A := 1 / 2
noncomputable def rate_C := 1 / 6

theorem pump_B_rate :
  ∃ B : ℝ, (rate_A + B - rate_C = 4 / 3) ∧ (B = 1) := by
  sorry

end pump_B_rate_l214_214751


namespace find_original_price_l214_214576

noncomputable def original_price_per_bottle (P : ℝ) : Prop :=
  let discounted_price := 0.80 * P
  let final_price_per_bottle := discounted_price - 2.00
  3 * final_price_per_bottle = 30

theorem find_original_price : ∃ P : ℝ, original_price_per_bottle P ∧ P = 15 :=
by
  sorry

end find_original_price_l214_214576


namespace total_net_worth_after_2_years_l214_214293

def initial_value : ℝ := 40000
def depreciation_rate : ℝ := 0.05
def initial_maintenance_cost : ℝ := 2000
def inflation_rate : ℝ := 0.03
def years : ℕ := 2

def value_at_end_of_year (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  List.foldl (λ acc _ => acc * (1 - rate)) initial_value (List.range years)

def cumulative_maintenance_cost (initial_maintenance_cost : ℝ) (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  List.foldl (λ acc year => acc + initial_maintenance_cost * ((1 + inflation_rate) ^ year)) 0 (List.range years)

def total_net_worth (initial_value : ℝ) (depreciation_rate : ℝ) (initial_maintenance_cost : ℝ) (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  value_at_end_of_year initial_value depreciation_rate years - cumulative_maintenance_cost initial_maintenance_cost inflation_rate years

theorem total_net_worth_after_2_years : total_net_worth initial_value depreciation_rate initial_maintenance_cost inflation_rate years = 32040 :=
  by
    sorry

end total_net_worth_after_2_years_l214_214293


namespace arithmetic_sequence_m_value_l214_214509

theorem arithmetic_sequence_m_value 
  (a : ℕ → ℝ) (d : ℝ) (h₁ : d ≠ 0) 
  (h₂ : a 3 + a 6 + a 10 + a 13 = 32) 
  (m : ℕ) (h₃ : a m = 8) : 
  m = 8 :=
sorry

end arithmetic_sequence_m_value_l214_214509


namespace sequence_bound_l214_214749

theorem sequence_bound (n : ℕ) (a : ℝ) (a_seq : ℕ → ℝ) 
  (h1 : a_seq 1 = a) 
  (h2 : a_seq n = a) 
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k < n - 1 → a_seq (k + 1) ≤ (a_seq k + a_seq (k + 2)) / 2) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → a_seq k ≤ a := 
by
  sorry

end sequence_bound_l214_214749


namespace red_balls_in_bag_l214_214689

theorem red_balls_in_bag : ∃ x : ℕ, (3 : ℚ) / (4 + (x : ℕ)) = 1 / 2 ∧ x = 2 := sorry

end red_balls_in_bag_l214_214689


namespace correct_transformation_l214_214438

-- Definitions of the points and their mapped coordinates
def C : ℝ × ℝ := (3, -2)
def D : ℝ × ℝ := (4, -3)
def C' : ℝ × ℝ := (1, 2)
def D' : ℝ × ℝ := (-2, 3)

-- Transformation function (as given in the problem)
def skew_reflection_and_vertical_shrink (p : ℝ × ℝ) : ℝ × ℝ :=
  match p with
  | (x, y) => (-y, x)

-- Theorem statement to be proved
theorem correct_transformation :
  skew_reflection_and_vertical_shrink C = C' ∧ skew_reflection_and_vertical_shrink D = D' :=
sorry

end correct_transformation_l214_214438


namespace person_B_correct_probability_l214_214777

-- Define probabilities
def P_A_correct : ℝ := 0.4
def P_A_incorrect : ℝ := 1 - P_A_correct
def P_B_correct_if_A_incorrect : ℝ := 0.5
def P_B_correct : ℝ := P_A_incorrect * P_B_correct_if_A_incorrect

-- Theorem statement
theorem person_B_correct_probability : P_B_correct = 0.3 :=
by
  -- Problem conditions implicitly used in definitions
  sorry

end person_B_correct_probability_l214_214777


namespace roots_cubic_properties_l214_214624

theorem roots_cubic_properties (a b c : ℝ) 
    (h1 : ∀ x : ℝ, x^3 - 2 * x^2 + 3 * x - 4 = 0 → x = a ∨ x = b ∨ x = c)
    (h_sum : a + b + c = 2)
    (h_prod_sum : a * b + b * c + c * a = 3)
    (h_prod : a * b * c = 4) :
  a^3 + b^3 + c^3 = 2 := by
  sorry

end roots_cubic_properties_l214_214624


namespace cakes_to_make_l214_214272

-- Define the conditions
def packages_per_cake : ℕ := 2
def cost_per_package : ℕ := 3
def total_cost : ℕ := 12

-- Define the proof problem
theorem cakes_to_make (h1 : packages_per_cake = 2) (h2 : cost_per_package = 3) (h3 : total_cost = 12) :
  (total_cost / cost_per_package) / packages_per_cake = 2 :=
by sorry

end cakes_to_make_l214_214272


namespace arithmetic_sequence_7th_term_l214_214148

theorem arithmetic_sequence_7th_term 
  (a d : ℝ)
  (n : ℕ)
  (h1 : 5 * a + 10 * d = 34)
  (h2 : 5 * a + 5 * (n - 1) * d = 146)
  (h3 : (n / 2 : ℝ) * (2 * a + (n - 1) * d) = 234) :
  a + 6 * d = 19 :=
by
  sorry

end arithmetic_sequence_7th_term_l214_214148


namespace total_shots_cost_l214_214741

def numDogs : ℕ := 3
def puppiesPerDog : ℕ := 4
def shotsPerPuppy : ℕ := 2
def costPerShot : ℕ := 5

theorem total_shots_cost : (numDogs * puppiesPerDog * shotsPerPuppy * costPerShot) = 120 := by
  sorry

end total_shots_cost_l214_214741


namespace jimmy_exams_l214_214879

theorem jimmy_exams (p l a : ℕ) (h_p : p = 50) (h_l : l = 5) (h_a : a = 5) (x : ℕ) :
  (20 * x - (l + a) ≥ p) ↔ (x ≥ 3) :=
by
  sorry

end jimmy_exams_l214_214879


namespace ab_calculation_l214_214588

noncomputable def triangle_area (a b : ℝ) : ℝ :=
  (1 / 2) * (4 / a) * (4 / b)

theorem ab_calculation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : triangle_area a b = 4) : a * b = 2 :=
by
  sorry

end ab_calculation_l214_214588


namespace common_chord_eq_l214_214511

theorem common_chord_eq :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 2*x + 8*y - 8 = 0) → (x^2 + y^2 - 4*x - 4*y - 2 = 0) →
    x + 2*y - 1 = 0 :=
by
  intros x y h1 h2
  sorry

end common_chord_eq_l214_214511


namespace problem_I_problem_II_l214_214431

noncomputable def f (x a : ℝ) : ℝ := 2 / x + a * Real.log x

theorem problem_I (a : ℝ) (h : a > 0) (h' : (2:ℝ) = (1 / (4 / a)) * (a^2) / 8):
  ∃ x : ℝ, f x a = f (1 / 2) a := sorry

theorem problem_II (a : ℝ) (h : a > 0) (h' : ∃ x : ℝ, f x a < 2) :
  (True : Prop) := sorry

end problem_I_problem_II_l214_214431


namespace min_value_trig_expression_l214_214063

theorem min_value_trig_expression : (∃ x : ℝ, 3 * Real.cos x - 4 * Real.sin x = -5) :=
by
  sorry

end min_value_trig_expression_l214_214063


namespace proof_by_contradiction_conditions_l214_214872

theorem proof_by_contradiction_conditions :
  ∀ (P Q : Prop),
    (∃ R : Prop, (R = ¬Q) ∧ (P → R) ∧ (R → P) ∧ (∀ T : Prop, (T = Q) → false)) →
    (∃ S : Prop, (S = ¬Q) ∧ P ∧ (∃ U : Prop, U) ∧ ¬Q) :=
by
  sorry

end proof_by_contradiction_conditions_l214_214872


namespace value_of_S_l214_214843

-- Defining the condition as an assumption
def one_third_one_eighth_S (S : ℝ) : Prop :=
  (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 120

-- The statement we need to prove
theorem value_of_S (S : ℝ) (h : one_third_one_eighth_S S) : S = 120 :=
by
  sorry

end value_of_S_l214_214843


namespace range_G_l214_214381

noncomputable def G (x : ℝ) : ℝ := |x + 2| - 2 * |x - 2|

theorem range_G : Set.range G = Set.Icc (-8 : ℝ) 8 := sorry

end range_G_l214_214381


namespace first_discount_percentage_l214_214178

theorem first_discount_percentage (original_price final_price : ℝ)
  (first_discount second_discount : ℝ) (h_orig : original_price = 200)
  (h_final : final_price = 144) (h_second_disc : second_discount = 0.20) :
  first_discount = 0.10 :=
by
  sorry

end first_discount_percentage_l214_214178


namespace man_overtime_hours_correctness_l214_214376

def man_worked_overtime_hours (r h_r t : ℕ): ℕ :=
  let regular_pay := r * h_r
  let overtime_pay := t - regular_pay
  let overtime_rate := 2 * r
  overtime_pay / overtime_rate

theorem man_overtime_hours_correctness : man_worked_overtime_hours 3 40 186 = 11 := by
  sorry

end man_overtime_hours_correctness_l214_214376


namespace ratio_removing_middle_digit_l214_214699

theorem ratio_removing_middle_digit 
  (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9) 
  (hb : 0 ≤ b ∧ b ≤ 9) 
  (hc : 0 ≤ c ∧ c ≤ 9)
  (h1 : 10 * b + c = 8 * a) 
  (h2 : 10 * a + b = 8 * c) : 
  (10 * a + c) / b = 17 :=
by sorry

end ratio_removing_middle_digit_l214_214699


namespace total_tiles_l214_214264

/-- A square-shaped floor is covered with congruent square tiles. 
If the total number of tiles on the two diagonals is 88 and the floor 
forms a perfect square with an even side length, then the number of tiles 
covering the floor is 1936. -/
theorem total_tiles (n : ℕ) (hn_even : n % 2 = 0) (h_diag : 2 * n = 88) : n^2 = 1936 := 
by 
  sorry

end total_tiles_l214_214264


namespace squared_difference_l214_214728

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l214_214728


namespace jack_correct_percentage_l214_214202

theorem jack_correct_percentage (y : ℝ) (h : y ≠ 0) :
  ((8 * y - (2 * y - 3)) / (8 * y)) * 100 = 75 + (75 / (2 * y)) :=
by
  sorry

end jack_correct_percentage_l214_214202


namespace remaining_walking_time_is_30_l214_214252

-- Define all the given conditions
def total_distance_to_store : ℝ := 2.5
def distance_already_walked : ℝ := 1.0
def time_per_mile : ℝ := 20.0

-- Define the target remaining walking time
def remaining_distance : ℝ := total_distance_to_store - distance_already_walked
def remaining_time : ℝ := remaining_distance * time_per_mile

-- Prove the remaining walking time is 30 minutes
theorem remaining_walking_time_is_30 : remaining_time = 30 :=
by
  -- Formal proof would go here using corresponding Lean tactics
  sorry

end remaining_walking_time_is_30_l214_214252


namespace dozens_in_each_box_l214_214988

theorem dozens_in_each_box (boxes total_mangoes : ℕ) (h1 : boxes = 36) (h2 : total_mangoes = 4320) :
  (total_mangoes / 12) / boxes = 10 :=
by
  -- The proof will go here.
  sorry

end dozens_in_each_box_l214_214988


namespace discriminant_of_quadratic_l214_214469

-- Define the quadratic equation coefficients
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := 1/2

-- Define the discriminant function
def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

-- State the theorem
theorem discriminant_of_quadratic :
  discriminant a b c = 81 / 4 :=
by
  -- We provide the result of the computation directly
  sorry

end discriminant_of_quadratic_l214_214469


namespace cost_price_of_article_l214_214892

variable (C : ℝ)
variable (h1 : (0.18 * C - 0.09 * C = 72))

theorem cost_price_of_article : C = 800 :=
by
  sorry

end cost_price_of_article_l214_214892


namespace count_non_integer_angles_l214_214304

open Int

def interior_angle (n : ℕ) : ℕ := 180 * (n - 2) / n

def is_integer_angle (n : ℕ) : Prop := 180 * (n - 2) % n = 0

theorem count_non_integer_angles : ∃ (count : ℕ), count = 2 ∧ ∀ n, 3 ≤ n ∧ n < 12 → is_integer_angle n ↔ ¬ (count = count + 1) :=
sorry

end count_non_integer_angles_l214_214304


namespace bob_daily_earnings_l214_214910

-- Define Sally's daily earnings
def Sally_daily_earnings : ℝ := 6

-- Define the total savings after a year for both Sally and Bob
def total_savings : ℝ := 1825

-- Define the number of days in a year
def days_in_year : ℝ := 365

-- Define Bob's daily earnings
variable (B : ℝ)

-- Define the proof statement
theorem bob_daily_earnings : (3 + B / 2) * days_in_year = total_savings → B = 4 :=
by
  sorry

end bob_daily_earnings_l214_214910


namespace pencils_left_l214_214828

theorem pencils_left (initial_pencils : ℕ := 79) (pencils_taken : ℕ := 4) : initial_pencils - pencils_taken = 75 :=
by
  sorry

end pencils_left_l214_214828


namespace fraction_zero_condition_l214_214946

theorem fraction_zero_condition (x : ℝ) (h : (abs x - 2) / (2 - x) = 0) : x = -2 :=
by
  sorry

end fraction_zero_condition_l214_214946


namespace find_rectangle_area_l214_214354

noncomputable def rectangle_area (a b : ℕ) : ℕ :=
  a * b

theorem find_rectangle_area (a b : ℕ) :
  (5 : ℚ) / 8 = (a : ℚ) / b ∧ (a + 6) * (b + 6) - a * b = 114 ∧ a + b = 13 →
  rectangle_area a b = 40 :=
by
  sorry

end find_rectangle_area_l214_214354


namespace largest_n_unique_k_l214_214505

theorem largest_n_unique_k : ∃! (n : ℕ), ∃ (k : ℤ),
  (7 / 16 : ℚ) < (n : ℚ) / (n + k : ℚ) ∧ (n : ℚ) / (n + k : ℚ) < (8 / 17 : ℚ) ∧ n = 112 := 
sorry

end largest_n_unique_k_l214_214505


namespace max_value_of_a_l214_214059

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y ≤ f x

theorem max_value_of_a
  (odd_f : odd_function f)
  (decr_f : decreasing_function f)
  (h : ∀ x : ℝ, f (Real.cos (2 * x) + Real.sin x) + f (Real.sin x - a) ≤ 0) :
  a ≤ -3 :=
sorry

end max_value_of_a_l214_214059


namespace sub_three_five_l214_214823

theorem sub_three_five : 3 - 5 = -2 := 
by 
  sorry

end sub_three_five_l214_214823


namespace initial_chocolate_bars_l214_214997

theorem initial_chocolate_bars (B : ℕ) 
  (H1 : Thomas_and_friends_take = B / 4)
  (H2 : One_friend_returns_5 = Thomas_and_friends_take - 5)
  (H3 : Piper_takes = Thomas_and_friends_take - 5 - 5)
  (H4 : Remaining_bars = B - Thomas_and_friends_take - Piper_takes)
  (H5 : Remaining_bars = 110) :
  B = 190 := 
sorry

end initial_chocolate_bars_l214_214997


namespace distance_between_hyperbola_vertices_l214_214931

theorem distance_between_hyperbola_vertices :
  ∀ (x y : ℝ), (x^2 / 121 - y^2 / 49 = 1) → (22 = 2 * 11) :=
by
  sorry

end distance_between_hyperbola_vertices_l214_214931


namespace albums_total_l214_214071

noncomputable def miriam_albums (katrina_albums : ℕ) := 5 * katrina_albums
noncomputable def katrina_albums (bridget_albums : ℕ) := 6 * bridget_albums
noncomputable def bridget_albums (adele_albums : ℕ) := adele_albums - 15
noncomputable def total_albums (adele_albums : ℕ) (bridget_albums : ℕ) (katrina_albums : ℕ) (miriam_albums : ℕ) := 
  adele_albums + bridget_albums + katrina_albums + miriam_albums

theorem albums_total (adele_has_30 : adele_albums = 30) : 
  total_albums 30 (bridget_albums 30) (katrina_albums (bridget_albums 30)) (miriam_albums (katrina_albums (bridget_albums 30))) = 585 := 
by 
  -- In Lean, replace the term 'sorry' below with the necessary steps to finish the proof
  sorry

end albums_total_l214_214071


namespace correct_equation_l214_214790

theorem correct_equation (x : ℝ) : 3 * x + 20 = 4 * x - 25 :=
by sorry

end correct_equation_l214_214790


namespace polygon_a_largest_area_l214_214220

open Real

/-- Lean 4 statement to prove that Polygon A has the largest area among the given polygons -/
theorem polygon_a_largest_area :
  let area_polygon_a := 4 + 2 * (1 / 2 * 2 * 2)
  let area_polygon_b := 3 + 3 * (1 / 2 * 1 * 1)
  let area_polygon_c := 6
  let area_polygon_d := 5 + (1 / 2) * π * 1^2
  let area_polygon_e := 7
  area_polygon_a > area_polygon_b ∧
  area_polygon_a > area_polygon_c ∧
  area_polygon_a > area_polygon_d ∧
  area_polygon_a > area_polygon_e :=
by
  let area_polygon_a := 4 + 2 * (1 / 2 * 2 * 2)
  let area_polygon_b := 3 + 3 * (1 / 2 * 1 * 1)
  let area_polygon_c := 6
  let area_polygon_d := 5 + (1 / 2) * π * 1^2
  let area_polygon_e := 7
  sorry

end polygon_a_largest_area_l214_214220


namespace range_of_m_l214_214473

noncomputable def proposition (m : ℝ) : Prop := ∀ x : ℝ, 4^x - 2^(x + 1) + m = 0

theorem range_of_m (m : ℝ) (h : ¬¬proposition m) : m ≤ 1 :=
by
  sorry

end range_of_m_l214_214473


namespace simon_students_l214_214165

theorem simon_students (S L : ℕ) (h1 : S = 4 * L) (h2 : S + L = 2500) : S = 2000 :=
by {
  sorry
}

end simon_students_l214_214165


namespace operation_results_in_m4_l214_214069

variable (m : ℤ)

theorem operation_results_in_m4 :
  (-m^2)^2 = m^4 :=
sorry

end operation_results_in_m4_l214_214069


namespace div_sqrt3_mul_inv_sqrt3_eq_one_l214_214329

theorem div_sqrt3_mul_inv_sqrt3_eq_one :
  (3 / Real.sqrt 3) * (1 / Real.sqrt 3) = 1 :=
by
  sorry

end div_sqrt3_mul_inv_sqrt3_eq_one_l214_214329


namespace factorize_expression_l214_214878

theorem factorize_expression (x y : ℝ) : 
  x^3 - x*y^2 = x * (x + y) * (x - y) :=
sorry

end factorize_expression_l214_214878


namespace factor_x4_plus_64_l214_214557

theorem factor_x4_plus_64 (x : ℝ) : 
  (x^4 + 64) = (x^2 - 4 * x + 8) * (x^2 + 4 * x + 8) :=
sorry

end factor_x4_plus_64_l214_214557


namespace total_flags_l214_214027

theorem total_flags (x : ℕ) (hx1 : 4 * x + 20 > 8 * (x - 1)) (hx2 : 4 * x + 20 < 8 * x) : 4 * 6 + 20 = 44 :=
by sorry

end total_flags_l214_214027


namespace cube_side_length_eq_three_l214_214407

theorem cube_side_length_eq_three (n : ℕ) (h1 : 6 * n^2 = 6 * n^3 / 3) : n = 3 := by
  -- The proof is omitted as per instructions, we use sorry to skip it.
  sorry

end cube_side_length_eq_three_l214_214407


namespace slope_of_line_l214_214744

theorem slope_of_line (x y : ℝ) : (∃ (m b : ℝ), (3 * y + 2 * x = 12) ∧ (m = -2 / 3) ∧ (y = m * x + b)) :=
sorry

end slope_of_line_l214_214744


namespace female_athletes_in_sample_l214_214914

theorem female_athletes_in_sample (M F S : ℕ) (hM : M = 56) (hF : F = 42) (hS : S = 28) :
  (F * (S / (M + F))) = 12 :=
by
  rw [hM, hF, hS]
  norm_num
  sorry

end female_athletes_in_sample_l214_214914


namespace polynomial_divisible_by_24_l214_214954

-- Defining the function
def f (n : ℕ) : ℕ :=
n^4 + 2*n^3 + 11*n^2 + 10*n

-- Statement of the theorem
theorem polynomial_divisible_by_24 (n : ℕ) (h : n > 0) : f n % 24 = 0 :=
sorry

end polynomial_divisible_by_24_l214_214954


namespace sum_of_GCF_and_LCM_l214_214733

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l214_214733


namespace MarthaEndBlocks_l214_214406

theorem MarthaEndBlocks (start_blocks found_blocks total_blocks : ℕ) 
  (h₁ : start_blocks = 11)
  (h₂ : found_blocks = 129) : 
  total_blocks = 140 :=
by
  sorry

end MarthaEndBlocks_l214_214406


namespace value_of_a_l214_214359

theorem value_of_a (a : ℝ) (h : 3 ∈ ({1, a, a - 2} : Set ℝ)) : a = 5 :=
by
  sorry

end value_of_a_l214_214359


namespace original_price_of_wand_l214_214147

-- Definitions as per the conditions
def price_paid (paid : Real) := paid = 8
def fraction_of_original (fraction : Real) := fraction = 1 / 8

-- Question and correct answer put as a theorem to prove
theorem original_price_of_wand (paid : Real) (fraction : Real) 
  (h1 : price_paid paid) (h2 : fraction_of_original fraction) : 
  (paid / fraction = 64) := 
by
  -- This 'sorry' indicates where the actual proof would go.
  sorry

end original_price_of_wand_l214_214147


namespace geometric_seq_a6_l214_214204

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q, ∀ n, a (n + 1) = a n * q

theorem geometric_seq_a6 {a : ℕ → ℝ} (h : geometric_sequence a) (h1 : a 1 * a 3 = 4) (h2 : a 4 = 4) : a 6 = 8 :=
sorry

end geometric_seq_a6_l214_214204


namespace length_of_bridge_l214_214756

theorem length_of_bridge (length_train : ℝ) (speed_kmh : ℝ) (time_sec : ℝ) (speed_ms : ℝ) (total_distance : ℝ) (bridge_length : ℝ) :
  length_train = 160 →
  speed_kmh = 45 →
  time_sec = 30 →
  speed_ms = 45 * (1000 / 3600) →
  total_distance = speed_ms * time_sec →
  bridge_length = total_distance - length_train →
  bridge_length = 215 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end length_of_bridge_l214_214756


namespace minimum_period_l214_214391

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem minimum_period (ω : ℝ) (hω : ω > 0) 
  (h : ∀ x1 x2 : ℝ, |f ω x1 - f ω x2| = 2 → |x1 - x2| = Real.pi / 2) :
  ∃ T > 0, ∀ x : ℝ, f ω (x + T) = f ω x ∧ T = Real.pi := sorry

end minimum_period_l214_214391


namespace allocation_schemes_l214_214200

theorem allocation_schemes (students factories: ℕ) (has_factory_a: Prop) (A_must_have_students: has_factory_a): students = 3 → factories = 4 → has_factory_a → (∃ n: ℕ, n = 4^3 - 3^3 ∧ n = 37) :=
by try { sorry }

end allocation_schemes_l214_214200


namespace problem_solution_l214_214499

theorem problem_solution :
  ∃ a b c d : ℚ, 
  4 * a + 2 * b + 5 * c + 8 * d = 67 ∧ 
  4 * (d + c) = b ∧ 
  2 * b + 3 * c = a ∧ 
  c + 1 = d ∧ 
  a * b * c * d = (1201 * 572 * 19 * 124) / (105 ^ 4) :=
sorry

end problem_solution_l214_214499


namespace choir_minimum_members_l214_214227

theorem choir_minimum_members (n : ℕ) :
  (∃ k1, n = 8 * k1) ∧ (∃ k2, n = 9 * k2) ∧ (∃ k3, n = 10 * k3) → n = 360 :=
by
  sorry

end choir_minimum_members_l214_214227


namespace cos_arccos_minus_arctan_eq_l214_214919

noncomputable def cos_arccos_minus_arctan: Real :=
  Real.cos (Real.arccos (4 / 5) - Real.arctan (1 / 2))

theorem cos_arccos_minus_arctan_eq : cos_arccos_minus_arctan = (11 * Real.sqrt 5) / 25 := by
  sorry

end cos_arccos_minus_arctan_eq_l214_214919


namespace gina_minutes_of_netflix_l214_214457

-- Define the conditions given in the problem
def gina_chooses_three_times_as_often (g s : ℕ) : Prop :=
  g = 3 * s

def total_shows_watched (g s : ℕ) : Prop :=
  g + s = 24

def duration_per_show : ℕ := 50

-- The theorem that encapsulates the problem statement and the correct answer
theorem gina_minutes_of_netflix (g s : ℕ) (h1 : gina_chooses_three_times_as_often g s) 
    (h2 : total_shows_watched g s) :
    g * duration_per_show = 900 :=
by
  sorry

end gina_minutes_of_netflix_l214_214457


namespace cannot_represent_1986_as_sum_of_squares_of_6_odd_integers_l214_214807

theorem cannot_represent_1986_as_sum_of_squares_of_6_odd_integers
  (a1 a2 a3 a4 a5 a6 : ℤ)
  (h1 : a1 % 2 = 1) 
  (h2 : a2 % 2 = 1) 
  (h3 : a3 % 2 = 1) 
  (h4 : a4 % 2 = 1) 
  (h5 : a5 % 2 = 1) 
  (h6 : a6 % 2 = 1) : 
  ¬ (1986 = a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2) := 
by 
  sorry

end cannot_represent_1986_as_sum_of_squares_of_6_odd_integers_l214_214807


namespace complex_numbers_equation_l214_214556

theorem complex_numbers_equation {a b : ℂ} (h : (a + b) / (a - b) - (a - b) / (a + b) = 0) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = 0 := 
by sorry

end complex_numbers_equation_l214_214556


namespace exponent_problem_l214_214795

theorem exponent_problem : (-1 : ℝ)^2003 / (-1 : ℝ)^2004 = -1 := by
  sorry

end exponent_problem_l214_214795


namespace calculate_fg1_l214_214021

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 + 2

theorem calculate_fg1 : f (g 1) = -1 :=
by {
  sorry
}

end calculate_fg1_l214_214021


namespace hyperbola_asymptotes_l214_214197

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := (y^2 / 4) - (x^2 / 9) = 1

-- Define the standard form of hyperbola asymptotes equations
def asymptotes_eq (x y : ℝ) : Prop := 2 * x + 3 * y = 0 ∨ 2 * x - 3 * y = 0

-- The final proof statement
theorem hyperbola_asymptotes (x y : ℝ) (h : hyperbola_eq x y) : asymptotes_eq x y :=
    sorry

end hyperbola_asymptotes_l214_214197


namespace range_of_m_l214_214072

theorem range_of_m (m : ℝ) :
  (∃ (m : ℝ), (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 + 1 = 0 ∧ x2^2 + m * x2 + 1 = 0) ∧ 
  (∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≤ 0)) ↔ (m ≤ 1 ∨ m ≥ 3 ∨ m < -2) :=
by
  sorry

end range_of_m_l214_214072


namespace distance_point_parabola_focus_l214_214480

theorem distance_point_parabola_focus (P : ℝ × ℝ) (x y : ℝ) (hP : P = (3, y)) (h_parabola : y^2 = 4 * 3) :
    dist P (0, -1) = 4 :=
by
  sorry

end distance_point_parabola_focus_l214_214480


namespace paint_cans_for_25_rooms_l214_214972

theorem paint_cans_for_25_rooms (cans rooms : ℕ) (H1 : cans * 30 = rooms) (H2 : cans * 25 = rooms - 5 * cans) :
  cans = 15 :=
by
  sorry

end paint_cans_for_25_rooms_l214_214972


namespace find_f1_l214_214166

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x
def odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem find_f1 (f : ℝ → ℝ)
  (h_periodic : periodic f 2)
  (h_odd : odd f) :
  f 1 = 0 :=
sorry

end find_f1_l214_214166


namespace herd_total_cows_l214_214085

theorem herd_total_cows (n : ℕ) (h1 : (1 / 3 : ℚ) * n + (1 / 5 : ℚ) * n + (1 / 6 : ℚ) * n + 19 = n) : n = 63 :=
sorry

end herd_total_cows_l214_214085


namespace total_hovering_time_is_24_hours_l214_214338

-- Define the initial conditions
def mountain_time_day1 : ℕ := 3
def central_time_day1 : ℕ := 4
def eastern_time_day1 : ℕ := 2

-- Define the additional time hovered in each zone on the second day
def additional_time_per_zone_day2 : ℕ := 2

-- Calculate the total time spent on each day
def total_time_day1 : ℕ := mountain_time_day1 + central_time_day1 + eastern_time_day1
def total_additional_time_day2 : ℕ := 3 * additional_time_per_zone_day2 -- there are three zones
def total_time_day2 : ℕ := total_time_day1 + total_additional_time_day2

-- Calculate the total time over the two days
def total_time_two_days : ℕ := total_time_day1 + total_time_day2

-- Prove that the total time over the two days is 24 hours
theorem total_hovering_time_is_24_hours : total_time_two_days = 24 := by
  sorry

end total_hovering_time_is_24_hours_l214_214338


namespace aaron_weekly_earnings_l214_214805

def minutes_worked_monday : ℕ := 90
def minutes_worked_tuesday : ℕ := 40
def minutes_worked_wednesday : ℕ := 135
def minutes_worked_thursday : ℕ := 45
def minutes_worked_friday : ℕ := 60
def minutes_worked_saturday1 : ℕ := 90
def minutes_worked_saturday2 : ℕ := 75
def hourly_rate : ℕ := 4

def total_minutes_worked : ℕ :=
  minutes_worked_monday + 
  minutes_worked_tuesday + 
  minutes_worked_wednesday +
  minutes_worked_thursday + 
  minutes_worked_friday +
  minutes_worked_saturday1 + 
  minutes_worked_saturday2

def total_hours_worked : ℕ := total_minutes_worked / 60

def total_earnings : ℕ := total_hours_worked * hourly_rate

theorem aaron_weekly_earnings : total_earnings = 36 := by 
  sorry -- The proof is omitted.

end aaron_weekly_earnings_l214_214805


namespace polygon_largest_area_l214_214341

-- Definition for the area calculation of each polygon based on given conditions
def area_A : ℝ := 3 * 1 + 2 * 0.5
def area_B : ℝ := 6 * 1
def area_C : ℝ := 4 * 1 + 3 * 0.5
def area_D : ℝ := 5 * 1 + 1 * 0.5
def area_E : ℝ := 7 * 1

-- Theorem stating the problem
theorem polygon_largest_area :
  area_E = max (max (max (max area_A area_B) area_C) area_D) area_E :=
by
  -- The proof steps would go here.
  sorry

end polygon_largest_area_l214_214341


namespace find_product_xy_l214_214422

theorem find_product_xy (x y : ℝ) 
  (h1 : (9 + 10 + 11 + x + y) / 5 = 10)
  (h2 : ((9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2) / 5 = 4) :
  x * y = 191 :=
sorry

end find_product_xy_l214_214422


namespace decimal_division_l214_214847

theorem decimal_division : (0.05 : ℝ) / (0.005 : ℝ) = 10 := 
by 
  sorry

end decimal_division_l214_214847


namespace arithmetic_sequence_general_term_and_sum_max_l214_214926

-- Definitions and conditions
def a1 : ℤ := 4
def d : ℤ := -2
def a (n : ℕ) : ℤ := a1 + (n - 1) * d
def Sn (n : ℕ) : ℤ := n * (a1 + (a n)) / 2

-- Prove the general term formula and maximum value
theorem arithmetic_sequence_general_term_and_sum_max :
  (∀ n, a n = -2 * n + 6) ∧ (∃ n, Sn n = 6) :=
by
  sorry

end arithmetic_sequence_general_term_and_sum_max_l214_214926


namespace gcd_420_135_l214_214890

theorem gcd_420_135 : Nat.gcd 420 135 = 15 := by
  sorry

end gcd_420_135_l214_214890


namespace cost_of_one_lesson_l214_214667

-- Define the conditions
def total_cost_for_lessons : ℝ := 360
def total_hours_of_lessons : ℝ := 18
def duration_of_one_lesson : ℝ := 1.5

-- Define the theorem statement
theorem cost_of_one_lesson :
  (total_cost_for_lessons / total_hours_of_lessons) * duration_of_one_lesson = 30 := by
  -- Proof goes here
  sorry

end cost_of_one_lesson_l214_214667


namespace football_team_throwers_l214_214291

theorem football_team_throwers
    (total_players : ℕ)
    (right_handed_players : ℕ)
    (one_third : ℚ)
    (number_throwers : ℕ)
    (number_non_throwers : ℕ)
    (right_handed_non_throwers : ℕ)
    (left_handed_non_throwers : ℕ)
    (h1 : total_players = 70)
    (h2 : right_handed_players = 63)
    (h3 : one_third = 1 / 3)
    (h4 : number_non_throwers = total_players - number_throwers)
    (h5 : right_handed_non_throwers = right_handed_players - number_throwers)
    (h6 : left_handed_non_throwers = one_third * number_non_throwers)
    (h7 : 2 * left_handed_non_throwers = right_handed_non_throwers)
    : number_throwers = 49 := 
by
  sorry

end football_team_throwers_l214_214291


namespace rem_value_is_correct_l214_214470

def rem (x y : ℚ) : ℚ :=
  x - y * (Int.floor (x / y))

theorem rem_value_is_correct : rem (-5/9) (7/3) = 16/9 := by
  sorry

end rem_value_is_correct_l214_214470


namespace spent_amount_l214_214233

def initial_amount : ℕ := 15
def final_amount : ℕ := 11

theorem spent_amount : initial_amount - final_amount = 4 :=
by
  sorry

end spent_amount_l214_214233


namespace shanna_initial_tomato_plants_l214_214846

theorem shanna_initial_tomato_plants (T : ℕ) 
  (h1 : 56 = (T / 2) * 7 + 2 * 7 + 3 * 7) : 
  T = 6 :=
by sorry

end shanna_initial_tomato_plants_l214_214846


namespace triple_apply_l214_214883

def f (x : ℝ) : ℝ := 5 * x - 4

theorem triple_apply : f (f (f 2)) = 126 :=
by
  rw [f, f, f]
  sorry

end triple_apply_l214_214883


namespace find_c_l214_214168

noncomputable def func_condition (f : ℝ → ℝ) (c : ℝ) :=
  ∀ x y : ℝ, (f x + 1) * (f y + 1) = f (x + y) + f (x * y + c)

theorem find_c :
  ∃ c : ℝ, ∀ (f : ℝ → ℝ), func_condition f c → (c = 1 ∨ c = -1) :=
sorry

end find_c_l214_214168


namespace last_digit_two_power_2015_l214_214358

/-- The last digit of powers of 2 cycles through 2, 4, 8, 6. Therefore, the last digit of 2^2015 is the same as 2^3, which is 8. -/
theorem last_digit_two_power_2015 : (2^2015) % 10 = 8 :=
by sorry

end last_digit_two_power_2015_l214_214358


namespace range_of_b_l214_214822

theorem range_of_b (a b : ℝ) (h1 : a ≠ 0) (h2 : a * b^2 > a) (h3 : a > a * b) : b < -1 :=
sorry

end range_of_b_l214_214822


namespace sara_sent_letters_l214_214380

theorem sara_sent_letters (J : ℕ)
  (h1 : 9 + 3 * J + J = 33) : J = 6 :=
by
  sorry

end sara_sent_letters_l214_214380


namespace mr_blue_carrots_l214_214629

theorem mr_blue_carrots :
  let steps_length := 3 -- length of each step in feet
  let garden_length_steps := 25 -- length of garden in steps
  let garden_width_steps := 35 -- width of garden in steps
  let length_feet := garden_length_steps * steps_length -- length of garden in feet
  let width_feet := garden_width_steps * steps_length -- width of garden in feet
  let area_feet2 := length_feet * width_feet -- area of garden in square feet
  let yield_rate := 3 / 4 -- yield rate of carrots in pounds per square foot
  let expected_yield := area_feet2 * yield_rate -- expected yield in pounds
  expected_yield = 5906.25
:= by
  sorry

end mr_blue_carrots_l214_214629


namespace exists_marked_sum_of_three_l214_214337

theorem exists_marked_sum_of_three (s : Finset ℕ) (h₀ : s.card = 22) (h₁ : ∀ x ∈ s, x ≤ 30) :
  ∃ a ∈ s, ∃ b ∈ s, ∃ c ∈ s, ∃ d ∈ s, a = b + c + d :=
by
  sorry

end exists_marked_sum_of_three_l214_214337


namespace height_of_wooden_box_l214_214246

theorem height_of_wooden_box 
  (height : ℝ)
  (h₁ : ∀ (length width : ℝ), length = 8 ∧ width = 10)
  (h₂ : ∀ (small_length small_width small_height : ℕ), small_length = 4 ∧ small_width = 5 ∧ small_height = 6)
  (h₃ : ∀ (num_boxes : ℕ), num_boxes = 4000000) :
  height = 6 := 
sorry

end height_of_wooden_box_l214_214246


namespace kaleb_money_earned_l214_214831

-- Definitions based on the conditions
def total_games : ℕ := 10
def non_working_games : ℕ := 8
def price_per_game : ℕ := 6

-- Calculate the number of working games
def working_games : ℕ := total_games - non_working_games

-- Calculate the total money earned by Kaleb
def money_earned : ℕ := working_games * price_per_game

-- The theorem to prove
theorem kaleb_money_earned : money_earned = 12 := by sorry

end kaleb_money_earned_l214_214831


namespace percentage_of_women_in_study_group_l214_214261

theorem percentage_of_women_in_study_group
  (W : ℝ) -- percentage of women in decimal form
  (h1 : 0 < W ∧ W ≤ 1) -- percentage of women should be between 0 and 1
  (h2 : 0.4 * W = 0.32) -- 40 percent of women are lawyers, and probability is 0.32
  : W = 0.8 :=
  sorry

end percentage_of_women_in_study_group_l214_214261


namespace grunters_at_least_4_wins_l214_214607

noncomputable def grunters_probability : ℚ :=
  let p_win := 3 / 5
  let p_loss := 2 / 5
  let p_4_wins := 5 * (p_win^4) * (p_loss)
  let p_5_wins := p_win^5
  p_4_wins + p_5_wins

theorem grunters_at_least_4_wins :
  grunters_probability = 1053 / 3125 :=
by sorry

end grunters_at_least_4_wins_l214_214607


namespace pair_a_n_uniq_l214_214158

theorem pair_a_n_uniq (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n) (h_eq : 3^n = a^2 - 16) : a = 5 ∧ n = 2 := 
by 
  sorry

end pair_a_n_uniq_l214_214158


namespace series_sum_correct_l214_214067

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l214_214067


namespace molecular_weight_of_ammonium_bromide_l214_214683

-- Define the atomic weights for the elements.
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90

-- Define the molecular weight of ammonium bromide based on its formula NH4Br.
def molecular_weight_NH4Br : ℝ := (1 * atomic_weight_N) + (4 * atomic_weight_H) + (1 * atomic_weight_Br)

-- State the theorem that the molecular weight of ammonium bromide is approximately 97.95 g/mol.
theorem molecular_weight_of_ammonium_bromide :
  molecular_weight_NH4Br = 97.95 :=
by
  -- The following line is a placeholder to mark the theorem as correct without proving it.
  sorry

end molecular_weight_of_ammonium_bromide_l214_214683


namespace number_of_white_balls_l214_214401

theorem number_of_white_balls (x : ℕ) (h : (5 : ℚ) / (5 + x) = 1 / 4) : x = 15 :=
by 
  sorry

end number_of_white_balls_l214_214401


namespace problem1_l214_214287

theorem problem1 (α : ℝ) (h : Real.tan α = 2) :
  Real.sin (Real.pi / 2 - α)^2 + 3 * Real.sin (α + Real.pi) * Real.sin (α + Real.pi / 2) = -1 :=
sorry

end problem1_l214_214287


namespace calculate_fg_l214_214512

def f (x : ℝ) : ℝ := x - 4

def g (x : ℝ) : ℝ := x^2 + 5

theorem calculate_fg : f (g (-3)) = 10 := by
  sorry

end calculate_fg_l214_214512


namespace find_sum_of_abc_l214_214437

theorem find_sum_of_abc
  (a b c x y : ℕ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a^2 + b^2 + c^2 = 2011)
  (h3 : Nat.gcd a (Nat.gcd b c) = x)
  (h4 : Nat.lcm a (Nat.lcm b c) = y)
  (h5 : x + y = 388)
  :
  a + b + c = 61 :=
sorry

end find_sum_of_abc_l214_214437


namespace triangle_XYZ_XY2_XZ2_difference_l214_214703

-- Define the problem parameters and conditions
def YZ : ℝ := 10
def XM : ℝ := 6
def midpoint_YZ (M : ℝ) := 2 * M = YZ

-- The main theorem to be proved
theorem triangle_XYZ_XY2_XZ2_difference :
  ∀ (XY XZ : ℝ), 
  (∀ (M : ℝ), midpoint_YZ M) →
  ((∃ (x : ℝ), (0 ≤ x ∧ x ≤ 10) ∧ XY^2 + XZ^2 = 2 * x^2 - 20 * x + 2 * (11 * x - x^2 - 11) + 100)) →
  (120 - 100 = 20) :=
by
  sorry

end triangle_XYZ_XY2_XZ2_difference_l214_214703


namespace log_of_y_pow_x_eq_neg4_l214_214456

theorem log_of_y_pow_x_eq_neg4 (x y : ℝ) (h : |x - 8 * y| + (4 * y - 1) ^ 2 = 0) : 
  Real.logb 2 (y ^ x) = -4 :=
sorry

end log_of_y_pow_x_eq_neg4_l214_214456


namespace walking_speed_l214_214799

theorem walking_speed (x : ℝ) (h1 : 20 / x = 40 / (x + 5)) : x + 5 = 10 :=
  by
  sorry

end walking_speed_l214_214799


namespace seventh_rack_dvds_l214_214390

def rack_dvds : ℕ → ℕ
| 0 => 3
| 1 => 4
| n + 2 => ((rack_dvds (n + 1)) - (rack_dvds n)) * 2 + (rack_dvds (n + 1))

theorem seventh_rack_dvds : rack_dvds 6 = 66 := 
by
  sorry

end seventh_rack_dvds_l214_214390


namespace gain_percent_l214_214869

theorem gain_percent (C S : ℝ) (h : 50 * C = 15 * S) :
  (S > C) →
  ((S - C) / C * 100) = 233.33 := 
sorry

end gain_percent_l214_214869


namespace percent_decrease_in_hours_l214_214830

theorem percent_decrease_in_hours (W H : ℝ) 
  (h1 : W > 0) 
  (h2 : H > 0)
  (new_wage : ℝ := W * 1.25)
  (H_new : ℝ := H / 1.25)
  (total_income_same : W * H = new_wage * H_new) :
  ((H - H_new) / H) * 100 = 20 := 
by
  sorry

end percent_decrease_in_hours_l214_214830


namespace largest_number_among_options_l214_214424

def option_a : ℝ := -abs (-4)
def option_b : ℝ := 0
def option_c : ℝ := 1
def option_d : ℝ := -( -3)

theorem largest_number_among_options : 
  max (max option_a (max option_b option_c)) option_d = option_d := by
  sorry

end largest_number_among_options_l214_214424


namespace symmetric_point_x_axis_l214_214209

def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ := (M.1, -M.2)

theorem symmetric_point_x_axis :
  ∀ (M : ℝ × ℝ), M = (3, -4) → symmetric_point M = (3, 4) :=
by
  intros M h
  rw [h]
  dsimp [symmetric_point]
  congr
  sorry

end symmetric_point_x_axis_l214_214209


namespace total_students_in_lab_l214_214538

def total_workstations : Nat := 16
def workstations_for_2_students : Nat := 10
def students_per_workstation_2 : Nat := 2
def students_per_workstation_3 : Nat := 3

theorem total_students_in_lab :
  let workstations_with_2_students := workstations_for_2_students
  let workstations_with_3_students := total_workstations - workstations_for_2_students
  let students_in_2_student_workstations := workstations_with_2_students * students_per_workstation_2
  let students_in_3_student_workstations := workstations_with_3_students * students_per_workstation_3
  students_in_2_student_workstations + students_in_3_student_workstations = 38 :=
by
  sorry

end total_students_in_lab_l214_214538


namespace economy_class_seats_l214_214579

-- Definitions based on the conditions
def first_class_people : ℕ := 3
def business_class_people : ℕ := 22
def economy_class_fullness (E : ℕ) : ℕ := E / 2

-- Problem statement: Proving E == 50 given the conditions
theorem economy_class_seats :
  ∃ E : ℕ,  economy_class_fullness E = first_class_people + business_class_people → E = 50 :=
by
  sorry

end economy_class_seats_l214_214579


namespace standard_deviation_less_than_l214_214580

theorem standard_deviation_less_than:
  ∀ (μ σ : ℝ)
  (h1 : μ = 55)
  (h2 : μ - 3 * σ > 48),
  σ < 7 / 3 :=
by
  intros μ σ h1 h2
  sorry

end standard_deviation_less_than_l214_214580


namespace number_of_digits_if_million_place_l214_214851

theorem number_of_digits_if_million_place (n : ℕ) (h : n = 1000000) : 7 = 7 := by
  sorry

end number_of_digits_if_million_place_l214_214851


namespace option_c_is_always_odd_l214_214005

theorem option_c_is_always_odd (n : ℤ) : ∃ (q : ℤ), n^2 + n + 5 = 2*q + 1 := by
  sorry

end option_c_is_always_odd_l214_214005


namespace range_of_k_l214_214802

theorem range_of_k (k : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → (k + 2) * x1 - 1 > (k + 2) * x2 - 1) → k < -2 := by
  sorry

end range_of_k_l214_214802


namespace systematic_sampling_seventeenth_group_l214_214501

theorem systematic_sampling_seventeenth_group :
  ∀ (total_students : ℕ) (sample_size : ℕ) (first_number : ℕ) (interval : ℕ),
  total_students = 800 →
  sample_size = 50 →
  first_number = 8 →
  interval = total_students / sample_size →
  first_number + 16 * interval = 264 :=
by
  intros total_students sample_size first_number interval h1 h2 h3 h4
  sorry

end systematic_sampling_seventeenth_group_l214_214501


namespace real_roots_of_quadratic_l214_214844

theorem real_roots_of_quadratic (m : ℝ) : ((m - 2) ≠ 0 ∧ (-4 * m + 24) ≥ 0) → (m ≤ 6 ∧ m ≠ 2) := 
by 
  sorry

end real_roots_of_quadratic_l214_214844


namespace transformed_curve_l214_214783

theorem transformed_curve (x y : ℝ) :
  (∃ (x1 y1 : ℝ), x1 = 3*x ∧ y1 = 2*y ∧ (x1^2 / 9 + y1^2 / 4 = 1)) →
  x^2 + y^2 = 1 :=
by
  sorry

end transformed_curve_l214_214783


namespace no_common_points_l214_214379

theorem no_common_points 
  (x x_o y y_o : ℝ) 
  (h_parabola : y^2 = 4 * x) 
  (h_inside : y_o^2 < 4 * x_o) : 
  ¬ ∃ (x y : ℝ), y * y_o = 2 * (x + x_o) ∧ y^2 = 4 * x :=
by
  sorry

end no_common_points_l214_214379


namespace original_average_weight_l214_214573

-- Definitions from conditions
def original_team_size : ℕ := 7
def new_player1_weight : ℝ := 110
def new_player2_weight : ℝ := 60
def new_team_size := original_team_size + 2
def new_average_weight : ℝ := 106

-- Statement to prove
theorem original_average_weight (W : ℝ) :
  (7 * W + 110 + 60 = 9 * 106) → W = 112 := by
  sorry

end original_average_weight_l214_214573


namespace basketball_court_perimeter_l214_214393

variables {Width Length : ℕ}

def width := 17
def length := 31

def perimeter (width length : ℕ) := 2 * (length + width)

theorem basketball_court_perimeter : 
  perimeter width length = 96 :=
sorry

end basketball_court_perimeter_l214_214393


namespace problem1_correct_problem2_correct_l214_214350

noncomputable def problem1 : ℚ :=
  (1/2 - 5/9 + 7/12) * (-36)

theorem problem1_correct : problem1 = -19 := 
by 
  sorry

noncomputable def mixed_number (a : ℤ) (b : ℚ) : ℚ := a + b

noncomputable def problem2 : ℚ :=
  (mixed_number (-199) (24/25)) * 5

theorem problem2_correct : problem2 = -999 - 4/5 :=
by
  sorry

end problem1_correct_problem2_correct_l214_214350


namespace thief_speed_is_43_75_l214_214357

-- Given Information
def speed_owner : ℝ := 50
def time_head_start : ℝ := 0.5
def total_time_to_overtake : ℝ := 4

-- Question: What is the speed of the thief's car v?
theorem thief_speed_is_43_75 (v : ℝ) (hv : 4 * v = speed_owner * (total_time_to_overtake - time_head_start)) : v = 43.75 := 
by {
  -- The proof of this theorem is omitted as it is not required.
  sorry
}

end thief_speed_is_43_75_l214_214357


namespace balls_picking_l214_214369

theorem balls_picking (red_bag blue_bag : ℕ) (h_red : red_bag = 3) (h_blue : blue_bag = 5) : (red_bag * blue_bag = 15) :=
by
  sorry

end balls_picking_l214_214369


namespace number_is_18_l214_214923

theorem number_is_18 (x : ℝ) (h : (7 / 3) * x = 42) : x = 18 :=
sorry

end number_is_18_l214_214923


namespace julia_ink_containers_l214_214441

-- Definitions based on conditions
def total_posters : Nat := 60
def posters_remaining : Nat := 45
def lost_containers : Nat := 1

-- Required to be proven statement
theorem julia_ink_containers : 
  (total_posters - posters_remaining) = 15 → 
  posters_remaining / 15 = 3 := 
by 
  sorry

end julia_ink_containers_l214_214441


namespace intersect_x_axis_iff_k_le_4_l214_214884

theorem intersect_x_axis_iff_k_le_4 (k : ℝ) :
  (∃ x : ℝ, (k-3) * x^2 + 2 * x + 1 = 0) ↔ k ≤ 4 :=
sorry

end intersect_x_axis_iff_k_le_4_l214_214884


namespace time_for_q_to_complete_work_alone_l214_214330

theorem time_for_q_to_complete_work_alone (P Q : ℝ) (h1 : (1 / P) + (1 / Q) = 1 / 40) (h2 : (20 / P) + (12 / Q) = 1) : Q = 64 / 3 :=
by
  sorry

end time_for_q_to_complete_work_alone_l214_214330


namespace A_inter_B_empty_l214_214577

def setA : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def setB : Set ℝ := {x | Real.log x / Real.log 4 > 1/2}

theorem A_inter_B_empty : setA ∩ setB = ∅ := by
  sorry

end A_inter_B_empty_l214_214577


namespace express_x_in_terms_of_y_l214_214447

theorem express_x_in_terms_of_y (x y : ℝ) (h : 2 * x - 3 * y = 7) : x = 7 / 2 + 3 / 2 * y :=
by
  sorry

end express_x_in_terms_of_y_l214_214447


namespace vasya_petya_distance_l214_214998

theorem vasya_petya_distance :
  ∀ (D : ℝ), 
    (3 : ℝ) ≠ 0 → (6 : ℝ) ≠ 0 →
    ((D / 3) + (D / 6) = 2.5) →
    ((D / 6) + (D / 3) = 3.5) →
    D = 12 := 
by
  intros D h3 h6 h1 h2
  sorry

end vasya_petya_distance_l214_214998


namespace min_value_of_a_l214_214055

noncomputable def P (x : ℕ) : ℤ := sorry

def smallest_value_of_a (a : ℕ) : Prop :=
  a > 0 ∧
  (P 1 = a ∧ P 3 = a ∧ P 5 = a ∧ P 7 = a ∧ P 9 = a ∧
   P 2 = -a ∧ P 4 = -a ∧ P 6 = -a ∧ P 8 = -a ∧ P 10 = -a)

theorem min_value_of_a : ∃ a : ℕ, smallest_value_of_a a ∧ a = 6930 :=
sorry

end min_value_of_a_l214_214055


namespace jack_christina_speed_l214_214649

noncomputable def speed_of_jack_christina (d_jack_christina : ℝ) (v_lindy : ℝ) (d_lindy : ℝ) (relative_speed_factor : ℝ := 2) : ℝ :=
d_lindy * relative_speed_factor / d_jack_christina

theorem jack_christina_speed :
  speed_of_jack_christina 240 10 400 = 3 := by
  sorry

end jack_christina_speed_l214_214649


namespace statement_I_l214_214842

section Problem
variable (g : ℝ → ℝ)

-- Conditions
def cond1 : Prop := ∀ x : ℝ, g x > 0
def cond2 : Prop := ∀ a b : ℝ, g a * g b = g (a + 2 * b)

-- Statement I to be proved
theorem statement_I (h1 : cond1 g) (h2 : cond2 g) : g 0 = 1 :=
by
  -- Proof is omitted
  sorry
end Problem

end statement_I_l214_214842


namespace distance_light_300_years_eq_l214_214025

-- Define the constant distance light travels in one year
def distance_light_year : ℕ := 9460800000000

-- Define the time period in years
def time_period : ℕ := 300

-- Define the expected distance light travels in 300 years in scientific notation
def expected_distance : ℝ := 28382 * 10^13

-- The theorem to prove
theorem distance_light_300_years_eq :
  (distance_light_year * time_period) = 2838200000000000 :=
by
  sorry

end distance_light_300_years_eq_l214_214025


namespace fraction_simplification_l214_214758

theorem fraction_simplification :
  (1722 ^ 2 - 1715 ^ 2) / (1729 ^ 2 - 1708 ^ 2) = 1 / 3 := by
  sorry

end fraction_simplification_l214_214758


namespace product_of_numbers_eq_zero_l214_214353

theorem product_of_numbers_eq_zero (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 1) 
  (h3 : a^3 + b^3 + c^3 = 1) : 
  a * b * c = 0 := 
by
  sorry

end product_of_numbers_eq_zero_l214_214353


namespace problem_statement_l214_214079

-- Assume F is a function defined such that given the point (4,4) is on the graph y = F(x)
def F : ℝ → ℝ := sorry

-- Hypothesis: (4, 4) is on the graph of y = F(x)
axiom H : F 4 = 4

-- We need to prove that F(4) = 4
theorem problem_statement : F 4 = 4 :=
by exact H

end problem_statement_l214_214079


namespace smallest_composite_no_prime_factors_below_15_correct_l214_214578

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l214_214578


namespace sum_of_coordinates_of_point_B_l214_214112

theorem sum_of_coordinates_of_point_B
  (A : ℝ × ℝ) (hA : A = (0, 0))
  (B : ℝ × ℝ) (hB : ∃ x : ℝ, B = (x, 3))
  (slope_AB : ∃ x : ℝ, (3 - 0)/(x - 0) = 3/4) :
  (∃ x : ℝ, B = (x, 3)) ∧ x + 3 = 7 :=
by
  sorry

end sum_of_coordinates_of_point_B_l214_214112


namespace problem1_problem2_l214_214450

variable (k : ℝ)

-- Definitions of proposition p and q
def p (k : ℝ) : Prop := ∀ x : ℝ, x^2 - k*x + 2*k + 5 ≥ 0

def q (k : ℝ) : Prop := (4 - k > 0) ∧ (1 - k < 0)

-- Theorem statements based on the proof problem
theorem problem1 (hq : q k) : 1 < k ∧ k < 4 :=
by sorry

theorem problem2 (hp_q : p k ∨ q k) (hp_and_q_false : ¬(p k ∧ q k)) : 
  (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10) :=
by sorry

end problem1_problem2_l214_214450


namespace company_employees_after_reduction_l214_214206

theorem company_employees_after_reduction :
  let original_number := 224.13793103448276
  let reduction := 0.13 * original_number
  let current_number := original_number - reduction
  current_number = 195 :=
by
  let original_number := 224.13793103448276
  let reduction := 0.13 * original_number
  let current_number := original_number - reduction
  sorry

end company_employees_after_reduction_l214_214206


namespace factor_expression_l214_214311

theorem factor_expression (y : ℝ) :
  5 * y * (y - 4) + 2 * (y - 4) = (5 * y + 2) * (y - 4) :=
by
  sorry

end factor_expression_l214_214311


namespace part_a_part_b_part_c_l214_214723

def quadradois (n : ℕ) : Prop :=
  ∃ (S1 S2 : ℕ), S1 ≠ S2 ∧ (S1 * S1 + S2 * S2 ≤ S1 * S1 + S2 * S2 + (n - 2))

theorem part_a : quadradois 6 := 
sorry

theorem part_b : quadradois 2015 := 
sorry

theorem part_c : ∀ (n : ℕ), n > 5 → quadradois n := 
sorry

end part_a_part_b_part_c_l214_214723


namespace min_value_of_sum_squares_on_circle_l214_214102

theorem min_value_of_sum_squares_on_circle :
  ∃ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 1 ∧ x^2 + y^2 = 6 - 2 * Real.sqrt 5 :=
sorry

end min_value_of_sum_squares_on_circle_l214_214102


namespace max_value_expression_l214_214142

theorem max_value_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1 / Real.sqrt 3) :
  27 * a * b * c + a * Real.sqrt (a^2 + 2 * b * c) + b * Real.sqrt (b^2 + 2 * c * a) + c * Real.sqrt (c^2 + 2 * a * b) ≤ 2 / (3 * Real.sqrt 3) :=
sorry

end max_value_expression_l214_214142


namespace poly_eq_l214_214933

-- Definition of the polynomials f(x) and g(x)
def f (x : ℝ) := x^4 + 4*x^3 + 8*x
def g (x : ℝ) := 10*x^4 + 30*x^3 + 29*x^2 + 2*x + 5

-- Define p(x) as a function that satisfies the given condition
def p (x : ℝ) := 9*x^4 + 26*x^3 + 29*x^2 - 6*x + 5

-- Prove that the function p(x) satisfies the equation
theorem poly_eq : ∀ x : ℝ, p x + f x = g x :=
by
  intro x
  -- Add a marker to indicate that this is where the proof would go
  sorry

end poly_eq_l214_214933


namespace input_x_for_y_16_l214_214714

noncomputable def output_y_from_input_x (x : Int) : Int :=
if x < 0 then (x + 1) * (x + 1)
else (x - 1) * (x - 1)

theorem input_x_for_y_16 (x : Int) (y : Int) (h : y = 16) :
  output_y_from_input_x x = y ↔ (x = 5 ∨ x = -5) :=
by
  sorry

end input_x_for_y_16_l214_214714


namespace units_digit_7_pow_451_l214_214707

theorem units_digit_7_pow_451 : (7^451 % 10) = 3 := by
  sorry

end units_digit_7_pow_451_l214_214707


namespace proof_fraction_problem_l214_214421

def fraction_problem :=
  (1 / 5 + 1 / 3) / (3 / 4 - 1 / 8) = 64 / 75

theorem proof_fraction_problem : fraction_problem :=
by
  sorry

end proof_fraction_problem_l214_214421


namespace find_price_of_pastry_l214_214400

-- Define the known values and conditions
variable (P : ℕ)  -- Price of a pastry
variable (usual_pastries : ℕ := 20)
variable (usual_bread : ℕ := 10)
variable (bread_price : ℕ := 4)
variable (today_pastries : ℕ := 14)
variable (today_bread : ℕ := 25)
variable (price_difference : ℕ := 48)

-- Define the usual daily total and today's total
def usual_total := usual_pastries * P + usual_bread * bread_price
def today_total := today_pastries * P + today_bread * bread_price

-- Define the problem statement
theorem find_price_of_pastry (h: usual_total - today_total = price_difference) : P = 18 :=
  by sorry

end find_price_of_pastry_l214_214400


namespace circle_intersection_range_l214_214962

noncomputable def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 = 25
noncomputable def circle2_eq (x y r : ℝ) : Prop := (x - 7)^2 + y^2 = r^2

theorem circle_intersection_range (r : ℝ) (h : r > 0) :
  (∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y r) ↔ 2 < r ∧ r < 12 :=
sorry

end circle_intersection_range_l214_214962


namespace stratified_sampling_l214_214798

theorem stratified_sampling 
  (male_students : ℕ)
  (female_students : ℕ)
  (sample_size : ℕ)
  (H_male_students : male_students = 40)
  (H_female_students : female_students = 30)
  (H_sample_size : sample_size = 7)
  (H_stratified_sample : sample_size = male_students_drawn + female_students_drawn) :
  male_students_drawn = 4 ∧ female_students_drawn = 3  :=
sorry

end stratified_sampling_l214_214798


namespace correct_operation_l214_214522

-- Define the operations given in the conditions
def optionA (m : ℝ) := m^2 + m^2 = 2 * m^4
def optionB (a : ℝ) := a^2 * a^3 = a^5
def optionC (m n : ℝ) := (m * n^2) ^ 3 = m * n^6
def optionD (m : ℝ) := m^6 / m^2 = m^3

-- Theorem stating that option B is the correct operation
theorem correct_operation (a m n : ℝ) : optionB a :=
by sorry

end correct_operation_l214_214522


namespace greatest_median_l214_214898

theorem greatest_median (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t) (h5 : (k + m + r + s + t) = 80) (h6 : t = 42) : r = 17 :=
by
  sorry

end greatest_median_l214_214898


namespace arithmetic_sequence_value_l214_214901

variable (a : ℕ → ℤ) (d : ℤ)
variable (h1 : a 1 + a 4 + a 7 = 39)
variable (h2 : a 2 + a 5 + a 8 = 33)
variable (h_arith : ∀ n, a (n + 1) = a n + d)

theorem arithmetic_sequence_value : a 5 + a 8 + a 11 = 15 := by
  sorry

end arithmetic_sequence_value_l214_214901


namespace not_enrolled_eq_80_l214_214583

variable (total_students : ℕ)
variable (french_students : ℕ)
variable (german_students : ℕ)
variable (spanish_students : ℕ)
variable (french_and_german : ℕ)
variable (german_and_spanish : ℕ)
variable (spanish_and_french : ℕ)
variable (all_three : ℕ)

noncomputable def students_not_enrolled_in_any_language 
  (total_students french_students german_students spanish_students french_and_german german_and_spanish spanish_and_french all_three : ℕ) : ℕ :=
  total_students - (french_students + german_students + spanish_students - french_and_german - german_and_spanish - spanish_and_french + all_three)

theorem not_enrolled_eq_80 : 
  students_not_enrolled_in_any_language 180 60 50 35 20 15 10 5 = 80 :=
  by
    unfold students_not_enrolled_in_any_language
    simp
    sorry

end not_enrolled_eq_80_l214_214583


namespace sequence_sum_relation_l214_214082

theorem sequence_sum_relation (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, 4 * S n = (a n + 1) ^ 2) →
  (S 1 = a 1) →
  a 1 = 1 →
  (∀ n, a (n + 1) = a n + 2) →
  a 2023 = 4045 :=
by
  sorry

end sequence_sum_relation_l214_214082


namespace average_output_l214_214171

theorem average_output (t1 t2: ℝ) (cogs1 cogs2 : ℕ) (h1 : t1 = cogs1 / 36) (h2 : t2 = cogs2 / 60) (h_sum_cogs : cogs1 = 60) (h_sum_more_cogs : cogs2 = 60) (h_sum_time : t1 + t2 = 60 / 36 + 60 / 60) : 
  (cogs1 + cogs2) / (t1 + t2) = 45 := by
  sorry

end average_output_l214_214171


namespace tangent_line_find_a_l214_214853

theorem tangent_line_find_a (a : ℝ) (f : ℝ → ℝ) (tangent : ℝ → ℝ) (x₀ : ℝ)
  (hf : ∀ x, f x = x + 1/x - a * Real.log x)
  (h_tangent : ∀ x, tangent x = x + 1)
  (h_deriv : deriv f x₀ = deriv tangent x₀)
  (h_eq : f x₀ = tangent x₀) :
  a = -1 :=
sorry

end tangent_line_find_a_l214_214853


namespace no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1_l214_214212

theorem no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1 :
  ∀ (a b n : ℕ), (a > 1) → (b > 1) → (a ∣ 2^n - 1) → (b ∣ 2^n + 1) → ∀ (k : ℕ), ¬ (a ∣ 2^k + 1 ∧ b ∣ 2^k - 1) :=
by
  intros a b n a_gt_1 b_gt_1 a_div_2n_minus_1 b_div_2n_plus_1 k
  sorry

end no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1_l214_214212


namespace jerry_bought_3_pounds_l214_214911

-- Definitions based on conditions:
def cost_mustard_oil := 2 * 13
def cost_pasta_sauce := 5
def total_money := 50
def money_left := 7
def cost_gluten_free_pasta_per_pound := 4

-- The proof goal based on the correct answer:
def pounds_gluten_free_pasta : Nat :=
  let total_spent := total_money - money_left
  let spent_on_mustard_and_sauce := cost_mustard_oil + cost_pasta_sauce
  let spent_on_pasta := total_spent - spent_on_mustard_and_sauce
  spent_on_pasta / cost_gluten_free_pasta_per_pound

theorem jerry_bought_3_pounds :
  pounds_gluten_free_pasta = 3 := by
  -- the proof should follow here
  sorry

end jerry_bought_3_pounds_l214_214911


namespace bc_together_l214_214700

theorem bc_together (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 20) : B + C = 320 :=
by
  sorry

end bc_together_l214_214700


namespace necessary_sufficient_condition_l214_214153

theorem necessary_sufficient_condition (a b x_0 : ℝ) (h : a > 0) :
  (x_0 = b / a) ↔ (∀ x : ℝ, (1/2) * a * x^2 - b * x ≥ (1/2) * a * x_0^2 - b * x_0) :=
sorry

end necessary_sufficient_condition_l214_214153


namespace Melanie_dimes_and_coins_l214_214497

-- Define all given conditions
def d1 : Nat := 7
def d2 : Nat := 8
def d3 : Nat := 4
def r : Float := 2.5

-- State the theorem to prove
theorem Melanie_dimes_and_coins :
  let d_t := d1 + d2 + d3
  let c_t := Float.ofNat d_t * r
  d_t = 19 ∧ c_t = 47.5 :=
by
  sorry

end Melanie_dimes_and_coins_l214_214497


namespace number_of_monomials_is_3_l214_214835

def isMonomial (term : String) : Bool :=
  match term with
  | "0" => true
  | "-a" => true
  | "-3x^2y" => true
  | _ => false

def monomialCount (terms : List String) : Nat :=
  terms.filter isMonomial |>.length

theorem number_of_monomials_is_3 :
  monomialCount ["1/x", "x+y", "0", "-a", "-3x^2y", "(x+1)/3"] = 3 :=
by
  sorry

end number_of_monomials_is_3_l214_214835


namespace largest_4digit_div_by_35_l214_214761

theorem largest_4digit_div_by_35 : ∃ n, (1000 ≤ n) ∧ (n ≤ 9999) ∧ (35 ∣ n) ∧ (∀ m, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (35 ∣ m) → m ≤ n) ∧ n = 9985 :=
by
  sorry

end largest_4digit_div_by_35_l214_214761


namespace gcd_3pow600_minus_1_3pow612_minus_1_l214_214875

theorem gcd_3pow600_minus_1_3pow612_minus_1 :
  Nat.gcd (3^600 - 1) (3^612 - 1) = 531440 :=
by
  sorry

end gcd_3pow600_minus_1_3pow612_minus_1_l214_214875


namespace vector_expression_l214_214343

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, -2)

-- The target relationship
theorem vector_expression :
  c = (- (3 / 2) • a + (1 / 2) • b) :=
sorry

end vector_expression_l214_214343


namespace find_a_minus_c_l214_214563

theorem find_a_minus_c (a b c : ℝ) (h1 : (a + b) / 2 = 80) (h2 : (b + c) / 2 = 180) : a - c = -200 :=
by 
  sorry

end find_a_minus_c_l214_214563


namespace height_of_wall_l214_214213

-- Definitions
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def wall_length : ℝ := 850
def wall_width : ℝ := 22.5
def num_bricks : ℝ := 6800

-- Total volume of bricks
def total_brick_volume : ℝ := num_bricks * brick_length * brick_width * brick_height

-- Volume of the wall
def wall_volume (height : ℝ) : ℝ := wall_length * wall_width * height

-- Proof statement
theorem height_of_wall : ∃ h : ℝ, wall_volume h = total_brick_volume ∧ h = 600 := 
sorry

end height_of_wall_l214_214213


namespace passes_through_point_l214_214389

theorem passes_through_point (a : ℝ) (h : a > 0) (h2 : a ≠ 1) : 
  (2, 1) ∈ {p : ℝ × ℝ | ∃ a, p.snd = a * p.fst - 2} :=
sorry

end passes_through_point_l214_214389


namespace find_b_of_expression_l214_214920

theorem find_b_of_expression (y : ℝ) (b : ℝ) (hy : y > 0)
  (h : (7 / 10) * y = (8 * y) / b + (3 * y) / 10) : b = 20 :=
sorry

end find_b_of_expression_l214_214920


namespace correct_statements_count_l214_214848

theorem correct_statements_count :
  (∀ x > 0, x > Real.sin x) ∧
  (¬ (∀ x > 0, x - Real.log x > 0) ↔ (∃ x > 0, x - Real.log x ≤ 0)) ∧
  ¬ (∀ p q : Prop, (p ∨ q) → (p ∧ q)) →
  2 = 2 :=
by sorry

end correct_statements_count_l214_214848


namespace sea_lions_count_l214_214540

theorem sea_lions_count (S P : ℕ) (h1 : 11 * S = 4 * P) (h2 : P = S + 84) : S = 48 := 
by {
  sorry
}

end sea_lions_count_l214_214540


namespace product_of_solutions_l214_214543

theorem product_of_solutions : 
  (∃ x1 x2 : ℝ, |5 * x1 - 1| + 4 = 54 ∧ |5 * x2 - 1| + 4 = 54 ∧ x1 * x2 = -99.96) :=
  by sorry

end product_of_solutions_l214_214543


namespace sum_of_coordinates_reflection_l214_214464

theorem sum_of_coordinates_reflection (y : ℝ) : 
  let C := (3, y)
  let D := (3, -y)
  (C.1 + C.2 + D.1 + D.2) = 6 :=
by
  let C := (3, y)
  let D := (3, -y)
  have h : C.1 + C.2 + D.1 + D.2 = 6 := sorry
  exact h

end sum_of_coordinates_reflection_l214_214464


namespace at_least_one_less_than_two_l214_214251

theorem at_least_one_less_than_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : 2 < a + b) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := 
by
  sorry

end at_least_one_less_than_two_l214_214251


namespace distribution_of_balls_into_boxes_l214_214606

noncomputable def partitions_of_6_into_4_boxes : ℕ := 9

theorem distribution_of_balls_into_boxes :
  let balls := 6
  let boxes := 4
  let ways := partitions_of_6_into_4_boxes
  ways = 9 :=
by
  let balls := 6
  let boxes := 4
  let ways := partitions_of_6_into_4_boxes
  sorry

end distribution_of_balls_into_boxes_l214_214606


namespace integral_one_over_x_l214_214205

theorem integral_one_over_x:
  ∫ x in (1 : ℝ)..(Real.exp 1), 1 / x = 1 := 
by 
  sorry

end integral_one_over_x_l214_214205


namespace total_questions_in_test_l214_214608

theorem total_questions_in_test :
  ∃ x, (5 * x = total_questions) ∧ 
       (20 : ℚ) / total_questions > (60 / 100 : ℚ) ∧ 
       (20 : ℚ) / total_questions < (70 / 100 : ℚ) ∧ 
       total_questions = 30 :=
by
  sorry

end total_questions_in_test_l214_214608


namespace p_iff_q_l214_214672

variables {a b c : ℝ}
def p (a b c : ℝ) : Prop := ∃ x : ℝ, x = 1 ∧ a * x^2 + b * x + c = 0
def q (a b c : ℝ) : Prop := a + b + c = 0

theorem p_iff_q (h : a ≠ 0) : p a b c ↔ q a b c :=
sorry

end p_iff_q_l214_214672


namespace jackson_chairs_l214_214352

theorem jackson_chairs (a b c d : ℕ) (h1 : a = 6) (h2 : b = 4) (h3 : c = 12) (h4 : d = 6) : a * b + c * d = 96 := 
by sorry

end jackson_chairs_l214_214352


namespace area_shaded_region_in_hexagon_l214_214520

theorem area_shaded_region_in_hexagon (s : ℝ) (r : ℝ) (h_s : s = 4) (h_r : r = 2) :
  let area_hexagon := ((3 * Real.sqrt 3) / 2) * s^2
  let area_semicircle := (π * r^2) / 2
  let total_area_semicircles := 8 * area_semicircle
  let area_shaded_region := area_hexagon - total_area_semicircles
  area_shaded_region = 24 * Real.sqrt 3 - 16 * π :=
by {
  sorry
}

end area_shaded_region_in_hexagon_l214_214520


namespace vlad_taller_by_41_inches_l214_214405

/-- Vlad's height is 6 feet and 3 inches. -/
def vlad_height_feet : ℕ := 6

def vlad_height_inches : ℕ := 3

/-- Vlad's sister's height is 2 feet and 10 inches. -/
def sister_height_feet : ℕ := 2

def sister_height_inches : ℕ := 10

/-- There are 12 inches in a foot. -/
def inches_in_a_foot : ℕ := 12

/-- Convert height in feet and inches to total inches. -/
def convert_to_inches (feet inches : ℕ) : ℕ :=
  feet * inches_in_a_foot + inches

/-- Proof that Vlad is 41 inches taller than his sister. -/
theorem vlad_taller_by_41_inches : convert_to_inches vlad_height_feet vlad_height_inches - convert_to_inches sister_height_feet sister_height_inches = 41 :=
by
  -- Start the proof
  sorry

end vlad_taller_by_41_inches_l214_214405


namespace additional_people_required_l214_214734

-- Define conditions
def people := 8
def time1 := 3
def total_work := people * time1 -- This gives us the constant k

-- Define the second condition where 12 people are needed to complete in 2 hours
def required_people (t : Nat) := total_work / t

-- The number of additional people required
def additional_people := required_people 2 - people

-- State the theorem
theorem additional_people_required : additional_people = 4 :=
by 
  show additional_people = 4
  sorry

end additional_people_required_l214_214734


namespace solution_set_of_fx_eq_zero_l214_214323

noncomputable def f (x : ℝ) : ℝ :=
if hx : x = 0 then 0 else if 0 < x then Real.log x / Real.log 2 else - (Real.log (-x) / Real.log 2)

lemma f_is_odd : ∀ x : ℝ, f (-x) = - f x :=
by sorry

lemma f_is_log_for_positive : ∀ x : ℝ, 0 < x → f x = Real.log x / Real.log 2 :=
by sorry

theorem solution_set_of_fx_eq_zero :
  {x : ℝ | f x = 0} = {-1, 0, 1} :=
by sorry

end solution_set_of_fx_eq_zero_l214_214323


namespace truffles_more_than_caramels_l214_214317

-- Define the conditions
def chocolates := 50
def caramels := 3
def nougats := 2 * caramels
def peanut_clusters := (64 * chocolates) / 100
def truffles := chocolates - (caramels + nougats + peanut_clusters)

-- Define the claim
theorem truffles_more_than_caramels : (truffles - caramels) = 6 := by
  sorry

end truffles_more_than_caramels_l214_214317


namespace zero_count_at_end_of_45_320_125_product_l214_214730

theorem zero_count_at_end_of_45_320_125_product :
  let p := 45 * 320 * 125
  45 = 5 * 3^2 ∧ 320 = 2^6 * 5 ∧ 125 = 5^3 →
  p = 2^6 * 3^2 * 5^5 →
  p % 10^5 = 0 ∧ p % 10^6 ≠ 0 :=
by
  sorry

end zero_count_at_end_of_45_320_125_product_l214_214730


namespace roots_inverse_cubed_l214_214985

-- Define the conditions and the problem statement
theorem roots_inverse_cubed (p q m r s : ℝ) (h1 : r + s = -q / p) (h2 : r * s = m / p) 
  (h3 : ∀ x : ℝ, p * x^2 + q * x + m = 0 → x = r ∨ x = s) : 
  1 / r^3 + 1 / s^3 = (-q^3 + 3 * q * m) / m^3 := 
sorry

end roots_inverse_cubed_l214_214985


namespace scorpion_segments_daily_total_l214_214198

theorem scorpion_segments_daily_total (seg1 : ℕ) (seg2 : ℕ) (additional : ℕ) (total_daily : ℕ) :
  (seg1 = 60) →
  (seg2 = 2 * seg1 * 2) →
  (additional = 10 * 50) →
  (total_daily = seg1 + seg2 + additional) →
  total_daily = 800 :=
by
  intros h1 h2 h3 h4
  sorry

end scorpion_segments_daily_total_l214_214198


namespace smallest_number_of_marbles_l214_214627

-- Define the conditions
variables (r w b g n : ℕ)
def valid_total (r w b g n : ℕ) := r + w + b + g = n
def valid_probability_4r (r w b g n : ℕ) := r * (r-1) * (r-2) * (r-3) = 24 * (w) * (r * (r - 1) * (r - 2) / 6)
def valid_probability_1w3r (r w b g n : ℕ) := r * (r-1) * (r-2) * (r-3) = 24 * (w * b * (r * (r - 1) / 2))
def valid_probability_1w1b2r (r w b g n : ℕ) := w * b * (r * (r - 1) / 2) = w * b * g * r

theorem smallest_number_of_marbles :
  ∃ n r w b g, valid_total r w b g n ∧
  valid_probability_4r r w b g n ∧
  valid_probability_1w3r r w b g n ∧
  valid_probability_1w1b2r r w b g n ∧ 
  n = 21 :=
  sorry

end smallest_number_of_marbles_l214_214627


namespace speed_of_mrs_a_l214_214199

theorem speed_of_mrs_a
  (distance_between : ℝ)
  (speed_mr_a : ℝ)
  (speed_bee : ℝ)
  (distance_bee_travelled : ℝ)
  (time_bee : ℝ)
  (remaining_distance : ℝ)
  (speed_mrs_a : ℝ) :
  distance_between = 120 ∧
  speed_mr_a = 30 ∧
  speed_bee = 60 ∧
  distance_bee_travelled = 180 ∧
  time_bee = distance_bee_travelled / speed_bee ∧
  remaining_distance = distance_between - (speed_mr_a * time_bee) ∧
  speed_mrs_a = remaining_distance / time_bee →
  speed_mrs_a = 10 := by
  sorry

end speed_of_mrs_a_l214_214199


namespace total_students_l214_214081

theorem total_students (students_in_front : ℕ) (position_from_back : ℕ) : 
  students_in_front = 6 ∧ position_from_back = 5 → 
  students_in_front + 1 + (position_from_back - 1) = 11 :=
by
  sorry

end total_students_l214_214081


namespace arithmetic_mean_of_two_numbers_l214_214514

def is_arithmetic_mean (x y z : ℚ) : Prop :=
  (x + z) / 2 = y

theorem arithmetic_mean_of_two_numbers :
  is_arithmetic_mean (9 / 12) (5 / 6) (7 / 8) :=
by
  sorry

end arithmetic_mean_of_two_numbers_l214_214514


namespace term_sequence_l214_214086

theorem term_sequence (n : ℕ) (h : (-1:ℤ) ^ (n + 1) * n * (n + 1) = -20) : n = 4 :=
sorry

end term_sequence_l214_214086


namespace rectangular_prism_diagonal_l214_214574

theorem rectangular_prism_diagonal 
  (a b c : ℝ)
  (h1 : 2 * a * b + 2 * b * c + 2 * c * a = 11)
  (h2 : 4 * (a + b + c) = 24) :
  (a^2 + b^2 + c^2 = 25) :=
by {
  -- Sorry to skip the proof steps
  sorry
}

end rectangular_prism_diagonal_l214_214574


namespace geometric_sequence_ratio_l214_214312

noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ := 
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_ratio 
  (S : ℕ → ℝ) 
  (hS12 : S 12 = 1)
  (hS6 : S 6 = 2)
  (geom_property : ∀ a r, (S n = a * (1 - r^n) / (1 - r))) :
  S 18 / S 6 = 3 / 4 :=
by
  sorry

end geometric_sequence_ratio_l214_214312


namespace simplify_expression_l214_214413

theorem simplify_expression :
  (Real.sqrt (Real.sqrt (81)) - Real.sqrt (8 + 1 / 2)) ^ 2 = (35 / 2) - 3 * Real.sqrt 34 :=
by
  sorry

end simplify_expression_l214_214413


namespace area_of_triangle_hyperbola_focus_l214_214685

theorem area_of_triangle_hyperbola_focus :
  let F₁ := (-Real.sqrt 2, 0)
  let F₂ := (Real.sqrt 2, 0)
  let hyperbola := {p : ℝ × ℝ | p.1 ^ 2 - p.2 ^ 2 = 1}
  let asymptote (p : ℝ × ℝ) := p.1 = p.2
  let circle := {p : ℝ × ℝ | (p.1 - F₁.1 / 2) ^ 2 + (p.2 - F₁.2 / 2) ^ 2 = (Real.sqrt 2) ^ 2}
  let P := (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
  let Q := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let area (p1 p2 p3 : ℝ × ℝ) := 0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))
  area F₁ P Q = Real.sqrt 2 := 
sorry

end area_of_triangle_hyperbola_focus_l214_214685


namespace B_is_subset_of_A_l214_214886
open Set

def A := {x : ℤ | ∃ n : ℤ, x = 2 * n}
def B := {y : ℤ | ∃ k : ℤ, y = 4 * k}

theorem B_is_subset_of_A : B ⊆ A :=
by sorry

end B_is_subset_of_A_l214_214886


namespace find_v1_l214_214340

def u (x : ℝ) : ℝ := 4 * x - 9

def v (y : ℝ) : ℝ := y^2 + 4 * y - 5

theorem find_v1 : v 1 = 11.25 := by
  sorry

end find_v1_l214_214340


namespace triangle_pentagon_side_ratio_l214_214966

theorem triangle_pentagon_side_ratio (triangle_perimeter : ℕ) (pentagon_perimeter : ℕ) 
  (h1 : triangle_perimeter = 60) (h2 : pentagon_perimeter = 60) :
  (triangle_perimeter / 3 : ℚ) / (pentagon_perimeter / 5 : ℚ) = 5 / 3 :=
by {
  sorry
}

end triangle_pentagon_side_ratio_l214_214966


namespace length_of_pipe_is_correct_l214_214656

-- Definitions of the conditions
def step_length : ℝ := 0.8
def steps_same_direction : ℤ := 210
def steps_opposite_direction : ℤ := 100

-- The distance moved by the tractor in one step
noncomputable def tractor_step_distance : ℝ := (steps_same_direction * step_length - steps_opposite_direction * step_length) / (steps_opposite_direction + steps_same_direction : ℝ)

-- The length of the pipe
noncomputable def length_of_pipe (steps_same_direction steps_opposite_direction : ℤ) (step_length : ℝ) : ℝ :=
 steps_same_direction * (step_length - tractor_step_distance)

-- Proof statement
theorem length_of_pipe_is_correct :
  length_of_pipe steps_same_direction steps_opposite_direction step_length = 108 :=
sorry

end length_of_pipe_is_correct_l214_214656


namespace cat_food_customers_l214_214937

/-
Problem: There was a big sale on cat food at the pet store. Some people bought cat food that day. The first 8 customers bought 3 cases each. The next four customers bought 2 cases each. The last 8 customers of the day only bought 1 case each. In total, 40 cases of cat food were sold. How many people bought cat food that day?
-/

theorem cat_food_customers:
  (8 * 3) + (4 * 2) + (8 * 1) = 40 →
  8 + 4 + 8 = 20 :=
by
  intro h
  linarith

end cat_food_customers_l214_214937


namespace trader_gain_pens_l214_214331

theorem trader_gain_pens (C S : ℝ) (h1 : S = 1.25 * C) 
                         (h2 : 80 * S = 100 * C) : S - C = 0.25 * C :=
by
  have h3 : S = 1.25 * C := h1
  have h4 : 80 * S = 100 * C := h2
  sorry

end trader_gain_pens_l214_214331


namespace problem1_problem2_l214_214423

noncomputable def f (x a b : ℝ) : ℝ := x^2 - (a+1)*x + b

theorem problem1 (h : ∀ x : ℝ, f x (-4) (-10) < 0 ↔ -5 < x ∧ x < 2) : f x (-4) (-10) < 0 :=
sorry

theorem problem2 (a : ℝ) : 
  (a > 1 → ∀ x : ℝ, f x a a > 0 ↔ x < 1 ∨ x > a) ∧
  (a = 1 → ∀ x : ℝ, f x a a > 0 ↔ x ≠ 1) ∧
  (a < 1 → ∀ x : ℝ, f x a a > 0 ↔ x < a ∨ x > 1) :=
sorry

end problem1_problem2_l214_214423


namespace stewart_farm_sheep_count_l214_214349

theorem stewart_farm_sheep_count
  (ratio : ℕ → ℕ → Prop)
  (S H : ℕ)
  (ratio_S_H : ratio S H)
  (one_sheep_seven_horses : ratio 1 7)
  (food_per_horse : ℕ)
  (total_food : ℕ)
  (food_per_horse_val : food_per_horse = 230)
  (total_food_val : total_food = 12880)
  (calc_horses : H = total_food / food_per_horse)
  (calc_sheep : S = H / 7) :
  S = 8 :=
by {
  /- Given the conditions, we need to show that S = 8 -/
  sorry
}

end stewart_farm_sheep_count_l214_214349


namespace reciprocal_opposite_neg_two_thirds_l214_214257

noncomputable def opposite (a : ℚ) : ℚ := -a
noncomputable def reciprocal (a : ℚ) : ℚ := 1 / a

theorem reciprocal_opposite_neg_two_thirds : reciprocal (opposite (-2 / 3)) = 3 / 2 :=
by sorry

end reciprocal_opposite_neg_two_thirds_l214_214257


namespace log_squared_sum_eq_one_l214_214242

open Real

theorem log_squared_sum_eq_one :
  (log 2)^2 * log 250 + (log 5)^2 * log 40 = 1 := by
  sorry

end log_squared_sum_eq_one_l214_214242


namespace total_soccer_balls_donated_l214_214641

def num_elementary_classes_per_school := 4
def num_middle_classes_per_school := 5
def num_schools := 2
def soccer_balls_per_class := 5

theorem total_soccer_balls_donated : 
  (num_elementary_classes_per_school + num_middle_classes_per_school) * num_schools * soccer_balls_per_class = 90 :=
by
  sorry

end total_soccer_balls_donated_l214_214641


namespace quadratic_inequality_solution_set_conclusions_l214_214283

variables {a b c : ℝ}

theorem quadratic_inequality_solution_set_conclusions (h1 : ∀ x, -1 ≤ x ∧ x ≤ 2 → ax^2 + bx + c ≥ 0)
(h2 : ∀ x, x < -1 ∨ x > 2 → ax^2 + bx + c < 0) :
(a + b = 0) ∧ (a + b + c > 0) ∧ (c > 0) ∧ ¬ (b < 0) := by
sorry

end quadratic_inequality_solution_set_conclusions_l214_214283


namespace walkways_area_l214_214640

theorem walkways_area (rows cols : ℕ) (bed_length bed_width walkthrough_width garden_length garden_width total_flower_beds bed_area total_bed_area total_garden_area : ℝ) 
  (h1 : rows = 4) (h2 : cols = 3) 
  (h3 : bed_length = 8) (h4 : bed_width = 3) 
  (h5 : walkthrough_width = 2)
  (h6 : garden_length = (cols * bed_length) + ((cols + 1) * walkthrough_width))
  (h7 : garden_width = (rows * bed_width) + ((rows + 1) * walkthrough_width))
  (h8 : total_garden_area = garden_length * garden_width)
  (h9 : total_flower_beds = rows * cols)
  (h10 : bed_area = bed_length * bed_width)
  (h11 : total_bed_area = total_flower_beds * bed_area)
  (h12 : total_garden_area - total_bed_area = 416) : 
  True := 
sorry

end walkways_area_l214_214640


namespace problem_solution_l214_214633

theorem problem_solution :
  (1/3⁻¹) - Real.sqrt 27 + 3 * Real.tan (Real.pi / 6) + (Real.pi - 3.14)^0 = 4 - 2 * Real.sqrt 3 := by
  sorry

end problem_solution_l214_214633


namespace locus_of_circle_center_l214_214861

theorem locus_of_circle_center (x y : ℝ) : 
    (exists C : ℝ × ℝ, (C.1, C.2) = (x,y)) ∧ 
    ((x - 0)^2 + (y - 3)^2 = r^2) ∧ 
    (y + 3 = 0) → x^2 = 12 * y :=
sorry

end locus_of_circle_center_l214_214861


namespace find_other_discount_l214_214234

def other_discount (list_price final_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) : Prop :=
  let price_after_first_discount := list_price - (first_discount / 100) * list_price
  final_price = price_after_first_discount - (second_discount / 100) * price_after_first_discount

theorem find_other_discount : 
  other_discount 70 59.22 10 6 :=
by
  sorry

end find_other_discount_l214_214234


namespace largest_among_trig_expressions_l214_214274

theorem largest_among_trig_expressions :
  let a := Real.tan 48 + 1 / Real.tan 48
  let b := Real.sin 48 + Real.cos 48
  let c := Real.tan 48 + Real.cos 48
  let d := 1 / Real.tan 48 + Real.sin 48
  a > b ∧ a > c ∧ a > d :=
by
  sorry

end largest_among_trig_expressions_l214_214274


namespace mary_animals_count_l214_214177

def initial_lambs := 18
def initial_alpacas := 5
def initial_baby_lambs := 7 * 4
def traded_lambs := 8
def traded_alpacas := 2
def received_goats := 3
def received_chickens := 10
def chickens_traded_for_alpacas := received_chickens / 2
def additional_lambs := 20
def additional_alpacas := 6

noncomputable def final_lambs := initial_lambs + initial_baby_lambs - traded_lambs + additional_lambs
noncomputable def final_alpacas := initial_alpacas - traded_alpacas + 2 + additional_alpacas
noncomputable def final_goats := received_goats
noncomputable def final_chickens := received_chickens - chickens_traded_for_alpacas

theorem mary_animals_count :
  final_lambs = 58 ∧ 
  final_alpacas = 11 ∧ 
  final_goats = 3 ∧ 
  final_chickens = 5 :=
by 
  sorry

end mary_animals_count_l214_214177


namespace simplify_expression_l214_214259

variable (a : Real)

theorem simplify_expression : (-2 * a) * a - (-2 * a)^2 = -6 * a^2 := by
  sorry

end simplify_expression_l214_214259


namespace martin_total_distance_l214_214402

theorem martin_total_distance (T S1 S2 t : ℕ) (hT : T = 8) (hS1 : S1 = 70) (hS2 : S2 = 85) (ht : t = T / 2) : S1 * t + S2 * t = 620 := 
by
  sorry

end martin_total_distance_l214_214402


namespace max_side_length_is_11_l214_214365

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l214_214365


namespace revenue_from_full_price_tickets_l214_214521

-- Definitions of the conditions
def total_tickets (f h : ℕ) : Prop := f + h = 180
def total_revenue (f h p : ℕ) : Prop := f * p + h * (p / 2) = 2750

-- Theorem statement
theorem revenue_from_full_price_tickets (f h p : ℕ) 
  (h_total_tickets : total_tickets f h) 
  (h_total_revenue : total_revenue f h p) : 
  f * p = 1000 :=
  sorry

end revenue_from_full_price_tickets_l214_214521


namespace integer_solutions_k_l214_214530

theorem integer_solutions_k (k n m : ℤ) (h1 : k + 1 = n^2) (h2 : 16 * k + 1 = m^2) :
  k = 0 ∨ k = 3 :=
by sorry

end integer_solutions_k_l214_214530


namespace square_area_l214_214144

theorem square_area :
  ∀ (x1 x2 : ℝ), (x1^2 + 2 * x1 + 1 = 8) ∧ (x2^2 + 2 * x2 + 1 = 8) ∧ (x1 ≠ x2) →
  (abs (x1 - x2))^2 = 36 :=
by
  sorry

end square_area_l214_214144


namespace tuesday_rainfall_l214_214300

-- Condition: average rainfall for the whole week is 3 cm
def avg_rainfall_week : ℝ := 3

-- Condition: number of days in a week
def days_in_week : ℕ := 7

-- Condition: total rainfall for the week
def total_rainfall_week : ℝ := avg_rainfall_week * days_in_week

-- Condition: total rainfall is twice the rainfall on Tuesday
def total_rainfall_equals_twice_T (T : ℝ) : ℝ := 2 * T

-- Theorem: Prove that the rainfall on Tuesday is 10.5 cm
theorem tuesday_rainfall : ∃ T : ℝ, total_rainfall_equals_twice_T T = total_rainfall_week ∧ T = 10.5 := by
  sorry

end tuesday_rainfall_l214_214300


namespace intersection_range_l214_214671

-- Define the line equation
def line (k x : ℝ) : ℝ := k * x - k + 1

-- Define the curve equation
def curve (x y m : ℝ) : Prop := x^2 + 2 * y^2 = m

-- State the problem: Given the line and the curve have a common point, prove the range of m is m >= 3
theorem intersection_range (k m : ℝ) (h : ∃ x y, line k x = y ∧ curve x y m) : m ≥ 3 :=
by {
  sorry
}

end intersection_range_l214_214671


namespace soccer_league_total_games_l214_214164

theorem soccer_league_total_games :
  let teams := 20
  let regular_games_per_team := 19 * 3
  let total_regular_games := (regular_games_per_team * teams) / 2
  let promotional_games_per_team := 3
  let total_promotional_games := promotional_games_per_team * teams
  let total_games := total_regular_games + total_promotional_games
  total_games = 1200 :=
by
  sorry

end soccer_league_total_games_l214_214164


namespace determine_digits_l214_214770

def digit (n : Nat) : Prop := n < 10

theorem determine_digits :
  ∃ (A B C D : Nat), digit A ∧ digit B ∧ digit C ∧ digit D ∧
    (1000 * A + 100 * B + 10 * B + B) ^ 2 = 10000 * A + 1000 * C + 100 * D + 10 * B + B ∧
    (1000 * C + 100 * D + 10 * D + D) ^ 3 = 10000 * A + 1000 * C + 100 * B + 10 * D + D ∧
    A = 9 ∧ B = 6 ∧ C = 2 ∧ D = 1 := 
by
  sorry

end determine_digits_l214_214770


namespace caps_difference_l214_214976

theorem caps_difference (Billie_caps Sammy_caps : ℕ) (Janine_caps := 3 * Billie_caps)
  (Billie_has : Billie_caps = 2) (Sammy_has : Sammy_caps = 8) :
  Sammy_caps - Janine_caps = 2 := by
  -- proof goes here
  sorry

end caps_difference_l214_214976


namespace number_of_cows_l214_214632

/-- 
The number of cows Mr. Reyansh has on his dairy farm 
given the conditions of water consumption and total water used in a week. 
-/
theorem number_of_cows (C : ℕ) 
  (h1 : ∀ (c : ℕ), (c = 80 * 7))
  (h2 : ∀ (s : ℕ), (s = 10 * C))
  (h3 : ∀ (d : ℕ), (d = 20 * 7))
  (h4 : 1960 * C = 78400) : 
  C = 40 :=
sorry

end number_of_cows_l214_214632


namespace num_pairs_of_nat_numbers_satisfying_eq_l214_214075

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end num_pairs_of_nat_numbers_satisfying_eq_l214_214075


namespace original_height_l214_214024

theorem original_height (total_travel : ℝ) (h : ℝ) (half: h/2 = (1/2 * h)): 
  (total_travel = h + 2 * (h / 2) + 2 * (h / 4)) → total_travel = 260 → h = 104 :=
by
  intro travel_eq
  intro travel_value
  sorry

end original_height_l214_214024


namespace average_of_remaining_two_numbers_l214_214026

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 2.5)
  (h2 : (a + b) / 2 = 1.1)
  (h3 : (c + d) / 2 = 1.4) : 
  (e + f) / 2 = 5 :=
by
  sorry

end average_of_remaining_two_numbers_l214_214026


namespace proof_equivalent_triples_l214_214477

noncomputable def valid_triples := 
  { (a, b, c) : ℝ × ℝ × ℝ |
    a * b + b * c + c * a = 1 ∧
    a^2 * b + c = b^2 * c + a ∧
    a^2 * b + c = c^2 * a + b }

noncomputable def desired_solutions := 
  { (a, b, c) |
    (a = 0 ∧ b = 1 ∧ c = 1) ∨
    (a = 0 ∧ b = 1 ∧ c = -1) ∨
    (a = 0 ∧ b = -1 ∧ c = 1) ∨
    (a = 0 ∧ b = -1 ∧ c = -1) ∨

    (a = 1 ∧ b = 1 ∧ c = 0) ∨
    (a = 1 ∧ b = -1 ∧ c = 0) ∨
    (a = -1 ∧ b = 1 ∧ c = 0) ∨
    (a = -1 ∧ b = -1 ∧ c = 0) ∨

    (a = 1 ∧ b = 0 ∧ c = 1) ∨
    (a = 1 ∧ b = 0 ∧ c = -1) ∨
    (a = -1 ∧ b = 0 ∧ c = 1) ∨
    (a = -1 ∧ b = 0 ∧ c = -1) ∨

    ((a = (Real.sqrt 3) / 3 ∧ b = (Real.sqrt 3) / 3 ∧ 
      c = (Real.sqrt 3) / 3) ∨
     (a = -(Real.sqrt 3) / 3 ∧ b = -(Real.sqrt 3) / 3 ∧ 
      c = -(Real.sqrt 3) / 3)) }

theorem proof_equivalent_triples :
  valid_triples = desired_solutions :=
sorry

end proof_equivalent_triples_l214_214477


namespace sequence_bound_l214_214628

variable {a : ℕ+ → ℝ}

theorem sequence_bound (h : ∀ k m : ℕ+, |a (k + m) - a k - a m| ≤ 1) :
    ∀ (p q : ℕ+), |a p / p - a q / q| < 1 / p + 1 / q :=
by
  sorry

end sequence_bound_l214_214628


namespace f_g_5_eq_163_l214_214960

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem f_g_5_eq_163 : f (g 5) = 163 := by
  sorry

end f_g_5_eq_163_l214_214960


namespace evaluate_power_l214_214808

theorem evaluate_power (x : ℝ) (hx : (8:ℝ)^(2 * x) = 11) : 
  2^(x + 1.5) = 11^(1 / 6) * 2 * Real.sqrt 2 :=
by 
  sorry

end evaluate_power_l214_214808


namespace sqrt49_times_sqrt25_eq_5sqrt7_l214_214478

noncomputable def sqrt49_times_sqrt25 : ℝ :=
  Real.sqrt (49 * Real.sqrt 25)

theorem sqrt49_times_sqrt25_eq_5sqrt7 :
  sqrt49_times_sqrt25 = 5 * Real.sqrt 7 :=
by
sorry

end sqrt49_times_sqrt25_eq_5sqrt7_l214_214478


namespace download_time_l214_214139

def first_segment_size : ℝ := 30
def first_segment_rate : ℝ := 5
def second_segment_size : ℝ := 40
def second_segment_rate1 : ℝ := 10
def second_segment_rate2 : ℝ := 2
def third_segment_size : ℝ := 20
def third_segment_rate1 : ℝ := 8
def third_segment_rate2 : ℝ := 4

theorem download_time :
  let time_first := first_segment_size / first_segment_rate
  let time_second := (10 / second_segment_rate1) + (10 / second_segment_rate2) + (10 / second_segment_rate1) + (10 / second_segment_rate2)
  let time_third := (10 / third_segment_rate1) + (10 / third_segment_rate2)
  time_first + time_second + time_third = 21.75 :=
by
  sorry

end download_time_l214_214139


namespace find_a_l214_214686

def setA (a : ℝ) : Set ℝ := {2, 4, a^3 - 2 * a^2 - a + 7}
def setB (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem find_a (a : ℝ) : 
  (setA a ∩ setB a = {2, 5}) → (a = -1 ∨ a = 2) :=
sorry

end find_a_l214_214686


namespace probability_XOXOXOX_is_one_over_thirty_five_l214_214582

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_arrangement : ℤ :=
  let total_arrangements := binomial_coefficient 7 4
  let favorable_outcomes := 1
  favorable_outcomes / total_arrangements

theorem probability_XOXOXOX_is_one_over_thirty_five :
  probability_of_arrangement = (1 : ℤ) / 35 := 
  by
  sorry

end probability_XOXOXOX_is_one_over_thirty_five_l214_214582


namespace find_largest_C_l214_214445

theorem find_largest_C : 
  ∃ (C : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + 10 ≥ C * (x + y + 2)) 
  ∧ (∀ D : ℝ, (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 10 ≥ D * (x + y + 2)) → D ≤ C) 
  ∧ C = Real.sqrt 5 :=
sorry

end find_largest_C_l214_214445


namespace find_salary_for_january_l214_214016

-- Definitions based on problem conditions
variables (J F M A May : ℝ)
variables (h1 : (J + F + M + A) / 4 = 8000)
variables (h2 : (F + M + A + May) / 4 = 8200)
variables (hMay : May = 6500)

-- Lean statement
theorem find_salary_for_january : J = 5700 :=
by {
  sorry
}

end find_salary_for_january_l214_214016


namespace find_f_and_q_l214_214515

theorem find_f_and_q (m : ℤ) (q : ℝ) :
  (∀ x > 0, (x : ℝ)^(-m^2 + 2*m + 3) = (x : ℝ)^4) ∧
  (∀ x ∈ [-1, 1], 2 * (x^2) - 8 * x + q - 1 > 0) →
  q > 7 :=
by
  sorry

end find_f_and_q_l214_214515


namespace find_starting_number_l214_214245

theorem find_starting_number : 
  ∃ x : ℕ, (∀ k : ℕ, (k < 12 → (x + 3 * k) ≤ 46) ∧ 12 = (46 - x) / 3 + 1) 
  ∧ x = 12 := 
by 
  sorry

end find_starting_number_l214_214245


namespace find_x_l214_214036

-- Define the conditions
def is_purely_imaginary (z : Complex) : Prop :=
  z.re = 0

-- Define the problem
theorem find_x (x : ℝ) (z : Complex) (h1 : z = Complex.ofReal (x^2 - 1) + Complex.I * (x + 1)) (h2 : is_purely_imaginary z) : x = 1 :=
sorry

end find_x_l214_214036


namespace sum_of_sequence_l214_214476

def sequence_t (n : ℕ) : ℚ :=
  if n % 2 = 1 then 1 / 7^n else 2 / 7^n

theorem sum_of_sequence :
  (∑' n:ℕ, sequence_t (n + 1)) = 3 / 16 :=
by
  sorry

end sum_of_sequence_l214_214476


namespace simplify_expression_l214_214495

theorem simplify_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ 2) (h4 : x ≠ 3) :
  ((x - 5) / (x - 3) - ((x^2 + 2 * x + 1) / (x^2 + x)) / ((x + 1) / (x - 2)) = 
  -6 / (x^2 - 3 * x)) :=
by
  sorry

end simplify_expression_l214_214495


namespace new_total_lines_l214_214955

-- Definitions and conditions
variable (L : ℕ)
def increased_lines : ℕ := L + 60
def percentage_increase := (60 : ℚ) / L = 1 / 3

-- Theorem statement
theorem new_total_lines : percentage_increase L → increased_lines L = 240 :=
by
  sorry

end new_total_lines_l214_214955


namespace p_sufficient_not_necessary_q_l214_214953

-- Define the conditions p and q
def p (x : ℝ) : Prop := 2 < x ∧ x < 4
def q (x : ℝ) : Prop := x > 2 ∨ x < -3

-- Prove the relationship between p and q
theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l214_214953


namespace addition_amount_first_trial_l214_214299

theorem addition_amount_first_trial :
  ∀ (a b : ℝ),
  20 ≤ a ∧ a ≤ 30 ∧ 20 ≤ b ∧ b ≤ 30 → (a = 20 + (30 - 20) * 0.618 ∨ b = 30 - (30 - 20) * 0.618) :=
by {
  sorry
}

end addition_amount_first_trial_l214_214299


namespace part1_part2_l214_214531

def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x - 4 ≤ 0}

-- Problem 1
theorem part1 (m : ℝ) : 
  (A ∩ B m = {x : ℝ | 1 ≤ x ∧ x ≤ 3}) → m = 3 :=
by sorry

-- Problem 2
theorem part2 (m : ℝ) : 
  (A ⊆ (B m)ᶜ) → (m < -3 ∨ m > 5) :=
by sorry

end part1_part2_l214_214531


namespace train_speed_is_correct_l214_214128

/-- Define the length of the train (in meters) -/
def train_length : ℕ := 120

/-- Define the length of the bridge (in meters) -/
def bridge_length : ℕ := 255

/-- Define the time to cross the bridge (in seconds) -/
def time_to_cross : ℕ := 30

/-- Define the total distance covered by the train while crossing the bridge -/
def total_distance : ℕ := train_length + bridge_length

/-- Define the speed of the train in meters per second -/
def speed_m_per_s : ℚ := total_distance / time_to_cross

/-- Conversion factor from m/s to km/hr -/
def m_per_s_to_km_per_hr : ℚ := 3.6

/-- The expected speed of the train in km/hr -/
def expected_speed_km_per_hr : ℕ := 45

/-- The theorem stating that the speed of the train is 45 km/hr -/
theorem train_speed_is_correct :
  (speed_m_per_s * m_per_s_to_km_per_hr) = expected_speed_km_per_hr := by
  sorry

end train_speed_is_correct_l214_214128


namespace find_original_cost_of_chips_l214_214772

def original_cost_chips (discount amount_spent : ℝ) : ℝ :=
  discount + amount_spent

theorem find_original_cost_of_chips :
  original_cost_chips 17 18 = 35 := by
  sorry

end find_original_cost_of_chips_l214_214772


namespace grace_wait_time_l214_214172

variable (hose1_rate : ℕ) (hose2_rate : ℕ) (pool_capacity : ℕ) (time_after_second_hose : ℕ)
variable (h : ℕ)

theorem grace_wait_time 
  (h1 : hose1_rate = 50)
  (h2 : hose2_rate = 70)
  (h3 : pool_capacity = 390)
  (h4 : time_after_second_hose = 2) : 
  50 * h + (50 + 70) * 2 = 390 → h = 3 :=
by
  sorry

end grace_wait_time_l214_214172


namespace arithmetic_sequence_value_l214_214646

theorem arithmetic_sequence_value (a : ℕ → ℕ) (m : ℕ) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 4) 
  (h_a5 : a 5 = m) 
  (h_a7 : a 7 = 16) : 
  m = 10 := 
by
  sorry

end arithmetic_sequence_value_l214_214646


namespace black_piece_probability_l214_214597

-- Definitions based on conditions
def total_pieces : ℕ := 10 + 5
def black_pieces : ℕ := 10

-- Probability calculation
def probability_black : ℚ := black_pieces / total_pieces

-- Statement to prove
theorem black_piece_probability : probability_black = 2/3 := by
  sorry -- proof to be filled in later

end black_piece_probability_l214_214597


namespace largest_divisor_composite_difference_l214_214670

-- Define when an integer is composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem largest_divisor_composite_difference (n : ℕ) (h : is_composite n) : 6 ∣ (n^4 - n) :=
by sorry

end largest_divisor_composite_difference_l214_214670


namespace sqrt3_mul_sqrt12_eq_6_l214_214195

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l214_214195


namespace marching_band_formations_l214_214915

theorem marching_band_formations :
  (∃ (s t : ℕ), s * t = 240 ∧ 8 ≤ t ∧ t ≤ 30) →
  ∃ (z : ℕ), z = 4 := sorry

end marching_band_formations_l214_214915


namespace max_f_l214_214546

noncomputable def f (x : ℝ) : ℝ :=
  min (min (2 * x + 2) (1 / 2 * x + 1)) (-3 / 4 * x + 7)

theorem max_f : ∃ x : ℝ, f x = 17 / 5 :=
by
  sorry

end max_f_l214_214546


namespace total_money_l214_214727

theorem total_money (gold_value silver_value cash : ℕ) (num_gold num_silver : ℕ)
  (h_gold : gold_value = 50)
  (h_silver : silver_value = 25)
  (h_num_gold : num_gold = 3)
  (h_num_silver : num_silver = 5)
  (h_cash : cash = 30) :
  gold_value * num_gold + silver_value * num_silver + cash = 305 :=
by
  sorry

end total_money_l214_214727


namespace ratio_of_albums_l214_214009

variable (M K B A : ℕ)
variable (s : ℕ)

-- Conditions
def adele_albums := (A = 30)
def bridget_albums := (B = A - 15)
def katrina_albums := (K = 6 * B)
def miriam_albums := (M = s * K)
def total_albums := (M + K + B + A = 585)

-- Proof statement
theorem ratio_of_albums (h1 : adele_albums A) (h2 : bridget_albums B A) (h3 : katrina_albums K B) 
(h4 : miriam_albums M s K) (h5 : total_albums M K B A) :
  s = 5 :=
by
  sorry

end ratio_of_albums_l214_214009


namespace log_eighteen_fifteen_l214_214360

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_eighteen_fifteen (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  log_base 18 15 = (b - a + 1) / (a + 2 * b) :=
by sorry

end log_eighteen_fifteen_l214_214360


namespace tangent_line_value_l214_214503

theorem tangent_line_value {a : ℝ} (h : a > 0) : 
  (∀ θ ρ, (ρ * (Real.cos θ + Real.sin θ) = a) → (ρ = 2 * Real.cos θ)) → 
  a = 1 + Real.sqrt 2 :=
sorry

end tangent_line_value_l214_214503


namespace petya_correct_square_l214_214001

theorem petya_correct_square :
  ∃ x a b : ℕ, (1 ≤ x ∧ x ≤ 9) ∧
              (x^2 = 10 * a + b) ∧ 
              (2 * x = 10 * b + a) ∧
              (x^2 = 81) :=
by
  sorry

end petya_correct_square_l214_214001


namespace compute_expression_l214_214792

-- Lean 4 statement for the mathematic equivalence proof problem
theorem compute_expression:
  (1004^2 - 996^2 - 1002^2 + 998^2) = 8000 := by
  sorry

end compute_expression_l214_214792


namespace sum_less_than_addends_then_both_negative_l214_214396

theorem sum_less_than_addends_then_both_negative {a b : ℝ} (h : a + b < a ∧ a + b < b) : a < 0 ∧ b < 0 := 
sorry

end sum_less_than_addends_then_both_negative_l214_214396


namespace committee_meeting_people_l214_214097

theorem committee_meeting_people (A B : ℕ) 
  (h1 : 2 * A + B = 10) 
  (h2 : A + 2 * B = 11) : 
  A + B = 7 :=
sorry

end committee_meeting_people_l214_214097


namespace max_single_painted_faces_l214_214617

theorem max_single_painted_faces (n : ℕ) (hn : n = 64) :
  ∃ max_cubes : ℕ, max_cubes = 32 := 
sorry

end max_single_painted_faces_l214_214617


namespace josh_ribbon_shortfall_l214_214482

-- Define the total amount of ribbon Josh has
def total_ribbon : ℝ := 18

-- Define the number of gifts
def num_gifts : ℕ := 6

-- Define the ribbon requirements for each gift
def ribbon_per_gift_wrapping : ℝ := 2
def ribbon_per_bow : ℝ := 1.5
def ribbon_per_tag : ℝ := 0.25
def ribbon_per_trim : ℝ := 0.5

-- Calculate the total ribbon required for all the tasks
def total_ribbon_needed : ℝ :=
  (ribbon_per_gift_wrapping * num_gifts) +
  (ribbon_per_bow * num_gifts) +
  (ribbon_per_tag * num_gifts) +
  (ribbon_per_trim * num_gifts)

-- Calculate the ribbon shortfall
def ribbon_shortfall : ℝ :=
  total_ribbon_needed - total_ribbon

-- Prove that Josh will be short by 7.5 yards of ribbon
theorem josh_ribbon_shortfall : ribbon_shortfall = 7.5 := by
  sorry

end josh_ribbon_shortfall_l214_214482


namespace billion_in_scientific_notation_l214_214623

theorem billion_in_scientific_notation :
  (4.55 * 10^9) = (4.55 * 10^9) := by
  sorry

end billion_in_scientific_notation_l214_214623


namespace quadrant_and_terminal_angle_l214_214658

def alpha : ℝ := -1910 

noncomputable def normalize_angle (α : ℝ) : ℝ := 
  let β := α % 360
  if β < 0 then β + 360 else β

noncomputable def in_quadrant_3 (β : ℝ) : Prop :=
  180 ≤ β ∧ β < 270

noncomputable def equivalent_theta (α : ℝ) (θ : ℝ) : Prop :=
  (α % 360 = θ % 360) ∧ (-720 ≤ θ ∧ θ < 0)

theorem quadrant_and_terminal_angle :
  in_quadrant_3 (normalize_angle alpha) ∧ 
  (equivalent_theta alpha (-110) ∨ equivalent_theta alpha (-470)) :=
by 
  sorry

end quadrant_and_terminal_angle_l214_214658


namespace original_daily_production_l214_214684

theorem original_daily_production (x N : ℕ) (h1 : N = (x - 3) * 31 + 60) (h2 : N = (x + 3) * 25 - 60) : x = 8 :=
sorry

end original_daily_production_l214_214684


namespace range_of_a_monotonically_decreasing_l214_214973

-- Definitions
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- Lean statement
theorem range_of_a_monotonically_decreasing {a : ℝ} : 
  (∀ x y : ℝ, -2 ≤ x → x ≤ 4 → -2 ≤ y → y ≤ 4 → x < y → f a y < f a x) ↔ a ≤ -3 := 
by 
  sorry

end range_of_a_monotonically_decreasing_l214_214973


namespace money_per_postcard_l214_214248

def postcards_per_day : ℕ := 30
def days : ℕ := 6
def total_earning : ℕ := 900
def total_postcards := postcards_per_day * days
def price_per_postcard := total_earning / total_postcards

theorem money_per_postcard :
  price_per_postcard = 5 := 
sorry

end money_per_postcard_l214_214248


namespace find_percentage_l214_214347

/-- 
Given some percentage P of 6,000, when subtracted from 1/10th of 6,000 (which is 600), 
the difference is 693. Prove that P equals 1.55.
-/
theorem find_percentage (P : ℝ) (h₁ : 6000 / 10 = 600) (h₂ : 600 - (P / 100) * 6000 = 693) : 
  P = 1.55 :=
  sorry

end find_percentage_l214_214347


namespace chime_2203_occurs_on_March_19_l214_214182

-- Define the initial conditions: chime patterns
def chimes_at_half_hour : Nat := 1
def chimes_at_hour (h : Nat) : Nat := if h = 12 then 12 else h % 12

-- Define the start time and the question parameters
def start_time_hours : Nat := 10
def start_time_minutes : Nat := 45
def start_day : Nat := 26 -- Assume February 26 as starting point, to facilitate day count accurately
def target_chime : Nat := 2203

-- Define the date calculation function (based on given solution steps)
noncomputable def calculate_chime_date (start_day : Nat) : Nat := sorry

-- The goal is to prove calculate_chime_date with given start conditions equals 19 (March 19th is the 19th day after the base day assumption of March 0)
theorem chime_2203_occurs_on_March_19 :
  calculate_chime_date start_day = 19 :=
sorry

end chime_2203_occurs_on_March_19_l214_214182


namespace total_points_scored_l214_214964

theorem total_points_scored (m1 m2 m3 m4 m5 m6 j1 j2 j3 j4 j5 j6 : ℕ) :
  m1 = 5 → j1 = m1 + 2 →
  m2 = 7 → j2 = m2 - 3 →
  m3 = 10 → j3 = m3 / 2 →
  m4 = 12 → j4 = m4 * 2 →
  m5 = 6 → j5 = m5 →
  j6 = 8 → m6 = j6 + 4 →
  m1 + m2 + m3 + m4 + m5 + m6 + j1 + j2 + j3 + j4 + j5 + j6 = 106 :=
by
  intros
  sorry

end total_points_scored_l214_214964


namespace ch_sub_ch_add_sh_sub_sh_add_l214_214958

noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

theorem ch_sub (x y : ℝ) : ch (x - y) = ch x * ch y - sh x * sh y := sorry
theorem ch_add (x y : ℝ) : ch (x + y) = ch x * ch y + sh x * sh y := sorry
theorem sh_sub (x y : ℝ) : sh (x - y) = sh x * ch y - ch x * sh y := sorry
theorem sh_add (x y : ℝ) : sh (x + y) = sh x * ch y + ch x * sh y := sorry

end ch_sub_ch_add_sh_sub_sh_add_l214_214958


namespace numerical_value_expression_l214_214980

theorem numerical_value_expression (x y z : ℚ) (h1 : x - 4 * y - 2 * z = 0) (h2 : 3 * x + 2 * y - z = 0) (h3 : z ≠ 0) : 
  (x^2 - 5 * x * y) / (2 * y^2 + z^2) = 164 / 147 :=
by sorry

end numerical_value_expression_l214_214980


namespace calc_xy_square_l214_214691

theorem calc_xy_square
  (x y z : ℝ)
  (h1 : 2 * x * (y + z) = 1 + y * z)
  (h2 : 1 / x - 2 / y = 3 / 2)
  (h3 : x + y + 1 / 2 = 0) :
  (x + y + z) ^ 2 = 1 :=
by
  sorry

end calc_xy_square_l214_214691


namespace big_joe_height_is_8_l214_214786

variable (Pepe_height Frank_height Larry_height Ben_height BigJoe_height : ℝ)

axiom Pepe_height_def : Pepe_height = 4.5
axiom Frank_height_def : Frank_height = Pepe_height + 0.5
axiom Larry_height_def : Larry_height = Frank_height + 1
axiom Ben_height_def : Ben_height = Larry_height + 1
axiom BigJoe_height_def : BigJoe_height = Ben_height + 1

theorem big_joe_height_is_8 :
  BigJoe_height = 8 :=
sorry

end big_joe_height_is_8_l214_214786


namespace johns_average_speed_l214_214409

-- Definitions based on conditions
def cycling_distance_uphill := 3 -- in km
def cycling_time_uphill := 45 / 60 -- in hr (45 minutes)

def cycling_distance_downhill := 3 -- in km
def cycling_time_downhill := 15 / 60 -- in hr (15 minutes)

def walking_distance := 2 -- in km
def walking_time := 20 / 60 -- in hr (20 minutes)

-- Definition for total distance traveled
def total_distance := cycling_distance_uphill + cycling_distance_downhill + walking_distance

-- Definition for total time spent traveling
def total_time := cycling_time_uphill + cycling_time_downhill + walking_time

-- Definition for average speed
def average_speed := total_distance / total_time

-- Proof statement
theorem johns_average_speed : average_speed = 6 := by
  sorry

end johns_average_speed_l214_214409


namespace solve_for_x_l214_214611

theorem solve_for_x (x : ℝ) (h : 0 < x) (h_property : (x / 100) * x^2 = 9) : x = 10 := by
  sorry

end solve_for_x_l214_214611


namespace quartic_poly_roots_l214_214995

noncomputable def roots_polynomial : List ℝ := [
  (1 + Real.sqrt 5) / 2,
  (1 - Real.sqrt 5) / 2,
  (3 + Real.sqrt 13) / 6,
  (3 - Real.sqrt 13) / 6
]

theorem quartic_poly_roots :
  ∀ x : ℝ, x ∈ roots_polynomial ↔ 3*x^4 - 4*x^3 - 5*x^2 - 4*x + 3 = 0 :=
by sorry

end quartic_poly_roots_l214_214995


namespace arithmetic_seq_common_diff_l214_214765

theorem arithmetic_seq_common_diff
  (a₃ a₇ S₁₀ : ℤ)
  (h₁ : a₃ + a₇ = 16)
  (h₂ : S₁₀ = 85)
  (a₃_eq : ∃ a₁ d : ℤ, a₃ = a₁ + 2 * d)
  (a₇_eq : ∃ a₁ d : ℤ, a₇ = a₁ + 6 * d)
  (S₁₀_eq : ∃ a₁ d : ℤ, S₁₀ = 10 * a₁ + 45 * d) :
  ∃ d : ℤ, d = 1 :=
by
  sorry

end arithmetic_seq_common_diff_l214_214765


namespace remainder_when_divided_by_x_minus_2_l214_214856

def p (x : ℕ) : ℕ := x^5 - 2 * x^3 + 4 * x + 5

theorem remainder_when_divided_by_x_minus_2 : p 2 = 29 := 
by {
  sorry
}

end remainder_when_divided_by_x_minus_2_l214_214856


namespace find_m_for_opposite_solutions_l214_214994

theorem find_m_for_opposite_solutions (x y m : ℝ) 
  (h1 : x = -y)
  (h2 : 3 * x + 5 * y = 2)
  (h3 : 2 * x + 7 * y = m - 18) : 
  m = 23 :=
sorry

end find_m_for_opposite_solutions_l214_214994


namespace tan_of_angle_123_l214_214653

variable (a : ℝ)
variable (h : Real.sin 123 = a)

theorem tan_of_angle_123 : Real.tan 123 = a / Real.cos 123 :=
by
  sorry

end tan_of_angle_123_l214_214653


namespace abs_diff_26th_term_l214_214535

def C (n : ℕ) : ℤ := 50 + 15 * (n - 1)
def D (n : ℕ) : ℤ := 85 - 20 * (n - 1)

theorem abs_diff_26th_term :
  |(C 26) - (D 26)| = 840 := by
  sorry

end abs_diff_26th_term_l214_214535


namespace Robin_hair_initial_length_l214_214306

theorem Robin_hair_initial_length (x : ℝ) (h1 : x + 8 - 20 = 2) : x = 14 :=
by
  sorry

end Robin_hair_initial_length_l214_214306


namespace find_y_l214_214116

def G (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y (y : ℕ) (h : G 3 y 5 18 = 500) : y = 6 :=
sorry

end find_y_l214_214116


namespace coprime_exists_pow_divisible_l214_214207

theorem coprime_exists_pow_divisible (a n : ℕ) (h_coprime : Nat.gcd a n = 1) : 
  ∃ m : ℕ, n ∣ a^m - 1 :=
by
  sorry

end coprime_exists_pow_divisible_l214_214207


namespace parabola_focus_coordinates_l214_214575

theorem parabola_focus_coordinates (x y : ℝ) (h : y = 2 * x^2) : (0, 1 / 8) = (0, 1 / 8) :=
by
  sorry

end parabola_focus_coordinates_l214_214575


namespace neg_p_l214_214332

-- Proposition p : For any x in ℝ, cos x ≤ 1
def p : Prop := ∀ (x : ℝ), Real.cos x ≤ 1

-- Negation of p: There exists an x₀ in ℝ such that cos x₀ > 1
theorem neg_p : ¬p ↔ (∃ (x₀ : ℝ), Real.cos x₀ > 1) := sorry

end neg_p_l214_214332


namespace marty_combinations_l214_214900

theorem marty_combinations : 
  let C := 5
  let P := 4
  C * P = 20 :=
by
  sorry

end marty_combinations_l214_214900


namespace determine_points_on_line_l214_214979

def pointA : ℝ × ℝ := (2, 5)
def pointB : ℝ × ℝ := (1, 2.2)
def line_eq (x y : ℝ) : ℝ := 3 * x - 5 * y + 8

theorem determine_points_on_line :
  (line_eq pointA.1 pointA.2 ≠ 0) ∧ (line_eq pointB.1 pointB.2 = 0) :=
by
  sorry

end determine_points_on_line_l214_214979


namespace average_difference_l214_214096

def differences : List ℤ := [15, -5, 25, 35, -15, 10, 20]
def days : ℤ := 7

theorem average_difference (diff : List ℤ) (n : ℤ) 
  (h : diff = [15, -5, 25, 35, -15, 10, 20]) (h_days : n = 7) : 
  (diff.sum / n : ℚ) = 12 := 
by 
  rw [h, h_days]
  norm_num
  sorry

end average_difference_l214_214096


namespace statement_1_statement_2_statement_3_statement_4_l214_214316

variables (a b c x0 : ℝ)
noncomputable def P (x : ℝ) : ℝ := a*x^2 + b*x + c

-- Statement ①
theorem statement_1 (h : a - b + c = 0) : P a b c (-1) = 0 := sorry

-- Statement ②
theorem statement_2 (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a*x1^2 + c = 0 ∧ a*x2^2 + c = 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ P a b c x1 = 0 ∧ P a b c x2 = 0 := sorry

-- Statement ③
theorem statement_3 (h : P a b c c = 0) : a*c + b + 1 = 0 := sorry

-- Statement ④
theorem statement_4 (h : P a b c x0 = 0) : b^2 - 4*a*c = (2*a*x0 + b)^2 := sorry

end statement_1_statement_2_statement_3_statement_4_l214_214316


namespace value_of_mn_l214_214912

theorem value_of_mn (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : m^4 - n^4 = 3439) : m * n = 90 := 
by sorry

end value_of_mn_l214_214912


namespace letter_ratio_l214_214769

theorem letter_ratio (G B M : ℕ) (h1 : G = B + 10) 
                     (h2 : B = 40) 
                     (h3 : G + B + M = 270) : 
                     M / (G + B) = 2 := 
by 
  sorry

end letter_ratio_l214_214769


namespace base7_digits_of_143_l214_214951

theorem base7_digits_of_143 : ∃ d1 d2 d3 : ℕ, (d1 < 7 ∧ d2 < 7 ∧ d3 < 7) ∧ (143 = d1 * 49 + d2 * 7 + d3) ∧ (d1 = 2 ∧ d2 = 6 ∧ d3 = 3) :=
by
  sorry

end base7_digits_of_143_l214_214951


namespace trout_to_bass_ratio_l214_214720

theorem trout_to_bass_ratio 
  (bass : ℕ) 
  (trout : ℕ) 
  (blue_gill : ℕ)
  (h1 : bass = 32) 
  (h2 : blue_gill = 2 * bass) 
  (h3 : bass + trout + blue_gill = 104) 
  : (trout / bass) = 1 / 4 :=
by 
  -- intermediate steps can be included here
  sorry

end trout_to_bass_ratio_l214_214720


namespace ratio_of_percent_changes_l214_214581

noncomputable def price_decrease_ratio (original_price : ℝ) (new_price : ℝ) : ℝ :=
(original_price - new_price) / original_price * 100

noncomputable def units_increase_ratio (original_units : ℝ) (new_units : ℝ) : ℝ :=
(new_units - original_units) / original_units * 100

theorem ratio_of_percent_changes 
  (original_price new_price original_units new_units : ℝ)
  (h1 : new_price = 0.7 * original_price)
  (h2 : original_price * original_units = new_price * new_units)
  : (units_increase_ratio original_units new_units) / (price_decrease_ratio original_price new_price) = 1.4285714285714286 :=
by
  sorry

end ratio_of_percent_changes_l214_214581


namespace quadratic_has_one_solution_implies_m_eq_3_l214_214042

theorem quadratic_has_one_solution_implies_m_eq_3 {m : ℝ} (h : ∃ x : ℝ, 3 * x^2 - 6 * x + m = 0 ∧ ∃! u, 3 * u^2 - 6 * u + m = 0) : m = 3 :=
by sorry

end quadratic_has_one_solution_implies_m_eq_3_l214_214042


namespace contrapositive_of_real_roots_l214_214019

variable {a : ℝ}

theorem contrapositive_of_real_roots :
  (1 + 4 * a < 0) → (a < 0) := by
  sorry

end contrapositive_of_real_roots_l214_214019


namespace integer_fraction_condition_l214_214913

theorem integer_fraction_condition (p : ℕ) (h_pos : 0 < p) :
  (∃ k : ℤ, k > 0 ∧ (5 * p + 15) = k * (3 * p - 9)) ↔ (4 ≤ p ∧ p ≤ 19) :=
by
  sorry

end integer_fraction_condition_l214_214913


namespace volleyball_tournament_first_place_score_l214_214602

theorem volleyball_tournament_first_place_score :
  ∃ (a b c d : ℕ), (a + b + c + d = 18) ∧ (a < b ∧ b < c ∧ c < d) ∧ (d = 6) :=
by
  sorry

end volleyball_tournament_first_place_score_l214_214602


namespace tilly_star_count_l214_214090

theorem tilly_star_count (stars_east : ℕ) (stars_west : ℕ) (total_stars : ℕ) 
  (h1 : stars_east = 120)
  (h2 : stars_west = 6 * stars_east)
  (h3 : total_stars = stars_east + stars_west) :
  total_stars = 840 :=
sorry

end tilly_star_count_l214_214090


namespace white_marbles_count_l214_214634

theorem white_marbles_count (total_marbles blue_marbles red_marbles : ℕ) (probability_red_or_white : ℚ)
    (h_total : total_marbles = 60)
    (h_blue : blue_marbles = 5)
    (h_red : red_marbles = 9)
    (h_probability : probability_red_or_white = 0.9166666666666666) :
    ∃ W : ℕ, W = total_marbles - blue_marbles - red_marbles ∧ probability_red_or_white = (red_marbles + W)/(total_marbles) ∧ W = 46 :=
by
  sorry

end white_marbles_count_l214_214634


namespace sqrt_product_simplify_l214_214902

theorem sqrt_product_simplify (x : ℝ) (hx : 0 ≤ x):
  Real.sqrt (48*x) * Real.sqrt (3*x) * Real.sqrt (50*x) = 60 * x * Real.sqrt x := 
by
  sorry

end sqrt_product_simplify_l214_214902


namespace abc_eq_zero_l214_214766

variable (a b c : ℝ) (n : ℕ)

theorem abc_eq_zero
  (h1 : a^n + b^n = c^n)
  (h2 : a^(n+1) + b^(n+1) = c^(n+1))
  (h3 : a^(n+2) + b^(n+2) = c^(n+2)) :
  a * b * c = 0 :=
sorry

end abc_eq_zero_l214_214766


namespace recurring_decimal_exceeds_by_fraction_l214_214817

theorem recurring_decimal_exceeds_by_fraction : 
  let y := (36 : ℚ) / 99
  let x := (36 : ℚ) / 100
  ((4 : ℚ) / 11) - x = (4 : ℚ) / 1100 :=
by
  sorry

end recurring_decimal_exceeds_by_fraction_l214_214817


namespace sum_of_coefficients_l214_214755

theorem sum_of_coefficients (a b : ℝ)
  (h1 : 15 * a^4 * b^2 = 135)
  (h2 : 6 * a^5 * b = -18) :
  (a + b)^6 = 64 := by
  sorry

end sum_of_coefficients_l214_214755


namespace min_sum_reciprocal_l214_214704

theorem min_sum_reciprocal (a b c : ℝ) (hp0 : 0 < a) (hp1 : 0 < b) (hp2 : 0 < c) (h : a + b + c = 1) : 
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
by
  sorry

end min_sum_reciprocal_l214_214704


namespace salary_increase_percentage_l214_214334

theorem salary_increase_percentage (old_salary new_salary : ℕ) (h1 : old_salary = 10000) (h2 : new_salary = 10200) : 
    ((new_salary - old_salary) / old_salary : ℚ) * 100 = 2 := 
by 
  sorry

end salary_increase_percentage_l214_214334


namespace smallest_number_diminished_by_35_l214_214754

def lcm_list (l : List ℕ) : ℕ := l.foldr Nat.lcm 1

def conditions : List ℕ := [5, 10, 15, 20, 25, 30, 35]

def lcm_conditions := lcm_list conditions

theorem smallest_number_diminished_by_35 :
  ∃ n, n - 35 = lcm_conditions :=
sorry

end smallest_number_diminished_by_35_l214_214754


namespace prob_dominant_trait_one_child_prob_at_least_one_dominant_trait_two_children_l214_214412

-- Define the probability of a genotype given two mixed genotype (rd) parents producing a child.
def prob_genotype_dd : ℚ := (1/2) * (1/2)
def prob_genotype_rr : ℚ := (1/2) * (1/2)
def prob_genotype_rd : ℚ := 2 * (1/2) * (1/2)

-- Assertion that the probability of a child displaying the dominant characteristic (dd or rd) is 3/4.
theorem prob_dominant_trait_one_child : 
  prob_genotype_dd + prob_genotype_rd = 3/4 := sorry

-- Define the probability of two children both being rr.
def prob_both_rr_two_children : ℚ := prob_genotype_rr * prob_genotype_rr

-- Assertion that the probability of at least one of two children displaying the dominant characteristic is 15/16.
theorem prob_at_least_one_dominant_trait_two_children : 
  1 - prob_both_rr_two_children = 15/16 := sorry

end prob_dominant_trait_one_child_prob_at_least_one_dominant_trait_two_children_l214_214412


namespace probability_one_from_each_l214_214927

-- Define the total number of cards
def total_cards : ℕ := 10

-- Define the number of cards from Amelia's name
def amelia_cards : ℕ := 6

-- Define the number of cards from Lucas's name
def lucas_cards : ℕ := 4

-- Define the probability that one letter is from each person's name
theorem probability_one_from_each : (amelia_cards / total_cards) * (lucas_cards / (total_cards - 1)) +
                                    (lucas_cards / total_cards) * (amelia_cards / (total_cards - 1)) = 8 / 15 :=
by
  sorry

end probability_one_from_each_l214_214927


namespace math_problem_l214_214106

theorem math_problem :
  8 / 4 - 3^2 + 4 * 2 + (Nat.factorial 5) = 121 :=
by
  sorry

end math_problem_l214_214106


namespace total_residents_l214_214764

open Set

/-- 
In a village, there are 912 residents who speak Bashkir, 
653 residents who speak Russian, 
and 435 residents who speak both languages.
Prove the total number of residents in the village is 1130.
-/
theorem total_residents (A B : Finset ℕ) (nA nB nAB : ℕ)
  (hA : nA = 912)
  (hB : nB = 653)
  (hAB : nAB = 435) :
  nA + nB - nAB = 1130 := by
  sorry

end total_residents_l214_214764


namespace remainder_of_3_pow_2023_mod_7_l214_214194

theorem remainder_of_3_pow_2023_mod_7 :
  (3^2023) % 7 = 3 := 
by
  sorry

end remainder_of_3_pow_2023_mod_7_l214_214194


namespace triangle_problems_l214_214827

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {S : ℝ}

theorem triangle_problems
  (h1 : (2 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a = Real.sqrt 13)
  (h3 : b + c = 5) :
  (A = π / 3) ∧ (S = Real.sqrt 3) :=
by
  sorry

end triangle_problems_l214_214827


namespace projection_inequality_l214_214256

theorem projection_inequality
  (a b c : ℝ)
  (h : c^2 = a^2 + b^2) :
  c ≥ (a + b) / Real.sqrt 2 :=
by
  sorry

end projection_inequality_l214_214256


namespace max_angle_AFB_l214_214679

noncomputable def focus_of_parabola := (2, 0)
def parabola (x y : ℝ) := y^2 = 8 * x
def on_parabola (A B : ℝ × ℝ) := parabola A.1 A.2 ∧ parabola B.1 B.2
def condition (x1 x2 : ℝ) (AB : ℝ) := x1 + x2 + 4 = (2 * Real.sqrt 3 / 3) * AB

theorem max_angle_AFB (A B : ℝ × ℝ) (x1 x2 : ℝ) (AB : ℝ)
  (h1 : on_parabola A B)
  (h2 : condition x1 x2 AB)
  (hA : A.1 = x1)
  (hB : B.1 = x2) :
  ∃ θ, θ ≤ Real.pi * 2 / 3 := 
  sorry

end max_angle_AFB_l214_214679


namespace combination_property_l214_214433

theorem combination_property (x : ℕ) (hx : 2 * x - 1 ≤ 11 ∧ x ≤ 11) :
  (Nat.choose 11 (2 * x - 1) = Nat.choose 11 x) → (x = 1 ∨ x = 4) :=
by
  sorry

end combination_property_l214_214433


namespace fill_in_the_blank_l214_214309

-- Definitions of the problem conditions
def parent := "being a parent"
def parent_with_special_needs := "being the parent of a child with special needs"

-- The sentence describing two situations of being a parent
def sentence1 := "Being a parent is not always easy"
def sentence2 := "being the parent of a child with special needs often carries with ___ extra stress."

-- The correct word to fill in the blank.
def correct_answer := "it"

-- Proof problem
theorem fill_in_the_blank : correct_answer = "it" :=
by
  sorry

end fill_in_the_blank_l214_214309


namespace total_goals_is_50_l214_214506

def team_a_first_half_goals := 8
def team_b_first_half_goals := team_a_first_half_goals / 2
def team_c_first_half_goals := 2 * team_b_first_half_goals
def team_a_first_half_missed_penalty := 1
def team_c_first_half_missed_penalty := 2

def team_a_second_half_goals := team_c_first_half_goals
def team_b_second_half_goals := team_a_first_half_goals
def team_c_second_half_goals := team_b_second_half_goals + 3
def team_a_second_half_successful_penalty := 1
def team_b_second_half_successful_penalty := 2

def total_team_a_goals := team_a_first_half_goals + team_a_second_half_goals + team_a_second_half_successful_penalty
def total_team_b_goals := team_b_first_half_goals + team_b_second_half_goals + team_b_second_half_successful_penalty
def total_team_c_goals := team_c_first_half_goals + team_c_second_half_goals

def total_goals := total_team_a_goals + total_team_b_goals + total_team_c_goals

theorem total_goals_is_50 : total_goals = 50 := by
  unfold total_goals
  unfold total_team_a_goals total_team_b_goals total_team_c_goals
  unfold team_a_first_half_goals team_b_first_half_goals team_c_first_half_goals
  unfold team_a_second_half_goals team_b_second_half_goals team_c_second_half_goals
  unfold team_a_second_half_successful_penalty team_b_second_half_successful_penalty
  sorry

end total_goals_is_50_l214_214506


namespace factor_values_l214_214098

theorem factor_values (a b : ℤ) :
  (∀ s : ℂ, s^2 - s - 1 = 0 → a * s^15 + b * s^14 + 1 = 0) ∧
  (∀ t : ℂ, t^2 - t - 1 = 0 → a * t^15 + b * t^14 + 1 = 0) →
  a = 377 ∧ b = -610 :=
by
  sorry

end factor_values_l214_214098


namespace curve_intersection_three_points_l214_214705

theorem curve_intersection_three_points (a : ℝ) :
  (∀ x y : ℝ, ((x^2 - y^2 = a^2) ∧ ((x-1)^2 + y^2 = 1)) → (a = 0)) :=
by
  sorry

end curve_intersection_three_points_l214_214705


namespace quadratic_roots_sum_product_l214_214346

theorem quadratic_roots_sum_product : 
  ∃ x1 x2 : ℝ, (x1^2 - 2*x1 - 4 = 0) ∧ (x2^2 - 2*x2 - 4 = 0) ∧ 
  (x1 ≠ x2) ∧ (x1 + x2 + x1 * x2 = -2) :=
sorry

end quadratic_roots_sum_product_l214_214346


namespace ratio_of_areas_of_triangles_l214_214678

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l214_214678


namespace length_of_water_fountain_l214_214007

theorem length_of_water_fountain :
  (∀ (L1 : ℕ), 20 * 14 = L1) ∧
  (35 * 3 = 21) →
  (20 * 14 = 56) := by
sorry

end length_of_water_fountain_l214_214007


namespace arithmetic_sequence_first_term_and_difference_l214_214532

theorem arithmetic_sequence_first_term_and_difference
  (a1 d : ℤ)
  (h1 : (a1 + 2 * d) * (a1 + 5 * d) = 406)
  (h2 : a1 + 8 * d = 2 * (a1 + 3 * d) + 6) : 
  a1 = 4 ∧ d = 5 :=
by 
  sorry

end arithmetic_sequence_first_term_and_difference_l214_214532


namespace p_sufficient_but_not_necessary_for_q_l214_214785

variable (x : ℝ) (p q : Prop)

def p_condition : Prop := 0 < x ∧ x < 1
def q_condition : Prop := x^2 < 2 * x

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p_condition x → q_condition x) ∧
  ¬ (∀ x : ℝ, q_condition x → p_condition x) := by
  sorry

end p_sufficient_but_not_necessary_for_q_l214_214785


namespace inscribed_triangle_perimeter_geq_half_l214_214377

theorem inscribed_triangle_perimeter_geq_half (a : ℝ) (s' : ℝ) (h_a_pos : a > 0) 
  (h_equilateral : ∀ (A B C : Type) (a b c : A), a = b ∧ b = c ∧ c = a) :
  2 * s' >= (3 * a) / 2 :=
by
  sorry

end inscribed_triangle_perimeter_geq_half_l214_214377


namespace solve_abs_eq_l214_214984

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 :=
  sorry

end solve_abs_eq_l214_214984


namespace solve_problem_l214_214208

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem solve_problem :
  is_prime 2017 :=
by
  have h1 : 2017 > 1 := by linarith
  have h2 : ∀ m : ℕ, m ∣ 2017 → m = 1 ∨ m = 2017 :=
    sorry
  exact ⟨h1, h2⟩

end solve_problem_l214_214208


namespace integer_ratio_condition_l214_214989

variable {x y : ℝ}

theorem integer_ratio_condition (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) (h3 : ∃ t : ℤ, x = t * y) : 
  ∃ t : ℤ, t = -2 := sorry

end integer_ratio_condition_l214_214989


namespace find_C_given_eq_statement_max_area_triangle_statement_l214_214286

open Real

noncomputable def find_C_given_eq (a b c A : ℝ) (C : ℝ) : Prop :=
  (2 * a = sqrt 3 * c * sin A - a * cos C) → 
  C = 2 * π / 3

noncomputable def max_area_triangle (a b c : ℝ) (C : ℝ) : Prop :=
  C = 2 * π / 3 →
  c = sqrt 3 →
  ∃ S, S = (sqrt 3 / 4) * a * b ∧ 
  ∀ a b : ℝ, a * b ≤ 1 → S = (sqrt 3 / 4)

-- Lean statements
theorem find_C_given_eq_statement (a b c A C : ℝ) : find_C_given_eq a b c A C := 
by sorry

theorem max_area_triangle_statement (a b c : ℝ) (C : ℝ) : max_area_triangle a b c C := 
by sorry

end find_C_given_eq_statement_max_area_triangle_statement_l214_214286


namespace unique_three_digit_numbers_l214_214032

theorem unique_three_digit_numbers (d1 d2 d3 : ℕ) :
  (d1 = 3 ∧ d2 = 0 ∧ d3 = 8) →
  ∃ nums : Finset ℕ, 
  (∀ n ∈ nums, (∃ h t u : ℕ, n = 100 * h + 10 * t + u ∧ 
                h ≠ 0 ∧ (h = d1 ∨ h = d2 ∨ h = d3) ∧ 
                (t = d1 ∨ t = d2 ∨ t = d3) ∧ (u = d1 ∨ u = d2 ∨ u = d3) ∧ 
                h ≠ t ∧ t ≠ u ∧ u ≠ h)) ∧ nums.card = 4 :=
by
  sorry

end unique_three_digit_numbers_l214_214032


namespace number_of_true_propositions_l214_214378

def inverse_proposition (x y : ℝ) : Prop :=
  ¬(x + y = 0 → (x ≠ -y))

def contrapositive_proposition (a b : ℝ) : Prop :=
  (a^2 ≤ b^2) → (a ≤ b)

def negation_proposition (x : ℝ) : Prop :=
  (x ≤ -3) → ¬(x^2 + x - 6 > 0)

theorem number_of_true_propositions : 
  (∃ (x y : ℝ), inverse_proposition x y) ∧
  (∃ (a b : ℝ), contrapositive_proposition a b) ∧
  ¬(∃ (x : ℝ), negation_proposition x) → 
  2 = 2 :=
by
  sorry

end number_of_true_propositions_l214_214378


namespace minimum_number_of_rooks_l214_214870

theorem minimum_number_of_rooks (n : ℕ) : 
  ∃ (num_rooks : ℕ), (∀ (cells_colored : ℕ), cells_colored = n^2 → num_rooks = n) :=
by sorry

end minimum_number_of_rooks_l214_214870


namespace problem_equivalent_l214_214956

theorem problem_equivalent :
  500 * 2019 * 0.0505 * 20 = 2019^2 :=
by
  sorry

end problem_equivalent_l214_214956


namespace letter_addition_problem_l214_214221

theorem letter_addition_problem (S I X : ℕ) (E L V N : ℕ) 
  (hS : S = 8) 
  (hX_odd : X % 2 = 1)
  (h_diff_digits : ∀ (a b c d e f : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ a)
  (h_sum : 2 * S * 100 + 2 * I * 10 + 2 * X = E * 10000 + L * 1000 + E * 100 + V * 10 + E + N) :
  I = 3 :=
by
  sorry

end letter_addition_problem_l214_214221


namespace composite_for_positive_integers_l214_214873

def is_composite (n : ℤ) : Prop :=
  ∃ a b : ℤ, 1 < a ∧ 1 < b ∧ n = a * b

theorem composite_for_positive_integers (n : ℕ) (h_pos : 1 < n) :
  is_composite (3^(2*n+1) - 2^(2*n+1) - 6*n) := 
sorry

end composite_for_positive_integers_l214_214873


namespace soccer_tournament_matches_l214_214180

theorem soccer_tournament_matches (x : ℕ) (h : 1 ≤ x) : (1 / 2 : ℝ) * x * (x - 1) = 45 := sorry

end soccer_tournament_matches_l214_214180


namespace min_vases_required_l214_214619

theorem min_vases_required (carnations roses tulips lilies : ℕ)
  (flowers_in_A flowers_in_B flowers_in_C : ℕ) 
  (total_flowers : ℕ) 
  (h_carnations : carnations = 10) 
  (h_roses : roses = 25) 
  (h_tulips : tulips = 15) 
  (h_lilies : lilies = 20)
  (h_flowers_in_A : flowers_in_A = 4) 
  (h_flowers_in_B : flowers_in_B = 6) 
  (h_flowers_in_C : flowers_in_C = 8)
  (h_total_flowers : total_flowers = carnations + roses + tulips + lilies) :
  total_flowers = 70 → 
  (exists vases_A vases_B vases_C : ℕ, 
    vases_A = 0 ∧ 
    vases_B = 1 ∧ 
    vases_C = 8 ∧ 
    total_flowers = vases_A * flowers_in_A + vases_B * flowers_in_B + vases_C * flowers_in_C) :=
by
  intros
  sorry

end min_vases_required_l214_214619


namespace banana_unique_permutations_l214_214529

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l214_214529


namespace mark_bought_5_pounds_of_apples_l214_214161

noncomputable def cost_of_tomatoes (pounds_tomatoes : ℕ) (cost_per_pound_tomato : ℝ) : ℝ :=
  pounds_tomatoes * cost_per_pound_tomato

noncomputable def cost_of_apples (total_spent : ℝ) (cost_of_tomatoes : ℝ) : ℝ :=
  total_spent - cost_of_tomatoes

noncomputable def pounds_of_apples (cost_of_apples : ℝ) (cost_per_pound_apples : ℝ) : ℝ :=
  cost_of_apples / cost_per_pound_apples

theorem mark_bought_5_pounds_of_apples (pounds_tomatoes : ℕ) (cost_per_pound_tomato : ℝ) 
  (total_spent : ℝ) (cost_per_pound_apples : ℝ) :
  pounds_tomatoes = 2 →
  cost_per_pound_tomato = 5 →
  total_spent = 40 →
  cost_per_pound_apples = 6 →
  pounds_of_apples (cost_of_apples total_spent (cost_of_tomatoes pounds_tomatoes cost_per_pound_tomato)) cost_per_pound_apples = 5 := by
  intros h1 h2 h3 h4
  sorry

end mark_bought_5_pounds_of_apples_l214_214161


namespace worm_length_difference_l214_214432

theorem worm_length_difference
  (worm1 worm2: ℝ)
  (h_worm1: worm1 = 0.8)
  (h_worm2: worm2 = 0.1) :
  worm1 - worm2 = 0.7 :=
by
  -- starting the proof
  sorry

end worm_length_difference_l214_214432


namespace complement_union_example_l214_214644

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3, 4}

-- Define set B
def B : Set ℕ := {2, 4}

-- State the theorem we want to prove
theorem complement_union_example : (U \ A) ∪ B = {2, 4, 5} :=
by
  sorry

end complement_union_example_l214_214644


namespace average_speed_of_tiger_exists_l214_214262

-- Conditions
def head_start_distance (v_t : ℝ) : ℝ := 5 * v_t
def zebra_distance : ℝ := 6 * 55
def tiger_distance (v_t : ℝ) : ℝ := 6 * v_t

-- Problem statement
theorem average_speed_of_tiger_exists (v_t : ℝ) (h : zebra_distance = head_start_distance v_t + tiger_distance v_t) : v_t = 30 :=
by
  sorry

end average_speed_of_tiger_exists_l214_214262


namespace smallest_sum_of_digits_l214_214725

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_sum_of_digits (N : ℕ) (hN_pos : 0 < N) 
  (h : sum_of_digits N = 3 * sum_of_digits (N + 1)) :
  sum_of_digits N = 12 :=
by {
  sorry
}

end smallest_sum_of_digits_l214_214725


namespace pow_mod_l214_214278

theorem pow_mod (h : 3^3 ≡ 1 [MOD 13]) : 3^21 ≡ 1 [MOD 13] :=
by
sorry

end pow_mod_l214_214278


namespace fraction_of_unoccupied_chairs_is_two_fifths_l214_214298

noncomputable def fraction_unoccupied_chairs (total_chairs : ℕ) (chair_capacity : ℕ) (attended_board_members : ℕ) : ℚ :=
  let total_capacity := total_chairs * chair_capacity
  let total_board_members := total_capacity
  let unoccupied_members := total_board_members - attended_board_members
  let unoccupied_chairs := unoccupied_members / chair_capacity
  unoccupied_chairs / total_chairs

theorem fraction_of_unoccupied_chairs_is_two_fifths :
  fraction_unoccupied_chairs 40 2 48 = 2 / 5 :=
by
  sorry

end fraction_of_unoccupied_chairs_is_two_fifths_l214_214298


namespace dilution_problem_l214_214271

theorem dilution_problem
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (desired_concentration : ℝ)
  (initial_alcohol_content : initial_concentration * initial_volume / 100 = 4.8)
  (desired_alcohol_content : 4.8 = desired_concentration * (initial_volume + N) / 100)
  (N : ℝ) :
  N = 11.2 :=
sorry

end dilution_problem_l214_214271


namespace binomial_10_10_binomial_10_9_l214_214132

-- Prove that \(\binom{10}{10} = 1\)
theorem binomial_10_10 : Nat.choose 10 10 = 1 :=
by sorry

-- Prove that \(\binom{10}{9} = 10\)
theorem binomial_10_9 : Nat.choose 10 9 = 10 :=
by sorry

end binomial_10_10_binomial_10_9_l214_214132


namespace lion_turn_angles_l214_214545

-- Define the radius of the circle
def radius (r : ℝ) := r = 10

-- Define the path length the lion runs in meters
def path_length (d : ℝ) := d = 30000

-- Define the final goal: The sum of all the angles of its turns is at least 2998 radians
theorem lion_turn_angles (r d : ℝ) (α : ℝ) (hr : radius r) (hd : path_length d) (hα : d ≤ 10 * α) : α ≥ 2998 := 
sorry

end lion_turn_angles_l214_214545


namespace negation_of_at_most_four_l214_214934

theorem negation_of_at_most_four (n : ℕ) : ¬(n ≤ 4) → n ≥ 5 := 
by
  sorry

end negation_of_at_most_four_l214_214934


namespace jacques_initial_gumballs_l214_214066

def joanna_initial_gumballs : ℕ := 40
def each_shared_gumballs_after_purchase : ℕ := 250

theorem jacques_initial_gumballs (J : ℕ) (h : 2 * (joanna_initial_gumballs + J + 4 * (joanna_initial_gumballs + J)) = 2 * each_shared_gumballs_after_purchase) : J = 60 :=
by
  sorry

end jacques_initial_gumballs_l214_214066


namespace trapezoid_area_l214_214586

theorem trapezoid_area (h : ℝ) : 
  let b1 : ℝ := 4 * h + 2
  let b2 : ℝ := 5 * h
  (b1 + b2) / 2 * h = (9 * h ^ 2 + 2 * h) / 2 :=
by 
  let b1 := 4 * h + 2
  let b2 := 5 * h
  sorry

end trapezoid_area_l214_214586


namespace total_books_l214_214600

-- Given conditions
def susan_books : Nat := 600
def lidia_books : Nat := 4 * susan_books

-- The theorem to prove
theorem total_books : susan_books + lidia_books = 3000 :=
by
  unfold susan_books lidia_books
  sorry

end total_books_l214_214600


namespace g_at_4_l214_214382

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2

theorem g_at_4 : g 4 = -2 :=
by
  -- Proof would go here
  sorry

end g_at_4_l214_214382


namespace rice_field_sacks_l214_214008

theorem rice_field_sacks (x : ℝ)
  (h1 : ∀ x, x + 1.20 * x = 44) : x = 20 :=
sorry

end rice_field_sacks_l214_214008


namespace number_consisting_of_11_hundreds_11_tens_and_11_units_l214_214525

theorem number_consisting_of_11_hundreds_11_tens_and_11_units :
  11 * 100 + 11 * 10 + 11 = 1221 :=
by
  sorry

end number_consisting_of_11_hundreds_11_tens_and_11_units_l214_214525


namespace positive_value_of_X_l214_214735

def hash_relation (X Y : ℕ) : ℕ := X^2 + Y^2

theorem positive_value_of_X (X : ℕ) (h : hash_relation X 7 = 290) : X = 17 :=
by sorry

end positive_value_of_X_l214_214735


namespace remainder_of_product_mod_7_l214_214065

   theorem remainder_of_product_mod_7 :
     (7 * 17 * 27 * 37 * 47 * 57 * 67) % 7 = 0 := 
   by
     sorry
   
end remainder_of_product_mod_7_l214_214065


namespace proof_case_a_proof_case_b_l214_214046

noncomputable def proof_problem_a (x y z p q : ℝ) (n : ℕ) 
  (h1 : y = x^n + p*x + q) 
  (h2 : z = y^n + p*y + q) 
  (h3 : x = z^n + p*z + q) : Prop :=
  x^2 * y + y^2 * z + z^2 * x >= x^2 * z + y^2 * x + z^2 * y

theorem proof_case_a (x y z p q : ℝ) 
  (h1 : y = x^2 + p*x + q) 
  (h2 : z = y^2 + p*y + q) 
  (h3 : x = z^2 + p*z + q) : 
  proof_problem_a x y z p q 2 h1 h2 h3 := 
sorry

theorem proof_case_b (x y z p q : ℝ) 
  (h1 : y = x^2010 + p*x + q) 
  (h2 : z = y^2010 + p*y + q) 
  (h3 : x = z^2010 + p*z + q) : 
  proof_problem_a x y z p q 2010 h1 h2 h3 := 
sorry

end proof_case_a_proof_case_b_l214_214046


namespace farmer_initial_plan_days_l214_214115

def initialDaysPlan
    (daily_hectares : ℕ)
    (increased_productivity : ℕ)
    (hectares_ploughed_first_two_days : ℕ)
    (hectares_remaining : ℕ)
    (days_ahead_schedule : ℕ)
    (total_hectares : ℕ)
    (days_actual : ℕ) : ℕ :=
  days_actual + days_ahead_schedule

theorem farmer_initial_plan_days : 
  ∀ (x days_ahead_schedule : ℕ) 
    (daily_hectares hectares_ploughed_first_two_days increased_productivity hectares_remaining total_hectares days_actual : ℕ),
  daily_hectares = 120 →
  increased_productivity = daily_hectares + daily_hectares / 4 →
  hectares_ploughed_first_two_days = 2 * daily_hectares →
  total_hectares = 1440 →
  days_ahead_schedule = 2 →
  days_actual = 10 →
  hectares_remaining = total_hectares - hectares_ploughed_first_two_days →
  hectares_remaining = increased_productivity * (days_actual - 2) →
  x = 12 :=
by
  intros x days_ahead_schedule daily_hectares hectares_ploughed_first_two_days increased_productivity hectares_remaining total_hectares days_actual
  intros h_daily_hectares h_increased_productivity h_hectares_ploughed_first_two_days h_total_hectares h_days_ahead_schedule h_days_actual h_hectares_remaining h_hectares_ploughed
  sorry

end farmer_initial_plan_days_l214_214115


namespace problem1_problem2_l214_214474

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2 * a * x - 3

theorem problem1 (a : ℝ) (h : f a (a + 1) - f a a = 9) : a = 2 :=
by sorry

theorem problem2 (a : ℝ) (h : ∃ x, f a x = -4 ∧ ∀ y, f a y ≥ -4) : a = 1 ∨ a = -1 :=
by sorry

end problem1_problem2_l214_214474


namespace number_of_people_in_group_l214_214669

theorem number_of_people_in_group (n : ℕ) (h1 : 110 - 60 = 5 * n) : n = 10 :=
by 
  sorry

end number_of_people_in_group_l214_214669


namespace green_turtles_1066_l214_214654

def number_of_turtles (G H : ℕ) : Prop :=
  H = 2 * G ∧ G + H = 3200

theorem green_turtles_1066 : ∃ G : ℕ, number_of_turtles G (2 * G) ∧ G = 1066 :=
by
  sorry

end green_turtles_1066_l214_214654


namespace complement_of_angle_l214_214003

theorem complement_of_angle (A : ℝ) (hA : A = 35) : 180 - A = 145 := by
  sorry

end complement_of_angle_l214_214003


namespace evaluate_cyclotomic_sum_l214_214356

theorem evaluate_cyclotomic_sum : 
  (Complex.I ^ 1520 + Complex.I ^ 1521 + Complex.I ^ 1522 + Complex.I ^ 1523 + Complex.I ^ 1524 = 2) :=
by sorry

end evaluate_cyclotomic_sum_l214_214356


namespace selena_book_pages_l214_214620

variable (S : ℕ)
variable (H : ℕ)

theorem selena_book_pages (cond1 : H = S / 2 - 20) (cond2 : H = 180) : S = 400 :=
by
  sorry

end selena_book_pages_l214_214620


namespace volume_difference_l214_214446

-- Define the dimensions of the first bowl
def length1 : ℝ := 14
def width1 : ℝ := 16
def height1 : ℝ := 9

-- Define the dimensions of the second bowl
def length2 : ℝ := 14
def width2 : ℝ := 16
def height2 : ℝ := 4

-- Define the volumes of the two bowls assuming they are rectangular prisms
def volume1 : ℝ := length1 * width1 * height1
def volume2 : ℝ := length2 * width2 * height2

-- Statement to prove the volume difference
theorem volume_difference : volume1 - volume2 = 1120 := by
  sorry

end volume_difference_l214_214446


namespace factorize_expr1_factorize_expr2_l214_214950

variable (x y a b : ℝ)

theorem factorize_expr1 : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := sorry

theorem factorize_expr2 : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := sorry

end factorize_expr1_factorize_expr2_l214_214950


namespace distributor_B_lower_avg_price_l214_214255

theorem distributor_B_lower_avg_price (p_1 p_2 : ℝ) (h : p_1 < p_2) :
  (p_1 + p_2) / 2 > (2 * p_1 * p_2) / (p_1 + p_2) :=
by {
  sorry
}

end distributor_B_lower_avg_price_l214_214255


namespace intersection_point_a_l214_214484

theorem intersection_point_a : ∃ (x y : ℝ), y = 4 * x - 32 ∧ y = -6 * x + 8 ∧ x = 4 ∧ y = -16 :=
sorry

end intersection_point_a_l214_214484


namespace tangent_line_exists_l214_214561

noncomputable def tangent_line_problem := ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  Int.gcd (Int.gcd a b) c = 1 ∧ 
  (∀ x y : ℝ, a * x + b * (x^2 + 52 / 25) = c ∧ a * (y^2 + 81 / 16) + b * y = c) ∧ 
  a + b + c = 168

theorem tangent_line_exists : tangent_line_problem := by
  sorry

end tangent_line_exists_l214_214561


namespace find_r_l214_214425

variable (p r s : ℝ)

theorem find_r (h : ∀ x : ℝ, (y : ℝ) = x^2 + p * x + r + s → (y = 10 ↔ x = -p / 2)) : r = 10 - s + p^2 / 4 := by
  sorry

end find_r_l214_214425


namespace completing_square_correct_l214_214307

-- Define the initial equation
def eq1 : Prop := ∀ x : ℝ, x^2 - 4*x - 1 = 0

-- Define the condition after moving the constant term
def eq2 : Prop := ∀ x : ℝ, x^2 - 4*x = 1

-- Define the condition after adding 4 to both sides
def eq3 : Prop := ∀ x : ℝ, x^2 - 4*x + 4 = 5

-- Define the final transformed equation
def final_eq : Prop := ∀ x : ℝ, (x - 2)^2 = 5

-- State the theorem
theorem completing_square_correct : 
  (eq1 → eq2) ∧ 
  (eq2 → eq3) ∧ 
  (eq3 → final_eq) :=
by
  sorry

end completing_square_correct_l214_214307


namespace donna_received_total_interest_l214_214225

-- Donna's investment conditions
def totalInvestment : ℝ := 33000
def investmentAt4Percent : ℝ := 13000
def investmentAt225Percent : ℝ := totalInvestment - investmentAt4Percent
def rate4Percent : ℝ := 0.04
def rate225Percent : ℝ := 0.0225

-- The interest calculation
def interestFrom4PercentInvestment : ℝ := investmentAt4Percent * rate4Percent
def interestFrom225PercentInvestment : ℝ := investmentAt225Percent * rate225Percent
def totalInterest : ℝ := interestFrom4PercentInvestment + interestFrom225PercentInvestment

-- The proof statement
theorem donna_received_total_interest :
  totalInterest = 970 := by
sorry

end donna_received_total_interest_l214_214225


namespace quadratic_inequality_solution_l214_214490

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x ^ 2 + 4 * x + 1 > 0) ↔ (a > 4) :=
sorry

end quadratic_inequality_solution_l214_214490


namespace number_of_adults_l214_214834

theorem number_of_adults
  (A C : ℕ)
  (h1 : A + C = 610)
  (h2 : 2 * A + C = 960) :
  A = 350 :=
by
  sorry

end number_of_adults_l214_214834


namespace units_digit_fraction_l214_214127

open Nat

theorem units_digit_fraction : 
  (15 * 16 * 17 * 18 * 19 * 20) % 500 % 10 = 2 := by
  sorry

end units_digit_fraction_l214_214127


namespace intersection_point_of_line_and_y_axis_l214_214084

theorem intersection_point_of_line_and_y_axis :
  {p : ℝ × ℝ | ∃ x, p = (x, 2 * x + 1) ∧ x = 0} = {(0, 1)} :=
by sorry

end intersection_point_of_line_and_y_axis_l214_214084


namespace combined_weight_l214_214877

-- Given constants
def JakeWeight : ℕ := 198
def WeightLost : ℕ := 8
def KendraWeight := (JakeWeight - WeightLost) / 2

-- Prove the combined weight of Jake and Kendra
theorem combined_weight : JakeWeight + KendraWeight = 293 := by
  sorry

end combined_weight_l214_214877


namespace least_n_for_perfect_square_l214_214564

theorem least_n_for_perfect_square (n : ℕ) :
  (∀ m : ℕ, 2^8 + 2^11 + 2^n = m * m) → n = 12 := sorry

end least_n_for_perfect_square_l214_214564


namespace symmetric_points_x_axis_l214_214061

theorem symmetric_points_x_axis (a b : ℝ) (P Q : ℝ × ℝ)
  (hP : P = (a + 2, -2))
  (hQ : Q = (4, b))
  (hx : (a + 2) = 4)
  (hy : b = 2) :
  (a^b) = 4 := by
sorry

end symmetric_points_x_axis_l214_214061


namespace min_value_of_a_l214_214652

theorem min_value_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 / 2 → x^2 + 2 * a * x + 1 ≥ 0) → a ≥ -5 / 4 := 
sorry

end min_value_of_a_l214_214652


namespace problem_abc_l214_214803

theorem problem_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by
  sorry

end problem_abc_l214_214803


namespace initial_birds_in_tree_l214_214674

theorem initial_birds_in_tree (x : ℕ) (h : x + 81 = 312) : x = 231 := 
by
  sorry

end initial_birds_in_tree_l214_214674


namespace factor_x4_plus_16_l214_214993

theorem factor_x4_plus_16 (x : ℝ) : x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end factor_x4_plus_16_l214_214993


namespace additional_land_cost_l214_214335

noncomputable def initial_land := 300
noncomputable def final_land := 900
noncomputable def cost_per_square_meter := 20

theorem additional_land_cost : (final_land - initial_land) * cost_per_square_meter = 12000 :=
by
  -- Define the amount of additional land purchased
  let additional_land := final_land - initial_land
  -- Calculate the cost of the additional land            
  show additional_land * cost_per_square_meter = 12000
  sorry

end additional_land_cost_l214_214335


namespace minimum_value_expression_l214_214815

theorem minimum_value_expression {a : ℝ} (h₀ : 1 < a) (h₁ : a < 4) : 
  (∃ m : ℝ, (∀ x : ℝ, 1 < x ∧ x < 4 → m ≤ (x / (4 - x) + 1 / (x - 1))) ∧ m = 2) :=
sorry

end minimum_value_expression_l214_214815


namespace count_six_digit_numbers_with_at_least_one_zero_l214_214223

theorem count_six_digit_numbers_with_at_least_one_zero : 
  900000 - 531441 = 368559 :=
by
  sorry

end count_six_digit_numbers_with_at_least_one_zero_l214_214223


namespace find_constants_and_extrema_l214_214427

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem find_constants_and_extrema (a b c : ℝ) (h : a ≠ 0) 
    (ext1 : deriv (f a b c) 1 = 0) (ext2 : deriv (f a b c) (-1) = 0) (value1 : f a b c 1 = -1) :
    a = -1/2 ∧ b = 0 ∧ c = 1/2 ∧ 
    (∃ x : ℝ, x = 1 ∧ deriv (deriv (f a b c)) x < 0) ∧
    (∃ x : ℝ, x = -1 ∧ deriv (deriv (f a b c)) x > 0) :=
sorry

end find_constants_and_extrema_l214_214427


namespace new_pressure_of_transferred_gas_l214_214282

theorem new_pressure_of_transferred_gas (V1 V2 : ℝ) (p1 k : ℝ) 
  (h1 : V1 = 3.5) (h2 : p1 = 8) (h3 : k = V1 * p1) (h4 : V2 = 7) :
  ∃ p2 : ℝ, p2 = 4 ∧ k = V2 * p2 :=
by
  use 4
  sorry

end new_pressure_of_transferred_gas_l214_214282


namespace distance_greater_than_two_l214_214916

theorem distance_greater_than_two (x : ℝ) (h : |x| > 2) : x > 2 ∨ x < -2 :=
sorry

end distance_greater_than_two_l214_214916


namespace recipe_calls_for_eight_cups_of_sugar_l214_214496

def cups_of_flour : ℕ := 6
def cups_of_salt : ℕ := 7
def additional_sugar_needed (salt : ℕ) : ℕ := salt + 1

theorem recipe_calls_for_eight_cups_of_sugar :
  additional_sugar_needed cups_of_salt = 8 :=
by
  -- condition 1: cups_of_flour = 6
  -- condition 2: cups_of_salt = 7
  -- condition 4: additional_sugar_needed = salt + 1
  -- prove formula: 7 + 1 = 8
  sorry

end recipe_calls_for_eight_cups_of_sugar_l214_214496


namespace range_of_k_l214_214145

theorem range_of_k (k : ℝ) (h : -3 < k ∧ k ≤ 0) : ∀ x : ℝ, k * x^2 + 2 * k * x - 3 < 0 :=
sorry

end range_of_k_l214_214145


namespace proposition_contrapositive_same_truth_value_l214_214163

variable {P : Prop}

theorem proposition_contrapositive_same_truth_value (P : Prop) :
  (P → P) = (¬P → ¬P) := 
sorry

end proposition_contrapositive_same_truth_value_l214_214163


namespace farmer_has_42_cows_left_l214_214932

-- Define the conditions
def initial_cows := 51
def added_cows := 5
def sold_fraction := 1 / 4

-- Lean statement to prove the number of cows left
theorem farmer_has_42_cows_left :
  (initial_cows + added_cows) - (sold_fraction * (initial_cows + added_cows)) = 42 :=
by
  -- skipping the proof part
  sorry

end farmer_has_42_cows_left_l214_214932


namespace find_a100_l214_214146

-- Define the arithmetic sequence with the given conditions
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Given conditions
variables {a d : ℤ}
variables (S_9 : ℤ) (a_10 : ℤ)

-- Conditions in Lean definition
def conditions (a d : ℤ) : Prop :=
  (9 / 2 * (2 * a + 8 * d) = 27) ∧ (a + 9 * d = 8)

-- Prove the final statement
theorem find_a100 : ∃ a d : ℤ, conditions a d → arithmetic_sequence a d 100 = 98 := 
by {
    sorry
}

end find_a100_l214_214146


namespace simplify_radical_subtraction_l214_214959

theorem simplify_radical_subtraction : 
  (Real.sqrt 18 - Real.sqrt 8) = Real.sqrt 2 := 
by
  sorry

end simplify_radical_subtraction_l214_214959


namespace point_A_lies_on_plane_l214_214167

-- Define the plane equation
def plane (x y z : ℝ) : Prop := 2 * x - y + 2 * z = 7

-- Define the specific point
def point_A : Prop := plane 2 3 3

-- The theorem stating that point A lies on the plane
theorem point_A_lies_on_plane : point_A :=
by
  -- Proof skipped
  sorry

end point_A_lies_on_plane_l214_214167


namespace triangle_area_l214_214643

noncomputable def area_ABC (AB BC : ℝ) (angle_B : ℝ) : ℝ :=
  1/2 * AB * BC * Real.sin angle_B

theorem triangle_area
  (A B C : Type)
  (AB : ℝ) (A_eq : ℝ) (B_eq : ℝ)
  (h_AB : AB = 6)
  (h_A : A_eq = Real.pi / 6)
  (h_B : B_eq = 2 * Real.pi / 3) :
  area_ABC AB AB (2 * Real.pi / 3) = 9 * Real.sqrt 3 :=
by
  simp [area_ABC, h_AB, h_A, h_B]
  sorry

end triangle_area_l214_214643


namespace long_fur_brown_dogs_l214_214015

-- Defining the basic parameters given in the problem
def total_dogs : ℕ := 45
def long_fur : ℕ := 26
def brown_dogs : ℕ := 30
def neither_long_fur_nor_brown : ℕ := 8

-- Statement of the theorem
theorem long_fur_brown_dogs : ∃ LB : ℕ, LB = 27 ∧ total_dogs = long_fur + brown_dogs - LB + neither_long_fur_nor_brown :=
by {
  -- skipping the proof
  sorry
}

end long_fur_brown_dogs_l214_214015


namespace abs_b_lt_abs_a_lt_2abs_b_l214_214022

variable {a b : ℝ}

theorem abs_b_lt_abs_a_lt_2abs_b (h : (6 * a + 9 * b) / (a + b) < (4 * a - b) / (a - b)) :
  |b| < |a| ∧ |a| < 2 * |b| :=
sorry

end abs_b_lt_abs_a_lt_2abs_b_l214_214022


namespace sqrt_three_irrational_l214_214322

theorem sqrt_three_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a:ℝ) / b = Real.sqrt 3 :=
sorry

end sqrt_three_irrational_l214_214322


namespace quarterly_to_annual_interest_rate_l214_214430

theorem quarterly_to_annual_interest_rate :
  ∃ s : ℝ, (1 + 0.02)^4 = 1 + s / 100 ∧ abs (s - 8.24) < 0.01 :=
by
  sorry

end quarterly_to_annual_interest_rate_l214_214430


namespace books_sold_l214_214862

theorem books_sold (initial_books sold_books remaining_books : ℕ) 
  (h_initial : initial_books = 242) 
  (h_remaining : remaining_books = 105)
  (h_relation : sold_books = initial_books - remaining_books) :
  sold_books = 137 := 
by
  sorry

end books_sold_l214_214862


namespace relation_between_3a5_3b5_l214_214105

theorem relation_between_3a5_3b5 (a b : ℝ) (h : a > b) : 3 * a + 5 > 3 * b + 5 := by
  sorry

end relation_between_3a5_3b5_l214_214105


namespace find_second_liquid_parts_l214_214659

-- Define the given constants
def first_liquid_kerosene_percentage : ℝ := 0.25
def second_liquid_kerosene_percentage : ℝ := 0.30
def first_liquid_parts : ℝ := 6
def mixture_kerosene_percentage : ℝ := 0.27

-- Define the amount of kerosene from each liquid
def kerosene_from_first_liquid := first_liquid_kerosene_percentage * first_liquid_parts
def kerosene_from_second_liquid (x : ℝ) := second_liquid_kerosene_percentage * x

-- Define the total parts of mixture
def total_mixture_parts (x : ℝ) := first_liquid_parts + x

-- Define the total kerosene in the mixture
def total_kerosene_in_mixture (x : ℝ) := mixture_kerosene_percentage * total_mixture_parts x

-- State the theorem
theorem find_second_liquid_parts (x : ℝ) :
  kerosene_from_first_liquid + kerosene_from_second_liquid x = total_kerosene_in_mixture x → 
  x = 4 :=
by
  sorry

end find_second_liquid_parts_l214_214659


namespace dinner_plates_percentage_l214_214694

/-- Define the cost of silverware and the total cost of both items -/
def silverware_cost : ℝ := 20
def total_cost : ℝ := 30

/-- Define the percentage of the silverware cost that the dinner plates cost -/
def percentage_of_silverware_cost := 50

theorem dinner_plates_percentage :
  ∃ (P : ℝ) (S : ℝ) (x : ℝ), S = silverware_cost ∧ (P + S = total_cost) ∧ (P = (x / 100) * S) ∧ x = percentage_of_silverware_cost :=
by {
  sorry
}

end dinner_plates_percentage_l214_214694


namespace initial_average_age_is_16_l214_214176

-- Given conditions
variable (N : ℕ) (newPersons : ℕ) (avgNewPersonsAge : ℝ) (totalPersonsAfter : ℕ) (avgAgeAfter : ℝ)
variable (initial_avg_age : ℝ) -- This represents the initial average age (A) we need to prove

-- The specific values from the problem
def N_value : ℕ := 20
def newPersons_value : ℕ := 20
def avgNewPersonsAge_value : ℝ := 15
def totalPersonsAfter_value : ℕ := 40
def avgAgeAfter_value : ℝ := 15.5

-- Theorem statement to prove that the initial average age is 16 years
theorem initial_average_age_is_16 (h1 : N = N_value) (h2 : newPersons = newPersons_value) 
  (h3 : avgNewPersonsAge = avgNewPersonsAge_value) (h4 : totalPersonsAfter = totalPersonsAfter_value) 
  (h5 : avgAgeAfter = avgAgeAfter_value) : initial_avg_age = 16 := by
  sorry

end initial_average_age_is_16_l214_214176


namespace average_headcount_spring_terms_l214_214498

def spring_headcount_02_03 := 10900
def spring_headcount_03_04 := 10500
def spring_headcount_04_05 := 10700

theorem average_headcount_spring_terms :
  (spring_headcount_02_03 + spring_headcount_03_04 + spring_headcount_04_05) / 3 = 10700 := by
  sorry

end average_headcount_spring_terms_l214_214498


namespace moles_of_CaCl2_l214_214492

/-- 
We are given the reaction: CaCO3 + 2 HCl → CaCl2 + CO2 + H2O 
with 2 moles of HCl and 1 mole of CaCO3. We need to prove that the number 
of moles of CaCl2 formed is 1.
-/
theorem moles_of_CaCl2 (HCl: ℝ) (CaCO3: ℝ) (reaction: CaCO3 + 2 * HCl = 1): CaCO3 = 1 → HCl = 2 → CaCl2 = 1 :=
by
  -- importing the required context for chemical equations and stoichiometry
  sorry

end moles_of_CaCl2_l214_214492


namespace determine_m_l214_214216

theorem determine_m {m : ℕ} : 
  (∃ (p : ℕ), p = 5 ∧ p = max (max (max 1 (1 + (m+1))) (3+1)) 4) → m = 3 := by
  sorry

end determine_m_l214_214216


namespace largest_angle_of_consecutive_interior_angles_pentagon_l214_214925

theorem largest_angle_of_consecutive_interior_angles_pentagon (x : ℕ)
  (h1 : (x - 3) + (x - 2) + (x - 1) + x + (x + 1) = 540) :
  x + 1 = 110 := sorry

end largest_angle_of_consecutive_interior_angles_pentagon_l214_214925


namespace number_of_pages_to_copy_l214_214590

-- Definitions based on the given conditions
def total_budget : ℕ := 5000
def service_charge : ℕ := 500
def copy_cost : ℕ := 3

-- Derived definition based on the conditions
def remaining_budget : ℕ := total_budget - service_charge

-- The statement we need to prove
theorem number_of_pages_to_copy : (remaining_budget / copy_cost) = 1500 :=
by {
  sorry
}

end number_of_pages_to_copy_l214_214590


namespace center_circle_sum_eq_neg1_l214_214596

theorem center_circle_sum_eq_neg1 
  (h k : ℝ) 
  (h_center : ∀ x y, (x - h)^2 + (y - k)^2 = 22) 
  (circle_eq : ∀ x y, x^2 + y^2 = 4*x - 6*y + 9) : 
  h + k = -1 := 
by 
  sorry

end center_circle_sum_eq_neg1_l214_214596


namespace sqrt_equality_l214_214651

theorem sqrt_equality (m : ℝ) (n : ℝ) (h1 : 0 < m) (h2 : -3 * m ≤ n) (h3 : n ≤ 3 * m) :
    (Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2))
     - Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2))
    = 2 * Real.sqrt (3 * m - n)) :=
sorry

end sqrt_equality_l214_214651


namespace sum_of_digits_triangular_array_l214_214173

theorem sum_of_digits_triangular_array (N : ℕ) (h : N * (N + 1) / 2 = 5050) : 
  Nat.digits 10 N = [1, 0, 0] := by
  sorry

end sum_of_digits_triangular_array_l214_214173


namespace minimum_keys_needed_l214_214921

theorem minimum_keys_needed (total_cabinets : ℕ) (boxes_per_cabinet : ℕ)
(boxes_needed : ℕ) (boxes_per_cabinet : ℕ) 
(warehouse_key : ℕ) (boxes_per_cabinet: ℕ)
(h1 : total_cabinets = 8)
(h2 : boxes_per_cabinet = 4)
(h3 : (boxes_needed = 52))
(h4 : boxes_per_cabinet = 4)
(h5 : warehouse_key = 1):
    6 + 2 + 1 = 9 := 
    sorry

end minimum_keys_needed_l214_214921


namespace revision_cost_is_3_l214_214682

def cost_first_time (pages : ℕ) : ℝ := 5 * pages

def cost_for_revisions (rev1 rev2 : ℕ) (rev_cost : ℝ) : ℝ := (rev1 * rev_cost) + (rev2 * 2 * rev_cost)

def total_cost (pages rev1 rev2 : ℕ) (rev_cost : ℝ) : ℝ := 
  cost_first_time pages + cost_for_revisions rev1 rev2 rev_cost

theorem revision_cost_is_3 :
  ∀ (pages rev1 rev2 : ℕ) (total : ℝ),
      pages = 100 →
      rev1 = 30 →
      rev2 = 20 →
      total = 710 →
      total_cost pages rev1 rev2 3 = total :=
by
  intros pages rev1 rev2 total pages_eq rev1_eq rev2_eq total_eq
  sorry

end revision_cost_is_3_l214_214682


namespace simplify_expr_l214_214126

theorem simplify_expr (a : ℝ) (h_a : a = (8:ℝ)^(1/2) * (1/2) - (3:ℝ)^(1/2)^(0) ) : 
  a = (2:ℝ)^(1/2) - 1 := 
by
  sorry

end simplify_expr_l214_214126


namespace gcd_90_450_l214_214695

theorem gcd_90_450 : Int.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l214_214695


namespace denver_wood_used_per_birdhouse_l214_214690

-- Definitions used in the problem
def cost_per_piee_of_wood : ℝ := 1.50
def profit_per_birdhouse : ℝ := 5.50
def price_for_two_birdhouses : ℝ := 32
def num_birdhouses_purchased : ℝ := 2

-- Property to prove
theorem denver_wood_used_per_birdhouse (W : ℝ) 
  (h : num_birdhouses_purchased * (cost_per_piee_of_wood * W + profit_per_birdhouse) = price_for_two_birdhouses) : 
  W = 7 :=
sorry

end denver_wood_used_per_birdhouse_l214_214690


namespace intersection_set_l214_214747

def M : Set ℤ := {1, 2, 3, 5, 7}
def N : Set ℤ := {x | ∃ k ∈ M, x = 2 * k - 1}
def I : Set ℤ := {1, 3, 5}

theorem intersection_set :
  M ∩ N = I :=
by sorry

end intersection_set_l214_214747


namespace minimum_value_is_14_div_27_l214_214018

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sin x)^8 + (Real.cos x)^8 + 1 / (Real.sin x)^6 + (Real.cos x)^6 + 1

theorem minimum_value_is_14_div_27 :
  ∃ x : ℝ, minimum_value_expression x = (14 / 27) :=
by
  sorry

end minimum_value_is_14_div_27_l214_214018


namespace factorization_of_x_squared_minus_one_l214_214443

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l214_214443


namespace a_b_finish_job_in_15_days_l214_214825

theorem a_b_finish_job_in_15_days (A B C : ℝ) 
  (h1 : A + B + C = 1 / 5)
  (h2 : C = 1 / 7.5) : 
  (1 / (A + B)) = 15 :=
by
  sorry

end a_b_finish_job_in_15_days_l214_214825


namespace joseph_total_distance_l214_214486

-- Distance Joseph runs on Monday
def d1 : ℕ := 900

-- Increment each day
def increment : ℕ := 200

-- Adjust distance calculation
def d2 := d1 + increment
def d3 := d2 + increment

-- Total distance calculation
def total_distance := d1 + d2 + d3

-- Prove that the total distance is 3300 meters
theorem joseph_total_distance : total_distance = 3300 :=
by sorry

end joseph_total_distance_l214_214486


namespace product_of_solutions_eq_zero_l214_214680

theorem product_of_solutions_eq_zero :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) → (x = 0 ∨ x = -4 / 7)) → (0 = 0) := 
by
  intros h
  sorry

end product_of_solutions_eq_zero_l214_214680


namespace find_n_from_sum_of_coeffs_l214_214215

-- The mathematical conditions and question translated to Lean

def sum_of_coefficients (n : ℕ) : ℕ := 6 ^ n
def binomial_coefficients_sum (n : ℕ) : ℕ := 2 ^ n

theorem find_n_from_sum_of_coeffs (n : ℕ) (M N : ℕ) (hM : M = sum_of_coefficients n) (hN : N = binomial_coefficients_sum n) (condition : M - N = 240) : n = 4 :=
by
  sorry

end find_n_from_sum_of_coeffs_l214_214215


namespace edward_initial_amount_l214_214613

-- Defining the conditions
def cost_books : ℕ := 6
def cost_pens : ℕ := 16
def cost_notebook : ℕ := 5
def cost_pencil_case : ℕ := 3
def amount_left : ℕ := 19

-- Mathematical statement to prove
theorem edward_initial_amount : 
  cost_books + cost_pens + cost_notebook + cost_pencil_case + amount_left = 49 :=
by
  sorry

end edward_initial_amount_l214_214613


namespace finiteness_of_triples_l214_214676

theorem finiteness_of_triples (x : ℚ) : ∃! (a b c : ℤ), a < 0 ∧ b^2 - 4*a*c = 5 ∧ (a*x^2 + b*x + c > 0) := sorry

end finiteness_of_triples_l214_214676


namespace calculation_correct_l214_214404

theorem calculation_correct : 67897 * 67898 - 67896 * 67899 = 2 := by
  sorry

end calculation_correct_l214_214404


namespace solve_for_x_l214_214824

theorem solve_for_x : (∃ x : ℝ, ((10 - 2 * x) ^ 2 = 4 * x ^ 2 + 16) ∧ x = 2.1) :=
by
  sorry

end solve_for_x_l214_214824


namespace mean_score_all_students_l214_214791

theorem mean_score_all_students
  (M A E : ℝ) (m a e : ℝ)
  (hM : M = 78)
  (hA : A = 68)
  (hE : E = 82)
  (h_ratio_ma : m / a = 4 / 5)
  (h_ratio_mae : (m + a) / e = 9 / 2)
  : (M * m + A * a + E * e) / (m + a + e) = 74.4 := by
  sorry

end mean_score_all_students_l214_214791


namespace check_interval_of_quadratic_l214_214310

theorem check_interval_of_quadratic (z : ℝ) : (z^2 - 40 * z + 344 ≤ 0) ↔ (20 - 2 * Real.sqrt 14 ≤ z ∧ z ≤ 20 + 2 * Real.sqrt 14) :=
sorry

end check_interval_of_quadratic_l214_214310


namespace problem_statement_l214_214190

theorem problem_statement :
  let a := -12
  let b := 45
  let c := -45
  let d := 54
  8 * a + 4 * b + 2 * c + d = 48 :=
by
  sorry

end problem_statement_l214_214190


namespace gardner_bakes_brownies_l214_214247

theorem gardner_bakes_brownies : 
  ∀ (cookies cupcakes brownies students sweet_treats_per_student total_sweet_treats total_cookies_and_cupcakes : ℕ),
  cookies = 20 →
  cupcakes = 25 →
  students = 20 →
  sweet_treats_per_student = 4 →
  total_sweet_treats = students * sweet_treats_per_student →
  total_cookies_and_cupcakes = cookies + cupcakes →
  brownies = total_sweet_treats - total_cookies_and_cupcakes →
  brownies = 35 :=
by
  intros cookies cupcakes brownies students sweet_treats_per_student total_sweet_treats total_cookies_and_cupcakes
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end gardner_bakes_brownies_l214_214247


namespace simplify_expression_l214_214814

theorem simplify_expression (x : ℝ) : (3 * x + 20) + (50 * x + 25) = 53 * x + 45 :=
by
  sorry

end simplify_expression_l214_214814


namespace line_intersects_circle_l214_214688

theorem line_intersects_circle (a : ℝ) (h : a ≠ 0) :
  ∃ p : ℝ × ℝ, (p.1 ^ 2 + p.2 ^ 2 = 9) ∧ (a * p.1 - p.2 + 2 * a = 0) :=
by
  sorry

end line_intersects_circle_l214_214688


namespace pie_left_is_30_percent_l214_214151

def Carlos_share : ℝ := 0.60
def remaining_after_Carlos : ℝ := 1 - Carlos_share
def Jessica_share : ℝ := 0.25 * remaining_after_Carlos
def final_remaining : ℝ := remaining_after_Carlos - Jessica_share

theorem pie_left_is_30_percent :
  final_remaining = 0.30 :=
sorry

end pie_left_is_30_percent_l214_214151


namespace sin_750_eq_one_half_l214_214893

theorem sin_750_eq_one_half :
  ∀ (θ: ℝ), (∀ n: ℤ, Real.sin (θ + n * 360) = Real.sin θ) → Real.sin 30 = 1 / 2 → Real.sin 750 = 1 / 2 :=
by 
  intros θ periodic_sine sin_30
  -- insert proof here
  sorry

end sin_750_eq_one_half_l214_214893


namespace quadratic_has_two_distinct_real_roots_determine_k_from_roots_relation_l214_214871

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4*a*c

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  let a := 1
  let b := 2*k - 1
  let c := -k - 1
  discriminant a b c > 0 := by
  sorry

theorem determine_k_from_roots_relation (x1 x2 k : ℝ) 
  (h1 : x1 + x2 = -(2*k - 1))
  (h2 : x1 * x2 = -k - 1)
  (h3 : x1 + x2 - 4*(x1 * x2) = 2) :
  k = -3/2 := by
  sorry

end quadratic_has_two_distinct_real_roots_determine_k_from_roots_relation_l214_214871


namespace fraction_equality_l214_214524

theorem fraction_equality (a b c : ℚ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ a / 2 ≠ 0) :
  (a - 2 * c) / (a - 2 * b) = 3 / 2 := 
by
  -- Use the hypthesis to derive that a = 2k, b = 3k, c = 4k and show the equality.
  sorry

end fraction_equality_l214_214524


namespace parallel_vectors_determine_t_l214_214374

theorem parallel_vectors_determine_t (t : ℝ) (h : (t, -6) = (k * -3, k * 2)) : t = 9 :=
by
  sorry

end parallel_vectors_determine_t_l214_214374


namespace james_distance_l214_214663

-- Definitions and conditions
def speed : ℝ := 80.0
def time : ℝ := 16.0

-- Proof problem statement
theorem james_distance : speed * time = 1280.0 := by
  sorry

end james_distance_l214_214663


namespace sculpture_and_base_height_l214_214468

def height_in_inches (feet: ℕ) (inches: ℕ) : ℕ :=
  feet * 12 + inches

theorem sculpture_and_base_height
  (sculpture_feet: ℕ) (sculpture_inches: ℕ) (base_inches: ℕ)
  (hf: sculpture_feet = 2)
  (hi: sculpture_inches = 10)
  (hb: base_inches = 8)
  : height_in_inches sculpture_feet sculpture_inches + base_inches = 42 :=
by
  -- Placeholder for the proof
  sorry

end sculpture_and_base_height_l214_214468


namespace find_a_plus_k_l214_214129

-- Define the conditions.
def foci1 : (ℝ × ℝ) := (2, 0)
def foci2 : (ℝ × ℝ) := (2, 4)
def ellipse_point : (ℝ × ℝ) := (7, 2)

-- Statement of the equivalent proof problem.
theorem find_a_plus_k (a b h k : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (∀ x y, ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ↔ (x, y) = ellipse_point) →
  h = 2 → k = 2 → a = 5 →
  a + k = 7 :=
by
  sorry

end find_a_plus_k_l214_214129


namespace factorize_a_cubed_minus_a_l214_214938

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l214_214938


namespace opposite_signs_add_same_signs_sub_l214_214952

-- Definitions based on the conditions
variables {a b : ℤ}

-- 1. Case when a and b have opposite signs
theorem opposite_signs_add (h₁ : |a| = 4) (h₂ : |b| = 3) (h₃ : a * b < 0) :
  a + b = 1 ∨ a + b = -1 := 
sorry

-- 2. Case when a and b have the same sign
theorem same_signs_sub (h₁ : |a| = 4) (h₂ : |b| = 3) (h₃ : a * b > 0) :
  a - b = 1 ∨ a - b = -1 := 
sorry

end opposite_signs_add_same_signs_sub_l214_214952


namespace simplified_expression_evaluation_l214_214701

theorem simplified_expression_evaluation (x : ℝ) (hx : x = Real.sqrt 7) :
    (2 * x + 3) * (2 * x - 3) - (x + 2)^2 + 4 * (x + 3) = 20 :=
by
  sorry

end simplified_expression_evaluation_l214_214701


namespace complex_problem_l214_214236

open Complex

noncomputable def z : ℂ := (1 + I) / Real.sqrt 2

theorem complex_problem :
  1 + z^50 + z^100 = I := 
by
  -- Subproofs or transformations will be here.
  sorry

end complex_problem_l214_214236


namespace jackie_walks_daily_l214_214265

theorem jackie_walks_daily (x : ℝ) :
  (∀ t : ℕ, t = 6 →
    6 * x = 6 * 1.5 + 3) →
  x = 2 :=
by
  sorry

end jackie_walks_daily_l214_214265


namespace pyramid_circumscribed_sphere_volume_l214_214906

theorem pyramid_circumscribed_sphere_volume 
  (PA ABCD : ℝ) 
  (square_base : Prop)
  (perpendicular_PA_base : Prop)
  (AB : ℝ)
  (PA_val : PA = 1)
  (AB_val : AB = 2) 
  : (∃ (volume : ℝ), volume = (4/3) * π * (3/2)^3 ∧ volume = 9 * π / 2) := 
by
  -- Provided the conditions, we need to prove that the volume of the circumscribed sphere is 9π/2
  sorry

end pyramid_circumscribed_sphere_volume_l214_214906


namespace line_through_origin_in_quadrants_l214_214762

theorem line_through_origin_in_quadrants (A B C : ℝ) :
  (-A * x - B * y + C = 0) ∧ (0 = 0) ∧ (exists x y, 0 < x * y) →
  (C = 0) ∧ (A * B < 0) :=
sorry

end line_through_origin_in_quadrants_l214_214762


namespace sqrt_six_estimation_l214_214062

theorem sqrt_six_estimation : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 :=
by 
  sorry

end sqrt_six_estimation_l214_214062


namespace lines_intersect_at_l214_214280

noncomputable def L₁ (t : ℝ) : ℝ × ℝ := (2 - t, -3 + 4 * t)
noncomputable def L₂ (u : ℝ) : ℝ × ℝ := (-1 + 5 * u, 6 - 7 * u)
noncomputable def point_of_intersection : ℝ × ℝ := (2 / 13, 69 / 13)

theorem lines_intersect_at :
  ∃ t u : ℝ, L₁ t = point_of_intersection ∧ L₂ u = point_of_intersection := 
sorry

end lines_intersect_at_l214_214280


namespace determine_number_of_quarters_l214_214230

def number_of_coins (Q D : ℕ) : Prop := Q + D = 23

def total_value (Q D : ℕ) : Prop := 25 * Q + 10 * D = 335

theorem determine_number_of_quarters (Q D : ℕ) 
  (h1 : number_of_coins Q D) 
  (h2 : total_value Q D) : 
  Q = 7 :=
by
  -- Equating and simplifying using h2, we find 15Q = 105, hence Q = 7
  sorry

end determine_number_of_quarters_l214_214230


namespace find_ratio_l214_214601

variable {x y z : ℝ}

theorem find_ratio
  (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (2 * x + y - z) / (3 * x - 2 * y + z) = 5 / 6 := by
  sorry

end find_ratio_l214_214601


namespace cube_side_length_l214_214118

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 := by
  sorry

end cube_side_length_l214_214118


namespace no_such_b_c_exist_l214_214940

theorem no_such_b_c_exist :
  ¬ ∃ (b c : ℝ), (∃ (k l : ℤ), (k ≠ l ∧ (k ^ 2 + b * ↑k + c = 0) ∧ (l ^ 2 + b * ↑l + c = 0))) ∧
                  (∃ (m n : ℤ), (m ≠ n ∧ (2 * (m ^ 2) + (b + 1) * ↑m + (c + 1) = 0) ∧ 
                                        (2 * (n ^ 2) + (b + 1) * ↑n + (c + 1) = 0))) :=
sorry

end no_such_b_c_exist_l214_214940


namespace area_of_square_containing_circle_l214_214816

theorem area_of_square_containing_circle (r : ℝ) (hr : r = 4) :
  ∃ (a : ℝ), a = 64 ∧ (∀ (s : ℝ), s = 2 * r → a = s * s) :=
by
  use 64
  sorry

end area_of_square_containing_circle_l214_214816


namespace find_p_l214_214461

noncomputable def p (x1 x2 x3 x4 n : ℝ) :=
  (x1 + x3) * (x2 + x3) + (x1 + x4) * (x2 + x4)

theorem find_p (x1 x2 x3 x4 n : ℝ) (h1 : x1 ≠ x2)
(h2 : (x1 + x3) * (x1 + x4) = n - 10)
(h3 : (x2 + x3) * (x2 + x4) = n - 10) :
  p x1 x2 x3 x4 n = n - 20 :=
sorry

end find_p_l214_214461


namespace domain_of_sqrt_log_l214_214415

theorem domain_of_sqrt_log {x : ℝ} : (2 < x ∧ x ≤ 5 / 2) ↔ 
  (5 - 2 * x > 0 ∧ 0 ≤ Real.logb (1 / 2) (5 - 2 * x)) :=
sorry

end domain_of_sqrt_log_l214_214415


namespace correct_operation_is_B_l214_214698

-- Definitions of the operations as conditions
def operation_A (x : ℝ) : Prop := 3 * x - x = 3
def operation_B (x : ℝ) : Prop := x^2 * x^3 = x^5
def operation_C (x : ℝ) : Prop := x^6 / x^2 = x^3
def operation_D (x : ℝ) : Prop := (x^2)^3 = x^5

-- Prove that the correct operation is B
theorem correct_operation_is_B (x : ℝ) : operation_B x :=
by
  show x^2 * x^3 = x^5
  sorry

end correct_operation_is_B_l214_214698


namespace domain_of_log_function_l214_214513

-- Define the problematic quadratic function
def quadratic_fn (x : ℝ) : ℝ := -x^2 + 4 * x - 3

-- Define the domain condition for our function
def domain_condition (x : ℝ) : Prop := quadratic_fn x > 0

-- The actual statement to prove, stating that the domain is (1, 3)
theorem domain_of_log_function :
  {x : ℝ | domain_condition x} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end domain_of_log_function_l214_214513


namespace card_area_after_reduction_width_l214_214076

def initial_length : ℕ := 5
def initial_width : ℕ := 8
def new_width := initial_width - 2
def expected_new_area : ℕ := 24

theorem card_area_after_reduction_width :
  initial_length * new_width = expected_new_area := 
by
  -- initial_length = 5, new_width = 8 - 2 = 6
  -- 5 * 6 = 30, which was corrected to 24 given the misinterpretation mentioned.
  sorry

end card_area_after_reduction_width_l214_214076


namespace multiplicative_inverse_137_391_l214_214732

theorem multiplicative_inverse_137_391 :
  ∃ (b : ℕ), (b ≤ 390) ∧ (137 * b) % 391 = 1 :=
sorry

end multiplicative_inverse_137_391_l214_214732


namespace cost_of_adult_ticket_l214_214192

-- Conditions provided in the original problem.
def total_people : ℕ := 23
def child_tickets_cost : ℕ := 10
def total_money_collected : ℕ := 246
def children_attended : ℕ := 7

-- Define some unknown amount A for the adult tickets cost to be solved.
variable (A : ℕ)

-- Define the Lean statement for the proof problem.
theorem cost_of_adult_ticket :
  16 * A = 176 →
  A = 11 :=
by
  -- Start the proof (this part will be filled out during the proof process).
  sorry

#check cost_of_adult_ticket  -- To ensure it type-checks

end cost_of_adult_ticket_l214_214192


namespace min_of_x_squared_y_squared_z_squared_l214_214239

theorem min_of_x_squared_y_squared_z_squared (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  x^2 + y^2 + z^2 ≥ 4 :=
by sorry

end min_of_x_squared_y_squared_z_squared_l214_214239


namespace find_HCF_of_two_numbers_l214_214074

theorem find_HCF_of_two_numbers (a b H : ℕ) 
  (H_HCF : Nat.gcd a b = H) 
  (H_LCM_Factors : Nat.lcm a b = H * 13 * 14) 
  (H_largest_number : 322 = max a b) : 
  H = 14 :=
sorry

end find_HCF_of_two_numbers_l214_214074


namespace multiple_of_24_l214_214942

theorem multiple_of_24 (n : ℕ) (h : n > 0) : 
  ∃ k₁ k₂ : ℕ, (6 * n - 1)^2 - 1 = 24 * k₁ ∧ (6 * n + 1)^2 - 1 = 24 * k₂ :=
by
  sorry

end multiple_of_24_l214_214942


namespace ab_relationship_l214_214615

theorem ab_relationship (a b : ℝ) (n : ℕ) (h1 : a^n = a + 1) (h2 : b^(2*n) = b + 3*a) (h3 : n ≥ 2) (h4 : 0 < a) (h5 : 0 < b) :
  a > b ∧ a > 1 ∧ b > 1 :=
sorry

end ab_relationship_l214_214615


namespace perimeter_of_billboard_l214_214722
noncomputable def perimeter_billboard : ℝ :=
  let width := 8
  let area := 104
  let length := area / width
  let perimeter := 2 * (length + width)
  perimeter

theorem perimeter_of_billboard (width area : ℝ) (P : width = 8 ∧ area = 104) :
    perimeter_billboard = 42 :=
by
  sorry

end perimeter_of_billboard_l214_214722


namespace inv_proportion_through_point_l214_214028

theorem inv_proportion_through_point (m : ℝ) (x y : ℝ) (h1 : y = m / x) (h2 : x = 2) (h3 : y = -3) : m = -6 := by
  sorry

end inv_proportion_through_point_l214_214028


namespace circle_center_l214_214936

theorem circle_center (n : ℝ) (r : ℝ) (h1 : r = 7) (h2 : ∀ x : ℝ, x^2 + (x^2 - n)^2 = 49 → x^4 - x^2 * (2*n - 1) + n^2 - 49 = 0)
  (h3 : ∃! y : ℝ, y^2 + (1 - 2*n) * y + n^2 - 49 = 0) :
  (0, n) = (0, 197 / 4) := 
sorry

end circle_center_l214_214936


namespace value_of_m_div_x_l214_214134

noncomputable def ratio_of_a_to_b (a b : ℝ) : Prop := a / b = 4 / 5
noncomputable def x_value (a : ℝ) : ℝ := a * 1.75
noncomputable def m_value (b : ℝ) : ℝ := b * 0.20

theorem value_of_m_div_x (a b : ℝ) (h1 : ratio_of_a_to_b a b) (h2 : 0 < a) (h3 : 0 < b) :
  (m_value b) / (x_value a) = 1 / 7 :=
by
  sorry

end value_of_m_div_x_l214_214134


namespace union_sets_l214_214033

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}

theorem union_sets : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_sets_l214_214033


namespace determine_defective_coin_l214_214930

-- Define the properties of the coins
structure Coin :=
(denomination : ℕ)
(weight : ℕ)

-- Given coins
def c1 : Coin := ⟨1, 1⟩
def c2 : Coin := ⟨2, 2⟩
def c3 : Coin := ⟨3, 3⟩
def c5 : Coin := ⟨5, 5⟩

-- Assume one coin is defective
variable (defective : Coin)
variable (differing_weight : ℕ)
#check differing_weight

theorem determine_defective_coin :
  (∃ (defective : Coin), ∀ (c : Coin), 
    c ≠ defective → c.weight = c.denomination) → 
  ((c2.weight + c3.weight = c5.weight → defective = c1) ∧
   (c1.weight + c2.weight = c3.weight → defective = c5) ∧
   (c2.weight ≠ 2 → defective = c2) ∧
   (c3.weight ≠ 3 → defective = c3)) :=
by
  sorry

end determine_defective_coin_l214_214930


namespace product_of_two_numbers_l214_214057

theorem product_of_two_numbers (x y : ℝ) 
  (h₁ : x + y = 50) 
  (h₂ : x - y = 6) : 
  x * y = 616 := 
by
  sorry

end product_of_two_numbers_l214_214057


namespace consecutive_squares_not_arithmetic_sequence_l214_214373

theorem consecutive_squares_not_arithmetic_sequence (x y z w : ℕ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w)
  (h_order: x < y ∧ y < z ∧ z < w) :
  ¬ (∃ d : ℕ, y^2 = x^2 + d ∧ z^2 = y^2 + d ∧ w^2 = z^2 + d) :=
sorry

end consecutive_squares_not_arithmetic_sequence_l214_214373


namespace smaller_number_l214_214949

theorem smaller_number (x y : ℝ) (h1 : y - x = (1 / 3) * y) (h2 : y = 71.99999999999999) : x = 48 :=
by
  sorry

end smaller_number_l214_214949


namespace identical_digits_has_37_factor_l214_214957

theorem identical_digits_has_37_factor (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) : 37 ∣ (100 * a + 10 * a + a) :=
by
  sorry

end identical_digits_has_37_factor_l214_214957


namespace geometric_sequence_common_ratio_l214_214186

theorem geometric_sequence_common_ratio (a_1 a_4 q : ℕ) (h1 : a_1 = 8) (h2 : a_4 = 64) (h3 : a_4 = a_1 * q^3) : q = 2 :=
by {
  -- Given: a_1 = 8
  --        a_4 = 64
  --        a_4 = a_1 * q^3
  -- Prove: q = 2
  sorry
}

end geometric_sequence_common_ratio_l214_214186


namespace combined_weight_of_jake_and_sister_l214_214974

theorem combined_weight_of_jake_and_sister
  (J : ℕ) (S : ℕ)
  (h₁ : J = 113)
  (h₂ : J - 33 = 2 * S)
  : J + S = 153 :=
sorry

end combined_weight_of_jake_and_sister_l214_214974


namespace domain_f_2x_l214_214826

-- Given conditions as definitions
def domain_f_x_minus_1 (x : ℝ) := 3 < x ∧ x ≤ 7

-- The main theorem statement that needs a proof
theorem domain_f_2x : (∀ x : ℝ, domain_f_x_minus_1 (x-1) → (1 < x ∧ x ≤ 3)) :=
by
  -- Proof steps will be here, however, as requested, they are omitted.
  sorry

end domain_f_2x_l214_214826


namespace algebraic_expression_value_l214_214122

theorem algebraic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 7 = -6 := by
  sorry

end algebraic_expression_value_l214_214122


namespace oranges_now_is_50_l214_214806

def initial_fruits : ℕ := 150
def remaining_fruits : ℕ := initial_fruits / 2
def num_limes (L : ℕ) (O : ℕ) : Prop := O = 2 * L
def total_remaining_fruits (L : ℕ) (O : ℕ) : Prop := O + L = remaining_fruits

theorem oranges_now_is_50 : ∃ O L : ℕ, num_limes L O ∧ total_remaining_fruits L O ∧ O = 50 := by
  sorry

end oranges_now_is_50_l214_214806


namespace prize_distribution_correct_l214_214463

def probability_A_correct : ℚ := 3 / 4
def probability_B_correct : ℚ := 4 / 5
def total_prize : ℚ := 190

-- Calculation of expected prizes
def probability_A_only_correct : ℚ := probability_A_correct * (1 - probability_B_correct)
def probability_B_only_correct : ℚ := probability_B_correct * (1 - probability_A_correct)
def probability_both_correct : ℚ := probability_A_correct * probability_B_correct

def normalized_probability : ℚ := probability_A_only_correct + probability_B_only_correct + probability_both_correct

def expected_prize_A : ℚ := (probability_A_only_correct / normalized_probability * total_prize) + (probability_both_correct / normalized_probability * (total_prize / 2))
def expected_prize_B : ℚ := (probability_B_only_correct / normalized_probability * total_prize) + (probability_both_correct / normalized_probability * (total_prize / 2))

theorem prize_distribution_correct :
  expected_prize_A = 90 ∧ expected_prize_B = 100 := 
by
  sorry

end prize_distribution_correct_l214_214463


namespace avg_age_all_l214_214035

-- Define the conditions
def avg_age_seventh_graders (n₁ : Nat) (a₁ : Nat) : Prop :=
  n₁ = 40 ∧ a₁ = 13

def avg_age_parents (n₂ : Nat) (a₂ : Nat) : Prop :=
  n₂ = 50 ∧ a₂ = 40

-- Define the problem to prove
def avg_age_combined (n₁ n₂ a₁ a₂ : Nat) : Prop :=
  (n₁ * a₁ + n₂ * a₂) / (n₁ + n₂) = 28

-- The main theorem
theorem avg_age_all (n₁ n₂ a₁ a₂ : Nat):
  avg_age_seventh_graders n₁ a₁ → avg_age_parents n₂ a₂ → avg_age_combined n₁ n₂ a₁ a₂ :=
by 
  intros h1 h2
  sorry

end avg_age_all_l214_214035


namespace geom_seq_log_eqn_l214_214140

theorem geom_seq_log_eqn {a : ℕ → ℝ} {b : ℕ → ℝ}
    (geom_seq : ∃ (r : ℝ) (a1 : ℝ), ∀ n : ℕ, a (n + 1) = a1 * r^n)
    (log_seq : ∀ n : ℕ, b n = Real.log (a (n + 1)) / Real.log 2)
    (b_eqn : b 1 + b 3 = 4) : a 2 = 4 :=
by
  sorry

end geom_seq_log_eqn_l214_214140


namespace find_x_if_perpendicular_l214_214301

-- Define vectors a and b in the given conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x - 5, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- The Lean theorem statement equivalent to the math problem
theorem find_x_if_perpendicular (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = 0) : x = 2 := by
  sorry

end find_x_if_perpendicular_l214_214301


namespace problem_1_problem_2_l214_214479

-- Define the factorial and permutation functions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def permutation (n k : ℕ) : ℕ :=
  factorial n / factorial (n - k)

-- Problem 1 statement
theorem problem_1 : permutation 6 6 - permutation 5 5 = 600 := by
  sorry

-- Problem 2 statement
theorem problem_2 : 
  15 * permutation 5 5 * (10^5) + 15 * permutation 4 4 * 11111 =
  15 * permutation 5 5 * (10^5) + 15 * permutation 4 4 * 11111 := by
  sorry

end problem_1_problem_2_l214_214479


namespace find_P_Q_l214_214599

noncomputable def P := 11 / 3
noncomputable def Q := -2 / 3

theorem find_P_Q :
  ∀ x : ℝ, x ≠ 7 → x ≠ -2 →
    (3 * x + 12) / (x ^ 2 - 5 * x - 14) = P / (x - 7) + Q / (x + 2) :=
by
  intros x hx1 hx2
  dsimp [P, Q]  -- Unfold the definitions of P and Q
  -- The actual proof would go here, but we are skipping it
  sorry

end find_P_Q_l214_214599


namespace value_of_abs_div_sum_l214_214527

theorem value_of_abs_div_sum (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (|a| / a + |b| / b = 2) ∨ (|a| / a + |b| / b = -2) ∨ (|a| / a + |b| / b = 0) := 
by
  sorry

end value_of_abs_div_sum_l214_214527


namespace probability_bijection_l214_214487

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 2, 3, 4, 5}

theorem probability_bijection : 
  let total_mappings := 5^4
  let bijections := 5 * 4 * 3 * 2
  let probability := bijections / total_mappings
  probability = 24 / 125 := 
by
  sorry

end probability_bijection_l214_214487


namespace certain_number_is_213_l214_214210

theorem certain_number_is_213 (x : ℝ) (h1 : x * 16 = 3408) (h2 : x * 1.6 = 340.8) : x = 213 :=
sorry

end certain_number_is_213_l214_214210


namespace sum_of_three_is_odd_implies_one_is_odd_l214_214399

theorem sum_of_three_is_odd_implies_one_is_odd 
  (a b c : ℤ) 
  (h : (a + b + c) % 2 = 1) : 
  a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1 := 
sorry

end sum_of_three_is_odd_implies_one_is_odd_l214_214399


namespace compute_fraction_l214_214465

theorem compute_fraction :
  ((1/3)^4 * (1/5) = (1/405)) :=
by
  sorry

end compute_fraction_l214_214465


namespace time_rachel_is_13_l214_214426

-- Definitions based on problem conditions
def time_matt := 12
def time_patty := time_matt / 3
def time_rachel := 2 * time_patty + 5

-- Theorem statement to prove Rachel's time to paint the house
theorem time_rachel_is_13 : time_rachel = 13 := 
by 
  sorry

end time_rachel_is_13_l214_214426


namespace volume_of_tetrahedron_eq_20_l214_214243

noncomputable def volume_tetrahedron (a b c : ℝ) : ℝ :=
  1 / 3 * a * b * c

theorem volume_of_tetrahedron_eq_20 {x y z : ℝ} (h1 : x^2 + y^2 = 25) (h2 : y^2 + z^2 = 41) (h3 : z^2 + x^2 = 34) :
  volume_tetrahedron 3 4 5 = 20 :=
by
  sorry

end volume_of_tetrahedron_eq_20_l214_214243


namespace patio_tiles_l214_214175

theorem patio_tiles (r c : ℕ) (h1 : r * c = 48) (h2 : (r + 4) * (c - 2) = 48) : r = 6 :=
sorry

end patio_tiles_l214_214175


namespace range_of_a_l214_214160

theorem range_of_a (b c a : ℝ) (h_intersect : ∀ x : ℝ, 
  (x ^ 2 - 2 * b * x + b ^ 2 + c = 1 - x → x = b )) 
  (h_vertex : c = a * b ^ 2) :
  a ≥ (-1 / 5) ∧ a ≠ 0 := 
by 
-- Proof skipped
sorry

end range_of_a_l214_214160


namespace problem_1_problem_2_l214_214885

noncomputable def f (x : ℝ) : ℝ := x^3 + 1 / (x + 1)

theorem problem_1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x ≥ 1 - x + x^2 := 
sorry

theorem problem_2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (h1 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≥ 1 - x + x^2) : f x > 3 / 4 := 
sorry

end problem_1_problem_2_l214_214885


namespace number_of_knights_l214_214854

/--
On the island of Liars and Knights, a circular arrangement is called correct if everyone standing in the circle
can say that among his two neighbors there is a representative of his tribe. One day, 2019 natives formed a correct
arrangement in a circle. A liar approached them and said: "Now together we can also form a correct arrangement in a circle."
Prove that the number of knights in the initial arrangement is 1346.
-/
theorem number_of_knights : 
  ∀ (K L : ℕ), 
    (K + L = 2019) → 
    (K ≥ 2 * L) → 
    (K ≤ 2 * L + 1) → 
  K = 1346 :=
by
  intros K L h1 h2 h3
  sorry

end number_of_knights_l214_214854


namespace Okeydokey_should_receive_25_earthworms_l214_214547

def applesOkeydokey : ℕ := 5
def applesArtichokey : ℕ := 7
def totalEarthworms : ℕ := 60
def totalApples : ℕ := applesOkeydokey + applesArtichokey
def okeydokeyProportion : ℚ := applesOkeydokey / totalApples
def okeydokeyEarthworms : ℚ := okeydokeyProportion * totalEarthworms

theorem Okeydokey_should_receive_25_earthworms : okeydokeyEarthworms = 25 := by
  sorry

end Okeydokey_should_receive_25_earthworms_l214_214547


namespace decreasing_interval_of_function_l214_214554

noncomputable def y (x : ℝ) : ℝ := (3 / Real.pi) ^ (x ^ 2 + 2 * x - 3)

theorem decreasing_interval_of_function :
  ∀ x ∈ Set.Ioi (-1 : ℝ), ∃ ε > 0, ∀ δ > 0, δ ≤ ε → y (x - δ) > y x :=
by
  sorry

end decreasing_interval_of_function_l214_214554


namespace route_down_distance_l214_214277

theorem route_down_distance
  (rate_up : ℕ)
  (time_up : ℕ)
  (rate_down_rate_factor : ℚ)
  (time_down : ℕ)
  (h1 : rate_up = 4)
  (h2 : time_up = 2)
  (h3 : rate_down_rate_factor = (3 / 2))
  (h4 : time_down = time_up) :
  rate_down_rate_factor * rate_up * time_up = 12 := 
by
  rw [h1, h2, h3]
  sorry

end route_down_distance_l214_214277


namespace min_value_expression_l214_214740

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 2 / b) * (a + 2 / b - 1010) + (b + 2 / a) * (b + 2 / a - 1010) + 101010 = -404040 :=
sorry

end min_value_expression_l214_214740


namespace range_of_x_l214_214568

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x

theorem range_of_x (x : ℝ) (h : f (x^2 + 2) < f (3 * x)) : 1 < x ∧ x < 2 :=
by sorry

end range_of_x_l214_214568


namespace highest_power_of_3_divides_l214_214453

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem highest_power_of_3_divides (n : ℕ) : ∃ k : ℕ, A_n n = 3^n * k ∧ ¬ (3 * A_n n = 3^(n+1) * k)
:= by
  sorry

end highest_power_of_3_divides_l214_214453


namespace solution_l214_214321

noncomputable def prove_a_greater_than_3 : Prop :=
  ∀ (x : ℝ) (a : ℝ), (a > 0) → (|x - 2| + |x - 3| + |x - 4| < a) → a > 3

theorem solution : prove_a_greater_than_3 :=
by
  intros x a h_pos h_ineq
  sorry

end solution_l214_214321


namespace original_ratio_l214_214718

theorem original_ratio (x y : ℕ) (h1 : x = y + 5) (h2 : (x - 5) / (y - 5) = 5 / 4) : x / y = 6 / 5 :=
by sorry

end original_ratio_l214_214718


namespace intersection_vertices_of_regular_octagon_l214_214010

noncomputable def set_A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1| + |p.2| = a ∧ a > 0}

def set_B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 * p.2| + 1 = |p.1| + |p.2|}

theorem intersection_vertices_of_regular_octagon (a : ℝ) :
  (∃ (p : ℝ × ℝ), p ∈ set_A a ∧ p ∈ set_B) ↔ (a = Real.sqrt 2 ∨ a = 2 + Real.sqrt 2) :=
  sorry

end intersection_vertices_of_regular_octagon_l214_214010


namespace find_y_in_terms_of_x_l214_214868

theorem find_y_in_terms_of_x (p : ℝ) (x y : ℝ) (h1 : x = 1 + 3^p) (h2 : y = 1 + 3^(-p)) : y = x / (x - 1) :=
by
  sorry

end find_y_in_terms_of_x_l214_214868


namespace gcd_of_18_and_30_l214_214813

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l214_214813


namespace probability_of_three_black_balls_l214_214668

def total_ball_count : ℕ := 4 + 8

def white_ball_count : ℕ := 4

def black_ball_count : ℕ := 8

def total_combinations : ℕ := Nat.choose total_ball_count 3

def black_combinations : ℕ := Nat.choose black_ball_count 3

def probability_three_black : ℚ := black_combinations / total_combinations

theorem probability_of_three_black_balls : 
  probability_three_black = 14 / 55 := 
sorry

end probability_of_three_black_balls_l214_214668


namespace inequality_holds_for_all_x_y_l214_214519

theorem inequality_holds_for_all_x_y (x y : ℝ) : 
  x^2 + y^2 + 1 ≥ x + y + x * y := 
by sorry

end inequality_holds_for_all_x_y_l214_214519


namespace greatest_three_digit_multiple_of_17_l214_214738

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l214_214738


namespace convert_to_scientific_notation_l214_214812

theorem convert_to_scientific_notation :
  (1670000000 : ℝ) = 1.67 * 10 ^ 9 := 
by
  sorry

end convert_to_scientific_notation_l214_214812


namespace option_B_is_equal_to_a_8_l214_214058

-- Statement: (a^2)^4 equals a^8
theorem option_B_is_equal_to_a_8 (a : ℝ) : (a^2)^4 = a^8 :=
by { sorry }

end option_B_is_equal_to_a_8_l214_214058


namespace find_g_at_75_l214_214410

noncomputable def g : ℝ → ℝ := sorry

-- Conditions
axiom g_property : ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y^2
axiom g_at_50 : g 50 = 25

-- The main result to be proved
theorem find_g_at_75 : g 75 = 100 / 9 :=
by
  sorry

end find_g_at_75_l214_214410


namespace find_other_endpoint_l214_214231

theorem find_other_endpoint (x_m y_m : ℤ) (x1 y1 : ℤ) 
(m_cond : x_m = (x1 + (-1)) / 2) (m_cond' : y_m = (y1 + (-4)) / 2) : 
(x_m, y_m) = (3, -1) ∧ (x1, y1) = (7, 2) → (-1, -4) = (-1, -4) :=
by
  sorry

end find_other_endpoint_l214_214231


namespace weight_loss_comparison_l214_214855

-- Define the conditions
def weight_loss_Barbi : ℝ := 1.5 * 24
def weight_loss_Luca : ℝ := 9 * 15
def weight_loss_Kim : ℝ := (2 * 12) + (3 * 60)

-- Define the combined weight loss of Luca and Kim
def combined_weight_loss_Luca_Kim : ℝ := weight_loss_Luca + weight_loss_Kim

-- Define the difference in weight loss between Luca and Kim combined and Barbi
def weight_loss_difference : ℝ := combined_weight_loss_Luca_Kim - weight_loss_Barbi

-- State the theorem to be proved
theorem weight_loss_comparison : weight_loss_difference = 303 := by
  sorry

end weight_loss_comparison_l214_214855


namespace unused_streetlights_remain_l214_214753

def total_streetlights : ℕ := 200
def squares : ℕ := 15
def streetlights_per_square : ℕ := 12

theorem unused_streetlights_remain :
  total_streetlights - (squares * streetlights_per_square) = 20 :=
sorry

end unused_streetlights_remain_l214_214753


namespace satisfies_conditions_l214_214481

noncomputable def m := 29 / 3

def real_part (m : ℝ) : ℝ := m^2 - 8*m + 15
def imag_part (m : ℝ) : ℝ := m^2 - 5*m - 14

theorem satisfies_conditions (m : ℝ) 
  (real_cond : m < 3 ∨ m > 5) 
  (imag_cond : -2 < m ∧ m < 7)
  (line_cond : real_part m = imag_part m): 
  m = 29 / 3 :=
by {
  sorry
}

end satisfies_conditions_l214_214481


namespace f_alpha_l214_214917

variables (α : Real) (x : Real)

noncomputable def f (x : Real) : Real := 
  (Real.cos (Real.pi + x) * Real.sin (2 * Real.pi - x)) / Real.cos (Real.pi - x)

lemma sin_alpha {α : Real} (hcos : Real.cos α = 1 / 3) (hα : 0 < α ∧ α < Real.pi) : 
  Real.sin α = 2 * Real.sqrt 2 / 3 :=
sorry

lemma tan_alpha {α : Real} (hsin : Real.sin α = 2 * Real.sqrt 2 / 3) (hcos : Real.cos α = 1 / 3) :
  Real.tan α = 2 * Real.sqrt 2 :=
sorry

theorem f_alpha {α : Real} (hcos : Real.cos α = 1 / 3) (hα : 0 < α ∧ α < Real.pi) :
  f α = -2 * Real.sqrt 2 / 3 :=
sorry

end f_alpha_l214_214917


namespace part1_part2_l214_214263

/- Define the function f(x) = |x-1| + |x-a| -/
def f (x a : ℝ) := abs (x - 1) + abs (x - a)

/- Part 1: Prove that if f(x) ≥ 2 implies the solution set {x | x ≤ 1/2 or x ≥ 5/2}, then a = 2 -/
theorem part1 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2 → (x ≤ 1/2 ∨ x ≥ 5/2)) : a = 2 :=
  sorry

/- Part 2: Prove that for all x ∈ ℝ, f(x) + |x-1| ≥ 1 implies a ∈ [2, +∞) -/
theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a + abs (x - 1) ≥ 1) : 2 ≤ a :=
  sorry

end part1_part2_l214_214263


namespace perimeter_C_correct_l214_214181

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l214_214181


namespace sum_of_numbers_l214_214493

-- Definitions for the numbers involved
def n1 : Nat := 1235
def n2 : Nat := 2351
def n3 : Nat := 3512
def n4 : Nat := 5123

-- Proof statement
theorem sum_of_numbers :
  n1 + n2 + n3 + n4 = 12221 := by
  sorry

end sum_of_numbers_l214_214493


namespace max_product_partition_l214_214267

theorem max_product_partition (k n : ℕ) (hkn : k ≥ n) 
  (q r : ℕ) (hqr : k = n * q + r) (h_r : 0 ≤ r ∧ r < n) : 
  ∃ (F : ℕ → ℕ), F k = q^(n-r) * (q+1)^r :=
by
  sorry

end max_product_partition_l214_214267


namespace sandrine_washed_160_dishes_l214_214439

-- Define the number of pears picked by Charles
def charlesPears : ℕ := 50

-- Define the number of bananas cooked by Charles as 3 times the number of pears he picked
def charlesBananas : ℕ := 3 * charlesPears

-- Define the number of dishes washed by Sandrine as 10 more than the number of bananas Charles cooked
def sandrineDishes : ℕ := charlesBananas + 10

-- Prove that Sandrine washed 160 dishes
theorem sandrine_washed_160_dishes : sandrineDishes = 160 := by
  -- The proof is omitted
  sorry

end sandrine_washed_160_dishes_l214_214439


namespace no_four_consecutive_product_square_l214_214344

/-- Prove that there do not exist four consecutive positive integers whose product is a perfect square. -/
theorem no_four_consecutive_product_square :
  ¬ ∃ (x : ℕ), ∃ (n : ℕ), n * n = x * (x + 1) * (x + 2) * (x + 3) :=
sorry

end no_four_consecutive_product_square_l214_214344


namespace log_bound_sum_l214_214864

theorem log_bound_sum (c d : ℕ) (h_c : c = 10) (h_d : d = 11) (h_bound : 10 < Real.log 1350 / Real.log 2 ∧ Real.log 1350 / Real.log 2 < 11) : c + d = 21 :=
by
  -- omitted proof
  sorry

end log_bound_sum_l214_214864


namespace range_of_independent_variable_l214_214068

theorem range_of_independent_variable (x : ℝ) : x ≠ -3 ↔ ∃ y : ℝ, y = 1 / (x + 3) :=
by 
  -- Proof is omitted
  sorry

end range_of_independent_variable_l214_214068


namespace janelle_initial_green_marbles_l214_214138

def initial_green_marbles (blue_bags : ℕ) (marbles_per_bag : ℕ) (gift_green : ℕ) (gift_blue : ℕ) (remaining_marbles : ℕ) : ℕ :=
  let blue_marbles := blue_bags * marbles_per_bag
  let remaining_blue_marbles := blue_marbles - gift_blue
  let remaining_green_marbles := remaining_marbles - remaining_blue_marbles
  remaining_green_marbles + gift_green

theorem janelle_initial_green_marbles :
  initial_green_marbles 6 10 6 8 72 = 26 :=
by
  rfl

end janelle_initial_green_marbles_l214_214138


namespace linear_eq_must_be_one_l214_214865

theorem linear_eq_must_be_one (m : ℝ) : (∀ x y : ℝ, (m + 1) * x + 3 * y ^ m = 5 → (m = 1)) :=
by
  intros x y h
  sorry

end linear_eq_must_be_one_l214_214865


namespace arithmetic_progression_term_l214_214411

variable (n r : ℕ)

-- Given the sum of the first n terms of an arithmetic progression is S_n = 3n + 4n^2
def S (n : ℕ) : ℕ := 3 * n + 4 * n^2

-- Prove that the r-th term of the sequence is 8r - 1
theorem arithmetic_progression_term :
  (S r) - (S (r - 1)) = 8 * r - 1 :=
by
  sorry

end arithmetic_progression_term_l214_214411


namespace sports_club_problem_l214_214196

theorem sports_club_problem (N B T Neither X : ℕ) (hN : N = 42) (hB : B = 20) (hT : T = 23) (hNeither : Neither = 6) :
  (B + T - X + Neither = N) → X = 7 :=
by
  intro h
  sorry

end sports_club_problem_l214_214196


namespace original_set_cardinality_l214_214029

-- Definitions based on conditions
def is_reversed_error (n : ℕ) : Prop :=
  ∃ (A B C : ℕ), 100 * A + 10 * B + C = n ∧ 100 * C + 10 * B + A = n + 198 ∧ C - A = 2

-- The theorem to prove
theorem original_set_cardinality : ∃ n : ℕ, is_reversed_error n ∧ n = 10 := by
  sorry

end original_set_cardinality_l214_214029


namespace range_of_a_l214_214002

variable (a x y : ℝ)

def proposition_p : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def proposition_q : Prop :=
  (1 - a) * (a - 3) < 0

theorem range_of_a (h1 : proposition_p a) (h2 : proposition_q a) : 
  (0 ≤ a ∧ a < 1) ∨ (3 < a ∧ a < 4) :=
by
  sorry

end range_of_a_l214_214002


namespace find_angle_BEC_l214_214489

theorem find_angle_BEC (A B C D E : Type) (angle_A angle_B angle_D angle_DEC angle_C angle_CED angle_BEC : ℝ) 
  (hA : angle_A = 50) (hB : angle_B = 90) (hD : angle_D = 70) (hDEC : angle_DEC = 20)
  (h_quadrilateral_sum: angle_A + angle_B + angle_C + angle_D = 360)
  (h_C : angle_C = 150)
  (h_CED : angle_CED = angle_C - angle_DEC)
  (h_BEC: angle_BEC = 180 - angle_B - angle_CED) : angle_BEC = 110 :=
by
  -- Definitions according to the given problem
  have h1 : angle_C = 360 - (angle_A + angle_B + angle_D) := by sorry
  have h2 : angle_CED = angle_C - angle_DEC := by sorry
  have h3 : angle_BEC = 180 - angle_B - angle_CED := by sorry

  -- Proving the required angle
  have h_goal : angle_BEC = 110 := by
    sorry  -- Actual proof steps go here

  exact h_goal

end find_angle_BEC_l214_214489


namespace calculate_expression_l214_214811

theorem calculate_expression : ∀ x y : ℝ, x = 7 → y = 3 → (x - y) ^ 2 * (x + y) = 160 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end calculate_expression_l214_214811


namespace two_pipes_fill_time_l214_214977

theorem two_pipes_fill_time (R : ℝ) (h1 : (3 : ℝ) * R * (8 : ℝ) = 1) : (2 : ℝ) * R * (12 : ℝ) = 1 :=
by 
  have hR : R = 1 / 24 := by linarith
  rw [hR]
  sorry

end two_pipes_fill_time_l214_214977


namespace maximum_m2_n2_l214_214724

theorem maximum_m2_n2 
  (m n : ℤ)
  (hm : 1 ≤ m ∧ m ≤ 1981) 
  (hn : 1 ≤ n ∧ n ≤ 1981)
  (h : (n^2 - m*n - m^2)^2 = 1) :
  m^2 + n^2 ≤ 3524578 :=
sorry

end maximum_m2_n2_l214_214724


namespace no_solution_for_equation_l214_214045

theorem no_solution_for_equation :
  ¬(∃ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ (x+2)/(x-2) - x/(x+2) = 16/(x^2-4)) :=
by
    sorry

end no_solution_for_equation_l214_214045


namespace pyramid_new_volume_l214_214328

-- Define constants
def V : ℝ := 100
def l : ℝ := 3
def w : ℝ := 2
def h : ℝ := 1.20

-- Define the theorem
theorem pyramid_new_volume : (l * w * h) * V = 720 := by
  sorry -- Proof is skipped

end pyramid_new_volume_l214_214328


namespace train_speed_l214_214327

-- Definition of the problem
def train_length : ℝ := 350
def time_to_cross_man : ℝ := 4.5
def expected_speed : ℝ := 77.78

-- Theorem statement
theorem train_speed :
  train_length / time_to_cross_man = expected_speed :=
sorry

end train_speed_l214_214327


namespace born_in_1890_l214_214023

theorem born_in_1890 (x : ℕ) (h1 : x^2 - x - 2 = 1890) (h2 : x^2 < 1950) : x = 44 :=
by {
    sorry
}

end born_in_1890_l214_214023


namespace compound_interest_time_l214_214592

theorem compound_interest_time (P r CI : ℝ) (n : ℕ) (A : ℝ) :
  P = 16000 ∧ r = 0.15 ∧ CI = 6218 ∧ n = 1 ∧ A = P + CI →
  t = 2 :=
by
  sorry

end compound_interest_time_l214_214592


namespace Canada_moose_population_l214_214887

theorem Canada_moose_population (moose beavers humans : ℕ) (h1 : beavers = 2 * moose) 
                              (h2 : humans = 19 * beavers) (h3 : humans = 38 * 10^6) : 
                              moose = 1 * 10^6 :=
by
  sorry

end Canada_moose_population_l214_214887


namespace f_6_plus_f_neg3_l214_214104

noncomputable def f : ℝ → ℝ := sorry

-- f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- f is increasing in the interval [3,6]
def is_increasing_interval (f : ℝ → ℝ) (a b : ℝ) := a ≤ b → ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

-- Define the given conditions
axiom h1 : is_odd_function f
axiom h2 : is_increasing_interval f 3 6
axiom h3 : f 6 = 8
axiom h4 : f 3 = -1

-- The statement to be proved
theorem f_6_plus_f_neg3 : f 6 + f (-3) = 9 :=
by
  sorry

end f_6_plus_f_neg3_l214_214104


namespace base_subtraction_l214_214742

def base8_to_base10 (n : Nat) : Nat :=
  -- base 8 number 54321 (in decimal representation)
  5 * 4096 + 4 * 512 + 3 * 64 + 2 * 8 + 1

def base5_to_base10 (n : Nat) : Nat :=
  -- base 5 number 4321 (in decimal representation)
  4 * 125 + 3 * 25 + 2 * 5 + 1

theorem base_subtraction :
  base8_to_base10 54321 - base5_to_base10 4321 = 22151 := by
  sorry

end base_subtraction_l214_214742


namespace solve_problem_l214_214896

-- Define the variables and conditions
def problem_statement : Prop :=
  ∃ x : ℕ, 865 * 48 = 240 * x ∧ x = 173

-- Statement to prove
theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l214_214896


namespace farm_field_area_l214_214491

theorem farm_field_area
  (plough_per_day_planned plough_per_day_actual fields_left : ℕ)
  (D : ℕ) 
  (condition1 : plough_per_day_planned = 100)
  (condition2 : plough_per_day_actual = 85)
  (condition3 : fields_left = 40)
  (additional_days : ℕ) 
  (condition4 : additional_days = 2)
  (initial_days : D + additional_days = 85 * (D + 2) + 40) :
  (100 * D + fields_left = 1440) :=
by
  sorry

end farm_field_area_l214_214491


namespace solve_quadratic_l214_214584

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = 2 + Real.sqrt 11) ∧ (x2 = 2 - (Real.sqrt 11)) ∧ 
  (∀ x : ℝ, x^2 - 4*x - 7 = 0 ↔ x = x1 ∨ x = x2) := 
sorry

end solve_quadratic_l214_214584


namespace fraction_filled_in_5_minutes_l214_214743

-- Conditions
def fill_time : ℕ := 55 -- Total minutes to fill the cistern
def duration : ℕ := 5  -- Minutes we are examining

-- The theorem to prove that the fraction filled in 'duration' minutes is 1/11
theorem fraction_filled_in_5_minutes : (duration : ℚ) / (fill_time : ℚ) = 1 / 11 :=
by
  have fraction_per_minute : ℚ := 1 / fill_time
  have fraction_in_5_minutes : ℚ := duration * fraction_per_minute
  sorry -- Proof steps would go here, if needed.

end fraction_filled_in_5_minutes_l214_214743


namespace gas_pressure_inversely_proportional_l214_214789

theorem gas_pressure_inversely_proportional
  (p v k : ℝ)
  (v_i v_f : ℝ)
  (p_i p_f : ℝ)
  (h1 : v_i = 3.5)
  (h2 : p_i = 8)
  (h3 : v_f = 7)
  (h4 : p * v = k)
  (h5 : p_i * v_i = k)
  (h6 : p_f * v_f = k) : p_f = 4 := by
  sorry

end gas_pressure_inversely_proportional_l214_214789


namespace rainfall_in_2011_l214_214752

-- Define the parameters
def avg_rainfall_2010 : ℝ := 37.2
def increase_from_2010_to_2011 : ℝ := 1.8
def months_in_a_year : ℕ := 12

-- Define the total rainfall in 2011
def total_rainfall_2011 : ℝ := 468

-- Prove that the total rainfall in Driptown in 2011 is 468 mm
theorem rainfall_in_2011 :
  avg_rainfall_2010 + increase_from_2010_to_2011 = 39.0 → 
  12 * (avg_rainfall_2010 + increase_from_2010_to_2011) = total_rainfall_2011 :=
by sorry

end rainfall_in_2011_l214_214752


namespace problem_part1_problem_part2_l214_214552

noncomputable def f (x a : ℝ) := |x - a| + x

theorem problem_part1 (a : ℝ) (h_a : a = 1) :
  {x : ℝ | f x a ≥ x + 2} = {x : ℝ | x ≥ 3} ∪ {x : ℝ | x ≤ -1} :=
by 
  simp [h_a, f]
  sorry

theorem problem_part2 (a : ℝ) (h_solution : {x : ℝ | f x a ≤ 3 * x} = {x : ℝ | x ≥ 2}) :
  a = 6 :=
by
  simp [f] at h_solution
  sorry

end problem_part1_problem_part2_l214_214552


namespace find_value_of_x_l214_214218

theorem find_value_of_x (b : ℕ) (x : ℝ) (h_b_pos : b > 0) (h_x_pos : x > 0) 
  (h_r1 : r = 4 ^ (2 * b)) (h_r2 : r = 2 ^ b * x ^ b) : x = 8 :=
by
  -- Proof omitted for brevity
  sorry

end find_value_of_x_l214_214218


namespace gcd_of_1230_and_920_is_10_l214_214642

theorem gcd_of_1230_and_920_is_10 : Int.gcd 1230 920 = 10 :=
sorry

end gcd_of_1230_and_920_is_10_l214_214642


namespace complex_exponential_sum_identity_l214_214935

theorem complex_exponential_sum_identity :
    12 * Complex.exp (Real.pi * Complex.I / 7) + 12 * Complex.exp (19 * Real.pi * Complex.I / 14) =
    24 * Real.cos (5 * Real.pi / 28) * Complex.exp (3 * Real.pi * Complex.I / 4) :=
sorry

end complex_exponential_sum_identity_l214_214935


namespace largest_of_five_numbers_l214_214941

theorem largest_of_five_numbers : ∀ (a b c d e : ℝ), 
  a = 0.938 → b = 0.9389 → c = 0.93809 → d = 0.839 → e = 0.893 → b = max a (max b (max c (max d e))) :=
by
  intros a b c d e ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end largest_of_five_numbers_l214_214941


namespace solve_part_a_solve_part_b_solve_part_c_l214_214965

-- Part (a)
theorem solve_part_a (x : ℝ) : 
  (2 * x^2 + 3 * x - 1)^2 - 5 * (2 * x^2 + 3 * x + 3) + 24 = 0 ↔ 
  x = 1 ∨ x = -2 ∨ x = 0.5 ∨ x = -2.5 := sorry

-- Part (b)
theorem solve_part_b (x : ℝ) : 
  (x - 1) * (x + 3) * (x + 4) * (x + 8) = -96 ↔ 
  x = 0 ∨ x = -7 ∨ x = (-7 + Real.sqrt 33) / 2 ∨ x = (-7 - Real.sqrt 33) / 2 := sorry

-- Part (c)
theorem solve_part_c (x : ℝ) (hx : x ≠ 0) : 
  (x - 1) * (x - 2) * (x - 4) * (x - 8) = 4 * x^2 ↔ 
  x = 4 + 2 * Real.sqrt 2 ∨ x = 4 - 2 * Real.sqrt 2 := sorry

end solve_part_a_solve_part_b_solve_part_c_l214_214965


namespace division_multiplication_order_l214_214266

theorem division_multiplication_order : 1100 / 25 * 4 / 11 = 16 := by
  sorry

end division_multiplication_order_l214_214266


namespace sqrt_calculation_l214_214591

theorem sqrt_calculation :
  Real.sqrt ((2:ℝ)^4 * 3^2 * 5^2) = 60 := 
by sorry

end sqrt_calculation_l214_214591


namespace point_in_second_quadrant_l214_214157

def point (x : ℤ) (y : ℤ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant : point (-1) 3 = true := by
  sorry

end point_in_second_quadrant_l214_214157


namespace no_solution_system_of_inequalities_l214_214092

theorem no_solution_system_of_inequalities :
  ¬ ∃ (x y : ℝ),
    11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧
    5 * x + y ≤ -10 :=
by
  sorry

end no_solution_system_of_inequalities_l214_214092


namespace roberts_test_score_l214_214249

structure ClassState where
  num_students : ℕ
  avg_19_students : ℕ
  class_avg_20_students : ℕ

def calculate_roberts_score (s : ClassState) : ℕ :=
  let total_19_students := s.num_students * s.avg_19_students
  let total_20_students := (s.num_students + 1) * s.class_avg_20_students
  total_20_students - total_19_students

theorem roberts_test_score 
  (state : ClassState) 
  (h1 : state.num_students = 19) 
  (h2 : state.avg_19_students = 74)
  (h3 : state.class_avg_20_students = 75) : 
  calculate_roberts_score state = 94 := by
  sorry

end roberts_test_score_l214_214249


namespace five_coins_total_cannot_be_30_cents_l214_214258

theorem five_coins_total_cannot_be_30_cents :
  ¬ ∃ (a b c d e : ℕ), 
  a + b + c + d + e = 5 ∧ 
  (a * 1 + b * 5 + c * 10 + d * 25 + e * 50) = 30 := 
sorry

end five_coins_total_cannot_be_30_cents_l214_214258


namespace contradiction_example_l214_214858

theorem contradiction_example (x : ℝ) (h : x^2 - 1 = 0) : x = -1 ∨ x = 1 :=
by
  sorry

end contradiction_example_l214_214858


namespace tetrahedron_surface_area_l214_214907

theorem tetrahedron_surface_area (a : ℝ) (h : a = Real.sqrt 2) :
  let R := (a * Real.sqrt 6) / 4
  let S := 4 * Real.pi * R^2
  S = 3 * Real.pi := by
  /- Proof here -/
  sorry

end tetrahedron_surface_area_l214_214907


namespace anne_distance_l214_214125

theorem anne_distance (speed time : ℕ) (h_speed : speed = 2) (h_time : time = 3) : 
  (speed * time) = 6 := by
  sorry

end anne_distance_l214_214125


namespace project_contribution_l214_214833

theorem project_contribution (total_cost : ℝ) (num_participants : ℝ) (expected_contribution : ℝ) 
  (h1 : total_cost = 25 * 10^9) 
  (h2 : num_participants = 300 * 10^6) 
  (h3 : expected_contribution = 83) : 
  total_cost / num_participants = expected_contribution := 
by 
  sorry

end project_contribution_l214_214833


namespace parallel_lines_l214_214550

theorem parallel_lines (a : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, a * x + 2 * y - 1 = k * (2 * x + a * y + 2)) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end parallel_lines_l214_214550


namespace problem1_problem2_problem3_l214_214696

theorem problem1 (a : ℝ) : |a + 2| = 4 → (a = 2 ∨ a = -6) :=
sorry

theorem problem2 (a : ℝ) (h₀ : -4 < a) (h₁ : a < 2) : |a + 4| + |a - 2| = 6 :=
sorry

theorem problem3 (a : ℝ) : ∃ x ∈ Set.Icc (-2 : ℝ) 1, |x-1| + |x+2| = 3 :=
sorry

end problem1_problem2_problem3_l214_214696


namespace find_x_l214_214661

theorem find_x (x : ℝ) (h : 5020 - (x / 100.4) = 5015) : x = 502 :=
sorry

end find_x_l214_214661


namespace solve_repeating_decimals_sum_l214_214589

def repeating_decimals_sum : Prop :=
  let x := (1 : ℚ) / 3
  let y := (4 : ℚ) / 999
  let z := (5 : ℚ) / 9999
  x + y + z = 3378 / 9999

theorem solve_repeating_decimals_sum : repeating_decimals_sum := 
by 
  sorry

end solve_repeating_decimals_sum_l214_214589


namespace factor_polynomial_l214_214903

theorem factor_polynomial (a b : ℕ) : 
  2 * a^3 - 3 * a^2 * b - 3 * a * b^2 + 2 * b^3 = (a + b) * (a - 2 * b) * (2 * a - b) :=
by sorry

end factor_polynomial_l214_214903


namespace train_speed_l214_214211

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 3500) (h_time : time = 80) : 
  length / time = 43.75 := 
by 
  sorry

end train_speed_l214_214211


namespace price_reduction_example_l214_214609

def original_price_per_mango (P : ℝ) : Prop :=
  (115 * P = 383.33)

def number_of_mangoes (P : ℝ) (n : ℝ) : Prop :=
  (n * P = 360)

def new_number_of_mangoes (n : ℝ) (R : ℝ) : Prop :=
  ((n + 12) * R = 360)

def percentage_reduction (P R : ℝ) (reduction : ℝ) : Prop :=
  (reduction = ((P - R) / P) * 100)

theorem price_reduction_example : 
  ∃ P R reduction, original_price_per_mango P ∧
    (∃ n, number_of_mangoes P n ∧ new_number_of_mangoes n R) ∧ 
    percentage_reduction P R reduction ∧ 
    reduction = 9.91 :=
by
  sorry

end price_reduction_example_l214_214609


namespace simplify_eval_expression_l214_214467

theorem simplify_eval_expression : 
  ∀ (a b : ℤ), a = -1 → b = 4 → ((a - b)^2 - 2 * a * (a + b) + (a + 2 * b) * (a - 2 * b)) = -32 := 
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_eval_expression_l214_214467


namespace trapezoid_area_l214_214763

-- Definitions based on conditions
def CL_div_LD (CL LD : ℝ) : Prop := CL / LD = 1 / 4

-- The main statement we want to prove
theorem trapezoid_area (BC CD : ℝ) (h1 : BC = 9) (h2 : CD = 30) (CL LD : ℝ) (h3 : CL_div_LD CL LD) : 
  1/2 * (BC + AD) * 24 = 972 :=
sorry

end trapezoid_area_l214_214763


namespace leak_time_to_empty_l214_214866

def pump_rate : ℝ := 0.1 -- P = 0.1 tanks/hour
def effective_rate : ℝ := 0.05 -- P - L = 0.05 tanks/hour

theorem leak_time_to_empty (P L : ℝ) (hp : P = pump_rate) (he : P - L = effective_rate) :
  1 / L = 20 := by
  sorry

end leak_time_to_empty_l214_214866


namespace liquid_X_percentage_correct_l214_214759

noncomputable def percent_liquid_X_in_solution_A := 0.8 / 100
noncomputable def percent_liquid_X_in_solution_B := 1.8 / 100

noncomputable def weight_solution_A := 400.0
noncomputable def weight_solution_B := 700.0

noncomputable def weight_liquid_X_in_A := percent_liquid_X_in_solution_A * weight_solution_A
noncomputable def weight_liquid_X_in_B := percent_liquid_X_in_solution_B * weight_solution_B

noncomputable def total_weight_solution := weight_solution_A + weight_solution_B
noncomputable def total_weight_liquid_X := weight_liquid_X_in_A + weight_liquid_X_in_B

noncomputable def percent_liquid_X_in_mixed_solution := (total_weight_liquid_X / total_weight_solution) * 100

theorem liquid_X_percentage_correct :
  percent_liquid_X_in_mixed_solution = 1.44 :=
by
  sorry

end liquid_X_percentage_correct_l214_214759


namespace statement_A_statement_D_l214_214908

variable (a b c d : ℝ)

-- Statement A: If ac² > bc², then a > b
theorem statement_A (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by
  sorry

-- Statement D: If a > b > 0, then a + 1/b > b + 1/a
theorem statement_D (h1 : a > b) (h2 : b > 0) : a + 1 / b > b + 1 / a := by
  sorry

end statement_A_statement_D_l214_214908


namespace find_multiple_of_hats_l214_214778

/-
   Given:
   - Fire chief Simpson has 15 hats.
   - Policeman O'Brien now has 34 hats.
   - Before he lost one, Policeman O'Brien had 5 more hats than a certain multiple of Fire chief Simpson's hats.
   Prove:
   The multiple of Fire chief Simpson's hats that Policeman O'Brien had before he lost one is 2.
-/

theorem find_multiple_of_hats :
  ∃ x : ℕ, 34 + 1 = 5 + 15 * x ∧ x = 2 :=
by
  sorry

end find_multiple_of_hats_l214_214778


namespace markup_constant_relationship_l214_214784

variable (C S : ℝ) (k : ℝ)
variable (fractional_markup : k * S = 0.25 * C)
variable (relation : S = C + k * S)

theorem markup_constant_relationship (fractional_markup : k * S = 0.25 * C) (relation : S = C + k * S) :
  k = 1 / 5 :=
by
  sorry

end markup_constant_relationship_l214_214784


namespace martina_success_rate_l214_214745

theorem martina_success_rate
  (games_played : ℕ) (games_won : ℕ) (games_remaining : ℕ)
  (games_won_remaining : ℕ) :
  games_played = 15 → 
  games_won = 9 → 
  games_remaining = 5 → 
  games_won_remaining = 5 → 
  ((games_won + games_won_remaining) / (games_played + games_remaining) : ℚ) * 100 = 70 := 
by
  intros h1 h2 h3 h4
  sorry

end martina_success_rate_l214_214745


namespace sum_of_xyz_l214_214099

theorem sum_of_xyz (x y z : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : z > 0)
  (h4 : x^2 + y^2 + x * y = 3)
  (h5 : y^2 + z^2 + y * z = 4)
  (h6 : z^2 + x^2 + z * x = 7) :
  x + y + z = Real.sqrt 13 :=
by sorry -- Proof omitted, but the statement formulation is complete and checks the equality under given conditions.

end sum_of_xyz_l214_214099


namespace total_puppies_count_l214_214841

theorem total_puppies_count (total_cost sale_cost others_cost: ℕ) 
  (three_puppies_on_sale: ℕ) 
  (one_sale_puppy_cost: ℕ)
  (one_other_puppy_cost: ℕ)
  (h1: total_cost = 800)
  (h2: three_puppies_on_sale = 3)
  (h3: one_sale_puppy_cost = 150)
  (h4: others_cost = total_cost - three_puppies_on_sale * one_sale_puppy_cost)
  (h5: one_other_puppy_cost = 175)
  (h6: ∃ other_puppies : ℕ, other_puppies = others_cost / one_other_puppy_cost) :
  ∃ total_puppies : ℕ,
  total_puppies = three_puppies_on_sale + (others_cost / one_other_puppy_cost) := 
sorry

end total_puppies_count_l214_214841


namespace sugar_price_difference_l214_214820

theorem sugar_price_difference (a b : ℝ) (h : (3 / 5 * a + 2 / 5 * b) - (2 / 5 * a + 3 / 5 * b) = 1.32) :
  a - b = 6.6 :=
by
  sorry

end sugar_price_difference_l214_214820


namespace tangent_lines_ln_e_proof_l214_214155

noncomputable def tangent_tangent_ln_e : Prop :=
  ∀ (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) 
  (h₁_eq : x₂ = -Real.log x₁)
  (h₂_eq : Real.log x₁ - 1 = (Real.exp x₂) * (1 - x₂)),
  (2 / (x₁ - 1) + x₂ = -1)

theorem tangent_lines_ln_e_proof : tangent_tangent_ln_e :=
  sorry

end tangent_lines_ln_e_proof_l214_214155


namespace find_slope_of_line_l214_214528

theorem find_slope_of_line (k m x0 : ℝ) (P Q : ℝ × ℝ) 
  (hP : P.2^2 = 4 * P.1) 
  (hQ : Q.2^2 = 4 * Q.1) 
  (hMid : (P.1 + Q.1) / 2 = x0 ∧ (P.2 + Q.2) / 2 = 2) 
  (hLineP : P.2 = k * P.1 + m) 
  (hLineQ : Q.2 = k * Q.1 + m) : k = 1 :=
by sorry

end find_slope_of_line_l214_214528


namespace round_trip_time_l214_214281

variable (dist : ℝ)
variable (speed_to_work : ℝ)
variable (speed_to_home : ℝ)

theorem round_trip_time (h_dist : dist = 24) (h_speed_to_work : speed_to_work = 60) (h_speed_to_home : speed_to_home = 40) :
    (dist / speed_to_work + dist / speed_to_home) = 1 := 
by 
  sorry

end round_trip_time_l214_214281


namespace value_of_knife_l214_214188

/-- Two siblings sold their flock of sheep. Each sheep was sold for as many florins as 
the number of sheep originally in the flock. They divided the revenue by giving out 
10 florins at a time. First, the elder brother took 10 florins, then the younger brother, 
then the elder again, and so on. In the end, the younger brother received less than 10 florins, 
so the elder brother gave him his knife, making their earnings equal. 
Prove that the value of the knife in florins is 2. -/
theorem value_of_knife (n : ℕ) (k m : ℕ) (h1 : n^2 = 20 * k + 10 + m) (h2 : 1 ≤ m ∧ m ≤ 9) : 
  (∃ b : ℕ, 10 - b = m + b ∧ b = 2) :=
by
  sorry

end value_of_knife_l214_214188


namespace vasya_triangle_rotation_l214_214078

theorem vasya_triangle_rotation :
  (∀ (θ1 θ2 θ3 : ℝ), (12 * θ1 = 360) ∧ (6 * θ2 = 360) ∧ (θ1 + θ2 + θ3 = 180) → ∃ n : ℕ, (n * θ3 = 360) ∧ n ≥ 4) :=
by
  -- The formal proof is omitted, inserting "sorry" to indicate incomplete proof
  sorry

end vasya_triangle_rotation_l214_214078


namespace school_fitness_event_participants_l214_214397

theorem school_fitness_event_participants :
  let p0 := 500 -- initial number of participants in 2000
  let r1 := 0.3 -- increase rate in 2001
  let r2 := 0.4 -- increase rate in 2002
  let r3 := 0.5 -- increase rate in 2003
  let p1 := p0 * (1 + r1) -- participants in 2001
  let p2 := p1 * (1 + r2) -- participants in 2002
  let p3 := p2 * (1 + r3) -- participants in 2003
  p3 = 1365 -- prove that number of participants in 2003 is 1365
:= sorry

end school_fitness_event_participants_l214_214397


namespace spinsters_count_l214_214418

theorem spinsters_count (S C : ℕ) (h_ratio : S / C = 2 / 7) (h_diff : C = S + 55) : S = 22 :=
by
  sorry

end spinsters_count_l214_214418


namespace marble_count_l214_214392

-- Define the variables for the number of marbles
variables (o p y : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := o = 1.3 * p
def condition2 : Prop := y = 1.5 * o

-- Define the total number of marbles based on the conditions
def total_marbles : ℝ := o + p + y

-- The theorem statement that needs to be proved
theorem marble_count (h1 : condition1 o p) (h2 : condition2 o y) : total_marbles o p y = 3.269 * o :=
by sorry

end marble_count_l214_214392


namespace total_hours_until_joy_sees_grandma_l214_214648

theorem total_hours_until_joy_sees_grandma
  (days_until_grandma: ℕ)
  (hours_in_a_day: ℕ)
  (timezone_difference: ℕ)
  (H_days : days_until_grandma = 2)
  (H_hours : hours_in_a_day = 24)
  (H_timezone : timezone_difference = 3) :
  (days_until_grandma * hours_in_a_day = 48) :=
by
  sorry

end total_hours_until_joy_sees_grandma_l214_214648


namespace arithmetic_square_root_16_l214_214325

theorem arithmetic_square_root_16 : Real.sqrt 16 = 4 := by
  sorry

end arithmetic_square_root_16_l214_214325


namespace arithmetic_square_root_of_9_l214_214103

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l214_214103


namespace ratio_of_speeds_l214_214697

theorem ratio_of_speeds (k r t V1 V2 : ℝ) (hk : 0 < k) (hr : 0 < r) (ht : 0 < t)
    (h1 : r * (V1 - V2) = k) (h2 : t * (V1 + V2) = k) :
    |r + t| / |r - t| = V1 / V2 :=
by
  sorry

end ratio_of_speeds_l214_214697


namespace false_statement_about_circles_l214_214810

variable (P Q : Type) [MetricSpace P] [MetricSpace Q]
variable (p q : ℝ)
variable (dist_PQ : ℝ)

theorem false_statement_about_circles 
  (hA : p - q = dist_PQ → false)
  (hB : p + q = dist_PQ → false)
  (hC : p + q < dist_PQ → false)
  (hD : p - q < dist_PQ → false) : 
  false :=
by sorry

end false_statement_about_circles_l214_214810


namespace rectangular_field_perimeter_l214_214716

variable (length width : ℝ)

theorem rectangular_field_perimeter (h_area : length * width = 50) (h_width : width = 5) : 2 * (length + width) = 30 := by
  sorry

end rectangular_field_perimeter_l214_214716


namespace solve_siblings_age_problem_l214_214630

def siblings_age_problem (x : ℕ) : Prop :=
  let age_eldest := 20
  let age_middle := 15
  let age_youngest := 10
  (age_eldest + x) + (age_middle + x) + (age_youngest + x) = 75 → x = 10

theorem solve_siblings_age_problem : siblings_age_problem 10 :=
by
  sorry

end solve_siblings_age_problem_l214_214630


namespace daniel_waist_size_correct_l214_214429

noncomputable def Daniel_waist_size_cm (inches_to_feet : ℝ) (feet_to_cm : ℝ) (waist_size_in_inches : ℝ) : ℝ := 
  (waist_size_in_inches * feet_to_cm) / inches_to_feet

theorem daniel_waist_size_correct :
  Daniel_waist_size_cm 12 30.5 34 = 86.4 :=
by
  -- This skips the proof for now
  sorry

end daniel_waist_size_correct_l214_214429


namespace total_trout_caught_l214_214124

theorem total_trout_caught (n_share j_share total_caught : ℕ) (h1 : n_share = 9) (h2 : j_share = 9) (h3 : total_caught = n_share + j_share) :
  total_caught = 18 :=
by
  sorry

end total_trout_caught_l214_214124


namespace ellipse_AB_length_l214_214947

theorem ellipse_AB_length :
  ∀ (F1 F2 A B : ℝ × ℝ) (x y : ℝ),
  (x^2 / 25 + y^2 / 9 = 1) →
  (F1 = (5, 0) ∨ F1 = (-5, 0)) →
  (F2 = (if F1 = (5, 0) then (-5, 0) else (5, 0))) →
  ({p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1} A ∨ {p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1} B) →
  ((A = F1) ∨ (B = F1)) →
  (abs (F2.1 - A.1) + abs (F2.2 - A.2) + abs (F2.1 - B.1) + abs (F2.2 - B.2) = 12) →
  abs (A.1 - B.1) + abs (A.2 - B.2) = 8 :=
by
  sorry

end ellipse_AB_length_l214_214947


namespace p3_mp_odd_iff_m_even_l214_214983

theorem p3_mp_odd_iff_m_even (p m : ℕ) (hp : p % 2 = 1) : (p^3 + m * p) % 2 = 1 ↔ m % 2 = 0 := sorry

end p3_mp_odd_iff_m_even_l214_214983


namespace team_selection_l214_214043

-- Define the number of boys and girls in the club
def boys : Nat := 10
def girls : Nat := 12

-- Define the number of boys and girls to be selected for the team
def boys_team : Nat := 4
def girls_team : Nat := 4

-- Calculate the number of combinations using Nat.choose
noncomputable def choosing_boys : Nat := Nat.choose boys boys_team
noncomputable def choosing_girls : Nat := Nat.choose girls girls_team

-- Calculate the total number of ways to form the team
noncomputable def total_combinations : Nat := choosing_boys * choosing_girls

-- Theorem stating the total number of combinations equals the correct answer
theorem team_selection :
  total_combinations = 103950 := by
  sorry

end team_selection_l214_214043


namespace percent_of_x_l214_214313

variable (x : ℝ) (h : x > 0)

theorem percent_of_x (p : ℝ) : 
  (p * x = 0.21 * x + 10) → 
  p = 0.21 + 10 / x :=
sorry

end percent_of_x_l214_214313


namespace find_a_14_l214_214549

variable {α : Type} [LinearOrderedField α]

-- Define the arithmetic sequence sum formula
def arithmetic_seq_sum (a_1 d : α) (n : ℕ) : α :=
  n * a_1 + n * (n - 1) / 2 * d

-- Define the nth term of an arithmetic sequence
def arithmetic_seq_nth (a_1 d : α) (n : ℕ) : α :=
  a_1 + (n - 1 : ℕ) * d

theorem find_a_14
  (a_1 d : α)
  (h1 : arithmetic_seq_sum a_1 d 11 = 55)
  (h2 : arithmetic_seq_nth a_1 d 10 = 9) :
  arithmetic_seq_nth a_1 d 14 = 13 :=
by
  sorry

end find_a_14_l214_214549


namespace imaginary_part_of_z_l214_214201

noncomputable def i : ℂ := Complex.I
noncomputable def z : ℂ := i / (i - 1)

theorem imaginary_part_of_z : z.im = -1 / 2 := by
  sorry

end imaginary_part_of_z_l214_214201


namespace deck_of_1000_transformable_l214_214241

def shuffle (n : ℕ) (deck : List ℕ) : List ℕ :=
  -- Definition of the shuffle operation as described in the problem
  sorry

noncomputable def transformable_in_56_shuffles (n : ℕ) : Prop :=
  ∀ (initial final : List ℕ) (h₁ : initial.length = n) (h₂ : final.length = n),
  -- Prove that any initial arrangement can be transformed to any final arrangement in at most 56 shuffles
  sorry

theorem deck_of_1000_transformable : transformable_in_56_shuffles 1000 :=
  -- Implement the proof here
  sorry

end deck_of_1000_transformable_l214_214241


namespace sin_product_identity_l214_214943

noncomputable def sin_15_deg := Real.sin (15 * Real.pi / 180)
noncomputable def sin_30_deg := Real.sin (30 * Real.pi / 180)
noncomputable def sin_75_deg := Real.sin (75 * Real.pi / 180)

theorem sin_product_identity :
  sin_15_deg * sin_30_deg * sin_75_deg = 1 / 8 :=
by
  sorry

end sin_product_identity_l214_214943


namespace map_a_distance_map_b_distance_miles_map_b_distance_km_l214_214048

theorem map_a_distance (distance_cm : ℝ) (scale_cm : ℝ) (scale_km : ℝ) (actual_distance : ℝ) : 
  distance_cm = 80.5 → scale_cm = 0.6 → scale_km = 6.6 → actual_distance = (distance_cm * scale_km) / scale_cm → actual_distance = 885.5 :=
by
  intros h1 h2 h3 h4
  sorry

theorem map_b_distance_miles (distance_cm : ℝ) (scale_cm : ℝ) (scale_miles : ℝ) (actual_distance_miles : ℝ) : 
  distance_cm = 56.3 → scale_cm = 1.1 → scale_miles = 7.7 → actual_distance_miles = (distance_cm * scale_miles) / scale_cm → actual_distance_miles = 394.1 :=
by
  intros h1 h2 h3 h4
  sorry

theorem map_b_distance_km (distance_miles : ℝ) (conversion_factor : ℝ) (actual_distance_km : ℝ) :
  conversion_factor = 1.60934 → distance_miles = 394.1 → actual_distance_km = distance_miles * conversion_factor → actual_distance_km = 634.3 :=
by
  intros h1 h2 h3
  sorry

end map_a_distance_map_b_distance_miles_map_b_distance_km_l214_214048


namespace triangle_perpendicular_division_l214_214996

variable (a b c : ℝ)
variable (b_gt_c : b > c)
variable (triangle : True)

theorem triangle_perpendicular_division (a b c : ℝ) (b_gt_c : b > c) :
  let CK := (1 / 2) * Real.sqrt (a^2 + b^2 - c^2)
  CK = (1 / 2) * Real.sqrt (a^2 + b^2 - c^2) :=
by
  sorry

end triangle_perpendicular_division_l214_214996


namespace changfei_class_l214_214260

theorem changfei_class (m n : ℕ) (h : m * (m - 1) + m * n + n = 51) : m + n = 9 :=
sorry

end changfei_class_l214_214260


namespace smallest_integer_solution_system_of_inequalities_solution_l214_214326

-- Define the conditions and problem
variable (x : ℝ)

-- Part 1: Prove smallest integer solution for 5x + 15 > x - 1
theorem smallest_integer_solution :
  5 * x + 15 > x - 1 → x = -3 := sorry

-- Part 2: Prove solution set for system of inequalities
theorem system_of_inequalities_solution :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 4 * x) / 3 > x - 1) → (-4 < x ∧ x ≤ 1) := sorry

end smallest_integer_solution_system_of_inequalities_solution_l214_214326


namespace train_speed_is_72_kmph_l214_214044

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 112
noncomputable def crossing_time : ℝ := 11.099112071034318

theorem train_speed_is_72_kmph :
  let total_distance := train_length + bridge_length
  let speed_m_per_s := total_distance / crossing_time
  let speed_kmph := speed_m_per_s * 3.6
  speed_kmph = 72 :=
by
  sorry

end train_speed_is_72_kmph_l214_214044


namespace contrapositive_equivalence_l214_214120

-- Definitions based on the conditions
variables (R S : Prop)

-- Statement of the proof
theorem contrapositive_equivalence (h : ¬R → S) : ¬S → R := 
sorry

end contrapositive_equivalence_l214_214120


namespace charity_tickets_l214_214536

theorem charity_tickets (f h p : ℕ) (H1 : f + h = 140) (H2 : f * p + h * (p / 2) = 2001) : f * p = 782 := 
sorry

end charity_tickets_l214_214536


namespace cube_tetrahedron_volume_ratio_l214_214837

theorem cube_tetrahedron_volume_ratio :
  let s := 2
  let v1 := (0, 0, 0)
  let v2 := (2, 2, 0)
  let v3 := (2, 0, 2)
  let v4 := (0, 2, 2)
  let a := Real.sqrt 8 -- Side length of the tetrahedron
  let volume_tetra := (a^3 * Real.sqrt 2) / 12
  let volume_cube := s^3
  volume_cube / volume_tetra = 6 * Real.sqrt 2 := 
by
  -- Proof content skipped
  intros
  sorry

end cube_tetrahedron_volume_ratio_l214_214837


namespace average_of_six_numbers_l214_214804

theorem average_of_six_numbers (a b c d e f : ℝ)
  (h1 : (a + b) / 2 = 3.4)
  (h2 : (c + d) / 2 = 3.8)
  (h3 : (e + f) / 2 = 6.6) :
  (a + b + c + d + e + f) / 6 = 4.6 :=
by sorry

end average_of_six_numbers_l214_214804


namespace billing_error_l214_214193

theorem billing_error (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) 
    (h : 100 * y + x - (100 * x + y) = 2970) : y - x = 30 ∧ 10 ≤ x ∧ x ≤ 69 ∧ 40 ≤ y ∧ y ≤ 99 := 
by
  sorry

end billing_error_l214_214193


namespace largest_number_in_sequence_is_48_l214_214782

theorem largest_number_in_sequence_is_48 
    (a_1 a_2 a_3 a_4 a_5 a_6 : ℕ) 
    (h1 : 0 < a_1) 
    (h2 : a_1 < a_2 ∧ a_2 < a_3 ∧ a_3 < a_4 ∧ a_4 < a_5 ∧ a_5 < a_6)
    (h3 : ∃ k_1 k_2 k_3 k_4 k_5 : ℕ, k_1 > 1 ∧ k_2 > 1 ∧ k_3 > 1 ∧ k_4 > 1 ∧ k_5 > 1 ∧ 
          a_2 = k_1 * a_1 ∧ a_3 = k_2 * a_2 ∧ a_4 = k_3 * a_3 ∧ a_5 = k_4 * a_4 ∧ a_6 = k_5 * a_5)
    (h4 : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 79) 
    : a_6 = 48 := 
by 
    sorry

end largest_number_in_sequence_is_48_l214_214782


namespace sum_ends_in_zero_squares_end_same_digit_l214_214285

theorem sum_ends_in_zero_squares_end_same_digit (a b : ℕ) (h : (a + b) % 10 = 0) : (a^2 % 10) = (b^2 % 10) := 
sorry

end sum_ends_in_zero_squares_end_same_digit_l214_214285


namespace C_eq_D_at_n_l214_214133

noncomputable def C_n (n : ℕ) : ℝ := 768 * (1 - (1 / (3^n)))
noncomputable def D_n (n : ℕ) : ℝ := (4096 / 5) * (1 - ((-1)^n / (4^n)))
noncomputable def n_ge_1 : ℕ := 4

theorem C_eq_D_at_n : ∀ n ≥ 1, C_n n = D_n n → n = n_ge_1 :=
by
  intro n hn heq
  sorry

end C_eq_D_at_n_l214_214133


namespace new_supervisor_salary_correct_l214_214518

noncomputable def salary_new_supervisor
  (avg_salary_old : ℝ)
  (old_supervisor_salary : ℝ)
  (avg_salary_new : ℝ)
  (workers_count : ℝ)
  (total_salary_workers : ℝ := (avg_salary_old * (workers_count + 1)) - old_supervisor_salary)
  (new_supervisor_salary : ℝ := (avg_salary_new * (workers_count + 1)) - total_salary_workers)
  : ℝ :=
  new_supervisor_salary

theorem new_supervisor_salary_correct :
  salary_new_supervisor 430 870 420 8 = 780 :=
by
  simp [salary_new_supervisor]
  sorry

end new_supervisor_salary_correct_l214_214518


namespace problem1_problem2_l214_214767

open Classical

theorem problem1 (x : ℝ) : -x^2 + 4 * x - 4 < 0 ↔ x ≠ 2 :=
sorry

theorem problem2 (x : ℝ) : (1 - x) / (x - 5) > 0 ↔ 1 < x ∧ x < 5 :=
sorry

end problem1_problem2_l214_214767


namespace jackson_pays_2100_l214_214739

def tile_cost (length : ℝ) (width : ℝ) (tiles_per_sqft : ℝ) (percent_green : ℝ) (cost_green : ℝ) (cost_red : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * percent_green
  let red_tiles := total_tiles - green_tiles
  let cost_green_total := green_tiles * cost_green
  let cost_red_total := red_tiles * cost_red
  cost_green_total + cost_red_total

theorem jackson_pays_2100 :
  tile_cost 10 25 4 0.4 3 1.5 = 2100 :=
by
  sorry

end jackson_pays_2100_l214_214739


namespace proof_find_C_proof_find_cos_A_l214_214693

noncomputable def find_C {a b c : ℝ} {B : ℝ} (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) : Prop :=
  ∃ (C : ℝ), 0 < C ∧ C < Real.pi ∧ C = Real.pi / 6

noncomputable def find_cos_A {a b c : ℝ} {B : ℝ} (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) (h2 : Real.cos B = 2 / 3) : Prop :=
  ∃ (A : ℝ), Real.cos A = (Real.sqrt 5 - 2 * Real.sqrt 3) / 6

theorem proof_find_C (a b c B : ℝ) (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) : find_C h1 :=
  sorry

theorem proof_find_cos_A (a b c B : ℝ) (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) (h2 : Real.cos B = 2 / 3) : find_cos_A h1 h2 :=
  sorry

end proof_find_C_proof_find_cos_A_l214_214693


namespace find_value_l214_214296

-- Definitions of the curve and the line
def curve (a b : ℝ) (P : ℝ × ℝ) : Prop := (P.1*P.1) / a - (P.2*P.2) / b = 1
def line (P : ℝ × ℝ) : Prop := P.1 + P.2 - 1 = 0

-- Definition of the dot product condition
def dot_product_zero (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0

-- Theorem statement
theorem find_value (a b : ℝ) (P Q : ℝ × ℝ)
  (hc1 : curve a b P)
  (hc2 : curve a b Q)
  (hl1 : line P)
  (hl2 : line Q)
  (h_dot : dot_product_zero P Q) :
  1 / a - 1 / b = 2 :=
sorry

end find_value_l214_214296


namespace field_trip_students_l214_214162

theorem field_trip_students (bus_cost admission_per_student budget : ℕ) (students : ℕ)
  (h1 : bus_cost = 100)
  (h2 : admission_per_student = 10)
  (h3 : budget = 350)
  (total_cost : students * admission_per_student + bus_cost ≤ budget) : 
  students = 25 :=
by
  sorry

end field_trip_students_l214_214162


namespace number_of_candy_packages_l214_214504

theorem number_of_candy_packages (total_candies pieces_per_package : ℕ) 
  (h_total_candies : total_candies = 405)
  (h_pieces_per_package : pieces_per_package = 9) :
  total_candies / pieces_per_package = 45 := by
  sorry

end number_of_candy_packages_l214_214504


namespace min_value_expression_l214_214224

open Real

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_condition : a * b * c = 1) :
  a^2 + 8 * a * b + 32 * b^2 + 24 * b * c + 8 * c^2 ≥ 36 :=
by
  sorry

end min_value_expression_l214_214224


namespace room_length_l214_214760

/-- Define the conditions -/
def width : ℝ := 3.75
def cost_paving : ℝ := 6187.5
def cost_per_sqm : ℝ := 300

/-- Prove that the length of the room is 5.5 meters -/
theorem room_length : 
  (cost_paving / cost_per_sqm) / width = 5.5 :=
by
  sorry

end room_length_l214_214760


namespace frank_initial_mushrooms_l214_214111

theorem frank_initial_mushrooms (pounds_eaten pounds_left initial_pounds : ℕ) 
  (h1 : pounds_eaten = 8) 
  (h2 : pounds_left = 7) 
  (h3 : initial_pounds = pounds_eaten + pounds_left) : 
  initial_pounds = 15 := 
by 
  sorry

end frank_initial_mushrooms_l214_214111


namespace length_of_tangent_point_to_circle_l214_214502

theorem length_of_tangent_point_to_circle :
  let P := (2, 3)
  let O := (0, 0)
  let r := 1
  let OP := Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2)
  let tangent_length := Real.sqrt (OP^2 - r^2)
  tangent_length = 2 * Real.sqrt 3 := by
  sorry

end length_of_tangent_point_to_circle_l214_214502


namespace eggs_left_in_box_l214_214428

theorem eggs_left_in_box (initial_eggs : ℕ) (taken_eggs : ℕ) (remaining_eggs : ℕ) : 
  initial_eggs = 47 → taken_eggs = 5 → remaining_eggs = initial_eggs - taken_eggs → remaining_eggs = 42 :=
by
  sorry

end eggs_left_in_box_l214_214428


namespace relation_y₁_y₂_y₃_l214_214121

def parabola (x : ℝ) : ℝ := - x^2 - 2 * x + 2
noncomputable def y₁ : ℝ := parabola (-2)
noncomputable def y₂ : ℝ := parabola (1)
noncomputable def y₃ : ℝ := parabola (2)

theorem relation_y₁_y₂_y₃ : y₁ > y₂ ∧ y₂ > y₃ := by
  have h₁ : y₁ = 2 := by
    unfold y₁ parabola
    norm_num
    
  have h₂ : y₂ = -1 := by
    unfold y₂ parabola
    norm_num
    
  have h₃ : y₃ = -6 := by
    unfold y₃ parabola
    norm_num
    
  rw [h₁, h₂, h₃]
  exact ⟨by norm_num, by norm_num⟩

end relation_y₁_y₂_y₃_l214_214121


namespace eulers_formula_convex_polyhedron_l214_214626

theorem eulers_formula_convex_polyhedron :
  ∀ (V E F T H : ℕ),
  (V - E + F = 2) →
  (F = 24) →
  (E = (3 * T + 6 * H) / 2) →
  100 * H + 10 * T + V = 240 :=
by
  intros V E F T H h1 h2 h3
  sorry

end eulers_formula_convex_polyhedron_l214_214626


namespace calculate_expression_l214_214449

theorem calculate_expression : 1000 * 2.998 * 2.998 * 100 = (29980)^2 := 
by
  sorry

end calculate_expression_l214_214449


namespace toothpicks_15th_stage_l214_214462
-- Import the required library

-- Define the arithmetic sequence based on the provided conditions.
def toothpicks (n : ℕ) : ℕ :=
  if n = 1 then 5 else 3 * (n - 1) + 5

-- State the theorem
theorem toothpicks_15th_stage : toothpicks 15 = 47 :=
by {
  -- Provide the proof here, but currently using sorry as instructed
  sorry
}

end toothpicks_15th_stage_l214_214462


namespace first_discount_percentage_l214_214945

-- Definitions based on the conditions provided
def listed_price : ℝ := 400
def final_price : ℝ := 334.4
def additional_discount : ℝ := 5

-- The equation relating these quantities
theorem first_discount_percentage (D : ℝ) (h : listed_price * (1 - D / 100) * (1 - additional_discount / 100) = final_price) : D = 12 :=
sorry

end first_discount_percentage_l214_214945


namespace intersection_P_Q_l214_214757

def P : Set ℝ := {x : ℝ | x < 1}
def Q : Set ℝ := {x : ℝ | x^2 < 4}

theorem intersection_P_Q : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := 
  sorry

end intersection_P_Q_l214_214757


namespace find_k_l214_214100

-- Define the variables and conditions
variables (x y k : ℤ)

-- State the theorem
theorem find_k (h1 : x = 2) (h2 : y = 1) (h3 : k * x - y = 3) : k = 2 :=
sorry

end find_k_l214_214100


namespace expression_value_range_l214_214454

theorem expression_value_range (a b c d e : ℝ) (h₁ : 0 ≤ a) (h₂ : a ≤ 1) (h₃ : 0 ≤ b) (h₄ : b ≤ 1) (h₅ : 0 ≤ c) (h₆ : c ≤ 1) (h₇ : 0 ≤ d) (h₈ : d ≤ 1) (h₉ : 0 ≤ e) (h₁₀ : e ≤ 1) :
  4 * Real.sqrt (2 / 3) ≤ (Real.sqrt (a^2 + (1 - b)^2 + e^2) + Real.sqrt (b^2 + (1 - c)^2 + e^2) + Real.sqrt (c^2 + (1 - d)^2 + e^2) + Real.sqrt (d^2 + (1 - a)^2 + e^2)) ∧ 
  (Real.sqrt (a^2 + (1 - b)^2 + e^2) + Real.sqrt (b^2 + (1 - c)^2 + e^2) + Real.sqrt (c^2 + (1 - d)^2 + e^2) + Real.sqrt (d^2 + (1 - a)^2 + e^2)) ≤ 8 :=
sorry

end expression_value_range_l214_214454


namespace max_rectangles_in_triangle_l214_214832

theorem max_rectangles_in_triangle : 
  (∃ (n : ℕ), n = 192 ∧ 
  ∀ (i j : ℕ), i + j < 7 → ∀ (a b : ℕ), a ≤ 6 - i ∧ b ≤ 6 - j → 
  ∃ (rectangles : ℕ), rectangles = (6 - i) * (6 - j)) :=
sorry

end max_rectangles_in_triangle_l214_214832


namespace inclination_angle_of_focal_chord_l214_214052

theorem inclination_angle_of_focal_chord
  (p : ℝ)
  (h_parabola : ∀ x y : ℝ, y^2 = 2 * p * x → True)
  (h_focal_chord_length : ∀ A B : ℝ, |A - B| = 8 * p → True) :
  ∃ θ : ℝ, (θ = π / 6 ∨ θ = 5 * π / 6) :=
by
  sorry

end inclination_angle_of_focal_chord_l214_214052


namespace solution_to_prime_equation_l214_214179

theorem solution_to_prime_equation (x y : ℕ) (p : ℕ) (h1 : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  x^3 + y^3 = p * (xy + p) ↔ (x = 8 ∧ y = 1 ∧ p = 19) ∨ (x = 1 ∧ y = 8 ∧ p = 19) ∨ 
              (x = 7 ∧ y = 2 ∧ p = 13) ∨ (x = 2 ∧ y = 7 ∧ p = 13) ∨ 
              (x = 5 ∧ y = 4 ∧ p = 7) ∨ (x = 4 ∧ y = 5 ∧ p = 7) := sorry

end solution_to_prime_equation_l214_214179


namespace ratio_of_height_to_radius_max_volume_l214_214440

theorem ratio_of_height_to_radius_max_volume (r h : ℝ) (h_surface_area : 2 * Real.pi * r^2 + 2 * Real.pi * r * h = 6 * Real.pi) :
  (exists (max_r : ℝ) (max_h : ℝ), 2 * r * max_r + 2 * r * max_h = 6 * Real.pi ∧ 
                                  max_r = 1 ∧ 
                                  max_h = 2 ∧ 
                                  (max_h / max_r) = 2) :=
by
  sorry

end ratio_of_height_to_radius_max_volume_l214_214440


namespace proof_problem_l214_214041

-- Necessary types and noncomputable definitions
noncomputable def a_seq : ℕ → ℕ := sorry
noncomputable def b_seq : ℕ → ℕ := sorry

-- The conditions in the problem are used as assumptions
axiom partition : ∀ (n : ℕ), n > 0 → a_seq n < a_seq (n + 1)
axiom b_def : ∀ (n : ℕ), n > 0 → b_seq n = a_seq n + n

-- The mathematical equivalent proof problem stated
theorem proof_problem (n : ℕ) (hn : n > 0) : a_seq n + b_seq n = a_seq (b_seq n) :=
sorry

end proof_problem_l214_214041


namespace sum_of_inserted_numbers_in_arithmetic_sequence_l214_214292

theorem sum_of_inserted_numbers_in_arithmetic_sequence :
  ∃ a2 a3 : ℤ, 2015 > a2 ∧ a2 > a3 ∧ a3 > 131 ∧ (2015 - a2) = (a2 - a3) ∧ (a2 - a3) = (a3 - 131) ∧ (a2 + a3) = 2146 := 
by
  sorry

end sum_of_inserted_numbers_in_arithmetic_sequence_l214_214292


namespace wendy_third_day_miles_l214_214523

theorem wendy_third_day_miles (miles_day1 miles_day2 total_miles : ℕ)
  (h1 : miles_day1 = 125)
  (h2 : miles_day2 = 223)
  (h3 : total_miles = 493) :
  total_miles - (miles_day1 + miles_day2) = 145 :=
by sorry

end wendy_third_day_miles_l214_214523


namespace range_of_a_l214_214364

theorem range_of_a (a : ℝ) : 
  ((-1 + a) ^ 2 + (-1 - a) ^ 2 < 4) ↔ (-1 < a ∧ a < 1) := 
by
  sorry

end range_of_a_l214_214364


namespace arithmetic_geometric_sequence_fraction_l214_214559

theorem arithmetic_geometric_sequence_fraction 
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : a1 + a2 = 10)
  (h2 : 1 * b3 = 9)
  (h3 : b2 ^ 2 = 9) : 
  b2 / (a1 + a2) = 3 / 10 := 
by 
  sorry

end arithmetic_geometric_sequence_fraction_l214_214559


namespace cosine_relationship_l214_214924

open Real

noncomputable def functional_relationship (x y : ℝ) : Prop :=
  y = -(4 / 5) * sqrt (1 - x ^ 2) + (3 / 5) * x

theorem cosine_relationship (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2)
  (h5 : cos (α + β) = - 4 / 5) (h6 : sin β = x) (h7 : cos α = y) (h8 : 4 / 5 < x) (h9 : x < 1) :
  functional_relationship x y :=
sorry

end cosine_relationship_l214_214924


namespace second_discount_percentage_l214_214056

/-- 
  Given:
  - The listed price of Rs. 560.
  - The final sale price after successive discounts of 20% and another discount is Rs. 313.6.
  Prove:
  - The second discount percentage is 30%.
-/
theorem second_discount_percentage (list_price final_price : ℝ) (first_discount_percentage : ℝ) : 
  list_price = 560 → 
  final_price = 313.6 → 
  first_discount_percentage = 20 → 
  ∃ (second_discount_percentage : ℝ), second_discount_percentage = 30 :=
by
  sorry

end second_discount_percentage_l214_214056


namespace log_sqrt_defined_l214_214403

open Real

-- Define the conditions for the logarithm and square root arguments
def log_condition (x : ℝ) : Prop := 4 * x - 7 > 0
def sqrt_condition (x : ℝ) : Prop := 2 * x - 3 ≥ 0

-- Define the combined condition
def combined_condition (x : ℝ) : Prop := x > 7 / 4

-- The proof statement
theorem log_sqrt_defined (x : ℝ) : combined_condition x ↔ log_condition x ∧ sqrt_condition x :=
by
  -- Work through the equivalence and proof steps
  sorry

end log_sqrt_defined_l214_214403


namespace at_least_one_not_beyond_20m_l214_214665

variables (p q : Prop)

theorem at_least_one_not_beyond_20m : (¬ p ∨ ¬ q) ↔ ¬ (p ∧ q) :=
by sorry

end at_least_one_not_beyond_20m_l214_214665


namespace rebus_solution_l214_214793

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l214_214793


namespace div_polynomial_not_div_l214_214279

theorem div_polynomial_not_div (n : ℕ) : ¬ (n + 2) ∣ (n^3 - 2 * n^2 - 5 * n + 7) := by
  sorry

end div_polynomial_not_div_l214_214279


namespace basketball_team_selection_l214_214986

noncomputable def count_ways_excluding_twins (n k : ℕ) : ℕ :=
  let total_ways := Nat.choose n k
  let exhaustive_cases := Nat.choose (n - 2) (k - 2)
  total_ways - exhaustive_cases

theorem basketball_team_selection :
  count_ways_excluding_twins 12 5 = 672 :=
by
  sorry

end basketball_team_selection_l214_214986


namespace surface_area_of_second_cube_l214_214064

theorem surface_area_of_second_cube (V1 V2: ℝ) (a2: ℝ):
  (V1 = 16 ∧ V2 = 4 * V1 ∧ a2 = (V2)^(1/3)) → 6 * a2^2 = 96 :=
by intros h; sorry

end surface_area_of_second_cube_l214_214064


namespace rooster_count_l214_214383

theorem rooster_count (total_chickens hens roosters : ℕ) 
  (h1 : total_chickens = roosters + hens)
  (h2 : roosters = 2 * hens)
  (h3 : total_chickens = 9000) 
  : roosters = 6000 := 
by
  sorry

end rooster_count_l214_214383


namespace fred_has_18_stickers_l214_214969

def jerry_stickers := 36
def george_stickers (jerry : ℕ) := jerry / 3
def fred_stickers (george : ℕ) := george + 6

theorem fred_has_18_stickers :
  let j := jerry_stickers
  let g := george_stickers j 
  fred_stickers g = 18 :=
by
  sorry

end fred_has_18_stickers_l214_214969


namespace range_of_m_three_zeros_l214_214614

theorem range_of_m_three_zeros (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (x^3 - 3*x + m = 0) ∧ (y^3 - 3*y + m = 0) ∧ (z^3 - 3*z + m = 0)) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end range_of_m_three_zeros_l214_214614


namespace initial_pigs_count_l214_214235

theorem initial_pigs_count (P : ℕ) (h1 : 2 + P + 6 + 3 + 5 + 2 = 21) : P = 3 :=
by
  sorry

end initial_pigs_count_l214_214235


namespace nina_money_proof_l214_214788

def total_money_nina_has (W M : ℝ) : Prop :=
  (10 * W = M) ∧ (14 * (W - 1.75) = M)

theorem nina_money_proof (W M : ℝ) (h : total_money_nina_has W M) : M = 61.25 :=
by 
  sorry

end nina_money_proof_l214_214788


namespace number_of_children_bikes_l214_214675

theorem number_of_children_bikes (c : ℕ) 
  (regular_bikes : ℕ) (wheels_per_regular_bike : ℕ) 
  (wheels_per_children_bike : ℕ) (total_wheels : ℕ)
  (h1 : regular_bikes = 7) 
  (h2 : wheels_per_regular_bike = 2) 
  (h3 : wheels_per_children_bike = 4) 
  (h4 : total_wheels = 58) 
  (h5 : total_wheels = (regular_bikes * wheels_per_regular_bike) + (c * wheels_per_children_bike)) 
  : c = 11 :=
by
  sorry

end number_of_children_bikes_l214_214675


namespace percentage_neither_bp_nor_ht_l214_214319

noncomputable def percentage_teachers_neither_condition (total: ℕ) (high_bp: ℕ) (heart_trouble: ℕ) (both: ℕ) : ℚ :=
  let either_condition := high_bp + heart_trouble - both
  let neither_condition := total - either_condition
  (neither_condition * 100 : ℚ) / total

theorem percentage_neither_bp_nor_ht :
  percentage_teachers_neither_condition 150 90 50 30 = 26.67 :=
by
  sorry

end percentage_neither_bp_nor_ht_l214_214319


namespace probability_no_defective_pens_l214_214639

theorem probability_no_defective_pens
  (total_pens : ℕ) (defective_pens : ℕ) (non_defective_pens : ℕ) (prob_first_non_defective : ℚ) (prob_second_non_defective : ℚ) :
  total_pens = 12 →
  defective_pens = 4 →
  non_defective_pens = total_pens - defective_pens →
  prob_first_non_defective = non_defective_pens / total_pens →
  prob_second_non_defective = (non_defective_pens - 1) / (total_pens - 1) →
  prob_first_non_defective * prob_second_non_defective = 14 / 33 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  sorry

end probability_no_defective_pens_l214_214639


namespace current_ratio_of_employees_l214_214031

-- Definitions for the number of current male employees and the ratio if 3 more men are hired
variables (M : ℕ) (F : ℕ)
variables (hM : M = 189)
variables (ratio_hired : (M + 3) / F = 8 / 9)

-- Conclusion we want to prove
theorem current_ratio_of_employees (M F : ℕ) (hM : M = 189) (ratio_hired : (M + 3) / F = 8 / 9) : 
  M / F = 7 / 8 :=
sorry

end current_ratio_of_employees_l214_214031


namespace B_work_rate_l214_214736

-- Definitions for the conditions
def A (t : ℝ) := 1 / 15 -- A's work rate per hour
noncomputable def B : ℝ := 1 / 10 - 1 / 15 -- Definition using the condition of the combined work rate

-- Lean 4 statement for the proof problem
theorem B_work_rate : B = 1 / 30 := by sorry

end B_work_rate_l214_214736


namespace breadth_of_hall_l214_214361

/-- Given a hall of length 20 meters and a uniform verandah width of 2.5 meters,
    with a cost of Rs. 700 for flooring the verandah at Rs. 3.50 per square meter,
    prove that the breadth of the hall is 15 meters. -/
theorem breadth_of_hall (h_length : ℝ) (v_width : ℝ) (cost : ℝ) (rate : ℝ) (b : ℝ) :
  h_length = 20 ∧ v_width = 2.5 ∧ cost = 700 ∧ rate = 3.50 →
  25 * (b + 5) - 20 * b = 200 →
  b = 15 :=
by
  intros hc ha
  sorry

end breadth_of_hall_l214_214361


namespace remainder_equivalence_l214_214709

theorem remainder_equivalence (x : ℕ) (r : ℕ) (hx_pos : 0 < x) 
  (h1 : ∃ q1, 100 = q1 * x + r) (h2 : ∃ q2, 197 = q2 * x + r) : 
  r = 3 :=
by
  sorry

end remainder_equivalence_l214_214709


namespace joe_total_cars_l214_214692

def initial_cars := 50
def multiplier := 3

theorem joe_total_cars : initial_cars + (multiplier * initial_cars) = 200 := by
  sorry

end joe_total_cars_l214_214692


namespace max_min_rounded_value_l214_214039

theorem max_min_rounded_value (n : ℝ) (h : 3.75 ≤ n ∧ n < 3.85) : 
  (∀ n, 3.75 ≤ n ∧ n < 3.85 → n ≤ 3.84 ∧ n ≥ 3.75) :=
sorry

end max_min_rounded_value_l214_214039


namespace a_sufficient_not_necessary_l214_214087

theorem a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (¬(1 / a < 1 → a > 1)) :=
by
  sorry

end a_sufficient_not_necessary_l214_214087


namespace good_students_l214_214638

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l214_214638


namespace average_price_per_dvd_l214_214315

-- Define the conditions
def num_movies_box1 : ℕ := 10
def price_per_movie_box1 : ℕ := 2
def num_movies_box2 : ℕ := 5
def price_per_movie_box2 : ℕ := 5

-- Define total calculations based on conditions
def total_cost_box1 : ℕ := num_movies_box1 * price_per_movie_box1
def total_cost_box2 : ℕ := num_movies_box2 * price_per_movie_box2

def total_cost : ℕ := total_cost_box1 + total_cost_box2
def total_movies : ℕ := num_movies_box1 + num_movies_box2

-- Define the average price per DVD and prove it to be 3
theorem average_price_per_dvd : total_cost / total_movies = 3 := by
  sorry

end average_price_per_dvd_l214_214315


namespace total_votes_cast_l214_214419

theorem total_votes_cast (V : ℝ) (h1 : V > 0) (h2 : 0.35 * V = candidate_votes) (h3 : candidate_votes + 2400 = rival_votes) (h4 : candidate_votes + rival_votes = V) : V = 8000 := 
by
  sorry

end total_votes_cast_l214_214419


namespace max_n_l214_214610

def sum_first_n_terms (S n : ℕ) (a : ℕ → ℕ) : Prop :=
  S = 2 * a n - n

theorem max_n (S : ℕ) (a : ℕ → ℕ) :
  (∀ n, sum_first_n_terms S n a) → ∀ n, (2 ^ n - 1 ≤ 10 * n) → n ≤ 5 :=
by
  sorry

end max_n_l214_214610


namespace housing_price_equation_l214_214083

-- Initial conditions
def january_price : ℝ := 8300
def march_price : ℝ := 8700
variables (x : ℝ)

-- Lean statement of the problem
theorem housing_price_equation :
  january_price * (1 + x)^2 = march_price := 
sorry

end housing_price_equation_l214_214083


namespace parallel_tangent_line_l214_214673

theorem parallel_tangent_line (b : ℝ) :
  (∃ b : ℝ, (∀ x y : ℝ, x + 2 * y + b = 0 → (x^2 + y^2 = 5))) →
  (b = 5 ∨ b = -5) :=
by
  sorry

end parallel_tangent_line_l214_214673


namespace problem1_problem2_l214_214135

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a + b + c = 1
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a + b + c = 1)

-- Problem 1: Prove that a^2 / b + b^2 / c + c^2 / a ≥ 1
theorem problem1 : a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
by sorry

-- Problem 2: Prove that ab + bc + ca ≤ 1 / 3
theorem problem2 : ab + bc + ca ≤ 1 / 3 :=
by sorry

end problem1_problem2_l214_214135


namespace baby_turtles_on_sand_l214_214845

theorem baby_turtles_on_sand (total_swept : ℕ) (total_hatched : ℕ) (h1 : total_hatched = 42) (h2 : total_swept = total_hatched / 3) :
  total_hatched - total_swept = 28 := by
  sorry

end baby_turtles_on_sand_l214_214845


namespace sum_of_x_y_l214_214460

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l214_214460


namespace quadratic_inequality_solution_set_l214_214054

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | ax^2 - (2 + a) * x + 2 > 0} = {x | 2 / a < x ∧ x < 1} :=
sorry

end quadratic_inequality_solution_set_l214_214054


namespace eyes_per_ant_proof_l214_214882

noncomputable def eyes_per_ant (s a e_s E : ℕ) : ℕ :=
  let e_spiders := s * e_s
  let e_ants := E - e_spiders
  e_ants / a

theorem eyes_per_ant_proof : eyes_per_ant 3 50 8 124 = 2 :=
by
  sorry

end eyes_per_ant_proof_l214_214882


namespace fg_of_3_is_2810_l214_214717

def f (x : ℕ) : ℕ := x^2 + 1
def g (x : ℕ) : ℕ := 2 * x^3 - 1

theorem fg_of_3_is_2810 : f (g 3) = 2810 := by
  sorry

end fg_of_3_is_2810_l214_214717


namespace savings_percentage_l214_214185

theorem savings_percentage
  (S : ℝ)
  (last_year_saved : ℝ := 0.06 * S)
  (this_year_salary : ℝ := 1.10 * S)
  (this_year_saved : ℝ := 0.10 * this_year_salary)
  (ratio := this_year_saved / last_year_saved * 100):
  ratio = 183.33 := 
sorry

end savings_percentage_l214_214185


namespace avg_marks_second_class_l214_214585

theorem avg_marks_second_class
  (x : ℝ)
  (avg_class1 : ℝ)
  (avg_total : ℝ)
  (n1 n2 : ℕ)
  (h1 : n1 = 30)
  (h2 : n2 = 50)
  (h3 : avg_class1 = 30)
  (h4: avg_total = 48.75)
  (h5 : (n1 * avg_class1 + n2 * x) / (n1 + n2) = avg_total) :
  x = 60 := by
  sorry

end avg_marks_second_class_l214_214585


namespace value_of_72_a_in_terms_of_m_and_n_l214_214394

theorem value_of_72_a_in_terms_of_m_and_n (a m n : ℝ) (hm : 2^a = m) (hn : 3^a = n) :
  72^a = m^3 * n^2 :=
by sorry

end value_of_72_a_in_terms_of_m_and_n_l214_214394


namespace students_standing_together_l214_214987

theorem students_standing_together (s : Finset ℕ) (h_size : s.card = 6) (a b : ℕ) (h_ab : a ∈ s ∧ b ∈ s) (h_ab_together : ∃ (l : List ℕ), l.length = 6 ∧ a :: b :: l = l):
  ∃ (arrangements : ℕ), arrangements = 240 := by
  sorry

end students_standing_together_l214_214987


namespace inequality_always_true_l214_214978

theorem inequality_always_true (x : ℝ) : (4 * x) / (x ^ 2 + 4) ≤ 1 := by
  sorry

end inequality_always_true_l214_214978


namespace relationship_of_y_values_l214_214712

theorem relationship_of_y_values (m n y1 y2 y3 : ℝ) (h1 : m < 0) (h2 : n > 0) 
  (hA : y1 = m * (-2) + n) (hB : y2 = m * (-3) + n) (hC : y3 = m * 1 + n) :
  y3 < y1 ∧ y1 < y2 := 
by 
  sorry

end relationship_of_y_values_l214_214712


namespace magician_method_N_2k_magician_method_values_l214_214416

-- (a) Prove that if there is a method for N = k, then there is a method for N = 2k.
theorem magician_method_N_2k (k : ℕ) (method_k : Prop) : 
  (∃ method_N_k : Prop, method_k → method_N_k) → 
  (∃ method_N_2k : Prop, method_k → method_N_2k) :=
sorry

-- (b) Find all values of N for which the magician and the assistant have a method.
theorem magician_method_values (N : ℕ) : 
  (∃ method : Prop, method) ↔ (∃ m : ℕ, N = 2^m) :=
sorry

end magician_method_N_2k_magician_method_values_l214_214416


namespace sam_distance_traveled_l214_214809

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l214_214809


namespace quadratic_complete_square_l214_214494

theorem quadratic_complete_square (a b c : ℝ) :
  (8*x^2 - 48*x - 288) = a*(x + b)^2 + c → a + b + c = -355 := 
  by
  sorry

end quadratic_complete_square_l214_214494


namespace complex_number_properties_l214_214314

theorem complex_number_properties (z : ℂ) (h : z^2 = 3 + 4 * Complex.I) : 
  (z.im = 1 ∨ z.im = -1) ∧ Complex.abs z = Real.sqrt 5 := 
by
  sorry

end complex_number_properties_l214_214314


namespace calc_price_per_litre_l214_214038

noncomputable def pricePerLitre (initial final totalCost : ℝ) : ℝ :=
  totalCost / (final - initial)

theorem calc_price_per_litre :
  pricePerLitre 10 50 36.60 = 91.5 :=
by
  sorry

end calc_price_per_litre_l214_214038


namespace radius_for_visibility_l214_214368

def is_concentric (hex_center : ℝ × ℝ) (circle_center : ℝ × ℝ) : Prop :=
  hex_center = circle_center

def regular_hexagon (side_length : ℝ) : Prop :=
  side_length = 3

theorem radius_for_visibility
  (r : ℝ)
  (hex_center : ℝ × ℝ)
  (circle_center : ℝ × ℝ)
  (P_visible: ℝ)
  (prob_Four_sides_visible: ℝ ) :
  is_concentric hex_center circle_center →
  regular_hexagon 3 →
  prob_Four_sides_visible = 1 / 3 →
  P_visible = 4 →
  r = 2.6 :=
by sorry

end radius_for_visibility_l214_214368


namespace largest_n_binary_operation_l214_214850

-- Define the binary operation @
def binary_operation (n : ℤ) : ℤ := n - (n * 5)

-- Define the theorem stating the desired property
theorem largest_n_binary_operation (x : ℤ) (h : x > -8) :
  ∃ (n : ℤ), n = 2 ∧ binary_operation n < x :=
sorry

end largest_n_binary_operation_l214_214850


namespace total_time_to_complete_work_l214_214088

noncomputable def mahesh_work_rate (W : ℕ) := W / 40
noncomputable def mahesh_work_done_in_20_days (W : ℕ) := 20 * (mahesh_work_rate W)
noncomputable def remaining_work (W : ℕ) := W - (mahesh_work_done_in_20_days W)
noncomputable def rajesh_work_rate (W : ℕ) := (remaining_work W) / 30

theorem total_time_to_complete_work (W : ℕ) :
    (mahesh_work_rate W) + (rajesh_work_rate W) = W / 24 →
    (mahesh_work_done_in_20_days W) = W / 2 →
    (remaining_work W) = W / 2 →
    (rajesh_work_rate W) = W / 60 →
    20 + 30 = 50 :=
by 
  intros _ _ _ _
  sorry

end total_time_to_complete_work_l214_214088


namespace polynomial_divisibility_l214_214737

theorem polynomial_divisibility (p q : ℝ) :
    (∀ x, x = -2 ∨ x = 3 → (x^6 - x^5 + x^4 - p*x^3 + q*x^2 - 7*x - 35) = 0) →
    (p, q) = (6.86, -36.21) :=
by
  sorry

end polynomial_divisibility_l214_214737


namespace part1_exists_n_part2_not_exists_n_l214_214507

open Nat

def is_prime (p : Nat) : Prop := p > 1 ∧ ∀ m : Nat, m ∣ p → m = 1 ∨ m = p

-- Part 1: Prove there exists an n such that n-96, n, n+96 are all primes
theorem part1_exists_n :
  ∃ (n : Nat), is_prime (n - 96) ∧ is_prime n ∧ is_prime (n + 96) :=
sorry

-- Part 2: Prove there does not exist an n such that n-1996, n, n+1996 are all primes
theorem part2_not_exists_n :
  ¬ (∃ (n : Nat), is_prime (n - 1996) ∧ is_prime n ∧ is_prime (n + 1996)) :=
sorry

end part1_exists_n_part2_not_exists_n_l214_214507


namespace profit_percentage_is_correct_l214_214444

noncomputable def cost_price : ℝ := 47.50
noncomputable def selling_price : ℝ := 65.97
noncomputable def list_price := selling_price / 0.90
noncomputable def profit := selling_price - cost_price
noncomputable def profit_percentage := (profit / cost_price) * 100

theorem profit_percentage_is_correct : profit_percentage = 38.88 := by
  sorry

end profit_percentage_is_correct_l214_214444


namespace scientific_notation_gdp_l214_214567

theorem scientific_notation_gdp :
  8837000000 = 8.837 * 10^9 := 
by
  sorry

end scientific_notation_gdp_l214_214567


namespace samantha_total_cost_l214_214719

-- Defining the conditions in Lean
def washer_cost : ℕ := 4
def dryer_cost_per_10_min : ℕ := 25
def loads : ℕ := 2
def num_dryers : ℕ := 3
def dryer_time : ℕ := 40

-- Proving the total cost Samantha spends is $11
theorem samantha_total_cost : (loads * washer_cost + num_dryers * (dryer_time / 10 * dryer_cost_per_10_min)) = 1100 :=
by
  sorry

end samantha_total_cost_l214_214719


namespace amanda_needs_how_many_bags_of_grass_seeds_l214_214655

theorem amanda_needs_how_many_bags_of_grass_seeds
    (lot_length : ℕ := 120)
    (lot_width : ℕ := 60)
    (concrete_length : ℕ := 40)
    (concrete_width : ℕ := 40)
    (bag_coverage : ℕ := 56) :
    (lot_length * lot_width - concrete_length * concrete_width) / bag_coverage = 100 := by
  sorry

end amanda_needs_how_many_bags_of_grass_seeds_l214_214655


namespace multiply_exponents_l214_214558

theorem multiply_exponents (x : ℝ) : (x^2) * (x^3) = x^5 := 
sorry

end multiply_exponents_l214_214558


namespace snail_kite_first_day_snails_l214_214109

theorem snail_kite_first_day_snails (x : ℕ) 
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 35) : 
  x = 3 :=
sorry

end snail_kite_first_day_snails_l214_214109


namespace percent_not_covering_politics_l214_214152

-- Definitions based on the conditions
def total_reporters : ℕ := 100
def local_politics_reporters : ℕ := 28
def percent_cover_local_politics : ℚ := 0.7

-- To be proved
theorem percent_not_covering_politics :
  let politics_reporters := local_politics_reporters / percent_cover_local_politics 
  (total_reporters - politics_reporters) / total_reporters = 0.6 := 
by
  sorry

end percent_not_covering_politics_l214_214152


namespace no_2007_in_display_can_2008_appear_in_display_l214_214562

-- Definitions of the operations as functions on the display number.
def button1 (n : ℕ) : ℕ := 1
def button2 (n : ℕ) : ℕ := if n % 2 = 0 then n / 2 else n
def button3 (n : ℕ) : ℕ := if n >= 3 then n - 3 else n
def button4 (n : ℕ) : ℕ := 4 * n

-- Initial condition
def initial_display : ℕ := 0

-- Define can_appear as a recursive function to determine if a number can appear on the display.
def can_appear (target : ℕ) : Prop :=
  ∃ n : ℕ, n = target ∧ (∃ f : (ℕ → ℕ) → ℕ, f initial_display = target)

-- Prove the statements:
theorem no_2007_in_display : ¬ can_appear 2007 :=
  sorry

theorem can_2008_appear_in_display : can_appear 2008 :=
  sorry

end no_2007_in_display_can_2008_appear_in_display_l214_214562


namespace difference_in_spectators_l214_214370

-- Define the parameters given in the problem
def people_game_2 : ℕ := 80
def people_game_1 : ℕ := people_game_2 - 20
def people_game_3 : ℕ := people_game_2 + 15
def people_last_week : ℕ := 200

-- Total people who watched the games this week
def people_this_week : ℕ := people_game_1 + people_game_2 + people_game_3

-- Theorem statement: Prove the difference in people watching the games between this week and last week is 35.
theorem difference_in_spectators : people_this_week - people_last_week = 35 :=
  sorry

end difference_in_spectators_l214_214370


namespace find_c_l214_214948

theorem find_c (c : ℝ) (h : ∃ a : ℝ, x^2 - 50 * x + c = (x - a)^2) : c = 625 :=
  by
  sorry

end find_c_l214_214948


namespace triangle_equilateral_l214_214342

noncomputable def is_equilateral (a b c : ℝ) (A B C : ℝ) : Prop :=
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c

theorem triangle_equilateral (A B C a b c : ℝ) (hB : B = 60) (hb : b^2 = a * c) :
  is_equilateral a b c A B C :=
by
  sorry

end triangle_equilateral_l214_214342


namespace candy_sold_tuesday_correct_l214_214448

variable (pieces_sold_monday pieces_left_by_wednesday initial_candy total_pieces_sold : ℕ)
variable (pieces_sold_tuesday : ℕ)

-- Conditions
def initial_candy_amount := 80
def candy_sold_on_monday := 15
def candy_left_by_wednesday := 7

-- Total candy sold by Wednesday
def total_candy_sold_by_wednesday := initial_candy_amount - candy_left_by_wednesday

-- Candy sold on Tuesday
def candy_sold_on_tuesday : ℕ := total_candy_sold_by_wednesday - candy_sold_on_monday

-- Proof statement
theorem candy_sold_tuesday_correct : candy_sold_on_tuesday = 58 := sorry

end candy_sold_tuesday_correct_l214_214448


namespace identify_incorrect_calculation_l214_214303

theorem identify_incorrect_calculation : 
  (∀ x : ℝ, x^2 * x^3 = x^5) ∧ 
  (∀ x : ℝ, x^3 + x^3 = 2 * x^3) ∧ 
  (∀ x : ℝ, x^6 / x^2 = x^4) ∧ 
  ¬ (∀ x : ℝ, (-3 * x)^2 = 6 * x^2) := 
by
  sorry

end identify_incorrect_calculation_l214_214303


namespace window_width_correct_l214_214897

def total_width_window (x : ℝ) : ℝ :=
  let pane_width := 4 * x
  let num_panes_per_row := 4
  let num_borders := 5
  num_panes_per_row * pane_width + num_borders * 3

theorem window_width_correct (x : ℝ) :
  total_width_window x = 16 * x + 15 := sorry

end window_width_correct_l214_214897


namespace abs_difference_l214_214250

theorem abs_difference (a b : ℝ) (h₁ : a * b = 9) (h₂ : a + b = 10) : |a - b| = 8 :=
sorry

end abs_difference_l214_214250


namespace solution_set_of_inequality_l214_214273

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_def : ∀ x : ℝ, x ≤ 0 → f x = x^2 + 2 * x) :
  {x : ℝ | f (x + 2) < 3} = {x : ℝ | -5 < x ∧ x < 1} :=
by sorry

end solution_set_of_inequality_l214_214273


namespace minimum_adjacent_white_pairs_l214_214621

theorem minimum_adjacent_white_pairs (total_black_cells : ℕ) (grid_size : ℕ) (total_pairs : ℕ) : 
  total_black_cells = 20 ∧ grid_size = 8 ∧ total_pairs = 112 → ∃ min_white_pairs : ℕ, min_white_pairs = 34 :=
by
  sorry

end minimum_adjacent_white_pairs_l214_214621


namespace sum_of_perimeters_l214_214214

theorem sum_of_perimeters (x y : ℝ) (h₁ : x^2 + y^2 = 125) (h₂ : x^2 - y^2 = 65) : 4 * x + 4 * y = 60 := 
by
  sorry

end sum_of_perimeters_l214_214214


namespace quadrilateral_is_trapezium_l214_214119

-- Define the angles of the quadrilateral and the sum of the angles condition
variables {x : ℝ}
def sum_of_angles (x : ℝ) : Prop := x + 5 * x + 2 * x + 4 * x = 360

-- State the theorem
theorem quadrilateral_is_trapezium (x : ℝ) (h : sum_of_angles x) : 
  30 + 150 = 180 ∧ 60 + 120 = 180 → is_trapezium :=
sorry

end quadrilateral_is_trapezium_l214_214119


namespace debby_drink_days_l214_214992

theorem debby_drink_days :
  ∀ (total_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ),
  total_bottles = 301 →
  bottles_per_day = 144 →
  remaining_bottles = 157 →
  (total_bottles - remaining_bottles) / bottles_per_day = 1 :=
by
  intros total_bottles bottles_per_day remaining_bottles ht he hb
  sorry

end debby_drink_days_l214_214992


namespace more_likely_second_machine_l214_214565

variable (P_B1 : ℝ := 0.8) -- Probability that a part is from the first machine
variable (P_B2 : ℝ := 0.2) -- Probability that a part is from the second machine
variable (P_A_given_B1 : ℝ := 0.01) -- Probability that a part is defective given it is from the first machine
variable (P_A_given_B2 : ℝ := 0.05) -- Probability that a part is defective given it is from the second machine

noncomputable def P_A : ℝ :=
  P_B1 * P_A_given_B1 + P_B2 * P_A_given_B2

noncomputable def P_B1_given_A : ℝ :=
  (P_B1 * P_A_given_B1) / P_A

noncomputable def P_B2_given_A : ℝ :=
  (P_B2 * P_A_given_B2) / P_A

theorem more_likely_second_machine :
  P_B2_given_A > P_B1_given_A :=
by
  sorry

end more_likely_second_machine_l214_214565


namespace largest_multiple_of_7_less_than_neg85_l214_214451

theorem largest_multiple_of_7_less_than_neg85 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n < -85 ∧ n = -91 :=
by
  sorry

end largest_multiple_of_7_less_than_neg85_l214_214451


namespace terminal_side_quadrant_l214_214050

theorem terminal_side_quadrant (α : ℝ) (k : ℤ) (hk : α = 45 + k * 180) :
  (∃ n : ℕ, k = 2 * n ∧ α = 45) ∨ (∃ n : ℕ, k = 2 * n + 1 ∧ α = 225) :=
sorry

end terminal_side_quadrant_l214_214050


namespace equation_of_line_AB_l214_214143

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (3, 2)

def equation_of_line (p1 p2 : point) : ℝ × ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  -- Calculate the slope
  let k := (y2 - y1) / (x2 - x1)
  -- Use point-slope form and simplify the equation to standard form
  (((1 : ℝ), -2, 1) : ℝ × ℝ × ℝ)

theorem equation_of_line_AB :
  equation_of_line A B = (1, -2, 1) :=
sorry

end equation_of_line_AB_l214_214143


namespace computation_result_l214_214189

theorem computation_result :
  2 + 8 * 3 - 4 + 7 * 2 / 2 * 3 = 43 :=
by
  sorry

end computation_result_l214_214189


namespace range_function_l214_214797

open Real

noncomputable def function_to_prove (x : ℝ) (a : ℕ) : ℝ := x + 2 * a / x

theorem range_function (a : ℕ) (h1 : a^2 - a < 2) (h2 : a ≠ 0) : 
  Set.range (function_to_prove · a) = {y : ℝ | y ≤ -2 * sqrt 2} ∪ {y : ℝ | y ≥ 2 * sqrt 2} :=
by
  sorry

end range_function_l214_214797


namespace subscriptions_sold_to_parents_l214_214510

-- Definitions for the conditions
variable (P : Nat) -- subscriptions sold to parents
def grandfather := 1
def next_door_neighbor := 2
def other_neighbor := 2 * next_door_neighbor
def subscriptions_other_than_parents := grandfather + next_door_neighbor + other_neighbor
def total_earnings := 55
def earnings_from_others := 5 * subscriptions_other_than_parents
def earnings_from_parents := total_earnings - earnings_from_others
def subscription_price := 5

-- Theorem stating the equivalent math proof
theorem subscriptions_sold_to_parents : P = earnings_from_parents / subscription_price :=
by
  sorry

end subscriptions_sold_to_parents_l214_214510


namespace xy_value_l214_214888

theorem xy_value (x y : ℝ) (h : |x - 1| + (x + y)^2 = 0) : x * y = -1 := 
by
  sorry

end xy_value_l214_214888


namespace team_total_points_l214_214787

-- Definition of Wade's average points per game
def wade_avg_points_per_game := 20

-- Definition of teammates' average points per game
def teammates_avg_points_per_game := 40

-- Definition of the number of games
def number_of_games := 5

-- The total points calculation problem
theorem team_total_points 
  (Wade_avg : wade_avg_points_per_game = 20)
  (Teammates_avg : teammates_avg_points_per_game = 40)
  (Games : number_of_games = 5) :
  5 * wade_avg_points_per_game + 5 * teammates_avg_points_per_game = 300 := 
by 
  -- The proof is omitted and marked as sorry
  sorry

end team_total_points_l214_214787


namespace find_number_l214_214137

theorem find_number : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 :=
by {
  sorry
}

end find_number_l214_214137


namespace james_bought_100_cattle_l214_214053

noncomputable def number_of_cattle (purchase_price : ℝ) (feeding_ratio : ℝ) (weight_per_cattle : ℝ) (price_per_pound : ℝ) (profit : ℝ) : ℝ :=
  let feeding_cost := purchase_price * feeding_ratio
  let total_feeding_cost := purchase_price + feeding_cost
  let total_cost := purchase_price + total_feeding_cost
  let selling_price_per_cattle := weight_per_cattle * price_per_pound
  let total_revenue := total_cost + profit
  total_revenue / selling_price_per_cattle

theorem james_bought_100_cattle :
  number_of_cattle 40000 0.20 1000 2 112000 = 100 :=
by {
  sorry
}

end james_bought_100_cattle_l214_214053


namespace odd_primes_pq_division_l214_214191

theorem odd_primes_pq_division (p q : ℕ) (m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
(hp_odd : ¬Even p) (hq_odd : ¬Even q) (hp_gt_hq : p > q) (hm_pos : 0 < m) : ¬(p * q ∣ m ^ (p - q) + 1) :=
by 
  sorry

end odd_primes_pq_division_l214_214191


namespace extremum_problem_l214_214051

def f (x a b : ℝ) := x^3 + a*x^2 + b*x + a^2

def f_prime (x a b : ℝ) := 3*x^2 + 2*a*x + b

theorem extremum_problem (a b : ℝ) 
  (cond1 : f_prime 1 a b = 0)
  (cond2 : f 1 a b = 10) :
  (a, b) = (4, -11) := 
sorry

end extremum_problem_l214_214051


namespace min_moves_queens_switch_places_l214_214387

-- Assume a type representing the board positions
inductive Position where
| first_rank | last_rank 

-- Assume a type representing the queens
inductive Queen where
| black | white

-- Function to count minimum moves for switching places
def min_moves_to_switch_places : ℕ :=
  sorry

theorem min_moves_queens_switch_places :
  min_moves_to_switch_places = 23 :=
  sorry

end min_moves_queens_switch_places_l214_214387


namespace parallelogram_angle_A_l214_214254

theorem parallelogram_angle_A 
  (A B : ℝ) (h1 : A + B = 180) (h2 : A - B = 40) :
  A = 110 :=
by sorry

end parallelogram_angle_A_l214_214254


namespace largest_multiple_of_three_l214_214774

theorem largest_multiple_of_three (n : ℕ) (h : 3 * n + (3 * n + 3) + (3 * n + 6) = 117) : 3 * n + 6 = 42 :=
by
  sorry

end largest_multiple_of_three_l214_214774


namespace row_3_seat_6_representation_l214_214073

-- Given Conditions
def seat_representation (r : ℕ) (s : ℕ) : (ℕ × ℕ) :=
  (r, s)

-- Proof Statement
theorem row_3_seat_6_representation :
  seat_representation 3 6 = (3, 6) :=
by
  sorry

end row_3_seat_6_representation_l214_214073


namespace max_profit_at_150_l214_214618

-- Define the conditions
def purchase_price : ℕ := 80
def total_items : ℕ := 1000
def selling_price_initial : ℕ := 100
def sales_volume_decrease : ℕ := 5

-- The profit function
def profit (x : ℕ) : ℤ :=
  (selling_price_initial + x) * (total_items - sales_volume_decrease * x) - purchase_price * total_items

-- The statement to prove: the selling price of 150 yuan/item maximizes the profit at 32500 yuan.
theorem max_profit_at_150 : profit 50 = 32500 := by
  sorry

end max_profit_at_150_l214_214618


namespace max_points_right_triangle_l214_214990

theorem max_points_right_triangle (n : ℕ) :
  (∀ (pts : Fin n → ℝ × ℝ), ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    let p1 := pts i
    let p2 := pts j
    let p3 := pts k
    let a := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
    let b := (p3.1 - p2.1)^2 + (p3.2 - p2.2)^2
    let c := (p3.1 - p1.1)^2 + (p3.2 - p1.2)^2
    a + b = c ∨ b + c = a ∨ c + a = b) →
  n ≤ 4 :=
sorry

end max_points_right_triangle_l214_214990


namespace sum_of_roots_eq_three_l214_214818

-- Definitions of the polynomials
def poly1 (x : ℝ) : ℝ := 3 * x^3 + 3 * x^2 - 9 * x + 27
def poly2 (x : ℝ) : ℝ := 4 * x^3 - 16 * x^2 + 5

-- Theorem stating the sum of the roots of the given equation is 3
theorem sum_of_roots_eq_three : 
  (∀ a b c d e f g h i : ℝ, 
    (poly1 a = 0) → (poly1 b = 0) → (poly1 c = 0) → 
    (poly2 d = 0) → (poly2 e = 0) → (poly2 f = 0) →
    a + b + c + d + e + f = 3) := 
by
  sorry

end sum_of_roots_eq_three_l214_214818


namespace binomial_12_6_eq_924_l214_214089

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_l214_214089
