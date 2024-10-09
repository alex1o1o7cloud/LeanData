import Mathlib

namespace find_f2_l100_10034

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 - a * x^3 + b * x - 6

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -22 :=
by
  sorry

end find_f2_l100_10034


namespace worker_cellphone_surveys_l100_10024

theorem worker_cellphone_surveys 
  (regular_rate : ℕ) 
  (num_surveys : ℕ) 
  (higher_rate : ℕ)
  (total_earnings : ℕ) 
  (earned : ℕ → ℕ → ℕ)
  (higher_earned : ℕ → ℕ → ℕ) 
  (h1 : regular_rate = 10) 
  (h2 : num_surveys = 50) 
  (h3 : higher_rate = 13) 
  (h4 : total_earnings = 605) 
  (h5 : ∀ x, earned regular_rate (num_surveys - x) + higher_earned higher_rate x = total_earnings)
  : (∃ x, x = 35 ∧ earned regular_rate (num_surveys - x) + higher_earned higher_rate x = total_earnings) :=
sorry

end worker_cellphone_surveys_l100_10024


namespace no_possible_path_l100_10026

theorem no_possible_path (n : ℕ) (h1 : n > 0) :
  ¬ ∃ (path : ℕ × ℕ → ℕ × ℕ), 
    (∀ (i : ℕ × ℕ), path i = if (i.1 < n - 1 ∧ i.2 < n - 1) then (i.1 + 1, i.2) else if i.2 < n - 1 then (i.1, i.2 + 1) else (i.1 - 1, i.2 - 1)) ∧
    (∀ (i j : ℕ × ℕ), i ≠ j → path i ≠ path j) ∧
    path (0, 0) = (0, 1) ∧
    path (n-1, n-1) = (n-1, 0) :=
sorry

end no_possible_path_l100_10026


namespace smaller_cube_size_l100_10080

theorem smaller_cube_size
  (original_cube_side : ℕ)
  (number_of_smaller_cubes : ℕ)
  (painted_cubes : ℕ)
  (unpainted_cubes : ℕ) :
  original_cube_side = 3 → 
  number_of_smaller_cubes = 27 → 
  painted_cubes = 26 → 
  unpainted_cubes = 1 →
  (∃ (side : ℕ), side = original_cube_side / 3 ∧ side = 1) :=
by
  intros h1 h2 h3 h4
  use 1
  have h : 1 = original_cube_side / 3 := sorry
  exact ⟨h, rfl⟩

end smaller_cube_size_l100_10080


namespace solve_inequality_l100_10003

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l100_10003


namespace molecular_weight_is_44_02_l100_10028

-- Definition of atomic weights and the number of atoms
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def count_N : ℕ := 2
def count_O : ℕ := 1

-- The compound's molecular weight calculation
def molecular_weight : ℝ := (count_N * atomic_weight_N) + (count_O * atomic_weight_O)

-- The proof statement that the molecular weight of the compound is approximately 44.02 amu
theorem molecular_weight_is_44_02 : molecular_weight = 44.02 := 
by
  sorry

#eval molecular_weight  -- Should output 44.02 (not part of the theorem, just for checking)

end molecular_weight_is_44_02_l100_10028


namespace repeating_decimal_eq_fraction_l100_10010

theorem repeating_decimal_eq_fraction :
  let a := (85 : ℝ) / 100
  let r := (1 : ℝ) / 100
  (∑' n : ℕ, a * (r ^ n)) = 85 / 99 := by
  let a := (85 : ℝ) / 100
  let r := (1 : ℝ) / 100
  exact sorry

end repeating_decimal_eq_fraction_l100_10010


namespace total_stamps_l100_10071

-- Definitions for the conditions.
def snowflake_stamps : ℕ := 11
def truck_stamps : ℕ := snowflake_stamps + 9
def rose_stamps : ℕ := truck_stamps - 13

-- Statement to prove the total number of stamps.
theorem total_stamps : (snowflake_stamps + truck_stamps + rose_stamps) = 38 :=
by
  sorry

end total_stamps_l100_10071


namespace a14_eq_33_l100_10072

variable {a : ℕ → ℝ}
variables (d : ℝ) (a1 : ℝ)

-- Defining the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℝ := a1 + n * d

-- Given conditions
axiom a5_eq_6 : arithmetic_sequence 4 = 6
axiom a8_eq_15 : arithmetic_sequence 7 = 15

-- Theorem statement
theorem a14_eq_33 : arithmetic_sequence 13 = 33 :=
by
  -- Proof skipped
  sorry

end a14_eq_33_l100_10072


namespace score_calculation_l100_10075

theorem score_calculation (N : ℕ) (C : ℕ) (hN: 1 ≤ N ∧ N ≤ 20) (hC: 1 ≤ C) : 
  ∃ (score: ℕ), score = Nat.floor (N / C) :=
by sorry

end score_calculation_l100_10075


namespace average_of_numbers_in_range_l100_10005

-- Define the set of numbers we are considering
def numbers_in_range : List ℕ := [10, 15, 20, 25, 30]

-- Define the sum of these numbers
def sum_in_range : ℕ := 10 + 15 + 20 + 25 + 30

-- Define the number of elements in our range
def count_in_range : ℕ := 5

-- Prove that the average of numbers in the range is 20
theorem average_of_numbers_in_range : (sum_in_range / count_in_range) = 20 := by
  -- TODO: Proof to be written, for now we use sorry as a placeholder
  sorry

end average_of_numbers_in_range_l100_10005


namespace negation_of_universal_proposition_l100_10085

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 > Real.log x)) ↔ (∃ x : ℝ, x^2 ≤ Real.log x) :=
by
  sorry

end negation_of_universal_proposition_l100_10085


namespace g_f_neg2_l100_10025

def f (x : ℤ) : ℤ := x^3 + 3

def g (x : ℤ) : ℤ := 2*x^2 + 2*x + 1

theorem g_f_neg2 : g (f (-2)) = 41 :=
by {
  -- proof steps skipped
  sorry
}

end g_f_neg2_l100_10025


namespace stratified_sampling_l100_10060

theorem stratified_sampling (total_students : ℕ) (ratio_grade1 ratio_grade2 ratio_grade3 : ℕ) (sample_size : ℕ) (h_ratio : ratio_grade1 = 3 ∧ ratio_grade2 = 3 ∧ ratio_grade3 = 4) (h_sample_size : sample_size = 50) : 
  (ratio_grade2 / (ratio_grade1 + ratio_grade2 + ratio_grade3) : ℚ) * sample_size = 15 := 
by
  sorry

end stratified_sampling_l100_10060


namespace polar_equation_of_circle_slope_of_line_l100_10068

-- Part 1: Polar equation of circle C
theorem polar_equation_of_circle (x y : ℝ) :
  (x - 2) ^ 2 + y ^ 2 = 9 -> ∃ (ρ θ : ℝ), ρ^2 - 4*ρ*Real.cos θ - 5 = 0 := 
sorry

-- Part 2: Slope of line L intersecting C at points A and B
theorem slope_of_line (α : ℝ) (L : ℝ → ℝ × ℝ) (A B : ℝ × ℝ) :
  (∀ t, L t = (t * Real.cos α, t * Real.sin α)) ∧ dist A B = 2 * Real.sqrt 7 ∧ 
  (∃ x y, (x - 2) ^ 2 + y ^ 2 = 9 ∧ L (Real.sqrt ((x - 2) ^ 2 + y ^ 2)) = (x, y))
  -> Real.tan α = 1 ∨ Real.tan α = -1 :=
sorry

end polar_equation_of_circle_slope_of_line_l100_10068


namespace ordered_pair_solution_l100_10016

theorem ordered_pair_solution :
  ∃ (x y : ℤ), 
    (x + y = (7 - x) + (7 - y)) ∧ 
    (x - y = (x - 2) + (y - 2)) ∧ 
    (x = 5 ∧ y = 2) :=
by
  sorry

end ordered_pair_solution_l100_10016


namespace friend_selling_price_l100_10059

-- Definitions and conditions
def original_cost_price : ℝ := 51724.14

def loss_percentage : ℝ := 0.13
def gain_percentage : ℝ := 0.20

def selling_price_man (CP : ℝ) : ℝ := (1 - loss_percentage) * CP
def selling_price_friend (SP1 : ℝ) : ℝ := (1 + gain_percentage) * SP1

-- Prove that the friend's selling price is 54,000 given the conditions
theorem friend_selling_price :
  selling_price_friend (selling_price_man original_cost_price) = 54000 :=
by
  sorry

end friend_selling_price_l100_10059


namespace smallest_integer_n_l100_10077

theorem smallest_integer_n (n : ℕ) (h₁ : 50 ∣ n^2) (h₂ : 294 ∣ n^3) : n = 210 :=
sorry

end smallest_integer_n_l100_10077


namespace six_digit_mod7_l100_10066

theorem six_digit_mod7 (a b c d e f : ℕ) (N : ℕ) (h : N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) (h_div7 : N % 7 = 0) :
    (10^5 * f + 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e) % 7 = 0 :=
by
  sorry

end six_digit_mod7_l100_10066


namespace factorize_expression_l100_10084

theorem factorize_expression (x y : ℝ) : 
  6 * x^3 * y^2 + 3 * x^2 * y^2 = 3 * x^2 * y^2 * (2 * x + 1) := 
by 
  sorry

end factorize_expression_l100_10084


namespace operation_difference_l100_10011

def operation (x y : ℕ) : ℕ := x * y - 3 * x + y

theorem operation_difference : operation 5 9 - operation 9 5 = 16 :=
by
  sorry

end operation_difference_l100_10011


namespace min_price_floppy_cd_l100_10021

theorem min_price_floppy_cd (x y : ℝ) (h1 : 4 * x + 5 * y ≥ 20) (h2 : 6 * x + 3 * y ≤ 24) : 3 * x + 9 * y ≥ 22 :=
by
  -- The proof is not provided as per the instructions.
  sorry

end min_price_floppy_cd_l100_10021


namespace unique_solution_nat_triplet_l100_10007

theorem unique_solution_nat_triplet (x y l : ℕ) (h : x^3 + y^3 - 53 = 7^l) : (x, y, l) = (3, 3, 0) :=
sorry

end unique_solution_nat_triplet_l100_10007


namespace roots_diff_eq_4_l100_10022

theorem roots_diff_eq_4 {r s : ℝ} (h₁ : r ≠ s) (h₂ : r > s) (h₃ : r^2 - 10 * r + 21 = 0) (h₄ : s^2 - 10 * s + 21 = 0) : r - s = 4 := 
by
  sorry

end roots_diff_eq_4_l100_10022


namespace triangle_similar_l100_10093

variables {a b c m_a m_b m_c t : ℝ}

-- Define the triangle ABC and its properties
def triangle_ABC (a b c m_a m_b m_c t : ℝ) : Prop :=
  t = (1 / 2) * a * m_a ∧
  t = (1 / 2) * b * m_b ∧
  t = (1 / 2) * c * m_c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧
  t > 0

-- Define the similarity condition for the triangles
def similitude_from_reciprocals (a b c m_a m_b m_c t : ℝ) : Prop :=
  (1 / m_a) / (1 / m_b) = a / b ∧
  (1 / m_b) / (1 / m_c) = b / c ∧
  (1 / m_a) / (1 / m_c) = a / c

theorem triangle_similar (a b c m_a m_b m_c t : ℝ) :
  triangle_ABC a b c m_a m_b m_c t →
  similitude_from_reciprocals a b c m_a m_b m_c t :=
by
  intro h
  sorry

end triangle_similar_l100_10093


namespace f_zero_f_odd_f_inequality_solution_l100_10062

open Real

-- Given definitions
variables {f : ℝ → ℝ}
variable (h_inc : ∀ x y, x < y → f x < f y)
variable (h_eq : ∀ x y, y * f x - x * f y = x * y * (x^2 - y^2))

-- Prove that f(0) = 0
theorem f_zero : f 0 = 0 := 
sorry

-- Prove that f is an odd function
theorem f_odd : ∀ x, f (-x) = -f x := 
sorry

-- Prove the range of x satisfying the given inequality
theorem f_inequality_solution : {x : ℝ | f (x^2 + 1) + f (3 * x - 5) < 0} = {x : ℝ | -4 < x ∧ x < 1} :=
sorry

end f_zero_f_odd_f_inequality_solution_l100_10062


namespace max_value_a_l100_10012

def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → |a - 2| ≤ |x + 1 / x|

theorem max_value_a : ∃ (a : ℝ), condition a ∧ (∀ b : ℝ, condition b → b ≤ 4) :=
  sorry

end max_value_a_l100_10012


namespace average_percentage_increase_l100_10002

def initial_income_A : ℝ := 60
def new_income_A : ℝ := 80
def initial_income_B : ℝ := 100
def new_income_B : ℝ := 130
def hours_worked_C : ℝ := 20
def initial_rate_C : ℝ := 8
def new_rate_C : ℝ := 10

theorem average_percentage_increase :
  let initial_weekly_income_C := hours_worked_C * initial_rate_C
  let new_weekly_income_C := hours_worked_C * new_rate_C
  let percentage_increase_A := (new_income_A - initial_income_A) / initial_income_A * 100
  let percentage_increase_B := (new_income_B - initial_income_B) / initial_income_B * 100
  let percentage_increase_C := (new_weekly_income_C - initial_weekly_income_C) / initial_weekly_income_C * 100
  let average_percentage_increase := (percentage_increase_A + percentage_increase_B + percentage_increase_C) / 3
  average_percentage_increase = 29.44 :=
by sorry

end average_percentage_increase_l100_10002


namespace pool_width_l100_10019

-- Define the given conditions
def hose_rate : ℝ := 60 -- cubic feet per minute
def drain_time : ℝ := 2000 -- minutes
def pool_length : ℝ := 150 -- feet
def pool_depth : ℝ := 10 -- feet

-- Calculate the total volume drained
def total_volume := hose_rate * drain_time -- cubic feet

-- Define a variable for the pool width
variable (W : ℝ)

-- The statement to prove
theorem pool_width :
  (total_volume = pool_length * W * pool_depth) → W = 80 :=
by
  sorry

end pool_width_l100_10019


namespace flowerbed_width_l100_10095

theorem flowerbed_width (w : ℝ) (h₁ : 22 = 2 * (2 * w - 1) + 2 * w) : w = 4 :=
sorry

end flowerbed_width_l100_10095


namespace Mike_owes_Laura_l100_10041

theorem Mike_owes_Laura :
  let rate_per_room := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let total_amount := (104 : ℚ) / 15
  rate_per_room * rooms_cleaned = total_amount :=
by
  sorry

end Mike_owes_Laura_l100_10041


namespace union_complement_real_domain_l100_10067

noncomputable def M : Set ℝ := {x | -2 < x ∧ x < 2}
noncomputable def N : Set ℝ := {x | -2 < x}

theorem union_complement_real_domain :
  M ∪ (Set.univ \ N) = {x : ℝ | x < 2} :=
by
  sorry

end union_complement_real_domain_l100_10067


namespace maximum_sum_of_digits_difference_l100_10038

-- Definition of the sum of the digits of a number
-- For the purpose of this statement, we'll assume the existence of a function sum_of_digits

def sum_of_digits (n : ℕ) : ℕ :=
  sorry -- Assume the function is defined elsewhere

-- Statement of the problem
theorem maximum_sum_of_digits_difference :
  ∃ x : ℕ, sum_of_digits (x + 2019) - sum_of_digits x = 12 :=
sorry

end maximum_sum_of_digits_difference_l100_10038


namespace width_of_park_l100_10009

theorem width_of_park (L : ℕ) (A_lawn : ℕ) (w_road : ℕ) (W : ℚ) :
  L = 60 → A_lawn = 2109 → w_road = 3 →
  60 * W - 2 * 60 * 3 = 2109 →
  W = 41.15 :=
by
  intros hL hA_lawn hw_road hEq
  -- The proof will go here
  sorry

end width_of_park_l100_10009


namespace part1_solution_part2_solution_l100_10079

theorem part1_solution (x : ℝ) (h1 : (2 * x) / (x - 2) + 3 / (2 - x) = 1) : x = 1 := by
  sorry

theorem part2_solution (x : ℝ) 
  (h1 : 2 * x - 1 ≥ 3 * (x - 1)) 
  (h2 : (5 - x) / 2 < x + 3) : -1 / 3 < x ∧ x ≤ 2 := by
  sorry

end part1_solution_part2_solution_l100_10079


namespace trigonometric_identity_l100_10051

theorem trigonometric_identity :
  let cos_30 := (Real.sqrt 3) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1 := by
  sorry

end trigonometric_identity_l100_10051


namespace arthur_bakes_muffins_l100_10099

-- Definitions of the conditions
def james_muffins : ℚ := 9.58333333299999
def multiplier : ℚ := 12.0

-- Statement of the problem
theorem arthur_bakes_muffins : 
  abs (multiplier * james_muffins - 115) < 1 :=
by
  sorry

end arthur_bakes_muffins_l100_10099


namespace express_in_scientific_notation_l100_10033

theorem express_in_scientific_notation :
  102200 = 1.022 * 10^5 :=
sorry

end express_in_scientific_notation_l100_10033


namespace quadratic_roots_real_equal_l100_10048

theorem quadratic_roots_real_equal (m : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ a = 3 ∧ b = 2 - m ∧ c = 6 ∧
    (b^2 - 4 * a * c = 0)) ↔ (m = 2 - 6 * Real.sqrt 2 ∨ m = 2 + 6 * Real.sqrt 2) :=
by
  sorry

end quadratic_roots_real_equal_l100_10048


namespace mod_abc_eq_zero_l100_10061

open Nat

theorem mod_abc_eq_zero
    (a b c : ℕ)
    (h1 : (a + 2 * b + 3 * c) % 9 = 1)
    (h2 : (2 * a + 3 * b + c) % 9 = 2)
    (h3 : (3 * a + b + 2 * c) % 9 = 3) :
    (a * b * c) % 9 = 0 := by
  sorry

end mod_abc_eq_zero_l100_10061


namespace parabola_distance_l100_10042

theorem parabola_distance (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1)
  (h_distance_focus : (P.1 - 1)^2 + P.2^2 = 9) : 
  Real.sqrt (P.1^2 + P.2^2) = 2 * Real.sqrt 3 :=
by
  sorry

end parabola_distance_l100_10042


namespace solve_system_part1_solve_system_part3_l100_10017

noncomputable def solution_part1 : Prop :=
  ∃ (x y : ℝ), (x + y = 2) ∧ (5 * x - 2 * (x + y) = 6) ∧ (x = 2) ∧ (y = 0)

-- Part (1) Statement
theorem solve_system_part1 : solution_part1 := sorry

noncomputable def solution_part3 : Prop :=
  ∃ (a b c : ℝ), (a + b = 3) ∧ (5 * a + 3 * c = 1) ∧ (a + b + c = 0) ∧ (a = 2) ∧ (b = 1) ∧ (c = -3)

-- Part (3) Statement
theorem solve_system_part3 : solution_part3 := sorry

end solve_system_part1_solve_system_part3_l100_10017


namespace negation_of_implication_l100_10036

theorem negation_of_implication (x : ℝ) : x^2 + x - 6 < 0 → x ≤ 2 :=
by
  -- proof goes here
  sorry

end negation_of_implication_l100_10036


namespace largest_fraction_l100_10045

theorem largest_fraction (x y z w : ℝ) (hx : 0 < x) (hxy : x < y) (hyz : y < z) (hzw : z < w) :
  max (max (max (max ((x + y) / (z + w)) ((x + w) / (y + z))) ((y + z) / (x + w))) ((y + w) / (x + z))) ((z + w) / (x + y)) = (z + w) / (x + y) :=
by sorry

end largest_fraction_l100_10045


namespace cubes_not_arithmetic_progression_l100_10013

theorem cubes_not_arithmetic_progression (x y z : ℤ) (h1 : y = (x + z) / 2) (h2 : x ≠ y) (h3 : y ≠ z) : x^3 + z^3 ≠ 2 * y^3 :=
by
  sorry

end cubes_not_arithmetic_progression_l100_10013


namespace zoo_revenue_is_61_l100_10049

def children_monday := 7
def children_tuesday := 4
def adults_monday := 5
def adults_tuesday := 2
def child_ticket_cost := 3
def adult_ticket_cost := 4

def total_children := children_monday + children_tuesday
def total_adults := adults_monday + adults_tuesday

def total_children_cost := total_children * child_ticket_cost
def total_adult_cost := total_adults * adult_ticket_cost

def total_zoo_revenue := total_children_cost + total_adult_cost

theorem zoo_revenue_is_61 : total_zoo_revenue = 61 := by
  sorry

end zoo_revenue_is_61_l100_10049


namespace smallest_integer_for_polynomial_div_l100_10001

theorem smallest_integer_for_polynomial_div (x : ℤ) : 
  (∃ k : ℤ, x = 6) ↔ ∃ y, y * (x - 5) = x^2 + 4 * x + 7 := 
by 
  sorry

end smallest_integer_for_polynomial_div_l100_10001


namespace pie_crusts_flour_l100_10039

theorem pie_crusts_flour (initial_crusts : ℕ)
  (initial_flour_per_crust : ℚ)
  (new_crusts : ℕ)
  (total_flour : ℚ)
  (h1 : initial_crusts = 40)
  (h2 : initial_flour_per_crust = 1/8)
  (h3 : new_crusts = 25)
  (h4 : total_flour = initial_crusts * initial_flour_per_crust) :
  (new_crusts * (total_flour / new_crusts) = total_flour) :=
by
  sorry

end pie_crusts_flour_l100_10039


namespace exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2_l100_10029

def omega (n : Nat) : Nat :=
  if n = 1 then 0 else n.factors.toFinset.card

theorem exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2 :
  ∃ᶠ n in atTop, ∃ k : Nat, n = 2^k ∧
    omega n < omega (n + 1) ∧
    omega (n + 1) < omega (n + 2) :=
sorry

end exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2_l100_10029


namespace slices_per_pizza_l100_10083

theorem slices_per_pizza (num_pizzas num_slices : ℕ) (h1 : num_pizzas = 17) (h2 : num_slices = 68) :
  (num_slices / num_pizzas) = 4 :=
by
  sorry

end slices_per_pizza_l100_10083


namespace rhombus_area_l100_10089

-- Define the parameters given in the problem
namespace MathProof

def perimeter (EFGH : ℝ) : ℝ := 80
def diagonal_EG (EFGH : ℝ) : ℝ := 30

-- Considering the rhombus EFGH with the given perimeter and diagonal
theorem rhombus_area : 
  ∃ (area : ℝ), area = 150 * Real.sqrt 7 ∧ 
  (perimeter EFGH = 80) ∧ 
  (diagonal_EG EFGH = 30) :=
  sorry
end MathProof

end rhombus_area_l100_10089


namespace neither_necessary_nor_sufficient_l100_10004

def set_M : Set ℝ := {x | x > 2}
def set_P : Set ℝ := {x | x < 3}

theorem neither_necessary_nor_sufficient (x : ℝ) :
  (x ∈ set_M ∨ x ∈ set_P) ↔ (x ∉ set_M ∩ set_P) :=
sorry

end neither_necessary_nor_sufficient_l100_10004


namespace solve_for_a_l100_10056

theorem solve_for_a (a x : ℝ) (h1 : 3 * x - 5 = x + a) (h2 : x = 2) : a = -1 :=
by
  sorry

end solve_for_a_l100_10056


namespace correct_answer_of_john_l100_10081

theorem correct_answer_of_john (x : ℝ) (h : 5 * x + 4 = 104) : (x + 5) / 4 = 6.25 :=
by
  sorry

end correct_answer_of_john_l100_10081


namespace maximum_M_k_l100_10052

-- Define the problem
def J (k : ℕ) : ℕ := 10^(k + 2) + 128

-- Define M(k) as the number of factors of 2 in the prime factorization of J(k)
def M (k : ℕ) : ℕ :=
  -- implementation details omitted
  sorry

-- The core theorem to prove
theorem maximum_M_k : ∃ k > 0, M k = 8 :=
by sorry

end maximum_M_k_l100_10052


namespace solve_for_x_l100_10046

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end solve_for_x_l100_10046


namespace hexagon_area_of_circle_l100_10047

noncomputable def radius (area : ℝ) : ℝ :=
  Real.sqrt (area / Real.pi)

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s * s * Real.sqrt 3) / 4

theorem hexagon_area_of_circle {r : ℝ} (h : π * r^2 = 100 * π) :
  6 * area_equilateral_triangle r = 150 * Real.sqrt 3 :=
by
  sorry

end hexagon_area_of_circle_l100_10047


namespace time_to_cross_tree_l100_10087

variable (length_train : ℕ) (time_platform : ℕ) (length_platform : ℕ)

theorem time_to_cross_tree (h1 : length_train = 1200) (h2 : time_platform = 190) (h3 : length_platform = 700) :
  let distance_platform := length_train + length_platform
  let speed_train := distance_platform / time_platform
  let time_to_cross_tree := length_train / speed_train
  time_to_cross_tree = 120 :=
by
  -- Using the conditions to prove the goal
  sorry

end time_to_cross_tree_l100_10087


namespace opposite_sign_pairs_l100_10064

theorem opposite_sign_pairs :
  ¬ ((- 2 ^ 3 < 0) ∧ (- (2 ^ 3) > 0)) ∧
  ¬ (|-4| < 0 ∧ -(-4) > 0) ∧
  ((- 3 ^ 4 < 0 ∧ (-(3 ^ 4)) = 81)) ∧
  ¬ (10 ^ 2 < 0 ∧ 2 ^ 10 > 0) :=
by
  sorry

end opposite_sign_pairs_l100_10064


namespace value_of_a_plus_d_l100_10088

variable (a b c d : ℝ)

theorem value_of_a_plus_d (h1 : a + b = 4) (h2 : b + c = 7) (h3 : c + d = 5) : a + d = 4 :=
sorry

end value_of_a_plus_d_l100_10088


namespace lucky_point_m2_is_lucky_point_A33_point_M_quadrant_l100_10094

noncomputable def lucky_point (m n : ℝ) : Prop := 2 * m = 4 + n ∧ ∃ (x y : ℝ), (x = m - 1) ∧ (y = (n + 2) / 2)

theorem lucky_point_m2 :
  lucky_point 2 0 := sorry

theorem is_lucky_point_A33 :
  lucky_point 4 4 := sorry

theorem point_M_quadrant (a : ℝ) :
  lucky_point (a + 1) (2 * (2 * a - 1) - 2) → (a = 1) := sorry

end lucky_point_m2_is_lucky_point_A33_point_M_quadrant_l100_10094


namespace roots_quadratic_eq_a2_b2_l100_10040

theorem roots_quadratic_eq_a2_b2 (a b : ℝ) (h1 : a^2 - 5 * a + 5 = 0) (h2 : b^2 - 5 * b + 5 = 0) : a^2 + b^2 = 15 :=
by
  sorry

end roots_quadratic_eq_a2_b2_l100_10040


namespace find_d_l100_10053

theorem find_d (d : ℝ) (h₁ : ∃ x, x = ⌊d⌋ ∧ 3 * x^2 + 19 * x - 84 = 0)
                (h₂ : ∃ y, y = d - ⌊d⌋ ∧ 5 * y^2 - 28 * y + 12 = 0 ∧ 0 ≤ y ∧ y < 1) :
  d = 3.2 :=
by
  sorry

end find_d_l100_10053


namespace tommys_books_l100_10097

-- Define the cost of each book
def book_cost : ℕ := 5

-- Define the amount Tommy already has
def tommy_money : ℕ := 13

-- Define the amount Tommy needs to save up
def tommy_goal : ℕ := 27

-- Prove the number of books Tommy wants to buy
theorem tommys_books : tommy_goal + tommy_money = 40 ∧ (tommy_goal + tommy_money) / book_cost = 8 :=
by
  sorry

end tommys_books_l100_10097


namespace find_number_l100_10054

theorem find_number (x : ℤ) (h : 3 * x + 3 * 12 + 3 * 13 + 11 = 134) : x = 16 :=
by
  sorry

end find_number_l100_10054


namespace largest_number_not_sum_of_two_composites_l100_10018

-- Definitions related to composite numbers and natural numbers
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)

-- Lean statement of the problem
theorem largest_number_not_sum_of_two_composites : ∀ n : ℕ,
  (∀ m : ℕ, m > n → cannot_be_sum_of_two_composites m → false) → n = 11 :=
by
  sorry

end largest_number_not_sum_of_two_composites_l100_10018


namespace percent_of_x_is_z_l100_10065

theorem percent_of_x_is_z (x y z : ℝ) (h1 : 0.45 * z = 1.2 * y) (h2 : y = 0.75 * x) : z = 2 * x :=
by
  sorry

end percent_of_x_is_z_l100_10065


namespace travel_time_reduction_l100_10027

theorem travel_time_reduction
  (original_speed : ℝ)
  (new_speed : ℝ)
  (time : ℝ)
  (distance : ℝ)
  (new_time : ℝ)
  (h1 : original_speed = 80)
  (h2 : new_speed = 50)
  (h3 : time = 3)
  (h4 : distance = original_speed * time)
  (h5 : new_time = distance / new_speed) :
  new_time = 4.8 := 
sorry

end travel_time_reduction_l100_10027


namespace Rachel_made_total_amount_l100_10082

def cost_per_bar : ℝ := 3.25
def total_bars_sold : ℕ := 25 - 7
def total_amount_made : ℝ := total_bars_sold * cost_per_bar

theorem Rachel_made_total_amount :
  total_amount_made = 58.50 :=
by
  sorry

end Rachel_made_total_amount_l100_10082


namespace cubic_roots_sum_of_cubes_l100_10055

theorem cubic_roots_sum_of_cubes (r s t a b c : ℚ) 
  (h1 : r + s + t = a) 
  (h2 : r * s + r * t + s * t = b)
  (h3 : r * s * t = c) 
  (h_poly : ∀ x : ℚ, x^3 - a*x^2 + b*x - c = 0 ↔ (x = r ∨ x = s ∨ x = t)) :
  r^3 + s^3 + t^3 = a^3 - 3 * a * b + 3 * c :=
sorry

end cubic_roots_sum_of_cubes_l100_10055


namespace gcd_f_50_51_l100_10086

def f (x : ℤ) : ℤ :=
  x ^ 2 - 2 * x + 2023

theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 11 := by
  sorry

end gcd_f_50_51_l100_10086


namespace taco_truck_profit_l100_10014

-- Definitions and conditions
def pounds_of_beef : ℕ := 100
def beef_per_taco : ℝ := 0.25
def price_per_taco : ℝ := 2
def cost_per_taco : ℝ := 1.5

-- Desired profit result
def expected_profit : ℝ := 200

-- The proof statement (to be completed)
theorem taco_truck_profit :
  let tacos := pounds_of_beef / beef_per_taco;
  let revenue := tacos * price_per_taco;
  let cost := tacos * cost_per_taco;
  let profit := revenue - cost;
  profit = expected_profit :=
by
  sorry

end taco_truck_profit_l100_10014


namespace worker_efficiency_l100_10063

theorem worker_efficiency (W_p W_q : ℚ) 
  (h1 : W_p = 1 / 24) 
  (h2 : W_p + W_q = 1 / 14) :
  (W_p - W_q) / W_q * 100 = 40 :=
by
  sorry

end worker_efficiency_l100_10063


namespace sum_of_coefficients_of_parabolas_kite_formed_l100_10073

theorem sum_of_coefficients_of_parabolas_kite_formed (a b : ℝ) 
  (h1 : ∃ (x : ℝ), y = ax^2 - 4)
  (h2 : ∃ (y : ℝ), y = 6 - bx^2)
  (h3 : (a > 0) ∧ (b > 0) ∧ (ax^2 - 4 = 0) ∧ (6 - bx^2 = 0))
  (h4 : kite_area = 18) :
  a + b = 125/36 := 
by sorry

end sum_of_coefficients_of_parabolas_kite_formed_l100_10073


namespace rectangle_area_diagonal_ratio_l100_10020

theorem rectangle_area_diagonal_ratio (d : ℝ) (x : ℝ) (h_ratio : 5 * x ≥ 0 ∧ 2 * x ≥ 0)
  (h_diagonal : d^2 = (5 * x)^2 + (2 * x)^2) :
  ∃ k : ℝ, (5 * x) * (2 * x) = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_diagonal_ratio_l100_10020


namespace f_in_neg_interval_l100_10069

variables (f : ℝ → ℝ)

-- Conditions
def is_even := ∀ x, f x = f (-x)
def symmetry := ∀ x, f (2 + x) = f (2 - x)
def in_interval := ∀ x, 0 < x ∧ x < 2 → f x = 1 / x

-- Target statement
theorem f_in_neg_interval
  (h_even : is_even f)
  (h_symm : symmetry f)
  (h_interval : in_interval f)
  (x : ℝ)
  (hx : -4 < x ∧ x < -2) :
  f x = 1 / (x + 4) :=
sorry

end f_in_neg_interval_l100_10069


namespace find_x_squared_plus_y_squared_l100_10091

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 6) : x^2 + y^2 = 13 :=
by
  sorry

end find_x_squared_plus_y_squared_l100_10091


namespace simplify_tan_pi_over_24_add_tan_7pi_over_24_l100_10058

theorem simplify_tan_pi_over_24_add_tan_7pi_over_24 :
  let a := Real.tan (Real.pi / 24)
  let b := Real.tan (7 * Real.pi / 24)
  a + b = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by
  -- conditions and definitions:
  let tan_eq_sin_div_cos := ∀ x, Real.tan x = Real.sin x / Real.cos x
  let sin_add := ∀ a b, Real.sin (a + b) = Real.sin a * Real.cos b + Real.cos a * Real.sin b
  let cos_mul := ∀ a b, Real.cos a * Real.cos b = 1 / 2 * (Real.cos (a + b) + Real.cos (a - b))
  let sin_pi_over_3 := Real.sin (Real.pi / 3) = Real.sqrt 3 / 2
  let cos_pi_over_3 := Real.cos (Real.pi / 3) = 1 / 2
  let cos_pi_over_4 := Real.cos (Real.pi / 4) = Real.sqrt 2 / 2
  have cond1 := tan_eq_sin_div_cos
  have cond2 := sin_add
  have cond3 := cos_mul
  have cond4 := sin_pi_over_3
  have cond5 := cos_pi_over_3
  have cond6 := cos_pi_over_4
  sorry

end simplify_tan_pi_over_24_add_tan_7pi_over_24_l100_10058


namespace actual_time_before_storm_l100_10030

-- Define valid hour digit ranges before the storm
def valid_first_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3
def valid_second_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define valid minute digit ranges before the storm
def valid_third_digit (d : ℕ) : Prop := d = 4 ∨ d = 5 ∨ d = 6
def valid_fourth_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define a valid time in HH:MM format
def valid_time (hh mm : ℕ) : Prop :=
  hh < 24 ∧ mm < 60

-- The proof problem
theorem actual_time_before_storm (hh hh' mm mm' : ℕ) 
  (h1 : valid_first_digit hh) (h2 : valid_second_digit hh') 
  (h3 : valid_third_digit mm) (h4 : valid_fourth_digit mm') 
  (h_valid : valid_time (hh * 10 + hh') (mm * 10 + mm')) 
  (h_display : (hh + 1) * 10 + (hh' - 1) = 20 ∧ (mm + 1) * 10 + (mm' - 1) = 50) :
  hh * 10 + hh' = 19 ∧ mm * 10 + mm' = 49 :=
by
  sorry

end actual_time_before_storm_l100_10030


namespace rectangle_area_l100_10043

noncomputable def side_of_square : ℝ := Real.sqrt 625

noncomputable def radius_of_circle : ℝ := side_of_square

noncomputable def length_of_rectangle : ℝ := (2 / 5) * radius_of_circle

def breadth_of_rectangle : ℝ := 10

theorem rectangle_area :
  length_of_rectangle * breadth_of_rectangle = 100 := 
by
  simp [length_of_rectangle, breadth_of_rectangle, radius_of_circle, side_of_square]
  sorry

end rectangle_area_l100_10043


namespace trigonometric_expression_eval_l100_10008

-- Conditions
variable (α : Real) (h1 : ∃ x : Real, 3 * x^2 - x - 2 = 0 ∧ x = Real.cos α) (h2 : α > π ∧ α < 3 * π / 2)

-- Question and expected answer
theorem trigonometric_expression_eval :
  (Real.sin (-α + 3 * π / 2) * Real.cos (3 * π / 2 + α) * Real.tan (π - α)^2) /
  (Real.cos (π / 2 + α) * Real.sin (π / 2 - α)) = 5 / 4 := sorry

end trigonometric_expression_eval_l100_10008


namespace geometric_sequence_arithmetic_sequence_l100_10057

def seq₃ := 7
def rec_rel (a : ℕ → ℕ) := ∀ n, n ≥ 2 → a n = 2 * a (n - 1) + a 2 - 2

-- Problem Part 1: Prove that {a_n+1} is a geometric sequence
theorem geometric_sequence (a : ℕ → ℕ) (h_rec_rel : rec_rel a) :
  ∃ r, ∀ n, n ≥ 1 → (a n + 1) = r * (a (n - 1) + 1) :=
sorry

-- Problem Part 2: Given a general formula, prove n, a_n, and S_n form an arithmetic sequence
def general_formula (a : ℕ → ℕ) := ∀ n, a n = 2^n - 1
def sum_formula (S : ℕ → ℕ) := ∀ n, S n = 2^(n+1) - n - 2

theorem arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h_general : general_formula a) (h_sum : sum_formula S) :
  ∀ n, n + S n = 2 * a n :=
sorry

end geometric_sequence_arithmetic_sequence_l100_10057


namespace outdoor_section_width_l100_10050

theorem outdoor_section_width (Length Area Width : ℝ) (h1 : Length = 6) (h2 : Area = 24) : Width = 4 :=
by
  -- We'll use "?" to represent the parts that need to be inferred by the proof assistant. 
  sorry

end outdoor_section_width_l100_10050


namespace ruler_cost_l100_10015

variable {s c r : ℕ}

theorem ruler_cost (h1 : s > 18) (h2 : r > 1) (h3 : c > r) (h4 : s * c * r = 1729) : c = 13 :=
by
  sorry

end ruler_cost_l100_10015


namespace Louis_ate_whole_boxes_l100_10098

def package_size := 6
def total_lemon_heads := 54

def whole_boxes : ℕ := total_lemon_heads / package_size

theorem Louis_ate_whole_boxes :
  whole_boxes = 9 :=
by
  sorry

end Louis_ate_whole_boxes_l100_10098


namespace sufficient_condition_hyperbola_l100_10096

theorem sufficient_condition_hyperbola (m : ℝ) (h : 5 < m) : 
  ∃ a b : ℝ, (a > 0) ∧ (b < 0) ∧ (∀ x y : ℝ, (x^2)/(a) + (y^2)/(b) = 1) := 
sorry

end sufficient_condition_hyperbola_l100_10096


namespace battery_current_l100_10074

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l100_10074


namespace min_a_for_monotonic_increase_l100_10006

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x ^ 3 + 2 * a * x ^ 2 + 2

theorem min_a_for_monotonic_increase :
  ∀ a : ℝ, (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → x^2 + 4 * a * x ≥ 0) ↔ a ≥ -1/4 := sorry

end min_a_for_monotonic_increase_l100_10006


namespace general_form_identity_expression_simplification_l100_10000

section
variable (a b x y : ℝ)

theorem general_form_identity : (a + b) * (a^2 - a * b + b^2) = a^3 + b^3 :=
by
  sorry

theorem expression_simplification : (x + y) * (x^2 - x * y + y^2) - (x - y) * (x^2 + x * y + y^2) = 2 * y^3 :=
by
  sorry
end

end general_form_identity_expression_simplification_l100_10000


namespace percentage_goods_lost_eq_l100_10037

-- Define the initial conditions
def initial_value : ℝ := 100
def profit_margin : ℝ := 0.10 * initial_value
def selling_price : ℝ := initial_value + profit_margin
def loss_percentage : ℝ := 0.12

-- Define the correct answer as a constant
def correct_percentage_loss : ℝ := 13.2

-- Define the target theorem
theorem percentage_goods_lost_eq : (0.12 * selling_price / initial_value * 100) = correct_percentage_loss := 
by
  -- sorry is used to skip the proof part as per instructions
  sorry

end percentage_goods_lost_eq_l100_10037


namespace find_angle_A_l100_10031

def triangle_ABC_angle_A (a b : ℝ) (B A : ℝ) (acute : Prop)
  (ha : a = 2 * Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 2)
  (hB: B = Real.pi / 4)
  (hacute: acute) : Prop :=
  A = Real.pi / 3

theorem find_angle_A 
  (a b A B : ℝ) (acute : Prop)
  (ha : a = 2 * Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 2)
  (hB: B = Real.pi / 4)
  (hacute: acute)
  (h_conditions : triangle_ABC_angle_A a b B A acute ha hb hB hacute) : 
  A = Real.pi / 3 := 
sorry

end find_angle_A_l100_10031


namespace aquafaba_needed_for_cakes_l100_10035

def tablespoons_of_aquafaba_for_egg_whites (n_egg_whites : ℕ) : ℕ :=
  2 * n_egg_whites

def total_egg_whites_needed (cakes : ℕ) (egg_whites_per_cake : ℕ) : ℕ :=
  cakes * egg_whites_per_cake

theorem aquafaba_needed_for_cakes (cakes : ℕ) (egg_whites_per_cake : ℕ) :
  tablespoons_of_aquafaba_for_egg_whites (total_egg_whites_needed cakes egg_whites_per_cake) = 32 :=
by
  have h1 : cakes = 2 := sorry
  have h2 : egg_whites_per_cake = 8 := sorry
  sorry

end aquafaba_needed_for_cakes_l100_10035


namespace ratio_x_y_z_l100_10044

theorem ratio_x_y_z (x y z : ℝ) (h1 : 0.10 * x = 0.20 * y) (h2 : 0.30 * y = 0.40 * z) :
  ∃ k : ℝ, x = 8 * k ∧ y = 4 * k ∧ z = 3 * k :=
by                         
  sorry

end ratio_x_y_z_l100_10044


namespace trig_identity_l100_10070

-- Define the given condition
def tan_half (α : ℝ) : Prop := Real.tan (α / 2) = 2

-- The main statement we need to prove
theorem trig_identity (α : ℝ) (h : tan_half α) : (1 + Real.cos α) / (Real.sin α) = 1 / 2 :=
  by
  sorry

end trig_identity_l100_10070


namespace parameter_range_exists_solution_l100_10032

theorem parameter_range_exists_solution :
  (∃ b : ℝ, -14 < b ∧ b < 9 ∧ ∃ a : ℝ, ∃ x y : ℝ,
    x^2 + y^2 + 2 * b * (b + x + y) = 81 ∧ y = 5 / ((x - a)^2 + 1)) :=
sorry

end parameter_range_exists_solution_l100_10032


namespace fraction_multiplication_simplifies_l100_10092

theorem fraction_multiplication_simplifies :
  (3 : ℚ) / 4 * (4 / 5) * (2 / 3) = 2 / 5 := 
by 
  -- Prove the equality step-by-step
  sorry

end fraction_multiplication_simplifies_l100_10092


namespace money_distribution_l100_10023

theorem money_distribution (a : ℕ) (h1 : 5 * a = 1500) : 7 * a - 3 * a = 1200 := by
  sorry

end money_distribution_l100_10023


namespace selection_problem_l100_10090

def group_size : ℕ := 10
def selected_group_size : ℕ := 3
def total_ways_without_C := Nat.choose 9 3
def ways_without_A_B_C := Nat.choose 7 3
def correct_answer := total_ways_without_C - ways_without_A_B_C

theorem selection_problem:
  (∃ (A B C : ℕ), total_ways_without_C - ways_without_A_B_C = 49) :=
by
  sorry

end selection_problem_l100_10090


namespace inequality_ineq_l100_10078

variable (x y z : Real)

theorem inequality_ineq {x y z : Real} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = 3) :
  (1 / (x^5 - x^2 + 3)) + (1 / (y^5 - y^2 + 3)) + (1 / (z^5 - z^2 + 3)) ≤ 1 :=
by 
  sorry

end inequality_ineq_l100_10078


namespace merchant_profit_l100_10076

theorem merchant_profit 
  (CP MP SP profit : ℝ)
  (markup_percentage discount_percentage : ℝ)
  (h1 : CP = 100)
  (h2 : markup_percentage = 0.40)
  (h3 : discount_percentage = 0.10)
  (h4 : MP = CP + (markup_percentage * CP))
  (h5 : SP = MP - (discount_percentage * MP))
  (h6 : profit = SP - CP) :
  profit / CP * 100 = 26 :=
by sorry

end merchant_profit_l100_10076
