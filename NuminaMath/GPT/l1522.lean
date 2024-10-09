import Mathlib

namespace visitors_on_that_day_l1522_152272

theorem visitors_on_that_day 
  (prev_visitors : ℕ) 
  (additional_visitors : ℕ) 
  (h1 : prev_visitors = 100)
  (h2 : additional_visitors = 566)
  : prev_visitors + additional_visitors = 666 := by
  sorry

end visitors_on_that_day_l1522_152272


namespace irrational_number_among_choices_l1522_152216

theorem irrational_number_among_choices : ∃ x ∈ ({17/6, -27/100, 0, Real.sqrt 2} : Set ℝ), Irrational x ∧ x = Real.sqrt 2 := by
  sorry

end irrational_number_among_choices_l1522_152216


namespace shrimp_appetizer_cost_l1522_152259

-- Define the conditions
def shrimp_per_guest : ℕ := 5
def number_of_guests : ℕ := 40
def cost_per_pound : ℕ := 17
def shrimp_per_pound : ℕ := 20

-- Define the proof statement
theorem shrimp_appetizer_cost : 
  (shrimp_per_guest * number_of_guests / shrimp_per_pound) * cost_per_pound = 170 := 
by
  sorry

end shrimp_appetizer_cost_l1522_152259


namespace compute_expression_l1522_152252

theorem compute_expression : 20 * (150 / 3 + 36 / 4 + 4 / 25 + 2) = 1223 + 1/5 :=
by
  sorry

end compute_expression_l1522_152252


namespace find_a_value_l1522_152234

theorem find_a_value (a : ℝ) (A B : Set ℝ) (hA : A = {3, 5}) (hB : B = {x | a * x - 1 = 0}) :
  B ⊆ A → a = 0 ∨ a = 1/3 ∨ a = 1/5 :=
by sorry

end find_a_value_l1522_152234


namespace exist_N_for_fn_eq_n_l1522_152265

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom f_condition1 (m n : ℕ+) : (f m, f n) ≤ (m, n) ^ 2014
axiom f_condition2 (n : ℕ+) : n ≤ f n ∧ f n ≤ n + 2014

theorem exist_N_for_fn_eq_n :
  ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → f n = n := sorry

end exist_N_for_fn_eq_n_l1522_152265


namespace milk_price_increase_day_l1522_152246

theorem milk_price_increase_day (total_cost : ℕ) (old_price : ℕ) (new_price : ℕ) (days : ℕ) (x : ℕ)
    (h1 : old_price = 1500)
    (h2 : new_price = 1600)
    (h3 : days = 30)
    (h4 : total_cost = 46200)
    (h5 : (x - 1) * old_price + (days + 1 - x) * new_price = total_cost) :
  x = 19 :=
by
  sorry

end milk_price_increase_day_l1522_152246


namespace fodder_lasting_days_l1522_152282

theorem fodder_lasting_days (buffalo_fodder_rate cow_fodder_rate ox_fodder_rate : ℕ)
  (initial_buffaloes initial_cows initial_oxen added_buffaloes added_cows initial_days : ℕ)
  (h1 : 3 * buffalo_fodder_rate = 4 * cow_fodder_rate)
  (h2 : 3 * buffalo_fodder_rate = 2 * ox_fodder_rate)
  (h3 : initial_days * (initial_buffaloes * buffalo_fodder_rate + initial_cows * cow_fodder_rate + initial_oxen * ox_fodder_rate) = 4320) :
  (4320 / ((initial_buffaloes + added_buffaloes) * buffalo_fodder_rate + (initial_cows + added_cows) * cow_fodder_rate + initial_oxen * ox_fodder_rate)) = 9 :=
by 
  sorry

end fodder_lasting_days_l1522_152282


namespace time_per_employee_updating_payroll_records_l1522_152233

-- Define the conditions
def minutes_making_coffee : ℕ := 5
def minutes_per_employee_status_update : ℕ := 2
def num_employees : ℕ := 9
def total_morning_routine_minutes : ℕ := 50

-- Define the proof statement encapsulating the problem
theorem time_per_employee_updating_payroll_records :
  (total_morning_routine_minutes - (minutes_making_coffee + minutes_per_employee_status_update * num_employees)) / num_employees = 3 := by
  sorry

end time_per_employee_updating_payroll_records_l1522_152233


namespace solve_for_x_l1522_152214

theorem solve_for_x : ∃ x : ℤ, 25 - (4 + 3) = 5 + x ∧ x = 13 :=
by {
  sorry
}

end solve_for_x_l1522_152214


namespace arith_seq_fraction_l1522_152206

theorem arith_seq_fraction (a : ℕ → ℝ) (d : ℝ) (h1 : ∀ n, a (n + 1) - a n = d)
  (h2 : d ≠ 0) (h3 : a 3 = 2 * a 1) :
  (a 1 + a 3) / (a 2 + a 4) = 3 / 4 :=
sorry

end arith_seq_fraction_l1522_152206


namespace brokerage_percentage_correct_l1522_152262

noncomputable def brokerage_percentage (market_value : ℝ) (income : ℝ) (investment : ℝ) (nominal_rate : ℝ) : ℝ :=
  let face_value := (income * 100) / nominal_rate
  let market_price := (face_value * market_value) / 100
  let brokerage_amount := investment - market_price
  (brokerage_amount / investment) * 100

theorem brokerage_percentage_correct :
  brokerage_percentage 110.86111111111111 756 8000 10.5 = 0.225 :=
by
  sorry

end brokerage_percentage_correct_l1522_152262


namespace lockers_number_l1522_152249

theorem lockers_number (total_cost : ℝ) (cost_per_digit : ℝ) (total_lockers : ℕ) 
  (locker_numbered_from_one : ∀ n : ℕ, n >= 1) :
  total_cost = 248.43 → cost_per_digit = 0.03 → total_lockers = 2347 :=
by
  intros h_total_cost h_cost_per_digit
  sorry

end lockers_number_l1522_152249


namespace find_y_value_l1522_152202

theorem find_y_value (k c x y : ℝ) (h1 : c = 3) 
                     (h2 : ∀ x : ℝ, y = k * x + c)
                     (h3 : ∃ k : ℝ, 15 = k * 5 + 3) :
  y = -21 :=
by 
  sorry

end find_y_value_l1522_152202


namespace domain_of_sqrt_l1522_152211

theorem domain_of_sqrt (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end domain_of_sqrt_l1522_152211


namespace ratio_of_interior_to_exterior_angle_in_regular_octagon_l1522_152260

theorem ratio_of_interior_to_exterior_angle_in_regular_octagon
  (n : ℕ) (regular_polygon : n = 8) : 
  let interior_angle := ((n - 2) * 180) / n
  let exterior_angle := 360 / n
  (interior_angle / exterior_angle) = 3 :=
by
  sorry

end ratio_of_interior_to_exterior_angle_in_regular_octagon_l1522_152260


namespace x_is_integer_l1522_152258

theorem x_is_integer
  (x : ℝ)
  (h_pos : 0 < x)
  (h1 : ∃ k1 : ℤ, x^2012 = x^2001 + k1)
  (h2 : ∃ k2 : ℤ, x^2001 = x^1990 + k2) : 
  ∃ n : ℤ, x = n :=
sorry

end x_is_integer_l1522_152258


namespace sum_three_numbers_l1522_152205

noncomputable def sum_of_three_numbers (a b c : ℝ) : ℝ :=
  a + b + c

theorem sum_three_numbers 
  (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = a + 20) 
  (h2 : (a + b + c) / 3 = c - 30) 
  (h3 : b = 10) :
  sum_of_three_numbers a b c = 60 :=
by
  sorry

end sum_three_numbers_l1522_152205


namespace capacity_of_initial_20_buckets_l1522_152254

theorem capacity_of_initial_20_buckets (x : ℝ) (h : 20 * x = 270) : x = 13.5 :=
by 
  sorry

end capacity_of_initial_20_buckets_l1522_152254


namespace money_distribution_l1522_152228

theorem money_distribution (a b c : ℝ) (h1 : 4 * (a - b - c) = 16)
                           (h2 : 6 * b - 2 * a - 2 * c = 16)
                           (h3 : 7 * c - a - b = 16) :
  a = 29 := 
by 
  sorry

end money_distribution_l1522_152228


namespace power_sum_identity_l1522_152256

theorem power_sum_identity (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) : 
  (a^7 + b^7 + c^7)^2 / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49 / 60 := 
by
  sorry

end power_sum_identity_l1522_152256


namespace Megatech_budget_allocation_l1522_152251

theorem Megatech_budget_allocation :
  let total_degrees := 360
  let degrees_astrophysics := 90
  let home_electronics := 19
  let food_additives := 10
  let genetically_modified_microorganisms := 24
  let industrial_lubricants := 8

  let percentage_astrophysics := (degrees_astrophysics / total_degrees) * 100
  let known_percentages_sum := home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants + percentage_astrophysics
  let percentage_microphotonics := 100 - known_percentages_sum

  percentage_microphotonics = 14 :=
by
  sorry

end Megatech_budget_allocation_l1522_152251


namespace teal_more_blue_proof_l1522_152266

theorem teal_more_blue_proof (P G B N : ℕ) (hP : P = 150) (hG : G = 90) (hB : B = 40) (hN : N = 25) : 
  (∃ (x : ℕ), x = 75) :=
by
  sorry

end teal_more_blue_proof_l1522_152266


namespace chandra_pairings_l1522_152295

theorem chandra_pairings : 
  let bowls := 5
  let glasses := 6
  (bowls * glasses) = 30 :=
by
  sorry

end chandra_pairings_l1522_152295


namespace smallest_t_l1522_152294

theorem smallest_t (p q r : ℕ) (h₁ : 0 < p) (h₂ : 0 < q) (h₃ : 0 < r) (h₄ : p + q + r = 2510) 
                   (k : ℕ) (t : ℕ) (h₅ : p! * q! * r! = k * 10^t) (h₆ : ¬(10 ∣ k)) : t = 626 := 
by sorry

end smallest_t_l1522_152294


namespace line_through_points_l1522_152267

theorem line_through_points :
  ∀ x y : ℝ, (∃ t : ℝ, (x, y) = (2 * t, -3 * (1 - t))) ↔ (x / 2) - (y / 3) = 1 :=
by
  sorry

end line_through_points_l1522_152267


namespace mean_of_S_eq_651_l1522_152200

theorem mean_of_S_eq_651 
  (s n : ℝ) 
  (h1 : (s + 1) / (n + 1) = s / n - 13) 
  (h2 : (s + 2001) / (n + 1) = s / n + 27) 
  (hn : n ≠ 0) : s / n = 651 := 
by 
  sorry

end mean_of_S_eq_651_l1522_152200


namespace sum_b4_b6_l1522_152293

theorem sum_b4_b6
  (b : ℕ → ℝ)
  (h₁ : ∀ n : ℕ, n > 0 → ∃ d : ℝ, ∀ m : ℕ, m > 0 → (1 / b (m + 1) - 1 / b m) = d)
  (h₂ : b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 = 90) :
  b 4 + b 6 = 20 := by
  sorry

end sum_b4_b6_l1522_152293


namespace sufficient_but_not_necessary_condition_l1522_152237

theorem sufficient_but_not_necessary_condition 
  (a b : ℝ) (h : a > b ∧ b > 0) : (a^2 > b^2) ∧ (¬ ∀ (a' b' : ℝ), a'^2 > b'^2 → a' > b' ∧ b' > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1522_152237


namespace exists_x_l1522_152215

noncomputable def g (x : ℝ) : ℝ := (2 / 7) ^ x + (3 / 7) ^ x + (6 / 7) ^ x

theorem exists_x (x : ℝ) : ∃ c : ℝ, g c = 1 :=
sorry

end exists_x_l1522_152215


namespace quadratic_inequality_solution_l1522_152280

open Real

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 15 < 0) : 3 < x ∧ x < 5 :=
sorry

end quadratic_inequality_solution_l1522_152280


namespace erica_duration_is_correct_l1522_152238

-- Define the durations for Dave, Chuck, and Erica
def dave_duration : ℝ := 10
def chuck_duration : ℝ := 5 * dave_duration
def erica_duration : ℝ := chuck_duration + 0.30 * chuck_duration

-- State the theorem
theorem erica_duration_is_correct : erica_duration = 65 := by
  sorry

end erica_duration_is_correct_l1522_152238


namespace count_triangles_with_center_inside_l1522_152244

theorem count_triangles_with_center_inside :
  let n := 201
  let num_triangles_with_center_inside (n : ℕ) : ℕ := 
    let half := n / 2
    let group_count := half * (half + 1) / 2
    group_count * n / 3
  num_triangles_with_center_inside n = 338350 :=
by
  sorry

end count_triangles_with_center_inside_l1522_152244


namespace negation_proof_l1522_152299

theorem negation_proof (x : ℝ) : ¬ (x^2 - x + 3 > 0) ↔ (x^2 - x + 3 ≤ 0) :=
by sorry

end negation_proof_l1522_152299


namespace geraldine_more_than_jazmin_l1522_152226

def geraldine_dolls : ℝ := 2186.0
def jazmin_dolls : ℝ := 1209.0
def difference_dolls : ℝ := 977.0

theorem geraldine_more_than_jazmin : geraldine_dolls - jazmin_dolls = difference_dolls :=
by sorry

end geraldine_more_than_jazmin_l1522_152226


namespace ratio_abc_xyz_l1522_152222

theorem ratio_abc_xyz
  (a b c x y z : ℝ)
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z) 
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a * x + b * y + c * z = 56) :
  (a + b + c) / (x + y + z) = 7 / 8 := 
sorry

end ratio_abc_xyz_l1522_152222


namespace max_elves_without_caps_proof_max_elves_with_caps_proof_l1522_152277

-- Defining the conditions and the problem statement
open Nat

-- We model the problem with the following:
axiom truth_teller : Type
axiom liar_with_caps : Type
axiom dwarf_with_caps : Type
axiom dwarf_without_caps : Type

noncomputable def max_elves_without_caps : ℕ :=
  59

noncomputable def max_elves_with_caps : ℕ :=
  30

-- Part (a): Given the conditions, we show that the maximum number of elves without caps is 59
theorem max_elves_without_caps_proof : max_elves_without_caps = 59 :=
by
  sorry

-- Part (b): Given the conditions, we show that the maximum number of elves with caps is 30
theorem max_elves_with_caps_proof : max_elves_with_caps = 30 :=
by
  sorry

end max_elves_without_caps_proof_max_elves_with_caps_proof_l1522_152277


namespace inequality_proof_l1522_152245

theorem inequality_proof
  (a b c d e f : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (he : 0 < e)
  (hf : 0 < f)
  (h_condition : abs (Real.sqrt (a * b) - Real.sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := 
sorry

end inequality_proof_l1522_152245


namespace sunil_total_amount_proof_l1522_152227

theorem sunil_total_amount_proof
  (CI : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) (P : ℝ) (A : ℝ)
  (h1 : CI = 492)
  (h2 : r = 0.05)
  (h3 : n = 1)
  (h4 : t = 2)
  (h5 : CI = P * ((1 + r / n) ^ (n * t) - 1))
  (h6 : A = P + CI) :
  A = 5292 :=
by
  -- Skip the proof.
  sorry

end sunil_total_amount_proof_l1522_152227


namespace distinct_prime_divisors_l1522_152255

theorem distinct_prime_divisors (a : ℤ) (n : ℕ) (h₁ : a > 3) (h₂ : Odd a) (h₃ : n > 0) : 
  ∃ (p : Finset ℤ), p.card ≥ n + 1 ∧ ∀ q ∈ p, Prime q ∧ q ∣ (a ^ (2 ^ n) - 1) :=
sorry

end distinct_prime_divisors_l1522_152255


namespace range_of_m_l1522_152298

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 4 * cos x + sin x ^ 2 + m - 4 = 0) ↔ 0 ≤ m ∧ m ≤ 8 :=
sorry

end range_of_m_l1522_152298


namespace solution_set_inequality_l1522_152247

theorem solution_set_inequality (x : ℝ) :
  (|x + 3| - |x - 3| > 3) ↔ (x > 3 / 2) := 
sorry

end solution_set_inequality_l1522_152247


namespace pow_evaluation_l1522_152248

theorem pow_evaluation : 81^(5/4) = 243 := 
by sorry

end pow_evaluation_l1522_152248


namespace f_7_minus_a_eq_neg_7_over_4_l1522_152264

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 2 else -Real.logb 3 x

variable (a : ℝ)

-- Given conditions
axiom h1 : f a = -2

-- The proof of the required condition
theorem f_7_minus_a_eq_neg_7_over_4 (h1 : f a = -2) : f (7 - a) = -7 / 4 := sorry

end f_7_minus_a_eq_neg_7_over_4_l1522_152264


namespace part_I_part_II_l1522_152273

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 + 4 * a * x - 3

-- Part (I)
theorem part_I (a : ℝ) (h_a : a > 0) (h_roots: ∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ f a x1 = 0 ∧ f a x2 = 0) : 
  0 < a ∧ a < 2 / 5 :=
sorry

-- Part (II)
theorem part_II (a : ℝ) (h_max : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f a x ≤ f a 2) : 
  a ≥ -1 / 3 :=
sorry

end part_I_part_II_l1522_152273


namespace cos_330_eq_sqrt3_div_2_l1522_152218

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l1522_152218


namespace middle_number_is_10_l1522_152289

theorem middle_number_is_10 (x y z : ℤ) (hx : x < y) (hy : y < z) 
    (h1 : x + y = 18) (h2 : x + z = 25) (h3 : y + z = 27) : y = 10 :=
by 
  sorry

end middle_number_is_10_l1522_152289


namespace largest_value_of_a_l1522_152288

theorem largest_value_of_a
  (a b c d e : ℕ)
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : e = d - 10)
  (h5 : e < 105) :
  a ≤ 6824 :=
by {
  -- Proof omitted
  sorry
}

end largest_value_of_a_l1522_152288


namespace cost_price_is_correct_l1522_152275

-- Define the conditions
def purchasing_clocks : ℕ := 150
def gain_60_clocks : ℝ := 0.12
def gain_90_clocks : ℝ := 0.18
def uniform_profit : ℝ := 0.16
def difference_in_profit : ℝ := 75

-- Define the cost price of each clock
noncomputable def C : ℝ := 125

-- Define and state the theorem
theorem cost_price_is_correct (C : ℝ) :
  (60 * C * (1 + gain_60_clocks) + 90 * C * (1 + gain_90_clocks)) - (150 * C * (1 + uniform_profit)) = difference_in_profit :=
sorry

end cost_price_is_correct_l1522_152275


namespace calculate_highest_score_l1522_152271

noncomputable def highest_score (avg_60 : ℕ) (delta_HL : ℕ) (avg_58 : ℕ) : ℕ :=
  let total_60 := 60 * avg_60
  let total_58 := 58 * avg_58
  let sum_HL := total_60 - total_58
  let L := (sum_HL - delta_HL) / 2
  let H := L + delta_HL
  H

theorem calculate_highest_score :
  highest_score 55 200 52 = 242 :=
by
  sorry

end calculate_highest_score_l1522_152271


namespace find_four_digit_number_abcd_exists_l1522_152236

theorem find_four_digit_number_abcd_exists (M : ℕ) (H1 : M > 0) (H2 : M % 10 ≠ 0) 
    (H3 : M % 100000 = M^2 % 100000) : ∃ abcd : ℕ, abcd = 2502 :=
by
  -- Proof is omitted
  sorry

end find_four_digit_number_abcd_exists_l1522_152236


namespace calculate_leakage_rate_l1522_152225

variable (B : ℕ) (T : ℕ) (R : ℝ)

-- B represents the bucket's capacity in ounces, T represents time in hours, R represents the rate of leakage per hour in ounces per hour.

def leakage_rate (B : ℕ) (T : ℕ) (R : ℝ) : Prop :=
  (B = 36) ∧ (T = 12) ∧ (B / 2 = T * R)

theorem calculate_leakage_rate : leakage_rate 36 12 1.5 :=
by 
  simp [leakage_rate]
  sorry

end calculate_leakage_rate_l1522_152225


namespace west_move_7m_l1522_152201

-- Definitions and conditions
def east_move (distance : Int) : Int := distance -- Moving east
def west_move (distance : Int) : Int := -distance -- Moving west is represented as negative

-- Problem: Prove that moving west by 7m is denoted by -7m given the conditions.
theorem west_move_7m : west_move 7 = -7 :=
by
  -- Proof will be handled here normally, but it's omitted as per instruction
  sorry

end west_move_7m_l1522_152201


namespace ken_change_l1522_152261

theorem ken_change (cost_per_pound : ℕ) (quantity : ℕ) (amount_paid : ℕ) (total_cost : ℕ) (change : ℕ) 
(h1 : cost_per_pound = 7)
(h2 : quantity = 2)
(h3 : amount_paid = 20)
(h4 : total_cost = cost_per_pound * quantity)
(h5 : change = amount_paid - total_cost) : change = 6 :=
by 
  sorry

end ken_change_l1522_152261


namespace base_of_exponent_l1522_152243

theorem base_of_exponent (x : ℤ) (m : ℕ) (h₁ : (-2 : ℤ)^(2 * m) = x^(12 - m)) (h₂ : m = 4) : x = -2 :=
by 
  sorry

end base_of_exponent_l1522_152243


namespace find_c_l1522_152240

-- Define the two points as given in the problem
def pointA : ℝ × ℝ := (-6, 1)
def pointB : ℝ × ℝ := (-3, 4)

-- Define the direction vector as subtraction of the two points
def directionVector : ℝ × ℝ := (pointB.1 - pointA.1, pointB.2 - pointA.2)

-- Define the target direction vector format with unknown c
def targetDirectionVector (c : ℝ) : ℝ × ℝ := (3, c)

-- The theorem stating that c must be 3
theorem find_c : ∃ c : ℝ, directionVector = targetDirectionVector c ∧ c = 3 := 
by
  -- Prove the statement or show it is derivable
  sorry

end find_c_l1522_152240


namespace not_possible_to_color_l1522_152290

theorem not_possible_to_color (f : ℕ → ℕ) (c1 c2 c3 : ℕ) :
  ∃ (x : ℕ), 1 < x ∧ f 2 = c1 ∧ f 4 = c1 ∧ 
  ∀ (a b : ℕ), 1 < a → 1 < b → f a ≠ f b → (f (a * b) ≠ f a ∧ f (a * b) ≠ f b) → 
  false :=
sorry

end not_possible_to_color_l1522_152290


namespace maria_cookies_l1522_152239

theorem maria_cookies :
  let c_initial := 19
  let c1 := c_initial - 5
  let c2 := c1 / 2
  let c_final := c2 - 2
  c_final = 5 :=
by
  sorry

end maria_cookies_l1522_152239


namespace find_n_l1522_152253

def alpha (n : ℕ) : ℚ := ((n - 2) * 180) / n
def alpha_plus_3 (n : ℕ) : ℚ := ((n + 1) * 180) / (n + 3)
def alpha_minus_2 (n : ℕ) : ℚ := ((n - 4) * 180) / (n - 2)

theorem find_n (n : ℕ) (h : alpha_plus_3 n - alpha n = alpha n - alpha_minus_2 n) : n = 12 :=
by
  -- The proof will be added here
  sorry

end find_n_l1522_152253


namespace sum_of_roots_abs_eqn_zero_l1522_152210

theorem sum_of_roots_abs_eqn_zero (x : ℝ) (hx : |x|^2 - 4*|x| - 5 = 0) : (5 + (-5) = 0) :=
  sorry

end sum_of_roots_abs_eqn_zero_l1522_152210


namespace decimal_to_binary_49_l1522_152270

theorem decimal_to_binary_49 : ((49:ℕ) = 6 * 2^4 + 3 * 2^3 + 0 * 2^2 + 0 * 2^1 + 0 * 2^0 + 1) ↔ (110001 = 110001) :=
by
  sorry

end decimal_to_binary_49_l1522_152270


namespace sum_divisible_by_3_l1522_152269

theorem sum_divisible_by_3 (a : ℤ) : 3 ∣ (a^3 + 2 * a) :=
sorry

end sum_divisible_by_3_l1522_152269


namespace part_a_part_b_l1522_152285

-- Part (a)
theorem part_a {x y n : ℕ} (h : x^3 + 2^n * y = y^3 + 2^n * x) : x = y :=
sorry

-- Part (b)
theorem part_b {x y : ℤ} {n : ℕ} (h : x ≠ 0 ∧ y ≠ 0 ∧ x^3 + 2^n * y = y^3 + 2^n * x) : |x| = |y| :=
sorry

end part_a_part_b_l1522_152285


namespace mutually_exclusive_events_l1522_152203

-- Define the bag, balls, and events
def bag := (5, 3) -- (red balls, white balls)

def draws (r w : Nat) := (r + w = 3)

def event_A (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.1 = 3 -- At least one red ball and all red balls
def event_B (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.2 = 3 -- At least one red ball and all white balls
def event_C (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.2 ≥ 1 -- At least one red ball and at least one white ball
def event_D (draw : ℕ × ℕ) := (draw.1 = 1 ∨ draw.1 = 2) ∧ draws draw.1 draw.2 -- Exactly one red ball and exactly two red balls

theorem mutually_exclusive_events : 
  ∀ draw : ℕ × ℕ, 
  (event_A draw ∨ event_B draw ∨ event_C draw ∨ event_D draw) → 
  (event_D draw ↔ (draw.1 = 1 ∧ draw.2 = 2) ∨ (draw.1 = 2 ∧ draw.2 = 1)) :=
by
  sorry

end mutually_exclusive_events_l1522_152203


namespace g_nested_result_l1522_152263

def g (n : ℕ) : ℕ :=
if n < 5 then
  n^2 + 1
else
  2 * n + 3

theorem g_nested_result : g (g (g 3)) = 49 := by
sorry

end g_nested_result_l1522_152263


namespace total_gas_cost_l1522_152257

def car_city_mpg : ℝ := 30
def car_highway_mpg : ℝ := 40
def city_miles : ℝ := 60 + 40 + 25
def highway_miles : ℝ := 200 + 150 + 180
def gas_price_per_gallon : ℝ := 3.00

theorem total_gas_cost : 
  (city_miles / car_city_mpg + highway_miles / car_highway_mpg) * gas_price_per_gallon = 52.25 := 
by
  sorry

end total_gas_cost_l1522_152257


namespace solution_set_of_inequality_l1522_152220

theorem solution_set_of_inequality {x : ℝ} :
  {x | |x| * (1 - 2 * x) > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | 0 < x ∧ x < 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l1522_152220


namespace find_weights_l1522_152223

def item_weights (a b c d e f g h : ℕ) : Prop :=
  1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧ 1 ≤ e ∧ 1 ≤ f ∧ 1 ≤ g ∧ 1 ≤ h ∧
  a > b ∧ b > c ∧ c > d ∧ d > e ∧ e > f ∧ f > g ∧ g > h ∧
  a ≤ 15 ∧ b ≤ 15 ∧ c ≤ 15 ∧ d ≤ 15 ∧ e ≤ 15 ∧ f ≤ 15 ∧ g ≤ 15 ∧ h ≤ 15

theorem find_weights (a b c d e f g h : ℕ) (hw : item_weights a b c d e f g h) 
    (h1 : d + e + f + g > a + b + c + h) 
    (h2 : e + f > d + g) 
    (h3 : e > f) : e = 11 ∧ g = 5 := sorry

end find_weights_l1522_152223


namespace shirts_sewn_on_tuesday_l1522_152232

theorem shirts_sewn_on_tuesday 
  (shirts_monday : ℕ) 
  (shirts_wednesday : ℕ) 
  (total_buttons : ℕ) 
  (buttons_per_shirt : ℕ) 
  (shirts_tuesday : ℕ) 
  (h1: shirts_monday = 4) 
  (h2: shirts_wednesday = 2) 
  (h3: total_buttons = 45) 
  (h4: buttons_per_shirt = 5) 
  (h5: shirts_tuesday * buttons_per_shirt + shirts_monday * buttons_per_shirt + shirts_wednesday * buttons_per_shirt = total_buttons) : 
  shirts_tuesday = 3 :=
by 
  sorry

end shirts_sewn_on_tuesday_l1522_152232


namespace sum_of_squares_of_roots_l1522_152231

/-- If r, s, and t are the roots of the cubic equation x³ - ax² + bx - c = 0, then r² + s² + t² = a² - 2b. -/
theorem sum_of_squares_of_roots (r s t a b c : ℝ) (h1 : r + s + t = a) (h2 : r * s + r * t + s * t = b) (h3 : r * s * t = c) :
    r ^ 2 + s ^ 2 + t ^ 2 = a ^ 2 - 2 * b := 
by 
  sorry

end sum_of_squares_of_roots_l1522_152231


namespace combined_average_score_girls_l1522_152229

open BigOperators

variable (A a B b C c : ℕ) -- number of boys and girls at each school
variable (x : ℕ) -- common value for number of boys and girls

axiom Adams_HS : 74 * (A : ℤ) + 81 * (a : ℤ) = 77 * (A + a)
axiom Baker_HS : 83 * (B : ℤ) + 92 * (b : ℤ) = 86 * (B + b)
axiom Carter_HS : 78 * (C : ℤ) + 85 * (c : ℤ) = 80 * (C + c)

theorem combined_average_score_girls :
  (A = a ∧ B = b ∧ C = c) →
  (A = B ∧ B = C) →
  (81 * (A : ℤ) + 92 * (B : ℤ) + 85 * (C : ℤ)) / (A + B + C) = 86 := 
by
  intro h1 h2
  sorry

end combined_average_score_girls_l1522_152229


namespace solve_logarithmic_equation_l1522_152217

/-- The solution to the equation log_2(9^x - 5) = 2 + log_2(3^x - 2) is x = 1. -/
theorem solve_logarithmic_equation (x : ℝ) :
  (Real.logb 2 (9^x - 5) = 2 + Real.logb 2 (3^x - 2)) → x = 1 :=
by
  sorry

end solve_logarithmic_equation_l1522_152217


namespace larger_interior_angle_trapezoid_pavilion_l1522_152224

theorem larger_interior_angle_trapezoid_pavilion :
  let n := 12
  let central_angle := 360 / n
  let smaller_angle := 180 - (central_angle / 2)
  let larger_angle := 180 - smaller_angle
  larger_angle = 97.5 :=
by
  sorry

end larger_interior_angle_trapezoid_pavilion_l1522_152224


namespace probability_of_F_l1522_152292

-- Definitions for the probabilities of regions D, E, and the total probability
def P_D : ℚ := 3 / 8
def P_E : ℚ := 1 / 4
def total_probability : ℚ := 1

-- The hypothesis
lemma total_probability_eq_one : P_D + P_E + (1 - P_D - P_E) = total_probability :=
by
  simp [P_D, P_E, total_probability]

-- The goal is to prove this statement
theorem probability_of_F : 1 - P_D - P_E = 3 / 8 :=
by
  -- Using the total_probability_eq_one hypothesis
  have h := total_probability_eq_one
  -- This is a structured approach where verification using hypothesis and simplification can be done
  sorry

end probability_of_F_l1522_152292


namespace trigonometric_inequality_l1522_152278

-- Let \( f(x) \) be defined as \( cos \, x \)
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- Given a, b, c are the sides of triangle ∆ABC opposite to angles A, B, C respectively
variables {a b c A B C : ℝ}

-- Condition: \( 3a^2 + 3b^2 - c^2 = 4ab \)
variable (h : 3 * a^2 + 3 * b^2 - c^2 = 4 * a * b)

-- Goal: Prove that \( f(\cos A) \leq f(\sin B) \)
theorem trigonometric_inequality (h1 : A + B + C = π) (h2 : a^2 + b^2 - 2 * a * b * Real.cos C = c^2) : 
  f (Real.cos A) ≤ f (Real.sin B) :=
by
  sorry

end trigonometric_inequality_l1522_152278


namespace perpendicular_tangents_add_l1522_152274

open Real

noncomputable def f1 (x : ℝ): ℝ := x^2 - 2 * x + 2
noncomputable def f2 (x : ℝ) (a : ℝ) (b : ℝ): ℝ := -x^2 + a * x + b

-- Definitions of derivatives for the given functions
noncomputable def f1' (x : ℝ): ℝ := 2 * x - 2
noncomputable def f2' (x : ℝ) (a : ℝ): ℝ := -2 * x + a

theorem perpendicular_tangents_add (x0 y0 a b : ℝ)
  (h1 : y0 = f1 x0)
  (h2 : y0 = f2 x0 a b)
  (h3 : f1' x0 * f2' x0 a = -1) :
  a + b = 5 / 2 := sorry

end perpendicular_tangents_add_l1522_152274


namespace probability_same_tribe_l1522_152279

def totalPeople : ℕ := 18
def peoplePerTribe : ℕ := 6
def tribes : ℕ := 3
def totalQuitters : ℕ := 2

def totalWaysToChooseQuitters := Nat.choose totalPeople totalQuitters
def waysToChooseFromTribe := Nat.choose peoplePerTribe totalQuitters
def totalWaysFromSameTribe := tribes * waysToChooseFromTribe

theorem probability_same_tribe (h1 : totalPeople = 18) (h2 : peoplePerTribe = 6) (h3 : tribes = 3) (h4 : totalQuitters = 2)
    (h5 : totalWaysToChooseQuitters = 153) (h6 : totalWaysFromSameTribe = 45) :
    (totalWaysFromSameTribe : ℚ) / totalWaysToChooseQuitters = 5 / 17 := by
  sorry

end probability_same_tribe_l1522_152279


namespace lisa_ratio_l1522_152208

theorem lisa_ratio (L J T : ℝ) 
  (h1 : L + J + T = 60) 
  (h2 : T = L / 2) 
  (h3 : L = T + 15) : 
  L / 60 = 1 / 2 :=
by 
  sorry

end lisa_ratio_l1522_152208


namespace student_2005_says_1_l1522_152242

def pattern : List ℕ := [1, 2, 3, 4, 3, 2]

def nth_number_in_pattern (n : ℕ) : ℕ :=
  List.nthLe pattern (n % 6) sorry  -- The index is (n-1) % 6 because Lean indices start at 0

theorem student_2005_says_1 : nth_number_in_pattern 2005 = 1 := 
  by
  -- The proof goes here
  sorry

end student_2005_says_1_l1522_152242


namespace value_of_expression_l1522_152291

theorem value_of_expression : ((25 + 8)^2 - (8^2 + 25^2) = 400) :=
by 
  sorry

end value_of_expression_l1522_152291


namespace expand_product_l1522_152296

theorem expand_product : (2 : ℝ) * (x + 2) * (x + 3) * (x + 4) = 2 * x^3 + 18 * x^2 + 52 * x + 48 :=
by
  sorry

end expand_product_l1522_152296


namespace water_usage_l1522_152276

def fee (x : ℕ) : ℕ :=
  if x ≤ 8 then 2 * x else 4 * x - 16

theorem water_usage (h : fee 9 = 20) : fee 9 = 20 := by
  sorry

end water_usage_l1522_152276


namespace factor_of_quadratic_polynomial_l1522_152213

theorem factor_of_quadratic_polynomial (t : ℚ) :
  (8 * t^2 + 22 * t + 5 = 0) ↔ (t = -1/4) ∨ (t = -5/2) :=
by sorry

end factor_of_quadratic_polynomial_l1522_152213


namespace max_cables_cut_l1522_152221

def initial_cameras : ℕ := 200
def initial_cables : ℕ := 345
def resulting_clusters : ℕ := 8

theorem max_cables_cut :
  ∃ (cables_cut : ℕ), resulting_clusters = 8 ∧ initial_cables - cables_cut = (initial_cables - cables_cut) - (resulting_clusters - 1) ∧ cables_cut = 153 :=
by
  sorry

end max_cables_cut_l1522_152221


namespace log_expansion_l1522_152297

theorem log_expansion (a : ℝ) (h : a = Real.log 4 / Real.log 5) : Real.log 64 / Real.log 5 - 2 * (Real.log 20 / Real.log 5) = a - 2 :=
by
  sorry

end log_expansion_l1522_152297


namespace proposition_p_l1522_152286

variable (x : ℝ)

-- Define condition
def negation_of_p : Prop := ∃ x, x < 1 ∧ x^2 < 1

-- Define proposition p
def p : Prop := ∀ x, x < 1 → x^2 ≥ 1

-- Theorem statement
theorem proposition_p (h : negation_of_p) : (p) :=
sorry

end proposition_p_l1522_152286


namespace draw_four_balls_in_order_l1522_152235

theorem draw_four_balls_in_order :
  let total_balls := 15
  let color_sequence_length := 4
  let colors_sequence := ["Red", "Green", "Blue", "Yellow"]
  total_balls * (total_balls - 1) * (total_balls - 2) * (total_balls - 3) = 32760 :=
by 
  sorry

end draw_four_balls_in_order_l1522_152235


namespace min_players_team_l1522_152241

theorem min_players_team : Nat.lcm (Nat.lcm (Nat.lcm 8 9) 10) 11 = 7920 := 
by 
  -- The proof will be filled here.
  sorry

end min_players_team_l1522_152241


namespace leila_spending_l1522_152281

theorem leila_spending (sweater jewelry total money_left : ℕ) (h1 : sweater = 40) (h2 : sweater * 4 = total) (h3 : money_left = 20) (h4 : total - sweater - jewelry = money_left) : jewelry - sweater = 60 :=
by
  sorry

end leila_spending_l1522_152281


namespace time_to_fill_pool_l1522_152287

def LindasPoolCapacity : ℕ := 30000
def CurrentVolume : ℕ := 6000
def NumberOfHoses : ℕ := 6
def RatePerHosePerMinute : ℕ := 3
def GallonsNeeded : ℕ := LindasPoolCapacity - CurrentVolume
def RatePerHosePerHour : ℕ := RatePerHosePerMinute * 60
def TotalHourlyRate : ℕ := NumberOfHoses * RatePerHosePerHour

theorem time_to_fill_pool : (GallonsNeeded / TotalHourlyRate) = 22 :=
by
  sorry

end time_to_fill_pool_l1522_152287


namespace sin_identity_l1522_152209

variable (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = Real.sqrt 3 / 2)

theorem sin_identity : Real.sin (3 * Real.pi / 4 - α) = Real.sqrt 3 / 2 := by
  sorry

end sin_identity_l1522_152209


namespace flagpole_shadow_length_correct_l1522_152204

noncomputable def flagpole_shadow_length (flagpole_height building_height building_shadow_length : ℕ) :=
  flagpole_height * building_shadow_length / building_height

theorem flagpole_shadow_length_correct :
  flagpole_shadow_length 18 20 50 = 45 :=
by
  sorry

end flagpole_shadow_length_correct_l1522_152204


namespace min_value_of_sum_l1522_152268

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + 2 * b = 1) : 
  (∃ x, x = (3 / a + 2 / b) ∧ x = 25) :=
sorry

end min_value_of_sum_l1522_152268


namespace pages_for_15_dollars_l1522_152212

theorem pages_for_15_dollars 
  (cpg : ℚ) -- cost per 5 pages in cents
  (budget : ℚ) -- budget in cents
  (h_cpg_pos : cpg = 7 * 1) -- 7 cents for 5 pages
  (h_budget_pos : budget = 1500 * 1) -- $15 = 1500 cents
  : (budget * (5 / cpg)).floor = 1071 :=
by {
  sorry
}

end pages_for_15_dollars_l1522_152212


namespace min_chord_length_eq_l1522_152230

-- Define the Circle C with center (1, 2) and radius 5
def isCircle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 25

-- Define the Line l parameterized by m
def isLine (m x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Prove that the minimal chord length intercepted by the circle occurs when the line l is 2x - y - 5 = 0
theorem min_chord_length_eq (x y : ℝ) : 
  (∀ m, isLine m x y → isCircle x y) → isLine 0 x y :=
sorry

end min_chord_length_eq_l1522_152230


namespace find_reduced_price_l1522_152284

noncomputable def reduced_price_per_kg 
  (total_spent : ℝ) (original_quantity : ℝ) (additional_quantity : ℝ) (price_reduction_rate : ℝ) : ℝ :=
  let original_price := total_spent / original_quantity
  let reduced_price := original_price * (1 - price_reduction_rate)
  reduced_price

theorem find_reduced_price 
  (total_spent : ℝ := 800)
  (original_quantity : ℝ := 20)
  (additional_quantity : ℝ := 5)
  (price_reduction_rate : ℝ := 0.15) :
  reduced_price_per_kg total_spent original_quantity additional_quantity price_reduction_rate = 34 :=
by
  sorry

end find_reduced_price_l1522_152284


namespace sequence_sum_l1522_152219

theorem sequence_sum (A B C D E F G H I J : ℤ)
  (h1 : D = 7)
  (h2 : A + B + C = 24)
  (h3 : B + C + D = 24)
  (h4 : C + D + E = 24)
  (h5 : D + E + F = 24)
  (h6 : E + F + G = 24)
  (h7 : F + G + H = 24)
  (h8 : G + H + I = 24)
  (h9 : H + I + J = 24) : 
  A + J = 105 :=
sorry

end sequence_sum_l1522_152219


namespace bill_salary_increase_l1522_152250

theorem bill_salary_increase (S P : ℝ) 
  (h1 : S + 0.16 * S = 812) 
  (h2 : S + P * S = 770.0000000000001) : 
  P = 0.1 :=
by {
  sorry
}

end bill_salary_increase_l1522_152250


namespace line_circle_intersection_l1522_152283

theorem line_circle_intersection (k : ℝ) :
  ∃ x y : ℝ, y = k * (x + 1 / 2) ∧ x^2 + y^2 = 1 :=
sorry

end line_circle_intersection_l1522_152283


namespace min_sum_m_n_l1522_152207

open Nat

theorem min_sum_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m * n - 2 * m - 3 * n - 20 = 0) : m + n = 20 :=
sorry

end min_sum_m_n_l1522_152207
