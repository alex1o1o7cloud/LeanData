import Mathlib

namespace NUMINAMATH_GPT_age_problem_l711_71105

theorem age_problem :
  ∃ (x y z : ℕ), 
    x - y = 3 ∧
    z = 2 * x + 2 * y - 3 ∧
    z = x + y + 20 ∧
    x = 13 ∧
    y = 10 ∧
    z = 43 :=
by 
  sorry

end NUMINAMATH_GPT_age_problem_l711_71105


namespace NUMINAMATH_GPT_milk_price_increase_day_l711_71129

theorem milk_price_increase_day (total_cost : ℕ) (old_price : ℕ) (new_price : ℕ) (days : ℕ) (x : ℕ)
    (h1 : old_price = 1500)
    (h2 : new_price = 1600)
    (h3 : days = 30)
    (h4 : total_cost = 46200)
    (h5 : (x - 1) * old_price + (days + 1 - x) * new_price = total_cost) :
  x = 19 :=
by
  sorry

end NUMINAMATH_GPT_milk_price_increase_day_l711_71129


namespace NUMINAMATH_GPT_money_distribution_l711_71149

theorem money_distribution (a b c : ℝ) (h1 : 4 * (a - b - c) = 16)
                           (h2 : 6 * b - 2 * a - 2 * c = 16)
                           (h3 : 7 * c - a - b = 16) :
  a = 29 := 
by 
  sorry

end NUMINAMATH_GPT_money_distribution_l711_71149


namespace NUMINAMATH_GPT_dice_number_divisible_by_7_l711_71102

theorem dice_number_divisible_by_7 :
  ∃ a b c : ℕ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) 
               ∧ (1001 * (100 * a + 10 * b + c)) % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_dice_number_divisible_by_7_l711_71102


namespace NUMINAMATH_GPT_least_number_to_add_l711_71108

theorem least_number_to_add (n : ℕ) (H : n = 433124) : ∃ k, k = 15 ∧ (n + k) % 17 = 0 := by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l711_71108


namespace NUMINAMATH_GPT_find_n_l711_71154

def alpha (n : ℕ) : ℚ := ((n - 2) * 180) / n
def alpha_plus_3 (n : ℕ) : ℚ := ((n + 1) * 180) / (n + 3)
def alpha_minus_2 (n : ℕ) : ℚ := ((n - 4) * 180) / (n - 2)

theorem find_n (n : ℕ) (h : alpha_plus_3 n - alpha n = alpha n - alpha_minus_2 n) : n = 12 :=
by
  -- The proof will be added here
  sorry

end NUMINAMATH_GPT_find_n_l711_71154


namespace NUMINAMATH_GPT_maria_cookies_l711_71160

theorem maria_cookies :
  let c_initial := 19
  let c1 := c_initial - 5
  let c2 := c1 / 2
  let c_final := c2 - 2
  c_final = 5 :=
by
  sorry

end NUMINAMATH_GPT_maria_cookies_l711_71160


namespace NUMINAMATH_GPT_min_value_of_sum_l711_71158

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + 2 * b = 1) : 
  (∃ x, x = (3 / a + 2 / b) ∧ x = 25) :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_l711_71158


namespace NUMINAMATH_GPT_visitors_on_that_day_l711_71151

theorem visitors_on_that_day 
  (prev_visitors : ℕ) 
  (additional_visitors : ℕ) 
  (h1 : prev_visitors = 100)
  (h2 : additional_visitors = 566)
  : prev_visitors + additional_visitors = 666 := by
  sorry

end NUMINAMATH_GPT_visitors_on_that_day_l711_71151


namespace NUMINAMATH_GPT_power_sum_identity_l711_71156

theorem power_sum_identity (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) : 
  (a^7 + b^7 + c^7)^2 / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49 / 60 := 
by
  sorry

end NUMINAMATH_GPT_power_sum_identity_l711_71156


namespace NUMINAMATH_GPT_student_2005_says_1_l711_71128

def pattern : List ℕ := [1, 2, 3, 4, 3, 2]

def nth_number_in_pattern (n : ℕ) : ℕ :=
  List.nthLe pattern (n % 6) sorry  -- The index is (n-1) % 6 because Lean indices start at 0

theorem student_2005_says_1 : nth_number_in_pattern 2005 = 1 := 
  by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_student_2005_says_1_l711_71128


namespace NUMINAMATH_GPT_john_candies_l711_71101

theorem john_candies (mark_candies : ℕ) (peter_candies : ℕ) (total_candies : ℕ) (equal_share : ℕ) (h1 : mark_candies = 30) (h2 : peter_candies = 25) (h3 : total_candies = 90) (h4 : equal_share * 3 = total_candies) : 
  (total_candies - mark_candies - peter_candies = 35) :=
by
  sorry

end NUMINAMATH_GPT_john_candies_l711_71101


namespace NUMINAMATH_GPT_conveyor_belt_efficiencies_and_min_cost_l711_71113

theorem conveyor_belt_efficiencies_and_min_cost :
  ∃ (efficiency_B efficiency_A : ℝ),
    efficiency_A = 1.5 * efficiency_B ∧
    18000 / efficiency_B - 18000 / efficiency_A = 10 ∧
    efficiency_B = 600 ∧
    efficiency_A = 900 ∧
    ∃ (cost_A cost_B : ℝ),
      cost_A = 8 * 20 ∧
      cost_B = 6 * 30 ∧
      cost_A = 160 ∧
      cost_B = 180 ∧
      cost_A < cost_B :=
by
  sorry

end NUMINAMATH_GPT_conveyor_belt_efficiencies_and_min_cost_l711_71113


namespace NUMINAMATH_GPT_perpendicular_tangents_add_l711_71163

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

end NUMINAMATH_GPT_perpendicular_tangents_add_l711_71163


namespace NUMINAMATH_GPT_min_chord_length_eq_l711_71183

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

end NUMINAMATH_GPT_min_chord_length_eq_l711_71183


namespace NUMINAMATH_GPT_max_cables_cut_l711_71179

def initial_cameras : ℕ := 200
def initial_cables : ℕ := 345
def resulting_clusters : ℕ := 8

theorem max_cables_cut :
  ∃ (cables_cut : ℕ), resulting_clusters = 8 ∧ initial_cables - cables_cut = (initial_cables - cables_cut) - (resulting_clusters - 1) ∧ cables_cut = 153 :=
by
  sorry

end NUMINAMATH_GPT_max_cables_cut_l711_71179


namespace NUMINAMATH_GPT_sunil_total_amount_proof_l711_71148

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

end NUMINAMATH_GPT_sunil_total_amount_proof_l711_71148


namespace NUMINAMATH_GPT_exist_N_for_fn_eq_n_l711_71126

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom f_condition1 (m n : ℕ+) : (f m, f n) ≤ (m, n) ^ 2014
axiom f_condition2 (n : ℕ+) : n ≤ f n ∧ f n ≤ n + 2014

theorem exist_N_for_fn_eq_n :
  ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → f n = n := sorry

end NUMINAMATH_GPT_exist_N_for_fn_eq_n_l711_71126


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l711_71167

open Real

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 15 < 0) : 3 < x ∧ x < 5 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l711_71167


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l711_71132

theorem sufficient_but_not_necessary_condition 
  (a b : ℝ) (h : a > b ∧ b > 0) : (a^2 > b^2) ∧ (¬ ∀ (a' b' : ℝ), a'^2 > b'^2 → a' > b' ∧ b' > 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l711_71132


namespace NUMINAMATH_GPT_partition_555_weights_l711_71100

theorem partition_555_weights :
  ∃ A B C : Finset ℕ, 
  (∀ x ∈ A, x ∈ Finset.range (555 + 1)) ∧ 
  (∀ y ∈ B, y ∈ Finset.range (555 + 1)) ∧ 
  (∀ z ∈ C, z ∈ Finset.range (555 + 1)) ∧ 
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ 
  A ∪ B ∪ C = Finset.range (555 + 1) ∧ 
  A.sum id = 51430 ∧ B.sum id = 51430 ∧ C.sum id = 51430 := sorry

end NUMINAMATH_GPT_partition_555_weights_l711_71100


namespace NUMINAMATH_GPT_decimal_to_binary_49_l711_71171

theorem decimal_to_binary_49 : ((49:ℕ) = 6 * 2^4 + 3 * 2^3 + 0 * 2^2 + 0 * 2^1 + 0 * 2^0 + 1) ↔ (110001 = 110001) :=
by
  sorry

end NUMINAMATH_GPT_decimal_to_binary_49_l711_71171


namespace NUMINAMATH_GPT_distinct_prime_divisors_l711_71147

theorem distinct_prime_divisors (a : ℤ) (n : ℕ) (h₁ : a > 3) (h₂ : Odd a) (h₃ : n > 0) : 
  ∃ (p : Finset ℤ), p.card ≥ n + 1 ∧ ∀ q ∈ p, Prime q ∧ q ∣ (a ^ (2 ^ n) - 1) :=
sorry

end NUMINAMATH_GPT_distinct_prime_divisors_l711_71147


namespace NUMINAMATH_GPT_inequality_proof_l711_71170

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

end NUMINAMATH_GPT_inequality_proof_l711_71170


namespace NUMINAMATH_GPT_combined_average_score_girls_l711_71127

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

end NUMINAMATH_GPT_combined_average_score_girls_l711_71127


namespace NUMINAMATH_GPT_sequence_squared_l711_71195

theorem sequence_squared (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n = a (n - 1) + 2 * (n - 1)) 
  : ∀ n, a n = n^2 := 
by
  sorry

end NUMINAMATH_GPT_sequence_squared_l711_71195


namespace NUMINAMATH_GPT_ratio_of_interior_to_exterior_angle_in_regular_octagon_l711_71153

theorem ratio_of_interior_to_exterior_angle_in_regular_octagon
  (n : ℕ) (regular_polygon : n = 8) : 
  let interior_angle := ((n - 2) * 180) / n
  let exterior_angle := 360 / n
  (interior_angle / exterior_angle) = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_interior_to_exterior_angle_in_regular_octagon_l711_71153


namespace NUMINAMATH_GPT_cos_theta_value_l711_71117

theorem cos_theta_value (θ : ℝ) (h_tan : Real.tan θ = -4/3) (h_range : 0 < θ ∧ θ < π) : Real.cos θ = -3/5 :=
by
  sorry

end NUMINAMATH_GPT_cos_theta_value_l711_71117


namespace NUMINAMATH_GPT_time_to_fill_pool_l711_71190

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

end NUMINAMATH_GPT_time_to_fill_pool_l711_71190


namespace NUMINAMATH_GPT_solution_set_inequality_l711_71188

theorem solution_set_inequality (x : ℝ) :
  (|x + 3| - |x - 3| > 3) ↔ (x > 3 / 2) := 
sorry

end NUMINAMATH_GPT_solution_set_inequality_l711_71188


namespace NUMINAMATH_GPT_max_elves_without_caps_proof_max_elves_with_caps_proof_l711_71177

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

end NUMINAMATH_GPT_max_elves_without_caps_proof_max_elves_with_caps_proof_l711_71177


namespace NUMINAMATH_GPT_sum_mod_17_l711_71114

theorem sum_mod_17 : (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 17 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_17_l711_71114


namespace NUMINAMATH_GPT_calculate_highest_score_l711_71150

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

end NUMINAMATH_GPT_calculate_highest_score_l711_71150


namespace NUMINAMATH_GPT_largest_value_of_a_l711_71136

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

end NUMINAMATH_GPT_largest_value_of_a_l711_71136


namespace NUMINAMATH_GPT_geraldine_more_than_jazmin_l711_71146

def geraldine_dolls : ℝ := 2186.0
def jazmin_dolls : ℝ := 1209.0
def difference_dolls : ℝ := 977.0

theorem geraldine_more_than_jazmin : geraldine_dolls - jazmin_dolls = difference_dolls :=
by sorry

end NUMINAMATH_GPT_geraldine_more_than_jazmin_l711_71146


namespace NUMINAMATH_GPT_arbitrarily_large_ratios_l711_71116

open Nat

theorem arbitrarily_large_ratios (a : ℕ → ℕ) (h_distinct: ∀ m n, m ≠ n → a m ≠ a n)
  (h_no_100_ones: ∀ n, ¬ (∃ k, a n / 10^k % 10^100 = 10^100 - 1)):
  ∀ M : ℕ, ∃ n : ℕ, a n / n ≥ M :=
by
  sorry

end NUMINAMATH_GPT_arbitrarily_large_ratios_l711_71116


namespace NUMINAMATH_GPT_remainder_n_plus_2023_l711_71118

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 5 = 2) : (n + 2023) % 5 = 0 :=
sorry

end NUMINAMATH_GPT_remainder_n_plus_2023_l711_71118


namespace NUMINAMATH_GPT_part_I_part_II_l711_71182

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 + 4 * a * x - 3

-- Part (I)
theorem part_I (a : ℝ) (h_a : a > 0) (h_roots: ∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ f a x1 = 0 ∧ f a x2 = 0) : 
  0 < a ∧ a < 2 / 5 :=
sorry

-- Part (II)
theorem part_II (a : ℝ) (h_max : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f a x ≤ f a 2) : 
  a ≥ -1 / 3 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l711_71182


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l711_71120

theorem hyperbola_asymptotes:
  ∀ (x y : ℝ),
  ( ∀ y, y = (1 + (4 / 5) * x) ∨ y = (1 - (4 / 5) * x) ) →
  (y-1)^2 / 16 - x^2 / 25 = 1 →
  (∃ m b: ℝ, m > 0 ∧ m = 4/5 ∧ b = 1) := by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l711_71120


namespace NUMINAMATH_GPT_line_circle_intersection_l711_71185

theorem line_circle_intersection (k : ℝ) :
  ∃ x y : ℝ, y = k * (x + 1 / 2) ∧ x^2 + y^2 = 1 :=
sorry

end NUMINAMATH_GPT_line_circle_intersection_l711_71185


namespace NUMINAMATH_GPT_proposition_p_l711_71123

variable (x : ℝ)

-- Define condition
def negation_of_p : Prop := ∃ x, x < 1 ∧ x^2 < 1

-- Define proposition p
def p : Prop := ∀ x, x < 1 → x^2 ≥ 1

-- Theorem statement
theorem proposition_p (h : negation_of_p) : (p) :=
sorry

end NUMINAMATH_GPT_proposition_p_l711_71123


namespace NUMINAMATH_GPT_x_is_integer_l711_71135

theorem x_is_integer
  (x : ℝ)
  (h_pos : 0 < x)
  (h1 : ∃ k1 : ℤ, x^2012 = x^2001 + k1)
  (h2 : ∃ k2 : ℤ, x^2001 = x^1990 + k2) : 
  ∃ n : ℤ, x = n :=
sorry

end NUMINAMATH_GPT_x_is_integer_l711_71135


namespace NUMINAMATH_GPT_fodder_lasting_days_l711_71143

theorem fodder_lasting_days (buffalo_fodder_rate cow_fodder_rate ox_fodder_rate : ℕ)
  (initial_buffaloes initial_cows initial_oxen added_buffaloes added_cows initial_days : ℕ)
  (h1 : 3 * buffalo_fodder_rate = 4 * cow_fodder_rate)
  (h2 : 3 * buffalo_fodder_rate = 2 * ox_fodder_rate)
  (h3 : initial_days * (initial_buffaloes * buffalo_fodder_rate + initial_cows * cow_fodder_rate + initial_oxen * ox_fodder_rate) = 4320) :
  (4320 / ((initial_buffaloes + added_buffaloes) * buffalo_fodder_rate + (initial_cows + added_cows) * cow_fodder_rate + initial_oxen * ox_fodder_rate)) = 9 :=
by 
  sorry

end NUMINAMATH_GPT_fodder_lasting_days_l711_71143


namespace NUMINAMATH_GPT_probability_same_tribe_l711_71166

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

end NUMINAMATH_GPT_probability_same_tribe_l711_71166


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l711_71184

/-- If r, s, and t are the roots of the cubic equation x³ - ax² + bx - c = 0, then r² + s² + t² = a² - 2b. -/
theorem sum_of_squares_of_roots (r s t a b c : ℝ) (h1 : r + s + t = a) (h2 : r * s + r * t + s * t = b) (h3 : r * s * t = c) :
    r ^ 2 + s ^ 2 + t ^ 2 = a ^ 2 - 2 * b := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l711_71184


namespace NUMINAMATH_GPT_drive_time_from_city_B_to_city_A_l711_71199

theorem drive_time_from_city_B_to_city_A
  (t : ℝ)
  (round_trip_distance : ℝ := 360)
  (saved_time_per_trip : ℝ := 0.5)
  (average_speed : ℝ := 80) :
  (80 * ((3 + t) - 2 * 0.5)) = 360 → t = 2.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_drive_time_from_city_B_to_city_A_l711_71199


namespace NUMINAMATH_GPT_find_square_number_divisible_by_9_between_40_and_90_l711_71193

theorem find_square_number_divisible_by_9_between_40_and_90 :
  ∃ x : ℕ, (∃ n : ℕ, x = n^2) ∧ (9 ∣ x) ∧ 40 < x ∧ x < 90 ∧ x = 81 :=
by
  sorry

end NUMINAMATH_GPT_find_square_number_divisible_by_9_between_40_and_90_l711_71193


namespace NUMINAMATH_GPT_calculate_leakage_rate_l711_71131

variable (B : ℕ) (T : ℕ) (R : ℝ)

-- B represents the bucket's capacity in ounces, T represents time in hours, R represents the rate of leakage per hour in ounces per hour.

def leakage_rate (B : ℕ) (T : ℕ) (R : ℝ) : Prop :=
  (B = 36) ∧ (T = 12) ∧ (B / 2 = T * R)

theorem calculate_leakage_rate : leakage_rate 36 12 1.5 :=
by 
  simp [leakage_rate]
  sorry

end NUMINAMATH_GPT_calculate_leakage_rate_l711_71131


namespace NUMINAMATH_GPT_draw_four_balls_in_order_l711_71130

theorem draw_four_balls_in_order :
  let total_balls := 15
  let color_sequence_length := 4
  let colors_sequence := ["Red", "Green", "Blue", "Yellow"]
  total_balls * (total_balls - 1) * (total_balls - 2) * (total_balls - 3) = 32760 :=
by 
  sorry

end NUMINAMATH_GPT_draw_four_balls_in_order_l711_71130


namespace NUMINAMATH_GPT_find_four_digit_number_abcd_exists_l711_71181

theorem find_four_digit_number_abcd_exists (M : ℕ) (H1 : M > 0) (H2 : M % 10 ≠ 0) 
    (H3 : M % 100000 = M^2 % 100000) : ∃ abcd : ℕ, abcd = 2502 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_find_four_digit_number_abcd_exists_l711_71181


namespace NUMINAMATH_GPT_lockers_number_l711_71142

theorem lockers_number (total_cost : ℝ) (cost_per_digit : ℝ) (total_lockers : ℕ) 
  (locker_numbered_from_one : ∀ n : ℕ, n >= 1) :
  total_cost = 248.43 → cost_per_digit = 0.03 → total_lockers = 2347 :=
by
  intros h_total_cost h_cost_per_digit
  sorry

end NUMINAMATH_GPT_lockers_number_l711_71142


namespace NUMINAMATH_GPT_gcd_48_180_l711_71121

theorem gcd_48_180 : Nat.gcd 48 180 = 12 := by
  have f1 : 48 = 2^4 * 3 := by norm_num
  have f2 : 180 = 2^2 * 3^2 * 5 := by norm_num
  sorry

end NUMINAMATH_GPT_gcd_48_180_l711_71121


namespace NUMINAMATH_GPT_brokerage_percentage_correct_l711_71164

noncomputable def brokerage_percentage (market_value : ℝ) (income : ℝ) (investment : ℝ) (nominal_rate : ℝ) : ℝ :=
  let face_value := (income * 100) / nominal_rate
  let market_price := (face_value * market_value) / 100
  let brokerage_amount := investment - market_price
  (brokerage_amount / investment) * 100

theorem brokerage_percentage_correct :
  brokerage_percentage 110.86111111111111 756 8000 10.5 = 0.225 :=
by
  sorry

end NUMINAMATH_GPT_brokerage_percentage_correct_l711_71164


namespace NUMINAMATH_GPT_leila_spending_l711_71168

theorem leila_spending (sweater jewelry total money_left : ℕ) (h1 : sweater = 40) (h2 : sweater * 4 = total) (h3 : money_left = 20) (h4 : total - sweater - jewelry = money_left) : jewelry - sweater = 60 :=
by
  sorry

end NUMINAMATH_GPT_leila_spending_l711_71168


namespace NUMINAMATH_GPT_compute_expression_l711_71174

theorem compute_expression : 20 * (150 / 3 + 36 / 4 + 4 / 25 + 2) = 1223 + 1/5 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l711_71174


namespace NUMINAMATH_GPT_capacity_of_initial_20_buckets_l711_71155

theorem capacity_of_initial_20_buckets (x : ℝ) (h : 20 * x = 270) : x = 13.5 :=
by 
  sorry

end NUMINAMATH_GPT_capacity_of_initial_20_buckets_l711_71155


namespace NUMINAMATH_GPT_bus_problem_l711_71106

theorem bus_problem
  (initial_children : ℕ := 18)
  (final_total_children : ℕ := 25) :
  final_total_children - initial_children = 7 :=
by
  sorry

end NUMINAMATH_GPT_bus_problem_l711_71106


namespace NUMINAMATH_GPT_find_e_l711_71197

theorem find_e 
  (a b c d e : ℕ) 
  (h1 : a = 16)
  (h2 : b = 2)
  (h3 : c = 3)
  (h4 : d = 12)
  (h5 : 32 / e = 288 / e) 
  : e = 9 := 
by
  sorry

end NUMINAMATH_GPT_find_e_l711_71197


namespace NUMINAMATH_GPT_time_per_employee_updating_payroll_records_l711_71187

-- Define the conditions
def minutes_making_coffee : ℕ := 5
def minutes_per_employee_status_update : ℕ := 2
def num_employees : ℕ := 9
def total_morning_routine_minutes : ℕ := 50

-- Define the proof statement encapsulating the problem
theorem time_per_employee_updating_payroll_records :
  (total_morning_routine_minutes - (minutes_making_coffee + minutes_per_employee_status_update * num_employees)) / num_employees = 3 := by
  sorry

end NUMINAMATH_GPT_time_per_employee_updating_payroll_records_l711_71187


namespace NUMINAMATH_GPT_erica_duration_is_correct_l711_71175

-- Define the durations for Dave, Chuck, and Erica
def dave_duration : ℝ := 10
def chuck_duration : ℝ := 5 * dave_duration
def erica_duration : ℝ := chuck_duration + 0.30 * chuck_duration

-- State the theorem
theorem erica_duration_is_correct : erica_duration = 65 := by
  sorry

end NUMINAMATH_GPT_erica_duration_is_correct_l711_71175


namespace NUMINAMATH_GPT_larger_interior_angle_trapezoid_pavilion_l711_71124

theorem larger_interior_angle_trapezoid_pavilion :
  let n := 12
  let central_angle := 360 / n
  let smaller_angle := 180 - (central_angle / 2)
  let larger_angle := 180 - smaller_angle
  larger_angle = 97.5 :=
by
  sorry

end NUMINAMATH_GPT_larger_interior_angle_trapezoid_pavilion_l711_71124


namespace NUMINAMATH_GPT_ratio_abc_xyz_l711_71161

theorem ratio_abc_xyz
  (a b c x y z : ℝ)
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z) 
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a * x + b * y + c * z = 56) :
  (a + b + c) / (x + y + z) = 7 / 8 := 
sorry

end NUMINAMATH_GPT_ratio_abc_xyz_l711_71161


namespace NUMINAMATH_GPT_find_a_value_l711_71145

theorem find_a_value (a : ℝ) (A B : Set ℝ) (hA : A = {3, 5}) (hB : B = {x | a * x - 1 = 0}) :
  B ⊆ A → a = 0 ∨ a = 1/3 ∨ a = 1/5 :=
by sorry

end NUMINAMATH_GPT_find_a_value_l711_71145


namespace NUMINAMATH_GPT_teal_more_blue_proof_l711_71138

theorem teal_more_blue_proof (P G B N : ℕ) (hP : P = 150) (hG : G = 90) (hB : B = 40) (hN : N = 25) : 
  (∃ (x : ℕ), x = 75) :=
by
  sorry

end NUMINAMATH_GPT_teal_more_blue_proof_l711_71138


namespace NUMINAMATH_GPT_min_players_team_l711_71141

theorem min_players_team : Nat.lcm (Nat.lcm (Nat.lcm 8 9) 10) 11 = 7920 := 
by 
  -- The proof will be filled here.
  sorry

end NUMINAMATH_GPT_min_players_team_l711_71141


namespace NUMINAMATH_GPT_find_c_l711_71140

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

end NUMINAMATH_GPT_find_c_l711_71140


namespace NUMINAMATH_GPT_ms_tom_investment_l711_71112

def invested_amounts (X Y : ℝ) : Prop :=
  X + Y = 100000 ∧ 0.17 * Y = 0.23 * X + 200 

theorem ms_tom_investment (X Y : ℝ) (h : invested_amounts X Y) : X = 42000 :=
by
  sorry

end NUMINAMATH_GPT_ms_tom_investment_l711_71112


namespace NUMINAMATH_GPT_find_m_l711_71198

theorem find_m (m : ℝ) : (m + 2) * (m - 2) + 3 * m * (m + 2) = 0 ↔ m = 1/2 ∨ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l711_71198


namespace NUMINAMATH_GPT_cost_price_is_correct_l711_71144

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

end NUMINAMATH_GPT_cost_price_is_correct_l711_71144


namespace NUMINAMATH_GPT_no_non_similar_triangles_with_geometric_angles_l711_71110

theorem no_non_similar_triangles_with_geometric_angles :
  ¬ ∃ (a r : ℤ), 0 < a ∧ 0 < r ∧ a ≠ ar ∧ a ≠ ar^2 ∧ ar ≠ ar^2 ∧
  a + ar + ar^2 = 180 :=
sorry

end NUMINAMATH_GPT_no_non_similar_triangles_with_geometric_angles_l711_71110


namespace NUMINAMATH_GPT_g_nested_result_l711_71165

def g (n : ℕ) : ℕ :=
if n < 5 then
  n^2 + 1
else
  2 * n + 3

theorem g_nested_result : g (g (g 3)) = 49 := by
sorry

end NUMINAMATH_GPT_g_nested_result_l711_71165


namespace NUMINAMATH_GPT_second_gym_signup_fee_covers_4_months_l711_71122

-- Define constants
def cheap_gym_monthly_fee : ℕ := 10
def cheap_gym_signup_fee : ℕ := 50
def total_spent_first_year : ℕ := 650

-- Define the monthly fee of the second gym
def second_gym_monthly_fee : ℕ := 3 * cheap_gym_monthly_fee

-- Calculate the amount spent on the second gym
def spent_on_second_gym : ℕ := total_spent_first_year - (12 * cheap_gym_monthly_fee + cheap_gym_signup_fee)

-- Define the number of months the sign-up fee covers
def months_covered_by_signup_fee : ℕ := spent_on_second_gym / second_gym_monthly_fee

theorem second_gym_signup_fee_covers_4_months :
  months_covered_by_signup_fee = 4 :=
by
  sorry

end NUMINAMATH_GPT_second_gym_signup_fee_covers_4_months_l711_71122


namespace NUMINAMATH_GPT_not_possible_to_color_l711_71176

theorem not_possible_to_color (f : ℕ → ℕ) (c1 c2 c3 : ℕ) :
  ∃ (x : ℕ), 1 < x ∧ f 2 = c1 ∧ f 4 = c1 ∧ 
  ∀ (a b : ℕ), 1 < a → 1 < b → f a ≠ f b → (f (a * b) ≠ f a ∧ f (a * b) ≠ f b) → 
  false :=
sorry

end NUMINAMATH_GPT_not_possible_to_color_l711_71176


namespace NUMINAMATH_GPT_smallest_integer_y_l711_71119

theorem smallest_integer_y (y : ℤ) (h : 7 - 5 * y < 22) : y ≥ -2 :=
by sorry

end NUMINAMATH_GPT_smallest_integer_y_l711_71119


namespace NUMINAMATH_GPT_pythagorean_relationship_l711_71115

theorem pythagorean_relationship (a b c : ℝ) (h : c^2 = a^2 + b^2) : c^2 = a^2 + b^2 :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_relationship_l711_71115


namespace NUMINAMATH_GPT_shrimp_appetizer_cost_l711_71152

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

end NUMINAMATH_GPT_shrimp_appetizer_cost_l711_71152


namespace NUMINAMATH_GPT_Megatech_budget_allocation_l711_71173

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

end NUMINAMATH_GPT_Megatech_budget_allocation_l711_71173


namespace NUMINAMATH_GPT_number_of_correct_judgments_is_zero_l711_71107

theorem number_of_correct_judgments_is_zero :
  (¬ ∀ (x : ℚ), -x ≠ |x|) ∧
  (¬ ∀ (x y : ℚ), -x = y → y = 1 / x) ∧
  (¬ ∀ (x y : ℚ), |x| = |y| → x = y) →
  0 = 0 :=
by
  intros h
  exact rfl

end NUMINAMATH_GPT_number_of_correct_judgments_is_zero_l711_71107


namespace NUMINAMATH_GPT_find_weights_l711_71162

def item_weights (a b c d e f g h : ℕ) : Prop :=
  1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧ 1 ≤ e ∧ 1 ≤ f ∧ 1 ≤ g ∧ 1 ≤ h ∧
  a > b ∧ b > c ∧ c > d ∧ d > e ∧ e > f ∧ f > g ∧ g > h ∧
  a ≤ 15 ∧ b ≤ 15 ∧ c ≤ 15 ∧ d ≤ 15 ∧ e ≤ 15 ∧ f ≤ 15 ∧ g ≤ 15 ∧ h ≤ 15

theorem find_weights (a b c d e f g h : ℕ) (hw : item_weights a b c d e f g h) 
    (h1 : d + e + f + g > a + b + c + h) 
    (h2 : e + f > d + g) 
    (h3 : e > f) : e = 11 ∧ g = 5 := sorry

end NUMINAMATH_GPT_find_weights_l711_71162


namespace NUMINAMATH_GPT_total_money_l711_71109

theorem total_money (a b c : ℕ) (h_ratio : (a / 2) / (b / 3) / (c / 4) = 1) (h_c : c = 306) : 
  a + b + c = 782 := 
by sorry

end NUMINAMATH_GPT_total_money_l711_71109


namespace NUMINAMATH_GPT_sum_divisible_by_3_l711_71159

theorem sum_divisible_by_3 (a : ℤ) : 3 ∣ (a^3 + 2 * a) :=
sorry

end NUMINAMATH_GPT_sum_divisible_by_3_l711_71159


namespace NUMINAMATH_GPT_total_gas_cost_l711_71192

def car_city_mpg : ℝ := 30
def car_highway_mpg : ℝ := 40
def city_miles : ℝ := 60 + 40 + 25
def highway_miles : ℝ := 200 + 150 + 180
def gas_price_per_gallon : ℝ := 3.00

theorem total_gas_cost : 
  (city_miles / car_city_mpg + highway_miles / car_highway_mpg) * gas_price_per_gallon = 52.25 := 
by
  sorry

end NUMINAMATH_GPT_total_gas_cost_l711_71192


namespace NUMINAMATH_GPT_shirts_sewn_on_tuesday_l711_71169

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

end NUMINAMATH_GPT_shirts_sewn_on_tuesday_l711_71169


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l711_71104

theorem simplify_expr1 (a : ℝ) : 2 * (a - 1) - (2 * a - 3) + 3 = 4 :=
by
  sorry

theorem simplify_expr2 (x : ℝ) : 3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l711_71104


namespace NUMINAMATH_GPT_find_reduced_price_l711_71186

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

end NUMINAMATH_GPT_find_reduced_price_l711_71186


namespace NUMINAMATH_GPT_bill_salary_increase_l711_71172

theorem bill_salary_increase (S P : ℝ) 
  (h1 : S + 0.16 * S = 812) 
  (h2 : S + P * S = 770.0000000000001) : 
  P = 0.1 :=
by {
  sorry
}

end NUMINAMATH_GPT_bill_salary_increase_l711_71172


namespace NUMINAMATH_GPT_smaller_circle_radius_l711_71194

theorem smaller_circle_radius (A1 A2 : ℝ) 
  (h1 : A1 + 2 * A2 = 25 * Real.pi) 
  (h2 : ∃ d : ℝ, A1 + d = A2 ∧ A2 + d = A1 + 2 * A2) : 
  ∃ r : ℝ, r^2 = 5 ∧ Real.pi * r^2 = A1 :=
by
  sorry

end NUMINAMATH_GPT_smaller_circle_radius_l711_71194


namespace NUMINAMATH_GPT_problem_conditions_l711_71103

def G (m : ℕ) : ℕ := m % 10

theorem problem_conditions (a b c : ℕ) (non_neg_m : ∀ m : ℕ, 0 ≤ m) :
  -- Condition ①
  ¬ (G (a - b) = G a - G b) ∧
  -- Condition ②
  (a - b = 10 * c → G a = G b) ∧
  -- Condition ③
  (G (a * b * c) = G (G a * G b * G c)) ∧
  -- Condition ④
  ¬ (G (3^2015) = 9) :=
by sorry

end NUMINAMATH_GPT_problem_conditions_l711_71103


namespace NUMINAMATH_GPT_pow_evaluation_l711_71189

theorem pow_evaluation : 81^(5/4) = 243 := 
by sorry

end NUMINAMATH_GPT_pow_evaluation_l711_71189


namespace NUMINAMATH_GPT_middle_number_is_10_l711_71137

theorem middle_number_is_10 (x y z : ℤ) (hx : x < y) (hy : y < z) 
    (h1 : x + y = 18) (h2 : x + z = 25) (h3 : y + z = 27) : y = 10 :=
by 
  sorry

end NUMINAMATH_GPT_middle_number_is_10_l711_71137


namespace NUMINAMATH_GPT_water_usage_l711_71180

def fee (x : ℕ) : ℕ :=
  if x ≤ 8 then 2 * x else 4 * x - 16

theorem water_usage (h : fee 9 = 20) : fee 9 = 20 := by
  sorry

end NUMINAMATH_GPT_water_usage_l711_71180


namespace NUMINAMATH_GPT_line_through_points_l711_71139

theorem line_through_points :
  ∀ x y : ℝ, (∃ t : ℝ, (x, y) = (2 * t, -3 * (1 - t))) ↔ (x / 2) - (y / 3) = 1 :=
by
  sorry

end NUMINAMATH_GPT_line_through_points_l711_71139


namespace NUMINAMATH_GPT_trigonometric_inequality_l711_71178

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

end NUMINAMATH_GPT_trigonometric_inequality_l711_71178


namespace NUMINAMATH_GPT_f_7_minus_a_eq_neg_7_over_4_l711_71125

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 2 else -Real.logb 3 x

variable (a : ℝ)

-- Given conditions
axiom h1 : f a = -2

-- The proof of the required condition
theorem f_7_minus_a_eq_neg_7_over_4 (h1 : f a = -2) : f (7 - a) = -7 / 4 := sorry

end NUMINAMATH_GPT_f_7_minus_a_eq_neg_7_over_4_l711_71125


namespace NUMINAMATH_GPT_ken_change_l711_71157

theorem ken_change (cost_per_pound : ℕ) (quantity : ℕ) (amount_paid : ℕ) (total_cost : ℕ) (change : ℕ) 
(h1 : cost_per_pound = 7)
(h2 : quantity = 2)
(h3 : amount_paid = 20)
(h4 : total_cost = cost_per_pound * quantity)
(h5 : change = amount_paid - total_cost) : change = 6 :=
by 
  sorry

end NUMINAMATH_GPT_ken_change_l711_71157


namespace NUMINAMATH_GPT_base_of_exponent_l711_71133

theorem base_of_exponent (x : ℤ) (m : ℕ) (h₁ : (-2 : ℤ)^(2 * m) = x^(12 - m)) (h₂ : m = 4) : x = -2 :=
by 
  sorry

end NUMINAMATH_GPT_base_of_exponent_l711_71133


namespace NUMINAMATH_GPT_count_triangles_with_center_inside_l711_71191

theorem count_triangles_with_center_inside :
  let n := 201
  let num_triangles_with_center_inside (n : ℕ) : ℕ := 
    let half := n / 2
    let group_count := half * (half + 1) / 2
    group_count * n / 3
  num_triangles_with_center_inside n = 338350 :=
by
  sorry

end NUMINAMATH_GPT_count_triangles_with_center_inside_l711_71191


namespace NUMINAMATH_GPT_pascal_triangle_fifth_number_twentieth_row_l711_71111

theorem pascal_triangle_fifth_number_twentieth_row : 
  (Nat.choose 20 4) = 4845 :=
by
  sorry

end NUMINAMATH_GPT_pascal_triangle_fifth_number_twentieth_row_l711_71111


namespace NUMINAMATH_GPT_part_a_part_b_l711_71134

-- Part (a)
theorem part_a {x y n : ℕ} (h : x^3 + 2^n * y = y^3 + 2^n * x) : x = y :=
sorry

-- Part (b)
theorem part_b {x y : ℤ} {n : ℕ} (h : x ≠ 0 ∧ y ≠ 0 ∧ x^3 + 2^n * y = y^3 + 2^n * x) : |x| = |y| :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l711_71134


namespace NUMINAMATH_GPT_option_A_option_B_option_D_l711_71196

-- Definitions of sequences
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a_1 + n * d

def geometric_seq (b_1 : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  b_1 * q ^ n

-- Option A: Prove that there exist d and q such that a_n = b_n
theorem option_A : ∃ (d q : ℤ), ∀ (a_1 b_1 : ℤ) (n : ℕ), 
  (arithmetic_seq a_1 d n = geometric_seq b_1 q n) := sorry

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

-- Option B: Prove the differences form an arithmetic sequence
theorem option_B (a_1 : ℤ) (d : ℤ) :
  ∀ n k : ℕ, k > 0 → 
  (sum_arithmetic_seq a_1 d ((k + 1) * n) - sum_arithmetic_seq a_1 d (k * n) =
   (sum_arithmetic_seq a_1 d n + k * n * n * d)) := sorry

-- Option D: Prove there exist real numbers A and a such that A * a^a_n = b_n
theorem option_D (a_1 : ℤ) (d : ℤ) (b_1 : ℤ) (q : ℤ) :
  ∀ n : ℕ, b_1 > 0 → q > 0 → 
  ∃ A a : ℝ, A * a^ (arithmetic_seq a_1 d n) = (geometric_seq b_1 q n) := sorry

end NUMINAMATH_GPT_option_A_option_B_option_D_l711_71196
