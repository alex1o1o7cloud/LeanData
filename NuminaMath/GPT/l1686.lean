import Mathlib

namespace NUMINAMATH_GPT_smallest_blocks_needed_for_wall_l1686_168618

noncomputable def smallest_number_of_blocks (wall_length : ℕ) (wall_height : ℕ) (block_length1 : ℕ) (block_length2 : ℕ) (block_length3 : ℝ) : ℕ :=
  let blocks_per_odd_row := wall_length / block_length1
  let blocks_per_even_row := wall_length / block_length1 - 1 + 2
  let odd_rows := wall_height / 2 + 1
  let even_rows := wall_height / 2
  odd_rows * blocks_per_odd_row + even_rows * blocks_per_even_row

theorem smallest_blocks_needed_for_wall :
  smallest_number_of_blocks 120 7 2 1 1.5 = 423 :=
by
  sorry

end NUMINAMATH_GPT_smallest_blocks_needed_for_wall_l1686_168618


namespace NUMINAMATH_GPT_tan_three_halves_pi_sub_alpha_l1686_168677

theorem tan_three_halves_pi_sub_alpha (α : ℝ) (h : Real.cos (π - α) = -3/5) :
    Real.tan (3 * π / 2 - α) = 3/4 ∨ Real.tan (3 * π / 2 - α) = -3/4 := by
  sorry

end NUMINAMATH_GPT_tan_three_halves_pi_sub_alpha_l1686_168677


namespace NUMINAMATH_GPT_probability_sqrt_lt_7_of_random_two_digit_number_l1686_168676

theorem probability_sqrt_lt_7_of_random_two_digit_number : 
  (∃ p : ℚ, (∀ n, 10 ≤ n ∧ n ≤ 99 → n < 49 → ∃ k, k = p) ∧ p = 13 / 30) := 
by
  sorry

end NUMINAMATH_GPT_probability_sqrt_lt_7_of_random_two_digit_number_l1686_168676


namespace NUMINAMATH_GPT_martha_black_butterflies_l1686_168627

theorem martha_black_butterflies
    (total_butterflies : ℕ)
    (total_blue_butterflies : ℕ)
    (total_yellow_butterflies : ℕ)
    (total_black_butterflies : ℕ)
    (h1 : total_butterflies = 19)
    (h2 : total_blue_butterflies = 6)
    (h3 : total_blue_butterflies = 2 * total_yellow_butterflies)
    (h4 : total_black_butterflies = total_butterflies - (total_blue_butterflies + total_yellow_butterflies))
    : total_black_butterflies = 10 :=
  sorry

end NUMINAMATH_GPT_martha_black_butterflies_l1686_168627


namespace NUMINAMATH_GPT_fourth_person_height_l1686_168639

noncomputable def height_of_fourth_person (H : ℕ) : ℕ := 
  let second_person := H + 2
  let third_person := H + 4
  let fourth_person := H + 10
  fourth_person

theorem fourth_person_height {H : ℕ} 
  (cond1 : 2 = 2)
  (cond2 : 6 = 6)
  (average_height : 76 = 76) 
  (height_sum : H + (H + 2) + (H + 4) + (H + 10) = 304) : 
  height_of_fourth_person H = 82 := sorry

end NUMINAMATH_GPT_fourth_person_height_l1686_168639


namespace NUMINAMATH_GPT_max_profit_l1686_168603

noncomputable def fixed_cost : ℝ := 2.5
noncomputable def var_cost (x : ℕ) : ℝ :=
  if x < 80 then (x^2 + 10 * x) * 1e4
  else (51 * x - 1450) * 1e4
noncomputable def revenue (x : ℕ) : ℝ := 500 * x * 1e4
noncomputable def profit (x : ℕ) : ℝ := revenue x - var_cost x - fixed_cost * 1e4

theorem max_profit (x : ℕ) :
  (∀ y : ℕ, profit y ≤ 43200 * 1e4) ∧ profit 100 = 43200 * 1e4 := by
  sorry

end NUMINAMATH_GPT_max_profit_l1686_168603


namespace NUMINAMATH_GPT_find_constants_monotonicity_l1686_168634

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem find_constants (a b c : ℝ) 
  (h1 : f' (-2/3) a b = 0)
  (h2 : f' 1 a b = 0) :
  a = -1/2 ∧ b = -2 :=
by sorry

theorem monotonicity (a b c : ℝ)
  (h1 : a = -1/2) 
  (h2 : b = -2) : 
  (∀ x : ℝ, x < -2/3 → f' x a b > 0) ∧ 
  (∀ x : ℝ, -2/3 < x ∧ x < 1 → f' x a b < 0) ∧ 
  (∀ x : ℝ, x > 1 → f' x a b > 0) :=
by sorry

end NUMINAMATH_GPT_find_constants_monotonicity_l1686_168634


namespace NUMINAMATH_GPT_circle_rolling_start_point_l1686_168657

theorem circle_rolling_start_point (x : ℝ) (h1 : ∃ x, (x + 2 * Real.pi = -1) ∨ (x - 2 * Real.pi = -1)) :
  x = -1 - 2 * Real.pi ∨ x = -1 + 2 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_rolling_start_point_l1686_168657


namespace NUMINAMATH_GPT_rate_per_meter_for_fencing_l1686_168632

theorem rate_per_meter_for_fencing
  (w : ℕ) (length : ℕ) (perimeter : ℕ) (cost : ℕ)
  (h1 : length = w + 10)
  (h2 : perimeter = 2 * (length + w))
  (h3 : perimeter = 340)
  (h4 : cost = 2210) : (cost / perimeter : ℝ) = 6.5 := by
  sorry

end NUMINAMATH_GPT_rate_per_meter_for_fencing_l1686_168632


namespace NUMINAMATH_GPT_mean_proportional_l1686_168678

theorem mean_proportional (a c x : ℝ) (ha : a = 9) (hc : c = 4) (hx : x^2 = a * c) : x = 6 := by
  sorry

end NUMINAMATH_GPT_mean_proportional_l1686_168678


namespace NUMINAMATH_GPT_tangent_points_are_on_locus_l1686_168648

noncomputable def tangent_points_locus (d : ℝ) : Prop :=
∀ (x y : ℝ), 
((x ≠ 0 ∨ y ≠ 0) ∧ (x-d ≠ 0)) ∧ (y = x) 
→ (y^2 - x*y + d*(x + y) = 0)

theorem tangent_points_are_on_locus (d : ℝ) : 
  tangent_points_locus d :=
by sorry

end NUMINAMATH_GPT_tangent_points_are_on_locus_l1686_168648


namespace NUMINAMATH_GPT_coefficient_of_x3_in_expansion_l1686_168633

theorem coefficient_of_x3_in_expansion :
  (∀ (x : ℝ), (Polynomial.coeff ((Polynomial.C x - 1)^5) 3) = 10) :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_x3_in_expansion_l1686_168633


namespace NUMINAMATH_GPT_number_of_acceptable_outfits_l1686_168684

-- Definitions based on conditions
def total_shirts := 5
def total_pants := 4
def restricted_shirts := 2
def restricted_pants := 1

-- Defining the problem statement
theorem number_of_acceptable_outfits : 
  (total_shirts * total_pants - restricted_shirts * restricted_pants + restricted_shirts * (total_pants - restricted_pants)) = 18 :=
by sorry

end NUMINAMATH_GPT_number_of_acceptable_outfits_l1686_168684


namespace NUMINAMATH_GPT_M_is_correct_ab_property_l1686_168622

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 1|
def M : Set ℝ := {x | f x < 4}

theorem M_is_correct : M = {x | -2 < x ∧ x < 2} :=
sorry

theorem ab_property (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 2 * |a + b| < |4 + a * b| :=
sorry

end NUMINAMATH_GPT_M_is_correct_ab_property_l1686_168622


namespace NUMINAMATH_GPT_rows_identical_l1686_168682

theorem rows_identical {n : ℕ} {a : Fin n → ℝ} {k : Fin n → Fin n}
  (h_inc : ∀ i j : Fin n, i < j → a i < a j)
  (h_perm : ∀ i j : Fin n, k i ≠ k j → a (k i) ≠ a (k j))
  (h_sum_inc : ∀ i j : Fin n, i < j → a i + a (k i) < a j + a (k j)) :
  ∀ i : Fin n, a i = a (k i) :=
by
  sorry

end NUMINAMATH_GPT_rows_identical_l1686_168682


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1686_168663

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 12 < 0 } = { x : ℝ | -4 < x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1686_168663


namespace NUMINAMATH_GPT_regular_pyramid_cannot_be_hexagonal_l1686_168665

theorem regular_pyramid_cannot_be_hexagonal (n : ℕ) (h₁ : n = 6) (base_edge_length slant_height : ℝ) 
  (reg_pyramid : base_edge_length = slant_height) : false :=
by
  sorry

end NUMINAMATH_GPT_regular_pyramid_cannot_be_hexagonal_l1686_168665


namespace NUMINAMATH_GPT_polynomial_coeff_sum_l1686_168601

theorem polynomial_coeff_sum :
  let f : ℕ → ℕ := λ x => (x^2 + 1) * (x - 1)^8
  let a_0 := f 2
  let a_sum := f 3
  a_sum - a_0 = 2555 :=
by
  let f : ℕ → ℕ := λ x => (x^2 + 1) * (x - 1)^8
  let a_0 := f 2
  let a_sum := f 3
  show a_sum - a_0 = 2555
  sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_l1686_168601


namespace NUMINAMATH_GPT_words_per_hour_after_two_hours_l1686_168668

theorem words_per_hour_after_two_hours 
  (total_words : ℕ) (initial_rate : ℕ) (initial_time : ℕ) (start_time_before_deadline : ℕ) 
  (words_written_in_first_phase : ℕ) (remaining_words : ℕ) (remaining_time : ℕ)
  (final_rate_per_hour : ℕ) :
  total_words = 1200 →
  initial_rate = 400 →
  initial_time = 2 →
  start_time_before_deadline = 4 →
  words_written_in_first_phase = initial_rate * initial_time →
  remaining_words = total_words - words_written_in_first_phase →
  remaining_time = start_time_before_deadline - initial_time →
  final_rate_per_hour = remaining_words / remaining_time →
  final_rate_per_hour = 200 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_words_per_hour_after_two_hours_l1686_168668


namespace NUMINAMATH_GPT_triangle_acute_angle_l1686_168688

theorem triangle_acute_angle 
  (a b c : ℝ) 
  (h1 : a^3 = b^3 + c^3)
  (h2 : a > b)
  (h3 : a > c)
  (h4 : b > 0) 
  (h5 : c > 0) 
  (h6 : a > 0) 
  : 
  (a^2 < b^2 + c^2) :=
sorry

end NUMINAMATH_GPT_triangle_acute_angle_l1686_168688


namespace NUMINAMATH_GPT_infinite_a_exists_l1686_168664

theorem infinite_a_exists (n : ℕ) : ∃ (a : ℕ), ∀ (k : ℕ), ∃ (a : ℕ), n^6 + 3 * a = (n^2 + 3 * k)^3 := 
sorry

end NUMINAMATH_GPT_infinite_a_exists_l1686_168664


namespace NUMINAMATH_GPT_Q_ratio_eq_one_l1686_168685

noncomputable def g (x : ℂ) : ℂ := x^2007 - 2 * x^2006 + 2

theorem Q_ratio_eq_one (Q : ℂ → ℂ) (s : ℕ → ℂ) (h_root : ∀ j : ℕ, j < 2007 → g (s j) = 0) 
  (h_Q : ∀ j : ℕ, j < 2007 → Q (s j + (1 / s j)) = 0) :
  Q 1 / Q (-1) = 1 := by
  sorry

end NUMINAMATH_GPT_Q_ratio_eq_one_l1686_168685


namespace NUMINAMATH_GPT_A_salary_l1686_168616

theorem A_salary (x y : ℝ) (h1 : x + y = 7000) (h2 : 0.05 * x = 0.15 * y) : x = 5250 :=
by
  sorry

end NUMINAMATH_GPT_A_salary_l1686_168616


namespace NUMINAMATH_GPT_range_of_a_minus_b_l1686_168619

theorem range_of_a_minus_b {a b : ℝ} (h1 : -2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) : 
  -4 < a - b ∧ a - b < 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_minus_b_l1686_168619


namespace NUMINAMATH_GPT_percent_correct_both_l1686_168681

-- Definitions based on given conditions in the problem
def P_A : ℝ := 0.63
def P_B : ℝ := 0.50
def P_not_A_and_not_B : ℝ := 0.20

-- Definition of the desired result using the inclusion-exclusion principle based on the given conditions
def P_A_and_B : ℝ := P_A + P_B - (1 - P_not_A_and_not_B)

-- Theorem stating our goal: proving the probability of both answering correctly is 0.33
theorem percent_correct_both : P_A_and_B = 0.33 := by
  sorry

end NUMINAMATH_GPT_percent_correct_both_l1686_168681


namespace NUMINAMATH_GPT_initial_mean_corrected_l1686_168607

theorem initial_mean_corrected (M : ℝ) (H : 30 * M + 30 = 30 * 151) : M = 150 :=
sorry

end NUMINAMATH_GPT_initial_mean_corrected_l1686_168607


namespace NUMINAMATH_GPT_find_value_m_sq_plus_2m_plus_n_l1686_168625

noncomputable def m_n_roots (x : ℝ) : Prop := x^2 + x - 1001 = 0

theorem find_value_m_sq_plus_2m_plus_n
  (m n : ℝ)
  (hm : m_n_roots m)
  (hn : m_n_roots n)
  (h_sum : m + n = -1)
  (h_prod : m * n = -1001) :
  m^2 + 2 * m + n = 1000 :=
sorry

end NUMINAMATH_GPT_find_value_m_sq_plus_2m_plus_n_l1686_168625


namespace NUMINAMATH_GPT_gcd_lcm_sum_correct_l1686_168697

def gcd_lcm_sum : ℕ :=
  let gcd_40_60 := Nat.gcd 40 60
  let lcm_20_15 := Nat.lcm 20 15
  gcd_40_60 + 2 * lcm_20_15

theorem gcd_lcm_sum_correct : gcd_lcm_sum = 140 := by
  -- Definitions based on conditions
  let gcd_40_60 := Nat.gcd 40 60
  let lcm_20_15 := Nat.lcm 20 15
  
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_correct_l1686_168697


namespace NUMINAMATH_GPT_computer_price_increase_l1686_168628

theorem computer_price_increase (d : ℝ) (h : 2 * d = 585) : d * 1.2 = 351 := by
  sorry

end NUMINAMATH_GPT_computer_price_increase_l1686_168628


namespace NUMINAMATH_GPT_f_1996x_l1686_168610

noncomputable def f : ℝ → ℝ := sorry

axiom f_equation (x y : ℝ) : f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f y)^2)

theorem f_1996x (x : ℝ) : f (1996 * x) = 1996 * f x :=
sorry

end NUMINAMATH_GPT_f_1996x_l1686_168610


namespace NUMINAMATH_GPT_polynomial_product_l1686_168695

theorem polynomial_product (a b c : ℝ) :
  a * (b - c) ^ 3 + b * (c - a) ^ 3 + c * (a - b) ^ 3 = (a - b) * (b - c) * (c - a) * (a + b + c) :=
by sorry

end NUMINAMATH_GPT_polynomial_product_l1686_168695


namespace NUMINAMATH_GPT_converse_even_power_divisible_l1686_168689

theorem converse_even_power_divisible (n : ℕ) (h_even : ∀ (k : ℕ), n = 2 * k → (3^n + 63) % 72 = 0) :
  (3^n + 63) % 72 = 0 → ∃ (k : ℕ), n = 2 * k :=
by sorry

end NUMINAMATH_GPT_converse_even_power_divisible_l1686_168689


namespace NUMINAMATH_GPT_correct_angle_calculation_l1686_168686

theorem correct_angle_calculation (α β : ℝ) (hα : 0 < α ∧ α < 90) (hβ : 90 < β ∧ β < 180) :
    22.5 < 0.25 * (α + β) ∧ 0.25 * (α + β) < 67.5 → 0.25 * (α + β) = 45.3 :=
by
  sorry

end NUMINAMATH_GPT_correct_angle_calculation_l1686_168686


namespace NUMINAMATH_GPT_company_max_revenue_l1686_168680

structure Conditions where
  max_total_time : ℕ -- maximum total time in minutes
  max_total_cost : ℕ -- maximum total cost in yuan
  rate_A : ℕ -- rate per minute for TV A in yuan
  rate_B : ℕ -- rate per minute for TV B in yuan
  revenue_A : ℕ -- revenue per minute for TV A in million yuan
  revenue_B : ℕ -- revenue per minute for TV B in million yuan

def company_conditions : Conditions :=
  { max_total_time := 300,
    max_total_cost := 90000,
    rate_A := 500,
    rate_B := 200,
    revenue_A := 3, -- as 0.3 million yuan converted to 3 tenths (integer representation)
    revenue_B := 2  -- as 0.2 million yuan converted to 2 tenths (integer representation)
  }

def advertising_strategy
  (conditions : Conditions)
  (time_A : ℕ) (time_B : ℕ) : Prop :=
  time_A + time_B ≤ conditions.max_total_time ∧
  time_A * conditions.rate_A + time_B * conditions.rate_B ≤ conditions.max_total_cost

def revenue
  (conditions : Conditions)
  (time_A : ℕ) (time_B : ℕ) : ℕ :=
  time_A * conditions.revenue_A + time_B * conditions.revenue_B

theorem company_max_revenue (time_A time_B : ℕ)
  (h : advertising_strategy company_conditions time_A time_B) :
  revenue company_conditions time_A time_B = 70 := 
  by
  have h1 : time_A = 100 := sorry
  have h2 : time_B = 200 := sorry
  sorry

end NUMINAMATH_GPT_company_max_revenue_l1686_168680


namespace NUMINAMATH_GPT_sum_of_fractions_l1686_168661

theorem sum_of_fractions : 
  (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) = 3 / 8) :=
by sorry

end NUMINAMATH_GPT_sum_of_fractions_l1686_168661


namespace NUMINAMATH_GPT_hyperbola_condition_l1686_168690

theorem hyperbola_condition (k : ℝ) : 
  (0 ≤ k ∧ k < 3) → (∃ a b : ℝ, a * b < 0 ∧ 
    (a = k + 1) ∧ (b = k - 5)) ∧ (∀ m : ℝ, -1 < m ∧ m < 5 → ∃ a b : ℝ, a * b < 0 ∧ 
    (a = m + 1) ∧ (b = m - 5)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l1686_168690


namespace NUMINAMATH_GPT_abs_inequality_condition_l1686_168650

theorem abs_inequality_condition (a : ℝ) : 
  (a < 2) ↔ ∀ x : ℝ, |x - 2| + |x| > a :=
sorry

end NUMINAMATH_GPT_abs_inequality_condition_l1686_168650


namespace NUMINAMATH_GPT_tank_fewer_eggs_in_second_round_l1686_168630

variables (T E_total T_r2_diff : ℕ)

theorem tank_fewer_eggs_in_second_round
  (h1 : E_total = 400)
  (h2 : E_total = (T + (T - 10)) + (30 + 60))
  (h3 : T_r2_diff = T - 30) :
  T_r2_diff = 130 := by
    sorry

end NUMINAMATH_GPT_tank_fewer_eggs_in_second_round_l1686_168630


namespace NUMINAMATH_GPT_age_difference_l1686_168612

theorem age_difference (x y : ℕ) (h1 : 3 * x + 4 * x = 42) (h2 : 18 - y = (24 - y) / 2) : 
  y = 12 :=
  sorry

end NUMINAMATH_GPT_age_difference_l1686_168612


namespace NUMINAMATH_GPT_five_minus_a_l1686_168673

theorem five_minus_a (a b : ℚ) (h1 : 5 + a = 3 - b) (h2 : 3 + b = 8 + a) : 5 - a = 17/2 :=
by
  sorry

end NUMINAMATH_GPT_five_minus_a_l1686_168673


namespace NUMINAMATH_GPT_circle_Γ_contains_exactly_one_l1686_168636

-- Condition definitions
variables (z1 z2 : ℂ) (Γ : ℂ → ℂ → Prop)
variable (hz1z2 : z1 * z2 = 1)
variable (hΓ_passes : Γ (-1) 1)
variable (hΓ_not_passes : ¬Γ z1 z2)

-- Math proof problem
theorem circle_Γ_contains_exactly_one (hz1z2 : z1 * z2 = 1)
    (hΓ_passes : Γ (-1) 1) (hΓ_not_passes : ¬Γ z1 z2) : 
  (Γ 0 z1 ↔ ¬Γ 0 z2) ∨ (Γ 0 z2 ↔ ¬Γ 0 z1) :=
sorry

end NUMINAMATH_GPT_circle_Γ_contains_exactly_one_l1686_168636


namespace NUMINAMATH_GPT_pages_in_book_l1686_168696

-- Define the initial conditions
variable (P : ℝ) -- total number of pages in the book
variable (h_read_20_percent : 0.20 * P = 320 * 0.20 / 0.80) -- Nate has read 20% of the book and the rest 80%

-- The goal is to show that P = 400
theorem pages_in_book (P : ℝ) :
  (0.80 * P = 320) → P = 400 :=
by
  sorry

end NUMINAMATH_GPT_pages_in_book_l1686_168696


namespace NUMINAMATH_GPT_profits_equal_l1686_168635

-- Define the profit variables
variables (profitA profitB profitC profitD : ℝ)

-- The conditions
def storeA_profit : profitA = 1.2 * profitB := sorry
def storeB_profit : profitB = 1.2 * profitC := sorry
def storeD_profit : profitD = profitA * 0.6 := sorry

-- The statement to be proven
theorem profits_equal : profitC = profitD :=
by sorry

end NUMINAMATH_GPT_profits_equal_l1686_168635


namespace NUMINAMATH_GPT_second_number_more_than_first_l1686_168654

-- Definitions of A and B based on the given ratio
def A : ℚ := 7 / 56
def B : ℚ := 8 / 56

-- Proof statement
theorem second_number_more_than_first : ((B - A) / A) * 100 = 100 / 7 :=
by
  -- skipped the proof
  sorry

end NUMINAMATH_GPT_second_number_more_than_first_l1686_168654


namespace NUMINAMATH_GPT_distance_to_SFL_l1686_168656

def distance_per_hour : ℕ := 27
def hours_travelled : ℕ := 3

theorem distance_to_SFL :
  (distance_per_hour * hours_travelled) = 81 := 
by
  sorry

end NUMINAMATH_GPT_distance_to_SFL_l1686_168656


namespace NUMINAMATH_GPT_rate_of_current_l1686_168674

-- Definitions of the conditions
def downstream_speed : ℝ := 30  -- in kmph
def upstream_speed : ℝ := 10    -- in kmph
def still_water_rate : ℝ := 20  -- in kmph

-- Calculating the rate of the current
def current_rate : ℝ := downstream_speed - still_water_rate

-- Proof statement
theorem rate_of_current :
  current_rate = 10 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_current_l1686_168674


namespace NUMINAMATH_GPT_ratio_of_polynomials_eq_962_l1686_168651

open Real

theorem ratio_of_polynomials_eq_962 :
  (10^4 + 400) * (26^4 + 400) * (42^4 + 400) * (58^4 + 400) /
  ((2^4 + 400) * (18^4 + 400) * (34^4 + 400) * (50^4 + 400)) = 962 := 
sorry

end NUMINAMATH_GPT_ratio_of_polynomials_eq_962_l1686_168651


namespace NUMINAMATH_GPT_train_length_l1686_168693

noncomputable def speed_kmph := 80
noncomputable def time_seconds := 5

 noncomputable def speed_mps := (speed_kmph * 1000) / 3600

 noncomputable def length_train : ℝ := speed_mps * time_seconds

theorem train_length : length_train = 111.1 := by
  sorry

end NUMINAMATH_GPT_train_length_l1686_168693


namespace NUMINAMATH_GPT_fiona_first_to_toss_eight_l1686_168644

theorem fiona_first_to_toss_eight :
  (∃ p : ℚ, p = 49/169 ∧
    (∀ n:ℕ, (7/8:ℚ)^(3*n) * (1/8) = if n = 0 then (49/512) else (49/512) * (343/512)^n)) :=
sorry

end NUMINAMATH_GPT_fiona_first_to_toss_eight_l1686_168644


namespace NUMINAMATH_GPT_regular_train_pass_time_l1686_168615

-- Define the lengths of the trains
def high_speed_train_length : ℕ := 400
def regular_train_length : ℕ := 600

-- Define the observation time for the passenger on the high-speed train
def observation_time : ℕ := 3

-- Define the problem to find the time x for the regular train passenger
theorem regular_train_pass_time :
  ∃ (x : ℕ), (regular_train_length / observation_time) * x = high_speed_train_length :=
by 
  sorry

end NUMINAMATH_GPT_regular_train_pass_time_l1686_168615


namespace NUMINAMATH_GPT_savings_account_amount_l1686_168647

noncomputable def final_amount : ℝ :=
  let initial_deposit : ℝ := 5000
  let first_quarter_rate : ℝ := 0.01
  let second_quarter_rate : ℝ := 0.0125
  let deposit_end_third_month : ℝ := 1000
  let withdrawal_end_fifth_month : ℝ := 500
  let amount_after_first_quarter := initial_deposit * (1 + first_quarter_rate)
  let amount_before_second_quarter := amount_after_first_quarter + deposit_end_third_month
  let amount_after_second_quarter := amount_before_second_quarter * (1 + second_quarter_rate)
  let final_amount := amount_after_second_quarter - withdrawal_end_fifth_month
  final_amount

theorem savings_account_amount :
  final_amount = 5625.625 :=
by
  sorry

end NUMINAMATH_GPT_savings_account_amount_l1686_168647


namespace NUMINAMATH_GPT_machine_A_sprockets_per_hour_l1686_168653

-- Definitions based on the problem conditions
def MachineP_time (A : ℝ) (T : ℝ) : ℝ := T + 10
def MachineQ_rate (A : ℝ) : ℝ := 1.1 * A
def MachineP_sprockets (A : ℝ) (T : ℝ) : ℝ := A * (T + 10)
def MachineQ_sprockets (A : ℝ) (T : ℝ) : ℝ := 1.1 * A * T

-- Lean proof statement to prove that Machine A produces 8 sprockets per hour
theorem machine_A_sprockets_per_hour :
  ∀ A T : ℝ, 
  880 = MachineP_sprockets A T ∧
  880 = MachineQ_sprockets A T →
  A = 8 :=
by
  intros A T h
  have h1 : 880 = MachineP_sprockets A T := h.left
  have h2 : 880 = MachineQ_sprockets A T := h.right
  sorry

end NUMINAMATH_GPT_machine_A_sprockets_per_hour_l1686_168653


namespace NUMINAMATH_GPT_simplify_fraction_l1686_168606

theorem simplify_fraction (m : ℝ) (hm : m ≠ 0) : (3 * m^3) / (6 * m^2) = m / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1686_168606


namespace NUMINAMATH_GPT_total_number_of_digits_l1686_168667

theorem total_number_of_digits (n S S₅ S₃ : ℕ) (h1 : S = 20 * n) (h2 : S₅ = 5 * 12) (h3 : S₃ = 3 * 33) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_digits_l1686_168667


namespace NUMINAMATH_GPT_solution_set_l1686_168692

def inequality_solution (x : ℝ) : Prop :=
  4 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9

theorem solution_set :
  { x : ℝ | inequality_solution x } = { x : ℝ | (63 / 26 : ℝ) < x ∧ x ≤ (28 / 11 : ℝ) } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1686_168692


namespace NUMINAMATH_GPT_compute_x_plus_y_l1686_168670

theorem compute_x_plus_y :
    ∃ (x y : ℕ), 4 * y = 7 * 84 ∧ 4 * 63 = 7 * x ∧ x + y = 183 :=
by
  sorry

end NUMINAMATH_GPT_compute_x_plus_y_l1686_168670


namespace NUMINAMATH_GPT_hypotenuse_eq_medians_l1686_168613

noncomputable def hypotenuse_length_medians (a b : ℝ) (h1 : b^2 + (9 * a^2) / 4 = 48) (h2 : a^2 + (9 * b^2) / 4 = 36) : ℝ :=
  3 * Real.sqrt (336 / 13)

-- definition
theorem hypotenuse_eq_medians {a b : ℝ} (h1 : b^2 + (9 * a^2) / 4 = 48) (h2 : a^2 + (9 * b^2) / 4 = 36) :
    Real.sqrt (9 * (a^2 + b^2)) = 3 * Real.sqrt (336 / 13) :=
sorry

end NUMINAMATH_GPT_hypotenuse_eq_medians_l1686_168613


namespace NUMINAMATH_GPT_cryptarithm_solution_l1686_168679

theorem cryptarithm_solution (A B : ℕ) (h_digit_A : A < 10) (h_digit_B : B < 10)
  (h_equation : 9 * (10 * A + B) = 110 * A + B) :
  A = 2 ∧ B = 5 :=
sorry

end NUMINAMATH_GPT_cryptarithm_solution_l1686_168679


namespace NUMINAMATH_GPT_tan_phi_l1686_168659

theorem tan_phi (φ : ℝ) (h1 : Real.cos (π / 2 + φ) = 2 / 3) (h2 : abs φ < π / 2) : 
  Real.tan φ = -2 * Real.sqrt 5 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_tan_phi_l1686_168659


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1686_168642

theorem solution_set_of_inequality (a : ℝ) :
  ¬ (∀ x : ℝ, ¬ (a * (x - a) * (a * x + a) ≥ 0)) ∧
  ¬ (∀ x : ℝ, (a - x ≤ 0 ∧ x - (-1) ≤ 0 → a * (x - a) * (a * x + a) ≥ 0)) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1686_168642


namespace NUMINAMATH_GPT_tangent_curve_line_a_eq_neg1_l1686_168620

theorem tangent_curve_line_a_eq_neg1 (a : ℝ) (x : ℝ) : 
  (∀ (x : ℝ), (e^x + a = x) ∧ (e^x = 1) ) → a = -1 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_tangent_curve_line_a_eq_neg1_l1686_168620


namespace NUMINAMATH_GPT_arnold_and_danny_age_l1686_168608

theorem arnold_and_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 9) : x = 4 :=
sorry

end NUMINAMATH_GPT_arnold_and_danny_age_l1686_168608


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1686_168609

theorem sufficient_but_not_necessary_condition (x : ℝ) (hx : x > 1) : x > 1 → x^2 > 1 ∧ ∀ y, (y^2 > 1 → ¬(y ≥ 1) → y < -1) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1686_168609


namespace NUMINAMATH_GPT_intersection_complement_l1686_168655

def A := {x : ℝ | -1 < x ∧ x < 6}
def B := {x : ℝ | x^2 < 4}
def complement_R (S : Set ℝ) := {x : ℝ | x ∉ S}

theorem intersection_complement :
  A ∩ (complement_R B) = {x : ℝ | 2 ≤ x ∧ x < 6} := by
sorry

end NUMINAMATH_GPT_intersection_complement_l1686_168655


namespace NUMINAMATH_GPT_area_R2_l1686_168691

-- Definitions from conditions
def side_R1 : ℕ := 3
def area_R1 : ℕ := 24
def diagonal_ratio : ℤ := 2

-- Introduction of the theorem
theorem area_R2 (similar: ℤ) (a b: ℕ) :
  a * b = area_R1 ∧
  a = 3 ∧
  b * 3 = 8 * a ∧
  (a^2 + b^2 = 292) ∧
  similar * (a^2 + b^2) = 2 * 2 * 73 →
  (6 * 16 = 96) := by
sorry

end NUMINAMATH_GPT_area_R2_l1686_168691


namespace NUMINAMATH_GPT_inequality_range_a_l1686_168641

open Real

theorem inequality_range_a (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 :=
sorry

end NUMINAMATH_GPT_inequality_range_a_l1686_168641


namespace NUMINAMATH_GPT_hot_dogs_left_over_l1686_168621

theorem hot_dogs_left_over : 25197629 % 6 = 5 := 
sorry

end NUMINAMATH_GPT_hot_dogs_left_over_l1686_168621


namespace NUMINAMATH_GPT_club_members_neither_subject_l1686_168623

theorem club_members_neither_subject (total members_cs members_bio members_both : ℕ)
  (h_total : total = 150)
  (h_cs : members_cs = 80)
  (h_bio : members_bio = 50)
  (h_both : members_both = 15) :
  total - ((members_cs - members_both) + (members_bio - members_both) + members_both) = 35 := by
  sorry

end NUMINAMATH_GPT_club_members_neither_subject_l1686_168623


namespace NUMINAMATH_GPT_PQRS_value_l1686_168617

theorem PQRS_value
  (P Q R S : ℝ)
  (hP : 0 < P)
  (hQ : 0 < Q)
  (hR : 0 < R)
  (hS : 0 < S)
  (h1 : Real.log (P * Q) / Real.log 10 + Real.log (P * S) / Real.log 10 = 2)
  (h2 : Real.log (Q * S) / Real.log 10 + Real.log (Q * R) / Real.log 10 = 3)
  (h3 : Real.log (R * P) / Real.log 10 + Real.log (R * S) / Real.log 10 = 5) :
  P * Q * R * S = 100000 := 
sorry

end NUMINAMATH_GPT_PQRS_value_l1686_168617


namespace NUMINAMATH_GPT_mod_multiplication_result_l1686_168605

theorem mod_multiplication_result :
  ∃ n : ℕ, 507 * 873 ≡ n [MOD 77] ∧ 0 ≤ n ∧ n < 77 ∧ n = 15 := by
  sorry

end NUMINAMATH_GPT_mod_multiplication_result_l1686_168605


namespace NUMINAMATH_GPT_fresh_fruit_sold_l1686_168662

variable (total_fruit frozen_fruit : ℕ)

theorem fresh_fruit_sold (h1 : total_fruit = 9792) (h2 : frozen_fruit = 3513) : 
  total_fruit - frozen_fruit = 6279 :=
by sorry

end NUMINAMATH_GPT_fresh_fruit_sold_l1686_168662


namespace NUMINAMATH_GPT_cost_price_of_watch_l1686_168646

/-
Let's state the problem conditions as functions
C represents the cost price
SP1 represents the selling price at 36% loss
SP2 represents the selling price at 4% gain
-/

def cost_price (C : ℝ) : ℝ := C

def selling_price_loss (C : ℝ) : ℝ := 0.64 * C

def selling_price_gain (C : ℝ) : ℝ := 1.04 * C

def price_difference (C : ℝ) : ℝ := (selling_price_gain C) - (selling_price_loss C)

theorem cost_price_of_watch : ∀ C : ℝ, price_difference C = 140 → C = 350 :=
by
   intro C H
   sorry

end NUMINAMATH_GPT_cost_price_of_watch_l1686_168646


namespace NUMINAMATH_GPT_no_solution_pos_integers_l1686_168671

theorem no_solution_pos_integers (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a + b + c + d - 3 ≠ a * b + c * d := 
by
  sorry

end NUMINAMATH_GPT_no_solution_pos_integers_l1686_168671


namespace NUMINAMATH_GPT_collinear_R_S_T_l1686_168637

theorem collinear_R_S_T
    (circle : Type)
    (P : circle)
    (A B C D : circle)
    (E F : Type → Type)
    (angle : ∀ (x y z : circle), ℝ)   -- Placeholder for angles
    (quadrilateral_inscribed_in_circle : ∀ (A B C D : circle), Prop)   -- Placeholder for the condition of the quadrilateral
    (extensions_intersect : ∀ (A B C D : circle) (E F : Type → Type), Prop)   -- Placeholder for extensions intersections
    (diagonals_intersect_at : ∀ (A C B D T : circle), Prop)   -- Placeholder for diagonals intersections
    (P_on_circle : ∀ (P : circle), Prop)        -- Point P is on the circle
    (PE_PF_intersect_again : ∀ (P R S : circle) (E F : Type → Type), Prop)   -- PE and PF intersect the circle again at R and S
    (R S T : circle) :
    quadrilateral_inscribed_in_circle A B C D →
    extensions_intersect A B C D E F →
    P_on_circle P →
    PE_PF_intersect_again P R S E F →
    diagonals_intersect_at A C B D T →
    ∃ collinearity : ∀ (R S T : circle), Prop,
    collinearity R S T := 
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_collinear_R_S_T_l1686_168637


namespace NUMINAMATH_GPT_hyperbola_standard_equation_l1686_168675

open Real

noncomputable def distance_from_center_to_focus (a b : ℝ) : ℝ := sqrt (a^2 - b^2)

theorem hyperbola_standard_equation (a b c : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : b = sqrt 3 * c)
  (h4 : a + c = 3 * sqrt 3) :
  ∃ h : a^2 = 12 ∧ b = 3, y^2 / 12 - x^2 / 9 = 1 :=
sorry

end NUMINAMATH_GPT_hyperbola_standard_equation_l1686_168675


namespace NUMINAMATH_GPT_find_integer_mod_condition_l1686_168660

theorem find_integer_mod_condition (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 4) (h3 : n ≡ -998 [ZMOD 5]) : n = 2 :=
sorry

end NUMINAMATH_GPT_find_integer_mod_condition_l1686_168660


namespace NUMINAMATH_GPT_determine_flower_responsibility_l1686_168602

-- Define the structure of the grid
structure Grid (m n : ℕ) :=
  (vertices : Fin m → Fin n → Bool) -- True if gardener lives at the vertex

-- Define a function to determine if 3 gardeners are nearest to a flower
def is_nearest (i j fi fj : ℕ) : Bool :=
  -- Assume this function gives true if the gardener at (i, j) is one of the 3 nearest to the flower at (fi, fj)
  sorry

-- The main theorem statement
theorem determine_flower_responsibility 
  {m n : ℕ} 
  (G : Grid m n) 
  (i j : Fin m) 
  (k : Fin n) 
  (h : G.vertices i k = true) 
  : ∃ (fi fj : ℕ), is_nearest (i : ℕ) (k : ℕ) fi fj = true := 
sorry

end NUMINAMATH_GPT_determine_flower_responsibility_l1686_168602


namespace NUMINAMATH_GPT_find_number_l1686_168604

theorem find_number (x : ℤ) (h : (85 + x) * 1 = 9637) : x = 9552 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1686_168604


namespace NUMINAMATH_GPT_average_rate_l1686_168669

theorem average_rate (distance_run distance_swim : ℝ) (rate_run rate_swim : ℝ) 
  (h1 : distance_run = 2) (h2 : distance_swim = 2) (h3 : rate_run = 10) (h4 : rate_swim = 5) : 
  (distance_run + distance_swim) / ((distance_run / rate_run) * 60 + (distance_swim / rate_swim) * 60) = 0.1111 :=
by
  sorry

end NUMINAMATH_GPT_average_rate_l1686_168669


namespace NUMINAMATH_GPT_plane_eq_unique_l1686_168645

open Int 

def plane_eq (A B C D x y z : ℤ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_eq_unique (x y z : ℤ) (A B C D : ℤ)
  (h₁ : x = 8) 
  (h₂ : y = -6) 
  (h₃ : z = 2) 
  (h₄ : A > 0)
  (h₅ : gcd (|A|) (gcd (|B|) (gcd (|C|) (|D|))) = 1) :
  plane_eq 4 (-3) 1 (-52) x y z :=
by
  sorry

end NUMINAMATH_GPT_plane_eq_unique_l1686_168645


namespace NUMINAMATH_GPT_find_sum_f_neg1_f_3_l1686_168672

noncomputable def f : ℝ → ℝ := sorry

-- condition: odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - f x

-- condition: symmetry around x=1
def symmetric_around_one (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (1 - x) = f (1 + x)

-- condition: specific value at x=1
def value_at_one (f : ℝ → ℝ) : Prop := f 1 = 2

-- Theorem to prove
theorem find_sum_f_neg1_f_3 (h1 : odd_function f) (h2 : symmetric_around_one f) (h3 : value_at_one f) : f (-1) + f 3 = -4 := by
  sorry

end NUMINAMATH_GPT_find_sum_f_neg1_f_3_l1686_168672


namespace NUMINAMATH_GPT_cone_base_radius_l1686_168643

/-- A hemisphere of radius 3 rests on the base of a circular cone and is tangent to the cone's lateral surface along a circle. 
Given that the height of the cone is 9, prove that the base radius of the cone is 10.5. -/
theorem cone_base_radius
  (r_h : ℝ) (h : ℝ) (r : ℝ) 
  (hemisphere_tangent_cone : r_h = 3)
  (cone_height : h = 9)
  (tangent_circle_height : r - r_h = 3) :
  r = 10.5 := by
  sorry

end NUMINAMATH_GPT_cone_base_radius_l1686_168643


namespace NUMINAMATH_GPT_count_real_numbers_a_with_integer_roots_l1686_168600

theorem count_real_numbers_a_with_integer_roots :
  ∃ (S : Finset ℝ), (∀ (a : ℝ), (∃ (x y : ℤ), x^2 + a*x + 9*a = 0 ∧ y^2 + a*y + 9*a = 0) ↔ a ∈ S) ∧ S.card = 8 :=
by
  sorry

end NUMINAMATH_GPT_count_real_numbers_a_with_integer_roots_l1686_168600


namespace NUMINAMATH_GPT_plot_length_60_l1686_168631

/-- The length of a rectangular plot is 20 meters more than its breadth. If the cost of fencing the plot at Rs. 26.50 per meter is Rs. 5300, then the length of the plot in meters is 60. -/
theorem plot_length_60 (b l : ℝ) (h1 : l = b + 20) (h2 : 2 * (l + b) * 26.5 = 5300) : l = 60 :=
by
  sorry

end NUMINAMATH_GPT_plot_length_60_l1686_168631


namespace NUMINAMATH_GPT_minimum_value_of_x_squared_l1686_168640

theorem minimum_value_of_x_squared : ∃ x : ℝ, x = 0 ∧ ∀ y : ℝ, y = x^2 → y ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_x_squared_l1686_168640


namespace NUMINAMATH_GPT_tax_percentage_excess_l1686_168698

/--
In Country X, each citizen is taxed an amount equal to 15 percent of the first $40,000 of income,
plus a certain percentage of all income in excess of $40,000. A citizen of Country X is taxed a total of $8,000
and her income is $50,000.

Prove that the percentage of the tax on the income in excess of $40,000 is 20%.
-/
theorem tax_percentage_excess (total_tax : ℝ) (first_income : ℝ) (additional_income : ℝ) (income : ℝ) (tax_first_part : ℝ) (tax_rate_first_part : ℝ) (tax_rate_excess : ℝ) (tax_excess : ℝ) :
  total_tax = 8000 →
  first_income = 40000 →
  additional_income = 10000 →
  income = first_income + additional_income →
  tax_rate_first_part = 0.15 →
  tax_first_part = tax_rate_first_part * first_income →
  tax_excess = total_tax - tax_first_part →
  tax_rate_excess * additional_income = tax_excess →
  tax_rate_excess = 0.20 :=
by
  intro h_total_tax h_first_income h_additional_income h_income h_tax_rate_first_part h_tax_first_part h_tax_excess h_tax_equation
  sorry

end NUMINAMATH_GPT_tax_percentage_excess_l1686_168698


namespace NUMINAMATH_GPT_find_number_eq_36_l1686_168626

theorem find_number_eq_36 (n : ℝ) (h : (n / 18) * (n / 72) = 1) : n = 36 :=
sorry

end NUMINAMATH_GPT_find_number_eq_36_l1686_168626


namespace NUMINAMATH_GPT_susan_ate_candies_l1686_168629

theorem susan_ate_candies (candies_tuesday candies_thursday candies_friday candies_left : ℕ) 
  (h_tuesday : candies_tuesday = 3) 
  (h_thursday : candies_thursday = 5) 
  (h_friday : candies_friday = 2) 
  (h_left : candies_left = 4) : candies_tuesday + candies_thursday + candies_friday - candies_left = 6 := by
  sorry

end NUMINAMATH_GPT_susan_ate_candies_l1686_168629


namespace NUMINAMATH_GPT_negate_proposition_l1686_168652

theorem negate_proposition : (∀ x : ℝ, x^3 - x^2 + 1 ≤ 1) ↔ ¬ (∃ x : ℝ, x^3 - x^2 + 1 > 1) :=
by
  sorry

end NUMINAMATH_GPT_negate_proposition_l1686_168652


namespace NUMINAMATH_GPT_complete_the_square_l1686_168699

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6 * x - 10 = 0

-- State the proof problem
theorem complete_the_square (x : ℝ) (h : quadratic_eq x) : (x - 3)^2 = 19 :=
by 
  -- Skip the proof using sorry
  sorry

end NUMINAMATH_GPT_complete_the_square_l1686_168699


namespace NUMINAMATH_GPT_max_min_distance_inequality_l1686_168638

theorem max_min_distance_inequality (n : ℕ) (D d : ℝ) (h1 : d > 0) 
    (exists_points : ∃ (points : Fin n → ℝ × ℝ), 
      (∀ i j : Fin n, i ≠ j → dist (points i) (points j) ≥ d) 
      ∧ (∀ i j : Fin n, dist (points i) (points j) ≤ D)) : 
    D / d > (Real.sqrt (n * Real.pi)) / 2 - 1 := 
  sorry

end NUMINAMATH_GPT_max_min_distance_inequality_l1686_168638


namespace NUMINAMATH_GPT_cos_210_eq_neg_sqrt3_over_2_l1686_168687

theorem cos_210_eq_neg_sqrt3_over_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_cos_210_eq_neg_sqrt3_over_2_l1686_168687


namespace NUMINAMATH_GPT_linear_function_in_quadrants_l1686_168614

section LinearFunctionQuadrants

variable (m : ℝ)

def passesThroughQuadrants (m : ℝ) : Prop :=
  (m + 1 > 0) ∧ (m - 1 > 0)

theorem linear_function_in_quadrants (h : passesThroughQuadrants m) : m > 1 :=
by sorry

end LinearFunctionQuadrants

end NUMINAMATH_GPT_linear_function_in_quadrants_l1686_168614


namespace NUMINAMATH_GPT_first_player_wins_l1686_168624

-- Define the polynomial with placeholders
def P (X : ℤ) (a3 a2 a1 a0 : ℤ) : ℤ :=
  X^4 + a3 * X^3 + a2 * X^2 + a1 * X + a0

-- The statement that the first player can always win
theorem first_player_wins :
  ∀ (a3 a2 a1 a0 : ℤ),
    (a0 ≠ 0) → (a1 ≠ 0) → (a2 ≠ 0) → (a3 ≠ 0) →
    ∃ (strategy : ℕ → ℤ),
      (∀ n, strategy n ≠ 0) ∧
      ¬ ∃ (x y : ℤ), x ≠ y ∧ P x (strategy 0) (strategy 1) (strategy 2) (strategy 3) = 0 ∧ P y (strategy 0) (strategy 1) (strategy 2) (strategy 3) = 0 :=
by
  sorry

end NUMINAMATH_GPT_first_player_wins_l1686_168624


namespace NUMINAMATH_GPT_jacket_price_equation_l1686_168658

theorem jacket_price_equation (x : ℝ) (h : 0.8 * (1 + 0.5) * x - x = 28) : 0.8 * (1 + 0.5) * x = x + 28 :=
by sorry

end NUMINAMATH_GPT_jacket_price_equation_l1686_168658


namespace NUMINAMATH_GPT_decimal_equiv_half_squared_l1686_168666

theorem decimal_equiv_half_squared :
  ((1 / 2 : ℝ) ^ 2) = 0.25 := by
  sorry

end NUMINAMATH_GPT_decimal_equiv_half_squared_l1686_168666


namespace NUMINAMATH_GPT_bottom_price_l1686_168683

open Nat

theorem bottom_price (B T : ℕ) (h1 : T = B + 300) (h2 : 3 * B + 3 * T = 21000) : B = 3350 := by
  sorry

end NUMINAMATH_GPT_bottom_price_l1686_168683


namespace NUMINAMATH_GPT_problem_l1686_168649

variables (y S : ℝ)

theorem problem (h : 5 * (2 * y + 3 * Real.sqrt 3) = S) : 10 * (4 * y + 6 * Real.sqrt 3) = 4 * S :=
sorry

end NUMINAMATH_GPT_problem_l1686_168649


namespace NUMINAMATH_GPT_smallest_possible_perimeter_of_scalene_triangle_with_prime_sides_l1686_168694

/-- Define what it means for a number to be a prime greater than 3 -/
def is_prime_gt_3 (n : ℕ) : Prop :=
  Prime n ∧ 3 < n

/-- Define a scalene triangle with side lengths that are distinct primes greater than 3 -/
def is_scalene_triangle_with_distinct_primes (a b c : ℕ) : Prop :=
  is_prime_gt_3 a ∧ is_prime_gt_3 b ∧ is_prime_gt_3 c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b > c ∧ b + c > a ∧ a + c > b

/-- The proof problem statement -/
theorem smallest_possible_perimeter_of_scalene_triangle_with_prime_sides :
  ∃ (a b c : ℕ), is_scalene_triangle_with_distinct_primes a b c ∧ Prime (a + b + c) ∧ (a + b + c = 23) :=
sorry

end NUMINAMATH_GPT_smallest_possible_perimeter_of_scalene_triangle_with_prime_sides_l1686_168694


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1686_168611

theorem repeating_decimal_to_fraction : (0.2727272727 : ℝ) = 3 / 11 := 
sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1686_168611
