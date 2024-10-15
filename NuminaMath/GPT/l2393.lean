import Mathlib

namespace NUMINAMATH_GPT_value_of_x_l2393_239303

theorem value_of_x (x y z : ℝ) (h1 : x = (1 / 2) * y) (h2 : y = (1 / 4) * z) (h3 : z = 80) : x = 10 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l2393_239303


namespace NUMINAMATH_GPT_mixed_water_temp_l2393_239380

def cold_water_temp : ℝ := 20   -- Temperature of cold water
def hot_water_temp : ℝ := 40    -- Temperature of hot water

theorem mixed_water_temp :
  (cold_water_temp + hot_water_temp) / 2 = 30 := 
by sorry

end NUMINAMATH_GPT_mixed_water_temp_l2393_239380


namespace NUMINAMATH_GPT_tens_digit_6_pow_45_l2393_239374

theorem tens_digit_6_pow_45 : (6 ^ 45 % 100) / 10 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_tens_digit_6_pow_45_l2393_239374


namespace NUMINAMATH_GPT_work_rate_combined_l2393_239317

theorem work_rate_combined (a b c : ℝ) (ha : a = 21) (hb : b = 6) (hc : c = 12) :
  (1 / ((1 / a) + (1 / b) + (1 / c))) = 84 / 25 := by
  sorry

end NUMINAMATH_GPT_work_rate_combined_l2393_239317


namespace NUMINAMATH_GPT_ratio_of_ages_l2393_239370

theorem ratio_of_ages (S M : ℕ) (h1 : M = S + 24) (h2 : M + 2 = (S + 2) * 2) (h3 : S = 22) : (M + 2) / (S + 2) = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_ages_l2393_239370


namespace NUMINAMATH_GPT_sequence_value_at_5_l2393_239392

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 / 3 ∧ ∀ n, 1 < n → a n = (-1) ^ n * 2 * a (n - 1)

theorem sequence_value_at_5 (a : ℕ → ℚ) (h : seq a) : a 5 = 16 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_sequence_value_at_5_l2393_239392


namespace NUMINAMATH_GPT_simplify_expression_l2393_239327

theorem simplify_expression : 1 - (1 / (2 + Real.sqrt 5)) + (1 / (2 - Real.sqrt 5)) = 1 - 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2393_239327


namespace NUMINAMATH_GPT_min_value_of_a3b2c_l2393_239360

theorem min_value_of_a3b2c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 / a + 1 / b + 1 / c = 9) : 
  a^3 * b^2 * c ≥ 1 / 2916 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_of_a3b2c_l2393_239360


namespace NUMINAMATH_GPT_number_of_candies_in_a_packet_l2393_239390

theorem number_of_candies_in_a_packet 
  (two_packets : ℕ)
  (candies_per_day_weekday : ℕ)
  (candies_per_day_weekend : ℕ)
  (weeks : ℕ)
  (total_candies : ℕ)
  (packet_size : ℕ)
  (H1 : two_packets = 2)
  (H2 : candies_per_day_weekday = 2)
  (H3 : candies_per_day_weekend = 1)
  (H4 : weeks = 3)
  (H5 : packet_size > 0)
  (H6 : total_candies = packets * packet_size)
  (H7 : total_candies = 3 * (5 * candies_per_day_weekday + 2 * candies_per_day_weekend))
  : packet_size = 18 :=
by
  sorry

end NUMINAMATH_GPT_number_of_candies_in_a_packet_l2393_239390


namespace NUMINAMATH_GPT_distance_foci_l2393_239341

noncomputable def distance_between_foci := 
  let F1 := (4, 5)
  let F2 := (-6, 9)
  Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) 

theorem distance_foci : 
  ∃ (F1 F2 : ℝ × ℝ), 
    F1 = (4, 5) ∧ 
    F2 = (-6, 9) ∧ 
    distance_between_foci = 2 * Real.sqrt 29 := by {
  sorry
}

end NUMINAMATH_GPT_distance_foci_l2393_239341


namespace NUMINAMATH_GPT_proof_equiv_expression_l2393_239339

variable (x y : ℝ)

def P : ℝ := x^2 + y^2
def Q : ℝ := x^2 - y^2

theorem proof_equiv_expression :
  ( (P x y)^2 + (Q x y)^2 ) / ( (P x y)^2 - (Q x y)^2 ) - 
  ( (P x y)^2 - (Q x y)^2 ) / ( (P x y)^2 + (Q x y)^2 ) = 
  (x^4 - y^4) / (x^2 * y^2) :=
by
  sorry

end NUMINAMATH_GPT_proof_equiv_expression_l2393_239339


namespace NUMINAMATH_GPT_value_of_expression_l2393_239394

theorem value_of_expression (x y : ℝ) (h1 : x = -2) (h2 : y = 1/2) :
  y * (x + y) + (x + y) * (x - y) - x^2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2393_239394


namespace NUMINAMATH_GPT_probability_of_winning_l2393_239355

def roll_is_seven (d1 d2 : ℕ) : Prop :=
  d1 + d2 = 7

theorem probability_of_winning (d1 d2 : ℕ) (h : roll_is_seven d1 d2) :
  (1/6 : ℚ) = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_winning_l2393_239355


namespace NUMINAMATH_GPT_neg_exists_lt_1000_l2393_239300

open Nat

theorem neg_exists_lt_1000 : (¬ ∃ n : ℕ, 2^n < 1000) = ∀ n : ℕ, 2^n ≥ 1000 := by
  sorry

end NUMINAMATH_GPT_neg_exists_lt_1000_l2393_239300


namespace NUMINAMATH_GPT_solve_for_p_l2393_239338

theorem solve_for_p (q p : ℝ) (h : p^2 * q = p * q + p^2) : 
  p = 0 ∨ p = q / (q - 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_p_l2393_239338


namespace NUMINAMATH_GPT_fill_time_l2393_239381

noncomputable def time_to_fill (X Y Z : ℝ) : ℝ :=
  1 / X + 1 / Y + 1 / Z

theorem fill_time 
  (V X Y Z : ℝ) 
  (h1 : X + Y = V / 3) 
  (h2 : X + Z = V / 2) 
  (h3 : Y + Z = V / 4) :
  1 / time_to_fill X Y Z = 24 / 13 :=
by
  sorry

end NUMINAMATH_GPT_fill_time_l2393_239381


namespace NUMINAMATH_GPT_plane_eq_l2393_239304

def gcd4 (a b c d : ℤ) : ℤ := Int.gcd (Int.gcd (Int.gcd (abs a) (abs b)) (abs c)) (abs d)

theorem plane_eq (A B C D : ℤ) (A_pos : A > 0) 
  (gcd_1 : gcd4 A B C D = 1) 
  (H_parallel : (A, B, C) = (3, 2, -4)) 
  (H_point : A * 2 + B * 3 + C * (-1) + D = 0) : 
  A = 3 ∧ B = 2 ∧ C = -4 ∧ D = -16 := 
sorry

end NUMINAMATH_GPT_plane_eq_l2393_239304


namespace NUMINAMATH_GPT_solve_n_m_l2393_239389

noncomputable def exponents_of_linear_equation (n m : ℕ) (x y : ℝ) : Prop :=
2 * x ^ (n - 3) - (1 / 3) * y ^ (2 * m + 1) = 0

theorem solve_n_m (n m : ℕ) (x y : ℝ) (h_linear : exponents_of_linear_equation n m x y) :
  n ^ m = 1 :=
sorry

end NUMINAMATH_GPT_solve_n_m_l2393_239389


namespace NUMINAMATH_GPT_not_valid_base_five_l2393_239391

theorem not_valid_base_five (k : ℕ) (h₁ : k = 5) : ¬(∀ d ∈ [3, 2, 5, 0, 1], d < k) :=
by
  sorry

end NUMINAMATH_GPT_not_valid_base_five_l2393_239391


namespace NUMINAMATH_GPT_min_value_of_expression_l2393_239393

theorem min_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x : ℝ, x = a^2 + b^2 + (1 / (a + b)^2) + (1 / (a * b)) ∧ x = Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l2393_239393


namespace NUMINAMATH_GPT_solution_values_sum_l2393_239366

theorem solution_values_sum (x y : ℝ) (p q r s : ℕ) 
  (hx : x + y = 5) 
  (hxy : 2 * x * y = 5) 
  (hx_form : x = (p + q * Real.sqrt r) / s ∨ x = (p - q * Real.sqrt r) / s) 
  (hpqs_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0) : 
  p + q + r + s = 23 := 
sorry

end NUMINAMATH_GPT_solution_values_sum_l2393_239366


namespace NUMINAMATH_GPT_quadratic_vertex_problem_l2393_239399

/-- 
    Given a quadratic equation y = ax^2 + bx + c, where (2, -3) 
    is the vertex of the parabola and it passes through (0, 1), 
    prove that a - b + c = 6. 
-/
theorem quadratic_vertex_problem 
    (a b c : ℤ)
    (h : ∀ x : ℝ, y = a * (x - 2)^2 - 3)
    (h_point : y = 1)
    (h_passes_through_origin : y = a * (0 - 2)^2 - 3) :
    a - b + c = 6 :=
sorry

end NUMINAMATH_GPT_quadratic_vertex_problem_l2393_239399


namespace NUMINAMATH_GPT_original_cube_volume_l2393_239383

theorem original_cube_volume 
  (a : ℕ) 
  (h : 3 * a * (a - a / 2) * a - a^3 = 2 * a^2) : 
  a = 4 → a^3 = 64 := 
by
  sorry

end NUMINAMATH_GPT_original_cube_volume_l2393_239383


namespace NUMINAMATH_GPT_tan_alpha_l2393_239314

variable (α : Real)
-- Condition 1: α is an angle in the second quadrant
-- This implies that π/2 < α < π and sin α = 4 / 5
variable (h1 : π / 2 < α ∧ α < π) 
variable (h2 : Real.sin α = 4 / 5)

theorem tan_alpha : Real.tan α = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_l2393_239314


namespace NUMINAMATH_GPT_total_bill_is_270_l2393_239359

-- Conditions as Lean definitions
def totalBill (T : ℝ) : Prop :=
  let eachShare := T / 10
  9 * (eachShare + 3) = T

-- Theorem stating that the total bill T is 270
theorem total_bill_is_270 (T : ℝ) (h : totalBill T) : T = 270 :=
sorry

end NUMINAMATH_GPT_total_bill_is_270_l2393_239359


namespace NUMINAMATH_GPT_min_value_x_plus_y_l2393_239358

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 / y + 1 / x = 4) :
  x + y ≥ 9 / 4 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l2393_239358


namespace NUMINAMATH_GPT_total_cars_all_own_l2393_239336

theorem total_cars_all_own :
  ∀ (C L S K : ℕ), 
  (C = 5) →
  (L = C + 4) →
  (K = 2 * C) →
  (S = K - 2) →
  (C + L + K + S = 32) :=
by
  intros C L S K
  intro hC
  intro hL
  intro hK
  intro hS
  sorry

end NUMINAMATH_GPT_total_cars_all_own_l2393_239336


namespace NUMINAMATH_GPT_eyes_saw_plane_l2393_239375

theorem eyes_saw_plane (total_students : ℕ) (fraction_looked_up : ℚ) (students_with_eyepatches : ℕ) :
  total_students = 200 → fraction_looked_up = 3/4 → students_with_eyepatches = 20 →
  ∃ eyes_saw_plane, eyes_saw_plane = 280 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_eyes_saw_plane_l2393_239375


namespace NUMINAMATH_GPT_number_of_books_to_break_even_is_4074_l2393_239324

-- Definitions from problem conditions
def fixed_costs : ℝ := 35630
def variable_cost_per_book : ℝ := 11.50
def selling_price_per_book : ℝ := 20.25

-- The target number of books to sell for break-even
def target_books_to_break_even : ℕ := 4074

-- Lean statement to prove that number of books to break even is 4074
theorem number_of_books_to_break_even_is_4074 :
  let total_costs (x : ℝ) := fixed_costs + variable_cost_per_book * x
  let total_revenue (x : ℝ) := selling_price_per_book * x
  ∃ x : ℝ, total_costs x = total_revenue x → x = target_books_to_break_even := by
  sorry

end NUMINAMATH_GPT_number_of_books_to_break_even_is_4074_l2393_239324


namespace NUMINAMATH_GPT_prove_cuboid_properties_l2393_239356

noncomputable def cuboid_length := 5
noncomputable def cuboid_width := 4
noncomputable def cuboid_height := 3

theorem prove_cuboid_properties :
  (min (cuboid_length * cuboid_width) (min (cuboid_length * cuboid_height) (cuboid_width * cuboid_height)) = 12) ∧
  (max (cuboid_length * cuboid_width) (max (cuboid_length * cuboid_height) (cuboid_width * cuboid_height)) = 20) ∧
  ((cuboid_length + cuboid_width + cuboid_height) * 4 = 48) ∧
  (2 * (cuboid_length * cuboid_width + cuboid_length * cuboid_height + cuboid_width * cuboid_height) = 94) ∧
  (cuboid_length * cuboid_width * cuboid_height = 60) :=
by
  sorry

end NUMINAMATH_GPT_prove_cuboid_properties_l2393_239356


namespace NUMINAMATH_GPT_original_average_is_6_2_l2393_239397

theorem original_average_is_6_2 (n : ℕ) (S : ℚ) (h1 : 6.2 = S / n) (h2 : 6.6 = (S + 4) / n) :
  6.2 = S / n :=
by
  sorry

end NUMINAMATH_GPT_original_average_is_6_2_l2393_239397


namespace NUMINAMATH_GPT_find_a_l2393_239321

def set_A : Set ℝ := { x | abs (x - 1) > 2 }
def set_B (a : ℝ) : Set ℝ := { x | x^2 - (a + 1) * x + a < 0 }
def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem find_a (a : ℝ) : (intersection set_A (set_B a)) = { x | 3 < x ∧ x < 5 } → a = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2393_239321


namespace NUMINAMATH_GPT_sum_is_18_less_than_abs_sum_l2393_239309

theorem sum_is_18_less_than_abs_sum : 
  (-5 + -4) = (|-5| + |-4| - 18) :=
by
  sorry

end NUMINAMATH_GPT_sum_is_18_less_than_abs_sum_l2393_239309


namespace NUMINAMATH_GPT_tshirt_cost_l2393_239323

-- Definitions based on conditions
def pants_cost : ℝ := 80
def shoes_cost : ℝ := 150
def discount : ℝ := 0.1
def total_paid : ℝ := 558

-- Variables based on the problem
variable (T : ℝ) -- Cost of one T-shirt
def num_tshirts : ℝ := 4
def num_pants : ℝ := 3
def num_shoes : ℝ := 2

-- Theorem: The cost of one T-shirt is $20
theorem tshirt_cost : T = 20 :=
by
  have total_cost : ℝ := (num_tshirts * T) + (num_pants * pants_cost) + (num_shoes * shoes_cost)
  have discounted_total : ℝ := (1 - discount) * total_cost
  have payment_condition : discounted_total = total_paid := sorry
  sorry -- detailed proof

end NUMINAMATH_GPT_tshirt_cost_l2393_239323


namespace NUMINAMATH_GPT_nested_expression_rational_count_l2393_239373

theorem nested_expression_rational_count : 
  let count := Nat.card {n : ℕ // 1 ≤ n ∧ n ≤ 2021 ∧ ∃ m : ℕ, m % 2 = 1 ∧ m * m = 1 + 4 * n}
  count = 44 := 
by sorry

end NUMINAMATH_GPT_nested_expression_rational_count_l2393_239373


namespace NUMINAMATH_GPT_simplify_complex_expression_l2393_239365

theorem simplify_complex_expression :
  (1 / (-8 ^ 2) ^ 4 * (-8) ^ 9) = -8 := by
  sorry

end NUMINAMATH_GPT_simplify_complex_expression_l2393_239365


namespace NUMINAMATH_GPT_inequality_with_sum_one_l2393_239329

theorem inequality_with_sum_one
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1)
  (x y : ℝ) :
  (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by
  sorry

end NUMINAMATH_GPT_inequality_with_sum_one_l2393_239329


namespace NUMINAMATH_GPT_cost_ratio_two_pastries_pies_l2393_239357

theorem cost_ratio_two_pastries_pies (s p : ℝ) (h1 : 2 * s = 3 * (2 * p)) :
  (s + p) / (2 * p) = 2 :=
by
  sorry

end NUMINAMATH_GPT_cost_ratio_two_pastries_pies_l2393_239357


namespace NUMINAMATH_GPT_find_n_tangent_l2393_239352

theorem find_n_tangent (n : ℤ) (h1 : -180 < n) (h2 : n < 180) (h3 : ∃ k : ℤ, 210 = n + 180 * k) : n = 30 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_find_n_tangent_l2393_239352


namespace NUMINAMATH_GPT_speed_of_boat_in_still_water_12_l2393_239396

theorem speed_of_boat_in_still_water_12 (d b c : ℝ) (h1 : d = (b - c) * 5) (h2 : d = (b + c) * 3) (hb : b = 12) : b = 12 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_boat_in_still_water_12_l2393_239396


namespace NUMINAMATH_GPT_find_c_plus_inv_b_l2393_239318

variable (a b c : ℝ)

def conditions := 
  (a * b * c = 1) ∧ 
  (a + 1/c = 7) ∧ 
  (b + 1/a = 16)

theorem find_c_plus_inv_b (h : conditions a b c) : 
  c + 1/b = 25 / 111 :=
sorry

end NUMINAMATH_GPT_find_c_plus_inv_b_l2393_239318


namespace NUMINAMATH_GPT_discount_percent_l2393_239335

theorem discount_percent (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : (SP - CP) / CP * 100 = 34.375) :
  ((MP - SP) / MP * 100) = 14 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_discount_percent_l2393_239335


namespace NUMINAMATH_GPT_allison_total_video_hours_l2393_239330

def total_video_hours_uploaded (total_days: ℕ) (half_days: ℕ) (first_half_rate: ℕ) (second_half_rate: ℕ): ℕ :=
  first_half_rate * half_days + second_half_rate * (total_days - half_days)

theorem allison_total_video_hours :
  total_video_hours_uploaded 30 15 10 20 = 450 :=
by
  sorry

end NUMINAMATH_GPT_allison_total_video_hours_l2393_239330


namespace NUMINAMATH_GPT_general_term_formula_l2393_239302

noncomputable def S (n : ℕ) : ℕ := 2^n - 1
noncomputable def a (n : ℕ) : ℕ := 2^(n-1)

theorem general_term_formula (n : ℕ) (hn : n > 0) : 
    a n = S n - S (n - 1) := 
by 
  sorry

end NUMINAMATH_GPT_general_term_formula_l2393_239302


namespace NUMINAMATH_GPT_additional_weekly_rate_l2393_239334

theorem additional_weekly_rate (rate_first_week : ℝ) (total_days_cost : ℝ) (days_first_week : ℕ) (total_days : ℕ) (cost_total : ℝ) (cost_first_week : ℝ) (days_after_first_week : ℕ) : 
  (rate_first_week * days_first_week = cost_first_week) → 
  (total_days = days_first_week + days_after_first_week) → 
  (cost_total = cost_first_week + (days_after_first_week * (rate_first_week * 7 / days_first_week))) →
  (rate_first_week = 18) →
  (cost_total = 350) →
  total_days = 23 → 
  (days_first_week = 7) → 
  cost_first_week = 126 →
  (days_after_first_week = 16) →
  rate_first_week * 7 / days_first_week * days_after_first_week = 14 := 
by 
  sorry

end NUMINAMATH_GPT_additional_weekly_rate_l2393_239334


namespace NUMINAMATH_GPT_find_a9_l2393_239384

noncomputable def polynomial_coefficients : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ),
  ∀ (x : ℤ),
    x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 =
    a₀ + a₁ * (1 + x) + a₂ * (1 + x)^2 + a₃ * (1 + x)^3 + a₄ * (1 + x)^4 + 
    a₅ * (1 + x)^5 + a₆ * (1 + x)^6 + a₇ * (1 + x)^7 + a₈ * (1 + x)^8 + 
    a₉ * (1 + x)^9 + a₁₀ * (1 + x)^10

theorem find_a9 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ) (h : polynomial_coefficients) : a₉ = -9 := by
  sorry

end NUMINAMATH_GPT_find_a9_l2393_239384


namespace NUMINAMATH_GPT_geometric_seq_condition_l2393_239395

/-- In a geometric sequence with common ratio q, sum of the first n terms S_n.
  Given q > 0, show that it is a necessary condition for {S_n} to be an increasing sequence,
  but not a sufficient condition. -/
theorem geometric_seq_condition (a1 q : ℝ) (S : ℕ → ℝ)
  (hS : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h1 : q > 0) : 
  (∀ n, S n < S (n + 1)) ↔ a1 > 0 :=
sorry

end NUMINAMATH_GPT_geometric_seq_condition_l2393_239395


namespace NUMINAMATH_GPT_evaluate_expression_l2393_239378

theorem evaluate_expression :
  (4^2 - 4) + (5^2 - 5) - (7^3 - 7) + (3^2 - 3) = -298 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2393_239378


namespace NUMINAMATH_GPT_expand_binomials_l2393_239346

variable {x y : ℝ}

theorem expand_binomials (x y : ℝ) : 
  (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := 
by
  sorry

end NUMINAMATH_GPT_expand_binomials_l2393_239346


namespace NUMINAMATH_GPT_line_through_A_parallel_y_axis_l2393_239320

theorem line_through_A_parallel_y_axis (x y: ℝ) (A: ℝ × ℝ) (h1: A = (-3, 1)) : 
  (∀ P: ℝ × ℝ, P ∈ {p : ℝ × ℝ | p.1 = -3} → (P = A ∨ P.1 = -3)) :=
by
  sorry

end NUMINAMATH_GPT_line_through_A_parallel_y_axis_l2393_239320


namespace NUMINAMATH_GPT_point_on_x_axis_l2393_239340

theorem point_on_x_axis (a : ℝ) (h : a + 2 = 0) : (a - 1, a + 2) = (-3, 0) :=
by
  sorry

end NUMINAMATH_GPT_point_on_x_axis_l2393_239340


namespace NUMINAMATH_GPT_eccentricity_hyperbola_l2393_239322

-- Conditions
def is_eccentricity_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  let e := (Real.sqrt 2) / 2
  (Real.sqrt (1 - b^2 / a^2) = e)

-- Objective: Find the eccentricity of the given the hyperbola.
theorem eccentricity_hyperbola (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : is_eccentricity_ellipse a b h1 h2) : 
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_eccentricity_hyperbola_l2393_239322


namespace NUMINAMATH_GPT_sin_add_arctan_arcsin_l2393_239379

theorem sin_add_arctan_arcsin :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan 3
  (Real.sin a = 4 / 5) →
  (Real.tan b = 3) →
  Real.sin (a + b) = (13 * Real.sqrt 10) / 50 :=
by
  intros _ _
  sorry

end NUMINAMATH_GPT_sin_add_arctan_arcsin_l2393_239379


namespace NUMINAMATH_GPT_area_EYH_trapezoid_l2393_239344

theorem area_EYH_trapezoid (EF GH : ℕ) (EF_len : EF = 15) (GH_len : GH = 35) 
(Area_trapezoid : (EF + GH) * 16 / 2 = 400) : 
∃ (EYH_area : ℕ), EYH_area = 84 := by
  sorry

end NUMINAMATH_GPT_area_EYH_trapezoid_l2393_239344


namespace NUMINAMATH_GPT_f_2_plus_f_5_eq_2_l2393_239368

noncomputable def f : ℝ → ℝ := sorry

open Real

-- Conditions: f(3^x) = x * log 9
axiom f_cond (x : ℝ) : f (3^x) = x * log 9

-- Question: f(2) + f(5) = 2
theorem f_2_plus_f_5_eq_2 : f 2 + f 5 = 2 := sorry

end NUMINAMATH_GPT_f_2_plus_f_5_eq_2_l2393_239368


namespace NUMINAMATH_GPT_smallest_digit_d_l2393_239367

theorem smallest_digit_d (d : ℕ) (hd : d < 10) :
  (∃ d, (20 - (8 + d)) % 11 = 0 ∧ d < 10) → d = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_digit_d_l2393_239367


namespace NUMINAMATH_GPT_B_pow_2017_eq_B_l2393_239362

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![ ![0, 1, 0], ![0, 0, 1], ![1, 0, 0] ]

theorem B_pow_2017_eq_B : B^2017 = B := by
  sorry

end NUMINAMATH_GPT_B_pow_2017_eq_B_l2393_239362


namespace NUMINAMATH_GPT_central_angle_of_regular_polygon_l2393_239372

theorem central_angle_of_regular_polygon (n : ℕ) (h : 360 ∣ 360 - 36 * n) :
  n = 10 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_regular_polygon_l2393_239372


namespace NUMINAMATH_GPT_part_a_part_b_l2393_239306

-- Define the parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Prove that (1, 1) lies on the parabola
theorem part_a : parabola 1 = 1 := by
  sorry

-- Prove that for any t, (t, t^2) lies on the parabola
theorem part_b (t : ℝ) : parabola t = t^2 := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l2393_239306


namespace NUMINAMATH_GPT_find_width_of_rectangle_l2393_239347

variable (w : ℝ) (l : ℝ) (P : ℝ)

def width_correct (h1 : P = 150) (h2 : l = w + 15) : Prop :=
  w = 30

-- Theorem statement in Lean
theorem find_width_of_rectangle (h1 : P = 150) (h2 : l = w + 15) (h3 : P = 2 * l + 2 * w) : width_correct w l P h1 h2 :=
by
  sorry

end NUMINAMATH_GPT_find_width_of_rectangle_l2393_239347


namespace NUMINAMATH_GPT_find_s_for_g_neg1_zero_l2393_239305

def g (x s : ℝ) : ℝ := 3 * x^4 + x^3 - 2 * x^2 - 4 * x + s

theorem find_s_for_g_neg1_zero (s : ℝ) : g (-1) s = 0 ↔ s = -4 := by
  sorry

end NUMINAMATH_GPT_find_s_for_g_neg1_zero_l2393_239305


namespace NUMINAMATH_GPT_ice_cubes_per_tray_l2393_239388

theorem ice_cubes_per_tray (total_ice_cubes : ℕ) (number_of_trays : ℕ) (h1 : total_ice_cubes = 72) (h2 : number_of_trays = 8) : 
  total_ice_cubes / number_of_trays = 9 :=
by
  sorry

end NUMINAMATH_GPT_ice_cubes_per_tray_l2393_239388


namespace NUMINAMATH_GPT_number_of_three_digit_prime_integers_l2393_239348

def prime_digits : Set Nat := {2, 3, 5, 7}

theorem number_of_three_digit_prime_integers : 
  (∃ count, count = 4 * 4 * 4 ∧ count = 64) :=
by
  sorry

end NUMINAMATH_GPT_number_of_three_digit_prime_integers_l2393_239348


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l2393_239343

theorem sum_arithmetic_sequence (S : ℕ → ℕ) :
  S 7 = 21 ∧ S 17 = 34 → S 27 = 27 :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_l2393_239343


namespace NUMINAMATH_GPT_no_real_solutions_l2393_239369

noncomputable def equation (x : ℝ) := x + 48 / (x - 3) + 1

theorem no_real_solutions : ∀ x : ℝ, equation x ≠ 0 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_real_solutions_l2393_239369


namespace NUMINAMATH_GPT_sequence_property_l2393_239312

theorem sequence_property (a : ℕ → ℕ) (h1 : ∀ n, n ≥ 1 → a n ∈ { x | x ≥ 1 }) 
  (h2 : ∀ n, n ≥ 1 → a (a n) + a n = 2 * n) : ∀ n, n ≥ 1 → a n = n :=
by
  sorry

end NUMINAMATH_GPT_sequence_property_l2393_239312


namespace NUMINAMATH_GPT_inequality_range_of_a_l2393_239382

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |2 * x - a| > x - 1) ↔ a < 3 ∨ a > 5 :=
by
  sorry

end NUMINAMATH_GPT_inequality_range_of_a_l2393_239382


namespace NUMINAMATH_GPT_percent_carnations_l2393_239342

theorem percent_carnations (F : ℕ) (H1 : 3 / 5 * F = pink) (H2 : 1 / 5 * F = white) 
(H3 : F - pink - white = red) (H4 : 1 / 2 * pink = pink_roses)
(H5 : pink - pink_roses = pink_carnations) (H6 : 1 / 2 * red = red_carnations)
(H7 : white = white_carnations) : 
100 * (pink_carnations + red_carnations + white_carnations) / F = 60 :=
sorry

end NUMINAMATH_GPT_percent_carnations_l2393_239342


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2393_239364

theorem arithmetic_sequence_sum (a1 d : ℝ)
  (h1 : a1 + 11 * d = -8)
  (h2 : 9 / 2 * (a1 + (a1 + 8 * d)) = -9) :
  16 / 2 * (a1 + (a1 + 15 * d)) = -72 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2393_239364


namespace NUMINAMATH_GPT_solution_count_l2393_239313

/-- There are 91 solutions to the equation x + y + z = 15 given that x, y, z are all positive integers. -/
theorem solution_count (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 15) : 
  ∃! n, n = 91 := 
by sorry

end NUMINAMATH_GPT_solution_count_l2393_239313


namespace NUMINAMATH_GPT_largest_negative_integer_solution_l2393_239376

theorem largest_negative_integer_solution :
  ∃ x : ℤ, x < 0 ∧ 50 * x + 14 % 24 = 10 % 24 ∧ ∀ y : ℤ, (y < 0 ∧ y % 12 = 10 % 12 → y ≤ x) :=
by
  sorry

end NUMINAMATH_GPT_largest_negative_integer_solution_l2393_239376


namespace NUMINAMATH_GPT_union_eq_l2393_239345

def A : Set ℤ := {-1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

theorem union_eq : A ∪ B = {-1, 0, 1, 2, 3} := 
by 
  sorry

end NUMINAMATH_GPT_union_eq_l2393_239345


namespace NUMINAMATH_GPT_find_roses_last_year_l2393_239333

-- Definitions based on conditions
def roses_last_year : ℕ := sorry
def roses_this_year := roses_last_year / 2
def roses_needed := 2 * roses_last_year
def rose_cost := 3 -- cost per rose in dollars
def total_spent := 54 -- total spent in dollars

-- Formulate the problem
theorem find_roses_last_year (h : 2 * roses_last_year - roses_this_year = 18)
  (cost_eq : total_spent / rose_cost = 18) :
  roses_last_year = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_roses_last_year_l2393_239333


namespace NUMINAMATH_GPT_R_transformed_is_R_l2393_239386

-- Define the initial coordinates of the rectangle PQRS
def P : ℝ × ℝ := (3, 4)
def Q : ℝ × ℝ := (6, 4)
def R : ℝ × ℝ := (6, 1)
def S : ℝ × ℝ := (3, 1)

-- Define the reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the translation down by 2 units
def translate_down_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 - 2)

-- Define the reflection across the line y = -x
def reflect_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

-- Define the translation up by 2 units
def translate_up_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 + 2)

-- Define the transformation to find R''
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_up_2 (reflect_y_neg_x (translate_down_2 (reflect_x p)))

-- Prove that the result of transforming R is (-3, -4)
theorem R_transformed_is_R'' : transform R = (-3, -4) :=
  by sorry

end NUMINAMATH_GPT_R_transformed_is_R_l2393_239386


namespace NUMINAMATH_GPT_smallest_b_for_factoring_l2393_239328

theorem smallest_b_for_factoring (b : ℕ) (p q : ℕ) (h1 : p * q = 1800) (h2 : p + q = b) : b = 85 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_for_factoring_l2393_239328


namespace NUMINAMATH_GPT_neg_p_sufficient_not_necessary_for_neg_q_l2393_239387

noncomputable def p (x : ℝ) : Prop := abs (x + 1) > 0
noncomputable def q (x : ℝ) : Prop := 5 * x - 6 > x^2

theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x, ¬ p x → ¬ q x) ∧ ¬ (∀ x, ¬ q x → ¬ p x) :=
by
  sorry

end NUMINAMATH_GPT_neg_p_sufficient_not_necessary_for_neg_q_l2393_239387


namespace NUMINAMATH_GPT_arithmetic_sequence_y_value_l2393_239331

theorem arithmetic_sequence_y_value (y : ℝ) (h₁ : 2 * y - 3 = -5 * y + 11) : y = 2 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_y_value_l2393_239331


namespace NUMINAMATH_GPT_range_of_independent_variable_l2393_239398

theorem range_of_independent_variable (x : ℝ) : 
  (y = 3 / (x + 2)) → (x ≠ -2) :=
by
  -- suppose the function y = 3 / (x + 2) is given
  -- we need to prove x ≠ -2 for the function to be defined
  sorry

end NUMINAMATH_GPT_range_of_independent_variable_l2393_239398


namespace NUMINAMATH_GPT_sin_alpha_eq_sin_beta_l2393_239325

theorem sin_alpha_eq_sin_beta (α β : Real) (k : Int) 
  (h_symmetry : α + β = 2 * k * Real.pi + Real.pi) : 
  Real.sin α = Real.sin β := 
by 
  sorry

end NUMINAMATH_GPT_sin_alpha_eq_sin_beta_l2393_239325


namespace NUMINAMATH_GPT_group_division_ways_l2393_239353

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem group_division_ways : 
  choose 30 10 * choose 20 10 * choose 10 10 = Nat.factorial 30 / (Nat.factorial 10 * Nat.factorial 10 * Nat.factorial 10) := 
by
  sorry

end NUMINAMATH_GPT_group_division_ways_l2393_239353


namespace NUMINAMATH_GPT_parabola_symmetric_y_axis_intersection_l2393_239326

theorem parabola_symmetric_y_axis_intersection :
  ∀ (x y : ℝ),
  (x = y ∨ x*x + y*y - 6*y = 0) ∧ (x*x = 3 * y) :=
by 
  sorry

end NUMINAMATH_GPT_parabola_symmetric_y_axis_intersection_l2393_239326


namespace NUMINAMATH_GPT_exponent_multiplication_l2393_239316

variable (a : ℝ) (m : ℤ)

theorem exponent_multiplication (a : ℝ) (m : ℤ) : a^(2 * m + 2) = a^(2 * m) * a^2 := 
sorry

end NUMINAMATH_GPT_exponent_multiplication_l2393_239316


namespace NUMINAMATH_GPT_intersection_A_B_l2393_239319

-- Define the set A
def A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}

-- Define the set B
def B := {x : ℝ | x^2 - x < 0}

-- The proof problem statement in Lean 4
theorem intersection_A_B : A ∩ B = {y : ℝ | 0 < y ∧ y < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2393_239319


namespace NUMINAMATH_GPT_factorial_division_l2393_239337

-- Define factorial using the standard library's factorial function
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- The problem statement
theorem factorial_division :
  (factorial 15) / (factorial 6 * factorial 9) = 834 :=
sorry

end NUMINAMATH_GPT_factorial_division_l2393_239337


namespace NUMINAMATH_GPT_beneficiary_received_32_176_l2393_239332

noncomputable def A : ℝ := 19520 / 0.728
noncomputable def B : ℝ := 1.20 * A
noncomputable def C : ℝ := 1.44 * A
noncomputable def D : ℝ := 1.728 * A

theorem beneficiary_received_32_176 :
    round B = 32176 :=
by
    sorry

end NUMINAMATH_GPT_beneficiary_received_32_176_l2393_239332


namespace NUMINAMATH_GPT_trips_needed_l2393_239308

def barbieCapacity : Nat := 4
def brunoCapacity : Nat := 8
def totalCoconuts : Nat := 144

theorem trips_needed : (totalCoconuts / (barbieCapacity + brunoCapacity)) = 12 := by
  sorry

end NUMINAMATH_GPT_trips_needed_l2393_239308


namespace NUMINAMATH_GPT_mutually_exclusive_event_l2393_239361

def Event := String  -- define a simple type for events

/-- Define the events -/
def at_most_one_hit : Event := "at most one hit"
def two_hits : Event := "two hits"

/-- Define a function to check mutual exclusiveness -/
def mutually_exclusive (e1 e2 : Event) : Prop := 
  e1 ≠ e2

theorem mutually_exclusive_event :
  mutually_exclusive at_most_one_hit two_hits :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_event_l2393_239361


namespace NUMINAMATH_GPT_circle_divides_CD_in_ratio_l2393_239315

variable (A B C D : Point)
variable (BC a : ℝ)
variable (AD : ℝ := (1 + Real.sqrt 15) * BC)
variable (radius : ℝ := (2 / 3) * BC)
variable (EF : ℝ := (Real.sqrt 7 / 3) * BC)
variable (is_isosceles_trapezoid : is_isosceles_trapezoid A B C D)
variable (circle_centered_at_C : circle_centered_at C radius)
variable (chord_EF : chord_intersects_base EF AD)

theorem circle_divides_CD_in_ratio (CD DK KC : ℝ) (H1 : CD = 2 * a)
  (H2 : DK + KC = CD) (H3 : KC = CD - DK) : DK / KC = 2 :=
sorry

end NUMINAMATH_GPT_circle_divides_CD_in_ratio_l2393_239315


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l2393_239385

theorem asymptotes_of_hyperbola (x y : ℝ) :
  (x ^ 2 / 4 - y ^ 2 / 9 = -1) →
  (y = (3 / 2) * x ∨ y = -(3 / 2) * x) :=
sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l2393_239385


namespace NUMINAMATH_GPT_simple_interest_principal_l2393_239371

theorem simple_interest_principal (R T SI : ℝ) (hR : R = 9 / 100) (hT : T = 1) (hSI : SI = 900) : 
  (SI / (R * T) = 10000) :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_principal_l2393_239371


namespace NUMINAMATH_GPT_train_cross_bridge_time_l2393_239350

/-
  Define the given conditions:
  - Length of the train (lt): 200 m
  - Speed of the train (st_kmh): 72 km/hr
  - Length of the bridge (lb): 132 m
-/

namespace TrainProblem

def length_of_train : ℕ := 200
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 132

/-
  Convert speed from km/hr to m/s
-/
def speed_of_train_ms : ℕ := speed_of_train_kmh * 1000 / 3600

/-
  Calculate total distance to be traveled (train length + bridge length).
-/
def total_distance : ℕ := length_of_train + length_of_bridge

/-
  Use the formula Time = Distance / Speed
-/
def time_to_cross_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_cross_bridge_time : 
  (length_of_train = 200) →
  (speed_of_train_kmh = 72) →
  (length_of_bridge = 132) →
  time_to_cross_bridge = 16.6 :=
by
  intros lt st lb
  sorry

end TrainProblem

end NUMINAMATH_GPT_train_cross_bridge_time_l2393_239350


namespace NUMINAMATH_GPT_total_spent_correct_l2393_239363

def cost_ornamental_plants : Float := 467.00
def cost_garden_tool_set : Float := 85.00
def cost_potting_soil : Float := 38.00

def discount_plants : Float := 0.15
def discount_tools : Float := 0.10
def discount_soil : Float := 0.00

def sales_tax_rate : Float := 0.08
def surcharge : Float := 12.00

def discounted_price (original_price : Float) (discount_rate : Float) : Float :=
  original_price * (1.0 - discount_rate)

def subtotal (price_plants : Float) (price_tools : Float) (price_soil : Float) : Float :=
  price_plants + price_tools + price_soil

def sales_tax (amount : Float) (tax_rate : Float) : Float :=
  amount * tax_rate

def total (subtotal : Float) (sales_tax : Float) (surcharge : Float) : Float :=
  subtotal + sales_tax + surcharge

def final_total_spent : Float :=
  let price_plants := discounted_price cost_ornamental_plants discount_plants
  let price_tools := discounted_price cost_garden_tool_set discount_tools
  let price_soil := cost_potting_soil
  let subtotal_amount := subtotal price_plants price_tools price_soil
  let tax_amount := sales_tax subtotal_amount sales_tax_rate
  total subtotal_amount tax_amount surcharge

theorem total_spent_correct : final_total_spent = 564.37 :=
  by sorry

end NUMINAMATH_GPT_total_spent_correct_l2393_239363


namespace NUMINAMATH_GPT_eggs_for_dinner_l2393_239354

-- Definitions of the conditions
def eggs_for_breakfast := 2
def eggs_for_lunch := 3
def total_eggs := 6

-- The quantity of eggs for dinner needs to be proved
theorem eggs_for_dinner :
  ∃ x : ℕ, x + eggs_for_breakfast + eggs_for_lunch = total_eggs ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_eggs_for_dinner_l2393_239354


namespace NUMINAMATH_GPT_linear_eq_k_l2393_239349

theorem linear_eq_k (k : ℕ) : (∀ x : ℝ, x^(k-1) + 3 = 0 ↔ k = 2) :=
by
  sorry

end NUMINAMATH_GPT_linear_eq_k_l2393_239349


namespace NUMINAMATH_GPT_total_hours_difference_l2393_239310

-- Definitions based on conditions
def hours_learning_english := 6
def hours_learning_chinese := 2
def hours_learning_spanish := 3
def hours_learning_french := 1

-- Calculation of total time spent on English and Chinese
def total_hours_english_chinese := hours_learning_english + hours_learning_chinese

-- Calculation of total time spent on Spanish and French
def total_hours_spanish_french := hours_learning_spanish + hours_learning_french

-- Calculation of the difference in hours spent
def hours_difference := total_hours_english_chinese - total_hours_spanish_french

-- Statement to prove
theorem total_hours_difference : hours_difference = 4 := by
  sorry

end NUMINAMATH_GPT_total_hours_difference_l2393_239310


namespace NUMINAMATH_GPT_portion_of_pizza_eaten_l2393_239351

-- Define the conditions
def total_slices : ℕ := 16
def slices_left : ℕ := 4
def slices_eaten : ℕ := total_slices - slices_left

-- Define the portion of pizza eaten
def portion_eaten := (slices_eaten : ℚ) / (total_slices : ℚ)

-- Statement to prove
theorem portion_of_pizza_eaten : portion_eaten = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_portion_of_pizza_eaten_l2393_239351


namespace NUMINAMATH_GPT_probability_triplet_1_2_3_in_10_rolls_l2393_239377

noncomputable def probability_of_triplet (n : ℕ) : ℝ :=
  let A0 := (6^10 : ℝ)
  let A1 := (8 * 6^7 : ℝ)
  let A2 := (15 * 6^4 : ℝ)
  let A3 := (4 * 6 : ℝ)
  let total := A0
  let p := (A0 - (total - (A1 - A2 + A3))) / total
  p

theorem probability_triplet_1_2_3_in_10_rolls : 
  abs (probability_of_triplet 10 - 0.0367) < 0.0001 :=
by
  sorry

end NUMINAMATH_GPT_probability_triplet_1_2_3_in_10_rolls_l2393_239377


namespace NUMINAMATH_GPT_sequence_term_number_l2393_239311

theorem sequence_term_number (n : ℕ) (a_n : ℕ) (h : a_n = 2 * n ^ 2 - 3) : a_n = 125 → n = 8 :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_number_l2393_239311


namespace NUMINAMATH_GPT_difference_between_20th_and_first_15_l2393_239307

def grains_on_square (k : ℕ) : ℕ := 2^k

def total_grains_on_first_15_squares : ℕ :=
  (Finset.range 15).sum (λ k => grains_on_square (k + 1))

def grains_on_20th_square : ℕ := grains_on_square 20

theorem difference_between_20th_and_first_15 :
  grains_on_20th_square - total_grains_on_first_15_squares = 983042 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_20th_and_first_15_l2393_239307


namespace NUMINAMATH_GPT_cars_with_neither_l2393_239301

theorem cars_with_neither (total_cars air_bag power_windows both : ℕ) 
                          (h1 : total_cars = 65) (h2 : air_bag = 45)
                          (h3 : power_windows = 30) (h4 : both = 12) : 
                          (total_cars - (air_bag + power_windows - both) = 2) :=
by
  sorry

end NUMINAMATH_GPT_cars_with_neither_l2393_239301
