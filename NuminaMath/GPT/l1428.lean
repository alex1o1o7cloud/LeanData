import Mathlib

namespace NUMINAMATH_GPT_circle_standard_equation_l1428_142808

noncomputable def circle_through_ellipse_vertices : Prop :=
  ∃ (a : ℝ) (r : ℝ), a < 0 ∧
    (∀ (x y : ℝ),   -- vertices of the ellipse
      ((x = 4 ∧ y = 0) ∨ (x = 0 ∧ (y = 2 ∨ y = -2)))
      → (x + a)^2 + y^2 = r^2) ∧
    ( a = -3/2 ∧ r = 5/2 ∧ 
      ∀ (x y : ℝ), (x + 3/2)^2 + y^2 = (5/2)^2
    )

theorem circle_standard_equation :
  circle_through_ellipse_vertices :=
sorry

end NUMINAMATH_GPT_circle_standard_equation_l1428_142808


namespace NUMINAMATH_GPT_find_missing_number_l1428_142890

theorem find_missing_number (n x : ℕ) (h : n * (n + 1) / 2 - x = 2012) : x = 4 := by
  sorry

end NUMINAMATH_GPT_find_missing_number_l1428_142890


namespace NUMINAMATH_GPT_reflected_light_eq_l1428_142826

theorem reflected_light_eq
  (incident_light : ∀ x y : ℝ, 2 * x - y + 6 = 0)
  (reflection_line : ∀ x y : ℝ, y = x) :
  ∃ x y : ℝ, x + 2 * y + 18 = 0 :=
sorry

end NUMINAMATH_GPT_reflected_light_eq_l1428_142826


namespace NUMINAMATH_GPT_difference_of_squares_divisible_by_18_l1428_142830

-- Definitions of odd integers.
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- The main theorem stating the equivalence.
theorem difference_of_squares_divisible_by_18 (a b : ℤ) 
  (ha : is_odd a) (hb : is_odd b) : 
  ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) % 18 = 0 := 
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_divisible_by_18_l1428_142830


namespace NUMINAMATH_GPT_StockPriceAdjustment_l1428_142877

theorem StockPriceAdjustment (P₀ P₁ P₂ P₃ P₄ : ℝ) (january_increase february_decrease march_increase : ℝ) :
  P₀ = 150 →
  january_increase = 0.10 →
  february_decrease = 0.15 →
  march_increase = 0.30 →
  P₁ = P₀ * (1 + january_increase) →
  P₂ = P₁ * (1 - february_decrease) →
  P₃ = P₂ * (1 + march_increase) →
  142.5 <= P₃ * (1 - 0.17) ∧ P₃ * (1 - 0.17) <= 157.5 :=
by
  intros hP₀ hJanuaryIncrease hFebruaryDecrease hMarchIncrease hP₁ hP₂ hP₃
  sorry

end NUMINAMATH_GPT_StockPriceAdjustment_l1428_142877


namespace NUMINAMATH_GPT_min_distance_sum_coordinates_l1428_142868

theorem min_distance_sum_coordinates (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, -1)) :
  ∃ P : ℝ × ℝ, P = (0, 3) ∧ ∀ Q : ℝ × ℝ, Q.1 = 0 → |A.1 - Q.1| + |A.2 - Q.2| + |B.1 - Q.1| + |B.2 - Q.2| ≥ |A.1 - (0 : ℝ)| + |A.2 - (3 : ℝ)| + |B.1 - (0 : ℝ)| + |B.2 - (3 : ℝ)| := 
sorry

end NUMINAMATH_GPT_min_distance_sum_coordinates_l1428_142868


namespace NUMINAMATH_GPT_correct_statements_eq_l1428_142815

-- Definitions used in the Lean 4 statement should only directly appear in the conditions
variable {a b c : ℝ} 

-- Use the condition directly
theorem correct_statements_eq (h : a / c = b / c) (hc : c ≠ 0) : a = b := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_correct_statements_eq_l1428_142815


namespace NUMINAMATH_GPT_problem_p_s_difference_l1428_142858

def P : ℤ := 12 - (3 * 4)
def S : ℤ := (12 - 3) * 4

theorem problem_p_s_difference : P - S = -36 := by
  sorry

end NUMINAMATH_GPT_problem_p_s_difference_l1428_142858


namespace NUMINAMATH_GPT_book_cost_price_l1428_142833

theorem book_cost_price (SP : ℝ) (P : ℝ) (C : ℝ) (hSP: SP = 260) (hP: P = 0.20) : C = 216.67 :=
by 
  sorry

end NUMINAMATH_GPT_book_cost_price_l1428_142833


namespace NUMINAMATH_GPT_train_crosses_signal_post_time_l1428_142856

theorem train_crosses_signal_post_time 
  (length_train : ℕ) 
  (length_bridge : ℕ) 
  (time_bridge_minutes : ℕ) 
  (time_signal_post_seconds : ℕ) 
  (h_length_train : length_train = 600) 
  (h_length_bridge : length_bridge = 1800) 
  (h_time_bridge_minutes : time_bridge_minutes = 2) 
  (h_time_signal_post : time_signal_post_seconds = 30) : 
  (length_train / ((length_train + length_bridge) / (time_bridge_minutes * 60))) = time_signal_post_seconds :=
by
  sorry

end NUMINAMATH_GPT_train_crosses_signal_post_time_l1428_142856


namespace NUMINAMATH_GPT_max_d_6_digit_multiple_33_l1428_142884

theorem max_d_6_digit_multiple_33 (x d e : ℕ) 
  (hx : 1 ≤ x ∧ x ≤ 9) 
  (hd : 0 ≤ d ∧ d ≤ 9) 
  (he : 0 ≤ e ∧ e ≤ 9)
  (h1 : (x * 100000 + 50000 + d * 1000 + 300 + 30 + e) ≥ 100000) 
  (h2 : (x + d + e + 11) % 3 = 0)
  (h3 : ((x + d - e - 5 + 11) % 11 = 0)) :
  d = 9 := 
sorry

end NUMINAMATH_GPT_max_d_6_digit_multiple_33_l1428_142884


namespace NUMINAMATH_GPT_find_principal_l1428_142848

theorem find_principal (P : ℝ) (r : ℝ) (t : ℝ) (CI SI : ℝ) 
  (h_r : r = 0.20) 
  (h_t : t = 2) 
  (h_diff : CI - SI = 144) 
  (h_CI : CI = P * (1 + r)^t - P) 
  (h_SI : SI = P * r * t) : 
  P = 3600 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l1428_142848


namespace NUMINAMATH_GPT_carpet_needed_correct_l1428_142891

def length_room : ℕ := 15
def width_room : ℕ := 9
def length_closet : ℕ := 3
def width_closet : ℕ := 2

def area_room : ℕ := length_room * width_room
def area_closet : ℕ := length_closet * width_closet
def area_to_carpet : ℕ := area_room - area_closet
def sq_ft_to_sq_yd (sqft: ℕ) : ℕ := (sqft + 8) / 9  -- Adding 8 to ensure proper rounding up

def carpet_needed : ℕ := sq_ft_to_sq_yd area_to_carpet

theorem carpet_needed_correct :
  carpet_needed = 15 := by
  sorry

end NUMINAMATH_GPT_carpet_needed_correct_l1428_142891


namespace NUMINAMATH_GPT_integer_parts_are_divisible_by_17_l1428_142847

-- Define that a is the greatest positive root of the given polynomial
def is_greatest_positive_root (a : ℝ) : Prop :=
  (∀ x : ℝ, x^3 - 3 * x^2 + 1 = 0 → x ≤ a) ∧ a > 0 ∧ (a^3 - 3 * a^2 + 1 = 0)

-- Define the main theorem to prove
theorem integer_parts_are_divisible_by_17 (a : ℝ)
  (h_root : is_greatest_positive_root a) :
  (⌊a ^ 1788⌋ % 17 = 0) ∧ (⌊a ^ 1988⌋ % 17 = 0) := 
sorry

end NUMINAMATH_GPT_integer_parts_are_divisible_by_17_l1428_142847


namespace NUMINAMATH_GPT_percentage_increase_l1428_142886

variable (P N N' : ℝ)
variable (h : P * 0.90 * N' = P * N * 1.035)

theorem percentage_increase :
  ((N' - N) / N) * 100 = 15 :=
by
  -- By given condition, we have the equation:
  -- P * 0.90 * N' = P * N * 1.035
  sorry

end NUMINAMATH_GPT_percentage_increase_l1428_142886


namespace NUMINAMATH_GPT_remainder_5_pow_2048_mod_17_l1428_142897

theorem remainder_5_pow_2048_mod_17 : (5 ^ 2048) % 17 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_5_pow_2048_mod_17_l1428_142897


namespace NUMINAMATH_GPT_kayla_apples_l1428_142852

variable (x y : ℕ)
variable (h1 : x + (10 + 4 * x) = 340)
variable (h2 : y = 10 + 4 * x)

theorem kayla_apples : y = 274 :=
by
  sorry

end NUMINAMATH_GPT_kayla_apples_l1428_142852


namespace NUMINAMATH_GPT_solve_for_y_l1428_142855

-- The given condition as a hypothesis
variables {x y : ℝ}

-- The theorem statement
theorem solve_for_y (h : 3 * x - y + 5 = 0) : y = 3 * x + 5 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l1428_142855


namespace NUMINAMATH_GPT_min_value_of_expr_min_value_at_specific_points_l1428_142827

noncomputable def min_value_expr (p q r : ℝ) : ℝ := 8 * p^4 + 18 * q^4 + 50 * r^4 + 1 / (8 * p * q * r)

theorem min_value_of_expr : ∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → min_value_expr p q r ≥ 6 :=
by
  intro p q r hp hq hr
  sorry

theorem min_value_at_specific_points : min_value_expr (1 / (8 : ℝ)^(1 / 4)) (1 / (18 : ℝ)^(1 / 4)) (1 / (50 : ℝ)^(1 / 4)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expr_min_value_at_specific_points_l1428_142827


namespace NUMINAMATH_GPT_mowing_work_rate_l1428_142843

variables (A B C : ℚ)

theorem mowing_work_rate :
  A + B = 1/28 → A + B + C = 1/21 → C = 1/84 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_mowing_work_rate_l1428_142843


namespace NUMINAMATH_GPT_arithmetic_sum_problem_l1428_142889

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The problem statement -/
theorem arithmetic_sum_problem
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_terms S a)
  (h_S10 : S 10 = 4) :
  a 3 + a 8 = 4 / 5 := 
sorry

end NUMINAMATH_GPT_arithmetic_sum_problem_l1428_142889


namespace NUMINAMATH_GPT_trigonometric_inequality_l1428_142805

-- Define the necessary mathematical objects and structures:
noncomputable def sin (x : ℝ) : ℝ := sorry -- Assume sine function as given

-- The theorem statement
theorem trigonometric_inequality {x y z A B C : ℝ} 
  (hA : A + B + C = π) -- A, B, C are angles of a triangle
  :
  ((x + y + z) / 2) ^ 2 ≥ x * y * (sin A) ^ 2 + y * z * (sin B) ^ 2 + z * x * (sin C) ^ 2 :=
sorry

end NUMINAMATH_GPT_trigonometric_inequality_l1428_142805


namespace NUMINAMATH_GPT_stickers_earned_correct_l1428_142806

-- Define the initial and final number of stickers.
def initial_stickers : ℕ := 39
def final_stickers : ℕ := 61

-- Define how many stickers Pat earned during the week
def stickers_earned : ℕ := final_stickers - initial_stickers

-- State the main theorem
theorem stickers_earned_correct : stickers_earned = 22 :=
by
  show final_stickers - initial_stickers = 22
  sorry

end NUMINAMATH_GPT_stickers_earned_correct_l1428_142806


namespace NUMINAMATH_GPT_weight_of_3_moles_HClO2_correct_l1428_142885

def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cl : ℝ := 35.453
def atomic_weight_O : ℝ := 15.999

def molecular_weight_HClO2 : ℝ := (1 * atomic_weight_H) + (1 * atomic_weight_Cl) + (2 * atomic_weight_O)
def weight_of_3_moles_HClO2 : ℝ := 3 * molecular_weight_HClO2

theorem weight_of_3_moles_HClO2_correct : weight_of_3_moles_HClO2 = 205.377 := by
  sorry

end NUMINAMATH_GPT_weight_of_3_moles_HClO2_correct_l1428_142885


namespace NUMINAMATH_GPT_at_least_one_negative_l1428_142820

theorem at_least_one_negative (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a^2 + 1 / b = b^2 + 1 / a) : a < 0 ∨ b < 0 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_negative_l1428_142820


namespace NUMINAMATH_GPT_shoes_produced_min_pairs_for_profit_l1428_142832

-- given conditions
def production_cost (n : ℕ) : ℕ := 4000 + 50 * n

-- Question (1)
theorem shoes_produced (C : ℕ) (h : C = 36000) : ∃ n : ℕ, production_cost n = C :=
by sorry

-- given conditions for part (2)
def selling_price (price_per_pair : ℕ) (n : ℕ) : ℕ := price_per_pair * n
def profit (price_per_pair : ℕ) (n : ℕ) : ℕ := selling_price price_per_pair n - production_cost n

-- Question (2)
theorem min_pairs_for_profit (price_per_pair profit_goal : ℕ) (h : price_per_pair = 90) (h1 : profit_goal = 8500) :
  ∃ n : ℕ, profit price_per_pair n ≥ profit_goal :=
by sorry

end NUMINAMATH_GPT_shoes_produced_min_pairs_for_profit_l1428_142832


namespace NUMINAMATH_GPT_positive_integer_pairs_l1428_142861

theorem positive_integer_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  a^b = b^(a^2) ↔ (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 16) ∨ (a = 3 ∧ b = 27) :=
by sorry

end NUMINAMATH_GPT_positive_integer_pairs_l1428_142861


namespace NUMINAMATH_GPT_evaluate_expression_l1428_142899

theorem evaluate_expression : 4^1 + 3^2 - 2^3 + 1^4 = 6 := by
  -- We will skip the proof steps here using sorry
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1428_142899


namespace NUMINAMATH_GPT_at_least_one_not_greater_than_neg_two_l1428_142813

open Real

theorem at_least_one_not_greater_than_neg_two
  {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + (1 / b) ≤ -2 ∨ b + (1 / c) ≤ -2 ∨ c + (1 / a) ≤ -2 :=
sorry

end NUMINAMATH_GPT_at_least_one_not_greater_than_neg_two_l1428_142813


namespace NUMINAMATH_GPT_quadractic_integer_roots_l1428_142851

theorem quadractic_integer_roots (n : ℕ) (h : n > 0) :
  (∃ x y : ℤ, x^2 - 4 * x + n = 0 ∧ y^2 - 4 * y + n = 0) ↔ (n = 3 ∨ n = 4) :=
by
  sorry

end NUMINAMATH_GPT_quadractic_integer_roots_l1428_142851


namespace NUMINAMATH_GPT_set_union_proof_l1428_142821

theorem set_union_proof (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {1, 2^a})
  (hB : B = {a, b}) 
  (h_inter : A ∩ B = {1/4}) :
  A ∪ B = {-2, 1, 1/4} := 
by 
  sorry

end NUMINAMATH_GPT_set_union_proof_l1428_142821


namespace NUMINAMATH_GPT_sum_first_eight_terms_geometric_sequence_l1428_142874

noncomputable def sum_of_geometric_sequence (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_eight_terms_geometric_sequence :
  sum_of_geometric_sequence (1/2) (1/3) 8 = 9840 / 6561 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_eight_terms_geometric_sequence_l1428_142874


namespace NUMINAMATH_GPT_smallest_integer_solution_l1428_142882

theorem smallest_integer_solution (x : ℤ) (h : 10 - 5 * x < -18) : x = 6 :=
sorry

end NUMINAMATH_GPT_smallest_integer_solution_l1428_142882


namespace NUMINAMATH_GPT_last_four_digits_of_5_pow_2017_l1428_142828

theorem last_four_digits_of_5_pow_2017 : (5 ^ 2017) % 10000 = 3125 :=
by sorry

end NUMINAMATH_GPT_last_four_digits_of_5_pow_2017_l1428_142828


namespace NUMINAMATH_GPT_infinite_series_correct_l1428_142845

noncomputable def infinite_series_sum : ℚ := 
  ∑' n : ℕ, (n+1)^2 * (1/999)^n

theorem infinite_series_correct : infinite_series_sum = 997005 / 996004 :=
  sorry

end NUMINAMATH_GPT_infinite_series_correct_l1428_142845


namespace NUMINAMATH_GPT_part1_part2_l1428_142816

-- Part (1)
theorem part1 : -6 * -2 + -5 * 16 = -68 := by
  sorry

-- Part (2)
theorem part2 : -1^4 + (1 / 4) * (2 * -6 - (-4)^2) = -8 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1428_142816


namespace NUMINAMATH_GPT_part1_part2_l1428_142801

def p (x : ℝ) : Prop := x^2 - 10*x + 16 ≤ 0
def q (x m : ℝ) : Prop := m > 0 ∧ x^2 - 4*m*x + 3*m^2 ≤ 0

theorem part1 (x : ℝ) : 
  (∃ (m : ℝ), m = 1 ∧ (p x ∨ q x m)) → 1 ≤ x ∧ x ≤ 8 :=
by
  intros
  sorry

theorem part2 (m : ℝ) :
  (∀ x, q x m → p x) ∧ ∃ x, ¬ q x m ∧ p x → 2 ≤ m ∧ m ≤ 8/3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_part1_part2_l1428_142801


namespace NUMINAMATH_GPT_algebra_expression_value_l1428_142829

theorem algebra_expression_value (x y : ℤ) (h : x - 2 * y + 2 = 5) : 2 * x - 4 * y - 1 = 5 :=
by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l1428_142829


namespace NUMINAMATH_GPT_determine_x_squared_plus_y_squared_l1428_142872

theorem determine_x_squared_plus_y_squared (x y : ℝ) 
(h : (x^2 + y^2 + 2) * (x^2 + y^2 - 3) = 6) : x^2 + y^2 = 4 :=
sorry

end NUMINAMATH_GPT_determine_x_squared_plus_y_squared_l1428_142872


namespace NUMINAMATH_GPT_neg_p_l1428_142873

theorem neg_p :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_neg_p_l1428_142873


namespace NUMINAMATH_GPT_largest_possible_s_l1428_142895

theorem largest_possible_s :
  ∃ s r : ℕ, (r ≥ s) ∧ (s ≥ 5) ∧ (122 * r - 120 * s = r * s) ∧ (s = 121) :=
by sorry

end NUMINAMATH_GPT_largest_possible_s_l1428_142895


namespace NUMINAMATH_GPT_prove_y_identity_l1428_142825

theorem prove_y_identity (y : ℤ) (h1 : y^2 = 2209) : (y + 2) * (y - 2) = 2205 :=
by
  sorry

end NUMINAMATH_GPT_prove_y_identity_l1428_142825


namespace NUMINAMATH_GPT_union_sets_intersection_complement_l1428_142838

open Set

noncomputable def U := (univ : Set ℝ)
def A := { x : ℝ | x ≥ 2 }
def B := { x : ℝ | x < 5 }

theorem union_sets : A ∪ B = univ := by
  sorry

theorem intersection_complement : (U \ A) ∩ B = { x : ℝ | x < 2 } := by
  sorry

end NUMINAMATH_GPT_union_sets_intersection_complement_l1428_142838


namespace NUMINAMATH_GPT_sum_of_digits_18_l1428_142862

def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

theorem sum_of_digits_18 (A B C D : ℕ) 
(h1 : A + D = 10)
(h2 : B + C + 1 = 10 + D)
(h3 : C + B + 1 = 10 + B)
(h4 : D + A + 1 = 11)
(h_distinct : distinct_digits A B C D) :
  A + B + C + D = 18 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_18_l1428_142862


namespace NUMINAMATH_GPT_smallest_positive_int_linear_combination_l1428_142819

theorem smallest_positive_int_linear_combination (m n : ℤ) :
  ∃ k : ℤ, 4509 * m + 27981 * n = k ∧ k > 0 ∧ k ≤ 4509 * m + 27981 * n → k = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_int_linear_combination_l1428_142819


namespace NUMINAMATH_GPT_find_repair_charge_l1428_142893

theorem find_repair_charge
    (cost_oil_change : ℕ)
    (cost_car_wash : ℕ)
    (num_oil_changes : ℕ)
    (num_repairs : ℕ)
    (num_car_washes : ℕ)
    (total_earnings : ℕ)
    (R : ℕ) :
    (cost_oil_change = 20) →
    (cost_car_wash = 5) →
    (num_oil_changes = 5) →
    (num_repairs = 10) →
    (num_car_washes = 15) →
    (total_earnings = 475) →
    5 * cost_oil_change + 10 * R + 15 * cost_car_wash = total_earnings →
    R = 30 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end NUMINAMATH_GPT_find_repair_charge_l1428_142893


namespace NUMINAMATH_GPT_problem1_problem2_l1428_142878

-- Statement for Question (1)
theorem problem1 (x : ℝ) (h : |x - 1| + x ≥ x + 2) : x ≤ -1 ∨ x ≥ 3 :=
  sorry

-- Statement for Question (2)
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + x ≤ 3 * x → x ≥ 2) : a = 6 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1428_142878


namespace NUMINAMATH_GPT_pair_d_same_function_l1428_142892

theorem pair_d_same_function : ∀ x : ℝ, x = (x ^ 5) ^ (1 / 5) := 
by
  intro x
  sorry

end NUMINAMATH_GPT_pair_d_same_function_l1428_142892


namespace NUMINAMATH_GPT_sequence_initial_term_l1428_142846

theorem sequence_initial_term (a : ℕ) :
  let a_1 := a
  let a_2 := 2
  let a_3 := a_1 + a_2
  let a_4 := a_1 + a_2 + a_3
  let a_5 := a_1 + a_2 + a_3 + a_4
  let a_6 := a_1 + a_2 + a_3 + a_4 + a_5
  a_6 = 56 → a = 5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sequence_initial_term_l1428_142846


namespace NUMINAMATH_GPT_segment_ratio_ae_ad_l1428_142875

/-- Given points B, C, and E lie on line segment AD, and the following conditions:
  1. The length of segment AB is twice the length of segment BD.
  2. The length of segment AC is 5 times the length of segment CD.
  3. The length of segment BE is one-third the length of segment EC.
Prove that the fraction of the length of segment AD that segment AE represents is 17/24. -/
theorem segment_ratio_ae_ad (AB BD AC CD BE EC AD AE : ℝ)
    (h1 : AB = 2 * BD)
    (h2 : AC = 5 * CD)
    (h3 : BE = (1/3) * EC)
    (h4 : AD = 6 * CD)
    (h5 : AE = 4.25 * CD) :
    AE / AD = 17 / 24 := 
  by 
  sorry

end NUMINAMATH_GPT_segment_ratio_ae_ad_l1428_142875


namespace NUMINAMATH_GPT_pizza_slices_with_both_toppings_l1428_142823

theorem pizza_slices_with_both_toppings :
  let T := 16
  let c := 10
  let p := 12
  let n := c + p - T
  n = 6 :=
by
  let T := 16
  let c := 10
  let p := 12
  let n := c + p - T
  show n = 6
  sorry

end NUMINAMATH_GPT_pizza_slices_with_both_toppings_l1428_142823


namespace NUMINAMATH_GPT_geom_prog_common_ratio_unique_l1428_142836

theorem geom_prog_common_ratio_unique (b q : ℝ) (hb : b > 0) (hq : q > 1) :
  (∃ b : ℝ, (q = (1 + Real.sqrt 5) / 2) ∧ 
    (0 < b ∧ b * q ≠ b ∧ b * q^2 ≠ b ∧ b * q^3 ≠ b) ∧ 
    ((2 * b * q = b + b * q^2) ∨ (2 * b * q = b + b * q^3) ∨ (2 * b * q^2 = b + b * q^3))) := 
sorry

end NUMINAMATH_GPT_geom_prog_common_ratio_unique_l1428_142836


namespace NUMINAMATH_GPT_johns_sister_age_l1428_142831

variable (j d s : ℝ)

theorem johns_sister_age 
  (h1 : j = d - 15)
  (h2 : j + d = 100)
  (h3 : s = j - 5) :
  s = 37.5 := 
sorry

end NUMINAMATH_GPT_johns_sister_age_l1428_142831


namespace NUMINAMATH_GPT_Kaleb_second_half_points_l1428_142866

theorem Kaleb_second_half_points (first_half_points total_points : ℕ) (h1 : first_half_points = 43) (h2 : total_points = 66) : total_points - first_half_points = 23 := by
  sorry

end NUMINAMATH_GPT_Kaleb_second_half_points_l1428_142866


namespace NUMINAMATH_GPT_evaluate_dollar_op_l1428_142857

def dollar_op (x y : ℤ) := x * (y + 2) + 2 * x * y

theorem evaluate_dollar_op : dollar_op 4 (-1) = -4 :=
by
  -- Proof steps here
  sorry

end NUMINAMATH_GPT_evaluate_dollar_op_l1428_142857


namespace NUMINAMATH_GPT_sam_balloons_l1428_142853

theorem sam_balloons (f d t S : ℝ) (h₁ : f = 10.0) (h₂ : d = 16.0) (h₃ : t = 40.0) (h₄ : f + S - d = t) : S = 46.0 := 
by 
  -- Replace "sorry" with a valid proof to solve this problem
  sorry

end NUMINAMATH_GPT_sam_balloons_l1428_142853


namespace NUMINAMATH_GPT_tile_5x7_rectangle_with_L_trominos_l1428_142839

theorem tile_5x7_rectangle_with_L_trominos :
  ∀ k : ℕ, ¬ (∃ (tile : ℕ → ℕ → ℕ), (∀ i j, tile (i+1) (j+1) = tile (i+3) (j+3)) ∧
    ∀ i j, (i < 5 ∧ j < 7) → (tile i j = k)) :=
by sorry

end NUMINAMATH_GPT_tile_5x7_rectangle_with_L_trominos_l1428_142839


namespace NUMINAMATH_GPT_new_rectangle_perimeters_l1428_142880

theorem new_rectangle_perimeters {l w : ℕ} (h_l : l = 4) (h_w : w = 2) :
  (∃ P, P = 2 * (8 + 2) ∨ P = 2 * (4 + 4)) ∧ (P = 20 ∨ P = 16) :=
by
  sorry

end NUMINAMATH_GPT_new_rectangle_perimeters_l1428_142880


namespace NUMINAMATH_GPT_range_of_a_l1428_142870

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x)^2 + a * Real.cos x + a

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 1) → a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1428_142870


namespace NUMINAMATH_GPT_find_angle_phi_l1428_142834

-- Definitions for the conditions given in the problem
def folded_paper_angle (φ : ℝ) : Prop := 0 < φ ∧ φ < 90

def angle_XOY := 144

-- The main statement to be proven
theorem find_angle_phi (φ : ℝ) (h1 : folded_paper_angle φ) : φ = 81 :=
sorry

end NUMINAMATH_GPT_find_angle_phi_l1428_142834


namespace NUMINAMATH_GPT_no_nat_solutions_l1428_142859

theorem no_nat_solutions (x y : ℕ) : (2 * x + y) * (2 * y + x) ≠ 2017 ^ 2017 := by sorry

end NUMINAMATH_GPT_no_nat_solutions_l1428_142859


namespace NUMINAMATH_GPT_arithmetic_progression_y_value_l1428_142883

theorem arithmetic_progression_y_value (x y : ℚ) 
  (h1 : x = 2)
  (h2 : 2 * y - x = (y + x + 3) - (2 * y - x))
  (h3 : (3 * y + x) - (y + x + 3) = (y + x + 3) - (2 * y - x)) : 
  y = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_y_value_l1428_142883


namespace NUMINAMATH_GPT_max_pairs_correct_l1428_142894

def max_pairs (n : ℕ) : ℕ :=
  if h : n > 1 then (n * n) / 4 else 0

theorem max_pairs_correct (n : ℕ) (h : n ≥ 2) :
  (max_pairs n = (n * n) / 4) :=
by sorry

end NUMINAMATH_GPT_max_pairs_correct_l1428_142894


namespace NUMINAMATH_GPT_lily_disproves_tom_claim_l1428_142867

-- Define the cards and the claim
inductive Card
| A : Card
| R : Card
| Circle : Card
| Square : Card
| Triangle : Card

def has_consonant (c : Card) : Prop :=
  match c with
  | Card.R => true
  | _ => false

def has_triangle (c : Card) : Card → Prop :=
  fun c' =>
    match c with
    | Card.R => c' = Card.Triangle
    | _ => true

def tom_claim (c : Card) (c' : Card) : Prop :=
  has_consonant c → has_triangle c c'

-- Proof problem statement:
theorem lily_disproves_tom_claim (c : Card) (c' : Card) : c = Card.R → ¬ has_triangle c c' → ¬ tom_claim c c' :=
by
  intros
  sorry

end NUMINAMATH_GPT_lily_disproves_tom_claim_l1428_142867


namespace NUMINAMATH_GPT_unique_outfits_count_l1428_142841

theorem unique_outfits_count (s : Fin 5) (p : Fin 6) (restricted_pairings : (Fin 1 × Fin 2) → Prop) 
  (r : restricted_pairings (0, 0) ∧ restricted_pairings (0, 1)) : ∃ n, n = 28 ∧ 
  ∃ (outfits : Fin 5 → Fin 6 → Prop), 
    (∀ s p, outfits s p) ∧ 
    (∀ p, ¬outfits 0 p ↔ p = 0 ∨ p = 1) := by
  sorry

end NUMINAMATH_GPT_unique_outfits_count_l1428_142841


namespace NUMINAMATH_GPT_inequality_always_true_l1428_142810

theorem inequality_always_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| := 
sorry

end NUMINAMATH_GPT_inequality_always_true_l1428_142810


namespace NUMINAMATH_GPT_negative_870_in_third_quadrant_l1428_142807

noncomputable def angle_in_third_quadrant (theta : ℝ) : Prop :=
  180 < theta ∧ theta < 270

theorem negative_870_in_third_quadrant:
  angle_in_third_quadrant 210 :=
by
  sorry

end NUMINAMATH_GPT_negative_870_in_third_quadrant_l1428_142807


namespace NUMINAMATH_GPT_f_2007_l1428_142850

def A : Set ℚ := {x : ℚ | x ≠ 0 ∧ x ≠ 1}

noncomputable def f : A → ℝ := sorry

theorem f_2007 :
  (∀ x : ℚ, x ∈ A → f ⟨x, sorry⟩ + f ⟨1 - (1/x), sorry⟩ = Real.log (|x|)) →
  f ⟨2007, sorry⟩ = Real.log (|2007|) :=
sorry

end NUMINAMATH_GPT_f_2007_l1428_142850


namespace NUMINAMATH_GPT_sqrt_450_eq_15_sqrt_2_l1428_142814

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_450_eq_15_sqrt_2_l1428_142814


namespace NUMINAMATH_GPT_polyhedron_with_12_edges_l1428_142864

def prism_edges (n : Nat) : Nat :=
  3 * n

def pyramid_edges (n : Nat) : Nat :=
  2 * n

def Quadrangular_prism : Nat := prism_edges 4
def Quadrangular_pyramid : Nat := pyramid_edges 4
def Pentagonal_pyramid : Nat := pyramid_edges 5
def Pentagonal_prism : Nat := prism_edges 5

theorem polyhedron_with_12_edges :
  (Quadrangular_prism = 12) ∧
  (Quadrangular_pyramid ≠ 12) ∧
  (Pentagonal_pyramid ≠ 12) ∧
  (Pentagonal_prism ≠ 12) := by
  sorry

end NUMINAMATH_GPT_polyhedron_with_12_edges_l1428_142864


namespace NUMINAMATH_GPT_age_of_B_l1428_142888

theorem age_of_B (a b c d : ℕ) 
  (h1: a + b + c + d = 112)
  (h2: a + c = 58)
  (h3: 2 * b + 3 * d = 135)
  (h4: b + d = 54) :
  b = 27 :=
by
  sorry

end NUMINAMATH_GPT_age_of_B_l1428_142888


namespace NUMINAMATH_GPT_election_debate_conditions_l1428_142871

theorem election_debate_conditions (n : ℕ) (h_n : n ≥ 3) :
  ¬ ∃ (p : ℕ), n = 2 * (2 ^ p - 2) + 1 :=
sorry

end NUMINAMATH_GPT_election_debate_conditions_l1428_142871


namespace NUMINAMATH_GPT_find_missing_number_l1428_142824

theorem find_missing_number (n : ℤ) (h : 1234562 - n * 3 * 2 = 1234490) : 
  n = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_number_l1428_142824


namespace NUMINAMATH_GPT_coeffs_sum_of_binomial_expansion_l1428_142854

theorem coeffs_sum_of_binomial_expansion :
  (3 * x - 2) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_0 = 64 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = -63 :=
by
  sorry

end NUMINAMATH_GPT_coeffs_sum_of_binomial_expansion_l1428_142854


namespace NUMINAMATH_GPT_sequence_solution_l1428_142802

-- Define the sequence x_n
def x (n : ℕ) : ℚ := n / (n + 2016)

-- Given condition: x_2016 = x_m * x_n
theorem sequence_solution (m n : ℕ) (h : x 2016 = x m * x n) : 
  m = 4032 ∧ n = 6048 := 
  by sorry

end NUMINAMATH_GPT_sequence_solution_l1428_142802


namespace NUMINAMATH_GPT_calculate_cells_after_12_days_l1428_142865

theorem calculate_cells_after_12_days :
  let initial_cells := 5
  let division_factor := 3
  let days := 12
  let period := 3
  let n := days / period
  initial_cells * division_factor ^ (n - 1) = 135 := by
  sorry

end NUMINAMATH_GPT_calculate_cells_after_12_days_l1428_142865


namespace NUMINAMATH_GPT_ages_of_boys_l1428_142818

theorem ages_of_boys (a b c : ℕ) (h1 : a + b + c = 29) (h2 : a = b) (h3 : c = 11) : a = 9 :=
by
  sorry

end NUMINAMATH_GPT_ages_of_boys_l1428_142818


namespace NUMINAMATH_GPT_domain_of_sqrt_and_fraction_l1428_142811

def domain_of_function (x : ℝ) : Prop :=
  2 * x - 3 ≥ 0 ∧ x ≠ 3

theorem domain_of_sqrt_and_fraction :
  {x : ℝ | domain_of_function x} = {x : ℝ | x ≥ 3 / 2} \ {3} :=
by sorry

end NUMINAMATH_GPT_domain_of_sqrt_and_fraction_l1428_142811


namespace NUMINAMATH_GPT_stateA_issues_more_than_stateB_l1428_142860

-- Definitions based on conditions
def stateA_format : ℕ := 26^5 * 10^1
def stateB_format : ℕ := 26^3 * 10^3

-- Proof problem statement
theorem stateA_issues_more_than_stateB : stateA_format - stateB_format = 10123776 := by
  sorry

end NUMINAMATH_GPT_stateA_issues_more_than_stateB_l1428_142860


namespace NUMINAMATH_GPT_part_I_solution_set_part_II_range_a_l1428_142837

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part_I_solution_set :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} :=
by
  sorry

theorem part_II_range_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 - a) ↔ (-1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_part_I_solution_set_part_II_range_a_l1428_142837


namespace NUMINAMATH_GPT_proof_math_problem_l1428_142803

-- Define the conditions
structure Conditions where
  person1_start_noon : ℕ -- Person 1 starts from Appleminster at 12:00 PM
  person2_start_2pm : ℕ -- Person 2 starts from Boniham at 2:00 PM
  meet_time : ℕ -- They meet at 4:55 PM
  finish_time_simultaneously : Bool -- They finish their journey simultaneously

-- Define the problem
def math_problem (c : Conditions) : Prop :=
  let arrival_time := 7 * 60 -- 7:00 PM in minutes
  c.person1_start_noon = 0 ∧ -- Noon as 0 minutes (12:00 PM)
  c.person2_start_2pm = 120 ∧ -- 2:00 PM as 120 minutes
  c.meet_time = 295 ∧ -- 4:55 PM as 295 minutes
  c.finish_time_simultaneously = true → arrival_time = 420 -- 7:00 PM in minutes

-- Prove the problem statement, skipping actual proof
theorem proof_math_problem (c : Conditions) : math_problem c :=
  by sorry

end NUMINAMATH_GPT_proof_math_problem_l1428_142803


namespace NUMINAMATH_GPT_unoccupied_seats_in_business_class_l1428_142812

def airplane_seating (fc bc ec : ℕ) (pfc pbc pec : ℕ) : Nat :=
  let num_ec := ec / 2
  let num_bc := num_ec - pfc
  bc - num_bc

theorem unoccupied_seats_in_business_class :
  airplane_seating 10 30 50 3 (50/2) (50/2) = 8 := by
    sorry

end NUMINAMATH_GPT_unoccupied_seats_in_business_class_l1428_142812


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1428_142800

theorem solution_set_of_inequality (x : ℝ) : {x | x * (x - 1) > 0} = { x | x < 0 } ∪ { x | x > 1 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1428_142800


namespace NUMINAMATH_GPT_q1_q2_q3_l1428_142863

-- (1) Given |a| = 3, |b| = 1, and a < b, prove a + b = -2 or -4.
theorem q1 (a b : ℚ) (h1 : |a| = 3) (h2 : |b| = 1) (h3 : a < b) : a + b = -2 ∨ a + b = -4 := sorry

-- (2) Given rational numbers a and b such that ab ≠ 0, prove the value of (a/|a|) + (b/|b|) is 2, -2, or 0.
theorem q2 (a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) : (a / |a|) + (b / |b|) = 2 ∨ (a / |a|) + (b / |b|) = -2 ∨ (a / |a|) + (b / |b|) = 0 := sorry

-- (3) Given rational numbers a, b, c such that a + b + c = 0 and abc < 0, prove the value of (b+c)/|a| + (a+c)/|b| + (a+b)/|c| is -1.
theorem q3 (a b c : ℚ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : (b + c) / |a| + (a + c) / |b| + (a + b) / |c| = -1 := sorry

end NUMINAMATH_GPT_q1_q2_q3_l1428_142863


namespace NUMINAMATH_GPT_calc_expr_solve_fractional_eq_l1428_142879

-- Problem 1: Calculate the expression
theorem calc_expr : (-2)^2 - (64:ℝ)^(1/3) + (-3)^0 - (1/3)^0 = 0 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

-- Problem 2: Solve the fractional equation
theorem solve_fractional_eq (x : ℝ) (h : x ≠ -1) : 
  (x / (x + 1) = 5 / (2 * x + 2) - 1) ↔ x = 3 / 4 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

end NUMINAMATH_GPT_calc_expr_solve_fractional_eq_l1428_142879


namespace NUMINAMATH_GPT_stacy_height_now_l1428_142844

-- Definitions based on the given conditions
def S_initial : ℕ := 50
def J_initial : ℕ := 45
def J_growth : ℕ := 1
def S_growth : ℕ := J_growth + 6

-- Prove statement about Stacy's current height
theorem stacy_height_now : S_initial + S_growth = 57 := by
  sorry

end NUMINAMATH_GPT_stacy_height_now_l1428_142844


namespace NUMINAMATH_GPT_triangle_area_13_14_15_l1428_142842

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let sin_C := Real.sqrt (1 - cos_C^2)
  (1/2) * a * b * sin_C

theorem triangle_area_13_14_15 : area_of_triangle 13 14 15 = 84 :=
by sorry

end NUMINAMATH_GPT_triangle_area_13_14_15_l1428_142842


namespace NUMINAMATH_GPT_books_read_in_8_hours_l1428_142898

def reading_speed := 100 -- pages per hour
def book_pages := 400 -- pages per book
def hours_available := 8 -- hours

theorem books_read_in_8_hours :
  (hours_available * reading_speed) / book_pages = 2 :=
by
  sorry

end NUMINAMATH_GPT_books_read_in_8_hours_l1428_142898


namespace NUMINAMATH_GPT_average_of_remaining_four_l1428_142840

theorem average_of_remaining_four (avg10 : ℕ → ℕ) (avg6 : ℕ → ℕ) 
  (h_avg10 : avg10 10 = 80) 
  (h_avg6 : avg6 6 = 58) : 
  (avg10 10 - avg6 6 * 6) / 4 = 113 :=
sorry

end NUMINAMATH_GPT_average_of_remaining_four_l1428_142840


namespace NUMINAMATH_GPT_smallest_n_for_divisibility_condition_l1428_142835

theorem smallest_n_for_divisibility_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) →
    (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (xyz ∣ (x + y + z)^n)) ∧
    n = 13 :=
by
  use 13
  sorry

end NUMINAMATH_GPT_smallest_n_for_divisibility_condition_l1428_142835


namespace NUMINAMATH_GPT_greatest_x_for_lcm_l1428_142869

theorem greatest_x_for_lcm (x : ℕ) (h_lcm : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 :=
by
  sorry

end NUMINAMATH_GPT_greatest_x_for_lcm_l1428_142869


namespace NUMINAMATH_GPT_range_of_x_l1428_142881

def p (x : ℝ) : Prop := x^2 - 5 * x + 6 ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) (hpq : p x ∨ q x) (hnq : ¬ q x) : x ≤ 0 ∨ x ≥ 4 :=
by sorry

end NUMINAMATH_GPT_range_of_x_l1428_142881


namespace NUMINAMATH_GPT_scientific_notation_460_billion_l1428_142817

theorem scientific_notation_460_billion : 460000000000 = 4.6 * 10^11 := 
sorry

end NUMINAMATH_GPT_scientific_notation_460_billion_l1428_142817


namespace NUMINAMATH_GPT_roots_of_equation_l1428_142849

theorem roots_of_equation :
  ∀ x : ℝ, x * (x - 1) + 3 * (x - 1) = 0 ↔ x = -3 ∨ x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_roots_of_equation_l1428_142849


namespace NUMINAMATH_GPT_decreasing_function_on_real_l1428_142809

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x + y) = f x + f y
axiom f_negative (x : ℝ) : x > 0 → f x < 0
axiom f_not_identically_zero : ∃ x, f x ≠ 0

theorem decreasing_function_on_real :
  ∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2 :=
sorry

end NUMINAMATH_GPT_decreasing_function_on_real_l1428_142809


namespace NUMINAMATH_GPT_age_comparison_l1428_142822

variable (P A F X : ℕ)

theorem age_comparison :
  P = 50 →
  P = 5 / 4 * A →
  P = 5 / 6 * F →
  X = 50 - A →
  X = 10 :=
by { sorry }

end NUMINAMATH_GPT_age_comparison_l1428_142822


namespace NUMINAMATH_GPT_students_not_receiving_A_l1428_142896

theorem students_not_receiving_A (total_students : ℕ) (students_A_physics : ℕ) (students_A_chemistry : ℕ) (students_A_both : ℕ) (h_total : total_students = 40) (h_A_physics : students_A_physics = 10) (h_A_chemistry : students_A_chemistry = 18) (h_A_both : students_A_both = 6) : (total_students - ((students_A_physics + students_A_chemistry) - students_A_both)) = 18 := 
by
  sorry

end NUMINAMATH_GPT_students_not_receiving_A_l1428_142896


namespace NUMINAMATH_GPT_problem_statement_l1428_142876

noncomputable def g : ℝ → ℝ := sorry

theorem problem_statement 
  (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - x * y^2 - x + 2) :
  ∃ (m t : ℕ), (m = 1) ∧ (t = 3) ∧ (m * t = 3) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1428_142876


namespace NUMINAMATH_GPT_Inez_initial_money_l1428_142804

theorem Inez_initial_money (X : ℝ) (h : X - (X / 2 + 50) = 25) : X = 150 :=
by
  sorry

end NUMINAMATH_GPT_Inez_initial_money_l1428_142804


namespace NUMINAMATH_GPT_find_positive_number_l1428_142887

theorem find_positive_number 
  (x : ℝ) (h_pos : x > 0) 
  (h_eq : (2 / 3) * x = (16 / 216) * (1 / x)) : 
  x = 1 / 3 :=
by
  -- This is indicating that we're skipping the actual proof steps
  sorry

end NUMINAMATH_GPT_find_positive_number_l1428_142887
