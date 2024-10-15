import Mathlib

namespace NUMINAMATH_GPT_find_general_term_l1028_102894

variable (a : ℕ → ℝ) (a1 : a 1 = 1)

def isGeometricSequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

def isArithmeticSequence (u v w : ℝ) :=
  2 * v = u + w

theorem find_general_term (h1 : a 1 = 1)
  (h2 : (isGeometricSequence a (1 / 2)))
  (h3 : isArithmeticSequence (1 / a 1) (1 / a 3) (1 / a 4 - 1)) :
  ∀ n, a n = (1 / 2) ^ (n - 1) :=
sorry

end NUMINAMATH_GPT_find_general_term_l1028_102894


namespace NUMINAMATH_GPT_cyclists_meet_time_l1028_102878

/-- 
  Two cyclists start on a circular track from a given point but in opposite directions with speeds of 7 m/s and 8 m/s.
  The circumference of the circle is 180 meters.
  After what time will they meet at the starting point? 
-/
theorem cyclists_meet_time :
  let speed1 := 7 -- m/s
  let speed2 := 8 -- m/s
  let circumference := 180 -- meters
  (circumference / (speed1 + speed2) = 12) :=
by
  let speed1 := 7 -- m/s
  let speed2 := 8 -- m/s
  let circumference := 180 -- meters
  sorry

end NUMINAMATH_GPT_cyclists_meet_time_l1028_102878


namespace NUMINAMATH_GPT_complex_ratio_max_min_diff_l1028_102859

noncomputable def max_minus_min_complex_ratio (z w : ℂ) : ℝ :=
max (1 : ℝ) (0 : ℝ) - min (1 : ℝ) (0 : ℝ)

theorem complex_ratio_max_min_diff (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) : 
  max_minus_min_complex_ratio z w = 1 :=
by sorry

end NUMINAMATH_GPT_complex_ratio_max_min_diff_l1028_102859


namespace NUMINAMATH_GPT_find_k_l1028_102876

theorem find_k (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 12 = 0 → ∃ y : ℝ, y = x + 3 ∧ y^2 - k * y + 12 = 0) →
  k = 3 :=
sorry

end NUMINAMATH_GPT_find_k_l1028_102876


namespace NUMINAMATH_GPT_find_a1_l1028_102813

variable (a : ℕ → ℕ)
variable (q : ℕ)
variable (h_q_pos : 0 < q)
variable (h_a2a6 : a 2 * a 6 = 8 * a 4)
variable (h_a2 : a 2 = 2)

theorem find_a1 :
  a 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_l1028_102813


namespace NUMINAMATH_GPT_min_shirts_to_save_money_l1028_102846

theorem min_shirts_to_save_money :
  let acme_cost (x : ℕ) := 75 + 12 * x
  let gamma_cost (x : ℕ) := 18 * x
  ∀ x : ℕ, acme_cost x < gamma_cost x → x ≥ 13 := 
by
  intros
  sorry

end NUMINAMATH_GPT_min_shirts_to_save_money_l1028_102846


namespace NUMINAMATH_GPT_ratio_Cheryl_C_to_Cyrus_Y_l1028_102850

noncomputable def Cheryl_C : ℕ := 126
noncomputable def Madeline_M : ℕ := 63
noncomputable def Total_pencils : ℕ := 231
noncomputable def Cyrus_Y : ℕ := Total_pencils - Cheryl_C - Madeline_M

theorem ratio_Cheryl_C_to_Cyrus_Y : 
  Cheryl_C = 2 * Madeline_M → 
  Madeline_M + Cheryl_C + Cyrus_Y = Total_pencils → 
  Cheryl_C / Cyrus_Y = 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_ratio_Cheryl_C_to_Cyrus_Y_l1028_102850


namespace NUMINAMATH_GPT_find_b_l1028_102898

theorem find_b (b : ℝ) (h : ∃ x : ℝ, x^2 + b*x - 35 = 0 ∧ x = -5) : b = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1028_102898


namespace NUMINAMATH_GPT_total_recruits_211_l1028_102893

theorem total_recruits_211 (P N D : ℕ) (total : ℕ) 
  (h1 : P = 50) 
  (h2 : N = 100) 
  (h3 : D = 170) 
  (h4 : ∃ (x y : ℕ), (x = 4 * y ∨ y = 4 * x) ∧ 
                      ((x, P) = (y, N) ∨ (x, N) = (y, D) ∨ (x, P) = (y, D))) :
  total = 211 :=
by
  sorry

end NUMINAMATH_GPT_total_recruits_211_l1028_102893


namespace NUMINAMATH_GPT_marble_ratio_l1028_102801

-- Definitions based on conditions
def dan_marbles : ℕ := 5
def mary_marbles : ℕ := 10

-- Statement of the theorem to prove the ratio is 2:1
theorem marble_ratio : mary_marbles / dan_marbles = 2 := by
  sorry

end NUMINAMATH_GPT_marble_ratio_l1028_102801


namespace NUMINAMATH_GPT_sum_of_all_four_is_zero_l1028_102828

variables {a b c d : ℤ}

theorem sum_of_all_four_is_zero (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum_rows : a + b = c + d) 
  (h_product_columns : a * c = b * d) :
  a + b + c + d = 0 := 
sorry

end NUMINAMATH_GPT_sum_of_all_four_is_zero_l1028_102828


namespace NUMINAMATH_GPT_line_through_intersection_and_origin_l1028_102804

-- Definitions of the lines
def l1 (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def l2 (x y : ℝ) : Prop := y = 1 - x

-- Prove that the line passing through the intersection of l1 and l2 and the origin has the equation 3x + 2y = 0
theorem line_through_intersection_and_origin (x y : ℝ) 
  (h1 : 2 * x - y + 7 = 0) (h2 : y = 1 - x) : 3 * x + 2 * y = 0 := 
sorry

end NUMINAMATH_GPT_line_through_intersection_and_origin_l1028_102804


namespace NUMINAMATH_GPT_cost_of_one_pack_of_gummy_bears_l1028_102888

theorem cost_of_one_pack_of_gummy_bears
    (num_chocolate_bars : ℕ)
    (num_gummy_bears : ℕ)
    (num_chocolate_chips : ℕ)
    (total_cost : ℕ)
    (cost_per_chocolate_bar : ℕ)
    (cost_per_chocolate_chip : ℕ)
    (cost_of_one_gummy_bear_pack : ℕ)
    (h1 : num_chocolate_bars = 10)
    (h2 : num_gummy_bears = 10)
    (h3 : num_chocolate_chips = 20)
    (h4 : total_cost = 150)
    (h5 : cost_per_chocolate_bar = 3)
    (h6 : cost_per_chocolate_chip = 5)
    (h7 : num_chocolate_bars * cost_per_chocolate_bar +
          num_gummy_bears * cost_of_one_gummy_bear_pack +
          num_chocolate_chips * cost_per_chocolate_chip = total_cost) :
    cost_of_one_gummy_bear_pack = 2 := by
  sorry

end NUMINAMATH_GPT_cost_of_one_pack_of_gummy_bears_l1028_102888


namespace NUMINAMATH_GPT_probability_snow_at_least_once_l1028_102892

-- Defining the probability of no snow on the first five days
def no_snow_first_five_days : ℚ := (4 / 5) ^ 5

-- Defining the probability of no snow on the next five days
def no_snow_next_five_days : ℚ := (2 / 3) ^ 5

-- Total probability of no snow during the first ten days
def no_snow_first_ten_days : ℚ := no_snow_first_five_days * no_snow_next_five_days

-- Probability of snow at least once during the first ten days
def snow_at_least_once_first_ten_days : ℚ := 1 - no_snow_first_ten_days

-- Desired proof statement
theorem probability_snow_at_least_once :
  snow_at_least_once_first_ten_days = 726607 / 759375 := by
  sorry

end NUMINAMATH_GPT_probability_snow_at_least_once_l1028_102892


namespace NUMINAMATH_GPT_total_cost_l1028_102879

variable (a b : ℝ)

theorem total_cost (ha : a ≥ 0) (hb : b ≥ 0) : 3 * a + 4 * b = 3 * a + 4 * b :=
by sorry

end NUMINAMATH_GPT_total_cost_l1028_102879


namespace NUMINAMATH_GPT_find_weight_of_a_l1028_102811

-- Define the weights
variables (a b c d e : ℝ)

-- Given conditions
def condition1 := (a + b + c) / 3 = 50
def condition2 := (a + b + c + d) / 4 = 53
def condition3 := (b + c + d + e) / 4 = 51
def condition4 := e = d + 3

-- Proof goal
theorem find_weight_of_a : condition1 a b c → condition2 a b c d → condition3 b c d e → condition4 d e → a = 73 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_weight_of_a_l1028_102811


namespace NUMINAMATH_GPT_solve_equation_l1028_102823

theorem solve_equation (x : ℝ) : 2 * x - 4 = 0 ↔ x = 2 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1028_102823


namespace NUMINAMATH_GPT_initial_minutes_planA_equivalence_l1028_102899

-- Conditions translated into Lean:
variable (x : ℝ)

-- Definitions for costs
def planA_cost_12 : ℝ := 0.60 + 0.06 * (12 - x)
def planB_cost_12 : ℝ := 0.08 * 12

-- Theorem we want to prove
theorem initial_minutes_planA_equivalence :
  (planA_cost_12 x = planB_cost_12) → x = 6 :=
by
  intro h
  -- complete proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_initial_minutes_planA_equivalence_l1028_102899


namespace NUMINAMATH_GPT_degree_measure_of_regular_hexagon_interior_angle_l1028_102873

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end NUMINAMATH_GPT_degree_measure_of_regular_hexagon_interior_angle_l1028_102873


namespace NUMINAMATH_GPT_total_apples_packed_correct_l1028_102820

-- Define the daily production of apples under normal conditions
def apples_per_box := 40
def boxes_per_day := 50
def days_per_week := 7
def apples_per_day := apples_per_box * boxes_per_day

-- Define the change in daily production for the next week
def fewer_apples := 500
def apples_per_day_next_week := apples_per_day - fewer_apples

-- Define the weekly production in normal and next conditions
def apples_first_week := apples_per_day * days_per_week
def apples_second_week := apples_per_day_next_week * days_per_week

-- Define the total apples packed in two weeks
def total_apples_packed := apples_first_week + apples_second_week

-- Prove the total apples packed is 24500
theorem total_apples_packed_correct : total_apples_packed = 24500 := by
  sorry

end NUMINAMATH_GPT_total_apples_packed_correct_l1028_102820


namespace NUMINAMATH_GPT_equivalent_region_l1028_102858

def satisfies_conditions (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 2 ∧ -1 ≤ x / (x + y) ∧ x / (x + y) ≤ 1

def region (x y : ℝ) : Prop :=
  y ≥ 0 ∧ y ≥ -2*x ∧ x^2 + y^2 ≤ 2

theorem equivalent_region (x y : ℝ) :
  satisfies_conditions x y = region x y := 
sorry

end NUMINAMATH_GPT_equivalent_region_l1028_102858


namespace NUMINAMATH_GPT_find_ratio_l1028_102802

theorem find_ratio (a b : ℝ) (h : a / b = 3 / 2) : (a + b) / a = 5 / 3 :=
sorry

end NUMINAMATH_GPT_find_ratio_l1028_102802


namespace NUMINAMATH_GPT_part1_part2_l1028_102874

theorem part1 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (hA : A + B + C = π) 
  (ha : a = 2) 
  (hcosC : Real.cos C = -1 / 4) 
  (hsinA_sinB : Real.sin A = 2 * Real.sin B) : b = 1 ∧ c = Real.sqrt 6 := 
  sorry

theorem part2
  (A B C : ℝ) 
  (a b c : ℝ) 
  (hA : A + B + C = π) 
  (ha : a = 2) 
  (hcosC : Real.cos C = -1 / 4)
  (hcosA_minus_pi_div_4 : Real.cos (A - π / 4) = 4 / 5) : c = 5 * Real.sqrt 30 / 2 := 
  sorry

end NUMINAMATH_GPT_part1_part2_l1028_102874


namespace NUMINAMATH_GPT_sqrt_x_minus_1_meaningful_l1028_102875

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) := by
  sorry

end NUMINAMATH_GPT_sqrt_x_minus_1_meaningful_l1028_102875


namespace NUMINAMATH_GPT_remainder_of_a_sq_plus_five_mod_seven_l1028_102887

theorem remainder_of_a_sq_plus_five_mod_seven (a : ℕ) (h : a % 7 = 4) : (a^2 + 5) % 7 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_a_sq_plus_five_mod_seven_l1028_102887


namespace NUMINAMATH_GPT_unknown_number_value_l1028_102853

theorem unknown_number_value (a : ℕ) (n : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 3 = 21 * n * 45 * 49) : 
  n = 75 :=
sorry

end NUMINAMATH_GPT_unknown_number_value_l1028_102853


namespace NUMINAMATH_GPT_smallest_sum_97_l1028_102803

theorem smallest_sum_97 (X Y Z W : ℕ) 
  (h1 : X + Y + Z = 3)
  (h2 : 4 * Z = 7 * Y)
  (h3 : 16 ∣ Y) : 
  X + Y + Z + W = 97 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_97_l1028_102803


namespace NUMINAMATH_GPT_theta1_gt_theta2_l1028_102819

theorem theta1_gt_theta2 (a : ℝ) (b : ℝ) (θ1 θ2 : ℝ)
  (h_range_θ1 : 0 ≤ θ1 ∧ θ1 ≤ π) (h_range_θ2 : 0 ≤ θ2 ∧ θ2 ≤ π)
  (x1 x2 : ℝ) (hx1 : x1 = a * Real.cos θ1) (hx2 : x2 = a * Real.cos θ2)
  (h_less : x1 < x2) : θ1 > θ2 :=
by
  sorry

end NUMINAMATH_GPT_theta1_gt_theta2_l1028_102819


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1028_102844

theorem solve_eq1 {x : ℝ} : 2 * x^2 - 1 = 49 ↔ x = 5 ∨ x = -5 := 
  sorry

theorem solve_eq2 {x : ℝ} : (x + 3)^3 = 64 ↔ x = 1 := 
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1028_102844


namespace NUMINAMATH_GPT_remainder_of_expression_l1028_102808

theorem remainder_of_expression :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by {
  -- Prove the expression step by step
  -- sorry
  sorry
}

end NUMINAMATH_GPT_remainder_of_expression_l1028_102808


namespace NUMINAMATH_GPT_decimal_representation_l1028_102826

theorem decimal_representation :
  (13 : ℝ) / (2 * 5^8) = 0.00001664 := 
  sorry

end NUMINAMATH_GPT_decimal_representation_l1028_102826


namespace NUMINAMATH_GPT_shooting_accuracy_l1028_102805

theorem shooting_accuracy (S : ℕ → ℕ) (H1 : ∀ n, S n < 10 * n / 9) (H2 : ∀ n, S n > 10 * n / 9) :
  ∃ n, 10 * (S n) = 9 * n :=
by
  sorry

end NUMINAMATH_GPT_shooting_accuracy_l1028_102805


namespace NUMINAMATH_GPT_find_min_value_c_l1028_102838

theorem find_min_value_c (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 2010) :
  (∃ x y : ℤ, 3 * x + y = 3005 ∧ y = abs (x - a) + abs (x - 2 * b) + abs (x - c) ∧
   (∀ x' y' : ℤ, 3 * x' + y' = 3005 → y' = abs (x' - a) + abs (x' - 2 * b) + abs (x' - c) → x = x' ∧ y = y')) →
  c ≥ 1014 :=
by
  sorry

end NUMINAMATH_GPT_find_min_value_c_l1028_102838


namespace NUMINAMATH_GPT_problem_statement_l1028_102869

variable (x : ℝ) (x₀ : ℝ)

def p : Prop := ∀ x > 0, x + 4 / x ≥ 4

def q : Prop := ∃ x₀ ∈ Set.Ioi (0 : ℝ), 2 * x₀ = 1 / 2

theorem problem_statement : p ∧ ¬q :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1028_102869


namespace NUMINAMATH_GPT_flagpole_height_l1028_102861

theorem flagpole_height (h : ℕ)
  (shadow_flagpole : ℕ := 72)
  (height_pole : ℕ := 18)
  (shadow_pole : ℕ := 27)
  (ratio_shadow : shadow_flagpole / shadow_pole = 8 / 3) :
  h = 48 :=
by
  sorry

end NUMINAMATH_GPT_flagpole_height_l1028_102861


namespace NUMINAMATH_GPT_number_of_possible_values_l1028_102856

theorem number_of_possible_values (x : ℕ) (h1 : x > 6) (h2 : x + 4 > 0) :
  ∃ (n : ℕ), n = 24 := 
sorry

end NUMINAMATH_GPT_number_of_possible_values_l1028_102856


namespace NUMINAMATH_GPT_largest_marbles_l1028_102847

theorem largest_marbles {n : ℕ} (h1 : n < 400) (h2 : n % 3 = 1) (h3 : n % 7 = 2) (h4 : n % 5 = 0) : n = 310 :=
  sorry

end NUMINAMATH_GPT_largest_marbles_l1028_102847


namespace NUMINAMATH_GPT_arith_sign_change_geo_sign_change_l1028_102857

-- Definitions for sequences
def arith_sequence (a₁ d : ℝ) : ℕ → ℝ
| 0 => a₁
| (n + 1) => arith_sequence a₁ d n + d

def geo_sequence (a₁ r : ℝ) : ℕ → ℝ
| 0 => a₁
| (n + 1) => geo_sequence a₁ r n * r

-- Problem statement
theorem arith_sign_change :
  ∀ (a₁ d : ℝ), (∃ N : ℕ, arith_sequence a₁ d N = 0) ∨ (∀ n m : ℕ, (arith_sequence a₁ d n) * (arith_sequence a₁ d m) ≥ 0) :=
sorry

theorem geo_sign_change :
  ∀ (a₁ r : ℝ), r < 0 → ∀ n : ℕ, (geo_sequence a₁ r n) * (geo_sequence a₁ r (n + 1)) < 0 :=
sorry

end NUMINAMATH_GPT_arith_sign_change_geo_sign_change_l1028_102857


namespace NUMINAMATH_GPT_people_in_circle_l1028_102872

theorem people_in_circle (n : ℕ) (h : ∃ k : ℕ, k * 2 + 7 = 18) : n = 22 :=
by
  sorry

end NUMINAMATH_GPT_people_in_circle_l1028_102872


namespace NUMINAMATH_GPT_product_y_coordinates_l1028_102842

theorem product_y_coordinates : 
  ∀ y : ℝ, (∀ P : ℝ × ℝ, P.1 = -1 ∧ (P.1 - 4)^2 + (P.2 - 3)^2 = 64 → P = (-1, y)) →
  ((3 + Real.sqrt 39) * (3 - Real.sqrt 39) = -30) :=
by
  intros y h
  sorry

end NUMINAMATH_GPT_product_y_coordinates_l1028_102842


namespace NUMINAMATH_GPT_largest_integer_x_l1028_102815

theorem largest_integer_x (x : ℤ) : 
  (0.2 : ℝ) < (x : ℝ) / 7 ∧ (x : ℝ) / 7 < (7 : ℝ) / 12 → x = 4 :=
sorry

end NUMINAMATH_GPT_largest_integer_x_l1028_102815


namespace NUMINAMATH_GPT_rectangle_length_l1028_102821

theorem rectangle_length (sq_side_len rect_width : ℕ) (sq_area : ℕ) (rect_len : ℕ) 
    (h1 : sq_side_len = 6) 
    (h2 : rect_width = 4) 
    (h3 : sq_area = sq_side_len * sq_side_len) 
    (h4 : sq_area = rect_width * rect_len) :
    rect_len = 9 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_length_l1028_102821


namespace NUMINAMATH_GPT_solve_for_a_l1028_102835

theorem solve_for_a (x : ℤ) (a : ℤ) (h : 3 * x + 2 * a + 1 = 2) (hx : x = -1) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1028_102835


namespace NUMINAMATH_GPT_fraction_product_sum_l1028_102854

theorem fraction_product_sum :
  (1/3) * (5/6) * (3/7) + (1/4) * (1/8) = 101/672 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_sum_l1028_102854


namespace NUMINAMATH_GPT_line_tangent_to_circle_l1028_102841

theorem line_tangent_to_circle (r : ℝ) :
  (∀ (x y : ℝ), (x + y = 4) → (x - 2)^2 + (y + 1)^2 = r) → r = 9 / 2 :=
sorry

end NUMINAMATH_GPT_line_tangent_to_circle_l1028_102841


namespace NUMINAMATH_GPT_find_scalars_l1028_102852

def M : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 7], ![-3, -1]]
def M_squared : Matrix (Fin 2) (Fin 2) ℤ := ![![-17, 7], ![-3, -20]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem find_scalars :
  ∃ p q : ℤ, M_squared = p • M + q • I ∧ (p, q) = (1, -19) := sorry

end NUMINAMATH_GPT_find_scalars_l1028_102852


namespace NUMINAMATH_GPT_even_function_a_value_l1028_102816

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x^2 + (a - 2) * x + a^2 - a - 2 = (a + 1) * x^2 - (a - 2) * x + a^2 - a - 2) → a = 2 := 
by sorry

end NUMINAMATH_GPT_even_function_a_value_l1028_102816


namespace NUMINAMATH_GPT_inverse_proposition_l1028_102877

theorem inverse_proposition :
  (∀ x : ℝ, x < 0 → x^2 > 0) → (∀ y : ℝ, y^2 > 0 → y < 0) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proposition_l1028_102877


namespace NUMINAMATH_GPT_find_k_l1028_102824

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (2, -3)

def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_k (k : ℝ) :
  is_perpendicular (k • vector_a - 2 • vector_b) vector_a ↔ k = -1 :=
sorry

end NUMINAMATH_GPT_find_k_l1028_102824


namespace NUMINAMATH_GPT_distance_from_P_to_origin_l1028_102834

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_from_P_to_origin :
  distance (-1) 2 0 0 = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_P_to_origin_l1028_102834


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1028_102812

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 / 8) → (∀ x : ℝ, x > 0 → 2 * x + a / x ≥ 1) :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1028_102812


namespace NUMINAMATH_GPT_correct_first_coupon_day_l1028_102851

def is_redemption_valid (start_day : ℕ) (interval : ℕ) (num_coupons : ℕ) (closed_day : ℕ) : Prop :=
  ∀ n : ℕ, n < num_coupons → (start_day + n * interval) % 7 ≠ closed_day

def wednesday : ℕ := 3  -- Assuming Sunday = 0, Monday = 1, ..., Saturday = 6

theorem correct_first_coupon_day : 
  is_redemption_valid wednesday 10 6 0 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_correct_first_coupon_day_l1028_102851


namespace NUMINAMATH_GPT_dress_cost_l1028_102896

theorem dress_cost (x : ℝ) 
  (h1 : 30 * x = 10 + x) 
  (h2 : 3 * ((10 + x) / 30) = x) : 
  x = 10 / 9 :=
by
  sorry

end NUMINAMATH_GPT_dress_cost_l1028_102896


namespace NUMINAMATH_GPT_initial_average_daily_production_l1028_102866

variable (A : ℝ) -- Initial average daily production
variable (n : ℕ) -- Number of days

theorem initial_average_daily_production (n_eq_5 : n = 5) (new_production_eq_90 : 90 = 90) 
  (new_average_eq_65 : (5 * A + 90) / 6 = 65) : A = 60 :=
by
  sorry

end NUMINAMATH_GPT_initial_average_daily_production_l1028_102866


namespace NUMINAMATH_GPT_line_parameterization_l1028_102883

theorem line_parameterization (r k : ℝ) (t : ℝ) :
  (∀ x y : ℝ, (x, y) = (r + 3 * t, 2 + k * t) → (y = 2 * x - 5) ) ∧
  (t = 0 → r = 7 / 2) ∧
  (t = 1 → k = 6) :=
by
  sorry

end NUMINAMATH_GPT_line_parameterization_l1028_102883


namespace NUMINAMATH_GPT_find_x_l1028_102889

theorem find_x (x : ℝ) (h_pos : x > 0) (h_eq : x * (⌊x⌋) = 132) : x = 12 := sorry

end NUMINAMATH_GPT_find_x_l1028_102889


namespace NUMINAMATH_GPT_robin_gum_packages_l1028_102806

theorem robin_gum_packages (P : ℕ) (h1 : 7 * P + 6 = 41) : P = 5 :=
by
  sorry

end NUMINAMATH_GPT_robin_gum_packages_l1028_102806


namespace NUMINAMATH_GPT_range_of_a_l1028_102864

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = x^2 - 2 * a * x + a^2 - 1) 
(h_sol : ∀ x, f (f x) ≥ 0) : a ≤ -2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1028_102864


namespace NUMINAMATH_GPT_wendy_total_sales_correct_l1028_102865

noncomputable def wendy_total_sales : ℝ :=
  let morning_apples := 40 * 1.50
  let morning_oranges := 30 * 1
  let morning_bananas := 10 * 0.75
  let afternoon_apples := 50 * 1.35
  let afternoon_oranges := 40 * 0.90
  let afternoon_bananas := 20 * 0.675
  let unsold_bananas := 20 * 0.375
  let unsold_oranges := 10 * 0.50
  let total_morning := morning_apples + morning_oranges + morning_bananas
  let total_afternoon := afternoon_apples + afternoon_oranges + afternoon_bananas
  let total_day_sales := total_morning + total_afternoon
  let total_unsold_sales := unsold_bananas + unsold_oranges
  total_day_sales + total_unsold_sales

theorem wendy_total_sales_correct :
  wendy_total_sales = 227 := by
  unfold wendy_total_sales
  sorry

end NUMINAMATH_GPT_wendy_total_sales_correct_l1028_102865


namespace NUMINAMATH_GPT_compute_x_y_sum_l1028_102885

theorem compute_x_y_sum (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^4 + (Real.log y / Real.log 3)^4 + 8 = 8 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^Real.sqrt 2 + y^Real.sqrt 2 = 13 :=
by
  sorry

end NUMINAMATH_GPT_compute_x_y_sum_l1028_102885


namespace NUMINAMATH_GPT_find_p_minus_q_l1028_102891

theorem find_p_minus_q (p q : ℝ) (h : ∀ x, x^2 - 6 * x + q = 0 ↔ (x - p)^2 = 7) : p - q = 1 :=
sorry

end NUMINAMATH_GPT_find_p_minus_q_l1028_102891


namespace NUMINAMATH_GPT_exists_good_pair_for_each_m_l1028_102867

def is_good_pair (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), m * n = a^2 ∧ (m + 1) * (n + 1) = b^2

theorem exists_good_pair_for_each_m : ∀ m : ℕ, ∃ n : ℕ, m < n ∧ is_good_pair m n := by
  intro m
  let n := m * (4 * m + 3)^2
  use n
  have h1 : m < n := sorry -- Proof that m < n
  have h2 : is_good_pair m n := sorry -- Proof that (m, n) is a good pair
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_exists_good_pair_for_each_m_l1028_102867


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_seq_l1028_102836

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * d

theorem common_difference_of_arithmetic_seq :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  arithmetic_sequence a d →
  (a 4 + a 8 = 10) →
  (a 10 = 6) →
  d = 1 / 4 :=
by
  intros a d h_seq h1 h2
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_seq_l1028_102836


namespace NUMINAMATH_GPT_find_a_plus_d_l1028_102822

theorem find_a_plus_d (a b c d : ℝ) (h₁ : ab + bc + ca + db = 42) (h₂ : b + c = 6) : a + d = 7 := 
sorry

end NUMINAMATH_GPT_find_a_plus_d_l1028_102822


namespace NUMINAMATH_GPT_find_cost_price_l1028_102817

-- Conditions
def initial_cost_price (C : ℝ) : Prop :=
  let SP := 1.07 * C
  let NCP := 0.92 * C
  let NSP := SP - 3
  NSP = 1.0304 * C

-- The problem is to prove the initial cost price C given the conditions
theorem find_cost_price (C : ℝ) (h : initial_cost_price C) : C = 75.7575 := 
  sorry

end NUMINAMATH_GPT_find_cost_price_l1028_102817


namespace NUMINAMATH_GPT_socks_problem_l1028_102868

/-
  Theorem: Given x + y + z = 15, 2x + 4y + 5z = 36, and x, y, z ≥ 1, 
  the number of $2 socks Jack bought is x = 4.
-/

theorem socks_problem
  (x y z : ℕ)
  (h1 : x + y + z = 15)
  (h2 : 2 * x + 4 * y + 5 * z = 36)
  (h3 : 1 ≤ x)
  (h4 : 1 ≤ y)
  (h5 : 1 ≤ z) :
  x = 4 :=
  sorry

end NUMINAMATH_GPT_socks_problem_l1028_102868


namespace NUMINAMATH_GPT_find_x_l1028_102840

theorem find_x (p q r s x : ℚ) (hpq : p ≠ q) (hq0 : q ≠ 0) 
    (h : (p + x) / (q - x) = r / s) 
    (hp : p = 3) (hq : q = 5) (hr : r = 7) (hs : s = 9) : 
    x = 1/2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_l1028_102840


namespace NUMINAMATH_GPT_larger_number_is_299_l1028_102843

theorem larger_number_is_299 (A B : ℕ) 
  (HCF_AB : Nat.gcd A B = 23) 
  (LCM_12_13 : Nat.lcm A B = 23 * 12 * 13) : 
  max A B = 299 := 
sorry

end NUMINAMATH_GPT_larger_number_is_299_l1028_102843


namespace NUMINAMATH_GPT_math_problem_l1028_102833

def a : ℕ := 2013
def b : ℕ := 2014

theorem math_problem :
  (a^3 - 2 * a^2 * b + 3 * a * b^2 - b^3 + 1) / (a * b) = a := by
  sorry

end NUMINAMATH_GPT_math_problem_l1028_102833


namespace NUMINAMATH_GPT_average_weight_of_24_boys_l1028_102818

theorem average_weight_of_24_boys (A : ℝ) : 
  (24 * A + 8 * 45.15) / 32 = 48.975 → A = 50.25 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_average_weight_of_24_boys_l1028_102818


namespace NUMINAMATH_GPT_add_in_base8_l1028_102863

def base8_add (a b : ℕ) (n : ℕ): ℕ :=
  a * (8 ^ n) + b

theorem add_in_base8 : base8_add 123 56 0 = 202 := by
  sorry

end NUMINAMATH_GPT_add_in_base8_l1028_102863


namespace NUMINAMATH_GPT_stuart_initial_marbles_l1028_102848

theorem stuart_initial_marbles
    (betty_marbles : ℕ)
    (stuart_marbles_after_given : ℕ)
    (percentage_given : ℚ)
    (betty_gave : ℕ):
    betty_marbles = 60 →
    stuart_marbles_after_given = 80 →
    percentage_given = 0.40 →
    betty_gave = percentage_given * betty_marbles →
    stuart_marbles_after_given = stuart_initial + betty_gave →
    stuart_initial = 56 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_stuart_initial_marbles_l1028_102848


namespace NUMINAMATH_GPT_proof_problem_l1028_102825

variable {a b c d e f : ℝ}

theorem proof_problem :
  (a * b * c = 130) →
  (b * c * d = 65) →
  (d * e * f = 250) →
  (a * f / (c * d) = 0.5) →
  (c * d * e = 1000) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_proof_problem_l1028_102825


namespace NUMINAMATH_GPT_part1_optimal_strategy_part2_optimal_strategy_l1028_102809

noncomputable def R (x1 x2 : ℝ) : ℝ := -2 * x1^2 - x2^2 + 13 * x1 + 11 * x2 - 28

theorem part1_optimal_strategy :
  ∃ x1 x2 : ℝ, x1 + x2 = 5 ∧ x1 = 2 ∧ x2 = 3 ∧
    ∀ y1 y2, y1 + y2 = 5 → (R y1 y2 - (y1 + y2) ≤ R x1 x2 - (x1 + x2)) := 
by
  sorry

theorem part2_optimal_strategy :
  ∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = 5 ∧
    ∀ y1 y2, (R y1 y2 - (y1 + y2) ≤ R x1 x2 - (x1 + x2)) := 
by
  sorry

end NUMINAMATH_GPT_part1_optimal_strategy_part2_optimal_strategy_l1028_102809


namespace NUMINAMATH_GPT_factor_expression_l1028_102827

theorem factor_expression (x : ℝ) : 
  72 * x^2 + 108 * x + 36 = 36 * (2 * x^2 + 3 * x + 1) :=
sorry

end NUMINAMATH_GPT_factor_expression_l1028_102827


namespace NUMINAMATH_GPT_factor_x4_minus_81_l1028_102800

theorem factor_x4_minus_81 : ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factor_x4_minus_81_l1028_102800


namespace NUMINAMATH_GPT_cyclist_C_speed_l1028_102870

variables (c d : ℕ) -- Speeds of cyclists C and D in mph
variables (d_eq : d = c + 6) -- Cyclist D travels 6 mph faster than cyclist C
variables (h1 : 80 = 65 + 15) -- Total distance from X to Y and back to the meet point
variables (same_time : 65 / c = 95 / d) -- Equating the travel times of both cyclists

theorem cyclist_C_speed : c = 13 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_cyclist_C_speed_l1028_102870


namespace NUMINAMATH_GPT_smallest_four_digit_in_pascals_triangle_l1028_102849

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_in_pascals_triangle_l1028_102849


namespace NUMINAMATH_GPT_problem_l1028_102880

open Real

theorem problem (x y : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_cond : x + y^(2016) ≥ 1) : 
  x^(2016) + y > 1 - 1/100 :=
by sorry

end NUMINAMATH_GPT_problem_l1028_102880


namespace NUMINAMATH_GPT_simplify_expression_l1028_102830

theorem simplify_expression : 
  1.5 * (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9)) = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1028_102830


namespace NUMINAMATH_GPT_tan3theta_l1028_102807

theorem tan3theta (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 := 
by
  sorry

end NUMINAMATH_GPT_tan3theta_l1028_102807


namespace NUMINAMATH_GPT_evaluate_expression_l1028_102839

theorem evaluate_expression (a : ℕ) (h : a = 2) : (7 * a ^ 2 - 10 * a + 3) * (3 * a - 4) = 22 :=
by
  -- Here would be the proof which is omitted as per instructions
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1028_102839


namespace NUMINAMATH_GPT_range_of_f_l1028_102884

noncomputable def f (x : ℝ) : ℝ := - (2 / (x - 1))

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, (0 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ 2) ∧ f x = y} = 
  {y : ℝ | y ≤ -2 ∨ 2 ≤ y} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1028_102884


namespace NUMINAMATH_GPT_marbles_difference_l1028_102897

theorem marbles_difference : 10 - 8 = 2 :=
by
  sorry

end NUMINAMATH_GPT_marbles_difference_l1028_102897


namespace NUMINAMATH_GPT_prime_factor_condition_l1028_102871

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 1
  | n + 2 => seq (n + 1) + seq n

theorem prime_factor_condition (p k : ℕ) (hp : Nat.Prime p) (h : p ∣ seq (2 * k) - 2) :
  p ∣ seq (2 * k - 1) - 1 :=
sorry

end NUMINAMATH_GPT_prime_factor_condition_l1028_102871


namespace NUMINAMATH_GPT_div_36_of_n_ge_5_l1028_102814

noncomputable def n := Nat

theorem div_36_of_n_ge_5 (n : ℕ) (hn : n ≥ 5) (h2 : ¬ (n % 2 = 0)) (h3 : ¬ (n % 3 = 0)) : 36 ∣ (n^2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_div_36_of_n_ge_5_l1028_102814


namespace NUMINAMATH_GPT_initial_packs_l1028_102832

def num_invitations_per_pack := 3
def num_friends := 9
def extra_invitations := 3
def total_invitations := num_friends + extra_invitations

theorem initial_packs (h : total_invitations = 12) : (total_invitations / num_invitations_per_pack) = 4 :=
by
  have h1 : total_invitations = 12 := by exact h
  have h2 : num_invitations_per_pack = 3 := by exact rfl
  have H_pack : total_invitations / num_invitations_per_pack = 4 := by sorry
  exact H_pack

end NUMINAMATH_GPT_initial_packs_l1028_102832


namespace NUMINAMATH_GPT_unique_real_solution_for_cubic_l1028_102860

theorem unique_real_solution_for_cubic {b : ℝ} :
  (∀ x : ℝ, (x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0) → ∃! x : ℝ, (x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0)) ↔ b > 3 :=
sorry

end NUMINAMATH_GPT_unique_real_solution_for_cubic_l1028_102860


namespace NUMINAMATH_GPT_games_bought_l1028_102829

def initial_money : ℕ := 35
def spent_money : ℕ := 7
def cost_per_game : ℕ := 4

theorem games_bought : (initial_money - spent_money) / cost_per_game = 7 := by
  sorry

end NUMINAMATH_GPT_games_bought_l1028_102829


namespace NUMINAMATH_GPT_least_number_of_square_tiles_l1028_102845

theorem least_number_of_square_tiles (length : ℕ) (breadth : ℕ) (gcd : ℕ) (area_room : ℕ) (area_tile : ℕ) (num_tiles : ℕ) :
  length = 544 → breadth = 374 → gcd = Nat.gcd length breadth → gcd = 2 →
  area_room = length * breadth → area_tile = gcd * gcd →
  num_tiles = area_room / area_tile → num_tiles = 50864 :=
by
  sorry

end NUMINAMATH_GPT_least_number_of_square_tiles_l1028_102845


namespace NUMINAMATH_GPT_total_number_of_sweets_l1028_102890

theorem total_number_of_sweets (num_crates : ℕ) (sweets_per_crate : ℕ) (total_sweets : ℕ) 
  (h1 : num_crates = 4) (h2 : sweets_per_crate = 16) : total_sweets = 64 := by
  sorry

end NUMINAMATH_GPT_total_number_of_sweets_l1028_102890


namespace NUMINAMATH_GPT_proof_problem_l1028_102881

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (Real.exp (Real.log a * x) - Real.exp (-Real.log a * x))

theorem proof_problem (
  a : ℝ
) (h1 : a > 1) :
  (∀ x, f a x = (a / (a^2 - 1)) * (Real.exp (Real.log a * x) - Real.exp (-Real.log a * x))) ∧
  (∀ x, f a (-x) = -f a x) ∧
  (∀ x1 x2, x1 < x2 → f a x1 < f a x2) ∧
  (∀ m, -1 < 1 - m ∧ 1 - m < m^2 - 1 ∧ m^2 - 1 < 1 → 1 < m ∧ m < Real.sqrt 2)
  :=
sorry

end NUMINAMATH_GPT_proof_problem_l1028_102881


namespace NUMINAMATH_GPT_distribute_pencils_l1028_102837

variables {initial_pencils : ℕ} {num_containers : ℕ} {additional_pencils : ℕ}

theorem distribute_pencils (h₁ : initial_pencils = 150) (h₂ : num_containers = 5)
                           (h₃ : additional_pencils = 30) :
  (initial_pencils + additional_pencils) / num_containers = 36 :=
by sorry

end NUMINAMATH_GPT_distribute_pencils_l1028_102837


namespace NUMINAMATH_GPT_problem_solution_l1028_102886

variable (α : ℝ)
variable (h : Real.cos α = 1 / 5)

theorem problem_solution : Real.cos (2 * α - 2017 * Real.pi) = 23 / 25 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1028_102886


namespace NUMINAMATH_GPT_topsoil_cost_l1028_102855

theorem topsoil_cost (cost_per_cubic_foot : ℕ) (cubic_yard_to_cubic_foot : ℕ) (volume_in_cubic_yards : ℕ) :
  cost_per_cubic_foot = 8 →
  cubic_yard_to_cubic_foot = 27 →
  volume_in_cubic_yards = 3 →
  volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 648 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_topsoil_cost_l1028_102855


namespace NUMINAMATH_GPT_remainder_3_pow_2000_mod_17_l1028_102862

theorem remainder_3_pow_2000_mod_17 : (3^2000 % 17) = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_3_pow_2000_mod_17_l1028_102862


namespace NUMINAMATH_GPT_total_carrots_l1028_102831

theorem total_carrots (sally_carrots fred_carrots : ℕ) (h1 : sally_carrots = 6) (h2 : fred_carrots = 4) : sally_carrots + fred_carrots = 10 := by
  sorry

end NUMINAMATH_GPT_total_carrots_l1028_102831


namespace NUMINAMATH_GPT_complex_identity_l1028_102882

theorem complex_identity (α β : ℝ) (h : Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = Complex.mk (-1 / 3) (5 / 8)) :
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = Complex.mk (-1 / 3) (-5 / 8) :=
by
  sorry

end NUMINAMATH_GPT_complex_identity_l1028_102882


namespace NUMINAMATH_GPT_problem_statement_l1028_102895

noncomputable def f (x : ℚ) : ℚ := (x^2 - x - 6) / (x^3 - 2 * x^2 - x + 2)

def a : ℕ := 1  -- number of holes
def b : ℕ := 2  -- number of vertical asymptotes
def c : ℕ := 1  -- number of horizontal asymptotes
def d : ℕ := 0  -- number of oblique asymptotes

theorem problem_statement : a + 2 * b + 3 * c + 4 * d = 8 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1028_102895


namespace NUMINAMATH_GPT_comparison_of_neg_square_roots_l1028_102810

noncomputable def compare_square_roots : Prop :=
  -2 * Real.sqrt 11 > -3 * Real.sqrt 5

theorem comparison_of_neg_square_roots : compare_square_roots :=
by
  -- Omitting the proof details
  sorry

end NUMINAMATH_GPT_comparison_of_neg_square_roots_l1028_102810
