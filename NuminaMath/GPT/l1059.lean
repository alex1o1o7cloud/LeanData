import Mathlib

namespace NUMINAMATH_GPT_algebraic_expression_value_l1059_105933

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 4) : 2*x^2 - 6*x + 5 = 11 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1059_105933


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_value_l1059_105903

variable {a_n : ℕ → ℝ}

theorem arithmetic_sequence_a5_value
  (h : a_n 2 + a_n 8 = 15 - a_n 5) :
  a_n 5 = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_value_l1059_105903


namespace NUMINAMATH_GPT_arrange_3x3_grid_l1059_105963

-- Define the problem conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := ¬ is_odd n

-- Define the function to count the number of such arrangements
noncomputable def count_arrangements : ℕ :=
  6 * 3^6 * 4^3 + 9 * 3^4 * 4^5 + 4^9

-- State the main theorem
theorem arrange_3x3_grid (nums : ℕ → Prop) (table : ℕ → ℕ → ℕ) (h : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 7) :
  (∀ i, is_odd (table i 0 + table i 1 + table i 2)) ∧ (∀ j, is_odd (table 0 j + table 1 j + table 2 j)) →
  count_arrangements = 6 * 3^6 * 4^3 + 9 * 3^4 * 4^5 + 4^9 :=
by sorry

end NUMINAMATH_GPT_arrange_3x3_grid_l1059_105963


namespace NUMINAMATH_GPT_female_athletes_drawn_is_7_l1059_105977

-- Given conditions as definitions
def male_athletes := 64
def female_athletes := 56
def drawn_male_athletes := 8

-- The function that represents the equation in stratified sampling
def stratified_sampling_eq (x : Nat) : Prop :=
  (drawn_male_athletes : ℚ) / (male_athletes) = (x : ℚ) / (female_athletes)

-- The theorem which states that the solution to the problem is x = 7
theorem female_athletes_drawn_is_7 : ∃ x : Nat, stratified_sampling_eq x ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_female_athletes_drawn_is_7_l1059_105977


namespace NUMINAMATH_GPT_problem_statement_l1059_105990

theorem problem_statement (a b c d : ℝ) 
  (hab : a ≤ b)
  (hbc : b ≤ c)
  (hcd : c ≤ d)
  (hsum : a + b + c + d = 0)
  (hinv_sum : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1059_105990


namespace NUMINAMATH_GPT_manuscript_copy_cost_l1059_105911

theorem manuscript_copy_cost (total_cost : ℝ) (binding_cost : ℝ) (num_manuscripts : ℕ) (pages_per_manuscript : ℕ) (x : ℝ) :
  total_cost = 250 ∧ binding_cost = 5 ∧ num_manuscripts = 10 ∧ pages_per_manuscript = 400 →
  x = (total_cost - binding_cost * num_manuscripts) / (num_manuscripts * pages_per_manuscript) →
  x = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_manuscript_copy_cost_l1059_105911


namespace NUMINAMATH_GPT_largest_divisible_l1059_105981

theorem largest_divisible (n : ℕ) (h1 : n > 0) (h2 : (n^3 + 200) % (n - 8) = 0) : n = 5376 :=
by
  sorry

end NUMINAMATH_GPT_largest_divisible_l1059_105981


namespace NUMINAMATH_GPT_find_xy_l1059_105980

theorem find_xy (x y : ℝ) (h : x * (x + 2 * y) = x^2 + 10) : x * y = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l1059_105980


namespace NUMINAMATH_GPT_circle_center_line_distance_l1059_105967

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_line_distance_l1059_105967


namespace NUMINAMATH_GPT_repeating_decimals_fraction_l1059_105944

theorem repeating_decimals_fraction :
  (0.81:ℚ) / (0.36:ℚ) = 9 / 4 :=
by
  have h₁ : (0.81:ℚ) = 81 / 99 := sorry
  have h₂ : (0.36:ℚ) = 36 / 99 := sorry
  sorry

end NUMINAMATH_GPT_repeating_decimals_fraction_l1059_105944


namespace NUMINAMATH_GPT_distance_walked_is_4_point_6_l1059_105935

-- Define the number of blocks Sarah walked in each direction
def blocks_west : ℕ := 8
def blocks_south : ℕ := 15

-- Define the length of each block in miles
def block_length : ℚ := 1 / 5

-- Calculate the total number of blocks
def total_blocks : ℕ := blocks_west + blocks_south

-- Calculate the total distance walked in miles
def total_distance_walked : ℚ := total_blocks * block_length

-- Statement to prove the total distance walked is 4.6 miles
theorem distance_walked_is_4_point_6 : total_distance_walked = 4.6 := sorry

end NUMINAMATH_GPT_distance_walked_is_4_point_6_l1059_105935


namespace NUMINAMATH_GPT_inequality_proof_l1059_105907

theorem inequality_proof (a b c x y z : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : c > 0) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) 
(h7 : a * y + b * x = c) (h8 : c * x + a * z = b) 
(h9 : b * z + c * y = a) :
x / (1 - y * z) + y / (1 - z * x) + z / (1 - x * y) ≤ 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1059_105907


namespace NUMINAMATH_GPT_positive_integer_divisibility_by_3_l1059_105926

theorem positive_integer_divisibility_by_3 (n : ℕ) (h : 0 < n) :
  (n * 2^n + 1) % 3 = 0 ↔ n % 6 = 1 ∨ n % 6 = 2 := 
sorry

end NUMINAMATH_GPT_positive_integer_divisibility_by_3_l1059_105926


namespace NUMINAMATH_GPT_remaining_students_l1059_105983

def students_remaining (n1 n2 n_leaving1 n_leaving2 : Nat) : Nat :=
  (n1 * 4 - n_leaving1) + (n2 * 2 - n_leaving2)

theorem remaining_students :
  students_remaining 15 18 8 5 = 83 := 
by
  sorry

end NUMINAMATH_GPT_remaining_students_l1059_105983


namespace NUMINAMATH_GPT_parallel_line_plane_no_common_points_l1059_105993

noncomputable def line := Type
noncomputable def plane := Type

variable {l : line}
variable {α : plane}

-- Definitions for parallel lines and planes, and relations between lines and planes
def parallel_to_plane (l : line) (α : plane) : Prop := sorry -- Definition of line parallel to plane
def within_plane (m : line) (α : plane) : Prop := sorry -- Definition of line within plane
def no_common_points (l m : line) : Prop := sorry -- Definition of no common points between lines

theorem parallel_line_plane_no_common_points
  (h₁ : parallel_to_plane l α)
  (l2 : line)
  (h₂ : within_plane l2 α) :
  no_common_points l l2 :=
sorry

end NUMINAMATH_GPT_parallel_line_plane_no_common_points_l1059_105993


namespace NUMINAMATH_GPT_shortest_side_of_similar_triangle_l1059_105971

theorem shortest_side_of_similar_triangle (a b : ℕ) (c : ℝ) 
  (h1 : a = 24) (h2 : c = 25) (h3 : a^2 + b^2 = c^2)
  (scale_factor : ℝ) (shortest_side_first : ℝ) (hypo_second : ℝ)
  (h4 : scale_factor = 100 / 25) 
  (h5 : hypo_second = 100) 
  (h6 : b = 7) 
  : (shortest_side_first * scale_factor = 28) :=
by
  sorry

end NUMINAMATH_GPT_shortest_side_of_similar_triangle_l1059_105971


namespace NUMINAMATH_GPT_extra_fruits_l1059_105985

theorem extra_fruits (r g s : Nat) (hr : r = 42) (hg : g = 7) (hs : s = 9) : r + g - s = 40 :=
by
  sorry

end NUMINAMATH_GPT_extra_fruits_l1059_105985


namespace NUMINAMATH_GPT_total_action_figures_l1059_105955

def jerry_original_count : Nat := 4
def jerry_added_count : Nat := 6

theorem total_action_figures : jerry_original_count + jerry_added_count = 10 :=
by
  sorry

end NUMINAMATH_GPT_total_action_figures_l1059_105955


namespace NUMINAMATH_GPT_cheapest_third_company_l1059_105927

theorem cheapest_third_company (x : ℕ) :
  (120 + 18 * x ≥ 150 + 15 * x) ∧ (220 + 13 * x ≥ 150 + 15 * x) → 36 ≤ x :=
by
  intro h
  cases h with
  | intro h1 h2 =>
    sorry

end NUMINAMATH_GPT_cheapest_third_company_l1059_105927


namespace NUMINAMATH_GPT_evaluate_expression_l1059_105922

theorem evaluate_expression :
  8^6 * 27^6 * 8^15 * 27^15 = 216^21 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1059_105922


namespace NUMINAMATH_GPT_sum_of_roots_equation_l1059_105972

noncomputable def sum_of_roots (a b c : ℝ) : ℝ :=
  (-b) / a

theorem sum_of_roots_equation :
  let a := 3
  let b := -15
  let c := 20
  sum_of_roots a b c = 5 := 
  by {
    sorry
  }

end NUMINAMATH_GPT_sum_of_roots_equation_l1059_105972


namespace NUMINAMATH_GPT_Jerry_remaining_pages_l1059_105918

theorem Jerry_remaining_pages (total_pages : ℕ) (saturday_read : ℕ) (sunday_read : ℕ) (remaining_pages : ℕ) 
  (h1 : total_pages = 93) (h2 : saturday_read = 30) (h3 : sunday_read = 20) (h4 : remaining_pages = 43) : 
  total_pages - saturday_read - sunday_read = remaining_pages := 
by
  sorry

end NUMINAMATH_GPT_Jerry_remaining_pages_l1059_105918


namespace NUMINAMATH_GPT_find_S10_l1059_105906

noncomputable def S (n : ℕ) : ℤ := 2 * (-2 ^ (n - 1)) + 1

theorem find_S10 : S 10 = -1023 :=
by
  sorry

end NUMINAMATH_GPT_find_S10_l1059_105906


namespace NUMINAMATH_GPT_remainder_pow_700_eq_one_l1059_105997

theorem remainder_pow_700_eq_one (number : ℤ) (h : number ^ 700 % 100 = 1) : number ^ 700 % 100 = 1 :=
  by
  exact h

end NUMINAMATH_GPT_remainder_pow_700_eq_one_l1059_105997


namespace NUMINAMATH_GPT_arrangement_ways_l1059_105961

def num_ways_arrange_boys_girls : Nat :=
  let boys := 2
  let girls := 3
  let ways_girls := Nat.factorial girls
  let ways_boys := Nat.factorial boys
  ways_girls * ways_boys

theorem arrangement_ways : num_ways_arrange_boys_girls = 12 :=
  by
    sorry

end NUMINAMATH_GPT_arrangement_ways_l1059_105961


namespace NUMINAMATH_GPT_correct_number_l1059_105974

theorem correct_number : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  -- proof starts here
  sorry

end NUMINAMATH_GPT_correct_number_l1059_105974


namespace NUMINAMATH_GPT_find_int_solutions_l1059_105931

theorem find_int_solutions (x : ℤ) :
  (∃ p : ℤ, Prime p ∧ 2*x^2 - x - 36 = p^2) ↔ (x = 5 ∨ x = 13) := 
sorry

end NUMINAMATH_GPT_find_int_solutions_l1059_105931


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l1059_105924

theorem arithmetic_seq_sum (a : ℕ → ℤ) (a1 a7 a3 a5 : ℤ) (S7 : ℤ)
  (h1 : a1 = a 1) (h7 : a7 = a 7) (h3 : a3 = a 3) (h5 : a5 = a 5)
  (h_arith : ∀ n m, a (n + m) = a n + a m - a 0)
  (h_S7 : (7 * (a1 + a7)) / 2 = 14) :
  a3 + a5 = 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l1059_105924


namespace NUMINAMATH_GPT_son_l1059_105929

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 35)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 33 := 
by
  sorry

end NUMINAMATH_GPT_son_l1059_105929


namespace NUMINAMATH_GPT_find_number_l1059_105901

theorem find_number (n : ℝ) :
  (n + 2 * 1.5)^5 = (1 + 3 * 1.5)^4 → n = 0.72 :=
sorry

end NUMINAMATH_GPT_find_number_l1059_105901


namespace NUMINAMATH_GPT_max_servings_possible_l1059_105932

def number_of_servings
  (peppers cucumbers tomatoes : Nat) (brynza : Nat) : Nat :=
  min (peppers) (min (brynza / 75) (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings_possible :
  number_of_servings 60 117 116 4200 = 56 := 
by 
  -- sorry statement allows skipping the proof
  sorry

end NUMINAMATH_GPT_max_servings_possible_l1059_105932


namespace NUMINAMATH_GPT_best_regression_effect_l1059_105914

theorem best_regression_effect (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.36)
  (h2 : R2_2 = 0.95)
  (h3 : R2_3 = 0.74)
  (h4 : R2_4 = 0.81):
  max (max (max R2_1 R2_2) R2_3) R2_4 = 0.95 := by
  sorry

end NUMINAMATH_GPT_best_regression_effect_l1059_105914


namespace NUMINAMATH_GPT_value_of_x_yplusz_l1059_105956

theorem value_of_x_yplusz (x y z : ℝ) (h : x * (x + y + z) = x^2 + 12) : x * (y + z) = 12 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_yplusz_l1059_105956


namespace NUMINAMATH_GPT_female_democrats_ratio_l1059_105919

theorem female_democrats_ratio 
  (M F : ℕ) 
  (H1 : M + F = 660)
  (H2 : (1 / 3 : ℝ) * 660 = 220)
  (H3 : ∃ dem_males : ℕ, dem_males = (1 / 4 : ℝ) * M)
  (H4 : ∃ dem_females : ℕ, dem_females = 110) :
  110 / F = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_female_democrats_ratio_l1059_105919


namespace NUMINAMATH_GPT_total_number_of_candles_l1059_105996

theorem total_number_of_candles
  (candles_bedroom : ℕ)
  (candles_living_room : ℕ)
  (candles_donovan : ℕ)
  (h1 : candles_bedroom = 20)
  (h2 : candles_bedroom = 2 * candles_living_room)
  (h3 : candles_donovan = 20) :
  candles_bedroom + candles_living_room + candles_donovan = 50 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_candles_l1059_105996


namespace NUMINAMATH_GPT_range_of_a_l1059_105909

theorem range_of_a (x a : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) : -1 ≤ a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1059_105909


namespace NUMINAMATH_GPT_average_of_remaining_two_l1059_105941

theorem average_of_remaining_two (a1 a2 a3 a4 a5 a6 : ℝ)
    (h_avg6 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 3.95)
    (h_avg2_1 : (a1 + a2) / 2 = 3.4)
    (h_avg2_2 : (a3 + a4) / 2 = 3.85) :
    (a5 + a6) / 2 = 4.6 := 
sorry

end NUMINAMATH_GPT_average_of_remaining_two_l1059_105941


namespace NUMINAMATH_GPT_investment_C_l1059_105915

theorem investment_C (A_invest B_invest profit_A total_profit C_invest : ℕ)
  (hA_invest : A_invest = 6300) 
  (hB_invest : B_invest = 4200) 
  (h_profit_A : profit_A = 3900) 
  (h_total_profit : total_profit = 13000) 
  (h_proportional : profit_A / total_profit = A_invest / (A_invest + B_invest + C_invest)) :
  C_invest = 10500 := by
  sorry

end NUMINAMATH_GPT_investment_C_l1059_105915


namespace NUMINAMATH_GPT_find_f_of_odd_function_periodic_l1059_105917

noncomputable def arctan (x : ℝ) : ℝ := sorry

theorem find_f_of_odd_function_periodic (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_periodic : ∀ x k : ℤ, f x = f (x + 3 * k))
    (α : ℝ) (h_tan : Real.tan α = 3) :
  f (2015 * Real.sin (2 * (arctan 3))) = 0 :=
sorry

end NUMINAMATH_GPT_find_f_of_odd_function_periodic_l1059_105917


namespace NUMINAMATH_GPT_price_restoration_l1059_105910

theorem price_restoration {P : ℝ} (hP : P > 0) :
  (P - 0.85 * P) / (0.85 * P) * 100 = 17.65 :=
by
  sorry

end NUMINAMATH_GPT_price_restoration_l1059_105910


namespace NUMINAMATH_GPT_evaluate_expression_l1059_105943

theorem evaluate_expression : (Real.sqrt (Real.sqrt 5 ^ 4))^3 = 125 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1059_105943


namespace NUMINAMATH_GPT_sequence_x_2022_l1059_105989

theorem sequence_x_2022 :
  ∃ (x : ℕ → ℤ), x 1 = 1 ∧ x 2 = 1 ∧ x 3 = -1 ∧
  (∀ n, 4 ≤ n → x n = x (n-1) * x (n-3)) ∧ x 2022 = 1 := by
  sorry

end NUMINAMATH_GPT_sequence_x_2022_l1059_105989


namespace NUMINAMATH_GPT_expand_expression_l1059_105925

theorem expand_expression (x y : ℕ) : 
  (x + 15) * (3 * y + 20) = 3 * x * y + 20 * x + 45 * y + 300 :=
by 
  sorry

end NUMINAMATH_GPT_expand_expression_l1059_105925


namespace NUMINAMATH_GPT_tan_neg_5pi_over_4_l1059_105992

theorem tan_neg_5pi_over_4 : Real.tan (-5 * Real.pi / 4) = -1 :=
by 
  sorry

end NUMINAMATH_GPT_tan_neg_5pi_over_4_l1059_105992


namespace NUMINAMATH_GPT_no_natural_m_n_prime_l1059_105970

theorem no_natural_m_n_prime (m n : ℕ) : ¬Prime (n^2 + 2018 * m * n + 2019 * m + n - 2019 * m^2) :=
by
  sorry

end NUMINAMATH_GPT_no_natural_m_n_prime_l1059_105970


namespace NUMINAMATH_GPT_no_100_roads_l1059_105958

theorem no_100_roads (k : ℕ) (hk : 3 * k % 2 = 0) : 100 ≠ 3 * k / 2 := 
by
  sorry

end NUMINAMATH_GPT_no_100_roads_l1059_105958


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_l1059_105969

theorem arithmetic_sequence_solution
  (a b c : ℤ)
  (h1 : a + 1 = b - a)
  (h2 : b - a = c - b)
  (h3 : c - b = -9 - c) :
  b = -5 ∧ a * c = 21 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_solution_l1059_105969


namespace NUMINAMATH_GPT_banquet_food_consumption_l1059_105934

theorem banquet_food_consumption (n : ℕ) (food_per_guest : ℕ) (total_food : ℕ) 
  (h1 : ∀ g : ℕ, g ≤ n -> g * food_per_guest ≤ total_food)
  (h2 : n = 169) 
  (h3 : food_per_guest = 2) :
  total_food = 338 := 
sorry

end NUMINAMATH_GPT_banquet_food_consumption_l1059_105934


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1059_105973

theorem solution_set_of_inequality (x : ℝ) : x * (x + 3) ≥ 0 ↔ x ≤ -3 ∨ x ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1059_105973


namespace NUMINAMATH_GPT_min_fencing_dims_l1059_105942

theorem min_fencing_dims (x : ℕ) (h₁ : x * (x + 5) ≥ 600) (h₂ : x = 23) : 
  2 * (x + (x + 5)) = 102 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_min_fencing_dims_l1059_105942


namespace NUMINAMATH_GPT_probability_bons_wins_even_rolls_l1059_105939
noncomputable def probability_of_Bons_winning (p6 : ℚ) (p_not6 : ℚ) : ℚ := 
  let r := p_not6^2
  let a := p_not6 * p6
  a / (1 - r)

theorem probability_bons_wins_even_rolls : 
  let p6 := (1 : ℚ) / 6
  let p_not6 := (5 : ℚ) / 6
  probability_of_Bons_winning p6 p_not6 = (5 : ℚ) / 11 := 
  sorry

end NUMINAMATH_GPT_probability_bons_wins_even_rolls_l1059_105939


namespace NUMINAMATH_GPT_log_add_property_l1059_105900

theorem log_add_property (log : ℝ → ℝ) (h1 : ∀ a b : ℝ, 0 < a → 0 < b → log a + log b = log (a * b)) (h2 : log 10 = 1) :
  log 5 + log 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_log_add_property_l1059_105900


namespace NUMINAMATH_GPT_g_15_33_eq_165_l1059_105916

noncomputable def g : ℕ → ℕ → ℕ := sorry

axiom g_self (x : ℕ) : g x x = x
axiom g_comm (x y : ℕ) : g x y = g y x
axiom g_equation (x y : ℕ) : (x + y) * g x y = y * g x (x + y)

theorem g_15_33_eq_165 : g 15 33 = 165 := by sorry

end NUMINAMATH_GPT_g_15_33_eq_165_l1059_105916


namespace NUMINAMATH_GPT_sally_popped_3_balloons_l1059_105936

-- Defining the conditions
def joans_initial_balloons : ℕ := 9
def jessicas_balloons : ℕ := 2
def total_balloons_now : ℕ := 6

-- Definition for the number of balloons Sally popped
def sally_balloons_popped : ℕ := joans_initial_balloons - (total_balloons_now - jessicas_balloons)

-- The theorem statement
theorem sally_popped_3_balloons : sally_balloons_popped = 3 := 
by
  -- Proof omitted; use sorry
  sorry

end NUMINAMATH_GPT_sally_popped_3_balloons_l1059_105936


namespace NUMINAMATH_GPT_cost_for_sugar_substitutes_l1059_105913

def packets_per_cup : ℕ := 1
def cups_per_day : ℕ := 2
def days : ℕ := 90
def packets_per_box : ℕ := 30
def price_per_box : ℕ := 4

theorem cost_for_sugar_substitutes : 
  (packets_per_cup * cups_per_day * days / packets_per_box) * price_per_box = 24 := by
  sorry

end NUMINAMATH_GPT_cost_for_sugar_substitutes_l1059_105913


namespace NUMINAMATH_GPT_discriminant_of_given_quadratic_l1059_105950

-- define the coefficients a, b, c
def a : ℚ := 2
def b : ℚ := 2 + 1/2
def c : ℚ := 1/2

-- define the discriminant function for a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

-- state the theorem
theorem discriminant_of_given_quadratic : discriminant a b c = 9/4 :=
by
  -- add the proof here
  sorry

end NUMINAMATH_GPT_discriminant_of_given_quadratic_l1059_105950


namespace NUMINAMATH_GPT_trajectory_midpoint_of_chord_l1059_105952

theorem trajectory_midpoint_of_chord :
  ∀ (M: ℝ × ℝ), (∃ (C D : ℝ × ℝ), (C.1^2 + C.2^2 = 25 ∧ D.1^2 + D.2^2 = 25 ∧ dist C D = 8) ∧ M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
  → M.1^2 + M.2^2 = 9 :=
sorry

end NUMINAMATH_GPT_trajectory_midpoint_of_chord_l1059_105952


namespace NUMINAMATH_GPT_determine_a_value_l1059_105930

theorem determine_a_value (a : ℤ) (h : ∀ x : ℝ, x^2 + 2 * (a:ℝ) * x + 1 > 0) : a = 0 := 
sorry

end NUMINAMATH_GPT_determine_a_value_l1059_105930


namespace NUMINAMATH_GPT_min_value_of_M_l1059_105945

noncomputable def min_val (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :=
  max (1/(a*c) + b) (max (1/a + b*c) (a/b + c))

theorem min_value_of_M (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (min_val a b c h1 h2 h3) >= 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_M_l1059_105945


namespace NUMINAMATH_GPT_find_difference_l1059_105978

theorem find_difference (x0 y0 : ℝ) 
  (h1 : x0^3 - 2023 * x0 = y0^3 - 2023 * y0 + 2020)
  (h2 : x0^2 + x0 * y0 + y0^2 = 2022) : 
  x0 - y0 = -2020 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_l1059_105978


namespace NUMINAMATH_GPT_find_s_t_l1059_105966

noncomputable def problem_constants (a b c : ℝ) : Prop :=
  (a^3 + 3 * a^2 + 4 * a - 11 = 0) ∧
  (b^3 + 3 * b^2 + 4 * b - 11 = 0) ∧
  (c^3 + 3 * c^2 + 4 * c - 11 = 0)

theorem find_s_t (a b c s t : ℝ) (h1 : problem_constants a b c) (h2 : (a + b) * (b + c) * (c + a) = -t)
  (h3 : (a + b) * (b + c) + (b + c) * (c + a) + (c + a) * (a + b) = s) :
s = 8 ∧ t = 23 :=
sorry

end NUMINAMATH_GPT_find_s_t_l1059_105966


namespace NUMINAMATH_GPT_inequality_condition_l1059_105968

theorem inequality_condition 
  (a b c : ℝ) : 
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (c > Real.sqrt (a^2 + b^2)) := 
sorry

end NUMINAMATH_GPT_inequality_condition_l1059_105968


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1059_105912

theorem necessary_but_not_sufficient (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1059_105912


namespace NUMINAMATH_GPT_three_digit_divisible_by_7_iff_last_two_digits_equal_l1059_105928

-- Define the conditions as given in the problem
variable (a b c : ℕ)

-- Ensure the sum of the digits is 7, as given by the problem conditions
def sum_of_digits_eq_7 := a + b + c = 7

-- Ensure that it is a three-digit number
def valid_three_digit_number := a ≠ 0

-- Define what it means to be divisible by 7
def divisible_by_7 (n : ℕ) := n % 7 = 0

-- Define the problem statement in Lean
theorem three_digit_divisible_by_7_iff_last_two_digits_equal (h1 : sum_of_digits_eq_7 a b c) (h2 : valid_three_digit_number a) :
  divisible_by_7 (100 * a + 10 * b + c) ↔ b = c :=
by sorry

end NUMINAMATH_GPT_three_digit_divisible_by_7_iff_last_two_digits_equal_l1059_105928


namespace NUMINAMATH_GPT_staff_price_l1059_105957

theorem staff_price (d : ℝ) : (d - 0.55 * d) / 2 = 0.225 * d := by
  sorry

end NUMINAMATH_GPT_staff_price_l1059_105957


namespace NUMINAMATH_GPT_incorrect_conversion_l1059_105953

/--
Incorrect conversion of -150° to radians.
-/
theorem incorrect_conversion : (¬(((-150 : ℝ) * (Real.pi / 180)) = (-7 * Real.pi / 6))) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_conversion_l1059_105953


namespace NUMINAMATH_GPT_worker_payment_l1059_105991

theorem worker_payment (x : ℕ) (daily_return : ℕ) (non_working_days : ℕ) (total_days : ℕ) 
    (net_earning : ℕ) 
    (H1 : daily_return = 25) 
    (H2 : non_working_days = 24) 
    (H3 : total_days = 30) 
    (H4 : net_earning = 0) 
    (H5 : ∀ w, net_earning = w * x - non_working_days * daily_return) : 
  x = 100 :=
by
  sorry

end NUMINAMATH_GPT_worker_payment_l1059_105991


namespace NUMINAMATH_GPT_gcd_of_abcd_dcba_l1059_105905

theorem gcd_of_abcd_dcba : 
  ∀ (a : ℕ), 0 ≤ a ∧ a ≤ 3 → 
  gcd (2332 * a + 7112) (2332 * (a + 1) + 7112) = 2 ∧ 
  gcd (2332 * (a + 1) + 7112) (2332 * (a + 2) + 7112) = 2 ∧ 
  gcd (2332 * (a + 2) + 7112) (2332 * (a + 3) + 7112) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_of_abcd_dcba_l1059_105905


namespace NUMINAMATH_GPT_quadratic_roots_l1059_105951

theorem quadratic_roots (k : ℝ) :
  (∃ x : ℝ, x = 2 ∧ 4 * x ^ 2 - k * x + 6 = 0) →
  k = 11 ∧ (∃ x : ℝ, x ≠ 2 ∧ 4 * x ^ 2 - 11 * x + 6 = 0 ∧ x = 3 / 4) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1059_105951


namespace NUMINAMATH_GPT_sequence_formula_l1059_105998

theorem sequence_formula (a : ℕ → ℚ) (h1 : a 2 = 3 / 2) (h2 : a 3 = 7 / 3) 
  (h3 : ∀ n : ℕ, ∃ r : ℚ, (∀ m : ℕ, m ≥ 2 → (m * a m + 1) / (n * a n + 1) = r ^ (m - n))) :
  a n = (2^n - 1) / n := 
sorry

end NUMINAMATH_GPT_sequence_formula_l1059_105998


namespace NUMINAMATH_GPT_unique_solution_of_quadratic_l1059_105964

theorem unique_solution_of_quadratic (b c x : ℝ) (h_eqn : 9 * x^2 + b * x + c = 0) (h_one_solution : ∀ y: ℝ, 9 * y^2 + b * y + c = 0 → y = x) (h_b2_4c : b^2 = 4 * c) : 
  x = -b / 18 := 
by 
  sorry

end NUMINAMATH_GPT_unique_solution_of_quadratic_l1059_105964


namespace NUMINAMATH_GPT_intersection_of_sets_l1059_105921

noncomputable def setA : Set ℝ := { x | |x - 2| ≤ 3 }
noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = 1 - x^2 }

theorem intersection_of_sets : (setA ∩ { x | 1 - x^2 ∈ setB }) = Set.Icc (-1) 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1059_105921


namespace NUMINAMATH_GPT_not_perfect_square_l1059_105902

theorem not_perfect_square : ¬ ∃ x : ℝ, x^2 = 7^2025 := by
  sorry

end NUMINAMATH_GPT_not_perfect_square_l1059_105902


namespace NUMINAMATH_GPT_Janet_earnings_l1059_105960

/--
Janet works as an exterminator and also sells molten metal casts of fire ant nests on the Internet.
She gets paid $70 an hour for exterminator work and makes $20/pound on her ant nest sculptures.
Given that she does 20 hours of exterminator work and sells a 5-pound sculpture and a 7-pound sculpture,
prove that Janet's total earnings are $1640.
-/
theorem Janet_earnings :
  let hourly_rate_exterminator := 70
  let hours_worked := 20
  let rate_per_pound := 20
  let sculpture_one_weight := 5
  let sculpture_two_weight := 7

  let exterminator_earnings := hourly_rate_exterminator * hours_worked
  let total_sculpture_weight := sculpture_one_weight + sculpture_two_weight
  let sculpture_earnings := rate_per_pound * total_sculpture_weight

  let total_earnings := exterminator_earnings + sculpture_earnings
  total_earnings = 1640 := 
by
  sorry

end NUMINAMATH_GPT_Janet_earnings_l1059_105960


namespace NUMINAMATH_GPT_ants_crushed_l1059_105949

theorem ants_crushed {original_ants left_ants crushed_ants : ℕ} 
  (h1 : original_ants = 102) 
  (h2 : left_ants = 42) 
  (h3 : crushed_ants = original_ants - left_ants) : 
  crushed_ants = 60 := 
by
  sorry

end NUMINAMATH_GPT_ants_crushed_l1059_105949


namespace NUMINAMATH_GPT_interest_rate_B_to_C_l1059_105987

theorem interest_rate_B_to_C
  (P : ℕ)                -- Principal amount
  (r_A : ℚ)              -- Interest rate A charges B per annum
  (t : ℕ)                -- Time period in years
  (gain_B : ℚ)           -- Gain of B in 3 years
  (H_P : P = 3500)
  (H_r_A : r_A = 0.10)
  (H_t : t = 3)
  (H_gain_B : gain_B = 315) :
  ∃ R : ℚ, R = 0.13 := 
by
  sorry

end NUMINAMATH_GPT_interest_rate_B_to_C_l1059_105987


namespace NUMINAMATH_GPT_divisibility_by_3_l1059_105946

theorem divisibility_by_3 (x y z : ℤ) (h : x^3 + y^3 = z^3) : 3 ∣ x ∨ 3 ∣ y ∨ 3 ∣ z := 
sorry

end NUMINAMATH_GPT_divisibility_by_3_l1059_105946


namespace NUMINAMATH_GPT_mateen_backyard_area_l1059_105923

theorem mateen_backyard_area :
  (∀ (L : ℝ), 30 * L = 1200) →
  (∀ (P : ℝ), 12 * P = 1200) →
  (∃ (L W : ℝ), 2 * L + 2 * W = 100 ∧ L * W = 400) := by
  intros hL hP
  use 40
  use 10
  apply And.intro
  sorry
  sorry

end NUMINAMATH_GPT_mateen_backyard_area_l1059_105923


namespace NUMINAMATH_GPT_factorize_expression_l1059_105937

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end NUMINAMATH_GPT_factorize_expression_l1059_105937


namespace NUMINAMATH_GPT_expected_value_is_750_l1059_105999

def winnings (roll : ℕ) : ℕ :=
  if roll % 2 = 0 then 3 * roll else 0

def expected_value : ℚ :=
  (winnings 2 / 8) + (winnings 4 / 8) + (winnings 6 / 8) + (winnings 8 / 8)

theorem expected_value_is_750 : expected_value = 7.5 := by
  sorry

end NUMINAMATH_GPT_expected_value_is_750_l1059_105999


namespace NUMINAMATH_GPT_platform_protection_l1059_105959

noncomputable def max_distance (r : ℝ) (n : ℕ) : ℝ :=
  if n > 2 then r / (Real.sin (180.0 / n)) else 0

noncomputable def coverage_ring_area (r : ℝ) (w : ℝ) : ℝ :=
  let inner_radius := r * (Real.sin 20.0)
  let outer_radius := inner_radius + w
  Real.pi * (outer_radius^2 - inner_radius^2)

theorem platform_protection :
  let r := 61
  let w := 22
  let n := 9
  max_distance r n = 60 / Real.sin 20.0 ∧
  coverage_ring_area r w = 2640 * Real.pi / Real.tan 20.0 := by
  sorry

end NUMINAMATH_GPT_platform_protection_l1059_105959


namespace NUMINAMATH_GPT_watermelon_seeds_l1059_105904

theorem watermelon_seeds (n_slices : ℕ) (total_seeds : ℕ) (B W : ℕ) 
  (h1: n_slices = 40) 
  (h2: B = W) 
  (h3 : n_slices * B + n_slices * W = total_seeds)
  (h4 : total_seeds = 1600) : B = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_watermelon_seeds_l1059_105904


namespace NUMINAMATH_GPT_fraction_to_decimal_l1059_105920

theorem fraction_to_decimal : (47 : ℝ) / 160 = 0.29375 :=
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1059_105920


namespace NUMINAMATH_GPT_pizza_volume_piece_l1059_105940

theorem pizza_volume_piece (h : ℝ) (d : ℝ) (n : ℝ) (V_piece : ℝ) 
  (h_eq : h = 1 / 2) (d_eq : d = 16) (n_eq : n = 8) : 
  V_piece = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_pizza_volume_piece_l1059_105940


namespace NUMINAMATH_GPT_midpoint_sum_coordinates_l1059_105962

theorem midpoint_sum_coordinates :
  let p1 : ℝ × ℝ := (8, -4)
  let p2 : ℝ × ℝ := (-2, 16)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 + midpoint.2) = 9 :=
by
  let p1 : ℝ × ℝ := (8, -4)
  let p2 : ℝ × ℝ := (-2, 16)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  show midpoint.1 + midpoint.2 = 9
  sorry

end NUMINAMATH_GPT_midpoint_sum_coordinates_l1059_105962


namespace NUMINAMATH_GPT_fish_population_l1059_105938

theorem fish_population (N : ℕ) (h1 : 50 > 0) (h2 : N > 0) 
  (tagged_first_catch : ℕ) (total_first_catch : ℕ)
  (tagged_second_catch : ℕ) (total_second_catch : ℕ)
  (h3 : tagged_first_catch = 50)
  (h4 : total_first_catch = 50)
  (h5 : tagged_second_catch = 2)
  (h6 : total_second_catch = 50)
  (h_percent : (tagged_first_catch : ℝ) / (N : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ))
  : N = 1250 := 
  by sorry

end NUMINAMATH_GPT_fish_population_l1059_105938


namespace NUMINAMATH_GPT_trapezium_other_side_length_l1059_105984

theorem trapezium_other_side_length (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ) :
  side1 = 20 ∧ distance = 15 ∧ area = 285 → side2 = 18 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_trapezium_other_side_length_l1059_105984


namespace NUMINAMATH_GPT_evaluate_expression_l1059_105976

theorem evaluate_expression : (-7)^3 / 7^2 - 4^4 + 5^2 = -238 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1059_105976


namespace NUMINAMATH_GPT_martin_travel_time_l1059_105986

-- Definitions based on the conditions
def distance : ℕ := 12
def speed : ℕ := 2

-- Statement of the problem to be proven
theorem martin_travel_time : (distance / speed) = 6 := by sorry

end NUMINAMATH_GPT_martin_travel_time_l1059_105986


namespace NUMINAMATH_GPT_painted_cube_ways_l1059_105982

theorem painted_cube_ways (b r g : ℕ) (cubes : ℕ) : 
  b = 1 → r = 2 → g = 3 → cubes = 3 := 
by
  intros
  sorry

end NUMINAMATH_GPT_painted_cube_ways_l1059_105982


namespace NUMINAMATH_GPT_expression_value_l1059_105954

-- Define the variables and the main statement
variable (w x y z : ℕ)

theorem expression_value :
  2^w * 3^x * 5^y * 11^z = 825 → w + 2 * x + 3 * y + 4 * z = 12 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_expression_value_l1059_105954


namespace NUMINAMATH_GPT_dan_present_age_l1059_105975

theorem dan_present_age : ∃ x : ℕ, (x + 18 = 8 * (x - 3)) ∧ x = 6 :=
by
  -- We skip the proof steps
  sorry

end NUMINAMATH_GPT_dan_present_age_l1059_105975


namespace NUMINAMATH_GPT_car_speed_ratio_l1059_105947

theorem car_speed_ratio (v_A v_B : ℕ) (h1 : v_B = 50) (h2 : 6 * v_A + 2 * v_B = 1000) :
  v_A / v_B = 3 :=
sorry

end NUMINAMATH_GPT_car_speed_ratio_l1059_105947


namespace NUMINAMATH_GPT_sum_of_coefficients_polynomial_expansion_l1059_105948

theorem sum_of_coefficients_polynomial_expansion :
  let polynomial := (2 * (1 : ℤ) + 3)^5
  ∃ b_5 b_4 b_3 b_2 b_1 b_0 : ℤ,
  polynomial = b_5 * 1^5 + b_4 * 1^4 + b_3 * 1^3 + b_2 * 1^2 + b_1 * 1 + b_0 ∧
  (b_5 + b_4 + b_3 + b_2 + b_1 + b_0) = 3125 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_polynomial_expansion_l1059_105948


namespace NUMINAMATH_GPT_probability_boarding_251_l1059_105965

theorem probability_boarding_251 :
  let interval_152 := 5
  let interval_251 := 7
  let total_events := interval_152 * interval_251
  let favorable_events := (interval_152 * interval_152) / 2
  (favorable_events / total_events : ℚ) = 5 / 14 :=
by 
  sorry

end NUMINAMATH_GPT_probability_boarding_251_l1059_105965


namespace NUMINAMATH_GPT_complex_vector_PQ_l1059_105995

theorem complex_vector_PQ (P Q : ℂ) (hP : P = 3 + 1 * I) (hQ : Q = 2 + 3 * I) : 
  (Q - P) = -1 + 2 * I :=
by sorry

end NUMINAMATH_GPT_complex_vector_PQ_l1059_105995


namespace NUMINAMATH_GPT_min_value_range_l1059_105908

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem min_value_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 2 * a^2 - a - 1) → (0 < a ∧ a ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_min_value_range_l1059_105908


namespace NUMINAMATH_GPT_solve_for_x_l1059_105988

def determinant_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem solve_for_x (x : ℝ) (h : determinant_2x2 (x+1) (x+2) (x-3) (x-1) = 2023) :
  x = 2018 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1059_105988


namespace NUMINAMATH_GPT_main_theorem_l1059_105979

variables {m n : ℕ} {x : ℝ}
variables {a : ℕ → ℕ}
noncomputable def relatively_prime (a : ℕ → ℕ) (n : ℕ) : Prop :=
∀ i j, i ≠ j → i < n → j < n → Nat.gcd (a i) (a j) = 1

noncomputable def distinct (a : ℕ → ℕ) (n : ℕ) : Prop :=
∀ i j, i ≠ j → i < n → j < n → a i ≠ a j

theorem main_theorem (hm : 1 < m) (hn : 1 < n) (hge : m ≥ n)
  (hrel_prime : relatively_prime a n)
  (hdistinct : distinct a n)
  (hbound : ∀ i, i < n → a i ≤ m)
  : ∃ i, i < n ∧ ‖a i * x‖ ≥ (2 / (m * (m + 1))) * ‖x‖ := 
sorry

end NUMINAMATH_GPT_main_theorem_l1059_105979


namespace NUMINAMATH_GPT_evaluate_g_at_3_l1059_105994

def g (x : ℤ) : ℤ := 3 * x^3 + 5 * x^2 - 2 * x - 7

theorem evaluate_g_at_3 : g 3 = 113 := by
  -- Proof of g(3) = 113 skipped
  sorry

end NUMINAMATH_GPT_evaluate_g_at_3_l1059_105994
