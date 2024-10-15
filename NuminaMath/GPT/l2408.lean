import Mathlib

namespace NUMINAMATH_GPT_max_area_rectangle_perimeter_156_l2408_240894

theorem max_area_rectangle_perimeter_156 (x y : ℕ) 
  (h : 2 * (x + y) = 156) : ∃x y, x * y = 1521 :=
by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_perimeter_156_l2408_240894


namespace NUMINAMATH_GPT_quadratic_inequality_solution_range_l2408_240801

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x, 1 < x ∧ x < 4 ∧ x^2 - 4 * x - 2 - a > 0) → a < -2 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_range_l2408_240801


namespace NUMINAMATH_GPT_pyramid_base_side_length_l2408_240854

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (s : ℝ)
  (h_area_lateral_face : area_lateral_face = 144)
  (h_slant_height : slant_height = 24) :
  (1 / 2) * s * slant_height = area_lateral_face → s = 12 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_base_side_length_l2408_240854


namespace NUMINAMATH_GPT_longer_side_length_l2408_240832

theorem longer_side_length (total_rope_length shorter_side_length longer_side_length : ℝ) 
  (h1 : total_rope_length = 100) 
  (h2 : shorter_side_length = 22) 
  : 2 * shorter_side_length + 2 * longer_side_length = total_rope_length -> longer_side_length = 28 :=
by sorry

end NUMINAMATH_GPT_longer_side_length_l2408_240832


namespace NUMINAMATH_GPT_adults_on_field_trip_l2408_240865

-- Define the conditions
def van_capacity : ℕ := 7
def num_students : ℕ := 33
def num_vans : ℕ := 6

-- Define the total number of people that can be transported given the number of vans and capacity per van
def total_people : ℕ := num_vans * van_capacity

-- The number of people that can be transported minus the number of students gives the number of adults
def num_adults : ℕ := total_people - num_students

-- Theorem to prove the number of adults is 9
theorem adults_on_field_trip : num_adults = 9 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_adults_on_field_trip_l2408_240865


namespace NUMINAMATH_GPT_series_sum_eq_neg_one_l2408_240815

   noncomputable def sum_series : ℝ :=
     ∑' k : ℕ, if k = 0 then 0 else (12 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))

   theorem series_sum_eq_neg_one : sum_series = -1 :=
   sorry
   
end NUMINAMATH_GPT_series_sum_eq_neg_one_l2408_240815


namespace NUMINAMATH_GPT_seventh_term_value_l2408_240845

open Nat

noncomputable def a : ℤ := sorry
noncomputable def d : ℤ := sorry
noncomputable def n : ℤ := sorry

-- Conditions as definitions
def sum_first_five : Prop := 5 * a + 10 * d = 34
def sum_last_five : Prop := 5 * a + 5 * (n - 1) * d = 146
def sum_all_terms : Prop := (n * (2 * a + (n - 1) * d)) / 2 = 234

-- Theorem statement
theorem seventh_term_value :
  sum_first_five ∧ sum_last_five ∧ sum_all_terms → a + 6 * d = 18 :=
by
  sorry

end NUMINAMATH_GPT_seventh_term_value_l2408_240845


namespace NUMINAMATH_GPT_problem_statement_l2408_240875

theorem problem_statement (n : ℕ) (h : ∀ (a b : ℕ), ¬ (n ∣ (2^a * 3^b + 1))) :
  ∀ (c d : ℕ), ¬ (n ∣ (2^c + 3^d)) := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2408_240875


namespace NUMINAMATH_GPT_minimum_average_cost_l2408_240889

noncomputable def average_cost (x : ℝ) : ℝ :=
  let y := (x^2) / 10 - 30 * x + 4000
  y / x

theorem minimum_average_cost : 
  ∃ (x : ℝ), 150 ≤ x ∧ x ≤ 250 ∧ (∀ (x' : ℝ), 150 ≤ x' ∧ x' ≤ 250 → average_cost x ≤ average_cost x') ∧ average_cost x = 10 := 
by
  sorry

end NUMINAMATH_GPT_minimum_average_cost_l2408_240889


namespace NUMINAMATH_GPT_exponential_decreasing_iff_frac_inequality_l2408_240833

theorem exponential_decreasing_iff_frac_inequality (a : ℝ) :
  (0 < a ∧ a < 1) ↔ (a ≠ 1 ∧ a * (a - 1) ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_exponential_decreasing_iff_frac_inequality_l2408_240833


namespace NUMINAMATH_GPT_range_of_m_l2408_240813

noncomputable def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
noncomputable def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) (h₀ : m > 0) (h₁ : ∀ x : ℝ, q x m → p x) : m ≥ 9 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2408_240813


namespace NUMINAMATH_GPT_dhoni_savings_percent_l2408_240822

variable (E : ℝ) -- Assuming E is Dhoni's last month's earnings

-- Condition 1: Dhoni spent 25% of his earnings on rent
def spent_on_rent (E : ℝ) : ℝ := 0.25 * E

-- Condition 2: Dhoni spent 10% less than what he spent on rent on a new dishwasher
def spent_on_dishwasher (E : ℝ) : ℝ := 0.225 * E

-- Prove the percentage of last month's earnings Dhoni had left over
theorem dhoni_savings_percent (E : ℝ) : 
    52.5 / 100 * E = E - (spent_on_rent E + spent_on_dishwasher E) :=
by
  sorry

end NUMINAMATH_GPT_dhoni_savings_percent_l2408_240822


namespace NUMINAMATH_GPT_white_paint_amount_l2408_240892

theorem white_paint_amount (total_blue_paint additional_blue_paint total_mix blue_parts red_parts white_parts green_parts : ℕ) 
    (h_ratio: blue_parts = 7 ∧ red_parts = 2 ∧ white_parts = 1 ∧ green_parts = 1)
    (total_blue_paint_eq: total_blue_paint = 140)
    (max_total_mix: additional_blue_paint ≤ 220 - total_blue_paint) 
    : (white_parts * (total_blue_paint / blue_parts)) = 20 := 
by 
  sorry

end NUMINAMATH_GPT_white_paint_amount_l2408_240892


namespace NUMINAMATH_GPT_smallest_number_of_rectangles_needed_l2408_240885

theorem smallest_number_of_rectangles_needed :
  ∃ n, (n * 12 = 144) ∧ (∀ k, (k * 12 = 144) → k ≥ n) := by
  sorry

end NUMINAMATH_GPT_smallest_number_of_rectangles_needed_l2408_240885


namespace NUMINAMATH_GPT_avg_weight_BC_l2408_240825

variable (A B C : ℝ)

def totalWeight_ABC := 3 * 45
def totalWeight_AB := 2 * 40
def weight_B := 31

theorem avg_weight_BC : ((B + C) / 2) = 43 :=
  by
    have totalWeight_ABC_eq : A + B + C = totalWeight_ABC := by sorry
    have totalWeight_AB_eq : A + B = totalWeight_AB := by sorry
    have weight_B_eq : B = weight_B := by sorry
    sorry

end NUMINAMATH_GPT_avg_weight_BC_l2408_240825


namespace NUMINAMATH_GPT_range_of_a_if_p_true_l2408_240812

theorem range_of_a_if_p_true : 
  (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 9 ∧ x^2 - a * x + 36 ≤ 0) → a ≥ 12 :=
sorry

end NUMINAMATH_GPT_range_of_a_if_p_true_l2408_240812


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l2408_240871

theorem arithmetic_sequence_properties 
  (a : ℕ → ℤ) 
  (h1 : a 1 + a 2 + a 3 = 21) 
  (h2 : a 1 * a 2 * a 3 = 231) :
  (a 2 = 7) ∧ (∀ n, a n = -4 * n + 15 ∨ a n = 4 * n - 1) := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l2408_240871


namespace NUMINAMATH_GPT_eric_running_time_l2408_240880

-- Define the conditions
variables (jog_time to_park_time return_time : ℕ)
axiom jog_time_def : jog_time = 10
axiom return_time_def : return_time = 90
axiom trip_relation : return_time = 3 * to_park_time

-- Define the question
def run_time : ℕ := to_park_time - jog_time

-- State the problem: Prove that given the conditions, the running time is 20 minutes.
theorem eric_running_time : run_time = 20 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_eric_running_time_l2408_240880


namespace NUMINAMATH_GPT_intersection_point_l2408_240810

theorem intersection_point (x y : ℝ) (h1 : y = x + 1) (h2 : y = -x + 1) : (x = 0) ∧ (y = 1) := 
by
  sorry

end NUMINAMATH_GPT_intersection_point_l2408_240810


namespace NUMINAMATH_GPT_remainder_b22_div_35_l2408_240835

def b_n (n : ℕ) : Nat :=
  ((List.range (n + 1)).drop 1).foldl (λ acc k => acc * 10^(Nat.digits 10 k).length + k) 0

theorem remainder_b22_div_35 : (b_n 22) % 35 = 17 :=
  sorry

end NUMINAMATH_GPT_remainder_b22_div_35_l2408_240835


namespace NUMINAMATH_GPT_geometric_sequence_t_value_l2408_240867

theorem geometric_sequence_t_value (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = t * 5^n - 2) → 
  (∀ n ≥ 1, a (n + 1) = S (n + 1) - S n) → 
  (a 1 ≠ 0) → -- Ensure the sequence is non-trivial.
  (∀ n, a (n + 1) / a n = 5) → 
  t = 5 := 
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_geometric_sequence_t_value_l2408_240867


namespace NUMINAMATH_GPT_eliminate_y_l2408_240838

theorem eliminate_y (x y : ℝ) (h1 : 2 * x + 3 * y = 1) (h2 : 3 * x - 6 * y = 7) :
  (4 * x + 6 * y) + (3 * x - 6 * y) = 9 :=
by
  sorry

end NUMINAMATH_GPT_eliminate_y_l2408_240838


namespace NUMINAMATH_GPT_pentagon_area_l2408_240844

theorem pentagon_area 
  (PQ QR RS ST TP : ℝ) 
  (angle_TPQ angle_PQR : ℝ) 
  (hPQ : PQ = 8) 
  (hQR : QR = 2) 
  (hRS : RS = 13) 
  (hST : ST = 13) 
  (hTP : TP = 8) 
  (hangle_TPQ : angle_TPQ = 90) 
  (hangle_PQR : angle_PQR = 90) : 
  PQ * QR + (1 / 2) * (TP - QR) * PQ + (1 / 2) * 10 * 12 = 100 := 
by
  sorry

end NUMINAMATH_GPT_pentagon_area_l2408_240844


namespace NUMINAMATH_GPT_number_of_people_quit_l2408_240846

-- Define the conditions as constants.
def initial_team_size : ℕ := 25
def new_members : ℕ := 13
def final_team_size : ℕ := 30

-- Define the question as a function.
def people_quit (Q : ℕ) : Prop :=
  initial_team_size - Q + new_members = final_team_size

-- Prove the main statement assuming the conditions.
theorem number_of_people_quit (Q : ℕ) (h : people_quit Q) : Q = 8 :=
by
  sorry -- Proof is not required, so we use sorry to skip it.

end NUMINAMATH_GPT_number_of_people_quit_l2408_240846


namespace NUMINAMATH_GPT_find_p_plus_q_l2408_240859

noncomputable def calculate_p_plus_q (DE EF FD WX : ℕ) (Area : ℕ → ℝ) : ℕ :=
  let s := (DE + EF + FD) / 2
  let triangle_area := (Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))) / 2
  let delta := triangle_area / (225 * WX)
  let gcd := Nat.gcd 41 225
  let p := 41 / gcd
  let q := 225 / gcd
  p + q

theorem find_p_plus_q : calculate_p_plus_q 13 30 19 15 (fun θ => 30 * θ - (41 / 225) * θ^2) = 266 := by
  sorry

end NUMINAMATH_GPT_find_p_plus_q_l2408_240859


namespace NUMINAMATH_GPT_sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one_l2408_240862

-- Defining the terms and the theorem
theorem sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one :
  (Real.sqrt 5 - 2) ^ 2023 * (Real.sqrt 5 + 2) ^ 2023 = 1 := 
by
  sorry

end NUMINAMATH_GPT_sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one_l2408_240862


namespace NUMINAMATH_GPT_insects_legs_l2408_240848

theorem insects_legs (L N : ℕ) (hL : L = 54) (hN : N = 9) : (L / N = 6) :=
by sorry

end NUMINAMATH_GPT_insects_legs_l2408_240848


namespace NUMINAMATH_GPT_sqrt_nested_l2408_240829

theorem sqrt_nested (x : ℝ) (hx : 0 ≤ x) : Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15 / 16) := by
  sorry

end NUMINAMATH_GPT_sqrt_nested_l2408_240829


namespace NUMINAMATH_GPT_find_k_l2408_240864

theorem find_k (x y k : ℝ) (h1 : 2 * x - y = 4) (h2 : k * x - 3 * y = 12) : k = 6 := by
  sorry

end NUMINAMATH_GPT_find_k_l2408_240864


namespace NUMINAMATH_GPT_legs_heads_difference_l2408_240840

variables (D C L H : ℕ)

theorem legs_heads_difference
    (hC : C = 18)
    (hL : L = 2 * D + 4 * C)
    (hH : H = D + C) :
    L - 2 * H = 36 :=
by
  have h1 : C = 18 := hC
  have h2 : L = 2 * D + 4 * C := hL
  have h3 : H = D + C := hH
  sorry

end NUMINAMATH_GPT_legs_heads_difference_l2408_240840


namespace NUMINAMATH_GPT_find_k_l2408_240869

open BigOperators

def a (n : ℕ) : ℕ := 2 ^ n

theorem find_k (k : ℕ) (h : a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8) + a (k+9) + a (k+10) = 2 ^ 15 - 2 ^ 5) : k = 4 :=
sorry

end NUMINAMATH_GPT_find_k_l2408_240869


namespace NUMINAMATH_GPT_N_is_even_l2408_240886

def sum_of_digits : ℕ → ℕ := sorry

theorem N_is_even 
  (N : ℕ)
  (h1 : sum_of_digits N = 100)
  (h2 : sum_of_digits (5 * N) = 50) : 
  Even N :=
sorry

end NUMINAMATH_GPT_N_is_even_l2408_240886


namespace NUMINAMATH_GPT_geom_sequence_ratio_l2408_240847

-- Definitions and assumptions for the problem
noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, 0 < r ∧ r < 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geom_sequence_ratio (a : ℕ → ℝ) (r : ℝ) 
  (h_geom: geom_seq a)
  (h_r: 0 < r ∧ r < 1)
  (h_seq: ∀ n : ℕ, a (n + 1) = a n * r)
  (ha1: a 7 * a 14 = 6)
  (ha2: a 4 + a 17 = 5) :
  (a 5 / a 18) = (3 / 2) :=
sorry

end NUMINAMATH_GPT_geom_sequence_ratio_l2408_240847


namespace NUMINAMATH_GPT_contrapositive_example_l2408_240806

theorem contrapositive_example :
  (∀ x : ℝ, x^2 < 4 → -2 < x ∧ x < 2) ↔ (∀ x : ℝ, (x ≥ 2 ∨ x ≤ -2) → x^2 ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_example_l2408_240806


namespace NUMINAMATH_GPT_system_of_equations_solution_l2408_240852

theorem system_of_equations_solution :
  ∀ (x y z : ℝ),
  4 * x + 2 * y + z = 20 →
  x + 4 * y + 2 * z = 26 →
  2 * x + y + 4 * z = 28 →
  20 * x^2 + 24 * x * y + 20 * y^2 + 12 * z^2 = 500 :=
by
  intros x y z h1 h2 h3
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l2408_240852


namespace NUMINAMATH_GPT_lena_more_candy_bars_than_nicole_l2408_240809

theorem lena_more_candy_bars_than_nicole
  (Lena Kevin Nicole : ℕ)
  (h1 : Lena = 16)
  (h2 : Lena + 5 = 3 * Kevin)
  (h3 : Kevin + 4 = Nicole) :
  Lena - Nicole = 5 :=
by
  sorry

end NUMINAMATH_GPT_lena_more_candy_bars_than_nicole_l2408_240809


namespace NUMINAMATH_GPT_range_of_reciprocal_sum_l2408_240842

theorem range_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
    4 ≤ (1/x + 1/y) :=
by
  sorry

end NUMINAMATH_GPT_range_of_reciprocal_sum_l2408_240842


namespace NUMINAMATH_GPT_hcf_of_given_numbers_l2408_240817

def hcf (x y : ℕ) : ℕ := Nat.gcd x y

theorem hcf_of_given_numbers :
  ∃ (A B : ℕ), A = 33 ∧ A * B = 363 ∧ hcf A B = 11 := 
by
  sorry

end NUMINAMATH_GPT_hcf_of_given_numbers_l2408_240817


namespace NUMINAMATH_GPT_fraction_taken_out_is_one_sixth_l2408_240872

-- Define the conditions
def original_cards : ℕ := 43
def cards_added_by_Sasha : ℕ := 48
def cards_left_after_Karen_took_out : ℕ := 83

-- Calculate the total number of cards initially after Sasha added hers
def total_cards_after_Sasha : ℕ := original_cards + cards_added_by_Sasha

-- Calculate the number of cards Karen took out
def cards_taken_out_by_Karen : ℕ := total_cards_after_Sasha - cards_left_after_Karen_took_out

-- Define the fraction of the cards Sasha added that Karen took out
def fraction_taken_out : ℚ := cards_taken_out_by_Karen / cards_added_by_Sasha

-- Proof statement: Fraction of the cards Sasha added that Karen took out is 1/6
theorem fraction_taken_out_is_one_sixth : fraction_taken_out = 1 / 6 :=
by
    -- Sorry is a placeholder for the proof, which is not required.
    sorry

end NUMINAMATH_GPT_fraction_taken_out_is_one_sixth_l2408_240872


namespace NUMINAMATH_GPT_even_digit_perfect_squares_odd_digit_perfect_squares_l2408_240839

-- Define the property of being a four-digit number
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Define the property of having even digits
def is_even_digit_number (n : ℕ) : Prop :=
  ∀ digit ∈ (n.digits 10), digit % 2 = 0

-- Define the property of having odd digits
def is_odd_digit_number (n : ℕ) : Prop :=
  ∀ digit ∈ (n.digits 10), digit % 2 = 1

-- Part (a) statement
theorem even_digit_perfect_squares :
  ∀ n : ℕ, is_four_digit n ∧ is_even_digit_number n ∧ ∃ m : ℕ, n = m * m ↔ 
    n = 4624 ∨ n = 6084 ∨ n = 6400 ∨ n = 8464 :=
sorry

-- Part (b) statement
theorem odd_digit_perfect_squares :
  ∀ n : ℕ, is_four_digit n ∧ is_odd_digit_number n ∧ ∃ m : ℕ, n = m * m → false :=
sorry

end NUMINAMATH_GPT_even_digit_perfect_squares_odd_digit_perfect_squares_l2408_240839


namespace NUMINAMATH_GPT_lateral_surface_area_of_cone_l2408_240843

theorem lateral_surface_area_of_cone (r l : ℝ) (h₁ : r = 3) (h₂ : l = 5) :
  π * r * l = 15 * π :=
by sorry

end NUMINAMATH_GPT_lateral_surface_area_of_cone_l2408_240843


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l2408_240888

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l2408_240888


namespace NUMINAMATH_GPT_hyperbola_real_axis_length_l2408_240896

variables {a b : ℝ} (ha : a > 0) (hb : b > 0) (h_asymptote_slope : b = 2 * a) (h_c : (a^2 + b^2) = 5)

theorem hyperbola_real_axis_length : 2 * a = 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_real_axis_length_l2408_240896


namespace NUMINAMATH_GPT_key_lime_yield_l2408_240856

def audrey_key_lime_juice_yield (cup_to_key_lime_juice_ratio: ℚ) (lime_juice_doubling_factor: ℚ) (tablespoons_per_cup: ℕ) (num_key_limes: ℕ) : ℚ :=
  let total_lime_juice_cups := cup_to_key_lime_juice_ratio * lime_juice_doubling_factor
  let total_lime_juice_tablespoons := total_lime_juice_cups * tablespoons_per_cup
  total_lime_juice_tablespoons / num_key_limes

-- Statement of the problem
theorem key_lime_yield :
  audrey_key_lime_juice_yield (1/4) 2 16 8 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_key_lime_yield_l2408_240856


namespace NUMINAMATH_GPT_line_intersects_circle_l2408_240890

theorem line_intersects_circle 
  (radius : ℝ) 
  (distance_center_line : ℝ) 
  (h_radius : radius = 4) 
  (h_distance : distance_center_line = 3) : 
  radius > distance_center_line := 
by 
  sorry

end NUMINAMATH_GPT_line_intersects_circle_l2408_240890


namespace NUMINAMATH_GPT_necessary_condition_real_roots_l2408_240816

theorem necessary_condition_real_roots (a : ℝ) :
  (a >= 1 ∨ a <= -2) → (∃ x : ℝ, x^2 - a * x + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_necessary_condition_real_roots_l2408_240816


namespace NUMINAMATH_GPT_select_student_D_l2408_240893

-- Define the scores and variances based on the conditions
def avg_A : ℝ := 96
def avg_B : ℝ := 94
def avg_C : ℝ := 93
def avg_D : ℝ := 96

def var_A : ℝ := 1.2
def var_B : ℝ := 1.2
def var_C : ℝ := 0.6
def var_D : ℝ := 0.4

-- Proof statement in Lean 4
theorem select_student_D (avg_A avg_B avg_C avg_D var_A var_B var_C var_D : ℝ) 
                         (h_avg_A : avg_A = 96)
                         (h_avg_B : avg_B = 94)
                         (h_avg_C : avg_C = 93)
                         (h_avg_D : avg_D = 96)
                         (h_var_A : var_A = 1.2)
                         (h_var_B : var_B = 1.2)
                         (h_var_C : var_C = 0.6)
                         (h_var_D : var_D = 0.4) 
                         (h_D_highest_avg : avg_D = max avg_A (max avg_B (max avg_C avg_D)))
                         (h_D_lowest_var : var_D = min (min (min var_A var_B) var_C) var_D) :
  avg_D = 96 ∧ var_D = 0.4 := 
by 
  -- As we're not asked to prove, we put sorry here to indicate the proof step is omitted.
  sorry

end NUMINAMATH_GPT_select_student_D_l2408_240893


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l2408_240826

theorem average_of_remaining_two_numbers (S S3 : ℝ) (h_avg5 : S / 5 = 8) (h_avg3 : S3 / 3 = 4) : S / 5 = 8 ∧ S3 / 3 = 4 → (S - S3) / 2 = 14 :=
by 
  sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l2408_240826


namespace NUMINAMATH_GPT_children_on_bus_after_events_l2408_240831

-- Definition of the given problem parameters
def initial_children : Nat := 21
def got_off : Nat := 10
def got_on : Nat := 5

-- The theorem we want to prove
theorem children_on_bus_after_events : initial_children - got_off + got_on = 16 :=
by
  -- This is where the proof would go, but we leave it as sorry for now
  sorry

end NUMINAMATH_GPT_children_on_bus_after_events_l2408_240831


namespace NUMINAMATH_GPT_fraction_of_3_4_is_4_27_l2408_240828

theorem fraction_of_3_4_is_4_27 (a b : ℚ) (h1 : a = 3/4) (h2 : b = 1/9) :
  b / a = 4 / 27 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_3_4_is_4_27_l2408_240828


namespace NUMINAMATH_GPT_inequality_solution_set_l2408_240803

theorem inequality_solution_set (x : ℝ) : (x - 1) * abs (x + 2) ≥ 0 ↔ (x ≥ 1 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l2408_240803


namespace NUMINAMATH_GPT_minutes_to_seconds_l2408_240879

theorem minutes_to_seconds (m : ℝ) (hm : m = 6.5) : m * 60 = 390 := by
  sorry

end NUMINAMATH_GPT_minutes_to_seconds_l2408_240879


namespace NUMINAMATH_GPT_number_of_students_l2408_240853

theorem number_of_students (x : ℕ) (h : x * (x - 1) = 210) : x = 15 := 
by sorry

end NUMINAMATH_GPT_number_of_students_l2408_240853


namespace NUMINAMATH_GPT_monotonicity_of_f_range_of_a_if_no_zeros_l2408_240850

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end NUMINAMATH_GPT_monotonicity_of_f_range_of_a_if_no_zeros_l2408_240850


namespace NUMINAMATH_GPT_binary_mult_div_to_decimal_l2408_240821

theorem binary_mult_div_to_decimal:
  let n1 := 2 ^ 5 + 2 ^ 4 + 2 ^ 2 + 2 ^ 1 -- This represents 101110_2
  let n2 := 2 ^ 6 + 2 ^ 4 + 2 ^ 2         -- This represents 1010100_2
  let d := 2 ^ 2                          -- This represents 100_2
  n1 * n2 / d = 2995 := 
by
  sorry

end NUMINAMATH_GPT_binary_mult_div_to_decimal_l2408_240821


namespace NUMINAMATH_GPT_largest_d_for_g_of_minus5_l2408_240811

theorem largest_d_for_g_of_minus5 (d : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + d = -5) → d ≤ -4 :=
by
-- Proof steps will be inserted here
sorry

end NUMINAMATH_GPT_largest_d_for_g_of_minus5_l2408_240811


namespace NUMINAMATH_GPT_problem_value_l2408_240800

theorem problem_value (x : ℤ) (h : x = -2023) : 
  abs (abs (abs x - x) - abs x) - x = 4046 :=
by
  sorry

end NUMINAMATH_GPT_problem_value_l2408_240800


namespace NUMINAMATH_GPT_question1_question2_l2408_240883

noncomputable def f (x b c : ℝ) := x^2 + b * x + c

theorem question1 (b c : ℝ) (h : ∀ x : ℝ, 2 * x + b ≤ f x b c) (x : ℝ) (hx : 0 ≤ x) :
  f x b c ≤ (x + c)^2 :=
sorry

theorem question2 (b c m : ℝ) (h : ∀ b c : ℝ, b ≠ c → f c b b - f b b b ≤ m * (c^2 - b^2)) :
  m ≥ 3/2 :=
sorry

end NUMINAMATH_GPT_question1_question2_l2408_240883


namespace NUMINAMATH_GPT_sum_of_largest_three_consecutive_numbers_l2408_240823

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_largest_three_consecutive_numbers_l2408_240823


namespace NUMINAMATH_GPT_math_problem_l2408_240849

theorem math_problem (a b c : ℝ) (h1 : a + 2 * b + 3 * c = 12) (h2 : a^2 + b^2 + c^2 = a * b + b * c + c * a) : a + b^2 + c^3 = 14 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2408_240849


namespace NUMINAMATH_GPT_prob_seven_heads_in_ten_tosses_l2408_240858

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.choose n k)

noncomputable def probability_of_heads (n k : ℕ) : ℚ :=
  (binomial_coefficient n k) * (0.5^k : ℚ) * (0.5^(n - k) : ℚ)

theorem prob_seven_heads_in_ten_tosses :
  probability_of_heads 10 7 = 15 / 128 :=
by
  sorry

end NUMINAMATH_GPT_prob_seven_heads_in_ten_tosses_l2408_240858


namespace NUMINAMATH_GPT_fraction_zero_l2408_240882

theorem fraction_zero (x : ℝ) (h : (x - 1) * (x + 2) = 0) (hne : x^2 - 1 ≠ 0) : x = -2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_zero_l2408_240882


namespace NUMINAMATH_GPT_coffee_ratio_l2408_240836

/-- Define the conditions -/
def initial_coffees_per_day := 4
def initial_price_per_coffee := 2
def price_increase_percentage := 50 / 100
def savings_per_day := 2

/-- Define the price calculations -/
def new_price_per_coffee := initial_price_per_coffee + (initial_price_per_coffee * price_increase_percentage)
def initial_daily_cost := initial_coffees_per_day * initial_price_per_coffee
def new_daily_cost := initial_daily_cost - savings_per_day
def new_coffees_per_day := new_daily_cost / new_price_per_coffee

/-- Prove the ratio -/
theorem coffee_ratio : (new_coffees_per_day / initial_coffees_per_day) = (1 : ℝ) / (2 : ℝ) :=
  by sorry

end NUMINAMATH_GPT_coffee_ratio_l2408_240836


namespace NUMINAMATH_GPT_meeting_time_l2408_240876

theorem meeting_time (x : ℝ) :
  (1/6) * x + (1/4) * (x - 1) = 1 :=
sorry

end NUMINAMATH_GPT_meeting_time_l2408_240876


namespace NUMINAMATH_GPT_number_of_students_taking_art_l2408_240827

noncomputable def total_students : ℕ := 500
noncomputable def students_taking_music : ℕ := 50
noncomputable def students_taking_both : ℕ := 10
noncomputable def students_taking_neither : ℕ := 440

theorem number_of_students_taking_art (A : ℕ) (h1: total_students = 500) (h2: students_taking_music = 50) 
  (h3: students_taking_both = 10) (h4: students_taking_neither = 440) : A = 20 :=
by 
  have h5 : total_students = students_taking_music - students_taking_both + A - students_taking_both + 
    students_taking_both + students_taking_neither := sorry
  have h6 : 500 = 40 + A - 10 + 10 + 440 := sorry
  have h7 : 500 = A + 480 := sorry
  have h8 : A = 20 := by linarith 
  exact h8

end NUMINAMATH_GPT_number_of_students_taking_art_l2408_240827


namespace NUMINAMATH_GPT_remainder_when_P_divided_by_ab_l2408_240873

-- Given conditions
variables {P a b c Q Q' R R' : ℕ}

-- Provided equations as conditions
def equation1 : P = a * Q + R :=
sorry

def equation2 : Q = (b + c) * Q' + R' :=
sorry

-- Proof problem statement
theorem remainder_when_P_divided_by_ab :
  P % (a * b) = (a * c * Q' + a * R' + R) % (a * b) :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_P_divided_by_ab_l2408_240873


namespace NUMINAMATH_GPT_yuan_older_than_david_l2408_240834

theorem yuan_older_than_david (David_age : ℕ) (Yuan_age : ℕ) 
  (h1 : Yuan_age = 2 * David_age) 
  (h2 : David_age = 7) : 
  Yuan_age - David_age = 7 := by
  sorry

end NUMINAMATH_GPT_yuan_older_than_david_l2408_240834


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_with_product_812_l2408_240897

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_with_product_812_l2408_240897


namespace NUMINAMATH_GPT_largest_divisor_three_consecutive_l2408_240891

theorem largest_divisor_three_consecutive (u v w : ℤ) (h1 : u + 1 = v) (h2 : v + 1 = w) (h3 : ∃ n : ℤ, (u = 5 * n) ∨ (v = 5 * n) ∨ (w = 5 * n)) : 
  ∀ d ∈ {d | ∀ a b c : ℤ, a * b * c = u * v * w → d ∣ a * b * c}, 
  15 ∈ {d | ∀ a b c : ℤ, a * b * c = u * v * w → d ∣ a * b * c} :=
sorry

end NUMINAMATH_GPT_largest_divisor_three_consecutive_l2408_240891


namespace NUMINAMATH_GPT_minimum_area_of_triangle_is_sqrt_58_div_2_l2408_240881

noncomputable def smallest_area_of_triangle (t s : ℝ) : ℝ :=
  (1/2) * Real.sqrt (5 * s^2 - 4 * s * t - 4 * s + 2 * t^2 + 10 * t + 13)

theorem minimum_area_of_triangle_is_sqrt_58_div_2 : ∃ t s : ℝ, smallest_area_of_triangle t s = Real.sqrt 58 / 2 := 
  by
  sorry

end NUMINAMATH_GPT_minimum_area_of_triangle_is_sqrt_58_div_2_l2408_240881


namespace NUMINAMATH_GPT_girls_attended_festival_l2408_240824

variable (g b : ℕ)

theorem girls_attended_festival :
  g + b = 1500 ∧ (2 / 3) * g + (1 / 2) * b = 900 → (2 / 3) * g = 600 := by
  sorry

end NUMINAMATH_GPT_girls_attended_festival_l2408_240824


namespace NUMINAMATH_GPT_add_fractions_l2408_240807

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by sorry

end NUMINAMATH_GPT_add_fractions_l2408_240807


namespace NUMINAMATH_GPT_initial_people_count_l2408_240863

theorem initial_people_count (C : ℝ) (n : ℕ) (h : n > 1) :
  ((C / (n - 1)) - (C / n) = 0.125) →
  n = 8 := by
  sorry

end NUMINAMATH_GPT_initial_people_count_l2408_240863


namespace NUMINAMATH_GPT_customers_left_is_31_l2408_240820

-- Define the initial number of customers
def initial_customers : ℕ := 33

-- Define the number of additional customers
def additional_customers : ℕ := 26

-- Define the final number of customers after some left and new ones came
def final_customers : ℕ := 28

-- Define the number of customers who left 
def customers_left (x : ℕ) : Prop :=
  (initial_customers - x) + additional_customers = final_customers

-- The proof statement that we aim to prove
theorem customers_left_is_31 : ∃ x : ℕ, customers_left x ∧ x = 31 :=
by
  use 31
  unfold customers_left
  sorry

end NUMINAMATH_GPT_customers_left_is_31_l2408_240820


namespace NUMINAMATH_GPT_quadratic_must_have_m_eq_neg2_l2408_240804

theorem quadratic_must_have_m_eq_neg2 (m : ℝ) (h : (m - 2) * x^|m| - 3 * x - 4 = 0) :
  (|m| = 2) ∧ (m ≠ 2) → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_must_have_m_eq_neg2_l2408_240804


namespace NUMINAMATH_GPT_solve_equation_l2408_240878

theorem solve_equation (x : ℝ) (h1 : 2 * x + 1 ≠ 0) (h2 : 4 * x ≠ 0) : 
  (3 / (2 * x + 1) = 5 / (4 * x)) ↔ (x = 2.5) :=
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l2408_240878


namespace NUMINAMATH_GPT_fraction_ordering_l2408_240884

theorem fraction_ordering :
  let a := (6 : ℚ) / 22
  let b := (8 : ℚ) / 32
  let c := (10 : ℚ) / 29
  a < b ∧ b < c :=
by
  sorry

end NUMINAMATH_GPT_fraction_ordering_l2408_240884


namespace NUMINAMATH_GPT_range_of_a_l2408_240851

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → x^2 + 2 * x - a > 0) → a < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2408_240851


namespace NUMINAMATH_GPT_total_votes_l2408_240808

/-- Let V be the total number of votes. Define the votes received by the candidate and rival. -/
def votes_cast (V : ℕ) (votes_candidate : ℕ) (votes_rival : ℕ) : Prop :=
  votes_candidate = 40 * V / 100 ∧ votes_rival = votes_candidate + 2000 ∧ votes_candidate + votes_rival = V

/-- Prove that the total number of votes is 10000 given the conditions. -/
theorem total_votes (V : ℕ) (votes_candidate : ℕ) (votes_rival : ℕ) :
  votes_cast V votes_candidate votes_rival → V = 10000 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_l2408_240808


namespace NUMINAMATH_GPT_toms_balloons_l2408_240899

-- Define the original number of balloons that Tom had
def original_balloons : ℕ := 30

-- Define the number of balloons that Tom gave to Fred
def balloons_given_to_Fred : ℕ := 16

-- Define the number of balloons that Tom has now
def balloons_left : ℕ := original_balloons - balloons_given_to_Fred

-- The theorem to prove
theorem toms_balloons : balloons_left = 14 := 
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_toms_balloons_l2408_240899


namespace NUMINAMATH_GPT_find_width_of_rectangle_l2408_240870

-- Given conditions
variable (P l w : ℕ)
variable (h1 : P = 240)
variable (h2 : P = 3 * l)

-- Prove the width of the rectangular field is 40 meters
theorem find_width_of_rectangle : w = 40 :=
  by 
  -- Add the necessary logical steps here
  sorry

end NUMINAMATH_GPT_find_width_of_rectangle_l2408_240870


namespace NUMINAMATH_GPT_classify_discuss_l2408_240877

theorem classify_discuss (a b c : ℚ) (h : a * b * c > 0) : 
  (|a| / a + |b| / b + |c| / c = 3) ∨ (|a| / a + |b| / b + |c| / c = -1) :=
sorry

end NUMINAMATH_GPT_classify_discuss_l2408_240877


namespace NUMINAMATH_GPT_sequence_general_term_l2408_240818

noncomputable def b_n (n : ℕ) : ℚ := 2 * n - 1
noncomputable def c_n (n : ℕ) : ℚ := n / (2 * n + 1)

theorem sequence_general_term (n : ℕ) : 
  b_n n + c_n n = (4 * n^2 + n - 1) / (2 * n + 1) :=
by sorry

end NUMINAMATH_GPT_sequence_general_term_l2408_240818


namespace NUMINAMATH_GPT_ball_and_ring_problem_l2408_240874

theorem ball_and_ring_problem (x y : ℕ) (m_x m_y : ℕ) : 
  m_x + 2 = y ∧ 
  m_y = x + 2 ∧
  x * m_x + y * m_y - 800 = 2 * (y - x) ∧
  x^2 + y^2 = 881 →
  (x = 25 ∧ y = 16) ∨ (x = 16 ∧ y = 25) := 
by 
  sorry

end NUMINAMATH_GPT_ball_and_ring_problem_l2408_240874


namespace NUMINAMATH_GPT_value_of_m_l2408_240868

theorem value_of_m (m : ℝ) : (∀ x : ℝ, (x^2 + 2 * m * x + m > 3 / 16)) ↔ (1 / 4 < m ∧ m < 3 / 4) :=
by sorry

end NUMINAMATH_GPT_value_of_m_l2408_240868


namespace NUMINAMATH_GPT_find_y_intercept_l2408_240837

theorem find_y_intercept (x1 y1 x2 y2 : ℝ) (h₁ : (x1, y1) = (2, -2)) (h₂ : (x2, y2) = (6, 6)) : 
  ∃ b : ℝ, (∀ x : ℝ, y = 2 * x + b) ∧ b = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_y_intercept_l2408_240837


namespace NUMINAMATH_GPT_minimum_roots_in_interval_l2408_240841

noncomputable def g : ℝ → ℝ := sorry

lemma symmetry_condition_1 (x : ℝ) : g (3 + x) = g (3 - x) := sorry
lemma symmetry_condition_2 (x : ℝ) : g (8 + x) = g (8 - x) := sorry
lemma initial_condition : g 1 = 0 := sorry

theorem minimum_roots_in_interval : 
  ∃ k, ∀ x, -1000 ≤ x ∧ x ≤ 1000 → g x = 0 ∧ 
  (2 * k) = 286 := sorry

end NUMINAMATH_GPT_minimum_roots_in_interval_l2408_240841


namespace NUMINAMATH_GPT_prime_base_values_l2408_240802

theorem prime_base_values :
  ∀ p : ℕ, Prime p →
    (2 * p^3 + p^2 + 6 + 4 * p^2 + p + 4 + 2 * p^2 + p + 5 + 2 * p^2 + 2 * p + 2 + 9 =
     4 * p^2 + 3 * p + 3 + 5 * p^2 + 7 * p + 2 + 3 * p^2 + 2 * p + 1) →
    false :=
by {
  sorry
}

end NUMINAMATH_GPT_prime_base_values_l2408_240802


namespace NUMINAMATH_GPT_tan_ratio_triangle_area_l2408_240860

theorem tan_ratio (a b c A B C : ℝ) (h1 : c = -3 * b * Real.cos A) :
  Real.tan A / Real.tan B = -4 := by
  sorry

theorem triangle_area (a b c A B C : ℝ) (h1 : c = -3 * b * Real.cos A)
  (h2 : c = 2) (h3 : Real.tan C = 3 / 4) :
  ∃ S : ℝ, S = 1 / 2 * b * c * Real.sin A ∧ S = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_ratio_triangle_area_l2408_240860


namespace NUMINAMATH_GPT_remainder_of_m_l2408_240861

theorem remainder_of_m (m : ℕ) (h₁ : m ^ 3 % 7 = 6) (h₂ : m ^ 4 % 7 = 4) : m % 7 = 3 := 
sorry

end NUMINAMATH_GPT_remainder_of_m_l2408_240861


namespace NUMINAMATH_GPT_product_of_consecutive_multiples_of_4_divisible_by_192_l2408_240819

theorem product_of_consecutive_multiples_of_4_divisible_by_192 :
  ∀ (n : ℤ), 192 ∣ (4 * n) * (4 * (n + 1)) * (4 * (n + 2)) :=
by
  intro n
  sorry

end NUMINAMATH_GPT_product_of_consecutive_multiples_of_4_divisible_by_192_l2408_240819


namespace NUMINAMATH_GPT_plane_arrival_time_l2408_240887

-- Define the conditions
def departure_time := 11 -- common departure time in hours (11:00)
def bus_speed := 100 -- bus speed in km/h
def train_speed := 300 -- train speed in km/h
def plane_speed := 900 -- plane speed in km/h
def bus_arrival := 20 -- bus arrival time in hours (20:00)
def train_arrival := 14 -- train arrival time in hours (14:00)

-- Given these conditions, we need to prove the plane arrival time
theorem plane_arrival_time : (departure_time + (900 / plane_speed)) = 12 := by
  sorry

end NUMINAMATH_GPT_plane_arrival_time_l2408_240887


namespace NUMINAMATH_GPT_anoop_joined_after_6_months_l2408_240855

theorem anoop_joined_after_6_months (arjun_investment : ℕ) (anoop_investment : ℕ) (months_in_year : ℕ)
  (arjun_time : ℕ) (anoop_time : ℕ) :
  arjun_investment * arjun_time = anoop_investment * anoop_time →
  anoop_investment = 2 * arjun_investment →
  arjun_time = months_in_year →
  anoop_time + arjun_time = months_in_year →
  anoop_time = 6 :=
by sorry

end NUMINAMATH_GPT_anoop_joined_after_6_months_l2408_240855


namespace NUMINAMATH_GPT_area_of_trapezoid_EFGH_l2408_240898

noncomputable def length (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

noncomputable def height_FG : ℝ :=
  6 - 2

noncomputable def area_trapezoid (E F G H : ℝ × ℝ) : ℝ :=
  let base1 := length E F
  let base2 := length G H
  let height := height_FG
  1/2 * (base1 + base2) * height

theorem area_of_trapezoid_EFGH :
  area_trapezoid (0, 0) (2, -3) (6, 0) (6, 4) = 2 * (Real.sqrt 13 + 4) :=
by
  sorry

end NUMINAMATH_GPT_area_of_trapezoid_EFGH_l2408_240898


namespace NUMINAMATH_GPT_math_problem_l2408_240895

noncomputable def proof_problem (k : ℝ) (a b k1 k2 : ℝ) : Prop :=
  (a*b) = 7/k ∧ (a + b) = (k-1)/k ∧ (k1^2 - 18*k1 + 1) = 0 ∧ (k2^2 - 18*k2 + 1) = 0 ∧ 
  (a/b + b/a = 3/7) → (k1/k2 + k2/k1 = 322)

theorem math_problem (k a b k1 k2 : ℝ) : proof_problem k a b k1 k2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2408_240895


namespace NUMINAMATH_GPT_simplify_expression_l2408_240866

-- Define the problem and its conditions
theorem simplify_expression :
  (81 * 10^12) / (9 * 10^4) = 900000000 :=
by
  sorry  -- Proof placeholder

end NUMINAMATH_GPT_simplify_expression_l2408_240866


namespace NUMINAMATH_GPT_extended_hexagon_area_l2408_240814

theorem extended_hexagon_area (original_area : ℝ) (side_length_extension : ℝ)
  (original_side_length : ℝ) (new_side_length : ℝ) :
  original_area = 18 ∧ side_length_extension = 1 ∧ original_side_length = 2 
  ∧ new_side_length = original_side_length + 2 * side_length_extension →
  36 = original_area + 6 * (0.5 * side_length_extension * (original_side_length + 1)) := 
sorry

end NUMINAMATH_GPT_extended_hexagon_area_l2408_240814


namespace NUMINAMATH_GPT_red_candies_l2408_240805

theorem red_candies (R Y B : ℕ) 
  (h1 : Y = 3 * R - 20)
  (h2 : B = Y / 2)
  (h3 : R + B = 90) :
  R = 40 :=
by
  sorry

end NUMINAMATH_GPT_red_candies_l2408_240805


namespace NUMINAMATH_GPT_pages_filled_with_images_ratio_l2408_240857

theorem pages_filled_with_images_ratio (total_pages intro_pages text_pages : ℕ) 
  (h_total : total_pages = 98)
  (h_intro : intro_pages = 11)
  (h_text : text_pages = 19)
  (h_blank : 2 * text_pages = total_pages - intro_pages - 2 * text_pages) :
  (total_pages - intro_pages - text_pages - text_pages) / total_pages = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_pages_filled_with_images_ratio_l2408_240857


namespace NUMINAMATH_GPT_average_billboards_per_hour_l2408_240830

def first_hour_billboards : ℕ := 17
def second_hour_billboards : ℕ := 20
def third_hour_billboards : ℕ := 23

theorem average_billboards_per_hour : 
  (first_hour_billboards + second_hour_billboards + third_hour_billboards) / 3 = 20 := 
by
  sorry

end NUMINAMATH_GPT_average_billboards_per_hour_l2408_240830
