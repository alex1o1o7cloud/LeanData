import Mathlib

namespace NUMINAMATH_GPT_sum_of_fourth_powers_l963_96390

theorem sum_of_fourth_powers (n : ℤ) (h : (n - 2)^2 + n^2 + (n + 2)^2 = 2450) :
  (n - 2)^4 + n^4 + (n + 2)^4 = 1881632 :=
sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_l963_96390


namespace NUMINAMATH_GPT_perfect_square_k_value_l963_96300

-- Given condition:
def is_perfect_square (P : ℤ) : Prop := ∃ (z : ℤ), P = z * z

-- Theorem to prove:
theorem perfect_square_k_value (a b k : ℤ) (h : is_perfect_square (4 * a^2 + k * a * b + 9 * b^2)) :
  k = 12 ∨ k = -12 :=
sorry

end NUMINAMATH_GPT_perfect_square_k_value_l963_96300


namespace NUMINAMATH_GPT_find_n_l963_96388

theorem find_n (n : ℕ) 
  (hM : ∀ M, M = n - 7 → 1 ≤ M)
  (hA : ∀ A, A = n - 2 → 1 ≤ A)
  (hT : ∀ M A, M = n - 7 → A = n - 2 → M + A < n) :
  n = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l963_96388


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l963_96347

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l963_96347


namespace NUMINAMATH_GPT_initial_items_in_cart_l963_96382

theorem initial_items_in_cart (deleted_items : ℕ) (items_left : ℕ) (initial_items : ℕ) 
  (h1 : deleted_items = 10) (h2 : items_left = 8) : initial_items = 18 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_initial_items_in_cart_l963_96382


namespace NUMINAMATH_GPT_correct_function_at_x_equals_1_l963_96332

noncomputable def candidate_A (x : ℝ) : ℝ := (x - 1)^3 + 3 * (x - 1)
noncomputable def candidate_B (x : ℝ) : ℝ := 2 * (x - 1)^2
noncomputable def candidate_C (x : ℝ) : ℝ := 2 * (x - 1)
noncomputable def candidate_D (x : ℝ) : ℝ := x - 1

theorem correct_function_at_x_equals_1 :
  (deriv candidate_A 1 = 3) ∧ 
  (deriv candidate_B 1 ≠ 3) ∧ 
  (deriv candidate_C 1 ≠ 3) ∧ 
  (deriv candidate_D 1 ≠ 3) := 
by
  sorry

end NUMINAMATH_GPT_correct_function_at_x_equals_1_l963_96332


namespace NUMINAMATH_GPT_polynomial_has_exactly_one_real_root_l963_96345

theorem polynomial_has_exactly_one_real_root :
  ∀ (x : ℝ), (2007 * x^3 + 2006 * x^2 + 2005 * x = 0) → x = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_has_exactly_one_real_root_l963_96345


namespace NUMINAMATH_GPT_rectangle_area_divisible_by_12_l963_96305

theorem rectangle_area_divisible_by_12 {a b c : ℕ} (h : a ^ 2 + b ^ 2 = c ^ 2) :
  12 ∣ (a * b) :=
sorry

end NUMINAMATH_GPT_rectangle_area_divisible_by_12_l963_96305


namespace NUMINAMATH_GPT_segments_to_start_l963_96355

-- Define the problem statement conditions in Lean 4
def concentric_circles : Prop := sorry -- Placeholder, as geometry involving tangents and arcs isn't directly supported

def chord_tangent_small_circle (AB : Prop) : Prop := sorry -- Placeholder, detailing tangency

def angle_ABC_eq_60 (A B C : Prop) : Prop := sorry -- Placeholder, situating angles in terms of Lean formalism

-- Proof statement
theorem segments_to_start (A B C : Prop) :
  concentric_circles →
  chord_tangent_small_circle (A ↔ B) →
  chord_tangent_small_circle (B ↔ C) →
  angle_ABC_eq_60 A B C →
  ∃ n : ℕ, n = 3 :=
sorry

end NUMINAMATH_GPT_segments_to_start_l963_96355


namespace NUMINAMATH_GPT_lucas_income_36000_l963_96316

variable (q I : ℝ)

-- Conditions as Lean 4 definitions
def tax_below_30000 : ℝ := 0.01 * q * 30000
def tax_above_30000 (I : ℝ) : ℝ := 0.01 * (q + 3) * (I - 30000)
def total_tax (I : ℝ) : ℝ := tax_below_30000 q + tax_above_30000 q I
def total_tax_condition (I : ℝ) : Prop := total_tax q I = 0.01 * (q + 0.5) * I

theorem lucas_income_36000 (h : total_tax_condition q I) : I = 36000 := by
  sorry

end NUMINAMATH_GPT_lucas_income_36000_l963_96316


namespace NUMINAMATH_GPT_train_speed_l963_96398

-- Define the conditions
def train_length : ℝ := 50 -- Length of the train in meters
def crossing_time : ℝ := 3 -- Time to cross the pole in seconds

-- Define the speed in meters per second and convert it to km/hr
noncomputable def speed_mps : ℝ := train_length / crossing_time
noncomputable def speed_kmph : ℝ := speed_mps * 3.6 -- Conversion factor

-- Theorem statement: Prove that the calculated speed in km/hr is 60 km/hr
theorem train_speed : speed_kmph = 60 := by
  sorry

end NUMINAMATH_GPT_train_speed_l963_96398


namespace NUMINAMATH_GPT_tom_and_jerry_drank_80_ounces_l963_96387

theorem tom_and_jerry_drank_80_ounces
    (T J : ℝ) 
    (initial_T : T = 40)
    (initial_J : J = 2 * T)
    (T_drank J_drank : ℝ)
    (T_remaining J_remaining : ℝ)
    (T_after_pour J_after_pour : ℝ)
    (T_final J_final : ℝ)
    (H1 : T_drank = (2 / 3) * T)
    (H2 : J_drank = (2 / 3) * J)
    (H3 : T_remaining = T - T_drank)
    (H4 : J_remaining = J - J_drank)
    (H5 : T_after_pour = T_remaining + (1 / 4) * J_remaining)
    (H6 : J_after_pour = J_remaining - (1 / 4) * J_remaining)
    (H7 : T_final = T_after_pour - 5)
    (H8 : J_final = J_after_pour + 5)
    (H9 : T_final = J_final + 4)
    : T_drank + J_drank = 80 :=
by
  sorry

end NUMINAMATH_GPT_tom_and_jerry_drank_80_ounces_l963_96387


namespace NUMINAMATH_GPT_quadratic_value_at_point_l963_96325

variable (a b c : ℝ)

-- Given: A quadratic function f(x) = ax^2 + bx + c that passes through the point (3,10)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_value_at_point
  (h : f a b c 3 = 10) :
  5 * a - 3 * b + c = -4 * a - 6 * b + 10 := by
  sorry

end NUMINAMATH_GPT_quadratic_value_at_point_l963_96325


namespace NUMINAMATH_GPT_leftovers_value_l963_96331

def quarters_in_roll : ℕ := 30
def dimes_in_roll : ℕ := 40
def james_quarters : ℕ := 77
def james_dimes : ℕ := 138
def lindsay_quarters : ℕ := 112
def lindsay_dimes : ℕ := 244
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftovers_value :
  let total_quarters := james_quarters + lindsay_quarters
  let total_dimes := james_dimes + lindsay_dimes
  let leftover_quarters := total_quarters % quarters_in_roll
  let leftover_dimes := total_dimes % dimes_in_roll
  leftover_quarters * quarter_value + leftover_dimes * dime_value = 2.45 :=
by
  sorry

end NUMINAMATH_GPT_leftovers_value_l963_96331


namespace NUMINAMATH_GPT_find_fraction_l963_96308

theorem find_fraction (f n : ℝ) (h1 : f * n - 5 = 5) (h2 : n = 50) : f = 1 / 5 :=
by
  -- skipping the proof as requested
  sorry

end NUMINAMATH_GPT_find_fraction_l963_96308


namespace NUMINAMATH_GPT_garden_perimeter_l963_96326

theorem garden_perimeter
  (width_garden : ℝ) (area_playground : ℝ)
  (length_playground : ℝ) (width_playground : ℝ)
  (area_garden : ℝ) (L : ℝ)
  (h1 : width_garden = 4) 
  (h2 : length_playground = 16)
  (h3 : width_playground = 12)
  (h4 : area_playground = length_playground * width_playground)
  (h5 : area_garden = area_playground)
  (h6 : area_garden = L * width_garden) :
  2 * L + 2 * width_garden = 104 :=
by
  sorry

end NUMINAMATH_GPT_garden_perimeter_l963_96326


namespace NUMINAMATH_GPT_find_AD_l963_96309

-- Defining points and distances in the context of a triangle
variables {A B C D: Type*}
variables (dist_AB : ℝ) (dist_AC : ℝ) (dist_BC : ℝ) (midpoint_D : Prop)

-- Given conditions
def triangle_conditions : Prop :=
  dist_AB = 26 ∧
  dist_AC = 26 ∧
  dist_BC = 24 ∧
  midpoint_D

-- Problem statement as a Lean theorem
theorem find_AD
  (h : triangle_conditions dist_AB dist_AC dist_BC midpoint_D) :
  ∃ (AD : ℝ), AD = 2 * Real.sqrt 133 :=
sorry

end NUMINAMATH_GPT_find_AD_l963_96309


namespace NUMINAMATH_GPT_actual_average_height_correct_l963_96360

noncomputable def actual_average_height (n : ℕ) (average_height : ℝ) (wrong_height : ℝ) (actual_height : ℝ) : ℝ :=
  let total_height := average_height * n
  let difference := wrong_height - actual_height
  let correct_total_height := total_height - difference
  correct_total_height / n

theorem actual_average_height_correct :
  actual_average_height 35 184 166 106 = 182.29 :=
by
  sorry

end NUMINAMATH_GPT_actual_average_height_correct_l963_96360


namespace NUMINAMATH_GPT_find_x_l963_96302

theorem find_x :
  ∃ x : ℝ, 12.1212 + x - 9.1103 = 20.011399999999995 ∧ x = 18.000499999999995 :=
sorry

end NUMINAMATH_GPT_find_x_l963_96302


namespace NUMINAMATH_GPT_cos120_sin_neg45_equals_l963_96301

noncomputable def cos120_plus_sin_neg45 : ℝ :=
  Real.cos (120 * Real.pi / 180) + Real.sin (-45 * Real.pi / 180)

theorem cos120_sin_neg45_equals : cos120_plus_sin_neg45 = - (1 + Real.sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos120_sin_neg45_equals_l963_96301


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l963_96324

variable (A : Set ℕ) (B : Set ℕ)

axiom h1 : A = {1, 2, 3, 4, 5}
axiom h2 : B = {3, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} :=
  by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l963_96324


namespace NUMINAMATH_GPT_product_primes_less_than_20_l963_96310

theorem product_primes_less_than_20 :
  (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 = 9699690) :=
by
  sorry

end NUMINAMATH_GPT_product_primes_less_than_20_l963_96310


namespace NUMINAMATH_GPT_josie_remaining_money_l963_96311

-- Conditions
def initial_amount : ℕ := 50
def cassette_tape_cost : ℕ := 9
def headphone_cost : ℕ := 25

-- Proof statement
theorem josie_remaining_money : initial_amount - (2 * cassette_tape_cost + headphone_cost) = 7 :=
by
  sorry

end NUMINAMATH_GPT_josie_remaining_money_l963_96311


namespace NUMINAMATH_GPT_unique_zero_iff_a_eq_half_l963_96334

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + a * (Real.exp (x - 1) + Real.exp (1 - x))

theorem unique_zero_iff_a_eq_half :
  (∃! x : ℝ, f x a = 0) ↔ a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_zero_iff_a_eq_half_l963_96334


namespace NUMINAMATH_GPT_meters_to_examine_10000_l963_96320

def projection_for_sample (total_meters_examined : ℕ) (rejection_rate : ℝ) (sample_size : ℕ) :=
  total_meters_examined = sample_size

theorem meters_to_examine_10000 : 
  projection_for_sample 10000 0.015 10000 := by
  sorry

end NUMINAMATH_GPT_meters_to_examine_10000_l963_96320


namespace NUMINAMATH_GPT_hexagon_perimeter_l963_96318

theorem hexagon_perimeter (s : ℝ) (h_area : s ^ 2 * (3 * Real.sqrt 3 / 2) = 54 * Real.sqrt 3) :
  6 * s = 36 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_l963_96318


namespace NUMINAMATH_GPT_value_of_expression_l963_96367

theorem value_of_expression (x : ℝ) (hx : x = -2) : (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l963_96367


namespace NUMINAMATH_GPT_marble_theorem_l963_96373

noncomputable def marble_problem (M : ℝ) : Prop :=
  let M_Pedro : ℝ := 0.7 * M
  let M_Ebony : ℝ := 0.85 * M_Pedro
  let M_Jimmy : ℝ := 0.7 * M_Ebony
  (M_Jimmy / M) * 100 = 41.65

theorem marble_theorem (M : ℝ) : marble_problem M := 
by
  sorry

end NUMINAMATH_GPT_marble_theorem_l963_96373


namespace NUMINAMATH_GPT_k_starts_at_10_l963_96343

variable (V_k V_l : ℝ)
variable (t_k t_l : ℝ)

-- Conditions
axiom k_faster_than_l : V_k = 1.5 * V_l
axiom l_speed : V_l = 50
axiom l_start_time : t_l = 9
axiom meet_time : t_k + 3 = 12
axiom distance_apart : V_l * 3 + V_k * (12 - t_k) = 300

-- Proof goal
theorem k_starts_at_10 : t_k = 10 :=
by
  sorry

end NUMINAMATH_GPT_k_starts_at_10_l963_96343


namespace NUMINAMATH_GPT_tunnel_length_l963_96339

def train_length : ℝ := 1.5
def exit_time_minutes : ℝ := 4
def speed_mph : ℝ := 45

theorem tunnel_length (d_train : ℝ := train_length)
                      (t_exit : ℝ := exit_time_minutes)
                      (v_mph : ℝ := speed_mph) :
  d_train + ((v_mph / 60) * t_exit - d_train) = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_tunnel_length_l963_96339


namespace NUMINAMATH_GPT_students_total_l963_96314

theorem students_total (T : ℝ) (h₁ : 0.675 * T = 594) : T = 880 :=
sorry

end NUMINAMATH_GPT_students_total_l963_96314


namespace NUMINAMATH_GPT_polygon_perimeter_l963_96379

theorem polygon_perimeter :
  let AB := 2
  let BC := 2
  let CD := 2
  let DE := 2
  let EF := 2
  let FG := 3
  let GH := 3
  let HI := 3
  let IJ := 3
  let JA := 4
  AB + BC + CD + DE + EF + FG + GH + HI + IJ + JA = 26 :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_perimeter_l963_96379


namespace NUMINAMATH_GPT_pages_left_l963_96372

variable (a b : ℕ)

theorem pages_left (a b : ℕ) : a - 8 * b = a - 8 * b :=
by
  sorry

end NUMINAMATH_GPT_pages_left_l963_96372


namespace NUMINAMATH_GPT_six_digit_number_divisible_9_22_l963_96344

theorem six_digit_number_divisible_9_22 (d : ℕ) (h0 : 0 ≤ d) (h1 : d ≤ 9)
  (h2 : 9 ∣ (220140 + d)) (h3 : 22 ∣ (220140 + d)) : 220140 + d = 520146 :=
sorry

end NUMINAMATH_GPT_six_digit_number_divisible_9_22_l963_96344


namespace NUMINAMATH_GPT_dandelions_survive_to_flower_l963_96327

def seeds_initial : ℕ := 300
def seeds_in_water : ℕ := seeds_initial / 3
def seeds_eaten_by_insects : ℕ := seeds_initial / 6
def seeds_remaining : ℕ := seeds_initial - seeds_in_water - seeds_eaten_by_insects
def seeds_to_flower : ℕ := seeds_remaining / 2

theorem dandelions_survive_to_flower : seeds_to_flower = 75 := by
  sorry

end NUMINAMATH_GPT_dandelions_survive_to_flower_l963_96327


namespace NUMINAMATH_GPT_remainder_when_divided_by_385_l963_96386

theorem remainder_when_divided_by_385 (x : ℤ)
  (h1 : 2 + x ≡ 4 [ZMOD 125])
  (h2 : 3 + x ≡ 9 [ZMOD 343])
  (h3 : 4 + x ≡ 25 [ZMOD 1331]) :
  x ≡ 307 [ZMOD 385] :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_385_l963_96386


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l963_96371

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 400) (hS : S = 2500) (hS_eq : S = a / (1 - r)) : r = 21 / 25 :=
by
  rw [ha, hS] at hS_eq
  -- This statement follows from algebraic manipulation outlined in the solution steps.
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l963_96371


namespace NUMINAMATH_GPT_kangaroo_jump_is_8_5_feet_longer_l963_96330

noncomputable def camel_step_length (total_distance : ℝ) (num_steps : ℕ) : ℝ := total_distance / num_steps
noncomputable def kangaroo_jump_length (total_distance : ℝ) (num_jumps : ℕ) : ℝ := total_distance / num_jumps
noncomputable def length_difference (jump_length step_length : ℝ) : ℝ := jump_length - step_length

theorem kangaroo_jump_is_8_5_feet_longer :
  let total_distance := 7920
  let num_gaps := 50
  let camel_steps_per_gap := 56
  let kangaroo_jumps_per_gap := 14
  let num_camel_steps := num_gaps * camel_steps_per_gap
  let num_kangaroo_jumps := num_gaps * kangaroo_jumps_per_gap
  let camel_step := camel_step_length total_distance num_camel_steps
  let kangaroo_jump := kangaroo_jump_length total_distance num_kangaroo_jumps
  length_difference kangaroo_jump camel_step = 8.5 := sorry

end NUMINAMATH_GPT_kangaroo_jump_is_8_5_feet_longer_l963_96330


namespace NUMINAMATH_GPT_number_of_ways_to_choose_one_person_l963_96365

-- Definitions for the conditions
def people_using_first_method : ℕ := 3
def people_using_second_method : ℕ := 5

-- Definition of the total number of ways to choose one person
def total_ways_to_choose_one_person : ℕ :=
  people_using_first_method + people_using_second_method

-- Statement of the theorem to be proved
theorem number_of_ways_to_choose_one_person :
  total_ways_to_choose_one_person = 8 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_one_person_l963_96365


namespace NUMINAMATH_GPT_sum_of_first_five_terms_l963_96393

theorem sum_of_first_five_terms : 
  ∀ (S : ℕ → ℕ) (a : ℕ → ℕ), 
    (a 1 = 1) ∧ 
    (∀ n ≥ 2, S n = S (n - 1) + n + 2) → 
    S 5 = 23 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_five_terms_l963_96393


namespace NUMINAMATH_GPT_arithmetic_expression_value_l963_96317

theorem arithmetic_expression_value :
  15 * 36 + 15 * 3^3 = 945 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_value_l963_96317


namespace NUMINAMATH_GPT_millimeters_of_78_74_inches_l963_96306

noncomputable def inchesToMillimeters (inches : ℝ) : ℝ :=
  inches * 25.4

theorem millimeters_of_78_74_inches :
  round (inchesToMillimeters 78.74) = 2000 :=
by
  -- This theorem should assert that converting 78.74 inches to millimeters and rounding to the nearest millimeter equals 2000
  sorry

end NUMINAMATH_GPT_millimeters_of_78_74_inches_l963_96306


namespace NUMINAMATH_GPT_linear_function_correct_max_profit_correct_min_selling_price_correct_l963_96341

-- Definition of the linear function
def linear_function (x : ℝ) : ℝ :=
  -2 * x + 360

-- Definition of monthly profit function
def profit_function (x : ℝ) : ℝ :=
  (-2 * x + 360) * (x - 30)

noncomputable def max_profit_statement : Prop :=
  ∃ x w, x = 105 ∧ w = 11250 ∧ profit_function x = w

noncomputable def min_selling_price (profit : ℝ) : Prop :=
  ∃ x, profit_function x ≥ profit ∧ x ≥ 80

-- The proof statements
theorem linear_function_correct : linear_function 30 = 300 ∧ linear_function 45 = 270 :=
  by
    sorry

theorem max_profit_correct : max_profit_statement :=
  by
    sorry

theorem min_selling_price_correct : min_selling_price 10000 :=
  by
    sorry

end NUMINAMATH_GPT_linear_function_correct_max_profit_correct_min_selling_price_correct_l963_96341


namespace NUMINAMATH_GPT_expression_value_l963_96315

theorem expression_value (x a b c : ℝ) 
  (ha : a + x^2 = 2006) 
  (hb : b + x^2 = 2007) 
  (hc : c + x^2 = 2008) 
  (h_abc : a * b * c = 3) :
  (a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1) := 
  sorry

end NUMINAMATH_GPT_expression_value_l963_96315


namespace NUMINAMATH_GPT_tennis_tournament_possible_l963_96389

theorem tennis_tournament_possible (p : ℕ) : 
  (∀ i j : ℕ, i ≠ j → ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  i = a ∨ i = b ∨ i = c ∨ i = d ∧ j = a ∨ j = b ∨ j = c ∨ j = d) → 
  ∃ k : ℕ, p = 8 * k + 1 := by
  sorry

end NUMINAMATH_GPT_tennis_tournament_possible_l963_96389


namespace NUMINAMATH_GPT_find_c_plus_d_l963_96383

noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then 2 * c * x + d else 9 - 2 * x

theorem find_c_plus_d (c d : ℝ) (h : ∀ x : ℝ, f c d (f c d x) = x) : c + d = 4.25 :=
by
  sorry

end NUMINAMATH_GPT_find_c_plus_d_l963_96383


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l963_96368

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (a_1 d : ℤ) 
  (h1: S 3 = (3 * a_1) + (3 * (2 * d) / 2))
  (h2: S 7 = (7 * a_1) + (7 * (6 * d) / 2)) :
  S 5 = (5 * a_1) + (5 * (4 * d) / 2) := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l963_96368


namespace NUMINAMATH_GPT_additional_telephone_lines_l963_96396

def telephone_lines_increase : ℕ :=
  let lines_six_digits := 9 * 10^5
  let lines_seven_digits := 9 * 10^6
  lines_seven_digits - lines_six_digits

theorem additional_telephone_lines : telephone_lines_increase = 81 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_additional_telephone_lines_l963_96396


namespace NUMINAMATH_GPT_average_difference_l963_96340

theorem average_difference :
  let avg1 := (24 + 35 + 58) / 3
  let avg2 := (19 + 51 + 29) / 3
  avg1 - avg2 = 6 := by
sorry

end NUMINAMATH_GPT_average_difference_l963_96340


namespace NUMINAMATH_GPT_total_guests_l963_96399

theorem total_guests (G : ℕ) 
  (hwomen: ∃ n, n = G / 2)
  (hmen: 15 = 15)
  (hchildren: ∃ n, n = G - (G / 2 + 15))
  (men_leaving: ∃ n, n = 1/5 * 15)
  (children_leaving: 4 = 4)
  (people_stayed: 43 = G - ((1/5 * 15) + 4))
  : G = 50 := by
  sorry

end NUMINAMATH_GPT_total_guests_l963_96399


namespace NUMINAMATH_GPT_maximum_value_of_function_l963_96303

noncomputable def f (x : ℝ) : ℝ := 10 * x - 4 * x^2

theorem maximum_value_of_function :
  ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f x_max = 25 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_maximum_value_of_function_l963_96303


namespace NUMINAMATH_GPT_k_value_if_root_is_one_l963_96342

theorem k_value_if_root_is_one (k : ℝ) (h : (k - 1) * 1 ^ 2 + 1 - k ^ 2 = 0) : k = 0 := 
by
  sorry

end NUMINAMATH_GPT_k_value_if_root_is_one_l963_96342


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l963_96392

theorem solution_set_of_quadratic_inequality (x : ℝ) : x^2 < x + 6 ↔ -2 < x ∧ x < 3 := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l963_96392


namespace NUMINAMATH_GPT_point_M_coordinates_l963_96328

theorem point_M_coordinates :
  (∃ (M : ℝ × ℝ), M.1 < 0 ∧ M.2 > 0 ∧ abs M.2 = 2 ∧ abs M.1 = 1 ∧ M = (-1, 2)) :=
by
  use (-1, 2)
  sorry

end NUMINAMATH_GPT_point_M_coordinates_l963_96328


namespace NUMINAMATH_GPT_possible_values_of_P_l963_96304

-- Definition of the conditions
variables (x y : ℕ) (h1 : x < y) (h2 : (x > 0)) (h3 : (y > 0))

-- Definition of P
def P : ℤ := (x^3 - y) / (1 + x * y)

-- Theorem statement
theorem possible_values_of_P : (P = 0) ∨ (P ≥ 2) :=
sorry

end NUMINAMATH_GPT_possible_values_of_P_l963_96304


namespace NUMINAMATH_GPT_pureAcidInSolution_l963_96337

/-- Define the conditions for the problem -/
def totalVolume : ℝ := 12
def percentageAcid : ℝ := 0.40

/-- State the theorem equivalent to the question:
    calculate the amount of pure acid -/
theorem pureAcidInSolution :
  totalVolume * percentageAcid = 4.8 := by
  sorry

end NUMINAMATH_GPT_pureAcidInSolution_l963_96337


namespace NUMINAMATH_GPT_solution_l963_96307

axiom f : ℝ → ℝ

def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)

def decreasing_function (f : ℝ → ℝ) := ∀ x y, x < y → y ≤ 0 → f x > f y

def main_problem : Prop :=
  even_function f ∧ decreasing_function f ∧ f (-2) = 0 → ∀ x, f x < 0 ↔ x > -2 ∧ x < 2

theorem solution : main_problem :=
by
  sorry

end NUMINAMATH_GPT_solution_l963_96307


namespace NUMINAMATH_GPT_notecard_area_new_dimension_l963_96376

theorem notecard_area_new_dimension :
  ∀ (length : ℕ) (width : ℕ) (shortened : ℕ),
    length = 7 →
    width = 5 →
    shortened = 2 →
    (width - shortened) * length = 21 →
    (length - shortened) * (width - shortened + shortened) = 25 :=
by
  intros length width shortened h_length h_width h_shortened h_area
  sorry

end NUMINAMATH_GPT_notecard_area_new_dimension_l963_96376


namespace NUMINAMATH_GPT_sign_of_slope_equals_sign_of_correlation_l963_96362

-- Definitions for conditions
def linear_relationship (x y : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ t, y t = a + b * x t

def correlation_coefficient (x y : ℝ → ℝ) (r : ℝ) : Prop :=
  r > -1 ∧ r < 1 ∧ ∀ t t', (y t - y t').sign = (x t - x t').sign

def regression_line_slope (b : ℝ) : Prop := True

-- Theorem to prove the sign of b is equal to the sign of r
theorem sign_of_slope_equals_sign_of_correlation (x y : ℝ → ℝ) (r b : ℝ) 
  (h1 : linear_relationship x y) 
  (h2 : correlation_coefficient x y r) 
  (h3 : regression_line_slope b) : 
  b.sign = r.sign := 
sorry

end NUMINAMATH_GPT_sign_of_slope_equals_sign_of_correlation_l963_96362


namespace NUMINAMATH_GPT_scientific_notation_of_twenty_million_l963_96338

-- Define the number 20 million
def twenty_million : ℂ :=
  20000000

-- Define the scientific notation to be proved correct
def scientific_notation : ℂ :=
  2 * 10 ^ 7

-- The theorem to prove the equivalence
theorem scientific_notation_of_twenty_million : twenty_million = scientific_notation :=
  sorry

end NUMINAMATH_GPT_scientific_notation_of_twenty_million_l963_96338


namespace NUMINAMATH_GPT_expected_pairs_socks_l963_96313

noncomputable def expected_socks_to_pair (p : ℕ) : ℕ :=
2 * p

theorem expected_pairs_socks (p : ℕ) : 
  (expected_socks_to_pair p) = 2 * p := 
by 
  sorry

end NUMINAMATH_GPT_expected_pairs_socks_l963_96313


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l963_96353

theorem sufficient_but_not_necessary (a b : ℝ) : (a > |b|) → (a^2 > b^2) ∧ ¬((a^2 > b^2) → (a > |b|)) := 
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l963_96353


namespace NUMINAMATH_GPT_transport_load_with_trucks_l963_96348

theorem transport_load_with_trucks
  (total_weight : ℕ)
  (box_max_weight : ℕ)
  (truck_capacity : ℕ)
  (num_trucks : ℕ)
  (H_weight : total_weight = 13500)
  (H_box : box_max_weight = 350)
  (H_truck : truck_capacity = 1500)
  (H_num_trucks : num_trucks = 11) :
  ∃ (boxes : ℕ), boxes * box_max_weight >= total_weight ∧ num_trucks * truck_capacity >= total_weight := 
sorry

end NUMINAMATH_GPT_transport_load_with_trucks_l963_96348


namespace NUMINAMATH_GPT_frank_oranges_correct_l963_96397

def betty_oranges : ℕ := 12
def sandra_oranges : ℕ := 3 * betty_oranges
def emily_oranges : ℕ := 7 * sandra_oranges
def frank_oranges : ℕ := 5 * emily_oranges

theorem frank_oranges_correct : frank_oranges = 1260 := by
  sorry

end NUMINAMATH_GPT_frank_oranges_correct_l963_96397


namespace NUMINAMATH_GPT_nine_pow_n_sub_one_l963_96329

theorem nine_pow_n_sub_one (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ (p1 p2 p3 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ (9^n - 1) = p1 * p2 * p3 ∧ (p1 = 61 ∨ p2 = 61 ∨ p3 = 61)) : 9^n - 1 = 59048 := 
sorry

end NUMINAMATH_GPT_nine_pow_n_sub_one_l963_96329


namespace NUMINAMATH_GPT_custom_op_subtraction_l963_96336

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_subtraction :
  (custom_op 4 2) - (custom_op 2 4) = -8 := by
  sorry

end NUMINAMATH_GPT_custom_op_subtraction_l963_96336


namespace NUMINAMATH_GPT_set_swept_by_all_lines_l963_96395

theorem set_swept_by_all_lines
  (a c x y : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < c)
  (h3 : c < a)
  (h4 : x^2 + y^2 ≤ a^2) : 
  (c^2 - a^2) * x^2 - a^2 * y^2 ≤ (c^2 - a^2) * c^2 :=
sorry

end NUMINAMATH_GPT_set_swept_by_all_lines_l963_96395


namespace NUMINAMATH_GPT_percent_of_a_is_b_l963_96377

theorem percent_of_a_is_b (a b c : ℝ) (h1 : c = 0.30 * a) (h2 : c = 0.25 * b) : b = 1.2 * a :=
by
  -- proof 
  sorry

end NUMINAMATH_GPT_percent_of_a_is_b_l963_96377


namespace NUMINAMATH_GPT_min_value_l963_96333

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 3 + 2 * Real.sqrt 2 ≤ 2 / a + 1 / b :=
by
  sorry

end NUMINAMATH_GPT_min_value_l963_96333


namespace NUMINAMATH_GPT_marcy_needs_6_tubs_of_lip_gloss_l963_96374

theorem marcy_needs_6_tubs_of_lip_gloss (people tubes_per_person tubes_per_tub : ℕ) 
  (h1 : people = 36) (h2 : tubes_per_person = 3) (h3 : tubes_per_tub = 2) :
  (people / tubes_per_person) / tubes_per_tub = 6 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_marcy_needs_6_tubs_of_lip_gloss_l963_96374


namespace NUMINAMATH_GPT_mirror_side_length_l963_96364

theorem mirror_side_length (width length : ℝ) (area_wall : ℝ) (area_mirror : ℝ) (side_length : ℝ) 
  (h1 : width = 28) 
  (h2 : length = 31.5) 
  (h3 : area_wall = width * length)
  (h4 : area_mirror = area_wall / 2) 
  (h5 : area_mirror = side_length ^ 2) : 
  side_length = 21 := 
by 
  sorry

end NUMINAMATH_GPT_mirror_side_length_l963_96364


namespace NUMINAMATH_GPT_determinant_of_given_matrix_l963_96321

noncomputable def given_matrix : Matrix (Fin 4) (Fin 4) ℤ :=
![![1, -3, 3, 2], ![0, 5, -1, 0], ![4, -2, 1, 0], ![0, 0, 0, 6]]

theorem determinant_of_given_matrix :
  Matrix.det given_matrix = -270 := by
  sorry

end NUMINAMATH_GPT_determinant_of_given_matrix_l963_96321


namespace NUMINAMATH_GPT_alcohol_quantity_l963_96361

theorem alcohol_quantity (A W : ℝ) (h1 : A / W = 2 / 5) (h2 : A / (W + 10) = 2 / 7) : A = 10 :=
by
  sorry

end NUMINAMATH_GPT_alcohol_quantity_l963_96361


namespace NUMINAMATH_GPT_average_salary_of_associates_l963_96319

theorem average_salary_of_associates 
  (num_managers : ℕ) (num_associates : ℕ)
  (avg_salary_managers : ℝ) (avg_salary_company : ℝ)
  (H_num_managers : num_managers = 15)
  (H_num_associates : num_associates = 75)
  (H_avg_salary_managers : avg_salary_managers = 90000)
  (H_avg_salary_company : avg_salary_company = 40000) :
  ∃ (A : ℝ), (num_managers * avg_salary_managers + num_associates * A) / (num_managers + num_associates) = avg_salary_company ∧ A = 30000 := by
  sorry

end NUMINAMATH_GPT_average_salary_of_associates_l963_96319


namespace NUMINAMATH_GPT_daily_rate_is_three_l963_96366

theorem daily_rate_is_three (r : ℝ) : 
  (∀ (initial bedbugs : ℝ), initial = 30 ∧ 
  (∀ days later_bedbugs, days = 4 ∧ later_bedbugs = 810 →
  later_bedbugs = initial * r ^ days)) → r = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_daily_rate_is_three_l963_96366


namespace NUMINAMATH_GPT_cake_fraction_eaten_l963_96322

theorem cake_fraction_eaten (total_slices kept_slices slices_eaten : ℕ) 
  (h1 : total_slices = 12)
  (h2 : kept_slices = 9)
  (h3 : slices_eaten = total_slices - kept_slices) :
  (slices_eaten : ℚ) / total_slices = 1 / 4 := 
sorry

end NUMINAMATH_GPT_cake_fraction_eaten_l963_96322


namespace NUMINAMATH_GPT_solution_set_of_inequality_l963_96394

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 2*x + 3 > 0 ↔ (-1 < x ∧ x < 3) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l963_96394


namespace NUMINAMATH_GPT_bounded_sequence_iff_l963_96357

theorem bounded_sequence_iff (x : ℕ → ℝ) (h : ∀ n, x (n + 1) = (n^2 + 1) * x n ^ 2 / (x n ^ 3 + n^2)) :
  (∃ C, ∀ n, x n < C) ↔ (0 < x 0 ∧ x 0 ≤ (Real.sqrt 5 - 1) / 2) ∨ x 0 ≥ 1 := sorry

end NUMINAMATH_GPT_bounded_sequence_iff_l963_96357


namespace NUMINAMATH_GPT_average_speed_of_trip_l963_96369

theorem average_speed_of_trip (d1 d2 s1 s2 : ℕ)
  (h1 : d1 = 30) (h2 : d2 = 30)
  (h3 : s1 = 60) (h4 : s2 = 30) :
  (d1 + d2) / (d1 / s1 + d2 / s2) = 40 :=
by sorry

end NUMINAMATH_GPT_average_speed_of_trip_l963_96369


namespace NUMINAMATH_GPT_find_a_l963_96378

theorem find_a (a x : ℝ) (h : x = 1) (h_eq : 2 - 3 * (a + x) = 2 * x) : a = -1 := by
  sorry

end NUMINAMATH_GPT_find_a_l963_96378


namespace NUMINAMATH_GPT_slope_of_line_l963_96312

theorem slope_of_line (x1 x2 y1 y2 : ℝ) (h1 : 1 = (x1 + x2) / 2) (h2 : 1 = (y1 + y2) / 2) 
                      (h3 : (x1^2 / 36) + (y1^2 / 9) = 1) (h4 : (x2^2 / 36) + (y2^2 / 9) = 1) :
  (y2 - y1) / (x2 - x1) = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l963_96312


namespace NUMINAMATH_GPT_calculate_rent_is_correct_l963_96385

noncomputable def requiredMonthlyRent 
  (purchase_cost : ℝ) 
  (monthly_set_aside_percent : ℝ)
  (annual_property_tax : ℝ)
  (annual_insurance : ℝ)
  (annual_return_percent : ℝ) : ℝ :=
  let annual_return := annual_return_percent * purchase_cost
  let total_yearly_expenses := annual_return + annual_property_tax + annual_insurance
  let monthly_expenses := total_yearly_expenses / 12
  let retention_rate := 1 - monthly_set_aside_percent
  monthly_expenses / retention_rate

theorem calculate_rent_is_correct 
  (purchase_cost : ℝ := 200000)
  (monthly_set_aside_percent : ℝ := 0.2)
  (annual_property_tax : ℝ := 5000)
  (annual_insurance : ℝ := 2400)
  (annual_return_percent : ℝ := 0.08) :
  requiredMonthlyRent purchase_cost monthly_set_aside_percent annual_property_tax annual_insurance annual_return_percent = 2437.50 :=
by
  sorry

end NUMINAMATH_GPT_calculate_rent_is_correct_l963_96385


namespace NUMINAMATH_GPT_contrapositive_false_l963_96350

theorem contrapositive_false : ¬ (∀ x : ℝ, x^2 = 1 → x = 1) → ∀ x : ℝ, x^2 = 1 → x ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_false_l963_96350


namespace NUMINAMATH_GPT_min_value_expr_l963_96384

theorem min_value_expr :
  ∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 :=
by sorry

end NUMINAMATH_GPT_min_value_expr_l963_96384


namespace NUMINAMATH_GPT_at_least_one_gt_one_l963_96363

variable (a b : ℝ)

theorem at_least_one_gt_one (h : a + b > 2) : a > 1 ∨ b > 1 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_gt_one_l963_96363


namespace NUMINAMATH_GPT_number_of_boys_l963_96349

def school_problem (x y : ℕ) : Prop :=
  (x + y = 400) ∧ (y = (x / 100) * 400)

theorem number_of_boys (x y : ℕ) (h : school_problem x y) : x = 80 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_l963_96349


namespace NUMINAMATH_GPT_sum_of_consecutive_odd_integers_l963_96380

-- Definitions of conditions
def consecutive_odd_integers (a b : ℤ) : Prop :=
  b = a + 2 ∧ (a % 2 = 1) ∧ (b % 2 = 1)

def five_times_smaller_minus_two_condition (a b : ℤ) : Prop :=
  b = 5 * a - 2

-- Theorem statement
theorem sum_of_consecutive_odd_integers (a b : ℤ)
  (h1 : consecutive_odd_integers a b)
  (h2 : five_times_smaller_minus_two_condition a b) : a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_odd_integers_l963_96380


namespace NUMINAMATH_GPT_initial_birds_l963_96352

-- Define the initial number of birds (B) and the fact that 13 more birds flew up to the tree
-- Define that the total number of birds after 13 more birds joined is 42
theorem initial_birds (B : ℕ) (h : B + 13 = 42) : B = 29 :=
by
  sorry

end NUMINAMATH_GPT_initial_birds_l963_96352


namespace NUMINAMATH_GPT_total_travel_time_l963_96335

theorem total_travel_time (distance1 distance2 speed time1: ℕ) (h1 : distance1 = 100) (h2 : time1 = 1) (h3 : distance2 = 300) (h4 : speed = distance1 / time1) :
  (time1 + distance2 / speed) = 4 :=
by
  sorry

end NUMINAMATH_GPT_total_travel_time_l963_96335


namespace NUMINAMATH_GPT_length_more_than_breadth_l963_96391

theorem length_more_than_breadth (b : ℝ) (x : ℝ) 
  (h1 : b + x = 55) 
  (h2 : 4 * b + 2 * x = 200) 
  (h3 : (5300 : ℝ) / 26.5 = 200)
  : x = 10 := 
by
  sorry

end NUMINAMATH_GPT_length_more_than_breadth_l963_96391


namespace NUMINAMATH_GPT_a_eq_zero_iff_purely_imaginary_l963_96358

open Complex

noncomputable def purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem a_eq_zero_iff_purely_imaginary (a b : ℝ) :
  (a = 0) ↔ purely_imaginary (a + b * Complex.I) :=
by
  sorry

end NUMINAMATH_GPT_a_eq_zero_iff_purely_imaginary_l963_96358


namespace NUMINAMATH_GPT_tile_floor_with_polygons_l963_96346

theorem tile_floor_with_polygons (x y z: ℕ) (h1: 3 ≤ x) (h2: 3 ≤ y) (h3: 3 ≤ z) 
  (h_seamless: ((1 - (2 / (x: ℝ))) * 180 + (1 - (2 / (y: ℝ))) * 180 + (1 - (2 / (z: ℝ))) * 180 = 360)) :
  (1 / (x: ℝ) + 1 / (y: ℝ) + 1 / (z: ℝ) = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_tile_floor_with_polygons_l963_96346


namespace NUMINAMATH_GPT_total_pencils_l963_96359

def initial_pencils : ℕ := 9
def additional_pencils : ℕ := 56

theorem total_pencils : initial_pencils + additional_pencils = 65 :=
by
  -- proof steps are not required, so we use sorry
  sorry

end NUMINAMATH_GPT_total_pencils_l963_96359


namespace NUMINAMATH_GPT_UBA_Capital_bought_8_SUVs_l963_96356

noncomputable def UBA_Capital_SUVs : ℕ := 
  let T := 9  -- Number of Toyotas
  let H := 1  -- Number of Hondas
  let SUV_Toyota := 9 / 10 * T  -- 90% of Toyotas are SUVs
  let SUV_Honda := 1 / 10 * H   -- 10% of Hondas are SUVs
  SUV_Toyota + SUV_Honda  -- Total number of SUVs

theorem UBA_Capital_bought_8_SUVs : UBA_Capital_SUVs = 8 := by
  sorry

end NUMINAMATH_GPT_UBA_Capital_bought_8_SUVs_l963_96356


namespace NUMINAMATH_GPT_arctan_sum_l963_96351

theorem arctan_sum (a b : ℝ) : 
  Real.arctan (a / (a + 2 * b)) + Real.arctan (b / (2 * a + b)) = Real.arctan (1 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_arctan_sum_l963_96351


namespace NUMINAMATH_GPT_pears_worth_l963_96323

variable (apples pears : ℚ)
variable (h : 3/4 * 16 * apples = 6 * pears)

theorem pears_worth (h : 3/4 * 16 * apples = 6 * pears) : 1 / 3 * 9 * apples = 1.5 * pears :=
by
  sorry

end NUMINAMATH_GPT_pears_worth_l963_96323


namespace NUMINAMATH_GPT_hitting_probability_l963_96370

theorem hitting_probability (A_hit B_hit : ℚ) (hA : A_hit = 4/5) (hB : B_hit = 5/6) :
  1 - ((1 - A_hit) * (1 - B_hit)) = 29/30 :=
by 
  sorry

end NUMINAMATH_GPT_hitting_probability_l963_96370


namespace NUMINAMATH_GPT_min_value_of_z_l963_96381

theorem min_value_of_z : ∃ x : ℝ, ∀ y : ℝ, 5 * x^2 + 20 * x + 25 ≤ 5 * y^2 + 20 * y + 25 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_z_l963_96381


namespace NUMINAMATH_GPT_annual_savings_l963_96354

-- defining the conditions
def current_speed := 10 -- in Mbps
def current_bill := 20 -- in dollars
def bill_30Mbps := 2 * current_bill -- in dollars
def bill_20Mbps := current_bill + 10 -- in dollars
def months_in_year := 12

-- calculating the annual costs
def annual_cost_30Mbps := bill_30Mbps * months_in_year
def annual_cost_20Mbps := bill_20Mbps * months_in_year

-- statement of the problem
theorem annual_savings : (annual_cost_30Mbps - annual_cost_20Mbps) = 120 := by
  sorry -- prove the statement

end NUMINAMATH_GPT_annual_savings_l963_96354


namespace NUMINAMATH_GPT_smallest_N_l963_96375

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end NUMINAMATH_GPT_smallest_N_l963_96375
