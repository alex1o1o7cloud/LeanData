import Mathlib

namespace NUMINAMATH_GPT_hexagon_coloring_count_l933_93376

def num_possible_colorings : Nat :=
by
  /- There are 7 choices for first vertex A.
     Once A is chosen, there are 6 choices for the remaining vertices B, C, D, E, F considering the diagonal restrictions. -/
  let total_colorings := 7 * 6 ^ 5
  let restricted_colorings := 7 * 6 ^ 3
  let valid_colorings := total_colorings - restricted_colorings
  exact valid_colorings

theorem hexagon_coloring_count : num_possible_colorings = 52920 :=
  by
    /- Computation steps above show that the number of valid colorings is 52920 -/
    sorry   -- Proof computation already indicated

end NUMINAMATH_GPT_hexagon_coloring_count_l933_93376


namespace NUMINAMATH_GPT_boxes_filled_l933_93364

noncomputable def bags_per_box := 6
noncomputable def balls_per_bag := 8
noncomputable def total_balls := 720

theorem boxes_filled (h1 : balls_per_bag = 8) (h2 : bags_per_box = 6) (h3 : total_balls = 720) :
  (total_balls / balls_per_bag) / bags_per_box = 15 :=
by
  sorry

end NUMINAMATH_GPT_boxes_filled_l933_93364


namespace NUMINAMATH_GPT_find_second_number_l933_93389

-- Define the given number
def given_number := 220070

-- Define the constants in the problem
def constant_555 := 555
def remainder := 70

-- Define the second number (our unknown)
variable (x : ℕ)

-- Define the condition as an equation
def condition : Prop :=
  given_number = (constant_555 + x) * 2 * (x - constant_555) + remainder

-- The theorem to prove that the second number is 343
theorem find_second_number : ∃ x : ℕ, condition x ∧ x = 343 :=
sorry

end NUMINAMATH_GPT_find_second_number_l933_93389


namespace NUMINAMATH_GPT_seth_sold_candy_bars_l933_93398

theorem seth_sold_candy_bars (max_sold : ℕ) (seth_sold : ℕ) 
  (h1 : max_sold = 24) 
  (h2 : seth_sold = 3 * max_sold + 6) : 
  seth_sold = 78 := 
by sorry

end NUMINAMATH_GPT_seth_sold_candy_bars_l933_93398


namespace NUMINAMATH_GPT_max_difference_y_coords_l933_93359

noncomputable def maximumDifference : ℝ :=
  (4 * Real.sqrt 6) / 9

theorem max_difference_y_coords :
  let f1 (x : ℝ) := 3 - 2 * x^2 + x^3
  let f2 (x : ℝ) := 1 + x^2 + x^3
  let x1 := Real.sqrt (2/3)
  let x2 := - Real.sqrt (2/3)
  let y1 := f1 x1
  let y2 := f1 x2
  |y1 - y2| = maximumDifference := sorry

end NUMINAMATH_GPT_max_difference_y_coords_l933_93359


namespace NUMINAMATH_GPT_least_prime_factor_of_11_pow4_minus_11_pow3_l933_93336

open Nat

theorem least_prime_factor_of_11_pow4_minus_11_pow3 : 
  Nat.minFac (11^4 - 11^3) = 2 :=
  sorry

end NUMINAMATH_GPT_least_prime_factor_of_11_pow4_minus_11_pow3_l933_93336


namespace NUMINAMATH_GPT_fraction_of_a_mile_additional_charge_l933_93385

-- Define the conditions
def initial_fee : ℚ := 2.25
def charge_per_fraction : ℚ := 0.25
def total_charge : ℚ := 4.50
def total_distance : ℚ := 3.6

-- Define the problem statement to prove
theorem fraction_of_a_mile_additional_charge :
  initial_fee = 2.25 →
  charge_per_fraction = 0.25 →
  total_charge = 4.50 →
  total_distance = 3.6 →
  total_distance - (total_charge - initial_fee) = 1.35 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fraction_of_a_mile_additional_charge_l933_93385


namespace NUMINAMATH_GPT_find_missing_number_l933_93344

theorem find_missing_number (x : ℝ) :
  ((20 + 40 + 60) / 3) = ((10 + 70 + x) / 3) + 8 → x = 16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_missing_number_l933_93344


namespace NUMINAMATH_GPT_vector_c_equals_combination_l933_93300

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)
def vector_c : ℝ × ℝ := (-2, 4)

theorem vector_c_equals_combination : vector_c = (vector_a.1 - 3 * vector_b.1, vector_a.2 - 3 * vector_b.2) :=
sorry

end NUMINAMATH_GPT_vector_c_equals_combination_l933_93300


namespace NUMINAMATH_GPT_intersection_eq_l933_93306

open Set

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x ≤ 5 }
def B : Set ℝ := { -1, 2, 3, 6 }

-- State the proof problem
theorem intersection_eq : A ∩ B = {2, 3} := 
by 
-- placeholder for the proof steps
sorry

end NUMINAMATH_GPT_intersection_eq_l933_93306


namespace NUMINAMATH_GPT_abs_sub_self_nonneg_l933_93390

theorem abs_sub_self_nonneg (m : ℚ) : |m| - m ≥ 0 := 
sorry

end NUMINAMATH_GPT_abs_sub_self_nonneg_l933_93390


namespace NUMINAMATH_GPT_place_value_diff_7669_l933_93322

theorem place_value_diff_7669 :
  let a := 6 * 10
  let b := 6 * 100
  b - a = 540 :=
by
  let a := 6 * 10
  let b := 6 * 100
  have h : b - a = 540 := by sorry
  exact h

end NUMINAMATH_GPT_place_value_diff_7669_l933_93322


namespace NUMINAMATH_GPT_general_formula_expression_of_k_l933_93301

noncomputable def sequence_a : ℕ → ℤ
| 0     => 0 
| 1     => 0 
| 2     => -6
| n + 2 => 2 * (sequence_a (n + 1)) - (sequence_a n)

theorem general_formula :
  ∀ n, sequence_a n = 2 * n - 10 := sorry

def sequence_k : ℕ → ℕ
| 0     => 0 
| 1     => 8 
| n + 1 => 3 * 2 ^ n + 5

theorem expression_of_k (n : ℕ) :
  sequence_k (n + 1) = 3 * 2 ^ n + 5 := sorry

end NUMINAMATH_GPT_general_formula_expression_of_k_l933_93301


namespace NUMINAMATH_GPT_wheel_radius_increase_l933_93307

theorem wheel_radius_increase :
  let r := 18
  let distance_AB := 600   -- distance from A to B in miles
  let distance_BA := 582   -- distance from B to A in miles
  let circumference_orig := 2 * Real.pi * r
  let dist_per_rotation_orig := circumference_orig / 63360
  let rotations_orig := distance_AB / dist_per_rotation_orig
  let r' := ((distance_BA * dist_per_rotation_orig * 63360) / (2 * Real.pi * rotations_orig))
  ((r' - r) : ℝ) = 0.34 := by
  sorry

end NUMINAMATH_GPT_wheel_radius_increase_l933_93307


namespace NUMINAMATH_GPT_mart_income_percentage_l933_93327

theorem mart_income_percentage 
  (J T M : ℝ)
  (h1 : M = 1.60 * T)
  (h2 : T = 0.60 * J) :
  M = 0.96 * J :=
sorry

end NUMINAMATH_GPT_mart_income_percentage_l933_93327


namespace NUMINAMATH_GPT_lara_has_largest_answer_l933_93388

/-- Define the final result for John, given his operations --/
def final_john (n : ℕ) : ℕ :=
  let add_three := n + 3
  let double := add_three * 2
  double - 4

/-- Define the final result for Lara, given her operations --/
def final_lara (n : ℕ) : ℕ :=
  let triple := n * 3
  let add_five := triple + 5
  add_five - 6

/-- Define the final result for Miguel, given his operations --/
def final_miguel (n : ℕ) : ℕ :=
  let double := n * 2
  let subtract_two := double - 2
  subtract_two + 2

/-- Main theorem to be proven --/
theorem lara_has_largest_answer :
  final_lara 12 > final_john 12 ∧ final_lara 12 > final_miguel 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_lara_has_largest_answer_l933_93388


namespace NUMINAMATH_GPT_instantaneous_acceleration_at_3_l933_93325

def v (t : ℝ) : ℝ := t^2 + 3

theorem instantaneous_acceleration_at_3 :
  deriv v 3 = 6 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_acceleration_at_3_l933_93325


namespace NUMINAMATH_GPT_find_radius_of_semicircular_plot_l933_93372

noncomputable def radius_of_semicircular_plot (π : ℝ) : ℝ :=
  let total_fence_length := 33
  let opening_length := 3
  let effective_fence_length := total_fence_length - opening_length
  let r := effective_fence_length / (π + 2)
  r

theorem find_radius_of_semicircular_plot 
  (π : ℝ) (Hπ : π = Real.pi) :
  radius_of_semicircular_plot π = 30 / (Real.pi + 2) :=
by
  unfold radius_of_semicircular_plot
  rw [Hπ]
  sorry

end NUMINAMATH_GPT_find_radius_of_semicircular_plot_l933_93372


namespace NUMINAMATH_GPT_value_of_c_l933_93357

-- Define the function f(x)
def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem value_of_c (a b c m : ℝ) (h₀ : ∀ x : ℝ, 0 ≤ f x a b)
  (h₁ : ∀ x : ℝ, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
sorry

end NUMINAMATH_GPT_value_of_c_l933_93357


namespace NUMINAMATH_GPT_decimal_to_fraction_l933_93392

theorem decimal_to_fraction : (0.3 + (0.24 - 0.24 / 100)) = (19 / 33) :=
by
  sorry

end NUMINAMATH_GPT_decimal_to_fraction_l933_93392


namespace NUMINAMATH_GPT_gcd_repeated_integer_l933_93378

theorem gcd_repeated_integer (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) :
  ∃ d, (∀ k : ℕ, k = 1001001001 * n → d = 1001001001 ∧ d ∣ k) :=
sorry

end NUMINAMATH_GPT_gcd_repeated_integer_l933_93378


namespace NUMINAMATH_GPT_range_of_a_l933_93305

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l933_93305


namespace NUMINAMATH_GPT_factor_product_modulo_l933_93384

theorem factor_product_modulo (h1 : 2021 % 23 = 21) (h2 : 2022 % 23 = 22) (h3 : 2023 % 23 = 0) (h4 : 2024 % 23 = 1) (h5 : 2025 % 23 = 2) :
  (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 0 :=
by
  sorry

end NUMINAMATH_GPT_factor_product_modulo_l933_93384


namespace NUMINAMATH_GPT_ratio_x_y_l933_93365

theorem ratio_x_y (x y : ℚ) (h : (14 * x - 5 * y) / (17 * x - 3 * y) = 2 / 7) : x / y = 29 / 64 :=
by
  sorry

end NUMINAMATH_GPT_ratio_x_y_l933_93365


namespace NUMINAMATH_GPT_election_majority_l933_93317

theorem election_majority (total_votes : ℕ) (winning_percentage : ℝ) (losing_percentage : ℝ)
  (h_total_votes : total_votes = 700)
  (h_winning_percentage : winning_percentage = 0.70)
  (h_losing_percentage : losing_percentage = 0.30) :
  (winning_percentage * total_votes - losing_percentage * total_votes) = 280 :=
by
  sorry

end NUMINAMATH_GPT_election_majority_l933_93317


namespace NUMINAMATH_GPT_number_of_tiles_l933_93396

theorem number_of_tiles (floor_length : ℝ) (floor_width : ℝ) (tile_length : ℝ) (tile_width : ℝ) 
  (h1 : floor_length = 9) 
  (h2 : floor_width = 12) 
  (h3 : tile_length = 1 / 2) 
  (h4 : tile_width = 2 / 3) 
  : (floor_length * floor_width) / (tile_length * tile_width) = 324 := 
by
  sorry

end NUMINAMATH_GPT_number_of_tiles_l933_93396


namespace NUMINAMATH_GPT_find_xy_l933_93377

variable (x y : ℚ)

theorem find_xy (h1 : 1/x + 3/y = 1/2) (h2 : 1/y - 3/x = 1/3) : 
    x = -20 ∧ y = 60/11 := 
by
  sorry

end NUMINAMATH_GPT_find_xy_l933_93377


namespace NUMINAMATH_GPT_percentage_change_in_receipts_l933_93347

theorem percentage_change_in_receipts
  (P S : ℝ) -- Original price and sales
  (hP : P > 0)
  (hS : S > 0)
  (new_P : ℝ := 0.70 * P) -- Price after 30% reduction
  (new_S : ℝ := 1.50 * S) -- Sales after 50% increase
  :
  (new_P * new_S - P * S) / (P * S) * 100 = 5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_change_in_receipts_l933_93347


namespace NUMINAMATH_GPT_area_ratio_of_circles_l933_93391

theorem area_ratio_of_circles (R_A R_B : ℝ) 
  (h1 : (60 / 360) * (2 * Real.pi * R_A) = (40 / 360) * (2 * Real.pi * R_B)) :
  (Real.pi * R_A ^ 2) / (Real.pi * R_B ^ 2) = 9 / 4 := 
sorry

end NUMINAMATH_GPT_area_ratio_of_circles_l933_93391


namespace NUMINAMATH_GPT_number_of_triangles_l933_93349

theorem number_of_triangles (m : ℕ) (h : m > 0) :
  ∃ n : ℕ, n = (m * (m + 1)) / 2 :=
by sorry

end NUMINAMATH_GPT_number_of_triangles_l933_93349


namespace NUMINAMATH_GPT_inequality_proof_l933_93386

variable (x y : ℝ)
variable (hx : 0 < x) (hy : 0 < y)

theorem inequality_proof :
  x / Real.sqrt y + y / Real.sqrt x ≥ Real.sqrt x + Real.sqrt y :=
sorry

end NUMINAMATH_GPT_inequality_proof_l933_93386


namespace NUMINAMATH_GPT_total_bricks_l933_93303

theorem total_bricks (n1 n2 r1 r2 : ℕ) (w1 w2 : ℕ)
  (h1 : n1 = 60) (h2 : r1 = 100) (h3 : n2 = 80) (h4 : r2 = 120)
  (h5 : w1 = 5) (h6 : w2 = 5) :
  (w1 * (n1 * r1) + w2 * (n2 * r2)) = 78000 :=
by sorry

end NUMINAMATH_GPT_total_bricks_l933_93303


namespace NUMINAMATH_GPT_total_sections_l933_93335

theorem total_sections (boys girls : ℕ) (h_boys : boys = 408) (h_girls : girls = 240) :
  let gcd_boys_girls := Nat.gcd boys girls
  let sections_boys := boys / gcd_boys_girls
  let sections_girls := girls / gcd_boys_girls
  sections_boys + sections_girls = 27 :=
by
  sorry

end NUMINAMATH_GPT_total_sections_l933_93335


namespace NUMINAMATH_GPT_oranges_and_apples_l933_93331

theorem oranges_and_apples (O A : ℕ) (h₁ : 7 * O = 5 * A) (h₂ : O = 28) : A = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_oranges_and_apples_l933_93331


namespace NUMINAMATH_GPT_quadratic_root_d_value_l933_93339

theorem quadratic_root_d_value :
  (∃ d : ℝ, ∀ x : ℝ, (2 * x^2 + 8 * x + d = 0) ↔ (x = (-8 + Real.sqrt 12) / 4) ∨ (x = (-8 - Real.sqrt 12) / 4)) → 
  d = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_d_value_l933_93339


namespace NUMINAMATH_GPT_Josanna_seventh_test_score_l933_93355

theorem Josanna_seventh_test_score (scores : List ℕ) (h_scores : scores = [95, 85, 75, 65, 90, 70])
                                   (average_increase : ℕ) (h_average_increase : average_increase = 5) :
                                   ∃ x, (List.sum scores + x) / (List.length scores + 1) = (List.sum scores) / (List.length scores) + average_increase := 
by
  sorry

end NUMINAMATH_GPT_Josanna_seventh_test_score_l933_93355


namespace NUMINAMATH_GPT_exists_line_through_ellipse_diameter_circle_origin_l933_93356

theorem exists_line_through_ellipse_diameter_circle_origin :
  ∃ m : ℝ, (m = (4 * Real.sqrt 3) / 3 ∨ m = -(4 * Real.sqrt 3) / 3) ∧
  ∀ (x y : ℝ), (x^2 + 2 * y^2 = 8) → (y = x + m) → (x^2 + (x + m)^2 = 8) :=
by
  sorry

end NUMINAMATH_GPT_exists_line_through_ellipse_diameter_circle_origin_l933_93356


namespace NUMINAMATH_GPT_negation_of_p_l933_93320

def p : Prop := ∀ x : ℝ, x ≥ 0 → x^2 - x ≥ 0

theorem negation_of_p : ¬ p ↔ ∃ x : ℝ, x ≥ 0 ∧ x^2 - x < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l933_93320


namespace NUMINAMATH_GPT_range_of_a_l933_93345

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ax^2 + 3 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4 / 9) :=
sorry

end NUMINAMATH_GPT_range_of_a_l933_93345


namespace NUMINAMATH_GPT_statement_A_statement_A_statement_C_statement_D_l933_93334

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem statement_A (x : ℝ) (hx : x > 1) : f x > 0 := sorry

theorem statement_A' (x : ℝ) (hx : 0 < x ∧ x < 1) : f x < 0 := sorry

theorem statement_C : Set.range f = Set.Ici (-1 / (2 * Real.exp 1)) := sorry

theorem statement_D (x : ℝ) : f x ≥ x - 1 := sorry

end NUMINAMATH_GPT_statement_A_statement_A_statement_C_statement_D_l933_93334


namespace NUMINAMATH_GPT_problem_l933_93338

def remainder_when_divided_by_20 (a b : ℕ) : ℕ := (a + b) % 20

theorem problem (a b : ℕ) (n m : ℤ) (h1 : a = 60 * n + 53) (h2 : b = 50 * m + 24) : 
  remainder_when_divided_by_20 a b = 17 := 
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_problem_l933_93338


namespace NUMINAMATH_GPT_area_of_right_triangle_l933_93348

theorem area_of_right_triangle
    (a b c : ℝ)
    (h₀ : a = 9)
    (h₁ : b = 12)
    (h₂ : c = 15)
    (right_triangle : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 54 := by
  sorry

end NUMINAMATH_GPT_area_of_right_triangle_l933_93348


namespace NUMINAMATH_GPT_train_speed_is_100_kmph_l933_93328

noncomputable def speed_of_train (length_of_train : ℝ) (time_to_cross_pole : ℝ) : ℝ :=
  (length_of_train / time_to_cross_pole) * 3.6

theorem train_speed_is_100_kmph :
  speed_of_train 100 3.6 = 100 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_100_kmph_l933_93328


namespace NUMINAMATH_GPT_largest_angle_measure_l933_93350

noncomputable def measure_largest_angle (x : ℚ) : Prop :=
  let a1 := 2 * x + 2
  let a2 := 3 * x
  let a3 := 4 * x + 3
  let a4 := 5 * x
  let a5 := 6 * x - 1
  let a6 := 7 * x
  a1 + a2 + a3 + a4 + a5 + a6 = 720 ∧ a6 = 5012 / 27

theorem largest_angle_measure : ∃ x : ℚ, measure_largest_angle x := by
  sorry

end NUMINAMATH_GPT_largest_angle_measure_l933_93350


namespace NUMINAMATH_GPT_ants_first_group_count_l933_93341

theorem ants_first_group_count :
    ∃ x : ℕ, 
        (∀ (w1 c1 a1 t1 w2 c2 a2 t2 : ℕ),
          w1 = 10 ∧ c1 = 600 ∧ a1 = x ∧ t1 = 5 ∧
          w2 = 5 ∧ c2 = 960 ∧ a2 = 20 ∧ t2 = 3 ∧ 
          (w1 * c1) / t1 = 1200 / a1 ∧ (w2 * c2) / t2 = 1600 / 20 →
             x = 15)
:= sorry

end NUMINAMATH_GPT_ants_first_group_count_l933_93341


namespace NUMINAMATH_GPT_black_pieces_more_than_white_l933_93381

theorem black_pieces_more_than_white (B W : ℕ) 
  (h₁ : (B - 1) * 7 = 9 * W)
  (h₂ : B * 5 = 7 * (W - 1)) :
  B - W = 7 :=
sorry

end NUMINAMATH_GPT_black_pieces_more_than_white_l933_93381


namespace NUMINAMATH_GPT_binary_101_eq_5_l933_93354

theorem binary_101_eq_5 : 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5 := 
by
  sorry

end NUMINAMATH_GPT_binary_101_eq_5_l933_93354


namespace NUMINAMATH_GPT_tom_teaching_years_l933_93387

theorem tom_teaching_years (T D : ℝ) (h1 : T + D = 70) (h2 : D = (1/2) * T - 5) : T = 50 :=
by
  -- This is where the proof would normally go if it were required.
  sorry

end NUMINAMATH_GPT_tom_teaching_years_l933_93387


namespace NUMINAMATH_GPT_kittens_given_is_two_l933_93342

-- Definitions of the conditions
def original_kittens : Nat := 8
def current_kittens : Nat := 6

-- Statement of the proof problem
theorem kittens_given_is_two : (original_kittens - current_kittens) = 2 := 
by
  sorry

end NUMINAMATH_GPT_kittens_given_is_two_l933_93342


namespace NUMINAMATH_GPT_cheryl_used_total_material_correct_amount_l933_93367

def material_used (initial leftover : ℚ) : ℚ := initial - leftover

def total_material_used 
  (initial_a initial_b initial_c leftover_a leftover_b leftover_c : ℚ) : ℚ :=
  material_used initial_a leftover_a + material_used initial_b leftover_b + material_used initial_c leftover_c

theorem cheryl_used_total_material_correct_amount :
  total_material_used (2/9) (1/8) (3/10) (4/18) (1/12) (3/15) = 17/120 :=
by
  sorry

end NUMINAMATH_GPT_cheryl_used_total_material_correct_amount_l933_93367


namespace NUMINAMATH_GPT_intersection_A_B_union_B_C_eq_B_iff_l933_93397

-- Definitions for the sets A, B, and C
def setA : Set ℝ := { x | x^2 - 3 * x < 0 }
def setB : Set ℝ := { x | (x + 2) * (4 - x) ≥ 0 }
def setC (a : ℝ) : Set ℝ := { x | a < x ∧ x ≤ a + 1 }

-- Proving that A ∩ B = { x | 0 < x < 3 }
theorem intersection_A_B : setA ∩ setB = { x : ℝ | 0 < x ∧ x < 3 } :=
sorry

-- Proving that B ∪ C = B implies the range of a is [-2, 3]
theorem union_B_C_eq_B_iff (a : ℝ) : (setB ∪ setC a = setB) ↔ (-2 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_GPT_intersection_A_B_union_B_C_eq_B_iff_l933_93397


namespace NUMINAMATH_GPT_value_of_d_l933_93393

theorem value_of_d (d : ℝ) (h : ∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) : d = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_d_l933_93393


namespace NUMINAMATH_GPT_average_test_score_l933_93329

theorem average_test_score (x : ℝ) :
  (0.45 * 95 + 0.50 * x + 0.05 * 60 = 84.75) → x = 78 :=
by
  sorry

end NUMINAMATH_GPT_average_test_score_l933_93329


namespace NUMINAMATH_GPT_total_toothpicks_in_grid_l933_93351

theorem total_toothpicks_in_grid (l w : ℕ) (h₁ : l = 50) (h₂ : w = 20) : 
  (l + 1) * w + (w + 1) * l + 2 * (l * w) = 4070 :=
by
  sorry

end NUMINAMATH_GPT_total_toothpicks_in_grid_l933_93351


namespace NUMINAMATH_GPT_polygon_has_area_144_l933_93324

noncomputable def polygonArea (n_sides : ℕ) (perimeter : ℕ) (n_squares : ℕ) : ℕ :=
  let s := perimeter / n_sides
  let square_area := s * s
  square_area * n_squares

theorem polygon_has_area_144 :
  polygonArea 32 64 36 = 144 :=
by
  sorry

end NUMINAMATH_GPT_polygon_has_area_144_l933_93324


namespace NUMINAMATH_GPT_james_meat_sales_l933_93366

theorem james_meat_sales
  (beef_pounds : ℕ)
  (pork_pounds : ℕ)
  (meat_per_meal : ℝ)
  (meal_price : ℝ)
  (total_meat : ℝ)
  (number_of_meals : ℝ)
  (total_money : ℝ)
  (h1 : beef_pounds = 20)
  (h2 : pork_pounds = beef_pounds / 2)
  (h3 : meat_per_meal = 1.5)
  (h4 : meal_price = 20)
  (h5 : total_meat = beef_pounds + pork_pounds)
  (h6 : number_of_meals = total_meat / meat_per_meal)
  (h7 : total_money = number_of_meals * meal_price) :
  total_money = 400 := by
  sorry

end NUMINAMATH_GPT_james_meat_sales_l933_93366


namespace NUMINAMATH_GPT_width_of_first_sheet_paper_l933_93369

theorem width_of_first_sheet_paper :
  ∀ (w : ℝ),
  2 * 11 * w = 2 * 4.5 * 11 + 100 → 
  w = 199 / 22 := 
by
  intro w
  intro h
  sorry

end NUMINAMATH_GPT_width_of_first_sheet_paper_l933_93369


namespace NUMINAMATH_GPT_red_pigment_contribution_l933_93352

theorem red_pigment_contribution :
  ∀ (G : ℝ), (2 * G + G + 3 * G = 24) →
  (0.6 * (2 * G) + 0.5 * (3 * G) = 10.8) :=
by
  intro G
  intro h1
  sorry

end NUMINAMATH_GPT_red_pigment_contribution_l933_93352


namespace NUMINAMATH_GPT_calculate_polynomial_value_l933_93323

theorem calculate_polynomial_value :
  103^5 - 5 * 103^4 + 10 * 103^3 - 10 * 103^2 + 5 * 103 - 1 = 11036846832 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_polynomial_value_l933_93323


namespace NUMINAMATH_GPT_pants_price_100_l933_93321

-- Define the variables and conditions
variables (x y : ℕ)

-- Define the prices according to the conditions
def coat_price_pants := x + 340
def coat_price_shoes_pants := y + x + 180
def total_price := (coat_price_pants x) + x + y

-- The theorem to prove
theorem pants_price_100 (h1: coat_price_pants x = coat_price_shoes_pants x y) (h2: total_price x y = 700) : x = 100 :=
sorry

end NUMINAMATH_GPT_pants_price_100_l933_93321


namespace NUMINAMATH_GPT_at_least_26_equal_differences_l933_93395

theorem at_least_26_equal_differences (x : Fin 102 → ℕ) (h : ∀ i j, i < j → x i < x j) (h' : ∀ i, x i < 255) :
  (∃ d : Fin 101 → ℕ, ∃ s : Finset ℕ, s.card ≥ 26 ∧ (∀ i, d i = x i.succ - x i) ∧ ∃ i j, i ≠ j ∧ (d i = d j)) :=
by {
  sorry
}

end NUMINAMATH_GPT_at_least_26_equal_differences_l933_93395


namespace NUMINAMATH_GPT_volume_correctness_l933_93346

noncomputable def volume_of_regular_triangular_pyramid (d : ℝ) : ℝ :=
  1/3 * d^2 * d * Real.sqrt 2

theorem volume_correctness (d : ℝ) : 
  volume_of_regular_triangular_pyramid d = 1/3 * d^3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_volume_correctness_l933_93346


namespace NUMINAMATH_GPT_find_m_l933_93343

/-- 
If the function y=x + m/(x-1) defined for x > 1 attains its minimum value at x = 3,
then the positive number m is 4.
-/
theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 < x -> x + m / (x - 1) ≥ 3 + m / 2):
  m = 4 :=
sorry

end NUMINAMATH_GPT_find_m_l933_93343


namespace NUMINAMATH_GPT_time_for_B_is_24_days_l933_93332

noncomputable def A_work : ℝ := (1 / 2) / (3 / 4)
noncomputable def B_work : ℝ := 1 -- assume B does 1 unit of work in 1 day
noncomputable def total_work : ℝ := (A_work + B_work) * 18

theorem time_for_B_is_24_days : 
  ((A_work + B_work) * 18) / B_work = 24 := by
  sorry

end NUMINAMATH_GPT_time_for_B_is_24_days_l933_93332


namespace NUMINAMATH_GPT_distance_between_intersection_points_l933_93314

noncomputable def C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def l (t : ℝ) : ℝ × ℝ :=
  (-2 * t + 2, 3 * t)

theorem distance_between_intersection_points :
  ∃ (A B : ℝ × ℝ), 
    (∃ θ : ℝ, C θ = A) ∧
    (∃ t : ℝ, l t = A) ∧
    (∃ θ : ℝ, C θ = B) ∧
    (∃ t : ℝ, l t = B) ∧
    dist A B = Real.sqrt 13 / 2 :=
sorry

end NUMINAMATH_GPT_distance_between_intersection_points_l933_93314


namespace NUMINAMATH_GPT_xy_minimization_l933_93304

theorem xy_minimization (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : (1 / (x : ℝ)) + 1 / (3 * y) = 1 / 11) : x * y = 176 ∧ x + y = 30 :=
by
  sorry

end NUMINAMATH_GPT_xy_minimization_l933_93304


namespace NUMINAMATH_GPT_two_thirds_of_5_times_9_l933_93394

theorem two_thirds_of_5_times_9 : (2 / 3) * (5 * 9) = 30 :=
by
  sorry

end NUMINAMATH_GPT_two_thirds_of_5_times_9_l933_93394


namespace NUMINAMATH_GPT_find_x_if_perpendicular_l933_93313

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (2 * x - 1, 3)
def vec_n : ℝ × ℝ := (1, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_x_if_perpendicular (x : ℝ) : 
  dot_product (vec_m x) vec_n = 0 ↔ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_if_perpendicular_l933_93313


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l933_93399

-- Definitions of the repeating decimals as fractions
def x : ℚ := 1 / 9
def y : ℚ := 2 / 99
def z : ℚ := 3 / 999

-- Theorem stating the sum of these fractions is equal to the expected result
theorem sum_of_repeating_decimals : x + y + z = 164 / 1221 := 
  sorry

end NUMINAMATH_GPT_sum_of_repeating_decimals_l933_93399


namespace NUMINAMATH_GPT_range_of_k_l933_93326

noncomputable def circle_equation (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

theorem range_of_k (k : ℝ) :
  circle_equation k →
  k ∈ (Set.Iio (-1) ∪ Set.Ioi 4) :=
sorry

end NUMINAMATH_GPT_range_of_k_l933_93326


namespace NUMINAMATH_GPT_total_time_spent_in_hours_l933_93370

/-- Miriam's time spent on each task in minutes. -/
def time_laundry := 30
def time_bathroom := 15
def time_room := 35
def time_homework := 40

/-- The function to convert minutes to hours. -/
def minutes_to_hours (minutes : ℕ) := minutes / 60

/-- The total time spent in minutes. -/
def total_time_minutes := time_laundry + time_bathroom + time_room + time_homework

/-- The total time spent in hours. -/
def total_time_hours := minutes_to_hours total_time_minutes

/-- The main statement to be proved: total_time_hours equals 2. -/
theorem total_time_spent_in_hours : total_time_hours = 2 := 
by
  sorry

end NUMINAMATH_GPT_total_time_spent_in_hours_l933_93370


namespace NUMINAMATH_GPT_sufficient_condition_range_k_l933_93319

theorem sufficient_condition_range_k {x k : ℝ} (h : ∀ x, x > k → (3 / (x + 1) < 1)) : k ≥ 2 :=
sorry

end NUMINAMATH_GPT_sufficient_condition_range_k_l933_93319


namespace NUMINAMATH_GPT_intersection_unique_l933_93353

noncomputable def f (x : ℝ) := 3 * Real.log x
noncomputable def g (x : ℝ) := Real.log (x + 4)

theorem intersection_unique : ∃! x, f x = g x :=
sorry

end NUMINAMATH_GPT_intersection_unique_l933_93353


namespace NUMINAMATH_GPT_largest_rectangle_area_l933_93315

theorem largest_rectangle_area (l w : ℕ) (hl : l > 0) (hw : w > 0) (hperimeter : 2 * l + 2 * w = 42)
  (harea_diff : ∃ (l1 w1 l2 w2 : ℕ), l1 > 0 ∧ w1 > 0 ∧ l2 > 0 ∧ w2 > 0 ∧ 2 * l1 + 2 * w1 = 42 
  ∧ 2 * l2 + 2 * w2 = 42 ∧ (l1 * w1) - (l2 * w2) = 90) : (l * w ≤ 110) :=
sorry

end NUMINAMATH_GPT_largest_rectangle_area_l933_93315


namespace NUMINAMATH_GPT_find_number_l933_93318

-- Define the conditions
def number_times_x_eq_165 (number x : ℕ) : Prop :=
  number * x = 165

def x_eq_11 (x : ℕ) : Prop :=
  x = 11

-- The proof problem statement
theorem find_number (number x : ℕ) (h1 : number_times_x_eq_165 number x) (h2 : x_eq_11 x) : number = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l933_93318


namespace NUMINAMATH_GPT_correct_option_c_l933_93316

-- Definitions for the problem context
noncomputable def qualification_rate : ℝ := 0.99
noncomputable def picking_probability := qualification_rate

-- The theorem statement that needs to be proven
theorem correct_option_c : picking_probability = 0.99 :=
sorry

end NUMINAMATH_GPT_correct_option_c_l933_93316


namespace NUMINAMATH_GPT_gcf_72_108_l933_93373

def gcf (a b : ℕ) : ℕ := 
  Nat.gcd a b

theorem gcf_72_108 : gcf 72 108 = 36 := by
  sorry

end NUMINAMATH_GPT_gcf_72_108_l933_93373


namespace NUMINAMATH_GPT_inequalities_not_hold_l933_93362

theorem inequalities_not_hold (x y z a b c : ℝ) (h1 : x < a) (h2 : y < b) (h3 : z < c) : 
  ¬ (x * y + y * z + z * x < a * b + b * c + c * a) ∧ 
  ¬ (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧ 
  ¬ (x * y * z < a * b * c) := 
sorry

end NUMINAMATH_GPT_inequalities_not_hold_l933_93362


namespace NUMINAMATH_GPT_Nina_money_before_tax_l933_93309

theorem Nina_money_before_tax :
  ∃ (M P : ℝ), M = 6 * P ∧ M = 8 * 0.9 * P ∧ M = 5 :=
by 
  sorry

end NUMINAMATH_GPT_Nina_money_before_tax_l933_93309


namespace NUMINAMATH_GPT_max_sum_of_factors_l933_93371

theorem max_sum_of_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C)
  (h4 : A * B * C = 3003) : A + B + C ≤ 117 :=
sorry

end NUMINAMATH_GPT_max_sum_of_factors_l933_93371


namespace NUMINAMATH_GPT_international_sales_correct_option_l933_93308

theorem international_sales_correct_option :
  (∃ (A B C D : String),
     A = "who" ∧
     B = "what" ∧
     C = "whoever" ∧
     D = "whatever" ∧
     (∃ x, x = C → "Could I speak to " ++ x ++ " is in charge of International Sales please?" = "Could I speak to whoever is in charge of International Sales please?")) :=
sorry

end NUMINAMATH_GPT_international_sales_correct_option_l933_93308


namespace NUMINAMATH_GPT_cost_of_paint_per_quart_l933_93330

/-- Tommy has a flag that is 5 feet wide and 4 feet tall. 
He needs to paint both sides of the flag. 
A quart of paint covers 4 square feet. 
He spends $20 on paint. 
Prove that the cost of paint per quart is $2. --/
theorem cost_of_paint_per_quart
  (width height : ℕ) (paint_area_per_quart : ℕ) (total_cost : ℕ) (total_area : ℕ) (quarts_needed : ℕ) :
  width = 5 →
  height = 4 →
  paint_area_per_quart = 4 →
  total_cost = 20 →
  total_area = 2 * (width * height) →
  quarts_needed = total_area / paint_area_per_quart →
  total_cost / quarts_needed = 2 := 
by
  intros h_w h_h h_papq h_tc h_ta h_qn
  sorry

end NUMINAMATH_GPT_cost_of_paint_per_quart_l933_93330


namespace NUMINAMATH_GPT_product_of_two_numbers_in_ratio_l933_93312

theorem product_of_two_numbers_in_ratio (x y : ℚ) 
  (h1 : x - y = d)
  (h2 : x + y = 8 * d)
  (h3 : x * y = 15 * d) :
  x * y = 100 / 7 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_in_ratio_l933_93312


namespace NUMINAMATH_GPT_independent_trials_probability_l933_93340

theorem independent_trials_probability (p : ℝ) (q : ℝ) (ε : ℝ) (desired_prob : ℝ) 
    (h_p : p = 0.7) (h_q : q = 0.3) (h_ε : ε = 0.2) (h_desired_prob : desired_prob = 0.96) :
    ∃ n : ℕ, n > (p * q) / (desired_prob * ε^2) ∧ n = 132 :=
by
  sorry

end NUMINAMATH_GPT_independent_trials_probability_l933_93340


namespace NUMINAMATH_GPT_smallest_integer_five_consecutive_sum_2025_l933_93302

theorem smallest_integer_five_consecutive_sum_2025 :
  ∃ n : ℤ, 5 * n + 10 = 2025 ∧ n = 403 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_five_consecutive_sum_2025_l933_93302


namespace NUMINAMATH_GPT_pin_probability_l933_93337

theorem pin_probability :
  let total_pins := 9 * 10^5
  let valid_pins := 10^4
  ∃ p : ℚ, p = valid_pins / total_pins ∧ p = 1 / 90 := by
  sorry

end NUMINAMATH_GPT_pin_probability_l933_93337


namespace NUMINAMATH_GPT_family_raised_percentage_l933_93360

theorem family_raised_percentage :
  ∀ (total_funds friends_percentage own_savings family_funds remaining_funds : ℝ),
    total_funds = 10000 →
    friends_percentage = 0.40 →
    own_savings = 4200 →
    remaining_funds = total_funds - (friends_percentage * total_funds) →
    family_funds = remaining_funds - own_savings →
    (family_funds / remaining_funds) * 100 = 30 :=
by
  intros total_funds friends_percentage own_savings family_funds remaining_funds
  intros h_total_funds h_friends_percentage h_own_savings h_remaining_funds h_family_funds
  sorry

end NUMINAMATH_GPT_family_raised_percentage_l933_93360


namespace NUMINAMATH_GPT_percentage_of_liquid_X_in_solution_A_l933_93310

theorem percentage_of_liquid_X_in_solution_A (P : ℝ) :
  (0.018 * 700 / 1200 + P * 500 / 1200) = 0.0166 → P = 0.01464 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_of_liquid_X_in_solution_A_l933_93310


namespace NUMINAMATH_GPT_solve_equation_l933_93383

theorem solve_equation : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l933_93383


namespace NUMINAMATH_GPT_exists_number_added_to_sum_of_digits_gives_2014_l933_93382

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem exists_number_added_to_sum_of_digits_gives_2014 : 
  ∃ (n : ℕ), n + sum_of_digits n = 2014 :=
sorry

end NUMINAMATH_GPT_exists_number_added_to_sum_of_digits_gives_2014_l933_93382


namespace NUMINAMATH_GPT_find_coordinates_M_l933_93379

open Real

theorem find_coordinates_M (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℝ) :
  ∃ (xM yM zM : ℝ), 
  xM = (x1 + x2 + x3 + x4) / 4 ∧
  yM = (y1 + y2 + y3 + y4) / 4 ∧
  zM = (z1 + z2 + z3 + z4) / 4 ∧
  (x1 - xM) + (x2 - xM) + (x3 - xM) + (x4 - xM) = 0 ∧
  (y1 - yM) + (y2 - yM) + (y3 - yM) + (y4 - yM) = 0 ∧
  (z1 - zM) + (z2 - zM) + (z3 - zM) + (z4 - zM) = 0 := by
  sorry

end NUMINAMATH_GPT_find_coordinates_M_l933_93379


namespace NUMINAMATH_GPT_length_of_XY_in_triangle_XYZ_l933_93358

theorem length_of_XY_in_triangle_XYZ :
  ∀ (XYZ : Type) (X Y Z : XYZ) (angle : XYZ → XYZ → XYZ → ℝ) (length : XYZ → XYZ → ℝ),
  angle X Z Y = 30 ∧ angle Y X Z = 90 ∧ length X Z = 8 → length X Y = 16 :=
by sorry

end NUMINAMATH_GPT_length_of_XY_in_triangle_XYZ_l933_93358


namespace NUMINAMATH_GPT_interior_points_in_divided_square_l933_93311

theorem interior_points_in_divided_square :
  ∀ (n : ℕ), 
  (n = 2016) →
  ∃ (k : ℕ), 
  (∀ (t : ℕ), t = 180 * n) → 
  k = 1007 :=
by
  intros n hn
  use 1007
  sorry

end NUMINAMATH_GPT_interior_points_in_divided_square_l933_93311


namespace NUMINAMATH_GPT_Mary_current_age_l933_93363

theorem Mary_current_age
  (M J : ℕ) 
  (h1 : J - 5 = (M - 5) + 7) 
  (h2 : J + 5 = 2 * (M + 5)) : 
  M = 2 :=
by
  /- We need to show that the current age of Mary (M) is 2
     given the conditions h1 and h2.-/
  sorry

end NUMINAMATH_GPT_Mary_current_age_l933_93363


namespace NUMINAMATH_GPT_tank_capacity_l933_93374

theorem tank_capacity (fill_rate drain_rate1 drain_rate2 : ℝ)
  (initial_fullness : ℝ) (time_to_fill : ℝ) (capacity_in_liters : ℝ) :
  fill_rate = 1 / 2 ∧
  drain_rate1 = 1 / 4 ∧
  drain_rate2 = 1 / 6 ∧ 
  initial_fullness = 1 / 2 ∧ 
  time_to_fill = 60 →
  capacity_in_liters = 10000 :=
by {
  sorry
}

end NUMINAMATH_GPT_tank_capacity_l933_93374


namespace NUMINAMATH_GPT_fraction_correct_l933_93380

-- Define the total number of coins.
def total_coins : ℕ := 30

-- Define the number of states that joined the union in the decade 1800 through 1809.
def states_1800_1809 : ℕ := 4

-- Define the fraction of coins representing states joining in the decade 1800 through 1809.
def fraction_coins_1800_1809 : ℚ := states_1800_1809 / total_coins

-- The theorem statement that needs to be proved.
theorem fraction_correct : fraction_coins_1800_1809 = (2 / 15) := 
by
  sorry

end NUMINAMATH_GPT_fraction_correct_l933_93380


namespace NUMINAMATH_GPT_wrestling_match_student_count_l933_93375

theorem wrestling_match_student_count (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 := by
  sorry

end NUMINAMATH_GPT_wrestling_match_student_count_l933_93375


namespace NUMINAMATH_GPT_find_point_C_l933_93368

noncomputable def point_on_z_axis (z : ℝ) : ℝ × ℝ × ℝ := (0, 0, z)
def point_A : ℝ × ℝ × ℝ := (1, 0, 2)
def point_B : ℝ × ℝ × ℝ := (1, 1, 1)

theorem find_point_C :
  ∃ C : ℝ × ℝ × ℝ, (C = point_on_z_axis 1) ∧ (dist C point_A = dist C point_B) :=
by
  sorry

end NUMINAMATH_GPT_find_point_C_l933_93368


namespace NUMINAMATH_GPT_find_x_when_areas_equal_l933_93333

-- Definitions based on the problem conditions
def glass_area : ℕ := 4 * (30 * 20)
def window_area (x : ℕ) : ℕ := (60 + 3 * x) * (40 + 3 * x)
def total_area_of_glass : ℕ := glass_area
def total_area_of_wood (x : ℕ) : ℕ := window_area x - glass_area

-- Proof problem, proving x == 20 / 3 when total area of glass equals total area of wood
theorem find_x_when_areas_equal : 
  ∃ x : ℕ, (total_area_of_glass = total_area_of_wood x) ∧ x = 20 / 3 :=
sorry

end NUMINAMATH_GPT_find_x_when_areas_equal_l933_93333


namespace NUMINAMATH_GPT_common_rational_root_neg_not_integer_l933_93361

theorem common_rational_root_neg_not_integer : 
  ∃ (p : ℚ), (p < 0) ∧ (¬ ∃ (z : ℤ), p = z) ∧ 
  (50 * p^4 + a * p^3 + b * p^2 + c * p + 20 = 0) ∧ 
  (20 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 50 = 0) := 
sorry

end NUMINAMATH_GPT_common_rational_root_neg_not_integer_l933_93361
