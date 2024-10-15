import Mathlib

namespace NUMINAMATH_GPT_sale_price_lower_than_original_l393_39330

noncomputable def original_price (p : ℝ) : ℝ := 
  p

noncomputable def increased_price (p : ℝ) : ℝ := 
  1.30 * p

noncomputable def sale_price (p : ℝ) : ℝ := 
  0.75 * increased_price p

theorem sale_price_lower_than_original (p : ℝ) : 
  sale_price p = 0.975 * p := 
sorry

end NUMINAMATH_GPT_sale_price_lower_than_original_l393_39330


namespace NUMINAMATH_GPT_trig_identity_proof_l393_39364

theorem trig_identity_proof :
  Real.sin (30 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) + 
  Real.sin (60 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) =
  Real.sqrt 2 / 2 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l393_39364


namespace NUMINAMATH_GPT_point_not_in_fourth_quadrant_l393_39319

theorem point_not_in_fourth_quadrant (a : ℝ) :
  ¬ ((a - 3 > 0) ∧ (a + 3 < 0)) :=
by
  sorry

end NUMINAMATH_GPT_point_not_in_fourth_quadrant_l393_39319


namespace NUMINAMATH_GPT_fg_of_1_l393_39307

def f (x : ℤ) : ℤ := x + 3
def g (x : ℤ) : ℤ := x^3 - x^2 - 6

theorem fg_of_1 : f (g 1) = -3 := by
  sorry

end NUMINAMATH_GPT_fg_of_1_l393_39307


namespace NUMINAMATH_GPT_find_a5_plus_a7_l393_39345

variable {a : ℕ → ℕ}

-- Assume a is a geometric sequence with common ratio q and first term a1.
def geometric_sequence (a : ℕ → ℕ) (a_1 : ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a_1 * q ^ n

-- Given conditions of the problem:
def conditions (a : ℕ → ℕ) : Prop :=
  a 2 + a 4 = 20 ∧ a 3 + a 5 = 40

-- The objective is to prove a_5 + a_7 = 160
theorem find_a5_plus_a7 (a : ℕ → ℕ) (a_1 q : ℕ) (h_geo : geometric_sequence a a_1 q) (h_cond : conditions a) : a 5 + a 7 = 160 :=
  sorry

end NUMINAMATH_GPT_find_a5_plus_a7_l393_39345


namespace NUMINAMATH_GPT_judgments_correct_l393_39344

variables {l m : Line} (a : Plane)

def is_perpendicular (l : Line) (a : Plane) : Prop := -- Definition of perpendicularity between a line and a plane
sorry

def is_parallel (l m : Line) : Prop := -- Definition of parallel lines
sorry

def is_contained_in (m : Line) (a : Plane) : Prop := -- Definition of a line contained in a plane
sorry

theorem judgments_correct 
  (hl : is_perpendicular l a)
  (hm : l ≠ m) :
  (∀ m, is_perpendicular m l → is_parallel m a) ∧ 
  (is_perpendicular m a → is_parallel m l) ∧
  (is_contained_in m a → is_perpendicular m l) ∧
  (is_parallel m l → is_perpendicular m a) :=
sorry

end NUMINAMATH_GPT_judgments_correct_l393_39344


namespace NUMINAMATH_GPT_solution_set_of_inequality_l393_39355

theorem solution_set_of_inequality (x : ℝ) :
  2 * |x - 1| - 1 < 0 ↔ (1 / 2) < x ∧ x < (3 / 2) :=
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l393_39355


namespace NUMINAMATH_GPT_A_wins_probability_is_3_over_4_l393_39312

def parity (n : ℕ) : Bool := n % 2 == 0

def number_of_dice_outcomes : ℕ := 36

def same_parity_outcome : ℕ := 18

def probability_A_wins : ℕ → ℕ → ℕ → ℚ
| total_outcomes, same_parity, different_parity =>
  (same_parity / total_outcomes : ℚ) * 1 + (different_parity / total_outcomes : ℚ) * (1 / 2)

theorem A_wins_probability_is_3_over_4 :
  probability_A_wins number_of_dice_outcomes same_parity_outcome (number_of_dice_outcomes - same_parity_outcome) = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_A_wins_probability_is_3_over_4_l393_39312


namespace NUMINAMATH_GPT_depth_of_right_frustum_l393_39359

-- Definitions
def volume_cm3 := 190000 -- Volume in cubic centimeters (190 liters)
def top_edge := 60 -- Length of the top edge in centimeters
def bottom_edge := 40 -- Length of the bottom edge in centimeters
def expected_depth := 75 -- Expected depth in centimeters

-- The following is the statement of the proof
theorem depth_of_right_frustum 
  (V : ℝ) (A1 A2 : ℝ) (h : ℝ)
  (hV : V = 190 * 1000)
  (hA1 : A1 = top_edge * top_edge)
  (hA2 : A2 = bottom_edge * bottom_edge)
  (h_avg : 2 * A1 / (top_edge + bottom_edge) = 2 * A2 / (top_edge + bottom_edge))
  : h = expected_depth := 
sorry

end NUMINAMATH_GPT_depth_of_right_frustum_l393_39359


namespace NUMINAMATH_GPT_MelAge_when_Katherine24_l393_39329

variable (Katherine Mel : ℕ)

-- Conditions
def isYounger (Mel Katherine : ℕ) : Prop :=
  Mel = Katherine - 3

def is24yearsOld (Katherine : ℕ) : Prop :=
  Katherine = 24

-- Statement to Prove
theorem MelAge_when_Katherine24 (Katherine Mel : ℕ) 
  (h1 : isYounger Mel Katherine) 
  (h2 : is24yearsOld Katherine) : 
  Mel = 21 := 
by 
  sorry

end NUMINAMATH_GPT_MelAge_when_Katherine24_l393_39329


namespace NUMINAMATH_GPT_band_to_orchestra_ratio_is_two_l393_39316

noncomputable def ratio_of_band_to_orchestra : ℤ :=
  let orchestra_students := 20
  let choir_students := 28
  let total_students := 88
  let band_students := (total_students - orchestra_students - choir_students)
  band_students / orchestra_students

theorem band_to_orchestra_ratio_is_two :
  let orchestra_students := 20
  let choir_students := 28
  let total_students := 88
  ratio_of_band_to_orchestra = 2 := by
  sorry

end NUMINAMATH_GPT_band_to_orchestra_ratio_is_two_l393_39316


namespace NUMINAMATH_GPT_find_abc_l393_39348

variables {a b c : ℕ}

theorem find_abc (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : abc ∣ ((a * b - 1) * (b * c - 1) * (c * a - 1))) : a = 2 ∧ b = 3 ∧ c = 5 :=
by {
    sorry
}

end NUMINAMATH_GPT_find_abc_l393_39348


namespace NUMINAMATH_GPT_yara_total_earnings_l393_39392

-- Lean code to represent the conditions and the proof statement

theorem yara_total_earnings
  (x : ℕ)  -- Yara's hourly wage
  (third_week_hours : ℕ := 18)
  (previous_week_hours : ℕ := 12)
  (extra_earnings : ℕ := 36)
  (third_week_earning : ℕ := third_week_hours * x)
  (previous_week_earning : ℕ := previous_week_hours * x)
  (total_earning : ℕ := third_week_earning + previous_week_earning) :
  third_week_earning = previous_week_earning + extra_earnings → 
  total_earning = 180 := 
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_yara_total_earnings_l393_39392


namespace NUMINAMATH_GPT_product_divisible_by_six_l393_39332

theorem product_divisible_by_six (a : ℤ) : 6 ∣ a * (a + 1) * (2 * a + 1) := 
sorry

end NUMINAMATH_GPT_product_divisible_by_six_l393_39332


namespace NUMINAMATH_GPT_liters_to_cubic_decimeters_eq_l393_39358

-- Define the condition for unit conversion
def liter_to_cubic_decimeter : ℝ :=
  1 -- since 1 liter = 1 cubic decimeter

-- Prove the equality for the given quantities
theorem liters_to_cubic_decimeters_eq :
  1.5 = 1.5 * liter_to_cubic_decimeter :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_liters_to_cubic_decimeters_eq_l393_39358


namespace NUMINAMATH_GPT_A_minus_B_l393_39384

theorem A_minus_B (A B : ℚ) (n : ℕ) :
  (A : ℚ) = 1 / 6 →
  (B : ℚ) = -1 / 12 →
  A - B = 1 / 4 :=
by
  intro hA hB
  rw [hA, hB]
  norm_num

end NUMINAMATH_GPT_A_minus_B_l393_39384


namespace NUMINAMATH_GPT_slices_left_for_lunch_tomorrow_l393_39382

def pizza_slices : ℕ := 12
def lunch_slices : ℕ := pizza_slices / 2
def remaining_after_lunch : ℕ := pizza_slices - lunch_slices
def dinner_slices : ℕ := remaining_after_lunch * 1/3
def slices_left : ℕ := remaining_after_lunch - dinner_slices

theorem slices_left_for_lunch_tomorrow : slices_left = 4 :=
by
  sorry

end NUMINAMATH_GPT_slices_left_for_lunch_tomorrow_l393_39382


namespace NUMINAMATH_GPT_solution_exists_for_any_y_l393_39394

theorem solution_exists_for_any_y (z : ℝ) : (∀ y : ℝ, ∃ x : ℝ, x^2 + y^2 + 4*z^2 + 2*x*y*z - 9 = 0) ↔ |z| ≤ 3 / 2 := 
sorry

end NUMINAMATH_GPT_solution_exists_for_any_y_l393_39394


namespace NUMINAMATH_GPT_correct_option_l393_39300

def option_A_1 : ℤ := (-2) ^ 2
def option_A_2 : ℤ := -(2 ^ 2)
def option_B_1 : ℤ := (|-2|) ^ 2
def option_B_2 : ℤ := -(2 ^ 2)
def option_C_1 : ℤ := (-2) ^ 3
def option_C_2 : ℤ := -(2 ^ 3)
def option_D_1 : ℤ := (|-2|) ^ 3
def option_D_2 : ℤ := -(2 ^ 3)

theorem correct_option : option_C_1 = option_C_2 ∧ 
  (option_A_1 ≠ option_A_2) ∧ 
  (option_B_1 ≠ option_B_2) ∧ 
  (option_D_1 ≠ option_D_2) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l393_39300


namespace NUMINAMATH_GPT_tony_water_trips_calculation_l393_39303

noncomputable def tony_drinks_water_after_every_n_trips (bucket_capacity_sand : ℤ) 
                                                        (sandbox_depth : ℤ) (sandbox_width : ℤ) 
                                                        (sandbox_length : ℤ) (sand_weight_cubic_foot : ℤ) 
                                                        (water_consumption : ℤ) (water_bottle_ounces : ℤ) 
                                                        (water_bottle_cost : ℤ) (money_with_tony : ℤ) 
                                                        (expected_change : ℤ) : ℤ :=
  let volume_sandbox := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := volume_sandbox * sand_weight_cubic_foot
  let trips_needed := total_sand_weight / bucket_capacity_sand
  let money_spent_on_water := money_with_tony - expected_change
  let water_bottles_bought := money_spent_on_water / water_bottle_cost
  let total_water_ounces := water_bottles_bought * water_bottle_ounces
  let drinking_sessions := total_water_ounces / water_consumption
  trips_needed / drinking_sessions

theorem tony_water_trips_calculation : 
  tony_drinks_water_after_every_n_trips 2 2 4 5 3 3 15 2 10 4 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_tony_water_trips_calculation_l393_39303


namespace NUMINAMATH_GPT_production_growth_rate_eq_l393_39318

theorem production_growth_rate_eq 
  (x : ℝ)
  (H : 100 + 100 * (1 + x) + 100 * (1 + x) ^ 2 = 364) : 
  100 + 100 * (1 + x) + 100 * (1 + x) ^ 2 = 364 :=
by {
  sorry
}

end NUMINAMATH_GPT_production_growth_rate_eq_l393_39318


namespace NUMINAMATH_GPT_initial_oranges_l393_39399

theorem initial_oranges (X : ℕ) (h1 : X - 37 + 7 = 10) : X = 40 :=
by
  sorry

end NUMINAMATH_GPT_initial_oranges_l393_39399


namespace NUMINAMATH_GPT_least_pos_int_N_l393_39301

theorem least_pos_int_N :
  ∃ N : ℕ, (N > 0) ∧ (N % 4 = 3) ∧ (N % 5 = 4) ∧ (N % 6 = 5) ∧ (N % 7 = 6) ∧ 
  (∀ m : ℕ, (m > 0) ∧ (m % 4 = 3) ∧ (m % 5 = 4) ∧ (m % 6 = 5) ∧ (m % 7 = 6) → N ≤ m) ∧ N = 419 :=
by
  sorry

end NUMINAMATH_GPT_least_pos_int_N_l393_39301


namespace NUMINAMATH_GPT_reasoning_is_invalid_l393_39398

-- Definitions based on conditions
variables {Line Plane : Type} (is_parallel_to : Line → Plane → Prop) (is_parallel_to' : Line → Line → Prop) (is_contained_in : Line → Plane → Prop)

-- Conditions
axiom major_premise (b : Line) (α : Plane) : is_parallel_to b α → ∀ (a : Line), is_contained_in a α → is_parallel_to' b a
axiom minor_premise1 (b : Line) (α : Plane) : is_parallel_to b α
axiom minor_premise2 (a : Line) (α : Plane) : is_contained_in a α

-- Conclusion
theorem reasoning_is_invalid : ∃ (a : Line) (b : Line) (α : Plane), ¬ (is_parallel_to b α → ∀ (a : Line), is_contained_in a α → is_parallel_to' b a) :=
sorry

end NUMINAMATH_GPT_reasoning_is_invalid_l393_39398


namespace NUMINAMATH_GPT_larger_number_is_21_l393_39351

theorem larger_number_is_21 (x y : ℤ) (h1 : x + y = 35) (h2 : x - y = 7) : x = 21 := 
by 
  sorry

end NUMINAMATH_GPT_larger_number_is_21_l393_39351


namespace NUMINAMATH_GPT_Kayla_total_items_l393_39326

theorem Kayla_total_items (T_bars : ℕ) (T_cans : ℕ) (K_bars : ℕ) (K_cans : ℕ)
  (h1 : T_bars = 2 * K_bars) (h2 : T_cans = 2 * K_cans)
  (h3 : T_bars = 12) (h4 : T_cans = 18) : 
  K_bars + K_cans = 15 :=
by {
  -- In order to focus only on statement definition, we use sorry here
  sorry
}

end NUMINAMATH_GPT_Kayla_total_items_l393_39326


namespace NUMINAMATH_GPT_problem_l393_39378

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (3 * x - Real.pi / 3)

theorem problem 
  (x₁ x₂ : ℝ)
  (hx₁x₂ : |f x₁ - f x₂| = 4)
  (x : ℝ)
  (hx : 0 ≤ x ∧ x ≤ Real.pi / 6)
  (m : ℝ) : m ≥ 1 / 3 :=
sorry

end NUMINAMATH_GPT_problem_l393_39378


namespace NUMINAMATH_GPT_largest_fraction_addition_l393_39360

-- Definitions for the problem conditions
def proper_fraction (a b : ℕ) : Prop :=
  a < b

def denom_less_than (d : ℕ) (bound : ℕ) : Prop :=
  d < bound

-- Main statement of the problem
theorem largest_fraction_addition :
  ∃ (a b : ℕ), (b > 0) ∧ proper_fraction (b + 7 * a) (7 * b) ∧ denom_less_than b 5 ∧ (a / b : ℚ) <= 3/4 := 
sorry

end NUMINAMATH_GPT_largest_fraction_addition_l393_39360


namespace NUMINAMATH_GPT_smallest_is_57_l393_39333

noncomputable def smallest_of_four_numbers (a b c d : ℕ) : ℕ :=
  if h1 : a + b + c = 234 ∧ a + b + d = 251 ∧ a + c + d = 284 ∧ b + c + d = 299
  then Nat.min (Nat.min a b) (Nat.min c d)
  else 0

theorem smallest_is_57 (a b c d : ℕ) (h1 : a + b + c = 234) (h2 : a + b + d = 251)
  (h3 : a + c + d = 284) (h4 : b + c + d = 299) :
  smallest_of_four_numbers a b c d = 57 :=
sorry

end NUMINAMATH_GPT_smallest_is_57_l393_39333


namespace NUMINAMATH_GPT_cone_plane_distance_l393_39365

theorem cone_plane_distance (H α : ℝ) : 
  (x = 2 * H * (Real.sin (α / 4)) ^ 2) :=
sorry

end NUMINAMATH_GPT_cone_plane_distance_l393_39365


namespace NUMINAMATH_GPT_sum_of_first_odd_numbers_l393_39337

theorem sum_of_first_odd_numbers (S1 S2 : ℕ) (n1 n2 : ℕ)
  (hS1 : S1 = n1^2) 
  (hS2 : S2 = n2^2) 
  (h1 : S1 = 2500)
  (h2 : S2 = 5625) : 
  n2 = 75 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_odd_numbers_l393_39337


namespace NUMINAMATH_GPT_distinct_integers_sum_l393_39313

theorem distinct_integers_sum {a b c d : ℤ} (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_product : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end NUMINAMATH_GPT_distinct_integers_sum_l393_39313


namespace NUMINAMATH_GPT_unit_vector_same_direction_l393_39388

-- Define the coordinates of points A and B as given in the conditions
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

-- Define the vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the magnitude of vector AB
noncomputable def magnitudeAB : ℝ := Real.sqrt (vectorAB.1^2 + vectorAB.2^2)

-- Define the unit vector in the direction of AB
noncomputable def unitVectorAB : ℝ × ℝ := (vectorAB.1 / magnitudeAB, vectorAB.2 / magnitudeAB)

-- The theorem we want to prove
theorem unit_vector_same_direction :
  unitVectorAB = (3 / 5, -4 / 5) :=
sorry

end NUMINAMATH_GPT_unit_vector_same_direction_l393_39388


namespace NUMINAMATH_GPT_total_capacity_of_two_tanks_l393_39369

-- Conditions
def tank_A_initial_fullness : ℚ := 3 / 4
def tank_A_final_fullness : ℚ := 7 / 8
def tank_A_added_volume : ℚ := 5

def tank_B_initial_fullness : ℚ := 2 / 3
def tank_B_final_fullness : ℚ := 5 / 6
def tank_B_added_volume : ℚ := 3

-- Proof statement
theorem total_capacity_of_two_tanks :
  let tank_A_total_capacity := tank_A_added_volume / (tank_A_final_fullness - tank_A_initial_fullness)
  let tank_B_total_capacity := tank_B_added_volume / (tank_B_final_fullness - tank_B_initial_fullness)
  tank_A_total_capacity + tank_B_total_capacity = 58 := 
sorry

end NUMINAMATH_GPT_total_capacity_of_two_tanks_l393_39369


namespace NUMINAMATH_GPT_odd_function_property_l393_39352

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 - x^2)) / x

theorem odd_function_property (a : ℝ) (h_a : -2 ≤ a ∧ a ≤ 2) (h_fa : f a = -4) : f (-a) = 4 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_property_l393_39352


namespace NUMINAMATH_GPT_angle_terminal_side_equivalence_l393_39338

theorem angle_terminal_side_equivalence (k : ℤ) : 
    ∃ k : ℤ, 405 = k * 360 + 45 :=
by
  sorry

end NUMINAMATH_GPT_angle_terminal_side_equivalence_l393_39338


namespace NUMINAMATH_GPT_christen_peeled_20_potatoes_l393_39309

-- Define the conditions and question
def homer_rate : ℕ := 3
def time_alone : ℕ := 4
def christen_rate : ℕ := 5
def total_potatoes : ℕ := 44

noncomputable def christen_potatoes : ℕ :=
  (total_potatoes - (homer_rate * time_alone)) / (homer_rate + christen_rate) * christen_rate

theorem christen_peeled_20_potatoes :
  christen_potatoes = 20 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_christen_peeled_20_potatoes_l393_39309


namespace NUMINAMATH_GPT_sum_infinite_series_eq_l393_39336

theorem sum_infinite_series_eq : 
  ∑' n : ℕ, (n + 1) * (1 / 999 : ℝ) ^ n = 1000 / 998 := by
sorry

end NUMINAMATH_GPT_sum_infinite_series_eq_l393_39336


namespace NUMINAMATH_GPT_Gilda_marbles_left_l393_39349

theorem Gilda_marbles_left (M : ℝ) (h1 : M > 0) :
  let remaining_after_pedro := M - 0.30 * M
  let remaining_after_ebony := remaining_after_pedro - 0.40 * remaining_after_pedro
  remaining_after_ebony / M * 100 = 42 :=
by
  sorry

end NUMINAMATH_GPT_Gilda_marbles_left_l393_39349


namespace NUMINAMATH_GPT_part1_part2_l393_39386

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (x - 1) + a
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x + Real.log x

theorem part1 (x : ℝ) (hx : 0 < x) :
  f x 0 ≥ g x 0 + 1 := sorry

theorem part2 {x0 : ℝ} (hx0 : ∃ y0 : ℝ, f x0 0 = g x0 0 ∧ ∀ x ≠ x0, f x 0 ≠ g x 0) :
  x0 < 2 := sorry

end NUMINAMATH_GPT_part1_part2_l393_39386


namespace NUMINAMATH_GPT_total_blocks_in_pyramid_l393_39377

-- Define the number of blocks in each layer
def blocks_in_layer (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n+1) => 3 * blocks_in_layer n

-- Prove the total number of blocks in the four-layer pyramid
theorem total_blocks_in_pyramid : 
  (blocks_in_layer 0) + (blocks_in_layer 1) + (blocks_in_layer 2) + (blocks_in_layer 3) = 40 :=
by
  sorry

end NUMINAMATH_GPT_total_blocks_in_pyramid_l393_39377


namespace NUMINAMATH_GPT_total_grapes_l393_39390

theorem total_grapes (r a n : ℕ) (h1 : r = 25) (h2 : a = r + 2) (h3 : n = a + 4) : r + a + n = 83 := by
  sorry

end NUMINAMATH_GPT_total_grapes_l393_39390


namespace NUMINAMATH_GPT_chocolate_bars_sold_last_week_l393_39305

-- Definitions based on conditions
def initial_chocolate_bars : Nat := 18
def chocolate_bars_sold_this_week : Nat := 7
def chocolate_bars_needed_to_sell : Nat := 6

-- Define the number of chocolate bars sold so far
def chocolate_bars_sold_so_far : Nat := chocolate_bars_sold_this_week + chocolate_bars_needed_to_sell

-- Target statement to prove
theorem chocolate_bars_sold_last_week :
  initial_chocolate_bars - chocolate_bars_sold_so_far = 5 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bars_sold_last_week_l393_39305


namespace NUMINAMATH_GPT_solve_for_x_l393_39335

theorem solve_for_x (x : ℚ) :
  (x^2 - 4*x + 3) / (x^2 - 7*x + 6) = (x^2 - 3*x - 10) / (x^2 - 2*x - 15) →
  x = -3 / 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l393_39335


namespace NUMINAMATH_GPT_amount_spent_on_milk_l393_39310

-- Define conditions
def monthly_salary (S : ℝ) := 0.10 * S = 1800
def rent := 5000
def groceries := 4500
def education := 2500
def petrol := 2000
def miscellaneous := 700
def total_expenses (S : ℝ) := S - 1800
def known_expenses := rent + groceries + education + petrol + miscellaneous

-- Define the proof problem
theorem amount_spent_on_milk (S : ℝ) (milk : ℝ) :
  monthly_salary S →
  total_expenses S = known_expenses + milk →
  milk = 1500 :=
by
  sorry

end NUMINAMATH_GPT_amount_spent_on_milk_l393_39310


namespace NUMINAMATH_GPT_x_power_expression_l393_39328

theorem x_power_expression (x : ℝ) (h : x^3 - 3 * x = 5) : x^5 - 27 * x^2 = -22 * x^2 + 9 * x + 15 :=
by
  --proof goes here
  sorry

end NUMINAMATH_GPT_x_power_expression_l393_39328


namespace NUMINAMATH_GPT_complement_of_67_is_23_l393_39366

-- Define complement function
def complement (x : ℝ) : ℝ := 90 - x

-- State the theorem
theorem complement_of_67_is_23 : complement 67 = 23 := 
by
  sorry

end NUMINAMATH_GPT_complement_of_67_is_23_l393_39366


namespace NUMINAMATH_GPT_total_pies_l393_39320

-- Define the number of each type of pie.
def apple_pies : Nat := 2
def pecan_pies : Nat := 4
def pumpkin_pies : Nat := 7

-- Prove the total number of pies.
theorem total_pies : apple_pies + pecan_pies + pumpkin_pies = 13 := by
  sorry

end NUMINAMATH_GPT_total_pies_l393_39320


namespace NUMINAMATH_GPT_vec_op_l393_39379

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (2, -2)
def two_a : ℝ × ℝ := (2 * 2, 2 * 1)
def result : ℝ × ℝ := (two_a.1 - b.1, two_a.2 - b.2)

theorem vec_op : (2 * a.1 - b.1, 2 * a.2 - b.2) = (2, 4) := by
  sorry

end NUMINAMATH_GPT_vec_op_l393_39379


namespace NUMINAMATH_GPT_max_possible_N_in_cities_l393_39340

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end NUMINAMATH_GPT_max_possible_N_in_cities_l393_39340


namespace NUMINAMATH_GPT_compute_f_of_1_plus_g_of_3_l393_39354

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 1

theorem compute_f_of_1_plus_g_of_3 : f (1 + g 3) = 29 := by 
  sorry

end NUMINAMATH_GPT_compute_f_of_1_plus_g_of_3_l393_39354


namespace NUMINAMATH_GPT_value_of_bc_l393_39353

theorem value_of_bc (a b c d : ℝ) (h1 : a + b = 14) (h2 : c + d = 3) (h3 : a + d = 8) : b + c = 9 :=
sorry

end NUMINAMATH_GPT_value_of_bc_l393_39353


namespace NUMINAMATH_GPT_tangent_line_circle_l393_39311

theorem tangent_line_circle (m : ℝ) : 
  (∀ (x y : ℝ), x + y + m = 0 → x^2 + y^2 = m) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_circle_l393_39311


namespace NUMINAMATH_GPT_square_root_of_25_squared_l393_39327

theorem square_root_of_25_squared :
  Real.sqrt (25 ^ 2) = 25 :=
sorry

end NUMINAMATH_GPT_square_root_of_25_squared_l393_39327


namespace NUMINAMATH_GPT_adjusted_area_difference_l393_39334

noncomputable def largest_circle_area (d : ℝ) : ℝ :=
  let r := d / 2
  r^2 * Real.pi

noncomputable def middle_circle_area (r : ℝ) : ℝ :=
  r^2 * Real.pi

noncomputable def smaller_circle_area (r : ℝ) : ℝ :=
  r^2 * Real.pi

theorem adjusted_area_difference (d_large r_middle r_small : ℝ) 
  (h_large : d_large = 30) (h_middle : r_middle = 10) (h_small : r_small = 5) :
  largest_circle_area d_large - middle_circle_area r_middle - smaller_circle_area r_small = 100 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_adjusted_area_difference_l393_39334


namespace NUMINAMATH_GPT_roots_quadratic_eq_l393_39341

theorem roots_quadratic_eq :
  (∃ a b : ℝ, (a + b = 8) ∧ (a * b = 8) ∧ (a^2 + b^2 = 48)) :=
sorry

end NUMINAMATH_GPT_roots_quadratic_eq_l393_39341


namespace NUMINAMATH_GPT_inverse_of_p_l393_39383

variables {p q r : Prop}

theorem inverse_of_p (m n : Prop) (hp : p = (m → n)) (hq : q = (¬m → ¬n)) (hr : r = (n → m)) : r = p ∧ r = (n → m) :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_p_l393_39383


namespace NUMINAMATH_GPT_bob_total_profit_l393_39393

def initial_cost (num_dogs : ℕ) (cost_per_dog : ℕ) : ℕ := num_dogs * cost_per_dog

def revenue (num_puppies : ℕ) (price_per_puppy : ℕ) : ℕ := num_puppies * price_per_puppy

def total_profit (initial_cost : ℕ) (revenue : ℕ) : ℕ := revenue - initial_cost

theorem bob_total_profit (c1 : initial_cost 2 250 = 500)
                        (c2 : revenue 6 350 = 2100)
                        (c3 : total_profit 500 2100 = 1600) :
  total_profit (initial_cost 2 250) (revenue 6 350) = 1600 := by
  sorry

end NUMINAMATH_GPT_bob_total_profit_l393_39393


namespace NUMINAMATH_GPT_balls_in_boxes_with_one_in_one_balls_in_boxes_with_two_empty_balls_in_boxes_with_three_empty_balls_in_boxes_A_not_less_B_l393_39361

noncomputable def ways_with_ball_in_box_one : Nat := 369
noncomputable def ways_with_two_empty_boxes : Nat := 360
noncomputable def ways_with_three_empty_boxes : Nat := 140
noncomputable def ways_ball_A_not_less_than_B : Nat := 375

theorem balls_in_boxes_with_one_in_one 
  (n_balls : Nat) (n_boxes : Nat) 
  (ball_1 : Nat) :
  n_balls = 4 → n_boxes = 5 → ball_1 = 1 → 
  ∃ ways, ways = ways_with_ball_in_box_one := 
sorry

theorem balls_in_boxes_with_two_empty 
  (n_balls : Nat) (n_boxes : Nat) 
  (empty_boxes : Nat) :
  n_balls = 4 → n_boxes = 5 → empty_boxes = 2 → 
  ∃ ways, ways = ways_with_two_empty_boxes := 
sorry

theorem balls_in_boxes_with_three_empty 
  (n_balls : Nat) (n_boxes : Nat) 
  (empty_boxes : Nat) :
  n_balls = 4 → n_boxes = 5 → empty_boxes = 3 → 
  ∃ ways, ways = ways_with_three_empty_boxes := 
sorry

theorem balls_in_boxes_A_not_less_B 
  (n_balls : Nat) (n_boxes : Nat) 
  (ball_A : Nat) (ball_B : Nat) :
  n_balls = 4 → n_boxes = 5 → ball_A ≠ ball_B →
  ∃ ways, ways = ways_ball_A_not_less_than_B := 
sorry

end NUMINAMATH_GPT_balls_in_boxes_with_one_in_one_balls_in_boxes_with_two_empty_balls_in_boxes_with_three_empty_balls_in_boxes_A_not_less_B_l393_39361


namespace NUMINAMATH_GPT_find_x_l393_39368

theorem find_x (x : ℕ) (hx : x > 0) : 1^(x + 3) + 2^(x + 2) + 3^x + 4^(x + 1) = 1958 → x = 4 :=
sorry

end NUMINAMATH_GPT_find_x_l393_39368


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l393_39314

theorem arithmetic_sequence_problem (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 9 = 81)
  (h3 : a (k - 4) = 191)
  (h4 : S k = 10000) :
  k = 100 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l393_39314


namespace NUMINAMATH_GPT_hammers_ordered_in_october_l393_39385

theorem hammers_ordered_in_october
  (ordered_in_june : Nat)
  (ordered_in_july : Nat)
  (ordered_in_august : Nat)
  (ordered_in_september : Nat)
  (pattern_increase : ∀ n : Nat, ordered_in_june + n = ordered_in_july ∧ ordered_in_july + (n + 1) = ordered_in_august ∧ ordered_in_august + (n + 2) = ordered_in_september) :
  ordered_in_september + 4 = 13 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_hammers_ordered_in_october_l393_39385


namespace NUMINAMATH_GPT_arithmetic_computation_l393_39321

theorem arithmetic_computation : 65 * 1515 - 25 * 1515 = 60600 := by
  sorry

end NUMINAMATH_GPT_arithmetic_computation_l393_39321


namespace NUMINAMATH_GPT_min_value_expression_l393_39389

theorem min_value_expression (x y : ℝ) : ∃ (a b : ℝ), x = a ∧ y = b ∧ (x^2 + y^2 - 8*x - 6*y + 30 = 5) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l393_39389


namespace NUMINAMATH_GPT_oblong_perimeter_182_l393_39380

variables (l w : ℕ) (x : ℤ)

def is_oblong (l w : ℕ) : Prop :=
l * w = 4624 ∧ l = 4 * x ∧ w = 3 * x

theorem oblong_perimeter_182 (l w x : ℕ) (hlw : is_oblong l w x) : 
  2 * l + 2 * w = 182 :=
by
  sorry

end NUMINAMATH_GPT_oblong_perimeter_182_l393_39380


namespace NUMINAMATH_GPT_sin_of_angle_l393_39381

theorem sin_of_angle (α : ℝ) (h : Real.cos (π + α) = -(1/3)) : Real.sin ((3 * π / 2) - α) = -(1/3) := 
by
  sorry

end NUMINAMATH_GPT_sin_of_angle_l393_39381


namespace NUMINAMATH_GPT_value_of_a_l393_39339

theorem value_of_a 
  (x y a : ℝ)
  (h1 : 2 * x + y = 3 * a)
  (h2 : x - 2 * y = 9 * a)
  (h3 : x + 3 * y = 24) :
  a = -4 :=
sorry

end NUMINAMATH_GPT_value_of_a_l393_39339


namespace NUMINAMATH_GPT_pauline_total_spending_l393_39387

theorem pauline_total_spending
  (total_before_tax : ℝ)
  (sales_tax_rate : ℝ)
  (h₁ : total_before_tax = 150)
  (h₂ : sales_tax_rate = 0.08) :
  total_before_tax + total_before_tax * sales_tax_rate = 162 :=
by {
  -- Proof here
  sorry
}

end NUMINAMATH_GPT_pauline_total_spending_l393_39387


namespace NUMINAMATH_GPT_possible_values_of_Q_l393_39395

theorem possible_values_of_Q (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (x + y) / z = (y + z) / x ∧ (y + z) / x = (z + x) / y) :
  ∃ Q : ℝ, Q = 8 ∨ Q = -1 := 
sorry

end NUMINAMATH_GPT_possible_values_of_Q_l393_39395


namespace NUMINAMATH_GPT_xiaoyangs_scores_l393_39367

theorem xiaoyangs_scores (average : ℕ) (diff : ℕ) (h_average : average = 96) (h_diff : diff = 8) :
  ∃ chinese_score math_score : ℕ, chinese_score = 92 ∧ math_score = 100 :=
by
  sorry

end NUMINAMATH_GPT_xiaoyangs_scores_l393_39367


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l393_39308

noncomputable def a := (3 / 5 : ℝ) ^ (2 / 5)
noncomputable def b := (2 / 5 : ℝ) ^ (3 / 5)
noncomputable def c := (2 / 5 : ℝ) ^ (2 / 5)

theorem relationship_between_a_b_c :
  a > c ∧ c > b :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l393_39308


namespace NUMINAMATH_GPT_prob_one_mistake_eq_l393_39317

-- Define the probability of making a mistake on a single question
def prob_mistake : ℝ := 0.1

-- Define the probability of answering correctly on a single question
def prob_correct : ℝ := 1 - prob_mistake

-- Define the probability of answering all three questions correctly
def three_correct : ℝ := prob_correct ^ 3

-- Define the probability of making at least one mistake in three questions
def prob_at_least_one_mistake := 1 - three_correct

-- The theorem states that the above probability is equal to 1 - 0.9^3
theorem prob_one_mistake_eq :
  prob_at_least_one_mistake = 1 - (0.9 ^ 3) :=
by
  sorry

end NUMINAMATH_GPT_prob_one_mistake_eq_l393_39317


namespace NUMINAMATH_GPT_log_inequality_region_l393_39342

theorem log_inequality_region (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hx1 : x ≠ 1) (hx2 : x ≠ y) :
  (0 < x ∧ x < 1 ∧ 0 < y ∧ y < x) 
  ∨ (1 < x ∧ y > x) ↔ (Real.log y / Real.log x ≥ Real.log (x * y) / Real.log (x / y)) :=
  sorry

end NUMINAMATH_GPT_log_inequality_region_l393_39342


namespace NUMINAMATH_GPT_divide_equally_l393_39347

-- Define the input values based on the conditions.
def brother_strawberries := 3 * 15
def kimberly_strawberries := 8 * brother_strawberries
def parents_strawberries := kimberly_strawberries - 93
def total_strawberries := brother_strawberries + kimberly_strawberries + parents_strawberries
def family_members := 4

-- Define the theorem to prove the question.
theorem divide_equally : 
    (total_strawberries / family_members) = 168 :=
by
    -- (proof goes here)
    sorry

end NUMINAMATH_GPT_divide_equally_l393_39347


namespace NUMINAMATH_GPT_zoey_finishes_on_wednesday_l393_39363

noncomputable def day_zoey_finishes (n : ℕ) : String :=
  let total_days := (n * (n + 1)) / 2
  match total_days % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | 6 => "Saturday"
  | _ => "Error"

theorem zoey_finishes_on_wednesday : day_zoey_finishes 18 = "Wednesday" :=
by
  -- Calculate that Zoey takes 171 days to read 18 books
  -- Recall that 171 mod 7 = 3, so she finishes on "Wednesday"
  sorry

end NUMINAMATH_GPT_zoey_finishes_on_wednesday_l393_39363


namespace NUMINAMATH_GPT_probability_at_least_one_female_is_five_sixths_l393_39343

-- Declare the total number of male and female students
def total_male_students := 6
def total_female_students := 4
def total_students := total_male_students + total_female_students
def selected_students := 3

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to select 3 students from 10 students
def total_ways_to_select_3 := binomial_coefficient total_students selected_students

-- Ways to select 3 male students from 6 male students
def ways_to_select_3_males := binomial_coefficient total_male_students selected_students

-- Probability of selecting at least one female student
def probability_of_at_least_one_female : ℚ := 1 - (ways_to_select_3_males / total_ways_to_select_3)

-- The theorem statement to be proved
theorem probability_at_least_one_female_is_five_sixths :
  probability_of_at_least_one_female = 5/6 := by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_female_is_five_sixths_l393_39343


namespace NUMINAMATH_GPT_carl_additional_hours_per_week_l393_39356

def driving_hours_per_day : ℕ := 2

def days_per_week : ℕ := 7

def total_hours_two_weeks_after_promotion : ℕ := 40

def driving_hours_per_week_before_promotion : ℕ := driving_hours_per_day * days_per_week

def driving_hours_per_week_after_promotion : ℕ := total_hours_two_weeks_after_promotion / 2

def additional_hours_per_week : ℕ := driving_hours_per_week_after_promotion - driving_hours_per_week_before_promotion

theorem carl_additional_hours_per_week : 
  additional_hours_per_week = 6 :=
by
  -- Using plain arithmetic based on given definitions
  sorry

end NUMINAMATH_GPT_carl_additional_hours_per_week_l393_39356


namespace NUMINAMATH_GPT_chess_team_boys_l393_39302

-- Definitions based on the conditions
def members : ℕ := 30
def attendees : ℕ := 20

-- Variables representing boys (B) and girls (G)
variables (B G : ℕ)

-- Defining the conditions
def condition1 : Prop := B + G = members
def condition2 : Prop := (2 * G) / 3 + B = attendees

-- The problem statement: proving that B = 0
theorem chess_team_boys (h1 : condition1 B G) (h2 : condition2 B G) : B = 0 :=
  sorry

end NUMINAMATH_GPT_chess_team_boys_l393_39302


namespace NUMINAMATH_GPT_sum_of_series_l393_39391

theorem sum_of_series : 
  (1 / (1 * 2) + 1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7)) = 6 / 7 := 
 by sorry

end NUMINAMATH_GPT_sum_of_series_l393_39391


namespace NUMINAMATH_GPT_geometric_to_arithmetic_l393_39371

theorem geometric_to_arithmetic (a_1 a_2 a_3 b_1 b_2 b_3: ℝ) (ha: a_1 > 0 ∧ a_2 > 0 ∧ a_3 > 0 ∧ b_1 > 0 ∧ b_2 > 0 ∧ b_3 > 0)
  (h_geometric_a : ∃ q : ℝ, a_2 = a_1 * q ∧ a_3 = a_1 * q^2)
  (h_geometric_b : ∃ q₁ : ℝ, b_2 = b_1 * q₁ ∧ b_3 = b_1 * q₁^2)
  (h_sum : a_1 + a_2 + a_3 = b_1 + b_2 + b_3)
  (h_arithmetic : 2 * a_2 * b_2 = a_1 * b_1 + a_3 * b_3) : 
  a_2 = b_2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_to_arithmetic_l393_39371


namespace NUMINAMATH_GPT_female_students_in_first_class_l393_39304

theorem female_students_in_first_class
  (females_in_second_class : ℕ)
  (males_in_first_class : ℕ)
  (males_in_second_class : ℕ)
  (males_in_third_class : ℕ)
  (females_in_third_class : ℕ)
  (extra_students : ℕ)
  (total_students_need_partners : ℕ)
  (total_males : ℕ := males_in_first_class + males_in_second_class + males_in_third_class)
  (total_females : ℕ := females_in_second_class + females_in_third_class)
  (females_in_first_class : ℕ)
  (females : ℕ := females_in_first_class + total_females) :
  (females_in_second_class = 18) →
  (males_in_first_class = 17) →
  (males_in_second_class = 14) →
  (males_in_third_class = 15) →
  (females_in_third_class = 17) →
  (extra_students = 2) →
  (total_students_need_partners = total_males - extra_students) →
  females = total_students_need_partners →
  females_in_first_class = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_female_students_in_first_class_l393_39304


namespace NUMINAMATH_GPT_chocolates_left_l393_39350

-- Definitions based on the conditions
def initially_bought := 3
def gave_away := 2
def additionally_bought := 3

-- Theorem statement to prove
theorem chocolates_left : initially_bought - gave_away + additionally_bought = 4 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_chocolates_left_l393_39350


namespace NUMINAMATH_GPT_polynomial_arithmetic_sequence_roots_l393_39357

theorem polynomial_arithmetic_sequence_roots (p q : ℝ) (h : ∃ a b c d : ℝ, 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a + 3*(b - a) = b ∧ b + 3*(c - b) = c ∧ c + 3*(d - c) = d ∧ 
  (a^4 + p * a^2 + q = 0) ∧ (b^4 + p * b^2 + q = 0) ∧ 
  (c^4 + p * c^2 + q = 0) ∧ (d^4 + p * d^2 + q = 0)) :
  p ≤ 0 ∧ q = 0.09 * p^2 := 
sorry

end NUMINAMATH_GPT_polynomial_arithmetic_sequence_roots_l393_39357


namespace NUMINAMATH_GPT_julia_tuesday_kids_l393_39397

-- Definitions based on conditions
def kids_on_monday : ℕ := 11
def tuesday_more_than_monday : ℕ := 1

-- The main statement to be proved
theorem julia_tuesday_kids : (kids_on_monday + tuesday_more_than_monday) = 12 := by
  sorry

end NUMINAMATH_GPT_julia_tuesday_kids_l393_39397


namespace NUMINAMATH_GPT_length_of_goods_train_l393_39376

-- Define the given conditions
def speed_kmph := 72
def platform_length := 260
def crossing_time := 26

-- Convert speed to m/s
def speed_mps := (speed_kmph * 5) / 18

-- Calculate distance covered
def distance_covered := speed_mps * crossing_time

-- Define the length of the train
def train_length := distance_covered - platform_length

theorem length_of_goods_train : train_length = 260 := by
  sorry

end NUMINAMATH_GPT_length_of_goods_train_l393_39376


namespace NUMINAMATH_GPT_boys_count_l393_39322

def total_pupils : ℕ := 485
def number_of_girls : ℕ := 232
def number_of_boys : ℕ := total_pupils - number_of_girls

theorem boys_count : number_of_boys = 253 := by
  -- The proof is omitted according to instruction
  sorry

end NUMINAMATH_GPT_boys_count_l393_39322


namespace NUMINAMATH_GPT_black_ants_employed_l393_39372

theorem black_ants_employed (total_ants : ℕ) (red_ants : ℕ) 
  (h1 : total_ants = 900) (h2 : red_ants = 413) :
    total_ants - red_ants = 487 :=
by
  -- The proof is given below.
  sorry

end NUMINAMATH_GPT_black_ants_employed_l393_39372


namespace NUMINAMATH_GPT_minimize_value_l393_39373

noncomputable def minimize_y (a b x : ℝ) : ℝ := (x - a) ^ 3 + (x - b) ^ 3

theorem minimize_value (a b : ℝ) : ∃ x : ℝ, minimize_y a b x = minimize_y a b a ∨ minimize_y a b x = minimize_y a b b :=
sorry

end NUMINAMATH_GPT_minimize_value_l393_39373


namespace NUMINAMATH_GPT_find_last_two_digits_l393_39374

noncomputable def tenth_digit (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ) : ℕ :=
d7 + d8

noncomputable def ninth_digit (d1 d2 d3 d4 d5 d6 d7 : ℕ) : ℕ :=
d6 + d7

theorem find_last_two_digits :
  ∃ d9 d10 : ℕ, d9 = ninth_digit 1 1 2 3 5 8 13 ∧ d10 = tenth_digit 1 1 2 3 5 8 13 21 :=
by
  sorry

end NUMINAMATH_GPT_find_last_two_digits_l393_39374


namespace NUMINAMATH_GPT_Liked_Both_Proof_l393_39324

section DessertProblem

variable (Total_Students Liked_Apple_Pie Liked_Chocolate_Cake Did_Not_Like_Either Liked_Both : ℕ)
variable (h1 : Total_Students = 50)
variable (h2 : Liked_Apple_Pie = 25)
variable (h3 : Liked_Chocolate_Cake = 20)
variable (h4 : Did_Not_Like_Either = 10)

theorem Liked_Both_Proof :
  Liked_Both = (Liked_Apple_Pie + Liked_Chocolate_Cake) - (Total_Students - Did_Not_Like_Either) :=
by
  sorry

end DessertProblem

end NUMINAMATH_GPT_Liked_Both_Proof_l393_39324


namespace NUMINAMATH_GPT_max_sum_first_n_terms_is_S_5_l393_39370

open Nat

-- Define the arithmetic sequence and the conditions.
variable {a : ℕ → ℝ} -- The arithmetic sequence {a_n}
variable {d : ℝ} -- The common difference of the arithmetic sequence
variable {S : ℕ → ℝ} -- The sum of the first n terms of the sequence a

-- Hypotheses corresponding to the conditions in the problem
lemma a_5_positive : a 5 > 0 := sorry
lemma a_4_plus_a_7_negative : a 4 + a 7 < 0 := sorry

-- Statement to prove that the maximum value of the sum of the first n terms is S_5 given the conditions
theorem max_sum_first_n_terms_is_S_5 :
  (∀ (n : ℕ), S n ≤ S 5) :=
sorry

end NUMINAMATH_GPT_max_sum_first_n_terms_is_S_5_l393_39370


namespace NUMINAMATH_GPT_range_of_m_l393_39346

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m*x^2 + m*x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l393_39346


namespace NUMINAMATH_GPT_find_x_l393_39315

theorem find_x (x : ℝ) : (x = 2 ∨ x = -2) ↔ (|x|^2 - 5 * |x| + 6 = 0 ∧ x^2 - 4 = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l393_39315


namespace NUMINAMATH_GPT_representable_by_expression_l393_39362

theorem representable_by_expression (n : ℕ) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (n = (x * y + y * z + z * x) / (x + y + z)) ↔ n ≠ 1 := by
  sorry

end NUMINAMATH_GPT_representable_by_expression_l393_39362


namespace NUMINAMATH_GPT_greatest_positive_integer_N_l393_39325

def condition (x : Int) (y : Int) : Prop :=
  (x^2 - x * y) % 1111 ≠ 0

theorem greatest_positive_integer_N :
  ∃ N : Nat, (∀ (x : Fin N) (y : Fin N), x ≠ y → condition x y) ∧ N = 1000 :=
by
  sorry

end NUMINAMATH_GPT_greatest_positive_integer_N_l393_39325


namespace NUMINAMATH_GPT_find_speed_of_stream_l393_39323

theorem find_speed_of_stream (x : ℝ) (h1 : ∃ x, 1 / (39 - x) = 2 * (1 / (39 + x))) : x = 13 :=
by
sorry

end NUMINAMATH_GPT_find_speed_of_stream_l393_39323


namespace NUMINAMATH_GPT_prove_final_value_is_111_l393_39306

theorem prove_final_value_is_111 :
  let initial_num := 16
  let doubled_num := initial_num * 2
  let added_five := doubled_num + 5
  let trebled_result := added_five * 3
  trebled_result = 111 :=
by
  sorry

end NUMINAMATH_GPT_prove_final_value_is_111_l393_39306


namespace NUMINAMATH_GPT_power_greater_than_any_l393_39396

theorem power_greater_than_any {p M : ℝ} (hp : p > 0) (hM : M > 0) : ∃ n : ℕ, (1 + p)^n > M :=
by
  sorry

end NUMINAMATH_GPT_power_greater_than_any_l393_39396


namespace NUMINAMATH_GPT_example_equation_l393_39375

-- Define what it means to be an equation in terms of containing an unknown and being an equality
def is_equation (expr : Prop) (contains_unknown : Prop) : Prop :=
  (contains_unknown ∧ expr)

-- Prove that 4x + 2 = 10 is an equation
theorem example_equation : is_equation (4 * x + 2 = 10) (∃ x : ℝ, true) :=
  by sorry

end NUMINAMATH_GPT_example_equation_l393_39375


namespace NUMINAMATH_GPT_area_of_third_face_l393_39331

-- Define the variables for the dimensions of the box: l, w, and h
variables (l w h: ℝ)

-- Given conditions
def face1_area := 120
def face2_area := 72
def volume := 720

-- The relationships between the dimensions and the given areas/volume
def face1_eq : Prop := l * w = face1_area
def face2_eq : Prop := w * h = face2_area
def volume_eq : Prop := l * w * h = volume

-- The statement we need to prove is that the area of the third face (l * h) is 60 cm² given the above equations
theorem area_of_third_face :
  face1_eq l w →
  face2_eq w h →
  volume_eq l w h →
  l * h = 60 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_area_of_third_face_l393_39331
