import Mathlib

namespace graph_not_in_third_quadrant_l9_9848

-- Define the conditions
variable (m : ℝ)
variable (h1 : 0 < m)
variable (h2 : m < 2)

-- Define the graph equation
noncomputable def line_eq (x : ℝ) : ℝ := (m - 2) * x + m

-- The proof problem: the graph does not pass through the third quadrant
theorem graph_not_in_third_quadrant : ¬ ∃ x y : ℝ, (x < 0 ∧ y < 0 ∧ y = (m - 2) * x + m) :=
sorry

end graph_not_in_third_quadrant_l9_9848


namespace num_two_digit_math_representation_l9_9454

-- Define the problem space
def unique_digits (n : ℕ) : Prop := 
  n >= 1 ∧ n <= 9

-- Representation of the characters' assignment
def representation (x y z w : ℕ) : Prop :=
  unique_digits x ∧ unique_digits y ∧ unique_digits z ∧ unique_digits w ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧ 
  x = z ∧ 3 * (10 * y + z) = 10 * w + x

-- The main theorem to prove
theorem num_two_digit_math_representation : 
  ∃ x y z w, representation x y z w :=
sorry

end num_two_digit_math_representation_l9_9454


namespace find_min_value_l9_9485

theorem find_min_value (a x y : ℝ) (h : y = -x^2 + 3 * Real.log x) : ∃ x, ∃ y, (a - x)^2 + (a + 2 - y)^2 = 8 :=
by
  sorry

end find_min_value_l9_9485


namespace probability_of_a_b_c_l9_9770

noncomputable def probability_condition : ℚ :=
  5 / 6 * 5 / 6 * 7 / 8

theorem probability_of_a_b_c : 
  let a_outcome := 6
  let b_outcome := 6
  let c_outcome := 8
  (1 / a_outcome) * (1 / b_outcome) * (1 / c_outcome) = probability_condition :=
sorry

end probability_of_a_b_c_l9_9770


namespace xy_identity_l9_9445

theorem xy_identity (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) : x^2 + y^2 = 5 :=
  sorry

end xy_identity_l9_9445


namespace frank_hawaiian_slices_l9_9822

theorem frank_hawaiian_slices:
  ∀ (total_slices dean_slices sammy_slices leftover_slices frank_slices : ℕ),
  total_slices = 24 →
  dean_slices = 6 →
  sammy_slices = 4 →
  leftover_slices = 11 →
  (total_slices - leftover_slices) = (dean_slices + sammy_slices + frank_slices) →
  frank_slices = 3 :=
by
  intros total_slices dean_slices sammy_slices leftover_slices frank_slices
  intros h_total h_dean h_sammy h_leftovers h_total_eaten
  sorry

end frank_hawaiian_slices_l9_9822


namespace angela_initial_action_figures_l9_9764

theorem angela_initial_action_figures (X : ℕ) (h1 : X - (1/4 : ℚ) * X - (1/3 : ℚ) * (3/4 : ℚ) * X = 12) : X = 24 :=
sorry

end angela_initial_action_figures_l9_9764


namespace closure_property_of_A_l9_9135

theorem closure_property_of_A 
  (a b c d k1 k2 : ℤ) 
  (x y : ℤ) 
  (Hx : x = a^2 + k1 * a * b + b^2) 
  (Hy : y = c^2 + k2 * c * d + d^2) : 
  ∃ m k : ℤ, x * y = m * (a^2 + k * a * b + b^2) := 
  by 
    -- this is where the proof would go
    sorry

end closure_property_of_A_l9_9135


namespace comb_10_3_eq_120_l9_9540

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l9_9540


namespace solution_per_beaker_l9_9518

theorem solution_per_beaker (solution_per_tube : ℕ) (num_tubes : ℕ) (num_beakers : ℕ)
    (h1 : solution_per_tube = 7) (h2 : num_tubes = 6) (h3 : num_beakers = 3) :
    (solution_per_tube * num_tubes) / num_beakers = 14 :=
by
  sorry

end solution_per_beaker_l9_9518


namespace factorize_expression_l9_9183

theorem factorize_expression (m : ℝ) : 3 * m^2 - 12 = 3 * (m + 2) * (m - 2) := 
sorry

end factorize_expression_l9_9183


namespace license_plate_increase_l9_9114

-- definitions from conditions
def old_plates_count : ℕ := 26 ^ 2 * 10 ^ 3
def new_plates_count : ℕ := 26 ^ 4 * 10 ^ 2

-- theorem stating the increase in the number of license plates
theorem license_plate_increase : 
  (new_plates_count : ℚ) / (old_plates_count : ℚ) = 26 ^ 2 / 10 :=
by
  sorry

end license_plate_increase_l9_9114


namespace exists_triangle_with_side_lengths_l9_9031

theorem exists_triangle_with_side_lengths (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : 
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

end exists_triangle_with_side_lengths_l9_9031


namespace total_number_of_cats_l9_9059

def Cat := Type -- Define a type of Cat.

variable (A B C: Cat) -- Declaring three cats A, B, and C.

variable (kittens_A: Fin 4 → {gender : Bool // (2 : Fin 4).val = 2 ∧ (2 : Fin 4).val = 2}) -- 4 kittens: 2 males, 2 females.
variable (kittens_B: Fin 3 → {gender : Bool // (1 : Fin 3).val = 1 ∧ (2 : Fin 3).val = 2}) -- 3 kittens: 1 male, 2 females.
variable (kittens_C: Fin 5 → {gender : Bool // (3 : Fin 5).val = 3 ∧ (2 : Fin 5).val = 2}) -- 5 kittens: 3 males, 2 females.

variable (extra_kittens: Fin 2 → {gender : Bool // (1 : Fin 2).val = 1 ∧ (1 : Fin 2).val = 1}) -- 2 kittens of the additional female kitten of Cat A.

theorem total_number_of_cats : 
  3 + 4 + 2 + 3 + 5 = 17 :=
by
  sorry

end total_number_of_cats_l9_9059


namespace olya_candies_l9_9023

theorem olya_candies (P M T O : ℕ) (h1 : P + M + T + O = 88) (h2 : 1 ≤ P) (h3 : 1 ≤ M) (h4 : 1 ≤ T) (h5 : 1 ≤ O) (h6 : M + T = 57) (h7 : P > M) (h8 : P > T) (h9 : P > O) : O = 1 :=
by
  sorry

end olya_candies_l9_9023


namespace car_dealership_sales_l9_9018

theorem car_dealership_sales (x : ℕ)
  (h1 : 5 * x = 30 * 8)
  (h2 : 30 + x = 78) : 
  x = 48 :=
sorry

end car_dealership_sales_l9_9018


namespace find_n_divisors_l9_9614

theorem find_n_divisors (n : ℕ) (h1 : 2287 % n = 2028 % n)
                        (h2 : 2028 % n = 1806 % n) : n = 37 := 
by
  sorry

end find_n_divisors_l9_9614


namespace man_work_alone_in_5_days_l9_9247

theorem man_work_alone_in_5_days (d : ℕ) (h1 : ∀ m : ℕ, (1 / (m : ℝ)) + 1 / 20 = 1 / 4):
  d = 5 := by
  sorry

end man_work_alone_in_5_days_l9_9247


namespace number_is_minus_three_l9_9875

variable (x a : ℝ)

theorem number_is_minus_three (h1 : a = 0.5) (h2 : x / (a - 3) = 3 / (a + 2)) : x = -3 :=
by
  sorry

end number_is_minus_three_l9_9875


namespace find_k_for_two_identical_solutions_l9_9397

theorem find_k_for_two_identical_solutions (k : ℝ) :
  (∃ x : ℝ, x^2 = 4 * x + k) ∧ (∀ x : ℝ, x^2 = 4 * x + k → x = 2) ↔ k = -4 :=
by
  sorry

end find_k_for_two_identical_solutions_l9_9397


namespace max_students_total_l9_9374

def max_students_class (a b : ℕ) (h : 3 * a + 5 * b = 115) : ℕ :=
  a + b

theorem max_students_total :
  ∃ a b : ℕ, 3 * a + 5 * b = 115 ∧ max_students_class a b (by sorry) = 37 :=
sorry

end max_students_total_l9_9374


namespace total_travel_ways_l9_9895

-- Define the number of car departures
def car_departures : ℕ := 3

-- Define the number of train departures
def train_departures : ℕ := 4

-- Define the number of ship departures
def ship_departures : ℕ := 2

-- The total number of ways to travel from location A to location B
def total_ways : ℕ := car_departures + train_departures + ship_departures

-- The theorem stating the total number of ways to travel given the conditions
theorem total_travel_ways :
  total_ways = 9 :=
by
  -- Proof goes here
  sorry

end total_travel_ways_l9_9895


namespace positive_solution_unique_m_l9_9115

theorem positive_solution_unique_m (m : ℝ) : ¬ (4 < m ∧ m < 2) :=
by
  sorry

end positive_solution_unique_m_l9_9115


namespace system_solution_l9_9423

theorem system_solution (x y : ℝ) (h1 : x + 5*y = 5) (h2 : 3*x - y = 3) : x + y = 2 := 
by
  sorry

end system_solution_l9_9423


namespace divide_square_into_smaller_squares_l9_9473

def P (n : ℕ) : Prop := sorry /- Define the property of dividing a square into n smaller squares -/

theorem divide_square_into_smaller_squares (n : ℕ) (h : n > 5) : P n :=
  sorry

end divide_square_into_smaller_squares_l9_9473


namespace Gina_tip_is_5_percent_l9_9992

noncomputable def Gina_tip_percentage : ℝ := 5

theorem Gina_tip_is_5_percent (bill_amount : ℝ) (good_tipper_percentage : ℝ)
    (good_tipper_extra_tip_cents : ℝ) (good_tipper_tip : ℝ) 
    (Gina_tip_extra_cents : ℝ):
    bill_amount = 26 ∧
    good_tipper_percentage = 20 ∧
    Gina_tip_extra_cents = 390 ∧
    good_tipper_tip = (20 / 100) * 26 ∧
    Gina_tip_extra_cents = 390 ∧
    (Gina_tip_percentage / 100) * bill_amount + (Gina_tip_extra_cents / 100) = good_tipper_tip
    → Gina_tip_percentage = 5 :=
by
  sorry

end Gina_tip_is_5_percent_l9_9992


namespace number_of_tins_per_day_for_rest_of_week_l9_9079
-- Import necessary library

-- Define conditions as Lean definitions
def d1 : ℕ := 50
def d2 : ℕ := 3 * d1
def d3 : ℕ := d2 - 50
def total_target : ℕ := 500

-- Define what we need to prove
theorem number_of_tins_per_day_for_rest_of_week :
  ∃ (dr : ℕ), d1 + d2 + d3 + 4 * dr = total_target ∧ dr = 50 :=
by
  sorry

end number_of_tins_per_day_for_rest_of_week_l9_9079


namespace slope_and_angle_of_inclination_l9_9431

noncomputable def line_slope_and_inclination : Prop :=
  ∀ (x y : ℝ), (x - y - 3 = 0) → (∃ m : ℝ, m = 1) ∧ (∃ θ : ℝ, θ = 45)

theorem slope_and_angle_of_inclination (x y : ℝ) (h : x - y - 3 = 0) : line_slope_and_inclination :=
by
  sorry

end slope_and_angle_of_inclination_l9_9431


namespace find_bullet_l9_9904

theorem find_bullet (x y : ℝ) (h₁ : 3 * x + y = 8) (h₂ : y = -1) : 2 * x - y = 7 :=
sorry

end find_bullet_l9_9904


namespace triple_solution_unique_l9_9489

theorem triple_solution_unique (a b c n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : 0 < n) :
  (a^2 + b^2 = n * Nat.lcm a b + n^2) ∧
  (b^2 + c^2 = n * Nat.lcm b c + n^2) ∧
  (c^2 + a^2 = n * Nat.lcm c a + n^2) →
  (a = n ∧ b = n ∧ c = n) :=
by
  sorry

end triple_solution_unique_l9_9489


namespace min_value_of_a_plus_b_minus_c_l9_9042

theorem min_value_of_a_plus_b_minus_c (a b c : ℝ)
  (h : ∀ x y : ℝ, 3 * x + 4 * y - 5 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ 3 * x + 4 * y + 5) :
  a = 3 ∧ b = 4 ∧ -5 ≤ c ∧ c ≤ 5 ∧ a + b - c = 2 :=
by {
  sorry
}

end min_value_of_a_plus_b_minus_c_l9_9042


namespace prove_composite_k_l9_9704

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := ∃ p q, p > 1 ∧ q > 1 ∧ n = p * q

def problem_statement (a b c d : ℕ) (h : a * b = c * d) : Prop :=
  is_composite (a^1984 + b^1984 + c^1984 + d^1984)

-- The theorem to prove
theorem prove_composite_k (a b c d : ℕ) (h : a * b = c * d) : 
  problem_statement a b c d h := sorry

end prove_composite_k_l9_9704


namespace contradiction_divisible_by_2_l9_9885

open Nat

theorem contradiction_divisible_by_2 (a b : ℕ) (h : (a * b) % 2 = 0) : a % 2 = 0 ∨ b % 2 = 0 :=
by
  sorry

end contradiction_divisible_by_2_l9_9885


namespace treasures_first_level_is_4_l9_9074

-- Definitions based on conditions
def points_per_treasure : ℕ := 5
def treasures_second_level : ℕ := 3
def score_second_level : ℕ := treasures_second_level * points_per_treasure
def total_score : ℕ := 35
def points_first_level : ℕ := total_score - score_second_level

-- Main statement to prove
theorem treasures_first_level_is_4 : points_first_level / points_per_treasure = 4 := 
by
  -- We are skipping the proof here and using sorry.
  sorry

end treasures_first_level_is_4_l9_9074


namespace max_participants_win_at_least_three_matches_l9_9067

theorem max_participants_win_at_least_three_matches (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 66 ∧ ∀ m : ℕ, (k * 3 ≤ m) ∧ (m ≤ 199) → k ≤ m / 3 := 
by
  sorry

end max_participants_win_at_least_three_matches_l9_9067


namespace find_z_l9_9702

theorem find_z (a b p q : ℝ) (z : ℝ) 
  (cond : (z + a + b = q * (p * z - a - b))) : 
  z = (a + b) * (q + 1) / (p * q - 1) :=
sorry

end find_z_l9_9702


namespace combined_rate_mpg_900_over_41_l9_9666

-- Declare the variables and conditions
variables {d : ℕ} (h_d_pos : d > 0)

def combined_mpg (d : ℕ) : ℚ :=
  let anna_car_gasoline := (d : ℚ) / 50
  let ben_car_gasoline  := (d : ℚ) / 20
  let carl_car_gasoline := (d : ℚ) / 15
  let total_gasoline    := anna_car_gasoline + ben_car_gasoline + carl_car_gasoline
  ((3 : ℚ) * d) / total_gasoline

-- Define the theorem statement
theorem combined_rate_mpg_900_over_41 :
  ∀ d : ℕ, d > 0 → combined_mpg d = 900 / 41 :=
by
  intros d h_d_pos
  rw [combined_mpg]
  -- Steps following the solution
  sorry -- proof omitted

end combined_rate_mpg_900_over_41_l9_9666


namespace probability_of_blue_buttons_l9_9879

theorem probability_of_blue_buttons
  (orig_red_A : ℕ) (orig_blue_A : ℕ)
  (removed_red : ℕ) (removed_blue : ℕ)
  (target_ratio : ℚ)
  (final_red_A : ℕ) (final_blue_A : ℕ)
  (final_red_B : ℕ) (final_blue_B : ℕ)
  (orig_buttons_A : orig_red_A + orig_blue_A = 16)
  (removed_buttons : removed_red = 3 ∧ removed_blue = 5)
  (final_buttons_A : final_red_A + final_blue_A = 8)
  (buttons_ratio : target_ratio = 2 / 3)
  (final_ratio_A : final_red_A + final_blue_A = target_ratio * 16)
  (red_in_A : final_red_A = orig_red_A - removed_red)
  (blue_in_A : final_blue_A = orig_blue_A - removed_blue)
  (red_in_B : final_red_B = removed_red)
  (blue_in_B : final_blue_B = removed_blue):
  (final_blue_A / (final_red_A + final_blue_A)) * (final_blue_B / (final_red_B + final_blue_B)) = 25 / 64 := 
by
  sorry

end probability_of_blue_buttons_l9_9879


namespace arithmetic_sequence_length_l9_9346

theorem arithmetic_sequence_length : 
  let a := 11
  let d := 5
  let l := 101
  ∃ n : ℕ, a + (n-1) * d = l ∧ n = 19 := 
by
  sorry

end arithmetic_sequence_length_l9_9346


namespace race_distance_l9_9302

theorem race_distance (d v_A v_B v_C : ℝ) (h1 : d / v_A = (d - 20) / v_B)
  (h2 : d / v_B = (d - 10) / v_C) (h3 : d / v_A = (d - 28) / v_C) : d = 100 :=
by
  sorry

end race_distance_l9_9302


namespace find_f_of_2011_l9_9742

-- Define the function f
def f (x : ℝ) (a b c : ℝ) := a * x^5 + b * x^3 + c * x + 7

-- The main statement we need to prove
theorem find_f_of_2011 (a b c : ℝ) (h : f (-2011) a b c = -17) : f 2011 a b c = 31 :=
by
  sorry

end find_f_of_2011_l9_9742


namespace quadratic_no_real_roots_l9_9624

theorem quadratic_no_real_roots (c : ℝ) (h : c > 1) : ∀ x : ℝ, x^2 + 2 * x + c ≠ 0 :=
by
  sorry

end quadratic_no_real_roots_l9_9624


namespace range_of_k_for_real_roots_l9_9106

theorem range_of_k_for_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by 
  sorry

end range_of_k_for_real_roots_l9_9106


namespace sqrt_inequality_l9_9727

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) :=
sorry

end sqrt_inequality_l9_9727


namespace sin_cos_fourth_power_l9_9516

theorem sin_cos_fourth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) : Real.sin θ ^ 4 + Real.cos θ ^ 4 = 63 / 64 :=
by
  sorry

end sin_cos_fourth_power_l9_9516


namespace segment_length_segment_fraction_three_segments_fraction_l9_9385

noncomputable def total_length : ℝ := 4
noncomputable def number_of_segments : ℕ := 5

theorem segment_length (L : ℝ) (n : ℕ) (hL : L = total_length) (hn : n = number_of_segments) :
  L / n = (4 / 5 : ℝ) := by
sorry

theorem segment_fraction (n : ℕ) (hn : n = number_of_segments) :
  (1 / n : ℝ) = (1 / 5 : ℝ) := by
sorry

theorem three_segments_fraction (n : ℕ) (hn : n = number_of_segments) :
  (3 / n : ℝ) = (3 / 5 : ℝ) := by
sorry

end segment_length_segment_fraction_three_segments_fraction_l9_9385


namespace face_value_of_share_l9_9370

theorem face_value_of_share (FV : ℝ) (market_value : ℝ) (dividend_rate : ℝ) (desired_return_rate : ℝ) 
  (H1 : market_value = 15) 
  (H2 : dividend_rate = 0.09) 
  (H3 : desired_return_rate = 0.12) 
  (H4 : dividend_rate * FV = desired_return_rate * market_value) :
  FV = 20 := 
by
  sorry

end face_value_of_share_l9_9370


namespace number_of_roots_l9_9210

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 2 * b * x + 3 * c

theorem number_of_roots (a b c x₁ x₂ : ℝ) (h_extreme : x₁ ≠ x₂)
    (h_fx1 : f a b c x₁ = x₁) :
    (∃ (r : ℝ), 3 * (f a b c r)^2 + 4 * a * (f a b c r) + 2 * b = 0) :=
sorry

end number_of_roots_l9_9210


namespace daragh_initial_bears_l9_9252

variables (initial_bears eden_initial_bears eden_final_bears favorite_bears shared_bears_per_sister : ℕ)
variables (sisters : ℕ)

-- Given conditions
axiom h1 : eden_initial_bears = 10
axiom h2 : eden_final_bears = 14
axiom h3 : favorite_bears = 8
axiom h4 : sisters = 3

-- Derived condition
axiom h5 : shared_bears_per_sister = eden_final_bears - eden_initial_bears
axiom h6 : initial_bears = favorite_bears + (shared_bears_per_sister * sisters)

-- The theorem to prove
theorem daragh_initial_bears : initial_bears = 20 :=
by
  -- Insert proof here
  sorry

end daragh_initial_bears_l9_9252


namespace right_angled_triangles_l9_9756

theorem right_angled_triangles (x y z : ℕ) : (x - 6) * (y - 6) = 18 ∧ (x^2 + y^2 = z^2)
  → (3 * (x + y + z) = x * y) :=
sorry

end right_angled_triangles_l9_9756


namespace equation_solution_count_l9_9192

open Real

theorem equation_solution_count :
  ∃ s : Finset ℝ, (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ sin (π / 4 * sin x) = cos (π / 4 * cos x)) ∧ s.card = 4 :=
by
  sorry

end equation_solution_count_l9_9192


namespace white_given_popped_l9_9534

-- Define the conditions
def white_kernels : ℚ := 1 / 2
def yellow_kernels : ℚ := 1 / 3
def blue_kernels : ℚ := 1 / 6

def white_kernels_pop : ℚ := 3 / 4
def yellow_kernels_pop : ℚ := 1 / 2
def blue_kernels_pop : ℚ := 1 / 3

def probability_white_popped : ℚ := white_kernels * white_kernels_pop
def probability_yellow_popped : ℚ := yellow_kernels * yellow_kernels_pop
def probability_blue_popped : ℚ := blue_kernels * blue_kernels_pop

def probability_popped : ℚ := probability_white_popped + probability_yellow_popped + probability_blue_popped

-- The theorem to be proved
theorem white_given_popped : (probability_white_popped / probability_popped) = (27 / 43) := 
by sorry

end white_given_popped_l9_9534


namespace arithmetic_and_geometric_sequence_l9_9915

theorem arithmetic_and_geometric_sequence (a : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + 2) 
  (h_geom_seq : (a 2)^2 = a 0 * a 3) : 
  a 1 + a 2 = -10 := 
sorry

end arithmetic_and_geometric_sequence_l9_9915


namespace largest_possible_number_of_sweets_in_each_tray_l9_9913

-- Define the initial conditions as given in the problem statement
def tim_sweets : ℕ := 36
def peter_sweets : ℕ := 44

-- Define the statement that we want to prove
theorem largest_possible_number_of_sweets_in_each_tray :
  Nat.gcd tim_sweets peter_sweets = 4 :=
by
  sorry

end largest_possible_number_of_sweets_in_each_tray_l9_9913


namespace frood_points_smallest_frood_points_l9_9322

theorem frood_points (n : ℕ) (h : n > 9) : (n * (n + 1)) / 2 > 5 * n :=
by {
  sorry
}

noncomputable def smallest_n : ℕ := 10

theorem smallest_frood_points (m : ℕ) (h : (m * (m + 1)) / 2 > 5 * m) : 10 ≤ m :=
by {
  sorry
}

end frood_points_smallest_frood_points_l9_9322


namespace hank_route_distance_l9_9372

theorem hank_route_distance 
  (d : ℝ) 
  (h1 : ∃ t1 : ℝ, t1 = d / 70 ∧ t1 = d / 70 + 1 / 60) 
  (h2 : ∃ t2 : ℝ, t2 = d / 75 ∧ t2 = d / 75 - 1 / 60) 
  (time_diff : (d / 70 - d / 75) = 1 / 30) : 
  d = 35 :=
sorry

end hank_route_distance_l9_9372


namespace gcd_seq_coprime_l9_9723

def seq (n : ℕ) : ℕ := 2^(2^n) + 1

theorem gcd_seq_coprime (n k : ℕ) (hnk : n ≠ k) : Nat.gcd (seq n) (seq k) = 1 :=
by
  sorry

end gcd_seq_coprime_l9_9723


namespace distribute_weights_l9_9003

theorem distribute_weights (max_weight : ℕ) (w_gbeans w_milk w_carrots w_apples w_bread w_rice w_oranges w_pasta : ℕ)
  (h_max_weight : max_weight = 20)
  (h_w_gbeans : w_gbeans = 4)
  (h_w_milk : w_milk = 6)
  (h_w_carrots : w_carrots = 2 * w_gbeans)
  (h_w_apples : w_apples = 3)
  (h_w_bread : w_bread = 1)
  (h_w_rice : w_rice = 5)
  (h_w_oranges : w_oranges = 2)
  (h_w_pasta : w_pasta = 3)
  : (w_gbeans + w_milk + w_carrots + w_apples + w_bread - 2 = max_weight) ∧ 
    (w_rice + w_oranges + w_pasta + 2 ≤ max_weight) :=
by
  sorry

end distribute_weights_l9_9003


namespace hyperbola_standard_equation_l9_9044

theorem hyperbola_standard_equation
  (passes_through : ∀ {x y : ℝ}, (x, y) = (1, 1) → 2 * x + y = 0 ∨ 2 * x - y = 0)
  (asymptote1 : ∀ {x y : ℝ}, 2 * x + y = 0 → y = -2 * x)
  (asymptote2 : ∀ {x y : ℝ}, 2 * x - y = 0 → y = 2 * x) :
  ∃ a b : ℝ, a = 4 / 3 ∧ b = 1 / 3 ∧ ∀ x y : ℝ, (x, y) = (1, 1) → (x^2 / a - y^2 / b = 1) := 
sorry

end hyperbola_standard_equation_l9_9044


namespace number_of_committees_correct_l9_9148

noncomputable def number_of_committees (teams members host_selection non_host_selection : ℕ) : ℕ :=
  have ways_to_choose_host := teams
  have ways_to_choose_four_from_seven := Nat.choose members host_selection
  have ways_to_choose_two_from_seven := Nat.choose members non_host_selection
  have total_non_host_combinations := ways_to_choose_two_from_seven ^ (teams - 1)
  ways_to_choose_host * ways_to_choose_four_from_seven * total_non_host_combinations

theorem number_of_committees_correct :
  number_of_committees 5 7 4 2 = 34134175 := by
  sorry

end number_of_committees_correct_l9_9148


namespace expression_evaluation_l9_9945

theorem expression_evaluation :
  (40 - (2040 - 210)) + (2040 - (210 - 40)) = 80 :=
by
  sorry

end expression_evaluation_l9_9945


namespace penny_half_dollar_same_probability_l9_9932

def probability_penny_half_dollar_same : ℚ :=
  1 / 2

theorem penny_half_dollar_same_probability :
  probability_penny_half_dollar_same = 1 / 2 :=
by
  sorry

end penny_half_dollar_same_probability_l9_9932


namespace probability_same_color_l9_9285

/-
Problem statement:
Given a bag contains 6 green balls and 7 white balls,
if two balls are drawn simultaneously, prove that the probability 
that both balls are the same color is 6/13.
-/

theorem probability_same_color
  (total_balls : ℕ := 6 + 7)
  (green_balls : ℕ := 6)
  (white_balls : ℕ := 7)
  (two_balls_drawn_simultaneously : Prop := true) :
  ((green_balls / total_balls) * ((green_balls - 1) / (total_balls - 1))) +
  ((white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))) = 6 / 13 :=
sorry

end probability_same_color_l9_9285


namespace multiplication_is_correct_l9_9792

theorem multiplication_is_correct : 209 * 209 = 43681 := sorry

end multiplication_is_correct_l9_9792


namespace eval_expression_l9_9137

theorem eval_expression : (-2 ^ 3) ^ (1/3 : ℝ) - (-1 : ℝ) ^ 0 = -3 := by 
  sorry

end eval_expression_l9_9137


namespace find_max_z_l9_9280

theorem find_max_z :
  ∃ (x y : ℝ), abs x + abs y ≤ 4 ∧ 2 * x + y ≤ 4 ∧ (2 * x - y) = (20 / 3) :=
by
  sorry

end find_max_z_l9_9280


namespace total_swim_distance_five_weeks_total_swim_time_five_weeks_l9_9659

-- Definitions of swim distances and times based on Jasmine's routine 
def monday_laps : ℕ := 10
def tuesday_laps : ℕ := 15
def tuesday_aerobics_time : ℕ := 20
def wednesday_laps : ℕ := 12
def wednesday_time_per_lap : ℕ := 2
def thursday_laps : ℕ := 18
def friday_laps : ℕ := 20

-- Proving total swim distance for five weeks
theorem total_swim_distance_five_weeks : (5 * (monday_laps + tuesday_laps + wednesday_laps + thursday_laps + friday_laps)) = 375 := 
by 
  sorry

-- Proving total swim time for five weeks (partially solvable)
theorem total_swim_time_five_weeks : (5 * (tuesday_aerobics_time + wednesday_laps * wednesday_time_per_lap)) = 220 := 
by 
  sorry

end total_swim_distance_five_weeks_total_swim_time_five_weeks_l9_9659


namespace calc_3_pow_6_mul_4_pow_6_l9_9447

theorem calc_3_pow_6_mul_4_pow_6 : (3^6) * (4^6) = 2985984 :=
by 
  sorry

end calc_3_pow_6_mul_4_pow_6_l9_9447


namespace three_pow_sub_two_pow_prime_power_prime_l9_9415

theorem three_pow_sub_two_pow_prime_power_prime (n : ℕ) (hn : n > 0) (hp : ∃ p k : ℕ, Nat.Prime p ∧ 3^n - 2^n = p^k) : Nat.Prime n := 
sorry

end three_pow_sub_two_pow_prime_power_prime_l9_9415


namespace range_of_ab_l9_9678

theorem range_of_ab (a b : ℝ) 
  (h1: ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 1 = 0 → (2 * a * x - b * y + 2 = 0)) : 
  ab ≤ 0 :=
sorry

end range_of_ab_l9_9678


namespace total_sales_correct_l9_9201

def normal_sales_per_month : ℕ := 21122
def additional_sales_in_june : ℕ := 3922
def sales_in_june : ℕ := normal_sales_per_month + additional_sales_in_june
def sales_in_july : ℕ := normal_sales_per_month
def total_sales : ℕ := sales_in_june + sales_in_july

theorem total_sales_correct :
  total_sales = 46166 :=
by
  -- Proof goes here
  sorry

end total_sales_correct_l9_9201


namespace cars_without_paying_l9_9654

theorem cars_without_paying (total_cars : ℕ) (percent_with_tickets : ℚ) (fraction_with_passes : ℚ)
  (h1 : total_cars = 300)
  (h2 : percent_with_tickets = 0.75)
  (h3 : fraction_with_passes = 1/5) :
  let cars_with_tickets := percent_with_tickets * total_cars
  let cars_with_passes := fraction_with_passes * cars_with_tickets
  total_cars - (cars_with_tickets + cars_with_passes) = 30 :=
by
  -- Placeholder proof
  sorry

end cars_without_paying_l9_9654


namespace number_of_ways_to_form_team_l9_9434

theorem number_of_ways_to_form_team (boys girls : ℕ) (select_boys select_girls : ℕ)
    (H_boys : boys = 7) (H_girls : girls = 9) (H_select_boys : select_boys = 2) (H_select_girls : select_girls = 3) :
    (Nat.choose boys select_boys) * (Nat.choose girls select_girls) = 1764 := by
  rw [H_boys, H_girls, H_select_boys, H_select_girls]
  sorry

end number_of_ways_to_form_team_l9_9434


namespace value_of_f_at_neg1_l9_9089

def f (x : ℤ) : ℤ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

theorem value_of_f_at_neg1 : f (-1) = 6 :=
by
  sorry

end value_of_f_at_neg1_l9_9089


namespace outer_boundary_diameter_l9_9531

-- Define the given conditions
def fountain_diameter : ℝ := 12
def walking_path_width : ℝ := 6
def garden_ring_width : ℝ := 10

-- Define what we need to prove
theorem outer_boundary_diameter :
  2 * (fountain_diameter / 2 + garden_ring_width + walking_path_width) = 44 :=
by
  sorry

end outer_boundary_diameter_l9_9531


namespace star_evaluation_l9_9497

def star (a b : ℕ) : ℕ := 3 + b^(a + 1)

theorem star_evaluation : star (star 2 3) 2 = 3 + 2^31 :=
by {
  sorry
}

end star_evaluation_l9_9497


namespace frustum_lateral_surface_area_l9_9286

/-- A frustum of a right circular cone has the following properties:
  * Lower base radius r1 = 8 inches
  * Upper base radius r2 = 2 inches
  * Height h = 6 inches
  The lateral surface area of such a frustum is 60 * √2 * π square inches.
-/
theorem frustum_lateral_surface_area : 
  let r1 := 8 
  let r2 := 2 
  let h := 6 
  let s := Real.sqrt (h^2 + (r1 - r2)^2)
  A = π * (r1 + r2) * s :=
  sorry

end frustum_lateral_surface_area_l9_9286


namespace sluice_fill_time_l9_9794

noncomputable def sluice_open_equal_time (x y : ℝ) (m : ℝ) : ℝ :=
  -- Define time (t) required for both sluice gates to be open equally to fill the lake
  m / 11

theorem sluice_fill_time :
  ∀ (x y : ℝ),
    (10 * x + 14 * y = 9900) →
    (18 * x + 12 * y = 9900) →
    sluice_open_equal_time x y 9900 = 900 := sorry

end sluice_fill_time_l9_9794


namespace clothing_discount_l9_9304

theorem clothing_discount (P : ℝ) :
  let first_sale_price := (4 / 5) * P
  let second_sale_price := first_sale_price * 0.60
  second_sale_price = (12 / 25) * P :=
by
  sorry

end clothing_discount_l9_9304


namespace geometric_sequence_sixth_term_correct_l9_9389

noncomputable def geometric_sequence_sixth_term (a r : ℝ) (pos_a : 0 < a) (pos_r : 0 < r)
    (third_term : a * r^2 = 27)
    (ninth_term : a * r^8 = 3) : ℝ :=
  a * r^5

theorem geometric_sequence_sixth_term_correct (a r : ℝ) (pos_a : 0 < a) (pos_r : 0 < r) 
    (third_term : a * r^2 = 27)
    (ninth_term : a * r^8 = 3) : geometric_sequence_sixth_term a r pos_a pos_r third_term ninth_term = 9 := 
sorry

end geometric_sequence_sixth_term_correct_l9_9389


namespace find_number_of_pencils_l9_9574

-- Define the conditions
def number_of_people : Nat := 6
def notebooks_per_person : Nat := 9
def number_of_notebooks : Nat := number_of_people * notebooks_per_person
def pencils_multiplier : Nat := 6
def number_of_pencils : Nat := pencils_multiplier * number_of_notebooks

-- Prove the main statement
theorem find_number_of_pencils : number_of_pencils = 324 :=
by
  sorry

end find_number_of_pencils_l9_9574


namespace interval_of_x_l9_9105

theorem interval_of_x (x : ℝ) : (4 * x > 2) ∧ (4 * x < 5) ∧ (5 * x > 2) ∧ (5 * x < 5) ↔ (x > 1/2) ∧ (x < 1) := 
by 
  sorry

end interval_of_x_l9_9105


namespace no_solution_exists_l9_9438

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l9_9438


namespace total_price_of_order_l9_9707

-- Define the price of each item
def price_ice_cream_bar : ℝ := 0.60
def price_sundae : ℝ := 1.40

-- Define the quantity of each item
def quantity_ice_cream_bar : ℕ := 125
def quantity_sundae : ℕ := 125

-- Calculate the costs
def cost_ice_cream_bar := quantity_ice_cream_bar * price_ice_cream_bar
def cost_sundae := quantity_sundae * price_sundae

-- Calculate the total cost
def total_cost := cost_ice_cream_bar + cost_sundae

-- Statement of the theorem
theorem total_price_of_order : total_cost = 250 := 
by {
  sorry
}

end total_price_of_order_l9_9707


namespace algebraic_expression_value_l9_9435

variable (a b : ℝ)

theorem algebraic_expression_value
  (h : a^2 + 2 * b^2 - 1 = 0) :
  (a - b)^2 + b * (2 * a + b) = 1 :=
by
  sorry

end algebraic_expression_value_l9_9435


namespace probability_male_is_2_5_l9_9032

variable (num_male_students num_female_students : ℕ)

def total_students (num_male_students num_female_students : ℕ) : ℕ :=
  num_male_students + num_female_students

def probability_of_male (num_male_students num_female_students : ℕ) : ℚ :=
  num_male_students / (total_students num_male_students num_female_students : ℚ)

theorem probability_male_is_2_5 :
  probability_of_male 2 3 = 2 / 5 := by
    sorry

end probability_male_is_2_5_l9_9032


namespace positive_y_percent_y_eq_16_l9_9978

theorem positive_y_percent_y_eq_16 (y : ℝ) (hy : 0 < y) (h : 0.01 * y * y = 16) : y = 40 :=
by
  sorry

end positive_y_percent_y_eq_16_l9_9978


namespace other_person_time_to_complete_job_l9_9474

-- Define the conditions
def SureshTime : ℕ := 15
def SureshWorkHours : ℕ := 9
def OtherPersonWorkHours : ℕ := 4

-- The proof problem: Prove that the other person can complete the job in 10 hours.
theorem other_person_time_to_complete_job (x : ℕ) 
  (h1 : ∀ SureshWorkHours SureshTime, SureshWorkHours * (1 / SureshTime) = (SureshWorkHours / SureshTime) ∧ 
       4 * (SureshWorkHours / SureshTime / 4) = 1) : 
  (x = 10) :=
sorry

end other_person_time_to_complete_job_l9_9474


namespace denomination_of_remaining_coins_l9_9672

/-
There are 324 coins total.
The total value of the coins is Rs. 70.
There are 220 coins of 20 paise each.
Find the denomination of the remaining coins.
-/

def total_coins := 324
def total_value := 7000 -- Rs. 70 converted into paise
def num_20_paise_coins := 220
def value_20_paise_coin := 20
  
theorem denomination_of_remaining_coins :
  let total_remaining_value := total_value - (num_20_paise_coins * value_20_paise_coin)
  let num_remaining_coins := total_coins - num_20_paise_coins
  num_remaining_coins > 0 →
  total_remaining_value / num_remaining_coins = 25 :=
by
  sorry

end denomination_of_remaining_coins_l9_9672


namespace ratio_female_to_male_members_l9_9826

theorem ratio_female_to_male_members (f m : ℕ)
  (h1 : 35 * f = SumAgesFemales)
  (h2 : 20 * m = SumAgesMales)
  (h3 : (35 * f + 20 * m) / (f + m) = 25) :
  f / m = 1 / 2 := by
  sorry

end ratio_female_to_male_members_l9_9826


namespace find_complex_z_l9_9569

theorem find_complex_z (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : z / (1 - 2 * i) = i) :
  z = 2 + i :=
sorry

end find_complex_z_l9_9569


namespace digit_possibilities_757_l9_9035

theorem digit_possibilities_757
  (N : ℕ)
  (h : N < 10) :
  (∃ d₀ d₁ d₂ : ℕ, (d₀ = 2 ∨ d₀ = 5 ∨ d₀ = 8) ∧
  (d₁ = 2 ∨ d₁ = 5 ∨ d₁ = 8) ∧
  (d₂ = 2 ∨ d₂ = 5 ∨ d₂ = 8) ∧
  (d₀ ≠ d₁) ∧
  (d₀ ≠ d₂) ∧
  (d₁ ≠ d₂)) :=
by
  sorry

end digit_possibilities_757_l9_9035


namespace break_even_price_l9_9967

noncomputable def initial_investment : ℝ := 1500
noncomputable def cost_per_tshirt : ℝ := 3
noncomputable def num_tshirts_break_even : ℝ := 83
noncomputable def total_cost_equipment_tshirts : ℝ := initial_investment + (cost_per_tshirt * num_tshirts_break_even)
noncomputable def price_per_tshirt := total_cost_equipment_tshirts / num_tshirts_break_even

theorem break_even_price : price_per_tshirt = 21.07 := by
  sorry

end break_even_price_l9_9967


namespace arithmetic_sequence_geo_ratio_l9_9900

theorem arithmetic_sequence_geo_ratio
  (a_n : ℕ → ℝ)
  (d : ℝ)
  (h_nonzero : d ≠ 0)
  (S : ℕ → ℝ)
  (h_seq : ∀ n, S n = (n * (2 * a_n 1 + (n - 1) * d)) / 2)
  (h_geo : (S 2) ^ 2 = S 1 * S 4) :
  (a_n 2 + a_n 3) / a_n 1 = 8 :=
by sorry

end arithmetic_sequence_geo_ratio_l9_9900


namespace determine_x_l9_9422

theorem determine_x 
  (w : ℤ) (hw : w = 90)
  (z : ℤ) (hz : z = 4 * w + 40)
  (y : ℤ) (hy : y = 3 * z + 15)
  (x : ℤ) (hx : x = 2 * y + 6) :
  x = 2436 := 
by
  sorry

end determine_x_l9_9422


namespace inverse_proportion_function_m_neg_l9_9590

theorem inverse_proportion_function_m_neg
  (x : ℝ) (y : ℝ) (m : ℝ)
  (h1 : y = m / x)
  (h2 : (x < 0 → y > 0) ∧ (x > 0 → y < 0)) :
  m < 0 :=
sorry

end inverse_proportion_function_m_neg_l9_9590


namespace intersection_A_B_l9_9175

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l9_9175


namespace line_equation_l9_9960

def line_through (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let m := (y₂ - y₁) / (x₂ - x₁)
  y - y₁ = m * (x - x₁)

noncomputable def is_trisection_point (A B QR : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (qx, qy) := QR
  (qx = (2 * x₂ + x₁) / 3 ∧ qy = (2 * y₂ + y₁) / 3) ∨
  (qx = (x₂ + 2 * x₁) / 3 ∧ qy = (y₂ + 2 * y₁) / 3)

theorem line_equation (A B P Q : ℝ × ℝ)
  (hA : A = (3, 4))
  (hB : B = (-4, 5))
  (hP : is_trisection_point B A P)
  (hQ : is_trisection_point B A Q) :
  line_through A P 1 3 ∨ line_through A P 2 1 → 
  (line_through A P 3 4 → P = (1, 3)) ∧ 
  (line_through A P 2 1 → P = (2, 1)) ∧ 
  (line_through A P x y → x - 4 * y + 13 = 0) := 
by 
  sorry

end line_equation_l9_9960


namespace number_of_zeros_among_50_numbers_l9_9845

theorem number_of_zeros_among_50_numbers :
  ∀ (m n p : ℕ), (m + n + p = 50) → (m * p = 500) → n = 5 :=
by
  intros m n p h1 h2
  sorry

end number_of_zeros_among_50_numbers_l9_9845


namespace closest_years_l9_9929

theorem closest_years (a b c d : ℕ) (h1 : 10 * a + b + 10 * c + d = 10 * b + c) :
  (a = 1 ∧ b = 8 ∧ c = 6 ∧ d = 8) ∨ (a = 2 ∧ b = 3 ∧ c = 0 ∧ d =7) ↔
  ((10 * 1 + 8 + 10 * 6 + 8 = 10 * 8 + 6) ∧ (10 * 2 + 3 + 10 * 0 + 7 = 10 * 3 + 0)) :=
sorry

end closest_years_l9_9929


namespace find_x_angle_l9_9550

theorem find_x_angle (x : ℝ) (h : x + x + 140 = 360) : x = 110 :=
by
  sorry

end find_x_angle_l9_9550


namespace team_order_l9_9760

-- Define the points of teams
variables (A B C D : ℕ)

-- State the conditions
def condition1 := A + C = B + D
def condition2 := B + A + 5 ≤ D + C
def condition3 := B + C ≥ A + D + 3

-- Statement of the theorem
theorem team_order (h1 : condition1 A B C D) (h2 : condition2 A B C D) (h3 : condition3 A B C D) :
  C > D ∧ D > B ∧ B > A :=
sorry

end team_order_l9_9760


namespace a_plus_b_is_24_l9_9267

theorem a_plus_b_is_24 (a b : ℤ) (h1 : 0 < b) (h2 : b < a) (h3 : a * (a + 3 * b) = 550) : a + b = 24 :=
sorry

end a_plus_b_is_24_l9_9267


namespace min_p_q_sum_l9_9905

theorem min_p_q_sum (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h : 162 * p = q^3) : p + q = 54 :=
sorry

end min_p_q_sum_l9_9905


namespace car_second_hour_speed_l9_9889

theorem car_second_hour_speed (s1 s2 : ℕ) (h1 : s1 = 100) (avg : (s1 + s2) / 2 = 80) : s2 = 60 :=
by
  sorry

end car_second_hour_speed_l9_9889


namespace problem_inequality_l9_9369

open Real

theorem problem_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_prod : x * y * z = 1) :
    1 / (x^3 * y) + 1 / (y^3 * z) + 1 / (z^3 * x) ≥ x * y + y * z + z * x :=
sorry

end problem_inequality_l9_9369


namespace gcd_problem_l9_9367

-- Define the conditions
def a (d : ℕ) : ℕ := d - 3
def b (d : ℕ) : ℕ := d - 2
def c (d : ℕ) : ℕ := d - 1

-- Define the number formed by digits in the specific form
def abcd (d : ℕ) : ℕ := 1000 * a d + 100 * b d + 10 * c d + d
def dcba (d : ℕ) : ℕ := 1000 * d + 100 * c d + 10 * b d + a d

-- Summing the two numbers
def num_sum (d : ℕ) : ℕ := abcd d + dcba d

-- The GCD of all num_sum(d) where d ranges from 3 to 9
def gcd_of_nums : ℕ := 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (num_sum 3) (num_sum 4)) (num_sum 5)) (num_sum 6)) (Nat.gcd (num_sum 7) (Nat.gcd (num_sum 8) (num_sum 9)))

theorem gcd_problem : gcd_of_nums = 1111 := sorry

end gcd_problem_l9_9367


namespace perfect_square_solutions_l9_9124

theorem perfect_square_solutions :
  {n : ℕ | ∃ m : ℕ, n^2 + 77 * n = m^2} = {4, 99, 175, 1444} :=
by
  sorry

end perfect_square_solutions_l9_9124


namespace sum_of_first_6033_terms_l9_9949

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_6033_terms (a r : ℝ) (h1 : geometric_sum a r 2011 = 200) 
  (h2 : geometric_sum a r 4022 = 380) : 
  geometric_sum a r 6033 = 542 :=
sorry

end sum_of_first_6033_terms_l9_9949


namespace number_of_zeros_f_l9_9152

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.log x

theorem number_of_zeros_f : ∃! n : ℕ, n = 2 ∧ ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end number_of_zeros_f_l9_9152


namespace quadratic_expression_odd_quadratic_expression_not_square_l9_9318

theorem quadratic_expression_odd (n : ℕ) : 
  (n^2 + n + 1) % 2 = 1 := 
by sorry

theorem quadratic_expression_not_square (n : ℕ) : 
  ¬ ∃ (m : ℕ), m^2 = n^2 + n + 1 := 
by sorry

end quadratic_expression_odd_quadratic_expression_not_square_l9_9318


namespace find_number_l9_9382

theorem find_number (x k : ℕ) (h1 : x / k = 4) (h2 : k = 16) : x = 64 := by
  sorry

end find_number_l9_9382


namespace efficiency_and_days_l9_9563

noncomputable def sakshi_efficiency : ℝ := 1 / 25
noncomputable def tanya_efficiency : ℝ := 1.25 * sakshi_efficiency
noncomputable def ravi_efficiency : ℝ := 0.70 * sakshi_efficiency
noncomputable def combined_efficiency : ℝ := sakshi_efficiency + tanya_efficiency + ravi_efficiency
noncomputable def days_to_complete_work : ℝ := 1 / combined_efficiency

theorem efficiency_and_days:
  combined_efficiency = 29.5 / 250 ∧
  days_to_complete_work = 250 / 29.5 :=
by
  sorry

end efficiency_and_days_l9_9563


namespace ratio_of_shares_l9_9700

-- Definitions for the given conditions
def capital_A : ℕ := 4500
def capital_B : ℕ := 16200
def months_A : ℕ := 12
def months_B : ℕ := 5 -- B joined after 7 months

-- Effective capital contributions
def effective_capital_A : ℕ := capital_A * months_A
def effective_capital_B : ℕ := capital_B * months_B

-- Defining the statement to prove
theorem ratio_of_shares : effective_capital_A / Nat.gcd effective_capital_A effective_capital_B = 2 ∧ effective_capital_B / Nat.gcd effective_capital_A effective_capital_B = 3 := by
  sorry

end ratio_of_shares_l9_9700


namespace distance_between_A_and_B_l9_9582

-- Definitions for the problem
def speed_fast_train := 65 -- speed of the first train in km/h
def speed_slow_train := 29 -- speed of the second train in km/h
def time_difference := 5   -- difference in hours

-- Given conditions and the final equation leading to the proof
theorem distance_between_A_and_B :
  ∃ (D : ℝ), D = 9425 / 36 :=
by
  existsi (9425 / 36 : ℝ)
  sorry

end distance_between_A_and_B_l9_9582


namespace find_x_values_l9_9919

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem find_x_values :
  { x : ℕ | combination 10 x = combination 10 (3 * x - 2) } = {1, 3} :=
by
  sorry

end find_x_values_l9_9919


namespace fruit_display_l9_9060

theorem fruit_display (bananas : ℕ) (Oranges : ℕ) (Apples : ℕ) (hBananas : bananas = 5)
  (hOranges : Oranges = 2 * bananas) (hApples : Apples = 2 * Oranges) :
  bananas + Oranges + Apples = 35 :=
by sorry

end fruit_display_l9_9060


namespace sales_price_reduction_l9_9433

theorem sales_price_reduction
  (current_sales : ℝ := 20)
  (current_profit_per_shirt : ℝ := 40)
  (sales_increase_per_dollar : ℝ := 2)
  (desired_profit : ℝ := 1200) :
  ∃ x : ℝ, (40 - x) * (20 + 2 * x) = 1200 ∧ x = 20 :=
by
  use 20
  sorry

end sales_price_reduction_l9_9433


namespace least_positive_integer_divisors_l9_9998

theorem least_positive_integer_divisors (n m k : ℕ) (h₁ : (∀ d : ℕ, d ∣ n ↔ d ≤ 2023))
(h₂ : n = m * 6^k) (h₃ : (∀ d : ℕ, d ∣ 6 → ¬(d ∣ m))) : m + k = 80 :=
sorry

end least_positive_integer_divisors_l9_9998


namespace option_d_satisfies_equation_l9_9587

theorem option_d_satisfies_equation (x y z : ℤ) (h1 : x = z) (h2 : y = x + 1) : x * (x - y) + y * (y - z) + z * (z - x) = 2 :=
by
  sorry

end option_d_satisfies_equation_l9_9587


namespace ones_digit_9_pow_53_l9_9113

theorem ones_digit_9_pow_53 :
  (9 ^ 53) % 10 = 9 :=
by
  sorry

end ones_digit_9_pow_53_l9_9113


namespace average_age_of_three_l9_9971

theorem average_age_of_three (Kimiko_age : ℕ) (Omi_age : ℕ) (Arlette_age : ℕ) 
  (h1 : Omi_age = 2 * Kimiko_age) 
  (h2 : Arlette_age = (3 * Kimiko_age) / 4) 
  (h3 : Kimiko_age = 28) : 
  (Kimiko_age + Omi_age + Arlette_age) / 3 = 35 := 
  by sorry

end average_age_of_three_l9_9971


namespace pear_juice_percentage_l9_9145

/--
Miki has a dozen oranges and pears. She extracts juice as follows:
5 pears -> 10 ounces of pear juice
3 oranges -> 12 ounces of orange juice
She uses 10 pears and 10 oranges to make a blend.
Prove that the percent of the blend that is pear juice is 33.33%.
-/
theorem pear_juice_percentage :
  let pear_juice_per_pear := 10 / 5
  let orange_juice_per_orange := 12 / 3
  let total_pear_juice := 10 * pear_juice_per_pear
  let total_orange_juice := 10 * orange_juice_per_orange
  let total_juice := total_pear_juice + total_orange_juice
  total_pear_juice / total_juice = 1 / 3 :=
by
  sorry

end pear_juice_percentage_l9_9145


namespace area_of_circle_l9_9123

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end area_of_circle_l9_9123


namespace zach_cookies_total_l9_9968

theorem zach_cookies_total :
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  cookies_monday + cookies_tuesday + cookies_wednesday = 92 :=
by
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  sorry

end zach_cookies_total_l9_9968


namespace solve_equation_l9_9182

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 := by
  sorry

end solve_equation_l9_9182


namespace sum_abc_l9_9109

theorem sum_abc (a b c : ℝ) 
  (h : (a - 6)^2 + (b - 3)^2 + (c - 2)^2 = 0) : 
  a + b + c = 11 := 
by 
  sorry

end sum_abc_l9_9109


namespace peter_pizza_fraction_l9_9075

def pizza_slices : ℕ := 16
def peter_initial_slices : ℕ := 2
def shared_slices : ℕ := 2
def shared_with_paul : ℕ := shared_slices / 2
def total_slices_peter_ate := peter_initial_slices + shared_with_paul
def fraction_peter_ate : ℚ := total_slices_peter_ate / pizza_slices

theorem peter_pizza_fraction :
  fraction_peter_ate = 3 / 16 :=
by
  -- Leave space for the proof, which is not required.
  sorry

end peter_pizza_fraction_l9_9075


namespace determine_y_l9_9680

def diamond (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

theorem determine_y (y : ℝ) (h : diamond 4 y = 30) : y = 5 / 3 :=
by sorry

end determine_y_l9_9680


namespace money_returned_l9_9958

theorem money_returned (individual group taken : ℝ)
  (h1 : individual = 12000)
  (h2 : group = 16000)
  (h3 : taken = 26400) :
  (individual + group - taken) = 1600 :=
by
  -- The proof has been omitted
  sorry

end money_returned_l9_9958


namespace choice_first_question_range_of_P2_l9_9428

theorem choice_first_question (P1 P2 a b : ℚ) (hP1 : P1 = 1/2) (hP2 : P2 = 1/3) :
  (P1 * (1 - P2) * a + P1 * P2 * (a + b) - P2 * (1 - P1) * b - P1 * P2 * (a + b) > 0) ↔ a > b / 2 :=
sorry

theorem range_of_P2 (a b P1 P2 : ℚ) (ha : a = 10) (hb : b = 20) (hP1 : P1 = 2/5) :
  P1 * (1 - P2) * a + P1 * P2 * (a + b) - P2 * (1 - P1) * b - P1 * P2 * (a + b) ≥ 0 ↔ (0 ≤ P2 ∧ P2 ≤ P1 / (2 - P1)) :=
sorry

end choice_first_question_range_of_P2_l9_9428


namespace remainder_is_90_l9_9886

theorem remainder_is_90:
  let larger_number := 2982
  let smaller_number := 482
  let quotient := 6
  (larger_number - smaller_number = 2500) ∧ 
  (larger_number = quotient * smaller_number + r) →
  (r = 90) :=
by
  sorry

end remainder_is_90_l9_9886


namespace length_AE_l9_9564

theorem length_AE (A B C D E : Type) 
  (AB AC AD AE : ℝ) 
  (angle_BAC : ℝ)
  (h1 : AB = 4.5) 
  (h2 : AC = 5) 
  (h3 : angle_BAC = 30) 
  (h4 : AD = 1.5) 
  (h5 : AD / AB = AE / AC) : 
  AE = 1.6667 := 
sorry

end length_AE_l9_9564


namespace cos_half_alpha_l9_9391

open Real -- open the Real namespace for convenience

theorem cos_half_alpha {α : ℝ} (h1 : cos α = 1 / 5) (h2 : 0 < α ∧ α < π) :
  cos (α / 2) = sqrt (15) / 5 :=
by
  sorry -- Proof is omitted

end cos_half_alpha_l9_9391


namespace my_current_age_l9_9401

-- Definitions based on the conditions
def bro_age (x : ℕ) : ℕ := 2 * x - 5

-- Main theorem to prove that my current age is 13 given the conditions
theorem my_current_age 
  (x y : ℕ)
  (h1 : y - 5 = 2 * (x - 5))
  (h2 : (x + 8) + (y + 8) = 50) :
  x = 13 :=
sorry

end my_current_age_l9_9401


namespace original_people_l9_9490

-- Declare the original number of people in the room
variable (x : ℕ)

-- Conditions
-- One third of the people in the room left
def remaining_after_one_third_left (x : ℕ) : ℕ := (2 * x) / 3

-- One quarter of the remaining people started to dance
def dancers (remaining : ℕ) : ℕ := remaining / 4

-- Number of people not dancing
def non_dancers (remaining : ℕ) (dancers : ℕ) : ℕ := remaining - dancers

-- Given that there are 18 people not dancing
variable (remaining : ℕ) (dancers : ℕ)
axiom non_dancers_number : non_dancers remaining dancers = 18

-- Theorem to prove
theorem original_people (h_rem: remaining = remaining_after_one_third_left x) 
(h_dancers: dancers = remaining / 4) : x = 36 := by
  sorry

end original_people_l9_9490


namespace cost_of_jam_l9_9348

theorem cost_of_jam (N B J H : ℕ) (h : N > 1) (cost_eq : N * (6 * B + 7 * J + 4 * H) = 462) : 7 * J * N = 462 :=
by
  sorry

end cost_of_jam_l9_9348


namespace beakers_with_copper_l9_9584

theorem beakers_with_copper :
  ∀ (total_beakers no_copper_beakers beakers_with_copper drops_per_beaker total_drops_used : ℕ),
    total_beakers = 22 →
    no_copper_beakers = 7 →
    drops_per_beaker = 3 →
    total_drops_used = 45 →
    total_drops_used = drops_per_beaker * beakers_with_copper →
    total_beakers = beakers_with_copper + no_copper_beakers →
    beakers_with_copper = 15 := 
-- inserting the placeholder proof 'sorry'
sorry

end beakers_with_copper_l9_9584


namespace max_min_difference_l9_9980

noncomputable def difference_max_min_z (x y z : ℝ) : ℝ :=
  if h₁ : x + y + z = 3 ∧ x^2 + y^2 + z^2 = 18 then 6 else 0

theorem max_min_difference (x y z : ℝ) (h₁ : x + y + z = 3) (h₂ : x^2 + y^2 + z^2 = 18) :
  difference_max_min_z x y z = 6 :=
by sorry

end max_min_difference_l9_9980


namespace angle_BAC_eq_angle_DAE_l9_9257

-- Define types and points A, B, C, D, E
variables (A B C D E : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
variables (P Q R S T : Point)

-- Define angles
variable {α β γ δ θ ω : Angle}

-- Establish the conditions
axiom angle_ABC_eq_angle_ADE : α = θ
axiom angle_AEC_eq_angle_ADB : β = ω

-- State the theorem
theorem angle_BAC_eq_angle_DAE
  (h1 : α = θ) -- Given \(\angle ABC = \angle ADE\)
  (h2 : β = ω) -- Given \(\angle AEC = \angle ADB\)
  : γ = δ := sorry

end angle_BAC_eq_angle_DAE_l9_9257


namespace ruth_hours_per_week_l9_9294

theorem ruth_hours_per_week :
  let daily_hours := 8
  let days_per_week := 5
  let monday_wednesday_friday := 3
  let tuesday_thursday := 2
  let percentage_to_hours (percent : ℝ) (hours : ℕ) : ℝ := percent * hours
  let total_weekly_hours := daily_hours * days_per_week
  let monday_wednesday_friday_math_hours := percentage_to_hours 0.25 daily_hours
  let monday_wednesday_friday_science_hours := percentage_to_hours 0.15 daily_hours
  let tuesday_thursday_math_hours := percentage_to_hours 0.2 daily_hours
  let tuesday_thursday_science_hours := percentage_to_hours 0.35 daily_hours
  let tuesday_thursday_history_hours := percentage_to_hours 0.15 daily_hours
  let weekly_math_hours := monday_wednesday_friday_math_hours * monday_wednesday_friday + tuesday_thursday_math_hours * tuesday_thursday
  let weekly_science_hours := monday_wednesday_friday_science_hours * monday_wednesday_friday + tuesday_thursday_science_hours * tuesday_thursday
  let weekly_history_hours := tuesday_thursday_history_hours * tuesday_thursday
  let total_hours := weekly_math_hours + weekly_science_hours + weekly_history_hours
  total_hours = 20.8 := by
  sorry

end ruth_hours_per_week_l9_9294


namespace difference_is_three_l9_9667

-- Define the range for two-digit numbers
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define whether a number is a multiple of three
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

-- Identify the smallest and largest two-digit multiples of three
def smallest_two_digit_multiple_of_three : ℕ := 12
def largest_two_digit_multiple_of_three : ℕ := 99

-- Identify the smallest and largest two-digit non-multiples of three
def smallest_two_digit_non_multiple_of_three : ℕ := 10
def largest_two_digit_non_multiple_of_three : ℕ := 98

-- Calculate Joey's sum
def joeys_sum : ℕ := smallest_two_digit_multiple_of_three + largest_two_digit_multiple_of_three

-- Calculate Zoë's sum
def zoes_sum : ℕ := smallest_two_digit_non_multiple_of_three + largest_two_digit_non_multiple_of_three

-- Prove the difference between Joey's and Zoë's sums is 3
theorem difference_is_three : joeys_sum - zoes_sum = 3 :=
by
  -- The proof is not given, so we use sorry here
  sorry

end difference_is_three_l9_9667


namespace inequality_xy_yz_zx_l9_9676

theorem inequality_xy_yz_zx {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y + 2 * y * z + 2 * z * x) / (x^2 + y^2 + z^2) <= 1 / 4 * (Real.sqrt 33 + 1) :=
sorry

end inequality_xy_yz_zx_l9_9676


namespace probability_three_girls_chosen_l9_9927

theorem probability_three_girls_chosen :
  let total_members := 15;
  let boys := 7;
  let girls := 8;
  let total_ways := Nat.choose total_members 3;
  let girls_ways := Nat.choose girls 3;
  total_ways = Nat.choose 15 3 ∧ girls_ways = Nat.choose 8 3 →
  (girls_ways : ℚ) / (total_ways : ℚ) = 8 / 65 := 
by  
  sorry

end probability_three_girls_chosen_l9_9927


namespace sequence_n_l9_9619

theorem sequence_n (a : ℕ → ℕ) (h : ∀ n : ℕ, 0 < n → (n^2 + 1) * a n = n * (a (n^2) + 1)) :
  ∀ n : ℕ, 0 < n → a n = n := 
by
  sorry

end sequence_n_l9_9619


namespace exists_four_numbers_with_equal_sum_l9_9732

theorem exists_four_numbers_with_equal_sum (S : Finset ℕ) (hS : S.card = 16) (h_range : ∀ n ∈ S, n ≤ 100) :
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ c ≠ d ∧ a ≠ c ∧ b ≠ d ∧ a + b = c + d :=
by
  sorry

end exists_four_numbers_with_equal_sum_l9_9732


namespace pairs_of_positive_integers_l9_9022

theorem pairs_of_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
    (∃ (m : ℕ), m ≥ 2 ∧ (x = m^3 + 2*m^2 - m - 1 ∧ y = m^3 + m^2 - 2*m - 1 ∨ 
                        x = m^3 + m^2 - 2*m - 1 ∧ y = m^3 + 2*m^2 - m - 1)) ∨
    (x = 1 ∧ y = 1) ↔ 
    (∃ n : ℝ, n^3 = 7*x^2 - 13*x*y + 7*y^2) ∧ (Int.natAbs (x - y) - 1 = n) :=
by
  sorry

end pairs_of_positive_integers_l9_9022


namespace distance_from_wall_to_picture_edge_l9_9758

theorem distance_from_wall_to_picture_edge
  (wall_width : ℕ)
  (picture_width : ℕ)
  (centered : Prop)
  (h1 : wall_width = 22)
  (h2 : picture_width = 4)
  (h3 : centered) :
  ∃ x : ℕ, x = 9 :=
by
  sorry

end distance_from_wall_to_picture_edge_l9_9758


namespace pieces_eaten_first_night_l9_9245

-- Define the initial numbers of candies
def debby_candies : Nat := 32
def sister_candies : Nat := 42
def candies_left : Nat := 39

-- Calculate the initial total number of candies
def initial_total_candies : Nat := debby_candies + sister_candies

-- Define the number of candies eaten the first night
def candies_eaten : Nat := initial_total_candies - candies_left

-- The problem statement with the proof goal
theorem pieces_eaten_first_night : candies_eaten = 35 := by
  sorry

end pieces_eaten_first_night_l9_9245


namespace luke_base_points_per_round_l9_9361

theorem luke_base_points_per_round
    (total_score : ℕ)
    (rounds : ℕ)
    (bonus : ℕ)
    (penalty : ℕ)
    (adjusted_total : ℕ) :
    total_score = 370 → rounds = 5 → bonus = 50 → penalty = 30 → adjusted_total = total_score + bonus - penalty → (adjusted_total / rounds) = 78 :=
by
  intros
  sorry

end luke_base_points_per_round_l9_9361


namespace symmetric_point_l9_9406

theorem symmetric_point (x y : ℝ) : 
  (x - 2 * y + 1 = 0) ∧ (y / x * 1 / 2 = -1) → (x = -2/5 ∧ y = 4/5) :=
by 
  sorry

end symmetric_point_l9_9406


namespace simplify_fraction_sum_l9_9874

theorem simplify_fraction_sum :
  (3 / 462) + (17 / 42) + (1 / 11) = 116 / 231 := 
by
  sorry

end simplify_fraction_sum_l9_9874


namespace find_value_of_N_l9_9493

theorem find_value_of_N :
  (2 * ((3.6 * 0.48 * 2.5) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002) :=
by {
  sorry
}

end find_value_of_N_l9_9493


namespace systematic_sample_contains_18_l9_9668

theorem systematic_sample_contains_18 (employees : Finset ℕ) (sample : Finset ℕ)
    (h1 : employees = Finset.range 52)
    (h2 : sample.card = 4)
    (h3 : ∀ n ∈ sample, n ∈ employees)
    (h4 : 5 ∈ sample)
    (h5 : 31 ∈ sample)
    (h6 : 44 ∈ sample) :
  18 ∈ sample :=
sorry

end systematic_sample_contains_18_l9_9668


namespace perfect_shells_l9_9777

theorem perfect_shells (P_spiral B_spiral P_total : ℕ) 
  (h1 : 52 = 2 * B_spiral)
  (h2 : B_spiral = P_spiral + 21)
  (h3 : P_total = P_spiral + 12) :
  P_total = 17 :=
by
  sorry

end perfect_shells_l9_9777


namespace trains_meet_in_approx_17_45_seconds_l9_9499

noncomputable def train_meet_time
  (length1 length2 distance_between : ℕ)
  (speed1_kmph speed2_kmph : ℕ)
  : ℕ :=
  let speed1_mps := (speed1_kmph * 1000) / 3600
  let speed2_mps := (speed2_kmph * 1000) / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := distance_between + length1 + length2
  total_distance / relative_speed

theorem trains_meet_in_approx_17_45_seconds :
  train_meet_time 100 200 660 90 108 = 17 := by
  sorry

end trains_meet_in_approx_17_45_seconds_l9_9499


namespace slope_of_line_passing_through_MN_l9_9731

theorem slope_of_line_passing_through_MN :
  let M := (-2, 1)
  let N := (1, 4)
  ∃ m : ℝ, m = (N.2 - M.2) / (N.1 - M.1) ∧ m = 1 :=
by
  sorry

end slope_of_line_passing_through_MN_l9_9731


namespace find_a_perpendicular_lines_l9_9935

theorem find_a_perpendicular_lines (a : ℝ) :
    (∀ x y : ℝ, a * x - y + 2 * a = 0 → (2 * a - 1) * x + a * y + a = 0) →
    (a = 0 ∨ a = 1) :=
by
  intro h
  sorry

end find_a_perpendicular_lines_l9_9935


namespace find_x_l9_9165

theorem find_x :
  ∃ X : ℝ, 0.25 * X + 0.20 * 40 = 23 ∧ X = 60 :=
by
  sorry

end find_x_l9_9165


namespace negation_of_proposition_is_false_l9_9132

theorem negation_of_proposition_is_false :
  (¬ ∀ (x : ℝ), x < 0 → x^2 > 0) = true :=
by
  sorry

end negation_of_proposition_is_false_l9_9132


namespace find_n_l9_9726

theorem find_n (n : ℝ) (h1 : ∀ x y : ℝ, (n + 1) * x^(n^2 - 5) = y) 
               (h2 : ∀ x > 0, (n + 1) * x^(n^2 - 5) > 0) :
               n = 2 :=
by
  sorry

end find_n_l9_9726


namespace problem_l9_9631

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x|

theorem problem :
  (∀ x, f x ≤ 1) ∧
  (∃ x, f x = 1) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → 
    ∃ x, (x = (a^2 / (b + 1) + b^2 / (a + 1)) ∧ x = 1 / 3)) :=
by {
  sorry
}

end problem_l9_9631


namespace standard_equation_hyperbola_l9_9021

-- Define necessary conditions
def condition_hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

def condition_asymptote (a b : ℝ) :=
  b / a = Real.sqrt 3

def condition_focus_hyperbola_parabola (a b : ℝ) :=
  (a^2 + b^2).sqrt = 4

-- Define the proof problem
theorem standard_equation_hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h_asymptote : condition_asymptote a b)
  (h_focus : condition_focus_hyperbola_parabola a b) :
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1) :=
sorry

end standard_equation_hyperbola_l9_9021


namespace reciprocal_of_repeating_decimal_l9_9588

theorem reciprocal_of_repeating_decimal :
  (1 / (0.33333333 : ℚ)) = 3 := by
  sorry

end reciprocal_of_repeating_decimal_l9_9588


namespace probability_at_least_one_blown_l9_9436

theorem probability_at_least_one_blown (P_A P_B P_AB : ℝ)  
  (hP_A : P_A = 0.085) 
  (hP_B : P_B = 0.074) 
  (hP_AB : P_AB = 0.063) : 
  P_A + P_B - P_AB = 0.096 :=
by
  sorry

end probability_at_least_one_blown_l9_9436


namespace david_money_l9_9606

theorem david_money (S : ℝ) (h_initial : 1500 - S = S - 500) : 1500 - S = 500 :=
by
  sorry

end david_money_l9_9606


namespace find_function_g_l9_9517

noncomputable def g (x : ℝ) : ℝ := (5^x - 3^x) / 8

theorem find_function_g (x y : ℝ) (h1 : g 2 = 2) (h2 : ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y) :
  g x = (5^x - 3^x) / 8 :=
by
  sorry

end find_function_g_l9_9517


namespace five_digit_number_divisible_by_B_is_multiple_of_1000_l9_9795

-- Definitions
def is_five_digit_number (A : ℕ) : Prop := 10000 ≤ A ∧ A < 100000
def B (A : ℕ) := (A / 1000 * 100) + (A % 100)
def is_four_digit_number (B : ℕ) : Prop := 1000 ≤ B ∧ B < 10000

-- Main theorem
theorem five_digit_number_divisible_by_B_is_multiple_of_1000
  (A : ℕ) (hA : is_five_digit_number A)
  (hAB : ∃ k : ℕ, B A = k) :
  A % 1000 = 0 := 
sorry

end five_digit_number_divisible_by_B_is_multiple_of_1000_l9_9795


namespace sum_of_second_and_third_smallest_is_804_l9_9868

noncomputable def sum_of_second_and_third_smallest : Nat :=
  let digits := [1, 6, 8]
  let second_smallest := 186
  let third_smallest := 618
  second_smallest + third_smallest

theorem sum_of_second_and_third_smallest_is_804 :
  sum_of_second_and_third_smallest = 804 :=
by
  sorry

end sum_of_second_and_third_smallest_is_804_l9_9868


namespace most_stable_performance_l9_9393

-- Define the variances for each player
def variance_A : ℝ := 0.66
def variance_B : ℝ := 0.52
def variance_C : ℝ := 0.58
def variance_D : ℝ := 0.62

-- State the theorem
theorem most_stable_performance : variance_B < variance_C ∧ variance_C < variance_D ∧ variance_D < variance_A :=
by
  -- Since we are tasked to write only the statement, the proof part is skipped.
  sorry

end most_stable_performance_l9_9393


namespace carl_marbles_l9_9708

-- Define initial conditions
def initial_marbles : ℕ := 12
def lost_marbles : ℕ := initial_marbles / 2
def remaining_marbles : ℕ := initial_marbles - lost_marbles
def additional_marbles : ℕ := 10
def new_marbles_from_mother : ℕ := 25

-- Define the final number of marbles Carl will put back in the jar
def total_marbles_put_back : ℕ := remaining_marbles + additional_marbles + new_marbles_from_mother

-- Statement to be proven
theorem carl_marbles : total_marbles_put_back = 41 :=
by
  sorry

end carl_marbles_l9_9708


namespace product_eval_l9_9769

theorem product_eval (a : ℤ) (h : a = 3) : (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  sorry

end product_eval_l9_9769


namespace digit_x_base_7_l9_9231

theorem digit_x_base_7 (x : ℕ) : 
    (4 * 7^3 + 5 * 7^2 + x * 7 + 2) % 9 = 0 → x = 4 := 
by {
    sorry
}

end digit_x_base_7_l9_9231


namespace sequence_sixth_term_is_364_l9_9324

theorem sequence_sixth_term_is_364 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 7) (h3 : a 3 = 20)
  (h4 : ∀ n, a (n + 1) = 1 / 3 * (a n + a (n + 2))) :
  a 6 = 364 :=
by
  -- Proof skipped
  sorry

end sequence_sixth_term_is_364_l9_9324


namespace initial_birds_179_l9_9496

theorem initial_birds_179 (B : ℕ) (h1 : B + 38 = 217) : B = 179 :=
sorry

end initial_birds_179_l9_9496


namespace rachel_left_24_brownies_at_home_l9_9488

-- Defining the conditions
def total_brownies : ℕ := 40
def brownies_brought_to_school : ℕ := 16

-- Formulation of the theorem
theorem rachel_left_24_brownies_at_home : (total_brownies - brownies_brought_to_school = 24) :=
by
  sorry

end rachel_left_24_brownies_at_home_l9_9488


namespace solve_quadratic_l9_9142

theorem solve_quadratic {x : ℝ} : x^2 = 2 * x ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_quadratic_l9_9142


namespace sum_of_two_digit_factors_is_162_l9_9491

-- Define the number
def num := 6545

-- Define the condition: num can be written as a product of two two-digit numbers
def are_two_digit_numbers (a b : ℕ) : Prop :=
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = num

-- The theorem to prove
theorem sum_of_two_digit_factors_is_162 : ∃ a b : ℕ, are_two_digit_numbers a b ∧ a + b = 162 :=
sorry

end sum_of_two_digit_factors_is_162_l9_9491


namespace sum_of_cubes_l9_9274

theorem sum_of_cubes (x y : ℝ) (h_sum : x + y = 3) (h_prod : x * y = 2) : x^3 + y^3 = 9 :=
by
  sorry

end sum_of_cubes_l9_9274


namespace solve_equation_l9_9921

theorem solve_equation (x : ℝ) : 
  16 * (x - 1) ^ 2 - 9 = 0 ↔ (x = 7 / 4 ∨ x = 1 / 4) := by
  sorry

end solve_equation_l9_9921


namespace dinosaur_dolls_distribution_l9_9878

-- Defining the conditions
def num_dolls : ℕ := 5
def num_friends : ℕ := 2

-- Lean theorem statement
theorem dinosaur_dolls_distribution :
  (num_dolls * (num_dolls - 1) = 20) :=
by
  -- Sorry placeholder for the proof
  sorry

end dinosaur_dolls_distribution_l9_9878


namespace value_of_n_l9_9399

theorem value_of_n 
  {a b n : ℕ} (ha : a > 0) (hb : b > 0) 
  (h : (1 + b)^n = 243) : 
  n = 5 := by 
  sorry

end value_of_n_l9_9399


namespace ninety_eight_squared_l9_9312

theorem ninety_eight_squared : 98^2 = 9604 := by
  sorry

end ninety_eight_squared_l9_9312


namespace sin_2theta_value_l9_9326

theorem sin_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ)^(2 * n) = 3) : Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
by
  sorry

end sin_2theta_value_l9_9326


namespace no_egg_arrangements_possible_l9_9751

noncomputable def num_egg_arrangements 
  (total_eggs : ℕ) 
  (type_A_eggs : ℕ) 
  (type_B_eggs : ℕ)
  (type_C_eggs : ℕ)
  (groups : ℕ)
  (ratio_A : ℕ) 
  (ratio_B : ℕ) 
  (ratio_C : ℕ) : ℕ :=
if (total_eggs = type_A_eggs + type_B_eggs + type_C_eggs) ∧ 
   (type_A_eggs / groups = ratio_A) ∧ 
   (type_B_eggs / groups = ratio_B) ∧ 
   (type_C_eggs / groups = ratio_C) then 0 else 0

theorem no_egg_arrangements_possible :
  num_egg_arrangements 35 15 12 8 5 2 3 1 = 0 := 
by sorry

end no_egg_arrangements_possible_l9_9751


namespace distance_behind_C_l9_9061

-- Conditions based on the problem
def distance_race : ℕ := 1000
def distance_B_when_A_finishes : ℕ := 50
def distance_C_when_B_finishes : ℕ := 100

-- Derived condition based on given problem details
def distance_run_by_B_when_A_finishes : ℕ := distance_race - distance_B_when_A_finishes
def distance_run_by_C_when_B_finishes : ℕ := distance_race - distance_C_when_B_finishes

-- Ratios
def ratio_B_to_A : ℚ := distance_run_by_B_when_A_finishes / distance_race
def ratio_C_to_B : ℚ := distance_run_by_C_when_B_finishes / distance_race

-- Combined ratio
def ratio_C_to_A : ℚ := ratio_C_to_B * ratio_B_to_A

-- Distance run by C when A finishes
def distance_run_by_C_when_A_finishes : ℚ := distance_race * ratio_C_to_A

-- Distance C is behind the finish line when A finishes
def distance_C_behind_when_A_finishes : ℚ := distance_race - distance_run_by_C_when_A_finishes

theorem distance_behind_C (d_race : ℕ) (d_BA : ℕ) (d_CB : ℕ)
  (hA : d_race = 1000) (hB : d_BA = 50) (hC : d_CB = 100) :
  distance_C_behind_when_A_finishes = 145 :=
  by sorry

end distance_behind_C_l9_9061


namespace range_of_a_l9_9627

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x ^ 2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3 : ℝ) 1 :=
by
  sorry

end range_of_a_l9_9627


namespace park_area_correct_l9_9669

noncomputable def rect_park_area (speed_km_hr : ℕ) (time_min : ℕ) (ratio_l_b : ℕ) : ℕ := by
  let speed_m_min := speed_km_hr * 1000 / 60
  let perimeter := speed_m_min * time_min
  let B := perimeter * 3 / 8
  let L := B / 3
  let area := L * B
  exact area

theorem park_area_correct : rect_park_area 12 8 3 = 120000 := by
  sorry

end park_area_correct_l9_9669


namespace extreme_values_f_a4_no_zeros_f_on_1e_l9_9169

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (a + 2) * Real.log x - 2 / x + 2

theorem extreme_values_f_a4 :
  f 4 (1 / 2) = 6 * Real.log 2 ∧ f 4 1 = 4 := sorry

theorem no_zeros_f_on_1e (a : ℝ) :
  (a ≤ 0 ∨ a ≥ 2 / (Real.exp 1 * (Real.exp 1 - 1))) →
  ∀ x, 1 < x → x < Real.exp 1 → f a x ≠ 0 := sorry

end extreme_values_f_a4_no_zeros_f_on_1e_l9_9169


namespace quadratic_condition_l9_9570

theorem quadratic_condition (m : ℝ) (h1 : m^2 - 2 = 2) (h2 : m + 2 ≠ 0) : m = 2 :=
by
  sorry

end quadratic_condition_l9_9570


namespace BDD1H_is_Spatial_in_Cube_l9_9541

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
(A B C D A1 B1 C1 D1 : Point3D)
(midpoint_B1C1 : Point3D)
(middle_B1C1 : midpoint_B1C1 = ⟨(B1.x + C1.x) / 2, (B1.y + C1.y) / 2, (B1.z + C1.z) / 2⟩)

def is_not_planar (a b c d : Point3D) : Prop :=
¬ ∃ α β γ δ : ℝ, α * a.x + β * a.y + γ * a.z + δ = 0 ∧ 
                α * b.x + β * b.y + γ * b.z + δ = 0 ∧ 
                α * c.x + β * c.y + γ * c.z + δ = 0 ∧ 
                α * d.x + β * d.y + γ * d.z + δ = 0

def BDD1H_is_spatial (cube : Cube) : Prop :=
is_not_planar cube.B cube.D cube.D1 cube.midpoint_B1C1

theorem BDD1H_is_Spatial_in_Cube (cube : Cube) : BDD1H_is_spatial cube :=
sorry

end BDD1H_is_Spatial_in_Cube_l9_9541


namespace sum_mean_median_mode_l9_9270

def numbers : List ℕ := [3, 5, 3, 0, 2, 5, 0, 2]

def mode (l : List ℕ) : ℝ := 4

def median (l : List ℕ) : ℝ := 2.5

def mean (l : List ℕ) : ℝ := 2.5

theorem sum_mean_median_mode : mean numbers + median numbers + mode numbers = 9 := by
  sorry

end sum_mean_median_mode_l9_9270


namespace triangle_area_l9_9714

theorem triangle_area : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y = 12
  let x_intercept := (4 : ℝ)
  let y_intercept := (6 : ℝ)
  ∃ (x y : ℝ), line_eq x y ∧ x = x_intercept ∧ y = y_intercept ∧
  ∃ (area : ℝ), area = 1 / 2 * x * y ∧ area = 12 :=
by
  sorry

end triangle_area_l9_9714


namespace length_of_AB_l9_9104

noncomputable def AB_CD_sum_240 (AB CD : ℝ) (h : ℝ) : Prop :=
  AB + CD = 240

noncomputable def ratio_of_areas (AB CD : ℝ) : Prop :=
  AB / CD = 5 / 3

theorem length_of_AB (AB CD : ℝ) (h : ℝ) (h_ratio : ratio_of_areas AB CD) (h_sum : AB_CD_sum_240 AB CD h) : AB = 150 :=
by
  unfold ratio_of_areas at h_ratio
  unfold AB_CD_sum_240 at h_sum
  sorry

end length_of_AB_l9_9104


namespace minimum_familiar_pairs_l9_9171

theorem minimum_familiar_pairs (n : ℕ) (students : Finset (Fin n)) 
  (familiar : Finset (Fin n × Fin n))
  (h_n : n = 175)
  (h_condition : ∀ (s : Finset (Fin n)), s.card = 6 → 
    ∃ (s1 s2 : Finset (Fin n)), s1 ∪ s2 = s ∧ s1.card = 3 ∧ s2.card = 3 ∧ 
    ∀ x ∈ s1, ∀ y ∈ s1, (x ≠ y → (x, y) ∈ familiar) ∧
    ∀ x ∈ s2, ∀ y ∈ s2, (x ≠ y → (x, y) ∈ familiar)) :
  ∃ m : ℕ, m = 15050 ∧ ∀ p : ℕ, (∃ g : Finset (Fin n × Fin n), g.card = p) → p ≥ m := 
sorry

end minimum_familiar_pairs_l9_9171


namespace smallest_k_l9_9460

theorem smallest_k (a b : ℚ) (h_a_period : ∀ n, a ≠ (10^30 - 1) * n)
  (h_b_period : ∀ n, b ≠ (10^30 - 1) * n)
  (h_diff_period : ∀ n, a - b ≠ (10^15 - 1) * n) :
  ∃ k : ℕ, k = 6 ∧ (a + (k:ℚ) * b) ≠ (10^15 - 1) :=
sorry

end smallest_k_l9_9460


namespace expression_evaluation_l9_9338

theorem expression_evaluation : -20 + 8 * (5 ^ 2 - 3) = 156 := by
  sorry

end expression_evaluation_l9_9338


namespace trivia_team_absentees_l9_9379

theorem trivia_team_absentees (total_members : ℕ) (total_points : ℕ) (points_per_member : ℕ) 
  (h1 : total_members = 5) 
  (h2 : total_points = 6) 
  (h3 : points_per_member = 2) : 
  total_members - (total_points / points_per_member) = 2 := 
by 
  sorry

end trivia_team_absentees_l9_9379


namespace outer_perimeter_fence_l9_9070

-- Definitions based on given conditions
def total_posts : Nat := 16
def post_width_feet : Real := 0.5 -- 6 inches converted to feet
def gap_length_feet : Real := 6 -- gap between posts in feet
def num_sides : Nat := 4 -- square field has 4 sides

-- Hypotheses that capture conditions and intermediate calculations
def num_corners : Nat := 4
def non_corner_posts : Nat := total_posts - num_corners
def non_corner_posts_per_side : Nat := non_corner_posts / num_sides
def posts_per_side : Nat := non_corner_posts_per_side + 2
def gaps_per_side : Nat := posts_per_side - 1
def length_gaps_per_side : Real := gaps_per_side * gap_length_feet
def total_post_width_per_side : Real := posts_per_side * post_width_feet
def length_one_side : Real := length_gaps_per_side + total_post_width_per_side
def perimeter : Real := num_sides * length_one_side

-- The theorem to prove
theorem outer_perimeter_fence : perimeter = 106 := by
  sorry

end outer_perimeter_fence_l9_9070


namespace area_between_circles_l9_9211

noncomputable def k_value (θ : ℝ) : ℝ := Real.tan θ

theorem area_between_circles {θ k : ℝ} (h₁ : k = Real.tan θ) (h₂ : θ = 4/3) (h_area : (3 * θ / 2) = 2) :
  k = Real.tan (4/3) :=
sorry

end area_between_circles_l9_9211


namespace percentage_fescue_in_Y_l9_9358

-- Define the seed mixtures and their compositions
structure SeedMixture :=
  (ryegrass : ℝ)  -- percentage of ryegrass

-- Seed mixture X
def X : SeedMixture := { ryegrass := 0.40 }

-- Seed mixture Y
def Y : SeedMixture := { ryegrass := 0.25 }

-- Mixture of X and Y contains 32 percent ryegrass
def mixture_percentage := 0.32

-- 46.67 percent of the weight of this mixture is X
def weight_X := 0.4667

-- Question: What percent of seed mixture Y is fescue
theorem percentage_fescue_in_Y : (1 - Y.ryegrass) = 0.75 := by
  sorry

end percentage_fescue_in_Y_l9_9358


namespace fewest_seats_to_be_occupied_l9_9299

theorem fewest_seats_to_be_occupied (n : ℕ) (h : n = 120) : ∃ m, m = 40 ∧
  ∀ a b, a + b = n → a ≥ m → ∀ x, (x > 0 ∧ x ≤ n) → (x > 1 → a = m → a + (b / 2) ≥ n / 3) :=
sorry

end fewest_seats_to_be_occupied_l9_9299


namespace movie_ticket_cost_l9_9076

-- Definitions from conditions
def total_spending : ℝ := 36
def combo_meal_cost : ℝ := 11
def candy_cost : ℝ := 2.5
def total_food_cost : ℝ := combo_meal_cost + 2 * candy_cost
def total_ticket_cost (x : ℝ) : ℝ := 2 * x

-- The theorem stating the proof problem
theorem movie_ticket_cost :
  ∃ (x : ℝ), total_ticket_cost x + total_food_cost = total_spending ∧ x = 10 :=
by
  sorry

end movie_ticket_cost_l9_9076


namespace yellow_less_than_three_times_red_l9_9890

def num_red : ℕ := 40
def less_than_three_times (Y : ℕ) : Prop := Y < 120
def blue_half_yellow (Y B : ℕ) : Prop := B = Y / 2
def remaining_after_carlos (B : ℕ) : Prop := 40 + B = 90
def difference_three_times_red (Y : ℕ) : ℕ := 3 * num_red - Y

theorem yellow_less_than_three_times_red (Y B : ℕ) 
  (h1 : less_than_three_times Y) 
  (h2 : blue_half_yellow Y B) 
  (h3 : remaining_after_carlos B) : 
  difference_three_times_red Y = 20 := by
  sorry

end yellow_less_than_three_times_red_l9_9890


namespace total_limes_picked_l9_9412

theorem total_limes_picked (Alyssa_limes Mike_limes : ℕ) 
        (hAlyssa : Alyssa_limes = 25) (hMike : Mike_limes = 32) : 
       Alyssa_limes + Mike_limes = 57 :=
by {
  sorry
}

end total_limes_picked_l9_9412


namespace sum_of_exterior_angles_of_triangle_l9_9375

theorem sum_of_exterior_angles_of_triangle
  {α β γ α' β' γ' : ℝ} 
  (h1 : α + β + γ = 180)
  (h2 : α + α' = 180)
  (h3 : β + β' = 180)
  (h4 : γ + γ' = 180) :
  α' + β' + γ' = 360 := 
by 
sorry

end sum_of_exterior_angles_of_triangle_l9_9375


namespace domain_of_f_l9_9279

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / Real.sqrt (x^2 - 4)

theorem domain_of_f :
  {x : ℝ | x^2 - 4 >= 0 ∧ x^2 - 4 ≠ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by
  sorry

end domain_of_f_l9_9279


namespace leak_drain_time_l9_9333

theorem leak_drain_time (P L : ℝ) (h1 : P = 0.5) (h2 : (P - L) = (6 / 13)) :
    (1 / L) = 26 := by
  sorry

end leak_drain_time_l9_9333


namespace ex1_ex2_l9_9509

-- Definition of the "multiplication-subtraction" operation.
def mult_sub (a b : ℚ) : ℚ :=
  if a = 0 then abs b else if b = 0 then abs a else if abs a = abs b then 0 else
  if (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) then abs a - abs b else -(abs a - abs b)

theorem ex1 : mult_sub (mult_sub (3) (-2)) (mult_sub (-9) 0) = -8 :=
  sorry

theorem ex2 : ∃ (a b c : ℚ), (mult_sub (mult_sub a b) c) ≠ (mult_sub a (mult_sub b c)) :=
  ⟨3, -2, 4, by simp [mult_sub]; sorry⟩

end ex1_ex2_l9_9509


namespace bees_flew_in_l9_9827

theorem bees_flew_in (initial_bees additional_bees total_bees : ℕ) 
  (h1 : initial_bees = 16) (h2 : total_bees = 25) 
  (h3 : initial_bees + additional_bees = total_bees) : additional_bees = 9 :=
by sorry

end bees_flew_in_l9_9827


namespace minimum_total_number_of_balls_l9_9810

theorem minimum_total_number_of_balls (x y z t : ℕ) 
  (h1 : x ≥ 4)
  (h2 : x ≥ 3 ∧ y ≥ 1)
  (h3 : x ≥ 2 ∧ y ≥ 1 ∧ z ≥ 1)
  (h4 : x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ t ≥ 1) :
  x + y + z + t = 21 :=
  sorry

end minimum_total_number_of_balls_l9_9810


namespace no_solutions_then_a_eq_zero_l9_9577

theorem no_solutions_then_a_eq_zero (a b : ℝ) :
  (∀ x y : ℝ, ¬ (y^2 = x^2 + a * x + b ∧ x^2 = y^2 + a * y + b)) → a = 0 :=
by
  sorry

end no_solutions_then_a_eq_zero_l9_9577


namespace largest_perfect_square_factor_of_1764_l9_9009

theorem largest_perfect_square_factor_of_1764 : ∃ m, m * m = 1764 ∧ ∀ n, n * n ∣ 1764 → n * n ≤ 1764 :=
by
  sorry

end largest_perfect_square_factor_of_1764_l9_9009


namespace plane_equation_exists_l9_9363

noncomputable def equation_of_plane (A B C D : ℤ) (hA : A > 0) (hGCD : Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) : Prop :=
∃ (x y z : ℤ),
  x = 1 ∧ y = -2 ∧ z = 2 ∧ D = -18 ∧
  (2 * x + (-3) * y + 5 * z + D = 0) ∧  -- Point (2, -3, 5) satisfies equation
  (4 * x + (-3) * y + 6 * z + D = 0) ∧  -- Point (4, -3, 6) satisfies equation
  (6 * x + (-4) * y + 8 * z + D = 0)    -- Point (6, -4, 8) satisfies equation

theorem plane_equation_exists : equation_of_plane 1 (-2) 2 (-18) (by decide) (by decide) :=
by
  -- Proof is omitted
  sorry

end plane_equation_exists_l9_9363


namespace asimov_books_l9_9223

theorem asimov_books (h p : Nat) (condition1 : h + p = 12) (condition2 : 30 * h + 20 * p = 300) : h = 6 := by
  sorry

end asimov_books_l9_9223


namespace calc_expr_l9_9694

noncomputable def expr_val : ℝ :=
  Real.sqrt 4 - |(-(1 / 4 : ℝ))| + (Real.pi - 2)^0 + 2^(-2 : ℝ)

theorem calc_expr : expr_val = 3 := by
  sorry

end calc_expr_l9_9694


namespace solve_inequality_l9_9544

theorem solve_inequality (x : ℝ) : ((x + 3) ^ 2 < 1) ↔ (-4 < x ∧ x < -2) := by
  sorry

end solve_inequality_l9_9544


namespace range_of_hx_l9_9575

open Real

theorem range_of_hx (h : ℝ → ℝ) (a b : ℝ) (H_def : ∀ x : ℝ, h x = 3 / (1 + 3 * x^4)) 
  (H_range : ∀ y : ℝ, (y > 0 ∧ y ≤ 3) ↔ ∃ x : ℝ, h x = y) : 
  a + b = 3 := 
sorry

end range_of_hx_l9_9575


namespace min_plus_max_value_of_x_l9_9912

theorem min_plus_max_value_of_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  let m := (10 - Real.sqrt 304) / 6
  let M := (10 + Real.sqrt 304) / 6
  m + M = 10 / 3 := by 
  sorry

end min_plus_max_value_of_x_l9_9912


namespace circumscribed_circle_area_l9_9555

theorem circumscribed_circle_area (x y c : ℝ)
  (h1 : x + y + c = 24)
  (h2 : x * y = 48)
  (h3 : x^2 + y^2 = c^2) :
  ∃ R : ℝ, (x + y + 2 * R = 24) ∧ (π * R^2 = 25 * π) := 
sorry

end circumscribed_circle_area_l9_9555


namespace length_EF_l9_9548

theorem length_EF
  (AB CD GH EF : ℝ)
  (h1 : AB = 180)
  (h2 : CD = 120)
  (h3 : AB = 2 * GH)
  (h4 : CD = 2 * EF) :
  EF = 45 :=
by
  sorry

end length_EF_l9_9548


namespace seventh_term_of_geometric_sequence_l9_9535

theorem seventh_term_of_geometric_sequence :
  ∀ (a r : ℝ), (a * r ^ 3 = 16) → (a * r ^ 8 = 2) → (a * r ^ 6 = 2) :=
by
  intros a r h1 h2
  sorry

end seventh_term_of_geometric_sequence_l9_9535


namespace duration_of_each_turn_l9_9849

-- Definitions based on conditions
def Wa := 1 / 4
def Wb := 1 / 12

-- Define the duration of each turn as T
def T : ℝ := 1 -- This is the correct answer we proved

-- Given conditions
def total_work_done := 6 * Wa + 6 * Wb

-- Lean statement to prove 
theorem duration_of_each_turn : T = 1 := by
  -- According to conditions, the total work done by a and b should equal the whole work
  have h1 : 3 * Wa + 3 * Wb = 1 := by sorry
  -- Let's conclude that T = 1
  sorry

end duration_of_each_turn_l9_9849


namespace joe_needs_more_cars_l9_9776

-- Definitions based on conditions
def current_cars : ℕ := 50
def total_cars : ℕ := 62

-- Theorem based on the problem question and correct answer
theorem joe_needs_more_cars : (total_cars - current_cars) = 12 :=
by
  sorry

end joe_needs_more_cars_l9_9776


namespace triangle_perimeter_ratio_l9_9337

theorem triangle_perimeter_ratio : 
  let side := 10
  let hypotenuse := Real.sqrt (side^2 + (side / 2) ^ 2)
  let triangle_perimeter := side + (side / 2) + hypotenuse
  let square_perimeter := 4 * side
  (triangle_perimeter / square_perimeter) = (15 + Real.sqrt 125) / 40 := 
by
  sorry

end triangle_perimeter_ratio_l9_9337


namespace measure_of_angle_C_l9_9866

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 360) (h2 : C = 5 * D) : C = 300 := 
by sorry

end measure_of_angle_C_l9_9866


namespace trevor_spending_proof_l9_9081

def trevor_spends (T R Q : ℕ) : Prop :=
  T = R + 20 ∧ R = 2 * Q ∧ 4 * T + 4 * R + 2 * Q = 680

theorem trevor_spending_proof (T R Q : ℕ) (h : trevor_spends T R Q) : T = 80 :=
by sorry

end trevor_spending_proof_l9_9081


namespace find_principal_l9_9864

theorem find_principal (r t1 t2 ΔI : ℝ) (h_r : r = 0.15) (h_t1 : t1 = 3.5) (h_t2 : t2 = 5) (h_ΔI : ΔI = 144) :
  ∃ P : ℝ, P = 640 :=
by
  sorry

end find_principal_l9_9864


namespace total_circles_l9_9181

theorem total_circles (n : ℕ) (h1 : ∀ k : ℕ, k = n + 14 → n^2 = (k * (k + 1) / 2)) : 
  n = 35 → n^2 = 1225 :=
by
  sorry

end total_circles_l9_9181


namespace magazine_cost_l9_9804

variable (b m : ℝ)

theorem magazine_cost (h1 : 2 * b + 2 * m = 26) (h2 : b + 3 * m = 27) : m = 7 :=
by
  sorry

end magazine_cost_l9_9804


namespace min_weighings_to_find_heaviest_l9_9016

-- Given conditions
variable (n : ℕ) (hn : n > 2)
variables (coins : Fin n) -- Representing coins with distinct masses
variables (scales : Fin n) -- Representing n scales where one is faulty

-- Theorem statement: Minimum number of weighings to find the heaviest coin
theorem min_weighings_to_find_heaviest : ∃ m, m = 2 * n - 1 := 
by
  existsi (2 * n - 1)
  rfl

end min_weighings_to_find_heaviest_l9_9016


namespace circumcircle_radius_is_one_l9_9215

-- Define the basic setup for the triangle with given sides and angles
variables {A B C : Real} -- Angles of the triangle
variables {a b c : Real} -- Sides of the triangle opposite these angles
variable (triangle_ABC : a = Real.sqrt 3 ∧ (c - 2 * b + 2 * Real.sqrt 3 * Real.cos C = 0)) -- Conditions on the sides

-- Define the circumcircle radius
noncomputable def circumcircle_radius (a b c : Real) (A B C : Real) := a / (2 * (Real.sin A))

-- Statement of the problem to be proven
theorem circumcircle_radius_is_one (h : a = Real.sqrt 3)
  (h1 : c - 2 * b + 2 * Real.sqrt 3 * Real.cos C = 0) :
  circumcircle_radius a b c A B C = 1 :=
sorry

end circumcircle_radius_is_one_l9_9215


namespace kanul_initial_amount_l9_9553

noncomputable def initial_amount : ℝ :=
  (5000 : ℝ) + 200 + 1200 + (11058.82 : ℝ) * 0.15 + 3000

theorem kanul_initial_amount (X : ℝ) 
  (raw_materials : ℝ := 5000) 
  (machinery : ℝ := 200) 
  (employee_wages : ℝ := 1200) 
  (maintenance_cost : ℝ := 0.15 * X)
  (remaining_balance : ℝ := 3000) 
  (expenses : ℝ := raw_materials + machinery + employee_wages + maintenance_cost) 
  (total_expenses : ℝ := expenses + remaining_balance) :
  X = total_expenses :=
by sorry

end kanul_initial_amount_l9_9553


namespace paper_area_l9_9796

variable (L W : ℕ)

theorem paper_area (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) : L * W = 140 := by
  sorry

end paper_area_l9_9796


namespace ant_food_cost_l9_9634

-- Definitions for the conditions
def number_of_ants : ℕ := 400
def food_per_ant : ℕ := 2
def job_charge : ℕ := 5
def leaf_charge : ℕ := 1 / 100 -- 1 penny is 1 cent which is 0.01 dollars
def leaves_raked : ℕ := 6000
def jobs_completed : ℕ := 4

-- Compute the total money earned from jobs
def money_from_jobs : ℕ := jobs_completed * job_charge

-- Compute the total money earned from raking leaves
def money_from_leaves : ℕ := leaves_raked * leaf_charge

-- Compute the total money earned
def total_money_earned : ℕ := money_from_jobs + money_from_leaves

-- Compute the total ounces of food needed
def total_food_needed : ℕ := number_of_ants * food_per_ant

-- Calculate the cost per ounce of food
def cost_per_ounce : ℕ := total_money_earned / total_food_needed

theorem ant_food_cost :
  cost_per_ounce = 1 / 10 := sorry

end ant_food_cost_l9_9634


namespace minimum_soldiers_to_add_l9_9284

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l9_9284


namespace kyle_vs_parker_l9_9276

-- Define the distances thrown by Parker, Grant, and Kyle.
def parker_distance : ℕ := 16
def grant_distance : ℕ := (125 * parker_distance) / 100
def kyle_distance : ℕ := 2 * grant_distance

-- Prove that Kyle threw the ball 24 yards farther than Parker.
theorem kyle_vs_parker : kyle_distance - parker_distance = 24 := 
by
  -- Sorry for proof
  sorry

end kyle_vs_parker_l9_9276


namespace fraction_sum_is_ten_l9_9336

theorem fraction_sum_is_ten :
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (5 / 10) + (6 / 10) + (7 / 10) + (8 / 10) + (9 / 10) + (55 / 10) = 10 :=
by
  sorry

end fraction_sum_is_ten_l9_9336


namespace cos_squared_formula_15deg_l9_9062

theorem cos_squared_formula_15deg :
  (Real.cos (15 * Real.pi / 180))^2 - (1 / 2) = (Real.sqrt 3) / 4 :=
by
  sorry

end cos_squared_formula_15deg_l9_9062


namespace curves_intersect_condition_l9_9487

noncomputable def curves_intersect_exactly_three_points (a : ℝ) : Prop :=
  ∃ x y : ℝ, 
    (x^2 + y^2 = a^2) ∧ (y = x^2 + a) ∧ 
    (y = a → x = 0) ∧ 
    ((2 * a + 1 < 0) → y = -(2 * a + 1) - 1)

theorem curves_intersect_condition (a : ℝ) : 
  curves_intersect_exactly_three_points a ↔ a < -1/2 :=
sorry

end curves_intersect_condition_l9_9487


namespace newer_model_distance_l9_9069

theorem newer_model_distance (d_old : ℝ) (p_increase : ℝ) (d_new : ℝ) (h1 : d_old = 300) (h2 : p_increase = 0.30) (h3 : d_new = d_old * (1 + p_increase)) : d_new = 390 :=
by
  sorry

end newer_model_distance_l9_9069


namespace inequality_solution_l9_9684

theorem inequality_solution (a : ℝ) (h : ∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) : a < -1 :=
sorry

end inequality_solution_l9_9684


namespace molecular_weight_3_moles_ascorbic_acid_l9_9141

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_formula_ascorbic_acid : List (ℝ × ℕ) :=
  [(atomic_weight_C, 6), (atomic_weight_H, 8), (atomic_weight_O, 6)]

def molecular_weight (formula : List (ℝ × ℕ)) : ℝ :=
  formula.foldl (λ acc (aw, count) => acc + aw * count) 0.0

def weight_of_moles (mw : ℝ) (moles : ℕ) : ℝ :=
  mw * moles

theorem molecular_weight_3_moles_ascorbic_acid :
  weight_of_moles (molecular_weight molecular_formula_ascorbic_acid) 3 = 528.372 :=
by
  sorry

end molecular_weight_3_moles_ascorbic_acid_l9_9141


namespace original_number_l9_9812

theorem original_number (x : ℝ) (h : x + 0.5 * x = 90) : x = 60 :=
by
  sorry

end original_number_l9_9812


namespace quadratic_sum_roots_l9_9636

-- We define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- The function f passes through points (r, k) and (s, k)
variables (a b c r s k : ℝ)
variable (ha : a ≠ 0)
variable (hr : f a b c r = k)
variable (hs : f a b c s = k)

-- What we want to prove
theorem quadratic_sum_roots :
  f a b c (r + s) = c :=
sorry

end quadratic_sum_roots_l9_9636


namespace boys_without_notebooks_l9_9710

theorem boys_without_notebooks
  (total_boys : ℕ) (students_with_notebooks : ℕ) (girls_with_notebooks : ℕ)
  (h1 : total_boys = 24) (h2 : students_with_notebooks = 30) (h3 : girls_with_notebooks = 17) :
  total_boys - (students_with_notebooks - girls_with_notebooks) = 11 :=
by
  sorry

end boys_without_notebooks_l9_9710


namespace problem_M_plus_N_l9_9757

theorem problem_M_plus_N (M N : ℝ) (H1 : 4/7 = M/77) (H2 : 4/7 = 98/(N^2)) : M + N = 57.1 := 
sorry

end problem_M_plus_N_l9_9757


namespace range_of_m_l9_9366

noncomputable def intersects_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, m * x^2 - 4 * x + 1 = 0

theorem range_of_m (m : ℝ) (h : intersects_x_axis m) : m ≤ 4 := by
  sorry

end range_of_m_l9_9366


namespace bug_twelfth_move_l9_9364

theorem bug_twelfth_move (Q : ℕ → ℚ)
  (hQ0 : Q 0 = 1)
  (hQ1 : Q 1 = 0)
  (hQ2 : Q 2 = 1/2)
  (h_recursive : ∀ n, Q (n + 1) = 1/2 * (1 - Q n)) :
  let m := 683
  let n := 2048
  (Nat.gcd m n = 1) ∧ (m + n = 2731) :=
by
  sorry

end bug_twelfth_move_l9_9364


namespace necessary_condition_l9_9972

theorem necessary_condition (x : ℝ) (h : (x-1) * (x-2) ≤ 0) : x^2 - 3 * x ≤ 0 :=
sorry

end necessary_condition_l9_9972


namespace marbles_given_l9_9146

theorem marbles_given (initial remaining given : ℕ) (h_initial : initial = 143) (h_remaining : remaining = 70) :
    given = initial - remaining → given = 73 :=
by
  intros
  sorry

end marbles_given_l9_9146


namespace solutions_exist_iff_l9_9185

variable (a b : ℝ)

theorem solutions_exist_iff :
  (∃ x y : ℝ, (x^2 + y^2 + xy = a) ∧ (x^2 - y^2 = b)) ↔ (-2 * a ≤ Real.sqrt 3 * b ∧ Real.sqrt 3 * b ≤ 2 * a) :=
sorry

end solutions_exist_iff_l9_9185


namespace equation_of_l_l9_9862

-- Defining the equations of the circles
def circle_O (x y : ℝ) := x^2 + y^2 = 4
def circle_C (x y : ℝ) := x^2 + y^2 + 4 * x - 4 * y + 4 = 0

-- Assuming the line l makes circles O and C symmetric
def symmetric (l : ℝ → ℝ → Prop) := ∀ (x y : ℝ), l x y → 
  (∃ (x' y' : ℝ), circle_O x y ∧ circle_C x' y' ∧ (x + x') / 2 = x' ∧ (y + y') / 2 = y')

-- Stating the theorem to be proven
theorem equation_of_l :
  ∀ l : ℝ → ℝ → Prop, symmetric l → (∀ x y : ℝ, l x y ↔ x - y + 2 = 0) :=
by
  sorry

end equation_of_l_l9_9862


namespace total_food_each_day_l9_9759

-- Definitions as per conditions
def soldiers_first_side : Nat := 4000
def food_per_soldier_first_side : Nat := 10
def soldiers_difference : Nat := 500
def food_difference : Nat := 2

-- Proving the total amount of food
theorem total_food_each_day : 
  let soldiers_second_side := soldiers_first_side - soldiers_difference
  let food_per_soldier_second_side := food_per_soldier_first_side - food_difference
  let total_food_first_side := soldiers_first_side * food_per_soldier_first_side
  let total_food_second_side := soldiers_second_side * food_per_soldier_second_side
  total_food_first_side + total_food_second_side = 68000 := by
  -- Proof is omitted
  sorry

end total_food_each_day_l9_9759


namespace vector_ab_l9_9836

theorem vector_ab
  (A B : ℝ × ℝ)
  (hA : A = (1, -1))
  (hB : B = (1, 2)) :
  (B.1 - A.1, B.2 - A.2) = (0, 3) :=
by
  sorry

end vector_ab_l9_9836


namespace least_number_subtracted_l9_9910

theorem least_number_subtracted (x : ℤ) (N : ℤ) :
  N = 2590 - x →
  (N % 9 = 6) →
  (N % 11 = 6) →
  (N % 13 = 6) →
  x = 10 :=
by
  sorry

end least_number_subtracted_l9_9910


namespace sin_angle_add_pi_over_4_l9_9283

open Real

theorem sin_angle_add_pi_over_4 (α : ℝ) (h1 : (cos α = -3/5) ∧ (sin α = 4/5)) : sin (α + π / 4) = sqrt 2 / 10 :=
by
  sorry

end sin_angle_add_pi_over_4_l9_9283


namespace volume_of_resulting_solid_is_9_l9_9086

-- Defining the initial cube with edge length 3
def initial_cube_edge_length : ℝ := 3

-- Defining the volume of the initial cube
def initial_cube_volume : ℝ := initial_cube_edge_length^3

-- Defining the volume of the resulting solid after some parts are cut off
def resulting_solid_volume : ℝ := 9

-- Theorem stating that given the initial conditions, the volume of the resulting solid is 9
theorem volume_of_resulting_solid_is_9 : resulting_solid_volume = 9 :=
by
  sorry

end volume_of_resulting_solid_is_9_l9_9086


namespace repeating_decimal_product_as_fraction_l9_9384

theorem repeating_decimal_product_as_fraction :
  let x := 37 / 999
  let y := 7 / 9
  x * y = 259 / 8991 := by {
    sorry
  }

end repeating_decimal_product_as_fraction_l9_9384


namespace polynomial_divisible_2520_l9_9001

theorem polynomial_divisible_2520 (n : ℕ) : (n^7 - 14 * n^5 + 49 * n^3 - 36 * n) % 2520 = 0 := 
sorry

end polynomial_divisible_2520_l9_9001


namespace loan_amount_needed_l9_9250

-- Define the total cost of tuition.
def total_tuition : ℝ := 30000

-- Define the amount Sabina has saved.
def savings : ℝ := 10000

-- Define the grant coverage rate.
def grant_coverage_rate : ℝ := 0.4

-- Define the remainder of the tuition after using savings.
def remaining_tuition : ℝ := total_tuition - savings

-- Define the amount covered by the grant.
def grant_amount : ℝ := grant_coverage_rate * remaining_tuition

-- Define the loan amount Sabina needs to apply for.
noncomputable def loan_amount : ℝ := remaining_tuition - grant_amount

-- State the theorem to prove the loan amount needed.
theorem loan_amount_needed : loan_amount = 12000 := by
  sorry

end loan_amount_needed_l9_9250


namespace g_difference_l9_9133

def g (n : ℕ) : ℚ :=
  1/4 * n * (n + 1) * (n + 2) * (n + 3)

theorem g_difference (r : ℕ) : g r - g (r - 1) = r * (r + 1) * (r + 2) :=
  sorry

end g_difference_l9_9133


namespace no_solution_l9_9251

theorem no_solution : ∀ x y z t : ℕ, 16^x + 21^y + 26^z ≠ t^2 :=
by
  intro x y z t
  sorry

end no_solution_l9_9251


namespace boys_in_2nd_l9_9013

def students_in_3rd := 19
def students_in_4th := 2 * students_in_3rd
def girls_in_2nd := 19
def total_students := 86
def students_in_2nd := total_students - students_in_3rd - students_in_4th

theorem boys_in_2nd : students_in_2nd - girls_in_2nd = 10 := by
  sorry

end boys_in_2nd_l9_9013


namespace find_a_plus_c_l9_9644

theorem find_a_plus_c (a b c d : ℝ)
  (h₁ : -(3 - a) ^ 2 + b = 6) (h₂ : (3 - c) ^ 2 + d = 6)
  (h₃ : -(7 - a) ^ 2 + b = 2) (h₄ : (7 - c) ^ 2 + d = 2) :
  a + c = 10 := sorry

end find_a_plus_c_l9_9644


namespace average_of_first_21_multiples_of_7_l9_9444

theorem average_of_first_21_multiples_of_7 :
  let a1 := 7
  let d := 7
  let n := 21
  let an := a1 + (n - 1) * d
  let Sn := n / 2 * (a1 + an)
  Sn / n = 77 :=
by
  let a1 := 7
  let d := 7
  let n := 21
  let an := a1 + (n - 1) * d
  let Sn := n / 2 * (a1 + an)
  have h1 : an = 147 := by
    sorry
  have h2 : Sn = 1617 := by
    sorry
  have h3 : Sn / n = 77 := by
    sorry
  exact h3

end average_of_first_21_multiples_of_7_l9_9444


namespace quadratic_distinct_real_roots_l9_9834

theorem quadratic_distinct_real_roots (a : ℝ) (h : a ≠ 1) : 
(a < 2) → 
(∃ x y : ℝ, x ≠ y ∧ (a-1)*x^2 - 2*x + 1 = 0 ∧ (a-1)*y^2 - 2*y + 1 = 0) :=
sorry

end quadratic_distinct_real_roots_l9_9834


namespace larger_number_is_42_l9_9954

theorem larger_number_is_42 (x y : ℕ) (h1 : x + y = 77) (h2 : 5 * x = 6 * y) : x = 42 :=
by
  sorry

end larger_number_is_42_l9_9954


namespace numbers_divisible_l9_9300

theorem numbers_divisible (n : ℕ) (d1 d2 : ℕ) (lcm_d1_d2 : ℕ) (limit : ℕ) (h_lcm: lcm d1 d2 = lcm_d1_d2) (h_limit : limit = 2011)
(h_d1 : d1 = 117) (h_d2 : d2 = 2) : 
  ∃ k : ℕ, k = 8 ∧ ∀ m : ℕ, m < limit → (m % lcm_d1_d2 = 0 ↔ ∃ i : ℕ, i < k ∧ m = lcm_d1_d2 * (i + 1)) :=
by
  sorry

end numbers_divisible_l9_9300


namespace regular_n_gon_center_inside_circle_l9_9455

-- Define a regular n-gon
structure RegularNGon (n : ℕ) :=
(center : ℝ × ℝ)
(vertices : Fin n → (ℝ × ℝ))

-- Define the condition to be able to roll and reflect the n-gon over any of its sides
def canReflectSymmetrically (n : ℕ) (g : RegularNGon n) : Prop := sorry

-- Definition of a circle with a given center and radius
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

-- Define the problem for determining if reflection can bring the center of n-gon inside any circle
def canCenterBeInsideCircle (n : ℕ) (g : RegularNGon n) (c : Circle) : Prop :=
  ∃ (f : ℝ × ℝ → ℝ × ℝ), -- Some function representing the reflections
    canReflectSymmetrically n g ∧ f g.center = c.center

-- State the main theorem determining for which n-gons the assertion is true
theorem regular_n_gon_center_inside_circle (n : ℕ) 
  (h : n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6) : 
  ∀ (g : RegularNGon n) (c : Circle), canCenterBeInsideCircle n g c :=
sorry

end regular_n_gon_center_inside_circle_l9_9455


namespace find_income_l9_9933

-- Definitions of percentages used in calculations
def rent_percentage : ℝ := 0.15
def education_percentage : ℝ := 0.15
def misc_percentage : ℝ := 0.10
def medical_percentage : ℝ := 0.15

-- Remaining amount after all expenses
def final_amount : ℝ := 5548

-- Income calculation function
def calc_income (X : ℝ) : ℝ :=
  let after_rent := X * (1 - rent_percentage)
  let after_education := after_rent * (1 - education_percentage)
  let after_misc := after_education * (1 - misc_percentage)
  let after_medical := after_misc * (1 - medical_percentage)
  after_medical

-- Theorem statement to prove the woman's income
theorem find_income (X : ℝ) (h : calc_income X = final_amount) : X = 10038.46 := by
  sorry

end find_income_l9_9933


namespace two_digit_numbers_div_by_7_with_remainder_1_l9_9047

theorem two_digit_numbers_div_by_7_with_remainder_1 :
  {n : ℕ | ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ (10 * a + b) % 7 = 1 ∧ (10 * b + a) % 7 = 1} 
  = {22, 29, 92, 99} := 
by
  sorry

end two_digit_numbers_div_by_7_with_remainder_1_l9_9047


namespace probability_of_sphere_in_cube_l9_9715

noncomputable def cube_volume : Real :=
  (4 : Real)^3

noncomputable def sphere_volume : Real :=
  (4 / 3) * Real.pi * (2 : Real)^3

noncomputable def probability : Real :=
  sphere_volume / cube_volume

theorem probability_of_sphere_in_cube : probability = Real.pi / 6 := by
  sorry

end probability_of_sphere_in_cube_l9_9715


namespace AndyCoordinatesAfter1500Turns_l9_9585

/-- Definition for Andy's movement rules given his starting position. -/
def AndyPositionAfterTurns (turns : ℕ) : ℤ × ℤ :=
  let rec move (x y : ℤ) (length : ℤ) (dir : ℕ) (remainingTurns : ℕ) : ℤ × ℤ :=
    match remainingTurns with
    | 0 => (x, y)
    | n+1 => 
        let (dx, dy) := match dir % 4 with
                        | 0 => (0, 1)
                        | 1 => (1, 0)
                        | 2 => (0, -1)
                        | _ => (-1, 0)
        move (x + dx * length) (y + dy * length) (length + 1) (dir + 1) n
  move (-30) 25 2 0 turns

theorem AndyCoordinatesAfter1500Turns :
  AndyPositionAfterTurns 1500 = (-280141, 280060) :=
by
  sorry

end AndyCoordinatesAfter1500Turns_l9_9585


namespace num_groups_of_consecutive_natural_numbers_l9_9800

theorem num_groups_of_consecutive_natural_numbers (n : ℕ) (h : 3 * n + 3 < 19) : n < 6 := 
  sorry

end num_groups_of_consecutive_natural_numbers_l9_9800


namespace find_B_l9_9377

theorem find_B (A B : ℕ) (h₁ : 6 * A + 10 * B + 2 = 77) (h₂ : A ≤ 9) (h₃ : B ≤ 9) : B = 1 := sorry

end find_B_l9_9377


namespace alberto_bikes_more_l9_9851

-- Definitions of given speeds
def alberto_speed : ℝ := 15
def bjorn_speed : ℝ := 11.25

-- The time duration considered
def time_hours : ℝ := 5

-- Calculate the distances each traveled
def alberto_distance : ℝ := alberto_speed * time_hours
def bjorn_distance : ℝ := bjorn_speed * time_hours

-- Calculate the difference in distances
def distance_difference : ℝ := alberto_distance - bjorn_distance

-- The theorem to be proved
theorem alberto_bikes_more : distance_difference = 18.75 := by
    sorry

end alberto_bikes_more_l9_9851


namespace rachel_picture_shelves_l9_9716

-- We define the number of books per shelf
def books_per_shelf : ℕ := 9

-- We define the number of mystery shelves
def mystery_shelves : ℕ := 6

-- We define the total number of books
def total_books : ℕ := 72

-- We create a theorem that states Rachel had 2 shelves of picture books
theorem rachel_picture_shelves : ∃ (picture_shelves : ℕ), 
  (mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = total_books) ∧
  picture_shelves = 2 := by
  sorry

end rachel_picture_shelves_l9_9716


namespace train_crossing_time_l9_9537

def train_length : ℝ := 150
def train_speed : ℝ := 179.99999999999997

theorem train_crossing_time : train_length / train_speed = 0.8333333333333333 := by
  sorry

end train_crossing_time_l9_9537


namespace victoria_gym_sessions_l9_9956

-- Define the initial conditions
def starts_on_monday := true
def sessions_per_two_week_cycle := 6
def total_sessions := 30

-- Define the sought day of the week when all gym sessions are completed
def final_day := "Thursday"

-- The theorem stating the problem
theorem victoria_gym_sessions : 
  starts_on_monday →
  sessions_per_two_week_cycle = 6 →
  total_sessions = 30 →
  final_day = "Thursday" := 
by
  intros
  exact sorry

end victoria_gym_sessions_l9_9956


namespace divides_sum_if_divides_polynomial_l9_9618

theorem divides_sum_if_divides_polynomial (x y : ℕ) : 
  x^2 ∣ x^2 + x * y + x + y → x^2 ∣ x + y :=
by
  sorry

end divides_sum_if_divides_polynomial_l9_9618


namespace cost_of_fencing_l9_9907

noncomputable def fencingCost :=
  let π := 3.14159
  let diameter := 32
  let costPerMeter := 1.50
  let circumference := π * diameter
  let totalCost := costPerMeter * circumference
  totalCost

theorem cost_of_fencing :
  let roundedCost := (fencingCost).round
  roundedCost = 150.80 :=
by
  sorry

end cost_of_fencing_l9_9907


namespace find_third_angle_of_triangle_l9_9477

theorem find_third_angle_of_triangle (a b c : ℝ) (h₁ : a = 40) (h₂ : b = 3 * c) (h₃ : a + b + c = 180) : c = 35 := 
by sorry

end find_third_angle_of_triangle_l9_9477


namespace volume_expression_correct_l9_9034

variable (x : ℝ)

def volume (x : ℝ) := x * (30 - 2 * x) * (20 - 2 * x)

theorem volume_expression_correct (h : x < 10) :
  volume x = 4 * x^3 - 100 * x^2 + 600 * x :=
by sorry

end volume_expression_correct_l9_9034


namespace usual_time_to_school_l9_9568

theorem usual_time_to_school (R T : ℝ) (h : (R * T = (6/5) * R * (T - 4))) : T = 24 :=
by 
  sorry

end usual_time_to_school_l9_9568


namespace basketball_weight_calc_l9_9316

-- Define the variables and conditions
variable (weight_basketball weight_watermelon : ℕ)
variable (h1 : 8 * weight_basketball = 4 * weight_watermelon)
variable (h2 : weight_watermelon = 32)

-- Statement to prove
theorem basketball_weight_calc : weight_basketball = 16 :=
by
  sorry

end basketball_weight_calc_l9_9316


namespace find_PO_l9_9155

variables {P : ℝ × ℝ} {O F : ℝ × ℝ}

def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1
def origin (O : ℝ × ℝ) : Prop := O = (0, 0)
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)
def isosceles_triangle (O P F : ℝ × ℝ) : Prop :=
  dist O P = dist O F ∨ dist O P = dist P F

theorem find_PO
  (P : ℝ × ℝ) (O : ℝ × ℝ) (F : ℝ × ℝ)
  (hO : origin O) (hF : focus F) (hP : on_parabola P) (h_iso : isosceles_triangle O P F) :
  dist O P = 1 ∨ dist O P = 3 / 2 :=
sorry

end find_PO_l9_9155


namespace smaller_number_of_two_digits_product_3774_l9_9005

theorem smaller_number_of_two_digits_product_3774 (a b : ℕ) (ha : 9 < a ∧ a < 100) (hb : 9 < b ∧ b < 100) (h : a * b = 3774) : a = 51 ∨ b = 51 :=
by
  sorry

end smaller_number_of_two_digits_product_3774_l9_9005


namespace largest_possible_b_l9_9259

theorem largest_possible_b (a b c : ℕ) (h₁ : 1 < c) (h₂ : c < b) (h₃ : b < a) (h₄ : a * b * c = 360): b = 12 :=
by
  sorry

end largest_possible_b_l9_9259


namespace sum_of_legs_l9_9657

theorem sum_of_legs (x : ℕ) (h : x^2 + (x + 1)^2 = 41^2) : x + (x + 1) = 57 :=
sorry

end sum_of_legs_l9_9657


namespace experiment_variance_l9_9317

noncomputable def probability_of_success : ℚ := 5/9

noncomputable def variance_of_binomial (n : ℕ) (p : ℚ) : ℚ :=
  n * p * (1 - p)

def number_of_experiments : ℕ := 30

theorem experiment_variance :
  variance_of_binomial number_of_experiments probability_of_success = 200/27 :=
by
  sorry

end experiment_variance_l9_9317


namespace sum_of_arithmetic_sequences_l9_9941

theorem sum_of_arithmetic_sequences (n : ℕ) (h : n ≠ 0) :
  (2 * n * (n + 3) = n * (n + 12)) → (n = 6) :=
by
  intro h_eq
  have h_nonzero : n ≠ 0 := h
  sorry

end sum_of_arithmetic_sequences_l9_9941


namespace paul_spending_l9_9161

theorem paul_spending :
  let cost_of_dress_shirts := 4 * 15
  let cost_of_pants := 2 * 40
  let cost_of_suit := 150
  let cost_of_sweaters := 2 * 30
  let total_cost := cost_of_dress_shirts + cost_of_pants + cost_of_suit + cost_of_sweaters
  let store_discount := 0.2 * total_cost
  let after_store_discount := total_cost - store_discount
  let coupon_discount := 0.1 * after_store_discount
  let final_amount := after_store_discount - coupon_discount
  final_amount = 252 :=
by
  -- Mathematically equivalent proof problem.
  sorry

end paul_spending_l9_9161


namespace exponential_inequality_l9_9241

theorem exponential_inequality (a x1 x2 : ℝ) (h1 : 1 < a) (h2 : x1 < x2) :
  |a ^ ((1 / 2) * (x1 + x2)) - a ^ x1| < |a ^ x2 - a ^ ((1 / 2) * (x1 + x2))| :=
by
  sorry

end exponential_inequality_l9_9241


namespace gino_gave_away_l9_9405

theorem gino_gave_away (initial_sticks given_away left_sticks : ℝ) 
  (h1 : initial_sticks = 63.0) (h2 : left_sticks = 13.0) 
  (h3 : left_sticks = initial_sticks - given_away) : 
  given_away = 50.0 :=
by
  sorry

end gino_gave_away_l9_9405


namespace percent_increase_l9_9033

theorem percent_increase (original new : ℕ) (h1 : original = 30) (h2 : new = 60) :
  ((new - original) / original) * 100 = 100 := 
by
  sorry

end percent_increase_l9_9033


namespace lcm_inequality_l9_9184

theorem lcm_inequality (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) : 
  Nat.lcm k m * Nat.lcm m n * Nat.lcm n k ≥ Nat.lcm (Nat.lcm k m) n ^ 2 :=
by sorry

end lcm_inequality_l9_9184


namespace find_q_l9_9816

open Polynomial

-- Define the conditions for the roots of the first polynomial
def roots_of_first_eq (a b m : ℝ) (h : a * b = 3) : Prop := 
  ∀ x, (x^2 - m*x + 3) = (x - a) * (x - b)

-- Define the problem statement
theorem find_q (a b m p q : ℝ) 
  (h1 : a * b = 3) 
  (h2 : ∀ x, (x^2 - m*x + 3) = (x - a) * (x - b)) 
  (h3 : ∀ x, (x^2 - p*x + q) = (x - (a + 2/b)) * (x - (b + 2/a))) :
  q = 25 / 3 :=
sorry

end find_q_l9_9816


namespace fraction_to_decimal_l9_9510

theorem fraction_to_decimal : (17 : ℝ) / 50 = 0.34 := 
by 
  sorry

end fraction_to_decimal_l9_9510


namespace prob1_prob2_prob3_l9_9371

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x^2 + 2
  else x

theorem prob1 :
  (∀ x, x ≥ 0 → f x = x^2 + 2) ∧
  (∀ x, x < 0 → f x = x) :=
by
  sorry

theorem prob2 : f 5 = 27 :=
by 
  sorry

theorem prob3 : ∀ (x : ℝ), f x = 0 → false :=
by
  sorry

end prob1_prob2_prob3_l9_9371


namespace infinite_integer_solutions_l9_9603

variable (x : ℤ)

theorem infinite_integer_solutions (x : ℤ) : 
  ∃ (k : ℤ), ∀ n : ℤ, n > 2 → k = n :=
by {
  sorry
}

end infinite_integer_solutions_l9_9603


namespace side_ratio_triangle_square_pentagon_l9_9951

-- Define the conditions
def perimeter_triangle (t : ℝ) := 3 * t = 18
def perimeter_square (s : ℝ) := 4 * s = 16
def perimeter_pentagon (p : ℝ) := 5 * p = 20

-- Statement to be proved
theorem side_ratio_triangle_square_pentagon 
  (t s p : ℝ)
  (ht : perimeter_triangle t)
  (hs : perimeter_square s)
  (hp : perimeter_pentagon p) : 
  (t / s = 3 / 2) ∧ (t / p = 3 / 2) := 
sorry

end side_ratio_triangle_square_pentagon_l9_9951


namespace part1_part2_l9_9621

def is_sum_solution_equation (a b x : ℝ) : Prop :=
  x = b + a

def part1_statement := ¬ is_sum_solution_equation 3 4.5 (4.5 / 3)

def part2_statement (m : ℝ) : Prop :=
  is_sum_solution_equation 5 (m + 1) (m + 6) → m = (-29 / 4)

theorem part1 : part1_statement :=
by 
  -- Proof here
  sorry

theorem part2 (m : ℝ) : part2_statement m :=
by 
  -- Proof here
  sorry

end part1_part2_l9_9621


namespace average_abcd_l9_9861

-- Define the average condition of the numbers 4, 6, 9, a, b, c, d given as 20
def average_condition (a b c d : ℝ) : Prop :=
  (4 + 6 + 9 + a + b + c + d) / 7 = 20

-- Prove that the average of a, b, c, and d is 30.25 given the above condition
theorem average_abcd (a b c d : ℝ) (h : average_condition a b c d) : 
  (a + b + c + d) / 4 = 30.25 :=
by
  sorry

end average_abcd_l9_9861


namespace maximal_inradius_of_tetrahedron_l9_9481

-- Define the properties and variables
variables (A B C D : ℝ) (h_A h_B h_C h_D : ℝ) (V r : ℝ)

-- Assumptions
variable (h_A_ge_1 : h_A ≥ 1)
variable (h_B_ge_1 : h_B ≥ 1)
variable (h_C_ge_1 : h_C ≥ 1)
variable (h_D_ge_1 : h_D ≥ 1)

-- Volume expressed in terms of altitudes and face areas
axiom vol_eq_Ah : V = (1 / 3) * A * h_A
axiom vol_eq_Bh : V = (1 / 3) * B * h_B
axiom vol_eq_Ch : V = (1 / 3) * C * h_C
axiom vol_eq_Dh : V = (1 / 3) * D * h_D

-- Volume expressed in terms of inradius and sum of face areas
axiom vol_eq_inradius : V = (1 / 3) * (A + B + C + D) * r

-- The theorem to prove
theorem maximal_inradius_of_tetrahedron : r = 1 / 4 :=
sorry

end maximal_inradius_of_tetrahedron_l9_9481


namespace common_difference_arithmetic_seq_l9_9649

theorem common_difference_arithmetic_seq (S n a1 d : ℕ) (h_sum : S = 650) (h_n : n = 20) (h_a1 : a1 = 4) :
  S = (n / 2) * (2 * a1 + (n - 1) * d) → d = 3 := by
  intros h_formula
  sorry

end common_difference_arithmetic_seq_l9_9649


namespace cafeteria_B_turnover_higher_in_May_l9_9526

noncomputable def initial_turnover (X a r : ℝ) : Prop :=
  ∃ (X a r : ℝ),
    (X + 8 * a = X * (1 + r) ^ 8) ∧
    ((X + 4 * a) < (X * (1 + r) ^ 4))

theorem cafeteria_B_turnover_higher_in_May (X a r : ℝ) :
    (X + 8 * a = X * (1 + r) ^ 8) → (X + 4 * a < X * (1 + r) ^ 4) :=
  sorry

end cafeteria_B_turnover_higher_in_May_l9_9526


namespace perfect_square_eq_m_val_l9_9532

theorem perfect_square_eq_m_val (m : ℝ) (h : ∃ a : ℝ, x^2 - m * x + 49 = (x - a)^2) : m = 14 ∨ m = -14 :=
by
  sorry

end perfect_square_eq_m_val_l9_9532


namespace distance_between_points_l9_9187

theorem distance_between_points 
  (v_A v_B : ℝ) 
  (d : ℝ) 
  (h1 : 4 * v_A + 4 * v_B = d)
  (h2 : 3.5 * (v_A + 3) + 3.5 * (v_B + 3) = d) : 
  d = 168 := 
by 
  sorry

end distance_between_points_l9_9187


namespace shelves_needed_l9_9122

def books_in_stock : Nat := 27
def books_sold : Nat := 6
def books_per_shelf : Nat := 7

theorem shelves_needed :
  let remaining_books := books_in_stock - books_sold
  let shelves := remaining_books / books_per_shelf
  shelves = 3 :=
by
  sorry

end shelves_needed_l9_9122


namespace gcd_10010_15015_l9_9746

theorem gcd_10010_15015 :
  let n1 := 10010
  let n2 := 15015
  ∃ d, d = Nat.gcd n1 n2 ∧ d = 5005 :=
by
  let n1 := 10010
  let n2 := 15015
  -- ... omitted proof steps
  sorry

end gcd_10010_15015_l9_9746


namespace roots_quadratic_eq_value_l9_9098

theorem roots_quadratic_eq_value (d e : ℝ) (h : 3 * d^2 + 4 * d - 7 = 0) (h' : 3 * e^2 + 4 * e - 7 = 0) : 
  (d - 2) * (e - 2) = 13 / 3 := 
by
  sorry

end roots_quadratic_eq_value_l9_9098


namespace find_distance_l9_9420

variable (A B : Point)
variable (distAB : ℝ) -- the distance between A and B
variable (meeting1 : ℝ) -- first meeting distance from A
variable (meeting2 : ℝ) -- second meeting distance from B

-- Conditions
axiom meeting_conditions_1 : meeting1 = 70
axiom meeting_conditions_2 : meeting2 = 90

-- Prove the distance between A and B is 120 km
def distance_from_A_to_B : ℝ := 120

theorem find_distance : distAB = distance_from_A_to_B := 
sorry

end find_distance_l9_9420


namespace solve_for_x_l9_9441

def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

theorem solve_for_x (x : ℝ) : (custom_mul 3 (custom_mul 6 x) = 2) → (x = 19 / 2) :=
sorry

end solve_for_x_l9_9441


namespace angle_BPE_l9_9091

-- Define the conditions given in the problem
def triangle_ABC (A B C : ℝ) : Prop := A = 60 ∧ 
  (∃ (B₁ B₂ B₃ : ℝ), B₁ = B / 3 ∧ B₂ = B / 3 ∧ B₃ = B / 3) ∧ 
  (∃ (C₁ C₂ C₃ : ℝ), C₁ = C / 3 ∧ C₂ = C / 3 ∧ C₃ = C / 3) ∧ 
  (B + C = 120)

-- State the theorem to proof
theorem angle_BPE (A B C x : ℝ) (h : triangle_ABC A B C) : x = 50 := by
  sorry

end angle_BPE_l9_9091


namespace no_positive_n_for_prime_expr_l9_9413

noncomputable def is_prime (p : ℤ) : Prop := p > 1 ∧ (∀ m : ℤ, 1 < m → m < p → ¬ (m ∣ p))

theorem no_positive_n_for_prime_expr : 
  ∀ n : ℕ, 0 < n → ¬ is_prime (n^3 - 9 * n^2 + 23 * n - 17) := by
  sorry

end no_positive_n_for_prime_expr_l9_9413


namespace proof_q1_a1_proof_q2_a2_proof_q3_a3_proof_q4_a4_l9_9419

variables (G : Type) [Group G] (kidney testis liver : G)
variables (SudanIII gentianViolet JanusGreenB dissociationFixative : G)

-- Conditions c1, c2, c3
def c1 : Prop := True -- Meiosis occurs in gonads, we simplify this in Lean to a true condition for brevity
def c2 : Prop := True -- Steps for slide preparation
def c3 : Prop := True -- Materials available

-- Questions
def q1 : G := testis
def q2 : G := dissociationFixative
def q3 : G := gentianViolet
def q4 : List G := [kidney, dissociationFixative, gentianViolet] -- Assume these are placeholders for correct cell types

-- Answers
def a1 : G := testis
def a2 : G := dissociationFixative
def a3 : G := gentianViolet
def a4 : List G := [testis, dissociationFixative, gentianViolet] -- Correct cells

-- Proving the equivalence of questions and answers given the conditions
theorem proof_q1_a1 : c1 ∧ c2 ∧ c3 → q1 = a1 := 
by sorry

theorem proof_q2_a2 : c1 ∧ c2 ∧ c3 → q2 = a2 := 
by sorry

theorem proof_q3_a3 : c1 ∧ c2 ∧ c3 → q3 = a3 := 
by sorry

theorem proof_q4_a4 : c1 ∧ c2 ∧ c3 → q4 = a4 := 
by sorry

end proof_q1_a1_proof_q2_a2_proof_q3_a3_proof_q4_a4_l9_9419


namespace arithmetic_sequence_sum_l9_9150

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h : ∀ n, a n = a 1 + (n - 1) * d) (h_6 : a 6 = 1) :
  a 2 + a 10 = 2 := 
sorry

end arithmetic_sequence_sum_l9_9150


namespace inheritance_amount_l9_9011

-- Definitions of conditions
def inheritance (y : ℝ) : Prop :=
  let federalTaxes := 0.25 * y
  let remainingAfterFederal := 0.75 * y
  let stateTaxes := 0.1125 * y
  let totalTaxes := federalTaxes + stateTaxes
  totalTaxes = 12000

-- Theorem statement
theorem inheritance_amount (y : ℝ) (h : inheritance y) : y = 33103 :=
sorry

end inheritance_amount_l9_9011


namespace caller_wins_both_at_35_l9_9690

theorem caller_wins_both_at_35 (n : ℕ) :
  ∀ n, (n % 5 = 0 ∧ n % 7 = 0) ↔ n = 35 :=
by
  sorry

end caller_wins_both_at_35_l9_9690


namespace intersection_complement_eq_l9_9478

open Set

variable (U A B : Set ℕ)

theorem intersection_complement_eq :
  (U = {1, 2, 3, 4, 5, 6}) →
  (A = {1, 3}) →
  (B = {3, 4, 5}) →
  A ∩ (U \ B) = {1} :=
by
  intros hU hA hB
  subst hU
  subst hA
  subst hB
  sorry

end intersection_complement_eq_l9_9478


namespace john_naps_70_days_l9_9204

def total_naps_in_days (naps_per_week nap_duration days_in_week total_days : ℕ) : ℕ :=
  let total_weeks := total_days / days_in_week
  let total_naps := total_weeks * naps_per_week
  total_naps * nap_duration

theorem john_naps_70_days
  (naps_per_week : ℕ)
  (nap_duration : ℕ)
  (days_in_week : ℕ)
  (total_days : ℕ)
  (h_naps_per_week : naps_per_week = 3)
  (h_nap_duration : nap_duration = 2)
  (h_days_in_week : days_in_week = 7)
  (h_total_days : total_days = 70) :
  total_naps_in_days naps_per_week nap_duration days_in_week total_days = 60 :=
by
  rw [h_naps_per_week, h_nap_duration, h_days_in_week, h_total_days]
  sorry

end john_naps_70_days_l9_9204


namespace product_of_real_roots_l9_9087

theorem product_of_real_roots (x : ℝ) (h : x^5 = 100) : x = 10^(2/5) := by
  sorry

end product_of_real_roots_l9_9087


namespace harry_travel_time_l9_9287

variables (bus_time1 bus_time2 : ℕ) (walk_ratio : ℕ)
-- Conditions based on the problem
-- Harry has already been sat on the bus for 15 minutes.
def part1_time : ℕ := 15

-- and he knows the rest of the journey will take another 25 minutes.
def part2_time : ℕ := 25

-- The total bus journey time
def total_bus_time : ℕ := part1_time + part2_time

-- The walk from the bus stop to his house will take half the amount of time the bus journey took.
def walk_time : ℕ := total_bus_time / 2

-- Total travel time
def total_travel_time : ℕ := total_bus_time + walk_time

-- Rewrite the proof problem statement
theorem harry_travel_time : total_travel_time = 60 := by
  sorry

end harry_travel_time_l9_9287


namespace equal_naturals_of_infinite_divisibility_l9_9065

theorem equal_naturals_of_infinite_divisibility
  (a b : ℕ)
  (h : ∀ᶠ n in Filter.atTop, (a^(n + 1) + b^(n + 1)) % (a^n + b^n) = 0) :
  a = b :=
sorry

end equal_naturals_of_infinite_divisibility_l9_9065


namespace cos_832_eq_cos_l9_9523

theorem cos_832_eq_cos (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * Real.pi / 180) = Real.cos (832 * Real.pi / 180)) : n = 112 := 
  sorry

end cos_832_eq_cos_l9_9523


namespace ice_cream_cost_l9_9243

theorem ice_cream_cost
  (num_pennies : ℕ) (num_nickels : ℕ) (num_dimes : ℕ) (num_quarters : ℕ) 
  (leftover_cents : ℤ) (num_family_members : ℕ)
  (h_pennies : num_pennies = 123)
  (h_nickels : num_nickels = 85)
  (h_dimes : num_dimes = 35)
  (h_quarters : num_quarters = 26)
  (h_leftover : leftover_cents = 48)
  (h_members : num_family_members = 5) :
  (123 * 0.01 + 85 * 0.05 + 35 * 0.1 + 26 * 0.25 - 0.48) / 5 = 3 :=
by
  sorry

end ice_cream_cost_l9_9243


namespace sam_drove_200_miles_l9_9952

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l9_9952


namespace probability_of_green_l9_9616

-- Define the conditions
def P_R : ℝ := 0.15
def P_O : ℝ := 0.35
def P_B : ℝ := 0.2
def total_probability (P_Y P_G : ℝ) : Prop := P_R + P_O + P_B + P_Y + P_G = 1

-- State the theorem to be proven
theorem probability_of_green (P_Y : ℝ) (P_G : ℝ) (h : total_probability P_Y P_G) (P_Y_assumption : P_Y = 0.15) : P_G = 0.15 :=
by
  sorry

end probability_of_green_l9_9616


namespace base_8_addition_l9_9050

theorem base_8_addition (X Y : ℕ) (h1 : Y + 2 % 8 = X % 8) (h2 : X + 3 % 8 = 2 % 8) : X + Y = 12 := by
  sorry

end base_8_addition_l9_9050


namespace min_value_correct_l9_9703

noncomputable def min_value (x y : ℝ) : ℝ :=
x * y / (x^2 + y^2)

theorem min_value_correct :
  ∃ x y : ℝ,
    (2 / 5 : ℝ) ≤ x ∧ x ≤ (1 / 2 : ℝ) ∧
    (1 / 3 : ℝ) ≤ y ∧ y ≤ (3 / 8 : ℝ) ∧
    min_value x y = (6 / 13 : ℝ) :=
by sorry

end min_value_correct_l9_9703


namespace positive_integer_expression_l9_9808

-- Define the existence conditions for a given positive integer n
theorem positive_integer_expression (n : ℕ) (h : 0 < n) : ∃ a b c : ℤ, (n = a^2 + b^2 + c^2 + c) := 
sorry

end positive_integer_expression_l9_9808


namespace intersection_points_l9_9581

theorem intersection_points (x y : ℝ) (h1 : x^2 - 4 * y^2 = 4) (h2 : x = 3 * y) : 
  (x, y) = (3, 1) ∨ (x, y) = (-3, -1) :=
sorry

end intersection_points_l9_9581


namespace luke_pages_lemma_l9_9376

def number_of_new_cards : ℕ := 3
def number_of_old_cards : ℕ := 9
def cards_per_page : ℕ := 3
def total_number_of_cards := number_of_new_cards + number_of_old_cards
def total_number_of_pages := total_number_of_cards / cards_per_page

theorem luke_pages_lemma : total_number_of_pages = 4 := by
  sorry

end luke_pages_lemma_l9_9376


namespace vacation_cost_division_l9_9613

theorem vacation_cost_division (n : ℕ) (total_cost : ℕ) 
  (cost_difference : ℕ)
  (cost_per_person_5 : ℕ) :
  total_cost = 1000 → 
  cost_difference = 50 → 
  cost_per_person_5 = total_cost / 5 →
  (total_cost / n) = cost_per_person_5 + cost_difference → 
  n = 4 := 
by
  intros h1 h2 h3 h4
  sorry

end vacation_cost_division_l9_9613


namespace second_train_length_l9_9212

noncomputable def length_of_second_train (speed1_kmph speed2_kmph time_sec length1_m : ℝ) : ℝ :=
  let relative_speed_mps := (speed1_kmph + speed2_kmph) * (1000 / 3600)
  let total_distance := relative_speed_mps * time_sec
  total_distance - length1_m

theorem second_train_length :
  length_of_second_train 60 48 9.99920006399488 140 = 159.9760019198464 :=
by
  sorry

end second_train_length_l9_9212


namespace find_number_l9_9782

theorem find_number (x : ℝ) (h : (x + 0.005) / 2 = 0.2025) : x = 0.400 :=
sorry

end find_number_l9_9782


namespace algebraic_expression_1_algebraic_expression_2_l9_9110

-- Problem 1
theorem algebraic_expression_1 (a : ℚ) (h : a = 4 / 5) : -24.7 * a + 1.3 * a - (33 / 5) * a = -24 := 
by 
  sorry

-- Problem 2
theorem algebraic_expression_2 (a b : ℕ) (ha : a = 899) (hb : b = 101) : a^2 + 2 * a * b + b^2 = 1000000 := 
by 
  sorry

end algebraic_expression_1_algebraic_expression_2_l9_9110


namespace board_tiling_condition_l9_9858

-- Define the problem in Lean

theorem board_tiling_condition (n : ℕ) : 
  (∃ m : ℕ, n * n = m + 4 * m) ↔ (∃ k : ℕ, n = 5 * k ∧ n > 5) := by 
sorry

end board_tiling_condition_l9_9858


namespace simple_interest_initial_amount_l9_9344

theorem simple_interest_initial_amount :
  ∃ P : ℝ, (P + P * 0.04 * 5 = 900) ∧ P = 750 :=
by
  sorry

end simple_interest_initial_amount_l9_9344


namespace symm_diff_complement_l9_9234

variable {U : Type} -- Universal set U
variable (A B : Set U) -- Sets A and B

-- Definition of symmetric difference
def symm_diff (X Y : Set U) : Set U := (X ∪ Y) \ (X ∩ Y)

theorem symm_diff_complement (A B : Set U) :
  (symm_diff A B) = (symm_diff (Aᶜ) (Bᶜ)) :=
sorry

end symm_diff_complement_l9_9234


namespace parallel_a_b_projection_a_onto_b_l9_9877

noncomputable section

open Real

def a : ℝ × ℝ := (sqrt 3, 1)
def b (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)

theorem parallel_a_b (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) (h_parallel : (a.1 / a.2) = (b θ).1 / (b θ).2) : θ = π / 6 := sorry

theorem projection_a_onto_b (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) (h_proj : (sqrt 3 * cos θ + sin θ) = -sqrt 3) : b θ = (-1, 0) := sorry

end parallel_a_b_projection_a_onto_b_l9_9877


namespace range_of_a_l9_9761

def set_A (a : ℝ) : Set ℝ := {-1, 0, a}
def set_B : Set ℝ := {x : ℝ | 1/3 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : (set_A a) ∩ set_B ≠ ∅) : 1/3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l9_9761


namespace smallest_integer_solution_of_inequality_l9_9451

theorem smallest_integer_solution_of_inequality : ∃ x : ℤ, (3 * x ≥ x - 5) ∧ (∀ y : ℤ, 3 * y ≥ y - 5 → y ≥ -2) := 
sorry

end smallest_integer_solution_of_inequality_l9_9451


namespace common_altitude_l9_9037

theorem common_altitude (A1 A2 b1 b2 h : ℝ)
    (hA1 : A1 = 800)
    (hA2 : A2 = 1200)
    (hb1 : b1 = 40)
    (hb2 : b2 = 60)
    (h1 : A1 = 1 / 2 * b1 * h)
    (h2 : A2 = 1 / 2 * b2 * h) :
    h = 40 := 
sorry

end common_altitude_l9_9037


namespace problem_proof_l9_9818

theorem problem_proof (a b x y : ℝ) (h1 : a + b = 0) (h2 : x * y = 1) : 5 * |a + b| - 5 * (x * y) = -5 :=
by
  sorry

end problem_proof_l9_9818


namespace breadth_of_rectangle_l9_9216

theorem breadth_of_rectangle 
  (Perimeter Length Breadth : ℝ)
  (h_perimeter_eq : Perimeter = 2 * (Length + Breadth))
  (h_given_perimeter : Perimeter = 480)
  (h_given_length : Length = 140) :
  Breadth = 100 := 
by
  sorry

end breadth_of_rectangle_l9_9216


namespace slope_of_line_l9_9400

theorem slope_of_line (x y : ℝ) : (4 * y = 5 * x - 20) → (y = (5/4) * x - 5) :=
by
  intro h
  sorry

end slope_of_line_l9_9400


namespace equivalent_conditions_l9_9936

theorem equivalent_conditions 
  (f : ℕ+ → ℕ+)
  (H1 : ∀ (m n : ℕ+), m ≤ n → (f m + n) ∣ (f n + m))
  (H2 : ∀ (m n : ℕ+), m ≥ n → (f m + n) ∣ (f n + m)) :
  (∀ (m n : ℕ+), m ≤ n → (f m + n) ∣ (f n + m)) ↔ 
  (∀ (m n : ℕ+), m ≥ n → (f m + n) ∣ (f n + m)) :=
sorry

end equivalent_conditions_l9_9936


namespace goldbach_10000_l9_9730

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

theorem goldbach_10000 :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p q : ℕ), (p, q) ∈ S → is_prime p ∧ is_prime q ∧ p + q = 10000) ∧ S.card > 3 :=
sorry

end goldbach_10000_l9_9730


namespace michael_ratio_zero_l9_9876

theorem michael_ratio_zero (M : ℕ) (h1: M ≤ 60) (h2: 15 = (60 - M) / 2 - 15) : M = 0 := by
  sorry 

end michael_ratio_zero_l9_9876


namespace marbles_remainder_l9_9057

theorem marbles_remainder (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 6) : (r + p) % 8 = 3 :=
by sorry

end marbles_remainder_l9_9057


namespace real_roots_condition_l9_9408

theorem real_roots_condition (k m : ℝ) (h : m ≠ 0) : (∃ x : ℝ, x^2 + k * x + m = 0) ↔ (m ≤ k^2 / 4) :=
by
  sorry

end real_roots_condition_l9_9408


namespace number_of_zeros_of_g_l9_9432

open Real

noncomputable def g (x : ℝ) : ℝ := cos (π * log x + x)

theorem number_of_zeros_of_g : ¬ ∃ (x : ℝ), 1 < x ∧ x < exp 2 ∧ g x = 0 :=
sorry

end number_of_zeros_of_g_l9_9432


namespace selling_price_is_correct_l9_9320

-- Define the constants used in the problem
noncomputable def cost_price : ℝ := 540
noncomputable def markup_percentage : ℝ := 0.15
noncomputable def discount_percentage : ℝ := 26.570048309178745 / 100

-- Define the conditions in the problem
noncomputable def marked_price : ℝ := cost_price * (1 + markup_percentage)
noncomputable def discount_amount : ℝ := marked_price * discount_percentage
noncomputable def selling_price : ℝ := marked_price - discount_amount

-- Theorem stating the problem
theorem selling_price_is_correct : selling_price = 456 := by 
  sorry

end selling_price_is_correct_l9_9320


namespace probability_no_3by3_red_grid_correct_l9_9426

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l9_9426


namespace gcd_example_l9_9470

theorem gcd_example : Nat.gcd (101^11 + 1) (101^11 + 101^3 + 1) = 1 := by
  sorry

end gcd_example_l9_9470


namespace max_sides_13_eq_13_max_sides_1950_eq_1950_l9_9891

noncomputable def max_sides (n : ℕ) : ℕ := n

theorem max_sides_13_eq_13 : max_sides 13 = 13 :=
by {
  sorry
}

theorem max_sides_1950_eq_1950 : max_sides 1950 = 1950 :=
by {
  sorry
}

end max_sides_13_eq_13_max_sides_1950_eq_1950_l9_9891


namespace tenth_term_arith_seq_l9_9662

variable (a1 d : Int) -- Initial term and common difference
variable (n : Nat) -- nth term

-- Definition of the nth term in an arithmetic sequence
def arithmeticSeq (a1 d : Int) (n : Nat) : Int :=
  a1 + (n - 1) * d

-- Specific values for the problem
def a_10 : Int :=
  arithmeticSeq 10 (-3) 10

-- The theorem we want to prove
theorem tenth_term_arith_seq : a_10 = -17 := by
  sorry

end tenth_term_arith_seq_l9_9662


namespace find_constant_a_l9_9112

theorem find_constant_a :
  (∃ (a : ℝ), a > 0 ∧ (a + 2 * a + 3 * a + 4 * a = 1)) →
  ∃ (a : ℝ), a = 1 / 10 :=
sorry

end find_constant_a_l9_9112


namespace part1_quantity_of_vegetables_part2_functional_relationship_part3_min_vegetable_a_l9_9172

/-- Part 1: Quantities of vegetables A and B wholesaled. -/
theorem part1_quantity_of_vegetables (x y : ℝ) 
  (h1 : x + y = 40) 
  (h2 : 4.8 * x + 4 * y = 180) : 
  x = 25 ∧ y = 15 :=
sorry

/-- Part 2: Functional relationship between m and n. -/
theorem part2_functional_relationship (n m : ℝ) 
  (h : n ≤ 80) 
  (h2 : m = 4.8 * n + 4 * (80 - n)) : 
  m = 0.8 * n + 320 :=
sorry

/-- Part 3: Minimum amount of vegetable A to ensure profit of at least 176 yuan -/
theorem part3_min_vegetable_a (n : ℝ) 
  (h : 0.8 * n + 128 ≥ 176) : 
  n ≥ 60 :=
sorry

end part1_quantity_of_vegetables_part2_functional_relationship_part3_min_vegetable_a_l9_9172


namespace intersection_of_A_and_B_l9_9897

noncomputable def A : Set ℕ := {x | x > 0 ∧ x ≤ 3}
def B : Set ℕ := {x | 0 < x ∧ x < 4}

theorem intersection_of_A_and_B : 
  A ∩ B = {1, 2, 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l9_9897


namespace at_least_one_boy_selected_l9_9351

-- Define the number of boys and girls
def boys : ℕ := 6
def girls : ℕ := 2

-- Define the total group and the total selected
def total_people : ℕ := boys + girls
def selected_people : ℕ := 3

-- Statement: In any selection of 3 people from the group, the selection contains at least one boy
theorem at_least_one_boy_selected :
  ∀ (selection : Finset ℕ), selection.card = selected_people → selection.card > girls :=
sorry

end at_least_one_boy_selected_l9_9351


namespace quadratic_roots_unique_l9_9039

theorem quadratic_roots_unique (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ (x = 2 * p ∨ x = p + q)) →
  p = 2 / 3 ∧ q = -8 / 3 :=
by
  sorry

end quadratic_roots_unique_l9_9039


namespace gcd_8917_4273_l9_9981

theorem gcd_8917_4273 : Int.gcd 8917 4273 = 1 :=
by
  sorry

end gcd_8917_4273_l9_9981


namespace polynomials_equal_at_all_x_l9_9784

variable {R : Type} [CommRing R]

def f (a_5 a_4 a_3 a_2 a_1 a_0 : R) (x : R) := a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
def g (b_3 b_2 b_1 b_0 : R) (x : R) := b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0
def h (c_2 c_1 c_0 : R) (x : R) := c_2 * x^2 + c_1 * x + c_0

theorem polynomials_equal_at_all_x 
    (a_5 a_4 a_3 a_2 a_1 a_0 b_3 b_2 b_1 b_0 c_2 c_1 c_0 : ℤ)
    (bound_a : ∀ i ∈ [a_5, a_4, a_3, a_2, a_1, a_0], |i| ≤ 4)
    (bound_b : ∀ i ∈ [b_3, b_2, b_1, b_0], |i| ≤ 1)
    (bound_c : ∀ i ∈ [c_2, c_1, c_0], |i| ≤ 1)
    (H : f a_5 a_4 a_3 a_2 a_1 a_0 10 = g b_3 b_2 b_1 b_0 10 * h c_2 c_1 c_0 10) :
    ∀ x, f a_5 a_4 a_3 a_2 a_1 a_0 x = g b_3 b_2 b_1 b_0 x * h c_2 c_1 c_0 x := by
  sorry

end polynomials_equal_at_all_x_l9_9784


namespace find_second_number_l9_9253

theorem find_second_number 
  (h1 : (20 + 40 + 60) / 3 = (10 + x + 45) / 3 + 5) :
  x = 50 :=
sorry

end find_second_number_l9_9253


namespace commutativity_l9_9717

universe u

variable {M : Type u} [Nonempty M]
variable (star : M → M → M)

axiom star_assoc_right {a b : M} : (star (star a b) b) = a
axiom star_assoc_left {a b : M} : star a (star a b) = b

theorem commutativity (a b : M) : star a b = star b a :=
by sorry

end commutativity_l9_9717


namespace johns_final_amount_l9_9500

def initial_amount : ℝ := 45.7
def deposit_amount : ℝ := 18.6
def withdrawal_amount : ℝ := 20.5

theorem johns_final_amount : initial_amount + deposit_amount - withdrawal_amount = 43.8 :=
by
  sorry

end johns_final_amount_l9_9500


namespace greatest_integer_radius_of_circle_l9_9872

theorem greatest_integer_radius_of_circle (r : ℕ) (A : ℝ) (hA : A < 80 * Real.pi) :
  r <= 8 ∧ r * r < 80 :=
sorry

end greatest_integer_radius_of_circle_l9_9872


namespace necessary_but_not_sufficient_condition_for_ellipse_l9_9996

theorem necessary_but_not_sufficient_condition_for_ellipse (m : ℝ) :
  (2 < m ∧ m < 6) ↔ ((∃ m, 2 < m ∧ m < 6 ∧ m ≠ 4) ∧ (∀ m, (2 < m ∧ m < 6) → ¬(m = 4))) := 
sorry

end necessary_but_not_sufficient_condition_for_ellipse_l9_9996


namespace fans_attended_show_l9_9813

-- Definitions from the conditions
def total_seats : ℕ := 60000
def sold_percentage : ℝ := 0.75
def fans_stayed_home : ℕ := 5000

-- The proof statement
theorem fans_attended_show :
  let sold_seats := sold_percentage * total_seats
  let fans_attended := sold_seats - fans_stayed_home
  fans_attended = 40000 :=
by
  -- Auto-generated proof placeholder.
  sorry

end fans_attended_show_l9_9813


namespace rectangle_length_l9_9227

theorem rectangle_length (P B L : ℝ) (h1 : P = 600) (h2 : B = 200) (h3 : P = 2 * (L + B)) : L = 100 :=
by
  sorry

end rectangle_length_l9_9227


namespace Eric_rent_days_l9_9622

-- Define the conditions given in the problem
def daily_rate := 50.00
def rate_14_days := 500.00
def total_cost := 800.00

-- State the problem as a theorem in Lean
theorem Eric_rent_days : ∀ (d : ℕ), (d : ℕ) = 20 :=
by
  sorry

end Eric_rent_days_l9_9622


namespace abc_eq_ab_bc_ca_l9_9012

variable {u v w A B C : ℝ}
variable (Huvw : u * v * w = 1)
variable (HA : A = u * v + u + 1)
variable (HB : B = v * w + v + 1)
variable (HC : C = w * u + w + 1)

theorem abc_eq_ab_bc_ca 
  (Huvw : u * v * w = 1)
  (HA : A = u * v + u + 1)
  (HB : B = v * w + v + 1)
  (HC : C = w * u + w + 1) : 
  A * B * C = A * B + B * C + C * A := 
by
  sorry

end abc_eq_ab_bc_ca_l9_9012


namespace parabola_coefficients_sum_l9_9128

theorem parabola_coefficients_sum (a b c : ℝ)
  (h_eqn : ∀ y, (-1) = a * y^2 + b * y + c)
  (h_vertex : (-1, -10) = (-a/(2*a), (4*a*c - b^2)/(4*a)))
  (h_pass_point : 0 = a * (-9)^2 + b * (-9) + c) 
  : a + b + c = 120 := 
sorry

end parabola_coefficients_sum_l9_9128


namespace find_y_when_x_is_4_l9_9859

variables (x y : ℕ)
def inversely_proportional (C : ℕ) (x y : ℕ) : Prop := x * y = C

theorem find_y_when_x_is_4 :
  inversely_proportional 240 x y → x = 4 → y = 60 :=
by
  sorry

end find_y_when_x_is_4_l9_9859


namespace notebook_cost_l9_9752

theorem notebook_cost (s n c : ℕ) (h1 : s > 20) (h2 : n > 2) (h3 : c > 2 * n) (h4 : s * c * n = 4515) : c = 35 :=
sorry

end notebook_cost_l9_9752


namespace books_and_games_left_to_experience_l9_9269

def booksLeft (B_total B_read : Nat) : Nat := B_total - B_read
def gamesLeft (G_total G_played : Nat) : Nat := G_total - G_played
def totalLeft (B_total B_read G_total G_played : Nat) : Nat := booksLeft B_total B_read + gamesLeft G_total G_played

theorem books_and_games_left_to_experience :
  totalLeft 150 74 50 17 = 109 := by
  sorry

end books_and_games_left_to_experience_l9_9269


namespace isosceles_triangle_area_48_l9_9745

noncomputable def isosceles_triangle_area (b h s : ℝ) : ℝ :=
  (1 / 2) * (2 * b) * h

theorem isosceles_triangle_area_48 :
  ∀ (b s : ℝ),
  b ^ 2 + 8 ^ 2 = s ^ 2 ∧ s + b = 16 →
  isosceles_triangle_area b 8 s = 48 :=
by
  intros b s h
  unfold isosceles_triangle_area
  sorry

end isosceles_triangle_area_48_l9_9745


namespace remainder_333_pow_333_mod_11_l9_9943

theorem remainder_333_pow_333_mod_11 : (333 ^ 333) % 11 = 5 := by
  sorry

end remainder_333_pow_333_mod_11_l9_9943


namespace derivative_correct_l9_9880

noncomputable def derivative_of_composite_function (x : ℝ) : Prop :=
  let y := (5 * x - 3) ^ 3
  let dy_dx := 3 * (5 * x - 3) ^ 2 * 5
  dy_dx = 15 * (5 * x - 3) ^ 2

theorem derivative_correct (x : ℝ) : derivative_of_composite_function x :=
by
  sorry

end derivative_correct_l9_9880


namespace javier_fraction_to_anna_zero_l9_9838

-- Variables
variable (l : ℕ) -- Lee's initial sticker count
variable (j : ℕ) -- Javier's initial sticker count
variable (a : ℕ) -- Anna's initial sticker count

-- Initial conditions
def conditions (l j a : ℕ) : Prop :=
  j = 4 * a ∧ a = 3 * l

-- Javier's final stickers count
def final_javier_stickers (ja : ℕ) (j : ℕ) : ℕ :=
  ja

-- Anna's final stickers count (af = final Anna's stickers)
def final_anna_stickers (af : ℕ) : ℕ :=
  af

-- Lee's final stickers count (lf = final Lee's stickers)
def final_lee_stickers (lf : ℕ) : ℕ :=
  lf

-- Final distribution requirements
def final_distribution (ja af lf : ℕ) : Prop :=
  ja = 2 * af ∧ ja = 3 * lf

-- Correct answer, fraction of stickers given to Anna
def fraction_given_to_anna (j ja : ℕ) : ℚ :=
  ((j - ja) : ℚ) / (j : ℚ)

-- Lean theorem statement to prove
theorem javier_fraction_to_anna_zero
  (l j a ja af lf : ℕ)
  (h_cond : conditions l j a)
  (h_final : final_distribution ja af lf) :
  fraction_given_to_anna j ja = 0 :=
by sorry

end javier_fraction_to_anna_zero_l9_9838


namespace inequality_inequality_l9_9502

theorem inequality_inequality (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by sorry

end inequality_inequality_l9_9502


namespace required_volume_proof_l9_9556

-- Defining the conditions
def initial_volume : ℝ := 60
def initial_concentration : ℝ := 0.10
def final_concentration : ℝ := 0.15

-- Defining the equation
def required_volume (V : ℝ) : Prop :=
  (initial_concentration * initial_volume + V = final_concentration * (initial_volume + V))

-- Stating the proof problem
theorem required_volume_proof :
  ∃ V : ℝ, required_volume V ∧ V = 3 / 0.85 :=
by {
  -- Proof skipped
  sorry
}

end required_volume_proof_l9_9556


namespace sum_of_first_9000_terms_of_geometric_sequence_l9_9938

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end sum_of_first_9000_terms_of_geometric_sequence_l9_9938


namespace find_d_l9_9863

theorem find_d (m a b d : ℕ) 
(hm : 0 < m) 
(ha : m^2 < a ∧ a < m^2 + m) 
(hb : m^2 < b ∧ b < m^2 + m) 
(hab : a ≠ b)
(hd : m^2 < d ∧ d < m^2 + m ∧ d ∣ (a * b)) : 
d = a ∨ d = b :=
sorry

end find_d_l9_9863


namespace lunch_break_duration_l9_9200

theorem lunch_break_duration
  (p h L : ℝ)
  (monday_eq : (9 - L) * (p + h) = 0.4)
  (tuesday_eq : (8 - L) * h = 0.33)
  (wednesday_eq : (12 - L) * p = 0.27) :
  L = 7.0 ∨ L * 60 = 420 :=
by
  sorry

end lunch_break_duration_l9_9200


namespace expression_value_l9_9911

theorem expression_value :
    (2.502 + 0.064)^2 - ((2.502 - 0.064)^2) / (2.502 * 0.064) = 4.002 :=
by
  -- the proof goes here
  sorry

end expression_value_l9_9911


namespace toys_produced_per_week_l9_9156

theorem toys_produced_per_week (daily_production : ℕ) (work_days_per_week : ℕ) (total_production : ℕ) :
  daily_production = 680 ∧ work_days_per_week = 5 → total_production = 3400 := by
  sorry

end toys_produced_per_week_l9_9156


namespace approximate_roots_l9_9100

noncomputable def f (x : ℝ) : ℝ := 0.3 * x^3 - 2 * x^2 - 0.2 * x + 0.5

theorem approximate_roots : 
  ∃ x₁ x₂ x₃ : ℝ, 
    (f x₁ = 0 ∧ |x₁ + 0.4| < 0.1) ∧ 
    (f x₂ = 0 ∧ |x₂ - 0.5| < 0.1) ∧ 
    (f x₃ = 0 ∧ |x₃ - 2.6| < 0.1) :=
by
  sorry

end approximate_roots_l9_9100


namespace num_valid_lists_l9_9167

-- Define a predicate for a list to satisfy the given constraints
def valid_list (l : List ℕ) : Prop :=
  l = List.range' 1 12 ∧ ∀ i, 1 < i ∧ i ≤ 12 → (l.indexOf (l.get! (i - 1) + 1) < i - 1 ∨ l.indexOf (l.get! (i - 1) - 1) < i - 1) ∧ ¬(l.indexOf (l.get! (i - 1) + 1) < i - 1 ∧ l.indexOf (l.get! (i - 1) - 1) < i - 1)

-- Prove that there is exactly one valid list of such nature
theorem num_valid_lists : ∃! l : List ℕ, valid_list l :=
  sorry

end num_valid_lists_l9_9167


namespace trader_profit_percent_equal_eight_l9_9687

-- Defining the initial conditions
def original_price (P : ℝ) := P
def purchased_price (P : ℝ) := 0.60 * original_price P
def selling_price (P : ℝ) := 1.80 * purchased_price P

-- Statement to be proved
theorem trader_profit_percent_equal_eight (P : ℝ) (h : P > 0) :
  ((selling_price P - original_price P) / original_price P) * 100 = 8 :=
by
  sorry

end trader_profit_percent_equal_eight_l9_9687


namespace number_of_questions_in_test_l9_9511

-- Definitions based on the conditions:
def marks_per_question : ℕ := 2
def jose_wrong_questions : ℕ := 5  -- number of questions Jose got wrong
def total_combined_score : ℕ := 210  -- total score of Meghan, Jose, and Alisson combined

-- Let A be Alisson's score
variables (A Jose Meghan : ℕ)

-- Conditions
axiom joe_more_than_alisson : Jose = A + 40
axiom megh_less_than_jose : Meghan = Jose - 20
axiom combined_scores : A + Jose + Meghan = total_combined_score

-- Function to compute the total possible score for Jose without wrong answers:
noncomputable def jose_improvement_score : ℕ := Jose + (jose_wrong_questions * marks_per_question)

-- Proof problem statement
theorem number_of_questions_in_test :
  (jose_improvement_score Jose) / marks_per_question = 50 :=
by
  -- Sorry is used here to indicate that the proof is omitted.
  sorry

end number_of_questions_in_test_l9_9511


namespace xyz_range_l9_9179

theorem xyz_range (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x^2 + y^2 + z^2 = 3) : 
  -1 ≤ x * y * z ∧ x * y * z ≤ 5 / 27 := 
sorry

end xyz_range_l9_9179


namespace remainder_when_divided_l9_9332

theorem remainder_when_divided (k : ℕ) (h_pos : 0 < k) (h_rem : 80 % k = 8) : 150 % (k^2) = 69 := by 
  sorry

end remainder_when_divided_l9_9332


namespace customers_stayed_behind_l9_9530

theorem customers_stayed_behind : ∃ x : ℕ, (x + (x + 5) = 11) ∧ x = 3 := by
  sorry

end customers_stayed_behind_l9_9530


namespace find_varphi_l9_9786

theorem find_varphi (φ : ℝ) (h1 : 0 < φ ∧ φ < 2 * Real.pi) 
    (h2 : ∀ x, x = 2 → Real.sin (Real.pi * x + φ) = 1) : 
    φ = Real.pi / 2 :=
-- The following is left as a proof placeholder
sorry

end find_varphi_l9_9786


namespace max_candy_leftover_l9_9272

theorem max_candy_leftover (x : ℕ) : (∃ k : ℕ, x = 12 * k + 11) → (x % 12 = 11) :=
by
  sorry

end max_candy_leftover_l9_9272


namespace bold_o_lit_cells_l9_9805

-- Define the conditions
def grid_size : ℕ := 5
def original_o_lit_cells : ℕ := 12 -- Number of cells lit in the original 'o'
def additional_lit_cells : ℕ := 12 -- Additional cells lit in the bold 'o'

-- Define the property to be proved
theorem bold_o_lit_cells : (original_o_lit_cells + additional_lit_cells) = 24 :=
by
  -- computation skipped
  sorry

end bold_o_lit_cells_l9_9805


namespace find_result_of_adding_8_l9_9118

theorem find_result_of_adding_8 (x : ℕ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end find_result_of_adding_8_l9_9118


namespace find_n_l9_9027

theorem find_n (n : ℕ) (m : ℕ) (h_pos_n : n > 0) (h_pos_m : m > 0) (h_div : (2^n - 1) ∣ (m^2 + 81)) : 
  ∃ k : ℕ, n = 2^k := 
sorry

end find_n_l9_9027


namespace quadratic_coefficients_l9_9158

theorem quadratic_coefficients :
  ∀ x : ℝ, 3 * x^2 = 5 * x - 1 → (∃ a b c : ℝ, a = 3 ∧ b = -5 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x h
  use 3, -5, 1
  sorry

end quadratic_coefficients_l9_9158


namespace subtracted_amount_l9_9198

theorem subtracted_amount (A N : ℝ) (h₁ : N = 200) (h₂ : 0.95 * N - A = 178) : A = 12 :=
by
  sorry

end subtracted_amount_l9_9198


namespace jaden_time_difference_l9_9561

-- Define the conditions as hypotheses
def jaden_time_as_girl (distance : ℕ) (time : ℕ) : Prop :=
  distance = 20 ∧ time = 240

def jaden_time_as_woman (distance : ℕ) (time : ℕ) : Prop :=
  distance = 8 ∧ time = 240

-- Define the proof problem
theorem jaden_time_difference
  (d_girl t_girl d_woman t_woman : ℕ)
  (H_girl : jaden_time_as_girl d_girl t_girl)
  (H_woman : jaden_time_as_woman d_woman t_woman)
  : (t_woman / d_woman) - (t_girl / d_girl) = 18 :=
by
  sorry

end jaden_time_difference_l9_9561


namespace isosceles_triangle_largest_angle_l9_9953

theorem isosceles_triangle_largest_angle (a b c : ℝ) (h1 : a = b) (h2 : b_angle = 50) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) 
  (h6 : a + b + c = 180) : c ≥ a ∨ c ≥ b → c = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l9_9953


namespace total_trash_cans_paid_for_l9_9164

-- Definitions based on conditions
def trash_cans_on_streets : ℕ := 14
def trash_cans_back_of_stores : ℕ := 2 * trash_cans_on_streets

-- Theorem to prove
theorem total_trash_cans_paid_for : trash_cans_on_streets + trash_cans_back_of_stores = 42 := 
by
  -- proof would go here, but we use sorry since proof is not required
  sorry

end total_trash_cans_paid_for_l9_9164


namespace four_numbers_are_perfect_squares_l9_9706

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem four_numbers_are_perfect_squares (a b c d : ℕ) (h1 : is_perfect_square (a * b * c))
                                                      (h2 : is_perfect_square (a * c * d))
                                                      (h3 : is_perfect_square (b * c * d))
                                                      (h4 : is_perfect_square (a * b * d)) : 
                                                      is_perfect_square a ∧
                                                      is_perfect_square b ∧
                                                      is_perfect_square c ∧
                                                      is_perfect_square d :=
by
  sorry

end four_numbers_are_perfect_squares_l9_9706


namespace gcd_of_A_and_B_l9_9625

theorem gcd_of_A_and_B (A B : ℕ) (h_lcm : Nat.lcm A B = 120) (h_ratio : A * 4 = B * 3) : Nat.gcd A B = 10 :=
sorry

end gcd_of_A_and_B_l9_9625


namespace distance_train_A_when_meeting_l9_9829

noncomputable def distance_traveled_by_train_A : ℝ :=
  let distance := 375
  let time_A := 36
  let time_B := 24
  let speed_A := distance / time_A
  let speed_B := distance / time_B
  let relative_speed := speed_A + speed_B
  let time_meeting := distance / relative_speed
  speed_A * time_meeting

theorem distance_train_A_when_meeting :
  distance_traveled_by_train_A = 150 := by
  sorry

end distance_train_A_when_meeting_l9_9829


namespace apple_distribution_l9_9176

theorem apple_distribution (x : ℕ) (h₁ : 1430 % x = 0) (h₂ : 1430 % (x + 45) = 0) (h₃ : 1430 / x - 1430 / (x + 45) = 9) : 
  1430 / x = 22 :=
by
  sorry

end apple_distribution_l9_9176


namespace sale_price_same_as_original_l9_9725

theorem sale_price_same_as_original (x : ℝ) :
  let increased_price := 1.25 * x
  let sale_price := 0.8 * increased_price
  sale_price = x := 
by
  let increased_price := 1.25 * x
  let sale_price := 0.8 * increased_price
  sorry

end sale_price_same_as_original_l9_9725


namespace negation_equiv_no_solution_l9_9774

-- Definition of there is at least one solution
def at_least_one_solution (P : α → Prop) : Prop := ∃ x, P x

-- Definition of no solution
def no_solution (P : α → Prop) : Prop := ∀ x, ¬ P x

-- Problem statement to prove that the negation of at_least_one_solution is equivalent to no_solution
theorem negation_equiv_no_solution (P : α → Prop) :
  ¬ at_least_one_solution P ↔ no_solution P := 
sorry

end negation_equiv_no_solution_l9_9774


namespace min_even_integers_zero_l9_9411

theorem min_even_integers_zero (x y a b m n : ℤ)
(h1 : x + y = 28) 
(h2 : x + y + a + b = 46) 
(h3 : x + y + a + b + m + n = 64) : 
∃ e, e = 0 :=
by {
  -- The conditions assure the sums of pairs are even including x, y, a, b, m, n.
  sorry
}

end min_even_integers_zero_l9_9411


namespace max_area_triangle_l9_9946

/-- Given two fixed points A and B on the plane with distance 2 between them, 
and a point P moving such that the ratio of distances |PA| / |PB| = sqrt(2), 
prove that the maximum area of triangle PAB is 2 * sqrt(2). -/
theorem max_area_triangle 
  (A B P : EuclideanSpace ℝ (Fin 2)) 
  (hAB : dist A B = 2)
  (h_ratio : dist P A = Real.sqrt 2 * dist P B)
  (h_non_collinear : ¬ ∃ k : ℝ, ∃ l : ℝ, k ≠ l ∧ A = k • B ∧ P = l • B) 
  : ∃ S_max : ℝ, S_max = 2 * Real.sqrt 2 := 
sorry

end max_area_triangle_l9_9946


namespace sin_2017pi_div_3_l9_9437

theorem sin_2017pi_div_3 : Real.sin (2017 * Real.pi / 3) = Real.sqrt 3 / 2 := 
  sorry

end sin_2017pi_div_3_l9_9437


namespace small_trucks_needed_l9_9339

-- Defining the problem's conditions
def total_flour : ℝ := 500
def large_truck_capacity : ℝ := 9.6
def num_large_trucks : ℝ := 40
def small_truck_capacity : ℝ := 4

-- Theorem statement to find the number of small trucks needed
theorem small_trucks_needed : (total_flour - (num_large_trucks * large_truck_capacity)) / small_truck_capacity = (500 - (40 * 9.6)) / 4 :=
by
  sorry

end small_trucks_needed_l9_9339


namespace option_C_correct_l9_9985

theorem option_C_correct (a b : ℝ) : ((a^2 * b)^3) / ((-a * b)^2) = a^4 * b := by
  sorry

end option_C_correct_l9_9985


namespace original_number_exists_l9_9867

theorem original_number_exists :
  ∃ x : ℝ, 10 * x = x + 2.7 ∧ x = 0.3 :=
by {
  sorry
}

end original_number_exists_l9_9867


namespace proof_problem_l9_9843

def intelligentFailRate (r1 r2 r3 : ℚ) : ℚ :=
  1 - r1 * r2 * r3

def phi (p : ℚ) : ℚ :=
  30 * p * (1 - p)^29

def derivativePhi (p : ℚ) : ℚ :=
  30 * (1 - p)^28 * (1 - 30 * p)

def qualifiedPassRate (intelligentPassRate comprehensivePassRate : ℚ) : ℚ :=
  intelligentPassRate * comprehensivePassRate

theorem proof_problem :
  let r1 := (99 : ℚ) / 100
  let r2 := (98 : ℚ) / 99
  let r3 := (97 : ℚ) / 98
  let p0 := (1 : ℚ) / 30
  let comprehensivePassRate := 1 - p0
  let qualifiedRate := qualifiedPassRate (r1 * r2 * r3) comprehensivePassRate
  (intelligentFailRate r1 r2 r3 = 3 / 100) ∧
  (derivativePhi p0 = 0) ∧
  (qualifiedRate < 96 / 100) :=
by
  sorry

end proof_problem_l9_9843


namespace tile_area_l9_9738

-- Define the properties and conditions of the tile

structure Tile where
  sides : Fin 9 → ℝ 
  six_of_length_1 : ∀ i : Fin 6, sides i = 1 
  congruent_quadrilaterals : Fin 3 → Quadrilateral

structure Quadrilateral where
  length : ℝ
  width : ℝ

-- Given the tile structure, calculate the area
noncomputable def area_of_tile (t: Tile) : ℝ := sorry

-- Statement: Prove the area of the tile given the conditions
theorem tile_area (t : Tile) : area_of_tile t = (4 * Real.sqrt 3 / 3) :=
  sorry

end tile_area_l9_9738


namespace frustumViews_l9_9821

-- Define the notion of a frustum
structure Frustum where
  -- You may add necessary geometric properties of a frustum if needed
  
-- Define a function to describe the view of the frustum
def frontView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type
def sideView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type
def topView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type

-- Define the properties of the views
def isCongruentIsoscelesTrapezoid (fig : Type) : Prop := sorry -- Define property for congruent isosceles trapezoid
def isTwoConcentricCircles (fig : Type) : Prop := sorry -- Define property for two concentric circles

-- State the theorem based on the given problem
theorem frustumViews (f : Frustum) :
  isCongruentIsoscelesTrapezoid (frontView f) ∧ 
  isCongruentIsoscelesTrapezoid (sideView f) ∧ 
  isTwoConcentricCircles (topView f) := 
sorry

end frustumViews_l9_9821


namespace express_y_in_terms_of_x_l9_9860

theorem express_y_in_terms_of_x (x y : ℝ) (h : 5 * x + y = 4) : y = 4 - 5 * x :=
by
  /- Proof to be filled in here. -/
  sorry

end express_y_in_terms_of_x_l9_9860


namespace degenerate_ellipse_single_point_c_l9_9096

theorem degenerate_ellipse_single_point_c (c : ℝ) :
  (∀ x y : ℝ, 2 * x^2 + y^2 + 8 * x - 10 * y + c = 0 → x = -2 ∧ y = 5) →
  c = 33 :=
by
  intros h
  sorry

end degenerate_ellipse_single_point_c_l9_9096


namespace g_neg6_eq_neg28_l9_9111

-- Define the given function g
def g (x : ℝ) : ℝ := 2 * x^7 - 3 * x^3 + 4 * x - 8

-- State the main theorem to prove g(-6) = -28 under the given conditions
theorem g_neg6_eq_neg28 (h1 : g 6 = 12) : g (-6) = -28 :=
by
  sorry

end g_neg6_eq_neg28_l9_9111


namespace zero_of_F_when_a_is_zero_range_of_a_if_P_and_Q_l9_9734

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x
noncomputable def g (a x : ℝ) : ℝ := Real.log (x^2 - 2*x + a)
noncomputable def F (a x : ℝ) : ℝ := f a x + g a x

theorem zero_of_F_when_a_is_zero (x : ℝ) : a = 0 → F a x = 0 → x = 3 := by
  sorry

theorem range_of_a_if_P_and_Q (a : ℝ) :
  (∀ x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ), a - 1/x ≤ 0) ∧
  (∀ x : ℝ, (x^2 - 2*x + a) > 0) →
  1 < a ∧ a ≤ 2 := by
  sorry

end zero_of_F_when_a_is_zero_range_of_a_if_P_and_Q_l9_9734


namespace quadratic_root_ratio_l9_9790

theorem quadratic_root_ratio (k : ℝ) (h : ∃ r : ℝ, r ≠ 0 ∧ 3 * r * r = k * r - 12 * r + k ∧ r * r = k + 9 * r - k) : k = 27 :=
sorry

end quadratic_root_ratio_l9_9790


namespace S_2011_value_l9_9093

-- Definitions based on conditions provided in the problem
def arithmetic_seq (a_n : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a_n (n + 1) = a_n n + d

def sum_seq (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
  ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2

-- Problem statement
theorem S_2011_value
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (h_arith : arithmetic_seq a_n)
  (h_sum : sum_seq S_n a_n)
  (h_init : a_n 1 = -2011)
  (h_cond : (S_n 2010) / 2010 - (S_n 2008) / 2008 = 2) :
  S_n 2011 = -2011 := 
sorry

end S_2011_value_l9_9093


namespace x_not_4_17_percent_less_than_z_x_is_8_0032_percent_less_than_z_l9_9199

def y_is_60_percent_greater_than_x (x y : ℝ) : Prop :=
  y = 1.60 * x

def z_is_40_percent_less_than_y (y z : ℝ) : Prop :=
  z = 0.60 * y

theorem x_not_4_17_percent_less_than_z (x y z : ℝ) (h1 : y_is_60_percent_greater_than_x x y) (h2 : z_is_40_percent_less_than_y y z) : 
  x ≠ 0.9583 * z :=
by {
  sorry
}

theorem x_is_8_0032_percent_less_than_z (x y z : ℝ) (h1 : y_is_60_percent_greater_than_x x y) (h2 : z_is_40_percent_less_than_y y z) : 
  x = 0.919968 * z :=
by {
  sorry
}

end x_not_4_17_percent_less_than_z_x_is_8_0032_percent_less_than_z_l9_9199


namespace range_of_a_l9_9830

theorem range_of_a (a : ℝ) : 
  (∀ (x1 : ℝ), ∃ (x2 : ℝ), |x1| = Real.log (a * x2^2 - 4 * x2 + 1)) → (0 ≤ a) :=
by
  sorry

end range_of_a_l9_9830


namespace trigonometric_identity_l9_9340

variable (α : Real)

theorem trigonometric_identity 
  (h : Real.sin (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (π / 3 - α) = Real.sqrt 3 / 3 :=
sorry

end trigonometric_identity_l9_9340


namespace exponent_equivalence_l9_9291

open Real

theorem exponent_equivalence (a : ℝ) (h : a > 0) : 
  (a^2 / (sqrt a * a^(2/3))) = a^(5/6) :=
  sorry

end exponent_equivalence_l9_9291


namespace prob_point_closer_to_six_than_zero_l9_9677

theorem prob_point_closer_to_six_than_zero : 
  let interval_start := 0
  let interval_end := 7
  let closer_to_six := fun x => x > ((interval_start + 6) / 2)
  let total_length := interval_end - interval_start
  let length_closer_to_six := interval_end - (interval_start + 6) / 2
  total_length > 0 -> length_closer_to_six / total_length = 4 / 7 :=
by
  sorry

end prob_point_closer_to_six_than_zero_l9_9677


namespace find_m_value_l9_9456

/-- 
If the function y = (m + 1)x^(m^2 + 3m + 4) is a quadratic function, 
then the value of m is -2.
--/
theorem find_m_value 
  (m : ℝ)
  (h1 : m^2 + 3 * m + 4 = 2)
  (h2 : m + 1 ≠ 0) : 
  m = -2 := 
sorry

end find_m_value_l9_9456


namespace min_employees_birthday_Wednesday_l9_9248

theorem min_employees_birthday_Wednesday (W D : ℕ) (h_eq : W + 6 * D = 50) (h_gt : W > D) : W = 8 :=
sorry

end min_employees_birthday_Wednesday_l9_9248


namespace factorize_expression_l9_9139

theorem factorize_expression (x : ℝ) : x^3 - 2 * x^2 + x = x * (x - 1)^2 :=
by sorry

end factorize_expression_l9_9139


namespace marble_count_calculation_l9_9064

theorem marble_count_calculation (y b g : ℕ) (x : ℕ)
  (h1 : y = 2 * x)
  (h2 : b = 3 * x)
  (h3 : g = 4 * x)
  (h4 : g = 32) : y + b + g = 72 :=
by
  sorry

end marble_count_calculation_l9_9064


namespace range_of_a_l9_9425

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 3| - |x + 1| ≤ a^2 - 3 * a) ↔ a ≤ -1 ∨ 4 ≤ a := 
sorry

end range_of_a_l9_9425


namespace number_divisible_by_11_l9_9360

theorem number_divisible_by_11 (N Q : ℕ) (h1 : N = 11 * Q) (h2 : Q + N + 11 = 71) : N = 55 :=
by
  sorry

end number_divisible_by_11_l9_9360


namespace monthly_salary_equals_l9_9698

-- Define the base salary
def base_salary : ℝ := 1600

-- Define the commission rate
def commission_rate : ℝ := 0.04

-- Define the sales amount for which the salaries are equal
def sales_amount : ℝ := 5000

-- Define the total earnings with a base salary and commission for 5000 worth of sales
def total_earnings : ℝ := base_salary + (commission_rate * sales_amount)

-- Define the monthly salary from Furniture by Design
def monthly_salary : ℝ := 1800

-- Prove that the monthly salary S is equal to 1800
theorem monthly_salary_equals :
  total_earnings = monthly_salary :=
by
  -- The proof is skipped with sorry.
  sorry

end monthly_salary_equals_l9_9698


namespace simplify_sum_l9_9674

theorem simplify_sum : 
  (-1: ℤ)^(2010) + (-1: ℤ)^(2011) + (1: ℤ)^(2012) + (-1: ℤ)^(2013) = -2 := by
  sorry

end simplify_sum_l9_9674


namespace find_ax6_by6_l9_9536

variable {a b x y : ℝ}

theorem find_ax6_by6
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 12)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 80) :
  a * x^6 + b * y^6 = 1531.25 :=
sorry

end find_ax6_by6_l9_9536


namespace average_coins_collected_per_day_l9_9615

noncomputable def average_coins (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (a + (a + (n - 1) * d)) / 2

theorem average_coins_collected_per_day :
  average_coins 10 5 7 = 25 := by
  sorry

end average_coins_collected_per_day_l9_9615


namespace dozen_pen_cost_l9_9458

-- Definitions based on the conditions
def cost_of_pen (x : ℝ) : ℝ := 5 * x
def cost_of_pencil (x : ℝ) : ℝ := x
def total_cost (x : ℝ) (y : ℝ) : ℝ := 3 * cost_of_pen x + y * cost_of_pencil x

open Classical
noncomputable def cost_dozen_pens (x : ℝ) : ℝ := 12 * cost_of_pen x

theorem dozen_pen_cost (x y : ℝ) (h : total_cost x y = 150) : cost_dozen_pens x = 60 * x :=
by
  sorry

end dozen_pen_cost_l9_9458


namespace price_of_sugar_and_salt_l9_9343

theorem price_of_sugar_and_salt:
  (∀ (sugar_price salt_price : ℝ), 2 * sugar_price + 5 * salt_price = 5.50 ∧ sugar_price = 1.50 →
  3 * sugar_price + salt_price = 5) := 
by 
  sorry

end price_of_sugar_and_salt_l9_9343


namespace intersection_points_l9_9839

def curve (x y : ℝ) : Prop := x^2 + y^2 = 1
def line (x y : ℝ) : Prop := y = x + 1

theorem intersection_points :
  {p : ℝ × ℝ | curve p.1 p.2 ∧ line p.1 p.2} = {(-1, 0), (0, 1)} :=
by 
  sorry

end intersection_points_l9_9839


namespace no_cubic_solution_l9_9594

theorem no_cubic_solution (t : ℤ) : ¬ ∃ k : ℤ, (7 * t + 3 = k ^ 3) := by
  sorry

end no_cubic_solution_l9_9594


namespace star_vertex_angle_l9_9002

-- Defining a function that calculates the star vertex angle for odd n-sided concave regular polygon
theorem star_vertex_angle (n : ℕ) (hn_odd : n % 2 = 1) (hn_gt3 : 3 < n) : 
  (180 - 360 / n) = (n - 2) * 180 / n := 
sorry

end star_vertex_angle_l9_9002


namespace determinant_problem_l9_9914

theorem determinant_problem (a b c d : ℝ)
  (h : Matrix.det ![![a, b], ![c, d]] = 4) :
  Matrix.det ![![a, 5*a + 3*b], ![c, 5*c + 3*d]] = 12 := by
  sorry

end determinant_problem_l9_9914


namespace factorization_example_l9_9168

theorem factorization_example (a b : ℕ) : (a - 2*b)^2 = a^2 - 4*a*b + 4*b^2 := 
by sorry

end factorization_example_l9_9168


namespace rectangular_field_length_l9_9461

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

noncomputable def length_rectangle (area width : ℝ) : ℝ :=
  area / width

theorem rectangular_field_length (base height width : ℝ) (h_base : base = 7.2) (h_height : height = 7) (h_width : width = 4) :
  length_rectangle (area_triangle base height) width = 6.3 :=
by
  -- sorry would be replaced by the actual proof.
  sorry

end rectangular_field_length_l9_9461


namespace six_cube_2d_faces_count_l9_9990

open BigOperators

theorem six_cube_2d_faces_count :
    let vertices := 64
    let edges_1d := 192
    let edges_2d := 240
    let small_cubes := 46656
    let faces_per_plane := 36
    let planes_count := 15 * 7^4
    faces_per_plane * planes_count = 1296150 := by
  sorry

end six_cube_2d_faces_count_l9_9990


namespace no_solution_set_1_2_4_l9_9773

theorem no_solution_set_1_2_4 
  (f : ℝ → ℝ) 
  (hf : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c)
  (t : ℝ) : ¬ ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f (|x1 - t|) = 0 ∧ f (|x2 - t|) = 0 ∧ f (|x3 - t|) = 0 ∧ (x1 = 1 ∧ x2 = 2 ∧ x3 = 4) := 
sorry

end no_solution_set_1_2_4_l9_9773


namespace additional_slow_workers_needed_l9_9807

-- Definitions based on conditions
def production_per_worker_fast (m : ℕ) (n : ℕ) (a : ℕ) : ℚ := m / (n * a)
def production_per_worker_slow (m : ℕ) (n : ℕ) (b : ℕ) : ℚ := m / (n * b)

def required_daily_production (p : ℕ) (q : ℕ) : ℚ := p / q

def contribution_fast_workers (m : ℕ) (n : ℕ) (a : ℕ) (c : ℕ) : ℚ :=
  (m * c) / (n * a)

def remaining_production (p : ℕ) (q : ℕ) (m : ℕ) (n : ℕ) (a : ℕ) (c : ℕ) : ℚ :=
  (p / q) - ((m * c) / (n * a))

def required_slow_workers (p : ℕ) (q : ℕ) (m : ℕ) (n : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) : ℚ :=
  ((p * n * a - q * m * c) * b) / (q * m * a)

theorem additional_slow_workers_needed (m n a b p q c : ℕ) :
  required_slow_workers p q m n a b c = ((p * n * a - q * m * c) * b) / (q * m * a) := by
  sorry

end additional_slow_workers_needed_l9_9807


namespace symmetrical_circle_l9_9663

-- Defining the given circle's equation
def given_circle_eq (x y: ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Defining the equation of the symmetrical circle
def symmetrical_circle_eq (x y: ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Proving the symmetry property
theorem symmetrical_circle (x y : ℝ) : 
  (given_circle_eq x y) → (symmetrical_circle_eq (-x) (-y)) :=
by
  sorry

end symmetrical_circle_l9_9663


namespace number_of_boys_in_biology_class_l9_9160

variable (B G : ℕ) (PhysicsClass BiologyClass : ℕ)

theorem number_of_boys_in_biology_class
  (h1 : G = 3 * B)
  (h2 : PhysicsClass = 200)
  (h3 : BiologyClass = PhysicsClass / 2)
  (h4 : BiologyClass = B + G) :
  B = 25 := by
  sorry

end number_of_boys_in_biology_class_l9_9160


namespace tank_capacity_correctness_l9_9711

noncomputable def tankCapacity : ℝ := 77.65

theorem tank_capacity_correctness (T : ℝ) 
  (h_initial: T * (5 / 8) + 11 = T * (23 / 30)) : 
  T = tankCapacity := 
by
  sorry

end tank_capacity_correctness_l9_9711


namespace angle_is_40_l9_9558

theorem angle_is_40 (x : ℝ) 
  : (180 - x = 2 * (90 - x) + 40) → x = 40 :=
by
  sorry

end angle_is_40_l9_9558


namespace intersection_of_sets_l9_9305

variable (M : Set ℤ) (N : Set ℤ)

theorem intersection_of_sets :
  M = {-2, -1, 0, 1, 2} →
  N = {x | x ≥ 3 ∨ x ≤ -2} →
  M ∩ N = {-2} :=
by
  intros hM hN
  sorry

end intersection_of_sets_l9_9305


namespace transform_fraction_l9_9740

theorem transform_fraction (x : ℝ) (h₁ : x ≠ 3) : - (1 / (3 - x)) = (1 / (x - 3)) := 
    sorry

end transform_fraction_l9_9740


namespace find_theta_even_fn_l9_9298

noncomputable def f (x θ : ℝ) := Real.sin (x + θ) + Real.cos (x + θ)

theorem find_theta_even_fn (θ : ℝ) (hθ: 0 ≤ θ ∧ θ ≤ π / 2) 
  (h: ∀ x : ℝ, f x θ = f (-x) θ) : θ = π / 4 :=
by sorry

end find_theta_even_fn_l9_9298


namespace longest_side_of_triangle_l9_9639

theorem longest_side_of_triangle (y : ℝ) 
  (side1 : ℝ := 8) (side2 : ℝ := y + 5) (side3 : ℝ := 3 * y + 2)
  (h_perimeter : side1 + side2 + side3 = 47) :
  max side1 (max side2 side3) = 26 :=
sorry

end longest_side_of_triangle_l9_9639


namespace simplify_140_210_l9_9538

noncomputable def simplify_fraction (num den : Nat) : Nat × Nat :=
  let d := Nat.gcd num den
  (num / d, den / d)

theorem simplify_140_210 :
  simplify_fraction 140 210 = (2, 3) :=
by
  have p140 : 140 = 2^2 * 5 * 7 := by rfl
  have p210 : 210 = 2 * 3 * 5 * 7 := by rfl
  sorry

end simplify_140_210_l9_9538


namespace keith_attended_games_l9_9051

def total_games : ℕ := 8
def missed_games : ℕ := 4
def attended_games (total : ℕ) (missed : ℕ) : ℕ := total - missed

theorem keith_attended_games : attended_games total_games missed_games = 4 := by
  sorry

end keith_attended_games_l9_9051


namespace combined_tax_rate_l9_9309

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (h1 : Mindy_income = 4 * Mork_income) :
  let Mork_tax := 0.45 * Mork_income;
  let Mindy_tax := 0.15 * Mindy_income;
  let combined_tax := Mork_tax + Mindy_tax;
  let combined_income := Mork_income + Mindy_income;
  combined_tax / combined_income * 100 = 21 := 
by
  sorry

end combined_tax_rate_l9_9309


namespace three_zeros_of_f_l9_9692

noncomputable def f (a x b : ℝ) : ℝ := (1/2) * a * x^2 - (a^2 + a + 2) * x + (2 * a + 2) * (Real.log x) + b

theorem three_zeros_of_f (a b : ℝ) (h1 : a > 3) (h2 : a^2 + a + 1 < b) (h3 : b < 2 * a^2 - 2 * a + 2) : 
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 b = 0 ∧ f a x2 b = 0 ∧ f a x3 b = 0 :=
by
  sorry

end three_zeros_of_f_l9_9692


namespace sum_of_interior_angles_l9_9977

noncomputable def exterior_angle (n : ℕ) := 360 / n

theorem sum_of_interior_angles (n : ℕ) (h : exterior_angle n = 45) :
  180 * (n - 2) = 1080 :=
by
  sorry

end sum_of_interior_angles_l9_9977


namespace math_problem_l9_9783

-- Statement of the theorem
theorem math_problem :
  (0.66)^3 - ((0.1)^3 / ((0.66)^2 + 0.066 + (0.1)^2)) = 0.3612 :=
by
  sorry -- Proof is not required

end math_problem_l9_9783


namespace first_team_engineers_l9_9177

theorem first_team_engineers (E : ℕ) 
  (teamQ_engineers : ℕ := 16) 
  (work_days_teamQ : ℕ := 30) 
  (work_days_first_team : ℕ := 32) 
  (working_capacity_ratio : ℚ := 3 / 2) :
  E * work_days_first_team * 3 = teamQ_engineers * work_days_teamQ * 2 → 
  E = 10 :=
by
  sorry

end first_team_engineers_l9_9177


namespace no_p_dependence_l9_9719

theorem no_p_dependence (m : ℕ) (p : ℕ) (hp : Prime p) (hm : m < p)
  (n : ℕ) (hn : 0 < n) (k : ℕ) 
  (h : m^2 + n^2 + p^2 - 2*m*n - 2*m*p - 2*n*p = k^2) : 
  ∀ q : ℕ, Prime q → m < q → (m^2 + n^2 + q^2 - 2*m*n - 2*m*q - 2*n*q = k^2) :=
by sorry

end no_p_dependence_l9_9719


namespace point_in_fourth_quadrant_l9_9789

def x : ℝ := 8
def y : ℝ := -3

theorem point_in_fourth_quadrant (h1 : x > 0) (h2 : y < 0) : (x > 0 ∧ y < 0) :=
by {
  sorry
}

end point_in_fourth_quadrant_l9_9789


namespace area_R3_l9_9335

-- Define the initial dimensions of rectangle R1
def length_R1 := 8
def width_R1 := 4

-- Define the dimensions of rectangle R2 after bisecting R1
def length_R2 := length_R1 / 2
def width_R2 := width_R1

-- Define the dimensions of rectangle R3 after bisecting R2
def length_R3 := length_R2 / 2
def width_R3 := width_R2

-- Prove that the area of R3 is 8
theorem area_R3 : (length_R3 * width_R3) = 8 := by
  -- Calculation for the theorem
  sorry

end area_R3_l9_9335


namespace total_handshakes_l9_9036

theorem total_handshakes (gremlins imps unfriendly_gremlins : ℕ) 
    (handshakes_among_friendly : ℕ) (handshakes_friendly_with_unfriendly : ℕ) 
    (handshakes_between_imps_and_gremlins : ℕ) 
    (h_friendly : gremlins = 30) (h_imps : imps = 20) 
    (h_unfriendly : unfriendly_gremlins = 10) 
    (h_handshakes_among_friendly : handshakes_among_friendly = 190) 
    (h_handshakes_friendly_with_unfriendly : handshakes_friendly_with_unfriendly = 200)
    (h_handshakes_between_imps_and_gremlins : handshakes_between_imps_and_gremlins = 600) : 
    handshakes_among_friendly + handshakes_friendly_with_unfriendly + handshakes_between_imps_and_gremlins = 990 := 
by 
    sorry

end total_handshakes_l9_9036


namespace frog_escape_probability_l9_9718

def jump_probability (N : ℕ) : ℚ := N / 14

def survival_probability (P : ℕ → ℚ) (N : ℕ) : ℚ :=
  if N = 0 then 0
  else if N = 14 then 1
  else jump_probability N * P (N - 1) + (1 - jump_probability N) * P (N + 1)

theorem frog_escape_probability :
  ∃ (P : ℕ → ℚ), P 0 = 0 ∧ P 14 = 1 ∧ (∀ (N : ℕ), 0 < N ∧ N < 14 → survival_probability P N = P N) ∧ P 3 = 325 / 728 :=
sorry

end frog_escape_probability_l9_9718


namespace minute_hour_hands_opposite_l9_9101

theorem minute_hour_hands_opposite (x : ℝ) (h1 : 10 * 60 ≤ x) (h2 : x ≤ 11 * 60) : 
  (5.5 * x = 442.5) :=
sorry

end minute_hour_hands_opposite_l9_9101


namespace find_solutions_l9_9566

-- Defining the system of equations as conditions
def cond1 (a b : ℕ) := a * b + 2 * a - b = 58
def cond2 (b c : ℕ) := b * c + 4 * b + 2 * c = 300
def cond3 (c d : ℕ) := c * d - 6 * c + 4 * d = 101

-- Theorem to prove the solutions satisfy the system of equations
theorem find_solutions (a b c d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0):
  cond1 a b ∧ cond2 b c ∧ cond3 c d ↔ (a, b, c, d) ∈ [(3, 26, 7, 13), (15, 2, 73, 7)] :=
by sorry

end find_solutions_l9_9566


namespace winning_votes_calculation_l9_9926

variables (V : ℚ) (winner_votes : ℚ)

-- Conditions
def percentage_of_votes_of_winner : ℚ := 0.60 * V
def percentage_of_votes_of_loser : ℚ := 0.40 * V
def vote_difference_spec : 0.60 * V - 0.40 * V = 288 := by sorry

-- Theorem to prove
theorem winning_votes_calculation (h1 : winner_votes = 0.60 * V)
  (h2 : 0.60 * V - 0.40 * V = 288) : winner_votes = 864 :=
by
  sorry

end winning_votes_calculation_l9_9926


namespace transformed_triangle_area_l9_9127

-- Define the function g and its properties
variable {R : Type*} [LinearOrderedField R]
variable (g : R → R)
variable (a b c : R)
variable (area_original : R)

-- Given conditions
-- The function g is defined such that the area of the triangle formed by 
-- points (a, g(a)), (b, g(b)), and (c, g(c)) is 24
axiom h₀ : {x | x = a ∨ x = b ∨ x = c} ⊆ Set.univ
axiom h₁ : area_original = 24

-- Define a function that computes the area of a triangle given three points
noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : R) : R := 
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem transformed_triangle_area (h₀ : {x | x = a ∨ x = b ∨ x = c} ⊆ Set.univ)
  (h₁ : area_triangle a (g a) b (g b) c (g c) = 24) :
  area_triangle (a / 3) (3 * g a) (b / 3) (3 * g b) (c / 3) (3 * g c) = 24 :=
sorry

end transformed_triangle_area_l9_9127


namespace total_canoes_by_end_of_april_l9_9815

def canoes_built_jan : Nat := 4

def canoes_built_next_month (prev_month : Nat) : Nat := 3 * prev_month

def canoes_built_feb : Nat := canoes_built_next_month canoes_built_jan
def canoes_built_mar : Nat := canoes_built_next_month canoes_built_feb
def canoes_built_apr : Nat := canoes_built_next_month canoes_built_mar

def total_canoes_built : Nat := canoes_built_jan + canoes_built_feb + canoes_built_mar + canoes_built_apr

theorem total_canoes_by_end_of_april : total_canoes_built = 160 :=
by
  sorry

end total_canoes_by_end_of_april_l9_9815


namespace books_per_shelf_l9_9147

theorem books_per_shelf 
  (initial_books : ℕ) 
  (sold_books : ℕ) 
  (num_shelves : ℕ) 
  (remaining_books : ℕ := initial_books - sold_books) :
  initial_books = 40 → sold_books = 20 → num_shelves = 5 → remaining_books / num_shelves = 4 :=
by
  sorry

end books_per_shelf_l9_9147


namespace first_candidate_percentage_l9_9368

-- Conditions
def total_votes : ℕ := 600
def second_candidate_votes : ℕ := 240
def first_candidate_votes : ℕ := total_votes - second_candidate_votes

-- Question and correct answer
theorem first_candidate_percentage : (first_candidate_votes * 100) / total_votes = 60 := by
  sorry

end first_candidate_percentage_l9_9368


namespace hotdogs_sold_l9_9580

-- Definitions of initial and remaining hotdogs
def initial : ℕ := 99
def remaining : ℕ := 97

-- The statement that needs to be proven
theorem hotdogs_sold : initial - remaining = 2 :=
by
  sorry

end hotdogs_sold_l9_9580


namespace greatest_possible_perimeter_l9_9793

theorem greatest_possible_perimeter :
  ∃ (x : ℤ), x ≥ 4 ∧ x ≤ 5 ∧ (x + 4 * x + 18 = 43 ∧
    ∀ (y : ℤ), y ≥ 4 ∧ y ≤ 5 → y + 4 * y + 18 ≤ 43) :=
by
  sorry

end greatest_possible_perimeter_l9_9793


namespace height_of_parallelogram_l9_9240

-- Define the problem statement
theorem height_of_parallelogram (A : ℝ) (b : ℝ) (h : ℝ) (h_eq : A = b * h) (A_val : A = 384) (b_val : b = 24) : h = 16 :=
by
  -- Skeleton proof, include the initial conditions and proof statement
  sorry

end height_of_parallelogram_l9_9240


namespace total_amount_is_2500_l9_9601

noncomputable def total_amount_divided (P1 : ℝ) (annual_income : ℝ) : ℝ :=
  let P2 := 2500 - P1
  let income_from_P1 := (5 / 100) * P1
  let income_from_P2 := (6 / 100) * P2
  income_from_P1 + income_from_P2

theorem total_amount_is_2500 : 
  (total_amount_divided 2000 130) = 130 :=
by
  sorry

end total_amount_is_2500_l9_9601


namespace initial_floors_l9_9217

-- Define the conditions given in the problem
def austin_time := 60 -- Time Austin takes in seconds to reach the ground floor
def jake_time := 90 -- Time Jake takes in seconds to reach the ground floor
def jake_steps_per_sec := 3 -- Jake descends 3 steps per second
def steps_per_floor := 30 -- There are 30 steps per floor

-- Define the total number of steps Jake descends
def total_jake_steps := jake_time * jake_steps_per_sec

-- Define the number of floors descended in terms of total steps and steps per floor
def num_floors := total_jake_steps / steps_per_floor

-- Theorem stating the number of floors is 9
theorem initial_floors : num_floors = 9 :=
by 
  -- Provide the basic proof structure
  sorry

end initial_floors_l9_9217


namespace drinkable_amount_l9_9780

variable {LiquidBeforeTest : ℕ}
variable {Threshold : ℕ}

def can_drink_more (LiquidBeforeTest : ℕ) (Threshold : ℕ): ℕ :=
  Threshold - LiquidBeforeTest

theorem drinkable_amount :
  LiquidBeforeTest = 24 ∧ Threshold = 32 →
  can_drink_more LiquidBeforeTest Threshold = 8 := by
  sorry

end drinkable_amount_l9_9780


namespace fraction_meaningful_l9_9353

theorem fraction_meaningful (x : ℝ) : (x ≠ 1) ↔ ¬ (x - 1 = 0) :=
by
  sorry

end fraction_meaningful_l9_9353


namespace right_triangle_perpendicular_ratio_l9_9791

theorem right_triangle_perpendicular_ratio {a b c r s : ℝ}
 (h : a^2 + b^2 = c^2)
 (perpendicular : r + s = c)
 (ratio_ab : a / b = 2 / 3) :
 r / s = 4 / 9 :=
sorry

end right_triangle_perpendicular_ratio_l9_9791


namespace cos_difference_identity_l9_9219

theorem cos_difference_identity (α β : ℝ) 
  (h1 : Real.sin α = 3 / 5) 
  (h2 : Real.sin β = 5 / 13) : Real.cos (α - β) = 63 / 65 := 
by 
  sorry

end cos_difference_identity_l9_9219


namespace problem_solution_l9_9330

noncomputable def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 1 then x^2 else sorry

lemma f_odd (x : ℝ) : f (-x) = -f x := sorry

lemma f_xplus1_even (x : ℝ) : f (x + 1) = f (-x + 1) := sorry

theorem problem_solution : f 2015 = -1 := 
by 
  sorry

end problem_solution_l9_9330


namespace simplify_expression_correct_l9_9380

def simplify_expression : ℚ :=
  (5^5 + 5^3) / (5^4 - 5^2)

theorem simplify_expression_correct : simplify_expression = 65 / 12 :=
  sorry

end simplify_expression_correct_l9_9380


namespace exists_special_number_l9_9102

theorem exists_special_number :
  ∃ N : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ 149 → k ∣ N) ∨ (k + 1 ∣ N) = false) :=
sorry

end exists_special_number_l9_9102


namespace find_additional_student_number_l9_9239

def classSize : ℕ := 52
def sampleSize : ℕ := 4
def sampledNumbers : List ℕ := [5, 31, 44]
def additionalStudentNumber : ℕ := 18

theorem find_additional_student_number (classSize sampleSize : ℕ) 
    (sampledNumbers : List ℕ) : additionalStudentNumber ∈ (5 :: 31 :: 44 :: []) →
    (sampledNumbers = [5, 31, 44]) →
    (additionalStudentNumber = 18) := by
  sorry

end find_additional_student_number_l9_9239


namespace louis_never_reaches_target_l9_9255

def stable (p : ℤ × ℤ) : Prop :=
  (p.1 + p.2) % 7 ≠ 0

def move1 (p : ℤ × ℤ) : ℤ × ℤ :=
  (p.2, p.1)

def move2 (p : ℤ × ℤ) : ℤ × ℤ :=
  (3 * p.1, -4 * p.2)

def move3 (p : ℤ × ℤ) : ℤ × ℤ :=
  (-2 * p.1, 5 * p.2)

def move4 (p : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 + 1, p.2 + 6)

def move5 (p : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 - 7, p.2)

-- Define the start and target points
def start : ℤ × ℤ := (0, 1)
def target : ℤ × ℤ := (0, 0)

theorem louis_never_reaches_target :
  ∀ p, (p = start → ¬ ∃ k, move1^[k] p = target) ∧
       (p = start → ¬ ∃ k, move2^[k] p = target) ∧
       (p = start → ¬ ∃ k, move3^[k] p = target) ∧
       (p = start → ¬ ∃ k, move4^[k] p = target) ∧
       (p = start → ¬ ∃ k, move5^[k] p = target) :=
by {
  sorry
}

end louis_never_reaches_target_l9_9255


namespace angle_difference_parallelogram_l9_9562

theorem angle_difference_parallelogram (A B : ℝ) (hA : A = 55) (h1 : A + B = 180) :
  B - A = 70 := 
by
  sorry

end angle_difference_parallelogram_l9_9562


namespace principal_amount_l9_9948

theorem principal_amount (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) : 
  A = 1120 → r = 0.05 → t = 6 → P = 1120 / (1 + 0.05 * 6) :=
by
  intros h1 h2 h3
  sorry

end principal_amount_l9_9948


namespace average_weight_increase_l9_9403

theorem average_weight_increase (A : ℝ) :
  let initial_weight := 8 * A
  let new_weight := initial_weight - 65 + 89
  let new_average := new_weight / 8
  let increase := new_average - A
  increase = (89 - 65) / 8 := 
by 
  sorry

end average_weight_increase_l9_9403


namespace evaluate_exponent_l9_9073

theorem evaluate_exponent : (3^2)^4 = 6561 := sorry

end evaluate_exponent_l9_9073


namespace minjun_current_height_l9_9857

variable (initial_height : ℝ) (growth_last_year : ℝ) (growth_this_year : ℝ)

theorem minjun_current_height
  (h_initial : initial_height = 1.1)
  (h_growth_last_year : growth_last_year = 0.2)
  (h_growth_this_year : growth_this_year = 0.1) :
  initial_height + growth_last_year + growth_this_year = 1.4 :=
by
  sorry

end minjun_current_height_l9_9857


namespace smallest_k_divides_polynomial_l9_9430

theorem smallest_k_divides_polynomial :
  ∃ k : ℕ, 0 < k ∧ (∀ z : ℂ, (z^10 + z^9 + z^8 + z^6 + z^5 + z^4 + z + 1) ∣ (z^k - 1)) ∧ k = 84 :=
by
  sorry

end smallest_k_divides_polynomial_l9_9430


namespace replace_digits_and_check_divisibility_l9_9661

theorem replace_digits_and_check_divisibility (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
    (30 * 10^5 + a * 10^4 + b * 10^2 + 3 ≠ 0 ∧ 
     (30 * 10^5 + a * 10^4 + b * 10^2 + 3) % 13 = 0) ↔ 
    (30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3000803 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3020303 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3030703 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3050203 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3060603 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3080103 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3090503) := sorry

end replace_digits_and_check_divisibility_l9_9661


namespace cos_trig_identity_l9_9524

theorem cos_trig_identity (α : Real) 
  (h : Real.cos (Real.pi / 6 - α) = 3 / 5) : 
  Real.cos (5 * Real.pi / 6 + α) = - (3 / 5) :=
by
  sorry

end cos_trig_identity_l9_9524


namespace space_per_bookshelf_l9_9820

-- Defining the conditions
def S_room : ℕ := 400
def S_reserved : ℕ := 160
def n_shelves : ℕ := 3

-- Theorem statement
theorem space_per_bookshelf (S_room S_reserved n_shelves : ℕ)
  (h1 : S_room = 400) (h2 : S_reserved = 160) (h3 : n_shelves = 3) :
  (S_room - S_reserved) / n_shelves = 80 :=
by
  -- Placeholder for the proof
  sorry

end space_per_bookshelf_l9_9820


namespace total_amount_owed_l9_9551

theorem total_amount_owed :
  ∃ (P remaining_balance processing_fee new_total discount: ℝ),
    0.05 * P = 50 ∧
    remaining_balance = P - 50 ∧
    processing_fee = 0.03 * remaining_balance ∧
    new_total = remaining_balance + processing_fee ∧
    discount = 0.10 * new_total ∧
    new_total - discount = 880.65 :=
sorry

end total_amount_owed_l9_9551


namespace clock_angle_230_l9_9258

theorem clock_angle_230 (h12 : ℕ := 12) (deg360 : ℕ := 360) 
  (hour_mark_deg : ℕ := deg360 / h12) (hour_halfway : ℕ := hour_mark_deg / 2)
  (hour_deg_230 : ℕ := hour_mark_deg * 3) (total_angle : ℕ := hour_halfway + hour_deg_230) :
  total_angle = 105 := 
by
  sorry

end clock_angle_230_l9_9258


namespace tan_150_eq_neg_inv_sqrt3_l9_9424

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l9_9424


namespace booklet_cost_l9_9833

theorem booklet_cost (b : ℝ) : 
  (10 * b < 15) ∧ (12 * b > 17) → b = 1.42 := by
  sorry

end booklet_cost_l9_9833


namespace speed_of_stream_l9_9310

theorem speed_of_stream (v : ℝ) (h_still : ∀ (d : ℝ), d / (3 - v) = 2 * d / (3 + v)) : v = 1 :=
by
  sorry

end speed_of_stream_l9_9310


namespace calc_expression_l9_9549

theorem calc_expression : 3 ^ 2022 * (1 / 3) ^ 2023 = 1 / 3 :=
by
  sorry

end calc_expression_l9_9549


namespace Andrena_more_than_Debelyn_l9_9819

-- Define initial dolls count for each person
def Debelyn_initial_dolls : ℕ := 20
def Christel_initial_dolls : ℕ := 24

-- Define dolls given by Debelyn and Christel
def Debelyn_gift_dolls : ℕ := 2
def Christel_gift_dolls : ℕ := 5

-- Define remaining dolls for Debelyn and Christel after giving dolls away
def Debelyn_final_dolls : ℕ := Debelyn_initial_dolls - Debelyn_gift_dolls
def Christel_final_dolls : ℕ := Christel_initial_dolls - Christel_gift_dolls

-- Define Andrena's dolls after transactions
def Andrena_dolls : ℕ := Christel_final_dolls + 2

-- Define the Lean statement for proving Andrena has 3 more dolls than Debelyn
theorem Andrena_more_than_Debelyn : Andrena_dolls = Debelyn_final_dolls + 3 := by
  -- Here you would prove the statement
  sorry

end Andrena_more_than_Debelyn_l9_9819


namespace sandro_children_ratio_l9_9080

theorem sandro_children_ratio (d : ℕ) (h1 : d + 3 = 21) : d / 3 = 6 :=
by
  sorry

end sandro_children_ratio_l9_9080


namespace ammonium_nitrate_formed_l9_9463

-- Definitions based on conditions in the problem
def NH3_moles : ℕ := 3
def HNO3_moles (NH3 : ℕ) : ℕ := NH3 -- 1:1 molar ratio with NH3 for HNO3

-- Definition of the outcome
def NH4NO3_moles (NH3 NH4NO3 : ℕ) : Prop :=
  NH4NO3 = NH3

-- The theorem to prove that 3 moles of NH3 combined with sufficient HNO3 produces 3 moles of NH4NO3
theorem ammonium_nitrate_formed (NH3 NH4NO3 : ℕ) (h : NH3 = 3) :
  NH4NO3_moles NH3 NH4NO3 → NH4NO3 = 3 :=
by
  intro hn
  rw [h] at hn
  exact hn

end ammonium_nitrate_formed_l9_9463


namespace initial_men_count_l9_9054

theorem initial_men_count (x : ℕ) 
  (h1 : ∀ t : ℕ, t = 25 * x) 
  (h2 : ∀ t : ℕ, t = 12 * 75) : 
  x = 36 := 
by
  sorry

end initial_men_count_l9_9054


namespace percent_germinated_is_31_l9_9976

-- Define given conditions
def seeds_first_plot : ℕ := 300
def seeds_second_plot : ℕ := 200
def germination_rate_first_plot : ℝ := 0.25
def germination_rate_second_plot : ℝ := 0.40

-- Calculate the number of germinated seeds in each plot
def germinated_first_plot : ℝ := germination_rate_first_plot * seeds_first_plot
def germinated_second_plot : ℝ := germination_rate_second_plot * seeds_second_plot

-- Calculate total number of seeds and total number of germinated seeds
def total_seeds : ℕ := seeds_first_plot + seeds_second_plot
def total_germinated : ℝ := germinated_first_plot + germinated_second_plot

-- Prove the percentage of the total number of seeds that germinated
theorem percent_germinated_is_31 :
  ((total_germinated / total_seeds) * 100) = 31 := 
by
  sorry

end percent_germinated_is_31_l9_9976


namespace volume_ratio_of_cubes_l9_9307

theorem volume_ratio_of_cubes (s2 : ℝ) : 
  let s1 := s2 * (Real.sqrt 3)
  let V1 := s1^3
  let V2 := s2^3
  V1 / V2 = 3 * (Real.sqrt 3) :=
by
  admit -- si



end volume_ratio_of_cubes_l9_9307


namespace arithmetic_sqrt_9_l9_9331

theorem arithmetic_sqrt_9 :
  (∃ y : ℝ, y * y = 9 ∧ y ≥ 0) → (∃ y : ℝ, y = 3) :=
by
  sorry

end arithmetic_sqrt_9_l9_9331


namespace possible_items_l9_9341

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l9_9341


namespace g_extreme_values_l9_9429

-- Definitions based on the conditions
def f (x : ℝ) := x^3 - 2 * x^2 + x
def g (x : ℝ) := f x + 1

-- Theorem statement
theorem g_extreme_values : 
  (g (1/3) = 31/27) ∧ (g 1 = 1) := sorry

end g_extreme_values_l9_9429


namespace solve_system_of_equations_l9_9268

theorem solve_system_of_equations (x y : Real) : 
  (3 * x^2 + 3 * y^2 - 2 * x^2 * y^2 = 3) ∧ 
  (x^4 + y^4 + (2/3) * x^2 * y^2 = 17) ↔
  ( (x = Real.sqrt 2 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3 )) ∨ 
    (x = -Real.sqrt 2 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨ 
    (x = Real.sqrt 3 ∧ (y = Real.sqrt 2 ∨ y = -Real.sqrt 2 )) ∨ 
    (x = -Real.sqrt 3 ∧ (y = Real.sqrt 2 ∨ y = -Real.sqrt 2 )) ) := 
  by
    sorry

end solve_system_of_equations_l9_9268


namespace find_b_l9_9642

theorem find_b (a b : ℝ) (h : ∀ x, 2 * x^2 - a * x + 4 < 0 ↔ 1 < x ∧ x < b) : b = 2 :=
sorry

end find_b_l9_9642


namespace price_of_wheat_flour_l9_9116

theorem price_of_wheat_flour
  (initial_amount : ℕ)
  (price_rice : ℕ)
  (num_rice : ℕ)
  (price_soda : ℕ)
  (num_soda : ℕ)
  (num_wheat_flour : ℕ)
  (remaining_balance : ℕ)
  (total_spent : ℕ)
  (amount_spent_on_rice_and_soda : ℕ)
  (amount_spent_on_wheat_flour : ℕ)
  (price_per_packet_wheat_flour : ℕ) 
  (h_initial_amount : initial_amount = 500)
  (h_price_rice : price_rice = 20)
  (h_num_rice : num_rice = 2)
  (h_price_soda : price_soda = 150)
  (h_num_soda : num_soda = 1)
  (h_num_wheat_flour : num_wheat_flour = 3)
  (h_remaining_balance : remaining_balance = 235)
  (h_total_spent : total_spent = initial_amount - remaining_balance)
  (h_amount_spent_on_rice_and_soda : amount_spent_on_rice_and_soda = price_rice * num_rice + price_soda * num_soda)
  (h_amount_spent_on_wheat_flour : amount_spent_on_wheat_flour = total_spent - amount_spent_on_rice_and_soda)
  (h_price_per_packet_wheat_flour : price_per_packet_wheat_flour = amount_spent_on_wheat_flour / num_wheat_flour) :
  price_per_packet_wheat_flour = 25 :=
by 
  sorry

end price_of_wheat_flour_l9_9116


namespace minimum_value_fraction_l9_9495

theorem minimum_value_fraction (m n : ℝ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
  (h_parallel : m / (4 - n) = 1 / 2) : 
  (1 / m + 8 / n) ≥ 9 / 2 :=
by
  sorry

end minimum_value_fraction_l9_9495


namespace divides_floor_factorial_div_l9_9652

theorem divides_floor_factorial_div {m n : ℕ} (h1 : 1 < m) (h2 : m < n + 2) (h3 : 3 < n) :
  (m - 1) ∣ (n! / m) :=
sorry

end divides_floor_factorial_div_l9_9652


namespace fruit_basket_count_l9_9327

/-- We have seven identical apples and twelve identical oranges.
    A fruit basket must contain at least one piece of fruit.
    Prove that the number of different fruit baskets we can make
    is 103. -/
theorem fruit_basket_count :
  let apples := 7
  let oranges := 12
  let total_possible_baskets := (apples + 1) * (oranges + 1) - 1
  total_possible_baskets = 103 :=
by
  let apples := 7
  let oranges := 12
  let total_possible_baskets := (apples + 1) * (oranges + 1) - 1
  show total_possible_baskets = 103
  sorry

end fruit_basket_count_l9_9327


namespace relationship_of_abc_l9_9554

theorem relationship_of_abc (a b c : ℝ) 
  (h1 : b + c = 6 - 4 * a + 3 * a^2) 
  (h2 : c - b = 4 - 4 * a + a^2) : 
  a < b ∧ b ≤ c := 
sorry

end relationship_of_abc_l9_9554


namespace range_m_n_l9_9520

noncomputable def f (m n x: ℝ) : ℝ := m * Real.exp x + x^2 + n * x

theorem range_m_n (m n: ℝ) :
  (∃ x, f m n x = 0) ∧ (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by
  sorry

end range_m_n_l9_9520


namespace excluded_students_count_l9_9806

theorem excluded_students_count 
  (N : ℕ) 
  (x : ℕ) 
  (average_marks : ℕ) 
  (excluded_average_marks : ℕ) 
  (remaining_average_marks : ℕ) 
  (total_students : ℕ)
  (h1 : average_marks = 80)
  (h2 : excluded_average_marks = 70)
  (h3 : remaining_average_marks = 90)
  (h4 : total_students = 10)
  (h5 : N = total_students)
  (h6 : 80 * N = 70 * x + 90 * (N - x))
  : x = 5 :=
by
  sorry

end excluded_students_count_l9_9806


namespace factorization_correct_l9_9928

def factor_expression (x : ℝ) : ℝ :=
  (12 * x^4 - 27 * x^3 + 45 * x) - (-3 * x^4 - 6 * x^3 + 9 * x)

theorem factorization_correct (x : ℝ) : 
  factor_expression x = 3 * x * (5 * x^3 - 7 * x^2 + 12) :=
by
  sorry

end factorization_correct_l9_9928


namespace smallest_positive_divisor_l9_9007

theorem smallest_positive_divisor
  (a b x₀ y₀ : ℤ)
  (h₀ : a ≠ 0 ∨ b ≠ 0)
  (h₁ : ∀ x y, a * x₀ + b * y₀ ≤ 0 ∨ a * x + b * y ≥ a * x₀ + b * y₀)
  (h₂ : 0 < a * x₀ + b * y₀):
  ∀ x y : ℤ, a * x₀ + b * y₀ ∣ a * x + b * y := 
sorry

end smallest_positive_divisor_l9_9007


namespace train_length_l9_9350

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (speed_conversion : speed_kmh = 40) 
  (time_condition : time_s = 27) : 
  (speed_kmh * 1000 / 3600 * time_s = 300) := 
by
  sorry

end train_length_l9_9350


namespace family_snails_l9_9545

def total_snails_family (n1 n2 n3 n4 : ℕ) (mother_find : ℕ) : ℕ :=
  n1 + n2 + n3 + mother_find

def first_ducklings_snails (num_ducklings : ℕ) (snails_per_duckling : ℕ) : ℕ :=
  num_ducklings * snails_per_duckling

def remaining_ducklings_snails (num_ducklings : ℕ) (mother_snails : ℕ) : ℕ :=
  num_ducklings * (mother_snails / 2)

def mother_find_snails (snails_group1 : ℕ) (snails_group2 : ℕ) : ℕ :=
  3 * (snails_group1 + snails_group2)

theorem family_snails : 
  ∀ (ducklings : ℕ) (group1_ducklings group2_ducklings : ℕ) 
    (snails1 snails2 : ℕ) 
    (total_ducklings : ℕ), 
    ducklings = 8 →
    group1_ducklings = 3 → 
    group2_ducklings = 3 → 
    snails1 = 5 →
    snails2 = 9 →
    total_ducklings = group1_ducklings + group2_ducklings + 2 →
    total_snails_family 
      (first_ducklings_snails group1_ducklings snails1)
      (first_ducklings_snails group2_ducklings snails2)
      (remaining_ducklings_snails 2 (mother_find_snails 
        (first_ducklings_snails group1_ducklings snails1)
        (first_ducklings_snails group2_ducklings snails2)))
      (mother_find_snails 
        (first_ducklings_snails group1_ducklings snails1)
        (first_ducklings_snails group2_ducklings snails2)) 
    = 294 :=
by intros; sorry

end family_snails_l9_9545


namespace average_score_of_entire_class_l9_9682

theorem average_score_of_entire_class :
  ∀ (num_students num_boys : ℕ) (avg_score_girls avg_score_boys : ℝ),
  num_students = 50 →
  num_boys = 20 →
  avg_score_girls = 85 →
  avg_score_boys = 80 →
  (avg_score_boys * num_boys + avg_score_girls * (num_students - num_boys)) / num_students = 83 :=
by
  intros num_students num_boys avg_score_girls avg_score_boys
  sorry

end average_score_of_entire_class_l9_9682


namespace victor_weight_is_correct_l9_9505

-- Define the given conditions
def bear_daily_food : ℕ := 90
def victors_food_in_3_weeks : ℕ := 15
def days_in_3_weeks : ℕ := 21

-- Define the equivalent weight of Victor based on the given conditions
def victor_weight : ℕ := bear_daily_food * days_in_3_weeks / victors_food_in_3_weeks

-- Prove that the weight of Victor is 126 pounds
theorem victor_weight_is_correct : victor_weight = 126 := by
  sorry

end victor_weight_is_correct_l9_9505


namespace find_2a_plus_b_l9_9462

open Real

-- Define the given conditions
variables (a b : ℝ)

-- a and b are acute angles
axiom acute_a : 0 < a ∧ a < π / 2
axiom acute_b : 0 < b ∧ b < π / 2

axiom condition1 : 4 * sin a ^ 2 + 3 * sin b ^ 2 = 1
axiom condition2 : 4 * sin (2 * a) - 3 * sin (2 * b) = 0

-- Define the theorem we want to prove
theorem find_2a_plus_b : 2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l9_9462


namespace range_of_sum_of_two_l9_9635

theorem range_of_sum_of_two (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) : 
  0 ≤ a + b ∧ a + b ≤ 4 / 3 :=
by
  -- Proof goes here.
  sorry

end range_of_sum_of_two_l9_9635


namespace dress_assignment_l9_9235

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l9_9235


namespace range_of_a_l9_9094

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a * x + 3 * x > 2 * a + 3) ↔ (x < 1)) → (a < -3 / 2) :=
by
  intro h
  sorry

end range_of_a_l9_9094


namespace alicia_tax_correct_l9_9373

theorem alicia_tax_correct :
  let hourly_wage_dollars := 25
  let hourly_wage_cents := hourly_wage_dollars * 100
  let basic_tax_rate := 0.01
  let additional_tax_rate := 0.0075
  let basic_tax := basic_tax_rate * hourly_wage_cents
  let excess_amount_cents := (hourly_wage_dollars - 20) * 100
  let additional_tax := additional_tax_rate * excess_amount_cents
  basic_tax + additional_tax = 28.75 := 
by
  sorry

end alicia_tax_correct_l9_9373


namespace y_is_defined_iff_x_not_equal_to_10_l9_9151

def range_of_independent_variable (x : ℝ) : Prop :=
  x ≠ 10

theorem y_is_defined_iff_x_not_equal_to_10 (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 10)) ↔ range_of_independent_variable x :=
by sorry

end y_is_defined_iff_x_not_equal_to_10_l9_9151


namespace sequence_number_theorem_l9_9117

def seq_count (n k : ℕ) : ℕ :=
  -- Sequence count function definition given the conditions.
  sorry -- placeholder, as we are only defining the statement, not the function itself.

theorem sequence_number_theorem (n k : ℕ) : seq_count n k = Nat.choose (n-1) k :=
by
  -- This is where the proof would go, currently omitted.
  sorry

end sequence_number_theorem_l9_9117


namespace probability_of_edge_endpoints_in_icosahedron_l9_9222

theorem probability_of_edge_endpoints_in_icosahedron :
  let vertices := 12
  let edges := 30
  let connections_per_vertex := 5
  (5 / (vertices - 1)) = (5 / 11) := by
  sorry

end probability_of_edge_endpoints_in_icosahedron_l9_9222


namespace find_richards_score_l9_9446

variable (R B : ℕ)

theorem find_richards_score (h1 : B = R - 14) (h2 : B = 48) : R = 62 := by
  sorry

end find_richards_score_l9_9446


namespace presidency_meeting_ways_l9_9025

theorem presidency_meeting_ways : 
  ∃ (ways : ℕ), ways = 4 * 6 * 3 * 225 := sorry

end presidency_meeting_ways_l9_9025


namespace draw_from_unit_D_l9_9656

variable (d : ℕ)

-- Variables representing the number of questionnaires drawn from A, B, C, and D
def QA : ℕ := 30 - d
def QB : ℕ := 30
def QC : ℕ := 30 + d
def QD : ℕ := 30 + 2 * d

-- Total number of questionnaires drawn
def TotalDrawn : ℕ := QA d + QB + QC d + QD d

theorem draw_from_unit_D :
  (TotalDrawn d = 150) →
  QD d = 60 := sorry

end draw_from_unit_D_l9_9656


namespace average_is_1380_l9_9896

def avg_of_numbers : Prop := 
  (1200 + 1300 + 1400 + 1510 + 1520 + 1530 + 1200) / 7 = 1380

theorem average_is_1380 : avg_of_numbers := by
  sorry

end average_is_1380_l9_9896


namespace tan_angle_sum_l9_9950

noncomputable def tan_sum (θ : ℝ) : ℝ := Real.tan (θ + (Real.pi / 4))

theorem tan_angle_sum :
  let x := 1
  let y := 2
  let θ := Real.arctan (y / x)
  tan_sum θ = -3 := by
  sorry

end tan_angle_sum_l9_9950


namespace ball_hits_ground_at_time_l9_9602

-- Given definitions from the conditions
def y (t : ℝ) : ℝ := -4.9 * t^2 + 5 * t + 8

-- Statement of the problem: proving the time t when the ball hits the ground
theorem ball_hits_ground_at_time :
  ∃ t : ℝ, y t = 0 ∧ t = 1.887 := 
sorry

end ball_hits_ground_at_time_l9_9602


namespace seven_does_not_always_divide_l9_9521

theorem seven_does_not_always_divide (n : ℤ) :
  ¬(7 ∣ (n ^ 2225 - n ^ 2005)) :=
by sorry

end seven_does_not_always_divide_l9_9521


namespace four_cards_probability_l9_9942

theorem four_cards_probability :
  let deck_size := 52
  let suits_size := 13
  ∀ (C D H S : ℕ), 
  C = 1 ∧ D = 13 ∧ H = 13 ∧ S = 13 →
  (C / deck_size) *
  (D / (deck_size - 1)) *
  (H / (deck_size - 2)) *
  (S / (deck_size - 3)) = (2197 / 499800) :=
by
  intros deck_size suits_size C D H S h
  sorry

end four_cards_probability_l9_9942


namespace student_weight_l9_9055

variable (S W : ℕ)

theorem student_weight (h1 : S - 5 = 2 * W) (h2 : S + W = 110) : S = 75 :=
by
  sorry

end student_weight_l9_9055


namespace total_get_well_cards_l9_9899

def dozens_to_cards (d : ℕ) : ℕ := d * 12
def hundreds_to_cards (h : ℕ) : ℕ := h * 100

theorem total_get_well_cards 
  (d_hospital : ℕ) (h_hospital : ℕ)
  (d_home : ℕ) (h_home : ℕ) :
  d_hospital = 25 ∧ h_hospital = 7 ∧ d_home = 39 ∧ h_home = 3 →
  (dozens_to_cards d_hospital + hundreds_to_cards h_hospital +
   dozens_to_cards d_home + hundreds_to_cards h_home) = 1768 :=
by
  intros
  sorry

end total_get_well_cards_l9_9899


namespace ratio_of_buyers_l9_9306

theorem ratio_of_buyers (B Y T : ℕ) (hB : B = 50) 
  (hT : T = Y + 40) (hTotal : B + Y + T = 140) : 
  (Y : ℚ) / B = 1 / 2 :=
by 
  sorry

end ratio_of_buyers_l9_9306


namespace hyperbola_asymptotes_l9_9865

theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, (x^2)/4 - y^2 = 1) →
  (∀ x : ℝ, y = x / 2 ∨ y = -x / 2) :=
by
  intro h1
  sorry

end hyperbola_asymptotes_l9_9865


namespace least_positive_divisible_l9_9846

/-- The first five different prime numbers are given as conditions: -/
def prime1 := 2
def prime2 := 3
def prime3 := 5
def prime4 := 7
def prime5 := 11

/-- The least positive whole number divisible by the first five primes is 2310. -/
theorem least_positive_divisible :
  ∃ n : ℕ, n > 0 ∧ (n % prime1 = 0) ∧ (n % prime2 = 0) ∧ (n % prime3 = 0) ∧ (n % prime4 = 0) ∧ (n % prime5 = 0) ∧ n = 2310 :=
sorry

end least_positive_divisible_l9_9846


namespace remainder_of_modified_expression_l9_9334

theorem remainder_of_modified_expression (x y u v : ℕ) (h : x = u * y + v) (hy_pos : y > 0) (hv_bound : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y + 4) % y = v + 4 :=
by sorry

end remainder_of_modified_expression_l9_9334


namespace f_is_odd_l9_9973

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.sqrt (1 + x^2))

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end f_is_odd_l9_9973


namespace number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l9_9026

/- Definitions for each number's expression using five eights -/
def number1 : Int := (8 / 8) ^ (8 / 8) * (8 / 8)
def number2 : Int := 8 / 8 + 8 / 8
def number3 : Int := (8 + 8 + 8) / 8
def number4 : Int := 8 / 8 + 8 / 8 + 8 / 8 + 8 / 8
def number5 : Int := (8 * 8 - 8) / 8 + 8 / 8

/- Theorem statements to be proven -/
theorem number1_is_1 : number1 = 1 := by
  sorry

theorem number2_is_2 : number2 = 2 := by
  sorry

theorem number3_is_3 : number3 = 3 := by
  sorry

theorem number4_is_4 : number4 = 4 := by
  sorry

theorem number5_is_5 : number5 = 5 := by
  sorry

end number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l9_9026


namespace binary_to_decimal_l9_9811

theorem binary_to_decimal (x : ℕ) (h : x = 0b110010) : x = 50 := by
  sorry

end binary_to_decimal_l9_9811


namespace minimum_distance_from_midpoint_to_y_axis_l9_9579

theorem minimum_distance_from_midpoint_to_y_axis (M N : ℝ × ℝ) (P : ℝ × ℝ)
  (hM : M.snd ^ 2 = M.fst) (hN : N.snd ^ 2 = N.fst)
  (hlength : (M.fst - N.fst)^2 + (M.snd - N.snd)^2 = 16)
  (hP : P = ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)) :
  abs P.fst = 7 / 4 :=
sorry

end minimum_distance_from_midpoint_to_y_axis_l9_9579


namespace value_of_expression_l9_9449

-- Given conditions as definitions
axiom cond1 (x y : ℝ) : -x + 2*y = 5

-- The theorem we want to prove
theorem value_of_expression (x y : ℝ) (h : -x + 2*y = 5) : 
  5 * (x - 2 * y)^2 - 3 * (x - 2 * y) - 60 = 80 :=
by
  -- The proof part is omitted here.
  sorry

end value_of_expression_l9_9449


namespace trillion_value_l9_9469

def ten_thousand : ℕ := 10^4
def million : ℕ := 10^6
def billion : ℕ := ten_thousand * million

theorem trillion_value : (ten_thousand * ten_thousand * billion) = 10^16 :=
by
  sorry

end trillion_value_l9_9469


namespace binom_2023_2_eq_l9_9955

theorem binom_2023_2_eq : Nat.choose 2023 2 = 2045323 := by
  sorry

end binom_2023_2_eq_l9_9955


namespace min_ab_diff_value_l9_9701

noncomputable def min_ab_diff (x y z : ℝ) : ℝ :=
  let A := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12)
  let B := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)
  A^2 - B^2

theorem min_ab_diff_value : ∀ (x y z : ℝ),
  0 ≤ x → 0 ≤ y → 0 ≤ z → min_ab_diff x y z = 36 :=
by
  intros x y z hx hy hz
  sorry

end min_ab_diff_value_l9_9701


namespace even_function_of_shift_sine_l9_9066

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := (x - 6)^2 * Real.sin (ω * x)

theorem even_function_of_shift_sine :
  ∃ ω : ℝ, (∀ x : ℝ, f x ω = f (-x) ω) → ω = π / 4 :=
by
  sorry

end even_function_of_shift_sine_l9_9066


namespace cement_percentage_of_second_concrete_l9_9017

theorem cement_percentage_of_second_concrete 
  (total_weight : ℝ) (final_percentage : ℝ) (partial_weight : ℝ) 
  (percentage_first_concrete : ℝ) :
  total_weight = 4500 →
  final_percentage = 0.108 →
  partial_weight = 1125 →
  percentage_first_concrete = 0.108 →
  ∃ percentage_second_concrete : ℝ, 
    percentage_second_concrete = 0.324 :=
by
  intros h1 h2 h3 h4
  let total_cement := total_weight * final_percentage
  let cement_first_concrete := partial_weight * percentage_first_concrete
  let cement_second_concrete := total_cement - cement_first_concrete
  let percentage_second_concrete := cement_second_concrete / partial_weight
  use percentage_second_concrete
  sorry

end cement_percentage_of_second_concrete_l9_9017


namespace remainder_is_three_l9_9058

def eleven_div_four_has_remainder_three (A : ℕ) : Prop :=
  11 = 4 * 2 + A

theorem remainder_is_three : eleven_div_four_has_remainder_three 3 :=
by
  sorry

end remainder_is_three_l9_9058


namespace propA_neither_sufficient_nor_necessary_l9_9744

def PropA (a b : ℕ) : Prop := a + b ≠ 4
def PropB (a b : ℕ) : Prop := a ≠ 1 ∧ b ≠ 3

theorem propA_neither_sufficient_nor_necessary (a b : ℕ) : 
  ¬((PropA a b → PropB a b) ∧ (PropB a b → PropA a b)) :=
by {
  sorry
}

end propA_neither_sufficient_nor_necessary_l9_9744


namespace cube_face_sharing_l9_9519

theorem cube_face_sharing (n : ℕ) :
  (∃ W B : ℕ, (W + B = n^3) ∧ (3 * W = 3 * B) ∧ W = B ∧ W = n^3 / 2) ↔ n % 2 = 0 :=
by
  sorry

end cube_face_sharing_l9_9519


namespace find_discriminant_l9_9737

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l9_9737


namespace minimum_colors_needed_l9_9347

def paint_fence_colors (B : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, B i ≠ B (i + 2)) ∧
  (∀ i : ℕ, B i ≠ B (i + 3)) ∧
  (∀ i : ℕ, B i ≠ B (i + 5))

theorem minimum_colors_needed : ∃ (c : ℕ), 
  (∀ B : ℕ → ℕ, paint_fence_colors B → c ≥ 3) ∧
  (∃ B : ℕ → ℕ, paint_fence_colors B ∧ c = 3) :=
sorry

end minimum_colors_needed_l9_9347


namespace tree_planting_total_l9_9576

theorem tree_planting_total (t4 t5 t6 : ℕ) 
  (h1 : t4 = 30)
  (h2 : t5 = 2 * t4)
  (h3 : t6 = 3 * t5 - 30) : 
  t4 + t5 + t6 = 240 := 
by 
  sorry

end tree_planting_total_l9_9576


namespace solve_equation_l9_9798

noncomputable def unique_solution (x : ℝ) : Prop :=
  2 * x * Real.log x + x - 1 = 0 → x = 1

-- Statement of our theorem
theorem solve_equation (x : ℝ) (h : 0 < x) : unique_solution x := sorry

end solve_equation_l9_9798


namespace problem_statement_l9_9645

-- Define the problem context
variables {a b c d : ℝ}

-- Define the conditions
def unit_square_condition (a b c d : ℝ) : Prop :=
  a^2 + b^2 + c^2 + d^2 ≥ 2 ∧ a^2 + b^2 + c^2 + d^2 ≤ 4 ∧ 
  a + b + c + d ≥ 2 * Real.sqrt 2 ∧ a + b + c + d ≤ 4

-- Provide the main theorem
theorem problem_statement (h : unit_square_condition a b c d) : 
  2 ≤ a^2 + b^2 + c^2 + d^2 ∧ a^2 + b^2 + c^2 + d^2 ≤ 4 ∧ 
  2 * Real.sqrt 2 ≤ a + b + c + d ∧ a + b + c + d ≤ 4 :=
  by 
  { sorry }  -- Proof to be completed

end problem_statement_l9_9645


namespace add_decimals_l9_9275

theorem add_decimals :
  5.467 + 3.92 = 9.387 :=
by
  sorry

end add_decimals_l9_9275


namespace angles_sum_132_l9_9208

theorem angles_sum_132
  (D E F p q : ℝ)
  (hD : D = 38)
  (hE : E = 58)
  (hF : F = 36)
  (five_sided_angle_sum : D + E + (360 - p) + 90 + (126 - q) = 540) : 
  p + q = 132 := 
by
  sorry

end angles_sum_132_l9_9208


namespace quadratic_two_distinct_roots_l9_9991

theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k*x^2 - 6*x + 9 = 0) ∧ (k*y^2 - 6*y + 9 = 0)) ↔ (k < 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_two_distinct_roots_l9_9991


namespace quadratic_is_complete_the_square_l9_9479

theorem quadratic_is_complete_the_square :
  ∃ a b c : ℝ, 15 * (x : ℝ)^2 + 150 * x + 2250 = a * (x + b)^2 + c 
  ∧ a + b + c = 1895 :=
sorry

end quadratic_is_complete_the_square_l9_9479


namespace distance_traveled_is_correct_l9_9416

noncomputable def speed_in_mph : ℝ := 23.863636363636363
noncomputable def seconds : ℝ := 2

-- constants for conversion
def miles_to_feet : ℝ := 5280
def hours_to_seconds : ℝ := 3600

-- speed in feet per second
noncomputable def speed_in_fps : ℝ := speed_in_mph * miles_to_feet / hours_to_seconds

-- distance traveled
noncomputable def distance : ℝ := speed_in_fps * seconds

theorem distance_traveled_is_correct : distance = 69.68 := by
  sorry

end distance_traveled_is_correct_l9_9416


namespace min_distance_PQ_l9_9693

theorem min_distance_PQ :
  ∀ (P Q : ℝ × ℝ), (P.1 - P.2 - 4 = 0) → (Q.1^2 = 4 * Q.2) →
  ∃ (d : ℝ), d = dist P Q ∧ d = 3 * Real.sqrt 2 / 2 :=
sorry

end min_distance_PQ_l9_9693


namespace Tanya_efficiency_higher_l9_9646

variable (Sakshi_days Tanya_days : ℕ)
variable (Sakshi_efficiency Tanya_efficiency increase_in_efficiency percentage_increase : ℚ)

theorem Tanya_efficiency_higher (h1: Sakshi_days = 20) (h2: Tanya_days = 16) :
  Sakshi_efficiency = 1 / 20 ∧ Tanya_efficiency = 1 / 16 ∧ 
  increase_in_efficiency = Tanya_efficiency - Sakshi_efficiency ∧ 
  percentage_increase = (increase_in_efficiency / Sakshi_efficiency) * 100 ∧
  percentage_increase = 25 := by
  sorry

end Tanya_efficiency_higher_l9_9646


namespace sum_geometric_series_nine_l9_9940

noncomputable def geometric_series_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  S n = a 0 * (1 - a 1 ^ n) / (1 - a 1)

theorem sum_geometric_series_nine
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (S_3 : S 3 = 12)
  (S_6 : S 6 = 60) :
  S 9 = 252 := by
  sorry

end sum_geometric_series_nine_l9_9940


namespace nine_digit_number_conditions_l9_9957

def nine_digit_number := 900900000

def remove_second_digit (n : ℕ) : ℕ := n / 100000000 * 10000000 + n % 10000000
def remove_third_digit (n : ℕ) : ℕ := n / 10000000 * 1000000 + n % 1000000
def remove_ninth_digit (n : ℕ) : ℕ := n / 10

theorem nine_digit_number_conditions :
  (remove_second_digit nine_digit_number) % 2 = 0 ∧
  (remove_third_digit nine_digit_number) % 3 = 0 ∧
  (remove_ninth_digit nine_digit_number) % 9 = 0 :=
by
  -- Proof steps would be included here.
  sorry

end nine_digit_number_conditions_l9_9957


namespace compound_interest_l9_9292

theorem compound_interest (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) (CI : ℝ) :
  SI = 40 → R = 5 → T = 2 → SI = (P * R * T) / 100 → CI = P * ((1 + R / 100) ^ T - 1) → CI = 41 :=
by sorry

end compound_interest_l9_9292


namespace sqrt_meaningful_range_l9_9323

theorem sqrt_meaningful_range (x : ℝ) (h : 3 * x - 5 ≥ 0) : x ≥ 5 / 3 :=
sorry

end sqrt_meaningful_range_l9_9323


namespace number_of_nonnegative_solutions_l9_9632

theorem number_of_nonnegative_solutions : ∃ (count : ℕ), count = 1 ∧ ∀ x : ℝ, x^2 + 9 * x = 0 → x ≥ 0 → x = 0 := by
  sorry

end number_of_nonnegative_solutions_l9_9632


namespace problem_sufficient_necessary_condition_l9_9020

open Set

variable {x : ℝ}

def P (x : ℝ) : Prop := abs (x - 2) < 3
def Q (x : ℝ) : Prop := x^2 - 8 * x + 15 < 0

theorem problem_sufficient_necessary_condition :
    (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬ Q x) :=
by
  sorry

end problem_sufficient_necessary_condition_l9_9020


namespace volume_of_spheres_l9_9528

noncomputable def sphere_volume (a : ℝ) : ℝ :=
  (4 / 3) * Real.pi * ((3 * a - a * Real.sqrt 3) / 4)^3

theorem volume_of_spheres (a : ℝ) : 
  ∃ r : ℝ, r = (3 * a - a * Real.sqrt 3) / 4 ∧ 
  sphere_volume a = (4 / 3) * Real.pi * r^3 := 
sorry

end volume_of_spheres_l9_9528


namespace value_of_a_l9_9939

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_a (a : ℝ) (f_symmetric : ∀ x y : ℝ, y = f x ↔ -y = 2^(-x + a)) (sum_f_condition : f (-2) + f (-4) = 1) :
  a = 2 :=
sorry

end value_of_a_l9_9939


namespace units_digit_of_42_pow_3_add_24_pow_3_l9_9775

theorem units_digit_of_42_pow_3_add_24_pow_3 :
    (42 ^ 3 + 24 ^ 3) % 10 = 2 :=
by
    have units_digit_42 := (42 % 10 = 2)
    have units_digit_24 := (24 % 10 = 4)
    sorry

end units_digit_of_42_pow_3_add_24_pow_3_l9_9775


namespace find_smaller_number_l9_9126

theorem find_smaller_number (a b : ℤ) (h1 : a + b = 18) (h2 : a - b = 24) : b = -3 :=
by
  sorry

end find_smaller_number_l9_9126


namespace min_value_of_f_range_of_a_l9_9019

def f (x : ℝ) : ℝ := 2 * |x - 2| - x + 5

theorem min_value_of_f : ∃ (m : ℝ), m = 3 ∧ ∀ x : ℝ, f x ≥ m :=
by
  use 3
  sorry

theorem range_of_a (a : ℝ) : (|a + 2| ≥ 3 ↔ a ≤ -5 ∨ a ≥ 1) :=
sorry

end min_value_of_f_range_of_a_l9_9019


namespace min_value_144_l9_9768

noncomputable def min_expression (a b c d : ℝ) : ℝ :=
  (a + b + c) / (a * b * c * d)

theorem min_value_144 (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_pos_d : 0 < d) (h_sum : a + b + c + d = 2) : min_expression a b c d ≥ 144 :=
by
  sorry

end min_value_144_l9_9768


namespace sum_of_subsets_l9_9696

theorem sum_of_subsets (a1 a2 a3 : ℝ) (h : (a1 + a2 + a3) + (a1 + a2 + a1 + a3 + a2 + a3) = 12) : 
  a1 + a2 + a3 = 4 := 
by 
  sorry

end sum_of_subsets_l9_9696


namespace average_remaining_ropes_l9_9244

theorem average_remaining_ropes 
  (n : ℕ) 
  (m : ℕ) 
  (l_avg : ℕ) 
  (l1_avg : ℕ) 
  (l2_avg : ℕ) 
  (h1 : n = 6)
  (h2 : m = 2)
  (hl_avg : l_avg = 80)
  (hl1_avg : l1_avg = 70)
  (htotal : l_avg * n = 480)
  (htotal1 : l1_avg * m = 140)
  (htotal2 : l_avg * n - l1_avg * m = 340):
  (340 : ℕ) / (4 : ℕ) = 85 := by
  sorry

end average_remaining_ropes_l9_9244


namespace evaluate_f_g_f_l9_9969

def f (x: ℝ) : ℝ := 5 * x + 4
def g (x: ℝ) : ℝ := 3 * x + 5

theorem evaluate_f_g_f :
  f (g (f 3)) = 314 :=
by
  sorry

end evaluate_f_g_f_l9_9969


namespace inverse_proportion_inequality_l9_9136

theorem inverse_proportion_inequality {x1 x2 : ℝ} (h1 : x1 > x2) (h2 : x2 > 0) : 
    -3 / x1 > -3 / x2 := 
by 
  sorry

end inverse_proportion_inequality_l9_9136


namespace jesse_bananas_l9_9365

def number_of_bananas_shared (friends : ℕ) (bananas_per_friend : ℕ) : ℕ :=
  friends * bananas_per_friend

theorem jesse_bananas :
  number_of_bananas_shared 3 7 = 21 :=
by
  sorry

end jesse_bananas_l9_9365


namespace decorations_count_l9_9228

/-
Danai is decorating her house for Halloween. She puts 12 plastic skulls all around the house.
She has 4 broomsticks, 1 for each side of the front and back doors to the house.
She puts up 12 spiderwebs around various areas of the house.
Danai puts twice as many pumpkins around the house as she put spiderwebs.
She also places a large cauldron on the dining room table.
Danai has the budget left to buy 20 more decorations and has 10 left to put up.
-/

def plastic_skulls := 12
def broomsticks := 4
def spiderwebs := 12
def pumpkins := 2 * spiderwebs
def cauldron := 1
def budget_remaining := 20
def undecorated_items := 10

def initial_decorations := plastic_skulls + broomsticks + spiderwebs + pumpkins + cauldron
def additional_decorations := budget_remaining + undecorated_items
def total_decorations := initial_decorations + additional_decorations

theorem decorations_count : total_decorations = 83 := by
  /- Detailed proof steps -/
  sorry

end decorations_count_l9_9228


namespace problem_solution_l9_9825

theorem problem_solution
  (x y : ℝ)
  (h1 : (x - y)^2 = 25)
  (h2 : x * y = -10) :
  x^2 + y^2 = 5 := sorry

end problem_solution_l9_9825


namespace find_the_number_l9_9747

theorem find_the_number :
  ∃ N : ℝ, ((4/5 : ℝ) * 25 = 20) ∧ (0.40 * N = 24) ∧ (N = 60) :=
by
  sorry

end find_the_number_l9_9747


namespace sum_of_reciprocals_l9_9159

variable (x y : ℝ)

theorem sum_of_reciprocals (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  (1 / x) + (1 / y) = 3 := 
sorry

end sum_of_reciprocals_l9_9159


namespace correct_survey_method_l9_9814

def service_life_of_light_tubes (survey_method : String) : Prop :=
  survey_method = "comprehensive"

def viewership_rate_of_spring_festival_gala (survey_method : String) : Prop :=
  survey_method = "comprehensive"

def crash_resistance_of_cars (survey_method : String) : Prop :=
  survey_method = "sample"

def fastest_student_for_sports_meeting (survey_method : String) : Prop :=
  survey_method = "sample"

theorem correct_survey_method :
  ¬(service_life_of_light_tubes "comprehensive") ∧
  ¬(viewership_rate_of_spring_festival_gala "comprehensive") ∧
  ¬(crash_resistance_of_cars "sample") ∧
  (fastest_student_for_sports_meeting "sample") :=
sorry

end correct_survey_method_l9_9814


namespace area_enclosed_by_curve_l9_9264

theorem area_enclosed_by_curve :
  ∃ (area : ℝ), (∀ (x y : ℝ), |x - 1| + |y - 1| = 1 → area = 2) :=
sorry

end area_enclosed_by_curve_l9_9264


namespace slower_train_speed_l9_9174

-- Conditions
variables (L : ℕ) -- Length of each train (in meters)
variables (v_f : ℕ) -- Speed of the faster train (in km/hr)
variables (t : ℕ) -- Time taken by the faster train to pass the slower one (in seconds)
variables (v_s : ℕ) -- Speed of the slower train (in km/hr)

-- Assumptions based on conditions of the problem
axiom length_eq : L = 30
axiom fast_speed : v_f = 42
axiom passing_time : t = 36

-- Conversion for km/hr to m/s
def km_per_hr_to_m_per_s (v : ℕ) : ℕ := (v * 5) / 18

-- Problem statement
theorem slower_train_speed : v_s = 36 :=
by
  let rel_speed := km_per_hr_to_m_per_s (v_f - v_s)
  have rel_speed_def : rel_speed = (42 - v_s) * 5 / 18 := by sorry
  have distance : 60 = rel_speed * t := by sorry
  have equation : 60 = (42 - v_s) * 10 := by sorry
  have solve_v_s : v_s = 36 := by sorry
  exact solve_v_s

end slower_train_speed_l9_9174


namespace perfect_square_divisors_of_240_l9_9993

theorem perfect_square_divisors_of_240 : 
  (∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, 0 < k ∧ k < n → ¬(k = 1 ∨ k = 4 ∨ k = 16)) := 
sorry

end perfect_square_divisors_of_240_l9_9993


namespace yan_ratio_l9_9567

variables (w x y : ℝ)

-- Given conditions
def yan_conditions : Prop :=
  w > 0 ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  (y / w = x / w + (x + y) / (7 * w))

-- The ratio of Yan's distance from his home to his distance from the stadium is 3/4
theorem yan_ratio (h : yan_conditions w x y) : 
  x / y = 3 / 4 :=
sorry

end yan_ratio_l9_9567


namespace compare_sums_of_square_roots_l9_9670

theorem compare_sums_of_square_roots
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (M : ℝ := Real.sqrt a + Real.sqrt b) 
  (N : ℝ := Real.sqrt (a + b)) :
  M > N :=
by
  sorry

end compare_sums_of_square_roots_l9_9670


namespace problem_statement_l9_9329

/-- A predicate that checks if the numbers from 1 to 2n can be split into two groups 
    such that the sum of the product of the elements of each group is divisible by 2n - 1. -/
def valid_split (n : ℕ) : Prop :=
  ∃ (a b : Finset ℕ), 
  a ∪ b = Finset.range (2 * n) ∧
  a ∩ b = ∅ ∧
  (2 * n) ∣ (a.prod id + b.prod id - 1)

theorem problem_statement : 
  ∀ n : ℕ, n > 0 → valid_split n ↔ (n = 1 ∨ ∃ a : ℕ, n = 2^a ∧ a ≥ 1) :=
by
  sorry

end problem_statement_l9_9329


namespace necessary_and_sufficient_condition_l9_9817

theorem necessary_and_sufficient_condition (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
    (∃ x : ℝ, 0 < x ∧ a^x = 2) ↔ (1 < a) := 
sorry

end necessary_and_sufficient_condition_l9_9817


namespace linear_equation_a_is_the_only_one_l9_9962

-- Definitions for each equation
def equation_a (x y : ℝ) : Prop := x + y = 2
def equation_b (x : ℝ) : Prop := x + 1 = -10
def equation_c (x y : ℝ) : Prop := x - 1/y = 6
def equation_d (x y : ℝ) : Prop := x^2 = 2 * y

-- Proof that equation_a is the only linear equation with two variables
theorem linear_equation_a_is_the_only_one (x y : ℝ) : 
  equation_a x y ∧ ¬equation_b x ∧ ¬(∃ y, equation_c x y) ∧ ¬(∃ y, equation_d x y) :=
by
  sorry

end linear_equation_a_is_the_only_one_l9_9962


namespace abs_sum_plus_two_eq_sum_abs_l9_9539

theorem abs_sum_plus_two_eq_sum_abs {a b c : ℤ} (h : |a + b + c| + 2 = |a| + |b| + |c|) :
  a^2 = 1 ∨ b^2 = 1 ∨ c^2 = 1 :=
sorry

end abs_sum_plus_two_eq_sum_abs_l9_9539


namespace total_distance_is_3_miles_l9_9648

-- Define conditions
def running_speed := 6   -- mph
def walking_speed := 2   -- mph
def running_time := 20 / 60   -- hours
def walking_time := 30 / 60   -- hours

-- Define total distance
def total_distance := (running_speed * running_time) + (walking_speed * walking_time)

theorem total_distance_is_3_miles : total_distance = 3 :=
by
  sorry

end total_distance_is_3_miles_l9_9648


namespace parallelogram_side_length_l9_9475

theorem parallelogram_side_length (a b : ℕ) (h1 : 2 * (a + b) = 16) (h2 : a = 5) : b = 3 :=
by
  sorry

end parallelogram_side_length_l9_9475


namespace cookie_problem_l9_9937

theorem cookie_problem (n : ℕ) (M A : ℕ) 
  (hM : M = n - 7) 
  (hA : A = n - 2) 
  (h_sum : M + A < n) 
  (hM_pos : M ≥ 1) 
  (hA_pos : A ≥ 1) : 
  n = 8 := 
sorry

end cookie_problem_l9_9937


namespace tan_75_eq_2_plus_sqrt_3_l9_9232

theorem tan_75_eq_2_plus_sqrt_3 : Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := 
sorry

end tan_75_eq_2_plus_sqrt_3_l9_9232


namespace sin_seven_pi_div_six_l9_9355

theorem sin_seven_pi_div_six : Real.sin (7 * Real.pi / 6) = -1 / 2 := 
  sorry

end sin_seven_pi_div_six_l9_9355


namespace angle_perpendicular_coterminal_l9_9206

theorem angle_perpendicular_coterminal (α β : ℝ) (k : ℤ) 
  (h_perpendicular : ∃ k, β = α + 90 + k * 360 ∨ β = α - 90 + k * 360) : 
  β = α + 90 + k * 360 ∨ β = α - 90 + k * 360 :=
sorry

end angle_perpendicular_coterminal_l9_9206


namespace min_value_fraction_ineq_l9_9765

-- Define the conditions and statement to be proved
theorem min_value_fraction_ineq (x : ℝ) (hx : x > 4) : 
  ∃ M, M = 4 * Real.sqrt 5 ∧ ∀ y : ℝ, y > 4 → (y + 16) / Real.sqrt (y - 4) ≥ M := 
sorry

end min_value_fraction_ineq_l9_9765


namespace find_k2_minus_b2_l9_9902

theorem find_k2_minus_b2 (k b : ℝ) (h1 : 3 = k * 1 + b) (h2 : 2 = k * (-1) + b) : k^2 - b^2 = -6 := 
by
  sorry

end find_k2_minus_b2_l9_9902


namespace original_number_l9_9748

-- Define the three-digit number and its permutations under certain conditions.
-- Prove the original number given the specific conditions stated.
theorem original_number (a b c : ℕ)
  (ha : a % 2 = 1) -- a being odd
  (m : ℕ := 100 * a + 10 * b + c)
  (sum_permutations : 100*a + 10*b + c + 100*a + 10*c + b + 100*b + 10*c + a + 
                      100*c + 10*a + b + 100*b + 10*a + c + 100*c + 10*b + a = 3300) :
  m = 192 := 
sorry

end original_number_l9_9748


namespace expected_balls_in_original_position_after_two_transpositions_l9_9196

-- Define the conditions
def num_balls : ℕ := 10

def probs_ball_unchanged : ℚ :=
  (1 / 50) + (16 / 25)

def expected_unchanged_balls (num_balls : ℕ) (probs_ball_unchanged : ℚ) : ℚ :=
  num_balls * probs_ball_unchanged

-- The theorem stating the expected number of balls in original positions
theorem expected_balls_in_original_position_after_two_transpositions
  (num_balls_eq : num_balls = 10)
  (prob_eq : probs_ball_unchanged = (1 / 50) + (16 / 25)) :
  expected_unchanged_balls num_balls probs_ball_unchanged = 7.2 := 
by
  sorry

end expected_balls_in_original_position_after_two_transpositions_l9_9196


namespace TomTotalWeight_l9_9689

def TomWeight : ℝ := 150
def HandWeight (personWeight: ℝ) : ℝ := 1.5 * personWeight
def VestWeight (personWeight: ℝ) : ℝ := 0.5 * personWeight
def TotalHandWeight (handWeight: ℝ) : ℝ := 2 * handWeight
def TotalWeight (totalHandWeight vestWeight: ℝ) : ℝ := totalHandWeight + vestWeight

theorem TomTotalWeight : TotalWeight (TotalHandWeight (HandWeight TomWeight)) (VestWeight TomWeight) = 525 := 
by
  sorry

end TomTotalWeight_l9_9689


namespace store_credit_card_discount_proof_l9_9892

def full_price : ℕ := 125
def sale_discount_percentage : ℕ := 20
def coupon_discount : ℕ := 10
def total_savings : ℕ := 44

def sale_discount := full_price * sale_discount_percentage / 100
def price_after_sale_discount := full_price - sale_discount
def price_after_coupon := price_after_sale_discount - coupon_discount
def store_credit_card_discount := total_savings - sale_discount - coupon_discount
def discount_percentage_of_store_credit := (store_credit_card_discount * 100) / price_after_coupon

theorem store_credit_card_discount_proof : discount_percentage_of_store_credit = 10 := by
  sorry

end store_credit_card_discount_proof_l9_9892


namespace range_of_a_squared_plus_b_l9_9163

variable (a b : ℝ)

theorem range_of_a_squared_plus_b (h1 : a < -2) (h2 : b > 4) : ∃ y, y = a^2 + b ∧ 8 < y :=
by
  sorry

end range_of_a_squared_plus_b_l9_9163


namespace total_hamburgers_for_lunch_l9_9131

theorem total_hamburgers_for_lunch 
  (initial_hamburgers: ℕ) 
  (additional_hamburgers: ℕ)
  (h1: initial_hamburgers = 9)
  (h2: additional_hamburgers = 3)
  : initial_hamburgers + additional_hamburgers = 12 := 
by
  sorry

end total_hamburgers_for_lunch_l9_9131


namespace total_ducats_is_160_l9_9881

variable (T : ℤ) (a b c d e : ℤ) -- Variables to represent the amounts taken by the robbers

-- Conditions
axiom h1 : a = 81                                            -- The strongest robber took 81 ducats
axiom h2 : b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e    -- Each remaining robber took a different amount
axiom h3 : a + b + c + d + e = T                             -- Total amount of ducats
axiom redistribution : 
  -- Redistribution process leads to each robber having the same amount
  2*b + 2*c + 2*d + 2*e = T ∧
  2*(2*c + 2*d + 2*e) = T ∧
  2*(2*(2*d + 2*e)) = T ∧
  2*(2*(2*(2*e))) = T

-- Proof that verifies the total ducats is 160
theorem total_ducats_is_160 : T = 160 :=
by
  sorry

end total_ducats_is_160_l9_9881


namespace matt_paper_piles_l9_9797

theorem matt_paper_piles (n : ℕ) (h_n1 : 1000 < n) (h_n2 : n < 2000)
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 4 = 1)
  (h5 : n % 5 = 1) (h6 : n % 6 = 1) (h7 : n % 7 = 1)
  (h8 : n % 8 = 1) : 
  ∃ k : ℕ, k ≠ 1 ∧ k ≠ n ∧ n = 1681 ∧ k = 41 :=
by
  use 41
  sorry

end matt_paper_piles_l9_9797


namespace window_treatments_cost_l9_9578

def cost_of_sheers (n : ℕ) (cost_per_pair : ℝ) : ℝ := n * cost_per_pair
def cost_of_drapes (n : ℕ) (cost_per_pair : ℝ) : ℝ := n * cost_per_pair
def total_cost (n : ℕ) (cost_sheers : ℝ) (cost_drapes : ℝ) : ℝ :=
  cost_of_sheers n cost_sheers + cost_of_drapes n cost_drapes

theorem window_treatments_cost :
  total_cost 3 40 60 = 300 :=
by
  sorry

end window_treatments_cost_l9_9578


namespace n_is_900_l9_9097

theorem n_is_900 
  (m n : ℕ) 
  (h1 : ∃ x y : ℤ, m = x^2 ∧ n = y^2) 
  (h2 : Prime (m - n)) : n = 900 := 
sorry

end n_is_900_l9_9097


namespace op_five_two_is_twentyfour_l9_9144

def op (x y : Int) : Int :=
  (x + y + 1) * (x - y)

theorem op_five_two_is_twentyfour : op 5 2 = 24 := by
  unfold op
  sorry

end op_five_two_is_twentyfour_l9_9144


namespace sample_size_of_survey_l9_9453

def eighth_grade_students : ℕ := 350
def selected_students : ℕ := 50

theorem sample_size_of_survey : selected_students = 50 :=
by sorry

end sample_size_of_survey_l9_9453


namespace num_real_roots_l9_9254

theorem num_real_roots (f : ℝ → ℝ)
  (h_eq : ∀ x, f x = 2 * x ^ 3 - 6 * x ^ 2 + 7)
  (h_interval : ∀ x, 0 < x ∧ x < 2 → f x < 0 ∧ f (2 - x) > 0) : 
  ∃! x, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end num_real_roots_l9_9254


namespace maximize_area_partition_l9_9803

noncomputable def optimLengthPartition (material: ℝ) (partitions: ℕ) : ℝ :=
  (material / (4 + partitions))

theorem maximize_area_partition :
  optimLengthPartition 24 (2 * 1) = 3 / 100 :=
by
  sorry

end maximize_area_partition_l9_9803


namespace tabitha_item_cost_l9_9494

theorem tabitha_item_cost :
  ∀ (start_money gave_mom invest fraction_remain spend item_count remain_money item_cost : ℝ),
    start_money = 25 →
    gave_mom = 8 →
    invest = (start_money - gave_mom) / 2 →
    fraction_remain = start_money - gave_mom - invest →
    spend = fraction_remain - remain_money →
    item_count = 5 →
    remain_money = 6 →
    item_cost = spend / item_count →
    item_cost = 0.5 :=
by
  intros
  sorry

end tabitha_item_cost_l9_9494


namespace scientific_notation_of_909_000_000_000_l9_9130

theorem scientific_notation_of_909_000_000_000 :
    ∃ (a : ℝ) (n : ℤ), 909000000000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 9.09 ∧ n = 11 := 
sorry

end scientific_notation_of_909_000_000_000_l9_9130


namespace positive_number_l9_9404

theorem positive_number (x : ℝ) (h1 : 0 < x) (h2 : (2 / 3) * x = (144 / 216) * (1 / x)) : x = 1 := sorry

end positive_number_l9_9404


namespace maxSUVMileage_l9_9583

noncomputable def maxSUVDistance : ℝ := 217.12

theorem maxSUVMileage 
    (tripGal : ℝ) (mpgHighway : ℝ) (mpgCity : ℝ)
    (regularHighwayRatio : ℝ) (regularCityRatio : ℝ)
    (peakHighwayRatio : ℝ) (peakCityRatio : ℝ) :
    tripGal = 23 →
    mpgHighway = 12.2 →
    mpgCity = 7.6 →
    regularHighwayRatio = 0.4 →
    regularCityRatio = 0.6 →
    peakHighwayRatio = 0.25 →
    peakCityRatio = 0.75 →
    max ((tripGal * regularHighwayRatio * mpgHighway) + (tripGal * regularCityRatio * mpgCity))
        ((tripGal * peakHighwayRatio * mpgHighway) + (tripGal * peakCityRatio * mpgCity)) = maxSUVDistance :=
by
  intros
  -- Proof would go here
  sorry

end maxSUVMileage_l9_9583


namespace probability_palindrome_divisible_by_11_l9_9609

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 = d5 ∧ d2 = d4

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def count_all_palindromes : ℕ :=
  9 * 10 * 10

def count_palindromes_div_by_11 : ℕ :=
  9 * 10

theorem probability_palindrome_divisible_by_11 :
  (count_palindromes_div_by_11 : ℚ) / count_all_palindromes = 1 / 10 :=
by sorry

end probability_palindrome_divisible_by_11_l9_9609


namespace equilibrium_and_stability_l9_9801

def system_in_equilibrium (G Q m r : ℝ) : Prop :=
    -- Stability conditions for points A and B, instability at C
    (G < (m-r)/(m-2*r)) ∧ (G > (m-r)/m)

-- Create a theorem to prove the system's equilibrium and stability
theorem equilibrium_and_stability (G Q m r : ℝ) 
  (h_gt_zero : G > 0) 
  (Q_gt_zero : Q > 0) 
  (m_gt_r : m > r) 
  (r_gt_zero : r > 0) : system_in_equilibrium G Q m r :=
by
  sorry   -- Proof omitted

end equilibrium_and_stability_l9_9801


namespace C_plus_D_l9_9237

theorem C_plus_D (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-4 * x^2 + 18 * x + 32) / (x - 3)) : 
  C + D = 28 := sorry

end C_plus_D_l9_9237


namespace earnings_in_total_l9_9501

-- Defining the conditions
def hourly_wage : ℝ := 12.50
def hours_per_week : ℝ := 40
def earnings_per_widget : ℝ := 0.16
def widgets_per_week : ℝ := 1250

-- Theorem statement
theorem earnings_in_total : 
  (hours_per_week * hourly_wage) + (widgets_per_week * earnings_per_widget) = 700 := 
by
  sorry

end earnings_in_total_l9_9501


namespace possible_values_on_Saras_card_l9_9090

theorem possible_values_on_Saras_card :
  ∀ (y : ℝ), (0 < y ∧ y < π / 2) →
  let sin_y := Real.sin y
  let cos_y := Real.cos y
  let tan_y := Real.tan y
  (∃ (s l k : ℝ), s = sin_y ∧ l = cos_y ∧ k = tan_y ∧
  (s = l ∨ s = k ∨ l = k) ∧ (s = l ∧ l ≠ k) ∧ s = l ∧ s = 1) :=
sorry

end possible_values_on_Saras_card_l9_9090


namespace david_money_left_l9_9888

noncomputable section
open Real

def money_left_after_week (rate_per_hour : ℝ) (hours_per_day : ℝ) (days_per_week : ℝ) : ℝ :=
  let total_hours := hours_per_day * days_per_week
  let total_money := total_hours * rate_per_hour
  let money_after_shoes := total_money / 2
  let money_after_mom := (total_money - money_after_shoes) / 2
  total_money - money_after_shoes - money_after_mom

theorem david_money_left :
  money_left_after_week 14 2 7 = 49 := by simp [money_left_after_week]; norm_num

end david_money_left_l9_9888


namespace stationery_box_cost_l9_9303

theorem stationery_box_cost (unit_price : ℕ) (quantity : ℕ) (total_cost : ℕ) :
  unit_price = 23 ∧ quantity = 3 ∧ total_cost = 3 * 23 → total_cost = 69 :=
by
  sorry

end stationery_box_cost_l9_9303


namespace gauss_polynomial_reciprocal_l9_9077

def gauss_polynomial (k l : ℤ) (x : ℝ) : ℝ := sorry -- Placeholder for actual polynomial definition

theorem gauss_polynomial_reciprocal (k l : ℤ) (x : ℝ) : 
  x^(k * l) * gauss_polynomial k l (1 / x) = gauss_polynomial k l x :=
sorry

end gauss_polynomial_reciprocal_l9_9077


namespace find_lowest_temperature_l9_9961

noncomputable def lowest_temperature 
(T1 T2 T3 T4 T5 : ℝ) : ℝ :=
if h : T1 + T2 + T3 + T4 + T5 = 200 ∧ max (max (max T1 T2) (max T3 T4)) T5 - min (min (min T1 T2) (min T3 T4)) T5 = 50 then
   min (min (min T1 T2) (min T3 T4)) T5
else 
  0

theorem find_lowest_temperature (T1 T2 T3 T4 T5 : ℝ) 
  (h_avg : T1 + T2 + T3 + T4 + T5 = 200)
  (h_range : max (max (max T1 T2) (max T3 T4)) T5 - min (min (min T1 T2) (min T3 T4)) T5 ≤ 50) : 
  lowest_temperature T1 T2 T3 T4 T5 = 30 := 
sorry

end find_lowest_temperature_l9_9961


namespace range_of_m_l9_9000

-- Define the two vectors a and b
def vector_a := (1, 2)
def vector_b (m : ℝ) := (m, 3 * m - 2)

-- Define the condition for non-collinearity
def non_collinear (m : ℝ) := ¬ (m / 1 = (3 * m - 2) / 2)

theorem range_of_m (m : ℝ) : non_collinear m ↔ m ≠ 2 :=
  sorry

end range_of_m_l9_9000


namespace ratio_of_fifth_terms_l9_9056

theorem ratio_of_fifth_terms (a_n b_n : ℕ → ℕ) (S T : ℕ → ℕ)
  (hs : ∀ n, S n = n * (a_n 1 + a_n n) / 2)
  (ht : ∀ n, T n = n * (b_n 1 + b_n n) / 2)
  (h : ∀ n, S n / T n = (7 * n + 2) / (n + 3)) :
  a_n 5 / b_n 5 = 65 / 12 :=
by
  sorry

end ratio_of_fifth_terms_l9_9056


namespace avg_weight_b_c_43_l9_9143

noncomputable def weights_are_correct (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧ (A + B) / 2 = 40 ∧ B = 31

theorem avg_weight_b_c_43 (A B C : ℝ) (h : weights_are_correct A B C) : (B + C) / 2 = 43 :=
by sorry

end avg_weight_b_c_43_l9_9143


namespace solve_quadratic_eq_l9_9290

theorem solve_quadratic_eq (x : ℝ) : x^2 - x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

end solve_quadratic_eq_l9_9290


namespace find_row_with_sum_2013_squared_l9_9908

-- Define the sum of the numbers in the nth row
def sum_of_row (n : ℕ) : ℕ := (2 * n - 1)^2

theorem find_row_with_sum_2013_squared : (∃ n : ℕ, sum_of_row n = 2013^2) ∧ (sum_of_row 1007 = 2013^2) :=
by
  sorry

end find_row_with_sum_2013_squared_l9_9908


namespace problems_per_page_l9_9629

-- Define the initial conditions
def total_problems : ℕ := 101
def finished_problems : ℕ := 47
def remaining_pages : ℕ := 6

-- State the theorem
theorem problems_per_page : 54 / remaining_pages = 9 :=
by
  -- Sorry is used to ignore the proof step
  sorry

end problems_per_page_l9_9629


namespace base_for_195₁₀_four_digit_even_final_digit_l9_9979

theorem base_for_195₁₀_four_digit_even_final_digit :
  ∃ b : ℕ, (b^3 ≤ 195 ∧ 195 < b^4) ∧ (∃ d : ℕ, 195 % b = d ∧ d % 2 = 0) ∧ b = 5 :=
by {
  sorry
}

end base_for_195₁₀_four_digit_even_final_digit_l9_9979


namespace initial_men_work_count_l9_9503

-- Define conditions given in the problem
def work_rate (M : ℕ) := 1 / (40 * M)
def initial_men_can_complete_work_in_40_days (M : ℕ) : Prop := M * work_rate M * 40 = 1
def work_done_by_initial_men_in_16_days (M : ℕ) := (M * 16) * work_rate M
def remaining_work_done_by_remaining_men_in_40_days (M : ℕ) := ((M - 14) * 40) * work_rate M

-- Define the main theorem to prove
theorem initial_men_work_count (M : ℕ) :
  initial_men_can_complete_work_in_40_days M →
  work_done_by_initial_men_in_16_days M = 2 / 5 →
  3 / 5 = (remaining_work_done_by_remaining_men_in_40_days M) →
  M = 15 :=
by
  intros h_initial h_16_days h_remaining
  have rate := h_initial
  sorry

end initial_men_work_count_l9_9503


namespace chocolate_bars_per_small_box_l9_9871

theorem chocolate_bars_per_small_box (total_chocolate_bars small_boxes : ℕ) 
  (h1 : total_chocolate_bars = 442) 
  (h2 : small_boxes = 17) : 
  total_chocolate_bars / small_boxes = 26 :=
by
  sorry

end chocolate_bars_per_small_box_l9_9871


namespace discount_per_issue_l9_9504

theorem discount_per_issue
  (normal_subscription_cost : ℝ) (months : ℕ) (issues_per_month : ℕ) 
  (promotional_discount : ℝ) :
  normal_subscription_cost = 34 →
  months = 18 →
  issues_per_month = 2 →
  promotional_discount = 9 →
  (normal_subscription_cost - promotional_discount) / (months * issues_per_month) = 0.25 :=
by
  intros h1 h2 h3 h4
  sorry

end discount_per_issue_l9_9504


namespace laura_change_l9_9262

-- Define the cost of a pair of pants and a shirt.
def cost_of_pants := 54
def cost_of_shirts := 33

-- Define the number of pants and shirts Laura bought.
def num_pants := 2
def num_shirts := 4

-- Define the amount Laura gave to the cashier.
def amount_given := 250

-- Calculate the total cost.
def total_cost := num_pants * cost_of_pants + num_shirts * cost_of_shirts

-- Define the expected change.
def expected_change := 10

-- The main theorem stating the problem and its solution.
theorem laura_change :
  amount_given - total_cost = expected_change :=
by
  sorry

end laura_change_l9_9262


namespace tens_digit_of_3_pow_2013_l9_9224

theorem tens_digit_of_3_pow_2013 : (3^2013 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_3_pow_2013_l9_9224


namespace cost_price_proof_l9_9855

def trader_sells_66m_for_660 : Prop := ∃ cp profit sp : ℝ, sp = 660 ∧ cp * 66 + profit * 66 = sp
def profit_5_per_meter : Prop := ∃ profit : ℝ, profit = 5
def cost_price_per_meter_is_5 : Prop := ∃ cp : ℝ, cp = 5

theorem cost_price_proof : trader_sells_66m_for_660 → profit_5_per_meter → cost_price_per_meter_is_5 :=
by
  intros h1 h2
  sorry

end cost_price_proof_l9_9855


namespace product_of_three_numbers_l9_9421

theorem product_of_three_numbers (x y z n : ℝ)
  (h_sum : x + y + z = 180)
  (h_n_eq_8x : n = 8 * x)
  (h_n_eq_y_minus_10 : n = y - 10)
  (h_n_eq_z_plus_10 : n = z + 10) :
  x * y * z = (180 / 17) * ((1440 / 17) ^ 2 - 100) := by
  sorry

end product_of_three_numbers_l9_9421


namespace largest_five_digit_integer_with_conditions_l9_9620

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digits_product (n : ℕ) : ℕ :=
  (n % 10) * ((n / 10) % 10) * ((n / 100) % 10) * ((n / 1000) % 10) * ((n / 10000) % 10)

def digits_sum (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10) + ((n / 10000) % 10)

theorem largest_five_digit_integer_with_conditions :
  ∃ n : ℕ, is_five_digit n ∧ digits_product n = 40320 ∧ digits_sum n < 35 ∧
  ∀ m : ℕ, is_five_digit m ∧ digits_product m = 40320 ∧ digits_sum m < 35 → n ≥ m :=
sorry

end largest_five_digit_integer_with_conditions_l9_9620


namespace child_tickets_sold_l9_9313

theorem child_tickets_sold (A C : ℕ) (h1 : A + C = 130) (h2 : 12 * A + 4 * C = 840) : C = 90 := by
  sorry

end child_tickets_sold_l9_9313


namespace card_area_after_shortening_l9_9464

/-- Given a card with dimensions 3 inches by 7 inches, prove that 
  if the length is shortened by 1 inch and the width is shortened by 2 inches, 
  then the resulting area is 10 square inches. -/
theorem card_area_after_shortening :
  let length := 3
  let width := 7
  let new_length := length - 1
  let new_width := width - 2
  new_length * new_width = 10 :=
by
  let length := 3
  let width := 7
  let new_length := length - 1
  let new_width := width - 2
  show new_length * new_width = 10
  sorry

end card_area_after_shortening_l9_9464


namespace area_of_each_small_concave_quadrilateral_l9_9989

noncomputable def inner_diameter : ℝ := 8
noncomputable def outer_diameter : ℝ := 10
noncomputable def total_area_covered_by_annuli : ℝ := 112.5
noncomputable def pi : ℝ := 3.14

theorem area_of_each_small_concave_quadrilateral (inner_diameter outer_diameter total_area_covered_by_annuli pi: ℝ)
    (h1 : inner_diameter = 8)
    (h2 : outer_diameter = 10)
    (h3 : total_area_covered_by_annuli = 112.5)
    (h4 : pi = 3.14) :
    (π * (outer_diameter / 2) ^ 2 - π * (inner_diameter / 2) ^ 2) * 5 - total_area_covered_by_annuli / 4 = 7.2 := 
sorry

end area_of_each_small_concave_quadrilateral_l9_9989


namespace length_of_one_side_of_regular_octagon_l9_9188

-- Define the conditions of the problem
def is_regular_octagon (n : ℕ) (P : ℝ) (length_of_side : ℝ) : Prop :=
  n = 8 ∧ P = 72 ∧ length_of_side = P / n

-- State the theorem
theorem length_of_one_side_of_regular_octagon : is_regular_octagon 8 72 9 :=
by
  -- The proof is omitted; only the statement is required
  sorry

end length_of_one_side_of_regular_octagon_l9_9188


namespace find_constants_l9_9916

variable (x : ℝ)

theorem find_constants 
  (h : ∀ x, (6 * x^2 + 3 * x) / ((x - 4) * (x - 2)^3) = 
  (13.5 / (x - 4)) + (-27 / (x - 2)) + (-15 / (x - 2)^3)) :
  true :=
by {
  sorry
}

end find_constants_l9_9916


namespace point_in_first_quadrant_l9_9974

theorem point_in_first_quadrant (x y : ℝ) (hx : x = 6) (hy : y = 2) : x > 0 ∧ y > 0 :=
by
  rw [hx, hy]
  exact ⟨by norm_num, by norm_num⟩

end point_in_first_quadrant_l9_9974


namespace quadratic_roots_l9_9607

theorem quadratic_roots (b c : ℝ) (h : ∀ x : ℝ, x^2 + bx + c = 0 ↔ x^2 - 5 * x + 2 = 0):
  c / b = -4 / 21 :=
  sorry

end quadratic_roots_l9_9607


namespace double_recipe_total_l9_9546

theorem double_recipe_total 
  (butter_ratio : ℕ) (flour_ratio : ℕ) (sugar_ratio : ℕ) 
  (flour_cups : ℕ) 
  (h_ratio : butter_ratio = 2) 
  (h_flour : flour_ratio = 5) 
  (h_sugar : sugar_ratio = 3) 
  (h_flour_cups : flour_cups = 15) : 
  2 * ((butter_ratio * (flour_cups / flour_ratio)) + flour_cups + (sugar_ratio * (flour_cups / flour_ratio))) = 60 := 
by 
  sorry

end double_recipe_total_l9_9546


namespace candy_pebbles_l9_9120

theorem candy_pebbles (C L : ℕ) 
  (h1 : L = 3 * C)
  (h2 : L = C + 8) :
  C = 4 :=
by
  sorry

end candy_pebbles_l9_9120


namespace perp_line_slope_zero_l9_9195

theorem perp_line_slope_zero {k : ℝ} (h : ∀ x : ℝ, ∃ y : ℝ, y = k * x + 1 ∧ x = 1 → false) : k = 0 :=
sorry

end perp_line_slope_zero_l9_9195


namespace paco_initial_cookies_l9_9178

theorem paco_initial_cookies :
  ∀ (total_cookies initially_ate initially_gave : ℕ),
    initially_ate = 14 →
    initially_gave = 13 →
    initially_ate = initially_gave + 1 →
    total_cookies = initially_ate + initially_gave →
    total_cookies = 27 :=
by
  intros total_cookies initially_ate initially_gave h_ate h_gave h_diff h_sum
  sorry

end paco_initial_cookies_l9_9178


namespace tony_rope_length_l9_9612

-- Define the lengths of the individual ropes.
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]

-- Define the total number of ropes Tony has.
def num_ropes : ℕ := rope_lengths.length

-- Calculate the total length of ropes before tying them together.
def total_length_before_tying : ℝ := rope_lengths.sum

-- Define the length lost per knot.
def length_lost_per_knot : ℝ := 1.2

-- Calculate the total number of knots needed.
def num_knots : ℕ := num_ropes - 1

-- Calculate the total length lost due to knots.
def total_length_lost : ℝ := num_knots * length_lost_per_knot

-- Calculate the total length of the rope after tying them all together.
def total_length_after_tying : ℝ := total_length_before_tying - total_length_lost

-- The theorem we want to prove.
theorem tony_rope_length : total_length_after_tying = 35 :=
by sorry

end tony_rope_length_l9_9612


namespace area_of_region_l9_9088

theorem area_of_region : 
  (∃ x y : ℝ, (x + 5)^2 + (y - 3)^2 = 32) → (π * 32 = 32 * π) :=
by 
  sorry

end area_of_region_l9_9088


namespace pipe_a_fills_cistern_l9_9999

theorem pipe_a_fills_cistern :
  ∀ (x : ℝ), (1 / x + 1 / 120 - 1 / 120 = 1 / 60) → x = 60 :=
by
  intro x
  intro h
  sorry

end pipe_a_fills_cistern_l9_9999


namespace standard_deviation_less_than_mean_l9_9781

theorem standard_deviation_less_than_mean 
  (μ : ℝ) (σ : ℝ) (x : ℝ) 
  (h1 : μ = 14.5) 
  (h2 : σ = 1.5) 
  (h3 : x = 11.5) : 
  (μ - x) / σ = 2 :=
by
  rw [h1, h2, h3]
  norm_num

end standard_deviation_less_than_mean_l9_9781


namespace greater_num_792_l9_9542

theorem greater_num_792 (x y : ℕ) (h1 : x + y = 1443) (h2 : x - y = 141) : x = 792 :=
by
  sorry

end greater_num_792_l9_9542


namespace triangle_third_side_lengths_l9_9297

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l9_9297


namespace pythagorean_triple_9_12_15_l9_9887

theorem pythagorean_triple_9_12_15 : 9^2 + 12^2 = 15^2 :=
by 
  sorry

end pythagorean_triple_9_12_15_l9_9887


namespace lisa_photos_last_weekend_l9_9191

def photos_of_animals : ℕ := 10
def photos_of_flowers : ℕ := 3 * photos_of_animals
def photos_of_scenery : ℕ := photos_of_flowers - 10
def total_photos_this_week : ℕ := photos_of_animals + photos_of_flowers + photos_of_scenery
def photos_last_weekend : ℕ := total_photos_this_week - 15

theorem lisa_photos_last_weekend : photos_last_weekend = 45 :=
by
  sorry

end lisa_photos_last_weekend_l9_9191


namespace probability_of_interval_l9_9592

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l9_9592


namespace molecular_weight_of_Carbonic_acid_l9_9466

theorem molecular_weight_of_Carbonic_acid :
  let H_weight := 1.008
  let C_weight := 12.011
  let O_weight := 15.999
  let H_atoms := 2
  let C_atoms := 1
  let O_atoms := 3
  (H_atoms * H_weight + C_atoms * C_weight + O_atoms * O_weight) = 62.024 :=
by 
  let H_weight := 1.008
  let C_weight := 12.011
  let O_weight := 15.999
  let H_atoms := 2
  let C_atoms := 1
  let O_atoms := 3
  sorry

end molecular_weight_of_Carbonic_acid_l9_9466


namespace input_equals_output_l9_9565

theorem input_equals_output (x : ℝ) :
  (x ≤ 1 → 2 * x - 3 = x) ∨ (x > 1 → x^2 - 3 * x + 3 = x) ↔ x = 3 :=
by
  sorry

end input_equals_output_l9_9565


namespace tangent_line_value_l9_9754

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2 * x

theorem tangent_line_value (a b : ℝ) (h : a ≤ 0) 
  (h_tangent : ∀ x : ℝ, f a x = 2 * x + b) : a - 2 * b = 2 :=
sorry

end tangent_line_value_l9_9754


namespace ratio_equivalence_l9_9729

theorem ratio_equivalence (x : ℚ) (h : x / 360 = 18 / 12) : x = 540 :=
by
  -- Proof goes here, to be filled in
  sorry

end ratio_equivalence_l9_9729


namespace third_snail_time_l9_9522

theorem third_snail_time
  (speed_first_snail : ℝ)
  (speed_second_snail : ℝ)
  (speed_third_snail : ℝ)
  (time_first_snail : ℝ)
  (distance : ℝ) :
  (speed_first_snail = 2) →
  (speed_second_snail = 2 * speed_first_snail) →
  (speed_third_snail = 5 * speed_second_snail) →
  (time_first_snail = 20) →
  (distance = speed_first_snail * time_first_snail) →
  (distance / speed_third_snail = 2) :=
by
  sorry

end third_snail_time_l9_9522


namespace quadratic_opposite_roots_l9_9901

theorem quadratic_opposite_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 + x2 = 0 ∧ x1 * x2 = k + 1) ↔ k = -2 :=
by
  sorry

end quadratic_opposite_roots_l9_9901


namespace consistent_values_for_a_l9_9628

def eq1 (x a : ℚ) : Prop := 10 * x^2 + x - a - 11 = 0
def eq2 (x a : ℚ) : Prop := 4 * x^2 + (a + 4) * x - 3 * a - 8 = 0

theorem consistent_values_for_a : ∃ x, (eq1 x 0 ∧ eq2 x 0) ∨ (eq1 x (-2) ∧ eq2 x (-2)) ∨ (eq1 x (54) ∧ eq2 x (54)) :=
by
  sorry

end consistent_values_for_a_l9_9628


namespace sum_of_a_and_b_l9_9015

theorem sum_of_a_and_b (a b : ℤ) (h1 : a + 2 * b = 8) (h2 : 2 * a + b = 4) : a + b = 4 := by
  sorry

end sum_of_a_and_b_l9_9015


namespace cube_surface_area_increase_l9_9903

theorem cube_surface_area_increase (s : ℝ) :
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  (A_new - A_original) / A_original * 100 = 224 :=
by
  -- Definitions from the conditions
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  -- Rest of the proof; replace sorry with the actual proof
  sorry

end cube_surface_area_increase_l9_9903


namespace prop_range_a_l9_9301

theorem prop_range_a (a : ℝ) 
  (p : ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → x^2 ≥ a)
  (q : ∃ (x : ℝ), x^2 + 2 * a * x + (2 - a) = 0)
  : a = 1 ∨ a ≤ -2 :=
sorry

end prop_range_a_l9_9301


namespace bananas_to_pears_l9_9735

theorem bananas_to_pears : ∀ (cost_banana cost_apple cost_pear : ℚ),
  (5 * cost_banana = 3 * cost_apple) →
  (9 * cost_apple = 6 * cost_pear) →
  (25 * cost_banana = 10 * cost_pear) :=
by
  intros cost_banana cost_apple cost_pear h1 h2
  sorry

end bananas_to_pears_l9_9735


namespace ratio_area_square_circle_eq_pi_l9_9721

theorem ratio_area_square_circle_eq_pi
  (a r : ℝ)
  (h : 4 * a = 4 * π * r) :
  (a^2 / (π * r^2)) = π := by
  sorry

end ratio_area_square_circle_eq_pi_l9_9721


namespace minimum_value_of_3x_plus_4y_l9_9598

theorem minimum_value_of_3x_plus_4y :
  ∀ (x y : ℝ), 0 < x → 0 < y → x + 3 * y = 5 * x * y → (3 * x + 4 * y) ≥ 24 / 5 :=
by
  sorry

end minimum_value_of_3x_plus_4y_l9_9598


namespace triangle_inequality_l9_9651

variable (a b c : ℝ)
variable (h1 : a * b + b * c + c * a = 18)
variable (h2 : 1 < a)
variable (h3 : 1 < b)
variable (h4 : 1 < c)

theorem triangle_inequality :
  (1 / (a - 1)^3 + 1 / (b - 1)^3 + 1 / (c - 1)^3) > (1 / (a + b + c - 3)) :=
by
  sorry

end triangle_inequality_l9_9651


namespace positive_diff_solutions_abs_eq_12_l9_9673

theorem positive_diff_solutions_abs_eq_12 : 
  ∀ (x1 x2 : ℤ), (|x1 - 4| = 12) ∧ (|x2 - 4| = 12) ∧ (x1 > x2) → (x1 - x2 = 24) :=
by
  sorry

end positive_diff_solutions_abs_eq_12_l9_9673


namespace calculate_expression_l9_9984

theorem calculate_expression:
  500 * 4020 * 0.0402 * 20 = 1616064000 := by
  sorry

end calculate_expression_l9_9984


namespace production_average_l9_9709

theorem production_average (n : ℕ) (P : ℕ) (hP : P = n * 50)
  (h1 : (P + 95) / (n + 1) = 55) : n = 8 :=
by
  -- skipping the proof
  sorry

end production_average_l9_9709


namespace unique_real_solution_system_l9_9321

/-- There is exactly one real solution (x, y, z, w) to the given system of equations:
  x + 1 = z + w + z * w * x,
  y - 1 = w + x + w * x * y,
  z + 2 = x + y + x * y * z,
  w - 2 = y + z + y * z * w
-/
theorem unique_real_solution_system :
  let eq1 (x y z w : ℝ) := x + 1 = z + w + z * w * x
  let eq2 (x y z w : ℝ) := y - 1 = w + x + w * x * y
  let eq3 (x y z w : ℝ) := z + 2 = x + y + x * y * z
  let eq4 (x y z w : ℝ) := w - 2 = y + z + y * z * w
  ∃! (x y z w : ℝ), eq1 x y z w ∧ eq2 x y z w ∧ eq3 x y z w ∧ eq4 x y z w := by {
  sorry
}

end unique_real_solution_system_l9_9321


namespace sum_lent_is_correct_l9_9295

variable (P : ℝ) -- Sum lent
variable (R : ℝ) -- Interest rate
variable (T : ℝ) -- Time period
variable (I : ℝ) -- Simple interest

-- Conditions
axiom interest_rate : R = 8
axiom time_period : T = 8
axiom simple_interest_formula : I = (P * R * T) / 100
axiom interest_condition : I = P - 900

-- The proof problem
theorem sum_lent_is_correct : P = 2500 := by
  -- The proof is skipped
  sorry

end sum_lent_is_correct_l9_9295


namespace magic_king_episodes_proof_l9_9640

-- Let's state the condition in terms of the number of seasons and episodes:
def total_episodes (seasons: ℕ) (episodes_first_half: ℕ) (episodes_second_half: ℕ) : ℕ :=
  (seasons / 2) * episodes_first_half + (seasons / 2) * episodes_second_half

-- Define the conditions for the "Magic King" show
def magic_king_total_episodes : ℕ :=
  total_episodes 10 20 25

-- The statement of the problem - to prove that the total episodes is 225
theorem magic_king_episodes_proof : magic_king_total_episodes = 225 :=
by
  sorry

end magic_king_episodes_proof_l9_9640


namespace not_possible_1006_2012_gons_l9_9029

theorem not_possible_1006_2012_gons :
  ∀ (n : ℕ), (∀ (k : ℕ), k ≤ 2011 → 2 * n ≤ k) → n ≠ 1006 :=
by
  intro n h
  -- Here goes the skipped proof part
  sorry

end not_possible_1006_2012_gons_l9_9029


namespace min_value_proof_l9_9655

noncomputable def min_value_expression (a b : ℝ) : ℝ :=
  (1 / (12 * a + 1)) + (1 / (8 * b + 1))

theorem min_value_proof (a b : ℝ) (h1 : 3 * a + 2 * b = 1) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  min_value_expression a b = 2 / 3 :=
sorry

end min_value_proof_l9_9655


namespace find_english_marks_l9_9779

variable (mathematics science social_studies english biology : ℕ)
variable (average_marks : ℕ)
variable (number_of_subjects : ℕ := 5)

-- Conditions
axiom score_math : mathematics = 76
axiom score_sci : science = 65
axiom score_ss : social_studies = 82
axiom score_bio : biology = 95
axiom average : average_marks = 77

-- The proof problem
theorem find_english_marks :
  english = 67 :=
  sorry

end find_english_marks_l9_9779


namespace cone_volume_l9_9410

theorem cone_volume :
  ∀ (l h : ℝ) (r : ℝ), l = 15 ∧ h = 9 ∧ h = 3 * r → 
  (1 / 3) * Real.pi * r^2 * h = 27 * Real.pi :=
by
  intros l h r
  intro h_eqns
  sorry

end cone_volume_l9_9410


namespace min_total_cost_of_container_l9_9571

-- Definitions from conditions
def container_volume := 4 -- m^3
def container_height := 1 -- m
def cost_per_square_meter_base : ℝ := 20
def cost_per_square_meter_sides : ℝ := 10

-- Proving the minimum total cost
theorem min_total_cost_of_container :
  ∃ (a b : ℝ), a * b = container_volume ∧
                (20 * (a + b) + 20 * (a * b)) = 160 :=
by
  sorry

end min_total_cost_of_container_l9_9571


namespace find_pairs_of_real_numbers_l9_9660

theorem find_pairs_of_real_numbers (x y : ℝ) :
  (∀ n : ℕ, n > 0 → x * ⌊n * y⌋ = y * ⌊n * x⌋) →
  (x = y ∨ x = 0 ∨ y = 0 ∨ (∃ a b : ℤ, x = a ∧ y = b)) :=
by
  sorry

end find_pairs_of_real_numbers_l9_9660


namespace trigonometric_identity_l9_9850

noncomputable def π := Real.pi
noncomputable def tan (x : ℝ) := Real.sin x / Real.cos x

theorem trigonometric_identity (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h : tan α = (1 + Real.sin β) / Real.cos β) :
  2 * α - β = π / 2 := 
sorry

end trigonometric_identity_l9_9850


namespace verify_euler_relation_for_transformed_cube_l9_9623

def euler_relation_for_transformed_cube : Prop :=
  let V := 12
  let A := 24
  let F := 14
  V + F = A + 2

theorem verify_euler_relation_for_transformed_cube :
  euler_relation_for_transformed_cube :=
by
  sorry

end verify_euler_relation_for_transformed_cube_l9_9623


namespace tax_calculation_l9_9190

variable (winnings : ℝ) (processing_fee : ℝ) (take_home : ℝ)
variable (tax_percentage : ℝ)

def given_conditions : Prop :=
  winnings = 50 ∧ processing_fee = 5 ∧ take_home = 35

def to_prove : Prop :=
  tax_percentage = 20

theorem tax_calculation (h : given_conditions winnings processing_fee take_home) : to_prove tax_percentage :=
by
  sorry

end tax_calculation_l9_9190


namespace find_a17_a18_a19_a20_l9_9695

variable {α : Type*} [Field α]

-- Definitions based on the given conditions:
def geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a n = a 0 * r ^ n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = (Finset.range n).sum a

-- Problem statement based on the question and conditions:
theorem find_a17_a18_a19_a20 (a S : ℕ → α) (h_geom : geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S) (hS4 : S 4 = 1) (hS8 : S 8 = 3) :
  a 17 + a 18 + a 19 + a 20 = 16 :=
sorry

end find_a17_a18_a19_a20_l9_9695


namespace dot_product_of_PA_PB_l9_9314

theorem dot_product_of_PA_PB
  (A B P: ℝ × ℝ)
  (h_circle : ∀ (x y : ℝ), x ^ 2 + y ^ 2 + 4 * x - 5 = 0 → (x, y) = A ∨ (x, y) = B)
  (h_midpoint : (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 1)
  (h_x_axis_intersect : P.2 = 0 ∧ (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = -5) :
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = -5 :=
sorry

end dot_product_of_PA_PB_l9_9314


namespace line_does_not_pass_third_quadrant_l9_9630

theorem line_does_not_pass_third_quadrant (a b c x y : ℝ) (h_ac : a * c < 0) (h_bc : b * c < 0) :
  ¬(x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0) :=
sorry

end line_does_not_pass_third_quadrant_l9_9630


namespace wrapping_paper_area_l9_9249

theorem wrapping_paper_area 
  (l w h : ℝ) :
  (l + 4 + 2 * h) ^ 2 = l^2 + 8 * l + 16 + 4 * l * h + 16 * h + 4 * h^2 := 
by 
  sorry

end wrapping_paper_area_l9_9249


namespace stock_yield_percentage_l9_9513

theorem stock_yield_percentage
  (annual_dividend : ℝ)
  (market_price : ℝ)
  (face_value : ℝ)
  (yield_percentage : ℝ)
  (H1 : annual_dividend = 0.14 * face_value)
  (H2 : market_price = 175)
  (H3 : face_value = 100)
  (H4 : yield_percentage = (annual_dividend / market_price) * 100) :
  yield_percentage = 8 := sorry

end stock_yield_percentage_l9_9513


namespace wreaths_per_greek_l9_9289

variable (m : ℕ) (m_pos : m > 0)

theorem wreaths_per_greek : ∃ x, x = 4 * m := 
sorry

end wreaths_per_greek_l9_9289


namespace jill_present_age_l9_9043

-- Define the main proof problem
theorem jill_present_age (H J : ℕ) (h1 : H + J = 33) (h2 : H - 6 = 2 * (J - 6)) : J = 13 :=
by
  sorry

end jill_present_age_l9_9043


namespace hyperbola_eccentricity_sqrt2_l9_9008

theorem hyperbola_eccentricity_sqrt2
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c = Real.sqrt (a^2 + b^2))
  (h : (c + a)^2 + (b^2 / a)^2 = 2 * c * (c + a)) :
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_sqrt2_l9_9008


namespace remainder_when_divided_by_10_l9_9831

theorem remainder_when_divided_by_10 : 
  (2468 * 7391 * 90523) % 10 = 4 :=
by
  sorry

end remainder_when_divided_by_10_l9_9831


namespace sqrt_difference_l9_9278

theorem sqrt_difference :
  (Real.sqrt 63 - 7 * Real.sqrt (1 / 7)) = 2 * Real.sqrt 7 :=
by
  sorry

end sqrt_difference_l9_9278


namespace relationship_between_A_and_B_l9_9705

theorem relationship_between_A_and_B (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let A := a^2
  let B := 2 * a - 1
  A > B :=
by
  let A := a^2
  let B := 2 * a - 1
  sorry

end relationship_between_A_and_B_l9_9705


namespace compare_f_values_l9_9221

noncomputable def f : ℝ → ℝ := sorry

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_monotonically_decreasing_on_nonnegative (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → x1 < x2 → f x2 < f x1

axiom even_property : is_even_function f
axiom decreasing_property : is_monotonically_decreasing_on_nonnegative f

theorem compare_f_values : f 3 < f (-2) ∧ f (-2) < f 1 :=
by {
  sorry
}

end compare_f_values_l9_9221


namespace form_regular_octagon_l9_9099

def concentric_squares_form_regular_octagon (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^2 / b^2 = 2) : Prop :=
  ∀ (p : ℂ), ∃ (h₃ : ∀ (pvertices : ℤ → ℂ), -- vertices of the smaller square
                ∀ (lperpendiculars : ℤ → ℂ), -- perpendicular line segments
                true), -- additional conditions representing the perpendicular lines construction
    -- proving that the formed shape is a regular octagon:
    true -- Placeholder for actual condition/check for regular octagon

theorem form_regular_octagon (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^2 / b^2 = 2) :
  concentric_squares_form_regular_octagon a b h₀ h₁ h₂ :=
by sorry

end form_regular_octagon_l9_9099


namespace car_speed_in_kmh_l9_9068

theorem car_speed_in_kmh (rev_per_min : ℕ) (circumference : ℕ) (speed : ℕ) 
  (h1 : rev_per_min = 400) (h2 : circumference = 4) : speed = 96 :=
  sorry

end car_speed_in_kmh_l9_9068


namespace new_person_weight_l9_9767

theorem new_person_weight (n : ℕ) (k : ℝ) (w_old w_new : ℝ) 
  (h_n : n = 6) 
  (h_k : k = 4.5) 
  (h_w_old : w_old = 75) 
  (h_avg_increase : w_new - w_old = n * k) : 
  w_new = 102 := 
sorry

end new_person_weight_l9_9767


namespace problem_1_solution_set_problem_2_minimum_value_a_l9_9084

-- Define the function f with given a value
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Problem 1: Prove the solution set for f(x) > 5 when a = -2 is {x | x < -4/3 ∨ x > 2}
theorem problem_1_solution_set (x : ℝ) : f x (-2) > 5 ↔ x < -4 / 3 ∨ x > 2 :=
by
  sorry

-- Problem 2: Prove the minimum value of a ensures f(x) ≤ a * |x + 3| is 1/2
theorem problem_2_minimum_value_a : (∀ x : ℝ, f x a ≤ a * |x + 3| ∨ a ≥ 1/2) :=
by
  sorry

end problem_1_solution_set_problem_2_minimum_value_a_l9_9084


namespace max_sum_of_positive_integers_with_product_144_l9_9853

theorem max_sum_of_positive_integers_with_product_144 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 144 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 75 := 
by
  sorry

end max_sum_of_positive_integers_with_product_144_l9_9853


namespace percentage_defective_meters_l9_9383

theorem percentage_defective_meters (total_meters : ℕ) (defective_meters : ℕ) (percentage : ℚ) :
  total_meters = 2500 →
  defective_meters = 2 →
  percentage = (defective_meters / total_meters) * 100 →
  percentage = 0.08 := 
sorry

end percentage_defective_meters_l9_9383


namespace number_of_girls_l9_9149

theorem number_of_girls
  (B : ℕ) (k : ℕ) (G : ℕ)
  (hB : B = 10) 
  (hk : k = 5)
  (h1 : B / k = 2)
  (h2 : G % k = 0) :
  G = 5 := 
sorry

end number_of_girls_l9_9149


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l9_9686

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l9_9686


namespace series_sum_correct_l9_9741

open Classical

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (2 * (k+1)) / 4^(k+1)

theorem series_sum_correct :
  series_sum = 8 / 9 :=
by
  sorry

end series_sum_correct_l9_9741


namespace problem_statement_l9_9753

-- Define the power function f and the property that it is odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the given conditions
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_cond : f 3 < f 2)

-- The statement we need to prove
theorem problem_statement : f (-3) > f (-2) := by
  sorry

end problem_statement_l9_9753


namespace integral_fx_l9_9107

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem integral_fx :
  ∫ x in -Real.pi..0, f x = -2 - (1/2) * Real.pi ^ 2 :=
by
  sorry

end integral_fx_l9_9107


namespace determine_m_l9_9675

variable {x y z : ℝ}

theorem determine_m (h : (5 / (x + y)) = (m / (x + z)) ∧ (m / (x + z)) = (13 / (z - y))) : m = 18 :=
by
  sorry

end determine_m_l9_9675


namespace solve_quadratic_polynomial_l9_9095

noncomputable def q (x : ℝ) : ℝ := -4.5 * x^2 + 4.5 * x + 135

theorem solve_quadratic_polynomial : 
  (q (-5) = 0) ∧ (q 6 = 0) ∧ (q 7 = -54) :=
by
  sorry

end solve_quadratic_polynomial_l9_9095


namespace part1_part2_l9_9658

-- Define the constants based on given conditions
def cost_price : ℕ := 5
def initial_selling_price : ℕ := 9
def initial_sales_volume : ℕ := 32
def price_increment : ℕ := 2
def sales_decrement : ℕ := 8

-- Part 1: Define the elements 
def selling_price_part1 : ℕ := 11
def profit_per_item_part1 : ℕ := 6
def daily_sales_volume_part1 : ℕ := 24

theorem part1 :
  (selling_price_part1 - cost_price = profit_per_item_part1) ∧ 
  (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price_part1 - initial_selling_price) = daily_sales_volume_part1) := 
by
  sorry

-- Part 2: Define the elements 
def target_daily_profit : ℕ := 140
def selling_price1_part2 : ℕ := 12
def selling_price2_part2 : ℕ := 10

theorem part2 :
  (((selling_price1_part2 - cost_price) *
    (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price1_part2 - initial_selling_price)) = target_daily_profit) ∨
  ((selling_price2_part2 - cost_price) *
    (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price2_part2 - initial_selling_price)) = target_daily_profit)) :=
by
  sorry

end part1_part2_l9_9658


namespace determine_original_number_l9_9842

theorem determine_original_number (a b c : ℕ) (m : ℕ) (N : ℕ) 
  (h1 : N = 4410) 
  (h2 : (a + b + c) % 2 = 0)
  (h3 : m = 100 * a + 10 * b + c)
  (h4 : N + m = 222 * (a + b + c)) : 
  a = 4 ∧ b = 4 ∧ c = 4 :=
by 
  sorry

end determine_original_number_l9_9842


namespace inequality_solution_set_l9_9959

theorem inequality_solution_set (x : ℝ) :
  (3 * (x + 2) - x > 4) ∧ ((1 + 2 * x) / 3 ≥ x - 1) ↔ (-1 < x ∧ x ≤ 4) :=
by
  sorry

end inequality_solution_set_l9_9959


namespace compute_expression_equals_375_l9_9918

theorem compute_expression_equals_375 : 15 * (30 / 6) ^ 2 = 375 := 
by 
  have frac_simplified : 30 / 6 = 5 := by sorry
  have power_calculated : 5 ^ 2 = 25 := by sorry
  have final_result : 15 * 25 = 375 := by sorry
  sorry

end compute_expression_equals_375_l9_9918


namespace div_expression_calc_l9_9180

theorem div_expression_calc :
  (3752 / (39 * 2) + 5030 / (39 * 10) = 61) :=
by
  sorry -- Proof of the theorem

end div_expression_calc_l9_9180


namespace point_satisfies_equation_l9_9771

theorem point_satisfies_equation (x y : ℝ) :
  (-1 ≤ x ∧ x ≤ 3) ∧ (-5 ≤ y ∧ y ≤ 1) ∧
  ((3 * x + 2 * y = 5) ∨ (-3 * x + 2 * y = -1) ∨ (3 * x - 2 * y = 13) ∨ (-3 * x - 2 * y = 7))
  → 3 * |x - 1| + 2 * |y + 2| = 6 := 
by 
  sorry

end point_satisfies_equation_l9_9771


namespace matching_pair_probability_l9_9471

theorem matching_pair_probability :
  let total_socks := 22
  let blue_socks := 12
  let red_socks := 10
  let total_ways := (total_socks * (total_socks - 1)) / 2
  let blue_ways := (blue_socks * (blue_socks - 1)) / 2
  let red_ways := (red_socks * (red_socks - 1)) / 2
  let matching_ways := blue_ways + red_ways
  total_ways = 231 →
  blue_ways = 66 →
  red_ways = 45 →
  matching_ways = 111 →
  (matching_ways : ℝ) / total_ways = 111 / 231 := by sorry

end matching_pair_probability_l9_9471


namespace percent_of_decimal_l9_9604

theorem percent_of_decimal : (3 / 8 / 100) * 240 = 0.9 :=
by
  sorry

end percent_of_decimal_l9_9604


namespace class_student_difference_l9_9559

theorem class_student_difference (A B : ℕ) (h : A - 4 = B + 4) : A - B = 8 := by
  sorry

end class_student_difference_l9_9559


namespace molecular_weight_of_compound_l9_9750

-- Definitions of the atomic weights.
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

-- Proof statement of the molecular weight of the compound.
theorem molecular_weight_of_compound :
  (1 * atomic_weight_K) + (1 * atomic_weight_Br) + (3 * atomic_weight_O) = 167.00 :=
  by
    sorry

end molecular_weight_of_compound_l9_9750


namespace total_paintable_area_correct_l9_9398

-- Bedroom dimensions and unoccupied wall space
def bedroom1_length : ℕ := 14
def bedroom1_width : ℕ := 12
def bedroom1_height : ℕ := 9
def bedroom1_unoccupied : ℕ := 70

def bedroom2_length : ℕ := 12
def bedroom2_width : ℕ := 11
def bedroom2_height : ℕ := 9
def bedroom2_unoccupied : ℕ := 65

def bedroom3_length : ℕ := 13
def bedroom3_width : ℕ := 12
def bedroom3_height : ℕ := 9
def bedroom3_unoccupied : ℕ := 68

-- Total paintable area calculation
def calculate_paintable_area (length width height unoccupied : ℕ) : ℕ :=
  2 * (length * height + width * height) - unoccupied

-- Total paintable area of all bedrooms
def total_paintable_area : ℕ :=
  calculate_paintable_area bedroom1_length bedroom1_width bedroom1_height bedroom1_unoccupied +
  calculate_paintable_area bedroom2_length bedroom2_width bedroom2_height bedroom2_unoccupied +
  calculate_paintable_area bedroom3_length bedroom3_width bedroom3_height bedroom3_unoccupied

theorem total_paintable_area_correct : 
  total_paintable_area = 1129 :=
by
  unfold total_paintable_area
  unfold calculate_paintable_area
  norm_num
  sorry

end total_paintable_area_correct_l9_9398


namespace hall_ratio_l9_9593

theorem hall_ratio (w l : ℕ) (h1 : w * l = 450) (h2 : l - w = 15) : w / l = 1 / 2 :=
by sorry

end hall_ratio_l9_9593


namespace simplify_expression_l9_9964

theorem simplify_expression : (0.4 * 0.5 + 0.3 * 0.2) = 0.26 := by
  sorry

end simplify_expression_l9_9964


namespace a_and_b_together_complete_work_in_12_days_l9_9643

-- Define the rate of work for b
def R_b : ℚ := 1 / 60

-- Define the rate of work for a based on the given condition that a is four times as fast as b
def R_a : ℚ := 4 * R_b

-- Define the combined rate of work for a and b working together
def R_a_plus_b : ℚ := R_a + R_b

-- Define the target time
def target_time : ℚ := 12

-- Proof statement
theorem a_and_b_together_complete_work_in_12_days :
  (R_a_plus_b * target_time) = 1 :=
by
  -- Proof omitted
  sorry

end a_and_b_together_complete_work_in_12_days_l9_9643


namespace common_ratio_of_geometric_sequence_l9_9595

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n+1) = a n * q) → 
  (a 1 + a 5 = 17) → 
  (a 2 * a 4 = 16) → 
  (∀ i j, i < j → a i < a j) → 
  q = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l9_9595


namespace determine_angle_A_max_triangle_area_l9_9213

-- Conditions: acute triangle with sides opposite to angles A, B, C as a, b, c.
variables {A B C a b c : ℝ}
-- Given condition on angles.
axiom angle_condition : 1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * Real.sin ((B + C) / 2) ^ 2 
-- Circumcircle radius
axiom circumcircle_radius : Real.pi > A ∧ A > 0 

-- Question I: Determine angle A
theorem determine_angle_A : A = Real.pi / 3 :=
by sorry

-- Given radius of the circumcircle
noncomputable def R := 2 * Real.sqrt 3 

-- Maximum area of triangle ABC
theorem max_triangle_area (a b c : ℝ) : ∃ area, area = 9 * Real.sqrt 3 :=
by sorry

end determine_angle_A_max_triangle_area_l9_9213


namespace total_amount_shared_l9_9987

noncomputable def z : ℝ := 300
noncomputable def y : ℝ := 1.2 * z
noncomputable def x : ℝ := 1.25 * y

theorem total_amount_shared (z y x : ℝ) (hz : z = 300) (hy : y = 1.2 * z) (hx : x = 1.25 * y) :
  x + y + z = 1110 :=
by
  simp [hx, hy, hz]
  -- Add intermediate steps here if necessary
  sorry

end total_amount_shared_l9_9987


namespace solve_for_k_l9_9809

theorem solve_for_k (p q : ℝ) (k : ℝ) (hpq : 3 * p^2 + 6 * p + k = 0) (hq : 3 * q^2 + 6 * q + k = 0) 
    (h_diff : |p - q| = (1 / 2) * (p^2 + q^2)) : k = -16 + 12 * Real.sqrt 2 ∨ k = -16 - 12 * Real.sqrt 2 :=
by
  sorry

end solve_for_k_l9_9809


namespace circle_center_l9_9970

theorem circle_center 
    (x y : ℝ)
    (h : x^2 + y^2 - 4 * x + 6 * y = 0) :
    (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = (x^2 - 4*x + 4) + (y^2 + 6*y + 9) 
    → (x, y) = (2, -3)) :=
sorry

end circle_center_l9_9970


namespace dance_boys_count_l9_9492

theorem dance_boys_count (d b : ℕ) (h1 : b = 2 * d) (h2 : b = d - 1 + 8) : b = 14 :=
by
  -- The proof is omitted, denoted by 'sorry'
  sorry

end dance_boys_count_l9_9492


namespace fraction_simplifies_l9_9220

theorem fraction_simplifies :
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end fraction_simplifies_l9_9220


namespace fraction_percent_of_y_l9_9787

theorem fraction_percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 10 + (3 * y) / 10) = 0.5 * y := by
  sorry

end fraction_percent_of_y_l9_9787


namespace increasing_interval_of_f_l9_9266

noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x

theorem increasing_interval_of_f :
  ∀ x : ℝ, 3 ≤ x → ∀ y : ℝ, 3 ≤ y → x < y → f x < f y := 
sorry

end increasing_interval_of_f_l9_9266


namespace max_area_100_max_fence_length_l9_9189

noncomputable def maximum_allowable_area (x y : ℝ) : Prop :=
  40 * x + 2 * 45 * y + 20 * x * y ≤ 3200

theorem max_area_100 (x y S : ℝ) (h : maximum_allowable_area x y) :
  S <= 100 :=
sorry

theorem max_fence_length (x y : ℝ) (h : maximum_allowable_area x y) (h1 : x * y = 100) :
  x = 15 :=
sorry

end max_area_100_max_fence_length_l9_9189


namespace find_number_l9_9225

theorem find_number (x : ℤ) (h : 3 * (2 * x + 15) = 75) : x = 5 :=
by
  sorry

end find_number_l9_9225


namespace problem_solution_l9_9514

def problem_conditions : Prop :=
  (∃ (students_total excellent_students: ℕ) 
     (classA_excellent classB_not_excellent: ℕ),
     students_total = 110 ∧
     excellent_students = 30 ∧
     classA_excellent = 10 ∧
     classB_not_excellent = 30)

theorem problem_solution
  (students_total excellent_students: ℕ)
  (classA_excellent classB_not_excellent: ℕ)
  (h : problem_conditions) :
  ∃ classA_not_excellent classB_excellent: ℕ,
    classA_not_excellent = 50 ∧
    classB_excellent = 20 ∧
    ((∃ χ_squared: ℝ, χ_squared = 7.5 ∧ χ_squared > 6.635) → true) ∧
    (∃ selectA selectB: ℕ, selectA = 5 ∧ selectB = 3) :=
by {
  sorry
}

end problem_solution_l9_9514


namespace range_of_a_l9_9633

variable (x a : ℝ)

-- Definitions of conditions as hypotheses
def condition_p (x : ℝ) := |x + 1| ≤ 2
def condition_q (x a : ℝ) := x ≤ a
def sufficient_not_necessary (p q : Prop) := p → q ∧ ¬(q → p)

-- The theorem statement
theorem range_of_a : sufficient_not_necessary (condition_p x) (condition_q x a) → 1 ≤ a ∧ ∀ b, b < 1 → sufficient_not_necessary (condition_p x) (condition_q x b) → false :=
by
  intro h
  sorry

end range_of_a_l9_9633


namespace initial_cell_count_l9_9486

-- Defining the constants and parameters given in the problem
def doubling_time : ℕ := 20 -- minutes
def culture_time : ℕ := 240 -- minutes (4 hours converted to minutes)
def final_bacterial_cells : ℕ := 4096

-- Definition to find the number of doublings
def num_doublings (culture_time doubling_time : ℕ) : ℕ :=
  culture_time / doubling_time

-- Definition for exponential growth formula
def exponential_growth (initial_cells : ℕ) (doublings : ℕ) : ℕ :=
  initial_cells * (2 ^ doublings)

-- The main theorem to be proven
theorem initial_cell_count :
  exponential_growth 1 (num_doublings culture_time doubling_time) = final_bacterial_cells :=
  sorry

end initial_cell_count_l9_9486


namespace math_problem_l9_9650

def otimes (a b : ℚ) : ℚ := (a^3) / (b^2)

theorem math_problem : ((otimes (otimes 2 4) 6) - (otimes 2 (otimes 4 6))) = -23327 / 288 := by sorry

end math_problem_l9_9650


namespace meteorological_forecast_l9_9157

theorem meteorological_forecast (prob_rain : ℝ) (h1 : prob_rain = 0.7) :
  (prob_rain = 0.7) → "There is a high probability of needing to carry rain gear when going out tomorrow." = "Correct" :=
by
  intro h
  sorry

end meteorological_forecast_l9_9157


namespace bill_harry_combined_l9_9893

-- Definitions based on the given conditions
def sue_nuts := 48
def harry_nuts := 2 * sue_nuts
def bill_nuts := 6 * harry_nuts

-- The theorem we want to prove
theorem bill_harry_combined : bill_nuts + harry_nuts = 672 :=
by
  sorry

end bill_harry_combined_l9_9893


namespace sum_of_squares_l9_9799

theorem sum_of_squares (a b c : ℝ) (h₁ : a + b + c = 31) (h₂ : ab + bc + ca = 10) :
  a^2 + b^2 + c^2 = 941 :=
by
  sorry

end sum_of_squares_l9_9799


namespace exists_two_digit_number_l9_9638

theorem exists_two_digit_number :
  ∃ x y : ℕ, (1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9) ∧ (10 * x + y = (x + y) * (x - y)) ∧ (10 * x + y = 48) :=
by
  sorry

end exists_two_digit_number_l9_9638


namespace chromium_percentage_l9_9048

noncomputable def chromium_percentage_in_new_alloy 
    (chromium_percentage_first: ℝ) 
    (weight_first: ℝ) 
    (chromium_percentage_second: ℝ) 
    (weight_second: ℝ) : ℝ :=
    (((chromium_percentage_first * weight_first / 100) + (chromium_percentage_second * weight_second / 100)) 
    / (weight_first + weight_second)) * 100

theorem chromium_percentage 
    (chromium_percentage_first: ℝ) 
    (weight_first: ℝ) 
    (chromium_percentage_second: ℝ) 
    (weight_second: ℝ) 
    (h1 : chromium_percentage_first = 10) 
    (h2 : weight_first = 15) 
    (h3 : chromium_percentage_second = 8) 
    (h4 : weight_second = 35) :
    chromium_percentage_in_new_alloy chromium_percentage_first weight_first chromium_percentage_second weight_second = 8.6 :=
by 
  rw [h1, h2, h3, h4]
  simp [chromium_percentage_in_new_alloy]
  norm_num


end chromium_percentage_l9_9048


namespace quadratic_roots_l9_9975

theorem quadratic_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*m = 0) ∧ (x2^2 + 2*x2 + 2*m = 0)) ↔ m < 1/2 :=
by sorry

end quadratic_roots_l9_9975


namespace min_sum_of_squares_l9_9722

theorem min_sum_of_squares 
  (x_1 x_2 x_3 : ℝ)
  (h1: x_1 + 3 * x_2 + 4 * x_3 = 72)
  (h2: x_1 = 3 * x_2)
  (h3: 0 < x_1)
  (h4: 0 < x_2)
  (h5: 0 < x_3) : 
  x_1^2 + x_2^2 + x_3^2 = 347.04 := 
sorry

end min_sum_of_squares_l9_9722


namespace Ali_is_8_l9_9920

open Nat

-- Definitions of the variables based on the conditions
def YusafAge (UmarAge : ℕ) : ℕ := UmarAge / 2
def AliAge (YusafAge : ℕ) : ℕ := YusafAge + 3

-- The specific given conditions
def UmarAge : ℕ := 10
def Yusaf : ℕ := YusafAge UmarAge
def Ali : ℕ := AliAge Yusaf

-- The theorem to be proved
theorem Ali_is_8 : Ali = 8 :=
by
  sorry

end Ali_is_8_l9_9920


namespace stationery_sales_calculation_l9_9986

-- Definitions
def total_sales : ℕ := 120
def fabric_percentage : ℝ := 0.30
def jewelry_percentage : ℝ := 0.20
def knitting_percentage : ℝ := 0.15
def home_decor_percentage : ℝ := 0.10
def stationery_percentage := 1 - (fabric_percentage + jewelry_percentage + knitting_percentage + home_decor_percentage)
def stationery_sales := stationery_percentage * total_sales

-- Statement to prove
theorem stationery_sales_calculation : stationery_sales = 30 := by
  -- Providing the initial values and assumptions to the context
  have h1 : total_sales = 120 := rfl
  have h2 : fabric_percentage = 0.30 := rfl
  have h3 : jewelry_percentage = 0.20 := rfl
  have h4 : knitting_percentage = 0.15 := rfl
  have h5 : home_decor_percentage = 0.10 := rfl
  
  -- Calculating the stationery percentage and sales
  have h_stationery_percentage : stationery_percentage = 1 - (fabric_percentage + jewelry_percentage + knitting_percentage + home_decor_percentage) := rfl
  have h_stationery_sales : stationery_sales = stationery_percentage * total_sales := rfl

  -- The calculated value should match the proof's requirement
  sorry

end stationery_sales_calculation_l9_9986


namespace problem_statement_l9_9983

noncomputable def alpha : ℝ := 3 + Real.sqrt 8
noncomputable def beta : ℝ := 3 - Real.sqrt 8
noncomputable def x : ℝ := alpha^(500)
noncomputable def N : ℝ := alpha^(500) + beta^(500)
noncomputable def n : ℝ := N - 1
noncomputable def f : ℝ := x - n
noncomputable def one_minus_f : ℝ := 1 - f

theorem problem_statement : x * one_minus_f = 1 :=
by
  -- Insert the proof here
  sorry

end problem_statement_l9_9983


namespace radius_B_eq_8_div_9_l9_9166

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Given conditions
variable (A B C D : Circle)
variable (h1 : A.radius = 1)
variable (h2 : A.radius + A.radius = D.radius)
variable (h3 : B.radius = C.radius)
variable (h4 : (A.center.1 - B.center.1)^2 + (A.center.2 - B.center.2)^2 = (A.radius + B.radius)^2)
variable (h5 : (A.center.1 - C.center.1)^2 + (A.center.2 - C.center.2)^2 = (A.radius + C.radius)^2)
variable (h6 : (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 = (B.radius + C.radius)^2)
variable (h7 : (D.center.1 - A.center.1)^2 + (D.center.2 - A.center.2)^2 = D.radius^2)

-- Prove the radius of circle B is 8/9
theorem radius_B_eq_8_div_9 : B.radius = 8 / 9 := 
by
  sorry

end radius_B_eq_8_div_9_l9_9166


namespace k_value_l9_9439

theorem k_value (k : ℝ) (h : (k / 4) + (-k / 3) = 2) : k = -24 :=
by
  sorry

end k_value_l9_9439


namespace peach_difference_proof_l9_9728

def red_peaches_odd := 12
def green_peaches_odd := 22
def red_peaches_even := 15
def green_peaches_even := 20
def num_baskets := 20
def num_odd_baskets := num_baskets / 2
def num_even_baskets := num_baskets / 2

def total_red_peaches := (red_peaches_odd * num_odd_baskets) + (red_peaches_even * num_even_baskets)
def total_green_peaches := (green_peaches_odd * num_odd_baskets) + (green_peaches_even * num_even_baskets)
def difference := total_green_peaches - total_red_peaches

theorem peach_difference_proof : difference = 150 := by
  sorry

end peach_difference_proof_l9_9728


namespace max_k_pos_l9_9121

-- Define the sequences {a_n} and {b_n}
def sequence_a (n k : ℤ) : ℤ := 2 * n + k - 1
def sequence_b (n : ℤ) : ℤ := 3 * n + 2

-- Conditions and given values
def S (n k : ℤ) : ℤ := n + k
def sum_first_9_b : ℤ := 153
def b_3 : ℤ := 11

-- Given the sequence {c_n}
def sequence_c (n k : ℤ) : ℤ := sequence_a n k - k * sequence_b n

-- Define the sum of the first n terms of the sequence {c_n}
def T (n k : ℤ) : ℤ := (n * (2 * sequence_c 1 k + (n - 1) * (2 - 3 * k))) / 2

-- Proof problem statement
theorem max_k_pos (k : ℤ) : (∀ n : ℤ, n > 0 → T n k > 0) → k ≤ 1 :=
sorry

end max_k_pos_l9_9121


namespace bricks_required_l9_9319

theorem bricks_required (courtyard_length_m : ℕ) (courtyard_width_m : ℕ)
  (brick_length_cm : ℕ) (brick_width_cm : ℕ)
  (h1 : courtyard_length_m = 30) (h2 : courtyard_width_m = 16)
  (h3 : brick_length_cm = 20) (h4 : brick_width_cm = 10) :
  (3000 * 1600) / (20 * 10) = 24000 :=
by sorry

end bricks_required_l9_9319


namespace incorrect_judgment_l9_9533

theorem incorrect_judgment : (∀ x : ℝ, x^2 - 1 ≥ -1) ∧ (4 + 2 ≠ 7) :=
by 
  sorry

end incorrect_judgment_l9_9533


namespace cone_base_circumference_l9_9028

theorem cone_base_circumference (r : ℝ) (θ : ℝ) (C : ℝ) : 
  r = 5 → θ = 300 → C = (θ / 360) * (2 * Real.pi * r) → C = (25 / 3) * Real.pi :=
by
  sorry

end cone_base_circumference_l9_9028


namespace volume_of_cuboid_l9_9197

-- Definitions of conditions
def side_length : ℕ := 6
def num_cubes : ℕ := 3
def volume_single_cube (side_length : ℕ) : ℕ := side_length ^ 3

-- The main theorem
theorem volume_of_cuboid : (num_cubes * volume_single_cube side_length) = 648 := by
  sorry

end volume_of_cuboid_l9_9197


namespace regression_value_l9_9229

theorem regression_value (x : ℝ) (y : ℝ) (h : y = 4.75 * x + 2.57) (hx : x = 28) : y = 135.57 :=
by
  sorry

end regression_value_l9_9229


namespace nadine_spent_money_l9_9930

theorem nadine_spent_money (table_cost : ℕ) (chair_cost : ℕ) (num_chairs : ℕ) 
    (h_table_cost : table_cost = 34) 
    (h_chair_cost : chair_cost = 11) 
    (h_num_chairs : num_chairs = 2) : 
    table_cost + num_chairs * chair_cost = 56 :=
by
  sorry

end nadine_spent_money_l9_9930


namespace total_messages_three_days_l9_9078

theorem total_messages_three_days :
  ∀ (A1 A2 A3 L1 L2 L3 : ℕ),
  A1 = L1 - 20 →
  L1 = 120 →
  L2 = (1 / 3 : ℚ) * L1 →
  A2 = 2 * A1 →
  A1 + L1 = A3 + L3 →
  (A1 + L1 + A2 + L2 + A3 + L3 = 680) := by
  intros A1 A2 A3 L1 L2 L3 h1 h2 h3 h4 h5
  sorry

end total_messages_three_days_l9_9078


namespace actual_length_of_tunnel_in_km_l9_9467

-- Define the conditions
def scale_factor : ℝ := 30000
def length_on_map_cm : ℝ := 7

-- Using the conditions, we need to prove the actual length is 2.1 km
theorem actual_length_of_tunnel_in_km :
  (length_on_map_cm * scale_factor / 100000) = 2.1 :=
by sorry

end actual_length_of_tunnel_in_km_l9_9467


namespace incorrect_option_C_l9_9480

theorem incorrect_option_C (a b : ℝ) (h1 : a > b) (h2 : b > a + b) : ¬ (ab > (a + b)^2) :=
by {
  sorry
}

end incorrect_option_C_l9_9480


namespace Jason_4week_visits_l9_9392

-- Definitions
def William_weekly_visits : ℕ := 2
def Jason_weekly_multiplier : ℕ := 4
def weeks_period : ℕ := 4

-- We need to prove that Jason goes to the library 32 times in 4 weeks.
theorem Jason_4week_visits : William_weekly_visits * Jason_weekly_multiplier * weeks_period = 32 := 
by sorry

end Jason_4week_visits_l9_9392


namespace quadratic_inequality_solution_l9_9778

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - (3 / 8) < 0) ↔ (-3 < k ∧ k < 0) :=
sorry

end quadratic_inequality_solution_l9_9778


namespace div_condition_l9_9387

theorem div_condition
  (a b : ℕ)
  (h₁ : a < 1000)
  (h₂ : b ≠ 0)
  (h₃ : b ∣ a ^ 21)
  (h₄ : b ^ 10 ∣ a ^ 21) :
  b ∣ a ^ 2 :=
sorry

end div_condition_l9_9387


namespace am_gm_inequality_l9_9443

-- Let's define the problem statement
theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : a + b + c ≥ 3 := by
  sorry

end am_gm_inequality_l9_9443


namespace sum_abs_eq_l9_9720

theorem sum_abs_eq (a b : ℝ) (ha : |a| = 10) (hb : |b| = 7) (hab : a > b) : a + b = 17 ∨ a + b = 3 :=
sorry

end sum_abs_eq_l9_9720


namespace parabola_focus_coordinates_l9_9712

theorem parabola_focus_coordinates (y x : ℝ) (h : y^2 = 8 * x) : (x, y) = (2, 0) :=
sorry

end parabola_focus_coordinates_l9_9712


namespace geometric_sequence_fourth_term_l9_9600

theorem geometric_sequence_fourth_term (x : ℝ) (r : ℝ) (h1 : (3 * x + 3)^2 = x * (6 * x + 6)) 
(h2 : r = (3 * x + 3) / x) :
  (6 * x + 6) * r = -24 :=
by {
  -- Definitions of x, r and condition h1, h2 are given.
  -- Conclusion must follow that the fourth term is -24.
  sorry
}

end geometric_sequence_fourth_term_l9_9600


namespace proof_theorem_l9_9994

noncomputable def proof_problem : Prop :=
  let a := 6
  let b := 15
  let c := 7
  let lhs := a * b * c
  let rhs := (Real.sqrt ((a^2) + (2 * a) + (b^3) - (b^2) + (3 * b))) / (c^2 + c + 1) + 629.001
  lhs = rhs

theorem proof_theorem : proof_problem :=
  by
  sorry

end proof_theorem_l9_9994


namespace part1_part2_l9_9414

-- Definition of the quadratic equation and its real roots condition
def quadratic_has_real_roots (k : ℝ) : Prop :=
  let Δ := (2 * k - 1)^2 - 4 * (k^2 - 1)
  Δ ≥ 0

-- Proving part (1): The range of real number k
theorem part1 (k : ℝ) (hk : quadratic_has_real_roots k) : k ≤ 5 / 4 := 
  sorry

-- Definition using the given condition in part (2)
def roots_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 = 16 + x₁ * x₂

-- Sum and product of roots of the quadratic equation
theorem part2 (k : ℝ) (h : quadratic_has_real_roots k) 
  (hx_sum : ∃ x₁ x₂ : ℝ, x₁ + x₂ = 1 - 2 * k ∧ x₁ * x₂ = k^2 - 1 ∧ roots_condition x₁ x₂) : k = -2 :=
  sorry

end part1_part2_l9_9414


namespace largest_of_seven_consecutive_l9_9739

theorem largest_of_seven_consecutive (n : ℕ) (h1 : (7 * n + 21 = 3020)) : (n + 6 = 434) :=
sorry

end largest_of_seven_consecutive_l9_9739


namespace minimum_shirts_to_save_money_l9_9691

-- Definitions for the costs
def EliteCost (n : ℕ) : ℕ := 30 + 8 * n
def OmegaCost (n : ℕ) : ℕ := 10 + 12 * n

-- Theorem to prove the given solution
theorem minimum_shirts_to_save_money : ∃ n : ℕ, 30 + 8 * n < 10 + 12 * n ∧ n = 6 :=
by {
  sorry
}

end minimum_shirts_to_save_money_l9_9691


namespace contractor_absent_days_l9_9498

theorem contractor_absent_days :
  ∃ (x y : ℝ), x + y = 30 ∧ 25 * x - 7.5 * y = 490 ∧ y = 8 :=
by {
  sorry
}

end contractor_absent_days_l9_9498


namespace fourth_child_sweets_l9_9173

theorem fourth_child_sweets (total_sweets : ℕ) (mother_sweets : ℕ) (child_sweets : ℕ) 
  (Y E T F: ℕ) (h1 : total_sweets = 120) (h2 : mother_sweets = total_sweets / 4) 
  (h3 : child_sweets = total_sweets - mother_sweets) 
  (h4 : E = 2 * Y) (h5 : T = F - 8) 
  (h6 : Y = (8 * (T + 6)) / 10) 
  (h7 : Y + E + (T + 6) + (F - 8) + F = child_sweets) : 
  F = 24 :=
by
  sorry

end fourth_child_sweets_l9_9173


namespace sum_eq_24_of_greatest_power_l9_9129

theorem sum_eq_24_of_greatest_power (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_b_gt_1 : b > 1) (h_a_pow_b_lt_500 : a^b < 500)
  (h_greatest : ∀ (x y : ℕ), (0 < x) → (0 < y) → (y > 1) → (x^y < 500) → (x^y ≤ a^b)) : a + b = 24 :=
  sorry

end sum_eq_24_of_greatest_power_l9_9129


namespace constant_is_arithmetic_l9_9448

def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem constant_is_arithmetic (a : ℕ → ℝ) (h : is_constant_sequence a) : is_arithmetic_sequence a := by
  sorry

end constant_is_arithmetic_l9_9448


namespace set_D_not_right_triangle_l9_9063

theorem set_D_not_right_triangle :
  let a := 11
  let b := 12
  let c := 15
  a ^ 2 + b ^ 2 ≠ c ^ 2
:=
by
  let a := 11
  let b := 12
  let c := 15
  sorry

end set_D_not_right_triangle_l9_9063


namespace expression_simplifies_to_neg_seven_l9_9683

theorem expression_simplifies_to_neg_seven (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
(h₃ : a + b + c = 0) (h₄ : ab + ac + bc ≠ 0) : 
    (a^7 + b^7 + c^7) / (abc * (ab + ac + bc)) = -7 :=
by
  sorry

end expression_simplifies_to_neg_seven_l9_9683


namespace range_of_a_l9_9315

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a * x > 0) → a < 1 :=
by
  sorry

end range_of_a_l9_9315


namespace sum_first_seven_terms_geometric_seq_l9_9852

theorem sum_first_seven_terms_geometric_seq :
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 2
  let S_7 := a * (1 - r^7) / (1 - r)
  S_7 = 127 / 192 := 
by
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 2
  let S_7 := a * (1 - r^7) / (1 - r)
  have h : S_7 = 127 / 192 := sorry
  exact h

end sum_first_seven_terms_geometric_seq_l9_9852


namespace steven_set_aside_pears_l9_9557

theorem steven_set_aside_pears :
  ∀ (apples pears grapes neededSeeds seedPerApple seedPerPear seedPerGrape : ℕ),
    apples = 4 →
    grapes = 9 →
    neededSeeds = 60 →
    seedPerApple = 6 →
    seedPerPear = 2 →
    seedPerGrape = 3 →
    (neededSeeds - 3) = (apples * seedPerApple + grapes * seedPerGrape + pears * seedPerPear) →
    pears = 3 :=
by
  intros apples pears grapes neededSeeds seedPerApple seedPerPear seedPerGrape
  intros h_apple h_grape h_needed h_seedApple h_seedPear h_seedGrape
  intros h_totalSeeds
  sorry

end steven_set_aside_pears_l9_9557


namespace polynomial_q_correct_l9_9697

noncomputable def polynomial_q (x : ℝ) : ℝ :=
  -x^6 + 12*x^5 + 9*x^4 + 14*x^3 - 5*x^2 + 17*x + 1

noncomputable def polynomial_rhs (x : ℝ) : ℝ :=
  x^6 + 12*x^5 + 13*x^4 + 14*x^3 + 17*x + 3

noncomputable def polynomial_2 (x : ℝ) : ℝ :=
  2*x^6 + 4*x^4 + 5*x^2 + 2

theorem polynomial_q_correct (x : ℝ) : 
  polynomial_q x = polynomial_rhs x - polynomial_2 x := 
by
  sorry

end polynomial_q_correct_l9_9697


namespace circle_radius_l9_9049

theorem circle_radius (r₂ : ℝ) : 
  (∃ r₁ : ℝ, r₁ = 5 ∧ (∀ d : ℝ, d = 7 → (d = r₁ + r₂ ∨ d = abs (r₁ - r₂)))) → (r₂ = 2 ∨ r₂ = 12) :=
by
  sorry

end circle_radius_l9_9049


namespace kolya_advantageous_methods_l9_9589

-- Define the context and conditions
variables (n : ℕ) (h₀ : n ≥ 2)
variables (a b : ℕ) (h₁ : a + b = 2*n + 1) (h₂ : a ≥ 2) (h₃ : b ≥ 2)

-- Define outcomes of the methods
def method1_outcome (a b : ℕ) := max a b + min (a - 1) (b - 1)
def method2_outcome (a b : ℕ) := min a b + min (a - 1) (b - 1)
def method3_outcome (a b : ℕ) := max (method1_outcome a b - 1) (method2_outcome a b - 1)

-- Prove which methods are the most and least advantageous
theorem kolya_advantageous_methods :
  method1_outcome a b >= method2_outcome a b ∧ method1_outcome a b >= method3_outcome a b :=
sorry

end kolya_advantageous_methods_l9_9589


namespace least_number_to_add_l9_9024

theorem least_number_to_add (n d : ℕ) (h₁ : n = 1054) (h₂ : d = 23) : ∃ x, (n + x) % d = 0 ∧ x = 4 := by
  sorry

end least_number_to_add_l9_9024


namespace term_with_largest_binomial_coeffs_and_largest_coefficient_l9_9236

theorem term_with_largest_binomial_coeffs_and_largest_coefficient :
  ∀ x : ℝ,
    (∀ k : ℕ, k = 2 → (Nat.choose 5 k) * (x ^ (2 / 3)) ^ (5 - k) * (3 * x ^ 2) ^ k = 90 * x ^ 6) ∧
    (∀ k : ℕ, k = 3 → (Nat.choose 5 k) * (x ^ (2 / 3)) ^ (5 - k) * (3 * x ^ 2) ^ k = 270 * x ^ (22 / 3)) ∧
    (∀ r : ℕ, r = 4 → (Nat.choose 5 4) * (x ^ (2 / 3)) ^ (5 - 4) * (3 * x ^ 2) ^ 4 = 405 * x ^ (26 / 3)) :=
by sorry

end term_with_largest_binomial_coeffs_and_largest_coefficient_l9_9236


namespace kayak_total_until_May_l9_9396

noncomputable def kayak_number (n : ℕ) : ℕ :=
  if n = 0 then 5
  else 3 * kayak_number (n - 1)

theorem kayak_total_until_May : kayak_number 0 + kayak_number 1 + kayak_number 2 + kayak_number 3 = 200 := by
  sorry

end kayak_total_until_May_l9_9396


namespace graph_passes_through_point_l9_9766

theorem graph_passes_through_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  ∃ p : ℝ × ℝ, p = (2, 0) ∧ ∀ x, (x = 2 → a ^ (x - 2) - 1 = 0) :=
by
  sorry

end graph_passes_through_point_l9_9766


namespace polynomial_solution_l9_9828

variable {R : Type*} [CommRing R]

theorem polynomial_solution (p : Polynomial R) :
  (∀ (a b c : R), 
    p.eval (a + b - 2 * c) + p.eval (b + c - 2 * a) + p.eval (c + a - 2 * b)
      = 3 * p.eval (a - b) + 3 * p.eval (b - c) + 3 * p.eval (c - a)
  ) →
  ∃ (a1 a2 : R), p = Polynomial.C a2 * Polynomial.X^2 + Polynomial.C a1 * Polynomial.X :=
by
  sorry

end polynomial_solution_l9_9828


namespace vasya_gift_ways_l9_9543

theorem vasya_gift_ways :
  let cars := 7
  let constructor_sets := 5
  (cars * constructor_sets) + (Nat.choose cars 2) + (Nat.choose constructor_sets 2) = 66 :=
by
  let cars := 7
  let constructor_sets := 5
  sorry

end vasya_gift_ways_l9_9543


namespace area_white_portion_l9_9038

/-- The dimensions of the sign --/
def sign_width : ℝ := 7
def sign_height : ℝ := 20

/-- The areas of letters "S", "A", "V", and "E" --/
def area_S : ℝ := 14
def area_A : ℝ := 16
def area_V : ℝ := 12
def area_E : ℝ := 12

/-- Calculate the total area of the sign --/
def total_area_sign : ℝ := sign_width * sign_height

/-- Calculate the total area covered by the letters --/
def total_area_letters : ℝ := area_S + area_A + area_V + area_E

/-- Calculate the area of the white portion of the sign --/
theorem area_white_portion : total_area_sign - total_area_letters = 86 := by
  sorry

end area_white_portion_l9_9038


namespace mul_65_35_eq_2275_l9_9072

theorem mul_65_35_eq_2275 : 65 * 35 = 2275 := by
  sorry

end mul_65_35_eq_2275_l9_9072


namespace inequality_solution_range_4_l9_9409

theorem inequality_solution_range_4 (a : ℝ) : 
  (∃ x : ℝ, |x - 2| - |x + 2| ≥ a) → a ≤ 4 :=
sorry

end inequality_solution_range_4_l9_9409


namespace total_wings_count_l9_9092

theorem total_wings_count (num_planes : ℕ) (wings_per_plane : ℕ) (h_planes : num_planes = 54) (h_wings : wings_per_plane = 2) : num_planes * wings_per_plane = 108 :=
by 
  sorry

end total_wings_count_l9_9092


namespace find_digits_of_six_two_digit_sum_equals_528_l9_9529

theorem find_digits_of_six_two_digit_sum_equals_528
  (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_digits : a < 10 ∧ b < 10 ∧ c < 10)
  (h_sum_six_numbers : (10 * a + b) + (10 * a + c) + (10 * b + c) + (10 * b + a) + (10 * c + a) + (10 * c + b) = 528) :
  (a = 7 ∧ b = 8 ∧ c = 9) := 
sorry

end find_digits_of_six_two_digit_sum_equals_528_l9_9529


namespace quadrilateral_with_equal_angles_is_parallelogram_l9_9281

axiom Quadrilateral (a b c d : Type) : Prop
axiom Parallelogram (a b c d : Type) : Prop
axiom equal_angles (a b c d : Type) : Prop

theorem quadrilateral_with_equal_angles_is_parallelogram 
  (a b c d : Type) 
  (q : Quadrilateral a b c d)
  (h : equal_angles a b c d) : Parallelogram a b c d := 
sorry

end quadrilateral_with_equal_angles_is_parallelogram_l9_9281


namespace inequality_solution_empty_solution_set_l9_9869

-- Problem 1: Prove the inequality and the solution range
theorem inequality_solution (x : ℝ) : (-7 < x ∧ x < 3) ↔ ( (x - 3)/(x + 7) < 0 ) :=
sorry

-- Problem 2: Prove the conditions for empty solution set
theorem empty_solution_set (a : ℝ) : (a > 0) ↔ ∀ x : ℝ, ¬ (x^2 - 4*a*x + 4*a^2 + a ≤ 0) :=
sorry

end inequality_solution_empty_solution_set_l9_9869


namespace correct_options_l9_9626

theorem correct_options :
  (1 + Real.tan 1) * (1 + Real.tan 44) = 2 ∧
  ¬((1 / Real.sin 10) - (Real.sqrt 3 / Real.cos 10) = 2) ∧
  (3 - Real.sin 70) / (2 - (Real.cos 10) ^ 2) = 2 ∧
  ¬(Real.tan 70 * Real.cos 10 * (Real.sqrt 3 * Real.tan 20 - 1) = 2) :=
sorry

end correct_options_l9_9626


namespace probability_two_white_balls_l9_9947

-- Definitions
def totalBalls : ℕ := 5
def whiteBalls : ℕ := 3
def blackBalls : ℕ := 2
def totalWaysToDrawTwoBalls : ℕ := Nat.choose totalBalls 2
def waysToDrawTwoWhiteBalls : ℕ := Nat.choose whiteBalls 2

-- Theorem statement
theorem probability_two_white_balls :
  (waysToDrawTwoWhiteBalls : ℚ) / totalWaysToDrawTwoBalls = 3 / 10 := by
  sorry

end probability_two_white_balls_l9_9947


namespace islanders_liars_l9_9647

theorem islanders_liars (n : ℕ) (h : n = 450) : (∃ L : ℕ, (L = 150 ∨ L = 450)) :=
sorry

end islanders_liars_l9_9647


namespace solve_inequality_l9_9004

theorem solve_inequality (x : ℝ) : (|2 * x - 1| < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by
  sorry

end solve_inequality_l9_9004


namespace find_max_z_plus_x_l9_9356

theorem find_max_z_plus_x : 
  (∃ (x y z t: ℝ), x^2 + y^2 = 4 ∧ z^2 + t^2 = 9 ∧ xt + yz ≥ 6 ∧ z + x = 5) :=
sorry

end find_max_z_plus_x_l9_9356


namespace books_at_end_of_year_l9_9071

def init_books : ℕ := 72
def monthly_books : ℕ := 12 -- 1 book each month for 12 months
def books_bought1 : ℕ := 5
def books_bought2 : ℕ := 2
def books_gift1 : ℕ := 1
def books_gift2 : ℕ := 4
def books_donated : ℕ := 12
def books_sold : ℕ := 3

theorem books_at_end_of_year :
  init_books + monthly_books + books_bought1 + books_bought2 + books_gift1 + books_gift2 - books_donated - books_sold = 81 :=
by
  sorry

end books_at_end_of_year_l9_9071


namespace find_x_l9_9515

noncomputable def is_solution (x : ℝ) : Prop :=
   (⌊x * ⌊x⌋⌋ = 29)

theorem find_x (x : ℝ) (h : is_solution x) : 5.8 ≤ x ∧ x < 6 :=
sorry

end find_x_l9_9515


namespace abc_product_l9_9030

theorem abc_product (A B C D : ℕ) 
  (h1 : A + B + C + D = 64)
  (h2 : A + 3 = B - 3)
  (h3 : A + 3 = C * 3)
  (h4 : A + 3 = D / 3) :
  A * B * C * D = 19440 := 
by
  sorry

end abc_product_l9_9030


namespace line_solutions_l9_9894

-- Definition for points
def point := ℝ × ℝ

-- Conditions for lines and points
def line1 (p : point) : Prop := 3 * p.1 + 4 * p.2 = 2
def line2 (p : point) : Prop := 2 * p.1 + p.2 = -2
def line3 : Prop := ∃ p : point, line1 p ∧ line2 p

def lineL (p : point) : Prop := 2 * p.1 + p.2 = -2 -- Line l we need to prove
def perp_lineL : Prop := ∃ p : point, lineL p ∧ p.1 - 2 * p.2 = 1

-- Symmetry condition for the line
def symmetric_line (p : point) : Prop := 2 * p.1 + p.2 = 2 -- Symmetric line we need to prove

-- Main theorem to prove
theorem line_solutions :
  line3 →
  perp_lineL →
  (∀ p, lineL p ↔ 2 * p.1 + p.2 = -2) ∧
  (∀ p, symmetric_line p ↔ 2 * p.1 + p.2 = 2) :=
sorry

end line_solutions_l9_9894


namespace gum_pieces_bought_correct_l9_9637

-- Define initial number of gum pieces
def initial_gum_pieces : ℕ := 10

-- Define number of friends Adrianna gave gum to
def friends_given_gum : ℕ := 11

-- Define the number of pieces Adrianna has left
def remaining_gum_pieces : ℕ := 2

-- Define a function to calculate the number of gum pieces Adrianna bought at the store
def gum_pieces_bought (initial_gum : ℕ) (given_gum : ℕ) (remaining_gum : ℕ) : ℕ :=
  (given_gum + remaining_gum) - initial_gum

-- Now state the theorem to prove the number of pieces bought is 3
theorem gum_pieces_bought_correct : 
  gum_pieces_bought initial_gum_pieces friends_given_gum remaining_gum_pieces = 3 :=
by
  sorry

end gum_pieces_bought_correct_l9_9637


namespace students_count_geometry_history_science_l9_9378

noncomputable def number_of_students (geometry_only history_only science_only 
                                      geometry_and_history geometry_and_science : ℕ) : ℕ :=
  geometry_only + history_only + science_only

theorem students_count_geometry_history_science (geometry_total history_only science_only 
                                                 geometry_and_history geometry_and_science : ℕ) :
  geometry_total = 30 →
  geometry_and_history = 15 →
  history_only = 15 →
  geometry_and_science = 8 →
  science_only = 10 →
  number_of_students (geometry_total - geometry_and_history - geometry_and_science)
                     history_only
                     science_only = 32 :=
by
  sorry

end students_count_geometry_history_science_l9_9378


namespace sum_of_bases_l9_9681

theorem sum_of_bases (R₁ R₂ : ℕ) 
    (h1 : (4 * R₁ + 5) / (R₁^2 - 1) = (3 * R₂ + 4) / (R₂^2 - 1))
    (h2 : (5 * R₁ + 4) / (R₁^2 - 1) = (4 * R₂ + 3) / (R₂^2 - 1)) : 
    R₁ + R₂ = 23 := 
sorry

end sum_of_bases_l9_9681


namespace sequence_arithmetic_and_find_an_l9_9512

theorem sequence_arithmetic_and_find_an (a : ℕ → ℝ)
  (h1 : a 9 = 1 / 7)
  (h2 : ∀ n, a (n + 1) = a n / (3 * a n + 1)) :
  (∀ n, 1 / a (n + 1) = 3 + 1 / a n) ∧ (∀ n, a n = 1 / (3 * n - 20)) :=
by
  sorry

end sequence_arithmetic_and_find_an_l9_9512


namespace bob_spends_more_time_l9_9085

def pages := 760
def time_per_page_bob := 45
def time_per_page_chandra := 30
def total_time_bob := pages * time_per_page_bob
def total_time_chandra := pages * time_per_page_chandra
def time_difference := total_time_bob - total_time_chandra

theorem bob_spends_more_time : time_difference = 11400 :=
by
  sorry

end bob_spends_more_time_l9_9085


namespace product_of_five_numbers_is_256_l9_9605

def possible_numbers : Set ℕ := {1, 2, 4}

theorem product_of_five_numbers_is_256 
  (x1 x2 x3 x4 x5 : ℕ) 
  (h1 : x1 ∈ possible_numbers) 
  (h2 : x2 ∈ possible_numbers) 
  (h3 : x3 ∈ possible_numbers) 
  (h4 : x4 ∈ possible_numbers) 
  (h5 : x5 ∈ possible_numbers) : 
  x1 * x2 * x3 * x4 * x5 = 256 :=
sorry

end product_of_five_numbers_is_256_l9_9605


namespace rachel_math_homework_pages_l9_9134

-- Define the number of pages of math homework and reading homework
def pagesReadingHomework : ℕ := 4

theorem rachel_math_homework_pages (M : ℕ) (h1 : M + 1 = pagesReadingHomework) : M = 3 :=
by
  sorry

end rachel_math_homework_pages_l9_9134


namespace cos_arcsin_l9_9083

theorem cos_arcsin (h3: ℝ) (h5: ℝ) (h_op: h3 = 3) (h_hyp: h5 = 5) : 
  Real.cos (Real.arcsin (3 / 5)) = 4 / 5 := 
sorry

end cos_arcsin_l9_9083


namespace geometric_sequence_common_ratio_l9_9345

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q))  -- Sum of geometric series
  (h2 : a 3 = S 3 + 1) : q = 3 :=
by sorry

end geometric_sequence_common_ratio_l9_9345


namespace negation_of_proposition_l9_9679

theorem negation_of_proposition (x : ℝ) : 
  ¬ (∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1 < 0) := 
sorry

end negation_of_proposition_l9_9679


namespace find_a2_plus_b2_l9_9246

theorem find_a2_plus_b2 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h: 8 * a^a * b^b = 27 * a^b * b^a) : a^2 + b^2 = 117 := by
  sorry

end find_a2_plus_b2_l9_9246


namespace smallest_perfect_square_5336100_l9_9906

def smallestPerfectSquareDivisibleBy (a b c d : Nat) (s : Nat) : Prop :=
  ∃ k : Nat, s = k * k ∧ s % a = 0 ∧ s % b = 0 ∧ s % c = 0 ∧ s % d = 0

theorem smallest_perfect_square_5336100 :
  smallestPerfectSquareDivisibleBy 6 14 22 30 5336100 :=
sorry

end smallest_perfect_square_5336100_l9_9906


namespace abs_eq_non_pos_2x_plus_4_l9_9883

-- Condition: |2x + 4| = 0
-- Conclusion: x = -2
theorem abs_eq_non_pos_2x_plus_4 (x : ℝ) : (|2 * x + 4| = 0) → x = -2 :=
by
  intro h
  -- Here lies the proof, but we use sorry to indicate the unchecked part.
  sorry

end abs_eq_non_pos_2x_plus_4_l9_9883


namespace rational_sum_zero_l9_9457

theorem rational_sum_zero {a b c : ℚ} (h : (a + b + c) * (a + b - c) = 4 * c^2) : a + b = 0 := 
sorry

end rational_sum_zero_l9_9457


namespace radius_of_inscribed_circle_l9_9359

variable (A p s r : ℝ)

theorem radius_of_inscribed_circle (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 := by
  sorry

end radius_of_inscribed_circle_l9_9359


namespace second_discount_percentage_l9_9271

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ) :
  original_price = 10000 →
  first_discount = 0.20 →
  final_price = 6840 →
  second_discount = 14.5 :=
by
  sorry

end second_discount_percentage_l9_9271


namespace intersection_M_N_l9_9743

open Set

def M := { x : ℝ | 0 < x ∧ x < 3 }
def N := { x : ℝ | x^2 - 5 * x + 4 ≥ 0 }

theorem intersection_M_N :
  { x | x ∈ M ∧ x ∈ N } = { x | 0 < x ∧ x ≤ 1 } :=
sorry

end intersection_M_N_l9_9743


namespace felix_chopped_down_trees_l9_9238

theorem felix_chopped_down_trees
  (sharpening_cost : ℕ)
  (trees_per_sharpening : ℕ)
  (total_spent : ℕ)
  (times_sharpened : ℕ)
  (trees_chopped_down : ℕ)
  (h1 : sharpening_cost = 5)
  (h2 : trees_per_sharpening = 13)
  (h3 : total_spent = 35)
  (h4 : times_sharpened = total_spent / sharpening_cost)
  (h5 : trees_chopped_down = trees_per_sharpening * times_sharpened) :
  trees_chopped_down ≥ 91 :=
by
  sorry

end felix_chopped_down_trees_l9_9238


namespace distance_from_negative_two_is_three_l9_9082

theorem distance_from_negative_two_is_three (x : ℝ) : abs (x + 2) = 3 → (x = -5) ∨ (x = 1) :=
  sorry

end distance_from_negative_two_is_three_l9_9082


namespace divide_24kg_into_parts_l9_9597

theorem divide_24kg_into_parts (W : ℕ) (part1 part2 : ℕ) (h_sum : part1 + part2 = 24) :
  (part1 = 9 ∧ part2 = 15) ∨ (part1 = 15 ∧ part2 = 9) :=
by
  sorry

end divide_24kg_into_parts_l9_9597


namespace true_statement_l9_9873

variables {Plane Line : Type}
variables (α β γ : Plane) (a b m n : Line)

-- Definitions for parallel and perpendicular relationships
def parallel (x y : Line) : Prop := sorry
def perpendicular (x y : Line) : Prop := sorry
def subset (l : Line) (p : Plane) : Prop := sorry
def intersect_line (p q : Plane) : Line := sorry

-- Given conditions for the problem
variables (h1 : (α ≠ β)) (h2 : (parallel α β))
variables (h3 : (intersect_line α γ = a)) (h4 : (intersect_line β γ = b))

-- Statement verifying the true condition based on the above givens
theorem true_statement : parallel a b :=
by sorry

end true_statement_l9_9873


namespace find_room_dimension_l9_9664

noncomputable def unknown_dimension_of_room 
  (cost_per_sq_ft : ℕ)
  (total_cost : ℕ)
  (w : ℕ)
  (l : ℕ)
  (h : ℕ)
  (door_h : ℕ)
  (door_w : ℕ)
  (window_h : ℕ)
  (window_w : ℕ)
  (num_windows : ℕ) : ℕ := sorry

theorem find_room_dimension :
  unknown_dimension_of_room 10 9060 25 15 12 6 3 4 3 3 = 25 :=
sorry

end find_room_dimension_l9_9664


namespace raft_drift_time_l9_9832

-- Define the conditions from the problem
variable (distance : ℝ := 1)
variable (steamboat_time : ℝ := 1) -- in hours
variable (motorboat_time : ℝ := 3 / 4) -- 45 minutes in hours
variable (motorboat_speed_ratio : ℝ := 2)

-- Variables for speeds
variable (vs vf : ℝ)

-- Conditions: the speeds and conditions of traveling from one village to another
variable (steamboat_eqn : vs + vf = distance / steamboat_time := by sorry)
variable (motorboat_eqn : (2 * vs) + vf = distance / motorboat_time := by sorry)

-- Time for the raft to travel the distance
theorem raft_drift_time : 90 = (distance / vf) * 60 := by
  -- Proof comes here
  sorry

end raft_drift_time_l9_9832


namespace tom_climbing_time_l9_9465

theorem tom_climbing_time (elizabeth_time : ℕ) (multiplier : ℕ) 
  (h1 : elizabeth_time = 30) (h2 : multiplier = 4) : (elizabeth_time * multiplier) / 60 = 2 :=
by
  sorry

end tom_climbing_time_l9_9465


namespace number_of_outcomes_exactly_two_evening_l9_9288

theorem number_of_outcomes_exactly_two_evening (chickens : Finset ℕ) (h_chickens : chickens.card = 4) 
    (day_places evening_places : ℕ) (h_day_places : day_places = 2) (h_evening_places : evening_places = 3) :
    ∃ n, n = (chickens.card.choose 2) ∧ n = 6 :=
by
  sorry

end number_of_outcomes_exactly_two_evening_l9_9288


namespace radius_calculation_l9_9772

noncomputable def radius_of_circle (n : ℕ) : ℝ :=
if 2 ≤ n ∧ n ≤ 11 then
  if n ≤ 7 then 1 else
  if n = 8 then 1.15 else
  if n = 9 then 1.30 else
  if n = 10 then 1.46 else
  1.61
else
  0  -- Outside the specified range

theorem radius_calculation (n : ℕ) (hn : 2 ≤ n ∧ n ≤ 11) :
  radius_of_circle n =
  if n ≤ 7 then 1 else
  if n = 8 then 1.15 else
  if n = 9 then 1.30 else
  if n = 10 then 1.46 else
  1.61 :=
sorry

end radius_calculation_l9_9772


namespace bryden_receives_10_dollars_l9_9468

-- Define the face value of one quarter
def face_value_quarter : ℝ := 0.25

-- Define the number of quarters Bryden has
def num_quarters : ℕ := 8

-- Define the multiplier for 500%
def multiplier : ℝ := 5

-- Calculate the total face value of eight quarters
def total_face_value : ℝ := num_quarters * face_value_quarter

-- Calculate the amount Bryden will receive
def amount_received : ℝ := total_face_value * multiplier

-- The proof goal: Bryden will receive 10 dollars
theorem bryden_receives_10_dollars : amount_received = 10 :=
by
  sorry

end bryden_receives_10_dollars_l9_9468


namespace option_b_correct_l9_9186

theorem option_b_correct (a : ℝ) : (a ^ 3) * (a ^ 2) = a ^ 5 := 
by
  sorry

end option_b_correct_l9_9186


namespace largest_4_digit_divisible_by_50_l9_9923

-- Define the condition for a 4-digit number
def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define the largest 4-digit number
def largest_4_digit : ℕ := 9999

-- Define the property that a number is exactly divisible by 50
def divisible_by_50 (n : ℕ) : Prop := n % 50 = 0

-- Main statement to be proved
theorem largest_4_digit_divisible_by_50 :
  ∃ n, is_4_digit n ∧ divisible_by_50 n ∧ ∀ m, is_4_digit m → divisible_by_50 m → m ≤ n ∧ n = 9950 :=
by
  sorry

end largest_4_digit_divisible_by_50_l9_9923


namespace ants_species_A_count_l9_9882

theorem ants_species_A_count (a b : ℕ) (h1 : a + b = 30) (h2 : 2^5 * a + 3^5 * b = 3281) : 32 * a = 608 :=
by
  sorry

end ants_species_A_count_l9_9882


namespace min_value_of_expression_min_value_achieved_l9_9296

theorem min_value_of_expression (x : ℝ) (h : x > 0) : 
  (x + 3 / (x + 1)) ≥ 2 * Real.sqrt 3 - 1 := 
sorry

theorem min_value_achieved (x : ℝ) (h : x = Real.sqrt 3 - 1) : 
  (x + 3 / (x + 1)) = 2 * Real.sqrt 3 - 1 := 
sorry

end min_value_of_expression_min_value_achieved_l9_9296


namespace simplified_expression_l9_9917

variable {x y : ℝ}

theorem simplified_expression 
  (P : ℝ := x^2 + y^2) 
  (Q : ℝ := x^2 - y^2) : 
  ( (P + 3 * Q) / (P - Q) - (P - 3 * Q) / (P + Q) ) = (2 * x^4 - y^4) / (x^2 * y^2) := 
  by sorry

end simplified_expression_l9_9917


namespace subtraction_solution_l9_9963

noncomputable def x : ℝ := 47.806

theorem subtraction_solution :
  (3889 : ℝ) + 12.808 - x = 3854.002 :=
by
  sorry

end subtraction_solution_l9_9963


namespace freq_distribution_correct_l9_9847

variable (freqTable_isForm : Prop)
variable (freqHistogram_isForm : Prop)
variable (freqTable_isAccurate : Prop)
variable (freqHistogram_isIntuitive : Prop)

theorem freq_distribution_correct :
  ((freqTable_isForm ∧ freqHistogram_isForm) ∧
   (freqTable_isAccurate ∧ freqHistogram_isIntuitive)) →
  True :=
by
  intros _
  exact trivial

end freq_distribution_correct_l9_9847


namespace symmetry_of_transformed_graphs_l9_9273

noncomputable def y_eq_f_x_symmetric_line (f : ℝ → ℝ) : Prop :=
∀ (x : ℝ), f (x - 19) = f (99 - x) ↔ x = 59

theorem symmetry_of_transformed_graphs (f : ℝ → ℝ) :
  y_eq_f_x_symmetric_line f :=
by {
  sorry
}

end symmetry_of_transformed_graphs_l9_9273


namespace remaining_load_after_three_deliveries_l9_9103

def initial_load : ℝ := 50000
def unload_first_store (load : ℝ) : ℝ := load - 0.10 * load
def unload_second_store (load : ℝ) : ℝ := load - 0.20 * load
def unload_third_store (load : ℝ) : ℝ := load - 0.15 * load

theorem remaining_load_after_three_deliveries : 
  unload_third_store (unload_second_store (unload_first_store initial_load)) = 30600 := 
by
  sorry

end remaining_load_after_three_deliveries_l9_9103


namespace books_loaned_out_l9_9427

theorem books_loaned_out (initial_books returned_percent final_books : ℕ) (h1 : initial_books = 75) (h2 : returned_percent = 65) (h3 : final_books = 61) : 
  ∃ x : ℕ, initial_books - final_books = x - (returned_percent * x / 100) ∧ x = 40 :=
by {
  sorry 
}

end books_loaned_out_l9_9427


namespace shopkeeper_loss_percent_l9_9402

noncomputable def loss_percentage (cost_price profit_percent theft_percent: ℝ) :=
  let selling_price := cost_price * (1 + profit_percent / 100)
  let value_lost := cost_price * (theft_percent / 100)
  let remaining_cost_price := cost_price * (1 - theft_percent / 100)
  (value_lost / remaining_cost_price) * 100

theorem shopkeeper_loss_percent
  (cost_price : ℝ)
  (profit_percent : ℝ := 10)
  (theft_percent : ℝ := 20)
  (expected_loss_percent : ℝ := 25)
  (h1 : profit_percent = 10) (h2 : theft_percent = 20) : 
  loss_percentage cost_price profit_percent theft_percent = expected_loss_percent := 
by
  sorry

end shopkeeper_loss_percent_l9_9402


namespace medicine_liquid_poured_l9_9508

theorem medicine_liquid_poured (x : ℝ) (h : 63 * (1 - x / 63) * (1 - x / 63) = 28) : x = 18 :=
by
  sorry

end medicine_liquid_poured_l9_9508


namespace log_27_gt_point_53_l9_9277

open Real

theorem log_27_gt_point_53 :
  log 27 > 0.53 :=
by
  sorry

end log_27_gt_point_53_l9_9277


namespace quadratic_passing_point_calc_l9_9381

theorem quadratic_passing_point_calc :
  (∀ (x y : ℤ), y = 2 * x ^ 2 - 3 * x + 4 → ∃ (x' y' : ℤ), x' = 2 ∧ y' = 6) →
  (2 * 2 - 3 * (-3) + 4 * 4 = 29) :=
by
  intro h
  -- The corresponding proof would follow by providing the necessary steps.
  -- For now, let's just use sorry to meet the requirement.
  sorry

end quadratic_passing_point_calc_l9_9381


namespace no_positive_integers_m_n_l9_9763

theorem no_positive_integers_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  m^3 + 11^3 ≠ n^3 :=
sorry

end no_positive_integers_m_n_l9_9763


namespace butterflies_in_the_garden_l9_9483

variable (total_butterflies : Nat) (fly_away : Nat)

def butterflies_left (total_butterflies : Nat) (fly_away : Nat) : Nat :=
  total_butterflies - fly_away

theorem butterflies_in_the_garden :
  (total_butterflies = 9) → (fly_away = 1 / 3 * total_butterflies) → butterflies_left total_butterflies fly_away = 6 :=
by
  intro h1 h2
  sorry

end butterflies_in_the_garden_l9_9483


namespace blue_pigment_percentage_l9_9006

-- Define weights and pigments in the problem
variables (S G : ℝ)
-- Conditions
def sky_blue_paint := 0.9 * S = 4.5
def total_weight := S + G = 10
def sky_blue_blue_pigment := 0.1
def green_blue_pigment := 0.7

-- Prove the percentage of blue pigment in brown paint is 40%
theorem blue_pigment_percentage :
  sky_blue_paint S →
  total_weight S G →
  (0.1 * (4.5 / 0.9) + 0.7 * (10 - (4.5 / 0.9))) / 10 * 100 = 40 :=
by
  intros h1 h2
  sorry

end blue_pigment_percentage_l9_9006


namespace train_length_l9_9154

noncomputable def relative_speed_kmh (vA vB : ℝ) : ℝ :=
  vA - vB

noncomputable def relative_speed_mps (relative_speed_kmh : ℝ) : ℝ :=
  relative_speed_kmh * (5 / 18)

noncomputable def distance_covered (relative_speed_mps : ℝ) (time_s : ℝ) : ℝ :=
  relative_speed_mps * time_s

theorem train_length (vA_kmh : ℝ) (vB_kmh : ℝ) (time_s : ℝ) (L : ℝ) 
  (h1 : vA_kmh = 42) (h2 : vB_kmh = 36) (h3 : time_s = 36) 
  (h4 : distance_covered (relative_speed_mps (relative_speed_kmh vA_kmh vB_kmh)) time_s = 2 * L) :
  L = 30 :=
by
  sorry

end train_length_l9_9154


namespace average_age_new_students_l9_9749

theorem average_age_new_students (O A_old A_new_avg A_new : ℕ) 
  (hO : O = 8) 
  (hA_old : A_old = 40) 
  (hA_new_avg : A_new_avg = 36)
  (h_total_age_before : O * A_old = 8 * 40)
  (h_total_age_after : (O + 8) * A_new_avg = 16 * 36)
  (h_age_new_students : (16 * 36) - (8 * 40) = A_new * 8) :
  A_new = 32 := 
by 
  sorry

end average_age_new_students_l9_9749


namespace minimum_inhabitants_to_ask_l9_9230

def knights_count : ℕ := 50
def civilians_count : ℕ := 15

theorem minimum_inhabitants_to_ask (knights civilians : ℕ) (h_knights : knights = knights_count) (h_civilians : civilians = civilians_count) :
  ∃ n, (∀ asked : ℕ, (asked ≥ n) → asked - civilians > civilians) ∧ n = 31 :=
by
  sorry

end minimum_inhabitants_to_ask_l9_9230


namespace inequality_solution_l9_9507

theorem inequality_solution {x : ℝ} (h : |x + 3| - |x - 1| > 0) : x > -1 :=
sorry

end inequality_solution_l9_9507


namespace village_population_rate_l9_9207

theorem village_population_rate
    (population_X : ℕ := 68000)
    (population_Y : ℕ := 42000)
    (increase_Y : ℕ := 800)
    (years : ℕ := 13) :
  ∃ R : ℕ, population_X - years * R = population_Y + years * increase_Y ∧ R = 1200 :=
by
  exists 1200
  sorry

end village_population_rate_l9_9207


namespace each_member_score_l9_9610

def total_members : ℝ := 5.0
def members_didnt_show_up : ℝ := 2.0
def total_points_by_showed_up_members : ℝ := 6.0

theorem each_member_score
  (h1 : total_members - members_didnt_show_up = 3.0)
  (h2 : total_points_by_showed_up_members = 6.0) :
  total_points_by_showed_up_members / (total_members - members_didnt_show_up) = 2.0 :=
sorry

end each_member_score_l9_9610


namespace f_even_of_g_odd_l9_9762

theorem f_even_of_g_odd (g : ℝ → ℝ) (f : ℝ → ℝ) (h1 : ∀ x, g (-x) = -g x) (h2 : ∀ x, f x = |g (x^5)|) : ∀ x, f (-x) = f x := 
by
  sorry

end f_even_of_g_odd_l9_9762


namespace max_students_on_field_trip_l9_9506

theorem max_students_on_field_trip 
  (bus_cost : ℕ := 100)
  (bus_capacity : ℕ := 25)
  (student_admission_cost_high : ℕ := 10)
  (student_admission_cost_low : ℕ := 8)
  (discount_threshold : ℕ := 20)
  (teacher_cost : ℕ := 0)
  (budget : ℕ := 350) :
  max_students ≤ bus_capacity ↔ bus_cost + 
  (if max_students ≥ discount_threshold then max_students * student_admission_cost_low
  else max_students * student_admission_cost_high) 
   ≤ budget := 
sorry

end max_students_on_field_trip_l9_9506


namespace adult_ticket_cost_l9_9218

variables (x : ℝ)

-- Conditions
def total_tickets := 510
def senior_tickets := 327
def senior_ticket_cost := 15
def total_receipts := 8748

-- Calculation based on the conditions
def adult_tickets := total_tickets - senior_tickets
def senior_receipts := senior_tickets * senior_ticket_cost
def adult_receipts := total_receipts - senior_receipts

-- Define the problem as an assertion to prove
theorem adult_ticket_cost :
  adult_receipts / adult_tickets = 21 := by
  -- Proof steps will go here, but for now, we'll use sorry.
  sorry

end adult_ticket_cost_l9_9218


namespace train_speed_l9_9476

def train_length : ℝ := 800
def crossing_time : ℝ := 12
def expected_speed : ℝ := 66.67 

theorem train_speed (h_len : train_length = 800) (h_time : crossing_time = 12) : 
  train_length / crossing_time = expected_speed := 
by {
  sorry
}

end train_speed_l9_9476


namespace find_cost_price_l9_9688

variable (CP : ℝ)

def selling_price (CP : ℝ) := CP * 1.40

theorem find_cost_price (h : selling_price CP = 1680) : CP = 1200 :=
by
  sorry

end find_cost_price_l9_9688


namespace smallest_n_inequality_l9_9841

theorem smallest_n_inequality :
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
           (∀ m : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) → n ≤ m) ∧
           n = 4 :=
by
  let n := 4
  sorry

end smallest_n_inequality_l9_9841


namespace no_such_natural_numbers_l9_9547

theorem no_such_natural_numbers :
  ¬(∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧
  (b ∣ a^2 - 1) ∧ (c ∣ a^2 - 1) ∧
  (a ∣ b^2 - 1) ∧ (c ∣ b^2 - 1) ∧
  (a ∣ c^2 - 1) ∧ (b ∣ c^2 - 1)) :=
by sorry

end no_such_natural_numbers_l9_9547


namespace sequences_of_lemon_recipients_l9_9354

theorem sequences_of_lemon_recipients :
  let students := 15
  let days := 5
  let total_sequences := students ^ days
  total_sequences = 759375 :=
by
  let students := 15
  let days := 5
  let total_sequences := students ^ days
  have h : total_sequences = 759375 := by sorry
  exact h

end sequences_of_lemon_recipients_l9_9354


namespace tangent_line_at_point_l9_9840

theorem tangent_line_at_point
  (x y : ℝ)
  (h_curve : y = x^3 - 3 * x^2 + 1)
  (h_point : (x, y) = (1, -1)) :
  ∃ m b : ℝ, (m = -3) ∧ (b = 2) ∧ (y = m * x + b) :=
sorry

end tangent_line_at_point_l9_9840


namespace meet_without_contact_probability_l9_9040

noncomputable def prob_meet_without_contact : ℝ :=
  let total_area := 1
  let outside_area := (1 / 8) * 2
  total_area - outside_area

theorem meet_without_contact_probability :
  prob_meet_without_contact = 3 / 4 :=
by
  sorry

end meet_without_contact_probability_l9_9040


namespace mean_of_five_numbers_l9_9108

theorem mean_of_five_numbers (x1 x2 x3 x4 x5 : ℚ) (h_sum : x1 + x2 + x3 + x4 + x5 = 1/3) : 
  (x1 + x2 + x3 + x4 + x5) / 5 = 1/15 :=
by 
  sorry

end mean_of_five_numbers_l9_9108


namespace island_solution_l9_9357

-- Definitions based on conditions
def is_liar (n : ℕ) (m : ℕ) : Prop := n = m + 2 ∨ n = m - 2
def is_truth_teller (n : ℕ) (m : ℕ) : Prop := n = m

-- Residents' statements
def first_resident_statement (liars : ℕ) (truth_tellers : ℕ) : Prop :=
  is_truth_teller liars 1001 ∧ is_truth_teller truth_tellers 1002 ∨
  is_liar liars 1001 ∧ is_liar truth_tellers 1002

def second_resident_statement (liars : ℕ) (truth_tellers : ℕ) : Prop :=
  is_truth_teller liars 1000 ∧ is_truth_teller truth_tellers 999 ∨
  is_liar liars 1000 ∧ is_liar truth_tellers 999

-- Proving the correct number of liars and truth-tellers, and identifying the residents
theorem island_solution :
  ∃ (liars : ℕ) (truth_tellers : ℕ),
    first_resident_statement (liars + 1) (truth_tellers + 1) ∧
    second_resident_statement (liars + 1) (truth_tellers + 1) ∧
    liars = 1000 ∧ truth_tellers = 1000 ∧
    first_resident_statement liars truth_tellers ∧ second_resident_statement liars truth_tellers :=
by
  sorry

end island_solution_l9_9357


namespace max_pieces_of_pie_l9_9653

theorem max_pieces_of_pie : ∃ (PIE PIECE : ℕ), 10000 ≤ PIE ∧ PIE < 100000
  ∧ 10000 ≤ PIECE ∧ PIECE < 100000
  ∧ ∃ (n : ℕ), n = 7 ∧ PIE = n * PIECE := by
  sorry

end max_pieces_of_pie_l9_9653


namespace max_candies_per_student_l9_9282

theorem max_candies_per_student (n_students : ℕ) (mean_candies : ℕ) (min_candies : ℕ) (max_candies : ℕ) :
  n_students = 50 ∧
  mean_candies = 7 ∧
  min_candies = 1 ∧
  max_candies = 20 →
  ∃ m : ℕ, m ≤ max_candies :=
by
  intro h
  use 20
  sorry

end max_candies_per_student_l9_9282


namespace inequality_for_positive_numbers_l9_9214

theorem inequality_for_positive_numbers (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  (a + b) * (a^4 + b^4) ≥ (a^2 + b^2) * (a^3 + b^3) :=
sorry

end inequality_for_positive_numbers_l9_9214


namespace total_cost_l9_9608

variables (p e n : ℕ) -- represent the costs of pencil, eraser, and notebook in cents

-- Given conditions
def conditions : Prop :=
  9 * p + 7 * e + 4 * n = 220 ∧
  p > n ∧ n > e ∧ e > 0

-- Prove the total cost
theorem total_cost (h : conditions p e n) : p + n + e = 26 :=
sorry

end total_cost_l9_9608


namespace citizen_income_l9_9261

theorem citizen_income (I : ℝ) 
  (h1 : I > 0)
  (h2 : 0.12 * 40000 + 0.20 * (I - 40000) = 8000) : 
  I = 56000 := 
sorry

end citizen_income_l9_9261


namespace find_m_pure_imaginary_l9_9870

theorem find_m_pure_imaginary (m : ℝ) (h : m^2 + m - 2 + (m^2 - 1) * I = (0 : ℝ) + (m^2 - 1) * I) :
  m = -2 :=
by {
  sorry
}

end find_m_pure_imaginary_l9_9870


namespace product_of_odd_primes_mod_32_l9_9755

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l9_9755


namespace x3_y3_sum_sq_sq_leq_4a10_equality_holds_when_x_eq_y_l9_9293

-- Conditions: x, y are positive real numbers and x + y = 2a
variables {x y a : ℝ}
variable (hxy : x + y = 2 * a)
variable (hx_pos : 0 < x)
variable (hy_pos : 0 < y)

-- Math proof problem: Prove the inequality
theorem x3_y3_sum_sq_sq_leq_4a10 : 
  x^3 * y^3 * (x^2 + y^2)^2 ≤ 4 * a^10 :=
by sorry

-- Equality condition: Equality holds when x = y
theorem equality_holds_when_x_eq_y (h : x = y) :
  x^3 * y^3 * (x^2 + y^2)^2 = 4 * a^10 :=
by sorry

end x3_y3_sum_sq_sq_leq_4a10_equality_holds_when_x_eq_y_l9_9293


namespace part1_solution_set_a_eq_1_part2_range_of_values_a_l9_9995

def f (x a : ℝ) : ℝ := |(2 * x - a)| + |(x - 3 * a)|

theorem part1_solution_set_a_eq_1 :
  ∀ x : ℝ, f x 1 ≤ 4 ↔ 0 ≤ x ∧ x ≤ 2 :=
by sorry

theorem part2_range_of_values_a :
  ∀ a : ℝ, (∀ x : ℝ, f x a ≥ |(x - a / 2)| + a^2 + 1) ↔
    ((-2 : ℝ) ≤ a ∧ a ≤ -1 / 2) ∨ (1 / 2 ≤ a ∧ a ≤ 2) :=
by sorry

end part1_solution_set_a_eq_1_part2_range_of_values_a_l9_9995


namespace tracy_feeds_dogs_times_per_day_l9_9459

theorem tracy_feeds_dogs_times_per_day : 
  let cups_per_meal_per_dog := 1.5
  let dogs := 2
  let total_pounds_per_day := 4
  let cups_per_pound := 2.25
  (total_pounds_per_day * cups_per_pound) / (dogs * cups_per_meal_per_dog) = 3 :=
by
  sorry

end tracy_feeds_dogs_times_per_day_l9_9459


namespace price_per_liter_l9_9611

theorem price_per_liter (cost : ℕ) (bottles : ℕ) (liters_per_bottle : ℕ) (total_cost : ℕ) (total_liters : ℕ) :
  bottles = 6 → liters_per_bottle = 2 → total_cost = 12 → total_liters = 12 → cost = total_cost / total_liters → cost = 1 :=
by
  intros h_bottles h_liters_per_bottle h_total_cost h_total_liters h_cost_div
  sorry

end price_per_liter_l9_9611


namespace no_integer_coordinates_between_A_and_B_l9_9525

section
variable (A B : ℤ × ℤ)
variable (Aeq : A = (2, 3))
variable (Beq : B = (50, 305))

theorem no_integer_coordinates_between_A_and_B :
  (∀ P : ℤ × ℤ, P.1 > 2 ∧ P.1 < 50 ∧ P.2 = (151 * P.1 - 230) / 24 → False) :=
by
  sorry
end

end no_integer_coordinates_between_A_and_B_l9_9525


namespace min_value_quadratic_expr_l9_9484

-- Define the quadratic function
def quadratic_expr (x : ℝ) : ℝ := 8 * x^2 - 24 * x + 1729

-- State the theorem to prove the minimum value
theorem min_value_quadratic_expr : (∃ x : ℝ, ∀ y : ℝ, quadratic_expr y ≥ quadratic_expr x) ∧ ∃ x : ℝ, quadratic_expr x = 1711 :=
by
  -- The proof will go here
  sorry

end min_value_quadratic_expr_l9_9484


namespace rectangle_area_l9_9856

theorem rectangle_area (a : ℕ) (h : 2 * (3 * a + 2 * a) = 160) : 3 * a * 2 * a = 1536 :=
by
  sorry

end rectangle_area_l9_9856


namespace sector_area_correct_l9_9194

noncomputable def sector_area (r α : ℝ) : ℝ :=
  (1 / 2) * r^2 * α

theorem sector_area_correct :
  sector_area 3 2 = 9 :=
by
  sorry

end sector_area_correct_l9_9194


namespace sum_of_distinct_nums_l9_9527

theorem sum_of_distinct_nums (m n p q : ℕ) (hmn : m ≠ n) (hmp : m ≠ p) (hmq : m ≠ q) 
(hnp : n ≠ p) (hnq : n ≠ q) (hpq : p ≠ q) (pos_m : 0 < m) (pos_n : 0 < n) 
(pos_p : 0 < p) (pos_q : 0 < q) (h : (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4) : 
  m + n + p + q = 24 :=
sorry

end sum_of_distinct_nums_l9_9527


namespace arithmetic_series_sum_after_multiplication_l9_9325

theorem arithmetic_series_sum_after_multiplication :
  let s : List ℕ := [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
  3 * s.sum = 3435 := by
  sorry

end arithmetic_series_sum_after_multiplication_l9_9325


namespace ceiling_and_floor_calculation_l9_9233

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l9_9233


namespace one_number_is_zero_l9_9452

variable {a b c : ℤ}
variable (cards : Fin 30 → ℤ)

theorem one_number_is_zero (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
    (h_cards : ∀ i : Fin 30, cards i = a ∨ cards i = b ∨ cards i = c)
    (h_sum_zero : ∀ (S : Finset (Fin 30)) (hS : S.card = 5),
        ∃ T : Finset (Fin 30), T.card = 5 ∧ (S ∪ T).sum cards = 0) :
    b = 0 := 
sorry

end one_number_is_zero_l9_9452


namespace number_of_strikers_l9_9573

theorem number_of_strikers (goalies defenders total_players midfielders strikers : ℕ)
  (h1 : goalies = 3)
  (h2 : defenders = 10)
  (h3 : midfielders = 2 * defenders)
  (h4 : total_players = 40)
  (h5 : total_players = goalies + defenders + midfielders + strikers) :
  strikers = 7 :=
by
  sorry

end number_of_strikers_l9_9573


namespace geometric_sequence_common_ratio_l9_9450

theorem geometric_sequence_common_ratio (a : ℕ → ℝ)
  (h : ∀ n, a n * a (n + 1) = 16^n) :
  ∃ r : ℝ, r = 4 ∧ ∀ n, a (n + 1) = a n * r :=
sorry

end geometric_sequence_common_ratio_l9_9450


namespace max_at_pi_six_l9_9442

theorem max_at_pi_six : ∃ (x : ℝ), (0 ≤ x ∧ x ≤ π / 2) ∧ (∀ y, (0 ≤ y ∧ y ≤ π / 2) → (x + 2 * Real.cos x) ≥ (y + 2 * Real.cos y)) ∧ x = π / 6 := sorry

end max_at_pi_six_l9_9442


namespace probability_abs_diff_l9_9203

variables (P : ℕ → ℚ) (m : ℚ)

def is_probability_distribution : Prop :=
  P 1 = m ∧ P 2 = 1/4 ∧ P 3 = 1/4 ∧ P 4 = 1/3 ∧ m + 1/4 + 1/4 + 1/3 = 1

theorem probability_abs_diff (h : is_probability_distribution P m) :
  P 1 + P 3 = 5 / 12 :=
by 
sorry

end probability_abs_diff_l9_9203


namespace Caitlin_age_l9_9724

theorem Caitlin_age (Aunt_Anna_age : ℕ) (h1 : Aunt_Anna_age = 54) (Brianna_age : ℕ) (h2 : Brianna_age = (2 * Aunt_Anna_age) / 3) (Caitlin_age : ℕ) (h3 : Caitlin_age = Brianna_age - 7) : 
  Caitlin_age = 29 := 
  sorry

end Caitlin_age_l9_9724


namespace coin_flips_heads_l9_9685

theorem coin_flips_heads (H T : ℕ) (flip_condition : H + T = 211) (tail_condition : T = H + 81) :
    H = 65 :=
by
  sorry

end coin_flips_heads_l9_9685


namespace f_2017_eq_one_l9_9209

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x - β)

-- Given conditions
variables {a b α β : ℝ}
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ α ≠ 0 ∧ β ≠ 0)
variable (h_f2016 : f 2016 a α b β = -1)

-- The goal
theorem f_2017_eq_one : f 2017 a α b β = 1 :=
sorry

end f_2017_eq_one_l9_9209


namespace polynomial_evaluation_x_eq_4_l9_9482

theorem polynomial_evaluation_x_eq_4 : 
  (4 ^ 4 + 4 ^ 3 + 4 ^ 2 + 4 + 1 = 341) := 
by 
  sorry

end polynomial_evaluation_x_eq_4_l9_9482


namespace largest_int_less_than_100_by_7_l9_9925

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l9_9925


namespace shoot_down_probability_l9_9837

-- Define the probabilities
def P_hit_nose := 0.2
def P_hit_middle := 0.4
def P_hit_tail := 0.1
def P_miss := 0.3

-- Define the condition: probability of shooting down the plane with at most 2 shots
def condition := (P_hit_tail + (P_hit_nose * P_hit_nose) + (P_miss * P_hit_tail))

-- Proving the probability matches the required value
theorem shoot_down_probability : condition = 0.23 :=
by
  sorry

end shoot_down_probability_l9_9837


namespace steak_chicken_ratio_l9_9349

variable (S C : ℕ)

theorem steak_chicken_ratio (h1 : S + C = 80) (h2 : 25 * S + 18 * C = 1860) : S = 3 * C :=
by
  sorry

end steak_chicken_ratio_l9_9349


namespace cone_lateral_surface_area_l9_9931

theorem cone_lateral_surface_area (r h : ℝ) (h_r : r = 3) (h_h : h = 4) : 
  (1/2) * (2 * Real.pi * r) * (Real.sqrt (r ^ 2 + h ^ 2)) = 15 * Real.pi := 
by
  sorry

end cone_lateral_surface_area_l9_9931


namespace one_circle_equiv_three_squares_l9_9140

-- Define the weights of circles and squares symbolically
variables {w_circle w_square : ℝ}

-- Equations based on the conditions in the problem
-- 3 circles balance 5 squares
axiom eq1 : 3 * w_circle = 5 * w_square

-- 2 circles balance 3 squares and 1 circle
axiom eq2 : 2 * w_circle = 3 * w_square + w_circle

-- We need to prove that 1 circle is equivalent to 3 squares
theorem one_circle_equiv_three_squares : w_circle = 3 * w_square := 
by sorry

end one_circle_equiv_three_squares_l9_9140


namespace solution_1_solution_2_l9_9308

noncomputable def problem_1 : Real :=
  Real.log 25 + Real.log 2 * Real.log 50 + (Real.log 2)^2

noncomputable def problem_2 : Real :=
  (Real.logb 3 2 + Real.logb 9 2) * (Real.logb 4 3 + Real.logb 8 3)

theorem solution_1 : problem_1 = 2 := by
  sorry

theorem solution_2 : problem_2 = 5 / 4 := by
  sorry

end solution_1_solution_2_l9_9308


namespace find_other_endpoint_l9_9966

def other_endpoint (midpoint endpoint: ℝ × ℝ) : ℝ × ℝ :=
  let (mx, my) := midpoint
  let (ex, ey) := endpoint
  (2 * mx - ex, 2 * my - ey)

theorem find_other_endpoint :
  other_endpoint (3, 1) (7, -4) = (-1, 6) :=
by
  -- Midpoint formula to find other endpoint
  sorry

end find_other_endpoint_l9_9966


namespace first_inequality_system_of_inequalities_l9_9596

-- First inequality problem
theorem first_inequality (x : ℝ) : 
  1 - (x - 3) / 6 > x / 3 → x < 3 := 
sorry

-- System of inequalities problem
theorem system_of_inequalities (x : ℝ) : 
  (x + 1 ≥ 3 * (x - 3)) ∧ ((x + 2) / 3 - (x - 1) / 4 > 1) → (1 < x ∧ x ≤ 5) := 
sorry

end first_inequality_system_of_inequalities_l9_9596


namespace solve_max_eq_l9_9125

theorem solve_max_eq (x : ℚ) (h : max x (-x) = 2 * x + 1) : x = -1 / 3 := by
  sorry

end solve_max_eq_l9_9125


namespace part_one_part_two_l9_9884

noncomputable def f (a x : ℝ) : ℝ := x * (a + Real.log x)
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem part_one (a : ℝ) (h : ∀ x > 0, f a x ≥ -1/Real.exp 1) : a = 0 := sorry

theorem part_two {a x : ℝ} (ha : a > 0) (hx : x > 0) :
  g x - f a x < 2 / Real.exp 1 := sorry

end part_one_part_two_l9_9884


namespace chalkboard_area_l9_9046

theorem chalkboard_area (width : ℝ) (h₁ : width = 3.5) (length : ℝ) (h₂ : length = 2.3 * width) : 
  width * length = 28.175 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end chalkboard_area_l9_9046


namespace right_triangle_side_length_l9_9362

theorem right_triangle_side_length
  (c : ℕ) (a : ℕ) (h_c : c = 13) (h_a : a = 12) :
  ∃ b : ℕ, b = 5 ∧ c^2 = a^2 + b^2 :=
by
  -- Definitions from conditions
  have h_c_square : c^2 = 169 := by rw [h_c]; norm_num
  have h_a_square : a^2 = 144 := by rw [h_a]; norm_num
  -- Prove the final result
  sorry

end right_triangle_side_length_l9_9362


namespace least_k_cubed_divisible_by_168_l9_9386

theorem least_k_cubed_divisible_by_168 : ∃ k : ℤ, (k ^ 3) % 168 = 0 ∧ ∀ n : ℤ, (n ^ 3) % 168 = 0 → k ≤ n :=
sorry

end least_k_cubed_divisible_by_168_l9_9386


namespace stable_scores_l9_9010

theorem stable_scores (S_A S_B S_C S_D : ℝ) (hA : S_A = 2.2) (hB : S_B = 6.6) (hC : S_C = 7.4) (hD : S_D = 10.8) : 
  S_A ≤ S_B ∧ S_A ≤ S_C ∧ S_A ≤ S_D :=
by
  sorry

end stable_scores_l9_9010


namespace equation_of_line_AB_l9_9138

noncomputable def center_of_circle : (ℝ × ℝ) := (-4, -1)

noncomputable def point_P : (ℝ × ℝ) := (2, 3)

noncomputable def slope_OP : ℝ :=
  let (x₁, y₁) := center_of_circle
  let (x₂, y₂) := point_P
  (y₂ - y₁) / (x₂ - x₁)

noncomputable def slope_AB : ℝ :=
  -1 / slope_OP

theorem equation_of_line_AB : (6 * x + 4 * y + 19 = 0) :=
  sorry

end equation_of_line_AB_l9_9138


namespace rectangular_prism_volume_l9_9352

theorem rectangular_prism_volume (a b c : ℝ) (h1 : a * b = 15) (h2 : b * c = 10) (h3 : a * c = 6) (h4 : c^2 = a^2 + b^2) : 
  a * b * c = 30 := 
sorry

end rectangular_prism_volume_l9_9352


namespace odd_function_value_l9_9053

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^3 - Real.sin x + b + 2

theorem odd_function_value (a b : ℝ) (h1 : ∀ x, f x b = -f (-x) b) (h2 : a - 4 + 2 * a - 2 = 0) : f a b + f (2 * -a) b = 0 := by
  sorry

end odd_function_value_l9_9053


namespace cos_alpha_minus_pi_over_6_l9_9733

theorem cos_alpha_minus_pi_over_6 (α : Real) 
  (h1 : Real.pi / 2 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.sin (α + Real.pi / 6) = 3 / 5) : 
  Real.cos (α - Real.pi / 6) = (3 * Real.sqrt 3 - 4) / 10 := 
by 
  sorry

end cos_alpha_minus_pi_over_6_l9_9733


namespace student_marks_l9_9835

def max_marks : ℕ := 600
def passing_percentage : ℕ := 30
def fail_by : ℕ := 100

theorem student_marks :
  ∃ x : ℕ, x + fail_by = (passing_percentage * max_marks) / 100 :=
sorry

end student_marks_l9_9835


namespace smallest_integer_proof_l9_9909

theorem smallest_integer_proof :
  ∃ (x : ℤ), x^2 = 3 * x + 75 ∧ ∀ (y : ℤ), y^2 = 3 * y + 75 → x ≤ y := 
  sorry

end smallest_integer_proof_l9_9909


namespace fivefold_composition_l9_9641

def f (x : ℚ) : ℚ := -2 / x

theorem fivefold_composition :
  f (f (f (f (f (3))))) = -2 / 3 := 
by
  -- Proof goes here
  sorry

end fivefold_composition_l9_9641


namespace money_left_after_bike_purchase_l9_9788

-- Definitions based on conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def quarter_value : ℝ := 0.25
def bike_cost : ℝ := 180

-- The theorem statement
theorem money_left_after_bike_purchase : (jars * quarters_per_jar * quarter_value) - bike_cost = 20 := by
  sorry

end money_left_after_bike_purchase_l9_9788


namespace cos_105_degree_value_l9_9407

noncomputable def cos105 : ℝ := Real.cos (105 * Real.pi / 180)

theorem cos_105_degree_value :
  cos105 = (Real.sqrt 2 - Real.sqrt 6) / 4 :=
by
  sorry

end cos_105_degree_value_l9_9407


namespace number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l9_9205

def is_consonant (c : Char) : Prop :=
  c = 'B' ∨ c = 'C' ∨ c = 'D' ∨ c = 'F'

def count_5_letter_words_with_at_least_one_consonant : Nat :=
  let total_words := 6 ^ 5
  let vowel_words := 2 ^ 5
  total_words - vowel_words

theorem number_of_5_letter_words_with_at_least_one_consonant_equals_7744 :
  count_5_letter_words_with_at_least_one_consonant = 7744 :=
by
  sorry

end number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l9_9205


namespace expenses_opposite_to_income_l9_9944

theorem expenses_opposite_to_income (income_5 : ℤ) (h_income : income_5 = 5) : -income_5 = -5 :=
by
  -- proof is omitted
  sorry

end expenses_opposite_to_income_l9_9944


namespace product_segment_doubles_l9_9922

-- Define the problem conditions and proof statement in Lean.
theorem product_segment_doubles
  (a b e : ℝ)
  (d : ℝ := (a * b) / e)
  (e' : ℝ := e / 2)
  (d' : ℝ := (a * b) / e') :
  d' = 2 * d := 
  sorry

end product_segment_doubles_l9_9922


namespace quadratic_inequality_solution_l9_9153

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + x - 12 > 0) → (x > 3 ∨ x < -4) :=
by
  sorry

end quadratic_inequality_solution_l9_9153


namespace percentage_shaded_l9_9617

def area_rect (width height : ℝ) : ℝ := width * height

def overlap_area (side_length : ℝ) (width_rect : ℝ) (length_rect: ℝ) (length_total: ℝ) : ℝ :=
  (side_length - (length_total - length_rect)) * width_rect

theorem percentage_shaded (sqr_side length_rect width_rect total_length total_width : ℝ) (h1 : sqr_side = 12) (h2 : length_rect = 9) (h3 : width_rect = 12)
  (h4 : total_length = 18) (h5 : total_width = 12) :
  (overlap_area sqr_side width_rect length_rect total_length) / (area_rect total_width total_length) * 100 = 12.5 :=
by
  sorry

end percentage_shaded_l9_9617


namespace fixed_point_of_function_l9_9390

theorem fixed_point_of_function :
  (4, 4) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, 2^(x-4) + 3) } :=
by
  sorry

end fixed_point_of_function_l9_9390


namespace largest_n_for_ap_interior_angles_l9_9045

theorem largest_n_for_ap_interior_angles (n : ℕ) (d : ℤ) (a : ℤ) :
  (∀ i ∈ Finset.range n, a + i * d < 180) → 720 = d * (n - 1) * n → n ≤ 27 :=
by
  sorry

end largest_n_for_ap_interior_angles_l9_9045


namespace pears_remaining_l9_9328

theorem pears_remaining (K_picked : ℕ) (M_picked : ℕ) (S_picked : ℕ)
                        (K_gave : ℕ) (M_gave : ℕ) (S_gave : ℕ)
                        (hK_pick : K_picked = 47)
                        (hM_pick : M_picked = 12)
                        (hS_pick : S_picked = 22)
                        (hK_give : K_gave = 46)
                        (hM_give : M_gave = 5)
                        (hS_give : S_gave = 15) :
  (K_picked - K_gave) + (M_picked - M_gave) + (S_picked - S_gave) = 15 :=
by
  sorry

end pears_remaining_l9_9328


namespace production_value_n_l9_9417

theorem production_value_n :
  -- Definitions based on conditions:
  (∀ a b : ℝ,
    (120 * a + 120 * b) / 60 = 6 ∧
    (100 * a + 100 * b) / 30 = 30) →
  (∃ n : ℝ, 80 * 3 * (a + b) = 480 * a + n * b) →
  n = 120 :=
by
  sorry

end production_value_n_l9_9417


namespace bananas_per_friend_l9_9162

-- Define the conditions
def total_bananas : ℕ := 40
def number_of_friends : ℕ := 40

-- Define the theorem to be proved
theorem bananas_per_friend : 
  (total_bananas / number_of_friends) = 1 :=
by
  sorry

end bananas_per_friend_l9_9162


namespace percent_of_z_l9_9671

variable {x y z : ℝ}

theorem percent_of_z (h₁ : x = 1.20 * y) (h₂ : y = 0.50 * z) : x = 0.60 * z :=
by
  sorry

end percent_of_z_l9_9671


namespace graph_of_3x2_minus_12y2_is_pair_of_straight_lines_l9_9014

theorem graph_of_3x2_minus_12y2_is_pair_of_straight_lines :
  ∀ (x y : ℝ), 3 * x^2 - 12 * y^2 = 0 ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  sorry

end graph_of_3x2_minus_12y2_is_pair_of_straight_lines_l9_9014


namespace train_length_l9_9785

noncomputable def length_of_first_train (l2 : ℝ) (v1 : ℝ) (v2 : ℝ) (t : ℝ) : ℝ :=
  let v1_m_per_s := v1 * 1000 / 3600
  let v2_m_per_s := v2 * 1000 / 3600
  let relative_speed := v1_m_per_s + v2_m_per_s
  let combined_length := relative_speed * t
  combined_length - l2

theorem train_length (l2 : ℝ) (v1 : ℝ) (v2 : ℝ) (t : ℝ) (h_l2 : l2 = 200) 
  (h_v1 : v1 = 100) (h_v2 : v2 = 200) (h_t : t = 3.6) : length_of_first_train l2 v1 v2 t = 100 := by
  sorry

end train_length_l9_9785


namespace prime_divides_sequence_term_l9_9736

theorem prime_divides_sequence_term (k : ℕ) (h_prime : Nat.Prime k) (h_ne_two : k ≠ 2) (h_ne_five : k ≠ 5) :
  ∃ n ≤ k, k ∣ (Nat.ofDigits 10 (List.replicate n 1)) :=
by
  sorry

end prime_divides_sequence_term_l9_9736


namespace sin_alpha_value_l9_9713

open Real

theorem sin_alpha_value (α β : ℝ) 
  (h1 : cos (α - β) = 3 / 5) 
  (h2 : sin β = -5 / 13) 
  (h3 : 0 < α ∧ α < π / 2) 
  (h4 : -π / 2 < β ∧ β < 0) 
  : sin α = 33 / 65 :=
sorry

end sin_alpha_value_l9_9713


namespace probability_only_one_l9_9854

-- Define the probabilities
def P_A : ℚ := 1 / 2
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Define the complement probabilities
def not_P (P : ℚ) : ℚ := 1 - P
def P_not_A := not_P P_A
def P_not_B := not_P P_B
def P_not_C := not_P P_C

-- Expressions for probabilities where only one student solves the problem
def only_A_solves : ℚ := P_A * P_not_B * P_not_C
def only_B_solves : ℚ := P_B * P_not_A * P_not_C
def only_C_solves : ℚ := P_C * P_not_A * P_not_B

-- Total probability that only one student solves the problem
def P_only_one : ℚ := only_A_solves + only_B_solves + only_C_solves

-- The theorem to prove that the total probability matches
theorem probability_only_one : P_only_one = 11 / 24 := by
  sorry

end probability_only_one_l9_9854


namespace selling_prices_maximize_profit_l9_9965

-- Definitions for the conditions
def total_items : ℕ := 200
def budget : ℤ := 5000
def cost_basketball : ℤ := 30
def cost_volleyball : ℤ := 24
def selling_price_ratio : ℚ := 3 / 2
def school_purchase_basketballs_value : ℤ := 1800
def school_purchase_volleyballs_value : ℤ := 1500
def basketballs_fewer_than_volleyballs : ℤ := 10

-- Part 1: Proof of selling prices
theorem selling_prices (x : ℚ) :
  (school_purchase_volleyballs_value / x - school_purchase_basketballs_value / (x * selling_price_ratio) = basketballs_fewer_than_volleyballs)
  → ∃ (basketball_price volleyball_price : ℚ), basketball_price = 45 ∧ volleyball_price = 30 :=
by
  sorry

-- Part 2: Proof of maximizing profit
theorem maximize_profit (a : ℕ) :
  (cost_basketball * a + cost_volleyball * (total_items - a) ≤ budget)
  → ∃ optimal_a : ℕ, (optimal_a = 33 ∧ total_items - optimal_a = 167) :=
by
  sorry

end selling_prices_maximize_profit_l9_9965


namespace avg_wx_half_l9_9202

noncomputable def avg_wx {w x y : ℝ} (h1 : 5 / w + 5 / x = 5 / y) (h2 : w * x = y) : ℝ :=
(w + x) / 2

theorem avg_wx_half {w x y : ℝ} (h1 : 5 / w + 5 / x = 5 / y) (h2 : w * x = y) :
  avg_wx h1 h2 = 1 / 2 :=
sorry

end avg_wx_half_l9_9202


namespace screen_time_morning_l9_9226

def total_screen_time : ℕ := 120
def evening_screen_time : ℕ := 75
def morning_screen_time : ℕ := 45

theorem screen_time_morning : total_screen_time - evening_screen_time = morning_screen_time := by
  sorry

end screen_time_morning_l9_9226


namespace L_like_reflexive_l9_9041

-- Definitions of the shapes and condition of being an "L-like shape"
inductive Shape
| A | B | C | D | E | LLike : Shape → Shape

-- reflection_equiv function representing reflection equivalence across a vertical dashed line
def reflection_equiv (s1 s2 : Shape) : Prop :=
sorry -- This would be defined according to the exact conditions of the shapes and reflection logic.

-- Given the shapes
axiom L_like : Shape
axiom A : Shape
axiom B : Shape
axiom C : Shape
axiom D : Shape
axiom E : Shape

-- The proof problem: Shape D is the mirrored reflection of the given "L-like shape" across a vertical dashed line
theorem L_like_reflexive :
  reflection_equiv L_like D :=
sorry

end L_like_reflexive_l9_9041


namespace quadratic_not_factored_l9_9311

theorem quadratic_not_factored
  (a b c : ℕ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (p : ℕ)
  (h_prime_p : Nat.Prime p)
  (h_p : a * 1991^2 + b * 1991 + c = p) :
  ¬ (∃ d₁ d₂ e₁ e₂ : ℤ, a = d₁ * d₂ ∧ b = d₁ * e₂ + d₂ * e₁ ∧ c = e₁ * e₂) :=
sorry

end quadratic_not_factored_l9_9311


namespace solution_set_l9_9260

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l9_9260


namespace equal_roots_for_specific_k_l9_9560

theorem equal_roots_for_specific_k (k : ℝ) :
  ((k - 1) * x^2 + 6 * x + 9 = 0) → (6^2 - 4*(k-1)*9 = 0) → (k = 2) :=
by sorry

end equal_roots_for_specific_k_l9_9560


namespace same_terminal_side_l9_9242

theorem same_terminal_side : ∃ k : ℤ, 36 + k * 360 = -324 :=
by
  use -1
  linarith

end same_terminal_side_l9_9242


namespace sum_of_abc_is_12_l9_9665

theorem sum_of_abc_is_12 (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
by
  sorry

end sum_of_abc_is_12_l9_9665


namespace image_of_center_l9_9052

-- Define the initial coordinates
def initial_coordinate : ℝ × ℝ := (-3, 4)

-- Function to reflect a point across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Function to translate a point up
def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

-- Definition of the final coordinate
noncomputable def final_coordinate : ℝ × ℝ :=
  translate_up (reflect_x initial_coordinate) 5

-- Theorem stating the final coordinate after transformations
theorem image_of_center : final_coordinate = (-3, 1) := by
  -- Proof is omitted
  sorry

end image_of_center_l9_9052


namespace brick_length_l9_9342

theorem brick_length (L : ℝ) :
  (∀ (V_wall V_brick : ℝ),
    V_wall = 29 * 100 * 2 * 100 * 0.75 * 100 ∧
    V_wall = 29000 * V_brick ∧
    V_brick = L * 10 * 7.5) →
  L = 20 :=
by
  intro h
  sorry

end brick_length_l9_9342


namespace veronica_pre_selected_photos_l9_9256

-- Definition: Veronica needs to include 3 or 4 of her pictures
def needs_3_or_4_photos : Prop := True

-- Definition: Veronica has pre-selected a certain number of photos
def pre_selected_photos : ℕ := 15

-- Definition: She has 15 choices
def choices : ℕ := 15

-- The proof statement
theorem veronica_pre_selected_photos : needs_3_or_4_photos → choices = pre_selected_photos :=
by
  intros
  sorry

end veronica_pre_selected_photos_l9_9256


namespace As_annual_income_l9_9699

theorem As_annual_income :
  let Cm := 14000
  let Bm := Cm + 0.12 * Cm
  let Am := (5 / 2) * Bm
  Am * 12 = 470400 := by
  sorry

end As_annual_income_l9_9699


namespace candies_per_child_rounded_l9_9988

/-- There are 15 pieces of candy divided equally among 7 children. The number of candies per child, rounded to the nearest tenth, is 2.1. -/
theorem candies_per_child_rounded :
  let candies := 15
  let children := 7
  Float.round (candies / children * 10) / 10 = 2.1 :=
by
  sorry

end candies_per_child_rounded_l9_9988


namespace daily_rental_cost_l9_9193

theorem daily_rental_cost (x : ℝ) (total_cost miles : ℝ)
  (cost_per_mile : ℝ) (daily_cost : ℝ) :
  total_cost = daily_cost + cost_per_mile * miles →
  total_cost = 46.12 →
  miles = 214 →
  cost_per_mile = 0.08 →
  daily_cost = 29 :=
by
  sorry

end daily_rental_cost_l9_9193


namespace find_p_l9_9170

theorem find_p (m n p : ℝ) 
  (h1 : m = (n / 2) - (2 / 5)) 
  (h2 : m + p = ((n + 4) / 2) - (2 / 5)) :
  p = 2 :=
sorry

end find_p_l9_9170


namespace man_swim_upstream_distance_l9_9823

theorem man_swim_upstream_distance (dist_downstream : ℝ) (time_downstream : ℝ) (time_upstream : ℝ) (speed_still_water : ℝ) 
  (effective_speed_downstream : ℝ) (speed_current : ℝ) (effective_speed_upstream : ℝ) (dist_upstream : ℝ) :
  dist_downstream = 36 →
  time_downstream = 6 →
  time_upstream = 6 →
  speed_still_water = 4.5 →
  effective_speed_downstream = dist_downstream / time_downstream →
  effective_speed_downstream = speed_still_water + speed_current →
  effective_speed_upstream = speed_still_water - speed_current →
  dist_upstream = effective_speed_upstream * time_upstream →
  dist_upstream = 18 :=
by
  intros h_dist_downstream h_time_downstream h_time_upstream h_speed_still_water
         h_effective_speed_downstream h_eq_speed_current h_effective_speed_upstream h_dist_upstream
  sorry

end man_swim_upstream_distance_l9_9823


namespace math_problem_l9_9394

theorem math_problem (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 :=
sorry

end math_problem_l9_9394


namespace sample_size_is_100_l9_9924

-- Conditions:
def scores_from_students := 100
def sampling_method := "simple random sampling"
def goal := "statistical analysis of senior three students' exam performance"

-- Problem statement:
theorem sample_size_is_100 :
  scores_from_students = 100 →
  sampling_method = "simple random sampling" →
  goal = "statistical analysis of senior three students' exam performance" →
  scores_from_students = 100 := by
sorry

end sample_size_is_100_l9_9924


namespace sarah_bought_3_bottle_caps_l9_9440

theorem sarah_bought_3_bottle_caps
  (orig_caps : ℕ)
  (new_caps : ℕ)
  (h_orig_caps : orig_caps = 26)
  (h_new_caps : new_caps = 29) :
  new_caps - orig_caps = 3 :=
by
  sorry

end sarah_bought_3_bottle_caps_l9_9440


namespace find_integer_pairs_l9_9388

theorem find_integer_pairs :
  ∀ x y : ℤ, x^2 = 2 + 6 * y^2 + y^4 ↔ (x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) :=
by {
  sorry
}

end find_integer_pairs_l9_9388


namespace bus_driver_total_compensation_l9_9934

-- Define the regular rate
def regular_rate : ℝ := 16

-- Define the number of regular hours
def regular_hours : ℕ := 40

-- Define the overtime rate as 75% higher than the regular rate
def overtime_rate : ℝ := regular_rate * 1.75

-- Define the total hours worked in the week
def total_hours_worked : ℕ := 48

-- Calculate the overtime hours
def overtime_hours : ℕ := total_hours_worked - regular_hours

-- Calculate the total compensation
def total_compensation : ℝ :=
  (regular_hours * regular_rate) + (overtime_hours * overtime_rate)

-- Theorem to prove that the total compensation is $864
theorem bus_driver_total_compensation : total_compensation = 864 := by
  -- Proof is omitted
  sorry

end bus_driver_total_compensation_l9_9934


namespace subset_implies_range_a_intersection_implies_range_a_l9_9591

noncomputable def setA : Set ℝ := {x | -1 < x ∧ x < 2}
noncomputable def setB (a : ℝ) : Set ℝ := {x | 2 * a - 1 < x ∧ x < 2 * a + 3}

theorem subset_implies_range_a (a : ℝ) : (setA ⊆ setB a) → (-1/2 ≤ a ∧ a ≤ 0) :=
by
  sorry

theorem intersection_implies_range_a (a : ℝ) : (setA ∩ setB a = ∅) → (a ≤ -2 ∨ a ≥ 3/2) :=
by
  sorry

end subset_implies_range_a_intersection_implies_range_a_l9_9591


namespace smallest_M_value_l9_9552

theorem smallest_M_value 
  (a b c d e : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) 
  (h_sum : a + b + c + d + e = 2010) : 
  (∃ M, M = max (a+b) (max (b+c) (max (c+d) (d+e))) ∧ M = 671) :=
by
  sorry

end smallest_M_value_l9_9552


namespace no_unhappy_days_l9_9844

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
  sorry

end no_unhappy_days_l9_9844


namespace ellipse_eccentricity_l9_9599

noncomputable def eccentricity_of_ellipse (a c : ℝ) : ℝ :=
  c / a

theorem ellipse_eccentricity (F1 A : ℝ) (v : ℝ) (a c : ℝ)
  (h1 : 4 * a = 10 * (a - c))
  (h2 : F1 = 0 ∧ A = 0 ∧ v ≠ 0) :
  eccentricity_of_ellipse a c = 3 / 5 := by
sorry

end ellipse_eccentricity_l9_9599


namespace slope_of_line_through_intersecting_points_of_circles_l9_9586

theorem slope_of_line_through_intersecting_points_of_circles :
  let circle1 (x y : ℝ) := x^2 + y^2 - 6*x + 4*y - 5 = 0
  let circle2 (x y : ℝ) := x^2 + y^2 - 10*x + 16*y + 24 = 0
  ∀ (C D : ℝ × ℝ), circle1 C.1 C.2 → circle2 C.1 C.2 → circle1 D.1 D.2 → circle2 D.1 D.2 → 
  let dx := D.1 - C.1
  let dy := D.2 - C.2
  dx ≠ 0 → dy / dx = 1 / 3 :=
by
  intros
  sorry

end slope_of_line_through_intersecting_points_of_circles_l9_9586


namespace measure_of_angle_Q_l9_9802

variables (R S T U Q : ℝ)
variables (angle_R angle_S angle_T angle_U : ℝ)

-- Given conditions
def sum_of_angles_in_pentagon : ℝ := 540
def angle_measure_R : ℝ := 120
def angle_measure_S : ℝ := 94
def angle_measure_T : ℝ := 115
def angle_measure_U : ℝ := 101

theorem measure_of_angle_Q :
  angle_R = angle_measure_R →
  angle_S = angle_measure_S →
  angle_T = angle_measure_T →
  angle_U = angle_measure_U →
  (angle_R + angle_S + angle_T + angle_U + Q = sum_of_angles_in_pentagon) →
  Q = 110 :=
by { sorry }

end measure_of_angle_Q_l9_9802


namespace total_distance_maria_l9_9395

theorem total_distance_maria (D : ℝ)
  (half_dist : D/2 + (D/2 - D/8) + 180 = D) :
  3 * D / 8 = 180 → 
  D = 480 :=
by
  sorry

end total_distance_maria_l9_9395


namespace shirt_cost_l9_9572

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 66) : S = 12 :=
by
  sorry

end shirt_cost_l9_9572


namespace inequality_proof_l9_9119

open Real

theorem inequality_proof
  (a b c x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hx_cond : 1 / x + 1 / y + 1 / z = 1) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a ^ x + b ^ y + c ^ z ≥ (4 * a * b * c * x * y * z) / (x + y + z - 3) ^ 2 :=
by
  sorry

end inequality_proof_l9_9119


namespace simplify_expression_l9_9982

theorem simplify_expression (y : ℝ) : (3 * y + 4 * y + 5 * y + 7) = (12 * y + 7) :=
by
  sorry

end simplify_expression_l9_9982


namespace trig_fraction_identity_l9_9472

noncomputable def cos_63 := Real.cos (Real.pi * 63 / 180)
noncomputable def cos_3 := Real.cos (Real.pi * 3 / 180)
noncomputable def cos_87 := Real.cos (Real.pi * 87 / 180)
noncomputable def cos_27 := Real.cos (Real.pi * 27 / 180)
noncomputable def cos_132 := Real.cos (Real.pi * 132 / 180)
noncomputable def cos_72 := Real.cos (Real.pi * 72 / 180)
noncomputable def cos_42 := Real.cos (Real.pi * 42 / 180)
noncomputable def cos_18 := Real.cos (Real.pi * 18 / 180)
noncomputable def tan_24 := Real.tan (Real.pi * 24 / 180)

theorem trig_fraction_identity :
  (cos_63 * cos_3 - cos_87 * cos_27) / 
  (cos_132 * cos_72 - cos_42 * cos_18) = 
  -tan_24 := 
by
  sorry

end trig_fraction_identity_l9_9472


namespace solution_set_inequality_l9_9265

theorem solution_set_inequality (x : ℝ) : |3 * x + 1| - |x - 1| < 0 ↔ -1 < x ∧ x < 0 := 
sorry

end solution_set_inequality_l9_9265


namespace triangle_inequality_l9_9263

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2*(b + c - a) + b^2*(c + a - b) + c^2*(a + b - c) ≤ 3*a*b*c :=
by
  sorry

end triangle_inequality_l9_9263


namespace max_t_subsets_of_base_set_l9_9824

theorem max_t_subsets_of_base_set (n : ℕ)
  (A : Fin (2 * n + 1) → Set (Fin n))
  (h : ∀ i j k : Fin (2 * n + 1), i < j → j < k → (A i ∩ A k) ⊆ A j) : 
  ∃ t : ℕ, t = 2 * n + 1 :=
by
  sorry

end max_t_subsets_of_base_set_l9_9824


namespace compute_n_binom_l9_9418

-- Definitions based on conditions
def n : ℕ := sorry  -- Assume n is a positive integer defined elsewhere
def k : ℕ := 4

-- The binomial coefficient definition
def binom (n k : ℕ) : ℕ :=
  if h₁ : k ≤ n then
    (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))
  else 0

-- The theorem to prove
theorem compute_n_binom : n * binom k 3 = 4 * n :=
by
  sorry

end compute_n_binom_l9_9418


namespace find_f_prime_2_l9_9898

theorem find_f_prime_2 (a : ℝ) (f' : ℝ → ℝ) 
    (h1 : f' 1 = -5)
    (h2 : ∀ x, f' x = 3 * a * x^2 + 2 * f' 2 * x) : f' 2 = -4 := by
    sorry

end find_f_prime_2_l9_9898


namespace solve_equation_l9_9997

theorem solve_equation (x : ℝ) :
  (x + 1)^2 = (2 * x - 1)^2 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_equation_l9_9997
