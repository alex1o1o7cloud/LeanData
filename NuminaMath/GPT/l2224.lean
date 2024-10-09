import Mathlib

namespace Evan_earnings_Markese_less_than_Evan_l2224_222409

-- Definitions from conditions
def MarkeseEarnings : ℕ := 16
def TotalEarnings : ℕ := 37

-- Theorem statements
theorem Evan_earnings (E : ℕ) (h : E + MarkeseEarnings = TotalEarnings) : E = 21 :=
by {
  sorry
}

theorem Markese_less_than_Evan (E : ℕ) (h : E + MarkeseEarnings = TotalEarnings) : E - MarkeseEarnings = 5 :=
by {
  sorry
}

end Evan_earnings_Markese_less_than_Evan_l2224_222409


namespace original_price_color_TV_l2224_222463

theorem original_price_color_TV (x : ℝ) 
  (h : 1.12 * x - x = 144) : 
  x = 1200 :=
sorry

end original_price_color_TV_l2224_222463


namespace jenny_ran_further_l2224_222453

-- Define the distances Jenny ran and walked
def ran_distance : ℝ := 0.6
def walked_distance : ℝ := 0.4

-- Define the difference between the distances Jenny ran and walked
def difference : ℝ := ran_distance - walked_distance

-- The proof statement
theorem jenny_ran_further : difference = 0.2 := by
  sorry

end jenny_ran_further_l2224_222453


namespace ratio_of_larger_to_smaller_l2224_222406

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a + b = 5 * (a - b)) :
  a / b = 3 / 2 := by
sorry

end ratio_of_larger_to_smaller_l2224_222406


namespace problem1_solution_correct_problem2_solution_correct_l2224_222449

def problem1 (x : ℤ) : Prop := (x - 1) ∣ (x + 3)
def problem2 (x : ℤ) : Prop := (x + 2) ∣ (x^2 + 2)
def solution1 (x : ℤ) : Prop := x = -3 ∨ x = -1 ∨ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5
def solution2 (x : ℤ) : Prop := x = -8 ∨ x = -5 ∨ x = -4 ∨ x = -3 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 4

theorem problem1_solution_correct : ∀ x: ℤ, problem1 x ↔ solution1 x := by
  sorry

theorem problem2_solution_correct : ∀ x: ℤ, problem2 x ↔ solution2 x := by
  sorry

end problem1_solution_correct_problem2_solution_correct_l2224_222449


namespace initial_deadline_l2224_222441

theorem initial_deadline (W : ℕ) (R : ℕ) (D : ℕ) :
    100 * 25 * 8 = (1/3 : ℚ) * W →
    (2/3 : ℚ) * W = 160 * R * 10 →
    D = 25 + R →
    D = 50 := 
by
  intros h1 h2 h3
  sorry

end initial_deadline_l2224_222441


namespace greatest_second_term_arithmetic_sequence_l2224_222415

theorem greatest_second_term_arithmetic_sequence:
  ∃ a d : ℕ, (a > 0) ∧ (d > 0) ∧ (2 * a + 3 * d = 29) ∧ (4 * a + 6 * d = 58) ∧ (((a + d : ℤ) / 3 : ℤ) = 10) :=
sorry

end greatest_second_term_arithmetic_sequence_l2224_222415


namespace dodecahedron_interior_diagonals_l2224_222447

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l2224_222447


namespace find_range_of_m_l2224_222446

noncomputable def quadratic_equation := 
  ∀ (m : ℝ), 
  ∃ x y : ℝ, 
  (m + 3) * x^2 - 4 * m * x + (2 * m - 1) = 0 ∧ 
  (m + 3) * y^2 - 4 * m * y + (2 * m - 1) = 0 ∧ 
  x * y < 0 ∧ 
  |x| > |y| ∧ 
  m ∈ Set.Ioo (-3:ℝ) (0:ℝ)

theorem find_range_of_m : quadratic_equation := 
by
  sorry

end find_range_of_m_l2224_222446


namespace work_completion_l2224_222444

theorem work_completion (W : ℕ) (a_rate b_rate combined_rate : ℕ) 
  (h1: combined_rate = W/8) 
  (h2: a_rate = W/12) 
  (h3: combined_rate = a_rate + b_rate) 
  : combined_rate = W/8 :=
by
  sorry

end work_completion_l2224_222444


namespace unit_digit_product_7858_1086_4582_9783_l2224_222419

theorem unit_digit_product_7858_1086_4582_9783 : 
  (7858 * 1086 * 4582 * 9783) % 10 = 8 :=
by
  -- Given that the unit digits of the numbers are 8, 6, 2, and 3.
  let d1 := 7858 % 10 -- This unit digit is 8
  let d2 := 1086 % 10 -- This unit digit is 6
  let d3 := 4582 % 10 -- This unit digit is 2
  let d4 := 9783 % 10 -- This unit digit is 3
  -- We need to prove that the unit digit of the product is 8
  sorry -- The actual proof steps are skipped

end unit_digit_product_7858_1086_4582_9783_l2224_222419


namespace correct_cd_value_l2224_222478

noncomputable def repeating_decimal (c d : ℕ) : ℝ :=
  1 + c / 10.0 + d / 100.0 + (c * 10 + d) / 990.0

theorem correct_cd_value (c d : ℕ) (h : (c = 9) ∧ (d = 9)) : 90 * (repeating_decimal 9 9 - (1 + 9 / 10.0 + 9 / 100.0)) = 0.9 :=
by
  sorry

end correct_cd_value_l2224_222478


namespace factorize_a_cubed_minus_25a_l2224_222425

variable {a : ℝ}

theorem factorize_a_cubed_minus_25a (a : ℝ) : a^3 - 25 * a = a * (a + 5) * (a - 5) := 
by sorry

end factorize_a_cubed_minus_25a_l2224_222425


namespace find_constants_l2224_222411

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x + 1

noncomputable def f_inv (x a b c : ℝ) : ℝ :=
  ( (x - a + Real.sqrt (x^2 - b * x + c)) / 2 )^(1/3) +
  ( (x - a - Real.sqrt (x^2 - b * x + c)) / 2 )^(1/3)

theorem find_constants (a b c : ℝ) (h1 : f_inv (1:ℝ) a b c = 0)
  (ha : a = 1) (hb : b = 2) (hc : c = 5) : a + 10 * b + 100 * c = 521 :=
by
  rw [ha, hb, hc]
  norm_num

end find_constants_l2224_222411


namespace find_m_plus_n_l2224_222461

-- Define the sets and variables
def M : Set ℝ := {x | x^2 - 4 * x < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < 5}
def K (n : ℝ) : Set ℝ := {x | 3 < x ∧ x < n}

theorem find_m_plus_n (m n : ℝ) 
  (hM: M = {x | 0 < x ∧ x < 4})
  (hK_true: K n = M ∩ N m) :
  m + n = 7 := 
  sorry

end find_m_plus_n_l2224_222461


namespace center_in_triangle_probability_l2224_222421

theorem center_in_triangle_probability (n : ℕ) :
  let vertices := 2 * n + 1
  let total_ways := vertices.choose 3
  let no_center_ways := vertices * (n.choose 2) / 2
  let p_no_center := no_center_ways / total_ways
  let p_center := 1 - p_no_center
  p_center = (n + 1) / (4 * n - 2) := sorry

end center_in_triangle_probability_l2224_222421


namespace evaluate_expression_l2224_222479

theorem evaluate_expression : 
  60 + 120 / 15 + 25 * 16 - 220 - 420 / 7 + 3 ^ 2 = 197 :=
by
  sorry

end evaluate_expression_l2224_222479


namespace fraction_simplification_l2224_222404

theorem fraction_simplification : (3^2040 + 3^2038) / (3^2040 - 3^2038) = 5 / 4 :=
by
  sorry

end fraction_simplification_l2224_222404


namespace greatest_divisor_4665_6905_l2224_222422

def digits_sum (n : ℕ) : ℕ :=
(n.digits 10).sum

theorem greatest_divisor_4665_6905 :
  ∃ n : ℕ, (n ∣ 4665) ∧ (n ∣ 6905) ∧ (digits_sum n = 4) ∧
  (∀ m : ℕ, ((m ∣ 4665) ∧ (m ∣ 6905) ∧ (digits_sum m = 4)) → (m ≤ n)) :=
sorry

end greatest_divisor_4665_6905_l2224_222422


namespace area_triangle_ABC_l2224_222473

theorem area_triangle_ABC (x y : ℝ) (h : x * y ≠ 0) (hAOB : 1 / 2 * |x * y| = 4) : 
  1 / 2 * |(x * (-2 * y) + x * (2 * y) + (-x) * (2 * y))| = 8 :=
by
  sorry

end area_triangle_ABC_l2224_222473


namespace cylinder_height_l2224_222466

theorem cylinder_height (r h : ℝ) (SA : ℝ) 
  (hSA : SA = 2 * Real.pi * r ^ 2 + 2 * Real.pi * r * h) 
  (hr : r = 3) (hSA_val : SA = 36 * Real.pi) : 
  h = 3 :=
by
  sorry

end cylinder_height_l2224_222466


namespace ratio_of_products_l2224_222407

variable (a b c d : ℚ) -- assuming a, b, c, d are rational numbers

theorem ratio_of_products (h1 : a = 3 * b) (h2 : b = 2 * c) (h3 : c = 5 * d) :
  a * c / (b * d) = 15 := by
  sorry

end ratio_of_products_l2224_222407


namespace smallest_integer_remainder_conditions_l2224_222499

theorem smallest_integer_remainder_conditions :
  ∃ b : ℕ, (b % 3 = 0) ∧ (b % 4 = 2) ∧ (b % 5 = 3) ∧ (∀ n : ℕ, (n % 3 = 0) ∧ (n % 4 = 2) ∧ (n % 5 = 3) → b ≤ n) :=
sorry

end smallest_integer_remainder_conditions_l2224_222499


namespace area_of_sector_l2224_222424

/-- The area of a sector of a circle with radius 10 meters and central angle 42 degrees is 35/3 * pi square meters. -/
theorem area_of_sector (r θ : ℕ) (h_r : r = 10) (h_θ : θ = 42) : 
  (θ / 360 : ℝ) * (Real.pi : ℝ) * (r : ℝ)^2 = (35 / 3 : ℝ) * (Real.pi : ℝ) :=
by {
  sorry
}

end area_of_sector_l2224_222424


namespace prove_q_ge_bd_and_p_eq_ac_l2224_222465

-- Definitions for the problem
variables (a b c d p q : ℕ)

-- Conditions given in the problem
axiom h1: a * d - b * c = 1
axiom h2: (a : ℚ) / b > (p : ℚ) / q
axiom h3: (p : ℚ) / q > (c : ℚ) / d

-- The theorem to be proved
theorem prove_q_ge_bd_and_p_eq_ac (a b c d p q : ℕ) (h1 : a * d - b * c = 1) 
  (h2 : (a : ℚ) / b > (p : ℚ) / q) (h3 : (p : ℚ) / q > (c : ℚ) / d) :
  q ≥ b + d ∧ (q = b + d → p = a + c) :=
by
  sorry

end prove_q_ge_bd_and_p_eq_ac_l2224_222465


namespace smallest_n_mult_y_perfect_cube_l2224_222497

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9

theorem smallest_n_mult_y_perfect_cube : ∃ n : ℕ, (∀ m : ℕ, y * n = m^3 → n = 1500) :=
sorry

end smallest_n_mult_y_perfect_cube_l2224_222497


namespace complex_expression_equals_neg3_l2224_222480

noncomputable def nonreal_root_of_x4_eq_1 : Type :=
{ζ : ℂ // ζ^4 = 1 ∧ ζ.im ≠ 0}

theorem complex_expression_equals_neg3 (ζ : nonreal_root_of_x4_eq_1) :
  (1 - ζ.val + ζ.val^3)^4 + (1 + ζ.val^2 - ζ.val^3)^4 = -3 :=
sorry

end complex_expression_equals_neg3_l2224_222480


namespace original_price_of_lens_is_correct_l2224_222493

-- Definitions based on conditions
def current_camera_price : ℝ := 4000
def new_camera_price : ℝ := current_camera_price + 0.30 * current_camera_price
def combined_price_paid : ℝ := 5400
def lens_discount : ℝ := 200
def combined_price_before_discount : ℝ := combined_price_paid + lens_discount

-- Calculated original price of the lens
def lens_original_price : ℝ := combined_price_before_discount - new_camera_price

-- The Lean theorem statement to prove the price is correct
theorem original_price_of_lens_is_correct : lens_original_price = 400 := by
  -- You do not need to provide the actual proof steps
  sorry

end original_price_of_lens_is_correct_l2224_222493


namespace vans_for_field_trip_l2224_222420

-- Definitions based on conditions
def students := 25
def adults := 5
def van_capacity := 5

-- Calculate total number of people
def total_people := students + adults

-- Calculate number of vans needed
def vans_needed := total_people / van_capacity

-- Theorem statement
theorem vans_for_field_trip : vans_needed = 6 := by
  -- Proof would go here
  sorry

end vans_for_field_trip_l2224_222420


namespace symmetric_point_of_P_l2224_222433

-- Let P be a point with coordinates (5, -3)
def P : ℝ × ℝ := (5, -3)

-- Definition of the symmetric point with respect to the x-axis
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Theorem stating that the symmetric point to P with respect to the x-axis is (5, 3)
theorem symmetric_point_of_P : symmetric_point P = (5, 3) := 
  sorry

end symmetric_point_of_P_l2224_222433


namespace min_value_expr_l2224_222458

theorem min_value_expr (x : ℝ) (h : x > 1) : ∃ m, m = 5 ∧ ∀ y, y = x + 4 / (x - 1) → y ≥ m :=
by
  sorry

end min_value_expr_l2224_222458


namespace complement_of_A_in_U_l2224_222462

def U : Set ℝ := {x | x ≤ 1}
def A : Set ℝ := {x | x < 0}

theorem complement_of_A_in_U : (U \ A) = {x | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end complement_of_A_in_U_l2224_222462


namespace greatest_possible_difference_l2224_222439

theorem greatest_possible_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) : 
  ∃ d, d = y - x ∧ d = 6 := 
by
  sorry

end greatest_possible_difference_l2224_222439


namespace find_x_squared_l2224_222438

variable (a b x p q : ℝ)

theorem find_x_squared (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : q ≠ p) (h4 : (a^2 + x^2) / (b^2 + x^2) = p / q) : 
  x^2 = (b^2 * p - a^2 * q) / (q - p) := 
by 
  sorry

end find_x_squared_l2224_222438


namespace find_x_of_arithmetic_mean_l2224_222455

theorem find_x_of_arithmetic_mean (x : ℝ) (h : (6 + 13 + 18 + 4 + x) / 5 = 10) : x = 9 :=
by
  sorry

end find_x_of_arithmetic_mean_l2224_222455


namespace distance_between_x_intercepts_l2224_222423

theorem distance_between_x_intercepts (x1 y1 : ℝ) 
  (m1 m2 : ℝ)
  (hx1 : x1 = 10) (hy1 : y1 = 15)
  (hm1 : m1 = 3) (hm2 : m2 = 5) :
  let x_intercept1 := (y1 - m1 * x1) / -m1
  let x_intercept2 := (y1 - m2 * x1) / -m2
  dist (x_intercept1, 0) (x_intercept2, 0) = 2 :=
by
  sorry

end distance_between_x_intercepts_l2224_222423


namespace expression_A_expression_B_expression_C_expression_D_l2224_222475

theorem expression_A :
  (Real.sin (7 * Real.pi / 180) * Real.cos (23 * Real.pi / 180) + 
   Real.sin (83 * Real.pi / 180) * Real.cos (67 * Real.pi / 180)) = 1 / 2 :=
sorry

theorem expression_B :
  (2 * Real.cos (75 * Real.pi / 180) * Real.sin (75 * Real.pi / 180)) = 1 / 2 :=
sorry

theorem expression_C :
  (Real.sqrt 3 * Real.cos (10 * Real.pi / 180) - Real.sin (10 * Real.pi / 180)) / 
   Real.sin (50 * Real.pi / 180) ≠ 1 / 2 :=
sorry

theorem expression_D :
  (1 / ((1 + Real.tan (27 * Real.pi / 180)) * (1 + Real.tan (18 * Real.pi / 180)))) = 1 / 2 :=
sorry

end expression_A_expression_B_expression_C_expression_D_l2224_222475


namespace arithmetic_sequence_l2224_222492

theorem arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h_n : n > 0) 
  (h_Sn : S (2 * n) - S (2 * n - 1) + a 2 = 424) : 
  a (n + 1) = 212 :=
sorry

end arithmetic_sequence_l2224_222492


namespace mutually_exclusive_event_l2224_222490

-- Define the events
def hits_first_shot : Prop := sorry  -- Placeholder for "hits the target on the first shot"
def hits_second_shot : Prop := sorry  -- Placeholder for "hits the target on the second shot"
def misses_first_shot : Prop := ¬ hits_first_shot
def misses_second_shot : Prop := ¬ hits_second_shot

-- Define the main events in the problem
def hitting_at_least_once : Prop := hits_first_shot ∨ hits_second_shot
def missing_both_times : Prop := misses_first_shot ∧ misses_second_shot

-- Statement of the theorem
theorem mutually_exclusive_event :
  missing_both_times ↔ ¬ hitting_at_least_once :=
by sorry

end mutually_exclusive_event_l2224_222490


namespace increase_by_1_or_prime_l2224_222491

theorem increase_by_1_or_prime (a : ℕ → ℕ) :
  a 0 = 6 →
  (∀ n, a (n + 1) = a n + Nat.gcd (a n) (n + 1)) →
  ∀ n, n < 1000000 → (∃ p, p = 1 ∨ Nat.Prime p ∧ a (n + 1) = a n + p) :=
by
  intro ha0 ha_step
  -- Proof omitted
  sorry

end increase_by_1_or_prime_l2224_222491


namespace find_x_l2224_222471

variable (x : ℝ)

def length := 4 * x
def width := x + 3

def area := length x * width x
def perimeter := 2 * length x + 2 * width x

theorem find_x (h : area x = 3 * perimeter x) : x = 5.342 := by
  sorry

end find_x_l2224_222471


namespace multiplication_of_powers_l2224_222468

theorem multiplication_of_powers :
  2^4 * 3^2 * 5^2 * 11 = 39600 := by
  sorry

end multiplication_of_powers_l2224_222468


namespace solve_for_a_l2224_222472

theorem solve_for_a (a y x : ℝ)
  (h1 : y = 5 * a)
  (h2 : x = 2 * a - 2)
  (h3 : y + 3 = x) :
  a = -5 / 3 :=
by
  sorry

end solve_for_a_l2224_222472


namespace brandon_investment_percentage_l2224_222432

noncomputable def jackson_initial_investment : ℕ := 500
noncomputable def brandon_initial_investment : ℕ := 500
noncomputable def jackson_final_investment : ℕ := 2000
noncomputable def difference_in_investments : ℕ := 1900
noncomputable def brandon_final_investment : ℕ := jackson_final_investment - difference_in_investments

theorem brandon_investment_percentage :
  (brandon_final_investment : ℝ) / (brandon_initial_investment : ℝ) * 100 = 20 := by
  sorry

end brandon_investment_percentage_l2224_222432


namespace volume_of_mixture_l2224_222485

theorem volume_of_mixture
    (weight_a : ℝ) (weight_b : ℝ) (ratio_a_b : ℝ) (total_weight : ℝ)
    (h1 : weight_a = 900) (h2 : weight_b = 700)
    (h3 : ratio_a_b = 3/2) (h4 : total_weight = 3280) :
    ∃ Va Vb : ℝ, (Va / Vb = ratio_a_b) ∧ (weight_a * Va + weight_b * Vb = total_weight) ∧ (Va + Vb = 4) := 
by
  sorry

end volume_of_mixture_l2224_222485


namespace sin_div_one_minus_tan_eq_neg_three_fourths_l2224_222408

variable (α : ℝ)

theorem sin_div_one_minus_tan_eq_neg_three_fourths
  (h : Real.sin (α - Real.pi / 4) = Real.sqrt 2 / 4) :
  (Real.sin α) / (1 - Real.tan α) = -3 / 4 := sorry

end sin_div_one_minus_tan_eq_neg_three_fourths_l2224_222408


namespace possible_third_side_l2224_222489

theorem possible_third_side {x : ℕ} (h_option_A : x = 2) (h_option_B : x = 3) (h_option_C : x = 6) (h_option_D : x = 13) : 3 < x ∧ x < 13 ↔ x = 6 :=
by
  sorry

end possible_third_side_l2224_222489


namespace value_of_x_plus_y_l2224_222476

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem value_of_x_plus_y
  (x y : ℝ)
  (h1 : x ≥ 1)
  (h2 : y ≥ 1)
  (h3 : x * y = 10)
  (h4 : x^(lg x) * y^(lg y) ≥ 10) :
  x + y = 11 :=
  sorry

end value_of_x_plus_y_l2224_222476


namespace javier_average_hits_l2224_222450

-- Define the total number of games Javier plays and the first set number of games
def total_games := 30
def first_set_games := 20

-- Define the hit averages for the first set of games and the desired season average
def average_hits_first_set := 2
def desired_season_average := 3

-- Define the total hits Javier needs to achieve the desired average by the end of the season
def total_hits_needed : ℕ := total_games * desired_season_average

-- Define the hits Javier made in the first set of games
def hits_made_first_set : ℕ := first_set_games * average_hits_first_set

-- Define the remaining games and the hits Javier needs to achieve in these games to meet his target
def remaining_games := total_games - first_set_games
def hits_needed_remaining_games : ℕ := total_hits_needed - hits_made_first_set

-- Define the average hits Javier needs in the remaining games to meet his target
def average_needed_remaining_games (remaining_games hits_needed_remaining_games : ℕ) : ℕ :=
  hits_needed_remaining_games / remaining_games

theorem javier_average_hits : 
  average_needed_remaining_games remaining_games hits_needed_remaining_games = 5 := 
by
  -- The proof is omitted.
  sorry

end javier_average_hits_l2224_222450


namespace shoveling_driveway_time_l2224_222431

theorem shoveling_driveway_time (S : ℝ) (Wayne_rate : ℝ) (combined_rate : ℝ) :
  (S = 1 / 7) → (Wayne_rate = 6 * S) → (combined_rate = Wayne_rate + S) → (combined_rate = 1) :=
by { sorry }

end shoveling_driveway_time_l2224_222431


namespace base_conversion_subtraction_l2224_222430

theorem base_conversion_subtraction :
  let n1_base9 := 3 * 9^2 + 2 * 9^1 + 4 * 9^0
  let n2_base7 := 1 * 7^2 + 6 * 7^1 + 5 * 7^0
  n1_base9 - n2_base7 = 169 :=
by
  sorry

end base_conversion_subtraction_l2224_222430


namespace div_by_9_digit_B_l2224_222452

theorem div_by_9_digit_B (B : ℕ) (h : (4 + B + B + 2) % 9 = 0) : B = 6 :=
by sorry

end div_by_9_digit_B_l2224_222452


namespace fair_hair_women_percentage_l2224_222435

-- Definitions based on conditions
def total_employees (E : ℝ) := E
def women_with_fair_hair (E : ℝ) := 0.28 * E
def fair_hair_employees (E : ℝ) := 0.70 * E

-- Theorem to prove
theorem fair_hair_women_percentage (E : ℝ) (hE : E > 0) :
  (women_with_fair_hair E) / (fair_hair_employees E) * 100 = 40 :=
by 
  -- Sorry denotes the proof is omitted
  sorry

end fair_hair_women_percentage_l2224_222435


namespace no_real_solutions_for_inequality_l2224_222494

theorem no_real_solutions_for_inequality (a : ℝ) :
  ¬∃ x : ℝ, ∀ y : ℝ, |(x^2 + a*x + 2*a)| ≤ 5 → y = x :=
sorry

end no_real_solutions_for_inequality_l2224_222494


namespace find_L_for_perfect_square_W_l2224_222487

theorem find_L_for_perfect_square_W :
  ∃ L W : ℕ, 1000 < W ∧ W < 2000 ∧ L > 1 ∧ W = 2 * L^3 ∧ ∃ m : ℕ, W = m^2 ∧ L = 8 :=
by sorry

end find_L_for_perfect_square_W_l2224_222487


namespace integer_solutions_positive_product_l2224_222457

theorem integer_solutions_positive_product :
  {a : ℤ | (5 + a) * (3 - a) > 0} = {-4, -3, -2, -1, 0, 1, 2} :=
by
  sorry

end integer_solutions_positive_product_l2224_222457


namespace find_D_c_l2224_222456

-- Define the given conditions
def daily_wage_ratio (W_a W_b W_c : ℝ) : Prop :=
  W_a / W_b = 3 / 4 ∧ W_a / W_c = 3 / 5 ∧ W_b / W_c = 4 / 5

def total_earnings (W_a W_b W_c : ℝ) (D_a D_b D_c : ℕ) : ℝ :=
  W_a * D_a + W_b * D_b + W_c * D_c

variables {W_a W_b W_c : ℝ} 
variables {D_a D_b D_c : ℕ} 

-- Given values according to the problem
def W_c_value : ℝ := 110
def D_a_value : ℕ := 6
def D_b_value : ℕ := 9
def total_earnings_value : ℝ := 1628

-- The target proof statement
theorem find_D_c 
  (h_ratio : daily_wage_ratio W_a W_b W_c)
  (h_Wc : W_c = W_c_value)
  (h_earnings : total_earnings W_a W_b W_c D_a_value D_b_value D_c = total_earnings_value) 
  : D_c = 4 := 
sorry

end find_D_c_l2224_222456


namespace least_possible_value_z_minus_x_l2224_222477

theorem least_possible_value_z_minus_x
  (x y z : ℤ)
  (h1 : x < y)
  (h2 : y < z)
  (h3 : y - x > 5)
  (hx_even : x % 2 = 0)
  (hy_odd : y % 2 = 1)
  (hz_odd : z % 2 = 1) :
  z - x = 9 :=
  sorry

end least_possible_value_z_minus_x_l2224_222477


namespace general_term_of_sequence_l2224_222442

def S (n : ℕ) : ℕ := n^2 + 3 * n + 1

def a (n : ℕ) : ℕ := 
  if n = 1 then 5 
  else 2 * n + 2

theorem general_term_of_sequence (n : ℕ) : 
  a n = if n = 1 then 5 else (S n - S (n - 1)) := 
by 
  sorry

end general_term_of_sequence_l2224_222442


namespace pit_A_no_replant_exactly_one_pit_no_replant_at_least_one_replant_l2224_222484

noncomputable def pit_a_no_replant_prob : ℝ := 0.875
noncomputable def one_pit_no_replant_prob : ℝ := 0.713
noncomputable def at_least_one_pit_replant_prob : ℝ := 0.330

theorem pit_A_no_replant (p : ℝ) (h1 : p = 0.5) : pit_a_no_replant_prob = 1 - (1 - p)^3 := by
  sorry

theorem exactly_one_pit_no_replant (p : ℝ) (h1 : p = 0.5) : one_pit_no_replant_prob = 1 - 3 * (1 - p)^3 * (p^3)^(2) := by
  sorry

theorem at_least_one_replant (p : ℝ) (h1 : p = 0.5) : at_least_one_pit_replant_prob = 1 - (1 - (1 - p)^3)^3 := by
  sorry

end pit_A_no_replant_exactly_one_pit_no_replant_at_least_one_replant_l2224_222484


namespace probability_of_picking_same_color_shoes_l2224_222460

theorem probability_of_picking_same_color_shoes
  (n_pairs_black : ℕ) (n_pairs_brown : ℕ) (n_pairs_gray : ℕ)
  (h_black_pairs : n_pairs_black = 8)
  (h_brown_pairs : n_pairs_brown = 4)
  (h_gray_pairs : n_pairs_gray = 3)
  (total_shoes : ℕ := 2 * (n_pairs_black + n_pairs_brown + n_pairs_gray)) :
  (16 / total_shoes * 8 / (total_shoes - 1) + 
   8 / total_shoes * 4 / (total_shoes - 1) + 
   6 / total_shoes * 3 / (total_shoes - 1)) = 89 / 435 :=
by
  sorry

end probability_of_picking_same_color_shoes_l2224_222460


namespace find_lesser_number_l2224_222403

theorem find_lesser_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 :=
by sorry

end find_lesser_number_l2224_222403


namespace solve_for_n_l2224_222428

theorem solve_for_n (n : ℝ) (h : 1 / (2 * n) + 1 / (4 * n) = 3 / 12) : n = 3 :=
sorry

end solve_for_n_l2224_222428


namespace domain_lg_function_l2224_222451

theorem domain_lg_function (x : ℝ) : (1 + x > 0 ∧ x - 1 > 0) ↔ (1 < x) :=
by
  sorry

end domain_lg_function_l2224_222451


namespace part1_part2_l2224_222464
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (3 * x - 1) + a * x + 3

theorem part1 (x : ℝ) : (f x 1) ≤ 5 ↔ (-1/2 : ℝ) ≤ x ∧ x ≤ 3/4 := by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, ∀ y : ℝ, f x a ≥ f y a) ↔ (-3 : ℝ) ≤ a ∧ a ≤ 3 := by
  sorry

end part1_part2_l2224_222464


namespace second_number_is_650_l2224_222401

theorem second_number_is_650 (x : ℝ) (h1 : 0.20 * 1600 = 0.20 * x + 190) : x = 650 :=
by sorry

end second_number_is_650_l2224_222401


namespace adam_remaining_loads_l2224_222483

-- Define the initial conditions
def total_loads : ℕ := 25
def washed_loads : ℕ := 6

-- Define the remaining loads as the total loads minus the washed loads
def remaining_loads (total_loads washed_loads : ℕ) : ℕ := total_loads - washed_loads

-- State the theorem to be proved
theorem adam_remaining_loads : remaining_loads total_loads washed_loads = 19 := by
  sorry

end adam_remaining_loads_l2224_222483


namespace lifespan_difference_l2224_222481

variable (H : ℕ)

theorem lifespan_difference (H : ℕ) (bat_lifespan : ℕ) (frog_lifespan : ℕ) (total_lifespan : ℕ) 
    (hb : bat_lifespan = 10)
    (hf : frog_lifespan = 4 * H)
    (ht : H + bat_lifespan + frog_lifespan = total_lifespan)
    (t30 : total_lifespan = 30) :
    bat_lifespan - H = 6 :=
by
  -- here would be the proof
  sorry

end lifespan_difference_l2224_222481


namespace percentage_reduction_is_20_l2224_222470

noncomputable def reduction_in_length (L W : ℝ) (x : ℝ) := 
  (L * (1 - x / 100)) * (W * 1.25) = L * W

theorem percentage_reduction_is_20 (L W : ℝ) : 
  reduction_in_length L W 20 := 
by 
  unfold reduction_in_length
  sorry

end percentage_reduction_is_20_l2224_222470


namespace sum_of_number_and_its_radical_conjugate_l2224_222416

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l2224_222416


namespace largest_divisor_of_product_of_consecutive_evens_l2224_222437

theorem largest_divisor_of_product_of_consecutive_evens (n : ℤ) : 
  ∃ d, d = 8 ∧ ∀ n, d ∣ (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) :=
sorry

end largest_divisor_of_product_of_consecutive_evens_l2224_222437


namespace table_covered_area_l2224_222459

-- Definitions based on conditions
def length := 12
def width := 1
def number_of_strips := 4
def overlapping_strips := 3

-- Calculating the area of one strip
def area_of_one_strip := length * width

-- Calculating total area assuming no overlaps
def total_area_no_overlap := number_of_strips * area_of_one_strip

-- Calculating the total overlap area
def overlap_area := overlapping_strips * (width * width)

-- Final area after subtracting overlaps
def final_covered_area := total_area_no_overlap - overlap_area

-- Theorem stating the proof problem
theorem table_covered_area : final_covered_area = 45 :=
by
  sorry

end table_covered_area_l2224_222459


namespace salesman_bonus_l2224_222413

theorem salesman_bonus (S B : ℝ) 
  (h1 : S > 10000) 
  (h2 : 0.09 * S + 0.03 * (S - 10000) = 1380) 
  : B = 0.03 * (S - 10000) :=
sorry

end salesman_bonus_l2224_222413


namespace alicia_local_tax_in_cents_l2224_222417

theorem alicia_local_tax_in_cents (hourly_wage : ℝ) (tax_rate : ℝ)
  (h_hourly_wage : hourly_wage = 30) (h_tax_rate : tax_rate = 0.021) :
  (hourly_wage * tax_rate * 100) = 63 := by
  sorry

end alicia_local_tax_in_cents_l2224_222417


namespace part1_part2_l2224_222488

theorem part1 (x y : ℕ) (h1 : 25 * x + 30 * y = 1500) (h2 : x = 2 * y - 4) : x = 36 ∧ y = 20 :=
by
  sorry

theorem part2 (x y : ℕ) (h1 : x + y = 60) (h2 : x ≥ 2 * y)
  (h_profit : ∃ p, p = 7 * x + 10 * y) : 
  ∃ x y profit, x = 40 ∧ y = 20 ∧ profit = 480 :=
by
  sorry

end part1_part2_l2224_222488


namespace find_y_l2224_222436

def F (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y : ∃ y : ℕ, F 3 y 5 15 = 490 ∧ y = 6 := by
  sorry

end find_y_l2224_222436


namespace road_unrepaired_is_42_percent_statement_is_false_l2224_222440

def road_length : ℝ := 1
def phase1_completion : ℝ := 0.40
def phase2_remaining_factor : ℝ := 0.30

def remaining_road (road : ℝ) (phase1 : ℝ) (phase2_factor : ℝ) : ℝ :=
  road - phase1 - (road - phase1) * phase2_factor

theorem road_unrepaired_is_42_percent (road_length : ℝ) (phase1_completion : ℝ) (phase2_remaining_factor : ℝ) :
  remaining_road road_length phase1_completion phase2_remaining_factor = 0.42 :=
by
  sorry

theorem statement_is_false : ¬(remaining_road road_length phase1_completion phase2_remaining_factor = 0.30) :=
by
  sorry

end road_unrepaired_is_42_percent_statement_is_false_l2224_222440


namespace remainder_sum_mod_13_l2224_222429

theorem remainder_sum_mod_13 (a b c d : ℕ) 
(h₁ : a % 13 = 3) (h₂ : b % 13 = 5) (h₃ : c % 13 = 7) (h₄ : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 :=
by sorry

end remainder_sum_mod_13_l2224_222429


namespace find_x0_l2224_222402

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem find_x0 (x0 : ℝ) (h : deriv f x0 = 0) : x0 = Real.exp 1 :=
by 
  sorry

end find_x0_l2224_222402


namespace max_contestants_l2224_222434

theorem max_contestants (n : ℕ) (h1 : n = 55) (h2 : ∀ (i j : ℕ), i < j → j < n → (j - i) % 5 ≠ 4) : ∃(k : ℕ), k = 30 := 
  sorry

end max_contestants_l2224_222434


namespace domain_of_f_l2224_222410

noncomputable def f (x : ℝ) : ℝ := (x + 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_f : 
  {x : ℝ | Real.sqrt (x^2 - 5 * x + 6) ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l2224_222410


namespace trajectory_of_center_l2224_222496

-- Define a structure for Point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the given point A
def A : Point := { x := -2, y := 0 }

-- Define a property for the circle being tangent to a line
def tangent_to_line (center : Point) (line_x : ℝ) : Prop :=
  center.x + line_x = 0

-- The main theorem to be proved
theorem trajectory_of_center :
  ∀ (C : Point), tangent_to_line C 2 → (C.y)^2 = -8 * C.x :=
sorry

end trajectory_of_center_l2224_222496


namespace find_w_l2224_222412

theorem find_w (a w : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 45 * w) : w = 49 :=
by
  sorry

end find_w_l2224_222412


namespace work_completion_time_l2224_222474

theorem work_completion_time (A_rate B_rate : ℝ) (hA : A_rate = 1/60) (hB : B_rate = 1/20) :
  1 / (A_rate + B_rate) = 15 :=
by
  sorry

end work_completion_time_l2224_222474


namespace units_digit_47_power_47_l2224_222467

theorem units_digit_47_power_47 : (47^47) % 10 = 3 :=
by
  sorry

end units_digit_47_power_47_l2224_222467


namespace option_A_option_B_option_C_option_D_l2224_222454

namespace Inequalities

theorem option_A (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  a + (1/a) > b + (1/b) :=
sorry

theorem option_B (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  (m + 1) / (n + 1) < m / n :=
sorry

theorem option_C (c a b : ℝ) (hc : c > 0) (ha : a > 0) (hb : b > 0) (hca : c > a) (hab : a > b) :
  a / (c - a) > b / (c - b) :=
sorry

theorem option_D (a b : ℝ) (ha : a > -1) (hb : b > -1) (hab : a ≥ b) :
  a / (a + 1) ≥ b / (b + 1) :=
sorry

end Inequalities

end option_A_option_B_option_C_option_D_l2224_222454


namespace find_y_given_x_eq_neg6_l2224_222482

theorem find_y_given_x_eq_neg6 :
  ∀ (y : ℤ), (∃ (x : ℤ), x = -6 ∧ x^2 - x + 6 = y - 6) → y = 54 :=
by
  intros y h
  obtain ⟨x, hx1, hx2⟩ := h
  rw [hx1] at hx2
  simp at hx2
  linarith

end find_y_given_x_eq_neg6_l2224_222482


namespace storks_initially_l2224_222400

-- Definitions for conditions
variable (S : ℕ) -- initial number of storks
variable (B : ℕ) -- initial number of birds

theorem storks_initially (h1 : B = 2) (h2 : S = B + 3 + 1) : S = 6 := by
  -- proof goes here
  sorry

end storks_initially_l2224_222400


namespace ryan_time_learning_l2224_222418

variable (t : ℕ) (c : ℕ)

/-- Ryan spends a total of 3 hours on both languages every day. Assume further that he spends 1 hour on learning Chinese every day, and you need to find how many hours he spends on learning English. --/
theorem ryan_time_learning (h_total : t = 3) (h_chinese : c = 1) : (t - c) = 2 := 
by
  -- Proof goes here
  sorry

end ryan_time_learning_l2224_222418


namespace hash_hash_hash_72_eq_12_5_l2224_222414

def hash (N : ℝ) : ℝ := 0.5 * N + 2

theorem hash_hash_hash_72_eq_12_5 : hash (hash (hash 72)) = 12.5 := 
by
  sorry

end hash_hash_hash_72_eq_12_5_l2224_222414


namespace find_first_number_l2224_222448

theorem find_first_number (y x : ℤ) (h1 : (y + 76 + x) / 3 = 5) (h2 : x = -63) : y = 2 :=
by
  -- To be filled in with the proof steps
  sorry

end find_first_number_l2224_222448


namespace total_seeds_in_watermelon_l2224_222498

theorem total_seeds_in_watermelon :
  let slices := 40
  let black_seeds_per_slice := 20
  let white_seeds_per_slice := 20
  let total_black_seeds := black_seeds_per_slice * slices
  let total_white_seeds := white_seeds_per_slice * slices
  total_black_seeds + total_white_seeds = 1600 := by
  sorry

end total_seeds_in_watermelon_l2224_222498


namespace rectangle_diagonal_opposite_vertex_l2224_222495

theorem rectangle_diagonal_opposite_vertex :
  ∀ (x y : ℝ),
    (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
      (x1, y1) = (5, 10) ∧ (x2, y2) = (15, -6) ∧ (x3, y3) = (11, 2) ∧
      (∃ (mx my : ℝ), mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2 ∧
        mx = (x + x3) / 2 ∧ my = (y + y3) / 2) ∧
      x = 9 ∧ y = 2) :=
by
  sorry

end rectangle_diagonal_opposite_vertex_l2224_222495


namespace maximum_F_value_l2224_222405

open Real

noncomputable def F (a b c x : ℝ) := abs ((a * x^2 + b * x + c) * (c * x^2 + b * x + a))

theorem maximum_F_value (a b c : ℝ) (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1)
    (hfx : abs (a * x^2 + b * x + c) ≤ 1) :
    ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ F a b c x = 2 := 
  sorry

end maximum_F_value_l2224_222405


namespace total_brown_mms_3rd_4th_bags_l2224_222469

def brown_mms_in_bags := (9 : ℕ) + (12 : ℕ) + (3 : ℕ)

def total_bags := 5

def average_mms_per_bag := 8

theorem total_brown_mms_3rd_4th_bags (x y : ℕ) 
  (h1 : brown_mms_in_bags + x + y = average_mms_per_bag * total_bags) : 
  x + y = 16 :=
by
  have h2 : brown_mms_in_bags + x + y = 40 := by sorry
  sorry

end total_brown_mms_3rd_4th_bags_l2224_222469


namespace strawberries_eaten_l2224_222445

-- Definitions based on the conditions
def strawberries_picked : ℕ := 35
def strawberries_remaining : ℕ := 33

-- Statement of the proof problem
theorem strawberries_eaten :
  strawberries_picked - strawberries_remaining = 2 :=
by
  sorry

end strawberries_eaten_l2224_222445


namespace width_of_road_correct_l2224_222486

-- Define the given conditions
def sum_of_circumferences (r R : ℝ) : Prop := 2 * Real.pi * r + 2 * Real.pi * R = 88
def radius_relation (r R : ℝ) : Prop := r = (1/3) * R
def width_of_road (R r : ℝ) := R - r

-- State the main theorem
theorem width_of_road_correct (R r : ℝ) (h1 : sum_of_circumferences r R) (h2 : radius_relation r R) :
    width_of_road R r = 22 / Real.pi := by
  sorry

end width_of_road_correct_l2224_222486


namespace sum_of_fourth_powers_l2224_222427

theorem sum_of_fourth_powers (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 37 / 6 := 
sorry

end sum_of_fourth_powers_l2224_222427


namespace old_man_gold_coins_l2224_222443

theorem old_man_gold_coins (x y : ℕ) (h1 : x - y = 1) (h2 : x^2 - y^2 = 25 * (x - y)) : x + y = 25 := 
sorry

end old_man_gold_coins_l2224_222443


namespace range_of_k_l2224_222426

noncomputable def f (k x : ℝ) := k * x - Real.exp x
noncomputable def g (x : ℝ) := Real.exp x / x

theorem range_of_k (k : ℝ) (h : ∃ x : ℝ, x ≠ 0 ∧ f k x = 0) :
  k < 0 ∨ k ≥ Real.exp 1 := sorry

end range_of_k_l2224_222426
