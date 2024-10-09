import Mathlib

namespace silvia_last_play_without_breach_l1473_147367

theorem silvia_last_play_without_breach (N : ℕ) : 
  36 * N < 2000 ∧ 72 * N ≥ 2000 ↔ N = 28 :=
by
  sorry

end silvia_last_play_without_breach_l1473_147367


namespace max_value_expr_max_l1473_147320

noncomputable def max_value_expr (x : ℝ) : ℝ :=
  (x^2 + 3 - (x^4 + 9).sqrt) / x

theorem max_value_expr_max (x : ℝ) (hx : 0 < x) :
  max_value_expr x ≤ (6 * (6:ℝ).sqrt) / (6 + 3 * (2:ℝ).sqrt) :=
sorry

end max_value_expr_max_l1473_147320


namespace train_length_l1473_147389

theorem train_length 
  (L : ℝ) -- Length of each train in meters.
  (speed_fast : ℝ := 56) -- Speed of the faster train in km/hr.
  (speed_slow : ℝ := 36) -- Speed of the slower train in km/hr.
  (time_pass : ℝ := 72) -- Time taken for the faster train to pass the slower train in seconds.
  (km_to_m_s : ℝ := 5 / 18) -- Conversion factor from km/hr to m/s.
  (relative_speed : ℝ := (speed_fast - speed_slow) * km_to_m_s) -- Relative speed in m/s.
  (distance_covered : ℝ := relative_speed * time_pass) -- Distance covered in meters.
  (equal_length : 2 * L = distance_covered) -- Condition of the problem: 2L = distance covered.
  : L = 200.16 :=
sorry

end train_length_l1473_147389


namespace maximum_fraction_l1473_147355

theorem maximum_fraction (A B : ℕ) (h1 : A ≠ B) (h2 : 0 < A ∧ A < 1000) (h3 : 0 < B ∧ B < 1000) :
  ∃ (A B : ℕ), (A = 500) ∧ (B = 499) ∧ (A ≠ B) ∧ (0 < A ∧ A < 1000) ∧ (0 < B ∧ B < 1000) ∧ (A - B = 1) ∧ (A + B = 999) ∧ (499 / 500 = 0.998) := sorry

end maximum_fraction_l1473_147355


namespace square_of_105_l1473_147334

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l1473_147334


namespace find_alpha_angle_l1473_147376

theorem find_alpha_angle :
  ∃ α : ℝ, (7 * α + 8 * α + 45) = 180 ∧ α = 9 :=
by 
  sorry

end find_alpha_angle_l1473_147376


namespace original_design_ratio_built_bridge_ratio_l1473_147306

-- Definitions
variables (v1 v2 r1 r2 : ℝ)

-- Conditions as per the problem
def original_height_relation : Prop := v1 = 3 * v2
def built_radius_relation : Prop := r2 = 2 * r1

-- Prove the required ratios
theorem original_design_ratio (h1 : original_height_relation v1 v2) (h2 : built_radius_relation r1 r2) : (v1 / r1 = 3 / 4) := sorry

theorem built_bridge_ratio (h1 : original_height_relation v1 v2) (h2 : built_radius_relation r1 r2) : (v2 / r2 = 1 / 8) := sorry

end original_design_ratio_built_bridge_ratio_l1473_147306


namespace binomial_expansion_calculation_l1473_147339

theorem binomial_expansion_calculation :
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 101^5 :=
by
  sorry

end binomial_expansion_calculation_l1473_147339


namespace chromium_percentage_in_new_alloy_l1473_147359

noncomputable def percentage_chromium_new_alloy (w1 w2 p1 p2 : ℝ) : ℝ :=
  ((p1 * w1 + p2 * w2) / (w1 + w2)) * 100

theorem chromium_percentage_in_new_alloy :
  percentage_chromium_new_alloy 15 35 0.12 0.10 = 10.6 :=
by
  sorry

end chromium_percentage_in_new_alloy_l1473_147359


namespace smallest_positive_integer_divisible_conditions_l1473_147384

theorem smallest_positive_integer_divisible_conditions :
  ∃ (M : ℕ), M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M = 419 :=
sorry

end smallest_positive_integer_divisible_conditions_l1473_147384


namespace largest_and_smallest_values_quartic_real_roots_l1473_147312

noncomputable def function_y (a b x : ℝ) : ℝ :=
  (4 * a^2 * x^2 + b^2 * (x^2 - 1)^2) / (x^2 + 1)^2

theorem largest_and_smallest_values (a b : ℝ) (h : a > b) :
  ∃ x y, function_y a b x = y^2 ∧ y = a ∧ y = b :=
by
  sorry

theorem quartic_real_roots (a b y : ℝ) (h₁ : a > b) (h₂ : y > b) (h₃ : y < a) :
  ∃ x₀ x₁ x₂ x₃, function_y a b x₀ = y^2 ∧ function_y a b x₁ = y^2 ∧ function_y a b x₂ = y^2 ∧ function_y a b x₃ = y^2 :=
by
  sorry

end largest_and_smallest_values_quartic_real_roots_l1473_147312


namespace least_number_to_add_l1473_147351

theorem least_number_to_add (LCM : ℕ) (a : ℕ) (x : ℕ) :
  LCM = 23 * 29 * 31 →
  a = 1076 →
  x = LCM - a →
  (a + x) % LCM = 0 :=
by
  sorry

end least_number_to_add_l1473_147351


namespace simplify_expression_l1473_147392

variable (x : ℝ)

theorem simplify_expression : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 :=
by
  sorry

end simplify_expression_l1473_147392


namespace real_root_fraction_l1473_147396

theorem real_root_fraction (a b : ℝ) 
  (h_cond_a : a^4 - 7 * a - 3 = 0) 
  (h_cond_b : b^4 - 7 * b - 3 = 0)
  (h_order : a > b) : 
  (a - b) / (a^4 - b^4) = 1 / 7 := 
sorry

end real_root_fraction_l1473_147396


namespace kenya_peanut_count_l1473_147336

-- Define the number of peanuts Jose has
def jose_peanuts : ℕ := 85

-- Define the number of additional peanuts Kenya has more than Jose
def additional_peanuts : ℕ := 48

-- Define the number of peanuts Kenya has
def kenya_peanuts : ℕ := jose_peanuts + additional_peanuts

-- Theorem to prove the number of peanuts Kenya has
theorem kenya_peanut_count : kenya_peanuts = 133 := by
  sorry

end kenya_peanut_count_l1473_147336


namespace arithmetic_sequence_fifth_term_l1473_147382

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 3) 
  (h2 : a + 11 * d = 9) : 
  a + 4 * d = -12 :=
by
  sorry

end arithmetic_sequence_fifth_term_l1473_147382


namespace proof_problem_l1473_147300

-- Definitions
def U : Set ℕ := {x | x < 7 ∧ x > 0}
def A : Set ℕ := {1, 2, 5}
def B : Set ℕ := {2, 3, 4, 5}

-- The equality proof statement
theorem proof_problem :
  (A ∩ B = {2, 5}) ∧
  ({x | x ∈ U ∧ ¬ (x ∈ A)} = {3, 4, 6}) ∧
  (A ∪ {x | x ∈ U ∧ ¬ (x ∈ B)} = {1, 2, 5, 6}) :=
by
  sorry

end proof_problem_l1473_147300


namespace complement_union_l1473_147358

open Set

variable (U M N : Set ℕ)

def complement_U (A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

theorem complement_union (hU : U = {0, 1, 2, 3, 4, 5, 6})
                          (hM : M = {1, 3, 5})
                          (hN : N = {2, 4, 6}) :
  (complement_U U M) ∪ (complement_U U N) = {0, 1, 2, 3, 4, 5, 6} :=
by 
  sorry

end complement_union_l1473_147358


namespace range_of_m_l1473_147385

def prop_p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0
def prop_q (m : ℝ) : Prop := ∃ (x y : ℝ), (x^2) / (m-6) - (y^2) / (m+3) = 1

theorem range_of_m (m : ℝ) : ¬ (prop_p m ∧ prop_q m) → m ≥ -3 :=
sorry

end range_of_m_l1473_147385


namespace juice_left_l1473_147311

theorem juice_left (total consumed : ℚ) (h_total : total = 1) (h_consumed : consumed = 4 / 6) :
  total - consumed = 2 / 6 ∨ total - consumed = 1 / 3 :=
by
  sorry

end juice_left_l1473_147311


namespace grid_cut_990_l1473_147332

theorem grid_cut_990 (grid : Matrix (Fin 1000) (Fin 1000) (Fin 2)) :
  (∃ (rows_to_remove : Finset (Fin 1000)), rows_to_remove.card = 990 ∧ 
   ∀ col : Fin 1000, ∃ row ∈ (Finset.univ \ rows_to_remove), grid row col = 1) ∨
  (∃ (cols_to_remove : Finset (Fin 1000)), cols_to_remove.card = 990 ∧ 
   ∀ row : Fin 1000, ∃ col ∈ (Finset.univ \ cols_to_remove), grid row col = 0) :=
sorry

end grid_cut_990_l1473_147332


namespace find_speed_of_faster_train_l1473_147393

noncomputable def speed_of_faster_train
  (length_each_train_m : ℝ)
  (speed_slower_kmph : ℝ)
  (time_pass_s : ℝ) : ℝ :=
  let distance_km := (2 * length_each_train_m / 1000)
  let time_pass_hr := (time_pass_s / 3600)
  let relative_speed_kmph := (distance_km / time_pass_hr)
  let speed_faster_kmph := (relative_speed_kmph - speed_slower_kmph)
  speed_faster_kmph

theorem find_speed_of_faster_train :
  speed_of_faster_train
    250   -- length_each_train_m
    30    -- speed_slower_kmph
    23.998080153587715 -- time_pass_s
  = 45 := sorry

end find_speed_of_faster_train_l1473_147393


namespace inequality_positive_numbers_l1473_147310

theorem inequality_positive_numbers (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (x + 2 * y + 3 * z)) + (y / (y + 2 * z + 3 * x)) + (z / (z + 2 * x + 3 * y)) ≤ 4 / 3 :=
by
  sorry

end inequality_positive_numbers_l1473_147310


namespace convinced_of_twelve_models_vitya_review_58_offers_l1473_147395

noncomputable def ln : ℝ → ℝ := Real.log

theorem convinced_of_twelve_models (n : ℕ) (h_n : n ≥ 13) :
  ∃ k : ℕ, (12 / n : ℝ) ^ k < 0.01 := sorry

theorem vitya_review_58_offers :
  ∃ k : ℕ, (12 / 13 : ℝ) ^ k < 0.01 ∧ k = 58 := sorry

end convinced_of_twelve_models_vitya_review_58_offers_l1473_147395


namespace Chandler_saves_enough_l1473_147325

theorem Chandler_saves_enough (total_cost gift_money weekly_earnings : ℕ)
  (h_cost : total_cost = 550)
  (h_gift : gift_money = 130)
  (h_weekly : weekly_earnings = 18) : ∃ x : ℕ, (130 + 18 * x) >= 550 ∧ x = 24 := 
by
  sorry

end Chandler_saves_enough_l1473_147325


namespace cyclic_proportion_l1473_147308

variable {A B C p q r : ℝ}

theorem cyclic_proportion (h1 : A / B = p) (h2 : B / C = q) (h3 : C / A = r) :
  ∃ x y z, A = x ∧ B = y ∧ C = z ∧ x / y = p ∧ y / z = q ∧ z / x = r ∧
  x = (p^2 * q / r)^(1/3:ℝ) ∧ y = (q^2 * r / p)^(1/3:ℝ) ∧ z = (r^2 * p / q)^(1/3:ℝ) :=
by sorry

end cyclic_proportion_l1473_147308


namespace m_over_n_eq_l1473_147313

variables (m n : ℝ)
variables (x y x1 y1 x2 y2 x0 y0 : ℝ)

-- Ellipse equation
axiom ellipse_eq : m * x^2 + n * y^2 = 1

-- Line equation
axiom line_eq : x + y = 1

-- Points M and N on the ellipse
axiom M_point : m * x1^2 + n * y1^2 = 1
axiom N_point : m * x2^2 + n * y2^2 = 1

-- Midpoint of MN is P
axiom P_midpoint : x0 = (x1 + x2) / 2 ∧ y0 = (y1 + y2) / 2

-- Slope of OP
axiom slope_OP : y0 / x0 = (Real.sqrt 2) / 2

theorem m_over_n_eq : m / n = (Real.sqrt 2) / 2 :=
sorry

end m_over_n_eq_l1473_147313


namespace smallest_possible_value_l1473_147373

open Nat

theorem smallest_possible_value (c d : ℕ) (hc : c > d) (hc_pos : 0 < c) (hd_pos : 0 < d) (odd_cd : ¬Even (c + d)) :
  (∃ (y : ℚ), y > 0 ∧ y = (c + d : ℚ) / (c - d) + (c - d : ℚ) / (c + d) ∧ y = 10 / 3) :=
by
  sorry

end smallest_possible_value_l1473_147373


namespace calculate_selling_price_l1473_147379

theorem calculate_selling_price (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 1500 → 
  loss_percentage = 0.17 →
  selling_price = cost_price - (loss_percentage * cost_price) →
  selling_price = 1245 :=
by 
  intros hc hl hs
  rw [hc, hl] at hs
  norm_num at hs
  exact hs

end calculate_selling_price_l1473_147379


namespace solve_xyz_l1473_147331

def is_solution (x y z : ℕ) : Prop :=
  x * y + y * z + z * x = 2 * (x + y + z)

theorem solve_xyz (x y z : ℕ) :
  is_solution x y z ↔ (x = 1 ∧ y = 2 ∧ z = 4) ∨
                     (x = 1 ∧ y = 4 ∧ z = 2) ∨
                     (x = 2 ∧ y = 1 ∧ z = 4) ∨
                     (x = 2 ∧ y = 4 ∧ z = 1) ∨
                     (x = 2 ∧ y = 2 ∧ z = 2) ∨
                     (x = 4 ∧ y = 1 ∧ z = 2) ∨
                     (x = 4 ∧ y = 2 ∧ z = 1) := sorry

end solve_xyz_l1473_147331


namespace least_5_digit_divisible_by_12_15_18_l1473_147321

theorem least_5_digit_divisible_by_12_15_18 : 
  ∃ n, n >= 10000 ∧ n < 100000 ∧ (180 ∣ n) ∧ n = 10080 :=
by
  -- Proof goes here
  sorry

end least_5_digit_divisible_by_12_15_18_l1473_147321


namespace minimum_value_polynomial_l1473_147305

def polynomial (x y : ℝ) : ℝ := 5 * x^2 - 4 * x * y + 4 * y^2 + 12 * x + 25

theorem minimum_value_polynomial : ∃ (m : ℝ), (∀ (x y : ℝ), polynomial x y ≥ m) ∧ m = 16 :=
by
  sorry

end minimum_value_polynomial_l1473_147305


namespace productivity_increase_l1473_147343

theorem productivity_increase (a b : ℝ) : (7 / 8) * (1 + 20 / 100) = 1.05 :=
by
  sorry

end productivity_increase_l1473_147343


namespace num_convex_numbers_without_repeats_l1473_147378

def is_convex_number (a b c : ℕ) : Prop :=
  a < b ∧ b > c

def is_valid_digit (n : ℕ) : Prop :=
  0 ≤ n ∧ n < 10

def distinct_digits (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem num_convex_numbers_without_repeats : 
  (∃ (numbers : Finset (ℕ × ℕ × ℕ)), 
    (∀ a b c, (a, b, c) ∈ numbers -> is_convex_number a b c ∧ is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ distinct_digits a b c) ∧
    numbers.card = 204) :=
sorry

end num_convex_numbers_without_repeats_l1473_147378


namespace odd_function_property_l1473_147322

noncomputable def odd_function := {f : ℝ → ℝ // ∀ x : ℝ, f (-x) = -f x}

theorem odd_function_property (f : odd_function) (h1 : f.1 1 = -2) : f.1 (-1) + f.1 0 = 2 := by
  sorry

end odd_function_property_l1473_147322


namespace find_function_range_of_a_l1473_147324

variables (a b : ℝ) (f : ℝ → ℝ) 

-- Given: f(x) = ax + b where a ≠ 0 
--        f(2x + 1) = 4x + 1
-- Prove: f(x) = 2x - 1
theorem find_function (h1 : ∀ x, f (2 * x + 1) = 4 * x + 1) : 
  ∃ a b, a = 2 ∧ b = -1 ∧ ∀ x, f x = a * x + b :=
by sorry

-- Given: A = {x | a - 1 < x < 2a +1 }
--        B = {x | 1 < f(x) < 3 }
--        B ⊆ A
-- Prove: 1/2 ≤ a ≤ 2
theorem range_of_a (Hf : ∀ x, f x = 2 * x - 1) (Hsubset: ∀ x, 1 < f x ∧ f x < 3 → a - 1 < x ∧ x < 2 * a + 1) :
  1 / 2 ≤ a ∧ a ≤ 2 :=
by sorry

end find_function_range_of_a_l1473_147324


namespace basketball_game_first_half_points_l1473_147326

noncomputable def total_points_first_half
  (eagles_points : ℕ → ℕ) (lions_points : ℕ → ℕ) (common_ratio : ℕ) (common_difference : ℕ) : ℕ :=
  eagles_points 0 + eagles_points 1 + lions_points 0 + lions_points 1

theorem basketball_game_first_half_points 
  (eagles_points lions_points : ℕ → ℕ)
  (common_ratio : ℕ) (common_difference : ℕ)
  (h1 : eagles_points 0 = lions_points 0)
  (h2 : ∀ n, eagles_points (n + 1) = common_ratio * eagles_points n)
  (h3 : ∀ n, lions_points (n + 1) = lions_points n + common_difference)
  (h4 : eagles_points 0 + eagles_points 1 + eagles_points 2 + eagles_points 3 =
        lions_points 0 + lions_points 1 + lions_points 2 + lions_points 3 + 3)
  (h5 : eagles_points 0 + eagles_points 1 + eagles_points 2 + eagles_points 3 ≤ 120)
  (h6 : lions_points 0 + lions_points 1 + lions_points 2 + lions_points 3 ≤ 120) :
  total_points_first_half eagles_points lions_points common_ratio common_difference = 15 :=
sorry

end basketball_game_first_half_points_l1473_147326


namespace proof_time_to_run_square_field_l1473_147397

def side : ℝ := 40
def speed_kmh : ℝ := 9
def perimeter (side : ℝ) : ℝ := 4 * side

noncomputable def speed_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

noncomputable def time_to_run (perimeter : ℝ) (speed_mps : ℝ) : ℝ := perimeter / speed_mps

theorem proof_time_to_run_square_field :
  time_to_run (perimeter side) (speed_mps speed_kmh) = 64 :=
by
  sorry

end proof_time_to_run_square_field_l1473_147397


namespace meet_time_same_departure_meet_time_staggered_departure_catch_up_time_same_departure_l1473_147327

-- Distance between locations A and B
def distance : ℝ := 448

-- Speed of the slow train
def slow_speed : ℝ := 60

-- Speed of the fast train
def fast_speed : ℝ := 80

-- Problem 1: Prove the two trains meet 3.2 hours after the fast train departs (both trains heading towards each other, departing at the same time)
theorem meet_time_same_departure : 
  (slow_speed + fast_speed) * 3.2 = distance :=
by
  sorry

-- Problem 2: Prove the two trains meet 3 hours after the fast train departs (slow train departs 28 minutes before the fast train)
theorem meet_time_staggered_departure : 
  (slow_speed * (28/60) + (slow_speed + fast_speed) * 3) = distance :=
by
  sorry

-- Problem 3: Prove the fast train catches up to the slow train 22.4 hours after departure (both trains heading in the same direction, departing at the same time)
theorem catch_up_time_same_departure : 
  (fast_speed - slow_speed) * 22.4 = distance :=
by
  sorry

end meet_time_same_departure_meet_time_staggered_departure_catch_up_time_same_departure_l1473_147327


namespace Dan_team_lost_games_l1473_147391

/-- Dan's high school played eighteen baseball games this year.
Two were at night and they won 15 games. Prove that they lost 3 games. -/
theorem Dan_team_lost_games (total_games won_games : ℕ) (h_total : total_games = 18) (h_won : won_games = 15) :
  total_games - won_games = 3 :=
by {
  sorry
}

end Dan_team_lost_games_l1473_147391


namespace find_equation_of_ellipse_C_l1473_147315

def equation_of_ellipse_C (a b : ℝ) : Prop :=
  ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1)

theorem find_equation_of_ellipse_C :
  ∀ (a b : ℝ), (a = 2) → (b = 1) →
  (equation_of_ellipse_C a b) →
  equation_of_ellipse_C 2 1 :=
by
  intros a b ha hb h
  sorry

end find_equation_of_ellipse_C_l1473_147315


namespace sum_of_numbers_l1473_147375

theorem sum_of_numbers (a b c : ℝ) (h1 : 2 * a + b = 46) (h2 : b + 2 * c = 53) (h3 : 2 * c + a = 29) :
  a + b + c = 48.8333 :=
by
  sorry

end sum_of_numbers_l1473_147375


namespace solve_for_x_l1473_147398

theorem solve_for_x : ∀ x : ℕ, x + 1315 + 9211 - 1569 = 11901 → x = 2944 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l1473_147398


namespace a_equals_b_l1473_147307

theorem a_equals_b (a b : ℕ) (h : a^3 + a + 4 * b^2 = 4 * a * b + b + b * a^2) : a = b := 
sorry

end a_equals_b_l1473_147307


namespace list_length_eq_12_l1473_147319

-- Define a list of numbers in the sequence
def seq : List ℝ := [1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5, 33.5, 37.5, 41.5, 45.5]

-- Define the theorem that states the number of elements in the sequence
theorem list_length_eq_12 : seq.length = 12 := 
by 
  -- Proof here
  sorry

end list_length_eq_12_l1473_147319


namespace gcd_solution_l1473_147369

theorem gcd_solution {m n : ℕ} (hm : m > 0) (hn : n > 0) (h : Nat.gcd m n = 10) : Nat.gcd (12 * m) (18 * n) = 60 := 
sorry

end gcd_solution_l1473_147369


namespace cost_price_is_925_l1473_147350

-- Definitions for the conditions
def SP : ℝ := 1110
def profit_percentage : ℝ := 0.20

-- Theorem to prove that the cost price is 925
theorem cost_price_is_925 (CP : ℝ) (h : SP = (CP * (1 + profit_percentage))) : CP = 925 := 
by sorry

end cost_price_is_925_l1473_147350


namespace equation1_solution_equation2_solution_l1473_147341

theorem equation1_solution (x : ℝ) : (x - 4)^2 - 9 = 0 ↔ (x = 7 ∨ x = 1) := 
sorry

theorem equation2_solution (x : ℝ) : (x + 1)^3 = -27 ↔ (x = -4) := 
sorry

end equation1_solution_equation2_solution_l1473_147341


namespace cubic_of_cubic_roots_correct_l1473_147302

variable (a b c : ℝ) (α β γ : ℝ)

-- Vieta's formulas conditions
axiom vieta1 : α + β + γ = -a
axiom vieta2 : α * β + β * γ + γ * α = b
axiom vieta3 : α * β * γ = -c

-- Define the polynomial whose roots are α³, β³, and γ³
def cubic_of_cubic_roots (x : ℝ) : ℝ :=
  x^3 + (a^3 - 3*a*b + 3*c)*x^2 + (b^3 + 3*c^2 - 3*a*b*c)*x + c^3

-- Prove that this polynomial has α³, β³, γ³ as roots
theorem cubic_of_cubic_roots_correct :
  ∀ x : ℝ, cubic_of_cubic_roots a b c x = 0 ↔ (x = α^3 ∨ x = β^3 ∨ x = γ^3) :=
sorry

end cubic_of_cubic_roots_correct_l1473_147302


namespace total_distance_traveled_l1473_147345

variable (vm vr t d_up d_down : ℝ)
variable (H_river_speed : vr = 3)
variable (H_row_speed : vm = 6)
variable (H_time : t = 1)

theorem total_distance_traveled (H_upstream : d_up = vm - vr) 
                                (H_downstream : d_down = vm + vr) 
                                (total_time : d_up / (vm - vr) + d_down / (vm + vr) = t) : 
                                2 * (d_up + d_down) = 4.5 := 
                                by
  sorry

end total_distance_traveled_l1473_147345


namespace sweets_ratio_l1473_147368

theorem sweets_ratio (number_orange_sweets : ℕ) (number_grape_sweets : ℕ) (max_sweets_per_tray : ℕ)
  (h1 : number_orange_sweets = 36) (h2 : number_grape_sweets = 44) (h3 : max_sweets_per_tray = 4) :
  (number_orange_sweets / max_sweets_per_tray) / (number_grape_sweets / max_sweets_per_tray) = 9 / 11 :=
by
  sorry

end sweets_ratio_l1473_147368


namespace symmetry_about_origin_l1473_147387

theorem symmetry_about_origin (m : ℝ) (A B : ℝ × ℝ) (hA : A = (2, -1)) (hB : B = (-2, m)) (h_sym : B = (-A.1, -A.2)) :
  m = 1 :=
by
  sorry

end symmetry_about_origin_l1473_147387


namespace sum_of_prime_factors_143_l1473_147323

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l1473_147323


namespace fraction_meaningful_l1473_147386

theorem fraction_meaningful (x : ℝ) : (¬ (x - 2 = 0)) ↔ (x ≠ 2) :=
by
  sorry

end fraction_meaningful_l1473_147386


namespace equality_of_a_and_b_l1473_147370

theorem equality_of_a_and_b
  (a b : ℕ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : 4 * a * b - 1 ∣ (4 * a ^ 2 - 1) ^ 2) : a = b := 
sorry

end equality_of_a_and_b_l1473_147370


namespace circle_area_l1473_147348

noncomputable def pointA : ℝ × ℝ := (2, 7)
noncomputable def pointB : ℝ × ℝ := (8, 5)

def is_tangent_with_intersection_on_x_axis (A B C : ℝ × ℝ) : Prop :=
  ∃ R : ℝ, ∃ r : ℝ, ∀ M : ℝ × ℝ, dist M C = R → dist A M = r ∧ dist B M = r

theorem circle_area (A B : ℝ × ℝ) (hA : A = (2, 7)) (hB : B = (8, 5))
    (h : ∃ C : ℝ × ℝ, is_tangent_with_intersection_on_x_axis A B C) 
    : ∃ R : ℝ, π * R^2 = 12.5 * π := 
sorry

end circle_area_l1473_147348


namespace Gunther_typing_correct_l1473_147353

def GuntherTypingProblem : Prop :=
  let first_phase := (160 * (120 / 3))
  let second_phase := (200 * (180 / 3))
  let third_phase := (50 * 60)
  let fourth_phase := (140 * (90 / 3))
  let total_words := first_phase + second_phase + third_phase + fourth_phase
  total_words = 26200

theorem Gunther_typing_correct : GuntherTypingProblem := by
  sorry

end Gunther_typing_correct_l1473_147353


namespace smallest_X_divisible_by_15_l1473_147329

theorem smallest_X_divisible_by_15 (T : ℕ) (h_pos : T > 0) (h_digits : ∀ (d : ℕ), d ∈ (Nat.digits 10 T) → d = 0 ∨ d = 1)
  (h_div15 : T % 15 = 0) : ∃ X : ℕ, X = T / 15 ∧ X = 74 :=
sorry

end smallest_X_divisible_by_15_l1473_147329


namespace line_equation_exists_l1473_147346

theorem line_equation_exists 
  (a b : ℝ) 
  (ha_pos: a > 0)
  (hb_pos: b > 0)
  (h_area: 1 / 2 * a * b = 2) 
  (h_diff: a - b = 3 ∨ b - a = 3) : 
  (∀ x y : ℝ, (x + 4 * y = 4 ∧ (x / a + y / b = 1)) ∨ (4 * x + y = 4 ∧ (x / a + y / b = 1))) :=
sorry

end line_equation_exists_l1473_147346


namespace find_tenth_term_l1473_147380

/- Define the general term formula -/
def a (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

/- Define the sum of the first n terms formula -/
def S (a1 d : ℤ) (n : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem find_tenth_term
  (a1 d : ℤ)
  (h1 : a a1 d 2 + a a1 d 5 = 19)
  (h2 : S a1 d 5 = 40) :
  a a1 d 10 = 29 := by
  /- Sorry used to skip the proof steps. -/
  sorry

end find_tenth_term_l1473_147380


namespace speed_of_current_11_00448_l1473_147344

/-- 
  The speed at which a man can row a boat in still water is 25 kmph.
  He takes 7.999360051195905 seconds to cover 80 meters downstream.
  Prove that the speed of the current is 11.00448 km/h.
-/
theorem speed_of_current_11_00448 :
  let speed_in_still_water_kmph := 25
  let distance_m := 80
  let time_s := 7.999360051195905
  (distance_m / time_s) * 3600 / 1000 - speed_in_still_water_kmph = 11.00448 :=
by
  sorry

end speed_of_current_11_00448_l1473_147344


namespace find_t_l1473_147304

variables {a b c r s t : ℝ}

theorem find_t (h1 : a + b + c = -3)
             (h2 : a * b + b * c + c * a = 4)
             (h3 : a * b * c = -1)
             (h4 : ∀ x, x^3 + 3*x^2 + 4*x + 1 = 0 → (x = a ∨ x = b ∨ x = c))
             (h5 : ∀ y, y^3 + r*y^2 + s*y + t = 0 → (y = a + b ∨ y = b + c ∨ y = c + a))
             : t = 11 :=
sorry

end find_t_l1473_147304


namespace percentage_loss_l1473_147301

theorem percentage_loss (selling_price_with_loss : ℝ)
    (desired_selling_price_for_profit : ℝ)
    (profit_percentage : ℝ) (actual_selling_price : ℝ)
    (calculated_loss_percentage : ℝ) :
    selling_price_with_loss = 16 →
    desired_selling_price_for_profit = 21.818181818181817 →
    profit_percentage = 20 →
    actual_selling_price = 18.181818181818182 →
    calculated_loss_percentage = 12 → 
    calculated_loss_percentage = (actual_selling_price - selling_price_with_loss) / actual_selling_price * 100 := 
sorry

end percentage_loss_l1473_147301


namespace largest_invertible_interval_l1473_147342

def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

theorem largest_invertible_interval (x : ℝ) (hx : x = 2) : 
  ∃ I : Set ℝ, (I = Set.univ ∩ {y | y ≥ 3 / 2}) ∧ ∀ y ∈ I, g y = 3 * (y - 3 / 2) ^ 2 - 11 / 4 ∧ g y ∈ I ∧ Function.Injective (g ∘ (fun z => z : I → ℝ)) :=
sorry

end largest_invertible_interval_l1473_147342


namespace tan_five_pi_over_four_eq_one_l1473_147363

theorem tan_five_pi_over_four_eq_one : Real.tan (5 * Real.pi / 4) = 1 :=
by sorry

end tan_five_pi_over_four_eq_one_l1473_147363


namespace lower_limit_of_range_l1473_147328

theorem lower_limit_of_range (x y : ℝ) (hx1 : 3 < x) (hx2 : x < 8) (hx3 : y < x) (hx4 : x < 10) (hx5 : x = 7) : 3 < y ∧ y ≤ 7 :=
by
  sorry

end lower_limit_of_range_l1473_147328


namespace sequence_formula_l1473_147388

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n / (1 + 2 * a n)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_formula_l1473_147388


namespace fraction_to_decimal_l1473_147360

theorem fraction_to_decimal : (7 / 32 : ℚ) = 0.21875 := 
by {
  sorry
}

end fraction_to_decimal_l1473_147360


namespace bottles_more_than_apples_l1473_147381

def regular_soda : ℕ := 72
def diet_soda : ℕ := 32
def apples : ℕ := 78

def total_bottles : ℕ := regular_soda + diet_soda

theorem bottles_more_than_apples : total_bottles - apples = 26 := by
  -- Proof will go here
  sorry

end bottles_more_than_apples_l1473_147381


namespace Jean_spots_l1473_147338

/--
Jean the jaguar has a total of 60 spots.
Half of her spots are located on her upper torso.
One-third of the spots are located on her back and hindquarters.
Jean has 30 spots on her upper torso.
Prove that Jean has 10 spots located on her sides.
-/
theorem Jean_spots (TotalSpots UpperTorsoSpots BackHindquartersSpots SidesSpots : ℕ)
  (h_half : UpperTorsoSpots = TotalSpots / 2)
  (h_back : BackHindquartersSpots = TotalSpots / 3)
  (h_total_upper : UpperTorsoSpots = 30)
  (h_total : TotalSpots = 60) :
  SidesSpots = 10 :=
by
  sorry

end Jean_spots_l1473_147338


namespace goals_scored_by_each_l1473_147372

theorem goals_scored_by_each (total_goals : ℕ) (percentage : ℕ) (two_players_goals : ℕ) (each_player_goals : ℕ)
  (H1 : total_goals = 300)
  (H2 : percentage = 20)
  (H3 : two_players_goals = (percentage * total_goals) / 100)
  (H4 : two_players_goals / 2 = each_player_goals) :
  each_player_goals = 30 := by
  sorry

end goals_scored_by_each_l1473_147372


namespace find_speeds_and_circumference_l1473_147349

variable (Va Vb : ℝ)
variable (l : ℝ)

axiom smaller_arc_condition : 10 * (Va + Vb) = 150
axiom larger_arc_condition : 14 * (Va + Vb) = l - 150
axiom travel_condition : l / Va = 90 / Vb 

theorem find_speeds_and_circumference :
  Va = 12 ∧ Vb = 3 ∧ l = 360 := by
  sorry

end find_speeds_and_circumference_l1473_147349


namespace tailor_cut_difference_l1473_147377

def dress_silk_cut : ℝ := 0.75
def dress_satin_cut : ℝ := 0.60
def dress_chiffon_cut : ℝ := 0.55
def pants_cotton_cut : ℝ := 0.50
def pants_polyester_cut : ℝ := 0.45

theorem tailor_cut_difference :
  (dress_silk_cut + dress_satin_cut + dress_chiffon_cut) - (pants_cotton_cut + pants_polyester_cut) = 0.95 :=
by
  sorry

end tailor_cut_difference_l1473_147377


namespace vector_magnitude_parallel_l1473_147303

theorem vector_magnitude_parallel (x : ℝ) 
  (h1 : 4 / x = 2 / 1) :
  ( Real.sqrt ((4 + x) ^ 2 + (2 + 1) ^ 2) ) = 3 * Real.sqrt 5 := 
sorry

end vector_magnitude_parallel_l1473_147303


namespace cookie_cost_l1473_147333

theorem cookie_cost
  (classes3 : ℕ) (students_per_class3 : ℕ)
  (classes4 : ℕ) (students_per_class4 : ℕ)
  (classes5 : ℕ) (students_per_class5 : ℕ)
  (hamburger_cost : ℝ) (carrot_cost : ℝ) (total_lunch_cost : ℝ) (cookie_cost : ℝ)
  (h1 : classes3 = 5) (h2 : students_per_class3 = 30)
  (h3 : classes4 = 4) (h4 : students_per_class4 = 28)
  (h5 : classes5 = 4) (h6 : students_per_class5 = 27)
  (h7 : hamburger_cost = 2.10) (h8 : carrot_cost = 0.50)
  (h9 : total_lunch_cost = 1036):
  ((classes3 * students_per_class3) + (classes4 * students_per_class4) + (classes5 * students_per_class5)) * (cookie_cost + hamburger_cost + carrot_cost) = total_lunch_cost → 
  cookie_cost = 0.20 := 
by 
  sorry

end cookie_cost_l1473_147333


namespace triangle_third_side_l1473_147317

theorem triangle_third_side (a b c : ℝ) (h1 : a = 5) (h2 : b = 7) (h3 : 2 < c ∧ c < 12) : c = 6 :=
sorry

end triangle_third_side_l1473_147317


namespace each_person_has_5_bags_l1473_147390

def people := 6
def weight_per_bag := 50
def max_plane_weight := 6000
def additional_capacity := 90

theorem each_person_has_5_bags :
  (max_plane_weight / weight_per_bag - additional_capacity) / people = 5 :=
by
  sorry

end each_person_has_5_bags_l1473_147390


namespace richard_older_than_david_l1473_147340

theorem richard_older_than_david
  (R D S : ℕ)   -- ages of Richard, David, Scott
  (x : ℕ)       -- the number of years Richard is older than David
  (h1 : R = D + x)
  (h2 : D = S + 8)
  (h3 : R + 8 = 2 * (S + 8))
  (h4 : D = 14) : 
  x = 6 := sorry

end richard_older_than_david_l1473_147340


namespace find_third_divisor_l1473_147365

theorem find_third_divisor 
  (h1 : ∃ (n : ℕ), n = 1014 - 3 ∧ n % 12 = 0 ∧ n % 16 = 0 ∧ n % 21 = 0 ∧ n % 28 = 0) 
  (h2 : 1011 - 3 = 1008) : 
  (∃ d, d = 3 ∧ 1008 % d = 0 ∧ 1008 % 12 = 0 ∧ 1008 % 16 = 0 ∧ 1008 % 21 = 0 ∧ 1008 % 28 = 0) :=
sorry

end find_third_divisor_l1473_147365


namespace find_a6_l1473_147352

-- Defining the conditions of the problem
def a1 := 2
def S3 := 12

-- Defining the necessary arithmetic sequence properties
def Sn (a1 d : ℕ) (n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2
def an (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Proof statement in Lean
theorem find_a6 (d : ℕ) (a1_val S3_val : ℕ) (h1 : a1_val = 2) (h2 : S3_val = 12) 
    (h3 : 3 * (2 * a1_val + (3 - 1) * d) / 2 = S3_val) : an a1_val d 6 = 12 :=
by 
  -- omitted proof
  sorry

end find_a6_l1473_147352


namespace jill_and_emily_total_peaches_l1473_147337

-- Define each person and their conditions
variables (Steven Jake Jill Maria Emily : ℕ)

-- Given conditions
def steven_has_peaches : Steven = 14 := sorry
def jake_has_fewer_than_steven : Jake = Steven - 6 := sorry
def jake_has_more_than_jill : Jake = Jill + 3 := sorry
def maria_has_twice_jake : Maria = 2 * Jake := sorry
def emily_has_fewer_than_maria : Emily = Maria - 9 := sorry

-- The theorem statement combining the conditions and the required result
theorem jill_and_emily_total_peaches (Steven Jake Jill Maria Emily : ℕ)
  (h1 : Steven = 14) 
  (h2 : Jake = Steven - 6) 
  (h3 : Jake = Jill + 3) 
  (h4 : Maria = 2 * Jake) 
  (h5 : Emily = Maria - 9) : 
  Jill + Emily = 12 := 
sorry

end jill_and_emily_total_peaches_l1473_147337


namespace peter_spent_on_repairs_l1473_147364

variable (C : ℝ)

def repairs_cost (C : ℝ) := 0.10 * C

def profit (C : ℝ) := 1.20 * C - C

theorem peter_spent_on_repairs :
  ∀ C, profit C = 1100 → repairs_cost C = 550 :=
by
  intro C
  sorry

end peter_spent_on_repairs_l1473_147364


namespace students_in_sixth_level_l1473_147357

theorem students_in_sixth_level (S : ℕ)
  (h1 : ∃ S₄ : ℕ, S₄ = 4 * S)
  (h2 : ∃ S₇ : ℕ, S₇ = 2 * (4 * S))
  (h3 : S + 4 * S + 2 * (4 * S) = 520) :
  S = 40 :=
by
  sorry

end students_in_sixth_level_l1473_147357


namespace range_a_l1473_147361

theorem range_a (a : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x ≤ 2) → x^2 - 2 * a * x + 1 ≥ 0) → a ≤ 1 :=
by
  sorry

end range_a_l1473_147361


namespace find_x_squared_plus_y_squared_l1473_147347

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 10344 / 169 :=
by
  sorry

end find_x_squared_plus_y_squared_l1473_147347


namespace sticks_per_chair_l1473_147356

-- defining the necessary parameters and conditions
def sticksPerTable := 9
def sticksPerStool := 2
def sticksPerHour := 5
def chairsChopped := 18
def tablesChopped := 6
def stoolsChopped := 4
def hoursKeptWarm := 34

-- calculation of total sticks needed
def totalSticksNeeded := sticksPerHour * hoursKeptWarm

-- the main theorem to prove the number of sticks a chair makes
theorem sticks_per_chair (C : ℕ) : (chairsChopped * C) + (tablesChopped * sticksPerTable) + (stoolsChopped * sticksPerStool) = totalSticksNeeded → C = 6 := by
  sorry

end sticks_per_chair_l1473_147356


namespace possible_distance_between_houses_l1473_147318

variable (d : ℝ)

theorem possible_distance_between_houses (h_d1 : 1 ≤ d) (h_d2 : d ≤ 5) : 1 ≤ d ∧ d ≤ 5 :=
by
  exact ⟨h_d1, h_d2⟩

end possible_distance_between_houses_l1473_147318


namespace trig_quadrant_l1473_147383

theorem trig_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  ∃ k : ℤ, α = (2 * k + 1) * π + α / 2 :=
sorry

end trig_quadrant_l1473_147383


namespace fractional_part_inequality_l1473_147374

noncomputable def frac (z : ℝ) : ℝ := z - ⌊z⌋

theorem fractional_part_inequality (x y : ℝ) : frac (x + y) ≤ frac x + frac y := 
sorry

end fractional_part_inequality_l1473_147374


namespace boys_to_girls_ratio_l1473_147394

theorem boys_to_girls_ratio (x y : ℕ) 
  (h1 : 149 * x + 144 * y = 147 * (x + y)) : 
  x = (3 / 2 : ℚ) * y :=
by
  sorry

end boys_to_girls_ratio_l1473_147394


namespace smallest_x_abs_eq_15_l1473_147354

theorem smallest_x_abs_eq_15 :
  ∃ x : ℝ, (|x - 8| = 15) ∧ ∀ y : ℝ, (|y - 8| = 15) → y ≥ x :=
sorry

end smallest_x_abs_eq_15_l1473_147354


namespace tommy_initial_balloons_l1473_147371

theorem tommy_initial_balloons (initial_balloons balloons_added total_balloons : ℝ)
  (h1 : balloons_added = 34.5)
  (h2 : total_balloons = 60.75)
  (h3 : total_balloons = initial_balloons + balloons_added) :
  initial_balloons = 26.25 :=
by sorry

end tommy_initial_balloons_l1473_147371


namespace geometry_problem_l1473_147330

noncomputable def vertices_on_hyperbola (A B C : ℝ × ℝ) : Prop :=
  (∃ x1 y1, A = (x1, y1) ∧ 2 * x1^2 - y1^2 = 4) ∧
  (∃ x2 y2, B = (x2, y2) ∧ 2 * x2^2 - y2^2 = 4) ∧
  (∃ x3 y3, C = (x3, y3) ∧ 2 * x3^2 - y3^2 = 4)

noncomputable def midpoints (A B C M N P : ℝ × ℝ) : Prop :=
  (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧
  (N = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) ∧
  (P = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))

noncomputable def slopes (A B C M N P : ℝ × ℝ) (k1 k2 k3 : ℝ) : Prop :=
  k1 ≠ 0 ∧ k2 ≠ 0 ∧ k3 ≠ 0 ∧
  k1 = M.2 / M.1 ∧ k2 = N.2 / N.1 ∧ k3 = P.2 / P.1

noncomputable def sum_of_slopes (A B C : ℝ × ℝ) (k1 k2 k3 : ℝ) : Prop :=
  ((A.2 - B.2) / (A.1 - B.1) +
   (B.2 - C.2) / (B.1 - C.1) +
   (C.2 - A.2) / (C.1 - A.1)) = -1

theorem geometry_problem 
  (A B C M N P : ℝ × ℝ) (k1 k2 k3 : ℝ) 
  (h1 : vertices_on_hyperbola A B C)
  (h2 : midpoints A B C M N P) 
  (h3 : slopes A B C M N P k1 k2 k3) 
  (h4 : sum_of_slopes A B C k1 k2 k3) :
  1/k1 + 1/k2 + 1/k3 = -1 / 2 :=
sorry

end geometry_problem_l1473_147330


namespace factorize_expression_l1473_147399

theorem factorize_expression (x y : ℝ) : x^2 * y - 2 * x * y^2 + y^3 = y * (x - y)^2 := 
sorry

end factorize_expression_l1473_147399


namespace olympics_year_zodiac_l1473_147362

-- Define the list of zodiac signs
def zodiac_cycle : List String :=
  ["rat", "ox", "tiger", "rabbit", "dragon", "snake", "horse", "goat", "monkey", "rooster", "dog", "pig"]

-- Function to compute the zodiac sign for a given year
def zodiac_sign (start_year : ℕ) (year : ℕ) : String :=
  let index := (year - start_year) % 12
  zodiac_cycle.getD index "unknown"

-- Proof statement: the zodiac sign of the year 2008 is "rabbit"
theorem olympics_year_zodiac :
  zodiac_sign 1 2008 = "rabbit" :=
by
  -- Proof omitted
  sorry

end olympics_year_zodiac_l1473_147362


namespace calculate_sin_product_l1473_147314

theorem calculate_sin_product (α β : ℝ) (h1 : Real.sin (α + β) = 0.2) (h2 : Real.cos (α - β) = 0.3) :
  Real.sin (α + π/4) * Real.sin (β + π/4) = 0.25 :=
by
  sorry

end calculate_sin_product_l1473_147314


namespace find_digits_l1473_147335

theorem find_digits (x y z : ℕ) (hx : x ≤ 9) (hy : y ≤ 9) (hz : z ≤ 9)
    (h_eq : (10*x+5) * (300 + 10*y + z) = 7850) : x = 2 ∧ y = 1 ∧ z = 4 :=
by {
  sorry
}

end find_digits_l1473_147335


namespace largest_integer_less_than_100_leaving_remainder_4_l1473_147316

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l1473_147316


namespace sufficient_but_not_necessary_l1473_147366

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 / x < 1 / 2) : x > 2 ∨ x < 0 :=
by
  sorry

end sufficient_but_not_necessary_l1473_147366


namespace largest_sum_36_l1473_147309

theorem largest_sum_36 : ∃ n : ℕ, ∃ a : ℕ, (n * a + (n * (n - 1)) / 2 = 36) ∧ ∀ m : ℕ, (m * a + (m * (m - 1)) / 2 = 36) → m ≤ 8 :=
by
  sorry

end largest_sum_36_l1473_147309
