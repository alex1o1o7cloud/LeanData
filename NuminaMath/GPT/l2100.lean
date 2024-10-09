import Mathlib

namespace painting_area_l2100_210066

def wall_height : ℝ := 10
def wall_length : ℝ := 15
def door_height : ℝ := 3
def door_length : ℝ := 5

noncomputable def area_of_wall : ℝ :=
  wall_height * wall_length

noncomputable def area_of_door : ℝ :=
  door_height * door_length

noncomputable def area_to_paint : ℝ :=
  area_of_wall - area_of_door

theorem painting_area :
  area_to_paint = 135 := by
  sorry

end painting_area_l2100_210066


namespace possible_values_of_a_l2100_210093

def line1 (x y : ℝ) := x + y + 1 = 0
def line2 (x y : ℝ) := 2 * x - y + 8 = 0
def line3 (a : ℝ) (x y : ℝ) := a * x + 3 * y - 5 = 0

theorem possible_values_of_a :
  {a : ℝ | ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 a x y} ⊆ {1/3, 3, -6} ∧
  {1/3, 3, -6} ⊆ {a : ℝ | ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 a x y} :=
sorry

end possible_values_of_a_l2100_210093


namespace dentist_age_l2100_210088

theorem dentist_age (x : ℝ) (h : (x - 8) / 6 = (x + 8) / 10) : x = 32 :=
  by
  sorry

end dentist_age_l2100_210088


namespace min_additional_cells_l2100_210072

-- Definitions based on conditions
def num_cells_shape : Nat := 32
def side_length_square : Nat := 9
def area_square : Nat := side_length_square * side_length_square

-- The statement to prove
theorem min_additional_cells (num_cells_given : Nat := num_cells_shape) 
(side_length : Nat := side_length_square)
(area : Nat := area_square) :
  area - num_cells_given = 49 :=
by
  sorry

end min_additional_cells_l2100_210072


namespace length_of_second_train_l2100_210024

/-- 
The length of the second train can be determined given the length and speed of the first train,
the speed of the second train, and the time they take to cross each other.
-/
theorem length_of_second_train (speed1_kmph : ℝ) (length1_m : ℝ) (speed2_kmph : ℝ) (time_s : ℝ) :
  (speed1_kmph = 120) →
  (length1_m = 230) →
  (speed2_kmph = 80) →
  (time_s = 9) →
  let relative_speed_m_per_s := (speed1_kmph * 1000 / 3600) + (speed2_kmph * 1000 / 3600)
  let total_distance := relative_speed_m_per_s * time_s
  let length2_m := total_distance - length1_m
  length2_m = 269.95 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  let relative_speed_m_per_s := (120 * 1000 / 3600) + (80 * 1000 / 3600)
  let total_distance := relative_speed_m_per_s * 9
  let length2_m := total_distance - 230
  exact sorry

end length_of_second_train_l2100_210024


namespace air_conditioner_consumption_l2100_210040

theorem air_conditioner_consumption :
  ∀ (total_consumption_8_hours : ℝ)
    (hours_8 : ℝ)
    (hours_per_day : ℝ)
    (days : ℝ),
    total_consumption_8_hours / hours_8 * hours_per_day * days = 27 :=
by
  intros total_consumption_8_hours hours_8 hours_per_day days
  sorry

end air_conditioner_consumption_l2100_210040


namespace intercepts_correct_l2100_210018

-- Define the equation of the line
def line_eq (x y : ℝ) := 5 * x - 2 * y - 10 = 0

-- Define the intercepts
def x_intercept : ℝ := 2
def y_intercept : ℝ := -5

-- Prove that the intercepts are as stated
theorem intercepts_correct :
  (∃ x, line_eq x 0 ∧ x = x_intercept) ∧
  (∃ y, line_eq 0 y ∧ y = y_intercept) :=
by
  sorry

end intercepts_correct_l2100_210018


namespace matrix_addition_l2100_210041

variable (A B : Matrix (Fin 2) (Fin 2) ℤ) -- Define matrices with integer entries

-- Define the specific matrices used in the problem
def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![ ![2, 3], ![-1, 4] ]

def matrix_B : Matrix (Fin 2) (Fin 2) ℤ := 
  ![ ![-1, 8], ![-3, 0] ]

-- Define the result matrix
def result_matrix : Matrix (Fin 2) (Fin 2) ℤ := 
  ![ ![3, 14], ![-5, 8] ]

-- The theorem to prove
theorem matrix_addition : 2 • matrix_A + matrix_B = result_matrix := by
  sorry -- Proof omitted

end matrix_addition_l2100_210041


namespace expressions_equal_iff_sum_zero_l2100_210034

theorem expressions_equal_iff_sum_zero (p q r : ℝ) : (p + qr = (p + q) * (p + r)) ↔ (p + q + r = 0) :=
sorry

end expressions_equal_iff_sum_zero_l2100_210034


namespace geom_series_sum_n_eq_728_div_729_l2100_210058

noncomputable def a : ℚ := 1 / 3
noncomputable def r : ℚ := 1 / 3
noncomputable def S_n (n : ℕ) : ℚ := a * ((1 - r^n) / (1 - r))

theorem geom_series_sum_n_eq_728_div_729 (n : ℕ) (h : S_n n = 728 / 729) : n = 6 :=
by
  sorry

end geom_series_sum_n_eq_728_div_729_l2100_210058


namespace new_trailers_added_l2100_210053

theorem new_trailers_added (n : ℕ) :
  let original_trailers := 15
  let original_age := 12
  let years_passed := 3
  let current_total_trailers := original_trailers + n
  let current_average_age := 10
  let total_age_three_years_ago := original_trailers * original_age
  let new_trailers_age := 3
  let total_current_age := (original_trailers * (original_age + years_passed)) + (n * new_trailers_age)
  (total_current_age / current_total_trailers = current_average_age) ↔ (n = 10) :=
by
  sorry

end new_trailers_added_l2100_210053


namespace total_cost_for_trip_l2100_210089

def cost_of_trip (students : ℕ) (teachers : ℕ) (seats_per_bus : ℕ) (cost_per_bus : ℕ) (toll_per_bus : ℕ) : ℕ :=
  let total_people := students + teachers
  let buses_required := (total_people + seats_per_bus - 1) / seats_per_bus -- ceiling division
  let total_rent_cost := buses_required * cost_per_bus
  let total_toll_cost := buses_required * toll_per_bus
  total_rent_cost + total_toll_cost

theorem total_cost_for_trip
  (students : ℕ := 252)
  (teachers : ℕ := 8)
  (seats_per_bus : ℕ := 41)
  (cost_per_bus : ℕ := 300000)
  (toll_per_bus : ℕ := 7500) :
  cost_of_trip students teachers seats_per_bus cost_per_bus toll_per_bus = 2152500 := by
  sorry -- Proof to be filled

end total_cost_for_trip_l2100_210089


namespace pondFishEstimate_l2100_210045

noncomputable def estimateTotalFish (initialFishMarked : ℕ) (caughtFishTenDaysLater : ℕ) (markedFishCaught : ℕ) : ℕ :=
  initialFishMarked * caughtFishTenDaysLater / markedFishCaught

theorem pondFishEstimate
    (initialFishMarked : ℕ)
    (caughtFishTenDaysLater : ℕ)
    (markedFishCaught : ℕ)
    (h1 : initialFishMarked = 30)
    (h2 : caughtFishTenDaysLater = 50)
    (h3 : markedFishCaught = 2) :
    estimateTotalFish initialFishMarked caughtFishTenDaysLater markedFishCaught = 750 := by
  sorry

end pondFishEstimate_l2100_210045


namespace problem_statement_l2100_210010

theorem problem_statement (f : ℕ → ℤ) (a b : ℤ) 
  (h1 : f 1 = 7) 
  (h2 : f 2 = 11)
  (h3 : ∀ x, f x = a * x^2 + b * x + 3) :
  f 3 = 15 := 
sorry

end problem_statement_l2100_210010


namespace calculation_correct_l2100_210047

theorem calculation_correct :
  15 * ( (1/3 : ℚ) + (1/4) + (1/6) )⁻¹ = 20 := sorry

end calculation_correct_l2100_210047


namespace total_animals_l2100_210068

theorem total_animals (B : ℕ) (h1 : 4 * B + 8 = 44) : B + 4 = 13 := by
  sorry

end total_animals_l2100_210068


namespace remainder_division_l2100_210084

theorem remainder_division
  (j : ℕ) (h_pos : 0 < j)
  (h_rem : ∃ b : ℕ, 72 = b * j^2 + 8) :
  150 % j = 6 :=
sorry

end remainder_division_l2100_210084


namespace probability_ace_king_queen_l2100_210092

-- Definitions based on the conditions
def total_cards := 52
def aces := 4
def kings := 4
def queens := 4

def probability_first_ace := aces / total_cards
def probability_second_king := kings / (total_cards - 1)
def probability_third_queen := queens / (total_cards - 2)

theorem probability_ace_king_queen :
  (probability_first_ace * probability_second_king * probability_third_queen) = (8 / 16575) :=
by sorry

end probability_ace_king_queen_l2100_210092


namespace alpha_half_quadrant_l2100_210042

open Real

theorem alpha_half_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π - π / 2 < α ∧ α < 2 * k * π) :
  (∃ k1 : ℤ, (2 * k1 + 1) * π - π / 4 < α / 2 ∧ α / 2 < (2 * k1 + 1) * π) ∨
  (∃ k2 : ℤ, 2 * k2 * π - π / 4 < α / 2 ∧ α / 2 < 2 * k2 * π) :=
sorry

end alpha_half_quadrant_l2100_210042


namespace simplify_and_evaluate_expr_l2100_210015

noncomputable def original_expr (x : ℝ) : ℝ := 
  ((x / (x - 1)) - (x / (x^2 - 1))) / ((x^2 - x) / (x^2 - 2*x + 1))

noncomputable def x_val : ℝ := Real.sqrt 2 - 1

theorem simplify_and_evaluate_expr : original_expr x_val = 1 - (Real.sqrt 2) / 2 :=
  by
    sorry

end simplify_and_evaluate_expr_l2100_210015


namespace problem_l2100_210062

def setA : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}
def setB : Set ℝ := {x : ℝ | x ≤ 3}

theorem problem : setA ∩ setB = setA := sorry

end problem_l2100_210062


namespace find_r_l2100_210090

theorem find_r (r : ℝ) (h : 5 * (r - 9) = 6 * (3 - 3 * r) + 6) : r = 3 :=
by
  sorry

end find_r_l2100_210090


namespace relationship_y1_y2_l2100_210001

theorem relationship_y1_y2
    (b : ℝ) 
    (y1 y2 : ℝ)
    (h1 : y1 = - (1 / 2) * (-2) + b) 
    (h2 : y2 = - (1 / 2) * 3 + b) : 
    y1 > y2 :=
sorry

end relationship_y1_y2_l2100_210001


namespace rectangle_area_perimeter_l2100_210051

theorem rectangle_area_perimeter (a b : ℝ) (h₁ : a * b = 6) (h₂ : a + b = 6) : a^2 + b^2 = 24 := 
by
  sorry

end rectangle_area_perimeter_l2100_210051


namespace largest_three_digit_number_with_7_in_hundreds_l2100_210014

def is_three_digit_number_with_7_in_hundreds (n : ℕ) : Prop := 
  100 ≤ n ∧ n < 1000 ∧ (n / 100) = 7

theorem largest_three_digit_number_with_7_in_hundreds : 
  ∀ (n : ℕ), is_three_digit_number_with_7_in_hundreds n → n ≤ 799 :=
by sorry

end largest_three_digit_number_with_7_in_hundreds_l2100_210014


namespace q_evaluation_at_3_point_5_l2100_210056

def q (x : ℝ) : ℝ :=
  |x - 3|^(1/3) + 2*|x - 3|^(1/5) + |x - 3|^(1/7)

theorem q_evaluation_at_3_point_5 : q 3.5 = 3 :=
by
  sorry

end q_evaluation_at_3_point_5_l2100_210056


namespace average_grade_of_females_is_92_l2100_210060

theorem average_grade_of_females_is_92 (F : ℝ) : 
  (∀ (overall_avg male_avg : ℝ) (num_male num_female : ℕ), 
    overall_avg = 90 ∧ male_avg = 82 ∧ num_male = 8 ∧ num_female = 32 → 
    overall_avg = (num_male * male_avg + num_female * F) / (num_male + num_female) → F = 92) :=
sorry

end average_grade_of_females_is_92_l2100_210060


namespace add_and_simplify_fractions_l2100_210002

theorem add_and_simplify_fractions :
  (1 / 462) + (23 / 42) = 127 / 231 :=
by
  sorry

end add_and_simplify_fractions_l2100_210002


namespace calculate_expression_l2100_210048

theorem calculate_expression :
  12 * 11 + 7 * 8 - 5 * 6 + 10 * 4 = 198 :=
by
  sorry

end calculate_expression_l2100_210048


namespace swiss_probability_is_30_percent_l2100_210098

def total_cheese_sticks : Nat := 22 + 34 + 29 + 45 + 20

def swiss_cheese_sticks : Nat := 45

def probability_swiss : Nat :=
  (swiss_cheese_sticks * 100) / total_cheese_sticks

theorem swiss_probability_is_30_percent :
  probability_swiss = 30 := by
  sorry

end swiss_probability_is_30_percent_l2100_210098


namespace set_inclusion_l2100_210037

def setM : Set ℝ := {θ | ∃ k : ℤ, θ = k * Real.pi / 4}

def setN : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4)}

def setP : Set ℝ := {a | ∃ k : ℤ, a = (k * Real.pi / 2) + (Real.pi / 4)}

theorem set_inclusion : setP ⊆ setN ∧ setN ⊆ setM := by
  sorry

end set_inclusion_l2100_210037


namespace smallest_unreachable_integer_l2100_210009

/-- The smallest positive integer that cannot be expressed in the form (2^a - 2^b) / (2^c - 2^d) where a, b, c, d are non-negative integers is 11. -/
theorem smallest_unreachable_integer : 
  ∀ (a b c d : ℕ), 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
  ∃ (n : ℕ), n = 11 ∧ ¬ ∃ (a b c d : ℕ), (2^a - 2^b) / (2^c - 2^d) = n :=
by
  sorry

end smallest_unreachable_integer_l2100_210009


namespace perp_vectors_dot_product_eq_zero_l2100_210099

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perp_vectors_dot_product_eq_zero (x : ℝ) (h : dot_product vector_a (vector_b x) = 0) : x = -8 :=
  by sorry

end perp_vectors_dot_product_eq_zero_l2100_210099


namespace jill_bought_5_packs_of_red_bouncy_balls_l2100_210016

theorem jill_bought_5_packs_of_red_bouncy_balls
  (r : ℕ) -- number of packs of red bouncy balls
  (yellow_packs : ℕ := 4)
  (bouncy_balls_per_pack : ℕ := 18)
  (extra_red_bouncy_balls : ℕ := 18)
  (total_yellow_bouncy_balls : ℕ := yellow_packs * bouncy_balls_per_pack)
  (total_red_bouncy_balls : ℕ := total_yellow_bouncy_balls + extra_red_bouncy_balls)
  (h : r * bouncy_balls_per_pack = total_red_bouncy_balls) :
  r = 5 :=
by sorry

end jill_bought_5_packs_of_red_bouncy_balls_l2100_210016


namespace line_through_center_and_perpendicular_l2100_210064

def center_of_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 1

def perpendicular_to_line (slope : ℝ) : Prop :=
  slope = 1

theorem line_through_center_and_perpendicular (x y : ℝ) :
  center_of_circle x y →
  perpendicular_to_line 1 →
  (x - y + 1 = 0) :=
by
  intros h_center h_perpendicular
  sorry

end line_through_center_and_perpendicular_l2100_210064


namespace notebook_cost_l2100_210076

open Nat

theorem notebook_cost
  (s : ℕ) (c : ℕ) (n : ℕ)
  (h_majority : s > 21)
  (h_notebooks : n > 2)
  (h_cost : c > n)
  (h_total : s * c * n = 2773) : c = 103 := 
sorry

end notebook_cost_l2100_210076


namespace binary_to_decimal_l2100_210019

theorem binary_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 13) :=
by
  sorry

end binary_to_decimal_l2100_210019


namespace opposite_of_neg_nine_l2100_210052

theorem opposite_of_neg_nine : -(-9) = 9 :=
by
  sorry

end opposite_of_neg_nine_l2100_210052


namespace match_end_time_is_17_55_l2100_210073

-- Definitions corresponding to conditions
def start_time : ℕ := 15 * 60 + 30  -- Convert 15:30 to minutes past midnight
def duration : ℕ := 145  -- Duration in minutes

-- Definition corresponding to the question
def end_time : ℕ := start_time + duration 

-- Assertion corresponding to the correct answer
theorem match_end_time_is_17_55 : end_time = 17 * 60 + 55 :=
by
  -- Proof steps and actual proof will go here
  sorry

end match_end_time_is_17_55_l2100_210073


namespace Tim_took_out_11_rulers_l2100_210011

-- Define the initial number of rulers
def initial_rulers := 14

-- Define the number of rulers left in the drawer
def rulers_left := 3

-- Define the number of rulers taken by Tim
def rulers_taken := initial_rulers - rulers_left

-- Statement to prove that the number of rulers taken by Tim is indeed 11
theorem Tim_took_out_11_rulers : rulers_taken = 11 := by
  sorry

end Tim_took_out_11_rulers_l2100_210011


namespace compare_y_coordinates_l2100_210049

theorem compare_y_coordinates (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁: (x₁ = -3) ∧ (y₁ = 2 * x₁ - 1)) 
  (h₂: (x₂ = -5) ∧ (y₂ = 2 * x₂ - 1)) : 
  y₁ > y₂ := 
by 
  sorry

end compare_y_coordinates_l2100_210049


namespace geometric_series_sum_l2100_210022

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h_a : a = 1) (h_r : r = 1 / 2) (h_n : n = 5) :
  ((a * (1 - r^n)) / (1 - r)) = 31 / 16 := 
by
  sorry

end geometric_series_sum_l2100_210022


namespace no_apples_info_l2100_210008

theorem no_apples_info (r d : ℕ) (condition1 : r = 79) (condition2 : d = 53) (condition3 : r = d + 26) : 
  ∀ a : ℕ, (a = a) → false :=
by
  intro a h
  sorry

end no_apples_info_l2100_210008


namespace geometric_sequence_a7_l2100_210044

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a1 : a 1 = 2) (h_a3 : a 3 = 4) : a 7 = 16 := 
sorry

end geometric_sequence_a7_l2100_210044


namespace find_u_plus_v_l2100_210003

theorem find_u_plus_v (u v : ℚ) 
  (h₁ : 3 * u + 7 * v = 17) 
  (h₂ : 5 * u - 3 * v = 9) : 
  u + v = 43 / 11 :=
sorry

end find_u_plus_v_l2100_210003


namespace residue_neg_1234_mod_31_l2100_210012

theorem residue_neg_1234_mod_31 : -1234 % 31 = 6 := 
by sorry

end residue_neg_1234_mod_31_l2100_210012


namespace a_100_value_l2100_210083

variables (S : ℕ → ℚ) (a : ℕ → ℚ)

def S_n (n : ℕ) : ℚ := S n
def a_n (n : ℕ) : ℚ := a n

axiom a1_eq_3 : a 1 = 3
axiom a_n_formula (n : ℕ) (hn : n ≥ 2) : a n = (3 * S n ^ 2) / (3 * S n - 2)

theorem a_100_value : a 100 = -3 / 88401 :=
sorry

end a_100_value_l2100_210083


namespace solve_fractional_equation_l2100_210085

theorem solve_fractional_equation : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ -3 → (1 / x = 2 / (x + 3) ↔ x = 3) :=
by
  intros x hx
  sorry

end solve_fractional_equation_l2100_210085


namespace shaded_quadrilateral_area_l2100_210038

noncomputable def area_of_shaded_quadrilateral : ℝ :=
  let side_lens : List ℝ := [3, 5, 7, 9]
  let total_base: ℝ := side_lens.sum
  let largest_square_height: ℝ := 9
  let height_base_ratio := largest_square_height / total_base
  let heights := side_lens.scanl (· + ·) 0 |>.tail.map (λ x => x * height_base_ratio)
  let a := heights.get! 0
  let b := heights.get! heights.length - 1
  (largest_square_height * (a + b)) / 2

theorem shaded_quadrilateral_area :
    let side_lens := [3, 5, 7, 9]
    let total_base := side_lens.sum
    let largest_square_height := 9
    let height_base_ratio := largest_square_height / total_base
    let heights := side_lens.scanl (· + ·) 0 |>.tail.map (λ x => x * height_base_ratio)
    let a := heights.get! 0
    let b := heights.get! heights.length - 1
    (largest_square_height * (a + b)) / 2 = 30.375 :=
by 
  sorry

end shaded_quadrilateral_area_l2100_210038


namespace area_of_ABCD_l2100_210061

theorem area_of_ABCD 
  (AB CD DA: ℝ) (angle_CDA: ℝ) (a b c: ℕ) 
  (H1: AB = 10) 
  (H2: BC = 6) 
  (H3: CD = 13) 
  (H4: DA = 13) 
  (H5: angle_CDA = 45) 
  (H_area: a = 8 ∧ b = 30 ∧ c = 2) :

  ∃ (a b c : ℝ), a + b + c = 40 := 
by
  sorry

end area_of_ABCD_l2100_210061


namespace total_area_of_WIN_sectors_l2100_210054

theorem total_area_of_WIN_sectors (r : ℝ) (A_total : ℝ) (Prob_WIN : ℝ) (A_WIN : ℝ) : 
  r = 15 → 
  A_total = π * r^2 → 
  Prob_WIN = 3/7 → 
  A_WIN = Prob_WIN * A_total → 
  A_WIN = 3/7 * 225 * π :=
by {
  intros;
  sorry
}

end total_area_of_WIN_sectors_l2100_210054


namespace day_100_days_from_friday_l2100_210035

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Define a function to get the day of the week after a given number of days
def dayOfWeekAfter (start : Day) (n : ℕ) : Day :=
  match start with
  | Sunday    => match n % 7 with
                  | 0 => Sunday
                  | 1 => Monday
                  | 2 => Tuesday
                  | 3 => Wednesday
                  | 4 => Thursday
                  | 5 => Friday
                  | 6 => Saturday
                  | _ => start
  | Monday    => match n % 7 with
                  | 0 => Monday
                  | 1 => Tuesday
                  | 2 => Wednesday
                  | 3 => Thursday
                  | 4 => Friday
                  | 5 => Saturday
                  | 6 => Sunday
                  | _ => start
  | Tuesday   => match n % 7 with
                  | 0 => Tuesday
                  | 1 => Wednesday
                  | 2 => Thursday
                  | 3 => Friday
                  | 4 => Saturday
                  | 5 => Sunday
                  | 6 => Monday
                  | _ => start
  | Wednesday => match n % 7 with
                  | 0 => Wednesday
                  | 1 => Thursday
                  | 2 => Friday
                  | 3 => Saturday
                  | 4 => Sunday
                  | 5 => Monday
                  | 6 => Tuesday
                  | _ => start
  | Thursday  => match n % 7 with
                  | 0 => Thursday
                  | 1 => Friday
                  | 2 => Saturday
                  | 3 => Sunday
                  | 4 => Monday
                  | 5 => Tuesday
                  | 6 => Wednesday
                  | _ => start
  | Friday    => match n % 7 with
                  | 0 => Friday
                  | 1 => Saturday
                  | 2 => Sunday
                  | 3 => Monday
                  | 4 => Tuesday
                  | 5 => Wednesday
                  | 6 => Thursday
                  | _ => start
  | Saturday  => match n % 7 with
                  | 0 => Saturday
                  | 1 => Sunday
                  | 2 => Monday
                  | 3 => Tuesday
                  | 4 => Wednesday
                  | 5 => Thursday
                  | 6 => Friday
                  | _ => start

-- The proof problem as a Lean theorem
theorem day_100_days_from_friday : dayOfWeekAfter Friday 100 = Sunday := by
  -- Proof will go here
  sorry

end day_100_days_from_friday_l2100_210035


namespace y_coordinate_of_C_range_l2100_210059

noncomputable def A : ℝ × ℝ := (0, 2)

def is_on_parabola (P : ℝ × ℝ) : Prop := (P.2)^2 = P.1 + 4

def is_perpendicular (A B C : ℝ × ℝ) : Prop := 
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  k_AB * k_BC = -1

def range_of_y_C (y_C : ℝ) : Prop := y_C ≤ 0 ∨ y_C ≥ 4

theorem y_coordinate_of_C_range (B C : ℝ × ℝ)
  (hB : is_on_parabola B) (hC : is_on_parabola C) (h_perpendicular : is_perpendicular A B C) : 
  range_of_y_C (C.2) :=
sorry

end y_coordinate_of_C_range_l2100_210059


namespace tangent_line_extreme_values_l2100_210000

-- Define the function f and its conditions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- Given conditions
def cond1 (a b : ℝ) : Prop := 3 * a * 2^2 + b = 0
def cond2 (a b : ℝ) : Prop := a * 2^3 + b * 2 + 2 = -14

-- Part 1: Tangent line equation at (1, f(1))
theorem tangent_line (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) : 
  ∃ m c : ℝ, m = -9 ∧ c = 9 ∧ (∀ y : ℝ, y = f a b 1 → 9 * 1 + y = 0) :=
sorry

-- Part 2: Extreme values on [-3, 3]
-- Define critical points and endpoints
def f_value_at (a b x : ℝ) := f a b x

theorem extreme_values (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) :
  ∃ min max : ℝ, min = -14 ∧ max = 18 ∧ 
  f_value_at a b 2 = min ∧ f_value_at a b (-2) = max :=
sorry

end tangent_line_extreme_values_l2100_210000


namespace bacteria_mass_at_4pm_l2100_210063

theorem bacteria_mass_at_4pm 
  (r s t u v w : ℝ)
  (x y z : ℝ)
  (h1 : x = 10.0 * (1 + r))
  (h2 : y = 15.0 * (1 + s))
  (h3 : z = 8.0 * (1 + t))
  (h4 : 28.9 = x * (1 + u))
  (h5 : 35.5 = y * (1 + v))
  (h6 : 20.1 = z * (1 + w)) :
  x = 28.9 / (1 + u) ∧ y = 35.5 / (1 + v) ∧ z = 20.1 / (1 + w) :=
by
  sorry

end bacteria_mass_at_4pm_l2100_210063


namespace tangent_line_parabola_k_l2100_210029

theorem tangent_line_parabola_k :
  ∃ (k : ℝ), (∀ (x y : ℝ), 4 * x + 7 * y + k = 0 → y^2 = 16 * x → (28 ^ 2 = 4 * 1 * 4 * k)) → k = 49 :=
by
  sorry

end tangent_line_parabola_k_l2100_210029


namespace third_grade_parts_in_batch_l2100_210039

-- Define conditions
variable (x y s : ℕ) (h_first_grade : 24 = 24) (h_second_grade : 36 = 36)
variable (h_sample_size : 20 = 20) (h_sample_third_grade : 10 = 10)

-- The problem: Prove the total number of third-grade parts in the batch is 60 and the number of second-grade parts sampled is 6
open Nat

theorem third_grade_parts_in_batch
  (h_total_parts : x - y = 60)
  (h_third_grade_proportion : y = (1 / 2) * x)
  (h_second_grade_proportion : s = (36 / 120) * 20) :
  y = 60 ∧ s = 6 := by
  sorry

end third_grade_parts_in_batch_l2100_210039


namespace unique_solution_l2100_210079

theorem unique_solution :
  ∀ a b c : ℕ,
    a > 0 → b > 0 → c > 0 →
    (3 * a * b * c + 11 * (a + b + c) = 6 * (a * b + b * c + c * a) + 18) →
    (a = 1 ∧ b = 2 ∧ c = 3) :=
by
  intros a b c ha hb hc h
  have h1 : a = 1 := sorry
  have h2 : b = 2 := sorry
  have h3 : c = 3 := sorry
  exact ⟨h1, h2, h3⟩

end unique_solution_l2100_210079


namespace quadratic_equation_unique_solution_l2100_210031

theorem quadratic_equation_unique_solution
  (a c : ℝ)
  (h_discriminant : 100 - 4 * a * c = 0)
  (h_sum : a + c = 12)
  (h_lt : a < c) :
  (a, c) = (6 - Real.sqrt 11, 6 + Real.sqrt 11) :=
sorry

end quadratic_equation_unique_solution_l2100_210031


namespace calc_fraction_power_l2100_210078

theorem calc_fraction_power (n m : ℤ) (h_n : n = 2023) (h_m : m = 2022) :
  (- (2 / 3 : ℚ))^n * ((3 / 2 : ℚ))^m = - (2 / 3) := by
  sorry

end calc_fraction_power_l2100_210078


namespace problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l2100_210017

-- Problem G6.1
theorem problem_G6_1 : (21 ^ 3 - 11 ^ 3) / (21 ^ 2 + 21 * 11 + 11 ^ 2) = 10 := 
  sorry

-- Problem G6.2
theorem problem_G6_2 (p q : ℕ) (h1 : (p : ℚ) * 6 = 4 * (q : ℚ)) : q = 3 * p / 2 := 
  sorry

-- Problem G6.3
theorem problem_G6_3 (q r : ℕ) (h1 : q % 7 = 3) (h2 : r % 7 = 5) (h3 : 18 < r) (h4 : r < 26) : r = 24 := 
  sorry

-- Problem G6.4
def star (a b : ℕ) : ℕ := a * b + 1

theorem problem_G6_4 : star (star 3 4) 2 = 27 := 
  sorry

end problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l2100_210017


namespace find_angle_A_determine_triangle_shape_l2100_210075

noncomputable def angle_A (A : ℝ) (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 7 / 2 ∧ m = (Real.cos (A / 2)^2, Real.cos (2 * A)) ∧ 
  n = (4, -1)

theorem find_angle_A : 
  ∃ A : ℝ,  (0 < A ∧ A < Real.pi) ∧ angle_A A (Real.cos (A / 2)^2, Real.cos (2 * A)) (4, -1) 
  := sorry

noncomputable def triangle_shape (a b c : ℝ) (A : ℝ) : Prop :=
  a = Real.sqrt 3 ∧ a^2 = b^2 + c^2 - b * c * Real.cos (A)

theorem determine_triangle_shape :
  ∀ (b c : ℝ), (b * c ≤ 3) → triangle_shape (Real.sqrt 3) b c (Real.pi / 3) →
  (b = Real.sqrt 3 ∧ c = Real.sqrt 3)
  := sorry


end find_angle_A_determine_triangle_shape_l2100_210075


namespace largest_of_four_consecutive_integers_with_product_840_l2100_210027

theorem largest_of_four_consecutive_integers_with_product_840 
  (a b c d : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c + 1 = d) (h_pos : 0 < a) (h_prod : a * b * c * d = 840) : d = 7 :=
sorry

end largest_of_four_consecutive_integers_with_product_840_l2100_210027


namespace ratio_w_y_l2100_210091

theorem ratio_w_y 
  (w x y z : ℚ) 
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) : 
  w / y = 16 / 3 :=
sorry

end ratio_w_y_l2100_210091


namespace domain_transformation_l2100_210043

-- Definitions of conditions
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

def domain_g (x : ℝ) : Prop := 1 < x ∧ x ≤ 3

-- Theorem stating the proof problem
theorem domain_transformation : 
  (∀ x, domain_f x → 0 ≤ x+1 ∧ x+1 ≤ 4) →
  (∀ x, (0 ≤ x+1 ∧ x+1 ≤ 4) → (x-1 > 0) → domain_g x) :=
by
  intros h1 x hx
  sorry

end domain_transformation_l2100_210043


namespace postcards_initial_count_l2100_210004

theorem postcards_initial_count (P : ℕ) 
  (h1 : ∀ n, n = P / 2)
  (h2 : ∀ n, n = (P / 2) * 15 / 5) 
  (h3 : P / 2 + 3 * P / 2 = 36) : 
  P = 18 := 
sorry

end postcards_initial_count_l2100_210004


namespace bug_paths_from_A_to_B_l2100_210033

-- Define the positions A and B and intermediate red and blue points in the lattice
inductive Position
| A
| B
| red1
| red2
| blue1
| blue2

open Position

-- Define the possible directed paths in the lattice
def paths : List (Position × Position) :=
[(A, red1), (A, red2), 
 (red1, blue1), (red1, blue2), 
 (red2, blue1), (red2, blue2), 
 (blue1, B), (blue1, B), (blue1, B), 
 (blue2, B), (blue2, B), (blue2, B)]

-- Define a function that calculates the number of unique paths from A to B without repeating any path
def count_paths : ℕ := sorry

-- The mathematical problem statement
theorem bug_paths_from_A_to_B : count_paths = 24 := sorry

end bug_paths_from_A_to_B_l2100_210033


namespace alice_bush_count_l2100_210032

theorem alice_bush_count :
  let side_length := 24
  let num_sides := 3
  let bush_space := 3
  (num_sides * side_length) / bush_space = 24 :=
by
  sorry

end alice_bush_count_l2100_210032


namespace necessary_condition_x_pow_2_minus_x_lt_0_l2100_210067

theorem necessary_condition_x_pow_2_minus_x_lt_0 (x : ℝ) : (x^2 - x < 0) → (-1 < x ∧ x < 1) := by
  intro hx
  sorry

end necessary_condition_x_pow_2_minus_x_lt_0_l2100_210067


namespace systematic_sampling_example_l2100_210023

theorem systematic_sampling_example :
  ∃ (selected : Finset ℕ), 
    selected = {10, 30, 50, 70, 90} ∧
    ∀ n ∈ selected, 1 ≤ n ∧ n ≤ 100 ∧ 
    (∃ k, k > 0 ∧ k * 20 - 10∈ selected ∧ k * 20 - 10 ∈ Finset.range 101) := 
by
  sorry

end systematic_sampling_example_l2100_210023


namespace solve_for_x_l2100_210094

theorem solve_for_x (x : ℤ) (h : 158 - x = 59) : x = 99 :=
by
  sorry

end solve_for_x_l2100_210094


namespace max_value_of_g_l2100_210081

def g : ℕ → ℕ
| n => if n < 7 then n + 7 else g (n - 3)

theorem max_value_of_g : ∀ (n : ℕ), g n ≤ 13 ∧ (∃ n0, g n0 = 13) := by
  sorry

end max_value_of_g_l2100_210081


namespace range_a_l2100_210007

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∃ x : ℝ, |x - a| + |x - 1| ≤ 3

theorem range_a (a : ℝ) : range_of_a a → -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_a_l2100_210007


namespace birds_remaining_on_fence_l2100_210005

noncomputable def initial_birds : ℝ := 15.3
noncomputable def birds_flew_away : ℝ := 6.5
noncomputable def remaining_birds : ℝ := initial_birds - birds_flew_away

theorem birds_remaining_on_fence : remaining_birds = 8.8 :=
by
  -- sorry is a placeholder for the proof, which is not required
  sorry

end birds_remaining_on_fence_l2100_210005


namespace buyers_muffin_mix_l2100_210025

variable (P C M CM: ℕ)

theorem buyers_muffin_mix
    (h_total: P = 100)
    (h_cake: C = 50)
    (h_both: CM = 17)
    (h_neither: P - (C + M - CM) = 27)
    : M = 73 :=
by sorry

end buyers_muffin_mix_l2100_210025


namespace Sue_made_22_buttons_l2100_210070

def Mari_buttons : Nat := 8
def Kendra_buttons : Nat := 5 * Mari_buttons + 4
def Sue_buttons : Nat := Kendra_buttons / 2

theorem Sue_made_22_buttons : Sue_buttons = 22 :=
by
  -- proof to be added
  sorry

end Sue_made_22_buttons_l2100_210070


namespace bonus_distributed_correctly_l2100_210071

def amount_received (A B C D E F : ℝ) :=
  -- Conditions
  (A = 2 * B) ∧ 
  (B = C) ∧ 
  (D = 2 * B - 1500) ∧ 
  (E = C + 2000) ∧ 
  (F = 1/2 * (A + D)) ∧ 
  -- Total bonus amount
  (A + B + C + D + E + F = 25000)

theorem bonus_distributed_correctly :
  ∃ (A B C D E F : ℝ), 
    amount_received A B C D E F ∧ 
    A = 4950 ∧ 
    B = 2475 ∧ 
    C = 2475 ∧ 
    D = 3450 ∧ 
    E = 4475 ∧ 
    F = 4200 :=
sorry

end bonus_distributed_correctly_l2100_210071


namespace tan_theta_half_l2100_210020

theorem tan_theta_half (θ : ℝ) (a b : ℝ × ℝ) 
  (h₀ : a = (Real.sin θ, 1)) 
  (h₁ : b = (-2, Real.cos θ)) 
  (h₂ : a.1 * b.1 + a.2 * b.2 = 0) : Real.tan θ = 1 / 2 :=
sorry

end tan_theta_half_l2100_210020


namespace intersection_correct_l2100_210065

open Set

def M : Set ℤ := {-1, 3, 5}
def N : Set ℤ := {-1, 0, 1, 2, 3}
def MN_intersection : Set ℤ := {-1, 3}

theorem intersection_correct : M ∩ N = MN_intersection := by
  sorry

end intersection_correct_l2100_210065


namespace candy_count_l2100_210082

theorem candy_count (S : ℕ) (H1 : 32 + S - 35 = 39) : S = 42 :=
by
  sorry

end candy_count_l2100_210082


namespace min_value_sum_inverse_squares_l2100_210021

theorem min_value_sum_inverse_squares (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_sum : a + b + c = 3) :
    (1 / (a + b)^2) + (1 / (a + c)^2) + (1 / (b + c)^2) >= 3 / 2 :=
sorry

end min_value_sum_inverse_squares_l2100_210021


namespace simon_change_l2100_210036

def pansy_price : ℝ := 2.50
def pansy_count : ℕ := 5
def hydrangea_price : ℝ := 12.50
def hydrangea_count : ℕ := 1
def petunia_price : ℝ := 1.00
def petunia_count : ℕ := 5
def discount_rate : ℝ := 0.10
def initial_payment : ℝ := 50.00

theorem simon_change : 
  let total_cost := (pansy_count * pansy_price) + (hydrangea_count * hydrangea_price) + (petunia_count * petunia_price)
  let discount := total_cost * discount_rate
  let cost_after_discount := total_cost - discount
  let change := initial_payment - cost_after_discount
  change = 23.00 :=
by
  sorry

end simon_change_l2100_210036


namespace population_growth_l2100_210080

theorem population_growth 
  (P₀ : ℝ) (P₂ : ℝ) (r : ℝ)
  (hP₀ : P₀ = 15540) 
  (hP₂ : P₂ = 25460.736)
  (h_growth : P₂ = P₀ * (1 + r)^2) :
  r = 0.28 :=
by 
  sorry

end population_growth_l2100_210080


namespace greene_family_amusement_park_spending_l2100_210097

def spent_on_admission : ℝ := 45
def original_ticket_cost : ℝ := 50
def spent_less_than_original_cost_on_food_and_beverages : ℝ := 13
def spent_on_souvenir_Mr_Greene : ℝ := 15
def spent_on_souvenir_Mrs_Greene : ℝ := 2 * spent_on_souvenir_Mr_Greene
def cost_per_game : ℝ := 9
def number_of_children : ℝ := 3
def spent_on_transportation : ℝ := 25
def tax_rate : ℝ := 0.08

def food_and_beverages_cost : ℝ := original_ticket_cost - spent_less_than_original_cost_on_food_and_beverages
def games_cost : ℝ := number_of_children * cost_per_game
def taxable_amount : ℝ := food_and_beverages_cost + spent_on_souvenir_Mr_Greene + spent_on_souvenir_Mrs_Greene + games_cost
def tax : ℝ := tax_rate * taxable_amount
def total_expenditure : ℝ := spent_on_admission + food_and_beverages_cost + spent_on_souvenir_Mr_Greene + spent_on_souvenir_Mrs_Greene + games_cost + spent_on_transportation + tax

theorem greene_family_amusement_park_spending : total_expenditure = 187.72 :=
by {
  sorry
}

end greene_family_amusement_park_spending_l2100_210097


namespace sachin_younger_than_rahul_l2100_210077

theorem sachin_younger_than_rahul :
  ∀ (sachin_age rahul_age : ℕ),
  (sachin_age / rahul_age = 6 / 9) →
  (sachin_age = 14) →
  (rahul_age - sachin_age = 7) :=
by
  sorry

end sachin_younger_than_rahul_l2100_210077


namespace longest_side_eq_24_l2100_210074

noncomputable def x : Real := 19 / 3

def side1 (x : Real) : Real := x + 3
def side2 (x : Real) : Real := 2 * x - 1
def side3 (x : Real) : Real := 3 * x + 5

def perimeter (x : Real) : Prop :=
  side1 x + side2 x + side3 x = 45

theorem longest_side_eq_24 : perimeter x → max (max (side1 x) (side2 x)) (side3 x) = 24 :=
by
  sorry

end longest_side_eq_24_l2100_210074


namespace find_d_minus_r_l2100_210087

theorem find_d_minus_r :
  ∃ d r : ℕ, d > 1 ∧ (1059 % d = r) ∧ (1417 % d = r) ∧ (2312 % d = r) ∧ (d - r = 15) :=
sorry

end find_d_minus_r_l2100_210087


namespace possible_values_of_quadratic_l2100_210095

theorem possible_values_of_quadratic (x : ℝ) (h : x^2 - 5 * x + 4 < 0) : 10 < x^2 + 4 * x + 5 ∧ x^2 + 4 * x + 5 < 37 :=
by
  sorry

end possible_values_of_quadratic_l2100_210095


namespace gcd_lcm_sum_l2100_210030

theorem gcd_lcm_sum : Nat.gcd 24 54 + Nat.lcm 40 10 = 46 := by
  sorry

end gcd_lcm_sum_l2100_210030


namespace g_at_52_l2100_210013

noncomputable def g : ℝ → ℝ := sorry

axiom g_multiplicative : ∀ (x y: ℝ), g (x * y) = y * g x
axiom g_at_1 : g 1 = 10

theorem g_at_52 : g 52 = 520 := sorry

end g_at_52_l2100_210013


namespace true_proposition_among_choices_l2100_210028

theorem true_proposition_among_choices (p q : Prop) (hp : p) (hq : ¬ q) :
  p ∧ ¬ q :=
by
  sorry

end true_proposition_among_choices_l2100_210028


namespace remaining_money_after_payments_l2100_210006

-- Conditions
def initial_money : ℕ := 100
def paid_colin : ℕ := 20
def paid_helen : ℕ := 2 * paid_colin
def paid_benedict : ℕ := paid_helen / 2
def total_paid : ℕ := paid_colin + paid_helen + paid_benedict

-- Proof
theorem remaining_money_after_payments : 
  initial_money - total_paid = 20 := by
  sorry

end remaining_money_after_payments_l2100_210006


namespace polynomial_remainder_l2100_210046

theorem polynomial_remainder :
  (4 * (2.5 : ℝ)^5 - 9 * (2.5 : ℝ)^4 + 7 * (2.5 : ℝ)^2 - 2.5 - 35 = 45.3125) :=
by sorry

end polynomial_remainder_l2100_210046


namespace area_inequality_l2100_210050

variable {a b c : ℝ} (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a)

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) : ℝ :=
  let p := semiperimeter a b c
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

theorem area_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  (2 * (area a b c h))^3 < (a * b * c)^2 := sorry

end area_inequality_l2100_210050


namespace solve_equation_l2100_210096

theorem solve_equation (x : ℝ) (h : (x^2 + x + 1) / (x + 1) = x + 2) : x = -1/2 :=
by sorry

end solve_equation_l2100_210096


namespace solve_inequality_system_l2100_210069

theorem solve_inequality_system (x : ℝ) :
  (x + 2 > -1) ∧ (x - 5 < 3 * (x - 1)) ↔ (x > -1) :=
by
  sorry

end solve_inequality_system_l2100_210069


namespace clark_paid_correct_amount_l2100_210026

-- Definitions based on the conditions
def cost_per_part : ℕ := 80
def number_of_parts : ℕ := 7
def total_discount : ℕ := 121

-- Given conditions
def total_cost_without_discount : ℕ := cost_per_part * number_of_parts
def expected_total_cost_after_discount : ℕ := 439

-- Theorem to prove the amount Clark paid after the discount is correct
theorem clark_paid_correct_amount : total_cost_without_discount - total_discount = expected_total_cost_after_discount := by
  sorry

end clark_paid_correct_amount_l2100_210026


namespace distinguishable_squares_count_l2100_210086

theorem distinguishable_squares_count :
  let colors := 5  -- Number of different colors
  let total_corner_sets :=
    5 + -- All four corners the same color
    5 * 4 + -- Three corners the same color
    Nat.choose 5 2 * 2 + -- Two pairs of corners with the same color
    5 * 4 * 3 * 2 -- All four corners different
  let total_corner_together := total_corner_sets
  let total := 
    (4 * 5 + -- One corner color used
    3 * (5 * 4 + Nat.choose 5 2 * 2) + -- Two corner colors used
    2 * (5 * 4 * 3 * 2) + -- Three corner colors used
    1 * (5 * 4 * 3 * 2)) -- Four corner colors used
  total_corner_together * colors / 10
= 540 :=
by
  sorry

end distinguishable_squares_count_l2100_210086


namespace transactions_proof_l2100_210057

def transactions_problem : Prop :=
  let mabel_transactions := 90
  let anthony_transactions := mabel_transactions + (0.10 * mabel_transactions)
  let cal_transactions := (2 / 3) * anthony_transactions
  let jade_transactions := 81
  jade_transactions - cal_transactions = 15

-- The proof is omitted (replace 'sorry' with an actual proof)
theorem transactions_proof : transactions_problem := by
  sorry

end transactions_proof_l2100_210057


namespace isosceles_obtuse_triangle_angles_l2100_210055

def isosceles (A B C : ℝ) : Prop := A = B ∨ B = C ∨ C = A
def obtuse (A B C : ℝ) : Prop := A > 90 ∨ B > 90 ∨ C > 90

noncomputable def sixty_percent_larger_angle : ℝ := 1.6 * 90

theorem isosceles_obtuse_triangle_angles 
  (A B C : ℝ) 
  (h_iso : isosceles A B C) 
  (h_obt : obtuse A B C) 
  (h_large_angle : A = sixty_percent_larger_angle ∨ B = sixty_percent_larger_angle ∨ C = sixty_percent_larger_angle) 
  (h_sum : A + B + C = 180) : 
  (A = 18 ∨ B = 18 ∨ C = 18) := 
sorry

end isosceles_obtuse_triangle_angles_l2100_210055
