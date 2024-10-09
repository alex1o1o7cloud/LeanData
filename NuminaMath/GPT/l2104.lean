import Mathlib

namespace arithmetic_square_root_of_16_l2104_210440

theorem arithmetic_square_root_of_16 : ∃! (x : ℝ), x^2 = 16 ∧ x ≥ 0 :=
by
  sorry

end arithmetic_square_root_of_16_l2104_210440


namespace joined_after_8_months_l2104_210442

theorem joined_after_8_months
  (investment_A investment_B : ℕ)
  (time_A time_B : ℕ)
  (profit_ratio : ℕ × ℕ)
  (h_A : investment_A = 36000)
  (h_B : investment_B = 54000)
  (h_ratio : profit_ratio = (2, 1))
  (h_time_A : time_A = 12)
  (h_eq : (investment_A * time_A) / (investment_B * time_B) = (profit_ratio.1 / profit_ratio.2)) :
  time_B = 4 := by
  sorry

end joined_after_8_months_l2104_210442


namespace seq_a2010_l2104_210456

-- Definitions and conditions
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ 
  a 2 = 3 ∧ 
  ∀ n ≥ 2, a (n + 1) = (a n * a (n - 1)) % 10

-- Proof statement
theorem seq_a2010 {a : ℕ → ℕ} (h : seq a) : a 2010 = 4 := 
  sorry

end seq_a2010_l2104_210456


namespace product_cubed_roots_l2104_210429

-- Given conditions
def cbrt (x : ℝ) : ℝ := x^(1/3)
def expr : ℝ := cbrt (1 + 27) * cbrt (1 + cbrt 27) * cbrt 9

-- Main statement to prove
theorem product_cubed_roots : expr = cbrt 1008 :=
by sorry

end product_cubed_roots_l2104_210429


namespace shadow_length_false_if_approaching_lamp_at_night_l2104_210479

theorem shadow_length_false_if_approaching_lamp_at_night
  (night : Prop)
  (approaches_lamp : Prop)
  (shadow_longer : Prop) :
  night → approaches_lamp → ¬shadow_longer :=
by
  -- assume it is night and person is approaching lamp
  intros h_night h_approaches
  -- proof is omitted
  sorry

end shadow_length_false_if_approaching_lamp_at_night_l2104_210479


namespace find_S11_l2104_210446

variable (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

axiom sum_of_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : ∀ n, S n = n * (a 1 + a n) / 2
axiom condition1 : is_arithmetic_sequence a
axiom condition2 : a 5 + a 7 = (a 6)^2

-- Proof (statement) that the sum of the first 11 terms is 22
theorem find_S11 : S 11 = 22 :=
  sorry

end find_S11_l2104_210446


namespace f_periodic_with_period_one_l2104_210449

noncomputable def is_periodic (f : ℝ → ℝ) :=
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f (x + c) = f x

theorem f_periodic_with_period_one
  (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f := 
sorry

end f_periodic_with_period_one_l2104_210449


namespace problem1_problem2_l2104_210496

-- Problem 1: Prove that (1) - 8 + 12 - 16 - 23 = -35
theorem problem1 : (1 - 8 + 12 - 16 - 23 = -35) :=
by
  sorry

-- Problem 2: Prove that (3 / 4) + (-1 / 6) - (1 / 3) - (-1 / 8) = 3 / 8
theorem problem2 : (3 / 4 + (-1 / 6) - 1 / 3 + 1 / 8 = 3 / 8) :=
by
  sorry

end problem1_problem2_l2104_210496


namespace second_group_members_l2104_210492

theorem second_group_members (total first third : ℕ) (h1 : total = 70) (h2 : first = 25) (h3 : third = 15) :
  (total - first - third) = 30 :=
by
  sorry

end second_group_members_l2104_210492


namespace tenth_graders_science_only_l2104_210468

theorem tenth_graders_science_only (total_students science_students art_students : ℕ) 
  (h1 : total_students = 140) 
  (h2 : science_students = 100) 
  (h3 : art_students = 75) : 
  (science_students - (science_students + art_students - total_students)) = 65 :=
by
  sorry

end tenth_graders_science_only_l2104_210468


namespace total_pears_picked_l2104_210459

theorem total_pears_picked (keith_pears jason_pears : ℕ) (h1 : keith_pears = 3) (h2 : jason_pears = 2) : keith_pears + jason_pears = 5 :=
by
  sorry

end total_pears_picked_l2104_210459


namespace solution_set_of_inequality_l2104_210485

theorem solution_set_of_inequality (x : ℝ) : x < (1 / x) ↔ (x < -1 ∨ (0 < x ∧ x < 1)) :=
by
  sorry

end solution_set_of_inequality_l2104_210485


namespace binom_sum_l2104_210444

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_sum : binom 7 4 + binom 6 5 = 41 := by
  sorry

end binom_sum_l2104_210444


namespace find_n_l2104_210425

theorem find_n {
    n : ℤ
   } (h1 : 0 ≤ n) (h2 : n < 103) (h3 : 99 * n ≡ 72 [ZMOD 103]) :
    n = 52 :=
sorry

end find_n_l2104_210425


namespace find_int_solutions_l2104_210491

theorem find_int_solutions (x y : ℤ) (h : x^4 - 2 * y^2 = 1) : (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
sorry

end find_int_solutions_l2104_210491


namespace simplest_form_expression_l2104_210408

theorem simplest_form_expression (x y a : ℤ) : 
  (∃ (E : ℚ → Prop), (E (1/3) ∨ E (1/(x-2)) ∨ E ((x^2 * y) / (2*x)) ∨ E (2*a / 8)) → (E (1/(x-2)) ↔ E (1/(x-2)))) :=
by 
  sorry

end simplest_form_expression_l2104_210408


namespace number_of_different_towers_l2104_210426

theorem number_of_different_towers
  (red blue yellow : ℕ)
  (total_height : ℕ)
  (total_cubes : ℕ)
  (discarded_cubes : ℕ)
  (ways_to_leave_out : ℕ)
  (multinomial_coefficient : ℕ) : 
  red = 3 → blue = 4 → yellow = 5 → total_height = 10 → total_cubes = 12 → discarded_cubes = 2 →
  ways_to_leave_out = 66 → multinomial_coefficient = 4200 →
  (ways_to_leave_out * multinomial_coefficient) = 277200 :=
by
  -- proof skipped
  sorry

end number_of_different_towers_l2104_210426


namespace calculate_length_of_bridge_l2104_210409

/-- Define the conditions based on given problem -/
def length_of_bridge (speed1 speed2 : ℕ) (length1 length2 : ℕ) (time : ℕ) : ℕ :=
    let distance_covered_train1 := speed1 * time
    let bridge_length_train1 := distance_covered_train1 - length1
    let distance_covered_train2 := speed2 * time
    let bridge_length_train2 := distance_covered_train2 - length2
    max bridge_length_train1 bridge_length_train2

/-- Given conditions -/
def speed_train1 := 15 -- in m/s
def length_train1 := 130 -- in meters
def speed_train2 := 20 -- in m/s
def length_train2 := 90 -- in meters
def crossing_time := 30 -- in seconds

theorem calculate_length_of_bridge : length_of_bridge speed_train1 speed_train2 length_train1 length_train2 crossing_time = 510 :=
by
  -- omitted proof
  sorry

end calculate_length_of_bridge_l2104_210409


namespace find_tuesday_temp_l2104_210423

variable (temps : List ℝ) (avg : ℝ) (len : ℕ) 

theorem find_tuesday_temp (h1 : temps = [99.1, 98.2, 99.3, 99.8, 99, 98.9, tuesday_temp])
                         (h2 : avg = 99)
                         (h3 : len = 7)
                         (h4 : (temps.sum / len) = avg) :
                         tuesday_temp = 98.7 := 
sorry

end find_tuesday_temp_l2104_210423


namespace four_kids_wash_three_whiteboards_in_20_minutes_l2104_210471

-- Condition: It takes one kid 160 minutes to wash six whiteboards
def time_per_whiteboard_for_one_kid : ℚ := 160 / 6

-- Calculation involving four kids
def time_per_whiteboard_for_four_kids : ℚ := time_per_whiteboard_for_one_kid / 4

-- The total time it takes for four kids to wash three whiteboards together
def total_time_for_four_kids_washing_three_whiteboards : ℚ := time_per_whiteboard_for_four_kids * 3

-- Statement to prove
theorem four_kids_wash_three_whiteboards_in_20_minutes : 
  total_time_for_four_kids_washing_three_whiteboards = 20 :=
by
  sorry

end four_kids_wash_three_whiteboards_in_20_minutes_l2104_210471


namespace apartment_building_floors_l2104_210447

theorem apartment_building_floors (K E P : ℕ) (h1 : 1 < K) (h2 : K < E) (h3 : E < P) (h4 : K * E * P = 715) : 
  E = 11 :=
sorry

end apartment_building_floors_l2104_210447


namespace tan_double_angle_l2104_210495

theorem tan_double_angle (θ : ℝ) (P : ℝ × ℝ) 
  (h_vertex : θ = 0) 
  (h_initial_side : ∀ x, θ = x)
  (h_terminal_side : P = (-1, 2)) : 
  Real.tan (2 * θ) = 4 / 3 := 
by 
  sorry

end tan_double_angle_l2104_210495


namespace reciprocal_of_repeating_decimal_6_l2104_210410

-- Define a repeating decimal .\overline{6}
noncomputable def repeating_decimal_6 : ℚ := 2 / 3

-- The theorem statement: reciprocal of .\overline{6} is 3/2
theorem reciprocal_of_repeating_decimal_6 :
  repeating_decimal_6⁻¹ = (3 / 2) :=
sorry

end reciprocal_of_repeating_decimal_6_l2104_210410


namespace triangle_perimeter_triangle_side_c_l2104_210474

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) (h1 : b * (Real.sin (A/2))^2 + a * (Real.sin (B/2))^2 = C / 2) (h2 : c = 2) : 
  a + b + c = 6 := 
sorry

theorem triangle_side_c (A B C : ℝ) (a b c : ℝ) (h1 : b * (Real.sin (A/2))^2 + a * (Real.sin (B/2))^2 = C / 2) 
(h2 : C = Real.pi / 3) (h3 : 2 * Real.sqrt 3 = (1/2) * a * b * Real.sin (Real.pi / 3)) : 
c = 2 * Real.sqrt 2 := 
sorry

end triangle_perimeter_triangle_side_c_l2104_210474


namespace third_test_point_l2104_210477

noncomputable def test_points : ℝ × ℝ × ℝ :=
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  (x1, x2, x3)

theorem third_test_point :
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  x1 > x2 → x3 = 3.528 :=
by
  intros
  sorry

end third_test_point_l2104_210477


namespace apples_left_correct_l2104_210445

noncomputable def apples_left (initial_apples : ℝ) (additional_apples : ℝ) (apples_for_pie : ℝ) : ℝ :=
  initial_apples + additional_apples - apples_for_pie

theorem apples_left_correct :
  apples_left 10.0 5.5 4.25 = 11.25 :=
by
  sorry

end apples_left_correct_l2104_210445


namespace largest_number_of_stores_visited_l2104_210499

-- Definitions of the conditions
def num_stores := 7
def total_visits := 21
def num_shoppers := 11
def two_stores_visitors := 7
def at_least_one_store (n : ℕ) : Prop := n ≥ 1

-- The goal statement
theorem largest_number_of_stores_visited :
  ∃ n, n ≤ num_shoppers ∧ 
       at_least_one_store n ∧ 
       (n * 2 + (num_shoppers - n)) <= total_visits ∧ 
       (num_shoppers - n) ≥ 3 → 
       n = 4 :=
sorry

end largest_number_of_stores_visited_l2104_210499


namespace perp_lines_l2104_210406

noncomputable def line_1 (k : ℝ) : ℝ → ℝ → ℝ := λ x y => (k - 3) * x + (5 - k) * y + 1
noncomputable def line_2 (k : ℝ) : ℝ → ℝ → ℝ := λ x y => 2 * (k - 3) * x - 2 * y + 3

theorem perp_lines (k : ℝ) : 
  let l1 := line_1 k
  let l2 := line_2 k
  (∀ x y, l1 x y = 0 → l2 x y = 0 → (k = 1 ∨ k = 4)) :=
by
    sorry

end perp_lines_l2104_210406


namespace cristina_catches_up_l2104_210404

theorem cristina_catches_up
  (t : ℝ)
  (cristina_speed : ℝ := 5)
  (nicky_speed : ℝ := 3)
  (nicky_head_start : ℝ := 54)
  (distance_cristina : ℝ := cristina_speed * t)
  (distance_nicky : ℝ := nicky_head_start + nicky_speed * t) :
  distance_cristina = distance_nicky → t = 27 :=
by
  intros h
  sorry

end cristina_catches_up_l2104_210404


namespace riza_son_age_l2104_210439

theorem riza_son_age (R S : ℕ) (h1 : R = S + 25) (h2 : R + S = 105) : S = 40 :=
by
  sorry

end riza_son_age_l2104_210439


namespace opposite_of_neg_2023_l2104_210431

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l2104_210431


namespace lateral_surface_area_of_cube_l2104_210458

-- Define the side length of the cube
def side_length : ℕ := 12

-- Define the area of one face of the cube
def area_of_one_face (s : ℕ) : ℕ := s * s

-- Define the lateral surface area of the cube
def lateral_surface_area (s : ℕ) : ℕ := 4 * (area_of_one_face s)

-- Prove the lateral surface area of a cube with side length 12 m is equal to 576 m²
theorem lateral_surface_area_of_cube : lateral_surface_area side_length = 576 := by
  sorry

end lateral_surface_area_of_cube_l2104_210458


namespace answer_choices_l2104_210405

theorem answer_choices (n : ℕ) (h : (n + 1) ^ 4 = 625) : n = 4 :=
by {
  sorry
}

end answer_choices_l2104_210405


namespace molecular_weight_calculation_l2104_210487

def molecular_weight (n_Ar n_Si n_H n_O : ℕ) (w_Ar w_Si w_H w_O : ℝ) : ℝ :=
  n_Ar * w_Ar + n_Si * w_Si + n_H * w_H + n_O * w_O

theorem molecular_weight_calculation :
  molecular_weight 2 3 12 8 39.948 28.085 1.008 15.999 = 304.239 :=
by
  sorry

end molecular_weight_calculation_l2104_210487


namespace min_tan_expression_l2104_210411

open Real

theorem min_tan_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
(h_eq : sin α * cos β - 2 * cos α * sin β = 0) :
  ∃ x, x = tan (2 * π + α) + tan (π / 2 - β) ∧ x = 2 * sqrt 2 :=
sorry

end min_tan_expression_l2104_210411


namespace equal_distances_sum_of_distances_moving_distances_equal_l2104_210464

-- Define the points A, B, origin O, and moving point P
def A : ℝ := -1
def B : ℝ := 3
def O : ℝ := 0

-- Define the moving point P
def P (x : ℝ) : ℝ := x

-- Define the velocities of each point
def vP : ℝ := -1
def vA : ℝ := -5
def vB : ℝ := -20

-- Proof statement ①: Distance from P to A and B are equal implies x = 1
theorem equal_distances (x : ℝ) (h : abs (x + 1) = abs (x - 3)) : x = 1 :=
sorry

-- Proof statement ②: Sum of distances from P to A and B is 5 implies x = -3/2 or 7/2
theorem sum_of_distances (x : ℝ) (h : abs (x + 1) + abs (x - 3) = 5) : x = -3/2 ∨ x = 7/2 :=
sorry

-- Proof statement ③: Moving distances equal at times t = 4/15 or 2/23
theorem moving_distances_equal (t : ℝ) (h : abs (4 * t + 1) = abs (19 * t - 3)) : t = 4/15 ∨ t = 2/23 :=
sorry

end equal_distances_sum_of_distances_moving_distances_equal_l2104_210464


namespace Ron_spends_15_dollars_l2104_210436

theorem Ron_spends_15_dollars (cost_per_bar : ℝ) (sections_per_bar : ℕ) (num_scouts : ℕ) (s'mores_per_scout : ℕ) :
  cost_per_bar = 1.50 ∧ sections_per_bar = 3 ∧ num_scouts = 15 ∧ s'mores_per_scout = 2 →
  cost_per_bar * (num_scouts * s'mores_per_scout / sections_per_bar) = 15 :=
by
  sorry

end Ron_spends_15_dollars_l2104_210436


namespace bananas_to_pears_ratio_l2104_210455

theorem bananas_to_pears_ratio (B P : ℕ) (hP : P = 50) (h1 : B + 10 = 160) (h2: ∃ k : ℕ, B = k * P) : B / P = 3 :=
by
  -- proof steps would go here
  sorry

end bananas_to_pears_ratio_l2104_210455


namespace terminating_decimals_count_l2104_210419

theorem terminating_decimals_count :
  (∀ m : ℤ, 1 ≤ m ∧ m ≤ 999 → ∃ k : ℕ, (m : ℝ) / 1000 = k / (2 ^ 3 * 5 ^ 3)) :=
by
  sorry

end terminating_decimals_count_l2104_210419


namespace trigonometric_identity_l2104_210424

theorem trigonometric_identity (θ : ℝ) (h : Real.sin θ + Real.cos θ = Real.sqrt 2) :
  Real.tan θ + 1 / Real.tan θ = 2 :=
by
  sorry

end trigonometric_identity_l2104_210424


namespace percentage_difference_l2104_210450

theorem percentage_difference (N : ℝ) (hN : N = 160) : 0.50 * N - 0.35 * N = 24 := by
  sorry

end percentage_difference_l2104_210450


namespace average_mb_per_hour_l2104_210417

theorem average_mb_per_hour
  (days : ℕ)
  (original_space  : ℕ)
  (compression_rate : ℝ)
  (total_hours : ℕ := days * 24)
  (effective_space : ℝ := original_space * (1 - compression_rate))
  (space_per_hour : ℝ := effective_space / total_hours) :
  days = 20 ∧ original_space = 25000 ∧ compression_rate = 0.10 → 
  (Int.floor (space_per_hour + 0.5)) = 47 := by
  intros
  sorry

end average_mb_per_hour_l2104_210417


namespace evaluate_expression_l2104_210470

theorem evaluate_expression : 5^2 - 5 + (6^2 - 6) - (7^2 - 7) + (8^2 - 8) = 64 :=
by sorry

end evaluate_expression_l2104_210470


namespace max_lines_with_specific_angles_l2104_210494

def intersecting_lines : ℕ := 6

theorem max_lines_with_specific_angles :
  ∀ (n : ℕ), (∀ (i j : ℕ), i ≠ j → (∃ θ : ℝ, θ = 30 ∨ θ = 60 ∨ θ = 90)) → n ≤ 6 :=
  sorry

end max_lines_with_specific_angles_l2104_210494


namespace johns_weight_l2104_210453

-- Definitions based on the given conditions
def max_weight : ℝ := 1000
def safety_percentage : ℝ := 0.20
def bar_weight : ℝ := 550

-- Theorem stating the mathematically equivalent proof problem
theorem johns_weight : 
  (johns_safe_weight : ℝ) = max_weight - safety_percentage * max_weight 
  → (johns_safe_weight - bar_weight = 250) :=
by
  sorry

end johns_weight_l2104_210453


namespace measure_of_angle_C_l2104_210430

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 180) (h2 : C = 7 * D) : C = 157.5 := 
by 
  sorry

end measure_of_angle_C_l2104_210430


namespace find_C_l2104_210413

theorem find_C (C : ℤ) (h : 2 * C - 3 = 11) : C = 7 :=
sorry

end find_C_l2104_210413


namespace necessary_but_not_sufficient_for_gt_l2104_210472

variable {a b : ℝ}

theorem necessary_but_not_sufficient_for_gt : a > b → a > b - 1 :=
by sorry

end necessary_but_not_sufficient_for_gt_l2104_210472


namespace average_temperature_l2104_210432

theorem average_temperature :
  ∀ (T : ℝ) (Tt : ℝ),
  -- Conditions
  (43 + T + T + T) / 4 = 48 → 
  Tt = 35 →
  -- Proof
  (T + T + T + Tt) / 4 = 46 :=
by
  intros T Tt H1 H2
  sorry

end average_temperature_l2104_210432


namespace trailing_zeroes_500_fact_l2104_210428

-- Define a function to count multiples of a given number in a range
def countMultiples (n m : Nat) : Nat :=
  m / n

-- Define a function to count trailing zeroes in the factorial
def trailingZeroesFactorial (n : Nat) : Nat :=
  countMultiples 5 n + countMultiples (5^2) n + countMultiples (5^3) n + countMultiples (5^4) n

theorem trailing_zeroes_500_fact : trailingZeroesFactorial 500 = 124 :=
by
  sorry

end trailing_zeroes_500_fact_l2104_210428


namespace minimum_moves_l2104_210407

theorem minimum_moves (n : ℕ) : 
  n > 0 → ∃ k l : ℕ, k + 2 * l ≥ ⌊ (n^2 : ℝ) / 2 ⌋₊ ∧ k + l ≥ ⌊ (n^2 : ℝ) / 3 ⌋₊ :=
by 
  intro hn
  sorry

end minimum_moves_l2104_210407


namespace tax_free_value_is_500_l2104_210435

-- Definitions of the given conditions
def total_value : ℝ := 730
def paid_tax : ℝ := 18.40
def tax_rate : ℝ := 0.08

-- Definition of the excess value
def excess_value (E : ℝ) := tax_rate * E = paid_tax

-- Definition of the tax-free threshold value
def tax_free_limit (V : ℝ) := total_value - (paid_tax / tax_rate) = V

-- The theorem to be proven
theorem tax_free_value_is_500 : 
  ∃ V : ℝ, (total_value - (paid_tax / tax_rate) = V) ∧ V = 500 :=
  by
    sorry -- Proof to be completed

end tax_free_value_is_500_l2104_210435


namespace trigonometric_identity_l2104_210454

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 1 / 3) : 
  Real.cos (Real.pi / 4 + α) * Real.cos (Real.pi / 4 - α) = 7 / 18 :=
by sorry

end trigonometric_identity_l2104_210454


namespace wire_length_is_180_l2104_210414

def wire_problem (length1 length2 : ℕ) (h1 : length1 = 106) (h2 : length2 = 74) (h3 : length1 = length2 + 32) : Prop :=
  (length1 + length2 = 180)

-- Use the definition as an assumption to write the theorem.
theorem wire_length_is_180 (length1 length2 : ℕ) 
  (h1 : length1 = 106) 
  (h2 : length2 = 74) 
  (h3 : length1 = length2 + 32) : 
  length1 + length2 = 180 :=
by
  rw [h1, h2] at h3
  sorry

end wire_length_is_180_l2104_210414


namespace num_rooms_with_2_windows_l2104_210489

theorem num_rooms_with_2_windows:
  ∃ (num_rooms_with_2_windows: ℕ),
  (∀ (num_rooms_with_4_windows num_rooms_with_3_windows: ℕ), 
    num_rooms_with_4_windows = 5 ∧ 
    num_rooms_with_3_windows = 8 ∧
    4 * num_rooms_with_4_windows + 3 * num_rooms_with_3_windows + 2 * num_rooms_with_2_windows = 122) → 
    num_rooms_with_2_windows = 39 :=
by
  sorry

end num_rooms_with_2_windows_l2104_210489


namespace first_year_fee_correct_l2104_210402

noncomputable def first_year_fee (n : ℕ) (annual_increase : ℕ) (sixth_year_fee : ℕ) : ℕ :=
  sixth_year_fee - (n - 1) * annual_increase

theorem first_year_fee_correct (n annual_increase sixth_year_fee value : ℕ) 
  (h_n : n = 6) (h_annual_increase : annual_increase = 10) 
  (h_sixth_year_fee : sixth_year_fee = 130) (h_value : value = 80) :
  first_year_fee n annual_increase sixth_year_fee = value :=
by {
  sorry
}

end first_year_fee_correct_l2104_210402


namespace complete_half_job_in_six_days_l2104_210462

theorem complete_half_job_in_six_days (x : ℕ) (h1 : 2 * x = x + 6) : x = 6 :=
  by
    sorry

end complete_half_job_in_six_days_l2104_210462


namespace max_diagonals_in_chessboard_l2104_210476

/-- The maximum number of non-intersecting diagonals that can be drawn in an 8x8 chessboard is 36. -/
theorem max_diagonals_in_chessboard : 
  ∃ (diagonals : Finset (ℕ × ℕ)), 
  diagonals.card = 36 ∧ 
  ∀ (d1 d2 : ℕ × ℕ), d1 ∈ diagonals → d2 ∈ diagonals → d1 ≠ d2 → d1.fst ≠ d2.fst ∧ d1.snd ≠ d2.snd := 
  sorry

end max_diagonals_in_chessboard_l2104_210476


namespace ladder_base_distance_l2104_210448

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l2104_210448


namespace arithmetic_mean_of_three_digit_multiples_of_8_l2104_210463

-- Define the conditions given in the problem
def smallest_three_digit_multiple_of_8 := 104
def largest_three_digit_multiple_of_8 := 992
def common_difference := 8

-- Define the sequence as an arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ :=
  smallest_three_digit_multiple_of_8 + n * common_difference

-- Calculate the number of terms in the sequence
def number_of_terms : ℕ :=
  (largest_three_digit_multiple_of_8 - smallest_three_digit_multiple_of_8) / common_difference + 1

-- Calculate the sum of the arithmetic sequence
def sum_of_sequence : ℕ :=
  (number_of_terms * (smallest_three_digit_multiple_of_8 + largest_three_digit_multiple_of_8)) / 2

-- Calculate the arithmetic mean
def arithmetic_mean : ℕ :=
  sum_of_sequence / number_of_terms

-- The statement to be proved
theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  arithmetic_mean = 548 :=
by
  sorry

end arithmetic_mean_of_three_digit_multiples_of_8_l2104_210463


namespace one_and_two_thirds_of_what_number_is_45_l2104_210478

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l2104_210478


namespace remainder_of_125_div_j_l2104_210457

theorem remainder_of_125_div_j (j : ℕ) (h1 : j > 0) (h2 : 75 % (j^2) = 3) : 125 % j = 5 :=
sorry

end remainder_of_125_div_j_l2104_210457


namespace remainder_when_divided_by_9_l2104_210481

theorem remainder_when_divided_by_9 (x : ℕ) (h1 : x > 0) (h2 : (5 * x) % 9 = 7) : x % 9 = 5 :=
sorry

end remainder_when_divided_by_9_l2104_210481


namespace sqrt_expression_nonneg_l2104_210467

theorem sqrt_expression_nonneg {b : ℝ} : b - 3 ≥ 0 ↔ b ≥ 3 := by
  sorry

end sqrt_expression_nonneg_l2104_210467


namespace tammy_avg_speed_second_day_l2104_210415

theorem tammy_avg_speed_second_day (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) :
  v + 0.5 = 4 :=
sorry

end tammy_avg_speed_second_day_l2104_210415


namespace repeated_root_and_m_value_l2104_210403

theorem repeated_root_and_m_value :
  (∃ x m : ℝ, (x = 2 ∨ x = -2) ∧ 
              (m / (x ^ 2 - 4) + 2 / (x + 2) = 1 / (x - 2)) ∧ 
              (m = 4 ∨ m = 8)) :=
sorry

end repeated_root_and_m_value_l2104_210403


namespace polynomial_expansion_identity_l2104_210482

variable (a0 a1 a2 a3 a4 : ℝ)

theorem polynomial_expansion_identity
  (h : (2 - (x : ℝ))^4 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) :
  a0 - a1 + a2 - a3 + a4 = 81 :=
sorry

end polynomial_expansion_identity_l2104_210482


namespace archie_initial_marbles_l2104_210451

theorem archie_initial_marbles (M : ℝ) (h1 : 0.6 * M + 0.5 * 0.4 * M = M - 20) : M = 100 :=
sorry

end archie_initial_marbles_l2104_210451


namespace one_cow_empties_pond_in_75_days_l2104_210422

-- Define the necessary variables and their types
variable (c a b : ℝ) -- c represents daily water inflow from the spring
                      -- a represents the total volume of the pond
                      -- b represents the daily consumption per cow

-- Define the conditions
def condition1 : Prop := a + 3 * c = 3 * 17 * b
def condition2 : Prop := a + 30 * c = 30 * 2 * b

-- Target statement we want to prove
theorem one_cow_empties_pond_in_75_days (h1 : condition1 c a b) (h2 : condition2 c a b) :
  ∃ t : ℝ, t = 75 := 
sorry -- Proof to be provided


end one_cow_empties_pond_in_75_days_l2104_210422


namespace cos_sum_series_l2104_210438

theorem cos_sum_series : 
  (∑' n : ℤ, if (n % 2 = 1 ∨ n % 2 = -1) then (1 : ℝ) / (n : ℝ)^2 else 0) = (π^2) / 8 := by
  sorry

end cos_sum_series_l2104_210438


namespace range_of_f_area_of_triangle_l2104_210441

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x - Real.pi / 6)

-- Problem Part (I)
theorem range_of_f : 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
      -1/2 ≤ f x ∧ f x ≤ 1/4) :=
sorry

-- Problem Part (II)
theorem area_of_triangle 
  (A B C : ℝ)
  (a b c : ℝ) 
  (hA0 : 0 < A ∧ A < Real.pi)
  (hS1 : a = Real.sqrt 3)
  (hS2 : b = 2 * c)
  (hF : f A = 1/4) :
  (∃ (area : ℝ), area = (1/2) * b * c * Real.sin A ∧ area = Real.sqrt 3 / 3)
:=
sorry

end range_of_f_area_of_triangle_l2104_210441


namespace frank_money_made_l2104_210401

theorem frank_money_made
  (spent_on_blades : ℕ)
  (number_of_games : ℕ)
  (cost_per_game : ℕ)
  (total_cost_games := number_of_games * cost_per_game)
  (total_money_made := spent_on_blades + total_cost_games)
  (H1 : spent_on_blades = 11)
  (H2 : number_of_games = 4)
  (H3 : cost_per_game = 2) :
  total_money_made = 19 :=
by
  sorry

end frank_money_made_l2104_210401


namespace every_algorithm_must_have_sequential_structure_l2104_210418

def is_sequential_structure (alg : Type) : Prop := sorry -- This defines what a sequential structure is

def must_have_sequential_structure (alg : Type) : Prop :=
∀ alg, is_sequential_structure alg

theorem every_algorithm_must_have_sequential_structure :
  must_have_sequential_structure nat := sorry

end every_algorithm_must_have_sequential_structure_l2104_210418


namespace sqrt_121_pm_11_l2104_210460

theorem sqrt_121_pm_11 :
  (∃ y : ℤ, y * y = 121) ∧ (∃ x : ℤ, x = 11 ∨ x = -11) → (∃ x : ℤ, x * x = 121 ∧ (x = 11 ∨ x = -11)) :=
by
  sorry

end sqrt_121_pm_11_l2104_210460


namespace probability_of_both_making_basket_l2104_210497

noncomputable def P : Set ℕ → ℚ :=
  sorry

def A : Set ℕ := sorry
def B : Set ℕ := sorry

axiom prob_A : P A = 2 / 5
axiom prob_B : P B = 1 / 2
axiom independent : P (A ∩ B) = P A * P B

theorem probability_of_both_making_basket :
  P (A ∩ B) = 1 / 5 :=
by
  rw [independent, prob_A, prob_B]
  norm_num

end probability_of_both_making_basket_l2104_210497


namespace dorothy_money_left_l2104_210493

-- Define the conditions
def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18

-- Define the calculation of the amount of money left after paying taxes
def money_left (income : ℝ) (rate : ℝ) : ℝ :=
  income - (rate * income)

-- State the main theorem to prove
theorem dorothy_money_left :
  money_left annual_income tax_rate = 49200 := 
by
  sorry

end dorothy_money_left_l2104_210493


namespace other_root_of_quadratic_l2104_210443

theorem other_root_of_quadratic (m : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + m = 0 → x = -1) → (∀ y : ℝ, y^2 - 4 * y + m = 0 → y = 5) :=
sorry

end other_root_of_quadratic_l2104_210443


namespace geometric_series_sum_example_l2104_210433

def geometric_series_sum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r ^ n - 1) / (r - 1))

theorem geometric_series_sum_example : geometric_series_sum 2 (-3) 8 = -3280 :=
by
  sorry

end geometric_series_sum_example_l2104_210433


namespace nursing_home_beds_l2104_210484

/-- A community plans to build a nursing home with 100 rooms, consisting of single, double, and triple rooms.
    Let t be the number of single rooms (1 nursing bed), double rooms (2 nursing beds) is twice the single rooms,
    and the rest are triple rooms (3 nursing beds).
    The equations are:
    - number of double rooms: 2 * t
    - number of single rooms: t
    - number of triple rooms: 100 - 3 * t
    - total number of nursing beds: t + 2 * (2 * t) + 3 * (100 - 3 * t) 
    Prove the following:
    1. If the total number of nursing beds is 200, then t = 25.
    2. The maximum number of nursing beds is 260.
    3. The minimum number of nursing beds is 180.
-/
theorem nursing_home_beds (t : ℕ) (h1 : 10 ≤ t ∧ t ≤ 30) (total_rooms : ℕ := 100) :
  (∀ total_beds, (total_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → total_beds = 200 → t = 25) ∧
  (∀ max_beds, (max_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → t = 10 → max_beds = 260) ∧
  (∀ min_beds, (min_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → t = 30 → min_beds = 180) := 
by
  sorry

end nursing_home_beds_l2104_210484


namespace mandy_yoga_time_l2104_210475

theorem mandy_yoga_time 
  (gym_ratio : ℕ)
  (bike_ratio : ℕ)
  (yoga_exercise_ratio : ℕ)
  (bike_time : ℕ) 
  (exercise_ratio : ℕ) 
  (yoga_ratio : ℕ)
  (h1 : gym_ratio = 2)
  (h2 : bike_ratio = 3)
  (h3 : yoga_exercise_ratio = 2)
  (h4 : exercise_ratio = 3)
  (h5 : bike_time = 18)
  (total_exercise_time : ℕ)
  (yoga_time : ℕ)
  (h6: total_exercise_time = ((gym_ratio * bike_time) / bike_ratio) + bike_time)
  (h7 : yoga_time = (yoga_exercise_ratio * total_exercise_time) / exercise_ratio) :
  yoga_time = 20 := 
by 
  sorry

end mandy_yoga_time_l2104_210475


namespace find_divisor_l2104_210465

variable (Dividend : ℕ) (Quotient : ℕ) (Divisor : ℕ)
variable (h1 : Dividend = 64)
variable (h2 : Quotient = 8)
variable (h3 : Dividend = Divisor * Quotient)

theorem find_divisor : Divisor = 8 := by
  sorry

end find_divisor_l2104_210465


namespace smallest_a_l2104_210466

theorem smallest_a (a : ℕ) (h_a : a > 8) : (∀ x : ℤ, ¬ Prime (x^4 + a^2)) ↔ a = 9 :=
by
  sorry

end smallest_a_l2104_210466


namespace book_distribution_l2104_210473

theorem book_distribution (x : ℕ) (books : ℕ) :
  (books = 3 * x + 8) ∧ (books < 5 * x - 5 + 2) → (x = 6 ∧ books = 26) :=
by
  sorry

end book_distribution_l2104_210473


namespace intersection_of_sets_l2104_210480

def is_angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

def is_acute_angle (α : ℝ) : Prop :=
  α < 90

theorem intersection_of_sets (α : ℝ) :
  (is_acute_angle α ∧ is_angle_in_first_quadrant α) ↔
  (∃ k : ℤ, k ≤ 0 ∧ k * 360 < α ∧ α < k * 360 + 90) := 
sorry

end intersection_of_sets_l2104_210480


namespace geom_seq_m_equals_11_l2104_210400

theorem geom_seq_m_equals_11 
  (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 1) 
  (h2 : |q| ≠ 1) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h4 : a m = (a 1) * (a 2) * (a 3) * (a 4) * (a 5)) : 
  m = 11 :=
sorry

end geom_seq_m_equals_11_l2104_210400


namespace democrats_and_republicans_seating_l2104_210420

theorem democrats_and_republicans_seating : 
  let n := 6
  let factorial := Nat.factorial
  let arrangements := (factorial n) * (factorial n)
  let circular_table := 1
  arrangements * circular_table = 518400 :=
by 
  sorry

end democrats_and_republicans_seating_l2104_210420


namespace non_egg_laying_chickens_count_l2104_210412

noncomputable def num_chickens : ℕ := 80
noncomputable def roosters : ℕ := num_chickens / 4
noncomputable def hens : ℕ := num_chickens - roosters
noncomputable def egg_laying_hens : ℕ := (3 * hens) / 4
noncomputable def hens_on_vacation : ℕ := (2 * egg_laying_hens) / 10
noncomputable def remaining_hens_after_vacation : ℕ := egg_laying_hens - hens_on_vacation
noncomputable def ill_hens : ℕ := (1 * remaining_hens_after_vacation) / 10
noncomputable def non_egg_laying_chickens : ℕ := roosters + hens_on_vacation + ill_hens

theorem non_egg_laying_chickens_count : non_egg_laying_chickens = 33 := by
  sorry

end non_egg_laying_chickens_count_l2104_210412


namespace smallest_k_l2104_210437

def v_seq (v : ℕ → ℝ) : Prop :=
  v 0 = 1/8 ∧ ∀ k, v (k + 1) = 3 * v k - 3 * (v k)^2

noncomputable def limit_M : ℝ := 0.5

theorem smallest_k 
  (v : ℕ → ℝ)
  (hv : v_seq v) :
  ∃ k : ℕ, |v k - limit_M| ≤ 1 / 2 ^ 500 ∧ ∀ n < k, ¬ (|v n - limit_M| ≤ 1 / 2 ^ 500) := 
sorry

end smallest_k_l2104_210437


namespace function_domain_l2104_210461

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l2104_210461


namespace probability_of_sine_inequality_l2104_210416

open Set Real

noncomputable def probability_sine_inequality (x : ℝ) : Prop :=
  ∃ (μ : MeasureTheory.Measure ℝ), μ (Ioc (-3) 3) = 1 ∧
    μ {x | sin (π / 6 * x) ≥ 1 / 2} = 1 / 3

theorem probability_of_sine_inequality : probability_sine_inequality x :=
by
  sorry

end probability_of_sine_inequality_l2104_210416


namespace count_two_digit_powers_of_three_l2104_210434

theorem count_two_digit_powers_of_three : 
  ∃ (n1 n2 : ℕ), 10 ≤ 3^n1 ∧ 3^n1 < 100 ∧ 10 ≤ 3^n2 ∧ 3^n2 < 100 ∧ n1 ≠ n2 ∧ ∀ n : ℕ, (10 ≤ 3^n ∧ 3^n < 100) → (n = n1 ∨ n = n2) ∧ n1 = 3 ∧ n2 = 4 := by
  sorry

end count_two_digit_powers_of_three_l2104_210434


namespace largest_positive_integer_n_exists_l2104_210490

theorem largest_positive_integer_n_exists (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n, 
    0 < n ∧ 
    (n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 9) ∧ 
    ∀ m, 0 < m → 
      (m^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 9) → 
      m ≤ n :=
  sorry

end largest_positive_integer_n_exists_l2104_210490


namespace area_of_triangle_l2104_210452

def line1 (x : ℝ) : ℝ := 3 * x + 6
def line2 (x : ℝ) : ℝ := -2 * x + 10

theorem area_of_triangle : 
  let inter_x := (10 - 6) / (3 + 2)
  let inter_y := line1 inter_x
  let base := (10 - 6 : ℝ)
  let height := inter_x
  base * height / 2 = 8 / 5 := 
by
  sorry

end area_of_triangle_l2104_210452


namespace expression_simplification_l2104_210483

theorem expression_simplification :
  (2 + 3) * (2^3 + 3^3) * (2^9 + 3^9) * (2^27 + 3^27) = 3^41 - 2^41 := 
sorry

end expression_simplification_l2104_210483


namespace sin_150_eq_half_l2104_210498

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_150_eq_half_l2104_210498


namespace greatest_number_of_consecutive_integers_sum_36_l2104_210488

theorem greatest_number_of_consecutive_integers_sum_36 :
  ∃ (N : ℕ), 
    (∃ a : ℤ, N * a + ((N - 1) * N) / 2 = 36) ∧ 
    (∀ N' : ℕ, (∃ a' : ℤ, N' * a' + ((N' - 1) * N') / 2 = 36) → N' ≤ 72) := by
  sorry

end greatest_number_of_consecutive_integers_sum_36_l2104_210488


namespace hexagon_angle_arith_prog_l2104_210469

theorem hexagon_angle_arith_prog (x d : ℝ) (hx : x > 0) (hd : d > 0) 
  (h_eq : 6 * x + 15 * d = 720) : x = 120 :=
by
  sorry

end hexagon_angle_arith_prog_l2104_210469


namespace total_length_remaining_l2104_210486

def initial_figure_height : ℕ := 10
def initial_figure_width : ℕ := 7
def top_right_removed : ℕ := 2
def middle_left_removed : ℕ := 2
def bottom_removed : ℕ := 3
def near_top_left_removed : ℕ := 1

def remaining_top_length : ℕ := initial_figure_width - top_right_removed
def remaining_left_length : ℕ := initial_figure_height - middle_left_removed
def remaining_bottom_length : ℕ := initial_figure_width - bottom_removed
def remaining_right_length : ℕ := initial_figure_height - near_top_left_removed

theorem total_length_remaining :
  remaining_top_length + remaining_left_length + remaining_bottom_length + remaining_right_length = 26 := by
  sorry

end total_length_remaining_l2104_210486


namespace problem_1_problem_2_l2104_210421

def A := {x : ℝ | 1 < 2 * x - 1 ∧ 2 * x - 1 < 7}
def B := {x : ℝ | x^2 - 2 * x - 3 < 0}

theorem problem_1 : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

theorem problem_2 : (A ∪ B)ᶜ = {x : ℝ | x ≤ -1 ∨ x ≥ 4} :=
sorry

end problem_1_problem_2_l2104_210421


namespace mismatching_socks_count_l2104_210427

-- Define the conditions given in the problem
def total_socks : ℕ := 65
def pairs_matching_ankle_socks : ℕ := 13
def pairs_matching_crew_socks : ℕ := 10

-- Define the calculated counts as per the conditions
def matching_ankle_socks : ℕ := pairs_matching_ankle_socks * 2
def matching_crew_socks : ℕ := pairs_matching_crew_socks * 2
def total_matching_socks : ℕ := matching_ankle_socks + matching_crew_socks

-- The statement to prove
theorem mismatching_socks_count : total_socks - total_matching_socks = 19 := by
  sorry

end mismatching_socks_count_l2104_210427
