import Mathlib

namespace NUMINAMATH_GPT_multiple_of_9_digit_l2071_207123

theorem multiple_of_9_digit :
  ∃ d : ℕ, d < 10 ∧ (5 + 6 + 7 + 8 + d) % 9 = 0 ∧ d = 1 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_9_digit_l2071_207123


namespace NUMINAMATH_GPT_max_value_of_sinx_over_2_minus_cosx_l2071_207115

theorem max_value_of_sinx_over_2_minus_cosx (x : ℝ) : 
  ∃ y_max, y_max = (Real.sqrt 3) / 3 ∧ ∀ y, y = (Real.sin x) / (2 - Real.cos x) → y ≤ y_max :=
sorry

end NUMINAMATH_GPT_max_value_of_sinx_over_2_minus_cosx_l2071_207115


namespace NUMINAMATH_GPT_area_of_sine_curve_l2071_207116

theorem area_of_sine_curve :
  let f := (fun x => Real.sin x)
  let a := -Real.pi
  let b := 2 * Real.pi
  (∫ x in a..b, f x) = 6 :=
by
  sorry

end NUMINAMATH_GPT_area_of_sine_curve_l2071_207116


namespace NUMINAMATH_GPT_no_solution_for_p_eq_7_l2071_207126

theorem no_solution_for_p_eq_7 : ∀ x : ℝ, x ≠ 4 → x ≠ 8 → ( (x-3)/(x-4) = (x-7)/(x-8) ) → false := by
  intro x h1 h2 h
  sorry

end NUMINAMATH_GPT_no_solution_for_p_eq_7_l2071_207126


namespace NUMINAMATH_GPT_factors_are_divisors_l2071_207186

theorem factors_are_divisors (a b c d : ℕ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) (h4 : d = 5) : 
  a ∣ 30 ∧ b ∣ 30 ∧ c ∣ 30 ∧ d ∣ 30 :=
by
  sorry

end NUMINAMATH_GPT_factors_are_divisors_l2071_207186


namespace NUMINAMATH_GPT_sequence_general_term_l2071_207111

noncomputable def a_n (n : ℕ) : ℝ :=
  sorry

-- The main statement
theorem sequence_general_term (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ m n : ℕ, |a (m + n) - a m - a n| ≤ 1 / (p * m + q * n)) :
  ∀ n : ℕ, a n = n :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l2071_207111


namespace NUMINAMATH_GPT_second_number_less_than_twice_first_l2071_207189

theorem second_number_less_than_twice_first (x y z : ℤ) (h1 : y = 37) (h2 : x + y = 57) (h3 : y = 2 * x - z) : z = 3 :=
by
  sorry

end NUMINAMATH_GPT_second_number_less_than_twice_first_l2071_207189


namespace NUMINAMATH_GPT_units_digit_six_l2071_207105

theorem units_digit_six (n : ℕ) (h : n > 0) : (6 ^ n) % 10 = 6 :=
by sorry

example : (6 ^ 7) % 10 = 6 :=
units_digit_six 7 (by norm_num)

end NUMINAMATH_GPT_units_digit_six_l2071_207105


namespace NUMINAMATH_GPT_sum_of_digits_x_squared_l2071_207138

theorem sum_of_digits_x_squared {r x p q : ℕ} (h_r : r ≤ 400) 
  (h_x_form : x = p * r^3 + p * r^2 + q * r + q) 
  (h_pq_condition : 7 * q = 17 * p) 
  (h_x2_form : ∃ (a b c : ℕ), x^2 = a * r^6 + b * r^5 + c * r^4 + d * r^3 + c * r^2 + b * r + a ∧ d = 0) :
  p + p + q + q = 400 := 
sorry

end NUMINAMATH_GPT_sum_of_digits_x_squared_l2071_207138


namespace NUMINAMATH_GPT_prob_A_and_B_succeed_prob_vaccine_A_successful_l2071_207161

-- Define the probabilities of success for Company A, Company B, and Company C
def P_A := (2 : ℚ) / 3
def P_B := (1 : ℚ) / 2
def P_C := (3 : ℚ) / 5

-- Define the theorem statements

-- Theorem for the probability that both Company A and Company B succeed
theorem prob_A_and_B_succeed : P_A * P_B = 1 / 3 := by
  sorry

-- Theorem for the probability that vaccine A is successfully developed
theorem prob_vaccine_A_successful : 1 - ((1 - P_A) * (1 - P_B)) = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_prob_A_and_B_succeed_prob_vaccine_A_successful_l2071_207161


namespace NUMINAMATH_GPT_bread_weight_eq_anton_weight_l2071_207133

-- Definitions of variables
variables (A B F X : ℝ)

-- Given conditions
axiom cond1 : X + F = A + B
axiom cond2 : B + X = A + F

-- Theorem to prove
theorem bread_weight_eq_anton_weight : X = A :=
by
  sorry

end NUMINAMATH_GPT_bread_weight_eq_anton_weight_l2071_207133


namespace NUMINAMATH_GPT_time_to_cover_escalator_l2071_207108

theorem time_to_cover_escalator (escalator_speed person_speed length : ℕ) (h1 : escalator_speed = 11) (h2 : person_speed = 3) (h3 : length = 126) : 
  length / (escalator_speed + person_speed) = 9 := by
  sorry

end NUMINAMATH_GPT_time_to_cover_escalator_l2071_207108


namespace NUMINAMATH_GPT_negation_proof_l2071_207124

theorem negation_proof :
  (∃ x₀ : ℝ, x₀ < 2) → ¬ (∀ x : ℝ, x < 2) :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l2071_207124


namespace NUMINAMATH_GPT_surface_area_of_equal_volume_cube_l2071_207104

def vol_rect_prism (l w h : ℝ) : ℝ := l * w * h
def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

theorem surface_area_of_equal_volume_cube :
  (vol_rect_prism 5 5 45 = surface_area_cube 10.5) :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_equal_volume_cube_l2071_207104


namespace NUMINAMATH_GPT_expression_value_l2071_207196

theorem expression_value (a b c d : ℝ) (h1 : a * b = 1) (h2 : c + d = 0) :
  -((a * b) ^ (1/3)) + (c + d).sqrt + 1 = 0 :=
by sorry

end NUMINAMATH_GPT_expression_value_l2071_207196


namespace NUMINAMATH_GPT_time_3339_minutes_after_midnight_l2071_207100

def minutes_since_midnight (minutes : ℕ) : ℕ × ℕ :=
  let hours := minutes / 60
  let remaining_minutes := minutes % 60
  (hours, remaining_minutes)

def time_after_midnight (start_time : ℕ × ℕ) (hours : ℕ) (minutes : ℕ) : ℕ × ℕ :=
  let (start_hours, start_minutes) := start_time
  let total_minutes := start_hours * 60 + start_minutes + hours * 60 + minutes
  let end_hours := total_minutes / 60
  let end_minutes := total_minutes % 60
  (end_hours, end_minutes)

theorem time_3339_minutes_after_midnight :
  time_after_midnight (0, 0) 55 39 = (7, 39) :=
by
  sorry

end NUMINAMATH_GPT_time_3339_minutes_after_midnight_l2071_207100


namespace NUMINAMATH_GPT_difference_between_two_numbers_l2071_207158

theorem difference_between_two_numbers :
  ∃ a b : ℕ, 
    a + 5 * b = 23405 ∧ 
    (∃ b' : ℕ, b = 10 * b' + 5 ∧ b' = 5 * a) ∧ 
    5 * b - a = 21600 :=
by {
  sorry
}

end NUMINAMATH_GPT_difference_between_two_numbers_l2071_207158


namespace NUMINAMATH_GPT_pascal_15_5th_number_l2071_207102

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end NUMINAMATH_GPT_pascal_15_5th_number_l2071_207102


namespace NUMINAMATH_GPT_initial_percentage_of_salt_l2071_207117

theorem initial_percentage_of_salt (P : ℝ) :
  (P / 100) * 80 = 8 → P = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_percentage_of_salt_l2071_207117


namespace NUMINAMATH_GPT_papaya_tree_growth_ratio_l2071_207113

theorem papaya_tree_growth_ratio :
  ∃ (a1 a2 a3 a4 a5 : ℝ),
    a1 = 2 ∧
    a2 = a1 * 1.5 ∧
    a3 = a2 * 1.5 ∧
    a4 = a3 * 2 ∧
    a1 + a2 + a3 + a4 + a5 = 23 ∧
    a5 = 4.5 ∧
    (a5 / a4) = 0.5 :=
sorry

end NUMINAMATH_GPT_papaya_tree_growth_ratio_l2071_207113


namespace NUMINAMATH_GPT_negative_solutions_iff_l2071_207107

theorem negative_solutions_iff (m x y : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) :
  (x < 0 ∧ y < 0) ↔ m < -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_negative_solutions_iff_l2071_207107


namespace NUMINAMATH_GPT_initial_balls_count_l2071_207193

variables (y w : ℕ)

theorem initial_balls_count (h1 : y = 2 * (w - 10)) (h2 : w - 10 = 5 * (y - 9)) :
  y = 10 ∧ w = 15 :=
sorry

end NUMINAMATH_GPT_initial_balls_count_l2071_207193


namespace NUMINAMATH_GPT_sock_pairs_l2071_207178

def total_ways (n_white n_brown n_blue n_red : ℕ) : ℕ :=
  n_blue * n_white + n_blue * n_brown + n_blue * n_red

theorem sock_pairs (n_white n_brown n_blue n_red : ℕ) (h_white : n_white = 5) (h_brown : n_brown = 4) (h_blue : n_blue = 2) (h_red : n_red = 1) :
  total_ways n_white n_brown n_blue n_red = 20 := by
  -- insert the proof steps here
  sorry

end NUMINAMATH_GPT_sock_pairs_l2071_207178


namespace NUMINAMATH_GPT_expression_evaluation_l2071_207135

theorem expression_evaluation : 2 + 3 * 4 - 5 + 6 * (2 - 1) = 15 := 
by sorry

end NUMINAMATH_GPT_expression_evaluation_l2071_207135


namespace NUMINAMATH_GPT_weight_of_A_l2071_207129

variable (A B C D E : ℕ)

axiom cond1 : A + B + C = 180
axiom cond2 : A + B + C + D = 260
axiom cond3 : E = D + 3
axiom cond4 : B + C + D + E = 256

theorem weight_of_A : A = 87 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_A_l2071_207129


namespace NUMINAMATH_GPT_find_t_l2071_207136

theorem find_t (t : ℝ) : 
  (∃ a b : ℝ, a^2 = t^2 ∧ b^2 = 5 * t ∧ (a - b = 2 * Real.sqrt 6 ∨ b - a = 2 * Real.sqrt 6)) → 
  (t = 2 ∨ t = 3 ∨ t = 6) := 
by
  sorry

end NUMINAMATH_GPT_find_t_l2071_207136


namespace NUMINAMATH_GPT_common_difference_l2071_207142

noncomputable def a : ℕ := 3
noncomputable def an : ℕ := 28
noncomputable def Sn : ℕ := 186

theorem common_difference (d : ℚ) (n : ℕ) (h1 : an = a + (n-1) * d) (h2 : Sn = n * (a + an) / 2) : d = 25 / 11 :=
sorry

end NUMINAMATH_GPT_common_difference_l2071_207142


namespace NUMINAMATH_GPT_evaluate_expression_at_y_minus3_l2071_207194

theorem evaluate_expression_at_y_minus3 :
  let y := -3
  (5 + y * (2 + y) - 4^2) / (y - 4 + y^2 - y) = -8 / 5 :=
by
  let y := -3
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_y_minus3_l2071_207194


namespace NUMINAMATH_GPT_white_surface_area_fraction_l2071_207184

theorem white_surface_area_fraction
    (total_cubes : ℕ)
    (white_cubes : ℕ)
    (red_cubes : ℕ)
    (edge_length : ℕ)
    (white_exposed_area : ℕ)
    (total_surface_area : ℕ)
    (fraction : ℚ)
    (h1 : total_cubes = 64)
    (h2 : white_cubes = 14)
    (h3 : red_cubes = 50)
    (h4 : edge_length = 4)
    (h5 : white_exposed_area = 6)
    (h6 : total_surface_area = 96)
    (h7 : fraction = 1 / 16)
    (h8 : white_cubes + red_cubes = total_cubes)
    (h9 : 6 * (edge_length * edge_length) = total_surface_area)
    (h10 : white_exposed_area / total_surface_area = fraction) :
    fraction = 1 / 16 := by
    sorry

end NUMINAMATH_GPT_white_surface_area_fraction_l2071_207184


namespace NUMINAMATH_GPT_smallest_n_l2071_207109

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 4 * n = k^2) (h2 : ∃ l : ℕ, 5 * n = l^3) : n = 100 :=
sorry

end NUMINAMATH_GPT_smallest_n_l2071_207109


namespace NUMINAMATH_GPT_find_alcohol_quantity_l2071_207106

theorem find_alcohol_quantity 
  (A W : ℝ) 
  (h1 : A / W = 2 / 5)
  (h2 : A / (W + 10) = 2 / 7) : 
  A = 10 :=
sorry

end NUMINAMATH_GPT_find_alcohol_quantity_l2071_207106


namespace NUMINAMATH_GPT_avg_of_last_three_l2071_207191

-- Define the conditions given in the problem
def avg_5 : Nat := 54
def avg_2 : Nat := 48
def num_list_length : Nat := 5
def first_two_length : Nat := 2

-- State the theorem
theorem avg_of_last_three
    (h_avg5 : 5 * avg_5 = 270)
    (h_avg2 : 2 * avg_2 = 96) :
  (270 - 96) / 3 = 58 :=
sorry

end NUMINAMATH_GPT_avg_of_last_three_l2071_207191


namespace NUMINAMATH_GPT_clock_angle_at_330_l2071_207176

/--
At 3:00, the hour hand is at 90 degrees from the 12 o'clock position.
The minute hand at 3:30 is at 180 degrees from the 12 o'clock position.
The hour hand at 3:30 has moved an additional 15 degrees (0.5 degrees per minute).
Prove that the smaller angle formed by the hour and minute hands of a clock at 3:30 is 75.0 degrees.
-/
theorem clock_angle_at_330 : 
  let hour_pos_at_3 := 90
  let min_pos_at_330 := 180
  let hour_additional := 15
  (min_pos_at_330 - (hour_pos_at_3 + hour_additional) = 75)
  :=
  by
  sorry

end NUMINAMATH_GPT_clock_angle_at_330_l2071_207176


namespace NUMINAMATH_GPT_opposite_of_a_is_2022_l2071_207183

theorem opposite_of_a_is_2022 (a : Int) (h : -a = -2022) : a = 2022 := by
  sorry

end NUMINAMATH_GPT_opposite_of_a_is_2022_l2071_207183


namespace NUMINAMATH_GPT_koala_fiber_intake_l2071_207112

theorem koala_fiber_intake (r a : ℝ) (hr : r = 0.20) (ha : a = 8) : (a / r) = 40 :=
by
  sorry

end NUMINAMATH_GPT_koala_fiber_intake_l2071_207112


namespace NUMINAMATH_GPT_lost_card_number_l2071_207181

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end NUMINAMATH_GPT_lost_card_number_l2071_207181


namespace NUMINAMATH_GPT_square_area_inscribed_triangle_l2071_207146

-- Definitions from the conditions of the problem
variable (EG : ℝ) (hF : ℝ)

-- Since EG = 12 inches and the altitude from F to EG is 7 inches
theorem square_area_inscribed_triangle 
(EG_eq : EG = 12) 
(hF_eq : hF = 7) :
  ∃ (AB : ℝ), AB ^ 2 = 36 :=
by 
  sorry

end NUMINAMATH_GPT_square_area_inscribed_triangle_l2071_207146


namespace NUMINAMATH_GPT_min_value_inequality_l2071_207172

open Real

theorem min_value_inequality (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ 47 :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l2071_207172


namespace NUMINAMATH_GPT_volume_box_values_l2071_207190

theorem volume_box_values :
  let V := (x + 3) * (x - 3) * (x^2 - 10*x + 25)
  ∃ (x_values : Finset ℕ),
    ∀ x ∈ x_values, V < 1000 ∧ x > 0 ∧ x_values.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_volume_box_values_l2071_207190


namespace NUMINAMATH_GPT_team_B_task_alone_optimal_scheduling_l2071_207103

-- Condition definitions
def task_completed_in_18_months (A : Nat → Prop) : Prop := A 18
def work_together_complete_task_in_10_months (A B : Nat → Prop) : Prop := 
  ∃ n m : ℕ, n = 2 ∧ A n ∧ B m ∧ m = 10 ∧ ∀ x y : ℕ, (x / y = 1 / 18 + 1 / (n + 10))

-- Question 1
theorem team_B_task_alone (B : Nat → Prop) : ∃ x : ℕ, x = 27 := sorry

-- Conditions for the second theorem
def team_a_max_time (a : ℕ) : Prop := a ≤ 6
def team_b_max_time (b : ℕ) : Prop := b ≤ 24
def positive_integers (a b : ℕ) : Prop := a > 0 ∧ b > 0 
def total_work_done (a b : ℕ) : Prop := (a / 18) + (b / 27) = 1

-- Question 2
theorem optimal_scheduling (A B : Nat → Prop) : 
  ∃ a b : ℕ, team_a_max_time a ∧ team_b_max_time b ∧ positive_integers a b ∧
             (a / 18 + b / 27 = 1) → min_cost := sorry

end NUMINAMATH_GPT_team_B_task_alone_optimal_scheduling_l2071_207103


namespace NUMINAMATH_GPT_subset_P_Q_l2071_207197

def P := {x : ℝ | x > 1}
def Q := {x : ℝ | x^2 - x > 0}

theorem subset_P_Q : P ⊆ Q :=
by
  sorry

end NUMINAMATH_GPT_subset_P_Q_l2071_207197


namespace NUMINAMATH_GPT_nonagon_perimeter_is_28_l2071_207171

-- Definitions based on problem conditions
def numSides : Nat := 9
def lengthSides1 : Nat := 3
def lengthSides2 : Nat := 4
def numSidesOfLength1 : Nat := 8
def numSidesOfLength2 : Nat := 1

-- Theorem statement proving that the perimeter is 28 units
theorem nonagon_perimeter_is_28 : 
  numSides = numSidesOfLength1 + numSidesOfLength2 →
  8 * lengthSides1 + 1 * lengthSides2 = 28 :=
by
  intros
  sorry

end NUMINAMATH_GPT_nonagon_perimeter_is_28_l2071_207171


namespace NUMINAMATH_GPT_add_neg_two_and_three_l2071_207149

theorem add_neg_two_and_three : -2 + 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_add_neg_two_and_three_l2071_207149


namespace NUMINAMATH_GPT_total_receipts_correct_l2071_207151

def cost_adult_ticket : ℝ := 5.50
def cost_children_ticket : ℝ := 2.50
def number_of_adults : ℕ := 152
def number_of_children : ℕ := number_of_adults / 2

def receipts_from_adults : ℝ := number_of_adults * cost_adult_ticket
def receipts_from_children : ℝ := number_of_children * cost_children_ticket
def total_receipts : ℝ := receipts_from_adults + receipts_from_children

theorem total_receipts_correct : total_receipts = 1026 := 
by
  -- Proof omitted, proof needed to validate theorem statement.
  sorry

end NUMINAMATH_GPT_total_receipts_correct_l2071_207151


namespace NUMINAMATH_GPT_find_x_value_l2071_207139

theorem find_x_value (x : ℚ) (h : 5 * (x - 10) = 3 * (3 - 3 * x) + 9) : x = 34 / 7 := by
  sorry

end NUMINAMATH_GPT_find_x_value_l2071_207139


namespace NUMINAMATH_GPT_problem1_remainder_of_9_power_100_mod_8_problem2_last_digit_of_2012_power_2012_l2071_207128

-- Problem 1: Prove the remainder of the Euclidean division of \(9^{100}\) by 8 is 1.
theorem problem1_remainder_of_9_power_100_mod_8 :
  (9 ^ 100) % 8 = 1 :=
by
sorry

-- Problem 2: Prove the last digit of \(2012^{2012}\) is 6.
theorem problem2_last_digit_of_2012_power_2012 :
  (2012 ^ 2012) % 10 = 6 :=
by
sorry

end NUMINAMATH_GPT_problem1_remainder_of_9_power_100_mod_8_problem2_last_digit_of_2012_power_2012_l2071_207128


namespace NUMINAMATH_GPT_prove_B_is_guilty_l2071_207118

variables (A B C : Prop)

def guilty_conditions (A B C : Prop) : Prop :=
  (A → ¬ B → C) ∧
  (C → B ∨ A) ∧
  (A → ¬ (A ∧ C)) ∧
  (A ∨ B ∨ C) ∧ 
  ¬ (¬ A ∧ ¬ B ∧ ¬ C)

theorem prove_B_is_guilty : guilty_conditions A B C → B :=
by
  intros h
  sorry

end NUMINAMATH_GPT_prove_B_is_guilty_l2071_207118


namespace NUMINAMATH_GPT_proof_problem_l2071_207150

variable (x y : ℝ)

theorem proof_problem 
  (h1 : 0.30 * x = 0.40 * 150 + 90)
  (h2 : 0.20 * x = 0.50 * 180 - 60)
  (h3 : y = 0.75 * x)
  (h4 : y^2 > x + 100) :
  x = 150 ∧ y = 112.5 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2071_207150


namespace NUMINAMATH_GPT_floor_e_eq_two_l2071_207173

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 := 
sorry

end NUMINAMATH_GPT_floor_e_eq_two_l2071_207173


namespace NUMINAMATH_GPT_units_digit_base7_of_multiplied_numbers_l2071_207170

-- Define the numbers in base 10
def num1 : ℕ := 325
def num2 : ℕ := 67

-- Define the modulus used for base 7
def base : ℕ := 7

-- Function to determine the units digit of the base-7 representation
def units_digit_base7 (n : ℕ) : ℕ := n % base

-- Prove that units_digit_base7 (num1 * num2) = 5
theorem units_digit_base7_of_multiplied_numbers :
  units_digit_base7 (num1 * num2) = 5 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_base7_of_multiplied_numbers_l2071_207170


namespace NUMINAMATH_GPT_skilled_new_worker_installation_avg_cost_electric_vehicle_cost_comparison_l2071_207119

-- Define the variables for the number of vehicles each type of worker can install
variables {x y : ℝ}

-- Define the conditions for system of equations
def skilled_and_new_workers_system1 (x y : ℝ) : Prop :=
  2 * x + y = 10

def skilled_and_new_workers_system2 (x y : ℝ) : Prop :=
  x + 3 * y = 10

-- Prove the number of vehicles each skilled worker and new worker can install
theorem skilled_new_worker_installation (x y : ℝ) (h1 : skilled_and_new_workers_system1 x y) (h2 : skilled_and_new_workers_system2 x y) : x = 4 ∧ y = 2 :=
by {
  -- Proof skipped
  sorry
}

-- Define the average cost equation for electric and gasoline vehicles
def avg_cost (m : ℝ) : Prop :=
  1 = 4 * (m / (m + 0.6))

-- Prove the average cost per kilometer of the electric vehicle
theorem avg_cost_electric_vehicle (m : ℝ) (h : avg_cost m) : m = 0.2 :=
by {
  -- Proof skipped
  sorry
}

-- Define annual cost equations and the comparison condition
variables {a : ℝ}
def annual_cost_electric_vehicle (a : ℝ) : ℝ :=
  0.2 * a + 6400

def annual_cost_gasoline_vehicle (a : ℝ) : ℝ :=
  0.8 * a + 4000

-- Prove that when the annual mileage is greater than 6667 kilometers, the annual cost of buying an electric vehicle is lower
theorem cost_comparison (a : ℝ) (h : a > 6667) : annual_cost_electric_vehicle a < annual_cost_gasoline_vehicle a :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_skilled_new_worker_installation_avg_cost_electric_vehicle_cost_comparison_l2071_207119


namespace NUMINAMATH_GPT_a4_b4_c4_double_square_l2071_207174

theorem a4_b4_c4_double_square (a b c : ℤ) (h : a = b + c) : 
  a^4 + b^4 + c^4 = 2 * ((a^2 - b * c)^2) :=
by {
  sorry -- proof is not provided as per instructions
}

end NUMINAMATH_GPT_a4_b4_c4_double_square_l2071_207174


namespace NUMINAMATH_GPT_sequence_formula_l2071_207167

theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : ∀ n, S n = 2 * a n + 1) : 
  ∀ n, a n = -2 ^ (n - 1) := 
by 
  sorry

end NUMINAMATH_GPT_sequence_formula_l2071_207167


namespace NUMINAMATH_GPT_smallest_n_for_congruence_l2071_207156

theorem smallest_n_for_congruence :
  ∃ n : ℕ, 827 * n % 36 = 1369 * n % 36 ∧ n > 0 ∧ (∀ m : ℕ, 827 * m % 36 = 1369 * m % 36 ∧ m > 0 → m ≥ 18) :=
by sorry

end NUMINAMATH_GPT_smallest_n_for_congruence_l2071_207156


namespace NUMINAMATH_GPT_kanul_initial_amount_l2071_207131

-- Definition based on the problem conditions
def spent_on_raw_materials : ℝ := 3000
def spent_on_machinery : ℝ := 2000
def spent_on_labor : ℝ := 1000
def percent_spent : ℝ := 0.15

-- Definition of the total amount initially had by Kanul
def total_amount_initial (X : ℝ) : Prop :=
  spent_on_raw_materials + spent_on_machinery + percent_spent * X + spent_on_labor = X

-- Theorem stating the conclusion based on the given conditions
theorem kanul_initial_amount : ∃ X : ℝ, total_amount_initial X ∧ X = 7058.82 :=
by {
  sorry
}

end NUMINAMATH_GPT_kanul_initial_amount_l2071_207131


namespace NUMINAMATH_GPT_abes_present_age_l2071_207152

theorem abes_present_age :
  ∃ A : ℕ, A + (A - 7) = 27 ∧ A = 17 :=
by
  sorry

end NUMINAMATH_GPT_abes_present_age_l2071_207152


namespace NUMINAMATH_GPT_crossnumber_unique_solution_l2071_207120

-- Definition of two-digit numbers
def two_digit_numbers (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Definition of prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Definition of square
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The given conditions reformulated
def crossnumber_problem : Prop :=
  ∃ (one_across one_down two_down three_across : ℕ),
    two_digit_numbers one_across ∧ is_prime one_across ∧
    two_digit_numbers one_down ∧ is_square one_down ∧
    two_digit_numbers two_down ∧ is_square two_down ∧
    two_digit_numbers three_across ∧ is_square three_across ∧
    one_across = 83 ∧ one_down = 81 ∧ two_down = 16 ∧ three_across = 16

theorem crossnumber_unique_solution : crossnumber_problem :=
by
  sorry

end NUMINAMATH_GPT_crossnumber_unique_solution_l2071_207120


namespace NUMINAMATH_GPT_bus_driver_compensation_l2071_207122

theorem bus_driver_compensation : 
  let regular_rate := 16
  let regular_hours := 40
  let total_hours_worked := 57
  let overtime_rate := regular_rate + (0.75 * regular_rate)
  let regular_pay := regular_hours * regular_rate
  let overtime_hours_worked := total_hours_worked - regular_hours
  let overtime_pay := overtime_hours_worked * overtime_rate
  let total_compensation := regular_pay + overtime_pay
  total_compensation = 1116 :=
by
  sorry

end NUMINAMATH_GPT_bus_driver_compensation_l2071_207122


namespace NUMINAMATH_GPT_sum_of_digits_of_greatest_prime_divisor_of_16385_is_13_l2071_207160

theorem sum_of_digits_of_greatest_prime_divisor_of_16385_is_13 : 
  ∃ p : ℕ, (p ∣ 16385 ∧ Nat.Prime p ∧ (∀ q : ℕ, q ∣ 16385 → Nat.Prime q → q ≤ p)) ∧ (Nat.digits 10 p).sum = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_greatest_prime_divisor_of_16385_is_13_l2071_207160


namespace NUMINAMATH_GPT_physical_fitness_test_l2071_207127

theorem physical_fitness_test (x : ℝ) (hx : x > 0) :
  (1000 / x - 1000 / (1.25 * x) = 30) :=
sorry

end NUMINAMATH_GPT_physical_fitness_test_l2071_207127


namespace NUMINAMATH_GPT_vector_parallel_x_value_l2071_207147

theorem vector_parallel_x_value :
  ∀ (x : ℝ), let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -3)
  (∃ k : ℝ, b = (k * 3, k * 1)) → x = -9 :=
by
  intro x
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -3)
  intro h
  sorry

end NUMINAMATH_GPT_vector_parallel_x_value_l2071_207147


namespace NUMINAMATH_GPT_range_of_f3_l2071_207195

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 2 + b * x + 1

theorem range_of_f3 {a b : ℝ}
  (h1 : -2 ≤ a - b ∧ a - b ≤ 0) 
  (h2 : -3 ≤ 4 * a + 2 * b ∧ 4 * a + 2 * b ≤ 1) :
  -7 ≤ f a b 3 ∧ f a b 3 ≤ 3 :=
sorry

end NUMINAMATH_GPT_range_of_f3_l2071_207195


namespace NUMINAMATH_GPT_lily_ducks_l2071_207125

variable (D G : ℕ)
variable (Rayden_ducks : ℕ := 3 * D)
variable (Rayden_geese : ℕ := 4 * G)
variable (Lily_geese : ℕ := 10) -- Given G = 10
variable (Rayden_extra : ℕ := 70) -- Given Rayden has 70 more ducks and geese

theorem lily_ducks (h : 3 * D + 4 * Lily_geese = D + Lily_geese + Rayden_extra) : D = 20 :=
by sorry

end NUMINAMATH_GPT_lily_ducks_l2071_207125


namespace NUMINAMATH_GPT_loan_balance_formula_l2071_207145

variable (c V : ℝ) (t n : ℝ)

theorem loan_balance_formula :
  V = c / (1 + t)^(3 * n) →
  n = (Real.log (c / V)) / (3 * Real.log (1 + t)) :=
by sorry

end NUMINAMATH_GPT_loan_balance_formula_l2071_207145


namespace NUMINAMATH_GPT_negation_of_p_l2071_207141

def p : Prop := ∀ x : ℝ, x > Real.sin x

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x ≤ Real.sin x :=
by sorry

end NUMINAMATH_GPT_negation_of_p_l2071_207141


namespace NUMINAMATH_GPT_length_in_scientific_notation_l2071_207180

theorem length_in_scientific_notation : (161000 : ℝ) = 1.61 * 10^5 := 
by 
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_length_in_scientific_notation_l2071_207180


namespace NUMINAMATH_GPT_binomial_expansion_l2071_207185

theorem binomial_expansion (a b : ℕ) (h_a : a = 34) (h_b : b = 5) :
  a^2 + 2*a*b + b^2 = 1521 :=
by
  rw [h_a, h_b]
  sorry

end NUMINAMATH_GPT_binomial_expansion_l2071_207185


namespace NUMINAMATH_GPT_exist_projections_l2071_207198

-- Define types for lines and points
variable {Point : Type} [MetricSpace Point]

-- Define the projection operator
def projection (t_i t_j : Set Point) (p : Point) : Point := 
  sorry -- projection definition will go here

-- Define t1, t2, ..., tk
variables (t : ℕ → Set Point) (k : ℕ)
  (hk : k > 1)  -- condition: k > 1
  (ht_distinct : ∀ i j, i ≠ j → t i ≠ t j)  -- condition: different lines

-- Define the proposition
theorem exist_projections : 
  ∃ (P : ℕ → Point), 
    (∀ i, 1 ≤ i ∧ i < k → P (i + 1) = projection (t i) (t (i + 1)) (P i)) ∧ 
    P 1 = projection (t k) (t 1) (P k) :=
sorry

end NUMINAMATH_GPT_exist_projections_l2071_207198


namespace NUMINAMATH_GPT_ratio_of_fusilli_to_penne_l2071_207154

def number_of_students := 800
def preferred_pasta_types := ["penne", "tortellini", "fusilli", "spaghetti"]
def students_prefer_fusilli := 320
def students_prefer_penne := 160

theorem ratio_of_fusilli_to_penne : (students_prefer_fusilli / students_prefer_penne) = 2 := by
  -- Here we would provide the proof, but since it's a statement, we use sorry
  sorry

end NUMINAMATH_GPT_ratio_of_fusilli_to_penne_l2071_207154


namespace NUMINAMATH_GPT_samBill_l2071_207199

def textMessageCostPerText := 8 -- cents
def extraMinuteCostPerMinute := 15 -- cents
def planBaseCost := 25 -- dollars
def includedPlanHours := 25
def centToDollar (cents: Nat) : Nat := cents / 100

def totalBill (texts: Nat) (hours: Nat) : Nat :=
  let textCost := centToDollar (texts * textMessageCostPerText)
  let extraHours := if hours > includedPlanHours then hours - includedPlanHours else 0
  let extraMinutes := extraHours * 60
  let extraMinuteCost := centToDollar (extraMinutes * extraMinuteCostPerMinute)
  planBaseCost + textCost + extraMinuteCost

theorem samBill :
  totalBill 150 26 = 46 := 
sorry

end NUMINAMATH_GPT_samBill_l2071_207199


namespace NUMINAMATH_GPT_find_other_solution_l2071_207166

theorem find_other_solution (x₁ : ℚ) (x₂ : ℚ) 
  (h₁ : x₁ = 3 / 4) 
  (h₂ : 72 * x₁^2 + 39 * x₁ - 18 = 0) 
  (eq : 72 * x₂^2 + 39 * x₂ - 18 = 0 ∧ x₂ ≠ x₁) : 
  x₂ = -31 / 6 := 
sorry

end NUMINAMATH_GPT_find_other_solution_l2071_207166


namespace NUMINAMATH_GPT_ratio_of_numbers_l2071_207137

theorem ratio_of_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (hsum_diff : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l2071_207137


namespace NUMINAMATH_GPT_solve_arithmetic_sequence_problem_l2071_207144

noncomputable def arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) : Prop :=
  (∀ n, a n = a 0 + n * (a 1 - a 0)) ∧  -- Condition: sequence is arithmetic
  (∀ n, S n = (n * (a 0 + a (n - 1))) / 2) ∧  -- Condition: sum of first n terms
  (m > 1) ∧  -- Condition: m > 1
  (a (m - 1) + a (m + 1) - a m ^ 2 = 0) ∧  -- Given condition
  (S (2 * m - 1) = 38)  -- Given that sum of first 2m-1 terms equals 38

-- The statement we need to prove
theorem solve_arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) :
  arithmetic_sequence_problem a S m → m = 10 :=
by
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_solve_arithmetic_sequence_problem_l2071_207144


namespace NUMINAMATH_GPT_calculate_train_length_l2071_207163

noncomputable def train_length (speed_kmph : ℕ) (time_secs : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_mps := (speed_kmph * 1000) / 3600
  let total_distance := speed_mps * time_secs
  total_distance - bridge_length_m

theorem calculate_train_length :
  train_length 60 14.998800095992321 140 = 110 :=
by
  sorry

end NUMINAMATH_GPT_calculate_train_length_l2071_207163


namespace NUMINAMATH_GPT_find_angle_C_find_area_l2071_207162

open Real

-- Definition of the problem conditions and questions

-- Condition: Given a triangle and the trigonometric relationship
variables {A B C : ℝ} {a b c : ℝ}

-- Condition 1: Trigonometric identity provided in the problem
axiom trig_identity : (sqrt 3) * c / (cos C) = a / (cos (3 * π / 2 + A))

-- First part of the problem
theorem find_angle_C (h1 : sqrt 3 * c / cos C = a / cos (3 * π / 2 + A)) : C = π / 6 :=
sorry

-- Second part of the problem
noncomputable def area_of_triangle (a b C : ℝ) : ℝ := 1 / 2 * a * b * sin C

variables {c' b' : ℝ}
-- Given conditions for the second question 
axiom condition_c_a : c' / a = 2
axiom condition_b : b' = 4 * sqrt 3

-- Definitions to align with the given problem
def c_from_a (a : ℝ) : ℝ := 2 * a

-- The final theorem for the second part
theorem find_area (hC : C = π / 6) (hc : c_from_a a = c') (hb : b' = 4 * sqrt 3) :
  area_of_triangle a b' C = 2 * sqrt 15 - 2 * sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_angle_C_find_area_l2071_207162


namespace NUMINAMATH_GPT_jack_estimate_larger_l2071_207168

variable {x y a b : ℝ}

theorem jack_estimate_larger (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (ha : 0 < a) (hb : 0 < b) : 
  (x + a) - (y - b) > x - y :=
by
  sorry

end NUMINAMATH_GPT_jack_estimate_larger_l2071_207168


namespace NUMINAMATH_GPT_find_ab_pairs_l2071_207164

open Set

-- Definitions
def f (a b x : ℝ) : ℝ := a * x + b

-- Main theorem
theorem find_ab_pairs (a b : ℝ) :
  (∀ x y : ℝ, (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → 
    f a b x * f a b y + f a b (x + y - x * y) ≤ 0) ↔ 
  (-1 ≤ b ∧ b ≤ 0 ∧ -(b + 1) ≤ a ∧ a ≤ -b) :=
by sorry

end NUMINAMATH_GPT_find_ab_pairs_l2071_207164


namespace NUMINAMATH_GPT_percent_employed_females_l2071_207110

theorem percent_employed_females (total_population employed_population employed_males : ℝ)
  (h1 : employed_population = 0.6 * total_population)
  (h2 : employed_males = 0.48 * total_population) :
  ((employed_population - employed_males) / employed_population) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_percent_employed_females_l2071_207110


namespace NUMINAMATH_GPT_chris_birthday_after_45_days_l2071_207159

theorem chris_birthday_after_45_days (k : ℕ) (h : k = 45) (tuesday : ℕ) (h_tuesday : tuesday = 2) : 
  (tuesday + k) % 7 = 5 := 
sorry

end NUMINAMATH_GPT_chris_birthday_after_45_days_l2071_207159


namespace NUMINAMATH_GPT_tangent_from_point_to_circle_l2071_207101

theorem tangent_from_point_to_circle :
  ∀ (x y : ℝ),
  (x - 6)^2 + (y - 3)^2 = 4 →
  (x = 10 → y = 0 →
    4 * x - 3 * y = 19) :=
by
  sorry

end NUMINAMATH_GPT_tangent_from_point_to_circle_l2071_207101


namespace NUMINAMATH_GPT_area_of_triangle_BP_Q_is_24_l2071_207165

open Real

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
1/2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem area_of_triangle_BP_Q_is_24
  (A B C P H Q : ℝ × ℝ)
  (h_triangle_ABC_right : C.1 = 0 ∧ C.2 = 0 ∧ B.2 = 0 ∧ A.2 ≠ 0)
  (h_BC_diameter : distance B C = 26)
  (h_tangent_AP : distance P B = distance P C ∧ P ≠ C)
  (h_PH_perpendicular_BC : P.1 = H.1 ∧ H.2 = 0)
  (h_PH_intersects_AB_at_Q : H.1 = Q.1 ∧ Q.2 ≠ 0)
  (h_BH_CH_ratio : 4 * distance B H = 9 * distance C H)
  : triangle_area B P Q = 24 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_BP_Q_is_24_l2071_207165


namespace NUMINAMATH_GPT_solve_for_x_l2071_207114

theorem solve_for_x (x z : ℝ) (h : z = 3 * x) :
  (4 * z^2 + z + 5 = 3 * (8 * x^2 + z + 3)) ↔ 
  (x = (1 + Real.sqrt 19) / 4 ∨ x = (1 - Real.sqrt 19) / 4) := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2071_207114


namespace NUMINAMATH_GPT_muscovy_more_than_cayuga_l2071_207143

theorem muscovy_more_than_cayuga
  (M C K : ℕ)
  (h1 : M + C + K = 90)
  (h2 : M = 39)
  (h3 : M = 2 * C + 3 + C) :
  M - C = 27 := by
  sorry

end NUMINAMATH_GPT_muscovy_more_than_cayuga_l2071_207143


namespace NUMINAMATH_GPT_molly_swam_28_meters_on_sunday_l2071_207182

def meters_swam_on_saturday : ℕ := 45
def total_meters_swum : ℕ := 73
def meters_swam_on_sunday := total_meters_swum - meters_swam_on_saturday

theorem molly_swam_28_meters_on_sunday : meters_swam_on_sunday = 28 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_molly_swam_28_meters_on_sunday_l2071_207182


namespace NUMINAMATH_GPT_square_root_condition_l2071_207177

-- Define the condition
def meaningful_square_root (x : ℝ) : Prop :=
  x - 5 ≥ 0

-- Define the theorem that x must be greater than or equal to 5 for the square root to be meaningful
theorem square_root_condition (x : ℝ) : meaningful_square_root x ↔ x ≥ 5 := by
  sorry

end NUMINAMATH_GPT_square_root_condition_l2071_207177


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2071_207155

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2071_207155


namespace NUMINAMATH_GPT_find_larger_page_l2071_207132

theorem find_larger_page {x y : ℕ} (h1 : y = x + 1) (h2 : x + y = 125) : y = 63 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_page_l2071_207132


namespace NUMINAMATH_GPT_apples_after_operations_l2071_207179

-- Define the initial conditions
def initial_apples : ℕ := 38
def used_apples : ℕ := 20
def bought_apples : ℕ := 28

-- State the theorem we want to prove
theorem apples_after_operations : initial_apples - used_apples + bought_apples = 46 :=
by
  sorry

end NUMINAMATH_GPT_apples_after_operations_l2071_207179


namespace NUMINAMATH_GPT_problem1_l2071_207140

theorem problem1 (a b : ℝ) : 
  ((-2 * a) ^ 3 * (- (a * b^2)) ^ 3 - 4 * a * b^2 * (2 * a^5 * b^4 + (1 / 2) * a * b^3 - 5)) / (-2 * a * b) = a * b^4 - 10 * b :=
sorry

end NUMINAMATH_GPT_problem1_l2071_207140


namespace NUMINAMATH_GPT_number_of_whole_numbers_between_sqrt2_and_3e_is_7_l2071_207188

noncomputable def number_of_whole_numbers_between_sqrt2_and_3e : ℕ :=
  let sqrt2 : ℝ := Real.sqrt 2
  let e : ℝ := Real.exp 1
  let small_int := Nat.ceil sqrt2 -- This is 2
  let large_int := Nat.floor (3 * e) -- This is 8
  large_int - small_int + 1 -- The number of integers between small_int and large_int (inclusive)

theorem number_of_whole_numbers_between_sqrt2_and_3e_is_7 :
  number_of_whole_numbers_between_sqrt2_and_3e = 7 := by
  sorry

end NUMINAMATH_GPT_number_of_whole_numbers_between_sqrt2_and_3e_is_7_l2071_207188


namespace NUMINAMATH_GPT_candies_per_friend_l2071_207187

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : additional_candies = 4) (h3 : num_friends = 6) : 
  (initial_candies + additional_candies) / num_friends = 4 := 
by
  sorry

end NUMINAMATH_GPT_candies_per_friend_l2071_207187


namespace NUMINAMATH_GPT_new_profit_percentage_l2071_207192

theorem new_profit_percentage (P SP NSP NP : ℝ) 
  (h1 : SP = 1.10 * P) 
  (h2 : SP = 879.9999999999993) 
  (h3 : NP = 0.90 * P) 
  (h4 : NSP = SP + 56) : 
  (NSP - NP) / NP * 100 = 30 := 
by
  sorry

end NUMINAMATH_GPT_new_profit_percentage_l2071_207192


namespace NUMINAMATH_GPT_game_spinner_probability_l2071_207153

theorem game_spinner_probability (P_A P_B P_D P_C : ℚ) (h₁ : P_A = 1/4) (h₂ : P_B = 1/3) (h₃ : P_D = 1/6) (h₄ : P_A + P_B + P_C + P_D = 1) :
  P_C = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_game_spinner_probability_l2071_207153


namespace NUMINAMATH_GPT_quadratic_real_roots_range_l2071_207175

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (m-1)*x^2 + x + 1 = 0) → (m ≤ 5/4 ∧ m ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_l2071_207175


namespace NUMINAMATH_GPT_reservoir_original_content_l2071_207134

noncomputable def original_content (T O : ℝ) : Prop :=
  (80 / 100) * T = O + 120 ∧
  O = (50 / 100) * T

theorem reservoir_original_content (T : ℝ) (h1 : (80 / 100) * T = (50 / 100) * T + 120) : 
  (50 / 100) * T = 200 :=
by
  sorry

end NUMINAMATH_GPT_reservoir_original_content_l2071_207134


namespace NUMINAMATH_GPT_length_of_goods_train_l2071_207121

theorem length_of_goods_train 
  (speed_kmph : ℝ) (platform_length : ℝ) (time_sec : ℝ) (train_length : ℝ) 
  (h1 : speed_kmph = 72)
  (h2 : platform_length = 270) 
  (h3 : time_sec = 26) 
  (h4 : train_length = (speed_kmph * 1000 / 3600 * time_sec) - platform_length)
  : train_length = 250 := 
  by
    sorry

end NUMINAMATH_GPT_length_of_goods_train_l2071_207121


namespace NUMINAMATH_GPT_number_of_unique_combinations_l2071_207157

-- Define the inputs and the expected output.
def n := 8
def r := 3
def expected_combinations := 56

-- We state our theorem indicating that the combination of 8 toppings chosen 3 at a time
-- equals 56.
theorem number_of_unique_combinations :
  (Nat.choose n r = expected_combinations) :=
by
  sorry

end NUMINAMATH_GPT_number_of_unique_combinations_l2071_207157


namespace NUMINAMATH_GPT_bamboo_capacity_l2071_207169

theorem bamboo_capacity :
  ∃ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 d : ℚ),
    a_1 + a_2 + a_3 = 4 ∧
    a_6 + a_7 + a_8 + a_9 = 3 ∧
    a_2 = a_1 + d ∧
    a_3 = a_1 + 2*d ∧
    a_4 = a_1 + 3*d ∧
    a_5 = a_1 + 4*d ∧
    a_7 = a_1 + 5*d ∧
    a_8 = a_1 + 6*d ∧
    a_9 = a_1 + 7*d ∧
    a_4 = 1 + 8/66 ∧
    a_5 = 1 + 1/66 :=
sorry

end NUMINAMATH_GPT_bamboo_capacity_l2071_207169


namespace NUMINAMATH_GPT_exists_distinct_nonzero_ints_for_poly_factorization_l2071_207130

theorem exists_distinct_nonzero_ints_for_poly_factorization :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ P Q : Polynomial ℤ, (P * Q = Polynomial.X * (Polynomial.X - Polynomial.C a) * 
   (Polynomial.X - Polynomial.C b) * (Polynomial.X - Polynomial.C c) + 1) ∧ 
   P.leadingCoeff = 1 ∧ Q.leadingCoeff = 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_distinct_nonzero_ints_for_poly_factorization_l2071_207130


namespace NUMINAMATH_GPT_tyler_saltwater_aquariums_l2071_207148

def num_animals_per_aquarium : ℕ := 39
def total_saltwater_animals : ℕ := 2184

theorem tyler_saltwater_aquariums : 
  total_saltwater_animals / num_animals_per_aquarium = 56 := 
by
  sorry

end NUMINAMATH_GPT_tyler_saltwater_aquariums_l2071_207148
