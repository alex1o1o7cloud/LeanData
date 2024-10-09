import Mathlib

namespace sufficient_not_necessary_condition_l2_260

variable (a : ℝ)

theorem sufficient_not_necessary_condition :
  (1 < a ∧ a < 2) → (a^2 - 3 * a ≤ 0) := by
  intro h
  sorry

end sufficient_not_necessary_condition_l2_260


namespace simplify_and_evaluate_expression_l2_201

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.pi^0 + 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (2 * x + 2)) = 2 / 3 := by
  sorry

end simplify_and_evaluate_expression_l2_201


namespace triangle_height_from_area_l2_252

theorem triangle_height_from_area {A b h : ℝ} (hA : A = 36) (hb : b = 8) 
    (formula : A = 1 / 2 * b * h) : h = 9 := 
by
  sorry

end triangle_height_from_area_l2_252


namespace percentage_neither_l2_273

theorem percentage_neither (total_teachers high_blood_pressure heart_trouble both_conditions : ℕ)
  (h1 : total_teachers = 150)
  (h2 : high_blood_pressure = 90)
  (h3 : heart_trouble = 60)
  (h4 : both_conditions = 30) :
  100 * (total_teachers - (high_blood_pressure + heart_trouble - both_conditions)) / total_teachers = 20 :=
by
  sorry

end percentage_neither_l2_273


namespace largest_n_divisibility_l2_276

theorem largest_n_divisibility (n : ℕ) (h : n + 12 ∣ n^3 + 144) : n ≤ 132 :=
  sorry

end largest_n_divisibility_l2_276


namespace geometric_series_sum_l2_209

theorem geometric_series_sum :
  let a := (1/2 : ℚ)
  let r := (-1/3 : ℚ)
  let n := 7
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 547 / 1458 :=
by
  sorry

end geometric_series_sum_l2_209


namespace gasoline_price_april_l2_299

theorem gasoline_price_april (P₀ : ℝ) (P₁ P₂ P₃ P₄ : ℝ) (x : ℝ)
  (h₁ : P₁ = P₀ * 1.20)  -- Price after January's increase
  (h₂ : P₂ = P₁ * 0.80)  -- Price after February's decrease
  (h₃ : P₃ = P₂ * 1.25)  -- Price after March's increase
  (h₄ : P₄ = P₃ * (1 - x / 100))  -- Price after April's decrease
  (h₅ : P₄ = P₀)  -- Price at the end of April equals the initial price
  : x = 17 := 
by
  sorry

end gasoline_price_april_l2_299


namespace nat_number_36_sum_of_digits_l2_251

-- Define the function that represents the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main statement
theorem nat_number_36_sum_of_digits (x : ℕ) (hx : x = 36 * sum_of_digits x) : x = 324 ∨ x = 648 := 
by 
  sorry

end nat_number_36_sum_of_digits_l2_251


namespace rectangle_area_l2_221

theorem rectangle_area (l w : ℕ) (h_diagonal : l^2 + w^2 = 17^2) (h_perimeter : l + w = 23) : l * w = 120 :=
by
  sorry

end rectangle_area_l2_221


namespace time_for_B_alone_to_complete_work_l2_285

theorem time_for_B_alone_to_complete_work :
  (∃ (A B C : ℝ), A = 1 / 4 
  ∧ B + C = 1 / 3
  ∧ A + C = 1 / 2)
  → 1 / B = 12 :=
by
  sorry

end time_for_B_alone_to_complete_work_l2_285


namespace symmetric_axis_of_g_l2_212

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6))

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - (Real.pi / 6))

theorem symmetric_axis_of_g :
  ∃ k : ℤ, (∃ x : ℝ, g x = 2 * Real.sin (k * Real.pi + (Real.pi / 2)) ∧ x = (k * Real.pi) / 2 + (Real.pi / 3)) :=
sorry

end symmetric_axis_of_g_l2_212


namespace num_perfect_squares_diff_consecutive_under_20000_l2_255

theorem num_perfect_squares_diff_consecutive_under_20000 : 
  ∃ n, n = 71 ∧ ∀ a, a ^ 2 < 20000 → ∃ b, a ^ 2 = (b + 1) ^ 2 - b ^ 2 ↔ a ^ 2 % 2 = 1 :=
by
  sorry

end num_perfect_squares_diff_consecutive_under_20000_l2_255


namespace value_of_t_l2_234

theorem value_of_t (t : ℝ) (x y : ℝ) (h1 : x = 1 - 2 * t) (h2 : y = 2 * t - 2) (h3 : x = y) : t = 3 / 4 := 
by
  sorry

end value_of_t_l2_234


namespace max_visible_cubes_from_point_l2_202

theorem max_visible_cubes_from_point (n : ℕ) (h : n = 12) :
  let total_cubes := n^3
  let face_cube_count := n * n
  let edge_count := n
  let visible_face_count := 3 * face_cube_count
  let double_counted_edges := 3 * (edge_count - 1)
  let corner_cube_count := 1
  visible_face_count - double_counted_edges + corner_cube_count = 400 := by
  sorry

end max_visible_cubes_from_point_l2_202


namespace rationalize_denominator_l2_254

theorem rationalize_denominator :
  ∃ (A B C : ℤ), 
  (A + B * Real.sqrt C) = (2 + Real.sqrt 5) / (3 - Real.sqrt 5) 
  ∧ A = 11 ∧ B = 5 ∧ C = 5 ∧ A * B * C = 275 := by
  sorry

end rationalize_denominator_l2_254


namespace complement_correct_l2_205

universe u

-- We define sets A and B
def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 3, 5}

-- Define the complement of B with respect to A
def complement (A B : Set ℕ) : Set ℕ := {x ∈ A | x ∉ B}

-- The theorem we need to prove
theorem complement_correct : complement A B = {2, 4} := 
  sorry

end complement_correct_l2_205


namespace train_speed_approx_72_km_hr_l2_206

noncomputable def train_length : ℝ := 150
noncomputable def bridge_length : ℝ := 132
noncomputable def crossing_time : ℝ := 14.098872090232781
noncomputable def total_distance : ℝ := train_length + bridge_length
noncomputable def speed_m_s : ℝ := total_distance / crossing_time
noncomputable def conversion_factor : ℝ := 3.6
noncomputable def speed_km_hr : ℝ := speed_m_s * conversion_factor

theorem train_speed_approx_72_km_hr : abs (speed_km_hr - 72) < 0.01 :=
sorry

end train_speed_approx_72_km_hr_l2_206


namespace total_journey_time_eq_5_l2_248

-- Define constants for speed and times
def speed1 : ℕ := 40
def speed2 : ℕ := 60
def total_distance : ℕ := 240
def time1 : ℕ := 3

-- Noncomputable definition to avoid computation issues
noncomputable def journey_time : ℕ :=
  let distance1 := speed1 * time1
  let distance2 := total_distance - distance1
  let time2 := distance2 / speed2
  time1 + time2

-- Theorem to state the total journey time
theorem total_journey_time_eq_5 : journey_time = 5 := by
  sorry

end total_journey_time_eq_5_l2_248


namespace pages_left_to_read_l2_223

-- Define the conditions
def total_pages : ℕ := 400
def percent_read : ℚ := 20 / 100
def pages_read := total_pages * percent_read

-- Define the question as a theorem
theorem pages_left_to_read (total_pages : ℕ) (percent_read : ℚ) (pages_read : ℚ) : ℚ :=
total_pages - pages_read

-- Assert the correct answer
example : pages_left_to_read total_pages percent_read pages_read = 320 := 
by
  sorry

end pages_left_to_read_l2_223


namespace range_of_x_l2_278

variable {x p : ℝ}

theorem range_of_x (H : 0 ≤ p ∧ p ≤ 4) : 
  (x^2 + p * x > 4 * x + p - 3) ↔ (x ∈ Set.Iio (-1) ∪ Set.Ioi 3) := 
by
  sorry

end range_of_x_l2_278


namespace frederick_final_amount_l2_263

-- Definitions of conditions
def P : ℝ := 2000
def r : ℝ := 0.05
def n : ℕ := 18

-- Define the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Theorem stating the question's answer
theorem frederick_final_amount : compound_interest P r n = 4813.24 :=
by
  sorry

end frederick_final_amount_l2_263


namespace A_elements_l2_290

open Set -- Open the Set namespace for easy access to set operations

def A : Set ℕ := {x | ∃ (n : ℕ), 12 = n * (6 - x)}

theorem A_elements : A = {0, 2, 3, 4, 5} :=
by
  -- proof steps here
  sorry

end A_elements_l2_290


namespace find_speed_of_P_l2_279

noncomputable def walking_speeds (v_P v_Q : ℝ) : Prop :=
  let distance_XY := 90
  let distance_meet_from_Y := 15
  let distance_P := distance_XY - distance_meet_from_Y
  let distance_Q := distance_XY + distance_meet_from_Y
  (v_Q = v_P + 3) ∧
  (distance_P / v_P = distance_Q / v_Q)

theorem find_speed_of_P : ∃ v_P : ℝ, walking_speeds v_P (v_P + 3) ∧ v_P = 7.5 :=
by
  sorry

end find_speed_of_P_l2_279


namespace proof_problem_l2_213

noncomputable def problem : Prop :=
  ∃ (m n l : Type) (α β : Type) 
    (is_line : ∀ x, x = m ∨ x = n ∨ x = l)
    (is_plane : ∀ x, x = α ∨ x = β)
    (perpendicular : ∀ (l α : Type), Prop)
    (parallel : ∀ (l α : Type), Prop)
    (belongs_to : ∀ (l α : Type), Prop),
    (parallel l α → ∃ l', parallel l' α ∧ parallel l l') ∧
    (perpendicular m α ∧ perpendicular m β → parallel α β)

theorem proof_problem : problem :=
sorry

end proof_problem_l2_213


namespace parallel_lines_perpendicular_lines_l2_291

theorem parallel_lines (a : ℝ) :
  (∃ b c : ℝ, (ax - y + b = 0) ∧ ((a + 2) * x - ay - c = 0)) →
  (∀ s1 s2 : ℝ, s1 = a → s2 = (a + 2) / a → s1 = s2) →
  a = 2 :=
by
  intros
  -- Proof goes here
  sorry

theorem perpendicular_lines (a : ℝ) :
  (∃ b c : ℝ, (ax - y + b = 0) ∧ ((a + 2) * x - ay - c = 0)) →
  (∀ s1 s2 : ℝ, s1 = a → s2 = (a + 2) / a → s1 * s2 = -1) →
  a = 0 ∨ a = -3 :=
by
  intros
  -- Proof goes here
  sorry

end parallel_lines_perpendicular_lines_l2_291


namespace ratio_of_perimeters_of_squares_l2_266

theorem ratio_of_perimeters_of_squares (A B : ℝ) (h: A / B = 16 / 25) : ∃ (P1 P2 : ℝ), P1 / P2 = 4 / 5 :=
by
  sorry

end ratio_of_perimeters_of_squares_l2_266


namespace JameMade112kProfit_l2_262

def JameProfitProblem : Prop :=
  let initial_purchase_cost := 40000
  let feeding_cost_rate := 0.2
  let num_cattle := 100
  let weight_per_cattle := 1000
  let sell_price_per_pound := 2
  let additional_feeding_cost := initial_purchase_cost * feeding_cost_rate
  let total_feeding_cost := initial_purchase_cost + additional_feeding_cost
  let total_purchase_and_feeding_cost := initial_purchase_cost + total_feeding_cost
  let total_revenue := num_cattle * weight_per_cattle * sell_price_per_pound
  let profit := total_revenue - total_purchase_and_feeding_cost
  profit = 112000

theorem JameMade112kProfit :
  JameProfitProblem :=
by
  -- Proof goes here
  sorry

end JameMade112kProfit_l2_262


namespace opposite_numbers_add_l2_277

theorem opposite_numbers_add : ∀ {a b : ℤ}, a + b = 0 → a + b + 3 = 3 :=
by
  intros
  sorry

end opposite_numbers_add_l2_277


namespace susan_books_l2_232

theorem susan_books (S : ℕ) (h1 : S + 4 * S = 3000) : S = 600 :=
by 
  sorry

end susan_books_l2_232


namespace recording_incorrect_l2_246

-- Definitions for given conditions
def qualifying_standard : ℝ := 1.5
def xiao_ming_jump : ℝ := 1.95
def xiao_liang_jump : ℝ := 1.23
def xiao_ming_recording : ℝ := 0.45
def xiao_liang_recording : ℝ := -0.23

-- The proof statement to verify the correctness of the recordings
theorem recording_incorrect :
  (xiao_ming_jump - qualifying_standard = xiao_ming_recording) ∧ 
  (xiao_liang_jump - qualifying_standard ≠ xiao_liang_recording) :=
by
  sorry

end recording_incorrect_l2_246


namespace initial_maintenance_time_l2_236

theorem initial_maintenance_time (x : ℝ) 
  (h1 : (1 + (1 / 3)) * x = 60) : 
  x = 45 :=
by
  sorry

end initial_maintenance_time_l2_236


namespace find_base_of_exponential_l2_293

theorem find_base_of_exponential (a : ℝ) 
  (h₁ : a > 0) 
  (h₂ : a ≠ 1) 
  (h₃ : a ^ 2 = 1 / 16) : 
  a = 1 / 4 := 
sorry

end find_base_of_exponential_l2_293


namespace sachin_age_is_49_l2_280

open Nat

-- Let S be Sachin's age and R be Rahul's age
def Sachin_age : ℕ := 49
def Rahul_age (S : ℕ) := S + 14

theorem sachin_age_is_49 (S R : ℕ) (h1 : R = S + 14) (h2 : S * 9 = R * 7) : S = 49 :=
by sorry

end sachin_age_is_49_l2_280


namespace probability_of_x_gt_5y_l2_247

theorem probability_of_x_gt_5y :
  let rectangle := {(x, y) | 0 ≤ x ∧ x ≤ 3000 ∧ 0 ≤ y ∧ y ≤ 2500}
  let area_of_rectangle := 3000 * 2500
  let triangle := {(x, y) | 0 ≤ x ∧ x ≤ 3000 ∧ 0 ≤ y ∧ y < x / 5}
  let area_of_triangle := (3000 * 600) / 2
  ∃ prob : ℚ, (area_of_triangle / area_of_rectangle = prob) ∧ prob = 3 / 25 := by
  sorry

end probability_of_x_gt_5y_l2_247


namespace least_addition_for_divisibility_least_subtraction_for_divisibility_least_addition_for_common_divisibility_l2_219

theorem least_addition_for_divisibility (n : ℕ) : (1100 + n) % 53 = 0 ↔ n = 9 := by
  sorry

theorem least_subtraction_for_divisibility (n : ℕ) : (1100 - n) % 71 = 0 ↔ n = 0 := by
  sorry

theorem least_addition_for_common_divisibility (X : ℕ) : (1100 + X) % (Nat.lcm 19 43) = 0 ∧ X = 534 := by
  sorry

end least_addition_for_divisibility_least_subtraction_for_divisibility_least_addition_for_common_divisibility_l2_219


namespace xiaohua_distance_rounds_l2_257

def length := 5
def width := 3
def perimeter (a b : ℕ) := (a + b) * 2
def total_distance (perimeter : ℕ) (laps : ℕ) := perimeter * laps

theorem xiaohua_distance_rounds :
  total_distance (perimeter length width) 3 = 30 :=
by sorry

end xiaohua_distance_rounds_l2_257


namespace repeating_decimal_mul_l2_208

theorem repeating_decimal_mul (x : ℝ) (hx : x = 0.3333333333333333) :
  x * 12 = 4 :=
sorry

end repeating_decimal_mul_l2_208


namespace jen_age_when_son_born_l2_225

theorem jen_age_when_son_born (S : ℕ) (Jen_present_age : ℕ) 
  (h1 : S = 16) (h2 : Jen_present_age = 3 * S - 7) : 
  Jen_present_age - S = 25 :=
by {
  sorry -- Proof would be here, but it is not required as per the instructions.
}

end jen_age_when_son_born_l2_225


namespace min_value_expression_l2_230

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x + y + z = 3) (h2 : z = (x + y) / 2) : 
  (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) = 3 / 2 :=
by sorry

end min_value_expression_l2_230


namespace neither_sufficient_nor_necessary_l2_289

theorem neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a + b > 0) ↔ (ab > 0)) := 
sorry

end neither_sufficient_nor_necessary_l2_289


namespace number_of_right_triangles_with_hypotenuse_is_12_l2_265

theorem number_of_right_triangles_with_hypotenuse_is_12 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b : ℕ), 
     (b < 150) →
     (a^2 + b^2 = (b + 2)^2) →
     ∃ (k : ℕ), a = 2 * k ∧ k^2 = b + 1) := 
  sorry

end number_of_right_triangles_with_hypotenuse_is_12_l2_265


namespace cara_between_friends_l2_204

theorem cara_between_friends (n : ℕ) (h : n = 6) : ∃ k : ℕ, k = 15 :=
by {
  sorry
}

end cara_between_friends_l2_204


namespace cyclic_quadrilateral_AD_correct_l2_243

noncomputable def cyclic_quadrilateral_AD_length : ℝ :=
  let R := 200 * Real.sqrt 2
  let AB := 200
  let BC := 200
  let CD := 200
  let AD := 500
  sorry

theorem cyclic_quadrilateral_AD_correct (R AB BC CD AD : ℝ) (hR : R = 200 * Real.sqrt 2) 
  (hAB : AB = 200) (hBC : BC = 200) (hCD : CD = 200) : AD = 500 :=
by
  have hRABBCDC: R = 200 * Real.sqrt 2 ∧ AB = 200 ∧ BC = 200 ∧ CD = 200 := ⟨hR, hAB, hBC, hCD⟩
  sorry

end cyclic_quadrilateral_AD_correct_l2_243


namespace total_adults_wearing_hats_l2_233

theorem total_adults_wearing_hats (total_adults : ℕ) (men_percentage : ℝ) (men_hats_percentage : ℝ) 
  (women_hats_percentage : ℝ) (total_men_wearing_hats : ℕ) (total_women_wearing_hats : ℕ) : 
  (total_adults = 1200) ∧ (men_percentage = 0.60) ∧ (men_hats_percentage = 0.15) 
  ∧ (women_hats_percentage = 0.10)
     → total_men_wearing_hats + total_women_wearing_hats = 156 :=
by
  -- Definitions
  let total_men := total_adults * men_percentage
  let total_women := total_adults - total_men
  let men_wearing_hats := total_men * men_hats_percentage
  let women_wearing_hats := total_women * women_hats_percentage
  sorry

end total_adults_wearing_hats_l2_233


namespace shaded_rectangle_area_l2_281

-- Define the square PQRS and its properties
def is_square (s : ℝ) := ∃ (PQ QR RS SP : ℝ), PQ = s ∧ QR = s ∧ RS = s ∧ SP = s

-- Define the conditions for the side lengths and segments
def side_length := 11
def top_left_height := 6
def top_right_height := 2
def width_bottom_right := 11 - 10
def width_top_right := 8

-- Calculate necessary dimensions
def shaded_rectangle_height := top_left_height - top_right_height
def shaded_rectangle_width := width_top_right - width_bottom_right

-- Proof statement
theorem shaded_rectangle_area (s : ℝ) (h1 : is_square s)
  (h2 : s = side_length)
  (h3 : shaded_rectangle_height = 4)
  (h4 : shaded_rectangle_width = 7) :
  4 * 7 = 28 := by
  sorry

end shaded_rectangle_area_l2_281


namespace avg_weight_increase_l2_256

theorem avg_weight_increase (A : ℝ) (X : ℝ) (hp1 : 8 * A - 65 + 105 = 8 * A + 40)
  (hp2 : 8 * (A + X) = 8 * A + 40) : X = 5 := 
by sorry

end avg_weight_increase_l2_256


namespace graph_symmetry_l2_218

theorem graph_symmetry (f : ℝ → ℝ) : 
  ∀ x : ℝ, f (x - 1) = f (-(x - 1)) ↔ x = 1 :=
by 
  sorry

end graph_symmetry_l2_218


namespace min_value_arithmetic_sequence_l2_249

theorem min_value_arithmetic_sequence (d : ℝ) (n : ℕ) (hd : d ≠ 0) (a1 : ℝ) (ha1 : a1 = 1)
(geo : (1 + 2 * d)^2 = 1 + 12 * d) (Sn : ℝ) (hSn : Sn = n^2) (an : ℝ) (han : an = 2 * n - 1) :
  ∀ (n : ℕ), n > 0 → (2 * Sn + 8) / (an + 3) ≥ 5 / 2 :=
by sorry

end min_value_arithmetic_sequence_l2_249


namespace find_slope_of_line_q_l2_288

theorem find_slope_of_line_q
  (k : ℝ)
  (h₁ : ∀ (x y : ℝ), (y = 3 * x + 5) → (y = k * x + 3) → (x = -4 ∧ y = -7))
  : k = 2.5 :=
sorry

end find_slope_of_line_q_l2_288


namespace find_polynomial_parameters_and_minimum_value_l2_220

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_polynomial_parameters_and_minimum_value 
  (a b c : ℝ)
  (h1 : f (-1) a b c = 7)
  (h2 : 3 * (-1)^2 + 2 * a * (-1) + b = 0)
  (h3 : 3 * 3^2 + 2 * a * 3 + b = 0)
  (h4 : a = -3)
  (h5 : b = -9)
  (h6 : c = 2) :
  f 3 (-3) (-9) 2 = -25 :=
by
  sorry

end find_polynomial_parameters_and_minimum_value_l2_220


namespace average_score_of_class_l2_272

-- Definitions based on the conditions
def class_size : ℕ := 20
def group1_size : ℕ := 10
def group2_size : ℕ := 10
def group1_avg_score : ℕ := 80
def group2_avg_score : ℕ := 60

-- Average score of the whole class
theorem average_score_of_class : 
  (group1_size * group1_avg_score + group2_size * group2_avg_score) / class_size = 70 := 
by sorry

end average_score_of_class_l2_272


namespace order_of_a_b_c_l2_216

noncomputable def a : ℝ := (Real.log (Real.sqrt 2)) / 2
noncomputable def b : ℝ := Real.log 3 / 6
noncomputable def c : ℝ := 1 / (2 * Real.exp 1)

theorem order_of_a_b_c : c > b ∧ b > a := by
  sorry

end order_of_a_b_c_l2_216


namespace solution_set_l2_207

noncomputable def f : ℝ → ℝ := sorry
axiom f'_lt_one_third (x : ℝ) : deriv f x < 1 / 3
axiom f_at_two : f 2 = 1

theorem solution_set : {x : ℝ | 0 < x ∧ x < 4} = {x : ℝ | f (Real.logb 2 x) > (Real.logb 2 x + 1) / 3} :=
by
  sorry

end solution_set_l2_207


namespace smallest_number_divided_into_18_and_60_groups_l2_244

theorem smallest_number_divided_into_18_and_60_groups : ∃ n : ℕ, (∀ m : ℕ, (m % 18 = 0 ∧ m % 60 = 0) → n ≤ m) ∧ (n % 18 = 0 ∧ n % 60 = 0) ∧ n = 180 :=
by
  use 180
  sorry

end smallest_number_divided_into_18_and_60_groups_l2_244


namespace price_for_70_cans_is_correct_l2_228

def regular_price_per_can : ℝ := 0.55
def discount_percentage : ℝ := 0.25
def purchase_quantity : ℕ := 70

def discount_per_can : ℝ := discount_percentage * regular_price_per_can
def discounted_price_per_can : ℝ := regular_price_per_can - discount_per_can

def price_for_72_cans : ℝ := 72 * discounted_price_per_can
def price_for_2_cans : ℝ := 2 * discounted_price_per_can

def final_price_for_70_cans : ℝ := price_for_72_cans - price_for_2_cans

theorem price_for_70_cans_is_correct
    (regular_price_per_can : ℝ := 0.55)
    (discount_percentage : ℝ := 0.25)
    (purchase_quantity : ℕ := 70)
    (disc_per_can : ℝ := discount_percentage * regular_price_per_can)
    (disc_price_per_can : ℝ := regular_price_per_can - disc_per_can)
    (price_72_cans : ℝ := 72 * disc_price_per_can)
    (price_2_cans : ℝ := 2 * disc_price_per_can):
    final_price_for_70_cans = 28.875 :=
by
  sorry

end price_for_70_cans_is_correct_l2_228


namespace time_for_c_l2_269

theorem time_for_c (a b work_completion: ℝ) (ha : a = 16) (hb : b = 6) (habc : work_completion = 3.2) : 
  (12 : ℝ) = 
  (48 * work_completion - 48) / 4 := 
sorry

end time_for_c_l2_269


namespace min_value_a_4b_l2_224

theorem min_value_a_4b (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 / (a - 1) + 1 / (b - 1) = 1) : a + 4 * b = 14 := 
sorry

end min_value_a_4b_l2_224


namespace cole_drive_time_l2_226

theorem cole_drive_time (d : ℝ) (h1 : d / 75 + d / 105 = 1) : (d / 75) * 60 = 35 :=
by
  -- Using the given equation: d / 75 + d / 105 = 1
  -- We solve it step by step and finally show that the time it took to drive to work is 35 minutes.
  sorry

end cole_drive_time_l2_226


namespace find_f_2008_l2_284

noncomputable def f : ℝ → ℝ := sorry

axiom f_zero (f : ℝ → ℝ) : f 0 = 2008
axiom f_inequality_1 (f : ℝ → ℝ) (x : ℝ) : f (x + 2) - f x ≤ 3 * 2^x
axiom f_inequality_2 (f : ℝ → ℝ) (x : ℝ) : f (x + 6) - f x ≥ 63 * 2^x

theorem find_f_2008 (f : ℝ → ℝ) : f 2008 = 2^2008 + 2007 :=
by
  apply sorry

end find_f_2008_l2_284


namespace product_of_two_numbers_l2_283

theorem product_of_two_numbers (x y : ℝ)
  (h1 : x + y = 25)
  (h2 : x - y = 3)
  : x * y = 154 := by
  sorry

end product_of_two_numbers_l2_283


namespace clients_using_radio_l2_210

theorem clients_using_radio (total_clients T R M TR TM RM TRM : ℕ)
  (h1 : total_clients = 180)
  (h2 : T = 115)
  (h3 : M = 130)
  (h4 : TR = 75)
  (h5 : TM = 85)
  (h6 : RM = 95)
  (h7 : TRM = 80) : R = 30 :=
by
  -- Using Inclusion-Exclusion Principle
  have h : total_clients = T + R + M - TR - TM - RM + TRM :=
    sorry  -- Proof of Inclusion-Exclusion principle for these sets
  rw [h1, h2, h3, h4, h5, h6, h7] at h
  -- Solve for R
  sorry

end clients_using_radio_l2_210


namespace jolyn_older_than_leon_l2_270

open Nat

def Jolyn := Nat
def Therese := Nat
def Aivo := Nat
def Leon := Nat

-- Conditions
variable (jolyn therese aivo leon : Nat)
variable (h1 : jolyn = therese + 2) -- Jolyn is 2 months older than Therese
variable (h2 : therese = aivo + 5) -- Therese is 5 months older than Aivo
variable (h3 : leon = aivo + 2) -- Leon is 2 months older than Aivo

theorem jolyn_older_than_leon :
  jolyn = leon + 5 := by
  sorry

end jolyn_older_than_leon_l2_270


namespace banana_permutations_l2_267

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l2_267


namespace exists_monotonic_subsequence_l2_282

open Function -- For function related definitions
open Finset -- For finite set operations

-- Defining the theorem with the given conditions and the goal to be proved
theorem exists_monotonic_subsequence (a : Fin 10 → ℝ) (h : ∀ i j : Fin 10, i ≠ j → a i ≠ a j) :
  ∃ (i1 i2 i3 i4 : Fin 10), i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧
  ((a i1 < a i2 ∧ a i2 < a i3 ∧ a i3 < a i4) ∨ (a i1 > a i2 ∧ a i2 > a i3 ∧ a i3 > a i4)) :=
by
  sorry -- Proof is omitted as per the instructions

end exists_monotonic_subsequence_l2_282


namespace no_real_roots_iff_k_gt_1_div_4_l2_229

theorem no_real_roots_iff_k_gt_1_div_4 (k : ℝ) :
  (∀ x : ℝ, ¬ (x^2 - x + k = 0)) ↔ k > 1 / 4 :=
by
  sorry

end no_real_roots_iff_k_gt_1_div_4_l2_229


namespace marbles_problem_l2_211

theorem marbles_problem :
  let red_marbles := 20
  let green_marbles := 3 * red_marbles
  let yellow_marbles := 0.20 * green_marbles
  let total_marbles := green_marbles + 3 * green_marbles
  total_marbles - (red_marbles + green_marbles + yellow_marbles) = 148 := by
  sorry

end marbles_problem_l2_211


namespace abby_and_damon_weight_l2_298

variables {a b c d : ℝ}

theorem abby_and_damon_weight (h1 : a + b = 260) (h2 : b + c = 245) 
(h3 : c + d = 270) (h4 : a + c = 220) : a + d = 285 := 
by 
  sorry

end abby_and_damon_weight_l2_298


namespace ribbon_left_l2_242

theorem ribbon_left (gifts : ℕ) (ribbon_per_gift Tom_ribbon_total : ℝ) (h1 : gifts = 8) (h2 : ribbon_per_gift = 1.5) (h3 : Tom_ribbon_total = 15) : Tom_ribbon_total - (gifts * ribbon_per_gift) = 3 := 
by
  sorry

end ribbon_left_l2_242


namespace sum_sequence_l2_217

noncomputable def sum_first_n_minus_1_terms (n : ℕ) : ℕ :=
  (2^n - n - 1)

theorem sum_sequence (n : ℕ) : 
  sum_first_n_minus_1_terms n = (2^n - n - 1) :=
by
  sorry 

end sum_sequence_l2_217


namespace find_m_of_symmetry_l2_241

-- Define the conditions for the parabola and the axis of symmetry
theorem find_m_of_symmetry (m : ℝ) :
  let a := (1 : ℝ)
  let b := (m - 2 : ℝ)
  let axis_of_symmetry := (0 : ℝ)
  (-b / (2 * a)) = axis_of_symmetry → m = 2 :=
by
  sorry

end find_m_of_symmetry_l2_241


namespace cheaper_fluid_cost_is_20_l2_239

variable (x : ℕ) -- Denote the cost per drum of the cheaper fluid as x

-- Given conditions:
variable (total_drums : ℕ) (cheaper_drums : ℕ) (expensive_cost : ℕ) (total_cost : ℕ)
variable (remaining_drums : ℕ) (total_expensive_cost : ℕ)

axiom total_drums_eq : total_drums = 7
axiom cheaper_drums_eq : cheaper_drums = 5
axiom expensive_cost_eq : expensive_cost = 30
axiom total_cost_eq : total_cost = 160
axiom remaining_drums_eq : remaining_drums = total_drums - cheaper_drums
axiom total_expensive_cost_eq : total_expensive_cost = remaining_drums * expensive_cost

-- The equation for the total cost:
axiom total_cost_eq2 : total_cost = cheaper_drums * x + total_expensive_cost

-- The goal: Prove that the cheaper fluid cost per drum is $20
theorem cheaper_fluid_cost_is_20 : x = 20 :=
by
  sorry

end cheaper_fluid_cost_is_20_l2_239


namespace prove_seq_formula_l2_297

noncomputable def seq (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 1
| 1     => 5
| n + 2 => (2 * (seq a (n + 1))^2 - 3 * (seq a (n + 1)) - 9) / (2 * (seq a n))

theorem prove_seq_formula : ∀ (n : ℕ), seq a n = 2^(n + 2) - 3 :=
by
  sorry  -- Proof not needed for the mathematical translation

end prove_seq_formula_l2_297


namespace p_6_is_126_l2_215

noncomputable def p (x : ℝ) : ℝ := sorry

axiom h1 : p 1 = 1
axiom h2 : p 2 = 2
axiom h3 : p 3 = 3
axiom h4 : p 4 = 4
axiom h5 : p 5 = 5

theorem p_6_is_126 : p 6 = 126 := sorry

end p_6_is_126_l2_215


namespace original_rent_l2_250

theorem original_rent {avg_rent_before avg_rent_after : ℝ} (total_before total_after increase_percentage diff_increase : ℝ) :
  avg_rent_before = 800 → 
  avg_rent_after = 880 → 
  total_before = 4 * avg_rent_before → 
  total_after = 4 * avg_rent_after → 
  diff_increase = total_after - total_before → 
  increase_percentage = 0.20 → 
  diff_increase = increase_percentage * R → 
  R = 1600 :=
by sorry

end original_rent_l2_250


namespace three_times_two_to_the_n_minus_one_gt_n_squared_plus_three_l2_264

theorem three_times_two_to_the_n_minus_one_gt_n_squared_plus_three (n : ℕ) (h : n ≥ 4) : 3 * 2^(n-1) > n^2 + 3 := by
  sorry

end three_times_two_to_the_n_minus_one_gt_n_squared_plus_three_l2_264


namespace value_of_m_l2_275

theorem value_of_m (a b m : ℚ) (h1 : 2 * a = m) (h2 : 5 * b = m) (h3 : a + b = 2) : m = 20 / 7 :=
by
  sorry

end value_of_m_l2_275


namespace rectangle_area_l2_238

theorem rectangle_area :
  ∀ (width length : ℝ), (length = 3 * width) → (width = 5) → (length * width = 75) :=
by
  intros width length h1 h2
  rw [h2, h1]
  sorry

end rectangle_area_l2_238


namespace exists_negative_value_of_f_l2_294

noncomputable def f : ℝ → ℝ := sorry

axiom f_monotonic (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) : f x < f y
axiom f_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f ((2 * x * y) / (x + y)) ≥ (f x + f y) / 2

theorem exists_negative_value_of_f : ∃ x > 0, f x < 0 := 
sorry

end exists_negative_value_of_f_l2_294


namespace total_stickers_purchased_l2_295

-- Definitions for the number of sheets and stickers per sheet for each folder
def num_sheets_per_folder := 10
def stickers_per_sheet_red := 3
def stickers_per_sheet_green := 2
def stickers_per_sheet_blue := 1

-- Theorem stating that the total number of stickers is 60
theorem total_stickers_purchased : 
  num_sheets_per_folder * (stickers_per_sheet_red + stickers_per_sheet_green + stickers_per_sheet_blue) = 60 := 
  by
  -- Skipping the proof
  sorry

end total_stickers_purchased_l2_295


namespace f_odd_and_minimum_period_pi_l2_235

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x

theorem f_odd_and_minimum_period_pi :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + π) = f x) :=
  sorry

end f_odd_and_minimum_period_pi_l2_235


namespace total_girls_is_68_l2_203

-- Define the initial conditions
def track_length : ℕ := 100
def student_spacing : ℕ := 2
def girls_per_cycle : ℕ := 2
def cycle_length : ℕ := 3

-- Calculate the number of students on one side
def students_on_one_side : ℕ := track_length / student_spacing + 1

-- Number of cycles of three students
def num_cycles : ℕ := students_on_one_side / cycle_length

-- Number of girls on one side
def girls_on_one_side : ℕ := num_cycles * girls_per_cycle

-- Total number of girls on both sides
def total_girls : ℕ := girls_on_one_side * 2

theorem total_girls_is_68 : total_girls = 68 := by
  -- proof will be provided here
  sorry

end total_girls_is_68_l2_203


namespace value_of_expression_eq_33_l2_287

theorem value_of_expression_eq_33 : (3^2 + 7^2 - 5^2 = 33) := by
  sorry

end value_of_expression_eq_33_l2_287


namespace largest_angle_of_triangle_l2_261

theorem largest_angle_of_triangle (A B C : ℝ) :
  A + B + C = 180 ∧ A + B = 126 ∧ abs (A - B) = 45 → max A (max B C) = 85.5 :=
by sorry

end largest_angle_of_triangle_l2_261


namespace min_distance_squared_l2_296

noncomputable def graph_function1 (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

noncomputable def point_on_graph1 (a b : ℝ) : Prop := b = graph_function1 a

noncomputable def graph_function2 (x : ℝ) : ℝ := x + 2

noncomputable def point_on_graph2 (c d : ℝ) : Prop := d = graph_function2 c

theorem min_distance_squared (a b c d : ℝ) 
  (hP : point_on_graph1 a b)
  (hQ : point_on_graph2 c d) :
  (a - c)^2 + (b - d)^2 = 8 := 
sorry

end min_distance_squared_l2_296


namespace soldier_initial_consumption_l2_292

theorem soldier_initial_consumption :
  ∀ (s d1 n : ℕ) (c2 d2 : ℝ), 
    s = 1200 → d1 = 30 → n = 528 → c2 = 2.5 → d2 = 25 → 
    36000 * (x : ℝ) = 108000 → x = 3 := 
by {
  sorry
}

end soldier_initial_consumption_l2_292


namespace football_cost_correct_l2_258

variable (total_spent_on_toys : ℝ := 12.30)
variable (spent_on_marbles : ℝ := 6.59)

theorem football_cost_correct :
  (total_spent_on_toys - spent_on_marbles = 5.71) :=
by
  sorry

end football_cost_correct_l2_258


namespace average_speed_comparison_l2_259

variables (u v : ℝ) (hu : u > 0) (hv : v > 0)

theorem average_speed_comparison (x y : ℝ) 
  (hx : x = 2 * u * v / (u + v)) 
  (hy : y = (u + v) / 2) : x ≤ y := 
sorry

end average_speed_comparison_l2_259


namespace rational_non_positive_l2_245

variable (a : ℚ)

theorem rational_non_positive (h : ∃ a : ℚ, True) : 
  -a^2 ≤ 0 :=
by
  sorry

end rational_non_positive_l2_245


namespace max_dot_product_between_ellipses_l2_214

noncomputable def ellipse1 (x y : ℝ) : Prop := (x^2 / 25 + y^2 / 9 = 1)
noncomputable def ellipse2 (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 9 = 1)

theorem max_dot_product_between_ellipses :
  ∀ (M N : ℝ × ℝ),
    ellipse1 M.1 M.2 →
    ellipse2 N.1 N.2 →
    ∃ θ φ : ℝ,
      M = (5 * Real.cos θ, 3 * Real.sin θ) ∧
      N = (3 * Real.cos φ, 3 * Real.sin φ) ∧
      (15 * Real.cos θ * Real.cos φ + 9 * Real.sin θ * Real.sin φ ≤ 15) :=
by
  sorry

end max_dot_product_between_ellipses_l2_214


namespace intersection_proof_l2_200

-- Definitions based on conditions
def circle1 (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 10) ^ 2 = 50
def circle2 (x y : ℝ) : Prop := x ^ 2 + y ^ 2 + 2 * (x - y) - 18 = 0

-- Correct answer tuple
def intersection_points : (ℝ × ℝ) × (ℝ × ℝ) := ((3, 3), (-3, 5))

-- The goal statement to prove
theorem intersection_proof :
  (circle1 3 3 ∧ circle2 3 3) ∧ (circle1 (-3) 5 ∧ circle2 (-3) 5) :=
by
  sorry

end intersection_proof_l2_200


namespace not_all_angles_less_than_60_l2_274

-- Definitions relating to interior angles of a triangle
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

theorem not_all_angles_less_than_60 (α β γ : ℝ) 
(h_triangle : triangle α β γ) 
(h1 : α < 60) 
(h2 : β < 60) 
(h3 : γ < 60) : False :=
    -- The proof steps would be placed here
sorry

end not_all_angles_less_than_60_l2_274


namespace speed_in_first_hour_l2_231

variable (x : ℕ)
-- Conditions: 
-- The speed of the car in the second hour:
def speed_in_second_hour : ℕ := 30
-- The average speed of the car:
def average_speed : ℕ := 60
-- The total time traveled:
def total_time : ℕ := 2

-- Proof problem: Prove that the speed of the car in the first hour is 90 km/h.
theorem speed_in_first_hour : x + speed_in_second_hour = average_speed * total_time → x = 90 := 
by 
  intro h
  sorry

end speed_in_first_hour_l2_231


namespace number_of_students_with_at_least_two_pets_l2_286

-- Definitions for the sets of students
def total_students := 50
def dog_students := 35
def cat_students := 40
def rabbit_students := 10
def dog_and_cat_students := 20
def dog_and_rabbit_students := 5
def cat_and_rabbit_students := 0  -- Assuming minimal overlap

-- Problem Statement
theorem number_of_students_with_at_least_two_pets :
  (dog_and_cat_students + dog_and_rabbit_students + cat_and_rabbit_students) = 25 :=
by
  sorry

end number_of_students_with_at_least_two_pets_l2_286


namespace collinear_points_b_value_l2_227

theorem collinear_points_b_value :
  ∃ b : ℝ, (3 - (-2)) * (11 - b) = (8 - 3) * (1 - b) → b = -9 :=
by
  sorry

end collinear_points_b_value_l2_227


namespace parabola_line_slope_l2_268

theorem parabola_line_slope (y1 y2 x1 x2 : ℝ) (h1 : y1 ^ 2 = 6 * x1) (h2 : y2 ^ 2 = 6 * x2) 
    (midpoint_condition : (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 2) :
  (y1 - y2) / (x1 - x2) = 3 / 2 :=
by
  -- here will be the actual proof using the given hypothesis
  sorry

end parabola_line_slope_l2_268


namespace no_solution_value_of_m_l2_222

theorem no_solution_value_of_m (m : ℤ) : ¬ ∃ x : ℤ, x ≠ 3 ∧ (x - 5) * (x - 3) = (m * (x - 3) + 2 * (x - 3) * (x - 3)) → m = -2 :=
by
  sorry

end no_solution_value_of_m_l2_222


namespace problem_statement_l2_271

theorem problem_statement : 
  (777 % 4 = 1) ∧ 
  (555 % 4 = 3) ∧ 
  (999 % 4 = 3) → 
  ( (999^2021 * 555^2021 - 1) % 4 = 0 ∧ 
    (777^2021 * 999^2021 - 1) % 4 ≠ 0 ∧ 
    (555^2021 * 777^2021 - 1) % 4 ≠ 0 ) := 
by {
  sorry
}

end problem_statement_l2_271


namespace point_on_inverse_proportion_function_l2_237

theorem point_on_inverse_proportion_function :
  ∀ (x y k : ℝ), k ≠ 0 ∧ y = k / x ∧ (2, -3) = (2, -(3 : ℝ)) → (x, y) = (-2, 3) → (y = -6 / x) :=
sorry

end point_on_inverse_proportion_function_l2_237


namespace arthur_initial_amount_l2_240

def initial_amount (X : ℝ) : Prop :=
  (1/5) * X = 40

theorem arthur_initial_amount (X : ℝ) (h : initial_amount X) : X = 200 :=
by
  sorry

end arthur_initial_amount_l2_240


namespace six_digit_start_5_not_possible_l2_253

theorem six_digit_start_5_not_possible :
  ∀ n : ℕ, (n ≥ 500000 ∧ n < 600000) → (¬ ∃ m : ℕ, (n * 10^6 + m) ^ 2 < 10^12 ∧ (n * 10^6 + m) ^ 2 ≥ 5 * 10^11 ∧ (n * 10^6 + m) ^ 2 < 6 * 10^11) :=
sorry

end six_digit_start_5_not_possible_l2_253
