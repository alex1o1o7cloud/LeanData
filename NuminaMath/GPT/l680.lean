import Mathlib

namespace NUMINAMATH_GPT_find_a_l680_68024

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + a - 1 = 0}

theorem find_a (a : ℝ) (h : A ∪ B a = A) : a = 2 ∨ a = 3 := by
  sorry

end NUMINAMATH_GPT_find_a_l680_68024


namespace NUMINAMATH_GPT_pens_given_away_l680_68097

theorem pens_given_away (initial_pens : ℕ) (pens_left : ℕ) (n : ℕ) (h1 : initial_pens = 56) (h2 : pens_left = 34) (h3 : n = initial_pens - pens_left) : n = 22 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_pens_given_away_l680_68097


namespace NUMINAMATH_GPT_correct_calculation_l680_68023

-- Define the variables used in the problem
variables (a x y : ℝ)

-- The main theorem statement
theorem correct_calculation : (2 * x * y^2 - x * y^2 = x * y^2) :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l680_68023


namespace NUMINAMATH_GPT_color_circles_with_four_colors_l680_68096

theorem color_circles_with_four_colors (n : ℕ) (circles : Fin n → (ℝ × ℝ)) (radius : ℝ):
  (∀ i j, i ≠ j → dist (circles i) (circles j) ≥ 2 * radius) →
  ∃ f : Fin n → Fin 4, ∀ i j, dist (circles i) (circles j) < 2 * radius → f i ≠ f j :=
by
  sorry

end NUMINAMATH_GPT_color_circles_with_four_colors_l680_68096


namespace NUMINAMATH_GPT_expand_product_l680_68098

theorem expand_product (x : ℝ) : 4 * (x + 3) * (2 * x + 7) = 8 * x ^ 2 + 52 * x + 84 := by
  sorry

end NUMINAMATH_GPT_expand_product_l680_68098


namespace NUMINAMATH_GPT_Amelia_weekly_sales_l680_68031

-- Conditions
def monday_sales : ℕ := 45
def tuesday_sales : ℕ := 45 - 16
def remaining_sales : ℕ := 16

-- Question to Answer
def total_weekly_sales : ℕ := 90

-- Lean 4 Statement to Prove
theorem Amelia_weekly_sales : monday_sales + tuesday_sales + remaining_sales = total_weekly_sales :=
by
  sorry

end NUMINAMATH_GPT_Amelia_weekly_sales_l680_68031


namespace NUMINAMATH_GPT_compound_interest_at_least_double_l680_68075

theorem compound_interest_at_least_double :
  ∀ t : ℕ, (0 < t) → (1.05 : ℝ)^t > 2 ↔ t ≥ 15 :=
by sorry

end NUMINAMATH_GPT_compound_interest_at_least_double_l680_68075


namespace NUMINAMATH_GPT_residue_of_neg_1237_mod_29_l680_68061

theorem residue_of_neg_1237_mod_29 :
  (-1237 : ℤ) % 29 = 10 :=
sorry

end NUMINAMATH_GPT_residue_of_neg_1237_mod_29_l680_68061


namespace NUMINAMATH_GPT_domain_of_function_l680_68083

def domain_of_f (x: ℝ) : Prop :=
x >= -1 ∧ x <= 48

theorem domain_of_function :
  ∀ x, (x + 1 >= 0 ∧ 7 - Real.sqrt (x + 1) >= 0 ∧ 4 - Real.sqrt (7 - Real.sqrt (x + 1)) >= 0)
  ↔ domain_of_f x := by
  sorry

end NUMINAMATH_GPT_domain_of_function_l680_68083


namespace NUMINAMATH_GPT_point_on_x_axis_l680_68081

theorem point_on_x_axis (a : ℝ) (h₁ : 1 - a = 0) : (3 * a - 6, 1 - a) = (-3, 0) :=
by
  sorry

end NUMINAMATH_GPT_point_on_x_axis_l680_68081


namespace NUMINAMATH_GPT_sin_B_value_cos_A_value_l680_68043

theorem sin_B_value (A B C S : Real)
  (h1: ∃ (a b c : Real), 
    (a * c * Real.cos (π - B) = (3/2) * (1/2) * a * c * Real.sin B) ∧ 
    (S = (1/2) * a * c * Real.sin B)) : 
  Real.sin B = 4/5 :=
sorry

theorem cos_A_value (A B C : Real)
  (h1: ∃ (a b c : Real), 
    (a * c * Real.cos (π - B) = (3/2) * (1/2) * a * c * Real.sin B) ∧ 
    (S = (1/2) * a * c * Real.sin B)) 
  (h2: A - C = π/4)
  (h3: Real.sin B = 4/5) 
  (h4: Real.cos B = -3/5): 
  Real.cos A = Real.sqrt (50 + 5 * Real.sqrt 2) / 10 :=
sorry

end NUMINAMATH_GPT_sin_B_value_cos_A_value_l680_68043


namespace NUMINAMATH_GPT_transformed_roots_equation_l680_68032

theorem transformed_roots_equation (α β : ℂ) (h1 : 3 * α^2 + 2 * α + 1 = 0) (h2 : 3 * β^2 + 2 * β + 1 = 0) :
  ∃ (y : ℂ), (y - (3 * α + 2)) * (y - (3 * β + 2)) = y^2 + 4 := 
sorry

end NUMINAMATH_GPT_transformed_roots_equation_l680_68032


namespace NUMINAMATH_GPT_boat_distance_against_stream_l680_68020

theorem boat_distance_against_stream 
  (v_b : ℝ)
  (v_s : ℝ)
  (distance_downstream : ℝ)
  (t : ℝ)
  (speed_downstream : v_s + v_b = 11)
  (speed_still_water : v_b = 8)
  (time : t = 1) :
  (v_b - (11 - v_b)) * t = 5 :=
by
  -- Here we're given the initial conditions and have to show the final distance against the stream is 5 km
  sorry

end NUMINAMATH_GPT_boat_distance_against_stream_l680_68020


namespace NUMINAMATH_GPT_part_one_part_two_l680_68021

variable {x : ℝ} {m : ℝ}

-- Question 1
theorem part_one (h : ∀ x : ℝ, mx^2 - mx - 1 < 0) : -4 < m ∧ m <= 0 :=
sorry

-- Question 2
theorem part_two (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → mx^2 - mx - 1 > -m + x - 1) : m > 1 :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l680_68021


namespace NUMINAMATH_GPT_rectangular_prism_edge_sum_l680_68006

theorem rectangular_prism_edge_sum
  (V A : ℝ)
  (hV : V = 8)
  (hA : A = 32)
  (l w h : ℝ)
  (geom_prog : l = w / h ∧ w = l * h ∧ h = l * (w / l)) :
  4 * (l + w + h) = 28 :=
by 
  sorry

end NUMINAMATH_GPT_rectangular_prism_edge_sum_l680_68006


namespace NUMINAMATH_GPT_sidney_cats_l680_68099

theorem sidney_cats (A : ℕ) :
  (4 * 7 * (3 / 4) + A * 7 = 42) →
  A = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sidney_cats_l680_68099


namespace NUMINAMATH_GPT_point_in_third_quadrant_coordinates_l680_68002

theorem point_in_third_quadrant_coordinates :
  ∀ (P : ℝ × ℝ), (P.1 < 0) ∧ (P.2 < 0) ∧ (|P.2| = 2) ∧ (|P.1| = 3) -> P = (-3, -2) :=
by
  intros P h
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_coordinates_l680_68002


namespace NUMINAMATH_GPT_pencil_count_l680_68012

theorem pencil_count (P N X : ℝ) 
  (h1 : 96 * P + 24 * N = 520) 
  (h2 : X * P + 4 * N = 60) 
  (h3 : P + N = 15.512820512820513) :
  X = 3 :=
by
  sorry

end NUMINAMATH_GPT_pencil_count_l680_68012


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l680_68078

theorem ratio_of_x_to_y (x y : ℤ) (h : (7 * x - 4 * y) * 9 = (20 * x - 3 * y) * 4) : x * 17 = y * -24 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_x_to_y_l680_68078


namespace NUMINAMATH_GPT_total_nephews_correct_l680_68025

def alden_nephews_10_years_ago : ℕ := 50

def alden_nephews_now : ℕ :=
  alden_nephews_10_years_ago * 2

def vihaan_nephews_now : ℕ :=
  alden_nephews_now + 60

def total_nephews : ℕ :=
  alden_nephews_now + vihaan_nephews_now

theorem total_nephews_correct : total_nephews = 260 := by
  sorry

end NUMINAMATH_GPT_total_nephews_correct_l680_68025


namespace NUMINAMATH_GPT_original_number_eq_nine_l680_68038

theorem original_number_eq_nine (N : ℕ) (h1 : ∃ k : ℤ, N - 4 = 5 * k) : N = 9 :=
sorry

end NUMINAMATH_GPT_original_number_eq_nine_l680_68038


namespace NUMINAMATH_GPT_rectangle_horizontal_length_l680_68077

theorem rectangle_horizontal_length (s v : ℕ) (h : ℕ) 
  (hs : s = 80) (hv : v = 100) 
  (eq_perimeters : 4 * s = 2 * (v + h)) : h = 60 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_horizontal_length_l680_68077


namespace NUMINAMATH_GPT_max_marks_paper_I_l680_68064

variable (M : ℝ)

theorem max_marks_paper_I (h1 : 0.65 * M = 112 + 58) : M = 262 :=
  sorry

end NUMINAMATH_GPT_max_marks_paper_I_l680_68064


namespace NUMINAMATH_GPT_map_scale_l680_68016

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end NUMINAMATH_GPT_map_scale_l680_68016


namespace NUMINAMATH_GPT_edward_friend_scores_l680_68014

theorem edward_friend_scores (total_points friend_points edward_points : ℕ) (h1 : total_points = 13) (h2 : edward_points = 7) (h3 : friend_points = total_points - edward_points) : friend_points = 6 := 
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_edward_friend_scores_l680_68014


namespace NUMINAMATH_GPT_bar_graph_represents_circle_graph_l680_68047

theorem bar_graph_represents_circle_graph (r b g : ℕ) 
  (h1 : r = g) 
  (h2 : b = 3 * r) : 
  (r = 1 ∧ b = 3 ∧ g = 1) :=
sorry

end NUMINAMATH_GPT_bar_graph_represents_circle_graph_l680_68047


namespace NUMINAMATH_GPT_find_operation_l680_68074

theorem find_operation (a b : Int) (h : a + b = 0) : (7 + (-7) = 0) := 
by
  sorry

end NUMINAMATH_GPT_find_operation_l680_68074


namespace NUMINAMATH_GPT_sum_of_roots_eq_h_over_4_l680_68027

theorem sum_of_roots_eq_h_over_4 (x1 x2 h b : ℝ) (h_ne : x1 ≠ x2)
  (hx1 : 4 * x1 ^ 2 - h * x1 = b) (hx2 : 4 * x2 ^ 2 - h * x2 = b) : x1 + x2 = h / 4 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_eq_h_over_4_l680_68027


namespace NUMINAMATH_GPT_min_distance_from_start_after_9_minutes_l680_68049

noncomputable def robot_min_distance : ℝ :=
  let movement_per_minute := 10
  sorry

theorem min_distance_from_start_after_9_minutes :
  robot_min_distance = 10 :=
sorry

end NUMINAMATH_GPT_min_distance_from_start_after_9_minutes_l680_68049


namespace NUMINAMATH_GPT_combine_polynomials_find_value_profit_or_loss_l680_68051

-- Problem 1, Part ①
theorem combine_polynomials (a b : ℝ) : -3 * (a+b)^2 - 6 * (a+b)^2 + 8 * (a+b)^2 = -(a+b)^2 := 
sorry

-- Problem 1, Part ②
theorem find_value (a b c d : ℝ) (h1 : a - 2 * b = 5) (h2 : 2 * b - c = -7) (h3 : c - d = 12) : 
  4 * (a - c) + 4 * (2 * b - d) - 4 * (2 * b - c) = 40 := 
sorry

-- Problem 2
theorem profit_or_loss (initial_cost : ℝ) (selling_prices : ℕ → ℝ) (base_price : ℝ) 
  (h_prices : selling_prices 0 = -3) (h_prices1 : selling_prices 1 = 7) 
  (h_prices2 : selling_prices 2 = -8) (h_prices3 : selling_prices 3 = 9) 
  (h_prices4 : selling_prices 4 = -2) (h_prices5 : selling_prices 5 = 0) 
  (h_prices6 : selling_prices 6 = -1) (h_prices7 : selling_prices 7 = -6) 
  (h_initial_cost : initial_cost = 400) (h_base_price : base_price = 56) : 
  (selling_prices 0 + selling_prices 1 + selling_prices 2 + selling_prices 3 + selling_prices 4 + selling_prices 5 + 
  selling_prices 6 + selling_prices 7 + 8 * base_price) - initial_cost > 0 := 
sorry

end NUMINAMATH_GPT_combine_polynomials_find_value_profit_or_loss_l680_68051


namespace NUMINAMATH_GPT_B_gain_l680_68041

-- Problem statement and conditions
def principalA : ℝ := 3500
def rateA : ℝ := 0.10
def periodA : ℕ := 2
def principalB : ℝ := 3500
def rateB : ℝ := 0.14
def periodB : ℕ := 3

-- Calculate amount A will receive from B after 2 years
noncomputable def amountA := principalA * (1 + rateA / 1) ^ periodA

-- Calculate amount B will receive from C after 3 years
noncomputable def amountB := principalB * (1 + rateB / 2) ^ (2 * periodB)

-- Calculate B's gain
noncomputable def gainB := amountB - amountA

-- The theorem to prove
theorem B_gain : gainB = 1019.20 := by
  sorry

end NUMINAMATH_GPT_B_gain_l680_68041


namespace NUMINAMATH_GPT_more_apples_than_pears_l680_68067

-- Define the variables
def apples := 17
def pears := 9

-- Theorem: The number of apples minus the number of pears equals 8
theorem more_apples_than_pears : apples - pears = 8 :=
by
  sorry

end NUMINAMATH_GPT_more_apples_than_pears_l680_68067


namespace NUMINAMATH_GPT_mouse_jumps_28_inches_further_than_grasshopper_l680_68048

theorem mouse_jumps_28_inches_further_than_grasshopper :
  let g_initial := 19
  let g_obstacle := 3
  let g_actual := g_initial - g_obstacle
  let f_difference := 10
  let f_actual := g_initial + f_difference
  let m_difference := 20
  let m_obstacle := 5
  let m_actual := f_actual + m_difference - m_obstacle
  let g_to_m_difference := m_actual - g_actual
  g_to_m_difference = 28 :=
by
  let g_initial := 19
  let g_obstacle := 3
  let g_actual := g_initial - g_obstacle
  let f_difference := 10
  let f_actual := g_initial + f_difference
  let m_difference := 20
  let m_obstacle := 5
  let m_actual := f_actual + m_difference - m_obstacle
  let g_to_m_difference := m_actual - g_actual
  show g_to_m_difference = 28
  sorry

end NUMINAMATH_GPT_mouse_jumps_28_inches_further_than_grasshopper_l680_68048


namespace NUMINAMATH_GPT_find_n_l680_68003

-- Definitions of the problem conditions
def sum_coefficients (n : ℕ) : ℕ := 4^n
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

-- The main theorem to be proved
theorem find_n (n : ℕ) (P S : ℕ) (hP : P = sum_coefficients n) (hS : S = sum_binomial_coefficients n) (h : P + S = 272) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l680_68003


namespace NUMINAMATH_GPT_area_of_rectangle_l680_68015

theorem area_of_rectangle (S R L B A : ℝ)
  (h1 : L = (2 / 5) * R)
  (h2 : R = S)
  (h3 : S^2 = 1600)
  (h4 : B = 10)
  (h5 : A = L * B) : 
  A = 160 := 
sorry

end NUMINAMATH_GPT_area_of_rectangle_l680_68015


namespace NUMINAMATH_GPT_weight_of_new_person_l680_68013

theorem weight_of_new_person (avg_increase : ℝ) (num_persons : ℕ) (old_weight new_weight : ℝ) 
  (h_avg_increase : avg_increase = 1.5) (h_num_persons : num_persons = 9) (h_old_weight : old_weight = 65) 
  (h_new_weight_increase : new_weight = old_weight + num_persons * avg_increase) : 
  new_weight = 78.5 :=
sorry

end NUMINAMATH_GPT_weight_of_new_person_l680_68013


namespace NUMINAMATH_GPT_Sheila_attends_picnic_probability_l680_68093

theorem Sheila_attends_picnic_probability :
  let P_rain := 0.5
  let P_no_rain := 0.5
  let P_Sheila_goes_if_rain := 0.3
  let P_Sheila_goes_if_no_rain := 0.7
  let P_friend_agrees := 0.5
  (P_rain * P_Sheila_goes_if_rain + P_no_rain * P_Sheila_goes_if_no_rain) * P_friend_agrees = 0.25 := 
by
  sorry

end NUMINAMATH_GPT_Sheila_attends_picnic_probability_l680_68093


namespace NUMINAMATH_GPT_james_coffee_weekdays_l680_68087

theorem james_coffee_weekdays :
  ∃ (c d : ℕ) (k : ℤ), (c + d = 5) ∧ 
                      (3 * c + 2 * d + 10 = k / 3) ∧ 
                      (k % 3 = 0) ∧ 
                      c = 2 :=
by 
  sorry

end NUMINAMATH_GPT_james_coffee_weekdays_l680_68087


namespace NUMINAMATH_GPT_units_digit_sum_42_4_24_4_l680_68033

theorem units_digit_sum_42_4_24_4 : (42^4 + 24^4) % 10 = 2 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_sum_42_4_24_4_l680_68033


namespace NUMINAMATH_GPT_gcd_n_cube_plus_16_n_plus_4_l680_68035

theorem gcd_n_cube_plus_16_n_plus_4 (n : ℕ) (h1 : n > 16) : 
  Nat.gcd (n^3 + 16) (n + 4) = Nat.gcd 48 (n + 4) :=
by
  sorry

end NUMINAMATH_GPT_gcd_n_cube_plus_16_n_plus_4_l680_68035


namespace NUMINAMATH_GPT_correct_student_answer_l680_68057

theorem correct_student_answer :
  (9 - (3^2) / 8 = 9 - (9 / 8)) ∧
  (24 - (4 * (3^2)) = 24 - 36) ∧
  ((36 - 12) / (3 / 2) = 24 * (2 / 3)) ∧
  ((-3)^2 / (1 / 3) * 3 = 9 * 3 * 3) →
  (24 * (2 / 3) = 16) :=
by
  sorry

end NUMINAMATH_GPT_correct_student_answer_l680_68057


namespace NUMINAMATH_GPT_factorize_x_pow_m_minus_x_pow_m_minus_2_l680_68070

theorem factorize_x_pow_m_minus_x_pow_m_minus_2 (x : ℝ) (m : ℕ) (h : m > 1) : 
  x ^ m - x ^ (m - 2) = (x ^ (m - 2)) * (x + 1) * (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x_pow_m_minus_x_pow_m_minus_2_l680_68070


namespace NUMINAMATH_GPT_equation_true_when_n_eq_2_l680_68062

theorem equation_true_when_n_eq_2 : (2 ^ (2 / 2)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_equation_true_when_n_eq_2_l680_68062


namespace NUMINAMATH_GPT_no_such_integers_l680_68019

theorem no_such_integers (x y z : ℤ) : ¬ ((x - y)^3 + (y - z)^3 + (z - x)^3 = 2011) :=
sorry

end NUMINAMATH_GPT_no_such_integers_l680_68019


namespace NUMINAMATH_GPT_calculate_expression_l680_68039

theorem calculate_expression : 6 * (8 + 1/3) = 50 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l680_68039


namespace NUMINAMATH_GPT_employee_b_pay_l680_68085

theorem employee_b_pay (total_pay : ℝ) (ratio_ab : ℝ) (pay_b : ℝ) 
  (h1 : total_pay = 570)
  (h2 : ratio_ab = 1.5 * pay_b)
  (h3 : total_pay = ratio_ab + pay_b) :
  pay_b = 228 := 
sorry

end NUMINAMATH_GPT_employee_b_pay_l680_68085


namespace NUMINAMATH_GPT_part_I_part_II_part_III_l680_68011

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem part_I : 
  ∀ x:ℝ, f x = x^3 - x :=
by sorry

theorem part_II : 
  ∃ (x1 x2 : ℝ), x1 ∈ Set.Icc (-1:ℝ) 1 ∧ x2 ∈ Set.Icc (-1:ℝ) 1 ∧ (3 * x1^2 - 1) * (3 * x2^2 - 1) = -1 :=
by sorry

theorem part_III (x_n y_m : ℝ) (hx : x_n ∈ Set.Icc (-1:ℝ) 1) (hy : y_m ∈ Set.Icc (-1:ℝ) 1) : 
  |f x_n - f y_m| < 1 :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_part_III_l680_68011


namespace NUMINAMATH_GPT_Quincy_sold_more_l680_68060

def ThorSales : ℕ := 200 / 10
def JakeSales : ℕ := ThorSales + 10
def QuincySales : ℕ := 200

theorem Quincy_sold_more (H : QuincySales = 200) : QuincySales - JakeSales = 170 := by
  sorry

end NUMINAMATH_GPT_Quincy_sold_more_l680_68060


namespace NUMINAMATH_GPT_parallelepiped_side_lengths_l680_68034

theorem parallelepiped_side_lengths (x y z : ℕ) 
  (h1 : x + y + z = 17) 
  (h2 : 2 * x * y + 2 * y * z + 2 * z * x = 180) 
  (h3 : x^2 + y^2 = 100) :
  x = 8 ∧ y = 6 ∧ z = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_parallelepiped_side_lengths_l680_68034


namespace NUMINAMATH_GPT_evaluate_f_diff_l680_68071

def f (x : ℝ) : ℝ := x^4 + 3 * x^3 + 2 * x^2 + 7 * x

theorem evaluate_f_diff:
  f 6 - f (-6) = 1380 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_diff_l680_68071


namespace NUMINAMATH_GPT_find_difference_l680_68092

-- Define the necessary constants and variables
variables (u v : ℝ)

-- Define the conditions
def condition1 := u + v = 360
def condition2 := u = (1/1.1) * v

-- Define the theorem to prove
theorem find_difference (h1 : condition1 u v) (h2 : condition2 u v) : v - u = 17 := 
sorry

end NUMINAMATH_GPT_find_difference_l680_68092


namespace NUMINAMATH_GPT_trucks_have_160_containers_per_truck_l680_68080

noncomputable def containers_per_truck: ℕ :=
  let boxes1 := 7 * 20
  let boxes2 := 5 * 12
  let total_boxes := boxes1 + boxes2
  let total_containers := total_boxes * 8
  let trucks := 10
  total_containers / trucks

theorem trucks_have_160_containers_per_truck:
  containers_per_truck = 160 :=
by
  sorry

end NUMINAMATH_GPT_trucks_have_160_containers_per_truck_l680_68080


namespace NUMINAMATH_GPT_even_factors_count_of_n_l680_68056

def n : ℕ := 2^3 * 3^2 * 7 * 5

theorem even_factors_count_of_n : ∃ k : ℕ, k = 36 ∧ ∀ (a b c d : ℕ), 
  1 ≤ a ∧ a ≤ 3 →
  b ≤ 2 →
  c ≤ 1 →
  d ≤ 1 →
  2^a * 3^b * 7^c * 5^d ∣ n :=
sorry

end NUMINAMATH_GPT_even_factors_count_of_n_l680_68056


namespace NUMINAMATH_GPT_find_original_speed_l680_68000

theorem find_original_speed (r : ℝ) (t : ℝ)
  (h_circumference : r * t = 15 / 5280)
  (h_increase : (r + 8) * (t - 1/10800) = 15 / 5280) :
  r = 7.5 :=
sorry

end NUMINAMATH_GPT_find_original_speed_l680_68000


namespace NUMINAMATH_GPT_train_crossing_time_l680_68040

/-!
## Problem Statement
A train 400 m in length crosses a telegraph post. The speed of the train is 90 km/h. Prove that it takes 16 seconds for the train to cross the telegraph post.
-/

-- Defining the given definitions based on the conditions in a)
def train_length : ℕ := 400
def train_speed_kmh : ℕ := 90
def train_speed_ms : ℚ := 25 -- Converting 90 km/h to 25 m/s

-- Proving the problem statement
theorem train_crossing_time : train_length / train_speed_ms = 16 := 
by
  -- convert conditions and show expected result
  sorry

end NUMINAMATH_GPT_train_crossing_time_l680_68040


namespace NUMINAMATH_GPT_gcd_sequence_l680_68082

theorem gcd_sequence (n : ℕ) : gcd ((7^n - 1)/6) ((7^(n+1) - 1)/6) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_sequence_l680_68082


namespace NUMINAMATH_GPT_intersection_A_B_l680_68089

def A : Set ℝ := { x | 2 * x^2 - 5 * x < 0 }
def B : Set ℝ := { x | 3^(x - 1) ≥ Real.sqrt 3 }

theorem intersection_A_B : A ∩ B = Set.Ico (3 / 2) (5 / 2) := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l680_68089


namespace NUMINAMATH_GPT_mary_needs_6_cups_of_flour_l680_68084

-- Define the necessary constants according to the conditions.
def flour_needed : ℕ := 6
def sugar_needed : ℕ := 13
def flour_more_than_sugar : ℕ := 8

-- Define the number of cups of flour Mary needs to add.
def flour_to_add (flour_put_in : ℕ) : ℕ := flour_needed - flour_put_in

-- Prove that Mary needs to add 6 more cups of flour.
theorem mary_needs_6_cups_of_flour (flour_put_in : ℕ) (h : flour_more_than_sugar = 8): flour_to_add flour_put_in = 6 :=
by {
  sorry -- the proof is omitted.
}

end NUMINAMATH_GPT_mary_needs_6_cups_of_flour_l680_68084


namespace NUMINAMATH_GPT_single_elimination_tournament_l680_68073

theorem single_elimination_tournament (teams : ℕ) (prelim_games : ℕ) (post_prelim_teams : ℕ) :
  teams = 24 →
  prelim_games = 4 →
  post_prelim_teams = teams - prelim_games →
  post_prelim_teams - 1 + prelim_games = 23 :=
by
  intros
  sorry

end NUMINAMATH_GPT_single_elimination_tournament_l680_68073


namespace NUMINAMATH_GPT_coefficient_x3_expansion_l680_68094

open Finset -- To use binomial coefficients and summation

theorem coefficient_x3_expansion (x : ℝ) : 
  (2 + x) ^ 3 = 8 + 12 * x + 6 * x^2 + 1 * x^3 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_x3_expansion_l680_68094


namespace NUMINAMATH_GPT_least_possible_value_expression_l680_68009

theorem least_possible_value_expression :
  ∃ min_value : ℝ, ∀ x : ℝ, ((x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019) ≥ min_value ∧ min_value = 2018 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_value_expression_l680_68009


namespace NUMINAMATH_GPT_min_value_reciprocals_l680_68017

open Real

theorem min_value_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + b = 1) :
  (1 / a + 1 / b) = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_reciprocals_l680_68017


namespace NUMINAMATH_GPT_find_number_l680_68076

theorem find_number (x : ℕ) (h : x + 20 + x + 30 + x + 40 + x + 10 = 4100) : x = 1000 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l680_68076


namespace NUMINAMATH_GPT_solve_for_b_l680_68022

theorem solve_for_b (a b c m : ℚ) (h : m = c * a * b / (a - b)) : b = (m * a) / (m + c * a) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_b_l680_68022


namespace NUMINAMATH_GPT_percentage_democrats_l680_68068

/-- In a certain city, some percent of the registered voters are Democrats and the rest are Republicans. In a mayoral race, 85 percent of the registered voters who are Democrats and 20 percent of the registered voters who are Republicans are expected to vote for candidate A. Candidate A is expected to get 59 percent of the registered voters' votes. Prove that 60 percent of the registered voters are Democrats. -/
theorem percentage_democrats (D R : ℝ) (h : D + R = 100) (h1 : 0.85 * D + 0.20 * R = 59) : 
  D = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_democrats_l680_68068


namespace NUMINAMATH_GPT_circle_radius_proof_l680_68091

def circle_radius : Prop :=
  let D := -2
  let E := 3
  let F := -3 / 4
  let r := 1 / 2 * Real.sqrt (D^2 + E^2 - 4 * F)
  r = 2

theorem circle_radius_proof : circle_radius :=
  sorry

end NUMINAMATH_GPT_circle_radius_proof_l680_68091


namespace NUMINAMATH_GPT_greatest_difference_54_l680_68004

theorem greatest_difference_54 (board : ℕ → ℕ → ℕ) (h : ∀ i j, 1 ≤ board i j ∧ board i j ≤ 100) :
  ∃ i j k l, (i = k ∨ j = l) ∧ (board i j - board k l ≥ 54 ∨ board k l - board i j ≥ 54) :=
sorry

end NUMINAMATH_GPT_greatest_difference_54_l680_68004


namespace NUMINAMATH_GPT_simplify_fraction_l680_68029

theorem simplify_fraction (a : ℝ) (h : a = 2) : (24 * a^5) / (72 * a^3) = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l680_68029


namespace NUMINAMATH_GPT_volume_of_pyramid_base_isosceles_right_triangle_l680_68046

theorem volume_of_pyramid_base_isosceles_right_triangle (a h : ℝ) (ha : a = 3) (hh : h = 4) :
  (1 / 3) * (1 / 2) * a * a * h = 6 := by
  sorry

end NUMINAMATH_GPT_volume_of_pyramid_base_isosceles_right_triangle_l680_68046


namespace NUMINAMATH_GPT_phoenix_hike_length_l680_68090

theorem phoenix_hike_length (a b c d : ℕ)
  (h1 : a + b = 22)
  (h2 : b + c = 26)
  (h3 : c + d = 30)
  (h4 : a + c = 26) :
  a + b + c + d = 52 :=
sorry

end NUMINAMATH_GPT_phoenix_hike_length_l680_68090


namespace NUMINAMATH_GPT_isosceles_base_angle_eq_43_l680_68072

theorem isosceles_base_angle_eq_43 (α β : ℝ) (h_iso : α = β) (h_sum : α + β + 94 = 180) : α = 43 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_base_angle_eq_43_l680_68072


namespace NUMINAMATH_GPT_grantRooms_is_2_l680_68059

/-- Danielle's apartment has 6 rooms. -/
def danielleRooms : ℕ := 6

/-- Heidi's apartment has 3 times as many rooms as Danielle's apartment. -/
def heidiRooms : ℕ := 3 * danielleRooms

/-- Grant's apartment has 1/9 as many rooms as Heidi's apartment. -/
def grantRooms : ℕ := heidiRooms / 9

/-- Prove that Grant's apartment has 2 rooms. -/
theorem grantRooms_is_2 : grantRooms = 2 := by
  sorry

end NUMINAMATH_GPT_grantRooms_is_2_l680_68059


namespace NUMINAMATH_GPT_books_sold_l680_68069

theorem books_sold (initial_books left_books sold_books : ℕ) (h1 : initial_books = 108) (h2 : left_books = 66) : sold_books = 42 :=
by
  have : sold_books = initial_books - left_books := sorry
  rw [h1, h2] at this
  exact this

end NUMINAMATH_GPT_books_sold_l680_68069


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l680_68079

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (h1 : a 1 * a 3 = 36)
  (h2 : a 4 = 54)
  (h_pos : ∀ n, a n > 0) :
  ∃ q, q > 0 ∧ ∀ n, a n = a 1 * q ^ (n - 1) ∧ q = 3 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l680_68079


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l680_68010

theorem area_of_triangle_ABC
  {A B C : Type*} 
  (AC BC : ℝ)
  (B : ℝ)
  (h1 : AC = Real.sqrt (13))
  (h2 : BC = 1)
  (h3 : B = Real.sqrt 3 / 2): 
  ∃ area : ℝ, area = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l680_68010


namespace NUMINAMATH_GPT_bamboo_fifth_section_volume_l680_68050

theorem bamboo_fifth_section_volume
  (a₁ q : ℝ)
  (h1 : a₁ * (a₁ * q) * (a₁ * q^2) = 3)
  (h2 : (a₁ * q^6) * (a₁ * q^7) * (a₁ * q^8) = 9) :
  a₁ * q^4 = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_bamboo_fifth_section_volume_l680_68050


namespace NUMINAMATH_GPT_solution_of_inequality_l680_68086

noncomputable def solutionSet (a x : ℝ) : Set ℝ :=
  if a > 0 then {x | -a < x ∧ x < 3 * a}
  else if a < 0 then {x | 3 * a < x ∧ x < -a}
  else ∅

theorem solution_of_inequality (a x : ℝ) :
  (x^2 - 2 * a * x - 3 * a^2 < 0 ↔ x ∈ solutionSet a x) :=
sorry

end NUMINAMATH_GPT_solution_of_inequality_l680_68086


namespace NUMINAMATH_GPT_eight_p_plus_one_composite_l680_68028

theorem eight_p_plus_one_composite 
  (p : ℕ) 
  (hp : Nat.Prime p) 
  (h8p_minus_one : Nat.Prime (8 * p - 1))
  : ¬ (Nat.Prime (8 * p + 1)) :=
sorry

end NUMINAMATH_GPT_eight_p_plus_one_composite_l680_68028


namespace NUMINAMATH_GPT_sum_diff_square_cube_l680_68044

/-- If the sum of two numbers is 25 and the difference between them is 15,
    then the difference between the square of the larger number and the cube of the smaller number is 275. -/
theorem sum_diff_square_cube (x y : ℝ) 
  (h1 : x + y = 25)
  (h2 : x - y = 15) :
  x^2 - y^3 = 275 :=
sorry

end NUMINAMATH_GPT_sum_diff_square_cube_l680_68044


namespace NUMINAMATH_GPT_initial_men_count_l680_68005

theorem initial_men_count (M : ℕ) (h1 : ∃ F : ℕ, F = M * 22) (h2 : ∃ F_remaining : ℕ, F_remaining = M * 20) (h3 : ∃ F_remaining_2 : ℕ, F_remaining_2 = (M + 1140) * 8) : 
  M = 760 := 
by
  -- Code to prove the theorem goes here.
  sorry

end NUMINAMATH_GPT_initial_men_count_l680_68005


namespace NUMINAMATH_GPT_train_length_360_l680_68054

variable (time_to_cross : ℝ) (speed_of_train : ℝ)

theorem train_length_360 (h1 : time_to_cross = 12) (h2 : speed_of_train = 30) :
  speed_of_train * time_to_cross = 360 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_train_length_360_l680_68054


namespace NUMINAMATH_GPT_carnival_tickets_l680_68066

theorem carnival_tickets (x : ℕ) (won_tickets : ℕ) (found_tickets : ℕ) (ticket_value : ℕ) (total_value : ℕ)
  (h1 : won_tickets = 5 * x)
  (h2 : found_tickets = 5)
  (h3 : ticket_value = 3)
  (h4 : total_value = 30)
  (h5 : total_value = (won_tickets + found_tickets) * ticket_value) :
  x = 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_carnival_tickets_l680_68066


namespace NUMINAMATH_GPT_ratio_of_albert_to_mary_l680_68037

variables (A M B : ℕ) (s : ℕ) 

-- Given conditions as hypotheses
noncomputable def albert_is_multiple_of_mary := A = s * M
noncomputable def albert_is_4_times_betty := A = 4 * B
noncomputable def mary_is_22_years_younger := M = A - 22
noncomputable def betty_is_11 := B = 11

-- Theorem to prove the ratio of Albert's age to Mary's age
theorem ratio_of_albert_to_mary 
  (h1 : albert_is_multiple_of_mary A M s) 
  (h2 : albert_is_4_times_betty A B) 
  (h3 : mary_is_22_years_younger A M) 
  (h4 : betty_is_11 B) : 
  A / M = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_albert_to_mary_l680_68037


namespace NUMINAMATH_GPT_probability_all_balls_same_color_probability_4_white_balls_l680_68008

-- Define initial conditions
def initial_white_balls : ℕ := 6
def initial_yellow_balls : ℕ := 4
def total_initial_balls : ℕ := initial_white_balls + initial_yellow_balls

-- Define the probability calculation for drawing balls as described
noncomputable def draw_probability_same_color_after_4_draws : ℚ :=
  (6 / 10) * (7 / 10) * (8 / 10) * (9 / 10)

noncomputable def draw_probability_4_white_balls_after_4_draws : ℚ :=
  (6 / 10) * (3 / 10) * (4 / 10) * (5 / 10) + 
  3 * ((4 / 10) * (5 / 10) * (4 / 10) * (5 / 10))

-- The theorem we want to prove about the probabilities
theorem probability_all_balls_same_color :
  draw_probability_same_color_after_4_draws = 189 / 625 := by
  sorry

theorem probability_4_white_balls :
  draw_probability_4_white_balls_after_4_draws = 19 / 125 := by
  sorry

end NUMINAMATH_GPT_probability_all_balls_same_color_probability_4_white_balls_l680_68008


namespace NUMINAMATH_GPT_find_number_exists_l680_68030

theorem find_number_exists (n : ℤ) : (50 < n ∧ n < 70) ∧
    (n % 5 = 3) ∧
    (n % 7 = 2) ∧
    (n % 8 = 2) → n = 58 := 
sorry

end NUMINAMATH_GPT_find_number_exists_l680_68030


namespace NUMINAMATH_GPT_proof_l680_68095

-- Define proposition p
def p : Prop := ∀ x : ℝ, x < 0 → 2^x > x

-- Define proposition q
def q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

theorem proof : p ∨ q :=
by
  have hp : p := 
    -- Here, you would provide the proof of p being true.
    sorry
  have hq : ¬ q :=
    -- Here, you would provide the proof of q being false, 
    -- i.e., showing that ∀ x, x^2 + x + 1 ≥ 0.
    sorry
  exact Or.inl hp

end NUMINAMATH_GPT_proof_l680_68095


namespace NUMINAMATH_GPT_simplify_expression_l680_68058

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l680_68058


namespace NUMINAMATH_GPT_josh_marbles_l680_68053

theorem josh_marbles (initial_marbles lost_marbles : ℕ) (h_initial : initial_marbles = 9) (h_lost : lost_marbles = 5) :
  initial_marbles - lost_marbles = 4 :=
by
  sorry

end NUMINAMATH_GPT_josh_marbles_l680_68053


namespace NUMINAMATH_GPT_sequence_periodicity_l680_68055

theorem sequence_periodicity (a : ℕ → ℝ) (h₁ : ∀ n, a (n + 1) = 1 / (1 - a n)) (h₂ : a 8 = 2) :
  a 1 = 1 / 2 := 
sorry

end NUMINAMATH_GPT_sequence_periodicity_l680_68055


namespace NUMINAMATH_GPT_min_value_of_sum_squares_l680_68088

theorem min_value_of_sum_squares (a b : ℝ) (h : (9 / a^2) + (4 / b^2) = 1) : a^2 + b^2 ≥ 25 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_squares_l680_68088


namespace NUMINAMATH_GPT_gerald_paid_l680_68026

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 :=
by
  sorry

end NUMINAMATH_GPT_gerald_paid_l680_68026


namespace NUMINAMATH_GPT_prob1_prob2_l680_68045

theorem prob1 : -2 + 5 - |(-8 : ℤ)| + (-5) = -10 := 
by
  sorry

theorem prob2 : (-2 : ℤ)^2 * 5 - (-2)^3 / 4 = 22 := 
by
  sorry

end NUMINAMATH_GPT_prob1_prob2_l680_68045


namespace NUMINAMATH_GPT_football_game_attendance_l680_68063

theorem football_game_attendance :
  ∃ y : ℕ, (∃ x : ℕ, x + y = 280 ∧ 60 * x + 25 * y = 14000) ∧ y = 80 :=
by
  sorry

end NUMINAMATH_GPT_football_game_attendance_l680_68063


namespace NUMINAMATH_GPT_min_c_plus_d_l680_68018

theorem min_c_plus_d (c d : ℤ) (h : c * d = 36) : c + d = -37 :=
sorry

end NUMINAMATH_GPT_min_c_plus_d_l680_68018


namespace NUMINAMATH_GPT_binomial_expansion_fraction_l680_68007

theorem binomial_expansion_fraction :
  let a0 := 32
  let a1 := -80
  let a2 := 80
  let a3 := -40
  let a4 := 10
  let a5 := -1
  (2 - x)^5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 →
  (a0 + a2 + a4) / (a1 + a3) = -61 / 60 :=
by
  sorry

end NUMINAMATH_GPT_binomial_expansion_fraction_l680_68007


namespace NUMINAMATH_GPT_possible_values_of_n_are_1_prime_or_prime_squared_l680_68036

/-- A function that determines if an n x n grid with n marked squares satisfies the condition
    that every rectangle of exactly n grid squares contains at least one marked square. -/
def satisfies_conditions (n : ℕ) (marked_squares : List (ℕ × ℕ)) : Prop :=
  n.succ.succ ≤ marked_squares.length ∧ ∀ (a b : ℕ), a * b = n → ∃ x y, (x, y) ∈ marked_squares ∧ x < n ∧ y < n

/-- The main theorem stating the possible values of n. -/
theorem possible_values_of_n_are_1_prime_or_prime_squared :
  ∀ (n : ℕ), (∃ p : ℕ, Prime p ∧ (n = 1 ∨ n = p ∨ n = p^2)) ↔ satisfies_conditions n marked_squares :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_n_are_1_prime_or_prime_squared_l680_68036


namespace NUMINAMATH_GPT_value_of_x_plus_y_l680_68001

theorem value_of_x_plus_y (x y : ℤ) (h1 : x + 2 = 10) (h2 : y - 1 = 6) : x + y = 15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l680_68001


namespace NUMINAMATH_GPT_sum_of_perimeters_of_squares_l680_68042

theorem sum_of_perimeters_of_squares (x : ℝ) (h₁ : x = 3) :
  let area1 := x^2 + 4 * x + 4
  let area2 := 4 * x^2 - 12 * x + 9
  let side1 := Real.sqrt area1
  let side2 := Real.sqrt area2
  let perim1 := 4 * side1
  let perim2 := 4 * side2
  perim1 + perim2 = 32 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_of_squares_l680_68042


namespace NUMINAMATH_GPT_beaver_group_count_l680_68052

theorem beaver_group_count (B : ℕ) (h1 : 3 * B = 60) : B = 20 :=
by sorry

end NUMINAMATH_GPT_beaver_group_count_l680_68052


namespace NUMINAMATH_GPT_trigonometric_identity_l680_68065

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = (2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l680_68065
