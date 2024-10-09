import Mathlib

namespace solve_for_x_l659_65964

theorem solve_for_x (x y : ℝ) (h₁ : x - y = 8) (h₂ : x + y = 16) (h₃ : x * y = 48) : x = 12 :=
sorry

end solve_for_x_l659_65964


namespace measure_of_angle_C_l659_65987

-- Definitions of the angles
def angles (A B C : ℝ) : Prop :=
  -- Conditions: measure of angle A is 1/4 of measure of angle B
  A = (1 / 4) * B ∧
  -- Lines p and q are parallel so alternate interior angles are equal
  C = A ∧
  -- Since angles B and C are supplementary
  B + C = 180

-- The problem in Lean 4 statement: Prove that C = 36 given the conditions
theorem measure_of_angle_C (A B C : ℝ) (h : angles A B C) : C = 36 := sorry

end measure_of_angle_C_l659_65987


namespace smallest_lcm_4_digit_integers_l659_65924

theorem smallest_lcm_4_digit_integers (k l : ℕ) (h1 : 1000 ≤ k ∧ k ≤ 9999) (h2 : 1000 ≤ l ∧ l ≤ 9999) (h3 : Nat.gcd k l = 11) : Nat.lcm k l = 92092 :=
by
  sorry

end smallest_lcm_4_digit_integers_l659_65924


namespace find_n_l659_65947

theorem find_n
  (n : ℤ)
  (h : n + (n + 1) + (n + 2) + (n + 3) = 30) :
  n = 6 :=
by
  sorry

end find_n_l659_65947


namespace range_of_a_l659_65931

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 3| - |x + 2| ≥ Real.log a / Real.log 2) ↔ (0 < a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l659_65931


namespace product_173_240_l659_65975

theorem product_173_240 :
  ∃ n : ℕ, n = 3460 ∧ n * 12 = 173 * 240 ∧ 173 * 240 = 41520 :=
by
  sorry

end product_173_240_l659_65975


namespace sum_possible_m_continuous_l659_65996

noncomputable def g (x m : ℝ) : ℝ :=
if x < m then x^2 + 4 * x + 3 else 3 * x + 9

theorem sum_possible_m_continuous :
  let m₁ := -3
  let m₂ := 2
  m₁ + m₂ = -1 :=
by
  sorry

end sum_possible_m_continuous_l659_65996


namespace negation_example_l659_65998

theorem negation_example (h : ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 3 * x - 1 > 0) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 3 * x - 1 ≤ 0 :=
sorry

end negation_example_l659_65998


namespace polygon_sides_l659_65906

theorem polygon_sides (n : ℕ) 
  (h : 3240 = 180 * (n - 2) - (360)) : n = 22 := 
by 
  sorry

end polygon_sides_l659_65906


namespace composite_product_division_l659_65926

-- Define the first 12 positive composite integers
def first_six_composites := [4, 6, 8, 9, 10, 12]
def next_six_composites := [14, 15, 16, 18, 20, 21]

-- Define the product of a list of integers
def product (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

-- Define the target problem statement
theorem composite_product_division :
  (product first_six_composites : ℚ) / (product next_six_composites : ℚ) = 1 / 49 := by
  sorry

end composite_product_division_l659_65926


namespace wheat_acres_l659_65940

theorem wheat_acres (x y : ℤ) 
  (h1 : x + y = 4500) 
  (h2 : 42 * x + 35 * y = 165200) : 
  y = 3400 :=
sorry

end wheat_acres_l659_65940


namespace lana_average_speed_l659_65991

theorem lana_average_speed (initial_reading : ℕ) (final_reading : ℕ) (time_first_day : ℕ) (time_second_day : ℕ) :
  initial_reading = 1991 → 
  final_reading = 2332 → 
  time_first_day = 5 → 
  time_second_day = 7 → 
  (final_reading - initial_reading) / (time_first_day + time_second_day : ℝ) = 28.4 :=
by
  intros h_init h_final h_first h_second
  rw [h_init, h_final, h_first, h_second]
  norm_num
  sorry

end lana_average_speed_l659_65991


namespace range_of_a_for_three_tangents_curve_through_point_l659_65914

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + 3 * x^2 + a * x + a - 2

noncomputable def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 6 * x + a

theorem range_of_a_for_three_tangents_curve_through_point :
  ∀ (a : ℝ), (∀ x0 : ℝ, 2 * x0^3 + 3 * x0^2 + 4 - a = 0 → 
    ((2 * -1^3 + 3 * -1^2 + 4 - a > 0) ∧ (2 * 0^3 + 3 * 0^2 + 4 - a < 0))) ↔ (4 < a ∧ a < 5) :=
by
  sorry

end range_of_a_for_three_tangents_curve_through_point_l659_65914


namespace find_m_l659_65953

theorem find_m (m x1 x2 : ℝ) 
  (h1 : x1 * x1 - 2 * (m + 1) * x1 + m^2 + 2 = 0)
  (h2 : x2 * x2 - 2 * (m + 1) * x2 + m^2 + 2 = 0)
  (h3 : (x1 + 1) * (x2 + 1) = 8) : 
  m = 1 :=
sorry

end find_m_l659_65953


namespace original_number_without_10s_digit_l659_65925

theorem original_number_without_10s_digit (h : ℕ) (n : ℕ) 
  (h_eq_1 : h = 1) 
  (n_eq : n = 2 * 1000 + h * 100 + 84) 
  (div_by_6: n % 6 = 0) : n = 2184 → 284 = 284 :=
by
  sorry

end original_number_without_10s_digit_l659_65925


namespace max_radius_approx_l659_65982

open Real

def angle_constraint (θ : ℝ) : Prop :=
  π / 4 ≤ θ ∧ θ ≤ 3 * π / 4

def wire_constraint (r θ : ℝ) : Prop :=
  16 = r * (2 + θ)

noncomputable def max_radius (θ : ℝ) : ℝ :=
  16 / (2 + θ)

theorem max_radius_approx :
  ∃ r θ, angle_constraint θ ∧ wire_constraint r θ ∧ abs (r - 3.673) < 0.001 :=
by
  sorry

end max_radius_approx_l659_65982


namespace find_b_l659_65913

theorem find_b (a b c : ℝ) (h1 : a = 6) (h2 : c = 3) (h3 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) : b = 15 :=
by
  rw [h1, h2] at h3
  sorry

end find_b_l659_65913


namespace hyperbola_asymptote_solution_l659_65958

theorem hyperbola_asymptote_solution (b : ℝ) (hb : b > 0)
  (h_asym : ∀ x y, (∀ y : ℝ, y = (1 / 2) * x ∨ y = - (1 / 2) * x) → (x^2 / 4 - y^2 / b^2 = 1)) :
  b = 1 :=
sorry

end hyperbola_asymptote_solution_l659_65958


namespace true_proposition_is_b_l659_65934

open Real

theorem true_proposition_is_b :
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∧
  (¬ ∀ n : ℝ, n^2 ≥ n) ∧
  (¬ ∀ n : ℝ, ∃ m : ℝ, m^2 < n) ∧
  (¬ ∀ n : ℝ, n^2 < n) :=
  by
    sorry

end true_proposition_is_b_l659_65934


namespace mosel_fills_315_boxes_per_week_l659_65974

-- Definitions for the conditions given in the problem.
def hens : ℕ := 270
def eggs_per_hen_per_day : ℕ := 1
def boxes_capacity : ℕ := 6
def days_per_week : ℕ := 7

-- Objective: Prove that the number of boxes filled each week is 315
theorem mosel_fills_315_boxes_per_week :
  let eggs_per_day := hens * eggs_per_hen_per_day
  let boxes_per_day := eggs_per_day / boxes_capacity
  let boxes_per_week := boxes_per_day * days_per_week
  boxes_per_week = 315 := by
  sorry

end mosel_fills_315_boxes_per_week_l659_65974


namespace students_taking_either_but_not_both_l659_65965

-- Definitions to encapsulate the conditions
def students_taking_both : ℕ := 15
def students_taking_mathematics : ℕ := 30
def students_taking_history_only : ℕ := 12

-- The goal is to prove the number of students taking mathematics or history but not both
theorem students_taking_either_but_not_both
  (hb : students_taking_both = 15)
  (hm : students_taking_mathematics = 30)
  (hh : students_taking_history_only = 12) : 
  students_taking_mathematics - students_taking_both + students_taking_history_only = 27 :=
by
  sorry

end students_taking_either_but_not_both_l659_65965


namespace find_point_M_l659_65933

def parabola (x y : ℝ) := x^2 = 4 * y
def focus_dist (M : ℝ × ℝ) := dist M (0, 1) = 2
def point_on_parabola (M : ℝ × ℝ) := parabola M.1 M.2

theorem find_point_M (M : ℝ × ℝ) (h1 : point_on_parabola M) (h2 : focus_dist M) :
  M = (2, 1) ∨ M = (-2, 1) := by
  sorry

end find_point_M_l659_65933


namespace equilateral_triangle_l659_65905

noncomputable def angles_arithmetic_seq (A B C : ℝ) : Prop := B - A = C - B

noncomputable def sides_geometric_seq (a b c : ℝ) : Prop := b / a = c / b

theorem equilateral_triangle 
  (A B C a b c : ℝ) 
  (h_angles : angles_arithmetic_seq A B C) 
  (h_sides : sides_geometric_seq a b c) 
  (h_triangle : A + B + C = π) 
  (h_pos_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  (A = B ∧ B = C) ∧ (a = b ∧ b = c) :=
sorry

end equilateral_triangle_l659_65905


namespace probability_of_white_balls_from_both_boxes_l659_65956

theorem probability_of_white_balls_from_both_boxes :
  let P_white_A := 3 / (3 + 2)
  let P_white_B := 2 / (2 + 3)
  P_white_A * P_white_B = 6 / 25 :=
by
  sorry

end probability_of_white_balls_from_both_boxes_l659_65956


namespace inequality_solution_set_l659_65943

theorem inequality_solution_set (x : ℝ) :
  (3 * x + 1) / (1 - 2 * x) ≥ 0 ↔ -1 / 3 ≤ x ∧ x < 1 / 2 :=
by
  sorry

end inequality_solution_set_l659_65943


namespace puppies_per_cage_calculation_l659_65927

noncomputable def initial_puppies : ℝ := 18.0
noncomputable def additional_puppies : ℝ := 3.0
noncomputable def total_puppies : ℝ := initial_puppies + additional_puppies
noncomputable def total_cages : ℝ := 4.2
noncomputable def puppies_per_cage : ℝ := total_puppies / total_cages

theorem puppies_per_cage_calculation :
  puppies_per_cage = 5.0 :=
by
  sorry

end puppies_per_cage_calculation_l659_65927


namespace arithmetic_seq_sum_x_y_l659_65972

theorem arithmetic_seq_sum_x_y :
  ∃ (x y : ℕ), (∀ n : ℕ, n > 0 → a_n = 3 + (n - 1) * 5) ∧ x + 33 = 33 ∧ x = 28 → x + y = 61 :=
by
  sorry

end arithmetic_seq_sum_x_y_l659_65972


namespace parallel_vectors_l659_65957

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2, 3)) (h₂ : b = (-1, 2)) :
  (m * a.1 + b.1) * (-1) - 4 * (m * a.2 + b.2) = 0 → m = -1 / 2 :=
by
  intro h
  rw [h₁, h₂] at h
  simp at h
  sorry

end parallel_vectors_l659_65957


namespace simplify_expression_l659_65983

theorem simplify_expression : 
  (6^8 - 4^7) * (2^3 - (-2)^3) ^ 10 = 1663232 * 16 ^ 10 := 
by {
  sorry
}

end simplify_expression_l659_65983


namespace final_selling_price_l659_65989

def actual_price : ℝ := 9941.52
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.05

noncomputable def final_price (P : ℝ) : ℝ :=
  P * (1 - discount1) * (1 - discount2) * (1 - discount3)

theorem final_selling_price :
  final_price actual_price = 6800.00 :=
by
  sorry

end final_selling_price_l659_65989


namespace remainder_when_eight_n_plus_five_divided_by_eleven_l659_65986

theorem remainder_when_eight_n_plus_five_divided_by_eleven
  (n : ℤ) (h : n % 11 = 4) : (8 * n + 5) % 11 = 4 := 
  sorry

end remainder_when_eight_n_plus_five_divided_by_eleven_l659_65986


namespace constant_function_of_zero_derivative_l659_65941

theorem constant_function_of_zero_derivative
  {f : ℝ → ℝ}
  (h : ∀ x : ℝ, deriv f x = 0) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end constant_function_of_zero_derivative_l659_65941


namespace no_infinite_lines_satisfying_conditions_l659_65955

theorem no_infinite_lines_satisfying_conditions :
  ¬ ∃ (l : ℕ → ℝ → ℝ → Prop)
      (k : ℕ → ℝ)
      (a b : ℕ → ℝ),
    (∀ n, l n 1 1) ∧
    (∀ n, k (n + 1) = a n - b n) ∧
    (∀ n, k n * k (n + 1) ≥ 0) := 
sorry

end no_infinite_lines_satisfying_conditions_l659_65955


namespace largest_positive_integer_l659_65917

def binary_op (n : ℕ) : ℤ := n - (n * 5)

theorem largest_positive_integer (n : ℕ) (h : binary_op n < 21) : n ≤ 1 := 
sorry

end largest_positive_integer_l659_65917


namespace percentage_discount_l659_65936

theorem percentage_discount (P D: ℝ) 
  (sale_price: P * (100 - D) / 100 = 78.2)
  (final_price_increase: 78.2 * 1.25 = P - 5.75):
  D = 24.44 :=
by
  sorry

end percentage_discount_l659_65936


namespace sum_three_consecutive_divisible_by_three_l659_65939

theorem sum_three_consecutive_divisible_by_three (n : ℤ) : 3 ∣ ((n - 1) + n + (n + 1)) :=
by
  sorry  -- Proof goes here

end sum_three_consecutive_divisible_by_three_l659_65939


namespace pesto_calculation_l659_65977

def basil_needed_per_pesto : ℕ := 4
def basil_harvest_per_week : ℕ := 16
def weeks : ℕ := 8
def total_basil_harvested : ℕ := basil_harvest_per_week * weeks
def total_pesto_possible : ℕ := total_basil_harvested / basil_needed_per_pesto

theorem pesto_calculation :
  total_pesto_possible = 32 :=
by
  sorry

end pesto_calculation_l659_65977


namespace function_properties_l659_65918

noncomputable def f (x : ℝ) : ℝ := x^2

theorem function_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end function_properties_l659_65918


namespace smallest_three_digit_number_exists_l659_65980

def is_valid_permutation_sum (x y z : ℕ) : Prop :=
  let perms := [100*x + 10*y + z, 100*x + 10*z + y, 100*y + 10*x + z, 100*z + 10*x + y, 100*y + 10*z + x, 100*z + 10*y + x]
  perms.sum = 2220

theorem smallest_three_digit_number_exists : ∃ (x y z : ℕ), x < y ∧ y < z ∧ x + y + z = 10 ∧ is_valid_permutation_sum x y z ∧ 100 * x + 10 * y + z = 127 :=
by {
  -- proof goal and steps would go here if we were to complete the proof
  sorry
}

end smallest_three_digit_number_exists_l659_65980


namespace entrepreneurs_not_attending_any_session_l659_65907

theorem entrepreneurs_not_attending_any_session 
  (total_entrepreneurs : ℕ) 
  (digital_marketing_attendees : ℕ) 
  (e_commerce_attendees : ℕ) 
  (both_sessions_attendees : ℕ)
  (h1 : total_entrepreneurs = 40)
  (h2 : digital_marketing_attendees = 22) 
  (h3 : e_commerce_attendees = 18) 
  (h4 : both_sessions_attendees = 8) : 
  total_entrepreneurs - (digital_marketing_attendees + e_commerce_attendees - both_sessions_attendees) = 8 :=
by sorry

end entrepreneurs_not_attending_any_session_l659_65907


namespace loss_percentage_l659_65938

theorem loss_percentage (C S : ℕ) (H1 : C = 750) (H2 : S = 600) : (C - S) * 100 / C = 20 := by
  sorry

end loss_percentage_l659_65938


namespace least_pos_int_for_multiple_of_5_l659_65968

theorem least_pos_int_for_multiple_of_5 (n : ℕ) (h1 : n = 725) : ∃ x : ℕ, x > 0 ∧ (725 + x) % 5 = 0 ∧ x = 5 :=
by
  sorry

end least_pos_int_for_multiple_of_5_l659_65968


namespace find_number_l659_65967

theorem find_number (x : ℤ) (h : x + x^2 + 15 = 96) : x = -9 :=
sorry

end find_number_l659_65967


namespace shirley_ends_with_106_l659_65919

-- Define the initial number of eggs and the number bought
def initialEggs : Nat := 98
def additionalEggs : Nat := 8

-- Define the final count as the sum of initial eggs and additional eggs
def finalEggCount : Nat := initialEggs + additionalEggs

-- State the theorem with the correct answer
theorem shirley_ends_with_106 :
  finalEggCount = 106 :=
by
  sorry

end shirley_ends_with_106_l659_65919


namespace one_serving_weight_l659_65929

-- Outline the main variables
def chicken_weight_pounds : ℝ := 4.5
def stuffing_weight_ounces : ℝ := 24
def num_servings : ℝ := 12
def conversion_factor : ℝ := 16 -- 1 pound = 16 ounces

-- Define the weights in ounces
def chicken_weight_ounces : ℝ := chicken_weight_pounds * conversion_factor

-- Total weight in ounces for all servings
def total_weight_ounces : ℝ := chicken_weight_ounces + stuffing_weight_ounces

-- Prove one serving weight in ounces
theorem one_serving_weight : total_weight_ounces / num_servings = 8 := by
  sorry

end one_serving_weight_l659_65929


namespace weekly_earnings_l659_65984

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def phone_repairs : ℕ := 5
def laptop_repairs : ℕ := 2
def computer_repairs : ℕ := 2

def total_earnings : ℕ := 
  phone_repairs * phone_repair_cost + 
  laptop_repairs * laptop_repair_cost + 
  computer_repairs * computer_repair_cost

theorem weekly_earnings : total_earnings = 121 := by
  sorry

end weekly_earnings_l659_65984


namespace triangle_area_l659_65945

theorem triangle_area {a b : ℝ} (h₁ : a = 3) (h₂ : b = 4) (h₃ : Real.sin (C : ℝ) = 1/2) :
  let area := (1 / 2) * a * b * (Real.sin C) 
  area = 3 := 
by
  rw [h₁, h₂, h₃]
  simp [Real.sin, mul_assoc]
  sorry

end triangle_area_l659_65945


namespace square_of_second_arm_l659_65963

theorem square_of_second_arm (a b c : ℝ) (h₁ : c = a + 2) (h₂ : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 :=
sorry

end square_of_second_arm_l659_65963


namespace sin_double_angle_l659_65971

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin α = 4 / 5) : Real.sin (2 * α) = -24 / 25 :=
by
  sorry

end sin_double_angle_l659_65971


namespace percent_increase_of_income_l659_65999

theorem percent_increase_of_income (original_income new_income : ℝ) 
  (h1 : original_income = 120) (h2 : new_income = 180) :
  ((new_income - original_income) / original_income) * 100 = 50 := 
by 
  rw [h1, h2]
  norm_num

end percent_increase_of_income_l659_65999


namespace ceil_sqrt_225_eq_15_l659_65995

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end ceil_sqrt_225_eq_15_l659_65995


namespace irene_overtime_pay_per_hour_l659_65901

def irene_base_pay : ℝ := 500
def irene_base_hours : ℕ := 40
def irene_total_hours_last_week : ℕ := 50
def irene_total_income_last_week : ℝ := 700

theorem irene_overtime_pay_per_hour :
  (irene_total_income_last_week - irene_base_pay) / (irene_total_hours_last_week - irene_base_hours) = 20 := 
by
  sorry

end irene_overtime_pay_per_hour_l659_65901


namespace river_current_speed_l659_65973

def motorboat_speed_still_water : ℝ := 20
def distance_between_points : ℝ := 60
def total_trip_time : ℝ := 6.25

theorem river_current_speed : ∃ v_T : ℝ, v_T = 4 ∧ 
  (distance_between_points / (motorboat_speed_still_water + v_T)) + 
  (distance_between_points / (motorboat_speed_still_water - v_T)) = total_trip_time := 
sorry

end river_current_speed_l659_65973


namespace speed_of_man_l659_65915

theorem speed_of_man (v_m v_s : ℝ) 
    (h1 : (v_m + v_s) * 4 = 32) 
    (h2 : (v_m - v_s) * 4 = 24) : v_m = 7 := 
by
  sorry

end speed_of_man_l659_65915


namespace amy_total_equals_bob_total_l659_65900

def original_price : ℝ := 120.00
def sales_tax_rate : ℝ := 0.08
def discount_rate : ℝ := 0.25
def additional_discount : ℝ := 0.10
def num_sweaters : ℕ := 4

def calculate_amy_total (original_price : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (additional_discount : ℝ) (num_sweaters : ℕ) : ℝ :=
  let price_with_tax := original_price * (1.0 + sales_tax_rate)
  let discounted_price := price_with_tax * (1.0 - discount_rate)
  let final_price := discounted_price * (1.0 - additional_discount)
  final_price * (num_sweaters : ℝ)
  
def calculate_bob_total (original_price : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (additional_discount : ℝ) (num_sweaters : ℕ) : ℝ :=
  let discounted_price := original_price * (1.0 - discount_rate)
  let further_discounted_price := discounted_price * (1.0 - additional_discount)
  let price_with_tax := further_discounted_price * (1.0 + sales_tax_rate)
  price_with_tax * (num_sweaters : ℝ)

theorem amy_total_equals_bob_total :
  calculate_amy_total original_price sales_tax_rate discount_rate additional_discount num_sweaters =
  calculate_bob_total original_price sales_tax_rate discount_rate additional_discount num_sweaters :=
by
  sorry

end amy_total_equals_bob_total_l659_65900


namespace angela_action_figures_l659_65970

theorem angela_action_figures (n s r g : ℕ) (hn : n = 24) (hs : s = n * 1 / 4) (hr : r = n - s) (hg : g = r * 1 / 3) :
  r - g = 12 :=
sorry

end angela_action_figures_l659_65970


namespace neg_p_implies_neg_q_l659_65992

variables {x : ℝ}

def condition_p (x : ℝ) : Prop := |x + 1| > 2
def condition_q (x : ℝ) : Prop := 5 * x - 6 > x^2
def neg_p (x : ℝ) : Prop := |x + 1| ≤ 2
def neg_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

theorem neg_p_implies_neg_q : (∀ x, neg_p x → neg_q x) :=
by 
  -- Proof is skipped according to the instructions
  sorry

end neg_p_implies_neg_q_l659_65992


namespace interval_length_l659_65981

theorem interval_length (a b m h : ℝ) (h_eq : h = m / |a - b|) : |a - b| = m / h := 
by 
  sorry

end interval_length_l659_65981


namespace length_of_side_of_largest_square_l659_65994

-- Definitions based on the conditions
def string_length : ℕ := 24

-- The main theorem corresponding to the problem statement.
theorem length_of_side_of_largest_square (h: string_length = 24) : 24 / 4 = 6 :=
by
  sorry

end length_of_side_of_largest_square_l659_65994


namespace compare_fractions_compare_integers_l659_65911

-- First comparison: Prove -4/7 > -2/3
theorem compare_fractions : - (4 : ℚ) / 7 > - (2 : ℚ) / 3 := 
by sorry

-- Second comparison: Prove -(-7) > -| -7 |
theorem compare_integers : -(-7) > -abs (-7) := 
by sorry

end compare_fractions_compare_integers_l659_65911


namespace sum_series_equals_three_fourths_l659_65910

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l659_65910


namespace number_of_new_students_l659_65990

variable (O N : ℕ)
variable (H1 : 48 * O + 32 * N = 44 * 160)
variable (H2 : O + N = 160)

theorem number_of_new_students : N = 40 := sorry

end number_of_new_students_l659_65990


namespace exists_prime_among_15_numbers_l659_65904

theorem exists_prime_among_15_numbers 
    (integers : Fin 15 → ℕ)
    (h1 : ∀ i, 1 < integers i)
    (h2 : ∀ i, integers i < 1998)
    (h3 : ∀ i j, i ≠ j → Nat.gcd (integers i) (integers j) = 1) :
    ∃ i, Nat.Prime (integers i) :=
by
  sorry

end exists_prime_among_15_numbers_l659_65904


namespace total_height_geometric_solid_l659_65949

-- Definitions corresponding to conditions
def radius_cylinder1 : ℝ := 1
def radius_cylinder2 : ℝ := 3
def height_water_surface_figure2 : ℝ := 20
def height_water_surface_figure3 : ℝ := 28

-- The total height of the geometric solid is 29 cm
theorem total_height_geometric_solid :
  ∃ height_total : ℝ,
    (height_water_surface_figure2 + height_total - height_water_surface_figure3) = 29 :=
sorry

end total_height_geometric_solid_l659_65949


namespace trig_identity_l659_65921

theorem trig_identity (α a : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
    (h3 : (Real.tan α) + (1 / (Real.tan α)) = a) : 
    (1 / Real.sin α) + (1 / Real.cos α) = Real.sqrt (a^2 + 2 * a) :=
by
  sorry

end trig_identity_l659_65921


namespace solution_l659_65928

theorem solution 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a^2 / (b - c)^2) + (b^2 / (c - a)^2) + (c^2 / (a - b)^2) = 0) :
  (a^3 / (b - c)^3) + (b^3 / (c - a)^3) + (c^3 / (a - b)^3) = 0 := 
sorry 

end solution_l659_65928


namespace find_value_of_expression_l659_65950

theorem find_value_of_expression (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) : 2 * a + 2 * b - 3 * (a * b) = 9 :=
by
  sorry

end find_value_of_expression_l659_65950


namespace johns_final_weight_is_200_l659_65952

-- Define the initial weight, percentage of weight loss, and weight gain
def initial_weight : ℝ := 220
def weight_loss_percentage : ℝ := 0.10
def weight_gain : ℝ := 2

-- Define a function to calculate the final weight
def final_weight (initial_weight : ℝ) (weight_loss_percentage : ℝ) (weight_gain : ℝ) : ℝ := 
  let weight_lost := initial_weight * weight_loss_percentage
  let weight_after_loss := initial_weight - weight_lost
  weight_after_loss + weight_gain

-- The proof problem is to show that the final weight is 200 pounds
theorem johns_final_weight_is_200 :
  final_weight initial_weight weight_loss_percentage weight_gain = 200 := 
by
  sorry

end johns_final_weight_is_200_l659_65952


namespace flowers_count_l659_65966

theorem flowers_count (lilies : ℕ) (sunflowers : ℕ) (daisies : ℕ) (total_flowers : ℕ) (roses : ℕ)
  (h1 : lilies = 40) (h2 : sunflowers = 40) (h3 : daisies = 40) (h4 : total_flowers = 160) :
  lilies + sunflowers + daisies + roses = 160 → roses = 40 := 
by
  sorry

end flowers_count_l659_65966


namespace div_expression_l659_65916

theorem div_expression : 180 / (12 + 13 * 2) = 90 / 19 := 
  sorry

end div_expression_l659_65916


namespace find_f_at_2_l659_65948

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem find_f_at_2 (a b : ℝ) 
  (h1 : 3 + 2 * a + b = 0) 
  (h2 : 1 + a + b + 1 = -2) : 
  f a b 2 = 3 := 
by
  dsimp [f]
  sorry

end find_f_at_2_l659_65948


namespace journey_time_l659_65903

variables (d1 d2 : ℝ) (T : ℝ)

theorem journey_time :
  (d1 / 30 + (150 - d1) / 4 = T) ∧
  (d1 / 30 + d2 / 30 + (150 - (d1 + d2)) / 4 = T) ∧
  (d2 / 4 + (150 - (d1 + d2)) / 4 = T) ∧
  (d1 = 3 / 2 * d2) 
  → T = 18 :=
by
  sorry

end journey_time_l659_65903


namespace quadratic_point_comparison_l659_65988

theorem quadratic_point_comparison (c y1 y2 y3 : ℝ) 
  (h1 : y1 = -(-2:ℝ)^2 + c)
  (h2 : y2 = -(1:ℝ)^2 + c)
  (h3 : y3 = -(3:ℝ)^2 + c) : y2 > y1 ∧ y1 > y3 := 
by
  sorry

end quadratic_point_comparison_l659_65988


namespace circumscribed_circle_area_l659_65902

theorem circumscribed_circle_area (side_length : ℝ) (h : side_length = 12) :
  ∃ (A : ℝ), A = 48 * π :=
by
  sorry

end circumscribed_circle_area_l659_65902


namespace intersection_sets_l659_65960

-- Define the sets A and B as given in the problem conditions
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {0, 2, 4}

-- Lean theorem statement for proving the intersection of sets A and B is {0, 2}
theorem intersection_sets : A ∩ B = {0, 2} := 
by
  sorry

end intersection_sets_l659_65960


namespace range_of_m_l659_65909

theorem range_of_m (m : ℝ) : (1^2 + 2*1 - m ≤ 0) ∧ (2^2 + 2*2 - m > 0) → 3 ≤ m ∧ m < 8 := by
  sorry

end range_of_m_l659_65909


namespace vasim_share_l659_65908

theorem vasim_share (x : ℕ) (F V R : ℕ) (h1 : F = 3 * x) (h2 : V = 5 * x) (h3 : R = 11 * x) (h4 : R - F = 2400) : V = 1500 :=
by sorry

end vasim_share_l659_65908


namespace base7_difference_l659_65942

theorem base7_difference (a b : ℕ) (h₁ : a = 12100) (h₂ : b = 3666) :
  ∃ c, c = 1111 ∧ (a - b = c) := by
sorry

end base7_difference_l659_65942


namespace person_a_catch_up_person_b_5_times_l659_65978

theorem person_a_catch_up_person_b_5_times :
  ∀ (num_flags laps_a laps_b : ℕ),
  num_flags = 2015 →
  laps_a = 23 →
  laps_b = 13 →
  (∃ t : ℕ, ∃ n : ℕ, 10 * t = num_flags * n ∧
             23 * t / 10 = k * num_flags ∧
             n % 2 = 0) →
  n = 10 →
  10 / (2 * 1) = 5 :=
by sorry

end person_a_catch_up_person_b_5_times_l659_65978


namespace roshini_sweets_cost_correct_l659_65923

noncomputable def roshini_sweet_cost_before_discounts_and_tax : ℝ := 10.54

theorem roshini_sweets_cost_correct (R F1 F2 F3 : ℝ) (h1 : R + F1 + F2 + F3 = 10.54)
    (h2 : R * 0.9 = (10.50 - 9.20) / 1.08)
    (h3 : F1 + F2 + F3 = 3.40 + 4.30 + 1.50) :
    R + F1 + F2 + F3 = roshini_sweet_cost_before_discounts_and_tax :=
by
  sorry

end roshini_sweets_cost_correct_l659_65923


namespace john_total_distance_l659_65997

theorem john_total_distance : 
  let daily_distance := 1700
  let days_run := 6
  daily_distance * days_run = 10200 :=
by
  sorry

end john_total_distance_l659_65997


namespace quadratic_two_distinct_real_roots_l659_65961

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l659_65961


namespace unique_solution_7x_eq_3y_plus_4_l659_65969

theorem unique_solution_7x_eq_3y_plus_4 (x y : ℕ) (hx : 1 ≤ x) (hy : 1 ≤ y) :
    7^x = 3^y + 4 ↔ (x = 1 ∧ y = 1) :=
by
  sorry

end unique_solution_7x_eq_3y_plus_4_l659_65969


namespace cube_volume_l659_65935

theorem cube_volume (S : ℝ) (hS : S = 294) : ∃ V : ℝ, V = 343 := by
  sorry

end cube_volume_l659_65935


namespace probability_of_5_distinct_dice_rolls_is_5_over_54_l659_65920

def count_distinct_dice_rolls : ℕ :=
  6 * 5 * 4 * 3 * 2

def total_dice_rolls : ℕ :=
  6 ^ 5

def probability_of_distinct_rolls : ℚ :=
  count_distinct_dice_rolls / total_dice_rolls

theorem probability_of_5_distinct_dice_rolls_is_5_over_54 : 
  probability_of_distinct_rolls = 5 / 54 :=
by
  sorry

end probability_of_5_distinct_dice_rolls_is_5_over_54_l659_65920


namespace total_pumpkin_weight_l659_65985

-- Conditions
def weight_first_pumpkin : ℝ := 4
def weight_second_pumpkin : ℝ := 8.7

-- Statement
theorem total_pumpkin_weight :
  weight_first_pumpkin + weight_second_pumpkin = 12.7 :=
by
  -- Proof can be done manually or via some automation here
  sorry

end total_pumpkin_weight_l659_65985


namespace infinite_pairs_natural_numbers_l659_65912

theorem infinite_pairs_natural_numbers :
  ∃ (infinite_pairs : ℕ × ℕ → Prop), (∀ a b : ℕ, infinite_pairs (a, b) ↔ (b ∣ (a^2 + 1) ∧ a ∣ (b^2 + 1))) ∧
    ∀ n : ℕ, ∃ (a b : ℕ), infinite_pairs (a, b) :=
sorry

end infinite_pairs_natural_numbers_l659_65912


namespace operation_result_l659_65946

def star (a b c : ℝ) : ℝ := (a + b + c) ^ 2

theorem operation_result (x : ℝ) : star (x - 1) (1 - x) 1 = 1 := 
by
  sorry

end operation_result_l659_65946


namespace yura_picture_dimensions_l659_65976

theorem yura_picture_dimensions (a b : ℕ) (h : (a + 2) * (b + 2) - a * b = a * b) :
  (a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) :=
by
  -- Place your proof here
  sorry

end yura_picture_dimensions_l659_65976


namespace find_sum_of_numbers_l659_65954

theorem find_sum_of_numbers 
  (a b : ℕ)
  (h₁ : a.gcd b = 5)
  (h₂ : a * b / a.gcd b = 120)
  (h₃ : (1 : ℚ) / a + 1 / b = 0.09166666666666666) :
  a + b = 55 := 
sorry

end find_sum_of_numbers_l659_65954


namespace max_f_value_l659_65937

noncomputable def f (x : ℝ) : ℝ := 9 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 15 :=
by
  sorry

end max_f_value_l659_65937


namespace trig_identity_proofs_l659_65930

theorem trig_identity_proofs (α : ℝ) 
  (h : Real.sin α + Real.cos α = 1 / 5) :
  (Real.sin α - Real.cos α = 7 / 5 ∨ Real.sin α - Real.cos α = -7 / 5) ∧
  (Real.sin α ^ 3 + Real.cos α ^ 3 = 37 / 125) :=
by
  sorry

end trig_identity_proofs_l659_65930


namespace legoland_kangaroos_l659_65951

theorem legoland_kangaroos :
  ∃ (K R : ℕ), R = 5 * K ∧ K + R = 216 ∧ R = 180 := by
  sorry

end legoland_kangaroos_l659_65951


namespace sum_is_composite_l659_65993

theorem sum_is_composite (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b = c * d) : 
  ∃ x y : ℕ, 1 < x ∧ 1 < y ∧ x * y = a + b + c + d :=
sorry

end sum_is_composite_l659_65993


namespace trapezium_second_side_length_l659_65922

theorem trapezium_second_side_length
  (side1 : ℝ)
  (height : ℝ)
  (area : ℝ) 
  (h1 : side1 = 20) 
  (h2 : height = 13) 
  (h3 : area = 247) : 
  ∃ side2 : ℝ, 0 ≤ side2 ∧ ∀ side2, area = 1 / 2 * (side1 + side2) * height → side2 = 18 :=
by
  use 18
  sorry

end trapezium_second_side_length_l659_65922


namespace rectangle_width_l659_65932

theorem rectangle_width (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * (w + l) = w * l) : w = 3 :=
by sorry

end rectangle_width_l659_65932


namespace odd_and_monotonic_l659_65944

-- Definitions based on the conditions identified
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_monotonic_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x ≤ f y
def f (x : ℝ) : ℝ := x ^ 3

-- Theorem statement without the proof
theorem odd_and_monotonic :
  is_odd f ∧ is_monotonic_increasing f :=
sorry

end odd_and_monotonic_l659_65944


namespace value_of_h_h_2_is_353_l659_65979

def h (x : ℕ) : ℕ := 3 * x^2 - x + 1

theorem value_of_h_h_2_is_353 : h (h 2) = 353 := 
by
  sorry

end value_of_h_h_2_is_353_l659_65979


namespace p_necessary_condition_q_l659_65962

variable (a b : ℝ) (p : ab = 0) (q : a^2 + b^2 ≠ 0)

theorem p_necessary_condition_q : (∀ a b : ℝ, (ab = 0) → (a^2 + b^2 ≠ 0)) ∧ (∃ a b : ℝ, (a^2 + b^2 ≠ 0) ∧ ¬ (ab = 0)) := sorry

end p_necessary_condition_q_l659_65962


namespace negation_equivalence_l659_65959

theorem negation_equivalence {Triangle : Type} (has_circumcircle : Triangle → Prop) :
  ¬ (∃ (t : Triangle), ¬ has_circumcircle t) ↔ (∀ (t : Triangle), has_circumcircle t) :=
by
  sorry

end negation_equivalence_l659_65959
