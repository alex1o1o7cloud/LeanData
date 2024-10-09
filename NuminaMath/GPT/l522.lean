import Mathlib

namespace angela_finished_9_problems_l522_52252

def martha_problems : Nat := 2

def jenna_problems : Nat := 4 * martha_problems - 2

def mark_problems : Nat := jenna_problems / 2

def total_problems : Nat := 20

def total_friends_problems : Nat := martha_problems + jenna_problems + mark_problems

def angela_problems : Nat := total_problems - total_friends_problems

theorem angela_finished_9_problems : angela_problems = 9 := by
  -- Placeholder for proof steps
  sorry

end angela_finished_9_problems_l522_52252


namespace inequality_ab5_bc5_ca5_l522_52271

theorem inequality_ab5_bc5_ca5 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b^5 + b * c^5 + c * a^5 ≥ a * b * c * (a^2 * b + b^2 * c + c^2 * a) :=
sorry

end inequality_ab5_bc5_ca5_l522_52271


namespace probability_below_8_l522_52253

theorem probability_below_8 
  (P10 P9 P8 : ℝ)
  (P10_eq : P10 = 0.24)
  (P9_eq : P9 = 0.28)
  (P8_eq : P8 = 0.19) :
  1 - (P10 + P9 + P8) = 0.29 := 
by
  sorry

end probability_below_8_l522_52253


namespace mike_hours_per_day_l522_52211

theorem mike_hours_per_day (total_hours : ℕ) (total_days : ℕ) (h_total_hours : total_hours = 15) (h_total_days : total_days = 5) : (total_hours / total_days) = 3 := by
  sorry

end mike_hours_per_day_l522_52211


namespace rita_canoe_distance_l522_52214

theorem rita_canoe_distance 
  (up_speed : ℕ) (down_speed : ℕ)
  (wind_up_decrease : ℕ) (wind_down_increase : ℕ)
  (total_time : ℕ) 
  (effective_up_speed : ℕ := up_speed - wind_up_decrease)
  (effective_down_speed : ℕ := down_speed + wind_down_increase)
  (T_up : ℚ := D / effective_up_speed)
  (T_down : ℚ := D / effective_down_speed) :
  (T_up + T_down = total_time) ->
  (D = 7) := 
by
  sorry

-- Parameters as defined in the problem
def up_speed : ℕ := 3
def down_speed : ℕ := 9
def wind_up_decrease : ℕ := 2
def wind_down_increase : ℕ := 4
def total_time : ℕ := 8

end rita_canoe_distance_l522_52214


namespace total_cups_needed_l522_52298

def servings : Float := 18.0
def cups_per_serving : Float := 2.0

theorem total_cups_needed : servings * cups_per_serving = 36.0 :=
by
  sorry

end total_cups_needed_l522_52298


namespace diagonal_square_grid_size_l522_52261

theorem diagonal_square_grid_size (n : ℕ) (h : 2 * n - 1 = 2017) : n = 1009 :=
by
  sorry

end diagonal_square_grid_size_l522_52261


namespace find_base_of_triangle_l522_52210

def triangle_base (area : ℝ) (height : ℝ) (base : ℝ) : Prop :=
  area = (base * height) / 2

theorem find_base_of_triangle : triangle_base 24 8 6 :=
by
  -- Simplification and computation steps are omitted as per the instruction
  sorry

end find_base_of_triangle_l522_52210


namespace necessary_but_not_sufficient_l522_52286

-- Definitions from the conditions
def p (a b : ℤ) : Prop := True  -- Since their integrality is given
def q (a b : ℤ) : Prop := ∃ (x : ℤ), (x^2 + a * x + b = 0)

theorem necessary_but_not_sufficient (a b : ℤ) : 
  (¬ (p a b → q a b)) ∧ (q a b → p a b) :=
by
  sorry

end necessary_but_not_sufficient_l522_52286


namespace like_terms_monomials_m_n_l522_52228

theorem like_terms_monomials_m_n (m n : ℕ) (h1 : 3 * x ^ m * y = - x ^ 3 * y ^ n) :
  m - n = 2 :=
by
  sorry

end like_terms_monomials_m_n_l522_52228


namespace cloud9_total_revenue_after_discounts_and_refunds_l522_52258

theorem cloud9_total_revenue_after_discounts_and_refunds :
  let individual_total := 12000
  let individual_early_total := 3000
  let group_a_total := 6000
  let group_a_participants := 8
  let group_b_total := 9000
  let group_b_participants := 15
  let group_c_total := 15000
  let group_c_participants := 22
  let individual_refund1 := 500
  let individual_refund1_count := 3
  let individual_refund2 := 300
  let individual_refund2_count := 2
  let group_refund := 800
  let group_refund_count := 2

  -- Discounts
  let early_booking_discount := 0.03
  let discount_between_5_and_10 := 0.05
  let discount_between_11_and_20 := 0.1
  let discount_21_and_more := 0.15

  -- Calculating individual bookings
  let individual_early_discount_total := individual_early_total * early_booking_discount
  let individual_total_after_discount := individual_total - individual_early_discount_total

  -- Calculating group bookings
  let group_a_discount := group_a_total * discount_between_5_and_10
  let group_a_early_discount := (group_a_total - group_a_discount) * early_booking_discount
  let group_a_total_after_discount := group_a_total - group_a_discount - group_a_early_discount

  let group_b_discount := group_b_total * discount_between_11_and_20
  let group_b_total_after_discount := group_b_total - group_b_discount

  let group_c_discount := group_c_total * discount_21_and_more
  let group_c_early_discount := (group_c_total - group_c_discount) * early_booking_discount
  let group_c_total_after_discount := group_c_total - group_c_discount - group_c_early_discount

  let total_group_after_discount := group_a_total_after_discount + group_b_total_after_discount + group_c_total_after_discount

  -- Calculating refunds
  let total_individual_refunds := (individual_refund1 * individual_refund1_count) + (individual_refund2 * individual_refund2_count)
  let total_group_refunds := group_refund

  let total_refunds := total_individual_refunds + total_group_refunds

  -- Final total calculation after all discounts and refunds
  let final_total := individual_total_after_discount + total_group_after_discount - total_refunds
  final_total = 35006.50 := by
  -- The rest of the proof would go here, but we use sorry to bypass the proof.
  sorry

end cloud9_total_revenue_after_discounts_and_refunds_l522_52258


namespace nails_no_three_collinear_l522_52224

-- Let's denote the 8x8 chessboard as an 8x8 grid of cells

-- Define a type for positions on the chessboard
def Position := (ℕ × ℕ)

-- Condition: 16 nails should be placed in such a way that no three are collinear. 
-- Let's create an inductive type to capture these conditions

def no_three_collinear (nails : List Position) : Prop :=
  ∀ (p1 p2 p3 : Position), p1 ∈ nails → p2 ∈ nails → p3 ∈ nails → 
  (p1.1 = p2.1 ∧ p2.1 = p3.1) → False ∧
  (p1.2 = p2.2 ∧ p2.2 = p3.2) → False ∧
  (p1.1 - p1.2 = p2.1 - p2.2 ∧ p2.1 - p2.2 = p3.1 - p3.2) → False

-- The main statement to prove
theorem nails_no_three_collinear :
  ∃ nails : List Position, List.length nails = 16 ∧ no_three_collinear nails :=
sorry

end nails_no_three_collinear_l522_52224


namespace trick_deck_cost_l522_52207

theorem trick_deck_cost :
  ∀ (x : ℝ), 3 * x + 2 * x = 35 → x = 7 :=
by
  sorry

end trick_deck_cost_l522_52207


namespace f_increasing_intervals_g_range_l522_52294

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := (1 + Real.sin x) * f x

theorem f_increasing_intervals : 
  (∀ x, 0 ≤ x → x ≤ Real.pi / 2 → 0 ≤ Real.cos x) ∧ (∀ x, 3 * Real.pi / 2 ≤ x → x ≤ 2 * Real.pi → 0 ≤ Real.cos x) :=
sorry

theorem g_range : 
  ∀ x, 0 ≤ x → x ≤ 2 * Real.pi → -1 / 2 ≤ g x ∧ g x ≤ 4 :=
sorry

end f_increasing_intervals_g_range_l522_52294


namespace shaded_region_area_computed_correctly_l522_52200

noncomputable def side_length : ℝ := 15
noncomputable def quarter_circle_radius : ℝ := side_length / 3
noncomputable def square_area : ℝ := side_length ^ 2
noncomputable def circle_area : ℝ := Real.pi * (quarter_circle_radius ^ 2)
noncomputable def shaded_region_area : ℝ := square_area - circle_area

theorem shaded_region_area_computed_correctly : 
  shaded_region_area = 225 - 25 * Real.pi := 
by 
  -- This statement only defines the proof problem.
  sorry

end shaded_region_area_computed_correctly_l522_52200


namespace smallest_positive_debt_pigs_goats_l522_52282

theorem smallest_positive_debt_pigs_goats :
  ∃ p g : ℤ, 350 * p + 240 * g = 10 :=
by
  sorry

end smallest_positive_debt_pigs_goats_l522_52282


namespace problem_lean_l522_52288

noncomputable def a : ℕ+ → ℝ := sorry

theorem problem_lean :
  a 11 = 1 / 52 ∧ (∀ n : ℕ+, 1 / a (n + 1) - 1 / a n = 5) → a 1 = 1 / 2 :=
by
  sorry

end problem_lean_l522_52288


namespace solve_factorial_equation_in_natural_numbers_l522_52247

theorem solve_factorial_equation_in_natural_numbers :
  ∃ n k : ℕ, n! + 3 * n + 8 = k^2 ↔ n = 2 ∧ k = 4 := by
sorry

end solve_factorial_equation_in_natural_numbers_l522_52247


namespace find_points_A_C_find_equation_line_l_l522_52221

variables (A B C : ℝ × ℝ)
variables (l : ℝ → ℝ)

-- Condition: the coordinates of point B are (2, 1)
def B_coord : Prop := B = (2, 1)

-- Condition: the equation of the line containing the altitude on side BC is x - 2y - 1 = 0
def altitude_BC (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Condition: the equation of the angle bisector of angle A is y = 0
def angle_bisector_A (y : ℝ) : Prop := y = 0

-- Statement of the theorems to be proved
theorem find_points_A_C
    (hB : B_coord B)
    (h_altitude_BC : altitude_BC 1 0)
    (h_angle_bisector_A : angle_bisector_A 0) :
  (A = (1, 0)) ∧ (C = (4, -3)) :=
sorry

theorem find_equation_line_l
    (hB : B_coord B)
    (h_altitude_BC : altitude_BC 1 0)
    (h_angle_bisector_A : angle_bisector_A 0)
    (hA : A = (1, 0)) :
  ((∀ x : ℝ, l x = x - 1)) :=
sorry

end find_points_A_C_find_equation_line_l_l522_52221


namespace problem1_problem2_l522_52287

theorem problem1 : 2 * Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 3 * Real.sqrt 2 :=
by
  -- Proof omitted
  sorry

theorem problem2 : (Real.sqrt 12 - Real.sqrt 24) / Real.sqrt 6 - 2 * Real.sqrt (1/2) = -2 :=
by
  -- Proof omitted
  sorry

end problem1_problem2_l522_52287


namespace sum_of_first_9_terms_is_27_l522_52243

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Definition for the geometric sequence
noncomputable def b_n (n : ℕ) : ℝ := sorry -- Definition for the arithmetic sequence

axiom a_geo_seq : ∃ r : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n * r
axiom b_ari_seq : ∃ d : ℝ, ∀ n : ℕ, b_n (n + 1) = b_n n + d
axiom a5_eq_3 : 3 * a_n 5 - a_n 3 * a_n 7 = 0
axiom b5_eq_a5 : b_n 5 = a_n 5

noncomputable def S_9 := (1 / 2) * 9 * (b_n 1 + b_n 9)

theorem sum_of_first_9_terms_is_27 : S_9 = 27 := by
  sorry

end sum_of_first_9_terms_is_27_l522_52243


namespace sum_of_prime_factors_eq_22_l522_52285

-- Conditions: n is defined as 3^6 - 1
def n : ℕ := 3^6 - 1

-- Statement: The sum of the prime factors of n is 22
theorem sum_of_prime_factors_eq_22 : 
  (∀ p : ℕ, p ∣ n → Prime p → p = 2 ∨ p = 7 ∨ p = 13) → 
  (2 + 7 + 13 = 22) :=
by sorry

end sum_of_prime_factors_eq_22_l522_52285


namespace optimal_garden_dimensions_l522_52260

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), l ≥ 100 ∧ w ≥ 60 ∧ l + w = 180 ∧ l * w = 8000 := by
  sorry

end optimal_garden_dimensions_l522_52260


namespace integer_solutions_count_2009_l522_52269

theorem integer_solutions_count_2009 :
  ∃ s : Finset (ℤ × ℤ × ℤ), (∀ (x y z : ℤ), (x, y, z) ∈ s ↔ x * y * z = 2009) ∧ s.card = 72 :=
  sorry

end integer_solutions_count_2009_l522_52269


namespace dina_has_60_dolls_l522_52274

variable (ivy_collectors_edition_dolls : ℕ)
variable (ivy_total_dolls : ℕ)
variable (dina_dolls : ℕ)

-- Conditions
def condition1 (ivy_total_dolls ivy_collectors_edition_dolls : ℕ) := ivy_collectors_edition_dolls = 20
def condition2 (ivy_total_dolls ivy_collectors_edition_dolls : ℕ) := (2 / 3 : ℚ) * ivy_total_dolls = ivy_collectors_edition_dolls
def condition3 (ivy_total_dolls dina_dolls : ℕ) := dina_dolls = 2 * ivy_total_dolls

-- Proof statement
theorem dina_has_60_dolls 
  (h1 : condition1 ivy_total_dolls ivy_collectors_edition_dolls) 
  (h2 : condition2 ivy_total_dolls ivy_collectors_edition_dolls) 
  (h3 : condition3 ivy_total_dolls dina_dolls) : 
  dina_dolls = 60 :=
sorry

end dina_has_60_dolls_l522_52274


namespace diametrically_opposite_points_l522_52226

theorem diametrically_opposite_points (n : ℕ) (h : (35 - 7 = n / 2)) : n = 56 := by
  sorry

end diametrically_opposite_points_l522_52226


namespace smallest_a_l522_52229

theorem smallest_a (a : ℤ) : 
  (112 ∣ (a * 43 * 62 * 1311)) ∧ (33 ∣ (a * 43 * 62 * 1311)) ↔ a = 1848 := 
sorry

end smallest_a_l522_52229


namespace polynomial_expression_value_l522_52292

theorem polynomial_expression_value (a : ℕ → ℤ) (x : ℤ) :
  (x + 2)^9 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 →
  ((a 1 + 3 * a 3 + 5 * a 5 + 7 * a 7 + 9 * a 9)^2 - (2 * a 2 + 4 * a 4 + 6 * a 6 + 8 * a 8)^2) = 3^12 :=
by
  sorry

end polynomial_expression_value_l522_52292


namespace students_taking_either_not_both_l522_52203

theorem students_taking_either_not_both (students_both : ℕ) (students_physics : ℕ) (students_only_chemistry : ℕ) :
  students_both = 12 →
  students_physics = 22 →
  students_only_chemistry = 9 →
  students_physics - students_both + students_only_chemistry = 19 :=
by
  intros h_both h_physics h_chemistry
  rw [h_both, h_physics, h_chemistry]
  repeat { sorry }

end students_taking_either_not_both_l522_52203


namespace pieces_of_gum_per_cousin_l522_52238

theorem pieces_of_gum_per_cousin (total_gum : ℕ) (num_cousins : ℕ) (h1 : total_gum = 20) (h2 : num_cousins = 4) : total_gum / num_cousins = 5 := by
  sorry

end pieces_of_gum_per_cousin_l522_52238


namespace largest_integer_divisor_of_p_squared_minus_3q_squared_l522_52255

theorem largest_integer_divisor_of_p_squared_minus_3q_squared (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h : q < p) :
  ∃ d : ℤ, (∀ p q : ℤ, p % 2 = 1 → q % 2 = 1 → q < p → d ∣ (p^2 - 3*q^2)) ∧ 
           (∀ k : ℤ, (∀ p q : ℤ, p % 2 = 1 → q % 2 = 1 → q < p → k ∣ (p^2 - 3*q^2)) → k ≤ d) ∧ d = 2 :=
sorry

end largest_integer_divisor_of_p_squared_minus_3q_squared_l522_52255


namespace paint_for_cube_l522_52244

theorem paint_for_cube (paint_per_unit_area : ℕ → ℕ → ℕ)
  (h2 : paint_per_unit_area 2 1 = 1) :
  paint_per_unit_area 6 1 = 9 :=
by
  sorry

end paint_for_cube_l522_52244


namespace smaller_of_two_digit_product_4680_l522_52266

theorem smaller_of_two_digit_product_4680 (a b : ℕ) (h1 : a * b = 4680) (h2 : 10 ≤ a) (h3 : a < 100) (h4 : 10 ≤ b) (h5 : b < 100): min a b = 40 :=
sorry

end smaller_of_two_digit_product_4680_l522_52266


namespace A_eq_three_l522_52248

theorem A_eq_three (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (A : ℤ)
  (h : A = ((a + 1 : ℕ) / (b : ℕ)) + (b : ℕ) / (a : ℕ)) : A = 3 := by
  sorry

end A_eq_three_l522_52248


namespace find_k_for_given_prime_l522_52296

theorem find_k_for_given_prime (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (k : ℕ) 
  (h : ∃ a : ℕ, k^2 - p * k = a^2) : 
  k = (p + 1)^2 / 4 :=
sorry

end find_k_for_given_prime_l522_52296


namespace second_person_fraction_removed_l522_52289

theorem second_person_fraction_removed (teeth_total : ℕ) 
    (removed1 removed3 removed4 : ℕ)
    (total_removed: ℕ)
    (h1: teeth_total = 32)
    (h2: removed1 = teeth_total / 4)
    (h3: removed3 = teeth_total / 2)
    (h4: removed4 = 4)
    (h5 : total_removed = 40):
    ((total_removed - (removed1 + removed3 + removed4)) : ℚ) / teeth_total = 3 / 8 :=
by
  sorry

end second_person_fraction_removed_l522_52289


namespace range_of_m_for_inequality_l522_52201

theorem range_of_m_for_inequality (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + m * x + m - 6 < 0) ↔ m < 8 := 
sorry

end range_of_m_for_inequality_l522_52201


namespace correct_average_after_error_l522_52220

theorem correct_average_after_error (n : ℕ) (a m_wrong m_correct : ℤ) 
  (h_n : n = 30) (h_a : a = 60) (h_m_wrong : m_wrong = 90) (h_m_correct : m_correct = 15) : 
  ((n * a + (m_correct - m_wrong)) / n : ℤ) = 57 := 
by
  sorry

end correct_average_after_error_l522_52220


namespace max_point_of_f_l522_52212

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Define the first derivative of the function
def f_prime (x : ℝ) : ℝ := 3 * x^2 - 12

-- Define the second derivative of the function
def f_double_prime (x : ℝ) : ℝ := 6 * x

-- Prove that a = -2 is the maximum value point of f(x)
theorem max_point_of_f : ∃ a : ℝ, (f_prime a = 0) ∧ (f_double_prime a < 0) ∧ (a = -2) :=
sorry

end max_point_of_f_l522_52212


namespace find_y_l522_52268

theorem find_y 
  (x y : ℕ) 
  (h1 : x % y = 9) 
  (h2 : x / y = 96) 
  (h3 : (x % y: ℝ) / y = 0.12) 
  : y = 75 := 
  by 
    sorry

end find_y_l522_52268


namespace guilt_of_X_and_Y_l522_52209

-- Definitions
variable (X Y : Prop)

-- Conditions
axiom condition1 : ¬X ∨ Y
axiom condition2 : X

-- Conclusion to prove
theorem guilt_of_X_and_Y : X ∧ Y := by
  sorry

end guilt_of_X_and_Y_l522_52209


namespace find_x_l522_52205

-- Define the angles AXB, CYX, and XYB as given in the problem.
def angle_AXB : ℝ := 150
def angle_CYX : ℝ := 130
def angle_XYB : ℝ := 55

-- Define a function that represents the sum of angles in a triangle.
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Define the angles.
def angle_XYZ : ℝ := angle_AXB - angle_XYB
def angle_YXZ : ℝ := 180 - angle_CYX
def angle_YXZ_proof (x : ℝ) : Prop := sum_of_angles_in_triangle angle_XYZ angle_YXZ x

-- State the theorem to be proved.
theorem find_x : angle_YXZ_proof 35 :=
sorry

end find_x_l522_52205


namespace package_cheaper_than_per_person_l522_52283

theorem package_cheaper_than_per_person (x : ℕ) :
  (90 * 6 + 10 * x < 54 * x + 8 * 3 * x) ↔ x ≥ 8 :=
by
  sorry

end package_cheaper_than_per_person_l522_52283


namespace find_third_number_l522_52256

theorem find_third_number (x : ℕ) : 9548 + 7314 = x + 13500 ↔ x = 3362 :=
by
  sorry

end find_third_number_l522_52256


namespace cistern_fill_time_l522_52250

theorem cistern_fill_time (hA : ∀ C : ℝ, 0 < C → ∀ t : ℝ, 0 < t → C / t = C / 10) 
                          (hB : ∀ C : ℝ, 0 < C → ∀ t : ℝ, 0 < t → C / t = -(C / 15)) :
  ∀ C : ℝ, 0 < C → ∃ t : ℝ, t = 30 := 
by 
  sorry

end cistern_fill_time_l522_52250


namespace distance_at_2_point_5_l522_52218

def distance_data : List (ℝ × ℝ) :=
  [(0, 0), (1, 10), (2, 40), (3, 90), (4, 160), (5, 250)]

def quadratic_relation (t s k : ℝ) : Prop :=
  s = k * t^2

theorem distance_at_2_point_5 :
  ∃ k : ℝ, (∀ (t s : ℝ), (t, s) ∈ distance_data → quadratic_relation t s k) ∧ quadratic_relation 2.5 62.5 k :=
by
  sorry

end distance_at_2_point_5_l522_52218


namespace quadratic_complete_square_l522_52217

theorem quadratic_complete_square :
  ∀ x : ℝ, x^2 - 4 * x + 5 = (x - 2)^2 + 1 :=
by
  intro x
  sorry

end quadratic_complete_square_l522_52217


namespace inverse_passes_through_3_4_l522_52279

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Given that f(x) has an inverse
def has_inverse := ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Given that y = f(x+1) passes through the point (3,3)
def condition := f (3 + 1) = 3

theorem inverse_passes_through_3_4 
  (h1 : has_inverse f) 
  (h2 : condition f) : 
  f⁻¹ 3 = 4 :=
sorry

end inverse_passes_through_3_4_l522_52279


namespace Eve_spend_l522_52242

noncomputable def hand_mitts := 14.00
noncomputable def apron := 16.00
noncomputable def utensils_set := 10.00
noncomputable def small_knife := 2 * utensils_set
noncomputable def total_cost_for_one_niece := hand_mitts + apron + utensils_set + small_knife
noncomputable def total_cost_for_three_nieces := 3 * total_cost_for_one_niece
noncomputable def discount := 0.25 * total_cost_for_three_nieces
noncomputable def final_cost := total_cost_for_three_nieces - discount

theorem Eve_spend : final_cost = 135.00 :=
by sorry

end Eve_spend_l522_52242


namespace decreasing_on_neg_l522_52262

variable (f : ℝ → ℝ)

-- Condition 1: f(x) is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Condition 2: f(x) is increasing on (0, +∞)
def increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

-- Theorem to prove: f(x) is decreasing on (-∞, 0)
theorem decreasing_on_neg (f : ℝ → ℝ) 
  (h_even : even_function f)
  (h_increasing : increasing_on_pos f) :
  ∀ x y, x < y → y < 0 → f y < f x :=
by 
  sorry

end decreasing_on_neg_l522_52262


namespace solution_sets_and_range_l522_52264

theorem solution_sets_and_range 
    (x a : ℝ) 
    (A : Set ℝ)
    (M : Set ℝ) :
    (∀ x, x ∈ A ↔ 1 ≤ x ∧ x ≤ 4) ∧
    (M = {x | (x - a) * (x - 2) ≤ 0} ) ∧
    (M ⊆ A) → (1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end solution_sets_and_range_l522_52264


namespace complex_equation_solution_l522_52273

theorem complex_equation_solution (x y : ℝ)
  (h : (x / (1 - (-ⅈ)) + y / (1 - 2 * (-ⅈ)) = 5 / (1 - 3 * (-ⅈ)))) :
  x + y = 4 :=
sorry

end complex_equation_solution_l522_52273


namespace comic_book_issue_pages_l522_52234

theorem comic_book_issue_pages (total_pages: ℕ) 
  (speed_month1 speed_month2 speed_month3: ℕ) 
  (bonus_pages: ℕ) (issue1_2_pages: ℕ) 
  (issue3_pages: ℕ)
  (h1: total_pages = 220)
  (h2: speed_month1 = 5)
  (h3: speed_month2 = 4)
  (h4: speed_month3 = 4)
  (h5: issue3_pages = issue1_2_pages + 4)
  (h6: bonus_pages = 3)
  (h7: (issue1_2_pages + bonus_pages) + 
       (issue1_2_pages + bonus_pages) + 
       (issue3_pages + bonus_pages) = total_pages) : 
  issue1_2_pages = 69 := 
by 
  sorry

end comic_book_issue_pages_l522_52234


namespace equal_constant_difference_l522_52216

theorem equal_constant_difference (x : ℤ) (k : ℤ) :
  x^2 - 6*x + 11 = k ∧ -x^2 + 8*x - 13 = k ∧ 3*x^2 - 16*x + 19 = k → x = 4 :=
by
  sorry

end equal_constant_difference_l522_52216


namespace f_2002_l522_52249

noncomputable def f : ℕ → ℝ := sorry

axiom f_condition (n : ℕ) (h : n > 1) : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

axiom f_2001 : f 2001 = 1

theorem f_2002 : f 2002 = 2 :=
  sorry

end f_2002_l522_52249


namespace base_number_mod_100_l522_52275

theorem base_number_mod_100 (base : ℕ) (h : base ^ 8 % 100 = 1) : base = 1 := 
sorry

end base_number_mod_100_l522_52275


namespace exists_root_in_interval_l522_52267

open Real

theorem exists_root_in_interval 
  (a b c r s : ℝ) 
  (ha : a ≠ 0) 
  (hr : a * r ^ 2 + b * r + c = 0) 
  (hs : -a * s ^ 2 + b * s + c = 0) : 
  ∃ t : ℝ, r < t ∧ t < s ∧ (a / 2) * t ^ 2 + b * t + c = 0 :=
by
  sorry

end exists_root_in_interval_l522_52267


namespace find_ages_of_siblings_l522_52223

-- Define the ages of the older brother and the younger sister as variables x and y
variables (x y : ℕ)

-- Define the conditions as provided in the problem
def condition1 : Prop := x = 4 * y
def condition2 : Prop := x + 3 = 3 * (y + 3)

-- State that the system of equations defined by condition1 and condition2 is consistent
theorem find_ages_of_siblings (x y : ℕ) (h1 : x = 4 * y) (h2 : x + 3 = 3 * (y + 3)) : 
  (x = 4 * y) ∧ (x + 3 = 3 * (y + 3)) :=
by 
  exact ⟨h1, h2⟩

end find_ages_of_siblings_l522_52223


namespace right_triangle_leg_square_l522_52225

theorem right_triangle_leg_square (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = a + 2) : 
  b^2 = 4 * a + 4 := 
by 
  sorry

end right_triangle_leg_square_l522_52225


namespace function_properties_l522_52239

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1 / x

theorem function_properties : 
  (∀ x : ℝ, x ≠ 0 → f (1 / x) + 2 * f x = 3 * x) ∧ 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x : ℝ, 0 < x → ∀ y : ℝ, x < y → f x < f y) := by
  -- Proof of the theorem would go here
  sorry

end function_properties_l522_52239


namespace coloring_possible_l522_52265

theorem coloring_possible (n k : ℕ) (h1 : 2 ≤ k ∧ k ≤ n) (h2 : n ≥ 2) :
  (n ≥ k ∧ k ≥ 3) ∨ (2 ≤ k ∧ k ≤ n ∧ n ≤ 3) :=
sorry

end coloring_possible_l522_52265


namespace find_fraction_l522_52208

theorem find_fraction
  (w x y F : ℝ)
  (h1 : 5 / w + F = 5 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  F = 10 := 
sorry

end find_fraction_l522_52208


namespace union_sets_l522_52237

-- Define the sets A and B based on the given conditions
def set_A : Set ℝ := {x | abs (x - 1) < 2}
def set_B : Set ℝ := {x | Real.log x / Real.log 2 < 3}

-- Problem statement: Prove that the union of sets A and B is {x | -1 < x < 9}
theorem union_sets : (set_A ∪ set_B) = {x | -1 < x ∧ x < 9} :=
by
  sorry

end union_sets_l522_52237


namespace intersection_M_N_l522_52215

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l522_52215


namespace display_total_cans_l522_52293

def row_num_cans (row : ℕ) : ℕ :=
  if row < 7 then 19 - 3 * (7 - row)
  else 19 + 3 * (row - 7)

def total_cans : ℕ :=
  List.sum (List.map row_num_cans (List.range 10))

theorem display_total_cans : total_cans = 145 := 
  sorry

end display_total_cans_l522_52293


namespace incorrect_statements_l522_52263

open Function

theorem incorrect_statements (a : ℝ) (x y x₁ y₁ x₂ y₂ k : ℝ) : 
  ¬ (a = -1 ↔ (∀ x y, a^2 * x - y + 1 = 0 ∧ x - a * y - 2 = 0 → (a = -1 ∨ a = 0))) ∧ 
  ¬ (∀ x y (x₁ y₁ x₂ y₂ : ℝ), (∃ (m : ℝ), (y - y₁) = m * (x - x₁) ∧ (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)) → 
    ((y - y₁) / (y₂ - y₁) = (x - x₁) / (x₂ - x₁))) :=
sorry

end incorrect_statements_l522_52263


namespace lemon_loaf_each_piece_weight_l522_52270

def pan_length := 20  -- cm
def pan_width := 18   -- cm
def pan_height := 5   -- cm
def total_pieces := 25
def density := 2      -- g/cm³

noncomputable def weight_of_each_piece : ℕ := by
  have volume := pan_length * pan_width * pan_height
  have volume_of_each_piece := volume / total_pieces
  have mass_of_each_piece := volume_of_each_piece * density
  exact mass_of_each_piece

theorem lemon_loaf_each_piece_weight :
  weight_of_each_piece = 144 :=
sorry

end lemon_loaf_each_piece_weight_l522_52270


namespace find_smaller_number_l522_52297

theorem find_smaller_number (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) : y = 18.5 :=
by
  sorry

end find_smaller_number_l522_52297


namespace jessica_mark_meet_time_jessica_mark_total_distance_l522_52257

noncomputable def jessica_start_time : ℚ := 7.75 -- 7:45 AM
noncomputable def mark_start_time : ℚ := 8.25 -- 8:15 AM
noncomputable def distance_between_towns : ℚ := 72
noncomputable def jessica_speed : ℚ := 14 -- miles per hour
noncomputable def mark_speed : ℚ := 18 -- miles per hour
noncomputable def t : ℚ := 81 / 32 -- time in hours when they meet

theorem jessica_mark_meet_time :
  7.75 + t = 10.28375 -- 10.17 hours in decimal
  :=
by
  -- Proof omitted.
  sorry

theorem jessica_mark_total_distance :
  jessica_speed * t + mark_speed * (t - (mark_start_time - jessica_start_time)) = distance_between_towns
  :=
by
  -- Proof omitted.
  sorry

end jessica_mark_meet_time_jessica_mark_total_distance_l522_52257


namespace sqrt_1001_1003_plus_1_eq_1002_verify_identity_sqrt_2014_2017_plus_1_eq_2014_2017_l522_52272

-- Define the first proof problem
theorem sqrt_1001_1003_plus_1_eq_1002 : Real.sqrt (1001 * 1003 + 1) = 1002 := 
by sorry

-- Define the second proof problem to verify the identity
theorem verify_identity (n : ℤ) : (n * (n + 3) + 1)^2 = n * (n + 1) * (n + 2) * (n + 3) + 1 :=
by sorry

-- Define the third proof problem
theorem sqrt_2014_2017_plus_1_eq_2014_2017 : Real.sqrt (2014 * 2015 * 2016 * 2017 + 1) = 2014 * 2017 :=
by sorry

end sqrt_1001_1003_plus_1_eq_1002_verify_identity_sqrt_2014_2017_plus_1_eq_2014_2017_l522_52272


namespace floor_sum_eq_55_l522_52278

noncomputable def x : ℝ := 9.42

theorem floor_sum_eq_55 : ∀ (x : ℝ), x = 9.42 → (⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋) = 55 := by
  intros
  sorry

end floor_sum_eq_55_l522_52278


namespace brownie_count_l522_52251

noncomputable def initial_brownies : ℕ := 20
noncomputable def to_school_administrator (n : ℕ) : ℕ := n / 2
noncomputable def remaining_after_administrator (n : ℕ) : ℕ := n - to_school_administrator n
noncomputable def to_best_friend (n : ℕ) : ℕ := remaining_after_administrator n / 2
noncomputable def remaining_after_best_friend (n : ℕ) : ℕ := remaining_after_administrator n - to_best_friend n
noncomputable def to_friend_simon : ℕ := 2
noncomputable def final_brownies : ℕ := remaining_after_best_friend initial_brownies - to_friend_simon

theorem brownie_count : final_brownies = 3 := by
  sorry

end brownie_count_l522_52251


namespace total_spending_in_4_years_l522_52206

def trevor_spending_per_year : ℕ := 80
def reed_to_trevor_diff : ℕ := 20
def reed_to_quinn_factor : ℕ := 2

theorem total_spending_in_4_years :
  ∃ (reed_spending quinn_spending : ℕ),
  (reed_spending = trevor_spending_per_year - reed_to_trevor_diff) ∧
  (reed_spending = reed_to_quinn_factor * quinn_spending) ∧
  ((trevor_spending_per_year + reed_spending + quinn_spending) * 4 = 680) :=
sorry

end total_spending_in_4_years_l522_52206


namespace find_a_of_parabola_and_hyperbola_intersection_l522_52299

theorem find_a_of_parabola_and_hyperbola_intersection
  (a : ℝ)
  (h_a_pos : a > 0)
  (h_asymptotes_intersect_directrix_distance : ∀ (x_A x_B : ℝ),
    -1 / (4 * a) = (1 / 2) * x_A ∧ -1 / (4 * a) = -(1 / 2) * x_B →
    |x_B - x_A| = 4) : a = 1 / 4 := by
  sorry

end find_a_of_parabola_and_hyperbola_intersection_l522_52299


namespace f_not_factorable_l522_52241

noncomputable def f (n : ℕ) (x : ℕ) : ℕ := x^n + 5 * x^(n - 1) + 3

theorem f_not_factorable (n : ℕ) (hn : n > 1) :
  ¬ ∃ g h : ℕ → ℕ, (∀ a b : ℕ, a ≠ 0 ∧ b ≠ 0 → g a * h b = f n a * f n b) ∧ 
    (∀ a b : ℕ, (g a = 0 ∧ h b = 0) → (a = 0 ∧ b = 0)) ∧ 
    (∃ pg qh : ℕ, pg ≥ 1 ∧ qh ≥ 1 ∧ g 1 = 1 ∧ h 1 = 1 ∧ (pg + qh = n)) := 
sorry

end f_not_factorable_l522_52241


namespace ball_initial_height_l522_52259

theorem ball_initial_height (c : ℝ) (d : ℝ) (h : ℝ) 
  (H1 : c = 4 / 5) 
  (H2 : d = 1080) 
  (H3 : d = h + 2 * h * c / (1 - c)) : 
  h = 216 :=
sorry

end ball_initial_height_l522_52259


namespace solution_set_of_absolute_value_inequality_l522_52232

theorem solution_set_of_absolute_value_inequality {x : ℝ} : 
  (|2 * x - 3| > 1) ↔ (x < 1 ∨ x > 2) := 
sorry

end solution_set_of_absolute_value_inequality_l522_52232


namespace number_of_solutions_l522_52277

theorem number_of_solutions : 
  ∃ n : ℕ, (∀ x y : ℕ, 3 * x + 4 * y = 766 → x % 2 = 0 → x > 0 → y > 0 → x = n * 2) ∧ n = 127 := 
by
  sorry

end number_of_solutions_l522_52277


namespace sum_of_three_numbers_is_neg_fifteen_l522_52219

theorem sum_of_three_numbers_is_neg_fifteen
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : (a + b + c) / 3 = a + 5)
  (h4 : (a + b + c) / 3 = c - 20)
  (h5 : b = 10) :
  a + b + c = -15 := by
  sorry

end sum_of_three_numbers_is_neg_fifteen_l522_52219


namespace number_of_cookies_on_the_fifth_plate_l522_52246

theorem number_of_cookies_on_the_fifth_plate
  (c : ℕ → ℕ)
  (h1 : c 1 = 5)
  (h2 : c 2 = 7)
  (h3 : c 3 = 10)
  (h4 : c 4 = 14)
  (h6 : c 6 = 25)
  (h_diff : ∀ n, c (n + 1) - c n = c (n + 2) - c (n + 1) + 1) :
  c 5 = 19 :=
by
  sorry

end number_of_cookies_on_the_fifth_plate_l522_52246


namespace arithmetic_sequence_a15_l522_52231

theorem arithmetic_sequence_a15 
  (a : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h1 : a 3 + a 13 = 20)
  (h2 : a 2 = -2) :
  a 15 = 24 := 
by
  sorry

end arithmetic_sequence_a15_l522_52231


namespace gcd_of_128_144_480_is_16_l522_52236

-- Define the three numbers
def a := 128
def b := 144
def c := 480

-- Define the problem statement in Lean
theorem gcd_of_128_144_480_is_16 : Int.gcd (Int.gcd a b) c = 16 :=
by
  -- Definitions using given conditions
  -- use Int.gcd function to define the problem precisely.
  -- The proof will be left as "sorry" since we don't need to solve it
  sorry

end gcd_of_128_144_480_is_16_l522_52236


namespace janes_score_is_110_l522_52233

-- Definitions and conditions
def sarah_score_condition (x y : ℕ) : Prop := x = y + 50
def average_score_condition (x y : ℕ) : Prop := (x + y) / 2 = 110
def janes_score (x y : ℕ) : ℕ := (x + y) / 2

-- The proof problem statement
theorem janes_score_is_110 (x y : ℕ) 
  (h_sarah : sarah_score_condition x y) 
  (h_avg   : average_score_condition x y) : 
  janes_score x y = 110 := 
by
  sorry

end janes_score_is_110_l522_52233


namespace sandy_change_from_twenty_dollar_bill_l522_52235

theorem sandy_change_from_twenty_dollar_bill :
  let cappuccino_cost := 2
  let iced_tea_cost := 3
  let cafe_latte_cost := 1.5
  let espresso_cost := 1
  let num_cappuccinos := 3
  let num_iced_teas := 2
  let num_cafe_lattes := 2
  let num_espressos := 2
  let total_cost := num_cappuccinos * cappuccino_cost
                  + num_iced_teas * iced_tea_cost
                  + num_cafe_lattes * cafe_latte_cost
                  + num_espressos * espresso_cost
  20 - total_cost = 3 := 
by
  sorry

end sandy_change_from_twenty_dollar_bill_l522_52235


namespace cost_of_blue_pill_l522_52213

variable (cost_total : ℝ) (days : ℕ) (daily_cost : ℝ)
variable (blue_pill : ℝ) (red_pill : ℝ)

-- Conditions
def condition1 (days : ℕ) : Prop := days = 21
def condition2 (blue_pill red_pill : ℝ) : Prop := blue_pill = red_pill + 2
def condition3 (cost_total daily_cost : ℝ) (days : ℕ) : Prop := cost_total = daily_cost * days
def condition4 (daily_cost blue_pill red_pill : ℝ) : Prop := daily_cost = blue_pill + red_pill

-- Target to prove
theorem cost_of_blue_pill
  (h1 : condition1 days)
  (h2 : condition2 blue_pill red_pill)
  (h3 : condition3 cost_total daily_cost days)
  (h4 : condition4 daily_cost blue_pill red_pill)
  (h5 : cost_total = 945) :
  blue_pill = 23.5 :=
by sorry

end cost_of_blue_pill_l522_52213


namespace probability_two_point_distribution_l522_52254

theorem probability_two_point_distribution 
  (P : ℕ → ℚ)
  (two_point_dist : P 0 + P 1 = 1)
  (condition : P 1 = (3 / 2) * P 0) :
  P 1 = 3 / 5 :=
by
  sorry

end probability_two_point_distribution_l522_52254


namespace solve_system_correct_l522_52202

noncomputable def solve_system (n : ℕ) (x : ℕ → ℝ) : Prop :=
  n > 2 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → x k + x (k + 1) = x (k + 2) ^ 2) ∧ 
  x (n + 1) = x 1 ∧ x (n + 2) = x 2 →
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → x i = 2

theorem solve_system_correct (n : ℕ) (x : ℕ → ℝ) : solve_system n x := 
sorry

end solve_system_correct_l522_52202


namespace sum_groups_is_250_l522_52240

-- Definitions based on the conditions
def group1 := [3, 13, 23, 33, 43]
def group2 := [7, 17, 27, 37, 47]

-- The proof problem
theorem sum_groups_is_250 : (group1.sum + group2.sum) = 250 :=
by
  sorry

end sum_groups_is_250_l522_52240


namespace simplify_expression_l522_52281

variable (q : ℝ)

theorem simplify_expression : ((6 * q + 2) - 3 * q * 5) * 4 + (5 - 2 / 4) * (7 * q - 14) = -4.5 * q - 55 :=
by sorry

end simplify_expression_l522_52281


namespace even_function_increasing_on_negative_half_l522_52227

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)

theorem even_function_increasing_on_negative_half (h1 : ∀ x, f (-x) = f x)
                                                  (h2 : ∀ a b : ℝ, a < b → b < 0 → f a < f b)
                                                  (h3 : x1 < 0 ∧ 0 < x2) (h4 : x1 + x2 > 0) 
                                                  : f (- x1) > f (x2) :=
by
  sorry

end even_function_increasing_on_negative_half_l522_52227


namespace rhombus_perimeter_l522_52230

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : ∃ p, p = 8 * Real.sqrt 41 := by
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  have h3 : s = 2 * Real.sqrt 41 := by sorry
  let p := 4 * s
  have h4 : p = 8 * Real.sqrt 41 := by sorry
  exact ⟨p, h4⟩

end rhombus_perimeter_l522_52230


namespace find_n_l522_52291

def sum_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (1 + n)

theorem find_n (k : ℕ) (h : k = 3) (hn : ∃ k, n = k^2)
  (hs : sum_first_n_even_numbers n = 90) : n = 9 :=
by
  sorry

end find_n_l522_52291


namespace lcm_of_4_6_10_18_l522_52245

theorem lcm_of_4_6_10_18 : Nat.lcm (Nat.lcm 4 6) (Nat.lcm 10 18) = 180 := by
  sorry

end lcm_of_4_6_10_18_l522_52245


namespace correct_calculation_option_l522_52222

theorem correct_calculation_option :
  (∀ a : ℝ, 3 * a^5 - a^5 ≠ 3) ∧
  (∀ a : ℝ, a^2 + a^5 ≠ a^7) ∧
  (∀ a : ℝ, a^5 + a^5 = 2 * a^5) ∧
  (∀ x y : ℝ, x^2 * y + x * y^2 ≠ 2 * x^3 * y^3) :=
by
  sorry

end correct_calculation_option_l522_52222


namespace Hans_current_age_l522_52204

variable {H : ℕ} -- Hans' current age

-- Conditions
def Josiah_age (H : ℕ) := 3 * H
def Hans_age_in_3_years (H : ℕ) := H + 3
def Josiah_age_in_3_years (H : ℕ) := Josiah_age H + 3
def sum_of_ages_in_3_years (H : ℕ) := Hans_age_in_3_years H + Josiah_age_in_3_years H

-- Theorem to prove
theorem Hans_current_age : sum_of_ages_in_3_years H = 66 → H = 15 :=
by
  sorry

end Hans_current_age_l522_52204


namespace probability_of_rolling_five_l522_52295

-- Define a cube with the given face numbers
def cube_faces : List ℕ := [1, 1, 2, 4, 5, 5]

-- Prove the probability of rolling a "5" is 1/3
theorem probability_of_rolling_five :
  (cube_faces.count 5 : ℚ) / cube_faces.length = 1 / 3 := by
  sorry

end probability_of_rolling_five_l522_52295


namespace ratio_avg_speeds_l522_52290

-- Definitions based on the problem conditions
def distance_A_B := 600
def time_Eddy := 3
def distance_A_C := 460
def time_Freddy := 4

-- Definition of average speeds
def avg_speed_Eddy := distance_A_B / time_Eddy
def avg_speed_Freddy := distance_A_C / time_Freddy

-- Theorem statement
theorem ratio_avg_speeds : avg_speed_Eddy / avg_speed_Freddy = 40 / 23 := 
sorry

end ratio_avg_speeds_l522_52290


namespace rectangle_area_l522_52280

variables (y w : ℝ)

-- Definitions from conditions
def is_width_of_rectangle : Prop := w = y / Real.sqrt 10
def is_length_of_rectangle : Prop := 3 * w = y / Real.sqrt 10

-- Theorem to be proved
theorem rectangle_area (h1 : is_width_of_rectangle y w) (h2 : is_length_of_rectangle y w) : 
  3 * (w^2) = 3 * (y^2 / 10) :=
by sorry

end rectangle_area_l522_52280


namespace negation_of_existence_lt_zero_l522_52276

theorem negation_of_existence_lt_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by sorry

end negation_of_existence_lt_zero_l522_52276


namespace age_difference_of_siblings_l522_52284

theorem age_difference_of_siblings (x : ℝ) 
  (h1 : 19 * x + 20 = 230) :
  |4 * x - 3 * x| = 210 / 19 := by
    sorry

end age_difference_of_siblings_l522_52284
