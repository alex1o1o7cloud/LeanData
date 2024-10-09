import Mathlib

namespace p_minus_q_value_l198_19844

theorem p_minus_q_value (p q : ℝ) (h1 : (x - 4) * (x + 4) = 24 * x - 96) (h2 : x^2 - 24 * x + 80 = 0) (h3 : p = 20) (h4 : q = 4) : p - q = 16 :=
by
  sorry

end p_minus_q_value_l198_19844


namespace value_of_d_l198_19852

theorem value_of_d (y : ℝ) (d : ℝ) (h1 : y > 0) (h2 : (4 * y) / 20 + (3 * y) / d = 0.5 * y) : d = 10 :=
by
  sorry

end value_of_d_l198_19852


namespace min_value_ab_l198_19846

theorem min_value_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a / 2) + b = 1) :
  (1 / a) + (1 / b) = (3 / 2) + Real.sqrt 2 :=
by sorry

end min_value_ab_l198_19846


namespace train_speed_l198_19849

theorem train_speed (v : ℝ) (d : ℝ) : 
  (v > 0) →
  (d > 0) →
  (d + (d - 55) = 495) →
  (d / v = (d - 55) / 25) →
  v = 31.25 := 
by
  intros hv hd hdist heqn
  -- We can leave the proof part out because we only need the statement
  sorry

end train_speed_l198_19849


namespace unique_zero_f_x1_minus_2x2_l198_19840

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

-- Condition a ≥ 0
variable (a : ℝ) (a_nonneg : 0 ≤ a)

-- Define the first part of the problem
theorem unique_zero_f : ∃! x, f a x = 0 :=
  sorry

-- Variables for the second part of the problem
variable (x₁ x₂ : ℝ)
variable (cond : f a x₁ = g a x₁ - g a x₂)

-- Define the second part of the problem
theorem x1_minus_2x2 : x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
  sorry

end unique_zero_f_x1_minus_2x2_l198_19840


namespace least_positive_integer_divisible_by_primes_gt_5_l198_19894

theorem least_positive_integer_divisible_by_primes_gt_5 : ∃ n : ℕ, n = 7 * 11 * 13 ∧ ∀ k : ℕ, (k > 0 ∧ (k % 7 = 0) ∧ (k % 11 = 0) ∧ (k % 13 = 0)) → k ≥ 1001 := 
sorry

end least_positive_integer_divisible_by_primes_gt_5_l198_19894


namespace assignment_statement_increases_l198_19868

theorem assignment_statement_increases (N : ℕ) : (N + 1 = N + 1) :=
sorry

end assignment_statement_increases_l198_19868


namespace solve_quadratic_inequality_l198_19853

theorem solve_quadratic_inequality :
  { x : ℝ | -3 * x^2 + 8 * x + 5 < 0 } = { x : ℝ | x < -1 ∨ x > 5 / 3 } :=
sorry

end solve_quadratic_inequality_l198_19853


namespace kims_total_points_l198_19896

theorem kims_total_points :
  let points_easy := 2
  let points_average := 3
  let points_hard := 5
  let answers_easy := 6
  let answers_average := 2
  let answers_hard := 4
  let total_points := (answers_easy * points_easy) + (answers_average * points_average) + (answers_hard * points_hard)
  total_points = 38 :=
by
  -- This is a placeholder to indicate that the proof is not included.
  sorry

end kims_total_points_l198_19896


namespace longest_side_range_of_obtuse_triangle_l198_19807

theorem longest_side_range_of_obtuse_triangle (a b c : ℝ) (h₁ : a = 1) (h₂ : b = 2) :
  a^2 + b^2 < c^2 → (Real.sqrt 5 < c ∧ c < 3) ∨ c = 2 :=
by
  sorry

end longest_side_range_of_obtuse_triangle_l198_19807


namespace math_problem_l198_19819

-- Definition of ⊕
def opp (a b : ℝ) : ℝ := a * b + a - b

-- Definition of ⊗
def tensor (a b : ℝ) : ℝ := (a * b) + a - b

theorem math_problem (a b : ℝ) :
  opp a b + tensor (b - a) b = b^2 - b := 
by
  sorry

end math_problem_l198_19819


namespace condition_3_implies_at_least_one_gt_one_condition_5_implies_at_least_one_gt_one_l198_19866

variables {a b : ℝ}

theorem condition_3_implies_at_least_one_gt_one (h : a + b > 2) : a > 1 ∨ b > 1 :=
sorry

theorem condition_5_implies_at_least_one_gt_one (h : ab > 1) : a > 1 ∨ b > 1 :=
sorry

end condition_3_implies_at_least_one_gt_one_condition_5_implies_at_least_one_gt_one_l198_19866


namespace parabola_transform_l198_19865

theorem parabola_transform (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + c = (x - 4)^2 - 3) → 
  b = 4 ∧ c = 6 := 
by
  sorry

end parabola_transform_l198_19865


namespace infinite_series_sum_l198_19823

theorem infinite_series_sum :
  ∑' n : ℕ, n / (8 : ℝ) ^ n = (8 / 49 : ℝ) :=
sorry

end infinite_series_sum_l198_19823


namespace largest_value_l198_19815

noncomputable def largest_possible_4x_3y (x y : ℝ) : ℝ :=
  4 * x + 3 * y

theorem largest_value (x y : ℝ) :
  x^2 + y^2 = 16 * x + 8 * y + 8 → (∃ x y, largest_possible_4x_3y x y = 9.64) :=
by
  sorry

end largest_value_l198_19815


namespace partI_partII_l198_19884

noncomputable def f (x m : ℝ) : ℝ := Real.log x - m * x
noncomputable def f' (x m : ℝ) : ℝ := (1 / x) - m

theorem partI (m : ℝ) : (∃ x : ℝ, x > 0 ∧ f x m = -1) → m = 1 := by
  sorry

theorem partII (x1 x2 : ℝ) (h1 : e ^ x1 ≤ x2) (h2 : f x1 1 = 0) (h3 : f x2 1 = 0) :
  ∃ y : ℝ, y = (x1 - x2) * f' (x1 + x2) 1 ∧ y = 2 / (1 + Real.exp 1) := by
  sorry

end partI_partII_l198_19884


namespace profit_calculation_correct_l198_19863

def main_actor_fee : ℕ := 500
def supporting_actor_fee : ℕ := 100
def extra_fee : ℕ := 50
def main_actor_food : ℕ := 10
def supporting_actor_food : ℕ := 5
def remaining_member_food : ℕ := 3
def post_production_cost : ℕ := 850
def revenue : ℕ := 10000

def total_actor_fees : ℕ := 2 * main_actor_fee + 3 * supporting_actor_fee + extra_fee
def total_food_cost : ℕ := 2 * main_actor_food + 4 * supporting_actor_food + 44 * remaining_member_food
def total_equipment_rental : ℕ := 2 * (total_actor_fees + total_food_cost)
def total_cost : ℕ := total_actor_fees + total_food_cost + total_equipment_rental + post_production_cost
def profit : ℕ := revenue - total_cost

theorem profit_calculation_correct : profit = 4584 :=
by
  -- proof omitted
  sorry

end profit_calculation_correct_l198_19863


namespace relationship_between_a_b_c_l198_19834

theorem relationship_between_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h₁ : a = (10 ^ 1988 + 1) / (10 ^ 1989 + 1))
  (h₂ : b = (10 ^ 1987 + 1) / (10 ^ 1988 + 1))
  (h₃ : c = (10 ^ 1987 + 9) / (10 ^ 1988 + 9)) :
  a < b ∧ b < c := 
sorry

end relationship_between_a_b_c_l198_19834


namespace find_numbers_l198_19885

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l198_19885


namespace dr_reeds_statement_l198_19827

variables (P Q : Prop)

theorem dr_reeds_statement (h : P → Q) : ¬Q → ¬P :=
by sorry

end dr_reeds_statement_l198_19827


namespace number_of_boxes_needed_l198_19893

theorem number_of_boxes_needed 
  (students : ℕ) (cookies_per_student : ℕ) (cookies_per_box : ℕ) 
  (total_students : students = 134) 
  (cookies_each : cookies_per_student = 7) 
  (cookies_in_box : cookies_per_box = 28) 
  (total_cookies : students * cookies_per_student = 938)
  : Nat.ceil (938 / 28) = 34 := 
by
  sorry

end number_of_boxes_needed_l198_19893


namespace sin_identity_l198_19897

theorem sin_identity (α : ℝ) (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := sorry

end sin_identity_l198_19897


namespace y_capital_l198_19874

theorem y_capital (X Y Z : ℕ) (Pz : ℕ) (Z_months_after_start : ℕ) (total_profit Z_share : ℕ)
    (hx : X = 20000)
    (hz : Z = 30000)
    (hz_profit : Z_share = 14000)
    (htotal_profit : total_profit = 50000)
    (hZ_months : Z_months_after_start = 5)
  : Y = 25000 := 
by
  -- Here we would have a proof, skipped with sorry for now
  sorry

end y_capital_l198_19874


namespace relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0_sufficiency_x_lt_1_necessity_x_lt_1_l198_19809

theorem relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0 (x : ℝ) :
  (x < 1) → (x^2 - 4 * x + 3 > 0) :=
by sorry

-- Define the sufficiency part
theorem sufficiency_x_lt_1 (x : ℝ) :
  (x < 1) → (x^2 - 4 * x + 3 > 0) :=
by sorry

-- Define the necessity part
theorem necessity_x_lt_1 (x : ℝ) :
  (x^2 - 4 * x + 3 > 0) → (x < 1 ∨ x > 3) :=
by sorry

end relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0_sufficiency_x_lt_1_necessity_x_lt_1_l198_19809


namespace a_eq_zero_l198_19850

theorem a_eq_zero (a b : ℤ) (h : ∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) : a = 0 :=
sorry

end a_eq_zero_l198_19850


namespace distance_center_to_point_l198_19816

theorem distance_center_to_point : 
  let center := (2, 3)
  let point  := (5, -2)
  let distance := Real.sqrt ((5 - 2)^2 + (-2 - 3)^2)
  distance = Real.sqrt 34 := by
  sorry

end distance_center_to_point_l198_19816


namespace exit_time_correct_l198_19878

def time_to_exit_wide : ℝ := 6
def time_to_exit_narrow : ℝ := 10

theorem exit_time_correct :
  ∃ x y : ℝ, x = 6 ∧ y = 10 ∧ 
  (1 / x + 1 / y = 4 / 15) ∧ 
  (y = x + 4) ∧ 
  (3.75 * (1 / x + 1 / y) = 1) :=
by
  use time_to_exit_wide
  use time_to_exit_narrow
  sorry

end exit_time_correct_l198_19878


namespace isosceles_triangle_largest_angle_l198_19871

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h : A = B) (h₁ : C + A + B = 180) (h₂ : C = 30) : 
  180 - 2 * 30 = 120 :=
by sorry

end isosceles_triangle_largest_angle_l198_19871


namespace find_a_l198_19821

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 := 
sorry

end find_a_l198_19821


namespace even_diagonal_moves_l198_19804

def King_Moves (ND D : ℕ) :=
  ND + D = 63 ∧ ND % 2 = 0

theorem even_diagonal_moves (ND D : ℕ) (traverse_board : King_Moves ND D) : D % 2 = 0 :=
by
  sorry

end even_diagonal_moves_l198_19804


namespace fraction_integer_l198_19839

theorem fraction_integer (x y : ℤ) (h₁ : ∃ k : ℤ, 3 * x + 4 * y = 5 * k) : ∃ m : ℤ, 4 * x - 3 * y = 5 * m :=
by
  sorry

end fraction_integer_l198_19839


namespace max_value_of_expression_l198_19830

theorem max_value_of_expression (x y z : ℝ) (h : 0 < x) (h' : 0 < y) (h'' : 0 < z) (hxyz : x * y * z = 1) :
  (∃ s, s = x ∧ ∃ t, t = y ∧ ∃ u, u = z ∧ 
  (x^2 * y / (x + y) + y^2 * z / (y + z) + z^2 * x / (z + x) ≤ 3 / 2)) :=
sorry

end max_value_of_expression_l198_19830


namespace ji_hoon_original_answer_l198_19864

-- Define the conditions: Ji-hoon's mistake
def ji_hoon_mistake (x : ℝ) := x - 7 = 0.45

-- The theorem statement
theorem ji_hoon_original_answer (x : ℝ) (h : ji_hoon_mistake x) : x * 7 = 52.15 :=
by
  sorry

end ji_hoon_original_answer_l198_19864


namespace maximum_area_of_rectangle_l198_19810

theorem maximum_area_of_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : ∃ A, A = 100 ∧ ∀ x' y', 2 * x' + 2 * y' = 40 → x' * y' ≤ A := by
  sorry

end maximum_area_of_rectangle_l198_19810


namespace find_theta_l198_19875

noncomputable def P := (Real.sin (3 * Real.pi / 4), Real.cos (3 * Real.pi / 4))

theorem find_theta
  (theta : ℝ)
  (h_theta_range : 0 ≤ theta ∧ theta < 2 * Real.pi)
  (h_P_theta : P = (Real.sin theta, Real.cos theta)) :
  theta = 7 * Real.pi / 4 :=
sorry

end find_theta_l198_19875


namespace probability_of_red_ball_l198_19822

theorem probability_of_red_ball :
  let total_balls := 9
  let red_balls := 6
  let probability := (red_balls : ℚ) / total_balls
  probability = (2 : ℚ) / 3 :=
by
  sorry

end probability_of_red_ball_l198_19822


namespace selling_price_l198_19803

def cost_price : ℝ := 76.92
def profit_rate : ℝ := 0.30

theorem selling_price : cost_price * (1 + profit_rate) = 100.00 := by
  sorry

end selling_price_l198_19803


namespace intersection_A_B_l198_19824

open Set

def A : Set ℝ := Icc 1 2

def B : Set ℤ := {x : ℤ | x^2 - 2 * x - 3 < 0}

theorem intersection_A_B :
  (A ∩ (coe '' B) : Set ℝ) = {1, 2} :=
sorry

end intersection_A_B_l198_19824


namespace outdoor_tables_count_l198_19888

variable (numIndoorTables : ℕ) (chairsPerIndoorTable : ℕ) (totalChairs : ℕ)
variable (chairsPerOutdoorTable : ℕ)

theorem outdoor_tables_count 
  (h1 : numIndoorTables = 8) 
  (h2 : chairsPerIndoorTable = 3) 
  (h3 : totalChairs = 60) 
  (h4 : chairsPerOutdoorTable = 3) :
  ∃ (numOutdoorTables : ℕ), numOutdoorTables = 12 := by
  admit

end outdoor_tables_count_l198_19888


namespace suzy_twice_mary_l198_19881

def suzy_current_age : ℕ := 20
def mary_current_age : ℕ := 8

theorem suzy_twice_mary (x : ℕ) : suzy_current_age + x = 2 * (mary_current_age + x) ↔ x = 4 := by
  sorry

end suzy_twice_mary_l198_19881


namespace solution_set_f_x_minus_2_pos_l198_19877

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then 2 * x - 4 else 2 * (-x) - 4

theorem solution_set_f_x_minus_2_pos :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry

end solution_set_f_x_minus_2_pos_l198_19877


namespace f_at_3_l198_19826

variable {R : Type} [LinearOrderedField R]

-- Define odd function
def is_odd_function (f : R → R) := ∀ x : R, f (-x) = -f x

-- Define the given function f and its properties
variables (f : R → R)
  (h_odd : is_odd_function f)
  (h_domain : ∀ x : R, true) -- domain is R implicitly
  (h_eq : ∀ x : R, f x + f (2 - x) = 4)

-- Prove that f(3) = 6
theorem f_at_3 : f 3 = 6 :=
  sorry

end f_at_3_l198_19826


namespace geometric_sequence_sum_l198_19859

variable {a : ℕ → ℕ}

-- Defining the geometric sequence and the conditions
def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def condition1 (a : ℕ → ℕ) : Prop :=
  a 1 = 3

def condition2 (a : ℕ → ℕ) : Prop :=
  a 1 + a 3 + a 5 = 21

-- The main theorem
theorem geometric_sequence_sum (a : ℕ → ℕ) (q : ℕ) 
  (h1 : condition1 a) (h2: condition2 a) (hq : is_geometric_sequence a q) : 
  a 3 + a 5 + a 7 = 42 := 
sorry

end geometric_sequence_sum_l198_19859


namespace hostel_provisions_l198_19820

theorem hostel_provisions (x : ℕ) (h1 : 250 * x = 200 * 40) : x = 32 :=
by
  sorry

end hostel_provisions_l198_19820


namespace find_swimming_speed_l198_19806

variable (S : ℝ)

def is_average_speed (x y avg : ℝ) : Prop :=
  avg = 2 * x * y / (x + y)

theorem find_swimming_speed
  (running_speed : ℝ := 7)
  (average_speed : ℝ := 4)
  (h : is_average_speed S running_speed average_speed) :
  S = 2.8 :=
by sorry

end find_swimming_speed_l198_19806


namespace base9_square_multiple_of_3_ab4c_l198_19873

theorem base9_square_multiple_of_3_ab4c (a b c : ℕ) (N : ℕ) (h1 : a ≠ 0)
  (h2 : N = a * 9^3 + b * 9^2 + 4 * 9 + c)
  (h3 : ∃ k : ℕ, N = k^2)
  (h4 : N % 3 = 0) :
  c = 0 :=
sorry

end base9_square_multiple_of_3_ab4c_l198_19873


namespace train_length_calculation_l198_19892

theorem train_length_calculation (L : ℝ) (t : ℝ) (v_faster : ℝ) (v_slower : ℝ) (relative_speed : ℝ) (total_distance : ℝ) :
  (v_faster = 60) →
  (v_slower = 40) →
  (relative_speed = (v_faster - v_slower) * 1000 / 3600) →
  (t = 48) →
  (total_distance = relative_speed * t) →
  (2 * L = total_distance) →
  L = 133.44 :=
by
  intros
  sorry

end train_length_calculation_l198_19892


namespace max_minus_min_l198_19812

noncomputable def f (x : ℝ) := if x > 0 then (x - 1) ^ 2 else (x + 1) ^ 2

theorem max_minus_min (n m : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ (-1 / 2) → n ≤ f x ∧ f x ≤ m) →
  m - n = 1 :=
by { sorry }

end max_minus_min_l198_19812


namespace average_marks_l198_19880

/-- Given that the total marks in physics, chemistry, and mathematics is 110 more than the marks obtained in physics. -/
theorem average_marks (P C M : ℕ) (h : P + C + M = P + 110) : (C + M) / 2 = 55 :=
by
  -- The proof goes here.
  sorry

end average_marks_l198_19880


namespace calculate_initial_budget_l198_19895

-- Definitions based on conditions
def cost_of_chicken := 12
def cost_per_pound_beef := 3
def pounds_of_beef := 5
def amount_left := 53

-- Derived definition for total cost of beef
def cost_of_beef := cost_per_pound_beef * pounds_of_beef
-- Derived definition for total spent
def total_spent := cost_of_chicken + cost_of_beef
-- Final calculation for initial budget
def initial_budget := total_spent + amount_left

-- Statement to prove
theorem calculate_initial_budget : initial_budget = 80 :=
by
  sorry

end calculate_initial_budget_l198_19895


namespace rational_m_abs_nonneg_l198_19813

theorem rational_m_abs_nonneg (m : ℚ) : m + |m| ≥ 0 :=
by sorry

end rational_m_abs_nonneg_l198_19813


namespace coordinates_of_point_on_x_axis_l198_19851

theorem coordinates_of_point_on_x_axis (m : ℤ) 
  (h : 2 * m + 8 = 0) : (m + 5, 2 * m + 8) = (1, 0) :=
sorry

end coordinates_of_point_on_x_axis_l198_19851


namespace percentage_of_sum_l198_19801

theorem percentage_of_sum (x y P : ℝ) (h1 : 0.50 * (x - y) = (P / 100) * (x + y)) (h2 : y = 0.25 * x) : P = 30 :=
by
  sorry

end percentage_of_sum_l198_19801


namespace function_C_is_quadratic_l198_19876

def isQuadratic (f : ℝ → ℝ) :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

def function_C (x : ℝ) : ℝ := (x + 1)^2 - 5

theorem function_C_is_quadratic : isQuadratic function_C :=
by
  sorry

end function_C_is_quadratic_l198_19876


namespace sum_of_first_n_terms_l198_19889

variable (a : ℕ → ℤ) (b : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Given conditions
axiom a_n_arith : ∀ n, a (n + 1) - a n = a 2 - a 1
axiom a_3 : a 3 = -6
axiom a_6 : a 6 = 0
axiom b_1 : b 1 = -8
axiom b_2 : b 2 = a 1 + a 2 + a 3

-- Correct answer to prove
theorem sum_of_first_n_terms : S n = 4 * (1 - 3^n) := sorry

end sum_of_first_n_terms_l198_19889


namespace reduced_price_per_kg_l198_19860

variable (P : ℝ)
variable (R : ℝ)
variable (Q : ℝ)

theorem reduced_price_per_kg
  (h1 : R = 0.75 * P)
  (h2 : 500 = Q * P)
  (h3 : 500 = (Q + 5) * R)
  (h4 : Q = 500 / P) :
  R = 25 :=
by
  sorry

end reduced_price_per_kg_l198_19860


namespace part1_part2_l198_19833

-- Definitions of sets A and B
def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def B (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ 3 - 2 * a }

-- Part 1: Prove that (complement of A union B = Universal Set) implies a in (-∞, 0]
theorem part1 (U : Set ℝ) (hU : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part 2: Prove that (A intersection B = B) implies a in [1/2, ∞)
theorem part2 (h : (A ∩ B a) = B a) : 1/2 ≤ a := sorry

end part1_part2_l198_19833


namespace chickens_and_sheep_are_ten_l198_19887

noncomputable def chickens_and_sheep_problem (C S : ℕ) : Prop :=
  (C + 4 * S = 2 * C) ∧ (2 * C + 4 * (S - 4) = 16 * (S - 4)) → (S + 2 = 10)

theorem chickens_and_sheep_are_ten (C S : ℕ) : chickens_and_sheep_problem C S :=
sorry

end chickens_and_sheep_are_ten_l198_19887


namespace fill_cistern_7_2_hours_l198_19854

theorem fill_cistern_7_2_hours :
  let R_fill := 1 / 4
  let R_empty := 1 / 9
  R_fill - R_empty = 5 / 36 →
  1 / (R_fill - R_empty) = 7.2 := 
by
  intros
  sorry

end fill_cistern_7_2_hours_l198_19854


namespace max_T_n_at_2_l198_19856

noncomputable def geom_seq (a n : ℕ) : ℕ :=
  a * 2 ^ n

noncomputable def S_n (a n : ℕ) : ℕ :=
  a * (2 ^ n - 1)

noncomputable def T_n (a n : ℕ) : ℕ :=
  (17 * S_n a n - S_n a (2 * n)) / geom_seq a n

theorem max_T_n_at_2 (a : ℕ) : (∀ n > 0, T_n a n ≤ T_n a 2) :=
by
  -- proof omitted
  sorry

end max_T_n_at_2_l198_19856


namespace cost_per_lunch_is_7_l198_19818

-- Definitions of the conditions
def total_children := 35
def total_chaperones := 5
def janet := 1
def additional_lunches := 3
def total_cost := 308

-- Calculate the total number of lunches
def total_lunches : Int :=
  total_children + total_chaperones + janet + additional_lunches

-- Statement to prove that the cost per lunch is 7
theorem cost_per_lunch_is_7 : total_cost / total_lunches = 7 := by
  sorry

end cost_per_lunch_is_7_l198_19818


namespace series_converges_l198_19883

theorem series_converges :
  ∑' n, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
sorry

end series_converges_l198_19883


namespace b8_expression_l198_19898

theorem b8_expression (a b : ℕ → ℚ)
  (ha0 : a 0 = 2)
  (hb0 : b 0 = 3)
  (ha : ∀ n, a (n + 1) = (a n) ^ 2 / (b n))
  (hb : ∀ n, b (n + 1) = (b n) ^ 2 / (a n)) :
  b 8 = 3 ^ 3281 / 2 ^ 3280 :=
by
  sorry

end b8_expression_l198_19898


namespace B_coordinates_when_A_is_origin_l198_19899

-- Definitions based on the conditions
def A_coordinates_when_B_is_origin := (2, 5)

-- Theorem to prove the coordinates of B when A is the origin
theorem B_coordinates_when_A_is_origin (x y : ℤ) :
    A_coordinates_when_B_is_origin = (2, 5) →
    (x, y) = (-2, -5) :=
by
  intro h
  -- skipping the proof steps
  sorry

end B_coordinates_when_A_is_origin_l198_19899


namespace total_value_is_155_l198_19870

def coin_count := 20
def silver_coin_count := 10
def silver_coin_value_total := 30
def gold_coin_count := 5
def regular_coin_value := 1

def silver_coin_value := silver_coin_value_total / 4
def gold_coin_value := 2 * silver_coin_value

def total_silver_value := silver_coin_count * silver_coin_value
def total_gold_value := gold_coin_count * gold_coin_value
def regular_coin_count := coin_count - (silver_coin_count + gold_coin_count)
def total_regular_value := regular_coin_count * regular_coin_value

def total_collection_value := total_silver_value + total_gold_value + total_regular_value

theorem total_value_is_155 : total_collection_value = 155 := 
by
  sorry

end total_value_is_155_l198_19870


namespace mul_inv_mod_391_l198_19814

theorem mul_inv_mod_391 (a : ℤ) (ha : 143 * a % 391 = 1) : a = 28 := by
  sorry

end mul_inv_mod_391_l198_19814


namespace true_statements_proved_l198_19862

-- Conditions
def A : Prop := ∃ n : ℕ, 25 = 5 * n
def B : Prop := (∃ m1 : ℕ, 209 = 19 * m1) ∧ (¬ ∃ m2 : ℕ, 63 = 19 * m2)
def C : Prop := (¬ ∃ k1 : ℕ, 90 = 30 * k1) ∧ (¬ ∃ k2 : ℕ, 49 = 30 * k2)
def D : Prop := (∃ l1 : ℕ, 34 = 17 * l1) ∧ (¬ ∃ l2 : ℕ, 68 = 17 * l2)
def E : Prop := ∃ q : ℕ, 140 = 7 * q

-- Correct statements
def TrueStatements : Prop := A ∧ B ∧ E ∧ ¬C ∧ ¬D

-- Lean statement to prove
theorem true_statements_proved : TrueStatements := 
by
  sorry

end true_statements_proved_l198_19862


namespace average_sleep_time_l198_19831

def sleep_times : List ℕ := [10, 9, 10, 8, 8]

theorem average_sleep_time : (sleep_times.sum / sleep_times.length) = 9 := by
  sorry

end average_sleep_time_l198_19831


namespace equilateral_triangle_l198_19869

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 0 < α ∧ α < π)
  (h5 : 0 < β ∧ β < π)
  (h6 : 0 < γ ∧ γ < π)
  (h7 : α + β + γ = π)
  (h8 : a * (1 - 2 * Real.cos α) + b * (1 - 2 * Real.cos β) + c * (1 - 2 * Real.cos γ) = 0) :
  α = β ∧ β = γ ∧ γ = α :=
by
  sorry

end equilateral_triangle_l198_19869


namespace difference_high_low_score_l198_19845

theorem difference_high_low_score :
  ∀ (num_innings : ℕ) (total_runs : ℕ) (exc_total_runs : ℕ) (high_score : ℕ) (low_score : ℕ),
  num_innings = 46 →
  total_runs = 60 * 46 →
  exc_total_runs = 58 * 44 →
  high_score = 194 →
  total_runs - exc_total_runs = high_score + low_score →
  high_score - low_score = 180 :=
by
  intros num_innings total_runs exc_total_runs high_score low_score h_innings h_total h_exc_total h_high_sum h_difference
  sorry

end difference_high_low_score_l198_19845


namespace no_polyhedron_with_area_ratio_ge_two_l198_19843

theorem no_polyhedron_with_area_ratio_ge_two (n : ℕ) (areas : Fin n → ℝ)
  (h : ∀ (i j : Fin n), i < j → (areas j) / (areas i) ≥ 2) : False := by
  sorry

end no_polyhedron_with_area_ratio_ge_two_l198_19843


namespace total_points_l198_19811

theorem total_points (total_players : ℕ) (paige_points : ℕ) (other_points : ℕ) (points_per_other_player : ℕ) :
  total_players = 5 →
  paige_points = 11 →
  points_per_other_player = 6 →
  other_points = (total_players - 1) * points_per_other_player →
  paige_points + other_points = 35 :=
by
  intro h_total_players h_paige_points h_points_per_other_player h_other_points
  sorry

end total_points_l198_19811


namespace determine_m_l198_19857

variables (m x : ℝ)
noncomputable def f (x : ℝ) := x^2 - 3*x + m
noncomputable def g (x : ℝ) := x^2 - 3*x + 5*m

theorem determine_m (h : 3 * f 5 = 2 * g 5) : m = 10 / 7 :=
by
  sorry

end determine_m_l198_19857


namespace labor_union_tree_equation_l198_19837

theorem labor_union_tree_equation (x : ℕ) : 2 * x + 21 = 3 * x - 24 := 
sorry

end labor_union_tree_equation_l198_19837


namespace min_value_96_l198_19835

noncomputable def min_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 32) : ℝ :=
x^2 + 4 * x * y + 4 * y^2 + 2 * z^2

theorem min_value_96 (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 32) :
  min_value x y z h_pos h_xyz = 96 :=
sorry

end min_value_96_l198_19835


namespace largest_fraction_l198_19847

theorem largest_fraction :
  let A := (5 : ℚ) / 11
  let B := (7 : ℚ) / 15
  let C := (29 : ℚ) / 59
  let D := (200 : ℚ) / 399
  let E := (251 : ℚ) / 501
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_fraction_l198_19847


namespace value_of_k_l198_19855

theorem value_of_k :
  ∃ k, k = 2 ∧ (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 5 ∧
                ∀ (s t : ℕ), (s, t) ∈ pairs → s = k * t) :=
by 
sorry

end value_of_k_l198_19855


namespace sufficient_but_not_necessary_l198_19825

def P (x : ℝ) : Prop := 2 < x ∧ x < 4
def Q (x : ℝ) : Prop := Real.log x < Real.exp 1

theorem sufficient_but_not_necessary (x : ℝ) : P x → Q x ∧ (¬ ∀ x, Q x → P x) := by
  sorry

end sufficient_but_not_necessary_l198_19825


namespace four_diff_digits_per_day_l198_19890

def valid_time_period (start_hour : ℕ) (end_hour : ℕ) : ℕ :=
  let total_minutes := (end_hour - start_hour + 1) * 60
  let valid_combinations :=
    match start_hour with
    | 0 => 0  -- start with appropriate calculation logic
    | 2 => 0  -- start with appropriate calculation logic
    | _ => 0  -- for general case, replace with correct logic
  total_minutes + valid_combinations  -- use proper aggregation

theorem four_diff_digits_per_day :
  valid_time_period 0 19 + valid_time_period 20 23 = 588 :=
by
  sorry

end four_diff_digits_per_day_l198_19890


namespace cubic_no_negative_roots_l198_19802

noncomputable def cubic_eq (x : ℝ) : ℝ := x^3 - 9 * x^2 + 23 * x - 15

theorem cubic_no_negative_roots {x : ℝ} : cubic_eq x = 0 → 0 ≤ x := sorry

end cubic_no_negative_roots_l198_19802


namespace susan_avg_speed_l198_19842

theorem susan_avg_speed 
  (speed1 : ℕ)
  (distance1 : ℕ)
  (speed2 : ℕ)
  (distance2 : ℕ)
  (no_stops : Prop) 
  (H1 : speed1 = 15)
  (H2 : distance1 = 40)
  (H3 : speed2 = 60)
  (H4 : distance2 = 20)
  (H5 : no_stops) :
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_distance / total_time
  avg_speed = 20 := by
  sorry

end susan_avg_speed_l198_19842


namespace sausages_fried_l198_19841

def num_eggs : ℕ := 6
def time_per_sausage : ℕ := 5
def time_per_egg : ℕ := 4
def total_time : ℕ := 39
def time_per_sauteurs (S : ℕ) : ℕ := S * time_per_sausage

theorem sausages_fried (S : ℕ) (h : num_eggs * time_per_egg + S * time_per_sausage = total_time) : S = 3 :=
by
  sorry

end sausages_fried_l198_19841


namespace batsman_average_after_25th_innings_l198_19848

theorem batsman_average_after_25th_innings :
  ∃ A : ℝ, 
    (∀ s : ℝ, s = 25 * A + 62.5 → 24 * A + 95 = s) →
    A + 2.5 = 35 :=
by
  sorry

end batsman_average_after_25th_innings_l198_19848


namespace find_a_l198_19872

theorem find_a (a : ℝ) : (∀ x : ℝ, |x - a| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  intro h
  have h1 : |(-1 : ℝ) - a| = 3 := sorry
  have h2 : |(5 : ℝ) - a| = 3 := sorry
  sorry

end find_a_l198_19872


namespace problem_solution_l198_19861

noncomputable def given_problem : ℝ := (Real.pi - 3)^0 - Real.sqrt 8 + 2 * Real.sin (45 * Real.pi / 180) + (1 / 2)⁻¹

theorem problem_solution : given_problem = 3 - Real.sqrt 2 := by
  sorry

end problem_solution_l198_19861


namespace divisible_by_3_l198_19832

theorem divisible_by_3 (x y : ℤ) (h : (x^2 + y^2) % 3 = 0) : x % 3 = 0 ∧ y % 3 = 0 :=
sorry

end divisible_by_3_l198_19832


namespace x_y_iff_pos_l198_19858

theorem x_y_iff_pos (x y : ℝ) : x + y > |x - y| ↔ x > 0 ∧ y > 0 := by
  sorry

end x_y_iff_pos_l198_19858


namespace infinite_points_in_region_l198_19828

theorem infinite_points_in_region : 
  ∀ x y : ℚ, 0 < x → 0 < y → x + 2 * y ≤ 6 → ¬(∃ n : ℕ, ∀ x y : ℚ, 0 < x → 0 < y → x + 2 * y ≤ 6 → sorry) :=
sorry

end infinite_points_in_region_l198_19828


namespace custom_star_calc_l198_19838

-- defining the custom operation "*"
def custom_star (a b : ℤ) : ℤ :=
  a * b - (b-1) * b

-- providing the theorem statement
theorem custom_star_calc : custom_star 2 (-3) = -18 :=
  sorry

end custom_star_calc_l198_19838


namespace problem1_solution_problem2_solution_l198_19879

noncomputable def problem1 : ℝ :=
  (Real.sqrt (1 / 3) + Real.sqrt 6) / Real.sqrt 3

noncomputable def problem2 : ℝ :=
  (Real.sqrt 3)^2 - Real.sqrt 4 + Real.sqrt ((-2)^2)

theorem problem1_solution :
  problem1 = 1 + 3 * Real.sqrt 2 :=
by
  sorry

theorem problem2_solution :
  problem2 = 3 :=
by
  sorry

end problem1_solution_problem2_solution_l198_19879


namespace remainder_when_2519_divided_by_3_l198_19886

theorem remainder_when_2519_divided_by_3 :
  2519 % 3 = 2 :=
by
  sorry

end remainder_when_2519_divided_by_3_l198_19886


namespace common_ratio_of_geometric_sequence_l198_19817

theorem common_ratio_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_nonzero : d ≠ 0) 
  (h_geom : (a 1)^2 = a 0 * a 2) :
  (a 2) / (a 0) = 3 / 2 := 
sorry

end common_ratio_of_geometric_sequence_l198_19817


namespace triangle_angle_A_l198_19829

theorem triangle_angle_A (a c C A : Real) (h1 : a = 1) (h2 : c = Real.sqrt 3) (h3 : C = 2 * Real.pi / 3) 
(h4 : Real.sin A = 1 / 2) : A = Real.pi / 6 :=
sorry

end triangle_angle_A_l198_19829


namespace digit_B_divisibility_l198_19836

theorem digit_B_divisibility :
  ∃ B : ℕ, B < 10 ∧
    (∃ n : ℕ, 658274 * 10 + B = 2 * n) ∧
    (∃ m : ℕ, 6582740 + B = 4 * m) ∧
    (B = 0 ∨ B = 5) ∧
    (∃ k : ℕ, 658274 * 10 + B = 7 * k) ∧
    (∃ p : ℕ, 6582740 + B = 8 * p) :=
sorry

end digit_B_divisibility_l198_19836


namespace fish_count_l198_19800

variables
  (x g s r : ℕ)
  (h1 : x - g = (2 / 3 : ℚ) * x - 1)
  (h2 : x - r = (2 / 3 : ℚ) * x + 4)
  (h3 : x = g + s + r)

theorem fish_count :
  s - g = 2 :=
by
  sorry

end fish_count_l198_19800


namespace polynomial_divisibility_l198_19882

open Polynomial

variables {R : Type*} [CommRing R]
variables {f g h k : R[X]}

theorem polynomial_divisibility (h1 : (X^2 + 1) * h + (X - 1) * f + (X - 2) * g = 0)
    (h2 : (X^2 + 1) * k + (X + 1) * f + (X + 2) * g = 0) :
    (X^2 + 1) ∣ (f * g) :=
sorry

end polynomial_divisibility_l198_19882


namespace appears_every_number_smallest_triplicate_number_l198_19891

open Nat

/-- Pascal's triangle is constructed such that each number 
    is the sum of the two numbers directly above it in the 
    previous row -/
def pascal (r k : ℕ) : ℕ :=
  if k > r then 0 else Nat.choose r k

/-- Every positive integer does appear at least once, but not 
    necessarily more than once for smaller numbers -/
theorem appears_every_number (n : ℕ) : ∃ r k : ℕ, pascal r k = n := sorry

/-- The smallest three-digit number in Pascal's triangle 
    that appears more than once is 102 -/
theorem smallest_triplicate_number : ∃ r1 k1 r2 k2 : ℕ, 
  100 ≤ pascal r1 k1 ∧ pascal r1 k1 < 1000 ∧ 
  pascal r1 k1 = 102 ∧ 
  r1 ≠ r2 ∧ k1 ≠ k2 ∧ 
  pascal r1 k1 = pascal r2 k2 := sorry

end appears_every_number_smallest_triplicate_number_l198_19891


namespace number_of_medium_boxes_l198_19808

def large_box_tape := 4
def medium_box_tape := 2
def small_box_tape := 1
def label_tape := 1

def num_large_boxes := 2
def num_small_boxes := 5
def total_tape := 44

theorem number_of_medium_boxes :
  let tape_used_large_boxes := num_large_boxes * (large_box_tape + label_tape)
  let tape_used_small_boxes := num_small_boxes * (small_box_tape + label_tape)
  let tape_used_medium_boxes := total_tape - (tape_used_large_boxes + tape_used_small_boxes)
  let medium_box_total_tape := medium_box_tape + label_tape
  let num_medium_boxes := tape_used_medium_boxes / medium_box_total_tape
  num_medium_boxes = 8 :=
by
  sorry

end number_of_medium_boxes_l198_19808


namespace problem_statement_l198_19867

theorem problem_statement (p q : ℚ) (hp : 3 / p = 4) (hq : 3 / q = 18) : p - q = 7 / 12 := 
by
  sorry

end problem_statement_l198_19867


namespace lara_harvest_raspberries_l198_19805

-- Define measurements of the garden
def length : ℕ := 10
def width : ℕ := 7

-- Define planting and harvesting constants
def plants_per_sq_ft : ℕ := 5
def raspberries_per_plant : ℕ := 12

-- Calculate expected number of raspberries
theorem lara_harvest_raspberries :  length * width * plants_per_sq_ft * raspberries_per_plant = 4200 := 
by sorry

end lara_harvest_raspberries_l198_19805
